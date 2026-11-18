# Changelog

All notable changes to HC4J. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is [semantic](https://semver.org/spec/v2.0.0.html), with the usual pre-1.0 caveat that
the API may still change between minor versions.

## [0.1.0] — unreleased

The first release with the full engine: linear algebra, a neural-network op layer, a tiered memory
manager and kernel fusion, all verified against CPU references on both DX12 and Vulkan.

### Added

**Neural-network ops** (`hc4j.ops.NnOps`, fluent forms on `Tensor`)
- 7 activations — ReLU, leaky ReLU, ELU, GELU (tanh approximation), SiLU, sigmoid, softplus — as
  elementwise ops, so they inherit strided inputs, in-place output and the streaming fallback.
- Row-wise softmax and log-softmax using the online max-and-sum recurrence; layer norm with Welford
  accumulation; RMS norm; both with optional affine parameters.
- Softmax cross entropy from logits and i32 class indices, computed without materialising the
  softmax. An out-of-range class yields NaN for that row.
- Bias broadcast over rows (`addRowBias`) and `linear` (GEMM + bias).
- `relu`, `sigmoid`, `gelu`, `silu` and `softplus` are available inside fused expressions.

**Collectives** (`collective.rs`, `hc4j.collective`)
- Ring AllReduce (sum) across every GPU on the machine: `n-1` scatter-reduce rounds and `n-1`
  allgather rounds, so each rank moves `2(n-1)/n` of the buffer instead of `n-1` copies.
- A bandwidth topology probe over every directed rank pair, reported as `Link` records.
- Ranks spread across the distinct physical GPUs, best first, falling back to extra logical devices
  on a GPU already in use (**loopback mode**) so the whole path runs on a single-GPU machine.
- Java API: `Collective.open(ranks)`, `allocate`, `allReduceSum`, `topology`, `CollectiveBuffer`.
- Transport is host-staged and synchronous; `Transport.PEER_TO_PEER` is defined but not produced,
  since wgpu exposes neither external-memory export/import nor timeline semaphores. Each hop
  blocks on `Device::poll(Wait)` where a raw-Vulkan transport would pipeline it.

**Reductions** (`ops/reduce.rs`)
- sum, mean, max and min over a whole tensor or along the last axis; argmax and argmin as i32
  indices.
- Folded loads so `sum(x²)` and mean squared error run as one reduction with no intermediate
  tensor.
- A full reduction is exactly two dispatches at any input size, with at most 4 KiB of scratch.

**Linear algebra**
- GEMM with five kernels chosen per shape — two register-tiled variants, a 16×16 shared-memory
  fallback, and two GEMV mappings with an optional subgroup reduction — plus an in-VRAM transpose.
- Out-of-core matmul: row-block streaming, then blocking over K when B does not fit either.

**Memory and scheduling**
- Three-tier memory manager (VRAM → host RAM → disk) with LRU eviction, page-in on use, streaming
  for tensors larger than the whole VRAM budget, and self-deleting spill files.
- Slab sub-allocation with epoch-based quarantine: 200 consecutive op outputs cost 0 driver
  allocations.
- `GpuBatch` records many ops into one command buffer and one submission.
- Kernel fusion: `a.lazy().sin().add(b).mul(c).eval()` compiles to one kernel, moving 4N bytes
  instead of 8N.

**API and tooling**
- Scalar arithmetic on `Tensor` (`add`/`sub`/`mul`/`div` with a float), each one fused kernel.
- `rsub`, `rdiv` and `rpow` on `FusedExpr` for a scalar on the left.
- `Tensor.residency()` and `Tensor.evict()`; host↔device byte counters in `EngineStats`.
- JMH benchmarks for elementwise ops, matmul and the NN layer.
- `README`, `docs/ARCHITECTURE.md`, `docs/API.md` and `docs/PERFORMANCE.md`.

### Fixed

- **A collective ring could recruit a software rasterizer.** Windows enumerates WARP
  ("Microsoft Basic Render Driver") beside the real GPU, and placing a rank there measured
  0.01 GB/s on that hop and made a 1 Mi-element AllReduce take 864 ms instead of 104 ms. Software
  adapters are now used only when the machine has nothing else.
- **One physical GPU enumerated under two backends looked like two devices.** A 2-rank ring on a
  single-GPU machine split itself across DX12 and Vulkan; adapters are now deduplicated by
  `(vendor, device, name)`, keeping the better-ranked backend.
- **Phantom submission epochs.** Cancelling a queued zero-fill left the stream's `pending` flag set
  with nothing to submit, so `retire_epoch()` could name an epoch no submission would ever claim and
  a slab region quarantined against it could never be reclaimed. It surfaced far from the cause, as
  `configure` failing with "epoch N was never submitted" while lowering a budget.
- **GELU returned NaN for large inputs.** The tanh argument reaches ~24000 at x = 88, and a driver
  evaluating `tanh` as `(e^2a−1)/(e^2a+1)` gets `inf/inf`. The cubic argument is now clamped to ±10.
- **Shader compilation under D3D12's FXC** took up to 190 s for the 128×128 GEMM, caused by WGSL
  workgroup-memory zero-initialisation scaling with the declared workgroup storage. Disabled, with
  explicit zeroing where a kernel needs it: 190 s → 5.5 s, and 2048³ throughput rose with it.
- **Command batching was 2.7× slower than not batching**, because each dispatch began its own
  compute pass (~2.4 ms per pass in `encoder.finish()`). One persistent pass per batch made 100
  small ops ~20× faster instead.
- **Dispatches past 67.1M elements were silently invalid** against the 65535-workgroup per-dimension
  limit. Grids now fold into 2-D and kernels rebuild their linear index from `num_workgroups`.
- **Adapter selection could return Vulkan on Windows** when falling back to a power preference. The
  preference now picks the GPU and the backend is re-ranked, so DX12 wins by default.
- **First matmul paid the pipeline compile** (569.5 ms). A background warm-up thread compiles the
  matmul pipelines at engine init: 569.5 → 22.4 ms.
- **Upload chunking cost 35%** on 40 MiB tensors; tensors below 256 MiB now upload in one call.
- `cos` could run `sin`'s pipeline — the cache was keyed on the entry point, which is `main` for
  every unary kernel.
- `meta` and `target` are WGSL reserved words, and WGSL has no shadowing; naga validation tests now
  cover every generated shader.
- `Tensor.fromArray` read out of bounds when the array length did not match the shape product.

### Release engineering

- `gradle/wrapper/gradle-wrapper.jar` is no longer excluded by `.gitignore`. Without it a fresh
  clone could not run `./gradlew` at all.
- Rust package renamed from `rust` to `hc4j`, with description, license and repository metadata and
  an MSRV of 1.88 (let-chains in edition 2024).
- Gradle and Cargo versions aligned at 0.1.0.
- Javadoc is a build gate (`-Xdoclint:all,-missing`): broken `{@link}`s, malformed tags and bad HTML
  now fail the build.

### Known limitations

- Forward pass only: no autograd and no optimizer. The ops compute activations, normalisations and
  losses, not their gradients.
- f32 throughout, plus i32 arithmetic. `SHADER_F16` is probed and reported but unused.
- No general broadcasting; the one exception is a bias over rows.
- Matmul is 2-D (with NumPy's 1-D conventions). No batched GEMM.
- Row-wise ops need a contiguous operand within one storage binding and have no host-streaming
  fallback.
- Collectives are host-staged and synchronous, with no true peer-to-peer path and no zero-copy
  bridge between a collective buffer and a `Tensor`.
- Outside `hc4j.collective` the engine is single-device.

## [0.0.1] — 2026-08-02

Initial proof of concept: the trigonometric suite over the Java Panama FFM bindings, with a wgpu
compute backend.
