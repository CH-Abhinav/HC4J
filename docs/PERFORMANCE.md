# HC4J Performance

Measured numbers, the optimisations that produced them, the two that were measured and rejected,
and how to reproduce any of it.

## Measurement setup

Unless stated otherwise, every number below was measured on:

```
Intel(R) Iris(R) Xe Graphics | IntegratedGpu | Dx12 | driver 31.0.101.4502
subgroups=false (size 16..16), shader_f16=false, workgroups/dim=65535, storage-binding=2047 MiB
JDK 25.0.4, Windows 11
```

This is a laptop integrated GPU, so absolute numbers move with thermal and power state between
runs — treat ratios as the result and absolute throughput as indicative. Several figures were
captured during development on the same machine and are labelled where they differ from a current
re-measurement.

Every benchmark calls `WgpuBackend.synchronize()` inside the timed region. GPU work is
asynchronous; without it you measure command recording, which is roughly 15–20 µs per op and tells
you nothing about the kernel.

## Current numbers

### Matmul (square, f32, `REGISTER_64_VEC4`)

| Shape | DX12 | | Vulkan | |
| --- | --- | --- | --- | --- |
| 512³ | 1.55 ms | 173.1 GFLOP/s | 1.85 ms | 144.8 GFLOP/s |
| 1024³ | 5.66 ms | 379.4 GFLOP/s | 7.50 ms | 286.3 GFLOP/s |
| 2048³ | 37.17 ms | 462.2 GFLOP/s | 54.67 ms | 314.2 GFLOP/s |

DX12 is consistently ahead of Vulkan on this adapter, which is why the adapter ranking prefers it
on Windows.

### JMH, 1 Mi f32 elements

| Benchmark | Time | |
| --- | --- | --- |
| `chainUnfused` — `sin(a)+b)*c` as 3 kernels | 3.099 ms | |
| `chainFused` — same as 1 kernel | **1.775 ms** | 1.75× |
| `manyOpsUnbatched` — 64 small ops, 64 submissions | 49.875 ms | |
| `manyOpsBatched` — same 64 ops, 1 submission | **11.944 ms** | 4.2× |
| `gemm` 512³ | 1.661 ms | |
| `gemvRows` 512×512 · 512 | 0.641 ms | submission-latency bound |
| `gemvCols` 512 · 512×512 | 0.804 ms | submission-latency bound |
| `transpose` 512×512 | 0.697 ms | submission-latency bound |

The GEMV and transpose figures sit at this machine's ~0.6 ms submit-and-wait floor; they measure
round-trip latency, not kernel throughput.

### Fusion and batching, as the smoke suite reports them

```
== Kernel fusion: sin(a).add(b).mul(c)
      unfused: 3 dispatches, 3 allocations, 32.0 MiB moved
      fused:   1 dispatch,   1 allocation,  16.0 MiB moved
      time/op: unfused 2.294 ms, fused 0.949 ms (2.42x)       [DX12]
      time/op: unfused 3.294 ms, fused 0.786 ms (4.19x)       [Vulkan]

== Slab sub-allocation and quarantine
      200 op outputs, 0 create_buffer calls

== Command batching
      100 small ops: unbatched 34.33 ms, batched 1.68 ms (20.4x)
```

The 8N → 4N memory-traffic claim is asserted, not estimated: `kernelBytes` from `EngineStats`
reports exactly 32.0 MiB for the three-kernel chain and 16.0 MiB for the fused one.

## What made the difference

### Workgroup zero-initialisation under FXC (the big one)

D3D12 shader compilation was pathological: 8 s for the transpose kernel, 24 s for the 64×64 GEMM,
**190 s** for the 128×128 GEMM. Neither `force_loop_bounding` nor `ShaderRuntimeChecks` moved the
needle. The cause was WGSL's workgroup-memory zero-initialisation, whose cost scales with the
amount of workgroup storage a kernel declares — and the register-tiled GEMM declares a lot.

Setting `zero_initialize_workgroup_memory: false` and zeroing explicitly where a kernel actually
needs it (`atomicStore(&slots, 0u)` in the subgroup GEMV):

| Kernel | Compile before | after |
| --- | --- | --- |
| transpose | 8.09 s | 0.064 s |
| GEMM 64×64 | 24.1 s | 0.50 s |
| GEMM 128×128 | 190 s | 5.5 s |

Runtime improved too, since the zero-fill ran per workgroup launch: 2048³ went from 351 GFLOP/s to
645 GFLOP/s in the run that measured the fix (today's re-measurement of the same code gives
462 GFLOP/s — see the note on laptop clocks above).

### Batching that was slower than not batching

The first batching implementation was **2.7× slower** than no batching at all: 146.75 ms versus
54.67 ms for 100 small ops. Profiling put the time in `encoder.finish()` — 241–340 ms for 100
passes, about 2.4 ms per compute pass. Beginning a `ComputePass` per dispatch was the entire cost.

The fix is one persistent `ComputePass` held across dispatches inside a batch
(`forget_lifetime()`), ended only when a recorder needs raw encoder commands. 100 ops now cost
~1.7–3 ms, a 20× win over unbatched instead of a 2.7× loss.

### First-matmul latency

Even at 0.5 s a GEMM pipeline compile is visible as a 569.5 ms first call. A background
`hc4j-warmup` thread compiles the matmul pipelines at engine init; the cache is lock-free during
compilation, so a caller that arrives first simply compiles it itself rather than blocking. First
matmul: **569.5 ms → 22.4 ms**. `HC4J_WARMUP=0` opts out.

### Driver allocations

Per-op `create_buffer` was replaced by slab sub-allocation, and the per-dispatch
`create_buffer_init` for kernel parameters by a 256-slot uniform ring with one `write_buffer` per
submission. Verified: 200 consecutive op outputs produce **0** `create_buffer` calls.

### Upload chunking

Chunking every upload into 32 MiB pieces cost **35%** on 40 MiB tensors. Tensors below 256 MiB now
upload in one `write_buffer` call; above that, chunking still bounds wgpu's internal staging
memory, which is the point of it.

### The 67M-element wall

A 1-D dispatch is capped at 65535 workgroups per dimension — ~67M elements at 4 elements per
thread, i.e. a 268 MB f32 tensor, beyond which dispatches were silently invalid. Dispatches now
fold into a balanced 2-D grid and kernels rebuild their linear index from `num_workgroups`. The
smoke suite checks every one of 67.2M elements for `add`, `exp` and `sin`, and asserts it still
took one dispatch per op.

## Measured and rejected

Two optimisations were built, measured, and left out. Both are in the project history rather than
the engine.

- **A double-buffered upload ring.** Steady-state uploads came out *slower* than
  `Queue::write_buffer`: 11.3 ms versus 9.8 ms. Only the very first upload favoured the ring
  (11.4 ms versus 54.8 ms), which is a one-time staging-buffer warm-up that the persistent readback
  ring already amortises for downloads. Not worth a second transfer path.
- **Bind-group caching.** Recording a batched op costs ~15–20 µs end to end, most of which is not
  bind-group creation. Caching would add invalidation complexity (slab regions move, tensors are
  evicted and paged back in) against a few microseconds.

A third idea — 3-D matmul blocking that also splits N — was skipped deliberately. The two existing
degradation steps (row blocks, then row+K blocks) already cover "operand larger than VRAM"; the
N split only matters when a single row of B does not fit, which on any supported device means a
matrix wider than ~500M elements.

## Reproducing

```bash
# Both GPU smoke suites, with the throughput, fusion and batching sections (182 checks)
./gradlew smokeTests

# Same, on Vulkan
HC4J_BACKEND=vulkan ./gradlew smokeTests

# The Rust suite: adapter ranking, memory tiers, naga shader validation, GPU vs CPU (48 tests)
cargo test --release --manifest-path src/main/rust/Cargo.toml

# JMH: 10 benchmarks, 2 warmup + 5 measurement iterations, 1 fork
./gradlew jmh                       # results in build/reports/jmh/results.json
```

For a quick targeted run, build the benchmark jar once and drive JMH directly:

```bash
./gradlew jmhJar
java --enable-native-access=ALL-UNNAMED -jar build/libs/HC4J-0.0.2-jmh.jar -l
java --enable-native-access=ALL-UNNAMED -jar build/libs/HC4J-0.0.2-jmh.jar \
     "ElementwiseBenchmark.(chainFused|chainUnfused)" -wi 1 -i 2 -f 1
```

Benchmarks live in `src/jmh/java/hc4j/bench/`: `ElementwiseBenchmark` (dispatch, in-place, fusion,
batching) and `MatmulBenchmark` (GEMM, both GEMV mappings, transpose; `-p size=512,1024,2048`).
The first `./gradlew jmh` needs network access to fetch `jmh-core` and `jmh-generator-bytecode`
(1.37, pinned in `build.gradle`); everything else builds offline.

## Tuning

| Knob | When to reach for it |
| --- | --- |
| `GpuBatch` | Any loop of small ops. This is the single largest available win. |
| `lazy()` / `eval()` | Elementwise chains — halves memory traffic and removes intermediates. |
| Caller-allocated ops | Hot loops: reuse one output tensor instead of allocating per op. |
| `Tensor.empty` over `zeros` | Outputs a kernel overwrites in full; skips a zero-fill pass. |
| `HC4J_GEMM_TILE=128` | Forces the larger GEMM block; the dispatcher only picks it on a discrete GPU at M,N ≥ 512, so this is how to try it on an integrated one. |
| `HC4J_VRAM_BUDGET_MB` | Lower it to exercise eviction and streaming deliberately; raise it if the device-class default is too conservative. |
| `HC4J_BACKEND` | Compare DX12 against Vulkan on the same machine — the gap is real and adapter-specific. |
| `HC4J_TRACE=1` | Paging, streaming and pipeline-compile diagnostics when a workload is unexpectedly slow. |

Where to look when something is slow: `EngineStats.submissions` (should be far below
`dispatches` under batching), `MemoryStats.evictions` / `pageIns` / `streamedOps` (thrashing, or an
unintentionally small budget), `MemoryStats.dedicatedBuffers` (tensors too large for slabs),
`EngineStats.hostToDeviceBytes` / `deviceToHostBytes` (round trips you did not intend).
