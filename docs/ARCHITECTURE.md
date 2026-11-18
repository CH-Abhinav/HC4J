# HC4J Architecture

How the engine is put together, and why each piece is shaped the way it is. For the user-facing
API see [API.md](API.md); for measurements see [PERFORMANCE.md](PERFORMANCE.md).

```
            Java (src/main/java)                   Rust cdylib (src/main/rust)
┌────────────────────────────────────┐      ┌───────────────────────────────────────┐
│ Tensor   FusedExpr   GpuBatch      │      │ lib.rs    engine, adapter selection,  │
│   │          │          │          │      │           pipeline cache, warm-up     │
│ TrignoOps ExponentialOps           │      │ ops/      elementwise, trigno,        │
│ ArithmeticOps  MatmulOps           │      │           exponential, arithmetic,    │
│   └──────────┬──────────┘          │      │           matmul, fusion              │
│        WgpuBackend                 │      │ memory/   manager, budget, lru, slab, │
│   (MethodHandle downcalls,         │ FFM  │           spill, transfer             │
│    Arena for off-heap params)      ├─────▶│ stream.rs single submission point     │
└────────────────────────────────────┘ u64  │ error.rs  status codes, panic guard   │
                                     handles└───────────────────────────────────────┘
                                                              │ wgpu v30
                                                   DX12 / Vulkan / Metal
```

Java never sees a pointer to GPU memory. A tensor is an opaque `u64` handle minted by the Rust
registry; the Rust side owns the buffer, its tier, its pins and its lifetime. That one decision is
what makes eviction, spilling, slab sub-allocation and streaming possible without the Java side
knowing any of it happened.

## 1. The FFI boundary

### Calling convention

Every op is **caller-allocated**: the output tensor's handle is an argument (`id_out`), never a
return value. This is what makes `a.sin(a)` and `a.add(b, a)` legal, lets Java reuse an output
buffer across a loop, and keeps allocation policy on the Java side where the `Tensor` lifecycle
lives. `Tensor` additionally offers convenience overloads that allocate the result, but they are
thin wrappers over the same entry points.

Four ABI families, all returning `int32_t` status:

```c
// Elementwise unary — 12 trig + 5 exponential ops
int32_t dispatch_<op>_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
        const uint32_t* shape, const uint32_t* strides_a, const uint32_t* strides_c,
        size_t length, uint32_t is_contiguous);

// Elementwise binary — add/sub/mul/div, dtype 0 = i32, 1 = f32
int32_t dispatch_<op>(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
        const uint32_t* shape, const uint32_t* strides_a, const uint32_t* strides_b,
        const uint32_t* strides_c, size_t length, uint32_t is_contiguous, uint32_t dtype);
int32_t dispatch_<op>_f32(/* as above, without dtype */);

// Linear algebra
int32_t dispatch_matmul_f32(uint64_t id_a, uint64_t id_b, uint64_t id_out,
        uint32_t m, uint32_t n, uint32_t k);
int32_t dispatch_matmul_f32_ex(/* as above */, uint32_t flags);   // 1 = A^T, 2 = B^T
int32_t dispatch_transpose_f32(uint64_t id_in, uint64_t id_out, uint32_t rows, uint32_t cols);
int32_t hc4j_matmul_plan(uint32_t m, uint32_t n, uint32_t k, uint32_t flags);  // diagnostic

// Fusion
int32_t hc4j_fused_dispatch(const uint32_t* program, uint32_t program_len,
        const uint64_t* inputs, uint32_t n_inputs, uint64_t id_out);

// Activations — the unary ABI plus an op code and an f32 parameter as raw bits
int32_t dispatch_activation_f32(/* as the unary ABI */, uint32_t act, uint32_t alpha_bits);

// Reductions — op: 0 sum, 1 mean, 2 max, 3 min
int32_t dispatch_reduce_f32(uint64_t id_a, uint64_t id_out, uint64_t length, uint32_t op);
int32_t dispatch_reduce_square_f32(/* as above */);                     // folds x*x
int32_t dispatch_reduce_diff2_f32(uint64_t id_a, uint64_t id_b, uint64_t id_out,
        uint64_t length, uint32_t op);                                  // folds (a-b)^2
int32_t dispatch_reduce_rows_f32(uint64_t id_a, uint64_t id_out,
        uint32_t rows, uint32_t cols, uint32_t op);
int32_t dispatch_argmax_rows(uint64_t id_a, uint64_t id_out,
        uint32_t rows, uint32_t cols, uint32_t largest);                // i32 output

// Neural-network row ops — a 0 handle means "absent" for gamma/beta
int32_t dispatch_softmax_rows_f32(uint64_t id_a, uint64_t id_out,
        uint32_t rows, uint32_t cols, uint32_t log_variant);
int32_t dispatch_layernorm_rows_f32(uint64_t id_a, uint64_t id_gamma, uint64_t id_beta,
        uint64_t id_out, uint32_t rows, uint32_t cols, uint32_t eps_bits);
int32_t dispatch_rmsnorm_rows_f32(uint64_t id_a, uint64_t id_gamma, uint64_t id_out,
        uint32_t rows, uint32_t cols, uint32_t eps_bits);
int32_t dispatch_cross_entropy_rows(uint64_t id_logits, uint64_t id_targets,
        uint64_t id_out, uint32_t rows, uint32_t cols);                 // i32 targets
int32_t dispatch_add_row_bias_f32(uint64_t id_a, uint64_t id_bias, uint64_t id_out,
        uint32_t rows, uint32_t cols);
```

Float parameters cross as **raw bits in a `uint32_t`** (`alpha_bits`, `eps_bits`) rather than as a
float argument, so the whole ABI is integers and handles — no float registers, no calling-convention
differences to get wrong between Panama and the platform ABI.

Contiguous calls pass null layout pointers and `is_contiguous = 1`, so the common case costs no
off-heap marshalling at all. Strided calls pass three (or four) `rank`-length `u32` arrays
describing a view, which is how a transposed tensor can be an operand without being copied.

Plus memory and engine control: `hc4j_gpu_alloc`, `hc4j_gpu_alloc_uninit`, `hc4j_gpu_write`,
`hc4j_gpu_download`, `hc4j_gpu_free`, `hc4j_mem_configure`, `hc4j_mem_stats`, `hc4j_mem_residency`,
`hc4j_mem_evict`, `hc4j_init_gpu`, `hc4j_batch_begin`, `hc4j_batch_end`, `hc4j_synchronize`,
`hc4j_engine_stats` — and the eleven `hc4j_collective_*` entry points of section 6. **66 exported
symbols** in total.

### Java side

`WgpuBackend` loads the library with `System.load` (walking up from the working directory to
`src/main/rust/target/release/`), then binds every symbol once into `static final MethodHandle`
constants via `Linker.nativeLinker().downcallHandle(SymbolLookup.loaderLookup().find(...), ...)`.
Binding in a static initialiser means a missing symbol fails loudly at class-load time rather than
at the first dispatch, and `invokeExact` on a static final handle is intrinsified by the JIT.

Off-heap scratch — stride arrays, the fused program, the `EngineStats`/`MemStats` structs — is
allocated in a `try (Arena arena = Arena.ofConfined())` block, so it is freed deterministically on
the calling thread and cannot be touched after the call. Native structs are mirrored as Java
records read field-by-field from a `MemorySegment`; the field count is asserted against the struct
layout, because a silent field-order mismatch would corrupt every counter.

### Errors never cross as panics

A Rust panic unwinding across `extern "C"` aborts the JVM. Every entry point therefore wraps its
body in `ffi_guard`, which `catch_unwind`s and maps the outcome to a status code:

| Code | Meaning | Java exception |
| --- | --- | --- |
| `0` | success | — |
| `-1` | handle not in the registry | `IllegalStateException` |
| `-2` | invalid parameters | `IllegalArgumentException` |
| `-3` | readback / mapping failure | `RuntimeException` |
| `-4` | every tier exhausted | `GpuOutOfMemoryException` |
| `-5` | device lost, timed out, or a contained panic | `IllegalStateException` |
| `-6` | wgpu rejected the shader, pipeline or submission | `IllegalStateException` |
| `-7` | spill-file I/O failure | `RuntimeException` |
| `-8` | unsupported layout or size | `UnsupportedOperationException` |

wgpu's default uncaptured-error handler panics, so the engine installs its own and runs every
fallible GPU call inside a scoped `push_error_scope` / `pop_error_scope` pair (an RAII
`ErrorScopeGuard`, popped in reverse order — wgpu keeps them in a thread-local stack). Validation
errors become `-6`; `OutOfMemory` from the driver becomes `-4` *after* the memory manager has tried
to make room. `Hc4jError::with_context` attaches a description when an error surfaces from a batch,
so "a batch of 37 ops: elementwise, matmul (register 64), …" is reported rather than a bare code.

## 2. Tiered memory

wgpu has no page faults: an allocation that does not fit either fails or, on WDDM, silently
oversubscribes into system memory at PCIe speed. So HC4J manages residency itself. Every tensor's
bytes live in **exactly one** tier:

```
   DEVICE ──evict──▶ HOST ──spill──▶ DISK
      ◀───page in────  ◀──────────────
```

- **Budgets.** VRAM and host RAM each have a lock-free atomic `Budget`; every byte is held by an
  RAII `Reservation`, so accounting cannot drift even on error paths. wgpu cannot portably query
  free VRAM, so the starting budget is `HC4J_VRAM_BUDGET_MB` or a device-class default (8 GiB
  discrete, 2 GiB integrated, 1 GiB otherwise). An optimistic default is safe because the first
  real driver OOM clamps the budget to what is actually committed and eviction resumes — the
  software budget converges on the hardware.
- **Eviction.** When VRAM is exhausted the least recently used *unpinned* device tensor is copied
  out through the persistent readback ring, one tensor at a time until the request fits. Recency is
  a `BTreeMap<tick, id>`, so the LRU victim is a `first_key_value` lookup.
- **Spilling.** If the host budget is also exhausted, bytes go to an anonymous, self-deleting file:
  Windows opens it with `FILE_FLAG_DELETE_ON_CLOSE`, Unix unlinks it immediately after creation.
  Spilled bytes can never outlive the process, even if the JVM is killed. All I/O is positional, so
  concurrent readers share one handle without a cursor.
- **Page-in.** An op whose operand is off-device pages it back in (and pins it) before dispatch.
- **Streaming.** If the operands cannot be made co-resident at all — a single tensor larger than
  the whole budget — the op runs tile by tile over transient scratch buffers instead of failing.
  Elementwise ops stream in both directions; matmul has two degradation steps (§5).

**Concurrency.** The table lock is never held across a GPU wait or disk I/O. A tensor being
migrated is marked `InTransit`; other threads block on a condvar until it settles. Freeing a tensor
that is pinned by in-flight work is deferred to its last unpin, so a recording thread can never
have a span pulled out from under it. Readers of a migrating tensor take a copy-on-write snapshot
(`Arc<Vec<u8>>` for host, `Arc<File>` for disk) and read it outside the lock.

## 3. Slab sub-allocation

`create_buffer` is a driver allocation — hundreds of microseconds, and it fragments the driver's
heap. A chain of elementwise ops would pay it per intermediate. So tensors up to **a quarter of a
slab** become aligned sub-ranges of a few large buffers; only larger tensors get dedicated buffers.
Slab size is `vram_budget / 8` clamped to [4 MiB, 64 MiB], and each slab holds a `Reservation` for
its *full* capacity, so the budget keeps tracking physical VRAM rather than live tensor bytes.

Sub-allocation is a best-fit `RangeAllocator` over a free list with coalescing on release.

Two invariants make this safe:

- **Epoch quarantine.** A freed region is not reusable until the submission epoch that was current
  when it was freed has retired. A Java `close()` immediately after a dispatch therefore cannot
  hand memory the GPU is still reading to a new tensor. Retirement is published by
  `Queue::on_submitted_work_done` and read lock-free (an atomic `submitted` counter plus a pending
  list), so the fast path never takes a lock.
- **Everything is `read_write`.** wgpu rejects a bind group where the same buffer appears both
  read-only and read-write in one dispatch (`any_exclusive && !bits.is_power_of_two()`). Two
  tensors from the same slab *are* the same buffer, so every storage binding in every kernel is
  declared `read_write`. This is also what makes in-place ops legal.

Bindings use `BufferBinding { buffer, offset, size }`, so a kernel sees exactly its region and
indexes from zero.

## 4. The command stream

Every `queue.submit` in HC4J goes through one `CommandStream`, which buys four properties:

- **Batching.** Inside a batch scope, ops record into one shared command encoder and *one shared
  `ComputePass`*, submitted at batch end (or every 256 passes). Beginning a pass per dispatch is
  what made an early batching attempt slower than no batching at all (see
  [PERFORMANCE.md](PERFORMANCE.md#batching-that-was-slower-than-not-batching)): `encoder.finish()`
  cost ~2.4 ms per pass. A recorder that needs raw encoder commands (a buffer copy) ends the open
  pass first, so pass and encoder commands stay correctly ordered.
- **Epochs.** Each submission gets a monotonically increasing epoch; the slab allocator quarantines
  against it.
- **Ordered zero-fills.** A zeroed slab region's `clear_buffer` is encoded *before* the next
  recorded op and flushed before any `queue.write_buffer`, so a clear can never land on top of
  data that was written after it was requested.
- **Uniform ring.** Per-dispatch kernel parameters are packed into 256-byte slots of one
  persistent uniform buffer (256 slots) and uploaded with a single `write_buffer` per submission,
  replacing a `create_buffer_init` — i.e. a driver allocation — per dispatch.

Epoch numbers are mirrored outside the lock (an atomic `submitted` plus a `pending` flag) so
`retire_epoch` can run from a `Drop` impl in any context. That mirror may over-report by one, and
the quarantine depends on it never naming an epoch that cannot arrive: `pending` is therefore
retracted the moment nothing is left to submit, including when a queued zero-fill is *cancelled*
(an allocation the next kernel fully overwrites). Before that, a cancelled clear could leave
`pending` set with an empty encoder, so a region freed just afterwards was quarantined against an
epoch no submission would ever claim — and the region could never be reclaimed. The symptom
appeared nowhere near the cause: `configure` failing with "epoch N was never submitted" while
lowering a budget.

Transfers also route through it: uploads use `Queue::write_buffer` so they are ordered after
recorded work, with at most 2 chunks of 32 MiB in flight to bound wgpu's internal staging memory,
except that tensors under 256 MiB upload in one shot (chunking them cost 35%). Downloads use a
persistent double-buffered ring of `MAP_READ` staging buffers: while the host drains slot *k* the
GPU is already copying into slot *k+1*.

## 5. Kernels

### Elementwise

One WGSL template per family, specialised by splicing the operation string, then cached. Each
thread handles 4 elements; workgroups are 256 threads.

**Grid folding.** A 1-D dispatch tops out at 65535 workgroups (some drivers report 65536), i.e.
~67M elements at 4 elements/thread — a 268 MB tensor. Dispatches therefore fold into a balanced
X×Y grid and kernels rebuild the linear id from `num_workgroups` rather than a hardcoded stride,
so the same kernel is correct under either grid shape and no tensor size can hit the limit.

**Strided views.** Non-contiguous operands pass shape and strides as `array<vec4<u32>, 2>` —
WGSL's uniform alignment rules make a naked `array<u32, 8>` stride 16 bytes per element, so the
layout is packed into two `vec4`s and unpacked in the shader. Trailing unit dimensions are pruned
in reverse so the index math is as short as the view allows, and the loop is branchless.

### Matmul

The dispatcher is a pure function of shape, transpose flags and device capabilities (unit-tested
without a GPU), choosing among five kernels:

| Shape | Kernel | Notes |
| --- | --- | --- |
| `N == 1` (matrix·vector) | `GEMV_ROWS` | 32 lanes per row, 256 when K ≥ 1024; `vec4` loads when K % 4 == 0 |
| `N == 1`, subgroups available, 256 lanes | `GEMV_ROWS_SUBGROUP` | `subgroupAdd` reduction, atomic slot claiming |
| `M == 1` (vector·matrix) | `GEMV_COLS` | one thread per column, 4/16/64 columns per workgroup |
| `min(M,N) ≥ 64`, `K ≥ 64` | `REGISTER_64_*` | 64×64 block, 4×4 per thread |
| …and discrete, `M,N ≥ 512` | `REGISTER_128_*` | 128×128 block, 8×8 per thread |
| anything else | `TILED_16` | 16×16 shared-memory tiles, handles tiny and ragged shapes |

The register-tiled kernels keep a `TM×TN` accumulator in registers, stage A k-major with one
column of padding (32×33-style, to break shared-memory bank conflicts), load B as `vec4` when
`K % 4 == 0 && N % 4 == 0`, unroll the K loop, and use `fma`. All of them take a `beta` flag so a
result can be accumulated into rather than overwritten — which is what makes K-blocking possible.

A **stored-transposed operand** is pre-transposed in VRAM by a 32×33 padded-tile kernel rather than
handled with a strided read, because the transposed access pattern costs more than the copy. For
GEMV, transposition is free: the row and column mappings read the same storage, so `A^T · x` simply
selects the other kernel.

**Out of core.** Two degradation steps, both using `beta`:

1. *Streamed* — B fits in VRAM, A does not: process A in row blocks, K whole.
2. *Blocked* — B does not fit either: block over rows **and** K, gathering A row by row and B in
   contiguous row chunks, accumulating with `beta = 1` for every K block after the first.

Neither path supports stored-transposed operands (`-8 UNSUPPORTED`): materialise with `transpose`.

### Reductions

The primitive the ML layer is built on, in two shapes.

**Full reduce** is always **exactly two dispatches**, whatever the input size. Pass 1 launches a
*bounded* grid — at most `MAX_PARTIALS` (1024) workgroups — and each workgroup walks its slice with
a grid-stride loop, reduces it in workgroup memory, and writes one partial. Pass 2 is a single
workgroup that strides over the partials. So the scratch buffer is at most 4 KiB regardless of
input size, and there is no recursion and no log-many passes. Inputs larger than one storage
binding are tiled, each tile contributing its own partials.

**Row reduce** (`[rows, cols] -> [rows]`) is one dispatch: one workgroup per row, striding over the
row and reducing in workgroup memory. Row indices come from a folded grid, so the row count is not
capped at 65535. Workgroups are 256 threads regardless of `cols`, which wastes lanes on narrow rows
— a lanes-per-row mapping like the GEMV kernel's would fix that, and has not been needed yet.

Three design points are worth naming:

- **The identities are finite**, `±f32::MAX` rather than `±inf`, because the pairwise combines
  (softmax's and Welford's) subtract two accumulators and `inf - inf` is NaN.
- **Load variants** let a reduction fold a cheap expression in instead of materialising it:
  `A[i]`, `A[i]*A[i]`, and `(A[i]-B[i])^2`. An L2 norm and an MSE are therefore one kernel with no
  intermediate tensor.
- **Mean divides in the final pass only**, guarded on a non-zero count in the uniform, so pass 1
  and pass 2 share one kernel.

The barriers sit in workgroup-uniform control flow in both loops: the tree loop's bound does not
depend on the invocation, the guard is inside the loop body and the barrier outside it, and the
early `row >= rows` return is taken by a whole workgroup at once.

### Neural-network ops

**Activations** are ordinary elementwise ops and reuse the unary executor, so they inherit tiling,
grid folding, in-place output and the streaming fallback. They are not WGSL built-ins, and WGSL has
neither overloading nor generics, so the unary template gained a *prelude* slot and separate
`vec4`/scalar callables. Each activation is written **once**, with `__T__` standing for the width,
and both functions are generated from that one body — the two paths cannot drift apart. A parameter
(a leaky-ReLU slope, an ELU alpha) rides in `dims.params.z` as raw f32 bits and is read back with
`bitcast`, so no new uniform layout was needed.

Numerics are the reason these are kernels and not one-liners:

- `sigmoid`, `silu` and `softplus` are written so every `exp` argument is non-positive
  (`e = exp(-|x|)`, then select), which keeps them finite for large `|x|` where the naive forms
  overflow.
- `gelu` (the tanh approximation) **clamps its cubic argument to ±10** before `tanh`. Unclamped, the
  argument reaches ~24000 at x = 88, and a driver that evaluates `tanh` as `(e^2a-1)/(e^2a+1)`
  returns `inf/inf` = NaN. This was a real failure, caught by comparing against an f64 reference at
  the saturating ends rather than only on random input in [-1, 1].
- `relu` uses `max(x, 0)`; leaky ReLU uses `select`, not `max`, because a slope above 1 would make
  `max` wrong.

**Row-wise ops** run one workgroup per row: a reduction over the row in workgroup memory, then a
write pass over the same row, as a single dispatch.

| Op | Statistics phase | Reads per row |
| --- | --- | --- |
| softmax, log-softmax | online `(max, sum exp)` recurrence | 2 |
| layer norm | Welford `(n, mean, M2)` triples, combined pairwise | 2 |
| RMS norm | sum of squares | 2 |
| cross entropy | online `(max, sum exp)`, then a gather at the target | 1 |

The **online softmax** recurrence (Milakov & Gimelshein) combines `(m1,s1)` and `(m2,s2)` as
`m = max(m1,m2)`, `s = s1·exp(m1-m) + s2·exp(m2-m)`, which folds the maximum and the sum into one
pass — a row is read twice rather than three times, and no `exp` ever sees a positive argument.
**Welford** accumulation gives layer norm its variance without `E[x²] - E[x]²`; a thread that saw
no elements contributes `(0,0,0)`, which the combine absorbs.

Cross entropy cannot report an error from the GPU, so an out-of-range class index writes **NaN**
for that row instead of reading a neighbouring one — the value is the signal, and the host contract
says so.

**Bias broadcast** (`out[r,c] = a[r,c] + bias[c]`) is its own flat kernel because fusion requires
identical shapes and cannot express it, and because a GEMM followed by a bias add is the shape of
every linear layer.

Affine variants are separate pipelines rather than one kernel with a flag: a bind-group layout is
derived from the shader, so "with gamma and beta" and "without" are different layouts. They are
cached under different names, which is all the pipeline cache needs.

### Fusion

`FusedExpr` builds an immutable expression tree on the Java side and lowers it to a **postfix u32
program** (`opcode << 24 | operand`, a `CONST` followed by one word of f32 bits) with an iterative
post-order walk, so a long chain cannot overflow the Java stack. Inputs are deduplicated by
identity, so `x.mul(x)` binds one buffer.

The Rust side validates the program (stack depth ≤ 64, ≤ 1024 words, operand bounds) and lowers it
to SSA WGSL with **local value numbering** and commutative canonicalisation, so a repeated
subexpression is computed once. Constants are emitted via `bitcast` to preserve exact bits. The
compiled pipeline is cached keyed on the exact program, so an expression shape reused in a loop
compiles once.

`sin(a) + b) * c` becomes one dispatch moving 4N floats (3 reads, 1 write) instead of three
dispatches moving 8N, with one allocation instead of three.

Activations fuse too (`relu`, `sigmoid`, `gelu`, `silu`, `softplus`). They are not built-ins, so the
generator emits each called activation's `vec4` helper — taken from the same `Activation` body the
standalone kernels use — and **only** the ones the program calls, since an unused function still
costs shader-compile time under D3D12. Which activations are called is collected during lowering,
not by scanning the program: a `CONST` payload is raw f32 bits whose top byte can look like any
opcode. The parameterised activations stay out, because a fused program has nowhere to carry a
per-op alpha.


## 6. Collectives

`collective.rs` adds a second, independent execution context: a ring of **ranks**, each with its
own `wgpu::Device`, queue and reduction pipeline, plus a topology probe and ring AllReduce.

```
        rank 0                 rank 1                 rank 2
   ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
   │ Device+Queue │       │ Device+Queue │       │ Device+Queue │
   │ add pipeline │       │ add pipeline │       │ add pipeline │
   │ data, recv   │       │ data, recv   │       │ data, recv   │
   └──────┬───────┘       └──────┬───────┘       └──────┬───────┘
          │  GPU→host→GPU        │  GPU→host→GPU        │
          └──────────────────────┴──────────────────────┘ (ring)
```

### Why it is separate from the engine

A rank is deliberately **not** a `GpuEngine`. The engine is a process-wide singleton with a tiered
memory manager, a command stream and an epoch-based slab allocator, all keyed to one device;
making it multi-device would mean threading a device index through every op and every allocation.
The collective instead owns its devices and buffers outright, which keeps the verified
single-device engine untouched. The cost is explicit: collective buffers are not `Tensor`s, so data
crosses through `write`/`read`. A zero-copy bridge for rank 0 is separate work.

What *is* shared: the backend selection (`HC4J_BACKEND`), the adapter and backend ranking functions,
and the convention that every storage binding is `read_write`.

### Picking devices for ranks

Two filters, both of which a naive implementation gets wrong — and both of which the probe caught:

- **One physical GPU can enumerate twice.** With DX12 and Vulkan both enabled, the same card
  appears once per backend. Ranks are therefore assigned over adapters deduplicated by
  `(vendor, device, name)`, keeping the better-ranked backend. Without this, a 2-rank ring on a
  single-GPU machine put rank 0 on Vulkan and rank 1 on DX12 and called them two devices.
- **Software adapters are excluded.** Windows enumerates WARP ("Microsoft Basic Render Driver",
  `DeviceType::Cpu`) beside the real GPU. A ring that placed a rank there measured 0.01 GB/s on
  that hop, and a 1 Mi-element AllReduce took 864 ms instead of 104 ms. They are used only when
  nothing else exists.

Remaining devices are sorted best-first, then ranks are assigned round-robin, so extra ranks beyond
the GPU count reuse a GPU — **loopback mode**, which exists so the whole path is exercisable on a
single-GPU machine. Loopback is reported per link, because an intra-GPU copy must not be read as a
PCIe measurement.

### Transport

Every hop is host-staged: the source copies its chunk into a persistent `MAP_READ` buffer, the host
maps and reads it, and the destination uploads it. wgpu has no external-memory export/import, so
`dma_buf` / `OPAQUE_WIN32` handle passing — the thing that makes real P2P fast — is not reachable
from this stack; it needs the raw `VkDevice` through `wgpu-hal`. The `Transport` enum already
carries the distinction so the Java API and the reports do not have to change when it lands.

Staging buffers are allocated once per AllReduce rather than once per hop, which is the same idea
as the engine's readback ring; the ring itself cannot be reused because it belongs to the engine's
device, not to a rank.

Hops are synchronous — a transfer completes before the reduction consuming it is recorded. Correct,
and dominated by round-trip latency rather than bandwidth. Pipelining the rounds is the obvious
next step.

The reference design orchestrates its staged transfers with Vulkan **timeline semaphores**, so the
CPU never blocks waiting for a stage to land. wgpu exposes no timeline-semaphore API, so each hop
instead blocks on `Device::poll(PollType::Wait)`. That is the second thing, after external memory,
that a `wgpu-hal` transport would recover — and it is why the hop cost here is a round trip rather
than a pipeline stage.

### The ring

`n - 1` rounds of scatter-reduce followed by `n - 1` rounds of allgather, over `n` chunks:

| Phase | Round `r`, rank `i` | Effect |
| --- | --- | --- |
| Scatter-reduce | sends chunk `(i - r) mod n` to `i + 1` | the receiver *adds* it into its own copy |
| Allgather | sends chunk `(i + 1 - r) mod n` to `i + 1` | the receiver *overwrites* its copy |

The index arithmetic has a property worth stating, because it is easy to assert incorrectly: each
rank receives exactly **one** add per chunk, for every chunk but its own — a chunk accumulates as it
travels rather than piling up on one rank. The chunk a rank receives in the *last* scatter-reduce
round is the one that has now seen every contribution, and that is where the allgather begins
(rank `r` starts holding chunk `r + 1`). Both properties are unit-tested for `n = 2..8` without a
GPU, alongside an end-to-end check that every rank ends up with the exact sum.

The reduction kernel is 256 threads wide with a folded 2-D grid, matching the engine's elementwise
convention; the reference design used one invocation per workgroup.

## 7. Engine startup

1. `requested_backends()` — `HC4J_BACKEND` narrows the set; the default on Windows is DX12 + Vulkan.
2. `enumerate_adapters` and rank: **discrete first**, then by backend (DX12 preferred on Windows),
   ties keeping enumeration order. `HC4J_ADAPTER=<substring>` pins a GPU by name.
3. If no discrete GPU exists, fall back to `PowerPreference::HighPerformance` — but the preference
   only picks the *GPU* (matched by vendor/device/type); the backend is re-ranked afterwards,
   because the preference happily returns a Vulkan adapter on Windows where DX12 is wanted.
4. Request a device with the adapter's own limits (the default 128 MiB storage-binding cap would
   make large tensors undispatchable), probe `SUBGROUP` and `SHADER_F16`, and record
   `subgroup_min_size`/`max_size` in `GpuEngine`.
5. Install the uncaptured-error handler and the device-lost callback, log the selection, and spawn
   a background `hc4j-warmup` thread that compiles the expensive matmul pipelines. The pipeline
   cache is lock-free during compilation, so a caller that gets there first just compiles it
   itself. `HC4J_WARMUP=0` opts out.

Pipelines are cached under `"{shader_name}::{entry_point}"` — keying on the entry point alone is a
real bug, because every unary kernel's entry point is `main`, so `cos` would run `sin`'s pipeline.
Compilation sets `zero_initialize_workgroup_memory: false`: under D3D12's FXC, zero-initialising
workgroup memory scaled catastrophically with its size (190 s for the 128×128 GEMM), so kernels
that need zeroed workgroup storage zero it explicitly.

## 8. Invariants worth keeping

- Java holds handles, never pointers. Only the Rust registry knows where the bytes are.
- One submission point. Anything that submits outside `CommandStream` breaks epochs, and therefore
  breaks quarantine.
- Every storage binding is `read_write`. Slab sharing and in-place ops both depend on it.
- Every entry point is panic-guarded, and every fallible GPU call runs inside an error scope.
- The table lock is never held across a GPU wait or I/O; `InTransit` + condvar covers the gap.
- Native struct layouts are mirrored with an asserted field count on the Java side.
- Kernels rebuild their linear index from `num_workgroups`, never from a hardcoded grid width.
- The matmul dispatcher stays pure and unit-tested; capability probing happens in `Caps::of`.
- `retire_epoch()` only ever names an epoch `wait_epoch()` can satisfy. It reads the lock-free
  `pending` mirror, so it may name one epoch past the last submission; that is safe **only** because
  `pending` is retracted whenever nothing is left to submit (a cancelled zero-fill, an empty flush)
  and `wait_epoch` treats the one-past case as satisfied by the last submission. Break either half
  and a slab region quarantined against a phantom epoch can never be reclaimed, which surfaces far
  away as `configure` failing with "epoch N was never submitted".
- Reduction identities are finite (`±f32::MAX`, not `±inf`): the pairwise combines subtract two
  accumulators.
- `workgroupBarrier()` stays in workgroup-uniform control flow. Loop bounds must not depend on the
  invocation, guards go inside the loop body with the barrier outside it, and an early return must
  be taken by the whole workgroup (so its condition may depend only on `workgroup_id`).
- WGSL has no shadowing and a long reserved-word list. `meta` and `target` both got through review
  and failed at shader-compile time; the naga validation tests exist to catch the next one.
- An adapter list is not a device list. Deduplicate by physical identity before treating entries as
  separate GPUs, and exclude `DeviceType::Cpu`: both mistakes produce a working-but-wrong topology
  that only a bandwidth number exposes.
- A collective owns its devices and buffers; it must not reach into the engine's memory manager or
  command stream. Anything that blurs that line pulls multi-device concerns into the single-device
  engine, which is the refactor this design exists to avoid.
- The ring schedule is unit-tested as pure index arithmetic for `n = 2..8`. A collective bug that
  only shows up as a wrong sum on four GPUs is not something to debug on hardware.
