# HC4J Java API

Everything a caller needs. Package `hc4j` for tensors and expressions, `hc4j.engine` for engine
control and statistics, `hc4j.ops` for the operation entry points `Tensor` delegates to.

Launch with `--enable-native-access=ALL-UNNAMED` (the Gradle tasks add it). The engine initialises
itself on first use and logs the adapter it selected:

```
[HC4J] GPU: Intel(R) Iris(R) Xe Graphics | IntegratedGpu | Dx12 | driver: 31.0.101.4502 | selected by: high-performance preference (no discrete GPU found)
[HC4J] features: subgroups=false (size 16..16) shader_f16=false | workgroups/dim=65535 storage-binding=2047 MiB
```

## Tensor lifecycle

A `Tensor` is a handle to VRAM-resident data and is `AutoCloseable`. Use try-with-resources;
`close()` is idempotent, and a tensor that is never closed leaks GPU memory for the life of the
process.

```java
Tensor.zeros(DType dtype, int... shape)    // allocated and zero-filled
Tensor.empty(DType dtype, int... shape)    // allocated, contents unspecified
Tensor.fromArray(float[] values, int... shape)   // uploads; values.length must equal the shape product
Tensor.fromArray(int[] values, int... shape)

float[] toFloatArray()      // f32 only, else IllegalStateException
int[]   toIntArray()        // i32 only

DType getDType();  long getSize();  int dim();  String toString();
```

`DType` is `f32` or `i32`. Shapes are `int...`; rank is unlimited for elementwise ops, 1–2 for
matmul. `empty` skips the zero-fill pass, so prefer it for outputs a kernel overwrites in full.

Residency is managed for you, but visible:

```java
WgpuBackend.Residency residency()   // DEVICE, HOST or DISK
void evict()                        // page out of VRAM; ops page it back in on demand
```

## Two calling conventions

Every op exists twice. The **caller-allocated** form takes the result tensor and returns it; the
**convenience** form allocates a fresh result:

```java
try (Tensor out = Tensor.empty(DType.f32, n)) {
    a.sin(out);              // no allocation
    a.add(b, out);
}
try (Tensor y = a.sin()) { }  // allocates y
```

Prefer the caller-allocated form in loops: it is the native convention, so it costs no allocation
and no eviction pressure.

**In place is legal.** The output may be an input — `a.sin(a)`, `a.add(b, a)`, `b.add(a, a)` —
because every element is read before it is written and all storage bindings are declared
`read_write`. The one exception is matmul and transpose, where the result must not alias an
operand (each output element reads many input elements).

### Elementwise ops

| Family | Methods |
| --- | --- |
| Trigonometric | `sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh` |
| Exponential | `exp log log2 log10 sqrt` |
| Arithmetic | `add sub mul div` (f32 and i32) |

Each in both forms: `a.sin()` and `a.sin(out)`, `a.add(b)` and `a.add(b, out)`.

Operands must have **identical shapes** — there is no broadcasting. Non-contiguous operands (a
transposed view) are handled by the strided path without a copy.

Precision follows WGSL, which guarantees only 2⁻¹¹ absolute error for `sin`/`cos`, and the inverse
functions inherit `atan2`'s (~4096 ULP). Measured error on Intel UHD/Iris is ~2.3e-5 for `sin`.
Tolerance your comparisons accordingly; `StrictMath` agreement is not a reasonable expectation.

## Matmul

```java
Tensor matmul(Tensor other)              // allocates the result
Tensor matmul(Tensor other, Tensor res)  // into res
MatmulOps.matmul(a, b, res, transposeA, transposeB)
MatmulOps.dims(a, b, transposeA, transposeB)   // validates and returns (m, n, k, resultShape)
MatmulOps.plan(m, n, k, transposeA, transposeB) // which kernel would run (diagnostic)

Tensor transpose()                       // 2-D, in VRAM
Tensor transpose(Tensor res)
MatmulOps.transpose(in, res)
```

Rank 1 and 2 operands, with NumPy's conventions:

| A | B | Result | Operation |
| --- | --- | --- | --- |
| `[m,k]` | `[k,n]` | `[m,n]` | GEMM |
| `[m,k]` | `[k]` | `[m]` | matrix · vector |
| `[k]` | `[k,n]` | `[n]` | vector · matrix |
| `[k]` | `[k]` | `[1]` | dot product |

`A.cols` must equal `B.rows` (after the transpose flags), `res` must be f32 with exactly the shape
`dims` reports, and `res` must not be `a` or `b`. Everything else is an
`IllegalArgumentException`.

The `transposeA` / `transposeB` flags mean "this operand is stored transposed" — the dispatcher
either pre-transposes it in VRAM (GEMM) or selects the mirrored kernel (GEMV), rather than making
you copy it. Note the one gap: when an operand is **larger than the VRAM budget**, the out-of-core
paths reject transpose flags with `UnsupportedOperationException`; materialise with `transpose`
first.

Error growth is the usual f32 GEMM bound, `k · u · Σ|a·b|`, so compare against a CPU reference
with a K-scaled tolerance, not a fixed epsilon.


## Neural-network ops

`hc4j.ops.NnOps`, with fluent forms on `Tensor`. Activations are ordinary elementwise ops; the rest
work along the **last axis**.

### The last-axis convention

Softmax, the normalisations, cross entropy and the `*Rows` reductions treat a tensor of any rank as
`[rows, cols]` with `cols` its last dimension and `rows` the product of the rest — NumPy's
`axis=-1`:

| Input | Rows × cols | Shape-preserving result | Reducing result |
| --- | --- | --- | --- |
| `[128]` | 1 × 128 | `[128]` | `[1]` |
| `[32, 128]` | 32 × 128 | `[32, 128]` | `[32]` |
| `[4, 8, 128]` | 32 × 128 | `[4, 8, 128]` | `[4, 8]` |

`NnOps.rows(t)` returns that split, and `Rows.reducedShape(t)` the reduced shape.

These kernels index their operands linearly, so a **strided view is rejected** rather than silently
read in the wrong order — materialise it first (`transpose` for a 2-D view). The operand must also
fit one storage binding (2047 MiB on this adapter) and be VRAM-resident.

Residency, by family: **activations** inherit the full elementwise tiering, streaming included, so
they work on tensors larger than VRAM. **Reductions** tile over the binding limit but need their
operands resident. **Row-wise ops** need one contiguous binding. Where residency cannot be reached,
the call raises `GpuOutOfMemoryException` instead of degrading — the elementwise fallback does not
extend to these.

### Activations

```java
a.relu()      a.relu(out)        // and gelu, silu, sigmoid, softplus
a.leakyRelu(0.01f)               a.leakyRelu(0.01f, out)
a.elu(1.0f)                      a.elu(1.0f, out)
NnOps.activation(NnOps.Activation.GELU, in, out, alpha);
```

Elementwise, so they inherit everything the elementwise path gives: strided inputs, in-place output
(`a.relu(a)`), tiling past the binding limit, and streaming for tensors larger than VRAM.

GELU is the **tanh approximation** (what GPT-2 and BERT use), not the erf form — WGSL has no `erf`.
Its cubic argument is clamped to ±10 before `tanh`, because a driver that evaluates `tanh` as
`(e^2a − 1)/(e^2a + 1)` returns `inf/inf` = NaN beyond that; sigmoid, SiLU and softplus are written
so every `exp` argument is non-positive, which keeps them finite for large `|x|`.

`relu`, `sigmoid`, `gelu`, `silu` and `softplus` are also available **inside a fused expression**:

```java
try (Tensor r = x.lazy().gelu().mul(2f).add(b).eval()) { }   // one kernel
```

The fused and standalone forms are generated from the same WGSL body. `leakyRelu` and `elu` are not
fusable: a fused program carries no per-op parameter.

### Reductions

```java
float s = a.sum();        float m = a.mean();
float hi = a.maxValue();  float lo = a.minValue();
float n2 = a.sumSquares();          // sum(x^2), no intermediate tensor
float err = pred.mse(target);       // mean((pred - target)^2), one pass

a.sum(res);                         // into a 1-element tensor, no readback
NnOps.reduce(NnOps.Reduction.SUM, in, res);
NnOps.reduceSquare(NnOps.Reduction.MEAN, in, res);
NnOps.reduceDiffSquared(NnOps.Reduction.MEAN, pred, target, res);
```

The `float`-returning forms allocate a one-element tensor, reduce, and read it back, which blocks
on the GPU; the `(res)` forms leave the result on the device, which is what you want inside a
`GpuBatch` or before further GPU work.

A full reduction is **exactly two dispatches** at any size: a bounded grid-stride pass writing at
most 1024 partials, then one workgroup reducing those. Tensors beyond the binding limit are tiled,
with partials per tile.

Along the last axis:

```java
a.sumRows()   a.meanRows()   a.maxRows()   a.minRows()     // and the (res) forms
a.argmaxRows()   a.argminRows()                            // i32 indices, ties -> lowest
NnOps.reduceRows(NnOps.Reduction.MEAN, in, res);
NnOps.argmaxRows(in, i32Res, true);
```

Summation is a workgroup tree, so error grows with log(n) rather than n; compare against a CPU
reference with a log-scaled tolerance, not a fixed epsilon.

### Softmax and normalisation

```java
a.softmax()        a.softmax(out)       // out may be a
a.logSoftmax()     a.logSoftmax(out)
a.layerNorm(eps)                        // no affine transform
a.layerNorm(gamma, beta, eps)           a.layerNorm(gamma, beta, eps, out)
a.rmsNorm(gamma, eps)                   a.rmsNorm(gamma, eps, out)   // gamma may be null
```

`gamma` and `beta` are per-column, so their length is the last axis. Layer norm needs both or
neither — passing one is an `IllegalArgumentException`.

Numerics: softmax uses the **online** max-and-sum recurrence, so a row is read twice instead of
three times and no `exp` sees a positive argument (logits around 590 come out finite). Layer norm
accumulates **Welford** triples and combines them pairwise in the tree, so the variance never comes
from `E[x²] − E[x]²`; it is the biased variance, as PyTorch's `LayerNorm` uses. Prefer
`logSoftmax()` to `softmax().log()`.

### Losses and layers

```java
try (Tensor loss = logits.crossEntropy(targets)) {   // [rows] of per-row loss
    float batchLoss = loss.mean();
}
logits.crossEntropy(targets, res);

a.addRowBias(bias)        a.addRowBias(bias, out)    // out may be a
x.linear(weights, bias)   x.linear(weights, bias, out)
```

`crossEntropy` takes **i32 class indices** with the reduced shape and computes
`logsumexp(row) − logits[row, target]` without materialising the softmax. An index outside
`[0, cols)` yields **NaN** for that row: a kernel cannot report an error, so the value is the
signal — check for NaN if the indices are not trusted.

`linear` is a GEMM plus a bias broadcast (`bias` may be null). It is two dispatches, so wrap it
with the activation in a `GpuBatch` to submit the layer once.

### Scalar arithmetic

```java
a.add(2f)   a.sub(2f)   a.mul(2f)   a.div(2f)       // and the (c, res) forms
```

Each compiles to one fused kernel, so `a.mul(2f)` costs the same as any single elementwise op.
`FusedExpr` also has `rsub`, `rdiv` and `rpow` for a scalar on the left.

## Fusion

`lazy()` starts a deferred expression; nothing runs until `eval()`, which compiles the whole tree
into **one** kernel — every input read once, the output written once, no intermediates.

```java
try (Tensor r = a.lazy().sin().add(b).mul(c).eval()) { }

// Into an existing tensor, which may be one of the inputs:
a.lazy().sin().mul(2f).eval(out);
```

| | Methods |
| --- | --- |
| Unary | `neg abs sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh exp log sqrt` |
| Activations | `relu sigmoid gelu silu softplus` — `leakyRelu` and `elu` take a parameter a fused program cannot carry, so they stay standalone |
| Binary (tensor or expression) | `add sub mul div max min pow` |
| Scalar | `add sub mul div pow` |
| Scalar on the left | `rsub rdiv rpow` — `rsub(1f)` is `1 - x` |

f32, identical shapes, finite constants. Expressions are immutable and re-evaluable; input tensors
are referenced, not copied, so they must stay open until `eval` returns. A program is limited to
1024 words (a constant costs two) and a stack depth of 64 — `eval()` a subexpression if you hit
that. Repeated subexpressions are value-numbered on the native side and computed once, and the
compiled pipeline is cached by program, so the same expression shape in a loop compiles once.


## Collectives (multi-GPU)

`hc4j.collective` runs a ring of GPU **ranks** with collective communication across them: a
bandwidth topology probe and ring AllReduce.

```java
try (Collective ring = Collective.open(2)) {
    ring.topology().forEach(System.out::println);
    // rank 0 -> 1: 0.77 GB/s HOST_STAGED (loopback)

    try (CollectiveBuffer a = ring.allocate(0, 1 << 20);
         CollectiveBuffer b = ring.allocate(1, 1 << 20)) {
        a.write(gradientsFromRank0);
        b.write(gradientsFromRank1);

        ring.allReduceSum(a, b);     // both now hold the element-wise sum
        float[] summed = a.read();
    }
}
```

### Ranks and loopback

A rank is a logical device. Ranks are spread across the machine's **distinct physical GPUs**, best
first (discrete before integrated); when there are fewer GPUs than ranks the remainder open extra
logical devices on a GPU already in use — **loopback mode** — so the ring, the transport and the
reduction all run end to end on a single-GPU machine.

Two details the probe forced:

- One physical GPU enumerated under two backends (DX12 *and* Vulkan on Windows) counts **once**.
  Spreading a ring across both would be one device addressed two ways, and the hop would measure a
  cross-backend host round trip.
- **Software adapters are excluded.** Windows enumerates "Microsoft Basic Render Driver" (WARP)
  next to the real GPU; a ring that put a rank there measured 0.01 GB/s and ran 8× slower overall.
  A software adapter is used only if the machine has nothing else.

### Topology

```java
List<Link> topology()     // one Link per directed rank pair
record Link(int src, int dst, double gbps, Transport transport, boolean loopback)
enum Transport { HOST_STAGED, PEER_TO_PEER }
```

The first `topology()` call probes every ordered pair with a 16 MiB transfer averaged over 8
iterations, then caches the result. `loopback` is true when both ranks sit on the same physical
GPU, in which case `gbps` is an intra-GPU copy and **not** a PCIe figure.

`Transport.PEER_TO_PEER` is never reported by this backend — see the transport note below.

### AllReduce

```java
void allReduceSum(CollectiveBuffer... perRank)
```

Textbook ring AllReduce: `n - 1` rounds of scatter-reduce, in which rank `i` sends chunk
`(i - round) mod n` to rank `i + 1`, which adds it into its own copy; then `n - 1` rounds of
allgather circulating the finished chunks. Each rank moves `2(n-1)/n` of the buffer rather than the
`n-1` copies an all-to-all would.

Requirements, all enforced with `IllegalArgumentException`: exactly one buffer per rank, passed in
rank order, every buffer the same length, and that length divisible by the rank count.

### Buffers

```java
CollectiveBuffer allocate(int rank, long elements)
void   write(float[] data)     // the whole buffer
float[] read()
```

Collective buffers are **separate from `Tensor` VRAM**: a collective owns its own devices, so data
crosses in and out through `write`/`read`. There is no zero-copy bridge to a `Tensor` yet. Closing
the collective releases every buffer it handed out, so closing a buffer afterwards is a no-op.

### Transport, and what it costs

Every hop is **host-staged**: the source rank copies its chunk into a mappable buffer, the host
reads it, and the destination rank uploads it. wgpu exposes no external-memory export/import, so
there is no `dma_buf` (Linux) or `OPAQUE_WIN32` (Windows) handle path — true peer-to-peer needs raw
Vulkan reached through `wgpu-hal`, which is why `Transport` already carries the distinction.

Hops are also **synchronous**: a transfer completes before the reduction consuming it is recorded.
The ring is correct but not pipelined, so expect the cost to be dominated by round trips rather
than bandwidth. Measured on one integrated GPU in loopback: 1 Mi f32 across 2 ranks takes about
100 ms, staging 4 MiB per rank.

Not thread-safe for concurrent AllReduce calls on one instance.

## Batching

Small ops are dominated by submission overhead. A `GpuBatch` scope records every op in it into one
command buffer and submits once at `close()`:

```java
try (GpuBatch batch = GpuBatch.open()) {
    for (int i = 0; i < 100; i++) {
        x.sin(x);
    }
} // one submission here
```

Scopes nest, and the batch is engine-wide: ops issued from other threads while a scope is open
join the same command buffer. Reading a result inside a scope is *correct* — a readback flushes the
recorded work first, so it never sees stale data — but it submits early and gives back exactly the
overhead the batch was there to avoid. Keep `toFloatArray()` outside the scope. Measured at 20× for
100 small ops.

## Engine control and statistics

```java
WgpuBackend.initGpu();                       // optional; implicit on first use
WgpuBackend.synchronize();                   // wait for all submitted work
WgpuBackend.configureMemory(vramBytes, hostBytes);  // 0 keeps the current value
MemoryStats WgpuBackend.memoryStats();
EngineStats WgpuBackend.engineStats();
```

`configureMemory` evicts and spills immediately down to the new limits, which is how the smoke
suites exercise the tiers on any GPU.

**`MemoryStats`** (21 counters): `vramBudget vramUsed hostBudget hostUsed spilledBytes
tensorsDevice tensorsHost tensorsEvicted evictions pageIns spills driverOoms streamedOps
allocations dedicatedBuffers slabsCreated regionAllocs slabCount slabBytes slabRegions
quarantined`.

**`EngineStats`** (13 counters): `submissions dispatches kernelBytes hostToDeviceBytes
deviceToHostBytes batches completedEpoch submittedEpoch features deviceType backend
subgroupMinSize subgroupMaxSize`, with helpers `hasSubgroups()`, `hasShaderF16()`, `isDiscrete()`,
`backendName()`, `deviceTypeName()`.

These are how you verify engine behaviour rather than guess at it: `dispatches` confirms a fused
expression was one kernel, `allocations` and `dedicatedBuffers` confirm slab reuse,
`submissions` confirms batching, `hostToDeviceBytes` confirms a transfer actually happened.

```java
EngineStats before = WgpuBackend.engineStats();
try (Tensor r = a.lazy().sin().add(b).mul(c).eval()) { }
EngineStats after = WgpuBackend.engineStats();
assert after.dispatches() - before.dispatches() == 1;
```

## Exceptions

| Exception | Cause |
| --- | --- |
| `IllegalArgumentException` | shape, dtype or aliasing violation; invalid native parameters |
| `IllegalStateException` | op on a closed tensor; unknown handle; device lost; shader or pipeline rejected |
| `GpuOutOfMemoryException` | VRAM, host RAM and the spill tier are all exhausted |
| `UnsupportedOperationException` | layout or size the chosen path cannot handle (e.g. a transposed out-of-core operand) |
| `RuntimeException` | readback/mapping failure, spill-file I/O failure |
| `UnsatisfiedLinkError` | the native library or a symbol is missing — rebuild the Rust backend |

A native panic is contained at the FFI boundary and surfaces as `IllegalStateException`; it never
takes the JVM down.

## Environment variables

| Variable | Effect |
| --- | --- |
| `HC4J_BACKEND` | `dx12` \| `vulkan` \| `metal` — narrows the backend set. Default on Windows: DX12 + Vulkan |
| `HC4J_ADAPTER` | case-insensitive substring; pins a GPU by name |
| `HC4J_VRAM_BUDGET_MB` | starting VRAM budget. Default: 8192 discrete, 2048 integrated, 1024 otherwise |
| `HC4J_HOST_BUDGET_MB` | host-RAM tier budget. Default 4096 |
| `HC4J_SPILL_DIR` | directory for spill files. Default `<temp>/hc4j-spill` |
| `HC4J_GEMM_TILE` | `64` \| `128` — forces the register-tiled GEMM block size |
| `HC4J_WARMUP` | `0` disables background matmul pipeline compilation |
| `HC4J_TRACE` | any non-empty value except `0` enables paging and compile-time diagnostics |

Budgets are a starting point, not a hard cap on correctness: a driver OOM clamps the VRAM budget to
what is actually committed, and the engine keeps running.

## Threading

The engine is internally synchronised and usable from multiple threads: handles are global, the
memory table is lock-protected (never across a GPU wait), and batching is engine-wide. A single
`Tensor` object is not itself synchronised — don't race two ops writing the same output, and don't
`close()` a tensor another thread is still using.
