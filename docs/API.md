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
| Binary (tensor or expression) | `add sub mul div max min pow` |
| Scalar | `add sub mul div pow` |
| Scalar on the left | `rsub rdiv rpow` — `rsub(1f)` is `1 - x` |

f32, identical shapes, finite constants. Expressions are immutable and re-evaluable; input tensors
are referenced, not copied, so they must stay open until `eval` returns. A program is limited to
1024 words (a constant costs two) and a stack depth of 64 — `eval()` a subexpression if you hit
that. Repeated subexpressions are value-numbered on the native side and computed once, and the
compiled pipeline is cached by program, so the same expression shape in a loop compiles once.

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
