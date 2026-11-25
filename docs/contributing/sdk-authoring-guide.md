# hc4x SDK Authoring Guide

Welcome to the engineering guide for building thin, idiomatic language SDKs under the **hc4x** umbrella (`hc4rs`, `hc4j`, `hc4py`, and community bindings).

All compute kernels, tiered memory management (`VRAM -> Host RAM -> NVMe Disk`), Vulkan P2P collectives, and the Exact Zero-Parameter Optimizer (Theorems 1–4 + Zero-Waste Split-KV Reclamation) live once in the native Rust engine (`hc4rs` / `hc4x-ffi`). New language SDKs are **thin, zero-business-logic wrappers** that expose idiomatic ergonomics while enforcing strict safety invariants.

---

## 1. Two-Path Architecture Overview

```
                    +---------------------------------------------------+
                    |              native/core/  (hc4rs)                |
                    |  Safe Rust Engine: wgpu 30, ash 0.38, naga 30,   |
                    |  3-tier memory, WGSL/SPIR-V kernels, collectives, |
                    |  Theorems 1-4 + Split-KV Exact Optimizer          |
                    +-------------------------+-------------------------+
                                              |
                    +-------------------------v-------------------------+
                    |    native/ffi/  (hc4x-ffi -> libhc4j / hc4j.dll)  |
                    |  native/include/hc4x.h (107 exports, 13 structs)  |
                    |  Every symbol wrapped in ffi_guard (zero panics)  |
                    +------------+------------------------+-------------+
                                 |                        |
         Path A: Native C-ABI    |                        |  Path B: JVM / JPMS
       (ctypes / bun:ffi / cgo)  |                        |  (Panama FFM via hc4j)
     +---------------------------v----+            +------v-----------------------+
     | Python:  hc4py (sdks/python/)  |            | Java 25: hc4j (core/ module) |
     | C / C++: direct hc4x.h include |            | Kotlin:  hc4kt (extension)   |
     | TS/Bun:  hc4ts (future)        |            | Scala:   hc4sc (future)      |
     | Go / C# / Swift (future)       |            | Clojure: hc4clj (future)     |
     +--------------------------------+            +------------------------------+
```

### Choosing Your Integration Path

| Target Language / Ecosystem | Recommended Path | Why |
| :--- | :--- | :--- |
| **Python, TypeScript/Bun, Go, C#, Swift, C/C++, Zig** | **Path A (Native C-ABI over `hc4x-ffi`)** | Consumes `hc4j.dll` / `libhc4j.so` / `libhc4j.dylib` via `native/include/hc4x.h` cleanly using each runtime's zero-overhead FFI (`ctypes`, `bun:ffi`, `cgo`, `.NET LibraryImport`, Swift C interop). |
| **Kotlin, Scala, Clojure, Groovy** | **Path B (JVM Bundle over `hc4j`)** | Depends on JPMS module `hc4j` directly (pure `java.base`, zero runtime dependencies). Inherits Panama FFM downcalls, `Cleaner` + ref-counted `VramHandle`, `GpuScope`, `DistributedExecutor`, and `.safetensors` loaders with zero C boilerplate. |
| **Rust** | **Direct Tier-1 (`hc4rs`)** | Add `hc4rs` path/crate dependency directly; use safe Rust `Tensor`, `engine`, `kernels`, and `collective::planner` APIs without crossing C-ABI. |

---

## 2. Canonical Domain Organization

To keep SDKs consistent across languages, organize every SDK into crisp functional modules mirroring `hc4rs` and `hc4j`:

| Module / Package | Responsibilities |
| :--- | :--- |
| `_ffi` / `internal.ffi` | Shared-library resolution, C struct layouts, status-code -> exception mapping, `hc4j_last_error` query |
| `dtype` | `f32` (`1`), `i32` (`0`), `f16` (`2`), item byte sizes, IEEE-754 half-precision host conversion helpers |
| `engine` | Lazy GPU initialization, synchronous barrier, command batching context manager (`batch`), adapter & memory telemetry |
| `tensor` | Ref-counted VRAM handle, strided view metadata (`shape`, `strides`, `offset`), NumPy / host buffer transfer, `close()` |
| `math` | Elementwise binary arithmetic, 12 trig ops, 5 exp/log/sqrt ops, unary math, `clamp`, comparisons (`eq..le`), `where`, `cast` |
| `linalg` | `matmul`, `batched_matmul`, `matmul_transposed_b` (`f32 x f32` and `f32 x f16`), 2-D `transpose`, `matmul_plan` |
| `nn` | 7 activations, `softmax`, `log_softmax`, `layer_norm`, `rms_norm`, `cross_entropy`, `add_row_bias`, reductions, `embedding`, `rope`, `kv_cache_append` |
| `fusion` | Lazy expression AST/DAG compiler producing `hc4j_fused_dispatch` bytecode |
| `collective` | Multi-GPU `Collective` runtime, topology discovery, `allreduce_sum`, `gemm_bias_act`, `split_kv_attn`, `merge_split_kv` |
| `optimizer` | Direct binding to `hc4j_optimizer_plan_work` and `hc4j_optimizer_plan_roles` (no local math reimplementation) |

---

## 3. Complete C-ABI Reference (`native/include/hc4x.h`)

Every exported symbol resides in `native/ffi/` (`#[unsafe(no_mangle)] pub extern "C"`), is panic-guarded via `hc4rs::engine::error::ffi_guard`, and returns `int32_t` (`>= 0` on success, `< 0` on error) or `uint64_t` handle (`0` on allocation failure).

### 3.1 Status Codes & Diagnostics (`hc4j_last_error`)

| Constant | Value | Meaning | Recommended Exception / Error |
| :--- | :---: | :--- | :--- |
| `HC4J_SUCCESS` | `0` | Success (non-negative counts may also return `> 0`) | — |
| `HC4J_ERR_NOT_FOUND` | `-1` | Invalid or freed VRAM buffer / collective handle ID | `NotFoundError` / `IllegalStateException` |
| `HC4J_ERR_INVALID_PARAM` | `-2` | Null pointer, zero length, dtype/shape mismatch | `InvalidParamError` / `IllegalArgumentException` |
| `HC4J_ERR_GPU_READBACK` | `-3` | Staging map/readback failure | `ReadbackError` / `RuntimeException` |
| `HC4J_ERR_OUT_OF_MEMORY` | `-4` | VRAM, host RAM, and Tier-2 disk spill exhausted | `OutOfMemoryError` / `GpuOutOfMemoryException` |
| `HC4J_ERR_DEVICE` | `-5` | GPU unavailable, device lost, or submission panic guard | `DeviceError` / `IllegalStateException` |
| `HC4J_ERR_GPU_VALIDATION` | `-6` | wgpu/Vulkan shader or pipeline validation error | `GpuValidationError` / `IllegalStateException` |
| `HC4J_ERR_HOST_IO` | `-7` | Tier-2 disk spill read/write failure | `HostIoError` / `RuntimeException` |
| `HC4J_ERR_UNSUPPORTED` | `-8` | Valid request unsupported by active layout/kernel | `UnsupportedError` / `UnsupportedOperationException` |

**Diagnostics Pattern (`hc4j_last_error`)**:
Whenever any C-ABI call returns `< 0` (or allocation returns `0`), immediately invoke:
```c
uint8_t diag[512];
int32_t len = hc4j_last_error(diag, sizeof(diag)); // thread-local UTF-8, NUL-terminated
```
Passing `(NULL, 0)` queries the full UTF-8 byte length without copying. `hc4j_last_error` uses `ffi_guard_preserve_error`, so querying it never overwrites the stored diagnostic.

### 3.2 Symbol Groups (107 Total Exports)

1. **Engine & Error Diagnostics (7)**:
   `hc4j_init_gpu`, `hc4j_batch_begin`, `hc4j_batch_end`, `hc4j_synchronize`, `hc4j_engine_stats`, `hc4j_adapter_info`, `hc4j_last_error`
2. **Tiered Tensor Memory Management (12)**:
   `hc4j_gpu_alloc`, `hc4j_gpu_alloc_uninit`, `hc4j_gpu_alloc_bytes`, `hc4j_gpu_write_bytes`, `hc4j_gpu_download_bytes`, `hc4j_gpu_write`, `hc4j_gpu_free`, `hc4j_gpu_download`, `hc4j_mem_configure`, `hc4j_mem_stats`, `hc4j_mem_residency`, `hc4j_mem_evict`
3. **Math, Elementwise, Compare, Selection & Casts (43)**:
   - Binary (`add`, `sub`, `mul`, `div`) x2 (`dispatch_<op>` typed + `dispatch_<op>_f32`) = 8
   - Trig (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`) = 12
   - Exp/Log (`exp`, `ln`, `log2`, `log10`, `sqrt`) = 5
   - Unary math & clamp (`abs`, `neg`, `rsqrt`, `floor`, `ceil`, `round`, `sign`, `clamp_f32`, `unary_i32`) = 9
   - Multi-operand & selection (`pow_f32`, `maximum`, `minimum`, `compare`, `where`) = 5
   - Casts & strided copy (`cast`, `cast_f32_to_f16`, `cast_f16_to_f32`, `copy_strided`) = 4
4. **Linear Algebra (6)**:
   `dispatch_matmul_f32`, `dispatch_matmul_f32_ex`, `dispatch_batched_matmul_f32`, `dispatch_matmul_f32_f16`, `dispatch_transpose_f32`, `hc4j_matmul_plan`
5. **Neural Networks, Reductions & Fusion (17)**:
   - NN & Transformer: `dispatch_activation_f32`, `dispatch_softmax_rows_f32`, `dispatch_layernorm_rows_f32`, `dispatch_rmsnorm_rows_f32`, `dispatch_cross_entropy_rows`, `dispatch_add_row_bias_f32`, `dispatch_kv_cache_append_f32`, `dispatch_embedding`, `dispatch_rope_f32`
   - Reductions: `dispatch_reduce_f32`, `dispatch_reduce_square_f32`, `dispatch_reduce_diff2_f32`, `dispatch_reduce_rows_f32`, `dispatch_argmax_rows`, `dispatch_reduce_axis_f32`, `dispatch_argmax_axis`
   - Fusion VM: `hc4j_fused_dispatch`
6. **Collective Runtime & Split-KV Merge (20)**:
   `hc4j_collective_create`, `hc4j_collective_destroy`, `hc4j_collective_ranks`, `hc4j_collective_link_count`, `hc4j_collective_link`, `hc4j_collective_device_name`, `hc4j_collective_backend_name`, `hc4j_collective_alloc`, `hc4j_collective_free`, `hc4j_collective_write`, `hc4j_collective_read`, `hc4j_collective_allreduce_sum`, `hc4j_collective_allreduce_sum_subset`, `hc4j_collective_gemm_bias_act`, `hc4j_collective_split_kv_attn`, `hc4j_collective_device_profile`, `hc4j_collective_ring_order`, `hc4j_collective_island_count`, `hc4j_collective_island`, `hc4j_collective_merge_split_kv_f32`
7. **Exact Theorems 1–4 Optimizer (2)**:
   `hc4j_optimizer_plan_work`, `hc4j_optimizer_plan_roles`

### 3.3 Exact `#[repr(C)]` Struct Layouts (13 Structs)

Every struct uses natural C alignment (`8`-byte aligned due to `u64`/`f64` fields, zero internal padding holes):

| C Struct (`hc4x.h`) | Byte Size | Field Layout Summary |
| :--- | :---: | :--- |
| `MemStats` | `168` | `21 x uint64_t` counters (`vram_budget..quarantined`) |
| `EngineStats` | `104` | `13 x uint64_t` (`submissions..subgroup_max_size`) |
| `AdapterInfoC` | `668` | `7 x uint32_t`, `uint8_t name[256]`, `driver[128]`, `driver_info[256]` |
| `LinkInfo` | `32` | `u32 src(0), u32 dst(4), f64 bw(8), u32 transport(16), u32 loopback(20), f64 lat_us(24)` |
| `DeviceProfileC` | `32` | `u32 rank(0), u32 vendor(4), u32 dev(8), u32 dev_type(12), u64 vram(16), f64 gflops(24)` |
| `IslandHeaderC` | `16` | `u32 id(0), u32 leader(4), u32 transport(8), u32 member_count(12)` |
| `HostCpuProfileC` | `32` | `u64 ram_bytes(0), f64 gflops(8), f64 ram_bw(16), f64 agg_pcie(24)` |
| `WorkloadSpecC` | `120` | `u64 units(0)..u64 disk_cap(72), u32 micro_batches(80), u32 allow_spilling(84), HostCpuProfileC(88)` |
| `ShardAllocationC` | `72` | `i32 rank(0), u32 excluded(4), u64 units(8)..u64 cpu_in_place_units(64)` |
| `WorkloadPlanHeaderC` | `40` | `u64 units(0), f64 makespan(8), f64 compute(16), f64 sync(24), u32 allocs(32), u32 active(36)` |
| `SplitKvSpecC` | `48` | `u64 tokens(0), u64 bytes_per_tok(8), f64 flops(16), u64 rpc_bytes(24), u64 reserved(32), f64 vram_util(40)` |
| `SplitKvAssignmentC` | `48` | `i32 rank(0), u32 role(4), u64 tokens(8), f64 attn_ms(16), f64 comm_ms(24), f64 total_ms(32), u64 mem(40)` |
| `RolePlanHeaderC` | `80` | `WorkloadPlanHeaderC tp_header(0..40), f64 tp_ms(40), f64 attn_ms(48), u64 tokens(56), u32 split_cnt(64), u32 reclaimed(68), u32 cpu_rec(72), u32 regime(76)` |

Always allocate output arrays sized `device_count + 1` to accommodate the optional `HOST_CPU_RANK = -1` entry!

---

## 4. `FusedExpr` Stack-Machine Bytecode Specification

`hc4j_fused_dispatch(program, program_len, inputs, n_inputs, id_out)` compiles and runs a single WGSL stack-machine kernel for arbitrary elementwise expressions over same-shape contiguous `f32` tensors.

### Word Encoding

Every instruction is a 32-bit unsigned integer:
```
word = ((opcode & 0xFF) << 24) | (operand & 0x00FFFFFF)
```
Maximum program length: **`MAX_PROGRAM_WORDS = 1024`** words.

### Opcodes & Operands

| Category | Opcode Hex | Meaning | Stack Effect / Trailing Words |
| :--- | :---: | :--- | :--- |
| Memory / Const | `0x01` (`LOAD`) | Push `inputs[operand][idx]` | `() -> (x)` |
| Memory / Const | `0x02` (`CONST`) | Push inline IEEE-754 `f32` constant | **Next word** holds raw `uint32_t` float32 bits; `() -> (c)` |
| Unary Math | `0x10` `NEG`, `0x11` `ABS` | `-x`, `|x|` | `(x) -> (r)` |
| Unary Trig | `0x12..0x1D` | `SIN(12), COS(13), TAN(14), ASIN(15), ACOS(16), ATAN(17), SINH(18), COSH(19), TANH(1A), ASINH(1B), ACOSH(1C), ATANH(1D)` | `(x) -> (r)` |
| Unary Exp/Log | `0x1E..0x20` | `EXP(1E), LOG(1F), SQRT(20)` | `(x) -> (r)` |
| Activations | `0x21..0x25` | `RELU(21), SIGMOID(22), GELU(23), SILU(24), SOFTPLUS(25)` | `(x) -> (r)` |
| Binary Ops | `0x40..0x46` | `ADD(40), SUB(41), MUL(42), DIV(43), MAX(44), MIN(45), POW(46)` | Pops `right`, pops `left`, pushes `op(left, right)` |

### Compilation Order

Emit instructions in **strict post-order**:
- For `Binary(op, left, right)`: emit `left`, then emit `right`, then emit `word(op, 0)`.
- For `Unary(op, x)`: emit `x`, then emit `word(op, 0)`.
- Deduplicate identical tensor references into unique indices `0 .. n_inputs - 1` so `inputs[]` passes each buffer handle once.
- Validate all `CONST` scalars are finite (`isfinite`) and materialize any non-contiguous / offset / layout-mismatched aliased `Load` tensors before calling `hc4j_fused_dispatch`.

---

## 5. Seven Mandatory Correctness & Safety Invariants

Every SDK implementation MUST uphold these seven invariants:

1. **Ref-Counted Handle Ownership Across Views**:
   `view()`, `reshape()`, `transpose()`, and `slice()` share the same GPU allocation ID. Wrap `vram_id` in a thread-safe reference-counted handle (`retain()` on view creation, `release()` on `close()` / finalizer). Call `hc4j_gpu_free` ONLY when refcount reaches `0`.
2. **`contiguous()` Short-Circuit View Invariant**:
   When a tensor already has `is_contiguous == true` and `offset == 0`, `contiguous()` MUST return a newly retained view sharing the same VRAM handle with **zero GPU copy**. Otherwise, allocate a fresh buffer and materialize via `dispatch_copy_strided`.
3. **Deterministic Close + Idempotency + Open Guard**:
   Support scoped deterministic cleanup (`AutoCloseable` / `try-with-resources`, Python `with`, Kotlin `.use {}`, C# `using`) AND an atomic `closed` flag making `close()` idempotent. Every accessor or kernel dispatch MUST fail fast on closed handles via `ensure_open()`.
4. **Layout-Mismatched In-Place Aliasing Guard**:
   Operations like `a.add_(a.T)` or `t.sin(out=t.T)` are legal user calls even though source and destination share `vram_id` with conflicting strides/offsets. Detect `src.vram_id == out.vram_id && !has_identical_layout(src, out)` and copy `src` into an independent temporary buffer **before** dispatching.
5. **Contiguous Zero-Offset Destination Guarantee**:
   All GPU compute kernels write packed row-major outputs. Validate `out.is_contiguous && out.offset == 0` before dispatching any write. Also forbid aliasing output with operand in matrix multiplications (`matmul`, `matmul_transposed_b`), `embedding`, and `kv_cache_append`.
6. **Zero Optimizer / Merge Re-Implementation**:
   Never re-implement Theorems 1–4 or online-softmax merging in Path A host languages. Call `hc4j_optimizer_plan_work`, `hc4j_optimizer_plan_roles`, and `hc4j_collective_merge_split_kv_f32` directly through FFI.
7. **Rich Error Propagation**:
   On any `< 0` status return or `0` handle from an allocation, query `hc4j_last_error`, format `{context}: {summary} [{rust_detail}]`, and raise the language's typed exception class.

---

## 6. Ten-Point SDK Conformance Checklist

Before submitting an SDK PR under `sdks/<language>/`, verify all 10 items:

- [ ] **1. Zero Third-Party Runtime Requirements**: Core package relies solely on the language's standard library FFI (`ctypes`, `bun:ffi`, `cgo`, etc.) or on `hc4j` (for Path B); numerical array libraries (`numpy`, etc.) are strictly optional fast paths.
- [ ] **2. Full Struct & Constant Parity**: Every FFI struct `sizeof` and field offset matches `native/include/hc4x.h` (verified by automated layout unit tests).
- [ ] **3. Panic-Safe Error Mapping**: All 9 status codes (`0 .. -8`) map to typed errors including `hc4j_last_error` context.
- [ ] **4. Handle Lifecycle & Leak Guard**: Ref-counted handle + deterministic `close()` + GC fallback finalizer verified across views and exceptions.
- [ ] **5. Aliasing & Stride Safety**: `a.add_(a.T)`, sliced offsets (`offset > 0`), and broadcast shape/stride calculations pass end-to-end GPU tests.
- [ ] **6. Zero-Copy Array Interop + Stdlib Fallback**: Direct native pointer upload/download verified alongside pure stdlib sequence fallback.
- [ ] **7. Complete Domain Coverage**: Covers `engine` (incl. `batch`), `math`, `linalg` (`matmul`, `f32 x f16`, `batched`, `matmul_plan`), `nn` (`norm`, `rope`, `kv_cache_append`, reductions), and `fusion` (`FusedExpr`).
- [ ] **8. Collective & Optimizer Integration**: Calls `hc4j_collective_*`, `hc4j_collective_merge_split_kv_f32`, `hc4j_optimizer_plan_work`, and `hc4j_optimizer_plan_roles` natively.
- [ ] **9. Live GPU + Offline/Optimizer Test Suite**: Automated tests run cleanly against `hc4j.dll`/`libhc4j.so` and skip GPU-only suites gracefully when headless.
- [ ] **10. Documentation & Licensing**: Apache 2.0 header/license metadata + idiomatic docstrings/examples (`README` + quickstart snippet).

---

## 7. Path A Starter Blueprints (Native C-ABI)

### 7.1 TypeScript / Bun (`bun:ffi`)

```ts
import { dlopen, FFIType, ptr } from "bun:ffi";

const lib = dlopen("libhc4j.so", {
  hc4j_init_gpu: { args: [], returns: FFIType.i32 },
  hc4j_last_error: { args: [FFIType.ptr, FFIType.u32], returns: FFIType.i32 },
  hc4j_gpu_alloc_bytes: { args: [FFIType.u64, FFIType.u32], returns: FFIType.u64 },
  hc4j_gpu_write_bytes: { args: [FFIType.u64, FFIType.ptr, FFIType.u64], returns: FFIType.i32 },
  hc4j_gpu_download_bytes: { args: [FFIType.u64, FFIType.ptr, FFIType.u64], returns: FFIType.i32 },
  hc4j_gpu_free: { args: [FFIType.u64], returns: FFIType.i32 },
  hc4j_optimizer_plan_work: {
    args: [FFIType.u32, FFIType.ptr, FFIType.u32, FFIType.ptr, FFIType.ptr,
           FFIType.ptr, FFIType.ptr, FFIType.u32, FFIType.ptr, FFIType.u32],
    returns: FFIType.i32,
  },
}).symbols;

// Float32Array maps zero-copy via ptr(f32Array):
const host = new Float32Array([1, 2, 3, 4]);
const vramId = lib.hc4j_gpu_alloc_bytes(BigInt(host.byteLength), 0);
lib.hc4j_gpu_write_bytes(vramId, ptr(host), BigInt(host.byteLength));
```

### 7.2 Go (`cgo` over `native/include/hc4x.h`)

```go
/*
#cgo CFLAGS: -I${SRCDIR}/../../native/include
#cgo LDFLAGS: -L${SRCDIR}/../../native/target/release -lhc4j
#include "hc4x.h"
*/
import "C"
import (
    "fmt"
    "unsafe"
)

func CheckStatus(code C.int32_t, ctx string) error {
    if code >= 0 {
        return nil
    }
    var buf [512]C.uint8_t
    n := C.hc4j_last_error(&buf[0], 512)
    detail := ""
    if n > 0 {
        detail = C.GoStringN((*C.char)(unsafe.Pointer(&buf[0])), C.int(n))
    }
    return fmt.Errorf("%s failed (%d): %s", ctx, int(code), detail)
}
```

### 7.3 C# / .NET 8 (`LibraryImport` + Explicit Structs)

```csharp
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
public struct DeviceProfileC
{
    public uint Rank, VendorId, DeviceId, DeviceType;
    public ulong VramBytes;
    public double ComputeGflops;
}

internal static partial class NativeMethods
{
    [LibraryImport("hc4j")]
    internal static partial int hc4j_init_gpu();

    [LibraryImport("hc4j")]
    internal static partial int hc4j_last_error(Span<byte> buf, uint cap);

    [LibraryImport("hc4j")]
    internal static partial ulong hc4j_gpu_alloc_bytes(ulong bytes, uint zeroed);
}
```

---

## 8. Path B Starter Blueprints (JVM Ecosystem over `hc4j`)

JVM languages (`Kotlin`, `Scala`, `Clojure`, `Groovy`) depend directly on Maven/Gradle artifact `hc4j` (`requires hc4j;` in JPMS or standard classpath + `--enable-native-access=hc4j`), inheriting Panama FFM, `Cleaner`-backed ref-counted `VramHandle`, `GpuScope`, `DistributedExecutor`, and `SplitKvSession` with **zero C-ABI marshaling**.

### Kotlin DSL Starter (`hc4kt`)

```kotlin
package hc4kt

import hc4j.Tensor
import hc4j.engine.GpuScope

inline fun <T> gpuScope(block: GpuScope.() -> T): T =
    GpuScope.open().use { scope -> scope.block() }

operator fun Tensor.plus(other: Tensor): Tensor = this.add(other)
operator fun Tensor.times(other: Tensor): Tensor = this.mul(other)
infix fun Tensor.matmul(other: Tensor): Tensor = this.matmul(other)

// Usage:
// val z = gpuScope {
//     val a = keep(Tensor.fromArray(floatArrayOf(1f, 2f, 3f, 4f), 2, 2))
//     val b = keep(Tensor.ones(2, 2))
//     (a + b).toFloatArray()
// }
```

### Scala / Clojure Pattern

- **Scala 3**: Export `extension (t: Tensor)` methods (`+`, `-`, `*`, `/`, `@`), `scala.util.Using.resource(GpuScope.open())`, and `IndexedSeq` / `Array[Float]` converters over `Tensor.fromArray`.
- **Clojure**: Wrap `hc4j.Tensor` and `hc4j.collective.WorkloadPlanner` inside idiomatic `with-open [scope (GpuScope/open)]` macros and persistent-vector helpers `(tensor->vec t)`.
