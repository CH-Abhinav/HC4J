# HC4J — Heterogeneous Compute for Java

GPU tensor execution for the JVM: a Rust `cdylib` built on [wgpu](https://wgpu.rs) with WGSL
compute kernels, bound into Java 25 through the Panama Foreign Function & Memory API. No JNI, no
native glue to compile, no CUDA. Tensors live in VRAM and are identified by an opaque `u64` handle;
Java holds the handle, the Rust engine owns the memory.

```java
try (Tensor a = Tensor.fromArray(x, 1024, 1024);
     Tensor b = Tensor.fromArray(y, 1024, 1024);
     Tensor c = a.matmul(b)) {
    float[] result = c.toFloatArray();
}
```

The engine runs on whatever the machine has — Direct3D 12 or Vulkan on Windows, Vulkan on Linux,
Metal on macOS — and picks the most capable adapter it can find, preferring a discrete GPU.

## What is in the box

| Area | Capability |
| --- | --- |
| Elementwise | 12 trigonometric, 5 exponential/log, 4 arithmetic ops, f32 (plus i32 arithmetic), contiguous and strided |
| Linear algebra | GEMM with register-tiled, shared-memory and GEMV kernels chosen per shape; in-VRAM transpose |
| Kernel fusion | `a.lazy().sin().add(b).mul(c).eval()` compiles to one kernel: 4N bytes moved instead of 8N, one allocation instead of three |
| Memory | Three-tier manager (VRAM → host RAM → disk) with LRU eviction, page-in on use, and streaming for tensors larger than the whole VRAM budget |
| Allocation | Slab sub-allocation with epoch-based quarantine: 200 consecutive op outputs cost 0 driver allocations |
| Batching | `GpuBatch` records many ops into one command buffer and one submission (20× for small ops) |
| Robustness | Every native entry point is panic-guarded; wgpu validation and OOM errors surface as Java exceptions, never as a JVM abort |

See [docs/API.md](docs/API.md) for the Java API, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how
the engine works, and [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for measured numbers and tuning.

## Requirements

- **JDK 25** (or any JDK 22+ with the FFM API final; the build's toolchain is set to 25).
- **Rust** stable with `cargo` (edition 2024).
- A GPU with a working DX12, Vulkan or Metal driver. Integrated GPUs are fully supported; the
  engine adapts its kernel choice and memory budget to the device class.
- Gradle wrapper included. If Gradle cannot find your JDK 25 — it reads the Windows registry and
  misses some installs — point it at the install explicitly in `~/.gradle/gradle.properties`:

  ```properties
  org.gradle.java.installations.paths=C:/Program Files/Java/jdk-25.0.4
  ```

## Build and run

The Gradle build compiles the Rust backend first (`cargo build --release`) and resolves the shared
library at runtime by walking up from the working directory to `src/main/rust/target/release/`.

```bash
./gradlew build          # Rust cdylib + Java classes + both GPU smoke suites
./gradlew run            # the hc4j.test.Main demo
./gradlew smokeTests     # 182 end-to-end GPU checks
./gradlew jmh            # JMH microbenchmarks (see docs/PERFORMANCE.md)
```

`check` depends on the smoke suites, so `build` needs a working GPU. Drop the
`tasks.named('check').configure { dependsOn 'smokeTests' }` line in `build.gradle` to keep the GPU
out of the default build.

The Rust side has its own suite — 48 tests covering adapter ranking, the memory tiers, shader
validation through naga, and GPU correctness against CPU references:

```bash
cargo test --release --manifest-path src/main/rust/Cargo.toml
```

Java launches need `--enable-native-access=ALL-UNNAMED`; the Gradle tasks add it for you.

## Quick start

```java
import hc4j.DType;
import hc4j.Tensor;
import hc4j.engine.GpuBatch;

// Upload, compute, download. close() frees the VRAM.
try (Tensor x = Tensor.fromArray(new float[] {0f, 1f, 2f, 3f}, 4)) {

    // Convenience form: allocates the result.
    try (Tensor y = x.sin()) {
        float[] host = y.toFloatArray();
    }

    // Caller-allocated form: no allocation per op, and in-place is legal.
    try (Tensor out = Tensor.empty(DType.f32, 4)) {
        x.sin(out);          // out = sin(x)
        out.exp(out);        // in place
        x.add(out, out);     // out = x + out
    }

    // One kernel for the whole expression.
    try (Tensor b = Tensor.fromArray(new float[] {1f, 1f, 1f, 1f}, 4);
         Tensor r = x.lazy().sin().mul(2f).add(b).eval()) {
        float[] host = r.toFloatArray();
    }

    // One submission for many small ops.
    try (Tensor t = Tensor.empty(DType.f32, 4)) {
        try (GpuBatch batch = GpuBatch.open()) {
            x.sin(t);
            t.cos(t);
            t.sqrt(t);
        } // ops are submitted here
    }
}
```

## Limitations

- **f32 only** outside arithmetic (which also does i32). No f16 compute yet, even where the adapter
  reports `SHADER_F16`; the feature is probed and exposed, not used.
- **No broadcasting.** Binary ops and fusion require identical shapes. Non-contiguous inputs work
  through the strided path (a transposed view reads correctly), but the shapes must still match.
- **Matmul is 2-D** (with the NumPy 1-D conventions: matrix·vector, vector·matrix, dot product).
  No batched GEMM.
- **Out-of-core matmul does not transpose.** When an operand is larger than the VRAM budget the
  streamed and K-blocked paths reject stored-transposed operands; materialise them with
  `transpose` first.
- **One device.** Multi-GPU and concurrent devices are out of scope.
- **Precision follows WGSL**, which guarantees only 2⁻¹¹ absolute error for `sin`/`cos` and
  inherits `atan2`'s ~4096 ULP for the inverse trig functions. Tests are toleranced accordingly —
  do not expect `StrictMath` agreement.

## Repository layout

```
src/main/java/hc4j/          Tensor, FusedExpr, DType
            hc4j/engine/     WgpuBackend (FFM downcalls), GpuBatch, stats records
            hc4j/ops/        TrignoOps, ExponentialOps, ArithmeticOps, MatmulOps, Layouts
            hc4j/test/       Main demo and the two smoke suites
src/main/rust/src/           lib.rs (engine, adapter selection, pipeline cache, C ABI)
                             stream.rs (command batching), error.rs (panic guard, status codes)
                  memory/    manager, budget, lru, slab, spill, transfer
                  ops/       elementwise, trigno, exponential, arithmetic, matmul, fusion
src/jmh/java/hc4j/bench/     JMH microbenchmarks
docs/                        ARCHITECTURE.md, API.md, PERFORMANCE.md
```

Roughly 7 kLOC of Rust and 2.6 kLOC of Java.
