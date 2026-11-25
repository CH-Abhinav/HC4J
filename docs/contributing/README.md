# Contributing to hc4x

Thank you for contributing to the **hc4x** heterogeneous GPU runtime and multi-language SDK ecosystem!

## Repository Layout

| Component | Path | Purpose |
| :--- | :--- | :--- |
| **Rust Core Engine (`hc4rs`)** | [`native/core/`](../../native/core/) | Pure safe Rust GPU engine (`wgpu 30`, `ash 0.38`, `naga 30`), 3-tier memory manager, WGSL/SPIR-V kernels, Vulkan P2P collectives, and Exact Theorems 1–4 optimizer |
| **C-ABI Shared Library (`hc4x-ffi`)** | [`native/ffi/`](../../native/ffi/) | Panic-guarded `cdylib` (`hc4j.dll` / `libhc4j.so` / `libhc4j.dylib`) exporting all 107 C-ABI entry points |
| **Canonical C99 Header** | [`native/include/hc4x.h`](../../native/include/hc4x.h) | Single source of truth for all 107 C-ABI signatures and 13 `#[repr(C)]` structs |
| **Java 25 Flagship Runtime (`hc4j`)** | [`core/`](../../core/), [`examples/`](../../examples/), [`benchmarks/`](../../benchmarks/) | Strict 5-package JPMS `java.base`-only SDK over Panama FFM, `GpuScope`, `DistributedExecutor`, and `.safetensors` LLM runner |
| **Python Thin SDK (`hc4py`)** | [`sdks/python/`](../../sdks/python/) | Zero-required-dependency Python 3.11+ `ctypes` wrapper with optional zero-copy NumPy bridge |
| **SDK Authoring Guide** | [`sdk-authoring-guide.md`](sdk-authoring-guide.md) | Architectural contract, C-ABI reference, invariants, and starter blueprints for Path A (FFI) and Path B (JVM) SDKs |

## Quick Verification Commands

- **Rust Workspace (Clippy + Unit & Layout Tests)**:
  ```bash
  cargo clippy --manifest-path native/Cargo.toml --workspace --all-targets -- -D warnings
  cargo test --manifest-path native/Cargo.toml --workspace
  ```
- **Java 25 Full Suite + Javadoc**:
  ```bash
  ./gradlew :core:test :examples:test :core:javadoc :examples:javadoc
  ```
- **Python SDK (`hc4py`)**:
  ```bash
  PYTHONPATH=sdks/python python -m unittest discover -s sdks/python/tests -v
  ```

## Where to Start When Adding a New Language SDK

Read [`sdk-authoring-guide.md`](sdk-authoring-guide.md) before writing any code. It defines the two integration paths (**Path A: Native C-ABI Bundles** and **Path B: JVM Ecosystem Bundles**), documents all 107 symbols and 13 structs in `hc4x.h`, and provides the mandatory 10-point conformance checklist.
