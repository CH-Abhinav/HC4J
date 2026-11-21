//! C ABI exports for matrix multiplication and transposition.

use crate::error::ffi_guard;
use crate::ops::matmul::dispatch::{run_matmul, run_transpose};
use crate::ops::matmul::plan::{Caps, plan};

/// `C[M×N] = A[M×K] · B[K×N]` into the caller-allocated `id_out`.
///
/// # Safety
/// No pointer arguments; `unsafe` matches the published signature. Handles
/// are validated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_matmul_f32(id_a: u64, id_b: u64, id_out: u64, m: u32, n: u32, k: u32) -> i32 {
    ffi_guard("dispatch_matmul_f32", || run_matmul(id_a, id_b, id_out, m, n, k, 0))
}

/// As `dispatch_matmul_f32`, with `flags` bit 0: A is stored transposed
/// (K×M), bit 1: B is stored transposed (N×K).
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_matmul_f32_ex(id_a: u64, id_b: u64, id_out: u64, m: u32, n: u32, k: u32, flags: u32) -> i32 {
    ffi_guard("dispatch_matmul_f32_ex", || run_matmul(id_a, id_b, id_out, m, n, k, flags))
}

/// `out[cols×rows] = transpose(in[rows×cols])`.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_transpose_f32(id_in: u64, id_out: u64, rows: u32, cols: u32) -> i32 {
    ffi_guard("dispatch_transpose_f32", || run_transpose(id_in, id_out, rows, cols))
}

/// The dispatcher's choice for a shape, as `Plan::code()`, or a negative
/// status code. Diagnostic: lets tests and tooling see which kernel runs.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_matmul_plan(m: u32, n: u32, k: u32, flags: u32) -> i32 {
    let mut code = 0;
    let status = ffi_guard("hc4j_matmul_plan", || {
        let engine = crate::get_engine()?;
        code = plan(&Caps::of(engine), m, n, k, flags)?.code();
        Ok(())
    });
    if status == crate::error::HC4J_SUCCESS { code } else { status }
}
