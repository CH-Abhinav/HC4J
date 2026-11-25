//! C-ABI entry points for matrix multiplication, batched GEMM, mixed `f32 x f16`,
//! matrix transpose, and matmul kernel planning diagnostics (6 symbols).

use hc4rs::engine::error::{HC4J_SUCCESS, Hc4jError, ffi_guard};
use hc4rs::engine::get_engine;
use hc4rs::kernels::linalg::{
    Caps, matmul_plan, run_batched_matmul, run_matmul, run_matmul_f32_f16, run_transpose,
};

/// # Safety
/// Caller ensures valid `TensorId`s sized for `(m, k)`, `(k, n)`, `(m, n)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_matmul_f32(id_a: u64, id_b: u64, id_out: u64, m: u32, n: u32, k: u32) -> i32 {
    ffi_guard("dispatch_matmul_f32", || run_matmul(id_a, id_b, id_out, m, n, k, 0))
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_matmul_f32_ex(
    id_a: u64,
    id_b: u64,
    id_out: u64,
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
) -> i32 {
    ffi_guard("dispatch_matmul_f32_ex", || run_matmul(id_a, id_b, id_out, m, n, k, flags))
}

/// # Safety
/// `ptr_batch_shape`, `ptr_a_batch_strides`, and `ptr_b_batch_strides` must each
/// point to `batch_rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_batched_matmul_f32(
    id_a: u64,
    id_b: u64,
    id_out: u64,
    m: u32,
    n: u32,
    k: u32,
    batch_rank: u32,
    ptr_batch_shape: *const u32,
    ptr_a_batch_strides: *const u32,
    ptr_b_batch_strides: *const u32,
    flags: u32,
) -> i32 {
    ffi_guard("dispatch_batched_matmul_f32", || {
        if ptr_batch_shape.is_null() || ptr_a_batch_strides.is_null() || ptr_b_batch_strides.is_null() {
            return Err(Hc4jError::InvalidParam("null pointer in batched matmul"));
        }
        if batch_rank == 0 || batch_rank > 4 {
            return Err(Hc4jError::InvalidParam("batch rank must be in 1..=4"));
        }
        let r = batch_rank as usize;
        let batch_shape = unsafe { std::slice::from_raw_parts(ptr_batch_shape, r) };
        let a_strides = unsafe { std::slice::from_raw_parts(ptr_a_batch_strides, r) };
        let b_strides = unsafe { std::slice::from_raw_parts(ptr_b_batch_strides, r) };
        run_batched_matmul(
            id_a,
            id_b,
            id_out,
            m,
            n,
            k,
            batch_rank,
            batch_shape,
            a_strides,
            b_strides,
            flags,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_matmul_f32_f16(
    id_a: u64,
    id_b_f16: u64,
    id_out: u64,
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
) -> i32 {
    ffi_guard("dispatch_matmul_f32_f16", || {
        run_matmul_f32_f16(id_a, id_b_f16, id_out, m, n, k, flags)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_transpose_f32(id_in: u64, id_out: u64, rows: u32, cols: u32) -> i32 {
    ffi_guard("dispatch_transpose_f32", || run_transpose(id_in, id_out, rows, cols))
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_matmul_plan(m: u32, n: u32, k: u32, flags: u32) -> i32 {
    let mut code = 0;
    let status = ffi_guard("hc4j_matmul_plan", || {
        let engine = get_engine()?;
        code = matmul_plan(&Caps::of(engine), m, n, k, flags)?.code();
        Ok(())
    });
    if status == HC4J_SUCCESS { code } else { status }
}
