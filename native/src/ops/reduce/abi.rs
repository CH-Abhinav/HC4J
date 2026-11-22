use crate::error::ffi_guard;

use super::dispatch::{run_full, run_row_argmax, run_rows};
use super::types::{Load, ReduceOp};

/// `out[0] = reduce(a)` over `length` elements. `op` is 0 sum, 1 mean,
/// 2 max, 3 min.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_reduce_f32(id_a: u64, id_out: u64, length: u64, op: u32) -> i32 {
    ffi_guard("dispatch_reduce_f32", || {
        run_full(ReduceOp::from_code(op)?, Load::Value, id_a, 0, id_out, length)
    })
}

/// `out[0] = reduce(a * a)`, the fold an L2 norm or a variance needs.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_reduce_square_f32(id_a: u64, id_out: u64, length: u64, op: u32) -> i32 {
    ffi_guard("dispatch_reduce_square_f32", || {
        run_full(ReduceOp::from_code(op)?, Load::Square, id_a, 0, id_out, length)
    })
}

/// `out[0] = reduce((a - b)^2)`. With `op = 1` (mean) this is MSE.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_reduce_diff2_f32(id_a: u64, id_b: u64, id_out: u64, length: u64, op: u32) -> i32 {
    ffi_guard("dispatch_reduce_diff2_f32", || {
        run_full(ReduceOp::from_code(op)?, Load::Diff2, id_a, id_b, id_out, length)
    })
}

/// `out[r] = reduce(a[r, :])` for a `[rows, cols]` tensor.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_reduce_rows_f32(id_a: u64, id_out: u64, rows: u32, cols: u32, op: u32) -> i32 {
    ffi_guard("dispatch_reduce_rows_f32", || {
        run_rows(ReduceOp::from_code(op)?, Load::Value, id_a, 0, id_out, rows, cols)
    })
}

/// `out[r] = argmax(a[r, :])` as i32, or argmin when `largest == 0`.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_argmax_rows(id_a: u64, id_out: u64, rows: u32, cols: u32, largest: u32) -> i32 {
    ffi_guard("dispatch_argmax_rows", || {
        run_row_argmax(id_a, id_out, rows, cols, largest != 0)
    })
}

/// Reduces `id_a` along an arbitrary axis, where `outer = prod(shape[..axis])`,
/// `axis_size = shape[axis]`, and `inner = prod(shape[axis+1..])`.
///
/// # Safety
/// No pointer parameters; `unsafe` matches the published C ABI signature.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_reduce_axis_f32(
    id_a: u64,
    id_out: u64,
    outer: u32,
    axis_size: u32,
    inner: u32,
    op: u32,
) -> i32 {
    ffi_guard("dispatch_reduce_axis_f32", || {
        let reduce_op = ReduceOp::from_code(op)?;
        super::dispatch::run_axis_reduce_f32(reduce_op, id_a, id_out, outer, axis_size, inner)
    })
}

/// Computes argmax or argmin along an arbitrary axis.
///
/// # Safety
/// No pointer parameters; `unsafe` matches the published C ABI signature.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_argmax_axis(
    id_a: u64,
    id_out: u64,
    outer: u32,
    axis_size: u32,
    inner: u32,
    largest: u32,
) -> i32 {
    ffi_guard("dispatch_argmax_axis", || {
        super::dispatch::run_axis_argmax(id_a, id_out, outer, axis_size, inner, largest != 0)
    })
}
