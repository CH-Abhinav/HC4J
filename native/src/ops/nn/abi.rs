//! C ABI exports for neural network operations.

use crate::error::{Hc4jError, ffi_guard};
use crate::memory::TensorId;
use crate::ops::elementwise::dispatch_unary_kernel;
use crate::ops::nn::activations::Activation;
use crate::ops::nn::bias::run_row_bias;
use crate::ops::nn::kv_cache::run_kv_cache_append_f32;
use crate::ops::nn::row_ops::{RowOp, run_row_op};

/// Elementwise activation, on the shared caller-allocated elementwise ABI
/// plus `act` (see `Activation::from_code`) and `alpha_bits`, the raw f32 bits
/// of the slope or alpha parameter.
///
/// # Safety
/// For strided calls, `shape`, `strides_a` and `strides_c` must each point to
/// `rank` readable `u32` values.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn dispatch_activation_f32(
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
    act: u32,
    alpha_bits: u32,
) -> i32 {
    let activation = match Activation::from_code(act) {
        Ok(a) => a,
        Err(err) => return err.code(),
    };
    let prelude = activation.prelude();
    let kernel = activation.kernel(&prelude, f32::from_bits(alpha_bits));
    unsafe {
        dispatch_unary_kernel(
            kernel,
            id_a,
            id_out,
            rank,
            ptr_shape,
            ptr_strides_a,
            ptr_strides_c,
            length,
            is_contiguous,
        )
    }
}

/// Row-wise softmax (`log_variant == 0`) or log-softmax over `[rows, cols]`.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_softmax_rows_f32(id_a: u64, id_out: u64, rows: u32, cols: u32, log_variant: u32) -> i32 {
    ffi_guard("dispatch_softmax_rows_f32", || {
        let op = if log_variant == 0 { RowOp::Softmax } else { RowOp::LogSoftmax };
        run_row_op(op, id_a, &[], id_out, rows, cols, 0)
    })
}

/// Row-wise layer norm. `id_gamma`/`id_beta` of 0 skip the affine transform;
/// otherwise both are required and must have `cols` elements.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_layernorm_rows_f32(
    id_a: u64,
    id_gamma: u64,
    id_beta: u64,
    id_out: u64,
    rows: u32,
    cols: u32,
    eps_bits: u32,
) -> i32 {
    ffi_guard("dispatch_layernorm_rows_f32", || {
        let affine = id_gamma != 0 || id_beta != 0;
        if affine && (id_gamma == 0 || id_beta == 0) {
            return Err(Hc4jError::InvalidParam("layer norm needs both gamma and beta, or neither"));
        }
        let extra: Vec<TensorId> = if affine { vec![id_gamma, id_beta] } else { Vec::new() };
        run_row_op(RowOp::LayerNorm { affine }, id_a, &extra, id_out, rows, cols, eps_bits)
    })
}

/// Row-wise RMS norm. `id_gamma` of 0 skips the scale.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_rmsnorm_rows_f32(
    id_a: u64,
    id_gamma: u64,
    id_out: u64,
    rows: u32,
    cols: u32,
    eps_bits: u32,
) -> i32 {
    ffi_guard("dispatch_rmsnorm_rows_f32", || {
        let affine = id_gamma != 0;
        let extra: Vec<TensorId> = if affine { vec![id_gamma] } else { Vec::new() };
        run_row_op(RowOp::RmsNorm { affine }, id_a, &extra, id_out, rows, cols, eps_bits)
    })
}

/// Per-row softmax cross entropy from logits `[rows, cols]` and i32 class
/// indices `[rows]`, into `[rows]`. An out-of-range index yields NaN.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_cross_entropy_rows(id_logits: u64, id_targets: u64, id_out: u64, rows: u32, cols: u32) -> i32 {
    ffi_guard("dispatch_cross_entropy_rows", || {
        run_row_op(RowOp::CrossEntropy, id_logits, &[id_targets], id_out, rows, cols, 0)
    })
}

/// `out[r, c] = a[r, c] + bias[c]`, the bias half of a linear layer.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_add_row_bias_f32(id_a: u64, id_bias: u64, id_out: u64, rows: u32, cols: u32) -> i32 {
    ffi_guard("dispatch_add_row_bias_f32", || run_row_bias(id_a, id_bias, id_out, rows, cols))
}

/// Appends `[seq_len, num_kv_heads, head_dim]` states from `id_src` into
/// `[num_kv_heads, max_seq_len, head_dim]` `id_cache` at `pos_offset`.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_kv_cache_append_f32(
    id_src: u64,
    id_cache: u64,
    num_kv_heads: u32,
    max_seq_len: u32,
    head_dim: u32,
    seq_len: u32,
    pos_offset: u32,
) -> i32 {
    ffi_guard("dispatch_kv_cache_append_f32", || {
        run_kv_cache_append_f32(id_src, id_cache, num_kv_heads, max_seq_len, head_dim, seq_len, pos_offset)
    })
}
