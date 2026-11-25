//! C-ABI entry points for neural-network layers, reductions, transformer
//! primitives (embedding, RoPE, KV cache), and fused expression dispatch (17 symbols).

use hc4rs::engine::error::{Hc4jError, Hc4jResult, ffi_guard};
use hc4rs::kernels::elementwise::UnaryStrided;
use hc4rs::kernels::fusion::run_fused_dispatch;
use hc4rs::kernels::nn::{
    run_activation_f32, run_cross_entropy_rows, run_embedding, run_kv_cache_append_f32, run_layernorm_rows_f32,
    run_rmsnorm_rows_f32, run_rope_f32, run_row_bias, run_softmax_rows_f32,
};
use hc4rs::kernels::reduce::{
    Load, ReduceOp, run_axis_argmax, run_axis_reduce_f32, run_full, run_row_argmax, run_rows,
};

unsafe fn unary_layout<'a>(
    is_contiguous: u32,
    rank: u32,
    shape: *const u32,
    sa: *const u32,
    sc: *const u32,
) -> Hc4jResult<Option<UnaryStrided<'a>>> {
    if is_contiguous != 0 {
        return Ok(None);
    }
    if rank == 0 {
        return Ok(Some((&[], &[], &[])));
    }
    if rank > 8 {
        return Err(Hc4jError::Unsupported("rank above 8"));
    }
    if shape.is_null() || sa.is_null() || sc.is_null() {
        return Err(Hc4jError::InvalidParam("null shape/stride pointer"));
    }
    let r = rank as usize;
    Ok(Some((
        unsafe { std::slice::from_raw_parts(shape, r) },
        unsafe { std::slice::from_raw_parts(sa, r) },
        unsafe { std::slice::from_raw_parts(sc, r) },
    )))
}

/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, layout pointers must point to `rank` readable `u32` values.
#[unsafe(no_mangle)]
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
    ffi_guard("dispatch_activation_f32", || {
        let layout = unsafe { unary_layout(is_contiguous, rank, ptr_shape, ptr_strides_a, ptr_strides_c)? };
        run_activation_f32(id_a, id_out, length, layout, act, alpha_bits)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_softmax_rows_f32(
    id_a: u64,
    id_out: u64,
    rows: u32,
    cols: u32,
    log_variant: u32,
) -> i32 {
    ffi_guard("dispatch_softmax_rows_f32", || {
        run_softmax_rows_f32(id_a, id_out, rows, cols, log_variant != 0)
    })
}

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
        run_layernorm_rows_f32(id_a, id_gamma, id_beta, id_out, rows, cols, eps_bits)
    })
}

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
        run_rmsnorm_rows_f32(id_a, id_gamma, id_out, rows, cols, eps_bits)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_cross_entropy_rows(
    id_logits: u64,
    id_targets: u64,
    id_out: u64,
    rows: u32,
    cols: u32,
) -> i32 {
    ffi_guard("dispatch_cross_entropy_rows", || {
        run_cross_entropy_rows(id_logits, id_targets, id_out, rows, cols)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_add_row_bias_f32(
    id_a: u64,
    id_bias: u64,
    id_out: u64,
    rows: u32,
    cols: u32,
) -> i32 {
    ffi_guard("dispatch_add_row_bias_f32", || {
        run_row_bias(id_a, id_bias, id_out, rows, cols)
    })
}

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
        run_kv_cache_append_f32(
            id_src,
            id_cache,
            num_kv_heads,
            max_seq_len,
            head_dim,
            seq_len,
            pos_offset,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_embedding(
    id_table: u64,
    id_indices: u64,
    id_out: u64,
    vocab_size: u32,
    hidden_dim: u32,
    num_tokens: u32,
    table_is_f16: u32,
) -> i32 {
    ffi_guard("dispatch_embedding", || {
        run_embedding(
            id_table,
            id_indices,
            id_out,
            vocab_size,
            hidden_dim,
            num_tokens,
            table_is_f16 != 0,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_rope_f32(
    id_in: u64,
    id_out: u64,
    batch: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    pos_offset: u32,
    theta_bits: u32,
) -> i32 {
    ffi_guard("dispatch_rope_f32", || {
        run_rope_f32(
            id_in,
            id_out,
            batch,
            seq_len,
            num_heads,
            head_dim,
            pos_offset,
            theta_bits,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_reduce_f32(id_a: u64, id_out: u64, length: u64, op: u32) -> i32 {
    ffi_guard("dispatch_reduce_f32", || {
        run_full(ReduceOp::from_code(op)?, Load::Value, id_a, 0, id_out, length)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_reduce_square_f32(id_a: u64, id_out: u64, length: u64, op: u32) -> i32 {
    ffi_guard("dispatch_reduce_square_f32", || {
        run_full(ReduceOp::from_code(op)?, Load::Square, id_a, 0, id_out, length)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_reduce_diff2_f32(
    id_a: u64,
    id_b: u64,
    id_out: u64,
    length: u64,
    op: u32,
) -> i32 {
    ffi_guard("dispatch_reduce_diff2_f32", || {
        run_full(ReduceOp::from_code(op)?, Load::Diff2, id_a, id_b, id_out, length)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_reduce_rows_f32(
    id_a: u64,
    id_out: u64,
    rows: u32,
    cols: u32,
    op: u32,
) -> i32 {
    ffi_guard("dispatch_reduce_rows_f32", || {
        run_rows(ReduceOp::from_code(op)?, Load::Value, id_a, 0, id_out, rows, cols)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_argmax_rows(
    id_a: u64,
    id_out: u64,
    rows: u32,
    cols: u32,
    largest: u32,
) -> i32 {
    ffi_guard("dispatch_argmax_rows", || {
        run_row_argmax(id_a, id_out, rows, cols, largest != 0)
    })
}

/// # Safety
/// No pointer parameters; `unsafe` preserves the C ABI signature.
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
        run_axis_reduce_f32(reduce_op, id_a, id_out, outer, axis_size, inner)
    })
}

/// # Safety
/// No pointer parameters; `unsafe` preserves the C ABI signature.
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
        run_axis_argmax(id_a, id_out, outer, axis_size, inner, largest != 0)
    })
}

/// # Safety
/// `program` must point to `program_len` readable `u32`s and `inputs` to `n_inputs` readable `u64` handles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_fused_dispatch(
    program: *const u32,
    program_len: u32,
    inputs: *const u64,
    n_inputs: u32,
    id_out: u64,
) -> i32 {
    ffi_guard("hc4j_fused_dispatch", || {
        if program.is_null() || inputs.is_null() || program_len == 0 || n_inputs == 0 {
            return Err(Hc4jError::InvalidParam("null or empty fused program"));
        }
        let program_slice = unsafe { std::slice::from_raw_parts(program, program_len as usize) };
        let input_slice = unsafe { std::slice::from_raw_parts(inputs, n_inputs as usize) };
        run_fused_dispatch(program_slice, input_slice, id_out)
    })
}
