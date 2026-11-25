//! C-ABI entry points for elementwise math, trig, exponential/log,
//! unary math, comparisons, ternary selection, strided copy, and type casts (43 symbols).

use hc4rs::engine::error::{Hc4jError, Hc4jResult, ffi_guard};
use hc4rs::kernels::elementwise::{BinaryStrided, TernaryStrided, UnaryStrided, run_copy_strided};
use hc4rs::kernels::math::{
    BinaryOp, ExpOp, F32UnaryOp, TrigOp, run_binary, run_cast, run_cast_f16_to_f32, run_cast_f32_to_f16,
    run_compare, run_exp_op, run_f32_unary, run_i32_unary, run_maximum, run_minimum, run_pow_f32, run_trig_op,
    run_where,
};

unsafe fn raw_slice<'a>(ptr: *const u32, rank: u32) -> Hc4jResult<&'a [u32]> {
    if rank == 0 {
        return Ok(&[]);
    }
    if rank > 8 {
        return Err(Hc4jError::Unsupported("rank above 8"));
    }
    if ptr.is_null() {
        return Err(Hc4jError::InvalidParam("null shape/stride pointer"));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, rank as usize) })
}

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
    Ok(Some((
        unsafe { raw_slice(shape, rank)? },
        unsafe { raw_slice(sa, rank)? },
        unsafe { raw_slice(sc, rank)? },
    )))
}

unsafe fn binary_layout<'a>(
    is_contiguous: u32,
    rank: u32,
    shape: *const u32,
    sa: *const u32,
    sb: *const u32,
    sc: *const u32,
) -> Hc4jResult<Option<BinaryStrided<'a>>> {
    if is_contiguous != 0 {
        return Ok(None);
    }
    Ok(Some((
        unsafe { raw_slice(shape, rank)? },
        unsafe { raw_slice(sa, rank)? },
        unsafe { raw_slice(sb, rank)? },
        unsafe { raw_slice(sc, rank)? },
    )))
}

unsafe fn ternary_layout<'a>(
    is_contiguous: u32,
    rank: u32,
    shape: *const u32,
    s0: *const u32,
    s1: *const u32,
    s2: *const u32,
    s3: *const u32,
) -> Hc4jResult<Option<TernaryStrided<'a>>> {
    if is_contiguous != 0 {
        return Ok(None);
    }
    Ok(Some((
        unsafe { raw_slice(shape, rank)? },
        unsafe { raw_slice(s0, rank)? },
        unsafe { raw_slice(s1, rank)? },
        unsafe { raw_slice(s2, rank)? },
        unsafe { raw_slice(s3, rank)? },
    )))
}

macro_rules! impl_binary_arith {
    ($fn_name:ident, $fn_f32:ident, $op_name:expr, $variant:expr) => {
        /// # Safety
        /// When `is_contiguous == 0` and `rank > 0`, layout pointers must point to `rank` readable `u32` values.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            id_a: u64,
            id_b: u64,
            id_out: u64,
            rank: u32,
            ptr_shape: *const u32,
            ptr_strides_a: *const u32,
            ptr_strides_b: *const u32,
            ptr_strides_c: *const u32,
            length: usize,
            is_contiguous: u32,
            dtype_code: u32,
        ) -> i32 {
            ffi_guard($op_name, || {
                let layout = unsafe { binary_layout(is_contiguous, rank, ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c)? };
                run_binary($variant, dtype_code, id_a, id_b, id_out, length, layout)
            })
        }

        /// # Safety
        /// When `is_contiguous == 0` and `rank > 0`, layout pointers must point to `rank` readable `u32` values.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_f32(
            id_a: u64,
            id_b: u64,
            id_out: u64,
            rank: u32,
            ptr_shape: *const u32,
            ptr_strides_a: *const u32,
            ptr_strides_b: *const u32,
            ptr_strides_c: *const u32,
            length: usize,
            is_contiguous: u32,
        ) -> i32 {
            unsafe {
                $fn_name(
                    id_a, id_b, id_out, rank, ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c, length,
                    is_contiguous, 1,
                )
            }
        }
    };
}

impl_binary_arith!(dispatch_add, dispatch_add_f32, "dispatch_add", BinaryOp::Add);
impl_binary_arith!(dispatch_sub, dispatch_sub_f32, "dispatch_sub", BinaryOp::Sub);
impl_binary_arith!(dispatch_mul, dispatch_mul_f32, "dispatch_mul", BinaryOp::Mul);
impl_binary_arith!(dispatch_div, dispatch_div_f32, "dispatch_div", BinaryOp::Div);

macro_rules! impl_trig_ffi {
    ($fn_name:ident, $op_name:expr, $variant:expr) => {
        /// # Safety
        /// When `is_contiguous == 0` and `rank > 0`, layout pointers must point to `rank` readable `u32` values.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            id_a: u64,
            id_out: u64,
            rank: u32,
            ptr_shape: *const u32,
            ptr_strides_a: *const u32,
            ptr_strides_c: *const u32,
            length: usize,
            is_contiguous: u32,
        ) -> i32 {
            ffi_guard($op_name, || {
                let layout = unsafe { unary_layout(is_contiguous, rank, ptr_shape, ptr_strides_a, ptr_strides_c)? };
                run_trig_op($variant, id_a, id_out, length, layout)
            })
        }
    };
}

impl_trig_ffi!(dispatch_sin_f32, "dispatch_sin_f32", TrigOp::Sin);
impl_trig_ffi!(dispatch_cos_f32, "dispatch_cos_f32", TrigOp::Cos);
impl_trig_ffi!(dispatch_tan_f32, "dispatch_tan_f32", TrigOp::Tan);
impl_trig_ffi!(dispatch_asin_f32, "dispatch_asin_f32", TrigOp::Asin);
impl_trig_ffi!(dispatch_acos_f32, "dispatch_acos_f32", TrigOp::Acos);
impl_trig_ffi!(dispatch_atan_f32, "dispatch_atan_f32", TrigOp::Atan);
impl_trig_ffi!(dispatch_sinh_f32, "dispatch_sinh_f32", TrigOp::Sinh);
impl_trig_ffi!(dispatch_cosh_f32, "dispatch_cosh_f32", TrigOp::Cosh);
impl_trig_ffi!(dispatch_tanh_f32, "dispatch_tanh_f32", TrigOp::Tanh);
impl_trig_ffi!(dispatch_asinh_f32, "dispatch_asinh_f32", TrigOp::Asinh);
impl_trig_ffi!(dispatch_acosh_f32, "dispatch_acosh_f32", TrigOp::Acosh);
impl_trig_ffi!(dispatch_atanh_f32, "dispatch_atanh_f32", TrigOp::Atanh);

macro_rules! impl_exp_ffi {
    ($fn_name:ident, $op_name:expr, $variant:expr) => {
        /// # Safety
        /// When `is_contiguous == 0` and `rank > 0`, layout pointers must point to `rank` readable `u32` values.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            id_a: u64,
            id_out: u64,
            rank: u32,
            ptr_shape: *const u32,
            ptr_strides_a: *const u32,
            ptr_strides_c: *const u32,
            length: usize,
            is_contiguous: u32,
        ) -> i32 {
            ffi_guard($op_name, || {
                let layout = unsafe { unary_layout(is_contiguous, rank, ptr_shape, ptr_strides_a, ptr_strides_c)? };
                run_exp_op($variant, id_a, id_out, length, layout)
            })
        }
    };
}

impl_exp_ffi!(dispatch_exp_f32, "dispatch_exp_f32", ExpOp::Exp);
impl_exp_ffi!(dispatch_ln_f32, "dispatch_ln_f32", ExpOp::Ln);
impl_exp_ffi!(dispatch_log2_f32, "dispatch_log2_f32", ExpOp::Log2);
impl_exp_ffi!(dispatch_log10_f32, "dispatch_log10_f32", ExpOp::Log10);
impl_exp_ffi!(dispatch_sqrt_f32, "dispatch_sqrt_f32", ExpOp::Sqrt);

macro_rules! impl_f32_unary_ffi {
    ($fn_name:ident, $op_name:expr, $variant:expr) => {
        /// # Safety
        /// When `is_contiguous == 0` and `rank > 0`, layout pointers must point to `rank` readable `u32` values.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            id_a: u64,
            id_out: u64,
            rank: u32,
            ptr_shape: *const u32,
            ptr_strides_a: *const u32,
            ptr_strides_c: *const u32,
            length: usize,
            is_contiguous: u32,
        ) -> i32 {
            ffi_guard($op_name, || {
                let layout = unsafe { unary_layout(is_contiguous, rank, ptr_shape, ptr_strides_a, ptr_strides_c)? };
                run_f32_unary($variant, id_a, id_out, length, layout)
            })
        }
    };
}

impl_f32_unary_ffi!(dispatch_abs_f32, "dispatch_abs_f32", F32UnaryOp::Abs);
impl_f32_unary_ffi!(dispatch_neg_f32, "dispatch_neg_f32", F32UnaryOp::Neg);
impl_f32_unary_ffi!(dispatch_rsqrt_f32, "dispatch_rsqrt_f32", F32UnaryOp::Rsqrt);
impl_f32_unary_ffi!(dispatch_floor_f32, "dispatch_floor_f32", F32UnaryOp::Floor);
impl_f32_unary_ffi!(dispatch_ceil_f32, "dispatch_ceil_f32", F32UnaryOp::Ceil);
impl_f32_unary_ffi!(dispatch_round_f32, "dispatch_round_f32", F32UnaryOp::Round);
impl_f32_unary_ffi!(dispatch_sign_f32, "dispatch_sign_f32", F32UnaryOp::Sign);

/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, layout pointers must point to `rank` readable `u32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_clamp_f32(
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
    lo: f32,
    hi: f32,
) -> i32 {
    ffi_guard("dispatch_clamp_f32", || {
        let layout = unsafe { unary_layout(is_contiguous, rank, ptr_shape, ptr_strides_a, ptr_strides_c)? };
        run_f32_unary(F32UnaryOp::Clamp { lo, hi }, id_a, id_out, length, layout)
    })
}

/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, layout pointers must point to `rank` readable `u32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_unary_i32(
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
    op_code: u32,
    p_z: u32,
    p_w: u32,
) -> i32 {
    ffi_guard("dispatch_unary_i32", || {
        let layout = unsafe { unary_layout(is_contiguous, rank, ptr_shape, ptr_strides_a, ptr_strides_c)? };
        run_i32_unary(op_code, id_a, id_out, length, layout, p_z, p_w)
    })
}

/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, layout pointers must point to `rank` readable `u32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_pow_f32(
    id_a: u64,
    id_b: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_b: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
) -> i32 {
    ffi_guard("dispatch_pow_f32", || {
        let layout = unsafe { binary_layout(is_contiguous, rank, ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c)? };
        run_pow_f32(id_a, id_b, id_out, length, layout)
    })
}

/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, layout pointers must point to `rank` readable `u32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_maximum(
    id_a: u64,
    id_b: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_b: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
    dtype_code: u32,
) -> i32 {
    ffi_guard("dispatch_maximum", || {
        let layout = unsafe { binary_layout(is_contiguous, rank, ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c)? };
        run_maximum(id_a, id_b, id_out, length, layout, dtype_code)
    })
}

/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, layout pointers must point to `rank` readable `u32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_minimum(
    id_a: u64,
    id_b: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_b: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
    dtype_code: u32,
) -> i32 {
    ffi_guard("dispatch_minimum", || {
        let layout = unsafe { binary_layout(is_contiguous, rank, ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c)? };
        run_minimum(id_a, id_b, id_out, length, layout, dtype_code)
    })
}

/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, layout pointers must point to `rank` readable `u32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_compare(
    id_a: u64,
    id_b: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_b: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
    dtype_code: u32,
    cmp_op: u32,
) -> i32 {
    ffi_guard("dispatch_compare", || {
        let layout = unsafe { binary_layout(is_contiguous, rank, ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c)? };
        run_compare(id_a, id_b, id_out, length, layout, dtype_code, cmp_op)
    })
}

/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, layout pointers must point to `rank` readable `u32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_where(
    id_cond: u64,
    id_x: u64,
    id_y: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_cond: *const u32,
    ptr_strides_x: *const u32,
    ptr_strides_y: *const u32,
    ptr_strides_out: *const u32,
    length: usize,
    is_contiguous: u32,
    cond_dtype: u32,
    val_dtype: u32,
) -> i32 {
    ffi_guard("dispatch_where", || {
        let layout = unsafe {
            ternary_layout(
                is_contiguous,
                rank,
                ptr_shape,
                ptr_strides_cond,
                ptr_strides_x,
                ptr_strides_y,
                ptr_strides_out,
            )?
        };
        run_where(id_cond, id_x, id_y, id_out, length, layout, cond_dtype, val_dtype)
    })
}

/// # Safety
/// Caller ensures valid `TensorId`s with at least `length` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_cast(
    id_src: u64,
    id_out: u64,
    length: u64,
    src_dtype: u32,
    dst_dtype: u32,
) -> i32 {
    ffi_guard("dispatch_cast", || run_cast(id_src, id_out, length, src_dtype, dst_dtype))
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_cast_f32_to_f16(id_src: u64, id_out: u64, length: u64) -> i32 {
    ffi_guard("dispatch_cast_f32_to_f16", || run_cast_f32_to_f16(id_src, id_out, length))
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_cast_f16_to_f32(id_src: u64, id_out: u64, length: u64) -> i32 {
    ffi_guard("dispatch_cast_f16_to_f32", || run_cast_f16_to_f32(id_src, id_out, length))
}

/// # Safety
/// If `rank > 0`, `shape` and `src_strides` must point to `rank` readable `u32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_copy_strided(
    src: u64,
    out: u64,
    src_offset: u64,
    rank: u32,
    shape: *const u32,
    src_strides: *const u32,
    len: usize,
) -> i32 {
    ffi_guard("dispatch_copy_strided", || {
        let shape_slice = unsafe { raw_slice(shape, rank)? };
        let stride_slice = unsafe { raw_slice(src_strides, rank)? };
        run_copy_strided(src, out, src_offset, shape_slice, stride_slice, len)
    })
}
