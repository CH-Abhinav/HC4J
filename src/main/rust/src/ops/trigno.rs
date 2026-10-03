//! Trigonometric suite (f32, elementwise).
//!
//! Every op uses the unified caller-allocated elementwise ABI:
//!
//! ```c
//! int32_t dispatch_<op>_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
//!         const uint32_t* shape, const uint32_t* strides_a, const uint32_t* strides_c,
//!         size_t length, uint32_t is_contiguous);
//! ```
//!
//! Contiguous calls pass null layout pointers and `is_contiguous = 1`;
//! `id_out == id_a` runs in place. The shared WGSL template, 2-D grid fold
//! and executors live in [`crate::ops::elementwise`].

use crate::ops::elementwise::dispatch_unary;
pub use crate::ops::elementwise::{UNARY_F32_SHADER_TEMPLATE, UnaryElemDims, generate_unary_shader};

macro_rules! impl_trig_op {
    ($fn_name:ident, $op:expr) => {
        /// # Safety
        /// For strided calls, `shape`, `strides_a` and `strides_c` must each
        /// point to `rank` readable `u32` values.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            id_a: u64, id_out: u64, rank: u32,
            ptr_shape: *const u32, ptr_strides_a: *const u32, ptr_strides_c: *const u32,
            length: usize, is_contiguous: u32,
        ) -> i32 {
            unsafe {
                dispatch_unary(
                    "trig", $op, $op, id_a, id_out, rank,
                    ptr_shape, ptr_strides_a, ptr_strides_c, length, is_contiguous,
                )
            }
        }
    };
}

// Standard Trigonometric
impl_trig_op!(dispatch_sin_f32, "sin");
impl_trig_op!(dispatch_cos_f32, "cos");
impl_trig_op!(dispatch_tan_f32, "tan");

// Inverse Trigonometric
impl_trig_op!(dispatch_asin_f32, "asin");
impl_trig_op!(dispatch_acos_f32, "acos");
impl_trig_op!(dispatch_atan_f32, "atan");

// Hyperbolic
impl_trig_op!(dispatch_sinh_f32, "sinh");
impl_trig_op!(dispatch_cosh_f32, "cosh");
impl_trig_op!(dispatch_tanh_f32, "tanh");

// Inverse Hyperbolic
impl_trig_op!(dispatch_asinh_f32, "asinh");
impl_trig_op!(dispatch_acosh_f32, "acosh");
impl_trig_op!(dispatch_atanh_f32, "atanh");
