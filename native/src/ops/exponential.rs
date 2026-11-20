//! Exponential and logarithmic suite (f32, elementwise).
//!
//! Same caller-allocated ABI, template, 2-D grid fold and executors as the
//! trigonometric suite ([`crate::ops::elementwise`]):
//!
//! ```c
//! int32_t dispatch_<op>_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
//!         const uint32_t* shape, const uint32_t* strides_a, const uint32_t* strides_c,
//!         size_t length, uint32_t is_contiguous);
//! ```

use crate::ops::elementwise::dispatch_unary;

macro_rules! impl_exponential_op {
    ($fn_name:ident, $op_name:expr, $wgsl_op:expr) => {
        /// # Safety
        /// For strided calls, `shape`, `strides_a` and `strides_c` must each
        /// point to `rank` readable `u32` values.
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
            unsafe {
                dispatch_unary(
                    "exponential",
                    $op_name,
                    $wgsl_op,
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
    };
}

// ============================================================================
// EXPONENTIAL
// ============================================================================

impl_exponential_op!(dispatch_exp_f32, "exp", "exp");

// ============================================================================
// NATURAL LOGARITHM
// ln(x) = log(x)
// ============================================================================

impl_exponential_op!(dispatch_ln_f32, "ln", "log");

// ============================================================================
// BASE-2 LOGARITHM
// ============================================================================

impl_exponential_op!(dispatch_log2_f32, "log2", "log2");

// ============================================================================
// BASE-10 LOGARITHM
// log10(x) = (1 / ln(10)) * ln(x). The op string is spliced in as `OP(x)`, so
// it must be a callable prefix, not a full expression.
// ============================================================================

impl_exponential_op!(dispatch_log10_f32, "log10", "0.4342944819032518 * log");

// ============================================================================
// SQUARE ROOT
// ============================================================================

impl_exponential_op!(dispatch_sqrt_f32, "sqrt", "sqrt");
