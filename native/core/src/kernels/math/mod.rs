//! Elementwise math kernels cleanly organized by domain:
//! arithmetic, exponential/log, trig/hyperbolic, unary math, compare/where, and casts.

pub mod arithmetic;
pub mod cast;
pub mod compare_mask;
pub mod exponential;
pub mod trig;
pub mod unary;

pub use arithmetic::{BinaryOp, DataType, ELEM_SHADER_TEMPLATE, generate_elem_shader, run_add_f32, run_binary, run_div_f32, run_mul_f32, run_sub_f32};
pub use cast::{
    CAST_F16_TO_F32_WGSL, CAST_F32_TO_F16_WGSL, CAST_F32_TO_I32_WGSL, CAST_I32_TO_F32_WGSL, run_cast,
    run_cast_f16_to_f32, run_cast_f32_to_f16,
};
pub use compare_mask::{
    COMPARE_SHADER_TEMPLATE, CompareOp, FUNCTIONAL_BINARY_SHADER_TEMPLATE, WHERE_SHADER_TEMPLATE,
    generate_compare_shader, generate_functional_binary_shader, generate_where_shader, run_compare, run_maximum,
    run_minimum, run_pow_f32, run_where,
};
pub use exponential::{ExpOp, run_exp_f32, run_exp_op, run_ln_f32, run_log2_f32, run_log10_f32, run_sqrt_f32};
pub use trig::{TrigOp, run_trig_op};
pub use unary::{
    F32UnaryOp, I32UnaryOp, SIGN_F32_PRELUDE, UNARY_I32_SHADER_TEMPLATE, generate_unary_i32_shader, run_abs_f32,
    run_ceil_f32, run_clamp_f32, run_f32_unary, run_floor_f32, run_i32_unary, run_neg_f32, run_round_f32,
    run_rsqrt_f32, run_sign_f32,
};
