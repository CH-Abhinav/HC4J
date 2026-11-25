//! Exponential and logarithmic suite (f32, elementwise).

use crate::engine::error::Hc4jResult;
use crate::kernels::elementwise::run_unary;
use crate::memory::TensorId;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExpOp {
    Exp,
    Ln,
    Log2,
    Log10,
    Sqrt,
}

impl ExpOp {
    pub fn name(&self) -> &'static str {
        match self {
            ExpOp::Exp => "exp",
            ExpOp::Ln => "ln",
            ExpOp::Log2 => "log2",
            ExpOp::Log10 => "log10",
            ExpOp::Sqrt => "sqrt",
        }
    }

    pub fn wgsl_op(&self) -> &'static str {
        match self {
            ExpOp::Exp => "exp",
            ExpOp::Ln => "log",
            ExpOp::Log2 => "log2",
            // log10(x) = (1 / ln(10)) * ln(x); spliced as `OP(x)` so it must be a callable prefix.
            ExpOp::Log10 => "0.4342944819032518 * log",
            ExpOp::Sqrt => "sqrt",
        }
    }
}

pub fn run_exp_op(
    op: ExpOp,
    id_a: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<(&[u32], &[u32], &[u32])>,
) -> Hc4jResult<()> {
    run_unary("exponential", op.name(), op.wgsl_op(), id_a, id_out, length, strided)
}

pub fn run_exp_f32(
    id_a: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<(&[u32], &[u32], &[u32])>,
) -> Hc4jResult<()> {
    run_exp_op(ExpOp::Exp, id_a, id_out, length, strided)
}

pub fn run_ln_f32(
    id_a: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<(&[u32], &[u32], &[u32])>,
) -> Hc4jResult<()> {
    run_exp_op(ExpOp::Ln, id_a, id_out, length, strided)
}

pub fn run_log2_f32(
    id_a: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<(&[u32], &[u32], &[u32])>,
) -> Hc4jResult<()> {
    run_exp_op(ExpOp::Log2, id_a, id_out, length, strided)
}

pub fn run_log10_f32(
    id_a: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<(&[u32], &[u32], &[u32])>,
) -> Hc4jResult<()> {
    run_exp_op(ExpOp::Log10, id_a, id_out, length, strided)
}

pub fn run_sqrt_f32(
    id_a: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<(&[u32], &[u32], &[u32])>,
) -> Hc4jResult<()> {
    run_exp_op(ExpOp::Sqrt, id_a, id_out, length, strided)
}
