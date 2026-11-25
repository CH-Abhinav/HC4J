//! Trigonometric and hyperbolic suite (f32, elementwise).

use crate::engine::error::Hc4jResult;
use crate::kernels::elementwise::run_unary;
pub use crate::kernels::elementwise::{UNARY_F32_SHADER_TEMPLATE, UnaryElemDims, generate_unary_shader};
use crate::memory::TensorId;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TrigOp {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
}

impl TrigOp {
    pub fn name(&self) -> &'static str {
        match self {
            TrigOp::Sin => "sin",
            TrigOp::Cos => "cos",
            TrigOp::Tan => "tan",
            TrigOp::Asin => "asin",
            TrigOp::Acos => "acos",
            TrigOp::Atan => "atan",
            TrigOp::Sinh => "sinh",
            TrigOp::Cosh => "cosh",
            TrigOp::Tanh => "tanh",
            TrigOp::Asinh => "asinh",
            TrigOp::Acosh => "acosh",
            TrigOp::Atanh => "atanh",
        }
    }
}

pub fn run_trig_op(
    op: TrigOp,
    id_a: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<(&[u32], &[u32], &[u32])>,
) -> Hc4jResult<()> {
    let name = op.name();
    run_unary("trig", name, name, id_a, id_out, length, strided)
}
