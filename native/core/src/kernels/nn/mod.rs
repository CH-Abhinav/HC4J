//! Neural-network ops: activations, softmax, normalisation, losses and bias
//! broadcast.
//!
//! **Activations** are ordinary elementwise ops and reuse the unary executor,
//! so they inherit tiling, the 2-D grid fold, in-place output and the
//! host-streaming fallback for tensors larger than VRAM. They are not WGSL
//! built-ins, so each one supplies a prelude defining a `vec4` and a scalar
//! function — WGSL has neither overloading nor generics. One body is written
//! per activation with `__T__` standing for the width, and both functions are
//! generated from it, so the two paths cannot drift apart.
//!
//! **Row-wise ops** (softmax, layer norm, RMS norm, cross entropy) run one
//! workgroup per row: a reduction over the row in workgroup memory, then a
//! write pass over the same row. Each is a single dispatch.
//!
//! Numerics follow the stable formulations:
//!
//! * softmax uses the *online* max-and-sum recurrence (Milakov & Gimelshein),
//!   so a row is read twice rather than three times, and no `exp` ever sees a
//!   positive argument.
//! * layer norm accumulates Welford triples `(n, mean, M2)` and combines them
//!   pairwise in the tree, so the variance never comes from `E[x²] - E[x]²`.
//! * sigmoid, SiLU and softplus are written so that every `exp` argument is
//!   non-positive, which is what makes them finite for large `|x|`.
//!
//! Row-wise operands must be VRAM-resident and within one storage binding.

pub mod activations;
pub mod bias;
pub mod embedding;
pub mod kv_cache;
pub mod rope;
pub mod row_ops;

pub use activations::*;
pub use bias::*;
pub use embedding::*;
pub use kv_cache::*;
pub use rope::*;
pub use row_ops::*;

use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::kernels::elementwise::run_unary_kernel;
use crate::memory::TensorId;

/// Dispatches an elementwise activation (`act_code` matching `Activation::from_code`).
pub fn run_activation_f32(
    id_a: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<(&[u32], &[u32], &[u32])>,
    act_code: u32,
    alpha_bits: u32,
) -> Hc4jResult<()> {
    let act = Activation::from_code(act_code)?;
    let alpha = f32::from_bits(alpha_bits);
    let prelude = act.prelude();
    run_unary_kernel(act.kernel(&prelude, alpha), id_a, id_out, length, strided)
}

/// Dispatches row-wise softmax (or log-softmax when `log_softmax` is true).
pub fn run_softmax_rows_f32(
    id_a: TensorId,
    id_out: TensorId,
    rows: u32,
    cols: u32,
    log_softmax: bool,
) -> Hc4jResult<()> {
    let op = if log_softmax { RowOp::LogSoftmax } else { RowOp::Softmax };
    run_row_op(op, id_a, &[], id_out, rows, cols, 0)
}

/// Dispatches row-wise layer normalization. Both `id_gamma` and `id_beta`
/// must either be `0` (plain) or both non-zero (affine).
pub fn run_layernorm_rows_f32(
    id_a: TensorId,
    id_gamma: TensorId,
    id_beta: TensorId,
    id_out: TensorId,
    rows: u32,
    cols: u32,
    eps_bits: u32,
) -> Hc4jResult<()> {
    match (id_gamma == 0, id_beta == 0) {
        (true, true) => run_row_op(RowOp::LayerNorm { affine: false }, id_a, &[], id_out, rows, cols, eps_bits),
        (false, false) => run_row_op(
            RowOp::LayerNorm { affine: true },
            id_a,
            &[id_gamma, id_beta],
            id_out,
            rows,
            cols,
            eps_bits,
        ),
        _ => Err(Hc4jError::InvalidParam("layernorm affine requires both gamma and beta")),
    }
}

/// Dispatches row-wise RMS normalization (`id_gamma == 0` disables affine scaling).
pub fn run_rmsnorm_rows_f32(
    id_a: TensorId,
    id_gamma: TensorId,
    id_out: TensorId,
    rows: u32,
    cols: u32,
    eps_bits: u32,
) -> Hc4jResult<()> {
    if id_gamma == 0 {
        run_row_op(RowOp::RmsNorm { affine: false }, id_a, &[], id_out, rows, cols, eps_bits)
    } else {
        run_row_op(
            RowOp::RmsNorm { affine: true },
            id_a,
            &[id_gamma],
            id_out,
            rows,
            cols,
            eps_bits,
        )
    }
}

/// Dispatches row-wise cross-entropy loss over `[rows, cols]` logits and `rows` i32 targets.
pub fn run_cross_entropy_rows(
    id_logits: TensorId,
    id_targets: TensorId,
    id_out: TensorId,
    rows: u32,
    cols: u32,
) -> Hc4jResult<()> {
    run_row_op(RowOp::CrossEntropy, id_logits, &[id_targets], id_out, rows, cols, 0)
}
