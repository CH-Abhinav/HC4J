//! Row bias broadcast operations.

use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::memory::{TensorId, manager};
use crate::kernels::elementwise::{WORKGROUP_SIZE, grid_for_threads};
use crate::kernels::nn::row_ops::{check_elements, row_dims, within_binding};
use crate::kernels::reduce::{row_bind_group, uniform_bytes};
use crate::engine::ErrorTrap;

/// Bias broadcast over rows: `out[r, c] = a[r, c] + bias[c]`. Fusion cannot
/// express this (it requires identical shapes), and a GEMM followed by a bias
/// add is the shape of every linear layer.
pub fn row_bias_wgsl() -> String {
    let wg = WORKGROUP_SIZE;
    format!(
        "struct Params {{\n    params: vec4<u32>,\n}}\n\n\
@group(0) @binding(0) var<storage, read_write> A: array<f32>;\n\
@group(0) @binding(1) var<storage, read_write> BIAS: array<f32>;\n\
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;\n\
@group(0) @binding(3) var<uniform> p: Params;\n\n\
@compute @workgroup_size({wg})\n\
fn main(\n    \
@builtin(global_invocation_id) gid: vec3<u32>,\n    \
@builtin(num_workgroups) nwg: vec3<u32>,\n\
) {{\n    \
// The host folds large grids into X*Y; rebuild the linear index.\n    \
let i = gid.x + gid.y * (nwg.x * {wg}u);\n    \
let total = p.params.x;\n    \
if (i >= total) {{ return; }}\n    \
let cols = p.params.y;\n    \
OUT[i] = A[i] + BIAS[i % cols];\n\
}}\n"
    )
}

/// `out[r, c] = a[r, c] + bias[c]`. `out` may be `a`.
pub fn run_row_bias(id_a: TensorId, id_bias: TensorId, id_out: TensorId, rows: u32, cols: u32) -> Hc4jResult<()> {
    let elems = row_dims(rows, cols)?;
    let mgr = manager()?;
    let engine = mgr.engine();
    within_binding(engine, elems)?;
    check_elements(id_a, elems, "input size does not match (rows, cols)")?;
    check_elements(id_bias, u64::from(cols), "bias length must equal cols")?;
    check_elements(id_out, elems, "output size does not match (rows, cols)")?;

    let resident = mgr.acquire_resident(&[id_a, id_bias, id_out])?;
    let pipeline = engine.get_or_compile("nn_row_bias", "main", &row_bias_wgsl())?;
    let grid = grid_for_threads(engine, elems)?;

    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, "nn (row bias)", |rec| {
        let uniform = rec.uniform(&uniform_bytes([
            u32::try_from(elems).map_err(|_| Hc4jError::Unsupported("bias add exceeds 2^32 elements"))?,
            cols,
            0,
            0,
        ]))?;
        let spans: Vec<&_> = resident.spans.iter().collect();
        let bg = row_bind_group(engine, &pipeline, &spans, uniform);
        rec.dispatch(&pipeline, &bg, grid, elems * 8);
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_bias_kernel_binds_storage_read_only() {
        assert!(!row_bias_wgsl().contains("storage, read>"));
    }
}
