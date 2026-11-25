//! Rotary Position Embedding (RoPE) kernel (NeoX half-split convention).

use crate::engine::ErrorTrap;
use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::kernels::elementwise::grid_for_threads;
use crate::kernels::linalg::matmul::dispatch::bind_group;
use crate::memory::{TensorId, manager};

pub const ROPE_F32_WGSL: &str = r#"
struct Params {
    batch: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    pos_offset: u32,
    theta_bits: u32,
    total_pairs: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> IN: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.x + gid.y * (nwg.x * 256u);
    if (idx >= p.total_pairs) { return; }

    let half = p.head_dim / 2u;
    let d = idx % half;
    let tmp1 = idx / half;
    let h = tmp1 % p.num_heads;
    let tmp2 = tmp1 / p.num_heads;
    let s = tmp2 % p.seq_len;
    let b = tmp2 / p.seq_len;

    let theta = bitcast<f32>(p.theta_bits);
    let pos = f32(p.pos_offset + s);
    let freq = exp(-log(theta) * f32(2u * d) / f32(p.head_dim));
    let angle = pos * freq;
    let c = cos(angle);
    let sn = sin(angle);

    let base = ((b * p.seq_len + s) * p.num_heads + h) * p.head_dim;
    let x0 = IN[base + d];
    let x1 = IN[base + d + half];
    OUT[base + d] = x0 * c - x1 * sn;
    OUT[base + d + half] = x0 * sn + x1 * c;
}
"#;

/// Dispatches Rotary Position Embedding (RoPE) over an `f32` tensor using NeoX half-split.
/// In-place safe (`id_in == id_out`).
#[allow(clippy::too_many_arguments)]
pub fn run_rope_f32(
    id_in: TensorId,
    id_out: TensorId,
    batch: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    pos_offset: u32,
    theta_bits: u32,
) -> Hc4jResult<()> {
    if batch == 0 || seq_len == 0 || num_heads == 0 || head_dim == 0 {
        return Err(Hc4jError::InvalidParam("batch, seq_len, num_heads, and head_dim must be positive"));
    }
    if !head_dim.is_multiple_of(2) {
        return Err(Hc4jError::InvalidParam("head_dim must be even for half-split RoPE"));
    }

    let total_elements = (batch as u64)
        .checked_mul(seq_len as u64)
        .and_then(|v| v.checked_mul(num_heads as u64))
        .and_then(|v| v.checked_mul(head_dim as u64))
        .ok_or(Hc4jError::InvalidParam("overflow"))?;

    let total_pairs = total_elements / 2;
    let req_bytes = total_elements.checked_mul(4).ok_or(Hc4jError::InvalidParam("overflow"))?;

    let mgr = manager()?;
    let engine = mgr.engine();

    let resident = mgr.acquire_resident(&[id_in, id_out])?;
    if resident.spans[0].size < req_bytes {
        return Err(Hc4jError::InvalidParam("input buffer smaller than required elements"));
    }
    if resident.spans[1].size < req_bytes {
        return Err(Hc4jError::InvalidParam("output buffer smaller than required elements"));
    }

    let total_pairs_u32 = u32::try_from(total_pairs).map_err(|_| Hc4jError::Unsupported("total pairs exceed u32"))?;
    let grid = grid_for_threads(engine, total_pairs)?;
    let pipeline = engine.get_or_compile("rope_f32", "main", ROPE_F32_WGSL)?;

    let params: [u32; 8] = [
        batch,
        seq_len,
        num_heads,
        head_dim,
        pos_offset,
        theta_bits,
        total_pairs_u32,
        0,
    ];
    let param_bytes: &[u8] = bytemuck::cast_slice(&params);

    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, "nn (rope)", |rec| {
        let uniform = rec.uniform(param_bytes)?;
        let bg = bind_group(
            engine,
            &pipeline,
            vec![resident.spans[0].binding(), resident.spans[1].binding(), uniform],
        );
        rec.dispatch(&pipeline, &bg, grid, req_bytes * 2);
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}
