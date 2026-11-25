//! Token embedding lookup kernel supporting f32 and f16 tables.

use crate::engine::ErrorTrap;
use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::kernels::elementwise::grid_for_threads;
use crate::kernels::linalg::matmul::dispatch::bind_group;
use crate::kernels::reduce::params_bytes;
use crate::memory::{TensorId, manager};

pub const EMBEDDING_F32_WGSL: &str = r#"
struct Params {
    vocab_size: u32,
    hidden_dim: u32,
    num_tokens: u32,
    total: u32,
}

@group(0) @binding(0) var<storage, read_write> TABLE: array<f32>;
@group(0) @binding(1) var<storage, read_write> INDICES: array<i32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.x + gid.y * (nwg.x * 256u);
    if (idx >= p.total) { return; }

    let tok = idx / p.hidden_dim;
    let col = idx % p.hidden_dim;
    let row = INDICES[tok];
    if (row < 0 || u32(row) >= p.vocab_size) {
        OUT[idx] = bitcast<f32>(0x7fc00000u);
    } else {
        OUT[idx] = TABLE[u32(row) * p.hidden_dim + col];
    }
}
"#;

pub const EMBEDDING_F16_WGSL: &str = r#"
struct Params {
    vocab_size: u32,
    hidden_dim: u32,
    num_tokens: u32,
    total: u32,
}

@group(0) @binding(0) var<storage, read_write> TABLE: array<u32>;
@group(0) @binding(1) var<storage, read_write> INDICES: array<i32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.x + gid.y * (nwg.x * 256u);
    if (idx >= p.total) { return; }

    let tok = idx / p.hidden_dim;
    let col = idx % p.hidden_dim;
    let row = INDICES[tok];
    if (row < 0 || u32(row) >= p.vocab_size) {
        OUT[idx] = bitcast<f32>(0x7fc00000u);
    } else {
        let table_idx = u32(row) * p.hidden_dim + col;
        let word = TABLE[table_idx >> 1u];
        let pair = unpack2x16float(word);
        OUT[idx] = select(pair.x, pair.y, (table_idx & 1u) != 0u);
    }
}
"#;

/// Dispatches token embedding lookup: `out[tok, col] = table[indices[tok], col]`.
/// Supports both f32 (`table_is_f16 == false`) and packed f16 (`table_is_f16 == true`) tables.
pub fn run_embedding(
    id_table: TensorId,
    id_indices: TensorId,
    id_out: TensorId,
    vocab_size: u32,
    hidden_dim: u32,
    num_tokens: u32,
    table_is_f16: bool,
) -> Hc4jResult<()> {
    if vocab_size == 0 || hidden_dim == 0 || num_tokens == 0 {
        return Err(Hc4jError::InvalidParam("vocab_size, hidden_dim, and num_tokens must be positive"));
    }
    if id_out == id_table || id_out == id_indices {
        return Err(Hc4jError::InvalidParam("embedding output must not alias table or indices"));
    }

    let mgr = manager()?;
    let engine = mgr.engine();

    let table_elems = (vocab_size as u64).checked_mul(hidden_dim as u64).ok_or(Hc4jError::InvalidParam("overflow"))?;
    let total = (num_tokens as u64).checked_mul(hidden_dim as u64).ok_or(Hc4jError::InvalidParam("overflow"))?;

    let req_table_bytes = if table_is_f16 {
        ((table_elems.checked_mul(2).ok_or(Hc4jError::InvalidParam("overflow"))?) + 3) & !3
    } else {
        table_elems.checked_mul(4).ok_or(Hc4jError::InvalidParam("overflow"))?
    };
    let req_indices_bytes = (num_tokens as u64).checked_mul(4).ok_or(Hc4jError::InvalidParam("overflow"))?;
    let req_out_bytes = total.checked_mul(4).ok_or(Hc4jError::InvalidParam("overflow"))?;

    let resident = mgr.acquire_resident(&[id_table, id_indices, id_out])?;
    if resident.spans[0].size < req_table_bytes {
        return Err(Hc4jError::InvalidParam("embedding table smaller than vocab_size * hidden_dim"));
    }
    if resident.spans[1].size < req_indices_bytes {
        return Err(Hc4jError::InvalidParam("indices buffer smaller than num_tokens"));
    }
    if resident.spans[2].size < req_out_bytes {
        return Err(Hc4jError::InvalidParam("output buffer smaller than num_tokens * hidden_dim"));
    }

    let total_u32 = u32::try_from(total).map_err(|_| Hc4jError::Unsupported("total outputs exceed u32"))?;
    let grid = grid_for_threads(engine, total)?;

    let (name, src) = if table_is_f16 {
        ("embedding_f16", EMBEDDING_F16_WGSL)
    } else {
        ("embedding_f32", EMBEDDING_F32_WGSL)
    };
    let pipeline = engine.get_or_compile(name, "main", src)?;

    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, "nn (embedding)", |rec| {
        let uniform = rec.uniform(&params_bytes([vocab_size, hidden_dim, num_tokens, total_u32]))?;
        let bg = bind_group(
            engine,
            &pipeline,
            vec![
                resident.spans[0].binding(),
                resident.spans[1].binding(),
                resident.spans[2].binding(),
                uniform,
            ],
        );
        rec.dispatch(&pipeline, &bg, grid, req_indices_bytes + total * 4);
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}
