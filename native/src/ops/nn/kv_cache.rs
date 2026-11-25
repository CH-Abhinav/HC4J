//! GPU-resident KV-cache append & transpose kernel.
//!
//! Copies and transposes incoming Key or Value slices of shape
//! `[seq_len, num_kv_heads, head_dim]` directly into a pre-allocated VRAM cache
//! of shape `[num_kv_heads, max_seq_len, head_dim]` at
//! `[0..num_kv_heads, pos_offset..pos_offset+seq_len, 0..head_dim]`.

use crate::error::{Hc4jError, Hc4jResult};
use crate::memory::{TensorId, manager};
use crate::ops::elementwise::grid_for_threads;
use crate::ops::matmul::dispatch::bind_group;
use crate::ErrorTrap;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KvAppendParams {
    params0: [u32; 4],
    params1: [u32; 4],
}

pub const KV_CACHE_APPEND_F32_WGSL: &str = r#"
struct KvAppendParams {
    params0: vec4<u32>,
    params1: vec4<u32>,
}

@group(0) @binding(0) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(1) var<storage, read_write> CACHE: array<f32>;
@group(0) @binding(2) var<uniform> p: KvAppendParams;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = gid.x + gid.y * (nwg.x * 256u);
    let num_kv_heads = p.params0.x;
    let max_seq_len = p.params0.y;
    let head_dim = p.params0.z;
    let pos_offset = p.params1.x;
    let total_elems = p.params1.y;

    if (tid >= total_elems) { return; }

    let d = tid % head_dim;
    let tmp = tid / head_dim;
    let h = tmp % num_kv_heads;
    let s = tmp / num_kv_heads;
    let dst = (h * max_seq_len + (pos_offset + s)) * head_dim + d;
    CACHE[dst] = SRC[tid];
}
"#;

/// Appends `[seq_len, num_kv_heads, head_dim]` f32 elements from `id_src`
/// into `[num_kv_heads, max_seq_len, head_dim]` `id_cache` starting at `pos_offset`.
pub fn run_kv_cache_append_f32(
    id_src: TensorId,
    id_cache: TensorId,
    num_kv_heads: u32,
    max_seq_len: u32,
    head_dim: u32,
    seq_len: u32,
    pos_offset: u32,
) -> Hc4jResult<()> {
    if num_kv_heads == 0 || max_seq_len == 0 || head_dim == 0 || seq_len == 0 {
        return Err(Hc4jError::InvalidParam(
            "kv_cache_append dimensions (num_kv_heads, max_seq_len, head_dim, seq_len) must be positive",
        ));
    }
    if id_src == id_cache {
        return Err(Hc4jError::InvalidParam("kv_cache_append source and cache must not alias"));
    }
    let end_pos = pos_offset
        .checked_add(seq_len)
        .ok_or(Hc4jError::InvalidParam("kv_cache_append pos_offset + seq_len overflows u32"))?;
    if end_pos > max_seq_len {
        return Err(Hc4jError::InvalidParam("kv_cache_append pos_offset + seq_len exceeds max_seq_len"));
    }

    let total_elems = u64::from(seq_len)
        .checked_mul(u64::from(num_kv_heads))
        .and_then(|v| v.checked_mul(u64::from(head_dim)))
        .ok_or(Hc4jError::InvalidParam("kv_cache_append src elements overflow"))?;
    let total_elems_u32 =
        u32::try_from(total_elems).map_err(|_| Hc4jError::Unsupported("kv_cache_append src elements exceed u32"))?;

    let cache_elems = u64::from(num_kv_heads)
        .checked_mul(u64::from(max_seq_len))
        .and_then(|v| v.checked_mul(u64::from(head_dim)))
        .ok_or(Hc4jError::InvalidParam("kv_cache_append cache elements overflow"))?;

    let req_src_bytes = total_elems
        .checked_mul(4)
        .ok_or(Hc4jError::InvalidParam("kv_cache_append src byte count overflow"))?;
    let req_cache_bytes = cache_elems
        .checked_mul(4)
        .ok_or(Hc4jError::InvalidParam("kv_cache_append cache byte count overflow"))?;

    let mgr = manager()?;
    let engine = mgr.engine();
    let max_binding = engine.limits.max_storage_buffer_binding_size;
    if req_src_bytes > max_binding || req_cache_bytes > max_binding {
        return Err(Hc4jError::Unsupported(
            "kv_cache_append operand exceeds max storage buffer binding limit",
        ));
    }

    if mgr.size_of(id_src)? != req_src_bytes {
        return Err(Hc4jError::InvalidParam("kv_cache_append src byte size mismatch"));
    }
    if mgr.size_of(id_cache)? != req_cache_bytes {
        return Err(Hc4jError::InvalidParam("kv_cache_append cache byte size mismatch"));
    }

    let resident = mgr.acquire_resident(&[id_src, id_cache])?;
    if resident.spans[0].size < req_src_bytes || resident.spans[1].size < req_cache_bytes {
        return Err(Hc4jError::InvalidParam("kv_cache_append span size smaller than required bytes"));
    }

    let grid = grid_for_threads(engine, total_elems)?;
    let pipeline = engine.get_or_compile("kv_cache_append_f32", "main", KV_CACHE_APPEND_F32_WGSL)?;
    let params = KvAppendParams {
        params0: [num_kv_heads, max_seq_len, head_dim, seq_len],
        params1: [pos_offset, total_elems_u32, 0, 0],
    };

    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, "nn (kv_cache_append)", |rec| {
        let uniform = rec.uniform(bytemuck::bytes_of(&params))?;
        let bg = bind_group(
            engine,
            &pipeline,
            vec![resident.spans[0].binding(), resident.spans[1].binding(), uniform],
        );
        rec.dispatch(&pipeline, &bg, grid, req_src_bytes * 2);
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}
