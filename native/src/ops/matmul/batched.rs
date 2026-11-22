//! Batched matrix multiplication for 3-D, 4-D and higher-rank tensors.
//!
//! Supports NumPy-style batch broadcasting and arbitrary batch ranks up to 4
//! (tensors up to 6-D total) with 16x16 shared-memory tiling.

use crate::ErrorTrap;
use crate::error::{Hc4jError, Hc4jResult};
use crate::memory::{manager, TensorId};
use crate::ops::matmul::dispatch::run_matmul;

/// Uniform parameter block for batched GEMM.
/// All fields are 16-byte aligned vec4<u32> to match WebGPU uniform buffer rules.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BatchedMatmulDims {
    /// x = M, y = N, z = K, w = batch_count
    pub dims: [u32; 4],
    /// x = trans_a, y = trans_b, z = batch_rank, w = batch_offset
    pub flags: [u32; 4],
    /// batch dimensions (padded with 1s)
    pub batch_shape: [u32; 4],
    /// A strides in elements for each batch dimension (0 for broadcast)
    pub a_batch_strides: [u32; 4],
    /// B strides in elements for each batch dimension (0 for broadcast)
    pub b_batch_strides: [u32; 4],
}

pub const BATCHED_TILED16_WGSL: &str = r#"
struct BatchedMatmulDims {
    dims: vec4<u32>,
    flags: vec4<u32>,
    batch_shape: vec4<u32>,
    a_batch_strides: vec4<u32>,
    b_batch_strides: vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: BatchedMatmulDims;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let M = dims.dims.x;
    let N = dims.dims.y;
    let K = dims.dims.z;
    let batch_count = dims.dims.w;

    let b = wid.z + dims.flags.w;
    if (b >= batch_count) {
        return;
    }

    // Decompose batch index `b` into multi-dimensional coordinates
    let rank = dims.flags.z;
    var a_base = 0u;
    var b_base = 0u;
    var rem = b;

    if (rank == 1u) {
        let c0 = rem;
        a_base = c0 * dims.a_batch_strides.x;
        b_base = c0 * dims.b_batch_strides.x;
    } else if (rank == 2u) {
        let c1 = rem % dims.batch_shape.y;
        rem = rem / dims.batch_shape.y;
        let c0 = rem;
        a_base = c0 * dims.a_batch_strides.x + c1 * dims.a_batch_strides.y;
        b_base = c0 * dims.b_batch_strides.x + c1 * dims.b_batch_strides.y;
    } else if (rank == 3u) {
        let c2 = rem % dims.batch_shape.z;
        rem = rem / dims.batch_shape.z;
        let c1 = rem % dims.batch_shape.y;
        rem = rem / dims.batch_shape.y;
        let c0 = rem;
        a_base = c0 * dims.a_batch_strides.x + c1 * dims.a_batch_strides.y + c2 * dims.a_batch_strides.z;
        b_base = c0 * dims.b_batch_strides.x + c1 * dims.b_batch_strides.y + c2 * dims.b_batch_strides.z;
    } else if (rank >= 4u) {
        let c3 = rem % dims.batch_shape.w;
        rem = rem / dims.batch_shape.w;
        let c2 = rem % dims.batch_shape.z;
        rem = rem / dims.batch_shape.z;
        let c1 = rem % dims.batch_shape.y;
        rem = rem / dims.batch_shape.y;
        let c0 = rem;
        a_base = c0 * dims.a_batch_strides.x + c1 * dims.a_batch_strides.y + c2 * dims.a_batch_strides.z + c3 * dims.a_batch_strides.w;
        b_base = c0 * dims.b_batch_strides.x + c1 * dims.b_batch_strides.y + c2 * dims.b_batch_strides.z + c3 * dims.b_batch_strides.w;
    }

    let c_base = b * (M * N);
    let row = (wid.y + wid.z * 0u) * 16u + lid.y;
    let col = wid.x * 16u + lid.x;
    var acc = 0.0;

    let tiles = (K + 15u) / 16u;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let ka = t * 16u + lid.x;
        var va = 0.0;
        if (row < M && ka < K) {
            if (dims.flags.x != 0u) {
                va = A[a_base + ka * M + row];
            } else {
                va = A[a_base + row * K + ka];
            }
        }
        tile_a[lid.y * 16u + lid.x] = va;

        let kb = t * 16u + lid.y;
        var vb = 0.0;
        if (kb < K && col < N) {
            if (dims.flags.y != 0u) {
                vb = B[b_base + col * K + kb];
            } else {
                vb = B[b_base + kb * N + col];
            }
        }
        tile_b[lid.y * 16u + lid.x] = vb;

        workgroupBarrier();

        for (var k = 0u; k < 16u; k = k + 1u) {
            acc = acc + tile_a[lid.y * 16u + k] * tile_b[k * 16u + lid.x];
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        C[c_base + row * N + col] = acc;
    }
}
"#;

/// Runs batched matrix multiplication `C = A @ B`.
#[allow(clippy::too_many_arguments)]
pub fn run_batched_matmul(
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    m: u32,
    n: u32,
    k: u32,
    batch_rank: u32,
    batch_shape: &[u32],
    a_batch_strides: &[u32],
    b_batch_strides: &[u32],
    flags: u32,
) -> Hc4jResult<()> {
    if id_out == id_a || id_out == id_b {
        return Err(Hc4jError::InvalidParam(
            "batched matmul output cannot alias an input buffer",
        ));
    }
    if m == 0 || n == 0 || k == 0 {
        return Err(Hc4jError::InvalidParam("m, n, and k must be strictly positive"));
    }
    if batch_rank == 0 || batch_rank > 4 {
        return Err(Hc4jError::InvalidParam("batch rank must be in 1..=4"));
    }
    if batch_shape.len() < batch_rank as usize
        || a_batch_strides.len() < batch_rank as usize
        || b_batch_strides.len() < batch_rank as usize
    {
        return Err(Hc4jError::InvalidParam("batch array slice length mismatch"));
    }

    let mut batch_count: u64 = 1;
    for &dim in &batch_shape[..batch_rank as usize] {
        if dim == 0 {
            return Err(Hc4jError::InvalidParam("batch dimension must be strictly positive"));
        }
        batch_count = batch_count
            .checked_mul(u64::from(dim))
            .ok_or(Hc4jError::InvalidParam("total batch count overflows u64"))?;
    }
    let batch_count_u32 = u32::try_from(batch_count)
        .map_err(|_| Hc4jError::Unsupported("total batch count exceeds u32"))?;

    let trans_a = flags & 1;
    let trans_b = (flags >> 1) & 1;

    // Fast Path: A is contiguous [batch_count * M, K], B is 2-D [K, N] (all b_batch_strides == 0),
    // and trans_a == 0 -> flatten A and delegate directly to run_matmul!
    let b_is_broadcast = b_batch_strides[..batch_rank as usize].iter().all(|&s| s == 0);
    if trans_a == 0 && b_is_broadcast {
        let mut expected_stride = u64::from(m) * u64::from(k);
        let mut a_is_contiguous = true;
        for i in (0..batch_rank as usize).rev() {
            if u64::from(a_batch_strides[i]) != expected_stride {
                a_is_contiguous = false;
                break;
            }
            expected_stride *= u64::from(batch_shape[i]);
        }
        if a_is_contiguous {
            let flattened_m = u64::from(batch_count_u32) * u64::from(m);
            if let Ok(flat_m_u32) = u32::try_from(flattened_m) {
                return run_matmul(id_a, id_b, id_out, flat_m_u32, n, k, flags);
            }
        }
    }

    // General Batched GEMM Shader
    let mut shape_padded = [1u32; 4];
    let mut a_strides_padded = [0u32; 4];
    let mut b_strides_padded = [0u32; 4];
    let b_len = batch_rank as usize;
    shape_padded[..b_len].copy_from_slice(&batch_shape[..b_len]);
    a_strides_padded[..b_len].copy_from_slice(&a_batch_strides[..b_len]);
    b_strides_padded[..b_len].copy_from_slice(&b_batch_strides[..b_len]);

    let mgr = manager()?;
    let engine = mgr.engine();
    let resident = mgr.acquire_resident(&[id_a, id_b, id_out])?;
    let (span_a, span_b, span_out) = (
        &resident.spans[0],
        &resident.spans[1],
        &resident.spans[2],
    );

    let max_binding = engine.limits.max_storage_buffer_binding_size;
    if span_a.size > max_binding || span_b.size > max_binding || span_out.size > max_binding {
        return Err(Hc4jError::Unsupported(
            "batched matmul tensor exceeds max storage buffer binding size",
        ));
    }

    let pipeline = engine.get_or_compile("batched_matmul_tiled16", "main", BATCHED_TILED16_WGSL)?;
    let layout = pipeline.get_bind_group_layout(0);
    let max_workgroups = engine.limits.max_compute_workgroups_per_dimension;

    let col_tiles = n.div_ceil(16);
    let row_tiles = m.div_ceil(16);
    if col_tiles > max_workgroups || row_tiles > max_workgroups {
        return Err(Hc4jError::Unsupported("tile grid exceeds workgroup limits"));
    }

    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, "batched_matmul", |rec| {
        let mut batch_offset = 0u32;
        while batch_offset < batch_count_u32 {
            let chunk = (batch_count_u32 - batch_offset).min(max_workgroups);
            let dims_block = BatchedMatmulDims {
                dims: [m, n, k, batch_count_u32],
                flags: [trans_a, trans_b, batch_rank, batch_offset],
                batch_shape: shape_padded,
                a_batch_strides: a_strides_padded,
                b_batch_strides: b_strides_padded,
            };
            let uniform_buf = rec.uniform(bytemuck::bytes_of(&dims_block))?;
            let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("HC4J Batched Matmul"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: span_a.binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: span_b.binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: span_out.binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buf,
                    },
                ],
            });
            let total_bytes = span_a.size + span_b.size + span_out.size;
            rec.dispatch(&pipeline, &bg, (col_tiles, row_tiles, chunk), total_bytes);
            batch_offset += chunk;
        }
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}
