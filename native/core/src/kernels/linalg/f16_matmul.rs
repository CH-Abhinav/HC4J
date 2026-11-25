//! Mixed-precision `f32 x f16` GEMM and GEMV (`unpack2x16float`, WGSL 1.0 portable).

use crate::engine::ErrorTrap;
use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::kernels::elementwise::grid_2d;
use crate::kernels::linalg::matmul::dispatch::{bind_group, tiled_grid};
use crate::kernels::linalg::matmul::plan::FLAG_TRANS_B;
use crate::kernels::reduce::params_bytes;
use crate::memory::{TensorId, manager};

pub const MATMUL_F32_F16_GEMV_TRANS_B_WGSL: &str = r#"
struct Params {
    params: vec4<u32>,
}

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B_f16: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let N = p.params.y;
    let K = p.params.z;
    let wg_idx = wid.x + wid.y * nwg.x;
    let local_row = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wg_idx * 8u + local_row;
    let k_pairs = K >> 1u;

    var sum = 0.0;
    if (row < N) {
        let b_base = row * k_pairs;
        for (var kp = lane; kp < k_pairs; kp = kp + 32u) {
            let packed_w = B_f16[b_base + kp];
            let w = unpack2x16float(packed_w);
            let a0 = A[kp * 2u];
            let a1 = A[kp * 2u + 1u];
            sum = fma(a0, w.x, fma(a1, w.y, sum));
        }
    }
    partial[lid.x] = sum;
    workgroupBarrier();

    if (lane < 16u) { partial[lid.x] = partial[lid.x] + partial[lid.x + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { partial[lid.x] = partial[lid.x] + partial[lid.x + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { partial[lid.x] = partial[lid.x] + partial[lid.x + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { partial[lid.x] = partial[lid.x] + partial[lid.x + 2u]; }
    workgroupBarrier();
    if (lane < 1u) { partial[lid.x] = partial[lid.x] + partial[lid.x + 1u]; }
    workgroupBarrier();

    if (lane == 0u && row < N) {
        OUT[row] = partial[lid.x];
    }
}
"#;

pub const MATMUL_F32_F16_TILED16_WGSL: &str = r#"
struct Params {
    params: vec4<u32>,
}

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B_f16: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

fn load_f16(elem_idx: u32) -> f32 {
    let word = B_f16[elem_idx >> 1u];
    let pair = unpack2x16float(word);
    return select(pair.x, pair.y, (elem_idx & 1u) != 0u);
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let M = p.params.x;
    let N = p.params.y;
    let K = p.params.z;
    let trans_b = p.params.w;
    let row = (wid.y + wid.z * nwg.y) * 16u + lid.y;
    let col = wid.x * 16u + lid.x;
    var acc = 0.0;
    let tiles = (K + 15u) / 16u;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let ka = t * 16u + lid.x;
        var va = 0.0;
        if (row < M && ka < K) { va = A[row * K + ka]; }
        tile_a[lid.y * 16u + lid.x] = va;

        let kb = t * 16u + lid.y;
        var vb = 0.0;
        if (kb < K && col < N) {
            let b_idx = select(kb * N + col, col * K + kb, trans_b != 0u);
            vb = load_f16(b_idx);
        }
        tile_b[lid.y * 16u + lid.x] = vb;

        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            acc = fma(tile_a[lid.y * 16u + kk], tile_b[kk * 16u + lid.x], acc);
        }
        workgroupBarrier();
    }
    if (row < M && col < N) {
        OUT[row * N + col] = acc;
    }
}
"#;

/// Executes mixed-precision `f32 x f16` matrix multiplication returning `f32`.
/// When `m == 1 && (flags & FLAG_TRANS_B) != 0 && k % 2 == 0`, uses the GEMV fast path.
pub fn run_matmul_f32_f16(
    id_a: TensorId,
    id_b_f16: TensorId,
    id_out: TensorId,
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
) -> Hc4jResult<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(Hc4jError::InvalidParam("m, n, and k must be positive"));
    }
    if id_out == id_a || id_out == id_b_f16 {
        return Err(Hc4jError::InvalidParam("matmul result must not alias an operand"));
    }

    let mgr = manager()?;
    let engine = mgr.engine();

    let elems_a = (m as u64).checked_mul(k as u64).ok_or(Hc4jError::InvalidParam("overflow"))?;
    let elems_b = (n as u64).checked_mul(k as u64).ok_or(Hc4jError::InvalidParam("overflow"))?;
    let elems_out = (m as u64).checked_mul(n as u64).ok_or(Hc4jError::InvalidParam("overflow"))?;

    let req_a_bytes = elems_a * 4;
    let req_b_bytes = ((elems_b * 2) + 3) & !3;
    let req_out_bytes = elems_out * 4;

    let resident = mgr.acquire_resident(&[id_a, id_b_f16, id_out])?;
    if resident.spans[0].size < req_a_bytes {
        return Err(Hc4jError::InvalidParam("A matrix smaller than M*K"));
    }
    if resident.spans[1].size < req_b_bytes {
        return Err(Hc4jError::InvalidParam("B matrix smaller than K*N"));
    }
    if resident.spans[2].size < req_out_bytes {
        return Err(Hc4jError::InvalidParam("output matrix smaller than M*N"));
    }

    let trans_b = (flags & FLAG_TRANS_B) != 0;
    let use_gemv = m == 1 && trans_b && k.is_multiple_of(2);

    let (pipeline_name, shader_source, grid, uniform_params) = if use_gemv {
        let n_wg = n.div_ceil(8);
        let (gx, gy) = grid_2d(n_wg, engine.limits.max_compute_workgroups_per_dimension);
        (
            "matmul_f32_f16_gemv_trans_b",
            MATMUL_F32_F16_GEMV_TRANS_B_WGSL,
            (gx, gy, 1),
            [m, n, k, flags],
        )
    } else {
        let grid = tiled_grid(engine, n.div_ceil(16), m.div_ceil(16))?;
        (
            "matmul_f32_f16_tiled16",
            MATMUL_F32_F16_TILED16_WGSL,
            grid,
            [m, n, k, if trans_b { 1 } else { 0 }],
        )
    };

    let pipeline = engine.get_or_compile(pipeline_name, "main", shader_source)?;
    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, "matmul (f32 x f16)", |rec| {
        let uniform = rec.uniform(&params_bytes(uniform_params))?;
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
        rec.dispatch(&pipeline, &bg, grid, req_a_bytes + req_b_bytes + req_out_bytes);
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}
