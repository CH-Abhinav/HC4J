//! Portable f16 storage casts (f32 <-> f16) and mixed-precision f32 x f16 GEMM / GEMV.
//! Uses standard WGSL 1.0 `pack2x16float` and `unpack2x16float` without requiring `enable f16;`.

use crate::error::{ffi_guard, Hc4jError};
use crate::memory::manager;
use crate::ops::elementwise::{grid_2d, grid_for_threads};
use crate::ops::matmul::dispatch::{bind_group, tiled_grid};
use crate::ops::matmul::plan::FLAG_TRANS_B;
use crate::ops::reduce::params_bytes;
use crate::ErrorTrap;

pub const CAST_F32_TO_F16_WGSL: &str = r#"
struct Params {
    params: vec4<u32>,
}

@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = gid.x + gid.y * (nwg.x * 256u);
    let length = p.params.x;
    let total_words = (length + 1u) >> 1u;
    if (tid >= total_words) { return; }

    let i0 = tid * 2u;
    let i1 = i0 + 1u;
    let v0 = src[i0];
    let v1 = select(0.0, src[i1], i1 < length);
    out[tid] = pack2x16float(vec2<f32>(v0, v1));
}
"#;

pub const CAST_F16_TO_F32_WGSL: &str = r#"
struct Params {
    params: vec4<u32>,
}

@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = gid.x + gid.y * (nwg.x * 256u);
    let length = p.params.x;
    let total_words = (length + 1u) >> 1u;
    if (tid >= total_words) { return; }

    let pair = unpack2x16float(src[tid]);
    let i0 = tid * 2u;
    let i1 = i0 + 1u;
    out[i0] = pair.x;
    if (i1 < length) {
        out[i1] = pair.y;
    }
}
"#;

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

/// Dispatches GPU cast from f32 to f16 (packed as u32 words).
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_cast_f32_to_f16(id_src: u64, id_out: u64, length: u64) -> i32 {
    ffi_guard("dispatch_cast_f32_to_f16", || {
        if length == 0 {
            return Err(Hc4jError::InvalidParam("length must be non-zero"));
        }
        let mgr = manager()?;
        let engine = mgr.engine();

        let req_src_bytes = length.checked_mul(4).ok_or(Hc4jError::InvalidParam("overflow"))?;
        let req_out_bytes = ((length.checked_mul(2).ok_or(Hc4jError::InvalidParam("overflow"))?) + 3) & !3;

        let resident = mgr.acquire_resident(&[id_src, id_out])?;
        if resident.spans[0].size < req_src_bytes {
            return Err(Hc4jError::InvalidParam("source buffer smaller than requested elements"));
        }
        if resident.spans[1].size < req_out_bytes {
            return Err(Hc4jError::InvalidParam("output buffer smaller than requested elements"));
        }

        let total_words = length.div_ceil(2);
        let grid = grid_for_threads(engine, total_words)?;
        let pipeline = engine.get_or_compile("cast_f32_to_f16", "main", CAST_F32_TO_F16_WGSL)?;

        let len_u32 = u32::try_from(length).map_err(|_| Hc4jError::Unsupported("length exceeds u32"))?;
        let trap = ErrorTrap::push(&engine.device);
        let recorded = engine.stream.record(1, "cast (f32->f16)", |rec| {
            let uniform = rec.uniform(&params_bytes([len_u32, 0, 0, 0]))?;
            let bg = bind_group(
                engine,
                &pipeline,
                vec![resident.spans[0].binding(), resident.spans[1].binding(), uniform],
            );
            rec.dispatch(&pipeline, &bg, grid, req_src_bytes + req_out_bytes);
            Ok(())
        });
        let trapped = trap.finish();
        recorded.and(trapped)
    })
}

/// Dispatches GPU cast from f16 (packed as u32 words) to f32.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_cast_f16_to_f32(id_src: u64, id_out: u64, length: u64) -> i32 {
    ffi_guard("dispatch_cast_f16_to_f32", || {
        if length == 0 {
            return Err(Hc4jError::InvalidParam("length must be non-zero"));
        }
        let mgr = manager()?;
        let engine = mgr.engine();

        let req_src_bytes = ((length.checked_mul(2).ok_or(Hc4jError::InvalidParam("overflow"))?) + 3) & !3;
        let req_out_bytes = length.checked_mul(4).ok_or(Hc4jError::InvalidParam("overflow"))?;

        let resident = mgr.acquire_resident(&[id_src, id_out])?;
        if resident.spans[0].size < req_src_bytes {
            return Err(Hc4jError::InvalidParam("source buffer smaller than requested elements"));
        }
        if resident.spans[1].size < req_out_bytes {
            return Err(Hc4jError::InvalidParam("output buffer smaller than requested elements"));
        }

        let total_words = length.div_ceil(2);
        let grid = grid_for_threads(engine, total_words)?;
        let pipeline = engine.get_or_compile("cast_f16_to_f32", "main", CAST_F16_TO_F32_WGSL)?;

        let len_u32 = u32::try_from(length).map_err(|_| Hc4jError::Unsupported("length exceeds u32"))?;
        let trap = ErrorTrap::push(&engine.device);
        let recorded = engine.stream.record(1, "cast (f16->f32)", |rec| {
            let uniform = rec.uniform(&params_bytes([len_u32, 0, 0, 0]))?;
            let bg = bind_group(
                engine,
                &pipeline,
                vec![resident.spans[0].binding(), resident.spans[1].binding(), uniform],
            );
            rec.dispatch(&pipeline, &bg, grid, req_src_bytes + req_out_bytes);
            Ok(())
        });
        let trapped = trap.finish();
        recorded.and(trapped)
    })
}

/// Dispatches mixed-precision f32 x f16 matmul returning f32.
/// When `m == 1 && (flags & FLAG_TRANS_B) != 0 && k % 2 == 0`, invokes the GEMV fast-path.
/// Otherwise invokes the general 16x16 tiled GEMM kernel.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_matmul_f32_f16(
    id_a: u64,
    id_b_f16: u64,
    id_out: u64,
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
) -> i32 {
    ffi_guard("dispatch_matmul_f32_f16", || {
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
    })
}
