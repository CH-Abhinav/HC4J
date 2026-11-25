//! GPU type-conversion kernels: `f32 <-> i32` and portable packed `f32 <-> f16`
//! using WGSL 1.0 `pack2x16float` / `unpack2x16float` (no `enable f16;` extension).

use crate::engine::ErrorTrap;
use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::engine::get_engine;
use crate::kernels::elementwise::{grid_for_threads, run_contiguous};
use crate::kernels::linalg::matmul::dispatch::bind_group;
use crate::kernels::math::arithmetic::DataType;
use crate::kernels::reduce::params_bytes;
use crate::memory::{TensorId, manager};

pub const CAST_F32_TO_I32_WGSL: &str = r#"
struct Dims {
    params: vec4<u32>,
}
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<i32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = gid.x + gid.y * (nwg.x * 256u);
    let length = dims.params.x;
    let base = tid << 2u;
    if (base >= length) { return; }
    if (base + 3u < length) {
        let v = vec4<f32>(src[base], src[base+1u], src[base+2u], src[base+3u]);
        let res = vec4<i32>(v);
        dst[base] = res.x;
        dst[base+1u] = res.y;
        dst[base+2u] = res.z;
        dst[base+3u] = res.w;
    } else {
        dst[base] = i32(src[base]);
        if (base + 1u < length) {
            dst[base+1u] = i32(src[base+1u]);
            if (base + 2u < length) {
                dst[base+2u] = i32(src[base+2u]);
            }
        }
    }
}
"#;

pub const CAST_I32_TO_F32_WGSL: &str = r#"
struct Dims {
    params: vec4<u32>,
}
@group(0) @binding(0) var<storage, read_write> src: array<i32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = gid.x + gid.y * (nwg.x * 256u);
    let length = dims.params.x;
    let base = tid << 2u;
    if (base >= length) { return; }
    if (base + 3u < length) {
        let v = vec4<i32>(src[base], src[base+1u], src[base+2u], src[base+3u]);
        let res = vec4<f32>(v);
        dst[base] = res.x;
        dst[base+1u] = res.y;
        dst[base+2u] = res.z;
        dst[base+3u] = res.w;
    } else {
        dst[base] = f32(src[base]);
        if (base + 1u < length) {
            dst[base+1u] = f32(src[base+1u]);
            if (base + 2u < length) {
                dst[base+2u] = f32(src[base+2u]);
            }
        }
    }
}
"#;

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

/// Converts tensor elements between `f32` and `i32` (or copies when identical).
pub fn run_cast(id_src: TensorId, id_out: TensorId, length: u64, src_dtype: u32, dst_dtype: u32) -> Hc4jResult<()> {
    let _ = u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("length exceeds u32"))?;
    let src_dt = DataType::from_u32(src_dtype).ok_or(Hc4jError::InvalidParam("invalid src_dtype"))?;
    let dst_dt = DataType::from_u32(dst_dtype).ok_or(Hc4jError::InvalidParam("invalid dst_dtype"))?;

    if src_dt == dst_dt {
        let pipeline = get_engine()?.get_or_compile(
            "cast_copy",
            "main",
            CAST_I32_TO_F32_WGSL.replace("f32(src", "src").as_str(),
        )?;
        return run_contiguous(
            &pipeline,
            &|len| bytemuck::bytes_of(&[len, 0u32, 0u32, 0u32]).to_vec(),
            &[id_src],
            id_out,
        );
    }

    let (name, shader) = match (src_dt, dst_dt) {
        (DataType::F32, DataType::I32) => ("cast_f32_to_i32", CAST_F32_TO_I32_WGSL),
        (DataType::I32, DataType::F32) => ("cast_i32_to_f32", CAST_I32_TO_F32_WGSL),
        _ => unreachable!(),
    };

    let pipeline = get_engine()?.get_or_compile(name, "main", shader)?;
    run_contiguous(
        &pipeline,
        &|len| bytemuck::bytes_of(&[len, 0u32, 0u32, 0u32]).to_vec(),
        &[id_src],
        id_out,
    )
}

fn run_packed_f16_cast(
    id_src: TensorId,
    id_out: TensorId,
    length: u64,
    src_is_f16: bool,
) -> Hc4jResult<()> {
    if length == 0 {
        return Err(Hc4jError::InvalidParam("length must be non-zero"));
    }
    let mgr = manager()?;
    let engine = mgr.engine();

    let packed_bytes = (length.checked_mul(2).ok_or(Hc4jError::InvalidParam("overflow"))? + 3) & !3;
    let f32_bytes = length.checked_mul(4).ok_or(Hc4jError::InvalidParam("overflow"))?;
    let (req_src_bytes, req_out_bytes, pipeline_name, wgsl, label) = if src_is_f16 {
        (packed_bytes, f32_bytes, "cast_f16_to_f32", CAST_F16_TO_F32_WGSL, "cast (f16->f32)")
    } else {
        (f32_bytes, packed_bytes, "cast_f32_to_f16", CAST_F32_TO_F16_WGSL, "cast (f32->f16)")
    };

    let resident = mgr.acquire_resident(&[id_src, id_out])?;
    if resident.spans[0].size < req_src_bytes {
        return Err(Hc4jError::InvalidParam("source buffer smaller than requested elements"));
    }
    if resident.spans[1].size < req_out_bytes {
        return Err(Hc4jError::InvalidParam("output buffer smaller than requested elements"));
    }

    let total_words = length.div_ceil(2);
    let grid = grid_for_threads(engine, total_words)?;
    let pipeline = engine.get_or_compile(pipeline_name, "main", wgsl)?;

    let len_u32 = u32::try_from(length).map_err(|_| Hc4jError::Unsupported("length exceeds u32"))?;
    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, label, |rec| {
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
}

/// Converts `length` contiguous `f32` elements into packed IEEE-754 `f16` (`u32` pairs).
pub fn run_cast_f32_to_f16(id_src: TensorId, id_out: TensorId, length: u64) -> Hc4jResult<()> {
    run_packed_f16_cast(id_src, id_out, length, false)
}

/// Unpacks `length` IEEE-754 `f16` elements (`u32` pairs) into contiguous `f32` elements.
pub fn run_cast_f16_to_f32(id_src: TensorId, id_out: TensorId, length: u64) -> Hc4jResult<()> {
    run_packed_f16_cast(id_src, id_out, length, true)
}
