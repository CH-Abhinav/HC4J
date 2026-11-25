//! Unary f32 and i32 elementwise math kernels (`abs`, `neg`, `rsqrt`, `floor`, `ceil`, `round`, `sign`, `clamp`).

use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::engine::get_engine;
use crate::kernels::elementwise::{UnaryElemDims, check_contiguous_length, prune_strided_layout, run_contiguous, run_strided, run_unary_kernel, shaders::UnaryKernel};
use crate::memory::TensorId;

const NEG_PRELUDE: &str = r#"
fn neg4(v: vec4<f32>) -> vec4<f32> { return -v; }
fn neg1(v: f32) -> f32 { return -v; }
"#;

pub const SIGN_F32_PRELUDE: &str = r#"
fn sign1_f32(v: f32) -> f32 {
    if (v > 0.0) { return 1.0; }
    if (v < 0.0) { return -1.0; }
    return 0.0;
}
fn sign4_f32(v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(sign1_f32(v.x), sign1_f32(v.y), sign1_f32(v.z), sign1_f32(v.w));
}
"#;

const CLAMP_F32_PRELUDE: &str = r#"
fn clamp4(v: vec4<f32>) -> vec4<f32> {
    let lo = bitcast<f32>(dims.params.z);
    let hi = bitcast<f32>(dims.params.w);
    return clamp(v, vec4<f32>(lo), vec4<f32>(hi));
}
fn clamp1(v: f32) -> f32 {
    let lo = bitcast<f32>(dims.params.z);
    let hi = bitcast<f32>(dims.params.w);
    return clamp(v, lo, hi);
}
"#;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum F32UnaryOp {
    Abs,
    Neg,
    Rsqrt,
    Floor,
    Ceil,
    Round,
    Sign,
    Clamp { lo: f32, hi: f32 },
}

pub fn run_f32_unary(
    op: F32UnaryOp,
    id_a: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<(&[u32], &[u32], &[u32])>,
) -> Hc4jResult<()> {
    let kernel = match op {
        F32UnaryOp::Abs => UnaryKernel::builtin("math", "abs", "abs"),
        F32UnaryOp::Neg => UnaryKernel {
            family: "math",
            op_name: "neg",
            prelude: NEG_PRELUDE,
            op4: "neg4",
            op1: "neg1",
            alpha_bits: 0,
            param_w_bits: 0,
        },
        F32UnaryOp::Rsqrt => UnaryKernel::builtin("math", "rsqrt", "inverseSqrt"),
        F32UnaryOp::Floor => UnaryKernel::builtin("math", "floor", "floor"),
        F32UnaryOp::Ceil => UnaryKernel::builtin("math", "ceil", "ceil"),
        F32UnaryOp::Round => UnaryKernel::builtin("math", "round", "round"),
        F32UnaryOp::Sign => UnaryKernel {
            family: "math",
            op_name: "sign",
            prelude: SIGN_F32_PRELUDE,
            op4: "sign4_f32",
            op1: "sign1_f32",
            alpha_bits: 0,
            param_w_bits: 0,
        },
        F32UnaryOp::Clamp { lo, hi } => {
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(Hc4jError::InvalidParam("clamp lo and hi must be finite and lo <= hi"));
            }
            UnaryKernel {
                family: "math",
                op_name: "clamp",
                prelude: CLAMP_F32_PRELUDE,
                op4: "clamp4",
                op1: "clamp1",
                alpha_bits: lo.to_bits(),
                param_w_bits: hi.to_bits(),
            }
        }
    };
    run_unary_kernel(kernel, id_a, id_out, length, strided)
}

pub fn run_abs_f32(id_a: TensorId, id_out: TensorId, length: usize, strided: Option<(&[u32], &[u32], &[u32])>) -> Hc4jResult<()> {
    run_f32_unary(F32UnaryOp::Abs, id_a, id_out, length, strided)
}
pub fn run_neg_f32(id_a: TensorId, id_out: TensorId, length: usize, strided: Option<(&[u32], &[u32], &[u32])>) -> Hc4jResult<()> {
    run_f32_unary(F32UnaryOp::Neg, id_a, id_out, length, strided)
}
pub fn run_rsqrt_f32(id_a: TensorId, id_out: TensorId, length: usize, strided: Option<(&[u32], &[u32], &[u32])>) -> Hc4jResult<()> {
    run_f32_unary(F32UnaryOp::Rsqrt, id_a, id_out, length, strided)
}
pub fn run_floor_f32(id_a: TensorId, id_out: TensorId, length: usize, strided: Option<(&[u32], &[u32], &[u32])>) -> Hc4jResult<()> {
    run_f32_unary(F32UnaryOp::Floor, id_a, id_out, length, strided)
}
pub fn run_ceil_f32(id_a: TensorId, id_out: TensorId, length: usize, strided: Option<(&[u32], &[u32], &[u32])>) -> Hc4jResult<()> {
    run_f32_unary(F32UnaryOp::Ceil, id_a, id_out, length, strided)
}
pub fn run_round_f32(id_a: TensorId, id_out: TensorId, length: usize, strided: Option<(&[u32], &[u32], &[u32])>) -> Hc4jResult<()> {
    run_f32_unary(F32UnaryOp::Round, id_a, id_out, length, strided)
}
pub fn run_sign_f32(id_a: TensorId, id_out: TensorId, length: usize, strided: Option<(&[u32], &[u32], &[u32])>) -> Hc4jResult<()> {
    run_f32_unary(F32UnaryOp::Sign, id_a, id_out, length, strided)
}
pub fn run_clamp_f32(id_a: TensorId, id_out: TensorId, length: usize, strided: Option<(&[u32], &[u32], &[u32])>, lo: f32, hi: f32) -> Hc4jResult<()> {
    run_f32_unary(F32UnaryOp::Clamp { lo, hi }, id_a, id_out, length, strided)
}

pub const UNARY_I32_SHADER_TEMPLATE: &str = r#"
struct Dims {
    params: vec4<u32>,
    shape: array<vec4<u32>, 2>,
    strides_a: array<vec4<u32>, 2>,
    strides_c: array<vec4<u32>, 2>,
}

@group(0) @binding(0) var<storage, read_write> arrayA: array<i32>;
@group(0) @binding(1) var<storage, read_write> arrayC: array<i32>;
@group(0) @binding(2) var<uniform> dims: Dims;

__WGSL_PRELUDE__

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = gid.x + gid.y * (nwg.x * 256u);
    let length = dims.params.x;

    if (dims.params.y == 1u) {
        let base = tid << 2u;
        if (base >= length) { return; }
        if (base + 3u < length) {
            let val_a = vec4<i32>(arrayA[base], arrayA[base+1u], arrayA[base+2u], arrayA[base+3u]);
            let res = __WGSL_OP4__(val_a);
            arrayC[base] = res.x;
            arrayC[base+1u] = res.y;
            arrayC[base+2u] = res.z;
            arrayC[base+3u] = res.w;
        } else {
            arrayC[base] = __WGSL_OP1__(arrayA[base]);
            if (base + 1u < length) {
                arrayC[base+1u] = __WGSL_OP1__(arrayA[base+1u]);
                if (base + 2u < length) {
                    arrayC[base+2u] = __WGSL_OP1__(arrayA[base+2u]);
                }
            }
        }
    } else {
        let i = tid;
        if (i >= length) { return; }
        var remaining = i;
        var offset_a = 0u;
        var offset_c = 0u;
        for (var d = 0u; d < 8u; d = d + 1u) {
            let dim_size = dims.shape[d >> 2u][d & 3u];
            if (dim_size > 1u) {
                let coord = remaining % dim_size;
                remaining = remaining / dim_size;
                offset_a = offset_a + coord * dims.strides_a[d >> 2u][d & 3u];
                offset_c = offset_c + coord * dims.strides_c[d >> 2u][d & 3u];
            }
        }
        arrayC[offset_c] = __WGSL_OP1__(arrayA[offset_a]);
    }
}
"#;

pub fn generate_unary_i32_shader(op_code: u32) -> String {
    match op_code {
        0 => UNARY_I32_SHADER_TEMPLATE
            .replace("__WGSL_PRELUDE__", "")
            .replace("__WGSL_OP4__", "abs")
            .replace("__WGSL_OP1__", "abs"),
        1 => UNARY_I32_SHADER_TEMPLATE
            .replace(
                "__WGSL_PRELUDE__",
                "fn neg4(v: vec4<i32>) -> vec4<i32> { return -v; }\nfn neg1(v: i32) -> i32 { return -v; }",
            )
            .replace("__WGSL_OP4__", "neg4")
            .replace("__WGSL_OP1__", "neg1"),
        2 => UNARY_I32_SHADER_TEMPLATE
            .replace("__WGSL_PRELUDE__", "")
            .replace("__WGSL_OP4__", "sign")
            .replace("__WGSL_OP1__", "sign"),
        3 => {
            let prelude = r#"
fn clamp4_i32(v: vec4<i32>) -> vec4<i32> {
    let lo = bitcast<i32>(dims.params.z);
    let hi = bitcast<i32>(dims.params.w);
    return clamp(v, vec4<i32>(lo), vec4<i32>(hi));
}
fn clamp1_i32(v: i32) -> i32 {
    let lo = bitcast<i32>(dims.params.z);
    let hi = bitcast<i32>(dims.params.w);
    return clamp(v, lo, hi);
}
"#;
            UNARY_I32_SHADER_TEMPLATE
                .replace("__WGSL_PRELUDE__", prelude)
                .replace("__WGSL_OP4__", "clamp4_i32")
                .replace("__WGSL_OP1__", "clamp1_i32")
        }
        _ => UNARY_I32_SHADER_TEMPLATE.to_string(),
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum I32UnaryOp {
    Abs,
    Neg,
    Sign,
    Clamp,
}

impl I32UnaryOp {
    pub fn from_code(code: u32) -> Hc4jResult<Self> {
        match code {
            0 => Ok(Self::Abs),
            1 => Ok(Self::Neg),
            2 => Ok(Self::Sign),
            3 => Ok(Self::Clamp),
            _ => Err(Hc4jError::InvalidParam("unknown i32 unary op_code")),
        }
    }

    pub fn code(self) -> u32 {
        match self {
            Self::Abs => 0,
            Self::Neg => 1,
            Self::Sign => 2,
            Self::Clamp => 3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Abs => "abs",
            Self::Neg => "neg",
            Self::Sign => "sign",
            Self::Clamp => "clamp",
        }
    }
}

/// Runs an elementwise unary op (`abs`, `neg`, `sign`, `clamp`) over `i32` tensors.
pub fn run_i32_unary(
    op_code: u32,
    id_a: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<(&[u32], &[u32], &[u32])>,
    p_z: u32,
    p_w: u32,
) -> Hc4jResult<()> {
    let op = I32UnaryOp::from_code(op_code)?;
    let shader_src = generate_unary_i32_shader(op.code());
    let pipeline_name = format!("math_unary_i32_{}", op.name());
    let pipeline = get_engine()?.get_or_compile(&pipeline_name, "main", &shader_src)?;

    match strided {
        None => {
            check_contiguous_length(id_out, length)?;
            run_contiguous(
                &pipeline,
                &|len| bytemuck::bytes_of(&UnaryElemDims::contiguous_with_params(len, p_z, p_w)).to_vec(),
                &[id_a],
                id_out,
            )
        }
        Some((shape, strides_a, strides_c)) => {
            let length_u32 = u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("length exceeds u32"))?;
            let (pruned_shape, pruned_strides) = prune_strided_layout(shape, &[strides_a, strides_c])?;
            let dims = UnaryElemDims {
                params: [length_u32, 0, p_z, p_w],
                shape: pruned_shape,
                strides_a: pruned_strides[0],
                strides_c: pruned_strides[1],
            };
            run_strided(&pipeline, &[id_a, id_out], bytemuck::bytes_of(&dims), length as u64)
        }
    }
}
