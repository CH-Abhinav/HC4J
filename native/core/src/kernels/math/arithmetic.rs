//! Binary elementwise arithmetic (add, sub, mul, div) for i32 and f32.

use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::engine::get_engine;
use crate::kernels::elementwise::{BinaryStrided, run_binary_elementwise};
use crate::memory::TensorId;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DataType {
    I32,
    F32,
}

impl DataType {
    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(DataType::I32),
            1 => Some(DataType::F32),
            _ => None,
        }
    }

    pub fn wgsl_scalar(&self) -> &'static str {
        match self {
            DataType::I32 => "i32",
            DataType::F32 => "f32",
        }
    }

    pub fn wgsl_vec4(&self) -> &'static str {
        match self {
            DataType::I32 => "vec4<i32>",
            DataType::F32 => "vec4<f32>",
        }
    }

    pub fn as_str(&self) -> &'static str {
        self.wgsl_scalar()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOp {
    pub fn name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
        }
    }

    pub fn wgsl_symbol(&self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
        }
    }
}

pub const ELEM_SHADER_TEMPLATE: &str = r#"
struct Dims {
    params: vec4<u32>, // x = length, y = contiguous flag
    shape: array<vec4<u32>, 2>,
    strides_a: array<vec4<u32>, 2>,
    strides_b: array<vec4<u32>, 2>,
    strides_c: array<vec4<u32>, 2>,
}

// read_write everywhere: slab regions and in-place ops share buffers.
@group(0) @binding(0) var<storage, read_write> arrayA: array<__SCALAR__>;
@group(0) @binding(1) var<storage, read_write> arrayB: array<__SCALAR__>;
@group(0) @binding(2) var<storage, read_write> arrayC: array<__SCALAR__>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    // Grids beyond the per-dimension limit are folded into X*Y by the host.
    let tid = gid.x + gid.y * (nwg.x * 256u);
    let length = dims.params.x;

    if (dims.params.y == 1u) {
        // --- CONTIGUOUS FAST PATH: one vec4 per invocation ---
        let base = tid << 2u;
        if (base >= length) { return; }
        if (base + 3u < length) {
            let a = __VEC4__(arrayA[base], arrayA[base + 1u], arrayA[base + 2u], arrayA[base + 3u]);
            let b = __VEC4__(arrayB[base], arrayB[base + 1u], arrayB[base + 2u], arrayB[base + 3u]);
            let r = a __OP__ b;
            arrayC[base] = r.x;
            arrayC[base + 1u] = r.y;
            arrayC[base + 2u] = r.z;
            arrayC[base + 3u] = r.w;
        } else {
            arrayC[base] = arrayA[base] __OP__ arrayB[base];
            if (base + 1u < length) {
                arrayC[base + 1u] = arrayA[base + 1u] __OP__ arrayB[base + 1u];
                if (base + 2u < length) {
                    arrayC[base + 2u] = arrayA[base + 2u] __OP__ arrayB[base + 2u];
                }
            }
        }
    } else {
        // --- STRIDED PATH: reverse-pruned layout, no rank loop ---
        let i = tid;
        if (i >= length) { return; }
        var remaining = i;
        var offset_a = 0u;
        var offset_b = 0u;
        var offset_c = 0u;
        for (var d = 0u; d < 8u; d = d + 1u) {
            let dim_size = dims.shape[d >> 2u][d & 3u];
            if (dim_size > 1u) {
                let coord = remaining % dim_size;
                remaining = remaining / dim_size;
                offset_a = offset_a + coord * dims.strides_a[d >> 2u][d & 3u];
                offset_b = offset_b + coord * dims.strides_b[d >> 2u][d & 3u];
                offset_c = offset_c + coord * dims.strides_c[d >> 2u][d & 3u];
            }
        }
        arrayC[offset_c] = arrayA[offset_a] __OP__ arrayB[offset_b];
    }
}
"#;

pub fn generate_elem_shader(dtype: DataType, wgsl_op: &str) -> String {
    ELEM_SHADER_TEMPLATE
        .replace("__SCALAR__", dtype.wgsl_scalar())
        .replace("__VEC4__", dtype.wgsl_vec4())
        .replace("__OP__", wgsl_op)
}

fn binary_pipeline(op_name: &str, wgsl_op: &str, dtype: DataType) -> Hc4jResult<wgpu::ComputePipeline> {
    let shader_name = format!("elem_{op_name}_{}", dtype.as_str());
    get_engine()?.get_or_compile(&shader_name, "main", &generate_elem_shader(dtype, wgsl_op))
}

/// Runs an elementwise binary arithmetic operation over contiguous or strided tensors.
pub fn run_binary(
    op: BinaryOp,
    dtype_code: u32,
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<BinaryStrided<'_>>,
) -> Hc4jResult<()> {
    let dtype = DataType::from_u32(dtype_code).ok_or(Hc4jError::InvalidParam("unknown dtype code"))?;
    let pipeline = binary_pipeline(op.name(), op.wgsl_symbol(), dtype)?;
    run_binary_elementwise(&pipeline, id_a, id_b, id_out, length, strided)
}

pub fn run_add_f32(
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<BinaryStrided<'_>>,
) -> Hc4jResult<()> {
    run_binary(BinaryOp::Add, 1, id_a, id_b, id_out, length, strided)
}

pub fn run_sub_f32(
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<BinaryStrided<'_>>,
) -> Hc4jResult<()> {
    run_binary(BinaryOp::Sub, 1, id_a, id_b, id_out, length, strided)
}

pub fn run_mul_f32(
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<BinaryStrided<'_>>,
) -> Hc4jResult<()> {
    run_binary(BinaryOp::Mul, 1, id_a, id_b, id_out, length, strided)
}

pub fn run_div_f32(
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<BinaryStrided<'_>>,
) -> Hc4jResult<()> {
    run_binary(BinaryOp::Div, 1, id_a, id_b, id_out, length, strided)
}
