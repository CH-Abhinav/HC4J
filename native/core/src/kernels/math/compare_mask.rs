//! Functional binary math (`pow`, `maximum`, `minimum`), comparisons (`==`, `!=`, `>`, `>=`, `<`, `<=`),
//! and ternary masked selection (`where`).

use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::engine::get_engine;
use crate::kernels::elementwise::{
    BinaryStrided, TernaryElemDims, TernaryStrided, check_contiguous_length, prune_strided_layout,
    run_binary_elementwise, run_strided,
};
use crate::kernels::math::arithmetic::DataType;
use crate::memory::TensorId;

pub const FUNCTIONAL_BINARY_SHADER_TEMPLATE: &str = r#"
struct Dims {
    params: vec4<u32>,
    shape: array<vec4<u32>, 2>,
    strides_a: array<vec4<u32>, 2>,
    strides_b: array<vec4<u32>, 2>,
    strides_c: array<vec4<u32>, 2>,
}

@group(0) @binding(0) var<storage, read_write> arrayA: array<__SCALAR__>;
@group(0) @binding(1) var<storage, read_write> arrayB: array<__SCALAR__>;
@group(0) @binding(2) var<storage, read_write> arrayC: array<__SCALAR__>;
@group(0) @binding(3) var<uniform> dims: Dims;

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
            let a = __VEC4__(arrayA[base], arrayA[base + 1u], arrayA[base + 2u], arrayA[base + 3u]);
            let b = __VEC4__(arrayB[base], arrayB[base + 1u], arrayB[base + 2u], arrayB[base + 3u]);
            let r = __FUNC__(a, b);
            arrayC[base] = r.x;
            arrayC[base + 1u] = r.y;
            arrayC[base + 2u] = r.z;
            arrayC[base + 3u] = r.w;
        } else {
            arrayC[base] = __FUNC__(arrayA[base], arrayB[base]);
            if (base + 1u < length) {
                arrayC[base + 1u] = __FUNC__(arrayA[base + 1u], arrayB[base + 1u]);
                if (base + 2u < length) {
                    arrayC[base + 2u] = __FUNC__(arrayA[base + 2u], arrayB[base + 2u]);
                }
            }
        }
    } else {
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
        arrayC[offset_c] = __FUNC__(arrayA[offset_a], arrayB[offset_b]);
    }
}
"#;

pub fn generate_functional_binary_shader(dtype: DataType, func_name: &str) -> String {
    FUNCTIONAL_BINARY_SHADER_TEMPLATE
        .replace("__SCALAR__", dtype.wgsl_scalar())
        .replace("__VEC4__", dtype.wgsl_vec4())
        .replace("__FUNC__", func_name)
}

pub fn run_pow_f32(
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<BinaryStrided<'_>>,
) -> Hc4jResult<()> {
    let shader = generate_functional_binary_shader(DataType::F32, "pow");
    let pipeline = get_engine()?.get_or_compile("func_bin_pow_f32", "main", &shader)?;
    run_binary_elementwise(&pipeline, id_a, id_b, id_out, length, strided)
}

pub fn run_maximum(
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<BinaryStrided<'_>>,
    dtype_code: u32,
) -> Hc4jResult<()> {
    let dtype = DataType::from_u32(dtype_code).ok_or(Hc4jError::InvalidParam("invalid dtype code"))?;
    let shader = generate_functional_binary_shader(dtype, "max");
    let name = format!("func_bin_max_{}", dtype.as_str());
    let pipeline = get_engine()?.get_or_compile(&name, "main", &shader)?;
    run_binary_elementwise(&pipeline, id_a, id_b, id_out, length, strided)
}

pub fn run_minimum(
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<BinaryStrided<'_>>,
    dtype_code: u32,
) -> Hc4jResult<()> {
    let dtype = DataType::from_u32(dtype_code).ok_or(Hc4jError::InvalidParam("invalid dtype code"))?;
    let shader = generate_functional_binary_shader(dtype, "min");
    let name = format!("func_bin_min_{}", dtype.as_str());
    let pipeline = get_engine()?.get_or_compile(&name, "main", &shader)?;
    run_binary_elementwise(&pipeline, id_a, id_b, id_out, length, strided)
}

pub const COMPARE_SHADER_TEMPLATE: &str = r#"
struct Dims {
    params: vec4<u32>,
    shape: array<vec4<u32>, 2>,
    strides_a: array<vec4<u32>, 2>,
    strides_b: array<vec4<u32>, 2>,
    strides_c: array<vec4<u32>, 2>,
}

@group(0) @binding(0) var<storage, read_write> arrayA: array<__INPUT_TYPE__>;
@group(0) @binding(1) var<storage, read_write> arrayB: array<__INPUT_TYPE__>;
@group(0) @binding(2) var<storage, read_write> arrayC: array<i32>;
@group(0) @binding(3) var<uniform> dims: Dims;

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
            let a = __VEC4__(arrayA[base], arrayA[base + 1u], arrayA[base + 2u], arrayA[base + 3u]);
            let b = __VEC4__(arrayB[base], arrayB[base + 1u], arrayB[base + 2u], arrayB[base + 3u]);
            let cond = a __CMP__ b;
            let r = select(vec4<i32>(0), vec4<i32>(1), cond);
            arrayC[base] = r.x;
            arrayC[base + 1u] = r.y;
            arrayC[base + 2u] = r.z;
            arrayC[base + 3u] = r.w;
        } else {
            let cond = arrayA[base] __CMP__ arrayB[base];
            arrayC[base] = select(0, 1, cond);
            if (base + 1u < length) {
                let cond1 = arrayA[base + 1u] __CMP__ arrayB[base + 1u];
                arrayC[base + 1u] = select(0, 1, cond1);
                if (base + 2u < length) {
                    let cond2 = arrayA[base + 2u] __CMP__ arrayB[base + 2u];
                    arrayC[base + 2u] = select(0, 1, cond2);
                }
            }
        }
    } else {
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
        let cond = arrayA[offset_a] __CMP__ arrayB[offset_b];
        arrayC[offset_c] = select(0, 1, cond);
    }
}
"#;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CompareOp {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

impl CompareOp {
    pub fn from_code(code: u32) -> Hc4jResult<Self> {
        match code {
            0 => Ok(Self::Eq),
            1 => Ok(Self::Ne),
            2 => Ok(Self::Gt),
            3 => Ok(Self::Ge),
            4 => Ok(Self::Lt),
            5 => Ok(Self::Le),
            _ => Err(Hc4jError::InvalidParam("invalid comparison op code")),
        }
    }

    pub fn code(self) -> u32 {
        match self {
            Self::Eq => 0,
            Self::Ne => 1,
            Self::Gt => 2,
            Self::Ge => 3,
            Self::Lt => 4,
            Self::Le => 5,
        }
    }
}

pub fn generate_compare_shader(dtype: DataType, cmp_op: u32) -> Result<String, Hc4jError> {
    let op_symbol = match cmp_op {
        0 => "==",
        1 => "!=",
        2 => ">",
        3 => ">=",
        4 => "<",
        5 => "<=",
        _ => return Err(Hc4jError::InvalidParam("invalid comparison op code")),
    };
    Ok(COMPARE_SHADER_TEMPLATE
        .replace("__INPUT_TYPE__", dtype.wgsl_scalar())
        .replace("__VEC4__", dtype.wgsl_vec4())
        .replace("__CMP__", op_symbol))
}

pub fn run_compare(
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<BinaryStrided<'_>>,
    dtype_code: u32,
    cmp_op: u32,
) -> Hc4jResult<()> {
    let dtype = DataType::from_u32(dtype_code).ok_or(Hc4jError::InvalidParam("invalid dtype code"))?;
    let shader = generate_compare_shader(dtype, cmp_op)?;
    let name = format!("compare_{}_{cmp_op}", dtype.as_str());
    let pipeline = get_engine()?.get_or_compile(&name, "main", &shader)?;
    run_binary_elementwise(&pipeline, id_a, id_b, id_out, length, strided)
}

pub const WHERE_SHADER_TEMPLATE: &str = r#"
struct TernaryDims {
    params: vec4<u32>,
    shape: array<vec4<u32>, 2>,
    strides_cond: array<vec4<u32>, 2>,
    strides_x: array<vec4<u32>, 2>,
    strides_y: array<vec4<u32>, 2>,
    strides_out: array<vec4<u32>, 2>,
}

@group(0) @binding(0) var<storage, read_write> arrayCond: array<__COND_SCALAR__>;
@group(0) @binding(1) var<storage, read_write> arrayX: array<__VAL_SCALAR__>;
@group(0) @binding(2) var<storage, read_write> arrayY: array<__VAL_SCALAR__>;
@group(0) @binding(3) var<storage, read_write> arrayOut: array<__VAL_SCALAR__>;
@group(0) @binding(4) var<uniform> dims: TernaryDims;

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
            let c = __COND_VEC4__(arrayCond[base], arrayCond[base + 1u], arrayCond[base + 2u], arrayCond[base + 3u]);
            let x = __VAL_VEC4__(arrayX[base], arrayX[base + 1u], arrayX[base + 2u], arrayX[base + 3u]);
            let y = __VAL_VEC4__(arrayY[base], arrayY[base + 1u], arrayY[base + 2u], arrayY[base + 3u]);
            let mask = c != __COND_ZERO_VEC4__;
            let r = select(y, x, mask);
            arrayOut[base] = r.x;
            arrayOut[base + 1u] = r.y;
            arrayOut[base + 2u] = r.z;
            arrayOut[base + 3u] = r.w;
        } else {
            let mask = arrayCond[base] != __COND_ZERO__;
            arrayOut[base] = select(arrayY[base], arrayX[base], mask);
            if (base + 1u < length) {
                let mask1 = arrayCond[base + 1u] != __COND_ZERO__;
                arrayOut[base + 1u] = select(arrayY[base + 1u], arrayX[base + 1u], mask1);
                if (base + 2u < length) {
                    let mask2 = arrayCond[base + 2u] != __COND_ZERO__;
                    arrayOut[base + 2u] = select(arrayY[base + 2u], arrayX[base + 2u], mask2);
                }
            }
        }
    } else {
        let i = tid;
        if (i >= length) { return; }
        var remaining = i;
        var offset_c = 0u;
        var offset_x = 0u;
        var offset_y = 0u;
        var offset_out = 0u;
        for (var d = 0u; d < 8u; d = d + 1u) {
            let dim_size = dims.shape[d >> 2u][d & 3u];
            if (dim_size > 1u) {
                let coord = remaining % dim_size;
                remaining = remaining / dim_size;
                offset_c = offset_c + coord * dims.strides_cond[d >> 2u][d & 3u];
                offset_x = offset_x + coord * dims.strides_x[d >> 2u][d & 3u];
                offset_y = offset_y + coord * dims.strides_y[d >> 2u][d & 3u];
                offset_out = offset_out + coord * dims.strides_out[d >> 2u][d & 3u];
            }
        }
        let mask = arrayCond[offset_c] != __COND_ZERO__;
        arrayOut[offset_out] = select(arrayY[offset_y], arrayX[offset_x], mask);
    }
}
"#;

pub fn generate_where_shader(cond_dtype: DataType, val_dtype: DataType) -> String {
    let (c_zero, c_zero_vec4) = match cond_dtype {
        DataType::I32 => ("0", "vec4<i32>(0)"),
        DataType::F32 => ("0.0", "vec4<f32>(0.0)"),
    };
    WHERE_SHADER_TEMPLATE
        .replace("__COND_SCALAR__", cond_dtype.wgsl_scalar())
        .replace("__COND_VEC4__", cond_dtype.wgsl_vec4())
        .replace("__COND_ZERO__", c_zero)
        .replace("__COND_ZERO_VEC4__", c_zero_vec4)
        .replace("__VAL_SCALAR__", val_dtype.wgsl_scalar())
        .replace("__VAL_VEC4__", val_dtype.wgsl_vec4())
}

#[allow(clippy::too_many_arguments)]
pub fn run_where(
    id_cond: TensorId,
    id_x: TensorId,
    id_y: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<TernaryStrided<'_>>,
    cond_dtype: u32,
    val_dtype: u32,
) -> Hc4jResult<()> {
    let cd = DataType::from_u32(cond_dtype).ok_or(Hc4jError::InvalidParam("invalid cond_dtype"))?;
    let vd = DataType::from_u32(val_dtype).ok_or(Hc4jError::InvalidParam("invalid val_dtype"))?;
    let shader = generate_where_shader(cd, vd);
    let name = format!("where_{}_{}", cd.as_str(), vd.as_str());
    let pipeline = get_engine()?.get_or_compile(&name, "main", &shader)?;

    let length_u32 = u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("length exceeds u32"))?;

    match strided {
        None => {
            check_contiguous_length(id_out, length)?;
            let dims = TernaryElemDims {
                params: [length_u32, 1, 0, 0],
                shape: [1; 8],
                strides_cond: [0; 8],
                strides_x: [0; 8],
                strides_y: [0; 8],
                strides_out: [0; 8],
            };
            run_strided(
                &pipeline,
                &[id_cond, id_x, id_y, id_out],
                bytemuck::bytes_of(&dims),
                length as u64,
            )
        }
        Some((shape, strides_cond, strides_x, strides_y, strides_out)) => {
            let (pruned_shape, pruned_strides) =
                prune_strided_layout(shape, &[strides_cond, strides_x, strides_y, strides_out])?;
            let dims = TernaryElemDims {
                params: [length_u32, 0, 0, 0],
                shape: pruned_shape,
                strides_cond: pruned_strides[0],
                strides_x: pruned_strides[1],
                strides_y: pruned_strides[2],
                strides_out: pruned_strides[3],
            };
            run_strided(
                &pipeline,
                &[id_cond, id_x, id_y, id_out],
                bytemuck::bytes_of(&dims),
                length as u64,
            )
        }
    }
}
