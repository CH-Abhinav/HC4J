//! Extra elementwise math, comparisons, ternary where, and type casting kernels.

use crate::error::{ffi_guard, Hc4jError};
use crate::get_engine;
use crate::ops::arithmetic::DataType;
use crate::ops::elementwise::{
    check_contiguous_length, layout::pruned_layout, layout::BinaryElemDims,
    layout::TernaryElemDims, layout::UnaryElemDims, run_contiguous, run_strided, shaders::UnaryKernel,
    dispatch_unary_kernel,
};

// ============================================================================
// 1. Unary f32 Operations
// ============================================================================

/// Computes elementwise absolute value `|x|` on f32.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_abs_f32(
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
) -> i32 {
    unsafe {
        dispatch_unary_kernel(
            UnaryKernel::builtin("math", "abs", "abs"),
            id_a,
            id_out,
            rank,
            ptr_shape,
            ptr_strides_a,
            ptr_strides_c,
            length,
            is_contiguous,
        )
    }
}

const NEG_PRELUDE: &str = r#"
fn neg4(v: vec4<f32>) -> vec4<f32> { return -v; }
fn neg1(v: f32) -> f32 { return -v; }
"#;

/// Computes elementwise negation `-x` on f32.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_neg_f32(
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
) -> i32 {
    let kernel = UnaryKernel {
        family: "math",
        op_name: "neg",
        prelude: NEG_PRELUDE,
        op4: "neg4",
        op1: "neg1",
        alpha_bits: 0,
        param_w_bits: 0,
    };
    unsafe {
        dispatch_unary_kernel(
            kernel,
            id_a,
            id_out,
            rank,
            ptr_shape,
            ptr_strides_a,
            ptr_strides_c,
            length,
            is_contiguous,
        )
    }
}

/// Computes elementwise reciprocal square root `1 / sqrt(x)` on f32.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_rsqrt_f32(
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
) -> i32 {
    unsafe {
        dispatch_unary_kernel(
            UnaryKernel::builtin("math", "rsqrt", "inverseSqrt"),
            id_a,
            id_out,
            rank,
            ptr_shape,
            ptr_strides_a,
            ptr_strides_c,
            length,
            is_contiguous,
        )
    }
}

/// Computes elementwise floor on f32.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_floor_f32(
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
) -> i32 {
    unsafe {
        dispatch_unary_kernel(
            UnaryKernel::builtin("math", "floor", "floor"),
            id_a,
            id_out,
            rank,
            ptr_shape,
            ptr_strides_a,
            ptr_strides_c,
            length,
            is_contiguous,
        )
    }
}

/// Computes elementwise ceiling on f32.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_ceil_f32(
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
) -> i32 {
    unsafe {
        dispatch_unary_kernel(
            UnaryKernel::builtin("math", "ceil", "ceil"),
            id_a,
            id_out,
            rank,
            ptr_shape,
            ptr_strides_a,
            ptr_strides_c,
            length,
            is_contiguous,
        )
    }
}

/// Computes elementwise round (to nearest even) on f32.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_round_f32(
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
) -> i32 {
    unsafe {
        dispatch_unary_kernel(
            UnaryKernel::builtin("math", "round", "round"),
            id_a,
            id_out,
            rank,
            ptr_shape,
            ptr_strides_a,
            ptr_strides_c,
            length,
            is_contiguous,
        )
    }
}

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

/// Computes elementwise sign (-1.0, 0.0, 1.0) on f32.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_sign_f32(
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
) -> i32 {
    let kernel = UnaryKernel {
        family: "math",
        op_name: "sign",
        prelude: SIGN_F32_PRELUDE,
        op4: "sign4_f32",
        op1: "sign1_f32",
        alpha_bits: 0,
        param_w_bits: 0,
    };
    unsafe {
        dispatch_unary_kernel(
            kernel,
            id_a,
            id_out,
            rank,
            ptr_shape,
            ptr_strides_a,
            ptr_strides_c,
            length,
            is_contiguous,
        )
    }
}

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

/// Computes elementwise clamp to `[lo, hi]` on f32.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_clamp_f32(
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
    lo: f32,
    hi: f32,
) -> i32 {
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return ffi_guard("dispatch_clamp_f32", || {
            Err(Hc4jError::InvalidParam("clamp lo and hi must be finite and lo <= hi"))
        });
    }
    let kernel = UnaryKernel {
        family: "math",
        op_name: "clamp",
        prelude: CLAMP_F32_PRELUDE,
        op4: "clamp4",
        op1: "clamp1",
        alpha_bits: lo.to_bits(),
        param_w_bits: hi.to_bits(),
    };
    unsafe {
        dispatch_unary_kernel(
            kernel,
            id_a,
            id_out,
            rank,
            ptr_shape,
            ptr_strides_a,
            ptr_strides_c,
            length,
            is_contiguous,
        )
    }
}

// ============================================================================
// 2. Unary i32 Operations (abs, neg, sign, clamp)
// ============================================================================

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

/// Computes elementwise unary op (abs, neg, sign, clamp) on i32.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_unary_i32(
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
    op_code: u32,
    p_z: u32,
    p_w: u32,
) -> i32 {
    ffi_guard("dispatch_unary_i32", || {
        let op_name = match op_code {
            0 => "abs",
            1 => "neg",
            2 => "sign",
            3 => "clamp",
            _ => return Err(Hc4jError::InvalidParam("unknown i32 unary op_code")),
        };
        let shader_src = generate_unary_i32_shader(op_code);
        let pipeline_name = format!("math_unary_i32_{op_name}");
        let pipeline = get_engine()?.get_or_compile(&pipeline_name, "main", &shader_src)?;

        if is_contiguous != 0 {
            check_contiguous_length(id_out, length)?;
            return run_contiguous(
                &pipeline,
                &|len| bytemuck::bytes_of(&UnaryElemDims::contiguous_with_params(len, p_z, p_w)).to_vec(),
                &[id_a],
                id_out,
            );
        }
        let length_u32 = u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("length exceeds u32"))?;
        let (shape, strides) = unsafe { pruned_layout(rank, ptr_shape, &[ptr_strides_a, ptr_strides_c])? };
        let dims = UnaryElemDims {
            params: [length_u32, 0, p_z, p_w],
            shape,
            strides_a: strides[0],
            strides_c: strides[1],
        };
        run_strided(&pipeline, &[id_a, id_out], bytemuck::bytes_of(&dims), length as u64)
    })
}

// ============================================================================
// 3. Functional Binary Math (pow, maximum, minimum)
// ============================================================================

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

/// Computes elementwise power `a ^ b` for f32 tensors.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_pow_f32(
    id_a: u64,
    id_b: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_b: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
) -> i32 {
    ffi_guard("dispatch_pow_f32", || {
        let shader = generate_functional_binary_shader(DataType::F32, "pow");
        let pipeline = get_engine()?.get_or_compile("func_bin_pow_f32", "main", &shader)?;

        if is_contiguous != 0 {
            check_contiguous_length(id_out, length)?;
            return run_contiguous(
                &pipeline,
                &|len| bytemuck::bytes_of(&BinaryElemDims::contiguous(len)).to_vec(),
                &[id_a, id_b],
                id_out,
            );
        }
        let length_u32 = u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("length exceeds u32"))?;
        let (shape, strides) = unsafe { pruned_layout(rank, ptr_shape, &[ptr_strides_a, ptr_strides_b, ptr_strides_c])? };
        let dims = BinaryElemDims {
            params: [length_u32, 0, 0, 0],
            shape,
            strides_a: strides[0],
            strides_b: strides[1],
            strides_c: strides[2],
        };
        run_strided(&pipeline, &[id_a, id_b, id_out], bytemuck::bytes_of(&dims), length as u64)
    })
}

/// Computes elementwise maximum `max(a, b)` for f32 or i32 tensors.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_maximum(
    id_a: u64,
    id_b: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_b: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
    dtype_code: u32,
) -> i32 {
    ffi_guard("dispatch_maximum", || {
        let dtype = DataType::from_u32(dtype_code).ok_or(Hc4jError::InvalidParam("invalid dtype code"))?;
        let shader = generate_functional_binary_shader(dtype, "max");
        let name = format!("func_bin_max_{}", dtype.as_str());
        let pipeline = get_engine()?.get_or_compile(&name, "main", &shader)?;

        if is_contiguous != 0 {
            check_contiguous_length(id_out, length)?;
            return run_contiguous(
                &pipeline,
                &|len| bytemuck::bytes_of(&BinaryElemDims::contiguous(len)).to_vec(),
                &[id_a, id_b],
                id_out,
            );
        }
        let length_u32 = u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("length exceeds u32"))?;
        let (shape, strides) = unsafe { pruned_layout(rank, ptr_shape, &[ptr_strides_a, ptr_strides_b, ptr_strides_c])? };
        let dims = BinaryElemDims {
            params: [length_u32, 0, 0, 0],
            shape,
            strides_a: strides[0],
            strides_b: strides[1],
            strides_c: strides[2],
        };
        run_strided(&pipeline, &[id_a, id_b, id_out], bytemuck::bytes_of(&dims), length as u64)
    })
}

/// Computes elementwise minimum `min(a, b)` for f32 or i32 tensors.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_minimum(
    id_a: u64,
    id_b: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_b: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
    dtype_code: u32,
) -> i32 {
    ffi_guard("dispatch_minimum", || {
        let dtype = DataType::from_u32(dtype_code).ok_or(Hc4jError::InvalidParam("invalid dtype code"))?;
        let shader = generate_functional_binary_shader(dtype, "min");
        let name = format!("func_bin_min_{}", dtype.as_str());
        let pipeline = get_engine()?.get_or_compile(&name, "main", &shader)?;

        if is_contiguous != 0 {
            check_contiguous_length(id_out, length)?;
            return run_contiguous(
                &pipeline,
                &|len| bytemuck::bytes_of(&BinaryElemDims::contiguous(len)).to_vec(),
                &[id_a, id_b],
                id_out,
            );
        }
        let length_u32 = u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("length exceeds u32"))?;
        let (shape, strides) = unsafe { pruned_layout(rank, ptr_shape, &[ptr_strides_a, ptr_strides_b, ptr_strides_c])? };
        let dims = BinaryElemDims {
            params: [length_u32, 0, 0, 0],
            shape,
            strides_a: strides[0],
            strides_b: strides[1],
            strides_c: strides[2],
        };
        run_strided(&pipeline, &[id_a, id_b, id_out], bytemuck::bytes_of(&dims), length as u64)
    })
}

// ============================================================================
// 4. Elementwise Comparisons (eq, ne, gt, ge, lt, le) -> output always i32
// ============================================================================

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

/// Computes elementwise comparison between tensors, returning an i32 0/1 mask.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_compare(
    id_a: u64,
    id_b: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_b: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
    dtype_code: u32,
    cmp_op: u32,
) -> i32 {
    ffi_guard("dispatch_compare", || {
        let dtype = DataType::from_u32(dtype_code).ok_or(Hc4jError::InvalidParam("invalid dtype code"))?;
        let shader = generate_compare_shader(dtype, cmp_op)?;
        let name = format!("compare_{}_{cmp_op}", dtype.as_str());
        let pipeline = get_engine()?.get_or_compile(&name, "main", &shader)?;

        if is_contiguous != 0 {
            check_contiguous_length(id_out, length)?;
            return run_contiguous(
                &pipeline,
                &|len| bytemuck::bytes_of(&BinaryElemDims::contiguous(len)).to_vec(),
                &[id_a, id_b],
                id_out,
            );
        }
        let length_u32 = u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("length exceeds u32"))?;
        let (shape, strides) = unsafe { pruned_layout(rank, ptr_shape, &[ptr_strides_a, ptr_strides_b, ptr_strides_c])? };
        let dims = BinaryElemDims {
            params: [length_u32, 0, 0, 0],
            shape,
            strides_a: strides[0],
            strides_b: strides[1],
            strides_c: strides[2],
        };
        run_strided(&pipeline, &[id_a, id_b, id_out], bytemuck::bytes_of(&dims), length as u64)
    })
}

// ============================================================================
// 5. Ternary Masked Select (dispatch_where)
// ============================================================================

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

/// Computes ternary where operation `cond != 0 ? x : y`.
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the shape and stride pointers must point to `rank` readable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_where(
    id_cond: u64,
    id_x: u64,
    id_y: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_cond: *const u32,
    ptr_strides_x: *const u32,
    ptr_strides_y: *const u32,
    ptr_strides_out: *const u32,
    length: usize,
    is_contiguous: u32,
    cond_dtype: u32,
    val_dtype: u32,
) -> i32 {
    ffi_guard("dispatch_where", || {
        let cd = DataType::from_u32(cond_dtype).ok_or(Hc4jError::InvalidParam("invalid cond_dtype"))?;
        let vd = DataType::from_u32(val_dtype).ok_or(Hc4jError::InvalidParam("invalid val_dtype"))?;
        let shader = generate_where_shader(cd, vd);
        let name = format!("where_{}_{}", cd.as_str(), vd.as_str());
        let pipeline = get_engine()?.get_or_compile(&name, "main", &shader)?;

        let length_u32 = u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("length exceeds u32"))?;

        if is_contiguous != 0 {
            check_contiguous_length(id_out, length)?;
            let dims = TernaryElemDims {
                params: [length_u32, 1, 0, 0],
                shape: [1; 8],
                strides_cond: [0; 8],
                strides_x: [0; 8],
                strides_y: [0; 8],
                strides_out: [0; 8],
            };
            return run_strided(
                &pipeline,
                &[id_cond, id_x, id_y, id_out],
                bytemuck::bytes_of(&dims),
                length as u64,
            );
        }

        let (shape, strides) = unsafe {
            pruned_layout(
                rank,
                ptr_shape,
                &[ptr_strides_cond, ptr_strides_x, ptr_strides_y, ptr_strides_out],
            )?
        };
        let dims = TernaryElemDims {
            params: [length_u32, 0, 0, 0],
            shape,
            strides_cond: strides[0],
            strides_x: strides[1],
            strides_y: strides[2],
            strides_out: strides[3],
        };
        run_strided(
            &pipeline,
            &[id_cond, id_x, id_y, id_out],
            bytemuck::bytes_of(&dims),
            length as u64,
        )
    })
}

// ============================================================================
// 6. GPU Type Casting (dispatch_cast)
// ============================================================================

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

/// Converts tensor elements between supported data types (f32 <-> i32).
///
/// # Safety
/// The caller must ensure that `id_src` and `id_out` are valid buffer IDs with at least `length` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_cast(
    id_src: u64,
    id_out: u64,
    length: u64,
    src_dtype: u32,
    dst_dtype: u32,
) -> i32 {
    ffi_guard("dispatch_cast", || {
        let _ = u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("length exceeds u32"))?;
        let src_dt = DataType::from_u32(src_dtype).ok_or(Hc4jError::InvalidParam("invalid src_dtype"))?;
        let dst_dt = DataType::from_u32(dst_dtype).ok_or(Hc4jError::InvalidParam("invalid dst_dtype"))?;

        if src_dt == dst_dt {
            // Same dtype: copy
            let pipeline = get_engine()?.get_or_compile("cast_copy", "main", CAST_I32_TO_F32_WGSL.replace("f32(src", "src").as_str())?;
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
    })
}
