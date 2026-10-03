//! Binary elementwise arithmetic (add, sub, mul, div) for i32 and f32.
//!
//! Caller-allocated ABI (unchanged from the original entry points):
//!
//! ```c
//! int32_t dispatch_<op>(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
//!         const uint32_t* shape, const uint32_t* strides_a, const uint32_t* strides_b,
//!         const uint32_t* strides_c, size_t length, uint32_t is_contiguous, uint32_t dtype);
//! ```
//!
//! `dtype` is 0 for i32 and 1 for f32; the `_f32` variants omit it.
//! `id_out` may equal `id_a` or `id_b` (in place).

use crate::error::{Hc4jError, Hc4jResult, ffi_guard};
use crate::get_engine;
use crate::ops::elementwise::{BinaryElemDims, check_contiguous_length, pruned_layout, run_contiguous, run_strided};

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

#[allow(clippy::too_many_arguments)]
unsafe fn dispatch_elem_op(
    op_name: &str,
    wgsl_op: &str,
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
    ffi_guard(op_name, || {
        let dtype = DataType::from_u32(dtype_code).ok_or(Hc4jError::InvalidParam("unknown dtype code"))?;
        let pipeline = binary_pipeline(op_name, wgsl_op, dtype)?;
        if is_contiguous != 0 {
            check_contiguous_length(id_out, length)?;
            return run_contiguous(
                &pipeline,
                &|len| bytemuck::bytes_of(&BinaryElemDims::contiguous(len)).to_vec(),
                &[id_a, id_b],
                id_out,
            );
        }
        let length = u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("strided length exceeds u32"))?;
        let (shape, strides) = unsafe { pruned_layout(rank, ptr_shape, &[ptr_strides_a, ptr_strides_b, ptr_strides_c])? };
        let dims = BinaryElemDims {
            params: [length, 0, 0, 0],
            shape,
            strides_a: strides[0],
            strides_b: strides[1],
            strides_c: strides[2],
        };
        run_strided(&pipeline, &[id_a, id_b, id_out], bytemuck::bytes_of(&dims), length as u64)
    })
}

macro_rules! impl_binary_op {
    ($fn_name:ident, $fn_f32:ident, $op_name:expr, $wgsl_op:expr) => {
        /// # Safety
        /// For strided calls, the shape and the three stride pointers must
        /// each point to `rank` readable `u32` values.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            id_a: u64, id_b: u64, id_out: u64, rank: u32,
            ptr_shape: *const u32, ptr_strides_a: *const u32, ptr_strides_b: *const u32, ptr_strides_c: *const u32,
            length: usize, is_contiguous: u32, dtype_code: u32,
        ) -> i32 {
            unsafe {
                dispatch_elem_op(
                    $op_name, $wgsl_op, id_a, id_b, id_out, rank,
                    ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c,
                    length, is_contiguous, dtype_code,
                )
            }
        }

        /// # Safety
        /// As the dtype-generic entry point.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_f32(
            id_a: u64, id_b: u64, id_out: u64, rank: u32,
            ptr_shape: *const u32, ptr_strides_a: *const u32, ptr_strides_b: *const u32, ptr_strides_c: *const u32,
            length: usize, is_contiguous: u32,
        ) -> i32 {
            unsafe {
                $fn_name(
                    id_a, id_b, id_out, rank, ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c,
                    length, is_contiguous, 1,
                )
            }
        }
    };
}

impl_binary_op!(dispatch_add, dispatch_add_f32, "add", "+");
impl_binary_op!(dispatch_sub, dispatch_sub_f32, "sub", "-");
impl_binary_op!(dispatch_mul, dispatch_mul_f32, "mul", "*");
impl_binary_op!(dispatch_div, dispatch_div_f32, "div", "/");
