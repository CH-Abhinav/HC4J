use crate::get_engine;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ElemDims {
    pub length: u32,
    pub rank: u32,
    pub is_contiguous: u32,
    pub _pad1: u32,
    pub shape: [u32; 8],
    pub strides_a: [u32; 8],
    pub strides_b: [u32; 8],
    pub strides_c: [u32; 8],
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DataType {
    I32,
    F32,
}

impl DataType {
    pub fn from_u32(val: u32) -> Self {
        match val {
            0 => DataType::I32,
            1 => DataType::F32,
            _ => DataType::F32,
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
        match self {
            DataType::I32 => "i32",
            DataType::F32 => "f32",
        }
    }
}

pub const ELEM_SHADER_TEMPLATE: &str = r#"
struct Dims {
    length: u32,
    rank: u32,
    is_contiguous: u32,
    _p1: u32,
    shape: array<u32, 8>,
    strides_a: array<u32, 8>,
    strides_b: array<u32, 8>,
    strides_c: array<u32, 8>,
}

@group(0) @binding(0) var<storage, read> arrayA: array<__SCALAR__>;
@group(0) @binding(1) var<storage, read> arrayB: array<__SCALAR__>;
@group(0) @binding(2) var<storage, read_write> arrayC: array<__SCALAR__>;
@group(0) @binding(3) var<storage, read> dims: Dims;

fn get_indices(logical_idx: u32) -> vec3<u32> { 
    var remaining = logical_idx;
    var offset_a = 0u;
    var offset_b = 0u;
    var offset_c = 0u;
    for (var d_idx = 0u; d_idx < dims.rank; d_idx = d_idx + 1u) {
        let d = dims.rank - 1u - d_idx;
        let dim_size = dims.shape[d];

        if (dim_size == 0u) { continue; }

        let coord = remaining % dim_size;
        remaining = remaining / dim_size;

        offset_a = offset_a + coord * dims.strides_a[d];
        offset_b = offset_b + coord * dims.strides_b[d];    
        offset_c = offset_c + coord * dims.strides_c[d];
    }
    return vec3<u32>(offset_a, offset_b, offset_c);
}

@compute @workgroup_size(256)
fn add_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (dims.is_contiguous == 1u) {
        let base = gid.x * 4u;
        if (base + 3u < dims.length) {
            let val_a = __VEC4__(arrayA[base], arrayA[base+1], arrayA[base+2], arrayA[base+3]);
            let val_b = __VEC4__(arrayB[base], arrayB[base+1], arrayB[base+2], arrayB[base+3]);
            let res = val_a + val_b;
            arrayC[base] = res.x;
            arrayC[base+1] = res.y;
            arrayC[base+2] = res.z;
            arrayC[base+3] = res.w;
        } else {
            for (var i = base; i < dims.length; i = i + 1u) {
                arrayC[i] = arrayA[i] + arrayB[i];
            }
        }
    } else { 
        let i = gid.x;
        if (i < dims.length) {
            let idxs = get_indices(i);
            arrayC[idxs.z] = arrayA[idxs.x] + arrayB[idxs.y];
        }
    }
}

@compute @workgroup_size(256)
fn sub_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (dims.is_contiguous == 1u) {
        let base = gid.x * 4u;
        if (base + 3u < dims.length) {
            let val_a = __VEC4__(arrayA[base], arrayA[base+1], arrayA[base+2], arrayA[base+3]);
            let val_b = __VEC4__(arrayB[base], arrayB[base+1], arrayB[base+2], arrayB[base+3]);
            let res = val_a - val_b;
            arrayC[base] = res.x;
            arrayC[base+1] = res.y;
            arrayC[base+2] = res.z;
            arrayC[base+3] = res.w;
        } else {
            for (var i = base; i < dims.length; i = i + 1u) {
                arrayC[i] = arrayA[i] - arrayB[i];
            }
        }
    } else {
        let i = gid.x;
        if (i < dims.length) {
            let idxs = get_indices(i);
            arrayC[idxs.z] = arrayA[idxs.x] - arrayB[idxs.y];
        }
    }
}

@compute @workgroup_size(256)
fn mul_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (dims.is_contiguous == 1u) {
        let base = gid.x * 4u;
        if (base + 3u < dims.length) {
            let val_a = __VEC4__(arrayA[base], arrayA[base+1], arrayA[base+2], arrayA[base+3]);
            let val_b = __VEC4__(arrayB[base], arrayB[base+1], arrayB[base+2], arrayB[base+3]);
            let res = val_a * val_b;
            arrayC[base] = res.x;
            arrayC[base+1] = res.y;
            arrayC[base+2] = res.z;
            arrayC[base+3] = res.w;
        } else {
            for (var i = base; i < dims.length; i = i + 1u) {
                arrayC[i] = arrayA[i] * arrayB[i];
            }
        }
    } else {
        let i = gid.x;
        if (i < dims.length) {
            let idxs = get_indices(i);
            arrayC[idxs.z] = arrayA[idxs.x] * arrayB[idxs.y];
        }
    }
}

@compute @workgroup_size(256)
fn div_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (dims.is_contiguous == 1u) {
        let base = gid.x * 4u;
        if (base + 3u < dims.length) {
            let val_a = __VEC4__(arrayA[base], arrayA[base+1], arrayA[base+2], arrayA[base+3]);
            let val_b = __VEC4__(arrayB[base], arrayB[base+1], arrayB[base+2], arrayB[base+3]);
            let res = val_a / val_b;
            arrayC[base] = res.x;
            arrayC[base+1] = res.y;
            arrayC[base+2] = res.z;
            arrayC[base+3] = res.w;
        } else {
            for (var i = base; i < dims.length; i = i + 1u) {
                arrayC[i] = arrayA[i] / arrayB[i];
            }
        }
    } else {
        let i = gid.x;
        if (i < dims.length) {
            let idxs = get_indices(i);
            arrayC[idxs.z] = arrayA[idxs.x] / arrayB[idxs.y];
        }
    }
}
"#;

pub fn generate_elem_shader(dtype: DataType) -> String {
    ELEM_SHADER_TEMPLATE
        .replace("__SCALAR__", dtype.wgsl_scalar())
        .replace("__VEC4__", dtype.wgsl_vec4())
        .replace("add_main", &format!("add_main_{}", dtype.as_str()))
        .replace("sub_main", &format!("sub_main_{}", dtype.as_str()))
        .replace("mul_main", &format!("mul_main_{}", dtype.as_str()))
        .replace("div_main", &format!("div_main_{}", dtype.as_str()))
}

fn dispatch_elem_op(
    op_name: &str,
    base_entry: &str,
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
    let dtype = DataType::from_u32(dtype_code);

    let mut dims = ElemDims {
        length: length as u32,
        rank,
        is_contiguous,
        _pad1: 0,
        shape: [1; 8],
        strides_a: [0; 8],
        strides_b: [0; 8],
        strides_c: [0; 8],
    };

    if is_contiguous == 0 && rank > 0 && !ptr_shape.is_null() {
        let r = std::cmp::min(rank as usize, 8);
        unsafe {
            dims.shape[..r].copy_from_slice(std::slice::from_raw_parts(ptr_shape, r));
            dims.strides_a[..r].copy_from_slice(std::slice::from_raw_parts(ptr_strides_a, r));
            dims.strides_b[..r].copy_from_slice(std::slice::from_raw_parts(ptr_strides_b, r));
            dims.strides_c[..r].copy_from_slice(std::slice::from_raw_parts(ptr_strides_c, r));
        }
    }

    let eng = get_engine();
    let shader_name = format!("elem_{}_{}", op_name, dtype.as_str());
    let entry_point = format!("{}_{}", base_entry, dtype.as_str());
    let wgsl_code = generate_elem_shader(dtype);
    let pipeline = eng.get_or_compile(&shader_name, &entry_point, &wgsl_code);
    
    execute_elementwise(id_a, id_b, id_out, dims, &pipeline, eng)
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_add(
    id_a: u64, id_b: u64, id_out: u64, rank: u32,
    ptr_shape: *const u32, ptr_strides_a: *const u32, ptr_strides_b: *const u32, ptr_strides_c: *const u32,
    length: usize, is_contiguous: u32, dtype_code: u32,
) -> i32 {
    dispatch_elem_op(
        "add", "add_main", id_a, id_b, id_out, rank,
        ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c,
        length, is_contiguous, dtype_code,
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_sub(
    id_a: u64, id_b: u64, id_out: u64, rank: u32,
    ptr_shape: *const u32, ptr_strides_a: *const u32, ptr_strides_b: *const u32, ptr_strides_c: *const u32,
    length: usize, is_contiguous: u32, dtype_code: u32,
) -> i32 {
    dispatch_elem_op(
        "sub", "sub_main", id_a, id_b, id_out, rank,
        ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c,
        length, is_contiguous, dtype_code,
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_mul(
    id_a: u64, id_b: u64, id_out: u64, rank: u32,
    ptr_shape: *const u32, ptr_strides_a: *const u32, ptr_strides_b: *const u32, ptr_strides_c: *const u32,
    length: usize, is_contiguous: u32, dtype_code: u32,
) -> i32 {
    dispatch_elem_op(
        "mul", "mul_main", id_a, id_b, id_out, rank,
        ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c,
        length, is_contiguous, dtype_code,
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_div(
    id_a: u64, id_b: u64, id_out: u64, rank: u32,
    ptr_shape: *const u32, ptr_strides_a: *const u32, ptr_strides_b: *const u32, ptr_strides_c: *const u32,
    length: usize, is_contiguous: u32, dtype_code: u32,
) -> i32 {
    dispatch_elem_op(
        "div", "div_main", id_a, id_b, id_out, rank,
        ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c,
        length, is_contiguous, dtype_code,
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_add_f32(
    id_a: u64, id_b: u64, id_out: u64, rank: u32,
    ptr_shape: *const u32, ptr_strides_a: *const u32, ptr_strides_b: *const u32, ptr_strides_c: *const u32,
    length: usize, is_contiguous: u32,
) -> i32 {
    dispatch_add(
        id_a, id_b, id_out, rank, ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c,
        length, is_contiguous, 1,
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_sub_f32(
    id_a: u64, id_b: u64, id_out: u64, rank: u32,
    ptr_shape: *const u32, ptr_strides_a: *const u32, ptr_strides_b: *const u32, ptr_strides_c: *const u32,
    length: usize, is_contiguous: u32,
) -> i32 {
    dispatch_sub(
        id_a, id_b, id_out, rank, ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c,
        length, is_contiguous, 1,
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_mul_f32(
    id_a: u64, id_b: u64, id_out: u64, rank: u32,
    ptr_shape: *const u32, ptr_strides_a: *const u32, ptr_strides_b: *const u32, ptr_strides_c: *const u32,
    length: usize, is_contiguous: u32,
) -> i32 {
    dispatch_mul(
        id_a, id_b, id_out, rank, ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c,
        length, is_contiguous, 1,
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch_div_f32(
    id_a: u64, id_b: u64, id_out: u64, rank: u32,
    ptr_shape: *const u32, ptr_strides_a: *const u32, ptr_strides_b: *const u32, ptr_strides_c: *const u32,
    length: usize, is_contiguous: u32,
) -> i32 {
    dispatch_div(
        id_a, id_b, id_out, rank, ptr_shape, ptr_strides_a, ptr_strides_b, ptr_strides_c,
        length, is_contiguous, 1,
    )
}

fn execute_elementwise(
    id_a: u64,
    id_b: u64,
    id_out: u64,
    dims: ElemDims,
    pipeline: &wgpu::ComputePipeline,
    eng: &crate::GpuEngine,
) -> i32 {
    let (buf_a, buf_b, buf_out) = {
        let guard = match crate::memory::get_registry().lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        
        let a = match guard.get(&id_a) {
            Some(b) => b.clone(),
            None => return crate::memory::HC4J_ERR_NOT_FOUND,
        };
        let b = match guard.get(&id_b) {
            Some(b) => b.clone(),
            None => return crate::memory::HC4J_ERR_NOT_FOUND,
        };
        let out = match guard.get(&id_out) {
            Some(b) => b.clone(),
            None => return crate::memory::HC4J_ERR_NOT_FOUND,
        };
        
        (a, b, out)
    };

     let meta = eng
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ElemDims Meta"),
            contents: bytemuck::cast_slice(&[dims]),
            usage: wgpu::BufferUsages::STORAGE, // <-- Changed to STORAGE
        });

    let bg = eng.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: meta.as_entire_binding(),
            },
        ],
    });

    let workgroups = if dims.is_contiguous == 1 {
        (dims.length + 1023) / 1024
    } else {
        (dims.length + 255) / 256
    };

    let mut enc = eng
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cp.set_pipeline(pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(workgroups, 1, 1);
    }

    eng.queue.submit(Some(enc.finish()));
    
    crate::memory::HC4J_SUCCESS
}