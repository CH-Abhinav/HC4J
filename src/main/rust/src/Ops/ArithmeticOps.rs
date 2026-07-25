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

pub const ELEM_SHADER: &str = r#"
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

@group(0) @binding(0) var<storage, read> arrayA: array<f32>;
@group(0) @binding(1) var<storage, read> arrayB: array<f32>;
@group(0) @binding(2) var<storage, read_write> arrayC: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

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
            let val_a = vec4<f32>(arrayA[base], arrayA[base+1], arrayA[base+2], arrayA[base+3]);
            let val_b = vec4<f32>(arrayB[base], arrayB[base+1], arrayB[base+2], arrayB[base+3]);
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
            let val_a = vec4<f32>(arrayA[base], arrayA[base+1], arrayA[base+2], arrayA[base+3]);
            let val_b = vec4<f32>(arrayB[base], arrayB[base+1], arrayB[base+2], arrayB[base+3]);
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
            let val_a = vec4<f32>(arrayA[base], arrayA[base+1], arrayA[base+2], arrayA[base+3]);
            let val_b = vec4<f32>(arrayB[base], arrayB[base+1], arrayB[base+2], arrayB[base+3]);
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
            let val_a = vec4<f32>(arrayA[base], arrayA[base+1], arrayA[base+2], arrayA[base+3]);
            let val_b = vec4<f32>(arrayB[base], arrayB[base+1], arrayB[base+2], arrayB[base+3]);
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

macro_rules! define_dispatch {
    ($func_name:ident, $pipeline_getter:ident) => {
        #[no_mangle]
        pub extern "C" fn $func_name(
            id_a: u64, id_b: u64, id_out: u64,
            rank: u32,
            ptr_shape: *const u32,
            ptr_strides_a: *const u32,
            ptr_strides_b: *const u32,
            ptr_strides_c: *const u32,
            length: usize,
            is_contiguous: u32,
        ) {
            let mut dims = ElemDims {
                length: length as u32, rank, is_contiguous, _pad1: 0,
                shape: [1; 8], strides_a: [0; 8], strides_b: [0; 8], strides_c: [0; 8],
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
            execute_elementwise(id_a, id_b, id_out, dims, &eng.$pipeline_getter, eng);
        }
    };
}

define_dispatch!(dispatch_add_f32, add_pipeline);
define_dispatch!(dispatch_sub_f32, sub_pipeline);
define_dispatch!(dispatch_mul_f32, mul_pipeline);
define_dispatch!(dispatch_div_f32, div_pipeline);

fn execute_elementwise(
    id_a: u64, id_b: u64, id_out: u64,
    dims: ElemDims, 
    pipeline: &wgpu::ComputePipeline, eng: &crate::GpuEngine
) {
    let registry = crate::memory::get_registry().lock().unwrap();
    let buf_a   = registry.get(&id_a).expect("HC4J: Unknown ID for buffer A");
    let buf_b   = registry.get(&id_b).expect("HC4J: Unknown ID for buffer B");
    let buf_out = registry.get(&id_out).expect("HC4J: Unknown ID for output buffer");

    let meta = eng.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("ElemDims Uniform"),
        contents: bytemuck::cast_slice(&[dims]),
        usage:    wgpu::BufferUsages::UNIFORM,
    });

    let bg = eng.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout:  &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: meta.as_entire_binding() },
        ],
    });

    let workgroups = if dims.is_contiguous == 1 {
        (dims.length + 1023) / 1024
    } else {
        (dims.length + 255) / 256
    };

    let mut enc = eng.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cp.set_pipeline(pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(workgroups, 1, 1);
    }

    eng.queue.submit(Some(enc.finish()));
}