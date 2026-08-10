use crate::get_engine;
use wgpu::util::DeviceExt;

// -------------------------------------------------------------------------
// ULTIMATE OPTIMIZATION: Zero-Branch Mathematical Identity Layout
// -------------------------------------------------------------------------
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UnaryElemDims {
    pub meta: [u32; 4],      // [length, is_contiguous, padding, padding]
    pub shape: [u32; 8],     // Maps to array<vec4<u32>, 2> in WGSL
    pub strides_a: [u32; 8], // Maps to array<vec4<u32>, 2> in WGSL
    pub strides_c: [u32; 8], // Maps to array<vec4<u32>, 2> in WGSL
}

pub const UNARY_F32_SHADER_TEMPLATE: &str = r#"
struct Dims {
    meta: vec4<u32>, 
    shape: array<vec4<u32>, 2>,
    strides_a: array<vec4<u32>, 2>,
    strides_c: array<vec4<u32>, 2>,
}

@group(0) @binding(0) var<storage, read> arrayA: array<f32>;
@group(0) @binding(1) var<storage, read_write> arrayC: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims; // UNIFORM: Hits fast constant cache!

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let length = dims.meta.x;

    if (dims.meta.y == 1u) {
        // --- CONTIGUOUS FAST PATH ---
        let base = gid.x << 2u; // base = gid.x * 4
        
        // Instant Warp Exit for out-of-bounds threads
        if (base >= length) { return; } 

        if (base + 3u < length) {
            // Highly optimized contiguous execution (4-wide vectorization)
            let val_a = vec4<f32>(arrayA[base], arrayA[base+1u], arrayA[base+2u], arrayA[base+3u]);
            let res = __WGSL_OP__(val_a);
            arrayC[base] = res.x;
            arrayC[base+1u] = res.y;
            arrayC[base+2u] = res.z;
            arrayC[base+3u] = res.w;
        } else {
            // Manually unrolled tail execution (avoids 'for' loop overhead on the GPU)
            arrayC[base] = __WGSL_OP__(arrayA[base]);
            if (base + 1u < length) {
                arrayC[base+1u] = __WGSL_OP__(arrayA[base+1u]);
                if (base + 2u < length) {
                    arrayC[base+2u] = __WGSL_OP__(arrayA[base+2u]);
                }
            }
        }
    } else { 
        // --- NON-CONTIGUOUS STRIDED PATH (ZERO-BRANCHING) ---
        let i = gid.x;
        
        // Instant Warp Exit
        if (i >= length) { return; } 
        
        var remaining = i;
        var offset_a = 0u;
        var offset_c = 0u;

        // 100% UNROLLED & BRANCHLESS LOOP!
        for (var d = 0u; d < 8u; d = d + 1u) {
            let dim_size = dims.shape[d >> 2u][d & 3u];
            
            // OPTIMIZATION: Skip expensive modulo/division hardware instructions if dimension is 1
            if (dim_size > 1u) {
                let coord = remaining % dim_size;
                remaining = remaining / dim_size;
                
                offset_a = offset_a + coord * dims.strides_a[d >> 2u][d & 3u];
                offset_c = offset_c + coord * dims.strides_c[d >> 2u][d & 3u];
            }
        }
        
        arrayC[offset_c] = __WGSL_OP__(arrayA[offset_a]);
    }
}
"#;

pub fn generate_unary_shader(wgsl_op: &str) -> String {
    UNARY_F32_SHADER_TEMPLATE.replace("__WGSL_OP__", wgsl_op)
}

fn dispatch_unary_op(
    op_name: &str,
    wgsl_op: &str,
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
) -> i32 {
    let mut dims = UnaryElemDims {
        meta: [length as u32, is_contiguous, 0, 0], 
        shape: [1; 8],     // Padding shape initialized to 1
        strides_a: [0; 8], // Padding strides initialized to 0
        strides_c: [0; 8], 
    };

    if is_contiguous == 0 && rank > 0 && !ptr_shape.is_null() {
        let r = std::cmp::min(rank as usize, 8);
        unsafe {
            let raw_shape = std::slice::from_raw_parts(ptr_shape, r);
            let raw_strides_a = std::slice::from_raw_parts(ptr_strides_a, r);
            let raw_strides_c = std::slice::from_raw_parts(ptr_strides_c, r);

            // OPTIMIZATION: Reverse-Pruning for Branchless GPU Math
            // We strip size 1 dimensions to save cycles. We also reverse the array 
            // order (innermost first) so the GPU doesn't need to know the rank.
            let mut effective_idx = 0;
            for i in (0..r).rev() {
                if raw_shape[i] > 1 {
                    dims.shape[effective_idx] = raw_shape[i];
                    dims.strides_a[effective_idx] = raw_strides_a[i];
                    dims.strides_c[effective_idx] = raw_strides_c[i];
                    effective_idx += 1;
                }
            }
        }
    }

    let eng = get_engine();
    let shader_name = format!("unary_{}_f32", op_name);
    let wgsl_code = generate_unary_shader(wgsl_op);
    
    let pipeline = eng.get_or_compile(&shader_name, "main", &wgsl_code);
    
    execute_unary(id_a, id_out, dims, &pipeline, eng)
}

fn execute_unary(
    id_a: u64,
    id_out: u64,
    dims: UnaryElemDims,
    pipeline: &wgpu::ComputePipeline,
    eng: &crate::GpuEngine,
) -> i32 {
    let is_contiguous = dims.meta[1] == 1;
    let length = dims.meta[0];

    // OPTIMIZATION: Instant-return safety check!
    if length == 0 {
        return crate::memory::HC4J_SUCCESS;
    }

    let (buf_a, buf_out) = {
        let guard = match crate::memory::get_registry().lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        
        let a = match guard.get(&id_a) {
            Some(b) => b.clone(),
            None => return crate::memory::HC4J_ERR_NOT_FOUND,
        };
        let out = match guard.get(&id_out) {
            Some(b) => b.clone(),
            None => return crate::memory::HC4J_ERR_NOT_FOUND,
        };
        
        (a, out)
    };

    let meta = eng
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UnaryElemDims Meta"),
            contents: bytemuck::cast_slice(&[dims]),
            usage: wgpu::BufferUsages::UNIFORM, 
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
                resource: buf_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: meta.as_entire_binding(),
            },
        ],
    });

    // Workgroup calculation based on vectorization density
    let workgroups = if is_contiguous {
        (length + 1023) / 1024 // 4 elements processed per thread
    } else {
        (length + 255) / 256   // 1 element processed per thread
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

// =====================================================================
// FFI EXPORTS (Macro Generated for Trigonometric Suite)
// =====================================================================

macro_rules! impl_unary_op {
    ($fn_name:ident, $op_str:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $fn_name(
            id_a: u64, id_out: u64, rank: u32,
            ptr_shape: *const u32, ptr_strides_a: *const u32, ptr_strides_c: *const u32,
            length: usize, is_contiguous: u32,
        ) -> i32 {
            dispatch_unary_op(
                $op_str, $op_str, id_a, id_out, rank, 
                ptr_shape, ptr_strides_a, ptr_strides_c, length, is_contiguous
            )
        }
    };
}

// Standard Trigonometric
impl_unary_op!(dispatch_sin_f32, "sin");
impl_unary_op!(dispatch_cos_f32, "cos");
impl_unary_op!(dispatch_tan_f32, "tan");

// Inverse Trigonometric
impl_unary_op!(dispatch_asin_f32, "asin");
impl_unary_op!(dispatch_acos_f32, "acos");
impl_unary_op!(dispatch_atan_f32, "atan");

// Hyperbolic
impl_unary_op!(dispatch_sinh_f32, "sinh");
impl_unary_op!(dispatch_cosh_f32, "cosh");
impl_unary_op!(dispatch_tanh_f32, "tanh");

// Inverse Hyperbolic
impl_unary_op!(dispatch_asinh_f32, "asinh");
impl_unary_op!(dispatch_acosh_f32, "acosh");
impl_unary_op!(dispatch_atanh_f32, "atanh");