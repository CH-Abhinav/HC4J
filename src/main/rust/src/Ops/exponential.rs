use crate::get_engine;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExponentialElemDims {
    pub meta: [u32; 4],
    pub shape: [u32; 8],
    pub strides_a: [u32; 8],
    pub strides_c: [u32; 8],
}

pub const EXPONENTIAL_F32_SHADER_TEMPLATE: &str = r#"
struct Dims {
    meta: vec4<u32>,
    shape: array<vec4<u32>, 2>,
    strides_a: array<vec4<u32>, 2>,
    strides_c: array<vec4<u32>, 2>,
}

@group(0) @binding(0)
var<storage, read> arrayA: array<f32>;
@group(0) @binding(1)
var<storage, read_write> arrayC: array<f32>;
@group(0) @binding(2)
var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let length = dims.meta.x;
    // ------------------------------------------------------------
    // CONTIGUOUS FAST PATH
    // 4 elements per GPU invocation
    // ------------------------------------------------------------
    if (dims.meta.y == 1u) {
        let base = gid.x << 2u;

        if (base >= length) {
            return;
        }

        if (base + 3u < length) {
            let values = vec4<f32>(
                arrayA[base],
                arrayA[base + 1u],
                arrayA[base + 2u],
                arrayA[base + 3u]
            );

            let result = __WGSL_OP__(values);

            arrayC[base] = result.x;
            arrayC[base + 1u] = result.y;
            arrayC[base + 2u] = result.z;
            arrayC[base + 3u] = result.w;
        } else {
            arrayC[base] = __WGSL_OP__(arrayA[base]);

            if (base + 1u < length) {
                arrayC[base + 1u] =
                    __WGSL_OP__(arrayA[base + 1u]);

                if (base + 2u < length) {
                    arrayC[base + 2u] =
                        __WGSL_OP__(arrayA[base + 2u]);
                }
            }
        }

        return;
    }

    // ------------------------------------------------------------
    // STRIDED / NON-CONTIGUOUS PATH
    // ------------------------------------------------------------
    let i = gid.x;

    if (i >= length) {
        return;
    }

    var remaining = i;
    var offset_a = 0u;
    var offset_c = 0u;

    for (var d = 0u; d < 8u; d = d + 1u) {
        let dim_size = dims.shape[d >> 2u][d & 3u];

        if (dim_size > 1u) {
            let coord = remaining % dim_size;
            remaining = remaining / dim_size;

            offset_a =
                offset_a +
                coord * dims.strides_a[d >> 2u][d & 3u];

            offset_c =
                offset_c +
                coord * dims.strides_c[d >> 2u][d & 3u];
        }
    }

    arrayC[offset_c] =
        __WGSL_OP__(arrayA[offset_a]);
}
"#;

// ============================================================================
// SHADER GENERATION
// ============================================================================

pub fn generate_exponential_shader(wgsl_op: &str) -> String {
    EXPONENTIAL_F32_SHADER_TEMPLATE.replace("__WGSL_OP__", wgsl_op)
}

// ============================================================================
// GENERIC EXPONENTIAL UNARY DISPATCH
// ============================================================================

fn dispatch_exponential_op(
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
    if length == 0 {
        return crate::memory::HC4J_SUCCESS;
    }

    if length > u32::MAX as usize {
        return crate::memory::HC4J_ERR_INVALID_PARAM;
    }

    let mut dims = ExponentialElemDims {
        meta: [length as u32, is_contiguous, 0, 0],
        shape: [1; 8],
        strides_a: [0; 8],
        strides_c: [0; 8],
    };

    if is_contiguous == 0 && rank > 0 {
        if ptr_shape.is_null()
            || ptr_strides_a.is_null()
            || ptr_strides_c.is_null()
        {
            return crate::memory::HC4J_ERR_INVALID_PARAM;
        }

        let r = std::cmp::min(rank as usize, 8);

        unsafe {
            let raw_shape =
                std::slice::from_raw_parts(ptr_shape, r);

            let raw_strides_a =
                std::slice::from_raw_parts(ptr_strides_a, r);

            let raw_strides_c =
                std::slice::from_raw_parts(ptr_strides_c, r);

            let mut effective_idx = 0usize;

            for i in (0..r).rev() {
                if raw_shape[i] > 1 {
                    dims.shape[effective_idx] = raw_shape[i];
                    dims.strides_a[effective_idx] =
                        raw_strides_a[i];
                    dims.strides_c[effective_idx] =
                        raw_strides_c[i];

                    effective_idx += 1;
                }
            }
        }
    }

    let engine = get_engine();

    let shader_name =
        format!("exponential_{}_f32", op_name);

    let wgsl_code =
        generate_exponential_shader(wgsl_op);

    let pipeline =
        engine.get_or_compile(
            &shader_name,
            "main",
            &wgsl_code,
        );

    execute_exponential(
        id_a,
        id_out,
        dims,
        &pipeline,
        engine,
    )
}

// ============================================================================
// GPU EXECUTION
// ============================================================================

fn execute_exponential(
    id_a: u64,
    id_out: u64,
    dims: ExponentialElemDims,
    pipeline: &wgpu::ComputePipeline,
    engine: &crate::GpuEngine,
) -> i32 {
    let length = dims.meta[0];

    if length == 0 {
        return crate::memory::HC4J_SUCCESS;
    }

    let is_contiguous = dims.meta[1] == 1;

    let (buffer_a, buffer_out) = {
        let guard =
            match crate::memory::get_registry().lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };

        let a = match guard.get(&id_a) {
            Some(buffer) => buffer.clone(),
            None =>
                return crate::memory::HC4J_ERR_NOT_FOUND,
        };

        let out = match guard.get(&id_out) {
            Some(buffer) => buffer.clone(),
            None =>
                return crate::memory::HC4J_ERR_NOT_FOUND,
        };

        (a, out)
    };

    let metadata_buffer =
        engine.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some(
                    "HC4J Exponential Dims"
                ),
                contents:
                    bytemuck::cast_slice(&[dims]),
                usage:
                    wgpu::BufferUsages::UNIFORM,
            },
        );

    let bind_group =
        engine.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some(
                    "HC4J Exponential Bind Group"
                ),
                layout:
                    &pipeline.get_bind_group_layout(0),

                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource:
                            buffer_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource:
                            buffer_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource:
                            metadata_buffer
                                .as_entire_binding(),
                    },
                ],
            },
        );

    let workgroups = if is_contiguous {
        (length as usize + 1023) / 1024
    } else {
        (length as usize + 255) / 256
    };

    let mut encoder =
        engine.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label:
                    Some("HC4J Exponential Encoder"),
            },
        );

    {
        let mut compute_pass =
            encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label:
                        Some("HC4J Exponential Compute"),
                    timestamp_writes: None,
                },
            );

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(
            0,
            &bind_group,
            &[],
        );

        compute_pass.dispatch_workgroups(
            workgroups as u32,
            1,
            1,
        );
    }

    engine.queue.submit(
        Some(encoder.finish())
    );

    crate::memory::HC4J_SUCCESS
}

// ============================================================================
// FFI MACRO
// ============================================================================

macro_rules! impl_exponential_op {
    ($fn_name:ident, $op_name:expr, $wgsl_op:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $fn_name(
            id_a: u64,
            id_out: u64,
            rank: u32,
            ptr_shape: *const u32,
            ptr_strides_a: *const u32,
            ptr_strides_c: *const u32,
            length: usize,
            is_contiguous: u32,
        ) -> i32 {
            dispatch_exponential_op(
                $op_name,
                $wgsl_op,
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
    };
}

// ============================================================================
// EXPONENTIAL
// ============================================================================

impl_exponential_op!(
    dispatch_exp_f32,
    "exp",
    "exp"
);

// ============================================================================
// NATURAL LOGARITHM
// ln(x) = log(x)
// ============================================================================

impl_exponential_op!(
    dispatch_ln_f32,
    "ln",
    "log"
);

// ============================================================================
// BASE-2 LOGARITHM
// ============================================================================

impl_exponential_op!(
    dispatch_log2_f32,
    "log2",
    "log2"
);

// ============================================================================
// BASE-10 LOGARITHM
// log10(x) = ln(x) / ln(10)
// ============================================================================

impl_exponential_op!(
    dispatch_log10_f32,
    "log10",
    "log(x) / log(10.0)"
);

// ============================================================================
// SQUARE ROOT
// ============================================================================

impl_exponential_op!(
    dispatch_sqrt_f32,
    "sqrt",
    "sqrt"
);