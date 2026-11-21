use crate::error::Hc4jResult;
use crate::get_engine;

/// Unary f32 template shared by the trigonometric and exponential suites.
pub const UNARY_F32_SHADER_TEMPLATE: &str = r#"
struct Dims {
    params: vec4<u32>, // `meta` is a reserved word in WGSL
    shape: array<vec4<u32>, 2>,
    strides_a: array<vec4<u32>, 2>,
    strides_c: array<vec4<u32>, 2>,
}

// read_write on both: slab regions and in-place ops share buffers.
@group(0) @binding(0) var<storage, read_write> arrayA: array<f32>;
@group(0) @binding(1) var<storage, read_write> arrayC: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims; // UNIFORM: Hits fast constant cache!

// Helper definitions for ops that are not WGSL built-ins. Empty for the
// built-in suites; activations declare a vec4 and a scalar function here,
// since WGSL has neither overloading nor generics. Functions may read
// `dims.params.z` for an op parameter (alpha).
__WGSL_PRELUDE__

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    // The host folds large grids into X*Y; rebuild the linear thread id.
    let tid = gid.x + gid.y * (nwg.x * 256u);
    let length = dims.params.x;

    if (dims.params.y == 1u) {
        // --- CONTIGUOUS FAST PATH ---
        let base = tid << 2u; // base = tid * 4

        // Instant Warp Exit for out-of-bounds threads
        if (base >= length) { return; }

        if (base + 3u < length) {
            // Highly optimized contiguous execution (4-wide vectorization)
            let val_a = vec4<f32>(arrayA[base], arrayA[base+1u], arrayA[base+2u], arrayA[base+3u]);
            let res = __WGSL_OP4__(val_a);
            arrayC[base] = res.x;
            arrayC[base+1u] = res.y;
            arrayC[base+2u] = res.z;
            arrayC[base+3u] = res.w;
        } else {
            // Manually unrolled tail execution (avoids 'for' loop overhead on the GPU)
            arrayC[base] = __WGSL_OP1__(arrayA[base]);
            if (base + 1u < length) {
                arrayC[base+1u] = __WGSL_OP1__(arrayA[base+1u]);
                if (base + 2u < length) {
                    arrayC[base+2u] = __WGSL_OP1__(arrayA[base+2u]);
                }
            }
        }
    } else {
        // --- NON-CONTIGUOUS STRIDED PATH (ZERO-BRANCHING) ---
        let i = tid;

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

        arrayC[offset_c] = __WGSL_OP1__(arrayA[offset_a]);
    }
}
"#;

/// Specialises the template for an op whose callable works on both `vec4<f32>`
/// and `f32` — every WGSL built-in, and prefixed expressions like
/// `"0.4342944819032518 * log"`.
pub fn generate_unary_shader(wgsl_op: &str) -> String {
    generate_unary_shader_with("", wgsl_op, wgsl_op)
}

/// Specialises the template with a prelude of WGSL helper definitions and
/// separate vec4 / scalar callables. WGSL has no overloading, so an op that
/// is not a built-in needs one function per width.
pub fn generate_unary_shader_with(prelude: &str, wgsl_op4: &str, wgsl_op1: &str) -> String {
    UNARY_F32_SHADER_TEMPLATE
        .replace("__WGSL_PRELUDE__", prelude)
        .replace("__WGSL_OP4__", wgsl_op4)
        .replace("__WGSL_OP1__", wgsl_op1)
}

pub fn unary_pipeline(family: &str, op_name: &str, wgsl_op: &str) -> Hc4jResult<wgpu::ComputePipeline> {
    let shader_name = format!("{family}_{op_name}_f32");
    get_engine()?.get_or_compile(&shader_name, "main", &generate_unary_shader(wgsl_op))
}

/// How to specialise the unary template for one op.
#[derive(Copy, Clone, Debug)]
pub struct UnaryKernel<'a> {
    /// Groups ops in the pipeline-cache key: `{family}_{op_name}_f32`.
    pub family: &'a str,
    pub op_name: &'a str,
    /// WGSL helper definitions, or `""` for built-in ops.
    pub prelude: &'a str,
    /// Callable applied to a `vec4<f32>` (the contiguous fast path).
    pub op4: &'a str,
    /// Callable applied to an `f32` (the tail and strided paths).
    pub op1: &'a str,
    /// Raw bits of the f32 op parameter exposed as `dims.params.z`.
    pub alpha_bits: u32,
}

impl<'a> UnaryKernel<'a> {
    /// An op whose callable is width-generic (a WGSL built-in).
    pub fn builtin(family: &'a str, op_name: &'a str, wgsl_op: &'a str) -> Self {
        Self { family, op_name, prelude: "", op4: wgsl_op, op1: wgsl_op, alpha_bits: 0 }
    }

    pub fn pipeline(&self) -> Hc4jResult<wgpu::ComputePipeline> {
        let shader_name = format!("{}_{}_f32", self.family, self.op_name);
        get_engine()?.get_or_compile(
            &shader_name,
            "main",
            &generate_unary_shader_with(self.prelude, self.op4, self.op1),
        )
    }
}
