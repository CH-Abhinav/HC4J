//! Neural-network ops: activations, softmax, normalisation, losses and bias
//! broadcast.
//!
//! **Activations** are ordinary elementwise ops and reuse the unary executor,
//! so they inherit tiling, the 2-D grid fold, in-place output and the
//! host-streaming fallback for tensors larger than VRAM. They are not WGSL
//! built-ins, so each one supplies a prelude defining a `vec4` and a scalar
//! function — WGSL has neither overloading nor generics. One body is written
//! per activation with `__T__` standing for the width, and both functions are
//! generated from it, so the two paths cannot drift apart.
//!
//! **Row-wise ops** (softmax, layer norm, RMS norm, cross entropy) run one
//! workgroup per row: a reduction over the row in workgroup memory, then a
//! write pass over the same row. Each is a single dispatch.
//!
//! Numerics follow the stable formulations:
//!
//! * softmax uses the *online* max-and-sum recurrence (Milakov & Gimelshein),
//!   so a row is read twice rather than three times, and no `exp` ever sees a
//!   positive argument.
//! * layer norm accumulates Welford triples `(n, mean, M2)` and combines them
//!   pairwise in the tree, so the variance never comes from `E[x²] - E[x]²`.
//! * sigmoid, SiLU and softplus are written so that every `exp` argument is
//!   non-positive, which is what makes them finite for large `|x|`.
//!
//! Row-wise operands must be VRAM-resident and within one storage binding.

use crate::error::{Hc4jError, Hc4jResult, ffi_guard};
use crate::memory::{TensorId, manager};
use crate::ops::elementwise::{
    UnaryKernel, WORKGROUP_SIZE, dispatch_unary_kernel, grid_for_threads,
};
use crate::ops::reduce::{row_bind_group, row_grid, uniform_bytes};
use crate::{ErrorTrap, GpuEngine};

// ============================================================================
// Activations
// ============================================================================

/// Elementwise activations. `alpha` is only read by the parameterised ones.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    LeakyRelu,
    Elu,
    /// The tanh approximation, as used by GPT-2/BERT — not the erf form,
    /// which WGSL has no built-in for.
    Gelu,
    Silu,
    Sigmoid,
    Softplus,
}

impl Activation {
    pub fn name(self) -> &'static str {
        match self {
            Activation::Relu => "relu",
            Activation::LeakyRelu => "leaky_relu",
            Activation::Elu => "elu",
            Activation::Gelu => "gelu",
            Activation::Silu => "silu",
            Activation::Sigmoid => "sigmoid",
            Activation::Softplus => "softplus",
        }
    }

    /// Codes mirrored by `hc4j.ops.NnOps.Activation`.
    pub fn from_code(code: u32) -> Hc4jResult<Self> {
        match code {
            0 => Ok(Activation::Relu),
            1 => Ok(Activation::LeakyRelu),
            2 => Ok(Activation::Elu),
            3 => Ok(Activation::Gelu),
            4 => Ok(Activation::Silu),
            5 => Ok(Activation::Sigmoid),
            6 => Ok(Activation::Softplus),
            _ => Err(Hc4jError::InvalidParam("unknown activation")),
        }
    }

    /// Whether the op reads `dims.params.z` as an f32 parameter.
    pub fn takes_alpha(self) -> bool {
        matches!(self, Activation::LeakyRelu | Activation::Elu)
    }

    /// The function body, with `__T__` for the operand type. `__T__(c)` is a
    /// splat for the vec4 width and a no-op conversion for the scalar one, so
    /// one body serves both.
    fn body(self) -> &'static str {
        match self {
            Activation::Relu => "return max(x, __T__(0.0));",
            // select, not max: a slope above 1 would make max() wrong.
            Activation::LeakyRelu => {
                "let a = bitcast<f32>(dims.params.z);\n    \
                 return select(__T__(a) * x, x, x > __T__(0.0));"
            }
            // min(x, 0) keeps the unselected arm from overflowing to inf.
            Activation::Elu => {
                "let a = bitcast<f32>(dims.params.z);\n    \
                 let neg = __T__(a) * (exp(min(x, __T__(0.0))) - __T__(1.0));\n    \
                 return select(neg, x, x > __T__(0.0));"
            }
            // tanh saturates by |a| = 10 (within 1 ulp of f32 1.0), and the
            // cubic reaches a ~= 24000 at x = 88. Unclamped, a driver that
            // evaluates tanh as (e^2a - 1)/(e^2a + 1) returns inf/inf = NaN,
            // so gelu(88) came out NaN instead of 88.
            Activation::Gelu => {
                "let cubic = __T__(0.7978845608028654) * (x + __T__(0.044715) * x * x * x);\n    \
                 let inner = clamp(cubic, __T__(-10.0), __T__(10.0));\n    \
                 return __T__(0.5) * x * (__T__(1.0) + tanh(inner));"
            }
            // e = exp(-|x|) is in (0, 1], so neither arm can overflow.
            Activation::Silu => {
                "let e = exp(-abs(x));\n    \
                 let num = select(e, __T__(1.0), x >= __T__(0.0));\n    \
                 return x * num / (__T__(1.0) + e);"
            }
            Activation::Sigmoid => {
                "let e = exp(-abs(x));\n    \
                 let num = select(e, __T__(1.0), x >= __T__(0.0));\n    \
                 return num / (__T__(1.0) + e);"
            }
            Activation::Softplus => {
                "return max(x, __T__(0.0)) + log(__T__(1.0) + exp(-abs(x)));"
            }
        }
    }

    /// WGSL prelude: the same body at both widths.
    pub fn prelude(self) -> String {
        format!("{}\n{}", self.vec4_fn("hc4j_act4"), self.scalar_fn("hc4j_act1"))
    }

    /// The vec4 form under a chosen name, for shaders that need several
    /// activations at once — the fused kernels work only in `vec4<f32>`.
    pub fn vec4_fn(self, name: &str) -> String {
        format!(
            "fn {name}(x: vec4<f32>) -> vec4<f32> {{\n    {}\n}}\n",
            self.body().replace("__T__", "vec4<f32>")
        )
    }

    fn scalar_fn(self, name: &str) -> String {
        format!(
            "fn {name}(x: f32) -> f32 {{\n    {}\n}}\n",
            self.body().replace("__T__", "f32")
        )
    }

    pub fn kernel(self, prelude: &str, alpha: f32) -> UnaryKernel<'_> {
        UnaryKernel {
            family: "nn",
            op_name: self.name(),
            prelude,
            op4: "hc4j_act4",
            op1: "hc4j_act1",
            alpha_bits: if self.takes_alpha() { alpha.to_bits() } else { 0 },
        }
    }
}

// ============================================================================
// Row-wise kernels
// ============================================================================

/// `params = [rows, cols, eps_or_alpha_bits, flags]`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RowOp {
    Softmax,
    LogSoftmax,
    LayerNorm { affine: bool },
    RmsNorm { affine: bool },
    CrossEntropy,
}

impl RowOp {
    pub fn name(self) -> String {
        match self {
            RowOp::Softmax => "nn_softmax_rows".to_string(),
            RowOp::LogSoftmax => "nn_log_softmax_rows".to_string(),
            RowOp::LayerNorm { affine } => format!("nn_layernorm_rows_{}", if affine { "affine" } else { "plain" }),
            RowOp::RmsNorm { affine } => format!("nn_rmsnorm_rows_{}", if affine { "affine" } else { "plain" }),
            RowOp::CrossEntropy => "nn_cross_entropy_rows".to_string(),
        }
    }

    /// Storage bindings, in order, excluding the trailing uniform.
    pub fn bindings(self) -> &'static [&'static str] {
        match self {
            RowOp::Softmax | RowOp::LogSoftmax => &["A", "OUT"],
            RowOp::LayerNorm { affine: false } | RowOp::RmsNorm { affine: false } => &["A", "OUT"],
            RowOp::LayerNorm { affine: true } => &["A", "GAMMA", "BETA", "OUT"],
            RowOp::RmsNorm { affine: true } => &["A", "GAMMA", "OUT"],
            RowOp::CrossEntropy => &["A", "TARGETS", "OUT"],
        }
    }

    pub fn wgsl(self) -> String {
        let wg = WORKGROUP_SIZE;
        let half = WORKGROUP_SIZE / 2;
        let mut decls = String::new();
        for (i, name) in self.bindings().iter().enumerate() {
            // Every storage binding is read_write: slab regions put several
            // tensors in one buffer, and wgpu rejects mixed access to it.
            let ty = if *name == "TARGETS" { "i32" } else { "f32" };
            decls.push_str(&format!(
                "@group(0) @binding({i}) var<storage, read_write> {name}: array<{ty}>;\n"
            ));
        }
        decls.push_str(&format!(
            "@group(0) @binding({}) var<uniform> p: Params;\n",
            self.bindings().len()
        ));

        let header = format!(
            "struct Params {{
    params: vec4<u32>,
}}

{decls}
"
        );

        match self {
            RowOp::Softmax | RowOp::LogSoftmax => {
                let write = if self == RowOp::Softmax {
                    "exp(A[i] - rmax) * inv_sum"
                } else {
                    "A[i] - rmax - log_sum"
                };
                let tail = if self == RowOp::Softmax {
                    "let inv_sum = 1.0 / ss[0];"
                } else {
                    "let log_sum = log(ss[0]);"
                };
                format!(
                    "{header}\
var<workgroup> sm: array<f32, {wg}u>;
var<workgroup> ss: array<f32, {wg}u>;

@compute @workgroup_size({wg})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let row = wid.x + wid.y * nwg.x;
    // Workgroup-uniform, so the whole group exits together and the barriers
    // below stay balanced.
    if (row >= p.params.x) {{ return; }}
    let cols = p.params.y;
    let base = row * cols;

    // Online (max, sum of exp) over this thread's slice: one pass, and every
    // exp argument is <= 0.
    var m = -3.4028235e38;
    var s = 0.0;
    var c = lid.x;
    while (c < cols) {{
        let v = A[base + c];
        let nm = max(m, v);
        s = s * exp(m - nm) + exp(v - nm);
        m = nm;
        c = c + {wg}u;
    }}
    sm[lid.x] = m;
    ss[lid.x] = s;
    workgroupBarrier();

    for (var st = {half}u; st > 0u; st = st >> 1u) {{
        if (lid.x < st) {{
            let m1 = sm[lid.x];
            let s1 = ss[lid.x];
            let m2 = sm[lid.x + st];
            let s2 = ss[lid.x + st];
            let nm = max(m1, m2);
            sm[lid.x] = nm;
            ss[lid.x] = s1 * exp(m1 - nm) + s2 * exp(m2 - nm);
        }}
        workgroupBarrier();
    }}

    let rmax = sm[0];
    {tail}
    var w = lid.x;
    while (w < cols) {{
        let i = base + w;
        OUT[i] = {write};
        w = w + {wg}u;
    }}
}}
"
                )
            }
            RowOp::LayerNorm { affine } => {
                let scale = if affine {
                    "OUT[i] = (A[i] - row_mean) * inv_std * GAMMA[w] + BETA[w];"
                } else {
                    "OUT[i] = (A[i] - row_mean) * inv_std;"
                };
                format!(
                    "{header}\
var<workgroup> wn: array<f32, {wg}u>;
var<workgroup> wmean: array<f32, {wg}u>;
var<workgroup> wm2: array<f32, {wg}u>;

@compute @workgroup_size({wg})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let row = wid.x + wid.y * nwg.x;
    if (row >= p.params.x) {{ return; }}
    let cols = p.params.y;
    let base = row * cols;
    let eps = bitcast<f32>(p.params.z);

    // Welford over this thread's slice.
    var n = 0.0;
    var mean = 0.0;
    var m2 = 0.0;
    var c = lid.x;
    while (c < cols) {{
        let v = A[base + c];
        n = n + 1.0;
        let d = v - mean;
        mean = mean + d / n;
        m2 = m2 + d * (v - mean);
        c = c + {wg}u;
    }}
    wn[lid.x] = n;
    wmean[lid.x] = mean;
    wm2[lid.x] = m2;
    workgroupBarrier();

    // Pairwise Welford combine. A thread that saw nothing contributes
    // (0, 0, 0), which the formula absorbs; only an all-empty pair needs the
    // guard, and cols >= 1 makes that impossible for the root.
    for (var st = {half}u; st > 0u; st = st >> 1u) {{
        if (lid.x < st) {{
            let n1 = wn[lid.x];
            let n2 = wn[lid.x + st];
            let nt = n1 + n2;
            let inv = select(0.0, 1.0 / nt, nt > 0.0);
            let d = wmean[lid.x + st] - wmean[lid.x];
            wn[lid.x] = nt;
            wmean[lid.x] = wmean[lid.x] + d * n2 * inv;
            wm2[lid.x] = wm2[lid.x] + wm2[lid.x + st] + d * d * n1 * n2 * inv;
        }}
        workgroupBarrier();
    }}

    // `mean` is the per-thread Welford accumulator above: WGSL has no
    // shadowing, so the row result needs its own name.
    let row_mean = wmean[0];
    // Biased variance, as PyTorch's LayerNorm uses.
    let variance = wm2[0] / max(wn[0], 1.0);
    let inv_std = 1.0 / sqrt(variance + eps);
    var w = lid.x;
    while (w < cols) {{
        let i = base + w;
        {scale}
        w = w + {wg}u;
    }}
}}
"
                )
            }
            RowOp::RmsNorm { affine } => {
                let scale = if affine {
                    "OUT[i] = A[i] * inv_rms * GAMMA[w];"
                } else {
                    "OUT[i] = A[i] * inv_rms;"
                };
                format!(
                    "{header}\
var<workgroup> ssq: array<f32, {wg}u>;

@compute @workgroup_size({wg})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let row = wid.x + wid.y * nwg.x;
    if (row >= p.params.x) {{ return; }}
    let cols = p.params.y;
    let base = row * cols;
    let eps = bitcast<f32>(p.params.z);

    var acc = 0.0;
    var c = lid.x;
    while (c < cols) {{
        let v = A[base + c];
        acc = acc + v * v;
        c = c + {wg}u;
    }}
    ssq[lid.x] = acc;
    workgroupBarrier();

    for (var st = {half}u; st > 0u; st = st >> 1u) {{
        if (lid.x < st) {{
            ssq[lid.x] = ssq[lid.x] + ssq[lid.x + st];
        }}
        workgroupBarrier();
    }}

    let mean_sq = ssq[0] / f32(cols);
    let inv_rms = 1.0 / sqrt(mean_sq + eps);
    var w = lid.x;
    while (w < cols) {{
        let i = base + w;
        {scale}
        w = w + {wg}u;
    }}
}}
"
                )
            }
            RowOp::CrossEntropy => format!(
                "{header}\
var<workgroup> sm: array<f32, {wg}u>;
var<workgroup> ss: array<f32, {wg}u>;

@compute @workgroup_size({wg})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let row = wid.x + wid.y * nwg.x;
    if (row >= p.params.x) {{ return; }}
    let cols = p.params.y;
    let base = row * cols;

    var m = -3.4028235e38;
    var s = 0.0;
    var c = lid.x;
    while (c < cols) {{
        let v = A[base + c];
        let nm = max(m, v);
        s = s * exp(m - nm) + exp(v - nm);
        m = nm;
        c = c + {wg}u;
    }}
    sm[lid.x] = m;
    ss[lid.x] = s;
    workgroupBarrier();

    for (var st = {half}u; st > 0u; st = st >> 1u) {{
        if (lid.x < st) {{
            let m1 = sm[lid.x];
            let s1 = ss[lid.x];
            let m2 = sm[lid.x + st];
            let s2 = ss[lid.x + st];
            let nm = max(m1, m2);
            sm[lid.x] = nm;
            ss[lid.x] = s1 * exp(m1 - nm) + s2 * exp(m2 - nm);
        }}
        workgroupBarrier();
    }}

    if (lid.x == 0u) {{
        let cls = TARGETS[row]; // `target` is a WGSL reserved word
        // An out-of-range class would read another row (or past the buffer):
        // write a NaN instead, so the mistake is visible in the output.
        if (cls < 0 || u32(cls) >= cols) {{
            OUT[row] = bitcast<f32>(0x7fc00000u);
        }} else {{
            let log_sum_exp = sm[0] + log(ss[0]);
            OUT[row] = log_sum_exp - A[base + u32(cls)];
        }}
    }}
}}
"
            ),
        }
    }
}

/// Bias broadcast over rows: `out[r, c] = a[r, c] + bias[c]`. Fusion cannot
/// express this (it requires identical shapes), and a GEMM followed by a bias
/// add is the shape of every linear layer.
pub fn row_bias_wgsl() -> String {
    let wg = WORKGROUP_SIZE;
    format!(
        "struct Params {{
    params: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> BIAS: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size({wg})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    // The host folds large grids into X*Y; rebuild the linear index.
    let i = gid.x + gid.y * (nwg.x * {wg}u);
    let total = p.params.x;
    if (i >= total) {{ return; }}
    let cols = p.params.y;
    OUT[i] = A[i] + BIAS[i % cols];
}}
"
    )
}

// ============================================================================
// Execution
// ============================================================================

fn check_elements(id: TensorId, expected: u64, what: &'static str) -> Hc4jResult<()> {
    if manager()?.size_of(id)? != expected * 4 {
        return Err(Hc4jError::InvalidParam(what));
    }
    Ok(())
}

fn row_dims(rows: u32, cols: u32) -> Hc4jResult<u64> {
    if rows == 0 || cols == 0 {
        return Err(Hc4jError::InvalidParam("row op needs non-zero rows and cols"));
    }
    u64::from(rows)
        .checked_mul(u64::from(cols))
        .ok_or(Hc4jError::InvalidParam("rows * cols overflows"))
}

fn within_binding(engine: &GpuEngine, elems: u64) -> Hc4jResult<()> {
    if elems * 4 > engine.limits.max_storage_buffer_binding_size {
        return Err(Hc4jError::Unsupported("row op operand exceeds the storage-binding limit"));
    }
    Ok(())
}

/// Runs a row-wise op. `extra` holds the optional affine/target handles in
/// binding order; `param_z` is the raw f32 bits of epsilon where the kernel
/// reads one.
pub fn run_row_op(
    op: RowOp,
    id_a: TensorId,
    extra: &[TensorId],
    id_out: TensorId,
    rows: u32,
    cols: u32,
    param_z: u32,
) -> Hc4jResult<()> {
    let elems = row_dims(rows, cols)?;
    let mgr = manager()?;
    let engine = mgr.engine();
    within_binding(engine, elems)?;
    check_elements(id_a, elems, "input size does not match (rows, cols)")?;

    // Softmax and the norms are elementwise in shape; the losses reduce a row
    // to one value.
    let out_elems = match op {
        RowOp::CrossEntropy => u64::from(rows),
        _ => elems,
    };
    check_elements(id_out, out_elems, "output size does not match the op's result shape")?;
    for &id in extra {
        let expected = match op {
            RowOp::CrossEntropy => u64::from(rows),
            _ => u64::from(cols),
        };
        check_elements(id, expected, "affine or target operand has the wrong length")?;
    }

    let mut operands = vec![id_a];
    operands.extend_from_slice(extra);
    operands.push(id_out);
    let resident = mgr.acquire_resident(&operands)?;

    let name = op.name();
    let pipeline = engine.get_or_compile(&name, "main", &op.wgsl())?;
    let grid = row_grid(engine, rows)?;
    let bytes = elems * 4 + out_elems * 4;
    let label: &'static str = match op {
        RowOp::Softmax | RowOp::LogSoftmax => "nn (softmax)",
        RowOp::LayerNorm { .. } => "nn (layer norm)",
        RowOp::RmsNorm { .. } => "nn (rms norm)",
        RowOp::CrossEntropy => "nn (cross entropy)",
    };

    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, label, |rec| {
        let uniform = rec.uniform(&uniform_bytes([rows, cols, param_z, 0]))?;
        let spans: Vec<&_> = resident.spans.iter().collect();
        let bg = row_bind_group(engine, &pipeline, &spans, uniform);
        rec.dispatch(&pipeline, &bg, grid, bytes);
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}

/// `out[r, c] = a[r, c] + bias[c]`. `out` may be `a`.
pub fn run_row_bias(id_a: TensorId, id_bias: TensorId, id_out: TensorId, rows: u32, cols: u32) -> Hc4jResult<()> {
    let elems = row_dims(rows, cols)?;
    let mgr = manager()?;
    let engine = mgr.engine();
    within_binding(engine, elems)?;
    check_elements(id_a, elems, "input size does not match (rows, cols)")?;
    check_elements(id_bias, u64::from(cols), "bias length must equal cols")?;
    check_elements(id_out, elems, "output size does not match (rows, cols)")?;

    let resident = mgr.acquire_resident(&[id_a, id_bias, id_out])?;
    let pipeline = engine.get_or_compile("nn_row_bias", "main", &row_bias_wgsl())?;
    let grid = grid_for_threads(engine, elems)?;

    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, "nn (row bias)", |rec| {
        let uniform = rec.uniform(&uniform_bytes([
            u32::try_from(elems).map_err(|_| Hc4jError::Unsupported("bias add exceeds 2^32 elements"))?,
            cols,
            0,
            0,
        ]))?;
        let spans: Vec<&_> = resident.spans.iter().collect();
        let bg = row_bind_group(engine, &pipeline, &spans, uniform);
        rec.dispatch(&pipeline, &bg, grid, elems * 8);
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}

// ============================================================================
// C ABI
// ============================================================================

/// Elementwise activation, on the shared caller-allocated elementwise ABI
/// plus `act` (see `Activation::from_code`) and `alpha_bits`, the raw f32 bits
/// of the slope or alpha parameter.
///
/// # Safety
/// For strided calls, `shape`, `strides_a` and `strides_c` must each point to
/// `rank` readable `u32` values.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn dispatch_activation_f32(
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
    act: u32,
    alpha_bits: u32,
) -> i32 {
    let activation = match Activation::from_code(act) {
        Ok(a) => a,
        Err(err) => return err.code(),
    };
    let prelude = activation.prelude();
    let kernel = activation.kernel(&prelude, f32::from_bits(alpha_bits));
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

/// Row-wise softmax (`log_variant == 0`) or log-softmax over `[rows, cols]`.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_softmax_rows_f32(id_a: u64, id_out: u64, rows: u32, cols: u32, log_variant: u32) -> i32 {
    ffi_guard("dispatch_softmax_rows_f32", || {
        let op = if log_variant == 0 { RowOp::Softmax } else { RowOp::LogSoftmax };
        run_row_op(op, id_a, &[], id_out, rows, cols, 0)
    })
}

/// Row-wise layer norm. `id_gamma`/`id_beta` of 0 skip the affine transform;
/// otherwise both are required and must have `cols` elements.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_layernorm_rows_f32(
    id_a: u64,
    id_gamma: u64,
    id_beta: u64,
    id_out: u64,
    rows: u32,
    cols: u32,
    eps_bits: u32,
) -> i32 {
    ffi_guard("dispatch_layernorm_rows_f32", || {
        let affine = id_gamma != 0 || id_beta != 0;
        if affine && (id_gamma == 0 || id_beta == 0) {
            return Err(Hc4jError::InvalidParam("layer norm needs both gamma and beta, or neither"));
        }
        let extra: Vec<TensorId> = if affine { vec![id_gamma, id_beta] } else { Vec::new() };
        run_row_op(RowOp::LayerNorm { affine }, id_a, &extra, id_out, rows, cols, eps_bits)
    })
}

/// Row-wise RMS norm. `id_gamma` of 0 skips the scale.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_rmsnorm_rows_f32(
    id_a: u64,
    id_gamma: u64,
    id_out: u64,
    rows: u32,
    cols: u32,
    eps_bits: u32,
) -> i32 {
    ffi_guard("dispatch_rmsnorm_rows_f32", || {
        let affine = id_gamma != 0;
        let extra: Vec<TensorId> = if affine { vec![id_gamma] } else { Vec::new() };
        run_row_op(RowOp::RmsNorm { affine }, id_a, &extra, id_out, rows, cols, eps_bits)
    })
}

/// Per-row softmax cross entropy from logits `[rows, cols]` and i32 class
/// indices `[rows]`, into `[rows]`. An out-of-range index yields NaN.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_cross_entropy_rows(id_logits: u64, id_targets: u64, id_out: u64, rows: u32, cols: u32) -> i32 {
    ffi_guard("dispatch_cross_entropy_rows", || {
        run_row_op(RowOp::CrossEntropy, id_logits, &[id_targets], id_out, rows, cols, 0)
    })
}

/// `out[r, c] = a[r, c] + bias[c]`, the bias half of a linear layer.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_add_row_bias_f32(id_a: u64, id_bias: u64, id_out: u64, rows: u32, cols: u32) -> i32 {
    ffi_guard("dispatch_add_row_bias_f32", || run_row_bias(id_a, id_bias, id_out, rows, cols))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_codes_round_trip() {
        let all = [
            Activation::Relu,
            Activation::LeakyRelu,
            Activation::Elu,
            Activation::Gelu,
            Activation::Silu,
            Activation::Sigmoid,
            Activation::Softplus,
        ];
        for (code, expected) in all.iter().enumerate() {
            assert_eq!(Activation::from_code(code as u32).expect("valid code"), *expected);
        }
        assert!(Activation::from_code(all.len() as u32).is_err());
    }

    #[test]
    fn both_widths_are_generated_from_one_body() {
        // The vec4 and scalar paths of the unary template must not drift, so
        // they come from the same body with __T__ substituted.
        for act in [Activation::Relu, Activation::Gelu, Activation::Silu] {
            let prelude = act.prelude();
            assert!(prelude.contains("fn hc4j_act4(x: vec4<f32>) -> vec4<f32>"));
            assert!(prelude.contains("fn hc4j_act1(x: f32) -> f32"));
            assert!(!prelude.contains("__T__"), "{}: unsubstituted type", act.name());
        }
    }

    #[test]
    fn only_parameterised_activations_read_alpha() {
        let prelude = Activation::LeakyRelu.prelude();
        assert!(prelude.contains("dims.params.z"));
        assert!(!Activation::Relu.prelude().contains("dims.params.z"));
        // An alpha passed to an op that ignores it must not reach the uniform,
        // where it would look like a parameter to a cached pipeline.
        assert_eq!(Activation::Relu.kernel("", 0.1).alpha_bits, 0);
        assert_eq!(Activation::LeakyRelu.kernel("", 0.1).alpha_bits, 0.1f32.to_bits());
    }

    #[test]
    fn every_exp_argument_is_non_positive() {
        // What keeps sigmoid, SiLU and softplus finite for large |x|.
        for act in [Activation::Sigmoid, Activation::Silu, Activation::Softplus] {
            let body = act.body();
            assert!(body.contains("exp(-abs(x))"), "{} must clamp its exp", act.name());
            assert!(!body.contains("exp(x)"), "{} has an unclamped exp", act.name());
        }
    }

    #[test]
    fn row_kernels_declare_the_bindings_they_use() {
        let ln = RowOp::LayerNorm { affine: true };
        assert_eq!(ln.bindings(), &["A", "GAMMA", "BETA", "OUT"]);
        let src = ln.wgsl();
        assert!(src.contains("binding(3) var<storage, read_write> OUT"));
        assert!(src.contains("binding(4) var<uniform>"));
        assert!(src.contains("GAMMA[w]") && src.contains("BETA[w]"));

        let plain = RowOp::LayerNorm { affine: false }.wgsl();
        assert!(!plain.contains("GAMMA"));
        assert!(plain.contains("binding(1) var<storage, read_write> OUT"));
    }

    #[test]
    fn targets_are_bound_as_i32() {
        let src = RowOp::CrossEntropy.wgsl();
        assert!(src.contains("TARGETS: array<i32>"));
        assert!(src.contains("0x7fc00000u"), "an out-of-range class must surface as NaN");
    }

    #[test]
    fn no_row_kernel_binds_storage_read_only() {
        for op in [
            RowOp::Softmax,
            RowOp::LogSoftmax,
            RowOp::LayerNorm { affine: true },
            RowOp::RmsNorm { affine: true },
            RowOp::CrossEntropy,
        ] {
            assert!(!op.wgsl().contains("storage, read>"), "{} binds read-only storage", op.name());
        }
        assert!(!row_bias_wgsl().contains("storage, read>"));
    }
}
