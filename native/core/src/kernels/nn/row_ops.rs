//! Row-wise neural network operations (softmax, layer norm, RMS norm, cross entropy).

use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::memory::{TensorId, manager};
use crate::kernels::elementwise::WORKGROUP_SIZE;
use crate::kernels::reduce::{row_bind_group, row_grid, uniform_bytes};
use crate::engine::{ErrorTrap, GpuEngine};

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
            "struct Params {{\n    params: vec4<u32>,\n}}\n\n{decls}\n"
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

pub(crate) fn check_elements(id: TensorId, expected: u64, what: &'static str) -> Hc4jResult<()> {
    if manager()?.size_of(id)? != expected * 4 {
        return Err(Hc4jError::InvalidParam(what));
    }
    Ok(())
}

pub(crate) fn row_dims(rows: u32, cols: u32) -> Hc4jResult<u64> {
    if rows == 0 || cols == 0 {
        return Err(Hc4jError::InvalidParam("row op needs non-zero rows and cols"));
    }
    u64::from(rows)
        .checked_mul(u64::from(cols))
        .ok_or(Hc4jError::InvalidParam("rows * cols overflows"))
}

pub(crate) fn within_binding(engine: &GpuEngine, elems: u64) -> Hc4jResult<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
