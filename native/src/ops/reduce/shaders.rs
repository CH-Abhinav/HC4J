use crate::ops::elementwise::WORKGROUP_SIZE;

use super::types::{Load, ReduceOp};

/// Pass over a flat range, reducing into one value per workgroup.
///
/// `params = [length, mean_count, 0, 0]`. The grid-stride loop means the
/// dispatch size is independent of the input length, and the barrier sits in
/// workgroup-uniform control flow in both loops.
pub fn flat_kernel_wgsl(op: ReduceOp, load: Load, finalize: bool) -> String {
    let bindings = match load.inputs() {
        2 => "@group(0) @binding(0) var<storage, read_write> A: array<f32>;\n\
              @group(0) @binding(1) var<storage, read_write> B: array<f32>;\n\
              @group(0) @binding(2) var<storage, read_write> OUT: array<f32>;\n\
              @group(0) @binding(3) var<uniform> p: Params;",
        _ => "@group(0) @binding(0) var<storage, read_write> A: array<f32>;\n\
              @group(0) @binding(1) var<storage, read_write> OUT: array<f32>;\n\
              @group(0) @binding(2) var<uniform> p: Params;",
    };
    let (ident, combine, load_expr) = (op.identity(), op.combine(), load.expr());
    let fin = if finalize { op.finalize() } else { "v" };
    let wg = WORKGROUP_SIZE;
    let half = WORKGROUP_SIZE / 2;
    format!(
        "struct Params {{
    params: vec4<u32>,
}}

{bindings}

var<workgroup> scratch: array<f32, {wg}u>;

fn combine(a: f32, b: f32) -> f32 {{
    return {combine};
}}

fn finalize(v: f32) -> f32 {{
    return {fin};
}}

@compute @workgroup_size({wg})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let group = wid.x + wid.y * nwg.x;
    let threads = nwg.x * nwg.y * {wg}u;
    let length = p.params.x;

    // Grid-stride fold: a bounded grid covers any length.
    var acc = {ident};
    var i = group * {wg}u + lid.x;
    while (i < length) {{
        acc = combine(acc, {load_expr});
        i = i + threads;
    }}
    scratch[lid.x] = acc;
    workgroupBarrier();

    // Tree reduction. The loop bound is workgroup-uniform and the barrier is
    // outside the guard, so every invocation reaches every barrier.
    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{
            scratch[lid.x] = combine(scratch[lid.x], scratch[lid.x + s]);
        }}
        workgroupBarrier();
    }}

    if (lid.x == 0u) {{
        OUT[p.params.z + group] = finalize(scratch[0]);
    }}
}}
"
    )
}

/// One workgroup per row of a `[rows, cols]` tensor.
///
/// `params = [rows, cols, mean_count, 0]`.
pub fn row_kernel_wgsl(op: ReduceOp, load: Load) -> String {
    let bindings = match load.inputs() {
        2 => "@group(0) @binding(0) var<storage, read_write> A: array<f32>;\n\
              @group(0) @binding(1) var<storage, read_write> B: array<f32>;\n\
              @group(0) @binding(2) var<storage, read_write> OUT: array<f32>;\n\
              @group(0) @binding(3) var<uniform> p: Params;",
        _ => "@group(0) @binding(0) var<storage, read_write> A: array<f32>;\n\
              @group(0) @binding(1) var<storage, read_write> OUT: array<f32>;\n\
              @group(0) @binding(2) var<uniform> p: Params;",
    };
    let (ident, combine, load_expr) = (op.identity(), op.combine(), load.expr());
    let fin = match op {
        ReduceOp::Mean => "select(v, v / f32(p.params.z), p.params.z > 0u)",
        _ => "v",
    };
    let wg = WORKGROUP_SIZE;
    let half = WORKGROUP_SIZE / 2;
    format!(
        "struct Params {{
    params: vec4<u32>,
}}

{bindings}

var<workgroup> scratch: array<f32, {wg}u>;

fn combine(a: f32, b: f32) -> f32 {{
    return {combine};
}}

fn finalize(v: f32) -> f32 {{
    return {fin};
}}

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

    var acc = {ident};
    var c = lid.x;
    while (c < cols) {{
        let i = base + c;
        acc = combine(acc, {load_expr});
        c = c + {wg}u;
    }}
    scratch[lid.x] = acc;
    workgroupBarrier();

    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{
            scratch[lid.x] = combine(scratch[lid.x], scratch[lid.x + s]);
        }}
        workgroupBarrier();
    }}

    if (lid.x == 0u) {{
        OUT[row] = finalize(scratch[0]);
    }}
}}
"
    )
}

/// General-axis reduction for inner > 1: one invocation per output element,
/// coalesced reads along inner dimension.
pub fn axis_kernel_wgsl(op: ReduceOp) -> String {
    let (ident, combine) = (op.identity(), op.combine());
    let is_mean = op == ReduceOp::Mean;
    format!(
        "struct Params {{
    params: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let idx = gid.x + gid.y * (nwg.x * 256u);
    let total_outputs = p.params.w;
    if (idx >= total_outputs) {{ return; }}

    let inner = p.params.z;
    let axis_size = p.params.y;
    let o = idx / inner;
    let i = idx % inner;
    let base = o * (axis_size * inner) + i;

    var acc = {ident};
    for (var k = 0u; k < axis_size; k = k + 1u) {{
        let b = A[base + k * inner];
        let a = acc;
        acc = {combine};
    }}
    if ({is_mean}) {{
        acc = acc / f32(axis_size);
    }}
    OUT[idx] = acc;
}}
"
    )
}

/// General-axis argmax / argmin for inner > 1: one invocation per output element,
/// ties resolve to lowest index.
pub fn axis_argmax_wgsl(largest: bool) -> String {
    let better = if largest { "v > best_val" } else { "v < best_val" };
    format!(
        "struct Params {{
    params: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT: array<i32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let idx = gid.x + gid.y * (nwg.x * 256u);
    let total_outputs = p.params.w;
    if (idx >= total_outputs) {{ return; }}

    let inner = p.params.z;
    let axis_size = p.params.y;
    let o = idx / inner;
    let i = idx % inner;
    let base = o * (axis_size * inner) + i;

    var best_val = A[base];
    var best_k = 0u;
    for (var k = 1u; k < axis_size; k = k + 1u) {{
        let v = A[base + k * inner];
        if ({better}) {{
            best_val = v;
            best_k = k;
        }}
    }}
    OUT[idx] = i32(best_k);
}}
"
    )
}

/// Row-wise argmax / argmin, writing i32 indices. Ties resolve to the lowest
/// index, matching NumPy and PyTorch.
pub fn row_argmax_wgsl(largest: bool) -> String {
    let better = if largest { "v > bv" } else { "v < bv" };
    let ident = if largest { "-3.4028235e38" } else { "3.4028235e38" };
    let wg = WORKGROUP_SIZE;
    let half = WORKGROUP_SIZE / 2;
    format!(
        "struct Params {{
    params: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT: array<i32>;
@group(0) @binding(2) var<uniform> p: Params;

var<workgroup> best_val: array<f32, {wg}u>;
var<workgroup> best_idx: array<u32, {wg}u>;

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

    var bv = {ident};
    var bi = 0u;
    var c = lid.x;
    while (c < cols) {{
        let v = A[base + c];
        if ({better}) {{
            bv = v;
            bi = c;
        }}
        c = c + {wg}u;
    }}
    best_val[lid.x] = bv;
    best_idx[lid.x] = bi;
    workgroupBarrier();

    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{
            let v = best_val[lid.x + s];
            let i = best_idx[lid.x + s];
            let cur = best_val[lid.x];
            if ({better_tree}) {{
                best_val[lid.x] = v;
                best_idx[lid.x] = i;
            }}
        }}
        workgroupBarrier();
    }}

    if (lid.x == 0u) {{
        OUT[row] = i32(best_idx[0]);
    }}
}}
",
        better_tree = if largest {
            "v > cur || (v == cur && i < best_idx[lid.x])"
        } else {
            "v < cur || (v == cur && i < best_idx[lid.x])"
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_input_loads_declare_two_bindings() {
        assert_eq!(Load::Diff2.inputs(), 2);
        let src = flat_kernel_wgsl(ReduceOp::Sum, Load::Diff2, false);
        assert!(src.contains("binding(1) var<storage, read_write> B"));
        assert!(src.contains("binding(3) var<uniform>"));
        let one = flat_kernel_wgsl(ReduceOp::Sum, Load::Value, false);
        assert!(!one.contains("> B:"));
        assert!(one.contains("binding(2) var<uniform>"));
    }

    #[test]
    fn every_storage_binding_is_read_write() {
        let sources = [
            flat_kernel_wgsl(ReduceOp::Sum, Load::Diff2, true),
            row_kernel_wgsl(ReduceOp::Mean, Load::Value),
            row_argmax_wgsl(true),
        ];
        for src in sources {
            assert!(!src.contains("storage, read>"), "found a read-only storage binding");
        }
    }

    #[test]
    fn flat_kernel_uses_partial_offset_param() {
        let src = flat_kernel_wgsl(ReduceOp::Sum, Load::Value, false);
        assert!(src.contains("OUT[p.params.z + group] = finalize(scratch[0]);"));
    }
}
