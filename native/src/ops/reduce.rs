//! Reductions: sum, mean, max, min — over a whole tensor or along its last
//! axis. The primitive softmax, normalisation and the losses are built on.
//!
//! **Full reduce** always costs exactly two dispatches, whatever the size.
//! Pass 1 runs a bounded number of workgroups (≤ [`MAX_PARTIALS`]) in a
//! grid-stride loop, each reducing its slice in workgroup memory and writing
//! one partial. Pass 2 is a single workgroup that reduces the partials. The
//! scratch buffer is therefore at most 4 KiB no matter how large the input,
//! and no recursion or log-many passes are needed.
//!
//! **Row reduce** (`[rows, cols] -> [rows]`) is one dispatch: one workgroup
//! per row, striding over the row and reducing in workgroup memory. Row
//! indices come from a folded grid, so the row count is not capped at 65535.
//!
//! Operands must be VRAM-resident. Unlike the elementwise ops there is no
//! host-streaming fallback, but a tensor larger than one storage binding is
//! handled: pass 1 tiles over sub-bindings and writes a partial per tile.
//!
//! **Load variants** let the reduction fold a cheap elementwise expression in
//! rather than materialising it: `sum(x*x)` for an L2 norm and
//! `sum((a-b)^2)` for MSE both run as one reduction with no intermediate
//! tensor.

use crate::error::{Hc4jError, Hc4jResult};
use crate::memory::{DeviceSpan, TensorId, manager};
use crate::ops::elementwise::{WORKGROUP_SIZE, grid_2d, max_binding_tile};
use crate::{ErrorTrap, GpuEngine};

/// Upper bound on pass-1 workgroups, and so on partials. One workgroup in
/// pass 2 strides over them, so this is also the largest number of values a
/// single workgroup must walk.
pub const MAX_PARTIALS: u32 = 1024;
/// Elements each pass-1 invocation folds before the tree reduction.
const ELEMS_PER_THREAD: u64 = 4;

/// Which fold a reduction applies.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    /// Sum, divided by the element count in the final pass.
    Mean,
    Max,
    Min,
}

impl ReduceOp {
    /// Identity element: the value a reduction of nothing returns.
    fn identity(self) -> &'static str {
        match self {
            // Finite rather than ±inf: the pairwise combines below subtract
            // two accumulators, and inf - inf is NaN.
            ReduceOp::Sum | ReduceOp::Mean => "0.0",
            ReduceOp::Max => "-3.4028235e38",
            ReduceOp::Min => "3.4028235e38",
        }
    }

    /// WGSL expression combining accumulators `a` and `b`.
    fn combine(self) -> &'static str {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => "a + b",
            ReduceOp::Max => "max(a, b)",
            ReduceOp::Min => "min(a, b)",
        }
    }

    /// Applied once, by the thread that writes the result.
    fn finalize(self) -> &'static str {
        match self {
            // params.y carries the count; 0 means "leave the sum alone",
            // which is what pass 1 and the non-mean ops pass.
            ReduceOp::Mean => "select(v, v / f32(p.params.y), p.params.y > 0u)",
            _ => "v",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Mean => "mean",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
        }
    }

    /// Code for the FFI, mirrored by `hc4j.ops.NnOps.Reduction`.
    pub fn from_code(code: u32) -> Hc4jResult<Self> {
        match code {
            0 => Ok(ReduceOp::Sum),
            1 => Ok(ReduceOp::Mean),
            2 => Ok(ReduceOp::Max),
            3 => Ok(ReduceOp::Min),
            _ => Err(Hc4jError::InvalidParam("unknown reduction op")),
        }
    }
}

/// What a reduction reads at each index.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Load {
    /// `A[i]`
    Value,
    /// `A[i] * A[i]` — sum of squares without an intermediate tensor.
    Square,
    /// `(A[i] - B[i])^2` — mean squared error in one pass.
    Diff2,
}

impl Load {
    fn inputs(self) -> usize {
        match self {
            Load::Value | Load::Square => 1,
            Load::Diff2 => 2,
        }
    }

    fn expr(self) -> &'static str {
        match self {
            Load::Value => "A[i]",
            Load::Square => "A[i] * A[i]",
            Load::Diff2 => "(A[i] - B[i]) * (A[i] - B[i])",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Load::Value => "val",
            Load::Square => "sq",
            Load::Diff2 => "diff2",
        }
    }
}

// ============================================================================
// WGSL
// ============================================================================

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
        OUT[group] = finalize(scratch[0]);
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
    // params.z carries the count, so Mean's finalize reads the right field.
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
    // `row` is workgroup-uniform, so the whole workgroup exits together and
    // the barriers below stay balanced.
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
            // Strict comparison keeps the lower index on a tie, except when
            // the challenger's index is lower, which the equality case covers.
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

// ============================================================================
// Execution
// ============================================================================

fn pipeline_for(engine: &GpuEngine, name: &str, source: &str) -> Hc4jResult<wgpu::ComputePipeline> {
    engine.get_or_compile(name, "main", source)
}

fn bind_group(
    engine: &GpuEngine,
    pipeline: &wgpu::ComputePipeline,
    resources: Vec<wgpu::BindingResource<'_>>,
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = resources
        .into_iter()
        .enumerate()
        .map(|(i, resource)| wgpu::BindGroupEntry { binding: i as u32, resource })
        .collect();
    engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("HC4J Reduce"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

fn params_bytes(p: [u32; 4]) -> Vec<u8> {
    bytemuck::bytes_of(&p).to_vec()
}

/// Validates that `id` holds `expected` f32 elements.
fn check_elements(id: TensorId, expected: u64) -> Hc4jResult<u64> {
    let size = manager()?.size_of(id)?;
    if size != expected * 4 {
        return Err(Hc4jError::InvalidParam("tensor size does not match the reduction shape"));
    }
    Ok(size)
}

/// `out[0] = reduce(load(a, b))` over `length` elements.
///
/// Two dispatches in one recorded unit: inside a batch this adds no
/// submission, and the partials buffer never outlives the op.
pub fn run_full(op: ReduceOp, load: Load, id_a: TensorId, id_b: TensorId, id_out: TensorId, length: u64) -> Hc4jResult<()> {
    if length == 0 {
        return Err(Hc4jError::InvalidParam("reduction over an empty tensor"));
    }
    let mgr = manager()?;
    let engine = mgr.engine();
    check_elements(id_a, length)?;
    if load.inputs() == 2 {
        check_elements(id_b, length)?;
    }
    check_elements(id_out, 1)?;

    let mut operands = vec![id_a];
    if load.inputs() == 2 {
        operands.push(id_b);
    }
    operands.push(id_out);
    let resident = mgr.acquire_resident(&operands)?;
    let (inputs, out) = resident.spans.split_at(load.inputs());

    // Tile over the storage-binding limit; each tile contributes its own
    // partials, which pass 2 then reduces together.
    let tile_bytes = max_binding_tile(engine);
    let total_bytes = length * 4;
    let tiles = total_bytes.div_ceil(tile_bytes);
    let per_tile_groups = MAX_PARTIALS as u64 / tiles.max(1);
    if per_tile_groups == 0 {
        return Err(Hc4jError::Unsupported("tensor needs more binding tiles than the reduction supports"));
    }

    let name_pass1 = format!("reduce_flat_{}_{}_p1", op.name(), load.name());
    let name_pass2 = format!("reduce_flat_{}_val_p2", op.name());
    let pass1 = pipeline_for(engine, &name_pass1, &flat_kernel_wgsl(op, load, false))?;
    let pass2 = pipeline_for(engine, &name_pass2, &flat_kernel_wgsl(op, Load::Value, true))?;

    let mut partial_counts: Vec<u32> = Vec::new();
    for tile in 0..tiles {
        let offset = tile * tile_bytes;
        let bytes = tile_bytes.min(total_bytes - offset);
        let elems = bytes / 4;
        let groups = u32::try_from(elems.div_ceil(WORKGROUP_SIZE as u64 * ELEMS_PER_THREAD))
            .unwrap_or(u32::MAX)
            .min(u32::try_from(per_tile_groups).unwrap_or(u32::MAX))
            .max(1);
        partial_counts.push(groups);
    }
    let partials_total: u32 = partial_counts.iter().sum();
    let partials = mgr.allocate_device(u64::from(partials_total) * 4, false)?;
    let partials_span = partials.span();

    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(tiles as u32 + 1, "reduce (full)", |rec| {
        let mut written = 0u64;
        for (tile, &groups) in partial_counts.iter().enumerate() {
            let offset = tile as u64 * tile_bytes;
            let bytes = tile_bytes.min(total_bytes - offset);
            let uniform = rec.uniform(&params_bytes([u32::try_from(bytes / 4).unwrap_or(u32::MAX), 0, 0, 0]))?;
            let mut resources: Vec<wgpu::BindingResource<'_>> =
                inputs.iter().map(|span| span.sub_binding(offset, bytes)).collect();
            resources.push(partials_span.sub_binding(written * 4, u64::from(groups) * 4));
            resources.push(uniform);
            let bg = bind_group(engine, &pass1, resources);
            rec.dispatch(&pass1, &bg, (groups, 1, 1), bytes * inputs.len() as u64);
            written += u64::from(groups);
        }

        // Pass 2: one workgroup strides over every partial.
        let count = if op == ReduceOp::Mean { u32::try_from(length).unwrap_or(u32::MAX) } else { 0 };
        let uniform = rec.uniform(&params_bytes([partials_total, count, 0, 0]))?;
        let bg = bind_group(
            engine,
            &pass2,
            vec![partials_span.binding(), out[0].binding(), uniform],
        );
        rec.dispatch(&pass2, &bg, (1, 1, 1), u64::from(partials_total) * 4);
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}

/// `out[r] = reduce(load(a[r, :], b[r, :]))` for each of `rows` rows.
pub fn run_rows(
    op: ReduceOp,
    load: Load,
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    rows: u32,
    cols: u32,
) -> Hc4jResult<()> {
    let (rows64, cols64) = (u64::from(rows), u64::from(cols));
    if rows == 0 || cols == 0 {
        return Err(Hc4jError::InvalidParam("row reduction needs non-zero rows and cols"));
    }
    let mgr = manager()?;
    let engine = mgr.engine();
    let elems = rows64
        .checked_mul(cols64)
        .ok_or(Hc4jError::InvalidParam("rows * cols overflows"))?;
    check_elements(id_a, elems)?;
    if load.inputs() == 2 {
        check_elements(id_b, elems)?;
    }
    check_elements(id_out, rows64)?;
    if elems * 4 > engine.limits.max_storage_buffer_binding_size {
        return Err(Hc4jError::Unsupported("row reduction operand exceeds the storage-binding limit"));
    }

    let mut operands = vec![id_a];
    if load.inputs() == 2 {
        operands.push(id_b);
    }
    operands.push(id_out);
    let resident = mgr.acquire_resident(&operands)?;
    let (inputs, out) = resident.spans.split_at(load.inputs());

    let name = format!("reduce_rows_{}_{}", op.name(), load.name());
    let pipeline = pipeline_for(engine, &name, &row_kernel_wgsl(op, load))?;
    let grid = row_grid(engine, rows)?;
    let count = if op == ReduceOp::Mean { cols } else { 0 };

    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, "reduce (rows)", |rec| {
        let uniform = rec.uniform(&params_bytes([rows, cols, count, 0]))?;
        let mut resources: Vec<wgpu::BindingResource<'_>> = inputs.iter().map(|span| span.binding()).collect();
        resources.push(out[0].binding());
        resources.push(uniform);
        let bg = bind_group(engine, &pipeline, resources);
        rec.dispatch(&pipeline, &bg, grid, elems * 4 * inputs.len() as u64 + rows64 * 4);
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}

/// `out[r] = argmax(a[r, :])` as i32 indices (or argmin when `largest` is false).
pub fn run_row_argmax(id_a: TensorId, id_out: TensorId, rows: u32, cols: u32, largest: bool) -> Hc4jResult<()> {
    if rows == 0 || cols == 0 {
        return Err(Hc4jError::InvalidParam("argmax needs non-zero rows and cols"));
    }
    let mgr = manager()?;
    let engine = mgr.engine();
    let elems = u64::from(rows)
        .checked_mul(u64::from(cols))
        .ok_or(Hc4jError::InvalidParam("rows * cols overflows"))?;
    check_elements(id_a, elems)?;
    check_elements(id_out, u64::from(rows))?;
    if elems * 4 > engine.limits.max_storage_buffer_binding_size {
        return Err(Hc4jError::Unsupported("argmax operand exceeds the storage-binding limit"));
    }

    let resident = mgr.acquire_resident(&[id_a, id_out])?;
    let (a, out) = (&resident.spans[0], &resident.spans[1]);
    let name = if largest { "reduce_rows_argmax" } else { "reduce_rows_argmin" };
    let pipeline = pipeline_for(engine, name, &row_argmax_wgsl(largest))?;
    let grid = row_grid(engine, rows)?;

    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, "reduce (argmax)", |rec| {
        let uniform = rec.uniform(&params_bytes([rows, cols, 0, 0]))?;
        let bg = bind_group(engine, &pipeline, vec![a.binding(), out.binding(), uniform]);
        rec.dispatch(&pipeline, &bg, grid, elems * 4 + u64::from(rows) * 4);
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}

/// One workgroup per row, folded so the row count is not capped at 65535.
pub fn row_grid(engine: &GpuEngine, rows: u32) -> Hc4jResult<(u32, u32, u32)> {
    let max = engine.limits.max_compute_workgroups_per_dimension;
    let (x, y) = grid_2d(rows, max);
    if y > max {
        return Err(Hc4jError::Unsupported("too many rows for a 2D grid"));
    }
    Ok((x, y, 1))
}

/// Shared by the row-wise kernels in [`crate::ops::nn`]: binds a span per
/// resource with the uniform last.
pub(crate) fn row_bind_group(
    engine: &GpuEngine,
    pipeline: &wgpu::ComputePipeline,
    spans: &[&DeviceSpan],
    uniform: wgpu::BindingResource<'_>,
) -> wgpu::BindGroup {
    let mut resources: Vec<wgpu::BindingResource<'_>> = spans.iter().map(|s| s.binding()).collect();
    resources.push(uniform);
    bind_group(engine, pipeline, resources)
}

pub(crate) fn uniform_bytes(p: [u32; 4]) -> Vec<u8> {
    params_bytes(p)
}

// ============================================================================
// C ABI
// ============================================================================

/// `out[0] = reduce(a)` over `length` elements. `op` is 0 sum, 1 mean,
/// 2 max, 3 min.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_reduce_f32(id_a: u64, id_out: u64, length: u64, op: u32) -> i32 {
    crate::error::ffi_guard("dispatch_reduce_f32", || {
        run_full(ReduceOp::from_code(op)?, Load::Value, id_a, 0, id_out, length)
    })
}

/// `out[0] = reduce(a * a)`, the fold an L2 norm or a variance needs.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_reduce_square_f32(id_a: u64, id_out: u64, length: u64, op: u32) -> i32 {
    crate::error::ffi_guard("dispatch_reduce_square_f32", || {
        run_full(ReduceOp::from_code(op)?, Load::Square, id_a, 0, id_out, length)
    })
}

/// `out[0] = reduce((a - b)^2)`. With `op = 1` (mean) this is MSE.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_reduce_diff2_f32(id_a: u64, id_b: u64, id_out: u64, length: u64, op: u32) -> i32 {
    crate::error::ffi_guard("dispatch_reduce_diff2_f32", || {
        run_full(ReduceOp::from_code(op)?, Load::Diff2, id_a, id_b, id_out, length)
    })
}

/// `out[r] = reduce(a[r, :])` for a `[rows, cols]` tensor.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_reduce_rows_f32(id_a: u64, id_out: u64, rows: u32, cols: u32, op: u32) -> i32 {
    crate::error::ffi_guard("dispatch_reduce_rows_f32", || {
        run_rows(ReduceOp::from_code(op)?, Load::Value, id_a, 0, id_out, rows, cols)
    })
}

/// `out[r] = argmax(a[r, :])` as i32, or argmin when `largest == 0`.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_argmax_rows(id_a: u64, id_out: u64, rows: u32, cols: u32, largest: u32) -> i32 {
    crate::error::ffi_guard("dispatch_argmax_rows", || {
        run_row_argmax(id_a, id_out, rows, cols, largest != 0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_are_finite_so_pairwise_combines_cannot_produce_nan() {
        // max/min seed with ±f32::MAX rather than ±inf: the softmax and
        // Welford combines subtract two accumulators.
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min] {
            let ident: f32 = op.identity().parse().expect("identity parses as f32");
            assert!(ident.is_finite(), "{} identity must be finite", op.name());
        }
    }

    #[test]
    fn op_codes_round_trip_and_reject_garbage() {
        for (code, expected) in [(0, ReduceOp::Sum), (1, ReduceOp::Mean), (2, ReduceOp::Max), (3, ReduceOp::Min)] {
            assert_eq!(ReduceOp::from_code(code).expect("valid code"), expected);
        }
        assert!(ReduceOp::from_code(4).is_err());
    }

    #[test]
    fn mean_divides_only_when_a_count_is_supplied() {
        // Pass 1 passes count 0 so partials stay sums; pass 2 passes the
        // element count. Both share one kernel, so the guard matters.
        assert!(ReduceOp::Mean.finalize().contains("p.params.y > 0u"));
        assert_eq!(ReduceOp::Sum.finalize(), "v");
    }

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
        // Slab regions put several tensors in one buffer, which wgpu rejects
        // if the same buffer is bound both read-only and read-write.
        let sources = [
            flat_kernel_wgsl(ReduceOp::Sum, Load::Diff2, true),
            row_kernel_wgsl(ReduceOp::Mean, Load::Value),
            row_argmax_wgsl(true),
        ];
        for src in sources {
            assert!(!src.contains("storage, read>"), "found a read-only storage binding");
        }
    }
}
