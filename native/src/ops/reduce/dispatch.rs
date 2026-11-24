use crate::error::{Hc4jError, Hc4jResult};
use crate::memory::{manager, DeviceSpan, TensorId};
use crate::ops::elementwise::{grid_2d, max_binding_tile, WORKGROUP_SIZE};
use crate::{ErrorTrap, GpuEngine};

use super::shaders::{flat_kernel_wgsl, row_argmax_wgsl, row_kernel_wgsl};
use super::types::{Load, ReduceOp, ELEMS_PER_THREAD, MAX_PARTIALS};

pub(crate) fn pipeline_for(engine: &GpuEngine, name: &str, source: &str) -> Hc4jResult<wgpu::ComputePipeline> {
    engine.get_or_compile(name, "main", source)
}

pub(crate) fn bind_group(
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

pub(crate) fn params_bytes(p: [u32; 4]) -> Vec<u8> {
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
            let uniform = rec.uniform(&params_bytes([
                u32::try_from(bytes / 4).unwrap_or(u32::MAX),
                0,
                u32::try_from(written).unwrap_or(u32::MAX),
                0,
            ]))?;
            let mut resources: Vec<wgpu::BindingResource<'_>> =
                inputs.iter().map(|span| span.sub_binding(offset, bytes)).collect();
            resources.push(partials_span.binding());
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

/// Reduces along an arbitrary axis where `inner > 1` (or delegates to `run_rows` if `inner == 1`).
pub fn run_axis_reduce_f32(
    op: ReduceOp,
    id_a: TensorId,
    id_out: TensorId,
    outer: u32,
    axis_size: u32,
    inner: u32,
) -> Hc4jResult<()> {
    if outer == 0 || axis_size == 0 || inner == 0 {
        return Err(Hc4jError::InvalidParam("outer, axis_size, and inner must be non-zero"));
    }
    if inner == 1 {
        return run_rows(op, Load::Value, id_a, 0, id_out, outer, axis_size);
    }

    let mgr = manager()?;
    let engine = mgr.engine();
    let total_elements = u64::from(outer) * u64::from(axis_size) * u64::from(inner);
    check_elements(id_a, total_elements)?;
    let total_outputs = u64::from(outer) * u64::from(inner);
    check_elements(id_out, total_outputs)?;

    let resident = mgr.acquire_resident(&[id_a, id_out])?;
    let (a, out) = (&resident.spans[0], &resident.spans[1]);

    let name = format!("reduce_axis_{}", op.name());
    let pipeline = pipeline_for(engine, &name, &super::shaders::axis_kernel_wgsl(op))?;
    let total_outputs_u32 = u32::try_from(total_outputs).map_err(|_| Hc4jError::Unsupported("total outputs exceeds u32"))?;
    let grid = crate::ops::elementwise::grid_for_threads(engine, total_outputs)?;

    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, "reduce (axis)", |rec| {
        let uniform = rec.uniform(&params_bytes([outer, axis_size, inner, total_outputs_u32]))?;
        let bg = bind_group(engine, &pipeline, vec![a.binding(), out.binding(), uniform]);
        rec.dispatch(&pipeline, &bg, grid, total_elements * 4 + total_outputs * 4);
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}

/// Computes argmax or argmin along an arbitrary axis where `inner > 1` (or delegates to `run_row_argmax` if `inner == 1`).
pub fn run_axis_argmax(
    id_a: TensorId,
    id_out: TensorId,
    outer: u32,
    axis_size: u32,
    inner: u32,
    largest: bool,
) -> Hc4jResult<()> {
    if outer == 0 || axis_size == 0 || inner == 0 {
        return Err(Hc4jError::InvalidParam("outer, axis_size, and inner must be non-zero"));
    }
    if inner == 1 {
        return run_row_argmax(id_a, id_out, outer, axis_size, largest);
    }

    let mgr = manager()?;
    let engine = mgr.engine();
    let total_elements = u64::from(outer) * u64::from(axis_size) * u64::from(inner);
    check_elements(id_a, total_elements)?;
    let total_outputs = u64::from(outer) * u64::from(inner);
    check_elements(id_out, total_outputs)?;

    let resident = mgr.acquire_resident(&[id_a, id_out])?;
    let (a, out) = (&resident.spans[0], &resident.spans[1]);

    let name = if largest { "reduce_axis_argmax" } else { "reduce_axis_argmin" };
    let pipeline = pipeline_for(engine, name, &super::shaders::axis_argmax_wgsl(largest))?;
    let total_outputs_u32 = u32::try_from(total_outputs).map_err(|_| Hc4jError::Unsupported("total outputs exceeds u32"))?;
    let grid = crate::ops::elementwise::grid_for_threads(engine, total_outputs)?;

    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, "reduce (axis argmax)", |rec| {
        let uniform = rec.uniform(&params_bytes([outer, axis_size, inner, total_outputs_u32]))?;
        let bg = bind_group(engine, &pipeline, vec![a.binding(), out.binding(), uniform]);
        rec.dispatch(&pipeline, &bg, grid, total_elements * 4 + total_outputs * 4);
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}
