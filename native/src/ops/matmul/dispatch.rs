//! Matrix multiplication and transpose dispatch.

use crate::error::{Hc4jError, Hc4jResult};
use crate::memory::{DeviceBlock, DeviceSpan, TensorId, manager};
use crate::ops::elementwise::grid_2d;
use crate::ops::matmul::plan::{Caps, Kernel, Operands, plan};
use crate::ops::matmul::shaders::{TRANSPOSE_WGSL, kernel_source};
use crate::ops::matmul::streaming::run_matmul_streaming;
use crate::stream::Recorder;
use crate::{ErrorTrap, GpuEngine};

pub(crate) fn dims_bytes(p: [u32; 4]) -> Vec<u8> {
    bytemuck::bytes_of(&p).to_vec()
}

/// Grid for a kernel tiled over (cols, rows) with the row tiles folded into
/// Y×Z when they exceed the per-dimension limit.
pub(crate) fn tiled_grid(engine: &GpuEngine, col_tiles: u32, row_tiles: u32) -> Hc4jResult<(u32, u32, u32)> {
    let max = engine.limits.max_compute_workgroups_per_dimension;
    if col_tiles > max {
        return Err(Hc4jError::Unsupported("matrix too wide for one dispatch"));
    }
    let (y, z) = grid_2d(row_tiles, max);
    if z > max {
        return Err(Hc4jError::Unsupported("matrix too tall for one dispatch"));
    }
    Ok((col_tiles.max(1), y, z))
}

/// Dispatch grid for a kernel producing an `rb x n` output block.
///
/// With no transpose flags the planner only picks the row-mapped GEMV when
/// `n == 1`, so its output length is `rb`; the column mapping always produces
/// `n` outputs.
pub(crate) fn block_grid(engine: &GpuEngine, kernel: Kernel, rb: u32, n: u32) -> Hc4jResult<(u32, u32, u32)> {
    let max = engine.limits.max_compute_workgroups_per_dimension;
    Ok(match kernel {
        Kernel::GemvRows { lanes, .. } => {
            let (x, y) = grid_2d(rb.div_ceil(256 / lanes), max);
            (x, y, 1)
        }
        Kernel::GemvCols { cols } => {
            let (x, y) = grid_2d(n.div_ceil(cols), max);
            (x, y, 1)
        }
        Kernel::Register { cfg, .. } => tiled_grid(engine, n.div_ceil(cfg.bn), rb.div_ceil(cfg.bm))?,
        Kernel::Tiled16 => tiled_grid(engine, n.div_ceil(16), rb.div_ceil(16))?,
    })
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
        label: Some("HC4J Matmul"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

pub(crate) fn encode_transpose(
    rec: &mut Recorder<'_>,
    engine: &GpuEngine,
    pipeline: &wgpu::ComputePipeline,
    src: &DeviceSpan,
    dst: &DeviceSpan,
    rows: u32,
    cols: u32,
) -> Hc4jResult<()> {
    let uniform = rec.uniform(&dims_bytes([rows, cols, 0, 0]))?;
    let bg = bind_group(engine, pipeline, vec![src.binding(), dst.binding(), uniform]);
    let grid = tiled_grid(engine, cols.div_ceil(32), rows.div_ceil(32))?;
    rec.dispatch(pipeline, &bg, grid, src.size + dst.size);
    Ok(())
}

pub(crate) fn transpose_pipeline(engine: &GpuEngine) -> Hc4jResult<wgpu::ComputePipeline> {
    engine.get_or_compile("transpose_f32", "main", TRANSPOSE_WGSL)
}

pub(crate) fn checked_bytes(a: u32, b: u32) -> Hc4jResult<u64> {
    (a as u64)
        .checked_mul(b as u64)
        .and_then(|e| e.checked_mul(4))
        .ok_or(Hc4jError::InvalidParam("matrix size overflows"))
}

pub fn run_matmul(id_a: TensorId, id_b: TensorId, id_out: TensorId, m: u32, n: u32, k: u32, flags: u32) -> Hc4jResult<()> {
    let mgr = manager()?;
    let engine = mgr.engine();
    let plan = plan(&Caps::of(engine), m, n, k, flags)?;
    if mgr.size_of(id_a)? != checked_bytes(m, k)?
        || mgr.size_of(id_b)? != checked_bytes(k, n)?
        || mgr.size_of(id_out)? != checked_bytes(m, n)?
    {
        return Err(Hc4jError::InvalidParam("tensor sizes do not match (m, n, k)"));
    }
    if id_out == id_a || id_out == id_b {
        return Err(Hc4jError::InvalidParam("matmul output must not alias an input"));
    }

    let resident = match mgr.acquire_resident(&[id_a, id_b, id_out]) {
        Ok(resident) => resident,
        // The working set does not fit in VRAM: stream A and C in row blocks.
        Err(Hc4jError::OutOfMemory) => {
            crate::hc4j_trace!("matmul {m}x{n}x{k} exceeds VRAM; streaming row blocks");
            return run_matmul_streaming(mgr, id_a, id_b, id_out, m, n, k, flags);
        }
        Err(err) => return Err(err),
    };
    let (a, b, c) = (&resident.spans[0], &resident.spans[1], &resident.spans[2]);
    if c.overlaps(a) || c.overlaps(b) {
        return Err(Hc4jError::InvalidParam("matmul output must not alias an input"));
    }
    let max_binding = engine.limits.max_storage_buffer_binding_size;
    if [a, b, c].iter().any(|s| s.size > max_binding) {
        return Err(Hc4jError::Unsupported("matmul operand exceeds the storage-binding limit"));
    }

    let (name, source) = kernel_source(plan.kernel);
    let pipeline = engine.get_or_compile(&name, "main", &source)?;

    match plan.operands {
        Operands::Gemv { matrix_is_a, out_len, k } => {
            let (matrix, vector) = if matrix_is_a { (a, b) } else { (b, a) };
            let grid = match plan.kernel {
                Kernel::GemvRows { lanes, .. } => {
                    let wgs = out_len.div_ceil(256 / lanes);
                    let (x, y) = grid_2d(wgs, engine.limits.max_compute_workgroups_per_dimension);
                    (x, y, 1)
                }
                Kernel::GemvCols { cols } => {
                    let (x, y) = grid_2d(out_len.div_ceil(cols), engine.limits.max_compute_workgroups_per_dimension);
                    (x, y, 1)
                }
                _ => return Err(Hc4jError::Device("GEMV plan with a GEMM kernel".to_string())),
            };
            let bytes = matrix.size + vector.size + c.size;
            let trap = ErrorTrap::push(&engine.device);
            let recorded = engine.stream.record(1, "matmul (gemv)", |rec| {
                let uniform = rec.uniform(&dims_bytes([out_len, k, 0, 0]))?;
                let bg = bind_group(engine, &pipeline, vec![matrix.binding(), vector.binding(), c.binding(), uniform]);
                rec.dispatch(&pipeline, &bg, grid, bytes);
                Ok(())
            });
            let trapped = trap.finish();
            recorded.and(trapped)
        }
        Operands::Gemm { pretranspose_a, pretranspose_b } => {
            // Scratch for in-VRAM transposition: slab regions or dedicated
            // buffers, released (quarantined) once the work is recorded.
            let scratch_a: Option<DeviceBlock> =
                if pretranspose_a { Some(mgr.allocate_device(a.size, false)?) } else { None };
            let scratch_b: Option<DeviceBlock> =
                if pretranspose_b { Some(mgr.allocate_device(b.size, false)?) } else { None };
            let transpose = if pretranspose_a || pretranspose_b { Some(transpose_pipeline(engine)?) } else { None };
            let (bm, bn) = match plan.kernel {
                Kernel::Register { cfg, .. } => (cfg.bm, cfg.bn),
                _ => (16, 16),
            };
            let grid = tiled_grid(engine, n.div_ceil(bn), m.div_ceil(bm))?;
            let slots = 1 + u32::from(pretranspose_a) + u32::from(pretranspose_b);

            // Transposes and the GEMM go into one recorded unit: one command
            // buffer, one submission (or none extra inside a batch).
            let trap = ErrorTrap::push(&engine.device);
            let recorded = engine.stream.record(slots, "matmul (gemm)", |rec| {
                let a_eff = match (&scratch_a, &transpose) {
                    (Some(scratch), Some(tp)) => {
                        // A stored K×M -> M×K.
                        encode_transpose(rec, engine, tp, a, scratch.span(), k, m)?;
                        scratch.span()
                    }
                    _ => a,
                };
                let b_eff = match (&scratch_b, &transpose) {
                    (Some(scratch), Some(tp)) => {
                        // B stored N×K -> K×N.
                        encode_transpose(rec, engine, tp, b, scratch.span(), n, k)?;
                        scratch.span()
                    }
                    _ => b,
                };
                let uniform = rec.uniform(&dims_bytes([m, n, k, 0]))?;
                let bg = bind_group(engine, &pipeline, vec![a_eff.binding(), b_eff.binding(), c.binding(), uniform]);
                rec.dispatch(&pipeline, &bg, grid, a_eff.size + b_eff.size + c.size);
                Ok(())
            });
            let trapped = trap.finish();
            drop((scratch_a, scratch_b));
            recorded.and(trapped)
        }
    }
}

pub fn run_transpose(id_in: TensorId, id_out: TensorId, rows: u32, cols: u32) -> Hc4jResult<()> {
    if rows == 0 || cols == 0 {
        return Err(Hc4jError::InvalidParam("transpose dimensions must be non-zero"));
    }
    let mgr = manager()?;
    let engine = mgr.engine();
    let bytes = checked_bytes(rows, cols)?;
    if mgr.size_of(id_in)? != bytes || mgr.size_of(id_out)? != bytes {
        return Err(Hc4jError::InvalidParam("tensor sizes do not match rows x cols"));
    }
    if id_in == id_out {
        return Err(Hc4jError::InvalidParam("transpose cannot run in place"));
    }
    let resident = mgr.acquire_resident(&[id_in, id_out])?;
    let (src, dst) = (&resident.spans[0], &resident.spans[1]);
    if src.size > engine.limits.max_storage_buffer_binding_size {
        return Err(Hc4jError::Unsupported("transpose operand exceeds the storage-binding limit"));
    }
    let pipeline = transpose_pipeline(engine)?;
    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine
        .stream
        .record(1, "transpose", |rec| encode_transpose(rec, engine, &pipeline, src, dst, rows, cols));
    let trapped = trap.finish();
    recorded.and(trapped)
}
