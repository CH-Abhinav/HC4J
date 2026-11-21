//! Out-of-core row-block and K-blocked streaming matrix multiplication.

use std::collections::VecDeque;

use crate::error::{Hc4jError, Hc4jResult};
use crate::memory::manager::try_with_capacity;
use crate::memory::spill;
use crate::memory::transfer::ReadbackRing;
use crate::memory::{DeviceBlock, Snapshot, TensorId, TieredMemoryManager};
use crate::ops::elementwise::{OutputSink, open_output_sink};
use crate::ops::matmul::dispatch::{bind_group, block_grid, dims_bytes, encode_transpose, transpose_pipeline};
use crate::ops::matmul::plan::{Caps, FLAG_TRANS_A, FLAG_TRANS_B, Operands, plan};
use crate::ops::matmul::shaders::kernel_source;
use crate::ErrorTrap;

/// Target scratch size per streamed operand block.
pub const STREAM_BLOCK_BYTES: u64 = 8 << 20;

/// Out-of-core matmul for working sets larger than VRAM.
///
/// `C` is computed in row blocks: `C[i..i+rb] = A[i..i+rb] · B`. A row block of
/// a row-major matrix is contiguous, so each block is one copy in and one copy
/// out, and K is never split — which keeps the kernels (and their accumulators)
/// exactly as they are in the resident path.
///
/// B stays VRAM-resident for the whole operation, since every block needs all
/// of it. If B alone cannot be made resident this returns `OutOfMemory`, and a
/// stored-transposed A returns `Unsupported` (transposing it would need the
/// whole matrix resident; transpose it explicitly first).
#[allow(clippy::too_many_arguments)]
pub fn run_matmul_streaming(
    mgr: &TieredMemoryManager,
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
) -> Hc4jResult<()> {
    let engine = mgr.engine();
    if flags & FLAG_TRANS_A != 0 {
        return Err(Hc4jError::Unsupported(
            "a stored-transposed A cannot be streamed; transpose it into its own tensor first",
        ));
    }
    let max_binding = engine.limits.max_storage_buffer_binding_size;

    // B must be resident: every row block multiplies by all of it. If even B
    // does not fit, fall back to blocking over K as well.
    let b_resident = match mgr.acquire_resident(&[id_b]) {
        Ok(resident) => resident,
        Err(Hc4jError::OutOfMemory) => {
            crate::hc4j_trace!("matmul {m}x{n}x{k}: B does not fit either; blocking over K");
            return run_matmul_blocked(mgr, id_a, id_b, id_out, m, n, k, flags);
        }
        Err(err) => return Err(err),
    };
    let b_span = &b_resident.spans[0];
    if b_span.size > max_binding {
        return Err(Hc4jError::Unsupported("matmul B exceeds the storage-binding limit"));
    }
    // Pre-transpose B once, outside the block loop.
    let b_scratch = if flags & FLAG_TRANS_B != 0 { Some(mgr.allocate_device(b_span.size, false)?) } else { None };
    if let Some(scratch) = &b_scratch {
        let pipeline = transpose_pipeline(engine)?;
        let trap = ErrorTrap::push(&engine.device);
        let recorded = engine
            .stream
            .record(1, "transpose (matmul B)", |rec| encode_transpose(rec, engine, &pipeline, b_span, scratch.span(), n, k));
        let trapped = trap.finish();
        recorded.and(trapped)?;
    }
    let b_eff = b_scratch.as_ref().map_or(b_span, |s| s.span());

    let (a_source, _, _a_pin) = mgr.pin_snapshot(id_a)?;
    let (out_snapshot, out_size, _out_pin) = mgr.pin_snapshot(id_out)?;
    let mut sink = match out_snapshot {
        Snapshot::Device(span) => OutputSink::Existing(span),
        _ => open_output_sink(mgr, out_size)?,
    };

    // Block rows so both scratch buffers stay near the target size.
    let a_row = (k as u64) * 4;
    let c_row = (n as u64) * 4;
    let mut rows = (STREAM_BLOCK_BYTES / a_row.max(c_row)).clamp(1, m as u64) as u32;
    let (scratch_a, scratch_c) = loop {
        let attempt = (|| -> Hc4jResult<(DeviceBlock, DeviceBlock)> {
            let a_blk = mgr.allocate_device(rows as u64 * a_row, false)?;
            let c_blk = mgr.allocate_device(rows as u64 * c_row, false)?;
            Ok((a_blk, c_blk))
        })();
        match attempt {
            Ok(pair) => break pair,
            Err(Hc4jError::OutOfMemory) if rows > 1 => rows /= 2,
            Err(err) => return Err(err),
        }
    };
    if scratch_a.span().size > max_binding || scratch_c.span().size > max_binding {
        return Err(Hc4jError::Unsupported("matmul row block exceeds the storage-binding limit"));
    }

    let mut host_block = Vec::new();
    if matches!(a_source, Snapshot::Disk(_)) {
        host_block = try_with_capacity(rows as u64 * a_row).ok_or(Hc4jError::OutOfMemory)?;
        host_block.resize((rows as u64 * a_row) as usize, 0);
    }

    let blocks = m.div_ceil(rows);
    let need_ring = sink.device_span().is_none();
    let mut stream_block = |ring: Option<&mut ReadbackRing>| -> Hc4jResult<()> {
        let mut ring = ring;
        let slots = ring.as_ref().map_or(1, |r| r.slot_count());
        let mut inflight: VecDeque<wgpu::SubmissionIndex> = VecDeque::new();
        for block in 0..blocks {
            let row0 = block * rows;
            let rb = rows.min(m - row0);
            let a_bytes = rb as u64 * a_row;
            let c_bytes = rb as u64 * c_row;
            let slot = (block as usize) % slots;

            // Retire the block that previously used this staging slot.
            if let Some(ring) = ring.as_mut() {
                ring.drain(engine, slot, &mut |chunk: &[u8]| sink.append(chunk))?;
            }

            // Stage this row block of A (contiguous) into scratch.
            let dst = scratch_a.span();
            match &a_source {
                Snapshot::Host(data) => {
                    let start = (row0 as u64 * a_row) as usize;
                    let slice = data
                        .get(start..start + a_bytes as usize)
                        .ok_or(Hc4jError::Readback("A shorter than its recorded size".to_string()))?;
                    engine.stream.write_buffer(&dst.buffer, dst.offset, slice);
                }
                Snapshot::Disk(file) => {
                    let buf = &mut host_block[..a_bytes as usize];
                    spill::read_exact_at(file, buf, row0 as u64 * a_row)?;
                    engine.stream.write_buffer(&dst.buffer, dst.offset, buf);
                }
                Snapshot::Device(_) => {} // copied on the GPU inside the recording below
            }

            let block_plan = plan(&Caps::of(engine), rb, n, k, 0)?;
            let (name, source) = kernel_source(block_plan.kernel);
            let pipeline = engine.get_or_compile(&name, "main", &source)?;
            let grid = block_grid(engine, block_plan.kernel, rb, n)?;

            let trap = ErrorTrap::push(&engine.device);
            let out_span = sink.device_span().cloned();
            let (_, submission) = engine.stream.submit_now(1, |rec| {
                if let Snapshot::Device(a_span) = &a_source {
                    let src = a_span;
                    rec.encoder()
                        .copy_buffer_to_buffer(&src.buffer, src.offset + row0 as u64 * a_row, &dst.buffer, dst.offset, a_bytes);
                }
                // GEMV kernels read [rows, K]; GEMM kernels read [M, N, K].
                let dims = match block_plan.operands {
                    Operands::Gemv { out_len, k: reduce, .. } => dims_bytes([out_len, reduce, 0, 0]),
                    Operands::Gemm { .. } => dims_bytes([rb, n, k, 0]),
                };
                let uniform = rec.uniform(&dims)?;
                let a_bind = scratch_a.span().sub_binding(0, a_bytes);
                let c_bind = scratch_c.span().sub_binding(0, c_bytes);
                // A GEMV block whose matrix is B (the m == 1 mapping) binds the
                // matrix first and the streamed block as the vector.
                let (first, second) = match block_plan.operands {
                    Operands::Gemv { matrix_is_a: false, .. } => (b_eff.binding(), a_bind),
                    _ => (a_bind, b_eff.binding()),
                };
                let bg = bind_group(engine, &pipeline, vec![first, second, c_bind, uniform]);
                rec.dispatch(&pipeline, &bg, grid, a_bytes + b_eff.size + c_bytes);
                match (&out_span, ring.as_ref()) {
                    // Output already in VRAM: copy the block straight back.
                    (Some(out), _) => {
                        let src = scratch_c.span();
                        rec.encoder().copy_buffer_to_buffer(
                            &src.buffer,
                            src.offset,
                            &out.buffer,
                            out.offset + row0 as u64 * c_row,
                            c_bytes,
                        );
                    }
                    (None, Some(ring)) => {
                        let src = scratch_c.span();
                        rec.encoder()
                            .copy_buffer_to_buffer(&src.buffer, src.offset, ring.buffer(slot), 0, c_bytes);
                    }
                    (None, None) => return Err(Hc4jError::Device("streamed matmul without an output sink".to_string())),
                }
                Ok(())
            })?;
            trap.finish()?;

            if let Some(ring) = ring.as_mut() {
                ring.arm(slot, c_bytes, submission);
            } else {
                inflight.push_back(submission);
                if inflight.len() > 2
                    && let Some(oldest) = inflight.pop_front()
                {
                    engine.wait_for(oldest)?;
                }
            }
        }
        if let Some(ring) = ring.as_mut() {
            for block in blocks.saturating_sub(slots as u32)..blocks {
                ring.drain(engine, block as usize % slots, &mut |chunk: &[u8]| sink.append(chunk))?;
            }
        }
        Ok(())
    };

    if need_ring {
        mgr.with_ring(|ring| stream_block(Some(ring)))?;
    } else {
        stream_block(None)?;
    }

    if let Some(residency) = sink.into_residency() {
        mgr.replace_residency(id_out, residency)?;
    }
    mgr.record_streamed_op();
    Ok(())
}

/// Copies `len` bytes of a non-resident operand into `host` at `at`.
fn gather_host(source: &Snapshot, src_offset: u64, len: u64, host: &mut [u8], at: usize) -> Hc4jResult<()> {
    let dst = host
        .get_mut(at..at + len as usize)
        .ok_or_else(|| Hc4jError::Device("staging buffer too small".to_string()))?;
    match source {
        Snapshot::Host(data) => {
            let src = data
                .get(src_offset as usize..(src_offset + len) as usize)
                .ok_or_else(|| Hc4jError::Readback("operand shorter than its recorded size".to_string()))?;
            dst.copy_from_slice(src);
            Ok(())
        }
        Snapshot::Disk(file) => Ok(spill::read_exact_at(file, dst, src_offset)?),
        Snapshot::Device(_) => Err(Hc4jError::Device("device operands are copied on the GPU".to_string())),
    }
}

/// Out-of-core matmul for working sets where even B cannot be made resident.
///
/// Blocks over rows of A/C **and** over K: `C[i] = Σ_j A[i, j] · B[j]`, with
/// the C block held in VRAM and accumulated across K blocks (`beta = 1` after
/// the first). A blocks are gathered row by row, since a block of columns is
/// strided in a row-major A; B blocks are contiguous row chunks.
///
/// N is kept whole, so one row of B plus one row of C must fit in VRAM. B is
/// re-read once per row block, which is the price of not having room for it;
/// the row block is made as large as the budget allows to amortise that.
#[allow(clippy::too_many_arguments)]
pub fn run_matmul_blocked(
    mgr: &TieredMemoryManager,
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
) -> Hc4jResult<()> {
    let engine = mgr.engine();
    if flags & FLAG_TRANS_A != 0 {
        return Err(Hc4jError::Unsupported(
            "a stored-transposed A cannot be streamed; transpose it into its own tensor first",
        ));
    }
    if flags & FLAG_TRANS_B != 0 {
        return Err(Hc4jError::Unsupported(
            "a stored-transposed B larger than VRAM cannot be transposed in place; transpose it into its own tensor first",
        ));
    }

    let (a_source, _, _a_pin) = mgr.pin_snapshot(id_a)?;
    let (b_source, _, _b_pin) = mgr.pin_snapshot(id_b)?;
    let (out_snapshot, out_size, _out_pin) = mgr.pin_snapshot(id_out)?;
    let mut sink = match out_snapshot {
        Snapshot::Device(span) => OutputSink::Existing(span),
        _ => open_output_sink(mgr, out_size)?,
    };

    // Block sizes: aim for ~STREAM_BLOCK_BYTES per operand block, then shrink
    // the larger dimension until all three blocks fit in VRAM.
    let row_c = (n as u64) * 4;
    let mut rows = (STREAM_BLOCK_BYTES / row_c).clamp(1, m as u64) as u32;
    let mut depth = (STREAM_BLOCK_BYTES / row_c).clamp(1, k as u64) as u32;
    let (a_block, b_block, c_block) = loop {
        let attempt = (|| -> Hc4jResult<(DeviceBlock, DeviceBlock, DeviceBlock)> {
            let a = mgr.allocate_device(rows as u64 * depth as u64 * 4, false)?;
            let b = mgr.allocate_device(depth as u64 * row_c, false)?;
            let c = mgr.allocate_device(rows as u64 * row_c, false)?;
            Ok((a, b, c))
        })();
        match attempt {
            Ok(blocks) => break blocks,
            Err(Hc4jError::OutOfMemory) if rows > 1 || depth > 1 => {
                if rows >= depth && rows > 1 {
                    rows /= 2;
                } else {
                    depth /= 2;
                }
            }
            Err(err) => return Err(err),
        }
    };
    let max_binding = engine.limits.max_storage_buffer_binding_size;
    if [a_block.span(), b_block.span(), c_block.span()].iter().any(|s| s.size > max_binding) {
        return Err(Hc4jError::Unsupported("matmul block exceeds the storage-binding limit"));
    }
    crate::hc4j_trace!("blocked matmul {m}x{n}x{k}: {rows} rows x {depth} deep");

    // Host staging for operands that are not device-resident.
    let mut a_stage = Vec::new();
    if !matches!(a_source, Snapshot::Device(_)) {
        a_stage = try_with_capacity(rows as u64 * depth as u64 * 4).ok_or(Hc4jError::OutOfMemory)?;
        a_stage.resize(rows as usize * depth as usize * 4, 0);
    }
    let mut b_stage = Vec::new();
    if matches!(b_source, Snapshot::Disk(_)) {
        b_stage = try_with_capacity(depth as u64 * row_c).ok_or(Hc4jError::OutOfMemory)?;
        b_stage.resize((depth as u64 * row_c) as usize, 0);
    }

    let row_blocks = m.div_ceil(rows);
    let k_blocks = k.div_ceil(depth);
    let need_ring = sink.device_span().is_none();

    let mut run = |ring: Option<&mut ReadbackRing>| -> Hc4jResult<()> {
        let mut ring = ring;
        let slots = ring.as_ref().map_or(1, |r| r.slot_count());
        let mut inflight: VecDeque<wgpu::SubmissionIndex> = VecDeque::new();
        for row_block in 0..row_blocks {
            let row0 = row_block * rows;
            let rb = rows.min(m - row0);
            let c_bytes = rb as u64 * row_c;
            let slot = (row_block as usize) % slots;
            if let Some(ring) = ring.as_mut() {
                // Retire the row block that previously used this slot.
                ring.drain(engine, slot, &mut |chunk: &[u8]| sink.append(chunk))?;
            }

            for k_block in 0..k_blocks {
                let k0 = k_block * depth;
                let kb = depth.min(k - k0);
                let a_bytes = rb as u64 * kb as u64 * 4;
                let b_bytes = kb as u64 * row_c;

                // Stage this A block (strided: one gather per row) and B block
                // (contiguous) unless they are already in VRAM.
                if !matches!(a_source, Snapshot::Device(_)) {
                    for r in 0..rb as u64 {
                        let src = ((row0 as u64 + r) * k as u64 + k0 as u64) * 4;
                        gather_host(&a_source, src, kb as u64 * 4, &mut a_stage, (r * kb as u64 * 4) as usize)?;
                    }
                    let dst = a_block.span();
                    engine.stream.write_buffer(&dst.buffer, dst.offset, &a_stage[..a_bytes as usize]);
                }
                match &b_source {
                    Snapshot::Host(data) => {
                        let start = (k0 as u64 * row_c) as usize;
                        let src = data
                            .get(start..start + b_bytes as usize)
                            .ok_or_else(|| Hc4jError::Readback("B shorter than its recorded size".to_string()))?;
                        let dst = b_block.span();
                        engine.stream.write_buffer(&dst.buffer, dst.offset, src);
                    }
                    Snapshot::Disk(_) => {
                        gather_host(&b_source, k0 as u64 * row_c, b_bytes, &mut b_stage, 0)?;
                        let dst = b_block.span();
                        engine.stream.write_buffer(&dst.buffer, dst.offset, &b_stage[..b_bytes as usize]);
                    }
                    Snapshot::Device(_) => {}
                }

                let block_plan = plan(&Caps::of(engine), rb, n, kb, 0)?;
                let (name, source) = kernel_source(block_plan.kernel);
                let pipeline = engine.get_or_compile(&name, "main", &source)?;
                let grid = block_grid(engine, block_plan.kernel, rb, n)?;
                let beta = u32::from(k_block > 0);

                let trap = ErrorTrap::push(&engine.device);
                let (_, submission) = engine.stream.submit_now(1, |rec| {
                    if let Snapshot::Device(a_span) = &a_source {
                        let dst = a_block.span();
                        for r in 0..rb as u64 {
                            let src = a_span.offset + ((row0 as u64 + r) * k as u64 + k0 as u64) * 4;
                            rec.encoder().copy_buffer_to_buffer(
                                &a_span.buffer,
                                src,
                                &dst.buffer,
                                dst.offset + r * kb as u64 * 4,
                                kb as u64 * 4,
                            );
                        }
                    }
                    if let Snapshot::Device(b_span) = &b_source {
                        let dst = b_block.span();
                        rec.encoder().copy_buffer_to_buffer(
                            &b_span.buffer,
                            b_span.offset + k0 as u64 * row_c,
                            &dst.buffer,
                            dst.offset,
                            b_bytes,
                        );
                    }
                    let dims = match block_plan.operands {
                        Operands::Gemv { out_len, k: reduce, .. } => dims_bytes([out_len, reduce, beta, 0]),
                        Operands::Gemm { .. } => dims_bytes([rb, n, kb, beta]),
                    };
                    let uniform = rec.uniform(&dims)?;
                    let a_bind = a_block.span().sub_binding(0, a_bytes);
                    let b_bind = b_block.span().sub_binding(0, b_bytes);
                    let c_bind = c_block.span().sub_binding(0, c_bytes);
                    let (first, second) = match block_plan.operands {
                        Operands::Gemv { matrix_is_a: false, .. } => (b_bind, a_bind),
                        _ => (a_bind, b_bind),
                    };
                    let bg = bind_group(engine, &pipeline, vec![first, second, c_bind, uniform]);
                    rec.dispatch(&pipeline, &bg, grid, a_bytes + b_bytes + c_bytes);
                    Ok(())
                })?;
                trap.finish()?;
                inflight.push_back(submission);
                if inflight.len() > 2
                    && let Some(oldest) = inflight.pop_front()
                {
                    engine.wait_for(oldest)?;
                }
            }

            // The row block is complete: hand it to the output.
            let out_span = sink.device_span().cloned();
            let trap = ErrorTrap::push(&engine.device);
            let staging = ring.as_ref().map(|r| r.buffer(slot).clone());
            let (_, submission) = engine.stream.submit_now(0, |rec| {
                let src = c_block.span();
                match (&out_span, &staging) {
                    (Some(out), _) => rec.encoder().copy_buffer_to_buffer(
                        &src.buffer,
                        src.offset,
                        &out.buffer,
                        out.offset + row0 as u64 * row_c,
                        c_bytes,
                    ),
                    (None, Some(staging)) => {
                        rec.encoder().copy_buffer_to_buffer(&src.buffer, src.offset, staging, 0, c_bytes)
                    }
                    (None, None) => return Err(Hc4jError::Device("blocked matmul without an output sink".to_string())),
                }
                Ok(())
            })?;
            trap.finish()?;
            if let Some(ring) = ring.as_mut() {
                ring.arm(slot, c_bytes, submission);
            } else {
                inflight.push_back(submission);
            }
        }
        if let Some(ring) = ring.as_mut() {
            for block in row_blocks.saturating_sub(slots as u32)..row_blocks {
                ring.drain(engine, block as usize % slots, &mut |chunk: &[u8]| sink.append(chunk))?;
            }
        }
        Ok(())
    };

    if need_ring {
        mgr.with_ring(|ring| run(Some(ring)))?;
    } else {
        run(None)?;
    }

    if let Some(residency) = sink.into_residency() {
        mgr.replace_residency(id_out, residency)?;
    }
    mgr.record_streamed_op();
    Ok(())
}
