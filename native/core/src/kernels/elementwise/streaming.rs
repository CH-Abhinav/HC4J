use std::collections::VecDeque;

use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::memory::budget::Reservation;
use crate::memory::manager::try_with_capacity;
use crate::memory::spill::{self, SpillFile};
use crate::memory::transfer::TRANSFER_CHUNK;
use crate::memory::{DeviceBlock, DeviceSpan, HostBlock, Residency, Snapshot, TensorId, TieredMemoryManager};
use crate::engine::ErrorTrap;

use super::grid::{grid_for_threads, max_binding_tile};
use super::{UniformFn, DEVICE_OUTPUT_INFLIGHT, ELEMS_PER_CONTIGUOUS_THREAD, MIN_STREAM_TILE};

/// Where streamed output blocks land: in place when the output tensor is
/// already VRAM-resident, otherwise in fresh storage that replaces the
/// output tensor's when the op completes. Shared with the streaming matmul.
pub enum OutputSink {
    /// The output tensor is VRAM-resident (and pinned): write in place.
    Existing(DeviceSpan),
    /// Fresh storage that replaces the output's when the op completes.
    Device(DeviceBlock),
    Host { data: Vec<u8>, reservation: Reservation },
    Disk { file: SpillFile, written: u64 },
}

impl OutputSink {
    pub fn device_span(&self) -> Option<&DeviceSpan> {
        match self {
            OutputSink::Existing(span) => Some(span),
            OutputSink::Device(block) => Some(block.span()),
            _ => None,
        }
    }

    pub fn append(&mut self, chunk: &[u8]) -> Hc4jResult<()> {
        match self {
            OutputSink::Existing(_) | OutputSink::Device(_) => {
                Err(Hc4jError::Device("device sink does not stream".to_string()))
            }
            OutputSink::Host { data, .. } => {
                // Capacity was reserved up front; never let extend reallocate
                // (an infallible reallocation aborts on failure).
                if data.len() + chunk.len() > data.capacity() {
                    return Err(Hc4jError::Readback("streamed output overran its tensor".to_string()));
                }
                data.extend_from_slice(chunk);
                Ok(())
            }
            OutputSink::Disk { file, written } => {
                file.write_all_at(chunk, *written)?;
                *written += chunk.len() as u64;
                Ok(())
            }
        }
    }

    /// New storage to install for the output, or `None` if it was written in
    /// place.
    pub fn into_residency(self) -> Option<Residency> {
        match self {
            OutputSink::Existing(_) => None,
            OutputSink::Device(block) => Some(Residency::DeviceResident(block)),
            OutputSink::Host { data, reservation } => Some(Residency::HostResident(HostBlock::new(data, reservation))),
            OutputSink::Disk { file, .. } => Some(Residency::Evicted(file)),
        }
    }
}

pub fn open_output_sink(mgr: &TieredMemoryManager, size: u64) -> Hc4jResult<OutputSink> {
    match mgr.allocate_device(size, false) {
        Ok(block) => return Ok(OutputSink::Device(block)),
        Err(Hc4jError::OutOfMemory) => {}
        Err(err) => return Err(err),
    }
    if let Some(reservation) = mgr.reserve_host(size)?
        && let Some(data) = try_with_capacity(size)
    {
        return Ok(OutputSink::Host { data, reservation });
    }
    Ok(OutputSink::Disk {
        file: SpillFile::create(mgr.spill_dir(), size)?,
        written: 0,
    })
}

/// Allocates tile-sized scratch for `staged` off-device inputs (and the
/// output if `need_out`), halving the tile until they fit. Scratch comes out
/// of the VRAM budget like any tensor, so it can evict LRU tensors too.
pub(crate) fn allocate_scratch(
    mgr: &TieredMemoryManager,
    staged: usize,
    need_out: bool,
    size: u64,
) -> Hc4jResult<(u64, Vec<DeviceBlock>, Option<DeviceBlock>)> {
    let engine = mgr.engine();
    let align = (engine.limits.min_storage_buffer_offset_alignment as u64).max(4);
    let whole = size.div_ceil(align) * align;
    let mut tile = max_binding_tile(engine).min(TRANSFER_CHUNK).min(whole);
    loop {
        let attempt = (|| -> Hc4jResult<(Vec<DeviceBlock>, Option<DeviceBlock>)> {
            let mut inputs = Vec::with_capacity(staged);
            for _ in 0..staged {
                inputs.push(mgr.allocate_device(tile, false)?);
            }
            let out = if need_out { Some(mgr.allocate_device(tile, false)?) } else { None };
            Ok((inputs, out))
        })();
        match attempt {
            Ok((inputs, out)) => return Ok((tile, inputs, out)),
            Err(Hc4jError::OutOfMemory) if tile / 2 >= MIN_STREAM_TILE.min(whole) && tile > align => {
                // Halving keeps the tile a multiple of the offset alignment.
                tile = (tile / 2).div_ceil(align) * align;
            }
            Err(err) => return Err(err),
        }
    }
}

pub(crate) fn stream_contiguous(
    mgr: &TieredMemoryManager,
    pipeline: &wgpu::ComputePipeline,
    uniform: &UniformFn<'_>,
    inputs: &[TensorId],
    output: TensorId,
    size: u64,
) -> Hc4jResult<()> {
    let engine = mgr.engine();
    // Pinned for the whole op: device-resident operands cannot be evicted,
    // and host/disk snapshots stay readable even if the tensor migrates.
    let mut sources = Vec::with_capacity(inputs.len());
    let mut pins = Vec::with_capacity(inputs.len() + 1);
    for &id in inputs {
        let (snapshot, _, pin) = mgr.pin_snapshot(id)?;
        sources.push(snapshot);
        pins.push(pin);
    }
    let (out_snapshot, _, out_pin) = mgr.pin_snapshot(output)?;
    pins.push(out_pin);
    let mut sink = match out_snapshot {
        Snapshot::Device(span) => OutputSink::Existing(span),
        _ => open_output_sink(mgr, size)?,
    };

    let staged: Vec<usize> = sources
        .iter()
        .enumerate()
        .filter(|(_, s)| !matches!(s, Snapshot::Device(_)))
        .map(|(i, _)| i)
        .collect();
    let need_out = sink.device_span().is_none();
    let (tile, scratch_in, scratch_out) = allocate_scratch(mgr, staged.len(), need_out, size)?;

    let mut disk_buf = Vec::new();
    if sources.iter().any(|s| matches!(s, Snapshot::Disk(_))) {
        disk_buf = try_with_capacity(tile).ok_or(Hc4jError::OutOfMemory)?;
        disk_buf.resize(tile as usize, 0);
    }
    let layout = pipeline.get_bind_group_layout(0);
    let n = sources.len();

    mgr.with_ring(|ring| {
        let slots = ring.slot_count() as u64;
        let tiles = size.div_ceil(tile);
        let mut device_inflight = VecDeque::with_capacity(DEVICE_OUTPUT_INFLIGHT + 1);

        for t in 0..tiles {
            let offset = t * tile;
            let len = tile.min(size - offset);
            let slot = (t % slots) as usize;
            if need_out {
                // Retire tile t - slots before reusing its staging slot.
                ring.drain(engine, slot, &mut |chunk: &[u8]| sink.append(chunk))?;
            }

            // Stage off-device inputs into scratch (ordered after earlier work).
            for (scratch, &k) in scratch_in.iter().zip(&staged) {
                let dst = scratch.span();
                match &sources[k] {
                    Snapshot::Host(data) => {
                        let slice = data
                            .get(offset as usize..(offset + len) as usize)
                            .ok_or(Hc4jError::Readback("host tensor shorter than recorded size".to_string()))?;
                        engine.stream.write_buffer(&dst.buffer, dst.offset, slice);
                    }
                    Snapshot::Disk(file) => {
                        let buf = &mut disk_buf[..len as usize];
                        spill::read_exact_at(file, buf, offset)?;
                        engine.stream.write_buffer(&dst.buffer, dst.offset, buf);
                    }
                    Snapshot::Device(_) => {}
                }
            }

            let trap = ErrorTrap::push(&engine.device);
            let staging = ring.buffer(slot);
            let ((), submission) = engine.stream.submit_now(1, |rec| {
                let uniform_binding = rec.uniform(&uniform((len / 4) as u32))?;
                let mut entries = Vec::with_capacity(n + 2);
                let mut staged_iter = scratch_in.iter();
                for (k, source) in sources.iter().enumerate() {
                    let resource = match source {
                        Snapshot::Device(span) => span.sub_binding(offset, len),
                        _ => staged_iter
                            .next()
                            .ok_or(Hc4jError::Device("input scratch missing".to_string()))?
                            .span()
                            .sub_binding(0, len),
                    };
                    entries.push(wgpu::BindGroupEntry { binding: k as u32, resource });
                }
                let out_resource = match (sink.device_span(), scratch_out.as_ref()) {
                    (Some(span), _) => span.sub_binding(offset, len),
                    (None, Some(scratch)) => scratch.span().sub_binding(0, len),
                    (None, None) => return Err(Hc4jError::Device("output scratch missing".to_string())),
                };
                entries.push(wgpu::BindGroupEntry { binding: n as u32, resource: out_resource });
                entries.push(wgpu::BindGroupEntry {
                    binding: n as u32 + 1,
                    resource: uniform_binding,
                });
                let bind_group = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("HC4J Stream Tile"),
                    layout: &layout,
                    entries: &entries,
                });
                let grid = grid_for_threads(engine, (len / 4).div_ceil(ELEMS_PER_CONTIGUOUS_THREAD))?;
                rec.dispatch(pipeline, &bind_group, grid, len * (n as u64 + 1));
                if let Some(scratch) = scratch_out.as_ref().filter(|_| need_out) {
                    let src = scratch.span();
                    rec.encoder().copy_buffer_to_buffer(&src.buffer, src.offset, staging, 0, len);
                }
                Ok(())
            })?;
            trap.finish()?;

            if need_out {
                ring.arm(slot, len, submission);
            } else {
                // No readback throttles the loop, so bound the upload staging
                // wgpu holds for in-flight tiles explicitly.
                device_inflight.push_back(submission);
                if device_inflight.len() > DEVICE_OUTPUT_INFLIGHT
                    && let Some(oldest) = device_inflight.pop_front()
                {
                    engine.wait_for(oldest)?;
                }
            }
        }
        if need_out {
            for t in tiles.saturating_sub(slots)..tiles {
                ring.drain(engine, (t % slots) as usize, &mut |chunk: &[u8]| sink.append(chunk))?;
            }
        }
        Ok(())
    })?;

    if let Some(residency) = sink.into_residency() {
        mgr.replace_residency(output, residency)?;
    }
    drop(pins);
    mgr.record_streamed_op();
    Ok(())
}
