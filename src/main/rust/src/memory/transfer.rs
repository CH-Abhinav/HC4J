//! Host<->device data movement in bounded chunks.
//!
//! Downloads go through a persistent, double-buffered ring of `MAP_READ`
//! staging buffers: while the host drains slot *k*, the GPU is already
//! copying the next chunk into slot *k+1*. Uploads go through
//! `Queue::write_buffer` (via the command stream, which orders them after
//! recorded work) with at most `UPLOAD_INFLIGHT` chunks in flight, which
//! bounds wgpu's internal staging memory no matter how large the tensor is.

use std::collections::VecDeque;
use std::fs::File;
use std::sync::mpsc;

use crate::error::{Hc4jError, Hc4jResult};
use crate::{ErrorTrap, GPU_WAIT_TIMEOUT, GpuEngine};

use super::spill;

/// Size of one staging slot and of one upload chunk. 32 MiB keeps the PCIe
/// link saturated while keeping staging memory small (2 slots = 64 MiB,
/// allocated in system memory on discrete GPUs).
pub const TRANSFER_CHUNK: u64 = 32 << 20;
const RING_SLOTS: usize = 2;
const UPLOAD_INFLIGHT: usize = 2;
/// In-memory uploads up to this size use one `write_buffer` with no explicit
/// submit: the copy rides along with the next dispatch. Chunking such sizes
/// measured ~35% slower on 40 MiB tensors (a submit per chunk) and buys
/// nothing; it only matters once wgpu's full-size transient staging copy
/// would itself strain host memory.
const SINGLE_SHOT_UPLOAD: u64 = 256 << 20;

struct PendingRead {
    len: u64,
    submission: wgpu::SubmissionIndex,
    mapped: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

struct Slot {
    buffer: wgpu::Buffer,
    pending: Option<PendingRead>,
}

pub struct ReadbackRing {
    slots: Vec<Slot>,
    chunk: u64,
}

impl ReadbackRing {
    pub fn new(engine: &GpuEngine, chunk: u64) -> Hc4jResult<Self> {
        let trap = ErrorTrap::push(&engine.device);
        let slots = (0..RING_SLOTS)
            .map(|i| Slot {
                buffer: engine.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(if i == 0 { "HC4J Readback Slot 0" } else { "HC4J Readback Slot 1" }),
                    size: chunk,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                pending: None,
            })
            .collect();
        trap.finish()?;
        Ok(Self { slots, chunk })
    }

    pub fn chunk(&self) -> u64 {
        self.chunk
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn buffer(&self, slot: usize) -> &wgpu::Buffer {
        &self.slots[slot].buffer
    }

    /// Requests a map of the first `len` bytes of `slot`. Must be called
    /// after `submission` (the one that copies into the slot) is submitted: a
    /// buffer with a pending map cannot be used by a later submission.
    pub fn arm(&mut self, slot: usize, len: u64, submission: wgpu::SubmissionIndex) {
        let (tx, rx) = mpsc::sync_channel(1);
        self.slots[slot].buffer.map_async(wgpu::MapMode::Read, 0..len, move |result| {
            let _ = tx.send(result);
        });
        self.slots[slot].pending = Some(PendingRead {
            len,
            submission,
            mapped: rx,
        });
    }

    /// Waits for the slot's pending read, hands the bytes to `sink`, and
    /// unmaps. A slot with nothing pending is a no-op.
    pub fn drain(
        &mut self,
        engine: &GpuEngine,
        slot: usize,
        sink: &mut dyn FnMut(&[u8]) -> Hc4jResult<()>,
    ) -> Hc4jResult<()> {
        let Some(pending) = self.slots[slot].pending.take() else {
            return Ok(());
        };
        engine.wait_for(pending.submission)?;
        match pending.mapped.recv_timeout(GPU_WAIT_TIMEOUT) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(Hc4jError::Readback(format!("map_async failed: {e}"))),
            Err(_) => return Err(Hc4jError::Readback("map callback never fired".to_string())),
        }
        let buffer = &self.slots[slot].buffer;
        let result = match buffer.get_mapped_range(0..pending.len) {
            Ok(view) => sink(&view),
            Err(e) => Err(Hc4jError::Readback(format!("get_mapped_range failed: {e}"))),
        };
        buffer.unmap();
        result
    }
}

/// Streams `len` bytes of `src` starting at `src_offset` into `sink`, in
/// order, one ring chunk at a time. On error the ring may hold mapped or
/// map-pending slots, so the caller must discard it (see
/// `TieredMemoryManager::with_ring`).
pub fn download(
    engine: &GpuEngine,
    ring: &mut ReadbackRing,
    src: &wgpu::Buffer,
    src_offset: u64,
    len: u64,
    sink: &mut dyn FnMut(&[u8]) -> Hc4jResult<()>,
) -> Hc4jResult<()> {
    if !len.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) || !src_offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
        return Err(Hc4jError::InvalidParam("download range must be 4-byte aligned"));
    }
    let chunk = ring.chunk();
    let slots = ring.slot_count() as u64;
    let chunks = len.div_ceil(chunk);
    for i in 0..chunks {
        let slot = (i % slots) as usize;
        // Retire the chunk that previously occupied this slot (chunk i - slots)
        // before reusing it; chunks therefore reach the sink in order.
        ring.drain(engine, slot, sink)?;

        let offset = i * chunk;
        let this_len = chunk.min(len - offset);
        let trap = ErrorTrap::push(&engine.device);
        let staging = ring.buffer(slot);
        // submit_now flushes batched work first, so the copy sees every
        // previously recorded write to `src`.
        let ((), submission) = engine.stream.submit_now(0, |rec| {
            rec.encoder().copy_buffer_to_buffer(src, src_offset + offset, staging, 0, this_len);
            Ok(())
        })?;
        // If the copy was invalid, mapping the slot would "succeed" and hand
        // back stale bytes, so check before arming.
        trap.finish()?;
        ring.arm(slot, this_len, submission);
    }
    for i in chunks.saturating_sub(slots)..chunks {
        ring.drain(engine, (i % slots) as usize, sink)?;
    }
    Ok(())
}

pub enum UploadSource<'a> {
    Bytes(&'a [u8]),
    File(&'a File),
}

/// Writes `len` bytes from `source` into `dst` at `dst_offset`, ordered after
/// all previously recorded GPU work.
pub fn upload(engine: &GpuEngine, dst: &wgpu::Buffer, dst_offset: u64, len: u64, source: UploadSource<'_>) -> Hc4jResult<()> {
    if !len.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) || !dst_offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
        return Err(Hc4jError::InvalidParam("upload range must be 4-byte aligned"));
    }
    if let UploadSource::Bytes(bytes) = source {
        if (bytes.len() as u64) < len {
            return Err(Hc4jError::InvalidParam("upload source shorter than requested length"));
        }
        if len <= SINGLE_SHOT_UPLOAD {
            let trap = ErrorTrap::push(&engine.device);
            engine.stream.write_buffer(dst, dst_offset, &bytes[..len as usize]);
            return trap.finish();
        }
    }

    let mut file_chunk: Vec<u8> = Vec::new();
    if matches!(source, UploadSource::File(_)) {
        let want = TRANSFER_CHUNK.min(len) as usize;
        file_chunk.try_reserve_exact(want).map_err(|_| Hc4jError::OutOfMemory)?;
        file_chunk.resize(want, 0);
    }

    let mut inflight: VecDeque<wgpu::SubmissionIndex> = VecDeque::with_capacity(UPLOAD_INFLIGHT + 1);
    let mut offset = 0u64;
    while offset < len {
        let this_len = TRANSFER_CHUNK.min(len - offset);
        let data: &[u8] = match &source {
            UploadSource::Bytes(bytes) => &bytes[offset as usize..(offset + this_len) as usize],
            UploadSource::File(file) => {
                let buf = &mut file_chunk[..this_len as usize];
                spill::read_exact_at(file, buf, offset)?;
                buf
            }
        };
        let trap = ErrorTrap::push(&engine.device);
        engine.stream.write_buffer(dst, dst_offset + offset, data);
        // Flush this chunk now so wgpu can recycle its staging copy once the
        // GPU consumes it, instead of accumulating the whole tensor.
        let submission = engine.stream.submit_empty();
        trap.finish()?;
        inflight.push_back(submission);
        if inflight.len() > UPLOAD_INFLIGHT
            && let Some(oldest) = inflight.pop_front()
        {
            engine.wait_for(oldest)?;
        }
        offset += this_len;
    }
    Ok(())
}
