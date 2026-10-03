//! The single submission point for all GPU work.
//!
//! Every `queue.submit` in HC4J goes through [`CommandStream`], which gives
//! the engine four properties:
//!
//! * **Batching.** Inside a batch scope ([`CommandStream::begin_batch`]), ops
//!   record their compute passes into one shared command encoder, submitted
//!   once at batch end (or every [`MAX_BATCH_PASSES`] passes). A loop of small
//!   ops then costs one submission instead of one per op.
//! * **Epochs.** Each submission gets a monotonically increasing epoch, and
//!   `Queue::on_submitted_work_done` publishes the highest completed one. The
//!   slab allocator quarantines freed regions until the epoch that last
//!   touched them retires, so a Java `close()` right after a dispatch can
//!   never hand memory the GPU is still reading to a new tensor.
//! * **Ordered zero-fills.** Zeroed slab regions get a `clear_buffer` that is
//!   encoded *before* the next recorded op, and flushed before any
//!   `queue.write_buffer`, so a clear can never land on top of written data.
//! * **Uniform ring.** Per-dispatch kernel parameters are packed into
//!   256-byte slots of one persistent uniform buffer and uploaded with a single
//!   `write_buffer` per submission, replacing a `create_buffer_init` (a driver
//!   allocation) per dispatch.
//!
//! Ordering rule: `queue.write_buffer` executes at the *start* of the next
//! submission, ahead of any command buffer in it. Writes that could race with
//! recorded-but-unsubmitted work therefore go through
//! [`CommandStream::write_buffer`], which flushes first.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::{Hc4jError, Hc4jResult};
use crate::{GPU_WAIT_TIMEOUT, lock_or_recover};

/// A batch is submitted early once it holds this many passes, bounding both
/// latency and the resources the open encoder keeps alive.
pub const MAX_BATCH_PASSES: u32 = 256;
/// Uniform slot size: the WebGPU maximum `min_uniform_buffer_offset_alignment`.
pub const UNIFORM_SLOT: u64 = 256;
pub const UNIFORM_SLOTS: u32 = 256;

struct PendingClear {
    buffer: wgpu::Buffer,
    offset: u64,
    size: u64,
}

#[derive(Default)]
struct StreamState {
    encoder: Option<wgpu::CommandEncoder>,
    /// The compute pass kept open on `encoder` across consecutive dispatches.
    /// wgpu-core does substantial work per pass at `finish()` (measured
    /// ~2.4 ms per pass for a 100-pass batch on Iris Xe / D3D12), so
    /// dispatches share one pass until an encoder-level command (copy,
    /// clear) or a flush needs the encoder back. Barriers between dispatches
    /// are still inserted from per-dispatch usage scopes.
    pass: Option<wgpu::ComputePass<'static>>,
    /// Dispatches recorded into `encoder` since it was opened.
    recorded: u32,
    batch_depth: u32,
    submitted_epoch: u64,
    pending_clears: Vec<PendingClear>,
    uniform_ring: Option<wgpu::Buffer>,
    /// First slot of the current upload window and the bytes staged for it.
    window_start: u32,
    window: Vec<u8>,
}

impl StreamState {
    fn window_slots(&self) -> u32 {
        (self.window.len() as u64 / UNIFORM_SLOT) as u32
    }

    fn has_work(&self) -> bool {
        self.encoder.is_some() || !self.pending_clears.is_empty() || !self.window.is_empty()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamStats {
    pub submissions: u64,
    pub dispatches: u64,
    /// Bytes bound to kernels as inputs plus outputs: the logical memory
    /// traffic the dispatched kernels must move.
    pub kernel_bytes: u64,
    pub batches: u64,
    pub completed_epoch: u64,
    pub submitted_epoch: u64,
}

#[derive(Default)]
struct Counters {
    submissions: AtomicU64,
    dispatches: AtomicU64,
    kernel_bytes: AtomicU64,
    batches: AtomicU64,
}

pub struct CommandStream {
    device: wgpu::Device,
    queue: wgpu::Queue,
    state: Mutex<StreamState>,
    completed_epoch: Arc<AtomicU64>,
    /// Mirrors of `state.submitted_epoch` and "work is pending", readable
    /// without the lock so `retire_epoch` can run from `Drop` impls in any
    /// context. `pending` may over-report (safe: reuse waits one extra
    /// submission) but never under-reports.
    submitted: AtomicU64,
    pending: AtomicBool,
    counters: Counters,
}

/// Handed to recording closures: dispatch recording, encoder access, and
/// uniform-slot allocation.
pub struct Recorder<'a> {
    encoder: &'a mut wgpu::CommandEncoder,
    pass: &'a mut Option<wgpu::ComputePass<'static>>,
    ring: &'a wgpu::Buffer,
    window: &'a mut Vec<u8>,
    window_start: u32,
    reserved: u32,
    used: u32,
    passes: u32,
    counters: &'a Counters,
}

impl<'a> Recorder<'a> {
    /// Stages `bytes` (at most 256) into the next reserved uniform slot and
    /// returns the binding for it.
    pub fn uniform(&mut self, bytes: &[u8]) -> Hc4jResult<wgpu::BindingResource<'a>> {
        if self.used >= self.reserved {
            return Err(Hc4jError::Device("uniform slot reservation exceeded".to_string()));
        }
        if bytes.is_empty() || bytes.len() as u64 > UNIFORM_SLOT || !bytes.len().is_multiple_of(4) {
            return Err(Hc4jError::InvalidParam("uniform payload must be 4..=256 bytes, 4-aligned"));
        }
        let slot = self.window_start + self.window_slots();
        let start = self.window.len();
        self.window.resize(start + UNIFORM_SLOT as usize, 0);
        self.window[start..start + bytes.len()].copy_from_slice(bytes);
        self.used += 1;
        Ok(wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.ring,
            offset: slot as u64 * UNIFORM_SLOT,
            size: wgpu::BufferSize::new(bytes.len().next_multiple_of(16) as u64),
        }))
    }

    fn window_slots(&self) -> u32 {
        (self.window.len() as u64 / UNIFORM_SLOT) as u32
    }

    /// The command encoder, for copies and clears. Ends the open compute pass
    /// first: the encoder is locked while a pass is being recorded.
    pub fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        *self.pass = None;
        self.encoder
    }

    /// Records one dispatch into the shared open compute pass (opening it if
    /// needed). `bytes_touched` is the input plus output bytes the kernel
    /// binds, for traffic accounting.
    pub fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        (x, y, z): (u32, u32, u32),
        bytes_touched: u64,
    ) {
        let Recorder { encoder, pass, .. } = self;
        let pass = pass.get_or_insert_with(|| {
            encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("HC4J Dispatch"),
                    timestamp_writes: None,
                })
                .forget_lifetime()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
        self.passes += 1;
        self.counters.dispatches.fetch_add(1, Ordering::Relaxed);
        self.counters.kernel_bytes.fetch_add(bytes_touched, Ordering::Relaxed);
    }
}

impl CommandStream {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            state: Mutex::new(StreamState::default()),
            completed_epoch: Arc::new(AtomicU64::new(0)),
            submitted: AtomicU64::new(0),
            pending: AtomicBool::new(false),
            counters: Counters::default(),
        }
    }

    fn uniform_ring(&self, state: &mut StreamState) -> wgpu::Buffer {
        state
            .uniform_ring
            .get_or_insert_with(|| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("HC4J Uniform Ring"),
                    size: UNIFORM_SLOT * UNIFORM_SLOTS as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .clone()
    }

    /// Submits everything pending: the open encoder, any unencoded clears and
    /// the staged uniform window.
    fn flush_locked(&self, state: &mut StreamState) -> Option<wgpu::SubmissionIndex> {
        if !state.has_work() {
            return None;
        }
        // End the shared compute pass: finish() needs an unlocked encoder.
        state.pass = None;
        let mut encoder = state.encoder.take().unwrap_or_else(|| self.new_encoder());
        // Clears queued after the last recorded op touch regions no recorded
        // command uses yet, so appending them here is safe.
        for clear in state.pending_clears.drain(..) {
            encoder.clear_buffer(&clear.buffer, clear.offset, Some(clear.size));
        }
        if !state.window.is_empty()
            && let Some(ring) = &state.uniform_ring
        {
            self.queue.write_buffer(ring, state.window_start as u64 * UNIFORM_SLOT, &state.window);
        }
        let index = self.queue.submit(Some(encoder.finish()));
        state.recorded = 0;
        // Every window restarts at slot 0: rewriting a slot is safe once the
        // submission that read it is queued, because the new data lands at
        // the start of a *later* submission. So any batch of up to
        // UNIFORM_SLOTS dispatches fits one window without a wrap flush.
        state.window_start = 0;
        state.window.clear();
        self.register_epoch(state);
        self.pending.store(false, Ordering::SeqCst);
        Some(index)
    }

    fn register_epoch(&self, state: &mut StreamState) {
        state.submitted_epoch += 1;
        let epoch = state.submitted_epoch;
        self.submitted.store(epoch, Ordering::SeqCst);
        let completed = Arc::clone(&self.completed_epoch);
        self.queue.on_submitted_work_done(move || {
            completed.fetch_max(epoch, Ordering::AcqRel);
        });
        self.counters.submissions.fetch_add(1, Ordering::Relaxed);
    }

    fn new_encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("HC4J Stream") })
    }

    /// Makes room for `slots` contiguous uniform slots in the current window,
    /// submitting the window first if it would wrap.
    fn reserve_slots(&self, state: &mut StreamState, slots: u32) -> Hc4jResult<()> {
        if slots > UNIFORM_SLOTS {
            return Err(Hc4jError::Unsupported("operation needs more uniform slots than the ring holds"));
        }
        if state.window_start + state.window_slots() + slots > UNIFORM_SLOTS {
            self.flush_locked(state);
            state.window_start = 0;
        }
        Ok(())
    }

    fn run_recorder<R>(
        &self,
        state: &mut StreamState,
        slots: u32,
        record: impl FnOnce(&mut Recorder<'_>) -> Hc4jResult<R>,
    ) -> Hc4jResult<R> {
        self.reserve_slots(state, slots)?;
        let ring = self.uniform_ring(state);
        let mut encoder = state.encoder.take().unwrap_or_else(|| self.new_encoder());
        // Zero-fills of freshly allocated regions must precede this op, and
        // need the encoder (not a pass).
        if !state.pending_clears.is_empty() {
            state.pass = None;
        }
        for clear in state.pending_clears.drain(..) {
            encoder.clear_buffer(&clear.buffer, clear.offset, Some(clear.size));
        }
        let window_start = state.window_start;
        let mut recorder = Recorder {
            encoder: &mut encoder,
            pass: &mut state.pass,
            ring: &ring,
            window: &mut state.window,
            window_start,
            reserved: slots,
            used: 0,
            passes: 0,
            counters: &self.counters,
        };
        let result = record(&mut recorder);
        let passes = recorder.passes;
        state.encoder = Some(encoder);
        state.recorded += passes;
        self.pending.store(true, Ordering::SeqCst);
        result
    }

    /// Records one op. Outside a batch it is submitted before returning (so
    /// the caller's error trap sees any submission error); inside a batch,
    /// submission is deferred to batch end.
    pub fn record<R>(&self, slots: u32, record: impl FnOnce(&mut Recorder<'_>) -> Hc4jResult<R>) -> Hc4jResult<R> {
        let mut state = lock_or_recover(&self.state);
        // On error the encoder is kept: passes recorded before the failure are
        // complete, and earlier batched ops must not be dropped.
        let result = self.run_recorder(&mut state, slots, record);
        if state.batch_depth == 0 || state.recorded >= MAX_BATCH_PASSES {
            self.flush_locked(&mut state);
        }
        result
    }

    /// Records and submits immediately, returning the submission index for
    /// code that must wait on or map its results (transfers, streaming).
    pub fn submit_now<R>(
        &self,
        slots: u32,
        record: impl FnOnce(&mut Recorder<'_>) -> Hc4jResult<R>,
    ) -> Hc4jResult<(R, wgpu::SubmissionIndex)> {
        let mut state = lock_or_recover(&self.state);
        // Earlier batched work must execute first.
        self.flush_locked(&mut state);
        let result = self.run_recorder(&mut state, slots, record);
        let index = self.flush_locked(&mut state);
        match (result, index) {
            (Ok(value), Some(index)) => Ok((value, index)),
            (Ok(_), None) => Err(Hc4jError::Device("nothing was submitted".to_string())),
            (Err(err), _) => Err(err),
        }
    }

    /// Submits any pending work. Returns the index of the submission, if any.
    pub fn flush(&self) -> Option<wgpu::SubmissionIndex> {
        let mut state = lock_or_recover(&self.state);
        self.flush_locked(&mut state)
    }

    /// Submits an empty command list after flushing, so `queue.write_buffer`
    /// data staged so far is handed to the GPU now. Returns its index.
    pub fn submit_empty(&self) -> wgpu::SubmissionIndex {
        let mut state = lock_or_recover(&self.state);
        if let Some(index) = self.flush_locked(&mut state) {
            return index;
        }
        let index = self.queue.submit(None);
        self.register_epoch(&mut state);
        index
    }

    /// `queue.write_buffer` ordered after all previously recorded work.
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        let mut state = lock_or_recover(&self.state);
        self.flush_locked(&mut state);
        self.queue.write_buffer(buffer, offset, data);
    }

    /// Queues a zero-fill that will be encoded before the next recorded op.
    pub fn queue_clear(&self, buffer: wgpu::Buffer, offset: u64, size: u64) {
        let mut state = lock_or_recover(&self.state);
        state.pending_clears.push(PendingClear { buffer, offset, size });
        self.pending.store(true, Ordering::SeqCst);
    }

    /// Drops a not-yet-encoded zero-fill (the caller is about to overwrite
    /// the whole region). Returns whether one was pending.
    pub fn cancel_clear(&self, buffer: &wgpu::Buffer, offset: u64) -> bool {
        let mut state = lock_or_recover(&self.state);
        let before = state.pending_clears.len();
        state.pending_clears.retain(|c| !(c.offset == offset && &c.buffer == buffer));
        state.pending_clears.len() != before
    }

    pub fn begin_batch(&self) {
        lock_or_recover(&self.state).batch_depth += 1;
    }

    /// Closes a batch scope; the outermost one submits the batch. Errors
    /// raised while submitting are returned here, since batched ops return
    /// before their work is validated by submission.
    pub fn end_batch(&self) -> Hc4jResult<()> {
        let mut state = lock_or_recover(&self.state);
        if state.batch_depth == 0 {
            return Err(Hc4jError::InvalidParam("end_batch without begin_batch"));
        }
        state.batch_depth -= 1;
        if state.batch_depth > 0 {
            return Ok(());
        }
        self.counters.batches.fetch_add(1, Ordering::Relaxed);
        let trap = crate::ErrorTrap::push(&self.device);
        self.flush_locked(&mut state);
        drop(state);
        trap.finish()
    }

    /// The epoch of the submission that will contain everything recorded so
    /// far: resources released now are safe to reuse once it retires.
    pub fn retire_epoch(&self) -> u64 {
        // Read `pending` first: a flush clears it only after bumping
        // `submitted`, so this order can over- but never under-estimate.
        let pending = self.pending.load(Ordering::SeqCst);
        self.submitted.load(Ordering::SeqCst) + u64::from(pending)
    }

    /// Highest epoch known complete. Runs pending completion callbacks first.
    pub fn completed_epoch(&self) -> u64 {
        let _ = self.device.poll(wgpu::PollType::Poll);
        self.completed_epoch.load(Ordering::Acquire)
    }

    /// Blocks until `epoch` has retired, flushing first if it covers work
    /// that is still only recorded.
    pub fn wait_epoch(&self, epoch: u64) -> Hc4jResult<()> {
        if self.completed_epoch() >= epoch {
            return Ok(());
        }
        let submitted = {
            let mut state = lock_or_recover(&self.state);
            if state.submitted_epoch < epoch {
                self.flush_locked(&mut state);
            }
            state.submitted_epoch
        };
        if submitted < epoch {
            return Err(Hc4jError::Device(format!("epoch {epoch} was never submitted")));
        }
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(GPU_WAIT_TIMEOUT),
            })
            .map_err(|e| Hc4jError::Device(format!("GPU wait failed: {e}")))?;
        if self.completed_epoch.load(Ordering::Acquire) >= epoch {
            Ok(())
        } else {
            Err(Hc4jError::Device(format!("epoch {epoch} did not retire")))
        }
    }

    pub fn stats(&self) -> StreamStats {
        let state = lock_or_recover(&self.state);
        StreamStats {
            submissions: self.counters.submissions.load(Ordering::Relaxed),
            dispatches: self.counters.dispatches.load(Ordering::Relaxed),
            kernel_bytes: self.counters.kernel_bytes.load(Ordering::Relaxed),
            batches: self.counters.batches.load(Ordering::Relaxed),
            completed_epoch: self.completed_epoch.load(Ordering::Acquire),
            submitted_epoch: state.submitted_epoch,
        }
    }
}
