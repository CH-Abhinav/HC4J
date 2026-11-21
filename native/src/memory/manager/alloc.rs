use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::error::{Hc4jError, Hc4jResult};
use crate::memory::transfer::{self, UploadSource};
use crate::memory::TensorId;
use crate::ErrorTrap;

use super::types::{
    align_up, try_zeroed, unshare, Backing, DeviceBlock, DeviceSpan, HostBlock, PinGuard, Residency,
    Snapshot, SpillFile, TensorAllocation, RegionAlloc, MIB, MIN_SLAB, MAX_SLAB, TENSOR_USAGE,
};
use super::TieredMemoryManager;

impl TieredMemoryManager {
    // ------------------------------------------------------------------
    // Lookup and registration
    // ------------------------------------------------------------------

    pub fn size_of(&self, id: TensorId) -> Hc4jResult<u64> {
        self.lock_inner()
            .allocs
            .get(&id)
            .filter(|a| !a.free_pending)
            .map(|a| a.size_bytes)
            .ok_or(Hc4jError::NotFound)
    }

    pub fn residency_code(&self, id: TensorId) -> Hc4jResult<i32> {
        let inner = self.wait_settled(id)?;
        inner.allocs.get(&id).map(|a| a.residency.code()).ok_or(Hc4jError::NotFound)
    }

    /// Adds a tensor whose storage the caller already built.
    pub fn register(&self, size_bytes: u64, residency: Residency) -> TensorId {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut inner = self.lock_inner();
        let tick = inner.next_tick();
        inner.allocs.insert(
            id,
            TensorAllocation::new(id, size_bytes, Residency::InTransit, tick),
        );
        let _ = inner.settle(id, residency);
        id
    }

    /// Swaps a tensor's storage for `residency`, e.g. after an op streamed a
    /// new result for a caller-allocated output. The old storage is dropped.
    pub fn replace_residency(&self, id: TensorId, residency: Residency) -> Hc4jResult<()> {
        let old = {
            let mut inner = self.wait_settled(id)?;
            let (old, _) = inner.take_for_transit(id).ok_or(Hc4jError::NotFound)?;
            (old, inner.settle(id, residency))
        };
        self.settled.notify_all();
        drop(old);
        Ok(())
    }

    pub fn free(&self, id: TensorId) -> Hc4jResult<()> {
        let removed = {
            let mut inner = self.wait_settled(id)?;
            let entry = inner.allocs.get_mut(&id).ok_or(Hc4jError::NotFound)?;
            if entry.pins > 0 {
                // In use by an in-flight op on another thread: the last unpin
                // frees it. Until then lookups report NotFound.
                entry.free_pending = true;
                return Ok(());
            }
            inner.remove_entry(id)
        };
        // Dropped outside the lock: freeing a large host Vec is slow, and a
        // slab region's release reads the stream epoch.
        drop(removed);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Allocation with fallback
    // ------------------------------------------------------------------

    /// Allocates a tensor: VRAM if it fits (evicting LRU tensors if needed),
    /// otherwise host RAM, otherwise a spill file. `zeroed` guarantees zero
    /// contents; op outputs that are fully overwritten skip it.
    pub fn allocate(&self, size_bytes: u64, zeroed: bool) -> Hc4jResult<TensorId> {
        self.counters.allocations.fetch_add(1, Ordering::Relaxed);
        let residency = match self.allocate_device(size_bytes, zeroed) {
            Ok(block) => Residency::DeviceResident(block),
            Err(Hc4jError::OutOfMemory) => {
                crate::hc4j_trace!("alloc {size_bytes} B: VRAM exhausted, placing tensor off-device");
                self.allocate_off_device(size_bytes)?
            }
            Err(err) => return Err(err),
        };
        Ok(self.register(size_bytes, residency))
    }

    pub(crate) fn allocate_off_device(&self, size_bytes: u64) -> Hc4jResult<Residency> {
        if let Some(reservation) = self.reserve_host(size_bytes)?
            && let Some(data) = try_zeroed(size_bytes)
        {
            return Ok(Residency::HostResident(HostBlock::new(data, reservation)));
        }
        Ok(Residency::Evicted(SpillFile::create(&self.spill_dir, size_bytes)?))
    }

    /// Slab capacity for the current budget: an eighth of it, clamped to
    /// 4..64 MiB and to what one storage binding can address.
    pub(crate) fn slab_size(&self) -> u64 {
        let limits = &self.engine.limits;
        let align = self.pool.align();
        let target = (self.vram.limit() / 8).clamp(MIN_SLAB, MAX_SLAB) / MIB * MIB;
        let cap = target.min(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        cap / align * align
    }

    /// Tensors up to this size are slab regions; larger ones are dedicated.
    pub(crate) fn region_max(&self) -> u64 {
        self.slab_size() / 4
    }

    /// The fallback allocator. Tries, in order: a region in an existing slab,
    /// a new slab, a dedicated buffer; then lets quarantined regions retire,
    /// releases spare slabs, and finally evicts one LRU tensor and retries.
    /// Returns `OutOfMemory` only when nothing evictable is left.
    pub fn allocate_device(&self, size_bytes: u64, zeroed: bool) -> Hc4jResult<DeviceBlock> {
        let size = align_up(size_bytes, wgpu::COPY_BUFFER_ALIGNMENT)
            .filter(|&s| s > 0)
            .ok_or(Hc4jError::InvalidParam("allocation size"))?;
        // Larger than the device can ever address: don't evict the working set
        // chasing an allocation that can never succeed.
        if size > self.engine.limits.max_buffer_size || size > self.vram.limit() {
            return Err(Hc4jError::OutOfMemory);
        }

        let mut driver_ooms = 0u32;
        loop {
            self.reclaim_slabs(false);
            if size <= self.region_max() {
                if let Some(block) = self.alloc_region(size_bytes, zeroed) {
                    return Ok(block);
                }
                match self.grow_slabs(size_bytes, zeroed) {
                    Ok(Some(block)) => return Ok(block),
                    Ok(None) => {}
                    Err(Hc4jError::OutOfMemory) => self.on_driver_oom(&mut driver_ooms, self.slab_size())?,
                    Err(err) => return Err(err),
                }
            }
            if let Some(reservation) = self.vram.try_reserve(size) {
                match self.create_buffer(size, "HC4J Tensor") {
                    Ok(buffer) => {
                        self.counters.dedicated_buffers.fetch_add(1, Ordering::Relaxed);
                        return Ok(DeviceBlock::new(
                            DeviceSpan { buffer, offset: 0, size: size_bytes },
                            Backing::Dedicated { _reservation: reservation },
                        ));
                    }
                    Err(Hc4jError::OutOfMemory) => {
                        drop(reservation);
                        self.on_driver_oom(&mut driver_ooms, size)?;
                    }
                    Err(err) => return Err(err),
                }
            }
            if self.drain_quarantine()? || self.release_spare_slabs() {
                continue;
            }
            if !self.evict_one()? {
                return Err(Hc4jError::OutOfMemory);
            }
        }
    }

    fn alloc_region(&self, size_bytes: u64, zeroed: bool) -> Option<DeviceBlock> {
        let region = self.pool.alloc(size_bytes)?;
        Some(self.region_block(region, size_bytes, zeroed))
    }

    fn region_block(&self, region: RegionAlloc, size_bytes: u64, zeroed: bool) -> DeviceBlock {
        self.counters.region_allocs.fetch_add(1, Ordering::Relaxed);
        if zeroed {
            // Reused ranges hold stale data (wgpu zero-initializes whole
            // buffers once, not sub-ranges), so zeroed regions get a clear
            // ordered before their first use.
            self.engine.stream.queue_clear(region.buffer.clone(), region.offset, region.rounded);
        }
        DeviceBlock::new(
            DeviceSpan {
                buffer: region.buffer,
                offset: region.offset,
                size: size_bytes,
            },
            Backing::Region {
                pool: Arc::clone(&self.pool),
                slab: region.slab,
                rounded: region.rounded,
                stream: &self.engine.stream,
            },
        )
    }

    /// Creates a new slab if the budget can hold one. `Ok(None)`: it cannot.
    fn grow_slabs(&self, size_bytes: u64, zeroed: bool) -> Hc4jResult<Option<DeviceBlock>> {
        let capacity = self.slab_size();
        if capacity < size_bytes {
            return Ok(None);
        }
        let Some(reservation) = self.vram.try_reserve(capacity) else {
            return Ok(None);
        };
        let buffer = self.create_buffer(capacity, "HC4J Slab")?;
        self.counters.slabs_created.fetch_add(1, Ordering::Relaxed);
        let region = self
            .pool
            .insert_with_region(buffer, capacity, reservation, size_bytes)
            .ok_or_else(|| Hc4jError::Device("fresh slab could not hold its first region".to_string()))?;
        Ok(Some(self.region_block(region, size_bytes, zeroed)))
    }

    /// Creates a buffer with OOM/validation errors trapped instead of panicking
    /// through wgpu's default handler.
    pub(crate) fn create_buffer(&self, size: u64, label: &str) -> Hc4jResult<wgpu::Buffer> {
        let trap = ErrorTrap::push(&self.engine.device);
        let buffer = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: TENSOR_USAGE,
            mapped_at_creation: false,
        });
        // On error `buffer` is an invalid handle and is simply dropped.
        trap.finish()?;
        Ok(buffer)
    }

    /// Returns retired quarantined regions to their slabs; drops surplus
    /// empty slabs (all of them under `pressure`).
    pub(crate) fn reclaim_slabs(&self, pressure: bool) -> usize {
        let released = self.pool.reclaim(self.engine.stream.completed_epoch(), pressure);
        let count = released.len();
        drop(released);
        count
    }

    pub(crate) fn release_spare_slabs(&self) -> bool {
        self.reclaim_slabs(true) > 0
    }

    /// Waits for quarantined regions to retire. Returns whether it waited.
    pub(crate) fn drain_quarantine(&self) -> Hc4jResult<bool> {
        let Some(epoch) = self.pool.newest_quarantine_epoch() else {
            return Ok(false);
        };
        let before = self.pool.quarantined();
        self.engine.stream.wait_epoch(epoch)?;
        self.reclaim_slabs(false);
        // Progress only if regions actually came back, so callers never spin.
        Ok(self.pool.quarantined() < before)
    }

    // ------------------------------------------------------------------
    // Host <-> tensor data movement (FFI write / download)
    // ------------------------------------------------------------------

    pub fn write(&self, id: TensorId, data: &[u8]) -> Hc4jResult<()> {
        enum Plan {
            Device(DeviceSpan),
            OffDevice(Residency),
        }
        let plan = {
            let mut inner = self.wait_settled(id)?;
            let size = inner.allocs.get(&id).map(|a| a.size_bytes).ok_or(Hc4jError::NotFound)?;
            if data.len() as u64 > size {
                return Err(Hc4jError::InvalidParam("write larger than tensor"));
            }
            inner.touch(id);
            let entry = inner.allocs.get_mut(&id).ok_or(Hc4jError::NotFound)?;
            if let Residency::DeviceResident(block) = &entry.residency {
                entry.pins += 1;
                Plan::Device(block.span().clone())
            } else {
                Plan::OffDevice(inner.take_for_transit(id).ok_or(Hc4jError::NotFound)?.0)
            }
        };

        match plan {
            Plan::Device(span) => {
                let _pin = PinGuard::new(self, vec![id]);
                if data.len() as u64 == span.size {
                    // Full overwrite: a pending zero-fill would be wasted work.
                    self.engine.stream.cancel_clear(&span.buffer, span.offset);
                }
                transfer::upload(self.engine, &span.buffer, span.offset, data.len() as u64, UploadSource::Bytes(data))
            }
            Plan::OffDevice(taken) => {
                let (residency, result) = self.write_off_device(taken, data);
                let orphan = self.lock_inner().settle(id, residency);
                self.settled.notify_all();
                drop(orphan);
                result
            }
        }
    }

    fn write_off_device(&self, taken: Residency, data: &[u8]) -> (Residency, Hc4jResult<()>) {
        match taken {
            Residency::HostResident(mut host) => {
                let result = unshare(&mut host.data).and_then(|bytes| {
                    bytes
                        .get_mut(..data.len())
                        .ok_or(Hc4jError::InvalidParam("write larger than tensor"))?
                        .copy_from_slice(data);
                    Ok(())
                });
                (Residency::HostResident(host), result)
            }
            Residency::Evicted(file) => match file.rewrite_prefix(&self.spill_dir, data) {
                Ok(next) => (Residency::Evicted(next), Ok(())),
                Err(err) => (Residency::Evicted(file), Err(err.into())),
            },
            other => (other, Err(Hc4jError::Device("write to a tensor in an invalid residency".to_string()))),
        }
    }

    pub fn read(&self, id: TensorId, out: &mut [u8]) -> Hc4jResult<()> {
        let (snapshot, size, _pin) = self.pin_snapshot(id)?;
        if out.len() as u64 > size {
            return Err(Hc4jError::InvalidParam("read larger than tensor"));
        }
        match snapshot {
            Snapshot::Device(span) => {
                let mut filled = 0usize;
                self.with_ring(|ring| {
                    transfer::download(self.engine, ring, &span.buffer, span.offset, out.len() as u64, &mut |chunk: &[u8]| {
                        out.get_mut(filled..filled + chunk.len())
                            .ok_or(Hc4jError::Readback("readback overran destination".to_string()))?
                            .copy_from_slice(chunk);
                        filled += chunk.len();
                        Ok(())
                    })
                })
            }
            Snapshot::Host(data) => {
                let src = data
                    .get(..out.len())
                    .ok_or(Hc4jError::Readback("host copy shorter than tensor".to_string()))?;
                out.copy_from_slice(src);
                Ok(())
            }
            Snapshot::Disk(file) => Ok(crate::memory::spill::read_exact_at(&file, out, 0)?),
        }
    }
}
