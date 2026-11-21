use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::error::{Hc4jError, Hc4jResult};
use crate::memory::budget::Reservation;
use crate::memory::transfer::{self, UploadSource};
use crate::memory::TensorId;

use super::types::{
    try_with_capacity, DeviceBlock, DeviceSpan, HostBlock, PinGuard, Residency, ResidentSet, Snapshot,
    SpillFile, MAX_DRIVER_OOM_RETRIES, MIN_VRAM_BUDGET,
};
use super::TieredMemoryManager;

impl TieredMemoryManager {
    // ------------------------------------------------------------------
    // Driver OOM clamping
    // ------------------------------------------------------------------

    pub(crate) fn on_driver_oom(&self, driver_ooms: &mut u32, size: u64) -> Hc4jResult<()> {
        *driver_ooms += 1;
        self.counters.driver_ooms.fetch_add(1, Ordering::Relaxed);
        let committed = self.vram.used();
        let limit = self.vram.clamp_limit(committed.max(MIN_VRAM_BUDGET));
        eprintln!("[HC4J] driver OOM allocating {size} B with {committed} B committed; VRAM budget clamped to {limit} B");
        if *driver_ooms > MAX_DRIVER_OOM_RETRIES {
            return Err(Hc4jError::OutOfMemory);
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // LRU eviction: VRAM -> host RAM (-> disk)
    // ------------------------------------------------------------------

    /// Demotes the least recently used unpinned device tensor. Returns
    /// `false` if there is none.
    pub(crate) fn evict_one(&self) -> Hc4jResult<bool> {
        let (id, block) = {
            let mut inner = self.lock_inner();
            let Some(id) = inner.oldest_unpinned(true) else {
                return Ok(false);
            };
            match inner.take_for_transit(id) {
                Some((Residency::DeviceResident(block), _)) => (id, block),
                Some((other, _)) => {
                    let _ = inner.settle(id, other);
                    return Ok(false);
                }
                None => return Ok(false),
            }
        };
        self.demote(id, block)?;
        Ok(true)
    }

    /// Evicts until `size` more bytes fit in the VRAM budget. Returns `false`
    /// if the budget cannot be met even after evicting everything unpinned.
    pub(crate) fn make_vram_room(&self, size: u64) -> Hc4jResult<bool> {
        loop {
            self.reclaim_slabs(true);
            if self.vram.used().saturating_add(size) <= self.vram.limit() {
                return Ok(true);
            }
            if self.drain_quarantine()? {
                continue;
            }
            if !self.evict_one()? {
                return Ok(false);
            }
        }
    }

    /// Explicitly pages `id` out of VRAM. Returns `false` if it was not
    /// device-resident or is pinned by an in-flight operation.
    pub fn evict(&self, id: TensorId) -> Hc4jResult<bool> {
        let block = {
            let mut inner = self.wait_settled(id)?;
            let eligible = inner
                .allocs
                .get(&id)
                .is_some_and(|a| a.pins == 0 && matches!(a.residency, Residency::DeviceResident(_)));
            if !eligible {
                return Ok(false);
            }
            match inner.take_for_transit(id) {
                Some((Residency::DeviceResident(block), _)) => block,
                Some((other, _)) => {
                    let _ = inner.settle(id, other);
                    return Ok(false);
                }
                None => return Err(Hc4jError::NotFound),
            }
        };
        self.demote(id, block)?;
        Ok(true)
    }

    /// Copies a device tensor down and installs it in the host or disk tier.
    /// On failure the tensor stays device-resident.
    fn demote(&self, id: TensorId, block: DeviceBlock) -> Hc4jResult<()> {
        let outcome = self.copy_out(block.span());
        let (result, leftover) = {
            let mut inner = self.lock_inner();
            match outcome {
                Ok(residency) => {
                    crate::hc4j_trace!("evicted tensor {id} ({} B) to tier {}", block.size(), residency.code());
                    self.counters.evictions.fetch_add(1, Ordering::Relaxed);
                    (Ok(()), (Some(block), inner.settle(id, residency)))
                }
                Err(err) => (Err(err), (None, inner.settle(id, Residency::DeviceResident(block)))),
            }
        };
        self.settled.notify_all();
        // The old device storage (budget or slab region) and any orphan are
        // released outside the lock.
        drop(leftover);
        result
    }

    fn copy_out(&self, src: &DeviceSpan) -> Hc4jResult<Residency> {
        let size = src.size;
        if let Some(reservation) = self.reserve_host(size)?
            && let Some(mut data) = try_with_capacity(size)
        {
            self.with_ring(|ring| {
                transfer::download(self.engine, ring, &src.buffer, src.offset, size, &mut |chunk: &[u8]| {
                    data.extend_from_slice(chunk);
                    Ok(())
                })
            })?;
            return Ok(Residency::HostResident(HostBlock::new(data, reservation)));
        }
        // Host tier full (or the allocator refused): stream straight to disk.
        let file = SpillFile::create(&self.spill_dir, size)?;
        let mut offset = 0u64;
        self.with_ring(|ring| {
            transfer::download(self.engine, ring, &src.buffer, src.offset, size, &mut |chunk: &[u8]| {
                file.write_all_at(chunk, offset)?;
                offset += chunk.len() as u64;
                Ok(())
            })
        })?;
        self.counters.spills.fetch_add(1, Ordering::Relaxed);
        Ok(Residency::Evicted(file))
    }

    // ------------------------------------------------------------------
    // Host tier: reservation with spill-to-disk
    // ------------------------------------------------------------------

    /// Reserves `size` bytes of host budget, spilling LRU host tensors to disk
    /// to make room. `None` means the host tier cannot take it; callers fall
    /// back to the disk tier.
    pub fn reserve_host(&self, size: u64) -> Hc4jResult<Option<Reservation>> {
        loop {
            if let Some(reservation) = self.host.try_reserve(size) {
                return Ok(Some(reservation));
            }
            if !self.make_host_room(size)? {
                return Ok(None);
            }
        }
    }

    pub(crate) fn make_host_room(&self, size: u64) -> Hc4jResult<bool> {
        loop {
            let (id, block) = {
                let mut inner = self.lock_inner();
                let limit = self.host.limit();
                let used = self.host.used();
                if used.saturating_add(size) <= limit {
                    return Ok(true);
                }
                let evictable = inner.evictable_bytes(false);
                if (used - evictable.min(used)).saturating_add(size) > limit {
                    return Ok(false);
                }
                let Some(id) = inner.oldest_unpinned(false) else {
                    return Ok(false);
                };
                match inner.take_for_transit(id) {
                    Some((Residency::HostResident(block), _)) => (id, block),
                    Some((other, _)) => {
                        let _ = inner.settle(id, other);
                        return Ok(false);
                    }
                    None => return Ok(false),
                }
            };
            self.spill(id, block)?;
        }
    }

    fn spill(&self, id: TensorId, block: HostBlock) -> Hc4jResult<()> {
        let outcome = SpillFile::create(&self.spill_dir, block.data.len() as u64).and_then(|file| {
            file.write_all_at(&block.data, 0)?;
            Ok(file)
        });
        let (result, leftover) = {
            let mut inner = self.lock_inner();
            match outcome {
                Ok(file) => {
                    crate::hc4j_trace!("spilled tensor {id} ({} B) to disk", block.data.len());
                    self.counters.spills.fetch_add(1, Ordering::Relaxed);
                    (Ok(()), (Some(block), inner.settle(id, Residency::Evicted(file))))
                }
                Err(err) => (Err(err.into()), (None, inner.settle(id, Residency::HostResident(block)))),
            }
        };
        self.settled.notify_all();
        drop(leftover);
        result
    }

    // ------------------------------------------------------------------
    // Page-in and pinning
    // ------------------------------------------------------------------

    /// Pins every tensor in `ids` and makes it VRAM-resident, paging in from
    /// host or disk as needed (which may evict other, unpinned tensors). The
    /// pins hold until the returned set is dropped, so spans stay valid
    /// through recording. Fails with `OutOfMemory` if the set cannot be
    /// co-resident; callers then take the streaming path.
    pub fn acquire_resident(&self, ids: &[TensorId]) -> Hc4jResult<ResidentSet<'_>> {
        // Sorted, deduplicated pin order: a.add(a) pins once, and concurrent
        // acquirers visit shared tensors in the same order.
        let mut unique = ids.to_vec();
        unique.sort_unstable();
        unique.dedup();

        let mut pins = PinGuard::new(self, Vec::with_capacity(unique.len()));
        let mut resolved = Vec::with_capacity(unique.len());
        for id in unique {
            let span = self.pin_resident(id)?;
            pins.ids.push(id);
            resolved.push((id, span));
        }
        let spans = ids
            .iter()
            .map(|id| {
                resolved
                    .iter()
                    .find(|(r, _)| r == id)
                    .map(|(_, s)| s.clone())
                    .ok_or(Hc4jError::NotFound)
            })
            .collect::<Hc4jResult<Vec<_>>>()?;
        Ok(ResidentSet { spans, _pins: pins })
    }

    /// Pins `id` and returns its device span, paging it in if needed. On
    /// error the pin is released before returning.
    fn pin_resident(&self, id: TensorId) -> Hc4jResult<DeviceSpan> {
        let (taken, size) = {
            let mut inner = self.wait_settled(id)?;
            inner.touch(id);
            let entry = inner.allocs.get_mut(&id).ok_or(Hc4jError::NotFound)?;
            entry.pins += 1;
            if let Residency::DeviceResident(block) = &entry.residency {
                return Ok(block.span().clone());
            }
            inner.take_for_transit(id).ok_or(Hc4jError::NotFound)?
        };

        match self.promote(taken, size) {
            Ok((block, previous)) => {
                let span = block.span().clone();
                let orphan = self.lock_inner().settle(id, Residency::DeviceResident(block));
                self.settled.notify_all();
                self.counters.page_ins.fetch_add(1, Ordering::Relaxed);
                crate::hc4j_trace!("paged tensor {id} ({size} B) into VRAM");
                // Releases the host copy / closes (and deletes) the spill file.
                drop((previous, orphan));
                Ok(span)
            }
            Err((err, previous)) => {
                {
                    let mut inner = self.lock_inner();
                    let _ = inner.settle(id, previous);
                    if let Some(entry) = inner.allocs.get_mut(&id) {
                        entry.pins = entry.pins.saturating_sub(1);
                    }
                }
                self.settled.notify_all();
                Err(err)
            }
        }
    }

    /// Allocates VRAM for `taken` and uploads it. Hands `taken` back on both
    /// paths: on success the caller drops it, on failure it is restored.
    fn promote(&self, taken: Residency, size: u64) -> Result<(DeviceBlock, Residency), (Hc4jError, Residency)> {
        let block = match self.allocate_device(size, false) {
            Ok(block) => block,
            Err(err) => return Err((err, taken)),
        };
        let dst = block.span();
        let uploaded = match &taken {
            Residency::HostResident(host) => {
                transfer::upload(self.engine, &dst.buffer, dst.offset, size, UploadSource::Bytes(&host.data))
            }
            Residency::Evicted(file) => {
                let handle = file.handle();
                transfer::upload(self.engine, &dst.buffer, dst.offset, size, UploadSource::File(&handle))
            }
            Residency::DeviceResident(_) | Residency::InTransit => {
                Err(Hc4jError::Device("promote from an invalid residency".to_string()))
            }
        };
        match uploaded {
            Ok(()) => Ok((block, taken)),
            Err(err) => Err((err, taken)),
        }
    }

    /// Pins `id` wherever it lives and returns a lock-free view of its bytes
    /// plus its size. Used by the streaming executor, which reads tensors too
    /// large to page in.
    pub fn pin_snapshot(&self, id: TensorId) -> Hc4jResult<(Snapshot, u64, PinGuard<'_>)> {
        let mut inner = self.wait_settled(id)?;
        inner.touch(id);
        let entry = inner.allocs.get_mut(&id).ok_or(Hc4jError::NotFound)?;
        let snapshot = match &entry.residency {
            Residency::DeviceResident(block) => Snapshot::Device(block.span().clone()),
            Residency::HostResident(host) => Snapshot::Host(Arc::clone(&host.data)),
            Residency::Evicted(file) => Snapshot::Disk(file.handle()),
            Residency::InTransit => return Err(Hc4jError::Device("snapshot of a tensor in transit".to_string())),
        };
        entry.pins += 1;
        let size = entry.size_bytes;
        drop(inner);
        Ok((snapshot, size, PinGuard::new(self, vec![id])))
    }
}
