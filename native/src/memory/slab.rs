//! Slab sub-allocation: small tensors live as aligned sub-ranges of a few
//! large buffers, so allocating one is a BTreeMap lookup instead of a
//! `create_buffer` driver call.
//!
//! * Each slab holds a VRAM budget [`Reservation`] for its full capacity, so
//!   the budget tracks physical VRAM, not just live tensor bytes.
//! * Freed regions are quarantined until the submission epoch current at
//!   free time has retired ([`crate::stream::CommandStream::retire_epoch`]).
//!   Only then is the range handed to a new tensor.
//! * Kernels declare every storage binding `read_write`: wgpu forbids one
//!   buffer being bound read-only and read-write in the same dispatch, and
//!   two tensors from one slab share a buffer.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Mutex;

use super::budget::Reservation;
use crate::lock_or_recover;

/// Best-fit sub-allocator over one slab's byte range. O(log n) alloc/free,
/// immediate coalescing of adjacent free ranges.
#[derive(Debug)]
pub struct RangeAllocator {
    capacity: u64,
    align: u64,
    /// Free ranges keyed by offset, for coalescing with neighbours.
    by_offset: BTreeMap<u64, u64>,
    /// Free ranges keyed by (len, offset), for best-fit lookup.
    by_size: BTreeSet<(u64, u64)>,
    allocated: u64,
}

impl RangeAllocator {
    pub fn new(capacity: u64, align: u64) -> Self {
        let align = align.max(4).next_power_of_two();
        let capacity = capacity / align * align;
        let mut ranges = Self {
            capacity,
            align,
            by_offset: BTreeMap::new(),
            by_size: BTreeSet::new(),
            allocated: 0,
        };
        if capacity > 0 {
            ranges.insert_free(0, capacity);
        }
        ranges
    }

    /// Size actually consumed by a request of `size` bytes.
    pub fn rounded(&self, size: u64) -> Option<u64> {
        let size = size.max(1).checked_add(self.align - 1)?;
        Some(size / self.align * self.align)
    }

    /// Returns the offset of a `size`-byte range aligned to `align`.
    pub fn alloc(&mut self, size: u64) -> Option<u64> {
        let size = self.rounded(size)?;
        let &(len, offset) = self.by_size.range((size, 0)..).next()?;
        self.remove_free(offset, len);
        if len > size {
            self.insert_free(offset + size, len - size);
        }
        self.allocated += size;
        Some(offset)
    }

    /// Frees a range from `alloc`. Returns `false` and changes nothing if the
    /// range overlaps free space (double free) or lies outside the slab.
    pub fn free(&mut self, offset: u64, size: u64) -> bool {
        let Some(size) = self.rounded(size) else { return false };
        let Some(end) = offset.checked_add(size) else { return false };
        if !offset.is_multiple_of(self.align) || end > self.capacity {
            return false;
        }
        // Free ranges are disjoint and sorted, so only the last one starting
        // before `end` can overlap [offset, end).
        if let Some((&start, &len)) = self.by_offset.range(..end).next_back()
            && start + len > offset
        {
            return false;
        }

        let (mut start, mut len) = (offset, size);
        if let Some((&prev, &prev_len)) = self.by_offset.range(..offset).next_back()
            && prev + prev_len == offset
        {
            self.remove_free(prev, prev_len);
            start = prev;
            len += prev_len;
        }
        if let Some(&next_len) = self.by_offset.get(&end) {
            self.remove_free(end, next_len);
            len += next_len;
        }
        self.insert_free(start, len);
        self.allocated = self.allocated.saturating_sub(size);
        true
    }

    pub fn allocated(&self) -> u64 {
        self.allocated
    }

    pub fn largest_free(&self) -> u64 {
        self.by_size.last().map_or(0, |&(len, _)| len)
    }

    /// 0.0 when all free space is one run; approaches 1.0 as it shatters.
    pub fn fragmentation(&self) -> f64 {
        let free = self.capacity - self.allocated;
        if free == 0 { 0.0 } else { 1.0 - self.largest_free() as f64 / free as f64 }
    }

    fn insert_free(&mut self, offset: u64, len: u64) {
        self.by_offset.insert(offset, len);
        self.by_size.insert((len, offset));
    }

    fn remove_free(&mut self, offset: u64, len: u64) {
        self.by_offset.remove(&offset);
        self.by_size.remove(&(len, offset));
    }
}

/// One slab. Dropping it releases the buffer and its budget reservation.
pub struct Slab {
    buffer: wgpu::Buffer,
    ranges: RangeAllocator,
    /// Regions handed out and not yet reclaimed (quarantined ones included).
    live: u32,
    _reservation: Reservation,
}

/// A region handed out by [`SlabPool::alloc`].
pub struct RegionAlloc {
    pub slab: u32,
    pub buffer: wgpu::Buffer,
    pub offset: u64,
    /// Bytes consumed in the slab (the request rounded up to the alignment).
    pub rounded: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SlabStats {
    pub slabs: u64,
    pub slab_bytes: u64,
    pub live_regions: u64,
    pub quarantined: u64,
}

struct Quarantined {
    epoch: u64,
    slab: u32,
    offset: u64,
    rounded: u64,
}

#[derive(Default)]
struct PoolInner {
    slabs: BTreeMap<u32, Slab>,
    quarantine: VecDeque<Quarantined>,
    next_id: u32,
}

/// Leaf lock: nothing is called while holding it, so dropping a region from
/// any context (including under the manager's table lock) cannot deadlock.
pub struct SlabPool {
    inner: Mutex<PoolInner>,
    align: u64,
}

impl SlabPool {
    pub fn new(align: u64) -> Self {
        Self {
            inner: Mutex::new(PoolInner::default()),
            align: align.max(4).next_power_of_two(),
        }
    }

    pub fn align(&self) -> u64 {
        self.align
    }

    /// Best fit across slabs: the slab whose largest hole is the smallest one
    /// that fits, which keeps other slabs empty enough to be released.
    pub fn alloc(&self, size: u64) -> Option<RegionAlloc> {
        let mut inner = lock_or_recover(&self.inner);
        let (&id, _) = inner
            .slabs
            .iter()
            .filter(|(_, s)| s.ranges.rounded(size).is_some_and(|r| s.ranges.largest_free() >= r))
            .min_by_key(|(_, s)| s.ranges.largest_free())?;
        let slab = inner.slabs.get_mut(&id)?;
        let rounded = slab.ranges.rounded(size)?;
        let offset = slab.ranges.alloc(size)?;
        slab.live += 1;
        Some(RegionAlloc {
            slab: id,
            buffer: slab.buffer.clone(),
            offset,
            rounded,
        })
    }

    /// Adds a slab and carves `first` bytes out of it under the same lock, so
    /// a concurrent `reclaim` cannot release the new, still-empty slab
    /// before its first region exists.
    pub fn insert_with_region(
        &self,
        buffer: wgpu::Buffer,
        capacity: u64,
        reservation: Reservation,
        first: u64,
    ) -> Option<RegionAlloc> {
        let mut inner = lock_or_recover(&self.inner);
        let id = inner.next_id;
        inner.next_id = inner.next_id.wrapping_add(1);
        let mut ranges = RangeAllocator::new(capacity, self.align);
        let rounded = ranges.rounded(first)?;
        let offset = ranges.alloc(first)?;
        inner.slabs.insert(
            id,
            Slab {
                buffer: buffer.clone(),
                ranges,
                live: 1,
                _reservation: reservation,
            },
        );
        Some(RegionAlloc { slab: id, buffer, offset, rounded })
    }

    /// Quarantines a freed region until `epoch` retires.
    pub fn release(&self, slab: u32, offset: u64, rounded: u64, epoch: u64) {
        lock_or_recover(&self.inner).quarantine.push_back(Quarantined {
            epoch,
            slab,
            offset,
            rounded,
        });
    }

    /// Returns quarantined regions whose epoch is `<= completed` to their
    /// slabs. Empty slabs beyond one spare (or all of them, under `pressure`)
    /// are removed and returned so the caller drops them, releasing VRAM,
    /// outside the lock.
    pub fn reclaim(&self, completed: u64, pressure: bool) -> Vec<Slab> {
        let mut inner = lock_or_recover(&self.inner);
        // Epoch tags are handed out in nondecreasing order.
        while inner.quarantine.front().is_some_and(|q| q.epoch <= completed) {
            let Some(q) = inner.quarantine.pop_front() else { break };
            if let Some(slab) = inner.slabs.get_mut(&q.slab) {
                if slab.ranges.free(q.offset, q.rounded) {
                    slab.live = slab.live.saturating_sub(1);
                } else {
                    eprintln!("[HC4J] slab {} rejected free of [{}, +{}) (double free?)", q.slab, q.offset, q.rounded);
                }
            }
        }
        let empty: Vec<u32> = inner.slabs.iter().filter(|(_, s)| s.live == 0).map(|(&id, _)| id).collect();
        let keep = if pressure { 0 } else { 1 };
        empty
            .into_iter()
            .skip(keep)
            .filter_map(|id| inner.slabs.remove(&id))
            .collect()
    }

    pub fn quarantined(&self) -> usize {
        lock_or_recover(&self.inner).quarantine.len()
    }

    /// Highest epoch among quarantined regions, if any are waiting.
    pub fn newest_quarantine_epoch(&self) -> Option<u64> {
        lock_or_recover(&self.inner).quarantine.back().map(|q| q.epoch)
    }

    pub fn stats(&self) -> SlabStats {
        let inner = lock_or_recover(&self.inner);
        SlabStats {
            slabs: inner.slabs.len() as u64,
            slab_bytes: inner.slabs.values().map(|s| s.ranges.capacity).sum(),
            live_regions: inner.slabs.values().map(|s| s.live as u64).sum(),
            quarantined: inner.quarantine.len() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RangeAllocator;

    #[test]
    fn range_allocator_coalesces_and_reuses() {
        let mut r = RangeAllocator::new(4096, 256);
        let a = r.alloc(100).unwrap();
        let b = r.alloc(300).unwrap();
        let c = r.alloc(256).unwrap();
        assert_eq!((a, b, c), (0, 256, 768));
        assert!(r.free(b, 300));
        assert!(r.free(a, 100));
        assert_eq!(r.alloc(768), Some(0), "a+b coalesced into one hole");
        assert!(r.free(0, 768));
        assert!(r.free(c, 256));
        assert_eq!(r.largest_free(), 4096);
        assert_eq!(r.allocated(), 0);
    }

    #[test]
    fn range_allocator_rejects_double_free_and_bogus_ranges() {
        let mut r = RangeAllocator::new(4096, 256);
        let a = r.alloc(256).unwrap();
        assert!(r.free(a, 256));
        assert!(!r.free(a, 256), "double free must be rejected");
        assert!(!r.free(4096, 256), "out of range");
        assert!(!r.free(3, 256), "misaligned");
        assert_eq!(r.largest_free(), 4096);
    }

    #[test]
    fn range_allocator_randomized_invariants() {
        let mut r = RangeAllocator::new(1 << 20, 256);
        let mut live: Vec<(u64, u64)> = Vec::new();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..20_000 {
            if live.is_empty() || next() % 3 != 0 {
                let size = 1 + next() % 20_000;
                if let Some(off) = r.alloc(size) {
                    let rounded = size.div_ceil(256) * 256;
                    assert_eq!(off % 256, 0);
                    assert!(off + rounded <= 1 << 20);
                    for &(o, s) in &live {
                        assert!(off + rounded <= o || o + s <= off, "overlap");
                    }
                    live.push((off, rounded));
                }
            } else {
                let i = (next() as usize) % live.len();
                let (o, s) = live.swap_remove(i);
                assert!(r.free(o, s));
            }
            assert_eq!(r.allocated(), live.iter().map(|&(_, s)| s).sum::<u64>());
        }
        for (o, s) in live.drain(..) {
            assert!(r.free(o, s));
        }
        assert_eq!(r.largest_free(), 1 << 20);
        assert_eq!(r.fragmentation(), 0.0);
    }
}
