use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use crate::engine::error::Hc4jResult;
use crate::engine::stream::CommandStream;

use super::super::budget::Reservation;
use super::super::lru::LruIndex;
pub(crate) use super::super::slab::{RegionAlloc, SlabPool};
pub(crate) use super::super::spill::SpillFile;
use super::super::TensorId;
use super::TieredMemoryManager;

pub(crate) const SETTLE_TIMEOUT: Duration = Duration::from_secs(300);
pub(crate) const MAX_DRIVER_OOM_RETRIES: u32 = 4;
pub(crate) const MIN_VRAM_BUDGET: u64 = 64 << 20;
pub(crate) const MIB: u64 = 1 << 20;
pub(crate) const MIN_SLAB: u64 = 4 * MIB;
pub(crate) const MAX_SLAB: u64 = 64 * MIB;

pub const TENSOR_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

/// A bindable byte range of a device buffer: a whole dedicated buffer or one
/// slab region.
#[derive(Clone, Debug)]
pub struct DeviceSpan {
    pub buffer: wgpu::Buffer,
    pub offset: u64,
    pub size: u64,
}

impl DeviceSpan {
    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        self.sub_binding(0, self.size)
    }

    /// Binds `len` bytes starting `offset` bytes into the span.
    pub fn sub_binding(&self, offset: u64, len: u64) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset + offset,
            size: wgpu::BufferSize::new(len),
        })
    }

    pub fn overlaps(&self, other: &DeviceSpan) -> bool {
        self.buffer == other.buffer
            && self.offset < other.offset + other.size
            && other.offset < self.offset + self.size
    }
}

pub(crate) enum Backing {
    Dedicated { _reservation: Reservation },
    Region {
        pool: Arc<SlabPool>,
        slab: u32,
        rounded: u64,
        stream: &'static CommandStream,
    },
}

/// Device storage for one tensor (or one transient scratch buffer). Dropping
/// it releases a dedicated buffer's budget, or quarantines a slab region
/// until every submission that could still touch it has retired.
pub struct DeviceBlock {
    pub(crate) span: DeviceSpan,
    pub(crate) backing: Backing,
}

impl DeviceBlock {
    pub(crate) fn new(span: DeviceSpan, backing: Backing) -> Self {
        Self { span, backing }
    }

    pub fn span(&self) -> &DeviceSpan {
        &self.span
    }

    pub fn size(&self) -> u64 {
        self.span.size
    }

    pub fn is_region(&self) -> bool {
        matches!(self.backing, Backing::Region { .. })
    }
}

impl Drop for DeviceBlock {
    fn drop(&mut self) {
        if let Backing::Region { pool, slab, rounded, stream } = &self.backing {
            pool.release(*slab, self.span.offset, *rounded, stream.retire_epoch());
        }
    }
}

/// Host-RAM copy of a tensor. The bytes are behind an `Arc` so snapshot
/// readers stay valid while the entry migrates; writers copy on write.
pub struct HostBlock {
    pub(crate) data: Arc<Vec<u8>>,
    pub(crate) _reservation: Reservation,
}

impl HostBlock {
    pub fn new(data: Vec<u8>, reservation: Reservation) -> Self {
        Self {
            data: Arc::new(data),
            _reservation: reservation,
        }
    }

    pub fn data(&self) -> &Arc<Vec<u8>> {
        &self.data
    }
}

pub enum Residency {
    /// Bound directly by compute pipelines.
    DeviceResident(DeviceBlock),
    /// Paged out to host RAM.
    HostResident(HostBlock),
    /// Spilled to a self-deleting file on host storage.
    Evicted(SpillFile),
    /// Being migrated by some thread; wait on `settled`.
    InTransit,
}

impl Residency {
    pub fn code(&self) -> i32 {
        match self {
            Residency::DeviceResident(_) => 0,
            Residency::HostResident(_) => 1,
            Residency::Evicted(_) => 2,
            Residency::InTransit => 3,
        }
    }
}

pub struct TensorAllocation {
    pub id: TensorId,
    pub size_bytes: u64,
    pub(crate) residency: Residency,
    /// Active users that need the tensor to stay where it is. Pinned tensors
    /// are never chosen for eviction or spilling.
    pub(crate) pins: u32,
    /// `free` was called while pinned; the last unpin removes the entry.
    pub(crate) free_pending: bool,
    /// Position in the LRU order; larger is more recent.
    pub(crate) tick: u64,
}

impl TensorAllocation {
    pub fn new(id: TensorId, size_bytes: u64, residency: Residency, tick: u64) -> Self {
        Self {
            id,
            size_bytes,
            residency,
            pins: 0,
            free_pending: false,
            tick,
        }
    }

    pub fn residency(&self) -> &Residency {
        &self.residency
    }

    pub fn pins(&self) -> u32 {
        self.pins
    }

    pub fn free_pending(&self) -> bool {
        self.free_pending
    }
}

/// A read-only view of a tensor's bytes that stays valid without the table
/// lock. `Device` is only handed out while the tensor is pinned.
pub enum Snapshot {
    Device(DeviceSpan),
    Host(Arc<Vec<u8>>),
    Disk(Arc<File>),
}

/// Unpins its tensors when dropped.
pub struct PinGuard<'m> {
    pub(crate) manager: &'m TieredMemoryManager,
    pub(crate) ids: Vec<TensorId>,
}

impl<'m> PinGuard<'m> {
    pub fn new(manager: &'m TieredMemoryManager, ids: Vec<TensorId>) -> Self {
        Self { manager, ids }
    }
}

impl Drop for PinGuard<'_> {
    fn drop(&mut self) {
        if self.ids.is_empty() {
            return;
        }
        let mut released = Vec::new();
        {
            let mut inner = self.manager.lock_inner();
            for id in &self.ids {
                let Some(entry) = inner.allocs.get_mut(id) else { continue };
                entry.pins = entry.pins.saturating_sub(1);
                if entry.pins == 0
                    && entry.free_pending
                    && let Some(entry) = inner.remove_entry(*id)
                {
                    released.push(entry);
                }
            }
        }
        // Deferred frees: storage is dropped outside the lock.
        drop(released);
    }
}

/// Device spans for a set of pinned, VRAM-resident tensors, in request order.
pub struct ResidentSet<'m> {
    pub spans: Vec<DeviceSpan>,
    pub(crate) _pins: PinGuard<'m>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MemStats {
    pub vram_budget: u64,
    pub vram_used: u64,
    pub host_budget: u64,
    pub host_used: u64,
    pub spilled_bytes: u64,
    pub tensors_device: u64,
    pub tensors_host: u64,
    pub tensors_evicted: u64,
    pub evictions: u64,
    pub page_ins: u64,
    pub spills: u64,
    pub driver_ooms: u64,
    pub streamed_ops: u64,
    /// Tensors allocated through the FFI (`hc4j_gpu_alloc*`).
    pub allocations: u64,
    /// `create_buffer` calls for dedicated tensor or scratch storage.
    pub dedicated_buffers: u64,
    pub slabs_created: u64,
    pub region_allocs: u64,
    pub slab_count: u64,
    pub slab_bytes: u64,
    pub slab_regions: u64,
    pub quarantined: u64,
}

#[derive(Default)]
pub(crate) struct Counters {
    pub evictions: AtomicU64,
    pub page_ins: AtomicU64,
    pub spills: AtomicU64,
    pub driver_ooms: AtomicU64,
    pub streamed_ops: AtomicU64,
    pub allocations: AtomicU64,
    pub dedicated_buffers: AtomicU64,
    pub slabs_created: AtomicU64,
    pub region_allocs: AtomicU64,
}

#[derive(Default)]
pub(crate) struct Inner {
    pub allocs: HashMap<TensorId, TensorAllocation>,
    pub device_lru: LruIndex,
    pub host_lru: LruIndex,
    pub clock: u64,
}

impl Inner {
    pub fn next_tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Marks `id` as most recently used within its tier.
    pub fn touch(&mut self, id: TensorId) {
        let tick = self.next_tick();
        let Some(entry) = self.allocs.get_mut(&id) else {
            return;
        };
        let old = std::mem::replace(&mut entry.tick, tick);
        match entry.residency {
            Residency::DeviceResident(_) => {
                self.device_lru.remove(old);
                self.device_lru.insert(tick, id);
            }
            Residency::HostResident(_) => {
                self.host_lru.remove(old);
                self.host_lru.insert(tick, id);
            }
            Residency::Evicted(_) | Residency::InTransit => {}
        }
    }

    /// Takes the entry's data out, leaving it `InTransit` and out of every LRU.
    pub fn take_for_transit(&mut self, id: TensorId) -> Option<(Residency, u64)> {
        let entry = self.allocs.get_mut(&id)?;
        let taken = std::mem::replace(&mut entry.residency, Residency::InTransit);
        match taken {
            Residency::DeviceResident(_) => self.device_lru.remove(entry.tick),
            Residency::HostResident(_) => self.host_lru.remove(entry.tick),
            Residency::Evicted(_) | Residency::InTransit => {}
        }
        Some((taken, entry.size_bytes))
    }

    /// Installs `residency` for `id`. If the entry vanished (it cannot while
    /// in transit, since `free` waits), the residency is handed back so the
    /// caller can drop it outside the lock.
    pub fn settle(&mut self, id: TensorId, residency: Residency) -> Option<Residency> {
        let Some(entry) = self.allocs.get_mut(&id) else {
            return Some(residency);
        };
        match residency {
            Residency::DeviceResident(_) => self.device_lru.insert(entry.tick, id),
            Residency::HostResident(_) => self.host_lru.insert(entry.tick, id),
            Residency::Evicted(_) | Residency::InTransit => {}
        }
        entry.residency = residency;
        None
    }

    pub fn remove_entry(&mut self, id: TensorId) -> Option<TensorAllocation> {
        let entry = self.allocs.remove(&id)?;
        match entry.residency {
            Residency::DeviceResident(_) => self.device_lru.remove(entry.tick),
            Residency::HostResident(_) => self.host_lru.remove(entry.tick),
            Residency::Evicted(_) | Residency::InTransit => {}
        }
        Some(entry)
    }

    pub fn evictable_bytes(&self, device_tier: bool) -> u64 {
        self.allocs
            .values()
            .filter(|a| a.pins == 0)
            .filter(|a| match a.residency {
                Residency::DeviceResident(_) => device_tier,
                Residency::HostResident(_) => !device_tier,
                _ => false,
            })
            .map(|a| a.size_bytes)
            .sum()
    }

    pub fn oldest_unpinned(&self, device_tier: bool) -> Option<TensorId> {
        let lru = if device_tier { &self.device_lru } else { &self.host_lru };
        lru.oldest_where(|id| self.allocs.get(&id).is_some_and(|a| a.pins == 0))
    }
}

pub fn align_up(value: u64, align: u64) -> Option<u64> {
    let rem = value % align;
    if rem == 0 { Some(value) } else { value.checked_add(align - rem) }
}

/// Fallible host allocations: Rust's infallible `Vec` APIs abort the process
/// (and the JVM) on allocation failure.
pub fn try_with_capacity(len: u64) -> Option<Vec<u8>> {
    let len = usize::try_from(len).ok()?;
    let mut data = Vec::new();
    data.try_reserve_exact(len).ok()?;
    Some(data)
}

pub(crate) fn try_zeroed(len: u64) -> Option<Vec<u8>> {
    let mut data = try_with_capacity(len)?;
    // Capacity is already reserved, so this cannot reallocate.
    data.resize(len as usize, 0);
    Some(data)
}

/// Copy-on-write access to shared host bytes.
pub(crate) fn unshare(data: &mut Arc<Vec<u8>>) -> Hc4jResult<&mut Vec<u8>> {
    if Arc::get_mut(data).is_none() {
        let mut copy = try_with_capacity(data.len() as u64).ok_or(crate::engine::error::Hc4jError::OutOfMemory)?;
        copy.extend_from_slice(data);
        *data = Arc::new(copy);
    }
    Arc::get_mut(data).ok_or(crate::engine::error::Hc4jError::Device("host block still shared after copy".to_string()))
}
