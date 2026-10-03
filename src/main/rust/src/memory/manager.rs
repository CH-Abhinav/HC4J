//! Software-managed tiered memory: VRAM -> host RAM -> disk.
//!
//! wgpu has no page faults, so an allocation that does not fit in VRAM either
//! fails or (on WDDM) silently oversubscribes. This manager keeps every
//! tensor's bytes in exactly one tier and moves them explicitly:
//!
//! * **Budgets.** VRAM and host RAM each have an atomic [`Budget`]. Every byte
//!   in a tier is held by an RAII [`Reservation`], so accounting cannot drift.
//!   Slabs reserve their full capacity: the budget tracks physical VRAM.
//! * **Slab sub-allocation.** Tensors up to a quarter of a slab become regions
//!   of a shared buffer ([`super::slab`]), so op outputs cost no driver call.
//!   Larger tensors get dedicated buffers.
//! * **Eviction.** When VRAM is exhausted, the least recently used *unpinned*
//!   device tensor is copied out through the persistent readback ring to host
//!   RAM, or to disk if the host budget is also exhausted, one at a time until
//!   the request fits.
//! * **Driver OOM.** Buffer creation runs inside an error scope. If the driver
//!   reports OutOfMemory while the software budget still has room, the budget
//!   was optimistic: it is clamped to what is committed and eviction resumes.
//! * **Concurrency.** The table lock is never held across GPU waits or I/O.
//!   An entry being migrated is marked `InTransit`; other threads wait on a
//!   condvar until it settles. Freeing a pinned tensor is deferred to its last
//!   unpin, so a slab region is never released while a recording thread still
//!   holds its span.

use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::error::{Hc4jError, Hc4jResult};
use crate::stream::CommandStream;
use crate::{ErrorTrap, GpuEngine, hc4j_trace, lock_or_recover};

use super::TensorId;
use super::budget::{Budget, Reservation};
use super::lru::LruIndex;
use super::slab::{RegionAlloc, SlabPool};
use super::spill::{self, SpillFile};
use super::transfer::{self, ReadbackRing, TRANSFER_CHUNK, UploadSource};

/// How long a thread waits for another thread's migration of the same tensor.
const SETTLE_TIMEOUT: Duration = Duration::from_secs(300);
const MAX_DRIVER_OOM_RETRIES: u32 = 4;
/// The budget is never clamped below this, so one spurious driver OOM cannot
/// shrink it to nothing.
const MIN_VRAM_BUDGET: u64 = 64 << 20;
const MIB: u64 = 1 << 20;
const MIN_SLAB: u64 = 4 * MIB;
const MAX_SLAB: u64 = 64 * MIB;

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

enum Backing {
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
    span: DeviceSpan,
    backing: Backing,
}

impl DeviceBlock {
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
    data: Arc<Vec<u8>>,
    _reservation: Reservation,
}

impl HostBlock {
    pub fn new(data: Vec<u8>, reservation: Reservation) -> Self {
        Self {
            data: Arc::new(data),
            _reservation: reservation,
        }
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
    residency: Residency,
    /// Active users that need the tensor to stay where it is. Pinned tensors
    /// are never chosen for eviction or spilling.
    pins: u32,
    /// `free` was called while pinned; the last unpin removes the entry.
    free_pending: bool,
    /// Position in the LRU order; larger is more recent.
    tick: u64,
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
    manager: &'m TieredMemoryManager,
    ids: Vec<TensorId>,
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
    _pins: PinGuard<'m>,
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
struct Counters {
    evictions: AtomicU64,
    page_ins: AtomicU64,
    spills: AtomicU64,
    driver_ooms: AtomicU64,
    streamed_ops: AtomicU64,
    allocations: AtomicU64,
    dedicated_buffers: AtomicU64,
    slabs_created: AtomicU64,
    region_allocs: AtomicU64,
}

#[derive(Default)]
struct Inner {
    allocs: HashMap<TensorId, TensorAllocation>,
    device_lru: LruIndex,
    host_lru: LruIndex,
    clock: u64,
}

impl Inner {
    fn next_tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Marks `id` as most recently used within its tier.
    fn touch(&mut self, id: TensorId) {
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
    fn take_for_transit(&mut self, id: TensorId) -> Option<(Residency, u64)> {
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
    fn settle(&mut self, id: TensorId, residency: Residency) -> Option<Residency> {
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

    fn remove_entry(&mut self, id: TensorId) -> Option<TensorAllocation> {
        let entry = self.allocs.remove(&id)?;
        match entry.residency {
            Residency::DeviceResident(_) => self.device_lru.remove(entry.tick),
            Residency::HostResident(_) => self.host_lru.remove(entry.tick),
            Residency::Evicted(_) | Residency::InTransit => {}
        }
        Some(entry)
    }

    fn evictable_bytes(&self, device_tier: bool) -> u64 {
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

    fn oldest_unpinned(&self, device_tier: bool) -> Option<TensorId> {
        let lru = if device_tier { &self.device_lru } else { &self.host_lru };
        lru.oldest_where(|id| self.allocs.get(&id).is_some_and(|a| a.pins == 0))
    }
}

pub struct TieredMemoryManager {
    engine: &'static GpuEngine,
    vram: Arc<Budget>,
    host: Arc<Budget>,
    pool: Arc<SlabPool>,
    inner: Mutex<Inner>,
    settled: Condvar,
    ring: Mutex<Option<ReadbackRing>>,
    spill_dir: PathBuf,
    next_id: AtomicU64,
    counters: Counters,
}

impl TieredMemoryManager {
    pub fn new(engine: &'static GpuEngine, vram_budget: u64, host_budget: u64, spill_dir: PathBuf) -> Self {
        Self {
            engine,
            vram: Budget::new(vram_budget),
            host: Budget::new(host_budget),
            pool: Arc::new(SlabPool::new(engine.limits.min_storage_buffer_offset_alignment as u64)),
            inner: Mutex::new(Inner::default()),
            settled: Condvar::new(),
            ring: Mutex::new(None),
            spill_dir,
            next_id: AtomicU64::new(1),
            counters: Counters::default(),
        }
    }

    pub fn engine(&self) -> &'static GpuEngine {
        self.engine
    }

    pub fn spill_dir(&self) -> &std::path::Path {
        &self.spill_dir
    }

    fn lock_inner(&self) -> MutexGuard<'_, Inner> {
        lock_or_recover(&self.inner)
    }

    /// Locks the table once `id` is not in transit.
    fn wait_settled(&self, id: TensorId) -> Hc4jResult<MutexGuard<'_, Inner>> {
        let deadline = Instant::now() + SETTLE_TIMEOUT;
        let mut inner = self.lock_inner();
        loop {
            match inner.allocs.get(&id) {
                None => return Err(Hc4jError::NotFound),
                Some(entry) if entry.free_pending => return Err(Hc4jError::NotFound),
                Some(entry) if !matches!(entry.residency, Residency::InTransit) => return Ok(inner),
                Some(_) => {}
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(Hc4jError::Device(format!("tensor {id} stuck in transit")));
            }
            inner = match self.settled.wait_timeout(inner, deadline - now) {
                Ok((guard, _)) => guard,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
    }

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
            TensorAllocation {
                id,
                size_bytes,
                residency: Residency::InTransit,
                pins: 0,
                free_pending: false,
                tick,
            },
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
                hc4j_trace!("alloc {size_bytes} B: VRAM exhausted, placing tensor off-device");
                self.allocate_off_device(size_bytes)?
            }
            Err(err) => return Err(err),
        };
        Ok(self.register(size_bytes, residency))
    }

    fn allocate_off_device(&self, size_bytes: u64) -> Hc4jResult<Residency> {
        if let Some(reservation) = self.reserve_host(size_bytes)?
            && let Some(data) = try_zeroed(size_bytes)
        {
            return Ok(Residency::HostResident(HostBlock::new(data, reservation)));
        }
        Ok(Residency::Evicted(SpillFile::create(&self.spill_dir, size_bytes)?))
    }

    /// Slab capacity for the current budget: an eighth of it, clamped to
    /// 4..64 MiB and to what one storage binding can address.
    fn slab_size(&self) -> u64 {
        let limits = &self.engine.limits;
        let align = self.pool.align();
        let target = (self.vram.limit() / 8).clamp(MIN_SLAB, MAX_SLAB) / MIB * MIB;
        let cap = target.min(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        cap / align * align
    }

    /// Tensors up to this size are slab regions; larger ones are dedicated.
    fn region_max(&self) -> u64 {
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
                        return Ok(DeviceBlock {
                            span: DeviceSpan { buffer, offset: 0, size: size_bytes },
                            backing: Backing::Dedicated { _reservation: reservation },
                        });
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
        DeviceBlock {
            span: DeviceSpan {
                buffer: region.buffer,
                offset: region.offset,
                size: size_bytes,
            },
            backing: Backing::Region {
                pool: Arc::clone(&self.pool),
                slab: region.slab,
                rounded: region.rounded,
                stream: &self.engine.stream,
            },
        }
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

    fn on_driver_oom(&self, driver_ooms: &mut u32, size: u64) -> Hc4jResult<()> {
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

    /// Creates a buffer with OOM/validation errors trapped instead of panicking
    /// through wgpu's default handler.
    fn create_buffer(&self, size: u64, label: &str) -> Hc4jResult<wgpu::Buffer> {
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
    fn reclaim_slabs(&self, pressure: bool) -> usize {
        let released = self.pool.reclaim(self.engine.stream.completed_epoch(), pressure);
        let count = released.len();
        drop(released);
        count
    }

    fn release_spare_slabs(&self) -> bool {
        self.reclaim_slabs(true) > 0
    }

    /// Waits for quarantined regions to retire. Returns whether it waited.
    fn drain_quarantine(&self) -> Hc4jResult<bool> {
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
    // LRU eviction: VRAM -> host RAM (-> disk)
    // ------------------------------------------------------------------

    /// Demotes the least recently used unpinned device tensor. Returns
    /// `false` if there is none.
    fn evict_one(&self) -> Hc4jResult<bool> {
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
    fn make_vram_room(&self, size: u64) -> Hc4jResult<bool> {
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
                    hc4j_trace!("evicted tensor {id} ({} B) to tier {}", block.size(), residency.code());
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

    fn make_host_room(&self, size: u64) -> Hc4jResult<bool> {
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
                    hc4j_trace!("spilled tensor {id} ({} B) to disk", block.data.len());
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

        let mut pins = PinGuard {
            manager: self,
            ids: Vec::with_capacity(unique.len()),
        };
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
                hc4j_trace!("paged tensor {id} ({size} B) into VRAM");
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
        Ok((snapshot, size, PinGuard { manager: self, ids: vec![id] }))
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
                let _pin = PinGuard { manager: self, ids: vec![id] };
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
            Snapshot::Disk(file) => Ok(spill::read_exact_at(&file, out, 0)?),
        }
    }

    /// Runs `body` with exclusive use of the persistent readback ring,
    /// creating it on first use. A failed transfer can leave slots mapped or
    /// map-pending, so on error the ring is discarded and rebuilt next time.
    pub fn with_ring<R>(&self, body: impl FnOnce(&mut ReadbackRing) -> Hc4jResult<R>) -> Hc4jResult<R> {
        let mut slot = lock_or_recover(&self.ring);
        if slot.is_none() {
            *slot = Some(ReadbackRing::new(self.engine, TRANSFER_CHUNK)?);
        }
        let Some(ring) = slot.as_mut() else {
            return Err(Hc4jError::Device("readback ring unavailable".to_string()));
        };
        let result = body(ring);
        if result.is_err() {
            *slot = None;
        }
        result
    }

    // ------------------------------------------------------------------
    // Configuration and introspection
    // ------------------------------------------------------------------

    /// Sets tier budgets (0 leaves a tier unchanged) and immediately trims
    /// each tier down to its new limit.
    pub fn configure(&self, vram_budget: u64, host_budget: u64) -> Hc4jResult<()> {
        if vram_budget > 0 {
            self.vram.set_limit(vram_budget);
        }
        if host_budget > 0 {
            self.host.set_limit(host_budget);
        }
        self.make_vram_room(0)?;
        self.make_host_room(0)?;
        Ok(())
    }

    pub fn record_streamed_op(&self) {
        self.counters.streamed_ops.fetch_add(1, Ordering::Relaxed);
    }

    pub fn stats(&self) -> MemStats {
        let slabs = self.pool.stats();
        let c = &self.counters;
        let mut stats = MemStats {
            vram_budget: self.vram.limit(),
            vram_used: self.vram.used(),
            host_budget: self.host.limit(),
            host_used: self.host.used(),
            evictions: c.evictions.load(Ordering::Relaxed),
            page_ins: c.page_ins.load(Ordering::Relaxed),
            spills: c.spills.load(Ordering::Relaxed),
            driver_ooms: c.driver_ooms.load(Ordering::Relaxed),
            streamed_ops: c.streamed_ops.load(Ordering::Relaxed),
            allocations: c.allocations.load(Ordering::Relaxed),
            dedicated_buffers: c.dedicated_buffers.load(Ordering::Relaxed),
            slabs_created: c.slabs_created.load(Ordering::Relaxed),
            region_allocs: c.region_allocs.load(Ordering::Relaxed),
            slab_count: slabs.slabs,
            slab_bytes: slabs.slab_bytes,
            slab_regions: slabs.live_regions,
            quarantined: slabs.quarantined,
            ..MemStats::default()
        };
        for alloc in self.lock_inner().allocs.values() {
            match alloc.residency {
                Residency::DeviceResident(_) => stats.tensors_device += 1,
                Residency::HostResident(_) => stats.tensors_host += 1,
                Residency::Evicted(_) => {
                    stats.tensors_evicted += 1;
                    stats.spilled_bytes += alloc.size_bytes;
                }
                Residency::InTransit => {}
            }
        }
        stats
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

fn try_zeroed(len: u64) -> Option<Vec<u8>> {
    let mut data = try_with_capacity(len)?;
    // Capacity is already reserved, so this cannot reallocate.
    data.resize(len as usize, 0);
    Some(data)
}

/// Copy-on-write access to shared host bytes.
fn unshare(data: &mut Arc<Vec<u8>>) -> Hc4jResult<&mut Vec<u8>> {
    if Arc::get_mut(data).is_none() {
        let mut copy = try_with_capacity(data.len() as u64).ok_or(Hc4jError::OutOfMemory)?;
        copy.extend_from_slice(data);
        *data = Arc::new(copy);
    }
    Arc::get_mut(data).ok_or(Hc4jError::Device("host block still shared after copy".to_string()))
}
