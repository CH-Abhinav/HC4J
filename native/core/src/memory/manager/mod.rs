pub mod alloc;
pub mod tiering;
pub mod types;

pub use types::*;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Instant};

use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::memory::budget::Budget;
use crate::memory::slab::SlabPool;
use crate::memory::transfer::{ReadbackRing, TRANSFER_CHUNK};
use crate::memory::TensorId;
use crate::engine::{lock_or_recover, GpuEngine};

pub struct TieredMemoryManager {
    pub(crate) engine: &'static GpuEngine,
    pub(crate) vram: Arc<Budget>,
    pub(crate) host: Arc<Budget>,
    pub(crate) pool: Arc<SlabPool>,
    pub(crate) inner: Mutex<Inner>,
    pub(crate) settled: Condvar,
    pub(crate) ring: Mutex<Option<ReadbackRing>>,
    pub(crate) spill_dir: PathBuf,
    pub(crate) next_id: AtomicU64,
    pub(crate) counters: Counters,
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

    pub fn spill_dir(&self) -> &Path {
        &self.spill_dir
    }

    pub(crate) fn lock_inner(&self) -> MutexGuard<'_, Inner> {
        lock_or_recover(&self.inner)
    }

    /// Locks the table once `id` is not in transit.
    pub(crate) fn wait_settled(&self, id: TensorId) -> Hc4jResult<MutexGuard<'_, Inner>> {
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
