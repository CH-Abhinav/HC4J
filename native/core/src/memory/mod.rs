//! Tensor storage. Every tensor handle crossing the FFI boundary is a
//! `TensorId` owned by the process-wide [`TieredMemoryManager`].

pub mod budget;
pub mod lru;
pub mod manager;
pub mod slab;
pub mod spill;
pub mod transfer;

use std::sync::OnceLock;

pub use crate::engine::error::{
    HC4J_ERR_DEVICE, HC4J_ERR_GPU_READBACK, HC4J_ERR_GPU_VALIDATION, HC4J_ERR_HOST_IO, HC4J_ERR_INVALID_PARAM,
    HC4J_ERR_NOT_FOUND, HC4J_ERR_OUT_OF_MEMORY, HC4J_ERR_UNSUPPORTED, HC4J_SUCCESS,
};
use crate::engine::error::{Hc4jError, Hc4jResult};
pub use manager::{DeviceBlock, DeviceSpan, HostBlock, MemStats, Residency, Snapshot, TieredMemoryManager};

pub type TensorId = u64;

const F32_BYTES: u64 = std::mem::size_of::<f32>() as u64;

static MANAGER: OnceLock<TieredMemoryManager> = OnceLock::new();

pub fn manager() -> Hc4jResult<&'static TieredMemoryManager> {
    let engine = crate::engine::get_engine()?;
    Ok(MANAGER.get_or_init(|| {
        TieredMemoryManager::new(
            engine,
            default_vram_budget(engine),
            env_megabytes("HC4J_HOST_BUDGET_MB").unwrap_or(4 << 30),
            spill::default_spill_dir(),
        )
    }))
}

/// wgpu cannot query free VRAM portably, so the starting budget comes from
/// `HC4J_VRAM_BUDGET_MB` or a device-class default. An optimistic default is
/// safe: the first real driver OOM clamps the budget to the observed capacity.
fn default_vram_budget(engine: &crate::engine::GpuEngine) -> u64 {
    env_megabytes("HC4J_VRAM_BUDGET_MB").unwrap_or(match engine.adapter_info.device_type {
        wgpu::DeviceType::DiscreteGpu => 8 << 30,
        wgpu::DeviceType::IntegratedGpu => 2 << 30,
        _ => 1 << 30,
    })
}

fn env_megabytes(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse::<u64>().ok()?.checked_mul(1 << 20)
}

fn checked_f32_bytes(length: usize) -> Hc4jResult<usize> {
    if length == 0 {
        return Err(Hc4jError::InvalidParam("zero length"));
    }
    (length as u64)
        .checked_mul(F32_BYTES)
        .filter(|&b| b <= isize::MAX as u64)
        .map(|b| b as usize)
        .ok_or(Hc4jError::InvalidParam("length overflows"))
}

fn alloc_elements(length: usize, zeroed: bool) -> Hc4jResult<TensorId> {
    let bytes = checked_f32_bytes(length)?;
    manager()?.allocate(bytes as u64, zeroed)
}

/// Allocates a zeroed tensor of `length` 4-byte elements. Falls back to host
/// RAM or disk when VRAM is exhausted.
pub fn alloc(length: usize) -> Hc4jResult<TensorId> {
    alloc_elements(length, true)
}

/// Like [`alloc`], but the contents are unspecified (a reused slab region
/// keeps stale bytes). For op outputs that kernels overwrite in full.
pub fn alloc_uninit(length: usize) -> Hc4jResult<TensorId> {
    alloc_elements(length, false)
}

fn align_up_u64(val: u64, align: u64) -> u64 {
    (val.saturating_add(align - 1)) & !(align - 1)
}

/// Allocates `bytes` in VRAM/host/spill, aligned up to 4 bytes for WebGPU/WGSL
/// compatibility. If `zeroed` is true, zero-fills the allocation.
pub fn alloc_bytes(bytes: u64, zeroed: bool) -> Hc4jResult<TensorId> {
    if bytes == 0 {
        return Err(Hc4jError::InvalidParam("bytes must be positive"));
    }
    let aligned = align_up_u64(bytes, 4);
    manager()?.allocate(aligned, zeroed)
}

/// Writes `data` bytes into tensor `id` (padding trailing partial words to a 4-byte boundary).
pub fn write_bytes(id: TensorId, data: &[u8]) -> Hc4jResult<()> {
    if data.is_empty() {
        return Err(Hc4jError::InvalidParam("empty write slice"));
    }
    if data.len().is_multiple_of(4) {
        manager()?.write(id, data)
    } else {
        let aligned = data
            .len()
            .checked_add(4 - (data.len() % 4))
            .ok_or(Hc4jError::InvalidParam("overflow"))?;
        let mut padded = vec![0u8; aligned];
        padded[..data.len()].copy_from_slice(data);
        manager()?.write(id, &padded)
    }
}

/// Downloads `out.len()` bytes from tensor `id` into `out`.
pub fn download_bytes(id: TensorId, out: &mut [u8]) -> Hc4jResult<()> {
    if out.is_empty() {
        return Err(Hc4jError::InvalidParam("empty download slice"));
    }
    if out.len().is_multiple_of(4) {
        manager()?.read(id, out)
    } else {
        let aligned = out
            .len()
            .checked_add(4 - (out.len() % 4))
            .ok_or(Hc4jError::InvalidParam("overflow"))?;
        let mut temp = vec![0u8; aligned];
        manager()?.read(id, &mut temp)?;
        out.copy_from_slice(&temp[..out.len()]);
        Ok(())
    }
}

/// Uploads `data` (`f32` slice) into tensor `id`.
pub fn write_f32(id: TensorId, data: &[f32]) -> Hc4jResult<()> {
    if data.is_empty() {
        return Err(Hc4jError::InvalidParam("empty write slice"));
    }
    manager()?.write(id, bytemuck::cast_slice(data))
}

/// Downloads `f32` contents from tensor `id` into `out`.
pub fn download_f32(id: TensorId, out: &mut [f32]) -> Hc4jResult<()> {
    if out.is_empty() {
        return Err(Hc4jError::InvalidParam("empty download slice"));
    }
    manager()?.read(id, bytemuck::cast_slice_mut(out))
}

/// Releases tensor `id` from all tiers.
pub fn free(id: TensorId) -> Hc4jResult<()> {
    manager()?.free(id)
}

/// Sets the VRAM and host-RAM budgets in bytes (`0` keeps the current value)
/// and immediately evicts/spills down to the new limits.
pub fn configure(vram_budget_bytes: u64, host_budget_bytes: u64) -> Hc4jResult<()> {
    manager()?.configure(vram_budget_bytes, host_budget_bytes)
}

/// Returns current tiered memory allocation/transfer statistics.
pub fn stats() -> Hc4jResult<MemStats> {
    Ok(manager()?.stats())
}

/// Returns the tier holding `id` (`0` = VRAM, `1` = host RAM, `2` = disk).
pub fn residency(id: TensorId) -> Hc4jResult<i32> {
    manager()?.residency_code(id)
}

/// Forces `id` out of VRAM. Succeeds as a no-op if already off-device or pinned.
pub fn evict(id: TensorId) -> Hc4jResult<()> {
    manager()?.evict(id).map(|_| ())
}
