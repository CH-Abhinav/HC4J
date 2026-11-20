//! Tensor storage. Every tensor handle crossing the FFI boundary is a
//! `TensorId` owned by the process-wide [`TieredMemoryManager`].

pub mod budget;
pub mod lru;
pub mod manager;
pub mod slab;
pub mod spill;
pub mod transfer;

use std::sync::OnceLock;

pub use crate::error::{
    HC4J_ERR_DEVICE, HC4J_ERR_GPU_READBACK, HC4J_ERR_GPU_VALIDATION, HC4J_ERR_HOST_IO, HC4J_ERR_INVALID_PARAM,
    HC4J_ERR_NOT_FOUND, HC4J_ERR_OUT_OF_MEMORY, HC4J_ERR_UNSUPPORTED, HC4J_SUCCESS,
};
use crate::error::{Hc4jError, Hc4jResult, ffi_guard};
pub use manager::{DeviceBlock, DeviceSpan, HostBlock, MemStats, Residency, Snapshot, TieredMemoryManager};

pub type TensorId = u64;

const F32_BYTES: u64 = std::mem::size_of::<f32>() as u64;

static MANAGER: OnceLock<TieredMemoryManager> = OnceLock::new();

pub fn manager() -> Hc4jResult<&'static TieredMemoryManager> {
    let engine = crate::get_engine()?;
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
fn default_vram_budget(engine: &crate::GpuEngine) -> u64 {
    env_megabytes("HC4J_VRAM_BUDGET_MB").unwrap_or(match engine.adapter_info.device_type {
        wgpu::DeviceType::DiscreteGpu => 8 << 30,
        wgpu::DeviceType::IntegratedGpu => 2 << 30,
        _ => 1 << 30,
    })
}

fn env_megabytes(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse::<u64>().ok()?.checked_mul(1 << 20)
}

/// Views `length` f32 elements at `ptr` as bytes.
///
/// # Safety
/// `ptr` must point to `length` readable f32 values that stay valid and
/// unaliased by writers for the returned lifetime.
unsafe fn host_bytes<'a>(ptr: *const f32, length: usize) -> Hc4jResult<&'a [u8]> {
    let bytes = checked_f32_bytes(ptr.is_null(), length)?;
    Ok(unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), bytes) })
}

/// # Safety
/// As [`host_bytes`], but for `length` writable f32 values.
unsafe fn host_bytes_mut<'a>(ptr: *mut f32, length: usize) -> Hc4jResult<&'a mut [u8]> {
    let bytes = checked_f32_bytes(ptr.is_null(), length)?;
    Ok(unsafe { std::slice::from_raw_parts_mut(ptr.cast::<u8>(), bytes) })
}

fn checked_f32_bytes(is_null: bool, length: usize) -> Hc4jResult<usize> {
    if is_null || length == 0 {
        return Err(Hc4jError::InvalidParam("null pointer or zero length"));
    }
    (length as u64)
        .checked_mul(F32_BYTES)
        .filter(|&b| b <= isize::MAX as u64)
        .map(|b| b as usize)
        .ok_or(Hc4jError::InvalidParam("length overflows"))
}

fn alloc_elements(op: &str, length: usize, zeroed: bool) -> u64 {
    let mut id = 0;
    let status = ffi_guard(op, || {
        let bytes = (length as u64)
            .checked_mul(F32_BYTES)
            .filter(|&b| b > 0)
            .ok_or(Hc4jError::InvalidParam("length"))?;
        id = manager()?.allocate(bytes, zeroed)?;
        Ok(())
    });
    if status == HC4J_SUCCESS { id } else { 0 }
}

/// Allocates a zeroed tensor of `length` 4-byte elements. Returns 0 on
/// failure. Falls back to host RAM or disk when VRAM is exhausted, so this
/// only fails for invalid sizes or when every tier is full.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_gpu_alloc(length: usize) -> u64 {
    alloc_elements("hc4j_gpu_alloc", length, true)
}

/// Like `hc4j_gpu_alloc`, but the contents are unspecified (a reused slab
/// region keeps stale bytes). For op outputs that kernels overwrite in full,
/// this skips a zero-fill pass over the whole tensor.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_gpu_alloc_uninit(length: usize) -> u64 {
    alloc_elements("hc4j_gpu_alloc_uninit", length, false)
}

/// # Safety
/// `ptr_in` must point to `length` readable f32 values for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_gpu_write(id: u64, ptr_in: *const f32, length: usize) -> i32 {
    ffi_guard("hc4j_gpu_write", || {
        let data = unsafe { host_bytes(ptr_in, length)? };
        manager()?.write(id, data)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_gpu_free(id: u64) -> i32 {
    ffi_guard("hc4j_gpu_free", || manager()?.free(id))
}

/// # Safety
/// `ptr_out` must point to `length` writable f32 values for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_gpu_download(id: u64, ptr_out: *mut f32, length: usize) -> i32 {
    ffi_guard("hc4j_gpu_download", || {
        let out = unsafe { host_bytes_mut(ptr_out, length)? };
        manager()?.read(id, out)
    })
}

/// Sets the VRAM and host-RAM budgets in bytes (0 keeps the current value)
/// and immediately evicts / spills down to the new limits.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_mem_configure(vram_budget_bytes: u64, host_budget_bytes: u64) -> i32 {
    ffi_guard("hc4j_mem_configure", || manager()?.configure(vram_budget_bytes, host_budget_bytes))
}

/// # Safety
/// `out` must be null or point to writable memory for one `MemStats`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_mem_stats(out: *mut MemStats) -> i32 {
    ffi_guard("hc4j_mem_stats", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null stats pointer"));
        }
        let stats = manager()?.stats();
        // Unaligned write so a caller that did not 8-byte-align the struct
        // cannot trigger UB.
        unsafe { out.write_unaligned(stats) };
        Ok(())
    })
}

/// Returns the tier holding `id` (0 = VRAM, 1 = host RAM, 2 = disk), or a
/// negative status code.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_mem_residency(id: u64) -> i32 {
    let mut code = 0;
    let status = ffi_guard("hc4j_mem_residency", || {
        code = manager()?.residency_code(id)?;
        Ok(())
    });
    if status == HC4J_SUCCESS { code } else { status }
}

/// Forces `id` out of VRAM. Succeeds as a no-op if it is already off-device
/// or pinned by an in-flight operation.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_mem_evict(id: u64) -> i32 {
    ffi_guard("hc4j_mem_evict", || manager()?.evict(id).map(|_| ()))
}
