//! C-ABI entry points for tiered VRAM / Host-RAM / Disk tensor allocation,
//! data transfer, eviction, and memory telemetry.

use hc4rs::MemStats;
use hc4rs::engine::error::{HC4J_SUCCESS, Hc4jError, Hc4jResult, ffi_guard};
use hc4rs::memory;

const F32_BYTES: u64 = std::mem::size_of::<f32>() as u64;

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
        id = if zeroed {
            memory::alloc(length)?
        } else {
            memory::alloc_uninit(length)?
        };
        Ok(())
    });
    if status == HC4J_SUCCESS { id } else { 0 }
}

/// Allocates a zeroed tensor of `length` 4-byte elements. Returns `0` on failure.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_gpu_alloc(length: usize) -> u64 {
    alloc_elements("hc4j_gpu_alloc", length, true)
}

/// Allocates an uninitialized tensor of `length` 4-byte elements. Returns `0` on failure.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_gpu_alloc_uninit(length: usize) -> u64 {
    alloc_elements("hc4j_gpu_alloc_uninit", length, false)
}

/// Allocates `bytes` (aligned up to 4 bytes). Returns `0` on failure.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_gpu_alloc_bytes(bytes: u64, zeroed: u32) -> u64 {
    let mut id = 0;
    let status = ffi_guard("hc4j_gpu_alloc_bytes", || {
        id = memory::alloc_bytes(bytes, zeroed != 0)?;
        Ok(())
    });
    if status == HC4J_SUCCESS { id } else { 0 }
}

/// # Safety
/// `ptr_in` must point to at least `bytes` readable bytes for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_gpu_write_bytes(id: u64, ptr_in: *const u8, bytes: usize) -> i32 {
    ffi_guard("hc4j_gpu_write_bytes", || {
        if ptr_in.is_null() || bytes == 0 {
            return Err(Hc4jError::InvalidParam("null pointer or zero bytes"));
        }
        let data = unsafe { std::slice::from_raw_parts(ptr_in, bytes) };
        memory::write_bytes(id, data)
    })
}

/// # Safety
/// `ptr_out` must point to at least `bytes` writable bytes for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_gpu_download_bytes(id: u64, ptr_out: *mut u8, bytes: usize) -> i32 {
    ffi_guard("hc4j_gpu_download_bytes", || {
        if ptr_out.is_null() || bytes == 0 {
            return Err(Hc4jError::InvalidParam("null pointer or zero bytes"));
        }
        let out = unsafe { std::slice::from_raw_parts_mut(ptr_out, bytes) };
        memory::download_bytes(id, out)
    })
}

/// # Safety
/// `ptr_in` must point to `length` readable `f32` (or 4-byte) values for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_gpu_write(id: u64, ptr_in: *const f32, length: usize) -> i32 {
    ffi_guard("hc4j_gpu_write", || {
        let bytes = checked_f32_bytes(ptr_in.is_null(), length)?;
        let data = unsafe { std::slice::from_raw_parts(ptr_in.cast::<u8>(), bytes) };
        memory::write_bytes(id, data)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_gpu_free(id: u64) -> i32 {
    ffi_guard("hc4j_gpu_free", || memory::free(id))
}

/// # Safety
/// `ptr_out` must point to `length` writable `f32` (or 4-byte) values for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_gpu_download(id: u64, ptr_out: *mut f32, length: usize) -> i32 {
    ffi_guard("hc4j_gpu_download", || {
        let bytes = checked_f32_bytes(ptr_out.is_null(), length)?;
        let out = unsafe { std::slice::from_raw_parts_mut(ptr_out.cast::<u8>(), bytes) };
        memory::download_bytes(id, out)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_mem_configure(vram_budget_bytes: u64, host_budget_bytes: u64) -> i32 {
    ffi_guard("hc4j_mem_configure", || {
        memory::configure(vram_budget_bytes, host_budget_bytes)
    })
}

/// # Safety
/// `out` must be non-null and point to writable memory for one [`MemStats`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_mem_stats(out: *mut MemStats) -> i32 {
    ffi_guard("hc4j_mem_stats", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null stats pointer"));
        }
        let stats = memory::stats()?;
        unsafe { out.write_unaligned(stats) };
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_mem_residency(id: u64) -> i32 {
    let mut code = 0;
    let status = ffi_guard("hc4j_mem_residency", || {
        code = memory::residency(id)?;
        Ok(())
    });
    if status == HC4J_SUCCESS { code } else { status }
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_mem_evict(id: u64) -> i32 {
    ffi_guard("hc4j_mem_evict", || memory::evict(id))
}
