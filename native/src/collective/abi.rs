use crate::error::{self, Hc4jError};

use super::backend::LinkInfo;

/// Opens a collective over `ranks` devices and writes its handle to `out`.
///
/// # Safety
/// `out` must point to one writable `u64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_create(ranks: u32, out: *mut u64) -> i32 {
    error::ffi_guard("hc4j_collective_create", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        let backend = super::create_backend(ranks)?;
        let handle = super::register_backend(backend);
        unsafe { out.write(handle) };
        Ok(())
    })
}

/// Releases a collective and every buffer it owns.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_destroy(handle: u64) -> i32 {
    error::ffi_guard("hc4j_collective_destroy", || super::destroy_backend(handle))
}

/// Number of ranks, or a negative status code.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_ranks(handle: u64) -> i32 {
    let mut ranks = 0;
    let status = error::ffi_guard("hc4j_collective_ranks", || {
        super::with_backend(handle, |b| {
            ranks = b.ranks() as i32;
            Ok(())
        })
    });
    if status == error::HC4J_SUCCESS { ranks } else { status }
}

/// Number of directed links in the topology, or a negative status code.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_link_count(handle: u64) -> i32 {
    let mut count = 0;
    let status = error::ffi_guard("hc4j_collective_link_count", || {
        super::with_backend(handle, |b| {
            count = b.links().len() as i32;
            Ok(())
        })
    });
    if status == error::HC4J_SUCCESS { count } else { status }
}

/// Copies link `index` into `out`.
///
/// # Safety
/// `out` must point to writable memory for one [`LinkInfo`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_link(handle: u64, index: u32, out: *mut LinkInfo) -> i32 {
    error::ffi_guard("hc4j_collective_link", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        super::with_backend(handle, |b| {
            let link = *b
                .links()
                .get(index as usize)
                .ok_or(Hc4jError::InvalidParam("link index out of range"))?;
            unsafe { out.write(link) };
            Ok(())
        })
    })
}

/// Writes rank `rank`'s adapter name into `buf` as UTF-8, truncating to `cap`.
/// Returns the number of bytes written, or a negative status code.
///
/// # Safety
/// `buf` must point to `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_device_name(handle: u64, rank: u32, buf: *mut u8, cap: u32) -> i32 {
    let mut len = 0;
    let status = error::ffi_guard("hc4j_collective_device_name", || {
        super::with_backend(handle, |b| {
            let name = b.device_name(rank as usize)?;
            if buf.is_null() || cap == 0 {
                return Err(Hc4jError::InvalidParam("null or empty name buffer"));
            }
            let bytes = name.as_bytes();
            let n = bytes.len().min(cap as usize);
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n) };
            len = n as i32;
            Ok(())
        })
    });
    if status == error::HC4J_SUCCESS { len } else { status }
}

/// Writes the collective backend's name ("vulkan-p2p" or "wgpu-staged") into `buf` as UTF-8,
/// truncating to `cap`. Returns the number of bytes written, or a negative status code.
///
/// # Safety
/// `buf` must point to `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_backend_name(handle: u64, buf: *mut u8, cap: u32) -> i32 {
    let mut len = 0;
    let status = error::ffi_guard("hc4j_collective_backend_name", || {
        super::with_backend(handle, |b| {
            let name = b.backend_name();
            if buf.is_null() || cap == 0 {
                return Err(Hc4jError::InvalidParam("null or empty name buffer"));
            }
            let bytes = name.as_bytes();
            let n = bytes.len().min(cap as usize);
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n) };
            len = n as i32;
            Ok(())
        })
    });
    if status == error::HC4J_SUCCESS { len } else { status }
}

/// Allocates `elements` f32 on `rank`, writing the buffer handle to `out`.
///
/// # Safety
/// `out` must point to one writable `u64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_alloc(handle: u64, rank: u32, elements: u64, out: *mut u64) -> i32 {
    error::ffi_guard("hc4j_collective_alloc", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        super::with_backend(handle, |b| {
            let id = b.alloc(rank as usize, elements)?;
            unsafe { out.write(id) };
            Ok(())
        })
    })
}

/// Releases a collective buffer.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_free(handle: u64, buffer_id: u64) -> i32 {
    error::ffi_guard("hc4j_collective_free", || {
        super::with_backend(handle, |b| b.free(buffer_id))
    })
}

/// Writes `data` into `buffer_id`.
///
/// # Safety
/// `ptr` must point to `elements` readable `f32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_write(handle: u64, buffer_id: u64, ptr: *const f32, elements: u64) -> i32 {
    error::ffi_guard("hc4j_collective_write", || {
        if ptr.is_null() && elements > 0 {
            return Err(Hc4jError::InvalidParam("null data pointer"));
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr, elements as usize) };
        super::with_backend(handle, |b| b.write(buffer_id, slice))
    })
}

/// Reads `buffer_id` into `out`.
///
/// # Safety
/// `out` must point to `elements` writable `f32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_read(handle: u64, buffer_id: u64, out: *mut f32, elements: u64) -> i32 {
    error::ffi_guard("hc4j_collective_read", || {
        if out.is_null() && elements > 0 {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(out, elements as usize) };
        super::with_backend(handle, |b| b.read(buffer_id, slice))
    })
}

/// Reduces `buffers` in place by element-wise sum across the ring.
///
/// # Safety
/// `ptr_buffers` must point to `count` readable `u64` buffer IDs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_allreduce_sum(handle: u64, ptr_buffers: *const u64, count: u32) -> i32 {
    error::ffi_guard("hc4j_collective_allreduce_sum", || {
        if ptr_buffers.is_null() && count > 0 {
            return Err(Hc4jError::InvalidParam("null buffers pointer"));
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr_buffers, count as usize) };
        super::with_backend(handle, |b| b.allreduce_sum(slice))
    })
}
