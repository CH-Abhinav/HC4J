use crate::error::{self, Hc4jError};

use super::backend::{DeviceProfileC, LinkInfo};

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

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IslandHeaderC {
    pub id: u32,
    pub leader_rank: u32,
    pub internal_transport: u32,
    pub member_count: u32,
}

/// Copies rank `rank`'s device profile into `out`.
///
/// # Safety
/// `out` must point to writable memory for one [`DeviceProfileC`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_device_profile(
    handle: u64,
    rank: u32,
    out: *mut DeviceProfileC,
) -> i32 {
    error::ffi_guard("hc4j_collective_device_profile", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        super::with_backend(handle, |b| {
            let prof = b
                .device_profiles()
                .iter()
                .find(|p| p.rank == rank)
                .copied()
                .ok_or(Hc4jError::InvalidParam("rank out of range"))?;
            unsafe { out.write(prof) };
            Ok(())
        })
    })
}

/// Writes planned ring order into `out_ranks` up to `cap` capacity.
/// Returns the number of ranks written, or a negative status code.
///
/// # Safety
/// `out_ranks` must point to writable memory for `cap` `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_ring_order(
    handle: u64,
    out_ranks: *mut u32,
    cap: u32,
) -> i32 {
    let mut written = 0;
    let status = error::ffi_guard("hc4j_collective_ring_order", || {
        if out_ranks.is_null() && cap > 0 {
            return Err(Hc4jError::InvalidParam("null out_ranks pointer"));
        }
        super::with_backend(handle, |b| {
            let order = b.ring_order();
            let count = (order.len() as u32).min(cap);
            if count > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(order.as_ptr(), out_ranks, count as usize);
                }
            }
            written = count as i32;
            Ok(())
        })
    });
    if status == error::HC4J_SUCCESS {
        written
    } else {
        status
    }
}

/// Number of detected islands, or a negative status code.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_island_count(handle: u64) -> i32 {
    let mut count = 0;
    let status = error::ffi_guard("hc4j_collective_island_count", || {
        super::with_backend(handle, |b| {
            count = b.islands().len() as i32;
            Ok(())
        })
    });
    if status == error::HC4J_SUCCESS {
        count
    } else {
        status
    }
}

/// Writes island `index` header into `out_header` and member ranks into `out_ranks` up to `cap`.
/// Returns the number of member ranks written, or a negative status code.
///
/// # Safety
/// `out_header` must point to writable memory for one [`IslandHeaderC`].
/// `out_ranks` must point to writable memory for `cap` `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_island(
    handle: u64,
    index: u32,
    out_header: *mut IslandHeaderC,
    out_ranks: *mut u32,
    cap: u32,
) -> i32 {
    let mut written = 0;
    let status = error::ffi_guard("hc4j_collective_island", || {
        super::with_backend(handle, |b| {
            let islands = b.islands();
            let isl = islands
                .get(index as usize)
                .ok_or(Hc4jError::InvalidParam("island index out of range"))?;

            if !out_header.is_null() {
                let header = IslandHeaderC {
                    id: isl.id,
                    leader_rank: isl.leader_rank,
                    internal_transport: isl.internal_transport,
                    member_count: isl.members.len() as u32,
                };
                unsafe { out_header.write(header) };
            }

            let count = (isl.members.len() as u32).min(cap);
            if !out_ranks.is_null() && count > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(isl.members.as_ptr(), out_ranks, count as usize);
                }
            }
            written = count as i32;
            Ok(())
        })
    });
    if status == error::HC4J_SUCCESS {
        written
    } else {
        status
    }
}
