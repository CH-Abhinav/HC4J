//! C-ABI entry points for multi-GPU collective communication, topology
//! inspection, fused GEMM+bias+activation, Split-KV decode attention, and
//! lossless online-softmax Split-KV merging (20 symbols).

use hc4rs::collective::{self, merge_split_kv_f32};
use hc4rs::engine::error::{HC4J_SUCCESS, Hc4jError, ffi_guard};
use hc4rs::{DeviceProfileC, LinkInfo};

/// Header describing a detected collective fast island (`#[repr(C)]`, 16 B).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IslandHeaderC {
    pub id: u32,
    pub leader_rank: u32,
    pub internal_transport: u32,
    pub member_count: u32,
}

/// # Safety
/// `out` must point to writable storage for one `u64` handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_create(ranks: u32, out: *mut u64) -> i32 {
    ffi_guard("hc4j_collective_create", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        let backend = collective::create_backend(ranks)?;
        let handle = collective::register_backend(backend);
        unsafe { out.write_unaligned(handle) };
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_destroy(handle: u64) -> i32 {
    ffi_guard("hc4j_collective_destroy", || collective::destroy_backend(handle))
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_ranks(handle: u64) -> i32 {
    let mut ranks = 0;
    let status = ffi_guard("hc4j_collective_ranks", || {
        collective::with_backend(handle, |b| {
            ranks = b.ranks() as i32;
            Ok(())
        })
    });
    if status == HC4J_SUCCESS { ranks } else { status }
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_link_count(handle: u64) -> i32 {
    let mut count = 0;
    let status = ffi_guard("hc4j_collective_link_count", || {
        collective::with_backend(handle, |b| {
            count = b.links().len() as i32;
            Ok(())
        })
    });
    if status == HC4J_SUCCESS { count } else { status }
}

/// # Safety
/// `out` must point to writable storage for one [`LinkInfo`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_link(handle: u64, index: u32, out: *mut LinkInfo) -> i32 {
    ffi_guard("hc4j_collective_link", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        collective::with_backend(handle, |b| {
            let link = *b
                .links()
                .get(index as usize)
                .ok_or(Hc4jError::InvalidParam("link index out of range"))?;
            unsafe { out.write_unaligned(link) };
            Ok(())
        })
    })
}

/// # Safety
/// `buf` must point to `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_device_name(handle: u64, rank: u32, buf: *mut u8, cap: u32) -> i32 {
    let mut len = 0;
    let status = ffi_guard("hc4j_collective_device_name", || {
        collective::with_backend(handle, |b| {
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
    if status == HC4J_SUCCESS { len } else { status }
}

/// # Safety
/// `buf` must point to `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_backend_name(handle: u64, buf: *mut u8, cap: u32) -> i32 {
    let mut len = 0;
    let status = ffi_guard("hc4j_collective_backend_name", || {
        collective::with_backend(handle, |b| {
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
    if status == HC4J_SUCCESS { len } else { status }
}

/// # Safety
/// `out` must point to writable storage for one `u64` buffer ID.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_alloc(handle: u64, rank: u32, elements: u64, out: *mut u64) -> i32 {
    ffi_guard("hc4j_collective_alloc", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        collective::with_backend(handle, |b| {
            let id = b.alloc(rank as usize, elements)?;
            unsafe { out.write_unaligned(id) };
            Ok(())
        })
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_free(handle: u64, buffer_id: u64) -> i32 {
    ffi_guard("hc4j_collective_free", || {
        collective::with_backend(handle, |b| b.free(buffer_id))
    })
}

/// # Safety
/// `ptr` must point to `elements` readable `f32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_write(
    handle: u64,
    buffer_id: u64,
    ptr: *const f32,
    elements: u64,
) -> i32 {
    ffi_guard("hc4j_collective_write", || {
        if ptr.is_null() && elements > 0 {
            return Err(Hc4jError::InvalidParam("null data pointer"));
        }
        let slice = if elements == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(ptr, elements as usize) }
        };
        collective::with_backend(handle, |b| b.write(buffer_id, slice))
    })
}

/// # Safety
/// `out` must point to `elements` writable `f32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_read(handle: u64, buffer_id: u64, out: *mut f32, elements: u64) -> i32 {
    ffi_guard("hc4j_collective_read", || {
        if out.is_null() && elements > 0 {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        let slice = if elements == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(out, elements as usize) }
        };
        collective::with_backend(handle, |b| b.read(buffer_id, slice))
    })
}

/// # Safety
/// `ptr_buffers` must point to `count` readable `u64` collective buffer IDs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_allreduce_sum(
    handle: u64,
    ptr_buffers: *const u64,
    count: u32,
) -> i32 {
    ffi_guard("hc4j_collective_allreduce_sum", || {
        if ptr_buffers.is_null() && count > 0 {
            return Err(Hc4jError::InvalidParam("null buffers pointer"));
        }
        let slice = if count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(ptr_buffers, count as usize) }
        };
        collective::with_backend(handle, |b| b.allreduce_sum(slice))
    })
}

/// # Safety
/// `ptr_buffers` must point to `count` readable `u64` collective buffer IDs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_allreduce_sum_subset(
    handle: u64,
    ptr_buffers: *const u64,
    count: u32,
) -> i32 {
    ffi_guard("hc4j_collective_allreduce_sum_subset", || {
        if ptr_buffers.is_null() && count > 0 {
            return Err(Hc4jError::InvalidParam("null buffers pointer"));
        }
        let slice = if count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(ptr_buffers, count as usize) }
        };
        collective::with_backend(handle, |b| b.allreduce_sum_subset(slice))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_gemm_bias_act(
    handle: u64,
    rank: u32,
    buf_a: u64,
    buf_b: u64,
    buf_bias: u64,
    buf_c: u64,
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
) -> i32 {
    ffi_guard("hc4j_collective_gemm_bias_act", || {
        collective::with_backend(handle, |b| {
            b.gemm_bias_act(rank as usize, buf_a, buf_b, buf_bias, buf_c, m, n, k, flags)
        })
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_split_kv_attn(
    handle: u64,
    rank: u32,
    buf_q: u64,
    buf_k: u64,
    buf_v: u64,
    buf_out_o: u64,
    buf_out_stats: u64,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    local_tokens: u32,
    scale_bits: u32,
) -> i32 {
    ffi_guard("hc4j_collective_split_kv_attn", || {
        let scale = f32::from_bits(scale_bits);
        collective::with_backend(handle, |b| {
            b.split_kv_attn(
                rank as usize,
                buf_q,
                buf_k,
                buf_v,
                buf_out_o,
                buf_out_stats,
                num_heads,
                num_kv_heads,
                head_dim,
                local_tokens,
                scale,
            )
        })
    })
}

/// # Safety
/// `out` must point to writable storage for one [`DeviceProfileC`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_device_profile(
    handle: u64,
    rank: u32,
    out: *mut DeviceProfileC,
) -> i32 {
    ffi_guard("hc4j_collective_device_profile", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        collective::with_backend(handle, |b| {
            let prof = b
                .device_profiles()
                .iter()
                .find(|p| p.rank == rank)
                .copied()
                .ok_or(Hc4jError::InvalidParam("rank out of range"))?;
            unsafe { out.write_unaligned(prof) };
            Ok(())
        })
    })
}

/// # Safety
/// If `cap > 0`, `out_ranks` must point to `cap` writable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_ring_order(
    handle: u64,
    out_ranks: *mut u32,
    cap: u32,
) -> i32 {
    let mut written = 0;
    let status = ffi_guard("hc4j_collective_ring_order", || {
        if out_ranks.is_null() && cap > 0 {
            return Err(Hc4jError::InvalidParam("null out_ranks pointer"));
        }
        collective::with_backend(handle, |b| {
            let order = b.ring_order();
            let count = (order.len()).min(cap as usize);
            if count > 0 {
                unsafe { std::ptr::copy_nonoverlapping(order.as_ptr(), out_ranks, count) };
            }
            written = count as i32;
            Ok(())
        })
    });
    if status == HC4J_SUCCESS { written } else { status }
}

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_island_count(handle: u64) -> i32 {
    let mut count = 0;
    let status = ffi_guard("hc4j_collective_island_count", || {
        collective::with_backend(handle, |b| {
            count = b.islands().len() as i32;
            Ok(())
        })
    });
    if status == HC4J_SUCCESS { count } else { status }
}

/// # Safety
/// `out_header` must point to writable [`IslandHeaderC`]; if `cap > 0`, `out_ranks`
/// must point to `cap` writable `u32` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_island(
    handle: u64,
    index: u32,
    out_header: *mut IslandHeaderC,
    out_ranks: *mut u32,
    cap: u32,
) -> i32 {
    let mut written = 0;
    let status = ffi_guard("hc4j_collective_island", || {
        if out_header.is_null() {
            return Err(Hc4jError::InvalidParam("null out_header pointer"));
        }
        if out_ranks.is_null() && cap > 0 {
            return Err(Hc4jError::InvalidParam("null out_ranks pointer"));
        }
        collective::with_backend(handle, |b| {
            let isl = b
                .islands()
                .get(index as usize)
                .ok_or(Hc4jError::InvalidParam("island index out of range"))?;
            let header = IslandHeaderC {
                id: isl.id,
                leader_rank: isl.leader_rank,
                internal_transport: isl.internal_transport,
                member_count: isl.members.len() as u32,
            };
            unsafe { out_header.write_unaligned(header) };
            let count = isl.members.len().min(cap as usize);
            if count > 0 {
                unsafe { std::ptr::copy_nonoverlapping(isl.members.as_ptr(), out_ranks, count) };
            }
            written = count as i32;
            Ok(())
        })
    });
    if status == HC4J_SUCCESS { written } else { status }
}

/// Lossless online-softmax merge over `worker_count` Split-KV worker partial
/// outputs (`ptr_partials_o[w]` of length `num_heads * head_dim`) and log-sum-exp
/// statistics (`ptr_partials_stats[w]` of length `2 * num_heads` stored as `[m, l]`),
/// writing `[num_heads, head_dim]` into `ptr_out`.
///
/// # Safety
/// `ptr_partials_o` and `ptr_partials_stats` must point to `worker_count` valid
/// non-null `f32` buffers sized `num_heads * head_dim` and `2 * num_heads`
/// respectively; `ptr_out` must point to `num_heads * head_dim` writable `f32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_merge_split_kv_f32(
    num_heads: u32,
    head_dim: u32,
    worker_count: u32,
    ptr_partials_o: *const *const f32,
    ptr_partials_stats: *const *const f32,
    ptr_out: *mut f32,
) -> i32 {
    ffi_guard("hc4j_collective_merge_split_kv_f32", || {
        if num_heads == 0 || head_dim == 0 || worker_count == 0 {
            return Err(Hc4jError::InvalidParam("num_heads, head_dim, and worker_count must be positive"));
        }
        if ptr_partials_o.is_null() || ptr_partials_stats.is_null() || ptr_out.is_null() {
            return Err(Hc4jError::InvalidParam("null pointer in Split-KV merge"));
        }
        let o_len = (num_heads as usize)
            .checked_mul(head_dim as usize)
            .ok_or(Hc4jError::InvalidParam("output element count overflows"))?;
        let stats_len = (num_heads as usize)
            .checked_mul(2)
            .ok_or(Hc4jError::InvalidParam("stats element count overflows"))?;
        let wc = worker_count as usize;
        let o_ptrs = unsafe { std::slice::from_raw_parts(ptr_partials_o, wc) };
        let s_ptrs = unsafe { std::slice::from_raw_parts(ptr_partials_stats, wc) };

        let mut partials_o = Vec::with_capacity(wc);
        let mut partials_stats = Vec::with_capacity(wc);
        for (&po, &ps) in o_ptrs.iter().zip(s_ptrs.iter()) {
            if po.is_null() || ps.is_null() {
                return Err(Hc4jError::InvalidParam("null worker buffer pointer in Split-KV merge"));
            }
            partials_o.push(unsafe { std::slice::from_raw_parts(po, o_len) });
            partials_stats.push(unsafe { std::slice::from_raw_parts(ps, stats_len) });
        }

        let out_slice = unsafe { std::slice::from_raw_parts_mut(ptr_out, o_len) };
        merge_split_kv_f32(num_heads, head_dim, &partials_o, &partials_stats, out_slice)
    })
}
