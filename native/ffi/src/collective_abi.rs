//! C-ABI entry points for multi-GPU collective communication, topology
//! inspection, fused GEMM+bias+activation, Split-KV decode attention,
//! lossless online-softmax Split-KV merging, and multi-threaded SIMD CPU-in-Place kernels.

use hc4rs::collective::{
    self, cpu_gemm_f32, cpu_gemm_f32_f16, cpu_mlp_slice_f32, cpu_split_kv_attn_f32, merge_split_kv_f32,
};
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

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_elem_mul(
    handle: u64,
    rank: u32,
    buf_a: u64,
    buf_b: u64,
    buf_out: u64,
    count: u32,
) -> i32 {
    ffi_guard("hc4j_collective_elem_mul", || {
        collective::with_backend(handle, |b| b.elem_mul(rank as usize, buf_a, buf_b, buf_out, count))
    })
}

/// # Safety
/// All matrix pointers must be valid for the respective `[M,K]`, `[K,N]` (or transposed), and `[M,N]` shapes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_cpu_gemm_f32(
    ptr_a: *const f32,
    ptr_b: *const f32,
    ptr_out: *mut f32,
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
) -> i32 {
    ffi_guard("hc4j_cpu_gemm_f32", || {
        if ptr_a.is_null() || ptr_b.is_null() || ptr_out.is_null() {
            return Err(Hc4jError::InvalidParam("null pointer in hc4j_cpu_gemm_f32"));
        }
        let mu = m as usize;
        let nu = n as usize;
        let ku = k as usize;
        let a_len = mu.checked_mul(ku).ok_or(Hc4jError::InvalidParam("A size overflow"))?;
        let b_len = ku.checked_mul(nu).ok_or(Hc4jError::InvalidParam("B size overflow"))?;
        let c_len = mu.checked_mul(nu).ok_or(Hc4jError::InvalidParam("out size overflow"))?;
        let a = unsafe { std::slice::from_raw_parts(ptr_a, a_len) };
        let b = unsafe { std::slice::from_raw_parts(ptr_b, b_len) };
        let out = unsafe { std::slice::from_raw_parts_mut(ptr_out, c_len) };
        cpu_gemm_f32(a, b, out, mu, nu, ku, flags)
    })
}

/// # Safety
/// `ptr_b_f16` must point to `k * n` IEEE-754 `u16` half elements matching `flags`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_cpu_gemm_f32_f16(
    ptr_a: *const f32,
    ptr_b_f16: *const u16,
    ptr_out: *mut f32,
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
) -> i32 {
    ffi_guard("hc4j_cpu_gemm_f32_f16", || {
        if ptr_a.is_null() || ptr_b_f16.is_null() || ptr_out.is_null() {
            return Err(Hc4jError::InvalidParam("null pointer in hc4j_cpu_gemm_f32_f16"));
        }
        let mu = m as usize;
        let nu = n as usize;
        let ku = k as usize;
        let a_len = mu.checked_mul(ku).ok_or(Hc4jError::InvalidParam("A size overflow"))?;
        let b_len = ku.checked_mul(nu).ok_or(Hc4jError::InvalidParam("B size overflow"))?;
        let c_len = mu.checked_mul(nu).ok_or(Hc4jError::InvalidParam("out size overflow"))?;
        let a = unsafe { std::slice::from_raw_parts(ptr_a, a_len) };
        let b = unsafe { std::slice::from_raw_parts(ptr_b_f16, b_len) };
        let out = unsafe { std::slice::from_raw_parts_mut(ptr_out, c_len) };
        cpu_gemm_f32_f16(a, b, out, mu, nu, ku, flags)
    })
}

/// # Safety
/// Pointers must be valid for `x [m, d_in]`, `w1 [d_in, h]`, optional `b1 [h]`, `w2 [h, d_out]`, and `out [m, d_out]`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn hc4j_cpu_mlp_slice_f32(
    ptr_x: *const f32,
    ptr_w1: *const f32,
    ptr_b1: *const f32,
    ptr_w2: *const f32,
    ptr_out: *mut f32,
    m: u32,
    d_in: u32,
    h: u32,
    d_out: u32,
    h_start: u32,
    units: u32,
    act_code: i32,
) -> i32 {
    ffi_guard("hc4j_cpu_mlp_slice_f32", || {
        if ptr_x.is_null() || ptr_w1.is_null() || ptr_w2.is_null() || ptr_out.is_null() {
            return Err(Hc4jError::InvalidParam("null pointer in hc4j_cpu_mlp_slice_f32"));
        }
        let mu = m as usize;
        let di = d_in as usize;
        let hu = h as usize;
        let do_ = d_out as usize;
        let x_len = mu.checked_mul(di).ok_or(Hc4jError::InvalidParam("x size overflow"))?;
        let w1_len = di.checked_mul(hu).ok_or(Hc4jError::InvalidParam("w1 size overflow"))?;
        let w2_len = hu.checked_mul(do_).ok_or(Hc4jError::InvalidParam("w2 size overflow"))?;
        let out_len = mu.checked_mul(do_).ok_or(Hc4jError::InvalidParam("out size overflow"))?;

        let x = unsafe { std::slice::from_raw_parts(ptr_x, x_len) };
        let w1 = unsafe { std::slice::from_raw_parts(ptr_w1, w1_len) };
        let b1 = if ptr_b1.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(ptr_b1, hu) })
        };
        let w2 = unsafe { std::slice::from_raw_parts(ptr_w2, w2_len) };
        let out = unsafe { std::slice::from_raw_parts_mut(ptr_out, out_len) };
        cpu_mlp_slice_f32(
            x,
            w1,
            b1,
            w2,
            out,
            mu,
            di,
            hu,
            do_,
            h_start as usize,
            units as usize,
            act_code,
        )
    })
}

/// # Safety
/// Pointers must be valid for `q [num_heads, head_dim]`, `k_shard`/`v_shard [num_kv_heads, tokens, head_dim]`,
/// `out_o [num_heads, head_dim]`, and `out_stats [num_heads, 2]`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn hc4j_cpu_split_kv_attn_f32(
    ptr_q: *const f32,
    ptr_k_shard: *const f32,
    ptr_v_shard: *const f32,
    ptr_out_o: *mut f32,
    ptr_out_stats: *mut f32,
    tokens: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    scale_bits: u32,
) -> i32 {
    ffi_guard("hc4j_cpu_split_kv_attn_f32", || {
        if ptr_q.is_null()
            || ptr_k_shard.is_null()
            || ptr_v_shard.is_null()
            || ptr_out_o.is_null()
            || ptr_out_stats.is_null()
        {
            return Err(Hc4jError::InvalidParam("null pointer in hc4j_cpu_split_kv_attn_f32"));
        }
        let tu = tokens as usize;
        let nh = num_heads as usize;
        let nkv = num_kv_heads as usize;
        let hd = head_dim as usize;
        let q_len = nh.checked_mul(hd).ok_or(Hc4jError::InvalidParam("q size overflow"))?;
        let kv_len = nkv
            .checked_mul(tu)
            .and_then(|v| v.checked_mul(hd))
            .ok_or(Hc4jError::InvalidParam("kv size overflow"))?;
        let s_len = nh.checked_mul(2).ok_or(Hc4jError::InvalidParam("stats size overflow"))?;
        let q = unsafe { std::slice::from_raw_parts(ptr_q, q_len) };
        let k_shard = unsafe { std::slice::from_raw_parts(ptr_k_shard, kv_len) };
        let v_shard = unsafe { std::slice::from_raw_parts(ptr_v_shard, kv_len) };
        let out_o = unsafe { std::slice::from_raw_parts_mut(ptr_out_o, q_len) };
        let out_stats = unsafe { std::slice::from_raw_parts_mut(ptr_out_stats, s_len) };
        cpu_split_kv_attn_f32(
            q,
            k_shard,
            v_shard,
            out_o,
            out_stats,
            tu,
            nh,
            nkv,
            hd,
            f32::from_bits(scale_bits),
        )
    })
}
