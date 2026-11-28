//! Balanced uneven chunk partitioning and zero-copy bidirectional pipelined Ring / Hierarchical AllReduce.

use ash::vk;

use crate::collective::backend::LinkInfo;
use crate::collective::planner::{IslandInfo, solve_optimal_ring};
use crate::collective::vulkan::device::GpuNode;
use crate::collective::vulkan::pipeline::{ReducePipeline, submit_cb_with_fence};
use crate::collective::vulkan::topology::TopoGraph;
use crate::collective::vulkan::transport::{EDGE_PIPELINE_SLOTS, EdgeChannelCache};
use crate::engine::error::{Hc4jError, Hc4jResult};

pub(crate) use crate::collective::planner::chunk_range;

#[inline]
fn default_slice_count(max_chunk_bytes: vk::DeviceSize, dual_queue: bool) -> usize {
    const MIB: vk::DeviceSize = 1024 * 1024;
    if max_chunk_bytes < MIB {
        if dual_queue { 2 } else { 1 }
    } else if max_chunk_bytes < 4 * MIB {
        4
    } else if max_chunk_bytes < 16 * MIB {
        8
    } else {
        16
    }
}

#[inline]
fn sub_slice_count(max_chunk_bytes: vk::DeviceSize, max_chunk_len: usize, dual_queue: bool) -> usize {
    let raw = std::env::var("HC4J_COLLECTIVE_SLICES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .map(|n| n.clamp(1, 32))
        .unwrap_or_else(|| default_slice_count(max_chunk_bytes, dual_queue));
    raw.min(max_chunk_len.max(1))
}

#[inline]
fn direct_reduce_enabled() -> bool {
    !std::env::var("HC4J_COLLECTIVE_DIRECT_REDUCE")
        .is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false"))
}

fn drain_ranks_idle(nodes: &[GpuNode], order: &[usize]) {
    for &rank in order {
        unsafe {
            nodes[rank].device.device_wait_idle().ok();
        }
    }
}

fn wait_slot_fences(
    nodes: &[GpuNode],
    order: &[usize],
    pipelines: &[Option<&mut ReducePipeline>],
    slot: usize,
    is_recv: bool,
    active: &mut [[bool; EDGE_PIPELINE_SLOTS]],
) -> Hc4jResult<()> {
    for (pos, flags) in active.iter_mut().enumerate() {
        if flags[slot] {
            let rank = order[pos];
            let sync = pipelines[rank].as_deref().unwrap().sync(slot);
            let fence = if is_recv { sync.recv_fence } else { sync.send_fence };
            unsafe {
                nodes[rank]
                    .device
                    .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                    .map_err(|e| Hc4jError::Device(format!("wait_for_fences failed: {e}")))?;
            }
            flags[slot] = false;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn submit_step_a_send_slice(
    nodes: &[GpuNode],
    pipelines: &[Option<&mut ReducePipeline>],
    channels: &EdgeChannelCache,
    order: &[usize],
    user_buffers: &[vk::Buffer],
    chunk_ranges: &[(usize, usize)],
    num_slices: usize,
    slice_idx: usize,
    slot: usize,
    send_active: &mut [[bool; EDGE_PIPELINE_SLOTS]],
) -> Hc4jResult<()> {
    let k_len = order.len();
    for (pos, &(chunk_start, chunk_len)) in chunk_ranges.iter().enumerate() {
        let (rel_start, slice_len) = chunk_range(chunk_len, num_slices, slice_idx);
        if slice_len == 0 {
            continue;
        }
        let i = order[pos];
        let next = order[(pos + 1) % k_len];
        let src_offset_bytes = ((chunk_start + rel_start) * 4) as vk::DeviceSize;
        let slice_bytes = (slice_len * 4) as vk::DeviceSize;
        let edge_src_buf = channels.edge(i, next).slot(slot).src_buffer();
        let sync = pipelines[i].as_deref().unwrap().sync(slot);

        submit_cb_with_fence(&nodes[i], nodes[i].send_queue, sync.send_cb, sync.send_fence, |cb| unsafe {
            let region = vk::BufferCopy {
                src_offset: src_offset_bytes,
                dst_offset: 0,
                size: slice_bytes,
            };
            nodes[i].device.cmd_copy_buffer(cb, user_buffers[i], edge_src_buf, &[region]);
        })?;
        send_active[pos][slot] = true;
    }
    Ok(())
}

#[inline]
fn copy_staged_slice_hosts(
    channels: &EdgeChannelCache,
    order: &[usize],
    chunk_ranges: &[(usize, usize)],
    num_slices: usize,
    slice_idx: usize,
    slot: usize,
) {
    let k_len = order.len();
    for (pos, &(_, chunk_len)) in chunk_ranges.iter().enumerate() {
        let (_, slice_len) = chunk_range(chunk_len, num_slices, slice_idx);
        if slice_len == 0 {
            continue;
        }
        let i = order[pos];
        let next = order[(pos + 1) % k_len];
        channels
            .edge(i, next)
            .slot(slot)
            .copy_host_if_staged((slice_len * 4) as vk::DeviceSize);
    }
}

#[allow(clippy::too_many_arguments)]
fn submit_step_b_recv_slice(
    nodes: &[GpuNode],
    pipelines: &mut [Option<&mut ReducePipeline>],
    channels: &EdgeChannelCache,
    order: &[usize],
    user_buffers: &[vk::Buffer],
    chunk_ranges: &[(usize, usize)],
    num_slices: usize,
    slice_idx: usize,
    slot: usize,
    reduce: bool,
    direct_reduce: bool,
    recv_active: &mut [[bool; EDGE_PIPELINE_SLOTS]],
) -> Hc4jResult<()> {
    let k_len = order.len();
    for (pos, &(chunk_start, chunk_len)) in chunk_ranges.iter().enumerate() {
        let (rel_start, slice_len) = chunk_range(chunk_len, num_slices, slice_idx);
        if slice_len == 0 {
            continue;
        }
        let i = order[pos];
        let next_pos = (pos + 1) % k_len;
        let next = order[next_pos];
        let dst_offset_bytes = ((chunk_start + rel_start) * 4) as vk::DeviceSize;
        let slice_bytes = (slice_len * 4) as vk::DeviceSize;
        let edge_slot = channels.edge(i, next).slot(slot);
        let edge_dst_buf = edge_slot.dst_buffer();
        let use_direct = direct_reduce && edge_slot.dst_supports_storage();
        let recv_pipe = pipelines[next].as_deref_mut().unwrap();
        let sync = *recv_pipe.sync(slot);

        submit_cb_with_fence(&nodes[next], nodes[next].queue, sync.recv_cb, sync.recv_fence, |cb| {
            if reduce {
                if use_direct {
                    recv_pipe.record_direct_reduce(
                        &nodes[next],
                        cb,
                        slot,
                        user_buffers[next],
                        dst_offset_bytes,
                        edge_dst_buf,
                        slice_len as u32,
                    );
                } else {
                    recv_pipe.record_copy_and_reduce(
                        &nodes[next],
                        cb,
                        slot,
                        user_buffers[next],
                        dst_offset_bytes,
                        edge_dst_buf,
                        slice_len as u32,
                    );
                }
            } else {
                unsafe {
                    let region = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: dst_offset_bytes,
                        size: slice_bytes,
                    };
                    nodes[next]
                        .device
                        .cmd_copy_buffer(cb, edge_dst_buf, user_buffers[next], &[region]);

                    let barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ);
                    nodes[next].device.cmd_pipeline_barrier(
                        cb,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                }
            }
        })?;
        recv_active[next_pos][slot] = true;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_ring_phase(
    nodes: &[GpuNode],
    pipelines: &mut [Option<&mut ReducePipeline>],
    channels: &EdgeChannelCache,
    order: &[usize],
    user_buffers: &[vk::Buffer],
    total_elements: usize,
    num_slices: usize,
    reduce: bool,
    direct_reduce: bool,
) -> Hc4jResult<()> {
    let k_len = order.len();
    let mut chunk_ranges = Vec::with_capacity(k_len);
    let mut send_active = vec![[false; EDGE_PIPELINE_SLOTS]; k_len];
    let mut recv_active = vec![[false; EDGE_PIPELINE_SLOTS]; k_len];

    for round in 0..(k_len - 1) {
        chunk_ranges.clear();
        for pos in 0..k_len {
            let chunk_idx = if reduce {
                (pos + k_len - round) % k_len
            } else {
                (pos + 1 + k_len - round) % k_len
            };
            chunk_ranges.push(chunk_range(total_elements, k_len, chunk_idx));
        }

        // Kick off slice 0 Step A on slot 0 and wait before first Step B read.
        submit_step_a_send_slice(
            nodes,
            pipelines,
            channels,
            order,
            user_buffers,
            &chunk_ranges,
            num_slices,
            0,
            0,
            &mut send_active,
        )?;
        wait_slot_fences(nodes, order, pipelines, 0, false, &mut send_active)?;
        copy_staged_slice_hosts(channels, order, &chunk_ranges, num_slices, 0, 0);

        for s in 0..num_slices {
            let curr_slot = s % EDGE_PIPELINE_SLOTS;
            submit_step_b_recv_slice(
                nodes,
                pipelines,
                channels,
                order,
                user_buffers,
                &chunk_ranges,
                num_slices,
                s,
                curr_slot,
                reduce,
                direct_reduce,
                &mut recv_active,
            )?;

            if s + 1 < num_slices {
                let next_slot = (s + 1) % EDGE_PIPELINE_SLOTS;
                // Ensure `next_slot` finished any prior Step B before reusing its buffers.
                wait_slot_fences(nodes, order, pipelines, next_slot, true, &mut recv_active)?;
                submit_step_a_send_slice(
                    nodes,
                    pipelines,
                    channels,
                    order,
                    user_buffers,
                    &chunk_ranges,
                    num_slices,
                    s + 1,
                    next_slot,
                    &mut send_active,
                )?;
                wait_slot_fences(nodes, order, pipelines, next_slot, false, &mut send_active)?;
                copy_staged_slice_hosts(channels, order, &chunk_ranges, num_slices, s + 1, next_slot);
            }
        }

        // Wait for all remaining Step B work before next round reads updated user buffers.
        wait_slot_fences(nodes, order, pipelines, 0, true, &mut recv_active)?;
        wait_slot_fences(nodes, order, pipelines, 1, true, &mut recv_active)?;
    }

    Ok(())
}

/// Executes a bidirectional, double-buffered ordered Ring AllReduce (sum) across `order`.
pub(crate) fn ring_allreduce_ordered(
    nodes: &[GpuNode],
    pipelines: &mut [Option<&mut ReducePipeline>],
    channels: &mut EdgeChannelCache,
    graph: &TopoGraph,
    user_buffers: &[vk::Buffer],
    total_elements: usize,
    order: &[usize],
) -> Hc4jResult<()> {
    let k_len = order.len();
    if k_len <= 1 || total_elements == 0 {
        return Ok(());
    }

    let max_chunk_len = (total_elements / k_len) + usize::from(!total_elements.is_multiple_of(k_len));
    let max_chunk_bytes = (max_chunk_len * 4) as vk::DeviceSize;
    let dual_queue = order.iter().all(|&r| nodes[r].has_dual_queue());
    let num_slices = sub_slice_count(max_chunk_bytes, max_chunk_len, dual_queue);
    let max_slice_len = (max_chunk_len / num_slices) + usize::from(!max_chunk_len.is_multiple_of(num_slices));
    let max_slice_bytes = (max_slice_len * 4) as vk::DeviceSize;
    let direct_reduce = direct_reduce_enabled();

    for &rank in order {
        pipelines[rank]
            .as_deref_mut()
            .unwrap()
            .ensure_recv_slots(&nodes[rank], max_slice_bytes)?;
    }
    for pos in 0..k_len {
        let i = order[pos];
        let next = order[(pos + 1) % k_len];
        let tier = *graph.tier(i, next);
        if channels.ensure_edge(&nodes[i], &nodes[next], i, next, tier, max_slice_bytes)? {
            pipelines[next].as_deref_mut().unwrap().invalidate_desc_cache();
        }
    }

    let res = (|| -> Hc4jResult<()> {
        // Phase 1: Scatter-Reduce (k_len - 1 rounds)
        run_ring_phase(
            nodes,
            pipelines,
            channels,
            order,
            user_buffers,
            total_elements,
            num_slices,
            true,
            direct_reduce,
        )?;
        // Phase 2: Allgather (k_len - 1 rounds)
        run_ring_phase(
            nodes,
            pipelines,
            channels,
            order,
            user_buffers,
            total_elements,
            num_slices,
            false,
            direct_reduce,
        )
    })();

    if res.is_err() {
        drain_ranks_idle(nodes, order);
    }
    res
}

/// Transfers `total_elements` from `src_rank` -> `dst_rank` across a persistent edge channel.
/// When `reduce_into_dst` is true, accumulates into `dst_buf` via fused copy+reduce CB;
/// otherwise copies directly into `dst_buf`.
#[allow(clippy::too_many_arguments)]
fn transfer_edge_persistent(
    nodes: &[GpuNode],
    pipelines: &mut [Option<&mut ReducePipeline>],
    channels: &mut EdgeChannelCache,
    graph: &TopoGraph,
    src_rank: usize,
    dst_rank: usize,
    src_buf: vk::Buffer,
    dst_buf: vk::Buffer,
    total_elements: usize,
    reduce_into_dst: bool,
) -> Hc4jResult<()> {
    if total_elements == 0 {
        return Ok(());
    }
    let total_bytes = (total_elements * 4) as vk::DeviceSize;
    let dual_queue = nodes[src_rank].has_dual_queue() && nodes[dst_rank].has_dual_queue();
    let num_slices = sub_slice_count(total_bytes, total_elements, dual_queue);
    let max_slice_len = (total_elements / num_slices) + usize::from(!total_elements.is_multiple_of(num_slices));
    let max_slice_bytes = (max_slice_len * 4) as vk::DeviceSize;
    let direct_reduce = direct_reduce_enabled();

    if reduce_into_dst {
        pipelines[dst_rank]
            .as_deref_mut()
            .unwrap()
            .ensure_recv_slots(&nodes[dst_rank], max_slice_bytes)?;
    }
    let tier = *graph.tier(src_rank, dst_rank);
    if channels.ensure_edge(&nodes[src_rank], &nodes[dst_rank], src_rank, dst_rank, tier, max_slice_bytes)? {
        pipelines[dst_rank].as_deref_mut().unwrap().invalidate_desc_cache();
    }

    let edge = channels.edge(src_rank, dst_rank);

    for s in 0..num_slices {
        let (elem_start, slice_len) = chunk_range(total_elements, num_slices, s);
        if slice_len == 0 {
            continue;
        }
        let slot = s % EDGE_PIPELINE_SLOTS;
        let offset_bytes = (elem_start * 4) as vk::DeviceSize;
        let slice_bytes = (slice_len * 4) as vk::DeviceSize;
        let slot_ref = edge.slot(slot);
        let src_sync = *pipelines[src_rank].as_deref().unwrap().sync(slot);
        let dst_sync = *pipelines[dst_rank].as_deref().unwrap().sync(slot);

        submit_cb_with_fence(&nodes[src_rank], nodes[src_rank].send_queue, src_sync.send_cb, src_sync.send_fence, |cb| unsafe {
            let region = vk::BufferCopy {
                src_offset: offset_bytes,
                dst_offset: 0,
                size: slice_bytes,
            };
            nodes[src_rank]
                .device
                .cmd_copy_buffer(cb, src_buf, slot_ref.src_buffer(), &[region]);
        })?;
        unsafe {
            nodes[src_rank]
                .device
                .wait_for_fences(std::slice::from_ref(&src_sync.send_fence), true, u64::MAX)
                .map_err(|e| Hc4jError::Device(format!("wait_for_fences failed: {e}")))?;
        }
        slot_ref.copy_host_if_staged(slice_bytes);

        let use_direct = direct_reduce && slot_ref.dst_supports_storage();
        let dst_pipe = pipelines[dst_rank].as_deref_mut().unwrap();
        submit_cb_with_fence(&nodes[dst_rank], nodes[dst_rank].queue, dst_sync.recv_cb, dst_sync.recv_fence, |cb| {
            if reduce_into_dst {
                if use_direct {
                    dst_pipe.record_direct_reduce(
                        &nodes[dst_rank],
                        cb,
                        slot,
                        dst_buf,
                        offset_bytes,
                        slot_ref.dst_buffer(),
                        slice_len as u32,
                    );
                } else {
                    dst_pipe.record_copy_and_reduce(
                        &nodes[dst_rank],
                        cb,
                        slot,
                        dst_buf,
                        offset_bytes,
                        slot_ref.dst_buffer(),
                        slice_len as u32,
                    );
                }
            } else {
                unsafe {
                    let region = vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: offset_bytes,
                        size: slice_bytes,
                    };
                    nodes[dst_rank]
                        .device
                        .cmd_copy_buffer(cb, slot_ref.dst_buffer(), dst_buf, &[region]);
                }
            }
        })?;
        unsafe {
            nodes[dst_rank]
                .device
                .wait_for_fences(std::slice::from_ref(&dst_sync.recv_fence), true, u64::MAX)
                .map_err(|e| Hc4jError::Device(format!("wait_for_fences failed: {e}")))?;
        }
    }

    Ok(())
}

/// Executes Hierarchical Fast-Island AllReduce:
/// Phase 1: Intra-Island Reduce to Leader
/// Phase 2: Inter-Island Leader Ring AllReduce
/// Phase 3: Intra-Island Broadcast from Leader
pub(crate) fn hierarchical_allreduce(
    nodes: &[GpuNode],
    pipelines: &mut [Option<&mut ReducePipeline>],
    channels: &mut EdgeChannelCache,
    graph: &TopoGraph,
    user_buffers: &[vk::Buffer],
    total_elements: usize,
    islands: &[IslandInfo],
) -> Hc4jResult<()> {
    if nodes.len() <= 1 || total_elements == 0 {
        return Ok(());
    }

    // Phase 1: Intra-Island Reduce to Leader
    for isl in islands {
        if isl.members.len() > 1 {
            let leader = isl.leader_rank as usize;
            for &m in &isl.members {
                let member = m as usize;
                if member == leader {
                    continue;
                }
                transfer_edge_persistent(
                    nodes,
                    pipelines,
                    channels,
                    graph,
                    member,
                    leader,
                    user_buffers[member],
                    user_buffers[leader],
                    total_elements,
                    true,
                )?;
            }
        }
    }

    // Phase 2: Inter-Island Leader Optimal-Ring AllReduce
    let leaders: Vec<usize> = islands.iter().map(|isl| isl.leader_rank as usize).collect();
    if leaders.len() > 1 {
        let mut leader_links = Vec::new();
        for (i, &l_i) in leaders.iter().enumerate() {
            for (j, &l_j) in leaders.iter().enumerate() {
                #[allow(clippy::collapsible_if)]
                if i != j {
                    if let Some(link) = graph
                        .links
                        .iter()
                        .find(|l| l.src_rank == l_i as u32 && l.dst_rank == l_j as u32)
                    {
                        leader_links.push(LinkInfo {
                            src_rank: i as u32,
                            dst_rank: j as u32,
                            bandwidth_gbps: link.bandwidth_gbps,
                            transport: link.transport,
                            loopback: link.loopback,
                            latency_us: link.latency_us,
                        });
                    }
                }
            }
        }
        let leader_ring = solve_optimal_ring(leaders.len(), &leader_links);
        let leader_order: Vec<usize> = leader_ring
            .iter()
            .map(|&idx| leaders[idx as usize])
            .collect();

        ring_allreduce_ordered(
            nodes,
            pipelines,
            channels,
            graph,
            user_buffers,
            total_elements,
            &leader_order,
        )?;
    }

    // Phase 3: Intra-Island Broadcast from Leader
    for isl in islands {
        if isl.members.len() > 1 {
            let leader = isl.leader_rank as usize;
            for &m in &isl.members {
                let member = m as usize;
                if member == leader {
                    continue;
                }
                transfer_edge_persistent(
                    nodes,
                    pipelines,
                    channels,
                    graph,
                    leader,
                    member,
                    user_buffers[leader],
                    user_buffers[member],
                    total_elements,
                    false,
                )?;
            }
        }
    }

    Ok(())
}

/// Executes AllReduce across the collective using the selected algorithm
/// (auto, ring, or hierarchical) based on `HC4J_COLLECTIVE_ALGO`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_allreduce(
    nodes: &[GpuNode],
    pipelines: &mut [Option<&mut ReducePipeline>],
    channels: &mut EdgeChannelCache,
    graph: &TopoGraph,
    user_buffers: &[vk::Buffer],
    total_elements: usize,
    ring_order: &[u32],
    islands: &[IslandInfo],
) -> Hc4jResult<()> {
    if nodes.len() <= 1 || total_elements == 0 {
        return Ok(());
    }

    let algo_var = std::env::var("HC4J_COLLECTIVE_ALGO")
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_else(|_| "auto".to_string());

    let ring_order_usize: Vec<usize> = ring_order.iter().map(|&r| r as usize).collect();

    match algo_var.as_str() {
        "auto" | "" => {
            if islands.len() > 1 && islands.iter().any(|isl| isl.members.len() > 1) {
                hierarchical_allreduce(nodes, pipelines, channels, graph, user_buffers, total_elements, islands)
            } else {
                ring_allreduce_ordered(
                    nodes,
                    pipelines,
                    channels,
                    graph,
                    user_buffers,
                    total_elements,
                    &ring_order_usize,
                )
            }
        }
        "ring" => ring_allreduce_ordered(
            nodes,
            pipelines,
            channels,
            graph,
            user_buffers,
            total_elements,
            &ring_order_usize,
        ),
        "hierarchical" => {
            if islands.len() <= 1 {
                ring_allreduce_ordered(
                    nodes,
                    pipelines,
                    channels,
                    graph,
                    user_buffers,
                    total_elements,
                    &ring_order_usize,
                )
            } else {
                hierarchical_allreduce(nodes, pipelines, channels, graph, user_buffers, total_elements, islands)
            }
        }
        _ => Err(Hc4jError::InvalidParam(
            "HC4J_COLLECTIVE_ALGO must be auto, ring, or hierarchical",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_chunk_range_partitioning() {
        let test_cases = [
            (0, 1), (0, 2), (0, 4),
            (1, 1), (1, 2), (1, 4),
            (2, 1), (2, 2), (2, 4),
            (5, 1), (5, 2), (5, 3), (5, 4),
            (6, 1), (6, 2), (6, 3), (6, 4),
            (7, 1), (7, 2), (7, 3), (7, 4),
            (1003, 1), (1003, 2), (1003, 3), (1003, 4),
        ];

        for (total, ranks) in test_cases {
            let mut covered = vec![false; total];
            let mut total_len = 0;

            for chunk_idx in 0..ranks {
                let (start, len) = chunk_range(total, ranks, chunk_idx);
                total_len += len;
                if len > 0 {
                    assert!(
                        start + len <= total,
                        "chunk overflow for total={total}, ranks={ranks}, chunk={chunk_idx}"
                    );
                    for i in start..start + len {
                        assert!(
                            !covered[i],
                            "overlap detected at index {i} for total={total}, ranks={ranks}"
                        );
                        covered[i] = true;
                    }
                }
            }

            assert_eq!(
                total_len, total,
                "sum of chunk lengths does not match total for total={total}, ranks={ranks}"
            );
            assert!(
                covered.iter().all(|&c| c),
                "gap detected in chunk coverage for total={total}, ranks={ranks}"
            );
        }
    }

    #[test]
    fn test_sub_slice_heuristic_and_env_override() {
        struct SlicesEnvGuard;
        impl Drop for SlicesEnvGuard {
            fn drop(&mut self) {
                unsafe { std::env::remove_var("HC4J_COLLECTIVE_SLICES") };
            }
        }
        let _guard = SlicesEnvGuard;
        unsafe { std::env::remove_var("HC4J_COLLECTIVE_SLICES") };

        const MIB: vk::DeviceSize = 1024 * 1024;
        assert_eq!(sub_slice_count(512 * 1024, 131_072, false), 1);
        assert_eq!(sub_slice_count(512 * 1024, 131_072, true), 2);
        assert_eq!(sub_slice_count(2 * MIB, 524_288, true), 4);
        assert_eq!(sub_slice_count(8 * MIB, 2_097_152, true), 8);
        assert_eq!(sub_slice_count(32 * MIB, 8_388_608, true), 16);
        // Clamped by max_chunk_len for tiny prime lengths
        assert_eq!(sub_slice_count(4, 1, true), 1);

        // Env override clamped to 1..=32 and max_chunk_len
        unsafe { std::env::set_var("HC4J_COLLECTIVE_SLICES", "64") };
        assert_eq!(sub_slice_count(32 * MIB, 8_388_608, true), 32);
        assert_eq!(sub_slice_count(12, 3, true), 3);

        unsafe { std::env::set_var("HC4J_COLLECTIVE_SLICES", "1") };
        assert_eq!(sub_slice_count(32 * MIB, 8_388_608, true), 1);
    }
}
