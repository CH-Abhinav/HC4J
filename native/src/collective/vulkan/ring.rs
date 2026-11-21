//! Balanced uneven chunk partitioning and N-rank Ring / Hierarchical AllReduce execution.

use ash::vk;

use crate::collective::backend::LinkInfo;
use crate::collective::planner::{solve_optimal_ring, IslandInfo};
use crate::collective::vulkan::device::{GpuNode, one_shot_submit};
use crate::collective::vulkan::pipeline::ReducePipeline;
use crate::collective::vulkan::topology::TopoGraph;
use crate::collective::vulkan::transport::transfer_route;
use crate::error::{Hc4jError, Hc4jResult};

/// Balanced chunk partitioning for arbitrary lengths and rank counts.
///
/// Divides `total` elements across `ranks` into balanced, non-overlapping contiguous slices.
/// The first `total % ranks` chunks have length `total / ranks + 1`, and the remaining chunks
/// have length `total / ranks`.
pub(crate) fn chunk_range(total: usize, ranks: usize, chunk_idx: usize) -> (usize, usize) {
    if ranks == 0 || chunk_idx >= ranks {
        return (0, 0);
    }
    let base = total / ranks;
    let rem = total % ranks;
    let len = base + usize::from(chunk_idx < rem);
    let start = chunk_idx * base + chunk_idx.min(rem);
    (start, len)
}

/// Executes an ordered Ring AllReduce (sum) across participating ranks in `order`.
pub(crate) fn ring_allreduce_ordered(
    nodes: &[GpuNode],
    pipelines: &mut [ReducePipeline],
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

    for &rank in order {
        pipelines[rank].ensure_scratch(&nodes[rank], max_chunk_bytes)?;
    }

    // ═══ Phase 1: Scatter-Reduce (k_len − 1 rounds) ══════════════════════════
    for round in 0..(k_len - 1) {
        for idx in 0..k_len {
            let i = order[idx];
            let next = order[(idx + 1) % k_len];
            let send_chunk_idx = ((idx + k_len) - round) % k_len;
            let (chunk_start, chunk_len) = chunk_range(total_elements, k_len, send_chunk_idx);

            if chunk_len == 0 {
                continue;
            }

            let chunk_bytes = (chunk_len * 4) as vk::DeviceSize;
            let chunk_offset_bytes = (chunk_start * 4) as vk::DeviceSize;

            let send_buf = pipelines[i].send_chunk_buf.as_ref().unwrap().buffer;
            let recv_buf = pipelines[next].recv_chunk_buf.as_ref().unwrap().buffer;

            // GPU i: copy slice from user buffer -> send_chunk_buf
            one_shot_submit(&nodes[i], |cb| unsafe {
                let region = vk::BufferCopy {
                    src_offset: chunk_offset_bytes,
                    dst_offset: 0,
                    size: chunk_bytes,
                };
                nodes[i].device.cmd_copy_buffer(cb, user_buffers[i], send_buf, &[region]);
            })?;

            // Transfer from GPU i (send_chunk_buf) -> GPU next (recv_chunk_buf)
            let tier = graph.tier(i, next);
            transfer_route(
                &nodes[i],
                &nodes[next],
                tier,
                send_buf,
                0,
                recv_buf,
                0,
                chunk_bytes,
            )?;

            // GPU next: accumulate recv_chunk_buf into user_buffers[next] at chunk_offset_bytes
            pipelines[next].dispatch_reduce(
                &nodes[next],
                user_buffers[next],
                chunk_offset_bytes,
                recv_buf,
                chunk_len as u32,
            )?;
        }
    }

    // ═══ Phase 2: Allgather (k_len − 1 rounds) ═══════════════════════════════
    for round in 0..(k_len - 1) {
        for idx in 0..k_len {
            let i = order[idx];
            let next = order[(idx + 1) % k_len];
            let send_chunk_idx = ((idx + 1 + k_len) - round) % k_len;
            let (chunk_start, chunk_len) = chunk_range(total_elements, k_len, send_chunk_idx);

            if chunk_len == 0 {
                continue;
            }

            let chunk_bytes = (chunk_len * 4) as vk::DeviceSize;
            let chunk_offset_bytes = (chunk_start * 4) as vk::DeviceSize;

            let send_buf = pipelines[i].send_chunk_buf.as_ref().unwrap().buffer;
            let recv_buf = pipelines[next].recv_chunk_buf.as_ref().unwrap().buffer;

            // GPU i: copy fully reduced slice from user buffer -> send_chunk_buf
            one_shot_submit(&nodes[i], |cb| unsafe {
                let region = vk::BufferCopy {
                    src_offset: chunk_offset_bytes,
                    dst_offset: 0,
                    size: chunk_bytes,
                };
                nodes[i].device.cmd_copy_buffer(cb, user_buffers[i], send_buf, &[region]);
            })?;

            // Transfer from GPU i -> GPU next
            let tier = graph.tier(i, next);
            transfer_route(
                &nodes[i],
                &nodes[next],
                tier,
                send_buf,
                0,
                recv_buf,
                0,
                chunk_bytes,
            )?;

            // GPU next: copy recv_chunk_buf into user_buffers[next] at chunk_offset_bytes
            one_shot_submit(&nodes[next], |cb| unsafe {
                let region = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: chunk_offset_bytes,
                    size: chunk_bytes,
                };
                nodes[next].device.cmd_copy_buffer(cb, recv_buf, user_buffers[next], &[region]);
            })?;
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
    pipelines: &mut [ReducePipeline],
    graph: &TopoGraph,
    user_buffers: &[vk::Buffer],
    total_elements: usize,
    islands: &[IslandInfo],
) -> Hc4jResult<()> {
    if nodes.len() <= 1 || total_elements == 0 {
        return Ok(());
    }

    let total_bytes = (total_elements * 4) as vk::DeviceSize;

    // Phase 1: Intra-Island Reduce to Leader
    for isl in islands {
        if isl.members.len() > 1 {
            let leader = isl.leader_rank as usize;
            pipelines[leader].ensure_scratch(&nodes[leader], total_bytes)?;
            let leader_recv_buf = pipelines[leader].recv_chunk_buf.as_ref().unwrap().buffer;

            for &m in &isl.members {
                let member = m as usize;
                if member == leader {
                    continue;
                }
                let tier = graph.tier(member, leader);
                transfer_route(
                    &nodes[member],
                    &nodes[leader],
                    tier,
                    user_buffers[member],
                    0,
                    leader_recv_buf,
                    0,
                    total_bytes,
                )?;
                pipelines[leader].dispatch_reduce(
                    &nodes[leader],
                    user_buffers[leader],
                    0,
                    leader_recv_buf,
                    total_elements as u32,
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
                let tier = graph.tier(leader, member);
                transfer_route(
                    &nodes[leader],
                    &nodes[member],
                    tier,
                    user_buffers[leader],
                    0,
                    user_buffers[member],
                    0,
                    total_bytes,
                )?;
            }
        }
    }

    Ok(())
}

/// Executes AllReduce across the collective using the selected algorithm
/// (auto, ring, or hierarchical) based on `HC4J_COLLECTIVE_ALGO`.
pub(crate) fn execute_allreduce(
    nodes: &[GpuNode],
    pipelines: &mut [ReducePipeline],
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
                hierarchical_allreduce(nodes, pipelines, graph, user_buffers, total_elements, islands)
            } else {
                ring_allreduce_ordered(
                    nodes,
                    pipelines,
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
                    graph,
                    user_buffers,
                    total_elements,
                    &ring_order_usize,
                )
            } else {
                hierarchical_allreduce(nodes, pipelines, graph, user_buffers, total_elements, islands)
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
}
