//! Balanced uneven chunk partitioning (Fix #7) and N-rank Ring AllReduce execution.

use ash::vk;

use crate::collective::vulkan::device::{GpuNode, one_shot_submit};
use crate::collective::vulkan::pipeline::ReducePipeline;
use crate::collective::vulkan::topology::TopoGraph;
use crate::collective::vulkan::transport::transfer_route;
use crate::error::Hc4jResult;

/// Balanced chunk partitioning for arbitrary lengths and rank counts (Fix #7).
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

/// Executes an N-rank Ring AllReduce (sum) across the collective.
pub(crate) fn execute_ring_allreduce(
    nodes: &[GpuNode],
    pipelines: &mut [ReducePipeline],
    graph: &TopoGraph,
    user_buffers: &[vk::Buffer],
    total_elements: usize,
) -> Hc4jResult<()> {
    let n = nodes.len();
    if n <= 1 || total_elements == 0 {
        // Fix #6: ranks == 1 is a valid no-op AllReduce
        return Ok(());
    }

    // Determine max chunk size in bytes to ensure scratch buffers are sufficiently large (Fix #4)
    let max_chunk_len = (total_elements / n) + usize::from(!total_elements.is_multiple_of(n));
    let max_chunk_bytes = (max_chunk_len * 4) as vk::DeviceSize;

    for i in 0..n {
        pipelines[i].ensure_scratch(&nodes[i], max_chunk_bytes)?;
    }

    // ═══ Phase 1: Scatter-Reduce (N−1 rounds) ════════════════════════════════
    for round in 0..(n - 1) {
        for i in 0..n {
            let next = (i + 1) % n;
            let send_chunk_idx = ((i + n) - round) % n;
            let (chunk_start, chunk_len) = chunk_range(total_elements, n, send_chunk_idx);

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

    // ═══ Phase 2: Allgather (N−1 rounds) ═════════════════════════════════════
    for round in 0..(n - 1) {
        for i in 0..n {
            let next = (i + 1) % n;
            let send_chunk_idx = ((i + 1 + n) - round) % n;
            let (chunk_start, chunk_len) = chunk_range(total_elements, n, send_chunk_idx);

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
                    assert!(start + len <= total, "chunk overflow for total={total}, ranks={ranks}, chunk={chunk_idx}");
                    for i in start..start + len {
                        assert!(!covered[i], "overlap detected at index {i} for total={total}, ranks={ranks}");
                        covered[i] = true;
                    }
                }
            }

            assert_eq!(total_len, total, "sum of chunk lengths does not match total for total={total}, ranks={ranks}");
            assert!(covered.iter().all(|&c| c), "gap detected in chunk coverage for total={total}, ranks={ranks}");
        }
    }
}
