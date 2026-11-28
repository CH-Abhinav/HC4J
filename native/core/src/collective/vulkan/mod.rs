//! Raw-Vulkan P2P Collective Backend (`VulkanP2pBackend`).
//!
//! Implements hardware-level peer-to-peer Ring AllReduce on Windows and Linux
//! using Vulkan 1.2 Timeline Semaphores, Windows `OPAQUE_WIN32` / `VK_EXT_external_memory_host`,
//! and Linux `dma_buf`.

pub(crate) mod device;
pub(crate) mod pipeline;
pub(crate) mod profile;
pub(crate) mod ring;
pub(crate) mod topology;
pub(crate) mod transport;

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use ash::vk;

use crate::collective::backend::{CollectiveBackend, DeviceProfileC, LinkInfo};
use crate::collective::planner::{detect_islands, solve_optimal_ring, IslandInfo};
use crate::collective::vulkan::device::{BufMem, GpuNode, alloc_device_local, alloc_staging, build_node, create_instance, destroy_buf, device_name, device_type_str, enum_physical_devices, one_shot_submit};
use crate::collective::vulkan::pipeline::{RankComputePipelines, ReducePipeline};
use crate::collective::vulkan::ring::execute_allreduce;
use crate::collective::vulkan::topology::{TopoGraph, build_topology_and_profiles};
use crate::collective::vulkan::transport::EdgeChannelCache;
use crate::engine::error::{Hc4jError, Hc4jResult};

/// Process-wide loaded Vulkan entry point. Kept resident for process lifetime so
/// dropping a `VulkanP2pBackend` never `dlclose`s `libvulkan.so.1` while `wgpu`'s
/// static GPU engine still holds active Vulkan objects on Linux.
static VULKAN_ENTRY: OnceLock<Result<ash::Entry, String>> = OnceLock::new();

fn get_vulkan_entry() -> Hc4jResult<&'static ash::Entry> {
    VULKAN_ENTRY
        .get_or_init(|| {
            unsafe { ash::Entry::load() }
                .map_err(|e| format!("failed to load Vulkan entry point: {e}"))
        })
        .as_ref()
        .map_err(|msg| Hc4jError::Device(msg.clone()))
}

pub struct VulkanP2pBackend {
    instance: ash::Instance,
    nodes: Vec<GpuNode>,
    rank_pipelines: Vec<Mutex<RankComputePipelines>>,
    edge_channels: Mutex<EdgeChannelCache>,
    graph: TopoGraph,
    profiles: Vec<DeviceProfileC>,
    ring_order: Vec<u32>,
    islands: Vec<IslandInfo>,
    buffers: Mutex<HashMap<u64, (usize, BufMem, usize)>>,
    next_buffer_id: AtomicU64,
}

impl VulkanP2pBackend {
    pub fn new(ranks: u32) -> Hc4jResult<Self> {
        if ranks == 0 {
            return Err(Hc4jError::InvalidParam("collective needs at least 1 rank"));
        }

        let entry = get_vulkan_entry()?;

        let instance = create_instance(entry)?;

        let phds = enum_physical_devices(&instance)?;
        let num_physical = phds.len();

        unsafe {
            if let Ok(len) = instance.enumerate_physical_device_groups_len() {
                let mut groups = vec![vk::PhysicalDeviceGroupProperties::default(); len];
                if instance.enumerate_physical_device_groups(&mut groups).is_ok() {
                    let sizes: Vec<u32> = groups.iter().map(|g| g.physical_device_count).collect();
                    eprintln!(
                        "[HC4J] collective device_groups: {} group(s) (sizes={:?})",
                        groups.len(),
                        sizes
                    );
                }
            }
        }

        let mut nodes = Vec::with_capacity(ranks as usize);
        for i in 0..ranks as usize {
            let phd = phds[i % num_physical];
            let node = build_node(&instance, phd, i)?;
            nodes.push(node);
        }

        let (graph, profiles) = build_topology_and_profiles(&nodes)?;
        let ring_order = solve_optimal_ring(nodes.len(), &graph.links);
        let islands = detect_islands(nodes.len(), &profiles, &graph.links);

        let mut rank_pipelines = Vec::with_capacity(ranks as usize);
        for node in &nodes {
            rank_pipelines.push(Mutex::new(RankComputePipelines::new(node)?));
        }

        // Output clean [HC4J] log messages matching HC4J style
        eprintln!("[HC4J] collective backend: vulkan-p2p");
        eprintln!(
            "[HC4J] collective: {} physical GPU(s) for {} ranks{}",
            num_physical,
            ranks,
            if num_physical < ranks as usize { " -> virtual loopback mode" } else { "" }
        );
        for (i, node) in nodes.iter().enumerate() {
            let p = &profiles[i];
            let vram_str = if p.vram_bytes >= 1024 * 1024 * 1024 {
                format!("{:.1} GiB", p.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
            } else {
                format!("{:.1} MiB", p.vram_bytes as f64 / (1024.0 * 1024.0))
            };
            eprintln!(
                "[HC4J] collective rank {}: {} | {} | VRAM={} | {:.1} GFLOP/s | {:.1} GB/s",
                i,
                device_name(&node.props),
                device_type_str(node.props.device_type),
                vram_str,
                p.compute_gflops,
                p.vram_bandwidth_gbps
            );
        }
        for link in &graph.links {
            let transport_label = match link.transport {
                1 => "P2P",
                2 => "HOST_SHARED_DMA",
                _ => "HOST_STAGED",
            };
            eprintln!(
                "[HC4J] collective rank {}->{}: {:.1} GB/s {} (loopback={})",
                link.src_rank,
                link.dst_rank,
                link.bandwidth_gbps,
                transport_label,
                link.loopback == 1
            );
        }

        let algo_var = std::env::var("HC4J_COLLECTIVE_ALGO")
            .map(|v| v.trim().to_ascii_lowercase())
            .unwrap_or_else(|_| "auto".to_string());
        let effective_algo = match algo_var.as_str() {
            "ring" => "ring",
            "hierarchical" => "hierarchical",
            _ => {
                if islands.len() > 1 && islands.iter().any(|isl| isl.members.len() > 1) {
                    "hierarchical"
                } else {
                    "ring"
                }
            }
        };

        let island_strs: Vec<String> = islands
            .iter()
            .map(|isl| {
                let members_str = isl
                    .members
                    .iter()
                    .map(|m| m.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                let tr_str = match isl.internal_transport {
                    1 => "P2P",
                    2 => "HOST_SHARED_DMA",
                    _ => "HOST_STAGED",
                };
                format!("[{members_str}]@{tr_str} leader={}", isl.leader_rank)
            })
            .collect();

        eprintln!(
            "[HC4J] collective plan: ring={:?} | islands={} ({}) | algo={}",
            ring_order,
            islands.len(),
            island_strs.join(", "),
            effective_algo
        );

        Ok(Self {
            instance,
            nodes,
            rank_pipelines,
            edge_channels: Mutex::new(EdgeChannelCache::new()),
            graph,
            profiles,
            ring_order,
            islands,
            buffers: Mutex::new(HashMap::new()),
            next_buffer_id: AtomicU64::new(1),
        })
    }
}

impl CollectiveBackend for VulkanP2pBackend {
    fn backend_name(&self) -> &'static str {
        "vulkan-p2p"
    }

    fn ranks(&self) -> usize {
        self.nodes.len()
    }

    fn device_name(&self, rank: usize) -> Hc4jResult<String> {
        let node = self.nodes.get(rank)
            .ok_or(Hc4jError::InvalidParam("rank out of range"))?;
        Ok(device_name(&node.props))
    }

    fn links(&self) -> &[LinkInfo] {
        &self.graph.links
    }

    fn alloc(&self, rank: usize, elements: u64) -> Hc4jResult<u64> {
        let node = self.nodes.get(rank)
            .ok_or(Hc4jError::InvalidParam("rank out of range"))?;

        if elements == 0 {
            return Err(Hc4jError::InvalidParam("buffer elements must be non-zero"));
        }

        let size = elements.checked_mul(4)
            .ok_or(Hc4jError::InvalidParam("buffer size overflows u64"))?;

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        let buf = alloc_device_local(node, size, usage)?;
        let id = self.next_buffer_id.fetch_add(1, Ordering::Relaxed);

        let mut map = self.buffers.lock().unwrap();
        map.insert(id, (rank, buf, elements as usize));

        Ok(id)
    }

    fn free(&self, buffer_id: u64) -> Hc4jResult<()> {
        let entry = {
            let mut map = self.buffers.lock().unwrap();
            map.remove(&buffer_id)
        };

        if let Some((rank, buf, _)) = entry {
            destroy_buf(&self.nodes[rank], &buf);
            Ok(())
        } else {
            Err(Hc4jError::InvalidParam("invalid collective buffer id"))
        }
    }

    fn write(&self, buffer_id: u64, data: &[f32]) -> Hc4jResult<()> {
        let (node, buffer, elements) = {
            let map = self.buffers.lock().unwrap();
            let &(rank, ref buf, elements) = map.get(&buffer_id)
                .ok_or(Hc4jError::InvalidParam("invalid collective buffer id"))?;
            (&self.nodes[rank], buf.buffer, elements)
        };

        if data.len() != elements {
            return Err(Hc4jError::InvalidParam("data length does not match buffer elements"));
        }

        let byte_len = (data.len() * 4) as vk::DeviceSize;
        let staging = alloc_staging(node, byte_len, vk::BufferUsageFlags::TRANSFER_SRC)?;

        unsafe {
            let ptr = node.device.map_memory(staging.memory, 0, staging.size, vk::MemoryMapFlags::empty())
                .map_err(|e| Hc4jError::Device(format!("map staging buffer failed: {e}")))? as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            node.device.unmap_memory(staging.memory);
        }

        let res = one_shot_submit(node, |cb| unsafe {
            let region = vk::BufferCopy { src_offset: 0, dst_offset: 0, size: byte_len };
            node.device.cmd_copy_buffer(cb, staging.buffer, buffer, &[region]);
        });

        destroy_buf(node, &staging);
        res
    }

    fn read(&self, buffer_id: u64, out: &mut [f32]) -> Hc4jResult<()> {
        let (node, buffer, elements) = {
            let map = self.buffers.lock().unwrap();
            let &(rank, ref buf, elements) = map.get(&buffer_id)
                .ok_or(Hc4jError::InvalidParam("invalid collective buffer id"))?;
            (&self.nodes[rank], buf.buffer, elements)
        };

        if out.len() != elements {
            return Err(Hc4jError::InvalidParam("out length does not match buffer elements"));
        }

        let byte_len = (out.len() * 4) as vk::DeviceSize;
        let staging = alloc_staging(node, byte_len, vk::BufferUsageFlags::TRANSFER_DST)?;

        one_shot_submit(node, |cb| unsafe {
            let region = vk::BufferCopy { src_offset: 0, dst_offset: 0, size: byte_len };
            node.device.cmd_copy_buffer(cb, buffer, staging.buffer, &[region]);
        })?;

        unsafe {
            let ptr = node.device.map_memory(staging.memory, 0, staging.size, vk::MemoryMapFlags::empty())
                .map_err(|e| Hc4jError::Device(format!("map staging buffer for read failed: {e}")))? as *const f32;
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), elements);
            node.device.unmap_memory(staging.memory);
        }

        destroy_buf(node, &staging);
        Ok(())
    }

    fn allreduce_sum(&self, buffers: &[u64]) -> Hc4jResult<()> {
        let n = self.nodes.len();
        if buffers.len() != n {
            return Err(Hc4jError::InvalidParam("buffer count must match collective ranks"));
        }
        self.allreduce_sum_subset(buffers)
    }

    fn allreduce_sum_subset(&self, buffers: &[u64]) -> Hc4jResult<()> {
        if buffers.is_empty() {
            return Err(Hc4jError::InvalidParam("allreduce_sum_subset requires at least 1 buffer"));
        }

        if buffers.len() == 1 {
            let map = self.buffers.lock().unwrap();
            if !map.contains_key(&buffers[0]) {
                return Err(Hc4jError::InvalidParam("invalid collective buffer handle"));
            }
            return Ok(());
        }

        let (sub_ranks, raw_buffers_map, elements) = {
            let map = self.buffers.lock().unwrap();
            let mut sub_ranks = Vec::with_capacity(buffers.len());
            let mut raw_map = HashMap::with_capacity(buffers.len());
            let mut elem_count = None;

            for &buf_id in buffers {
                let &(rank, ref buf, count) = map.get(&buf_id)
                    .ok_or(Hc4jError::InvalidParam("invalid collective buffer handle"))?;

                if raw_map.contains_key(&rank) {
                    return Err(Hc4jError::InvalidParam("duplicate rank in active subset"));
                }
                if let Some(expected) = elem_count {
                    if count != expected {
                        return Err(Hc4jError::InvalidParam("all collective buffers must have identical size"));
                    }
                } else {
                    elem_count = Some(count);
                }
                sub_ranks.push(rank);
                raw_map.insert(rank, buf.buffer);
            }

            (sub_ranks, raw_map, elem_count.unwrap_or(0))
        };

        let sub_n = sub_ranks.len();
        let mut sub_links = Vec::with_capacity(sub_n * (sub_n - 1));
        for (new_i, &orig_i) in sub_ranks.iter().enumerate() {
            for (new_j, &orig_j) in sub_ranks.iter().enumerate() {
                let found = if new_i != new_j {
                    self.graph.links.iter().find(|l| l.src_rank == orig_i as u32 && l.dst_rank == orig_j as u32)
                } else {
                    None
                };
                if let Some(link) = found {
                    let mut l = *link;
                    l.src_rank = new_i as u32;
                    l.dst_rank = new_j as u32;
                    sub_links.push(l);
                }
            }
        }

        let mut sub_profiles = Vec::with_capacity(sub_n);
        for (new_i, &orig_i) in sub_ranks.iter().enumerate() {
            let mut p = self.profiles[orig_i];
            p.rank = new_i as u32;
            sub_profiles.push(p);
        }

        let sub_ring = solve_optimal_ring(sub_n, &sub_links);
        let payload_bytes = (elements * 4) as u64;
        let sub_islands = crate::collective::planner::detect_islands_for_payload(sub_n, &sub_profiles, &sub_links, payload_bytes);

        let ring_order: Vec<u32> = sub_ring.iter().map(|&idx| sub_ranks[idx as usize] as u32).collect();
        let islands: Vec<IslandInfo> = sub_islands.into_iter().map(|isl| {
            IslandInfo {
                id: isl.id,
                leader_rank: sub_ranks[isl.leader_rank as usize] as u32,
                members: isl.members.into_iter().map(|m| sub_ranks[m as usize] as u32).collect(),
                internal_transport: isl.internal_transport,
            }
        }).collect();

        // Lock participating ranks in strictly ascending order to avoid deadlock
        let mut sorted_ranks = sub_ranks.clone();
        sorted_ranks.sort_unstable();
        let mut guards: Vec<(usize, _)> = sorted_ranks
            .iter()
            .map(|&r| (r, self.rank_pipelines[r].lock().unwrap()))
            .collect();

        let mut pipes: Vec<Option<&mut ReducePipeline>> = (0..self.nodes.len()).map(|_| None).collect();
        for (r, guard) in &mut guards {
            pipes[*r] = Some(&mut guard.reduce);
        }

        let mut raw_buffers = vec![vk::Buffer::null(); self.nodes.len()];
        for (&r, &b) in &raw_buffers_map {
            raw_buffers[r] = b;
        }

        let mut channels = self.edge_channels.lock().unwrap();
        execute_allreduce(
            &self.nodes,
            &mut pipes,
            &mut channels,
            &self.graph,
            &raw_buffers,
            elements,
            &ring_order,
            &islands,
        )
    }

    fn gemm_bias_act(
        &self,
        rank: usize,
        buf_a: u64,
        buf_b: u64,
        buf_bias: u64,
        buf_c: u64,
        m: u32,
        n: u32,
        k: u32,
        flags: u32,
    ) -> Hc4jResult<()> {
        if rank >= self.nodes.len() {
            return Err(Hc4jError::InvalidParam("rank out of range"));
        }
        let node = &self.nodes[rank];

        let (raw_a, raw_b, raw_bias, raw_c) = {
            let map = self.buffers.lock().unwrap();
            let (r_a, b_a, elems_a) = map.get(&buf_a).copied().ok_or(Hc4jError::InvalidParam("invalid buf_a"))?;
            let (r_b, b_b, elems_b) = map.get(&buf_b).copied().ok_or(Hc4jError::InvalidParam("invalid buf_b"))?;
            let (r_c, b_c, elems_c) = map.get(&buf_c).copied().ok_or(Hc4jError::InvalidParam("invalid buf_c"))?;

            if r_a != rank || r_b != rank || r_c != rank {
                return Err(Hc4jError::InvalidParam("buffers must belong to specified rank"));
            }
            if elems_a < (m * k) as usize || elems_b < (k * n) as usize || elems_c < (m * n) as usize {
                return Err(Hc4jError::InvalidParam("buffer size too small for gemm shape"));
            }

            let raw_bias = if buf_bias != 0 {
                let (r_bias, b_bias, elems_bias) = map.get(&buf_bias).copied().ok_or(Hc4jError::InvalidParam("invalid buf_bias"))?;
                if r_bias != rank {
                    return Err(Hc4jError::InvalidParam("bias buffer must belong to specified rank"));
                }
                if elems_bias < n as usize {
                    return Err(Hc4jError::InvalidParam("bias buffer size too small"));
                }
                Some(b_bias.buffer)
            } else {
                None
            };

            (b_a.buffer, b_b.buffer, raw_bias, b_c.buffer)
        };

        let pipes = self.rank_pipelines[rank].lock().unwrap();
        pipes.gemm.dispatch(node, raw_a, raw_b, raw_bias, raw_c, m, n, k, flags)
    }

    fn split_kv_attn(
        &self,
        rank: usize,
        buf_q: u64,
        buf_k: u64,
        buf_v: u64,
        buf_out_o: u64,
        buf_out_stats: u64,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        local_tokens: u32,
        scale: f32,
    ) -> Hc4jResult<()> {
        if rank >= self.nodes.len() {
            return Err(Hc4jError::InvalidParam("rank out of range"));
        }
        let node = &self.nodes[rank];

        let (raw_q, raw_k, raw_v, raw_o, raw_stats) = {
            let map = self.buffers.lock().unwrap();
            let (r_q, b_q, elems_q) = map.get(&buf_q).copied().ok_or(Hc4jError::InvalidParam("invalid buf_q"))?;
            let (r_k, b_k, elems_k) = map.get(&buf_k).copied().ok_or(Hc4jError::InvalidParam("invalid buf_k"))?;
            let (r_v, b_v, elems_v) = map.get(&buf_v).copied().ok_or(Hc4jError::InvalidParam("invalid buf_v"))?;
            let (r_o, b_o, elems_o) = map.get(&buf_out_o).copied().ok_or(Hc4jError::InvalidParam("invalid buf_out_o"))?;
            let (r_s, b_s, elems_s) = map.get(&buf_out_stats).copied().ok_or(Hc4jError::InvalidParam("invalid buf_out_stats"))?;

            if r_q != rank || r_k != rank || r_v != rank || r_o != rank || r_s != rank {
                return Err(Hc4jError::InvalidParam("split_kv buffers must belong to specified rank"));
            }
            if elems_q < (num_heads * head_dim) as usize
                || elems_k < (num_kv_heads * local_tokens * head_dim) as usize
                || elems_v < (num_kv_heads * local_tokens * head_dim) as usize
                || elems_o < (num_heads * head_dim) as usize
                || elems_s < (num_heads * 2) as usize
            {
                return Err(Hc4jError::InvalidParam("split_kv buffer size too small"));
            }

            (b_q.buffer, b_k.buffer, b_v.buffer, b_o.buffer, b_s.buffer)
        };

        let pipes = self.rank_pipelines[rank].lock().unwrap();
        pipes.split_kv.dispatch(
            node,
            raw_q,
            raw_k,
            raw_v,
            raw_o,
            raw_stats,
            num_heads,
            num_kv_heads,
            head_dim,
            local_tokens,
            scale,
        )
    }

    fn elem_mul(
        &self,
        rank: usize,
        buf_a: u64,
        buf_b: u64,
        buf_out: u64,
        count: u32,
    ) -> Hc4jResult<()> {
        if rank >= self.nodes.len() {
            return Err(Hc4jError::InvalidParam("rank out of range"));
        }
        if count == 0 {
            return Ok(());
        }
        let node = &self.nodes[rank];
        let (raw_a, raw_b, raw_out) = {
            let map = self.buffers.lock().unwrap();
            let (r_a, b_a, elems_a) = map.get(&buf_a).copied().ok_or(Hc4jError::InvalidParam("invalid buf_a"))?;
            let (r_b, b_b, elems_b) = map.get(&buf_b).copied().ok_or(Hc4jError::InvalidParam("invalid buf_b"))?;
            let (r_o, b_o, elems_o) = map.get(&buf_out).copied().ok_or(Hc4jError::InvalidParam("invalid buf_out"))?;
            if r_a != rank || r_b != rank || r_o != rank {
                return Err(Hc4jError::InvalidParam("elem_mul buffers must belong to specified rank"));
            }
            if elems_a < count as usize || elems_b < count as usize || elems_o < count as usize {
                return Err(Hc4jError::InvalidParam("elem_mul buffer size too small"));
            }
            (b_a.buffer, b_b.buffer, b_o.buffer)
        };
        let pipes = self.rank_pipelines[rank].lock().unwrap();
        pipes.elem_mul.dispatch(node, raw_a, raw_b, raw_out, count)
    }

    fn device_profiles(&self) -> &[DeviceProfileC] {
        &self.profiles
    }

    fn ring_order(&self) -> &[u32] {
        &self.ring_order
    }

    fn islands(&self) -> &[IslandInfo] {
        &self.islands
    }
}

// Fix #8: Clean RAII teardown in exact reverse Vulkan dependency order
impl Drop for VulkanP2pBackend {
    fn drop(&mut self) {
        // 1. Wait idle across all nodes
        for node in &self.nodes {
            unsafe {
                node.device.device_wait_idle().ok();
            }
        }

        // 2. Destroy cached persistent edge channels before devices are destroyed
        if let Ok(mut channels) = self.edge_channels.lock() {
            channels.destroy_all(&self.nodes);
        }

        // 3. Free all user allocated buffers
        let mut user_bufs = self.buffers.lock().unwrap();
        for (_, (rank, buf, _)) in user_bufs.drain() {
            destroy_buf(&self.nodes[rank], &buf);
        }

        // 4. Destroy all pipelines, scratch buffers, fences, CBs, and descriptor pools
        for (i, mutex) in self.rank_pipelines.iter().enumerate() {
            let mut pipes = mutex.lock().unwrap();
            pipes.destroy(&self.nodes[i]);
        }

        // 5. Destroy all GpuNodes (command pools & logical devices)
        for node in &self.nodes {
            node.destroy();
        }

        // 6. Destroy Vulkan instance
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_p2p_1_rank_allreduce() {
        let backend = match VulkanP2pBackend::new(1) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skipping Vulkan test (ICD unavailable): {e}");
                return;
            }
        };

        assert_eq!(backend.ranks(), 1);
        assert_eq!(backend.backend_name(), "vulkan-p2p");

        let b0 = backend.alloc(0, 6).expect("alloc on rank 0");
        let initial = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        backend.write(b0, &initial).expect("write b0");

        backend.allreduce_sum(&[b0]).expect("allreduce_sum 1 rank");

        let mut result = vec![0.0; 6];
        backend.read(b0, &mut result).expect("read b0");
        assert_eq!(result, initial);

        backend.free(b0).expect("free b0");
    }

    #[test]
    fn test_vulkan_p2p_2_rank_allreduce() {
        let backend = match VulkanP2pBackend::new(2) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skipping Vulkan test (ICD unavailable): {e}");
                return;
            }
        };

        assert_eq!(backend.ranks(), 2);
        let b0 = backend.alloc(0, 6).expect("alloc on rank 0");
        let b1 = backend.alloc(1, 6).expect("alloc on rank 1");

        let data0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data1 = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        backend.write(b0, &data0).expect("write b0");
        backend.write(b1, &data1).expect("write b1");

        backend.allreduce_sum(&[b0, b1]).expect("allreduce_sum");

        let mut res0 = vec![0.0; 6];
        let mut res1 = vec![0.0; 6];
        backend.read(b0, &mut res0).expect("read b0");
        backend.read(b1, &mut res1).expect("read b1");

        let expected = vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0];
        assert_eq!(res0, expected);
        assert_eq!(res1, expected);

        backend.free(b0).expect("free b0");
        backend.free(b1).expect("free b1");
    }

    #[test]
    fn test_vulkan_p2p_4_rank_allreduce_uneven() {
        let backend = match VulkanP2pBackend::new(4) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skipping Vulkan test (ICD unavailable): {e}");
                return;
            }
        };

        assert_eq!(backend.ranks(), 4);
        let len = 1003; // Prime length to stress uneven chunking
        let mut bufs = Vec::new();
        for r in 0..4 {
            let id = backend.alloc(r, len as u64).expect("alloc rank");
            let data: Vec<f32> = (0..len).map(|i| (r * len + i + 1) as f32).collect();
            backend.write(id, &data).expect("write rank data");
            bufs.push(id);
        }

        backend.allreduce_sum(&bufs).expect("allreduce_sum uneven 4-rank");

        let expected: Vec<f32> = (0..len)
            .map(|i| (0..4).map(|r| (r * len + i + 1) as f32).sum())
            .collect();

        for &id in &bufs {
            let mut res = vec![0.0; len];
            backend.read(id, &mut res).expect("read rank data");
            for i in 0..len {
                assert!(
                    (res[i] - expected[i]).abs() < 1e-4,
                    "mismatch at element {i}: got {}, expected {}",
                    res[i],
                    expected[i]
                );
            }
            backend.free(id).expect("free rank buf");
        }
    }

    #[test]
    fn test_vulkan_p2p_large_pipelined_and_prime_allreduce() {
        let backend = match VulkanP2pBackend::new(2) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skipping Vulkan test (ICD unavailable): {e}");
                return;
            }
        };

        // Cover uneven prime lengths (1, 7, 1003) and large >= 2 MiB chunk sizes
        // that exercise double-buffered sub-chunk pipelining (4 MiB = S=4, 16 MiB = S=8).
        for &len in &[1usize, 7, 1003, 1_048_576, 4_194_304] {
            let b0 = backend.alloc(0, len as u64).expect("alloc b0");
            let b1 = backend.alloc(1, len as u64).expect("alloc b1");

            let d0: Vec<f32> = (0..len).map(|i| ((i % 97) as f32) * 0.25 + 1.5).collect();
            let d1: Vec<f32> = (0..len).map(|i| ((i % 53) as f32) * 0.5 - 3.25).collect();

            backend.write(b0, &d0).expect("write b0");
            backend.write(b1, &d1).expect("write b1");

            // Run twice so pass 2 exercises 100% persistent EdgeChannel reuse without reallocation
            backend.allreduce_sum(&[b0, b1]).expect("allreduce_sum pass 1");
            backend.allreduce_sum(&[b0, b1]).expect("allreduce_sum pass 2");

            let mut out0 = vec![0.0f32; len];
            let mut out1 = vec![0.0f32; len];
            backend.read(b0, &mut out0).expect("read b0");
            backend.read(b1, &mut out1).expect("read b1");

            for i in 0..len {
                let expected = 2.0 * (d0[i] + d1[i]);
                assert!(
                    (out0[i] - expected).abs() < 1e-4 && (out1[i] - expected).abs() < 1e-4,
                    "len={len} mismatch at {i}: got ({}, {}), expected {expected}",
                    out0[i],
                    out1[i]
                );
            }

            backend.free(b0).expect("free b0");
            backend.free(b1).expect("free b1");
        }
    }

    #[test]
    fn test_vulkan_p2p_25_iterations_stress() {
        let backend = match VulkanP2pBackend::new(2) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skipping Vulkan test (ICD unavailable): {e}");
                return;
            }
        };

        let len = 256;
        let b0 = backend.alloc(0, len as u64).expect("alloc b0");
        let b1 = backend.alloc(1, len as u64).expect("alloc b1");

        for iter in 0..25 {
            let val0 = (iter + 1) as f32;
            let val1 = (iter * 10 + 2) as f32;
            let d0 = vec![val0; len];
            let d1 = vec![val1; len];

            backend.write(b0, &d0).expect("write b0");
            backend.write(b1, &d1).expect("write b1");

            backend.allreduce_sum(&[b0, b1]).expect("allreduce_sum iteration");

            let mut out0 = vec![0.0; len];
            let mut out1 = vec![0.0; len];
            backend.read(b0, &mut out0).expect("read b0");
            backend.read(b1, &mut out1).expect("read b1");

            let expected_val = val0 + val1;
            assert_eq!(out0[0], expected_val, "iter {iter} rank 0 mismatch");
            assert_eq!(out1[0], expected_val, "iter {iter} rank 1 mismatch");
        }

        backend.free(b0).expect("free b0");
        backend.free(b1).expect("free b1");
    }

    #[test]
    fn test_vulkan_p2p_forced_tiers() {
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        use crate::collective::backend::TRANSPORT_HOST_SHARED_DMA;
        use crate::collective::backend::{TRANSPORT_HOST_STAGED, TRANSPORT_PEER_TO_PEER};

        struct ForceTierEnvGuard;
        impl Drop for ForceTierEnvGuard {
            fn drop(&mut self) {
                unsafe { std::env::remove_var("HC4J_COLLECTIVE_FORCE_TIER") };
            }
        }
        let _env_guard = ForceTierEnvGuard;

        let tiers = [
            ("staged", TRANSPORT_HOST_STAGED),
            ("p2p", TRANSPORT_PEER_TO_PEER),
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            ("host_dma", TRANSPORT_HOST_SHARED_DMA),
        ];

        for (tier_name, expected_code) in tiers {
            // SAFETY: Tests are run sequentially via --test-threads=1 or inside this sequential loop
            unsafe { std::env::set_var("HC4J_COLLECTIVE_FORCE_TIER", tier_name) };

            let backend = match VulkanP2pBackend::new(2) {
                Ok(b) => b,
                Err(Hc4jError::Unsupported(msg)) => {
                    eprintln!("Skipping forced tier {tier_name} (unsupported: {msg})");
                    continue;
                }
                Err(e) => {
                    eprintln!("Skipping forced tier {tier_name} (Vulkan unavailable: {e})");
                    continue;
                }
            };

            assert_eq!(
                backend.links()[0].transport, expected_code,
                "Forced tier {tier_name} did not yield expected transport code {expected_code}"
            );

            let b0 = backend.alloc(0, 2).expect("alloc b0");
            let b1 = backend.alloc(1, 2).expect("alloc b1");

            backend.write(b0, &[1.0, 2.0]).expect("write b0");
            backend.write(b1, &[10.0, 20.0]).expect("write b1");

            backend.allreduce_sum(&[b0, b1]).expect("allreduce_sum");

            let mut out0 = vec![0.0; 2];
            let mut out1 = vec![0.0; 2];
            backend.read(b0, &mut out0).expect("read b0");
            backend.read(b1, &mut out1).expect("read b1");

            assert_eq!(out0, vec![11.0, 22.0]);
            assert_eq!(out1, vec![11.0, 22.0]);

            backend.free(b0).expect("free b0");
            backend.free(b1).expect("free b1");

            // Also verify persistent channel growth + >= 2 MiB sub-chunk pipelining on every tier
            let big_len = 1_048_576usize;
            let big0 = backend.alloc(0, big_len as u64).expect("alloc big0");
            let big1 = backend.alloc(1, big_len as u64).expect("alloc big1");
            let d0: Vec<f32> = (0..big_len).map(|i| ((i % 31) as f32) * 0.25 + 1.0).collect();
            let d1: Vec<f32> = (0..big_len).map(|i| ((i % 19) as f32) * 0.5 - 2.0).collect();
            backend.write(big0, &d0).expect("write big0");
            backend.write(big1, &d1).expect("write big1");
            backend.allreduce_sum(&[big0, big1]).expect("allreduce_sum pipelined");
            let mut r0 = vec![0.0f32; big_len];
            let mut r1 = vec![0.0f32; big_len];
            backend.read(big0, &mut r0).expect("read big0");
            backend.read(big1, &mut r1).expect("read big1");
            for i in (0..big_len).step_by(997) {
                let exp = d0[i] + d1[i];
                assert!(
                    (r0[i] - exp).abs() < 1e-4 && (r1[i] - exp).abs() < 1e-4,
                    "tier={tier_name} pipelined mismatch at {i}: got ({}, {}), expected {exp}",
                    r0[i],
                    r1[i]
                );
            }
            backend.free(big0).expect("free big0");
            backend.free(big1).expect("free big1");
        }

        unsafe { std::env::remove_var("HC4J_COLLECTIVE_FORCE_TIER") };
    }

    #[test]
    fn test_explicit_wgpu_backend_still_passes() {
        use crate::collective::wgpu_backend::WgpuStagedBackend;
        let backend = WgpuStagedBackend::create(2).expect("WgpuStagedBackend create");
        assert_eq!(backend.backend_name(), "wgpu-staged");

        let b0 = backend.alloc(0, 4).expect("alloc b0");
        let b1 = backend.alloc(1, 4).expect("alloc b1");

        backend.write(b0, &[1.0, 2.0, 3.0, 4.0]).expect("write b0");
        backend.write(b1, &[10.0, 20.0, 30.0, 40.0]).expect("write b1");

        backend.allreduce_sum(&[b0, b1]).expect("allreduce_sum");

        let mut res = vec![0.0; 4];
        backend.read(b0, &mut res).expect("read b0");
        assert_eq!(res, vec![11.0, 22.0, 33.0, 44.0]);

        backend.free(b0).expect("free b0");
        backend.free(b1).expect("free b1");
    }

    #[test]
    fn test_vulkan_device_profiles_and_plan() {
        let backend = match VulkanP2pBackend::new(2) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skipping Vulkan test (ICD unavailable): {e}");
                return;
            }
        };

        let profiles = backend.device_profiles();
        assert_eq!(profiles.len(), 2);
        for p in profiles {
            assert!(p.vram_bytes > 0, "vram_bytes should be > 0, got {}", p.vram_bytes);
            assert!(p.compute_gflops > 0.0, "compute_gflops should be > 0.0, got {}", p.compute_gflops);
            assert!(p.vram_bandwidth_gbps > 0.0, "vram_bandwidth_gbps should be > 0.0, got {}", p.vram_bandwidth_gbps);
        }

        let ring = backend.ring_order();
        assert_eq!(ring.len(), 2);
        assert_eq!(ring, &[0, 1]);

        let islands = backend.islands();
        assert!(!islands.is_empty());
    }

    #[test]
    fn test_vulkan_ring_vs_hierarchical_equivalence() {
        let rank_counts = [2, 4];
        let len = 1003; // Prime uneven length

        for &ranks in &rank_counts {
            // Run under HC4J_COLLECTIVE_ALGO=ring
            let res_ring = {
                unsafe { std::env::set_var("HC4J_COLLECTIVE_ALGO", "ring") };
                let backend = match VulkanP2pBackend::new(ranks as u32) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("Skipping Vulkan test for {ranks} ranks: {e}");
                        return;
                    }
                };
                let mut bufs = Vec::new();
                for r in 0..ranks {
                    let id = backend.alloc(r, len as u64).expect("alloc rank");
                    let data: Vec<f32> = (0..len).map(|i| (r * len + i + 1) as f32).collect();
                    backend.write(id, &data).expect("write rank data");
                    bufs.push(id);
                }
                backend.allreduce_sum(&bufs).expect("allreduce_sum ring");
                let mut out = vec![0.0f32; len];
                backend.read(bufs[0], &mut out).expect("read rank 0");
                for &id in &bufs {
                    backend.free(id).expect("free rank buf");
                }
                out
            };

            // Run under HC4J_COLLECTIVE_ALGO=hierarchical
            let res_hierarchical = {
                unsafe { std::env::set_var("HC4J_COLLECTIVE_ALGO", "hierarchical") };
                let backend = match VulkanP2pBackend::new(ranks as u32) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("Skipping Vulkan test for {ranks} ranks: {e}");
                        return;
                    }
                };
                let mut bufs = Vec::new();
                for r in 0..ranks {
                    let id = backend.alloc(r, len as u64).expect("alloc rank");
                    let data: Vec<f32> = (0..len).map(|i| (r * len + i + 1) as f32).collect();
                    backend.write(id, &data).expect("write rank data");
                    bufs.push(id);
                }
                backend.allreduce_sum(&bufs).expect("allreduce_sum hierarchical");
                let mut out = vec![0.0f32; len];
                backend.read(bufs[0], &mut out).expect("read rank 0");
                for &id in &bufs {
                    backend.free(id).expect("free rank buf");
                }
                out
            };

            unsafe { std::env::remove_var("HC4J_COLLECTIVE_ALGO") };

            // Verify numerical equivalence <= 1e-4
            for i in 0..len {
                assert!(
                    (res_ring[i] - res_hierarchical[i]).abs() <= 1e-4,
                    "mismatch at element {i} for ranks={ranks}: ring={}, hierarchical={}",
                    res_ring[i],
                    res_hierarchical[i]
                );
            }
        }
    }

    fn cpu_gemm_bias_act(
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        n: usize,
        k: usize,
        act: u32,
    ) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for i in 0..k {
                    acc += a[row * k + i] as f64 * b[i * n + col] as f64;
                }
                if let Some(bias_slice) = bias {
                    acc += bias_slice[col] as f64;
                }
                let x = acc as f32;
                let val = match act {
                    0 => x,
                    1 => {
                        let cubic = 0.7978845608028654 * (x as f64 + 0.044715 * (x as f64).powi(3));
                        let inner = cubic.clamp(-10.0, 10.0);
                        (0.5 * x as f64 * (1.0 + inner.tanh())) as f32
                    }
                    2 => {
                        let x_f64 = x as f64;
                        let e = (-x_f64.abs()).exp();
                        let num = if x_f64 >= 0.0 { 1.0 } else { e };
                        (x_f64 * num / (1.0 + e)) as f32
                    }
                    3 => x.max(0.0),
                    _ => x,
                };
                c[row * n + col] = val;
            }
        }
        c
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_split_kv_attn(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        local_tokens: usize,
        scale: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut out_o = vec![0.0f32; num_heads * head_dim];
        let mut out_stats = vec![0.0f32; num_heads * 2];
        let group_size = num_heads / num_kv_heads;

        for h in 0..num_heads {
            let kv_h = h / group_size;
            let mut scores = Vec::with_capacity(local_tokens);
            let mut m_h = -f64::INFINITY;

            for t in 0..local_tokens {
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += q[h * head_dim + d] as f64 * k[(kv_h * local_tokens + t) * head_dim + d] as f64;
                }
                let score = dot * scale as f64;
                if score > m_h {
                    m_h = score;
                }
                scores.push(score);
            }

            let mut l_h = 0.0f64;
            let mut weights = Vec::with_capacity(local_tokens);
            for &s in &scores {
                let w = (s - m_h).exp();
                weights.push(w);
                l_h += w;
            }

            out_stats[2 * h] = m_h as f32;
            out_stats[2 * h + 1] = l_h as f32;

            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for t in 0..local_tokens {
                    acc += weights[t] * v[(kv_h * local_tokens + t) * head_dim + d] as f64;
                }
                out_o[h * head_dim + d] = (acc / l_h) as f32;
            }
        }

        (out_o, out_stats)
    }

    #[test]
    fn test_vulkan_p2p_allreduce_sum_subset() {
        let backend = match VulkanP2pBackend::new(4) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skipping Vulkan test (ICD unavailable): {e}");
                return;
            }
        };

        let len = 1003;
        let b0 = backend.alloc(0, len as u64).unwrap();
        let b1 = backend.alloc(1, len as u64).unwrap();
        let b2 = backend.alloc(2, len as u64).unwrap();
        let b3 = backend.alloc(3, len as u64).unwrap();

        let data0: Vec<f32> = (0..len).map(|i| (i + 1) as f32).collect();
        let data1: Vec<f32> = (0..len).map(|i| (i * 2 + 3) as f32).collect();
        let data2: Vec<f32> = (0..len).map(|i| (i * 3 + 5) as f32).collect();
        let data3: Vec<f32> = (0..len).map(|i| (i * 4 + 7) as f32).collect();

        backend.write(b0, &data0).unwrap();
        backend.write(b1, &data1).unwrap();
        backend.write(b2, &data2).unwrap();
        backend.write(b3, &data3).unwrap();

        // 1-of-2 subset: 1-rank is valid no-op
        backend.allreduce_sum_subset(&[b0]).unwrap();
        let mut check_b0 = vec![0.0f32; len];
        backend.read(b0, &mut check_b0).unwrap();
        assert_eq!(check_b0, data0, "1-rank subset must be a no-op");

        // 2-of-4 subset: {0, 2}
        backend.allreduce_sum_subset(&[b0, b2]).unwrap();

        let mut res0 = vec![0.0f32; len];
        let mut res1 = vec![0.0f32; len];
        let mut res2 = vec![0.0f32; len];
        let mut res3 = vec![0.0f32; len];

        backend.read(b0, &mut res0).unwrap();
        backend.read(b1, &mut res1).unwrap();
        backend.read(b2, &mut res2).unwrap();
        backend.read(b3, &mut res3).unwrap();

        for i in 0..len {
            let expected_sum = data0[i] + data2[i];
            assert!((res0[i] - expected_sum).abs() < 1e-4);
            assert!((res2[i] - expected_sum).abs() < 1e-4);
            assert_eq!(res1[i], data1[i], "pruned rank 1 must be untouched");
            assert_eq!(res3[i], data3[i], "pruned rank 3 must be untouched");
        }

        // 3-of-4 subset: {0, 1, 3}
        backend.allreduce_sum_subset(&[b0, b1, b3]).unwrap();
        backend.read(b0, &mut res0).unwrap();
        backend.read(b1, &mut res1).unwrap();
        backend.read(b2, &mut res2).unwrap();
        backend.read(b3, &mut res3).unwrap();

        for i in 0..len {
            let expected_sum = (data0[i] + data2[i]) + data1[i] + data3[i];
            assert!((res0[i] - expected_sum).abs() < 1e-3);
            assert!((res1[i] - expected_sum).abs() < 1e-3);
            assert!((res3[i] - expected_sum).abs() < 1e-3);
            // Rank 2 should remain untouched from previous step
            assert!((res2[i] - (data0[i] + data2[i])).abs() < 1e-4);
        }

        backend.free(b0).unwrap();
        backend.free(b1).unwrap();
        backend.free(b2).unwrap();
        backend.free(b3).unwrap();
    }

    #[test]
    fn test_vulkan_p2p_gemm_bias_act_ragged() {
        let backend = match VulkanP2pBackend::new(1) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skipping Vulkan test (ICD unavailable): {e}");
                return;
            }
        };

        let m = 17usize;
        let k = 29usize;
        let n = 13usize;

        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect();
        let bias_data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let buf_a = backend.alloc(0, (m * k) as u64).unwrap();
        let buf_b = backend.alloc(0, (k * n) as u64).unwrap();
        let buf_bias = backend.alloc(0, n as u64).unwrap();
        let buf_c = backend.alloc(0, (m * n) as u64).unwrap();

        backend.write(buf_a, &a_data).unwrap();
        backend.write(buf_b, &b_data).unwrap();
        backend.write(buf_bias, &bias_data).unwrap();

        // 1. NONE, no bias
        backend.gemm_bias_act(0, buf_a, buf_b, 0, buf_c, m as u32, n as u32, k as u32, 0).unwrap();
        let mut c_res = vec![0.0f32; m * n];
        backend.read(buf_c, &mut c_res).unwrap();
        let cpu_ref = cpu_gemm_bias_act(&a_data, &b_data, None, m, n, k, 0);
        for i in 0..m * n {
            assert!((c_res[i] - cpu_ref[i]).abs() <= 1e-4, "NONE mismatch at {i}: got {}, expected {}", c_res[i], cpu_ref[i]);
        }

        // 2. BIAS + GELU (flags = 1 | (1 << 1) = 3)
        backend.gemm_bias_act(0, buf_a, buf_b, buf_bias, buf_c, m as u32, n as u32, k as u32, 3).unwrap();
        backend.read(buf_c, &mut c_res).unwrap();
        let cpu_ref = cpu_gemm_bias_act(&a_data, &b_data, Some(&bias_data), m, n, k, 1);
        for i in 0..m * n {
            assert!((c_res[i] - cpu_ref[i]).abs() <= 1e-4, "GELU mismatch at {i}: got {}, expected {}", c_res[i], cpu_ref[i]);
        }

        // 3. BIAS + SILU (flags = 1 | (2 << 1) = 5)
        backend.gemm_bias_act(0, buf_a, buf_b, buf_bias, buf_c, m as u32, n as u32, k as u32, 5).unwrap();
        backend.read(buf_c, &mut c_res).unwrap();
        let cpu_ref = cpu_gemm_bias_act(&a_data, &b_data, Some(&bias_data), m, n, k, 2);
        for i in 0..m * n {
            assert!((c_res[i] - cpu_ref[i]).abs() <= 1e-4, "SILU mismatch at {i}: got {}, expected {}", c_res[i], cpu_ref[i]);
        }

        // 4. BIAS + RELU (flags = 1 | (3 << 1) = 7)
        backend.gemm_bias_act(0, buf_a, buf_b, buf_bias, buf_c, m as u32, n as u32, k as u32, 7).unwrap();
        backend.read(buf_c, &mut c_res).unwrap();
        let cpu_ref = cpu_gemm_bias_act(&a_data, &b_data, Some(&bias_data), m, n, k, 3);
        for i in 0..m * n {
            assert!((c_res[i] - cpu_ref[i]).abs() <= 1e-4, "RELU mismatch at {i}: got {}, expected {}", c_res[i], cpu_ref[i]);
        }

        backend.free(buf_a).unwrap();
        backend.free(buf_b).unwrap();
        backend.free(buf_bias).unwrap();
        backend.free(buf_c).unwrap();
    }

    #[test]
    fn test_vulkan_p2p_split_kv_attn_mha_and_gqa() {
        let backend = match VulkanP2pBackend::new(1) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skipping Vulkan test (ICD unavailable): {e}");
                return;
            }
        };

        // Test 1: MHA (num_heads=4, num_kv_heads=4, head_dim=64, local_tokens=300)
        {
            let num_heads = 4usize;
            let num_kv_heads = 4usize;
            let head_dim = 64usize;
            let local_tokens = 300usize;
            let scale = 0.125f32;

            let q_data: Vec<f32> = (0..num_heads * head_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
            let k_data: Vec<f32> = (0..num_kv_heads * local_tokens * head_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
            let v_data: Vec<f32> = (0..num_kv_heads * local_tokens * head_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();

            let buf_q = backend.alloc(0, (num_heads * head_dim) as u64).unwrap();
            let buf_k = backend.alloc(0, (num_kv_heads * local_tokens * head_dim) as u64).unwrap();
            let buf_v = backend.alloc(0, (num_kv_heads * local_tokens * head_dim) as u64).unwrap();
            let buf_o = backend.alloc(0, (num_heads * head_dim) as u64).unwrap();
            let buf_stats = backend.alloc(0, (num_heads * 2) as u64).unwrap();

            backend.write(buf_q, &q_data).unwrap();
            backend.write(buf_k, &k_data).unwrap();
            backend.write(buf_v, &v_data).unwrap();

            backend.split_kv_attn(
                0,
                buf_q,
                buf_k,
                buf_v,
                buf_o,
                buf_stats,
                num_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                local_tokens as u32,
                scale,
            ).unwrap();

            let mut out_o = vec![0.0f32; num_heads * head_dim];
            let mut out_stats = vec![0.0f32; num_heads * 2];
            backend.read(buf_o, &mut out_o).unwrap();
            backend.read(buf_stats, &mut out_stats).unwrap();

            let (ref_o, ref_stats) = cpu_split_kv_attn(
                &q_data,
                &k_data,
                &v_data,
                num_heads,
                num_kv_heads,
                head_dim,
                local_tokens,
                scale,
            );

            for h in 0..num_heads {
                assert!((out_stats[2 * h] - ref_stats[2 * h]).abs() <= 1e-4, "m_h mismatch at head {h}: got {}, exp {}", out_stats[2 * h], ref_stats[2 * h]);
                assert!((out_stats[2 * h + 1] - ref_stats[2 * h + 1]).abs() / ref_stats[2 * h + 1] <= 1e-3, "l_h mismatch at head {h}: got {}, exp {}", out_stats[2 * h + 1], ref_stats[2 * h + 1]);
            }
            for i in 0..num_heads * head_dim {
                assert!((out_o[i] - ref_o[i]).abs() <= 1e-4, "MHA out_o mismatch at {i}: got {}, exp {}", out_o[i], ref_o[i]);
            }

            backend.free(buf_q).unwrap();
            backend.free(buf_k).unwrap();
            backend.free(buf_v).unwrap();
            backend.free(buf_o).unwrap();
            backend.free(buf_stats).unwrap();
        }

        // Test 2: GQA (num_heads=8, num_kv_heads=2, head_dim=64, local_tokens=513)
        {
            let num_heads = 8usize;
            let num_kv_heads = 2usize;
            let head_dim = 64usize;
            let local_tokens = 513usize;
            let scale = 0.125f32;

            let q_data: Vec<f32> = (0..num_heads * head_dim).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
            let k_data: Vec<f32> = (0..num_kv_heads * local_tokens * head_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
            let v_data: Vec<f32> = (0..num_kv_heads * local_tokens * head_dim).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();

            let buf_q = backend.alloc(0, (num_heads * head_dim) as u64).unwrap();
            let buf_k = backend.alloc(0, (num_kv_heads * local_tokens * head_dim) as u64).unwrap();
            let buf_v = backend.alloc(0, (num_kv_heads * local_tokens * head_dim) as u64).unwrap();
            let buf_o = backend.alloc(0, (num_heads * head_dim) as u64).unwrap();
            let buf_stats = backend.alloc(0, (num_heads * 2) as u64).unwrap();

            backend.write(buf_q, &q_data).unwrap();
            backend.write(buf_k, &k_data).unwrap();
            backend.write(buf_v, &v_data).unwrap();

            backend.split_kv_attn(
                0,
                buf_q,
                buf_k,
                buf_v,
                buf_o,
                buf_stats,
                num_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                local_tokens as u32,
                scale,
            ).unwrap();

            let mut out_o = vec![0.0f32; num_heads * head_dim];
            let mut out_stats = vec![0.0f32; num_heads * 2];
            backend.read(buf_o, &mut out_o).unwrap();
            backend.read(buf_stats, &mut out_stats).unwrap();

            let (ref_o, ref_stats) = cpu_split_kv_attn(
                &q_data,
                &k_data,
                &v_data,
                num_heads,
                num_kv_heads,
                head_dim,
                local_tokens,
                scale,
            );

            for h in 0..num_heads {
                assert!((out_stats[2 * h] - ref_stats[2 * h]).abs() <= 1e-4, "GQA m_h mismatch at head {h}: got {}, exp {}", out_stats[2 * h], ref_stats[2 * h]);
                assert!((out_stats[2 * h + 1] - ref_stats[2 * h + 1]).abs() / ref_stats[2 * h + 1] <= 1e-3, "GQA l_h mismatch at head {h}: got {}, exp {}", out_stats[2 * h + 1], ref_stats[2 * h + 1]);
            }
            for i in 0..num_heads * head_dim {
                assert!((out_o[i] - ref_o[i]).abs() <= 1e-4, "GQA out_o mismatch at {i}: got {}, exp {}", out_o[i], ref_o[i]);
            }

            backend.free(buf_q).unwrap();
            backend.free(buf_k).unwrap();
            backend.free(buf_v).unwrap();
            backend.free(buf_o).unwrap();
            backend.free(buf_stats).unwrap();
        }
    }
}
