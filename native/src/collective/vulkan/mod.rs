//! Raw-Vulkan P2P Collective Backend (`VulkanP2pBackend`).
//!
//! Implements hardware-level peer-to-peer Ring AllReduce on Windows and Linux
//! using Vulkan 1.2 Timeline Semaphores, Windows `OPAQUE_WIN32` / `VK_EXT_external_memory_host`,
//! and Linux `dma_buf`.

pub(crate) mod device;
pub(crate) mod pipeline;
pub(crate) mod ring;
pub(crate) mod topology;
pub(crate) mod transport;

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use ash::vk;

use crate::collective::backend::{CollectiveBackend, LinkInfo};
use crate::collective::vulkan::device::{BufMem, GpuNode, alloc_device_local, alloc_staging, build_node, create_instance, destroy_buf, device_name, device_type_str, enum_physical_devices, one_shot_submit};
use crate::collective::vulkan::pipeline::ReducePipeline;
use crate::collective::vulkan::ring::execute_ring_allreduce;
use crate::collective::vulkan::topology::{TopoGraph, build_topology};
use crate::error::{Hc4jError, Hc4jResult};

pub struct VulkanP2pBackend {
    _entry: ash::Entry,
    instance: ash::Instance,
    nodes: Vec<GpuNode>,
    pipelines: Mutex<Vec<ReducePipeline>>,
    graph: TopoGraph,
    buffers: Mutex<HashMap<u64, (usize, BufMem, usize)>>,
    next_buffer_id: AtomicU64,
}

impl VulkanP2pBackend {
    pub fn new(ranks: u32) -> Hc4jResult<Self> {
        if ranks == 0 {
            return Err(Hc4jError::InvalidParam("collective needs at least 1 rank"));
        }

        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| Hc4jError::Device(format!("failed to load Vulkan entry point: {e}")))?;

        let instance = create_instance(&entry)?;

        let phds = enum_physical_devices(&instance)?;
        let num_physical = phds.len();

        let mut nodes = Vec::with_capacity(ranks as usize);
        for i in 0..ranks as usize {
            let phd = phds[i % num_physical];
            let node = build_node(&instance, phd, i)?;
            nodes.push(node);
        }

        let graph = build_topology(&nodes)?;

        let mut pipelines = Vec::with_capacity(ranks as usize);
        for node in &nodes {
            pipelines.push(ReducePipeline::new(node)?);
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
            eprintln!(
                "[HC4J] collective rank {}: {} | {}",
                i,
                device_name(&node.props),
                device_type_str(node.props.device_type)
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

        Ok(Self {
            _entry: entry,
            instance,
            nodes,
            pipelines: Mutex::new(pipelines),
            graph,
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

        if n <= 1 {
            // Fix #6: 1-rank AllReduce is a valid no-op
            return Ok(());
        }

        let (raw_buffers, elements) = {
            let map = self.buffers.lock().unwrap();
            let mut raw = Vec::with_capacity(n);
            let mut elem_count = None;

            for (i, &buf_id) in buffers.iter().enumerate() {
                let &(rank, ref buf, count) = map.get(&buf_id)
                    .ok_or(Hc4jError::InvalidParam("invalid collective buffer handle"))?;
                if rank != i {
                    return Err(Hc4jError::InvalidParam("buffer handle does not belong to expected rank"));
                }
                if let Some(expected) = elem_count {
                    if count != expected {
                        return Err(Hc4jError::InvalidParam("all collective buffers must have identical size"));
                    }
                } else {
                    elem_count = Some(count);
                }
                raw.push(buf.buffer);
            }

            (raw, elem_count.unwrap_or(0))
        };

        let mut pipelines = self.pipelines.lock().unwrap();
        execute_ring_allreduce(&self.nodes, &mut pipelines, &self.graph, &raw_buffers, elements)
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

        // 2. Free all user allocated buffers
        let mut user_bufs = self.buffers.lock().unwrap();
        for (_, (rank, buf, _)) in user_bufs.drain() {
            destroy_buf(&self.nodes[rank], &buf);
        }

        // 3. Destroy all pipelines, scratch buffers, and descriptor pools
        let mut pipelines = self.pipelines.lock().unwrap();
        for (i, pipe) in pipelines.iter_mut().enumerate() {
            pipe.destroy(&self.nodes[i]);
        }

        // 4. Destroy all GpuNodes (command pools & logical devices)
        for node in &self.nodes {
            node.destroy();
        }

        // 5. Destroy Vulkan instance
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
        use crate::collective::backend::{
            TRANSPORT_HOST_SHARED_DMA, TRANSPORT_HOST_STAGED, TRANSPORT_PEER_TO_PEER,
        };

        let tiers = [
            ("staged", TRANSPORT_HOST_STAGED),
            ("p2p", TRANSPORT_PEER_TO_PEER),
            #[cfg(target_os = "windows")]
            ("host_dma", TRANSPORT_HOST_SHARED_DMA),
        ];

        for (tier_name, expected_code) in tiers {
            // SAFETY: Tests are run sequentially within this test function
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
}
