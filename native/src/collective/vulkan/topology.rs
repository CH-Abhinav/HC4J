//! P2P topology micro-benchmarking, dynamic relative tier selection, and LinkInfo construction.

use std::collections::HashMap;
use std::time::Instant;
use ash::vk;

use crate::collective::backend::{DeviceProfileC, LinkInfo};
use crate::collective::cache::{
    compute_fingerprint, get_cache_path, load_cache, serialize_cache, should_bypass_cache,
    write_cache_atomic, CachedDevice, CachedEdge, PhysDeviceSummary,
};
use crate::collective::vulkan::device::{
    GpuNode, alloc_device_local, alloc_staging, destroy_buf, device_name, one_shot_submit,
};
use crate::collective::vulkan::profile::{
    map_device_type, measure_compute_gflops, query_vram_bytes,
};
use crate::collective::vulkan::transport::TransportTier;
use crate::error::{Hc4jError, Hc4jResult};

#[cfg(target_os = "windows")]
use crate::collective::vulkan::transport::{
    VirtualAllocGuard, Win32HandleGuard, alloc_exportable_win32, alloc_host_imported, export_win32_handle,
    import_win32_handle,
};

#[cfg(target_os = "linux")]
use crate::collective::vulkan::transport::{
    alloc_exportable_dmabuf, export_dma_buf_fd, import_dma_buf_fd,
};

/// Size of the micro-benchmark buffer: 4 MiB (accurate measurement without startup delay).
pub(crate) const BENCHMARK_BUFFER_SIZE: vk::DeviceSize = 4 * 1024 * 1024;

/// Number of copy iterations for averaging.
pub(crate) const BENCHMARK_ITERATIONS: u32 = 4;

/// Acceptance ratio for direct P2P / Host DMA compared to HOST_STAGED baseline.
/// Direct tiers avoid CPU memcpy and host staging overhead, so accept them whenever
/// their throughput is at least 90% of the staged baseline.
pub(crate) const RELATIVE_ACCEPT_RATIO: f64 = 0.90;

/// Default estimated bandwidth for Tier 2 host-staged transfers if benchmark fails.
pub(crate) const TIER2_DEFAULT_BW_GBPS: f64 = 1.2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ForceTier {
    Auto,
    P2p,
    HostDma,
    Staged,
}

pub(crate) fn parse_force_tier(val: Option<&str>) -> Hc4jResult<ForceTier> {
    match val {
        None => Ok(ForceTier::Auto),
        Some(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
                Ok(ForceTier::Auto)
            } else if trimmed.eq_ignore_ascii_case("p2p") {
                Ok(ForceTier::P2p)
            } else if trimmed.eq_ignore_ascii_case("host_dma") {
                Ok(ForceTier::HostDma)
            } else if trimmed.eq_ignore_ascii_case("staged") {
                Ok(ForceTier::Staged)
            } else {
                Err(Hc4jError::InvalidParam(
                    "HC4J_COLLECTIVE_FORCE_TIER must be one of: auto, p2p, host_dma, staged",
                ))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TopoEdge {
    pub src: usize,
    pub dst: usize,
    pub tier: TransportTier,
}

pub(crate) struct TopoGraph {
    pub edges: Vec<TopoEdge>,
    pub links: Vec<LinkInfo>,
}

impl TopoGraph {
    pub fn tier(&self, src: usize, dst: usize) -> &TransportTier {
        self.edges
            .iter()
            .find(|e| e.src == src && e.dst == dst)
            .map(|e| &e.tier)
            .expect("BUG: no topology edge for requested GPU pair")
    }
}

/// Probe Tier 1 end-to-end hop:
/// src_data -> export_buf on src, then imported_buf -> dst_recv on dst.
#[cfg(target_os = "windows")]
pub(crate) fn probe_tier1(src: &GpuNode, dst: &GpuNode) -> Result<f64, &'static str> {
    if !src.ext.external_memory_win32 || !dst.ext.external_memory_win32 {
        return Err("ext-missing");
    }
    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    let src_data = alloc_device_local(src, BENCHMARK_BUFFER_SIZE, usage)
        .map_err(|_| "src-alloc-failed")?;

    let export_buf = match alloc_exportable_win32(src, BENCHMARK_BUFFER_SIZE, usage) {
        Some(b) => b,
        None => {
            destroy_buf(src, &src_data);
            return Err("export-alloc-failed");
        }
    };

    let handle = match export_win32_handle(src, export_buf.memory) {
        Some(h) => h,
        None => {
            destroy_buf(src, &export_buf);
            destroy_buf(src, &src_data);
            return Err("export-failed");
        }
    };
    let _guard = Win32HandleGuard(handle);

    let dst_buf = match import_win32_handle(dst, BENCHMARK_BUFFER_SIZE, handle, usage) {
        Some(b) => b,
        None => {
            destroy_buf(src, &export_buf);
            destroy_buf(src, &src_data);
            return Err("import-rejected");
        }
    };

    let dst_recv = match alloc_device_local(dst, BENCHMARK_BUFFER_SIZE, usage) {
        Ok(b) => b,
        Err(_) => {
            destroy_buf(dst, &dst_buf);
            destroy_buf(src, &export_buf);
            destroy_buf(src, &src_data);
            return Err("dst-alloc-failed");
        }
    };

    let region = vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: BENCHMARK_BUFFER_SIZE,
    };

    // Warm-up iteration
    let _ = one_shot_submit(src, |cb| unsafe {
        src.device.cmd_copy_buffer(cb, src_data.buffer, export_buf.buffer, &[region]);
    });
    let _ = one_shot_submit(dst, |cb| unsafe {
        dst.device.cmd_copy_buffer(cb, dst_buf.buffer, dst_recv.buffer, &[region]);
    });

    let t0 = Instant::now();
    for _ in 0..BENCHMARK_ITERATIONS {
        let _ = one_shot_submit(src, |cb| unsafe {
            src.device.cmd_copy_buffer(cb, src_data.buffer, export_buf.buffer, &[region]);
        });
        let _ = one_shot_submit(dst, |cb| unsafe {
            dst.device.cmd_copy_buffer(cb, dst_buf.buffer, dst_recv.buffer, &[region]);
        });
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total_bytes = BENCHMARK_BUFFER_SIZE as f64 * BENCHMARK_ITERATIONS as f64;
    let gbps = if elapsed > 0.0 { total_bytes / elapsed / 1e9 } else { 0.0 };

    destroy_buf(dst, &dst_recv);
    destroy_buf(dst, &dst_buf);
    destroy_buf(src, &export_buf);
    destroy_buf(src, &src_data);

    Ok(gbps)
}

/// Probe Tier 1 on Linux (dma_buf).
#[cfg(target_os = "linux")]
pub(crate) fn probe_tier1(src: &GpuNode, dst: &GpuNode) -> Result<f64, &'static str> {
    if !src.ext.external_memory_fd || !dst.ext.external_memory_fd {
        return Err("ext-missing");
    }
    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    let src_data = alloc_device_local(src, BENCHMARK_BUFFER_SIZE, usage)
        .map_err(|_| "src-alloc-failed")?;

    let export_buf = match alloc_exportable_dmabuf(src, BENCHMARK_BUFFER_SIZE, usage) {
        Some(b) => b,
        None => {
            destroy_buf(src, &src_data);
            return Err("export-alloc-failed");
        }
    };

    let fd = match export_dma_buf_fd(src, export_buf.memory) {
        Some(fd) => fd,
        None => {
            destroy_buf(src, &export_buf);
            destroy_buf(src, &src_data);
            return Err("export-failed");
        }
    };

    let dst_buf = match import_dma_buf_fd(dst, BENCHMARK_BUFFER_SIZE, fd, usage) {
        Some(b) => b,
        None => {
            destroy_buf(src, &export_buf);
            destroy_buf(src, &src_data);
            return Err("import-rejected");
        }
    };

    let dst_recv = match alloc_device_local(dst, BENCHMARK_BUFFER_SIZE, usage) {
        Ok(b) => b,
        Err(_) => {
            destroy_buf(dst, &dst_buf);
            destroy_buf(src, &export_buf);
            destroy_buf(src, &src_data);
            return Err("dst-alloc-failed");
        }
    };

    let region = vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: BENCHMARK_BUFFER_SIZE,
    };

    // Warm-up iteration
    let _ = one_shot_submit(src, |cb| unsafe {
        src.device.cmd_copy_buffer(cb, src_data.buffer, export_buf.buffer, &[region]);
    });
    let _ = one_shot_submit(dst, |cb| unsafe {
        dst.device.cmd_copy_buffer(cb, dst_buf.buffer, dst_recv.buffer, &[region]);
    });

    let t0 = Instant::now();
    for _ in 0..BENCHMARK_ITERATIONS {
        let _ = one_shot_submit(src, |cb| unsafe {
            src.device.cmd_copy_buffer(cb, src_data.buffer, export_buf.buffer, &[region]);
        });
        let _ = one_shot_submit(dst, |cb| unsafe {
            dst.device.cmd_copy_buffer(cb, dst_buf.buffer, dst_recv.buffer, &[region]);
        });
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total_bytes = BENCHMARK_BUFFER_SIZE as f64 * BENCHMARK_ITERATIONS as f64;
    let gbps = if elapsed > 0.0 { total_bytes / elapsed / 1e9 } else { 0.0 };

    destroy_buf(dst, &dst_recv);
    destroy_buf(dst, &dst_buf);
    destroy_buf(src, &export_buf);
    destroy_buf(src, &src_data);

    Ok(gbps)
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
pub(crate) fn probe_tier1(_src: &GpuNode, _dst: &GpuNode) -> Result<f64, &'static str> {
    Err("unsupported-os")
}

/// Probe Tier 1.5 end-to-end hop using VK_EXT_external_memory_host (Windows):
/// src_data -> host_buf_src on src, then host_buf_dst -> dst_recv on dst.
#[cfg(target_os = "windows")]
pub(crate) fn probe_tier1_5(src: &GpuNode, dst: &GpuNode) -> Result<f64, &'static str> {
    if !src.ext.external_memory_host || !dst.ext.external_memory_host {
        return Err("ext-missing");
    }
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE,
    };

    let align = src.min_host_ptr_align.max(dst.min_host_ptr_align).max(4096);
    let aligned_size = (BENCHMARK_BUFFER_SIZE + align - 1) & !(align - 1);

    let host_ptr = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            aligned_size as usize,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        )
    };
    if host_ptr.is_null() {
        return Err("host-alloc-failed");
    }
    let _guard = VirtualAllocGuard { ptr: host_ptr };

    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    let host_buf_src = match alloc_host_imported(src, host_ptr, aligned_size, usage) {
        Some(b) => b,
        None => return Err("src-import-failed"),
    };
    let host_buf_dst = match alloc_host_imported(dst, host_ptr, aligned_size, usage) {
        Some(b) => b,
        None => {
            destroy_buf(src, &host_buf_src);
            return Err("dst-import-failed");
        }
    };

    let src_data = match alloc_device_local(src, BENCHMARK_BUFFER_SIZE, usage) {
        Ok(b) => b,
        Err(_) => {
            destroy_buf(src, &host_buf_src);
            destroy_buf(dst, &host_buf_dst);
            return Err("src-alloc-failed");
        }
    };

    let dst_recv = match alloc_device_local(dst, BENCHMARK_BUFFER_SIZE, usage) {
        Ok(b) => b,
        Err(_) => {
            destroy_buf(src, &src_data);
            destroy_buf(src, &host_buf_src);
            destroy_buf(dst, &host_buf_dst);
            return Err("dst-alloc-failed");
        }
    };

    let region = vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: BENCHMARK_BUFFER_SIZE,
    };

    // Warm-up full hop
    let _ = one_shot_submit(src, |cb| unsafe {
        src.device.cmd_copy_buffer(cb, src_data.buffer, host_buf_src.buffer, &[region]);
    });
    let _ = one_shot_submit(dst, |cb| unsafe {
        dst.device.cmd_copy_buffer(cb, host_buf_dst.buffer, dst_recv.buffer, &[region]);
    });

    let t0 = Instant::now();
    for _ in 0..BENCHMARK_ITERATIONS {
        let _ = one_shot_submit(src, |cb| unsafe {
            src.device.cmd_copy_buffer(cb, src_data.buffer, host_buf_src.buffer, &[region]);
        });
        let _ = one_shot_submit(dst, |cb| unsafe {
            dst.device.cmd_copy_buffer(cb, host_buf_dst.buffer, dst_recv.buffer, &[region]);
        });
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total_bytes = BENCHMARK_BUFFER_SIZE as f64 * BENCHMARK_ITERATIONS as f64;
    let gbps = if elapsed > 0.0 { total_bytes / elapsed / 1e9 } else { 0.0 };

    destroy_buf(dst, &dst_recv);
    destroy_buf(src, &src_data);
    destroy_buf(dst, &host_buf_dst);
    destroy_buf(src, &host_buf_src);

    Ok(gbps)
}

#[cfg(not(target_os = "windows"))]
pub(crate) fn probe_tier1_5(_src: &GpuNode, _dst: &GpuNode) -> Result<f64, &'static str> {
    Err("unsupported-os")
}

/// Probe Tier 2 (HOST_STAGED) complete end-to-end hop:
/// src_data -> staging_src -> host memcpy -> staging_dst -> dst_recv.
pub(crate) fn probe_tier2(src: &GpuNode, dst: &GpuNode) -> f64 {
    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    let src_data = match alloc_device_local(src, BENCHMARK_BUFFER_SIZE, usage) {
        Ok(b) => b,
        Err(_) => return TIER2_DEFAULT_BW_GBPS,
    };
    let dst_recv = match alloc_device_local(dst, BENCHMARK_BUFFER_SIZE, usage) {
        Ok(b) => b,
        Err(_) => {
            destroy_buf(src, &src_data);
            return TIER2_DEFAULT_BW_GBPS;
        }
    };
    let staging_src = match alloc_staging(src, BENCHMARK_BUFFER_SIZE, usage) {
        Ok(b) => b,
        Err(_) => {
            destroy_buf(dst, &dst_recv);
            destroy_buf(src, &src_data);
            return TIER2_DEFAULT_BW_GBPS;
        }
    };
    let staging_dst = match alloc_staging(dst, BENCHMARK_BUFFER_SIZE, usage) {
        Ok(b) => b,
        Err(_) => {
            destroy_buf(src, &staging_src);
            destroy_buf(dst, &dst_recv);
            destroy_buf(src, &src_data);
            return TIER2_DEFAULT_BW_GBPS;
        }
    };

    let region = vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: BENCHMARK_BUFFER_SIZE,
    };

    let execute_staged_hop = || -> bool {
        let ok_src = one_shot_submit(src, |cb| unsafe {
            src.device.cmd_copy_buffer(cb, src_data.buffer, staging_src.buffer, &[region]);
        }).is_ok();
        if !ok_src { return false; }

        let copy_ok = (|| -> Option<()> {
            let src_ptr = unsafe {
                src.device.map_memory(staging_src.memory, 0, BENCHMARK_BUFFER_SIZE, vk::MemoryMapFlags::empty())
            }.ok()?;
            let dst_ptr = unsafe {
                dst.device.map_memory(staging_dst.memory, 0, BENCHMARK_BUFFER_SIZE, vk::MemoryMapFlags::empty())
            }.ok()?;

            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr as *const u8, dst_ptr as *mut u8, BENCHMARK_BUFFER_SIZE as usize);
                src.device.unmap_memory(staging_src.memory);
                dst.device.unmap_memory(staging_dst.memory);
            }
            Some(())
        })().is_some();
        if !copy_ok { return false; }

        one_shot_submit(dst, |cb| unsafe {
            dst.device.cmd_copy_buffer(cb, staging_dst.buffer, dst_recv.buffer, &[region]);
        }).is_ok()
    };

    // Warm-up iteration
    if !execute_staged_hop() {
        destroy_buf(dst, &staging_dst);
        destroy_buf(src, &staging_src);
        destroy_buf(dst, &dst_recv);
        destroy_buf(src, &src_data);
        return TIER2_DEFAULT_BW_GBPS;
    }

    let t0 = Instant::now();
    for _ in 0..BENCHMARK_ITERATIONS {
        if !execute_staged_hop() {
            destroy_buf(dst, &staging_dst);
            destroy_buf(src, &staging_src);
            destroy_buf(dst, &dst_recv);
            destroy_buf(src, &src_data);
            return TIER2_DEFAULT_BW_GBPS;
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total_bytes = BENCHMARK_BUFFER_SIZE as f64 * BENCHMARK_ITERATIONS as f64;
    let gbps = if elapsed > 0.0 { total_bytes / elapsed / 1e9 } else { TIER2_DEFAULT_BW_GBPS };

    destroy_buf(dst, &staging_dst);
    destroy_buf(src, &staging_src);
    destroy_buf(dst, &dst_recv);
    destroy_buf(src, &src_data);

    gbps
}

/// Pure decision function: selects the optimal transport tier comparing direct tiers
/// against the measured Tier 2 (HOST_STAGED) baseline.
pub(crate) fn select_best_tier(
    t1: Result<f64, &'static str>,
    t1_5: Result<f64, &'static str>,
    t2_gbps: f64,
) -> (TransportTier, f64) {
    let t1_accept = match t1 {
        Ok(bw) => t2_gbps <= 0.0 || bw >= t2_gbps * RELATIVE_ACCEPT_RATIO,
        Err(_) => false,
    };
    let t1_5_accept = match t1_5 {
        Ok(bw) => t2_gbps <= 0.0 || bw >= t2_gbps * RELATIVE_ACCEPT_RATIO,
        Err(_) => false,
    };

    #[cfg(target_os = "windows")]
    match (t1_accept, t1_5_accept) {
        (true, true) => {
            let bw1 = t1.unwrap();
            let bw1_5 = t1_5.unwrap();
            if bw1_5 > bw1 * 1.15 {
                (TransportTier::Tier1_5HostMem { bandwidth_gbps: bw1_5 }, bw1_5)
            } else {
                (TransportTier::Tier1P2P { bandwidth_gbps: bw1 }, bw1)
            }
        }
        (true, false) => {
            let bw = t1.unwrap();
            (TransportTier::Tier1P2P { bandwidth_gbps: bw }, bw)
        }
        (false, true) => {
            let bw = t1_5.unwrap();
            (TransportTier::Tier1_5HostMem { bandwidth_gbps: bw }, bw)
        }
        (false, false) => {
            (TransportTier::Tier2Staging { bandwidth_gbps: t2_gbps }, t2_gbps)
        }
    }

    #[cfg(not(target_os = "windows"))]
    match t1_accept {
        true => {
            let bw = t1.unwrap();
            (TransportTier::Tier1P2P { bandwidth_gbps: bw }, bw)
        }
        false => {
            (TransportTier::Tier2Staging { bandwidth_gbps: t2_gbps }, t2_gbps)
        }
    }
}

#[allow(dead_code)]
pub(crate) fn build_topology(nodes: &[GpuNode]) -> Hc4jResult<TopoGraph> {
    build_topology_and_profiles(nodes).map(|(graph, _)| graph)
}

pub(crate) fn build_topology_and_profiles(
    nodes: &[GpuNode],
) -> Hc4jResult<(TopoGraph, Vec<DeviceProfileC>)> {
    let mut unique_phds: Vec<vk::PhysicalDevice> = Vec::new();
    let mut unique_node_indices: Vec<usize> = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        if !unique_phds.contains(&node.physical_device) {
            unique_phds.push(node.physical_device);
            unique_node_indices.push(i);
        }
    }
    let num_physical = unique_phds.len();

    let include_cpu = std::env::var("HC4J_COLLECTIVE_INCLUDE_CPU")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let device_names: Vec<String> = unique_node_indices
        .iter()
        .map(|&idx| device_name(&nodes[idx].props))
        .collect();

    let summaries: Vec<PhysDeviceSummary> = unique_node_indices
        .iter()
        .enumerate()
        .map(|(k, &idx)| {
            let node = &nodes[idx];
            PhysDeviceSummary {
                vendor_id: node.props.vendor_id,
                device_id: node.props.device_id,
                driver_version: node.props.driver_version,
                device_type: map_device_type(node.props.device_type),
                device_name: &device_names[k],
            }
        })
        .collect();

    let fingerprint = compute_fingerprint(include_cpu, &summaries);
    let cache_path = get_cache_path(&fingerprint);
    let bypass = should_bypass_cache();

    #[allow(clippy::collapsible_if)]
    if !bypass {
        if let Some(cached) = load_cache(&cache_path, num_physical) {
            let mut all_found = true;
            for i in 0..nodes.len() {
                for j in 0..nodes.len() {
                    if i != j {
                        let src_phys = unique_phds
                            .iter()
                            .position(|&p| p == nodes[i].physical_device)
                            .unwrap();
                        let dst_phys = unique_phds
                            .iter()
                            .position(|&p| p == nodes[j].physical_device)
                            .unwrap();
                        if !cached
                            .edges
                            .iter()
                            .any(|e| e.src_phys_idx == src_phys && e.dst_phys_idx == dst_phys)
                        {
                            all_found = false;
                            break;
                        }
                    }
                }
                if !all_found {
                    break;
                }
            }

            if all_found {
                eprintln!(
                    "[HC4J] collective topology: loaded from cache ({})",
                    cache_path.display()
                );
                let mut profiles = Vec::with_capacity(nodes.len());
                for (r, node) in nodes.iter().enumerate() {
                    let phys_idx = unique_phds
                        .iter()
                        .position(|&p| p == node.physical_device)
                        .unwrap();
                    let dev = &cached.devices[phys_idx];
                    profiles.push(DeviceProfileC {
                        rank: r as u32,
                        vendor_id: node.props.vendor_id,
                        device_id: node.props.device_id,
                        device_type: map_device_type(node.props.device_type),
                        vram_bytes: dev.vram_bytes,
                        compute_gflops: dev.compute_gflops,
                    });
                }

                let mut edges = Vec::new();
                let mut links = Vec::new();
                for i in 0..nodes.len() {
                    for j in 0..nodes.len() {
                        if i != j {
                            let src_phys = unique_phds
                                .iter()
                                .position(|&p| p == nodes[i].physical_device)
                                .unwrap();
                            let dst_phys = unique_phds
                                .iter()
                                .position(|&p| p == nodes[j].physical_device)
                                .unwrap();
                            let loopback = src_phys == dst_phys;
                            let edge = cached
                                .edges
                                .iter()
                                .find(|e| {
                                    e.src_phys_idx == src_phys && e.dst_phys_idx == dst_phys
                                })
                                .unwrap();
                            let tier = match edge.transport {
                                1 => TransportTier::Tier1P2P {
                                    bandwidth_gbps: edge.bandwidth_gbps,
                                },
                                #[cfg(target_os = "windows")]
                                2 => TransportTier::Tier1_5HostMem {
                                    bandwidth_gbps: edge.bandwidth_gbps,
                                },
                                _ => TransportTier::Tier2Staging {
                                    bandwidth_gbps: edge.bandwidth_gbps,
                                },
                            };
                            edges.push(TopoEdge { src: i, dst: j, tier });
                            links.push(LinkInfo {
                                src_rank: i as u32,
                                dst_rank: j as u32,
                                bandwidth_gbps: edge.bandwidth_gbps,
                                transport: edge.transport,
                                loopback: u32::from(loopback),
                            });
                        }
                    }
                }
                return Ok((TopoGraph { edges, links }, profiles));
            }
        }
    }

    // Cache miss or bypass: measure profiles
    let mut cached_devices = Vec::with_capacity(num_physical);
    let mut phys_profiles = Vec::with_capacity(num_physical);
    for (k, &idx) in unique_node_indices.iter().enumerate() {
        let vram = query_vram_bytes(&nodes[idx]);
        let gflops = measure_compute_gflops(&nodes[idx])?;
        cached_devices.push(CachedDevice {
            phys_idx: k,
            vram_bytes: vram,
            compute_gflops: gflops,
        });
        phys_profiles.push((vram, gflops));
    }

    let mut profiles = Vec::with_capacity(nodes.len());
    for (r, node) in nodes.iter().enumerate() {
        let phys_idx = unique_phds
            .iter()
            .position(|&p| p == node.physical_device)
            .unwrap();
        let (vram, gflops) = phys_profiles[phys_idx];
        profiles.push(DeviceProfileC {
            rank: r as u32,
            vendor_id: node.props.vendor_id,
            device_id: node.props.device_id,
            device_type: map_device_type(node.props.device_type),
            vram_bytes: vram,
            compute_gflops: gflops,
        });
    }

    let force_tier_str = std::env::var("HC4J_COLLECTIVE_FORCE_TIER").ok();
    let force_tier = parse_force_tier(force_tier_str.as_deref())?;

    let mut edges = Vec::new();
    let mut links = Vec::new();
    let mut probe_cache: HashMap<(vk::PhysicalDevice, vk::PhysicalDevice), (TransportTier, f64)> =
        HashMap::new();

    for i in 0..nodes.len() {
        for j in 0..nodes.len() {
            if i != j {
                let pair_key = (nodes[i].physical_device, nodes[j].physical_device);
                let loopback = nodes[i].physical_device == nodes[j].physical_device;

                let (tier, final_bw) = if let Some(&cached) = probe_cache.get(&pair_key) {
                    cached
                } else {
                    let (selected_tier, bw) = match force_tier {
                        ForceTier::Auto => {
                            let t2_gbps = probe_tier2(&nodes[i], &nodes[j]);
                            let t1_res = probe_tier1(&nodes[i], &nodes[j]);
                            let t1_5_res = probe_tier1_5(&nodes[i], &nodes[j]);

                            let t1_str = match t1_res {
                                Ok(bw) => format!("{bw:.1} GB/s"),
                                Err(e) => e.to_string(),
                            };
                            let t1_5_str = match t1_5_res {
                                Ok(bw) => format!("{bw:.1} GB/s"),
                                Err(e) => e.to_string(),
                            };

                            let (t, b) = select_best_tier(t1_res, t1_5_res, t2_gbps);

                            #[cfg(target_os = "windows")]
                            let t1_tag = "win32";
                            #[cfg(not(target_os = "windows"))]
                            let t1_tag = "dmabuf";

                            eprintln!(
                                "[HC4J] collective probe {}->{}: T1({})={} | T1.5(host_dma)={} | T2(staged)={:.1} GB/s -> {} (loopback={})",
                                i, j, t1_tag, t1_str, t1_5_str, t2_gbps, t.name(), loopback
                            );

                            (t, b)
                        }
                        ForceTier::P2p => {
                            let t1_res = probe_tier1(&nodes[i], &nodes[j]);
                            match t1_res {
                                Ok(bw) => {
                                    let t = TransportTier::Tier1P2P { bandwidth_gbps: bw };
                                    #[cfg(target_os = "windows")]
                                    let t1_tag = "win32";
                                    #[cfg(not(target_os = "windows"))]
                                    let t1_tag = "dmabuf";
                                    eprintln!(
                                        "[HC4J] collective probe {}->{}: T1({})={:.1} GB/s -> PEER_TO_PEER [FORCED] (loopback={})",
                                        i, j, t1_tag, bw, loopback
                                    );
                                    (t, bw)
                                }
                                Err(reason) => {
                                    eprintln!(
                                        "[HC4J] Forced Tier 1 P2P unavailable for probe {}->{}: {reason}",
                                        i, j
                                    );
                                    return Err(Hc4jError::Unsupported(
                                        "Forced Tier 1 P2P unavailable",
                                    ));
                                }
                            }
                        }
                        ForceTier::HostDma => {
                            let t1_5_res = probe_tier1_5(&nodes[i], &nodes[j]);
                            match t1_5_res {
                                Ok(bw) => {
                                    #[cfg(target_os = "windows")]
                                    {
                                        let t = TransportTier::Tier1_5HostMem { bandwidth_gbps: bw };
                                        eprintln!(
                                            "[HC4J] collective probe {}->{}: T1.5(host_dma)={:.1} GB/s -> HOST_SHARED_DMA [FORCED] (loopback={})",
                                            i, j, bw, loopback
                                        );
                                        (t, bw)
                                    }
                                    #[cfg(not(target_os = "windows"))]
                                    {
                                        return Err(Hc4jError::Unsupported(
                                            "Forced Tier 1.5 HOST_SHARED_DMA unavailable: unsupported-os",
                                        ));
                                    }
                                }
                                Err(reason) => {
                                    eprintln!(
                                        "[HC4J] Forced Tier 1.5 HOST_SHARED_DMA unavailable for probe {}->{}: {reason}",
                                        i, j
                                    );
                                    return Err(Hc4jError::Unsupported(
                                        "Forced Tier 1.5 HOST_SHARED_DMA unavailable",
                                    ));
                                }
                            }
                        }
                        ForceTier::Staged => {
                            let t2_gbps = probe_tier2(&nodes[i], &nodes[j]);
                            let t = TransportTier::Tier2Staging { bandwidth_gbps: t2_gbps };
                            eprintln!(
                                "[HC4J] collective probe {}->{}: T2(staged)={:.1} GB/s -> HOST_STAGED [FORCED] (loopback={})",
                                i, j, t2_gbps, loopback
                            );
                            (t, t2_gbps)
                        }
                    };

                    probe_cache.insert(pair_key, (selected_tier, bw));
                    (selected_tier, bw)
                };

                edges.push(TopoEdge { src: i, dst: j, tier });
                links.push(LinkInfo {
                    src_rank: i as u32,
                    dst_rank: j as u32,
                    bandwidth_gbps: final_bw,
                    transport: tier.transport_code(),
                    loopback: u32::from(loopback),
                });
            }
        }
    }

    if !bypass {
        let mut cached_edges = Vec::new();
        for (&(src_phd, dst_phd), &(tier, bw)) in &probe_cache {
            let src_phys = unique_phds.iter().position(|&p| p == src_phd).unwrap();
            let dst_phys = unique_phds.iter().position(|&p| p == dst_phd).unwrap();
            cached_edges.push(CachedEdge {
                src_phys_idx: src_phys,
                dst_phys_idx: dst_phys,
                transport: tier.transport_code(),
                bandwidth_gbps: bw,
            });
        }
        let serialized = serialize_cache(&cached_devices, &cached_edges);
        let _ = write_cache_atomic(&cache_path, &serialized);
        eprintln!("[HC4J] collective topology: probed and cached");
    }

    Ok((TopoGraph { edges, links }, profiles))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_best_tier_igpu() {
        // T1=4.5, T1.5=3.2, T2=2.8 -> PEER_TO_PEER (T1 >= 2.8 * 0.9 = 2.52)
        let (tier, bw) = select_best_tier(Ok(4.5), Ok(3.2), 2.8);
        assert_eq!(tier.name(), "PEER_TO_PEER");
        assert_eq!(bw, 4.5);
    }

    #[test]
    fn test_select_best_tier_cross_vendor_wddm() {
        // T1=Err("import-rejected"), T1.5=1.9, T2=1.1 -> HOST_SHARED_DMA on Windows, HOST_STAGED on Linux
        let (tier, bw) = select_best_tier(Err("import-rejected"), Ok(1.9), 1.1);
        #[cfg(target_os = "windows")]
        {
            assert_eq!(tier.name(), "HOST_SHARED_DMA");
            assert_eq!(bw, 1.9);
        }
        #[cfg(not(target_os = "windows"))]
        {
            assert_eq!(tier.name(), "HOST_STAGED");
            assert_eq!(bw, 1.1);
        }
    }

    #[test]
    fn test_select_best_tier_slow_p2p() {
        // T1=0.8, T1.5=Err("ext-missing"), T2=4.0 -> HOST_STAGED (0.8 < 4.0 * 0.9 = 3.6)
        let (tier, bw) = select_best_tier(Ok(0.8), Err("ext-missing"), 4.0);
        assert_eq!(tier.name(), "HOST_STAGED");
        assert_eq!(bw, 4.0);
    }

    #[test]
    fn test_select_best_tier_host_dma_faster() {
        // T1=3.0, T1.5=5.0, T2=2.5 -> HOST_SHARED_DMA on Windows (5.0 > 3.0 * 1.15)
        let (tier, bw) = select_best_tier(Ok(3.0), Ok(5.0), 2.5);
        #[cfg(target_os = "windows")]
        {
            assert_eq!(tier.name(), "HOST_SHARED_DMA");
            assert_eq!(bw, 5.0);
        }
        #[cfg(not(target_os = "windows"))]
        {
            assert_eq!(tier.name(), "PEER_TO_PEER");
            assert_eq!(bw, 3.0);
        }
    }

    #[test]
    fn test_select_best_tier_fallback_both_fail() {
        // T1=Err, T1.5=Err, T2=2.0 -> HOST_STAGED
        let (tier, bw) = select_best_tier(Err("export-failed"), Err("host-alloc-failed"), 2.0);
        assert_eq!(tier.name(), "HOST_STAGED");
        assert_eq!(bw, 2.0);
    }

    #[test]
    fn test_parse_force_tier() {
        assert_eq!(parse_force_tier(None).unwrap(), ForceTier::Auto);
        assert_eq!(parse_force_tier(Some("")).unwrap(), ForceTier::Auto);
        assert_eq!(parse_force_tier(Some("auto")).unwrap(), ForceTier::Auto);
        assert_eq!(parse_force_tier(Some("AUTO")).unwrap(), ForceTier::Auto);
        assert_eq!(parse_force_tier(Some("p2p")).unwrap(), ForceTier::P2p);
        assert_eq!(parse_force_tier(Some("P2P")).unwrap(), ForceTier::P2p);
        assert_eq!(parse_force_tier(Some("host_dma")).unwrap(), ForceTier::HostDma);
        assert_eq!(parse_force_tier(Some("HOST_DMA")).unwrap(), ForceTier::HostDma);
        assert_eq!(parse_force_tier(Some("staged")).unwrap(), ForceTier::Staged);
        assert_eq!(parse_force_tier(Some("STAGED")).unwrap(), ForceTier::Staged);
        assert!(parse_force_tier(Some("invalid")).is_err());
    }
}
