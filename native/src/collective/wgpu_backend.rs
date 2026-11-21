use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::error::{Hc4jError, Hc4jResult};

use super::backend::{is_peer_class, CollectiveBackend, DeviceProfileC, LinkInfo, Transport};
use super::planner::{detect_islands, solve_optimal_ring, IslandInfo};

/// Elements each invocation of the reduction kernel handles.
const WORKGROUP_SIZE: u32 = 256;
/// Buffer size for the bandwidth probe. 16 MiB is large enough to measure a
/// link rather than per-submission overhead.
const PROBE_BYTES: u64 = 16 << 20;
/// Averaging iterations in the probe.
const PROBE_ITERATIONS: u32 = 8;

// ============================================================================
// A rank
// ============================================================================

/// One rank: its own logical device, queue and reduction pipeline.
pub struct Node {
    pub rank: usize,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    /// Index into the adapter list, so loopback ranks can be recognised.
    pub adapter_index: usize,
    add: wgpu::ComputePipeline,
    max_workgroups: u32,
}

/// `accum[offset + i] += incoming[i]`, the scatter-reduce step.
///
/// Both storage bindings are `read_write`, as everywhere in HC4J: wgpu rejects
/// a buffer bound both ways in one dispatch, and the accumulator and the
/// incoming chunk can share one allocation.
const ADD_WGSL: &str = r#"
struct Params {
    params: vec4<u32>, // x = count, y = destination offset in elements
}

@group(0) @binding(0) var<storage, read_write> accum: array<f32>;
@group(0) @binding(1) var<storage, read_write> incoming: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    // The host folds large grids into X*Y; rebuild the linear index.
    let i = gid.x + gid.y * (nwg.x * 256u);
    if (i >= p.params.x) { return; }
    let at = p.params.y + i;
    accum[at] = accum[at] + incoming[i];
}
"#;

impl Node {
    async fn open(adapter: &wgpu::Adapter, adapter_index: usize, rank: usize) -> Result<Self, String> {
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("HC4J Collective Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("rank {rank}: failed to open a logical device: {e}"))?;

        // wgpu's default handler panics, which would cross the FFI boundary.
        device.on_uncaptured_error(std::sync::Arc::new(move |err: wgpu::Error| {
            eprintln!("[HC4J] collective rank {rank} device error: {err}");
        }));

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HC4J Collective Add"),
            source: wgpu::ShaderSource::Wgsl(ADD_WGSL.into()),
        });
        let add = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HC4J Collective Add"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                // Same reason as the engine: zero-initialising workgroup
                // storage is pathologically slow under D3D12's FXC.
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });

        Ok(Self {
            rank,
            device,
            queue,
            info: adapter.get_info(),
            adapter_index,
            add,
            max_workgroups: limits.max_compute_workgroups_per_dimension,
        })
    }

    /// Grid for `count` invocations, folded so neither dimension exceeds the
    /// per-dimension limit.
    fn grid(&self, count: u64) -> Hc4jResult<(u32, u32, u32)> {
        let groups = u32::try_from(count.div_ceil(u64::from(WORKGROUP_SIZE)))
            .map_err(|_| Hc4jError::Unsupported("collective dispatch exceeds 2^32 workgroups"))?;
        let max = self.max_workgroups.max(1);
        if groups <= max {
            return Ok((groups.max(1), 1, 1));
        }
        let y = groups.div_ceil(max);
        if y > max {
            return Err(Hc4jError::Unsupported("collective dispatch too large for a 2D grid"));
        }
        Ok((groups.div_ceil(y), y, 1))
    }

    fn storage(&self, elements: u64, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: elements * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn readback(&self, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HC4J Collective Staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Blocks until everything submitted on this rank has completed.
    fn wait(&self) -> Hc4jResult<()> {
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map(|_| ())
            .map_err(|e| Hc4jError::Device(format!("rank {}: GPU wait failed: {e}", self.rank)))
    }

    /// `dst[dst_offset .. +count] += src[0 .. count]`, submitted and awaited.
    fn add_into(&self, dst: &wgpu::Buffer, dst_offset_elems: u64, src: &wgpu::Buffer, count: u64) -> Hc4jResult<()> {
        let params: [u32; 4] = [
            u32::try_from(count).map_err(|_| Hc4jError::Unsupported("chunk exceeds 2^32 elements"))?,
            u32::try_from(dst_offset_elems).map_err(|_| Hc4jError::Unsupported("offset exceeds 2^32 elements"))?,
            0,
            0,
        ];
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HC4J Collective Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HC4J Collective Add"),
            layout: &self.add.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: uniform.as_entire_binding() },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("HC4J Collective") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HC4J Collective Add"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.add);
            pass.set_bind_group(0, &bind_group, &[]);
            let (x, y, z) = self.grid(count)?;
            pass.dispatch_workgroups(x, y, z);
        }
        self.queue.submit(Some(encoder.finish()));
        self.wait()
    }

    /// Copies `bytes` from `src` at `src_offset` into a mappable buffer and
    /// returns the host bytes.
    fn download_into(
        &self,
        src: &wgpu::Buffer,
        src_offset: u64,
        staging: &wgpu::Buffer,
        bytes: u64,
        out: &mut [u8],
    ) -> Hc4jResult<()> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("HC4J Collective Read") });
        encoder.copy_buffer_to_buffer(src, src_offset, staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        staging.slice(0..bytes).map_async(wgpu::MapMode::Read, |_| {});
        self.wait()?;
        let result = match staging.get_mapped_range(0..bytes) {
            Ok(view) => {
                out[..bytes as usize].copy_from_slice(&view);
                Ok(())
            }
            Err(e) => Err(Hc4jError::Readback(format!("collective readback failed: {e}"))),
        };
        staging.unmap();
        result
    }

    /// Uploads host bytes into `dst` at `dst_offset` and waits.
    fn upload(&self, dst: &wgpu::Buffer, dst_offset: u64, data: &[u8]) -> Hc4jResult<()> {
        self.queue.write_buffer(dst, dst_offset, data);
        self.queue.submit(std::iter::empty());
        self.wait()
    }
}

// ============================================================================
// WgpuStagedBackend (safe wgpu host-staged collective)
// ============================================================================

struct Buf {
    rank: usize,
    buffer: wgpu::Buffer,
    elements: u64,
}

/// Safe wgpu host-staged collective backend.
pub struct WgpuStagedBackend {
    nodes: Vec<Node>,
    links: OnceLock<Vec<LinkInfo>>,
    profiles: Vec<DeviceProfileC>,
    ring_order: OnceLock<Vec<u32>>,
    islands: OnceLock<Vec<IslandInfo>>,
    buffers: Mutex<HashMap<u64, Buf>>,
    next_buffer: Mutex<u64>,
}

pub type Collective = WgpuStagedBackend;

impl WgpuStagedBackend {
    /// Opens `ranks` devices, spreading them over the available adapters and
    /// reusing adapters (loopback) when there are fewer adapters than ranks.
    pub fn create(ranks: usize) -> Hc4jResult<Self> {
        if ranks < 2 {
            return Err(Hc4jError::InvalidParam("a collective needs at least 2 ranks"));
        }
        pollster::block_on(Self::create_async(ranks))
            .map_err(Hc4jError::Device)
    }

    async fn create_async(ranks: usize) -> Result<Self, String> {
        let backends = crate::requested_backends();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let all = instance.enumerate_adapters(backends).await;
        if all.is_empty() {
            return Err("no GPU adapter available for a collective".to_string());
        }

        let mut physical: Vec<usize> = Vec::new();
        for (i, adapter) in all.iter().enumerate() {
            let info = adapter.get_info();
            let same = physical.iter().position(|&j| {
                let other = all[j].get_info();
                other.vendor == info.vendor && other.device == info.device && other.name == info.name
            });
            match same {
                Some(slot) => {
                    let incumbent = all[physical[slot]].get_info().backend;
                    if crate::backend_rank(info.backend) > crate::backend_rank(incumbent) {
                        physical[slot] = i;
                    }
                }
                None => physical.push(i),
            }
        }

        let real: Vec<usize> = physical
            .iter()
            .copied()
            .filter(|&i| all[i].get_info().device_type != wgpu::DeviceType::Cpu)
            .collect();
        if real.is_empty() {
            eprintln!("[HC4J] collective: no hardware GPU found; falling back to a software adapter");
        } else {
            physical = real;
        }

        physical.sort_by_key(|&i| {
            let info = all[i].get_info();
            std::cmp::Reverse((crate::device_rank(info.device_type), crate::backend_rank(info.backend)))
        });

        let mut nodes = Vec::with_capacity(ranks);
        for rank in 0..ranks {
            let slot = rank % physical.len();
            nodes.push(Node::open(&all[physical[slot]], slot, rank).await?);
        }

        if physical.len() < ranks {
            eprintln!(
                "[HC4J] collective: {} physical GPU(s) for {ranks} ranks -> loopback mode",
                physical.len()
            );
        }
        let mut profiles = Vec::with_capacity(ranks);
        for node in &nodes {
            eprintln!(
                "[HC4J] collective rank {}: {} | {:?} | {:?}",
                node.rank, node.info.name, node.info.device_type, node.info.backend
            );
            let dev_type = match node.info.device_type {
                wgpu::DeviceType::IntegratedGpu => 1,
                wgpu::DeviceType::DiscreteGpu => 2,
                wgpu::DeviceType::VirtualGpu => 3,
                wgpu::DeviceType::Cpu => 4,
                _ => 0,
            };
            let vram_bytes = node.device.limits().max_storage_buffer_binding_size;
            let compute_gflops = match dev_type {
                2 => 1000.0,
                1 => 100.0,
                4 => 20.0,
                _ => 50.0,
            };
            profiles.push(DeviceProfileC {
                rank: node.rank as u32,
                vendor_id: node.info.vendor,
                device_id: node.info.device,
                device_type: dev_type,
                vram_bytes,
                compute_gflops,
            });
        }

        Ok(Self {
            nodes,
            links: OnceLock::new(),
            profiles,
            ring_order: OnceLock::new(),
            islands: OnceLock::new(),
            buffers: Mutex::new(HashMap::new()),
            next_buffer: Mutex::new(1),
        })
    }

    pub fn ranks(&self) -> usize {
        self.nodes.len()
    }

    pub fn links(&self) -> &[LinkInfo] {
        self.links.get_or_init(|| {
            probe_topology(&self.nodes).unwrap_or_else(|e| {
                eprintln!("[HC4J] collective topology probe failed: {e}");
                Vec::new()
            })
        })
    }

    pub fn device_name(&self, rank: usize) -> Hc4jResult<&str> {
        self.nodes
            .get(rank)
            .map(|n| n.info.name.as_str())
            .ok_or(Hc4jError::InvalidParam("rank out of range"))
    }

    pub fn alloc(&self, rank: usize, elements: u64) -> Hc4jResult<u64> {
        let node = self.nodes.get(rank).ok_or(Hc4jError::InvalidParam("rank out of range"))?;
        if elements == 0 {
            return Err(Hc4jError::InvalidParam("collective buffer must be non-empty"));
        }
        let buffer = node.storage(elements, "HC4J Collective Buffer");
        let mut ids = self.next_buffer.lock().unwrap_or_else(|p| p.into_inner());
        let id = *ids;
        *ids += 1;
        self.buffers
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id, Buf { rank, buffer, elements });
        Ok(id)
    }

    pub fn free(&self, id: u64) -> Hc4jResult<()> {
        self.buffers
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&id)
            .map(|_| ())
            .ok_or(Hc4jError::NotFound)
    }

    fn with_buf<R>(&self, id: u64, f: impl FnOnce(&Buf, &Node) -> Hc4jResult<R>) -> Hc4jResult<R> {
        let guard = self.buffers.lock().unwrap_or_else(|p| p.into_inner());
        let buf = guard.get(&id).ok_or(Hc4jError::NotFound)?;
        let node = &self.nodes[buf.rank];
        f(buf, node)
    }

    pub fn write(&self, id: u64, data: &[f32]) -> Hc4jResult<()> {
        self.with_buf(id, |buf, node| {
            if data.len() as u64 != buf.elements {
                return Err(Hc4jError::InvalidParam("write length does not match the buffer"));
            }
            node.upload(&buf.buffer, 0, bytemuck::cast_slice(data))
        })
    }

    pub fn read(&self, id: u64, out: &mut [f32]) -> Hc4jResult<()> {
        self.with_buf(id, |buf, node| {
            if out.len() as u64 != buf.elements {
                return Err(Hc4jError::InvalidParam("read length does not match the buffer"));
            }
            let bytes = buf.elements * 4;
            let staging = node.readback(bytes);
            node.download_into(&buf.buffer, 0, &staging, bytes, bytemuck::cast_slice_mut(out))
        })
    }

    pub fn all_reduce_sum(&self, ids: &[u64]) -> Hc4jResult<()> {
        let n = self.nodes.len();
        if ids.len() != n {
            return Err(Hc4jError::InvalidParam("all_reduce needs one buffer per rank"));
        }
        let guard = self.buffers.lock().unwrap_or_else(|p| p.into_inner());
        let bufs: Vec<&Buf> = ids
            .iter()
            .map(|id| guard.get(id).ok_or(Hc4jError::NotFound))
            .collect::<Hc4jResult<_>>()?;
        for (rank, buf) in bufs.iter().enumerate() {
            if buf.rank != rank {
                return Err(Hc4jError::InvalidParam("buffer rank does not match its position"));
            }
            if buf.elements != bufs[0].elements {
                return Err(Hc4jError::InvalidParam("every rank must contribute the same length"));
            }
        }
        let total = bufs[0].elements;
        if !total.is_multiple_of(n as u64) {
            return Err(Hc4jError::InvalidParam("element count must divide by the rank count"));
        }
        let chunk = total / n as u64;
        let chunk_bytes = chunk * 4;

        let recv: Vec<wgpu::Buffer> = self
            .nodes
            .iter()
            .map(|node| node.storage(chunk, "HC4J Collective Recv"))
            .collect();
        let staging: Vec<wgpu::Buffer> = self.nodes.iter().map(|node| node.readback(chunk_bytes)).collect();
        let mut hop = vec![0u8; chunk_bytes as usize];

        // Scatter-reduce
        for round in 0..n - 1 {
            for i in 0..n {
                let next = (i + 1) % n;
                let send = (i + n - round) % n;
                self.hop(&bufs, &staging, &recv, &mut hop, i, next, send as u64 * chunk, chunk_bytes)?;
                self.nodes[next].add_into(&bufs[next].buffer, send as u64 * chunk, &recv[next], chunk)?;
            }
        }

        // Allgather
        for round in 0..n - 1 {
            for i in 0..n {
                let next = (i + 1) % n;
                let send = (i + 1 + n - round) % n;
                let offset = send as u64 * chunk;
                self.hop(&bufs, &staging, &recv, &mut hop, i, next, offset, chunk_bytes)?;
                let node = &self.nodes[next];
                let mut encoder = node
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("HC4J Collective Copy") });
                encoder.copy_buffer_to_buffer(&recv[next], 0, &bufs[next].buffer, offset * 4, chunk_bytes);
                node.queue.submit(Some(encoder.finish()));
                node.wait()?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn hop(
        &self,
        bufs: &[&Buf],
        staging: &[wgpu::Buffer],
        recv: &[wgpu::Buffer],
        hop: &mut [u8],
        src: usize,
        dst: usize,
        offset_elems: u64,
        chunk_bytes: u64,
    ) -> Hc4jResult<()> {
        self.nodes[src].download_into(
            &bufs[src].buffer,
            offset_elems * 4,
            &staging[src],
            chunk_bytes,
            hop,
        )?;
        self.nodes[dst].upload(&recv[dst], 0, hop)
    }
}

impl CollectiveBackend for WgpuStagedBackend {
    fn backend_name(&self) -> &'static str {
        "wgpu-staged"
    }

    fn ranks(&self) -> usize {
        self.ranks()
    }

    fn device_name(&self, rank: usize) -> Hc4jResult<String> {
        self.device_name(rank).map(|s| s.to_string())
    }

    fn links(&self) -> &[LinkInfo] {
        self.links()
    }

    fn alloc(&self, rank: usize, elements: u64) -> Hc4jResult<u64> {
        self.alloc(rank, elements)
    }

    fn free(&self, buffer_id: u64) -> Hc4jResult<()> {
        self.free(buffer_id)
    }

    fn write(&self, buffer_id: u64, data: &[f32]) -> Hc4jResult<()> {
        self.write(buffer_id, data)
    }

    fn read(&self, buffer_id: u64, out: &mut [f32]) -> Hc4jResult<()> {
        self.read(buffer_id, out)
    }

    fn allreduce_sum(&self, buffers: &[u64]) -> Hc4jResult<()> {
        self.all_reduce_sum(buffers)
    }

    fn device_profiles(&self) -> &[DeviceProfileC] {
        &self.profiles
    }

    fn ring_order(&self) -> &[u32] {
        self.ring_order
            .get_or_init(|| solve_optimal_ring(self.ranks(), self.links()))
    }

    fn islands(&self) -> &[IslandInfo] {
        self.islands
            .get_or_init(|| detect_islands(self.ranks(), &self.profiles, self.links()))
    }
}

// ============================================================================
// Topology probe
// ============================================================================

fn probe_topology(nodes: &[Node]) -> Result<Vec<LinkInfo>, String> {
    let elements = PROBE_BYTES / 4;
    let mut links = Vec::new();

    for src in 0..nodes.len() {
        for dst in 0..nodes.len() {
            if src == dst {
                continue;
            }
            let send = nodes[src].storage(elements, "HC4J Probe Src");
            let recv = nodes[dst].storage(elements, "HC4J Probe Dst");
            let staging = nodes[src].readback(PROBE_BYTES);
            let mut hop = vec![0u8; PROBE_BYTES as usize];

            let run = |hop: &mut Vec<u8>| -> Hc4jResult<()> {
                nodes[src].download_into(&send, 0, &staging, PROBE_BYTES, hop)?;
                nodes[dst].upload(&recv, 0, hop)
            };

            run(&mut hop).map_err(|e| format!("topology probe {src}->{dst} failed: {e}"))?;
            let t0 = Instant::now();
            for _ in 0..PROBE_ITERATIONS {
                run(&mut hop).map_err(|e| format!("topology probe {src}->{dst} failed: {e}"))?;
            }
            let secs = t0.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
            let gbps = (PROBE_BYTES as f64 * f64::from(PROBE_ITERATIONS)) / secs / 1e9;
            let loopback = nodes[src].adapter_index == nodes[dst].adapter_index;

            eprintln!(
                "[HC4J] collective rank {src}->{dst}: {gbps:.1} GB/s host-staged, {} peer-class{}",
                if is_peer_class(gbps) { "at" } else { "below" },
                if loopback { " (loopback, same adapter)" } else { "" }
            );
            links.push(LinkInfo {
                src_rank: src as u32,
                dst_rank: dst as u32,
                bandwidth_gbps: gbps,
                transport: Transport::HostStaged.code(),
                loopback: u32::from(loopback),
            });
        }
    }
    Ok(links)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_available() -> bool {
        crate::get_engine().is_ok()
    }

    fn ramp(rank: usize, elements: usize) -> Vec<f32> {
        (0..elements).map(|i| (rank * elements + i + 1) as f32).collect()
    }

    fn expected_sum(ranks: usize, elements: usize) -> Vec<f32> {
        (0..elements)
            .map(|i| (0..ranks).map(|r| (r * elements + i + 1) as f32).sum())
            .collect()
    }

    fn run_allreduce(ranks: usize, elements: usize) {
        let coll = Collective::create(ranks).expect("collective should open");
        assert_eq!(coll.ranks(), ranks);

        let ids: Vec<u64> = (0..ranks)
            .map(|r| {
                let id = coll.alloc(r, elements as u64).expect("alloc");
                coll.write(id, &ramp(r, elements)).expect("write");
                id
            })
            .collect();

        coll.all_reduce_sum(&ids).expect("all_reduce_sum");

        let want = expected_sum(ranks, elements);
        for (r, &id) in ids.iter().enumerate() {
            let mut got = vec![0f32; elements];
            coll.read(id, &mut got).expect("read");
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-6 * w.abs().max(1.0),
                    "ranks={ranks} elements={elements}: rank {r} element {i} = {g}, want {w}"
                );
            }
        }
        for id in ids {
            coll.free(id).expect("free");
        }
    }

    #[test]
    fn ring_allreduce_sums_across_two_ranks() {
        if !gpu_available() {
            return;
        }
        run_allreduce(2, 6);
    }

    #[test]
    fn ring_allreduce_sums_across_four_ranks() {
        if !gpu_available() {
            return;
        }
        run_allreduce(4, 4096);
    }

    #[test]
    fn ring_allreduce_rejects_mismatched_shapes() {
        if !gpu_available() {
            return;
        }
        let coll = Collective::create(2).expect("collective should open");
        let a = coll.alloc(0, 8).expect("alloc");
        let b = coll.alloc(1, 8).expect("alloc");
        let odd = coll.alloc(1, 4).expect("alloc");

        let seven_a = coll.alloc(0, 7).expect("alloc");
        let seven_b = coll.alloc(1, 7).expect("alloc");
        assert!(matches!(
            coll.all_reduce_sum(&[seven_a, seven_b]),
            Err(Hc4jError::InvalidParam(_))
        ));

        assert!(matches!(coll.all_reduce_sum(&[a, odd]), Err(Hc4jError::InvalidParam(_))));
        assert!(matches!(coll.all_reduce_sum(&[a]), Err(Hc4jError::InvalidParam(_))));
        assert!(matches!(coll.all_reduce_sum(&[b, a]), Err(Hc4jError::InvalidParam(_))));
        assert!(matches!(coll.all_reduce_sum(&[a, 99_999]), Err(Hc4jError::NotFound)));
    }

    #[test]
    fn topology_covers_every_directed_pair() {
        if !gpu_available() {
            return;
        }
        let coll = Collective::create(2).expect("collective should open");
        let links = coll.links();
        assert_eq!(links.len(), 2, "two ranks have two directed edges");
        for link in links {
            assert_ne!(link.src_rank, link.dst_rank);
            assert!(link.bandwidth_gbps > 0.0, "a measured hop should report throughput");
            assert_eq!(link.transport, Transport::HostStaged.code());
        }
        let names: Vec<&str> = (0..2).map(|r| coll.device_name(r).expect("name")).collect();
        if names[0] == names[1] {
            assert!(links.iter().all(|l| l.loopback == 1));
        }
    }

    #[test]
    fn a_collective_needs_at_least_two_ranks() {
        assert!(matches!(Collective::create(0), Err(Hc4jError::InvalidParam(_))));
        assert!(matches!(Collective::create(1), Err(Hc4jError::InvalidParam(_))));
    }

    #[test]
    fn transport_codes_match_the_java_enum() {
        assert_eq!(Transport::HostStaged.code(), 0);
        assert_eq!(Transport::PeerToPeer.code(), 1);
    }

    #[test]
    fn peer_class_threshold_rejects_staged_bandwidth() {
        assert!(!is_peer_class(3.9));
        assert!(is_peer_class(25.0));
    }

    #[test]
    fn the_add_kernel_binds_both_operands_read_write() {
        assert_eq!(ADD_WGSL.matches("storage, read_write").count(), 2);
        assert!(!ADD_WGSL.contains("storage, read>"));
    }

    #[test]
    fn ring_schedule_reduces_each_chunk_exactly_once() {
        for n in 2..=8usize {
            let mut adds = vec![vec![0u32; n]; n];
            let mut last = vec![usize::MAX; n];
            for round in 0..n - 1 {
                for i in 0..n {
                    let next = (i + 1) % n;
                    let send = (i + n - round) % n;
                    adds[next][send] += 1;
                    if round == n - 2 {
                        last[next] = send;
                    }
                }
            }
            for (r, row) in adds.iter().enumerate() {
                assert_eq!(row[r], 0, "n={n}: rank {r} must not receive its own chunk");
                for (c, &count) in row.iter().enumerate() {
                    if c != r {
                        assert_eq!(count, 1, "n={n}: rank {r} chunk {c} received {count} times");
                    }
                }
            }
            for (r, &chunk) in last.iter().enumerate() {
                assert_eq!(chunk, (r + 1) % n, "n={n}: rank {r} should finish chunk {}", (r + 1) % n);
            }
        }
    }

    #[test]
    fn ring_allgather_visits_every_rank_and_chunk() {
        for n in 2..=8usize {
            let mut seen = vec![vec![false; n]; n];
            for (i, row) in seen.iter_mut().enumerate() {
                row[(i + 1) % n] = true;
            }
            for round in 0..n - 1 {
                for i in 0..n {
                    let next = (i + 1) % n;
                    let send = (i + 1 + n - round) % n;
                    assert!(seen[i][send], "n={n}: rank {i} sent chunk {send} before holding it");
                    seen[next][send] = true;
                }
            }
            for (r, row) in seen.iter().enumerate() {
                assert!(row.iter().all(|&s| s), "n={n}: rank {r} is missing a chunk after allgather");
            }
        }
    }
}
