use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::error::{Hc4jError, Hc4jResult};

use super::backend::{is_peer_class, CollectiveBackend, DeviceProfileC, LinkInfo, Transport};
use super::planner::{chunk_range, detect_islands, solve_optimal_ring, IslandInfo};

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
    pub(crate) gemm_bias_act: wgpu::ComputePipeline,
    pub(crate) split_kv_attn: wgpu::ComputePipeline,
    pub(crate) dummy_bias: wgpu::Buffer,
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

        let gemm_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HC4J Collective Gemm"),
            source: wgpu::ShaderSource::Wgsl(include_str!("vulkan/gemm_bias_act.wgsl").into()),
        });
        let gemm_bias_act = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HC4J Collective Gemm"),
            layout: None,
            module: &gemm_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });

        let split_kv_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HC4J Collective Split-KV"),
            source: wgpu::ShaderSource::Wgsl(include_str!("vulkan/split_kv_attn.wgsl").into()),
        });
        let split_kv_attn = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HC4J Collective Split-KV"),
            layout: None,
            module: &split_kv_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });

        let dummy_bias = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HC4J Dummy Bias"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            rank,
            device,
            queue,
            info: adapter.get_info(),
            adapter_index,
            add,
            gemm_bias_act,
            split_kv_attn,
            dummy_bias,
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
        drop(guard);
        self.allreduce_sum_subset(ids)
    }

    pub fn allreduce_sum_subset(&self, ids: &[u64]) -> Hc4jResult<()> {
        let k = ids.len();
        if k == 0 {
            return Err(Hc4jError::InvalidParam("allreduce_sum_subset requires at least 1 buffer"));
        }
        if k == 1 {
            let guard = self.buffers.lock().unwrap_or_else(|p| p.into_inner());
            if !guard.contains_key(&ids[0]) {
                return Err(Hc4jError::NotFound);
            }
            return Ok(());
        }

        let guard = self.buffers.lock().unwrap_or_else(|p| p.into_inner());
        let bufs: Vec<&Buf> = ids
            .iter()
            .map(|id| guard.get(id).ok_or(Hc4jError::NotFound))
            .collect::<Hc4jResult<_>>()?;

        let mut seen_ranks = std::collections::HashSet::new();
        for buf in &bufs {
            if !seen_ranks.insert(buf.rank) {
                return Err(Hc4jError::InvalidParam("duplicate rank in subset"));
            }
            if buf.elements != bufs[0].elements {
                return Err(Hc4jError::InvalidParam("every rank must contribute the same length"));
            }
        }
        let total = bufs[0].elements;
        let sub_ranks: Vec<usize> = bufs.iter().map(|b| b.rank).collect();

        let max_chunk_len = (total as usize / k) + usize::from(!total.is_multiple_of(k as u64));
        let max_chunk_bytes = (max_chunk_len * 4) as u64;

        let recv: HashMap<usize, wgpu::Buffer> = sub_ranks
            .iter()
            .map(|&r| (r, self.nodes[r].storage(max_chunk_len as u64, "HC4J Collective Recv")))
            .collect();
        let staging: HashMap<usize, wgpu::Buffer> = sub_ranks
            .iter()
            .map(|&r| (r, self.nodes[r].readback(max_chunk_bytes)))
            .collect();
        let mut hop = vec![0u8; max_chunk_bytes as usize];

        // Scatter-reduce (k - 1 rounds)
        for round in 0..k - 1 {
            for idx in 0..k {
                let i = sub_ranks[idx];
                let next = sub_ranks[(idx + 1) % k];
                let send_chunk_idx = (idx + k - round) % k;
                let (chunk_start, chunk_len) = chunk_range(total as usize, k, send_chunk_idx);
                if chunk_len == 0 {
                    continue;
                }
                let chunk_bytes = (chunk_len * 4) as u64;

                self.nodes[i].download_into(
                    &bufs[idx].buffer,
                    (chunk_start * 4) as u64,
                    &staging[&i],
                    chunk_bytes,
                    &mut hop[..chunk_bytes as usize],
                )?;
                self.nodes[next].upload(&recv[&next], 0, &hop[..chunk_bytes as usize])?;
                self.nodes[next].add_into(
                    &bufs[(idx + 1) % k].buffer,
                    chunk_start as u64,
                    &recv[&next],
                    chunk_len as u64,
                )?;
            }
        }

        // Allgather (k - 1 rounds)
        for round in 0..k - 1 {
            for idx in 0..k {
                let i = sub_ranks[idx];
                let next = sub_ranks[(idx + 1) % k];
                let send_chunk_idx = (idx + 1 + k - round) % k;
                let (chunk_start, chunk_len) = chunk_range(total as usize, k, send_chunk_idx);
                if chunk_len == 0 {
                    continue;
                }
                let chunk_bytes = (chunk_len * 4) as u64;

                self.nodes[i].download_into(
                    &bufs[idx].buffer,
                    (chunk_start * 4) as u64,
                    &staging[&i],
                    chunk_bytes,
                    &mut hop[..chunk_bytes as usize],
                )?;
                self.nodes[next].upload(&recv[&next], 0, &hop[..chunk_bytes as usize])?;

                let node = &self.nodes[next];
                let mut encoder = node
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("HC4J Collective Copy") });
                encoder.copy_buffer_to_buffer(
                    &recv[&next],
                    0,
                    &bufs[(idx + 1) % k].buffer,
                    (chunk_start * 4) as u64,
                    chunk_bytes,
                );
                node.queue.submit(Some(encoder.finish()));
                node.wait()?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bias_act(
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
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        let node = &self.nodes[rank];
        let guard = self.buffers.lock().unwrap_or_else(|p| p.into_inner());
        let b_a = guard.get(&buf_a).ok_or(Hc4jError::InvalidParam("invalid buf_a"))?;
        let b_b = guard.get(&buf_b).ok_or(Hc4jError::InvalidParam("invalid buf_b"))?;
        let b_c = guard.get(&buf_c).ok_or(Hc4jError::InvalidParam("invalid buf_c"))?;

        if b_a.rank != rank || b_b.rank != rank || b_c.rank != rank {
            return Err(Hc4jError::InvalidParam("buffers must belong to specified rank"));
        }
        if b_a.elements < (m * k) as u64 || b_b.elements < (k * n) as u64 || b_c.elements < (m * n) as u64 {
            return Err(Hc4jError::InvalidParam("buffer size too small for gemm shape"));
        }

        let bias_buffer = if buf_bias != 0 {
            let b_bias = guard.get(&buf_bias).ok_or(Hc4jError::InvalidParam("invalid buf_bias"))?;
            if b_bias.rank != rank {
                return Err(Hc4jError::InvalidParam("bias buffer must belong to specified rank"));
            }
            if b_bias.elements < n as u64 {
                return Err(Hc4jError::InvalidParam("bias buffer size too small"));
            }
            &b_bias.buffer
        } else {
            &node.dummy_bias
        };

        let params: [u32; 4] = [m, n, k, flags];
        let uniform = node.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HC4J Gemm Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        node.queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));

        let bind_group = node.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HC4J Gemm BindGroup"),
            layout: &node.gemm_bias_act.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_b.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bias_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_c.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: uniform.as_entire_binding() },
            ],
        });

        let gx = n.div_ceil(16);
        let gy = m.div_ceil(16);

        let mut encoder = node
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("HC4J Gemm Encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HC4J Gemm Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&node.gemm_bias_act);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(gx, gy, 1);
        }
        node.queue.submit(Some(encoder.finish()));
        node.wait()?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn split_kv_attn(
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
        if num_heads == 0 || head_dim == 0 {
            return Ok(());
        }

        let node = &self.nodes[rank];
        let guard = self.buffers.lock().unwrap_or_else(|p| p.into_inner());
        let b_q = guard.get(&buf_q).ok_or(Hc4jError::InvalidParam("invalid buf_q"))?;
        let b_k = guard.get(&buf_k).ok_or(Hc4jError::InvalidParam("invalid buf_k"))?;
        let b_v = guard.get(&buf_v).ok_or(Hc4jError::InvalidParam("invalid buf_v"))?;
        let b_o = guard.get(&buf_out_o).ok_or(Hc4jError::InvalidParam("invalid buf_out_o"))?;
        let b_s = guard.get(&buf_out_stats).ok_or(Hc4jError::InvalidParam("invalid buf_out_stats"))?;

        if b_q.rank != rank || b_k.rank != rank || b_v.rank != rank || b_o.rank != rank || b_s.rank != rank {
            return Err(Hc4jError::InvalidParam("buffers must belong to specified rank"));
        }
        if b_q.elements < (num_heads * head_dim) as u64
            || b_k.elements < (num_kv_heads * local_tokens * head_dim) as u64
            || b_v.elements < (num_kv_heads * local_tokens * head_dim) as u64
            || b_o.elements < (num_heads * head_dim) as u64
            || b_s.elements < (num_heads * 2) as u64
        {
            return Err(Hc4jError::InvalidParam("buffer size too small for split_kv"));
        }

        let params: [u32; 8] = [
            num_heads,
            num_kv_heads,
            head_dim,
            local_tokens,
            scale.to_bits(),
            0,
            0,
            0,
        ];
        let uniform = node.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HC4J Split-KV Params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        node.queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&params));

        let bind_group = node.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HC4J Split-KV BindGroup"),
            layout: &node.split_kv_attn.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_q.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_k.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_v.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_o.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_s.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: uniform.as_entire_binding() },
            ],
        });

        let mut encoder = node
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("HC4J Split-KV Encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HC4J Split-KV Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&node.split_kv_attn);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(num_heads, 1, 1);
        }
        node.queue.submit(Some(encoder.finish()));
        node.wait()?;

        Ok(())
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

    fn allreduce_sum_subset(&self, buffers: &[u64]) -> Hc4jResult<()> {
        self.allreduce_sum_subset(buffers)
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
        self.gemm_bias_act(rank, buf_a, buf_b, buf_bias, buf_c, m, n, k, flags)
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
        self.split_kv_attn(
            rank,
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

            // Quick 4 KiB latency ping (8 iterations)
            let ping_bytes = 4096u64;
            let ping_staging = nodes[src].readback(ping_bytes);
            let mut ping_hop = vec![0u8; ping_bytes as usize];
            let ping_run = |hop: &mut Vec<u8>| -> Hc4jResult<()> {
                nodes[src].download_into(&send, 0, &ping_staging, ping_bytes, hop)?;
                nodes[dst].upload(&recv, 0, hop)
            };
            let latency_us = (|| -> Option<f64> {
                ping_run(&mut ping_hop).ok()?;
                let t_ping = Instant::now();
                for _ in 0..8 {
                    ping_run(&mut ping_hop).ok()?;
                }
                let elapsed_secs = t_ping.elapsed().as_secs_f64();
                Some((elapsed_secs * 1e6 / 8.0).max(1.0))
            })().unwrap_or(500.0);

            eprintln!(
                "[HC4J] collective rank {src}->{dst}: {gbps:.1} GB/s ({latency_us:.0} us) host-staged, {} peer-class{}",
                if is_peer_class(gbps) { "at" } else { "below" },
                if loopback { " (loopback, same adapter)" } else { "" }
            );
            links.push(LinkInfo {
                src_rank: src as u32,
                dst_rank: dst as u32,
                bandwidth_gbps: gbps,
                transport: Transport::HostStaged.code(),
                loopback: u32::from(loopback),
                latency_us,
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
    fn test_wgpu_staged_allreduce_sum_subset() {
        let backend = match Collective::create(4) {
            Ok(b) => b,
            Err(_) => return,
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

        // 1-rank is valid no-op
        backend.allreduce_sum_subset(&[b0]).unwrap();
        let mut check_b0 = vec![0.0f32; len];
        backend.read(b0, &mut check_b0).unwrap();
        assert_eq!(check_b0, data0);

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
            assert_eq!(res1[i], data1[i]);
            assert_eq!(res3[i], data3[i]);
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
            assert!((res2[i] - (data0[i] + data2[i])).abs() < 1e-4);
        }

        backend.free(b0).unwrap();
        backend.free(b1).unwrap();
        backend.free(b2).unwrap();
        backend.free(b3).unwrap();
    }

    #[test]
    fn test_wgpu_staged_gemm_bias_act_ragged() {
        let backend = match Collective::create(2) {
            Ok(b) => b,
            Err(_) => return,
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
            assert!((c_res[i] - cpu_ref[i]).abs() <= 1e-4);
        }

        // 2. BIAS + GELU (flags = 1 | (1 << 1) = 3)
        backend.gemm_bias_act(0, buf_a, buf_b, buf_bias, buf_c, m as u32, n as u32, k as u32, 3).unwrap();
        backend.read(buf_c, &mut c_res).unwrap();
        let cpu_ref = cpu_gemm_bias_act(&a_data, &b_data, Some(&bias_data), m, n, k, 1);
        for i in 0..m * n {
            assert!((c_res[i] - cpu_ref[i]).abs() <= 1e-4);
        }

        // 3. BIAS + SILU (flags = 1 | (2 << 1) = 5)
        backend.gemm_bias_act(0, buf_a, buf_b, buf_bias, buf_c, m as u32, n as u32, k as u32, 5).unwrap();
        backend.read(buf_c, &mut c_res).unwrap();
        let cpu_ref = cpu_gemm_bias_act(&a_data, &b_data, Some(&bias_data), m, n, k, 2);
        for i in 0..m * n {
            assert!((c_res[i] - cpu_ref[i]).abs() <= 1e-4);
        }

        // 4. BIAS + RELU (flags = 1 | (3 << 1) = 7)
        backend.gemm_bias_act(0, buf_a, buf_b, buf_bias, buf_c, m as u32, n as u32, k as u32, 7).unwrap();
        backend.read(buf_c, &mut c_res).unwrap();
        let cpu_ref = cpu_gemm_bias_act(&a_data, &b_data, Some(&bias_data), m, n, k, 3);
        for i in 0..m * n {
            assert!((c_res[i] - cpu_ref[i]).abs() <= 1e-4);
        }

        backend.free(buf_a).unwrap();
        backend.free(buf_b).unwrap();
        backend.free(buf_bias).unwrap();
        backend.free(buf_c).unwrap();
    }

    #[test]
    fn test_wgpu_staged_split_kv_attn_mha_and_gqa() {
        let backend = match Collective::create(2) {
            Ok(b) => b,
            Err(_) => return,
        };

        // Test MHA
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
                assert!((out_stats[2 * h] - ref_stats[2 * h]).abs() <= 1e-4);
                assert!((out_stats[2 * h + 1] - ref_stats[2 * h + 1]).abs() / ref_stats[2 * h + 1] <= 1e-3);
            }
            for i in 0..num_heads * head_dim {
                assert!((out_o[i] - ref_o[i]).abs() <= 1e-4);
            }

            backend.free(buf_q).unwrap();
            backend.free(buf_k).unwrap();
            backend.free(buf_v).unwrap();
            backend.free(buf_o).unwrap();
            backend.free(buf_stats).unwrap();
        }

        // Test GQA
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
                assert!((out_stats[2 * h] - ref_stats[2 * h]).abs() <= 1e-4);
                assert!((out_stats[2 * h + 1] - ref_stats[2 * h + 1]).abs() / ref_stats[2 * h + 1] <= 1e-3);
            }
            for i in 0..num_heads * head_dim {
                assert!((out_o[i] - ref_o[i]).abs() <= 1e-4);
            }

            backend.free(buf_q).unwrap();
            backend.free(buf_k).unwrap();
            backend.free(buf_v).unwrap();
            backend.free(buf_o).unwrap();
            backend.free(buf_stats).unwrap();
        }
    }
}
