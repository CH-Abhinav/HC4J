//! Multi-device collective communication: topology discovery and Ring
//! AllReduce across every GPU on the node.
//!
//! # Shape
//!
//! A [`Collective`] owns one [`Node`] per rank, each with its own `wgpu::Device`
//! and queue. When the machine exposes fewer adapters than ranks, the extra
//! ranks open additional logical devices on an adapter already in use —
//! **loopback mode** — so the ring, the transport and the reduction are
//! exercised end to end on a single-GPU box. Loopback measures intra-GPU
//! bandwidth, not PCIe, and the topology report says so.
//!
//! Ranks are deliberately independent of the process-wide [`crate::GpuEngine`]:
//! a collective owns its devices and its buffers outright. That keeps the
//! single-device engine (and every op built on it) untouched, at the cost of
//! needing [`Collective::write`] / [`Collective::read`] to move data in and
//! out. A zero-copy bridge for rank 0 is a separate piece of work.
//!
//! # Transport
//!
//! Every hop is **host-staged**: the source rank copies its chunk into a
//! persistent `MAP_READ` buffer, the host reads it, and the destination rank
//! uploads it. wgpu exposes no external-memory export/import, so true
//! peer-to-peer (`dma_buf` on Linux, an `OPAQUE_WIN32` handle on Windows)
//! cannot be expressed here — reaching it needs raw Vulkan through
//! `wgpu-hal`, which is why [`Transport`] already carries the distinction the
//! Java side reports. Staging buffers are allocated once per AllReduce, not
//! once per hop.
//!
//! # Algorithm
//!
//! Textbook ring AllReduce over `n` ranks and `n` chunks: `n - 1` rounds of
//! scatter-reduce, in which rank `i` sends chunk `(i - round) mod n` to
//! `i + 1`, which adds it into its own copy; then `n - 1` rounds of allgather
//! circulating the finished chunks. Each rank moves `2(n-1)/n` of the buffer
//! instead of the `n-1` copies a naive all-to-all would.
//!
//! Hops are synchronous: a transfer completes before its reduction is
//! recorded. The ring is therefore correct but not overlapped; pipelining the
//! rounds is the obvious next optimisation.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::error::{Hc4jError, Hc4jResult};

/// Elements each invocation of the reduction kernel handles.
const WORKGROUP_SIZE: u32 = 256;
/// Buffer size for the bandwidth probe. 16 MiB is large enough to measure a
/// link rather than per-submission overhead.
const PROBE_BYTES: u64 = 16 << 20;
/// Averaging iterations in the probe.
const PROBE_ITERATIONS: u32 = 8;
/// Above this, a link is reported as fast enough to treat as peer-class.
/// Host-staged hops on consumer hardware land far below it.
const PEER_CLASS_GBPS: f64 = 8.0;

// ============================================================================
// Transport classification
// ============================================================================

/// How a directed rank-to-rank hop moves its bytes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Transport {
    /// Pinned host staging. What the wgpu backend always uses.
    HostStaged,
    /// A true device-to-device path (exported OS memory handle). Reserved for
    /// a raw-Vulkan transport; the wgpu backend never reports it.
    PeerToPeer,
}

impl Transport {
    /// Code shared with `hc4j.collective.Transport`.
    pub fn code(self) -> u32 {
        match self {
            Transport::HostStaged => 0,
            Transport::PeerToPeer => 1,
        }
    }
}

/// One directed edge of the topology graph.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Link {
    pub src: u32,
    pub dst: u32,
    /// Measured throughput of the hop, in GB/s.
    pub gbps: f64,
    /// [`Transport::code`].
    pub transport: u32,
    /// 1 when both ranks sit on the same physical adapter (loopback), so the
    /// figure is an intra-GPU copy rather than a link between two devices.
    pub loopback: u32,
}

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
// Collective
// ============================================================================

/// A ring of ranks, with its topology measured on demand.
pub struct Collective {
    nodes: Vec<Node>,
    /// Probed on first request, not at creation: the measurement moves
    /// hundreds of MiB per rank pair, and AllReduce does not depend on it.
    links: OnceLock<Vec<Link>>,
    buffers: Mutex<HashMap<u64, Buf>>,
    next_buffer: Mutex<u64>,
}

struct Buf {
    rank: usize,
    buffer: wgpu::Buffer,
    elements: u64,
}

impl Collective {
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

        // One physical GPU is enumerated once per backend, so with DX12 and
        // Vulkan both enabled the same card shows up twice. Spreading a ring
        // across those would not be two devices — it would be one device
        // addressed two ways, with the hop measuring a cross-backend host
        // round trip. Deduplicate by physical identity and keep the
        // best-ranked backend for each.
        let mut physical: Vec<usize> = Vec::new();
        // (sorted and filtered below)
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

        // Drop software adapters. Windows enumerates "Microsoft Basic Render
        // Driver" (WARP) alongside the real GPU, and spreading a ring onto a
        // CPU rasterizer would measure a hop into software — the probe showed
        // 0.01 GB/s. Keep them only if there is nothing else at all.
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

        // Best device first, so a 2-rank ring on a mixed machine takes the
        // discrete GPU before the integrated one.
        physical.sort_by_key(|&i| {
            let info = all[i].get_info();
            std::cmp::Reverse((crate::device_rank(info.device_type), crate::backend_rank(info.backend)))
        });

        // Prefer distinct physical devices; once they run out, open extra
        // logical devices on the ones already in use (loopback).
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
        for node in &nodes {
            eprintln!(
                "[HC4J] collective rank {}: {} | {:?} | {:?}",
                node.rank, node.info.name, node.info.device_type, node.info.backend
            );
        }

        Ok(Self {
            nodes,
            links: OnceLock::new(),
            buffers: Mutex::new(HashMap::new()),
            next_buffer: Mutex::new(1),
        })
    }

    pub fn ranks(&self) -> usize {
        self.nodes.len()
    }

    /// The topology graph, probing every directed rank pair on first call.
    /// A failed probe yields an empty graph rather than failing the
    /// collective: the bandwidth figures are diagnostic.
    pub fn links(&self) -> &[Link] {
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

    /// Allocates `elements` f32 on `rank` and returns its handle.
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

    /// Uploads `data` into the whole of buffer `id`.
    pub fn write(&self, id: u64, data: &[f32]) -> Hc4jResult<()> {
        self.with_buf(id, |buf, node| {
            if data.len() as u64 != buf.elements {
                return Err(Hc4jError::InvalidParam("write length does not match the buffer"));
            }
            node.upload(&buf.buffer, 0, bytemuck::cast_slice(data))
        })
    }

    /// Reads the whole of buffer `id` back to the host.
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

    /// Ring AllReduce (sum) over one buffer per rank, in place.
    ///
    /// `ids[r]` must live on rank `r`, all buffers must hold the same number of
    /// elements, and that count must be divisible by the rank count.
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

        // Per-rank scratch, allocated once for the whole operation: a receive
        // buffer, a mappable staging buffer, and a host hop buffer.
        let recv: Vec<wgpu::Buffer> = self
            .nodes
            .iter()
            .map(|node| node.storage(chunk, "HC4J Collective Recv"))
            .collect();
        let staging: Vec<wgpu::Buffer> = self.nodes.iter().map(|node| node.readback(chunk_bytes)).collect();
        let mut hop = vec![0u8; chunk_bytes as usize];

        // Scatter-reduce: after n-1 rounds, rank r holds the finished sum of
        // chunk (r + 1) mod n.
        for round in 0..n - 1 {
            for i in 0..n {
                let next = (i + 1) % n;
                let send = (i + n - round) % n;
                self.hop(&bufs, &staging, &recv, &mut hop, i, next, send as u64 * chunk, chunk_bytes)?;
                self.nodes[next].add_into(&bufs[next].buffer, send as u64 * chunk, &recv[next], chunk)?;
            }
        }

        // Allgather: circulate the finished chunks without reducing.
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

    /// One host-staged hop: rank `src`'s chunk at `offset` into rank `dst`'s
    /// receive buffer.
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

// ============================================================================
// Topology probe
// ============================================================================

/// Times a host-staged hop between every ordered pair of ranks.
fn probe_topology(nodes: &[Node]) -> Result<Vec<Link>, String> {
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

            // Warm-up, then measure.
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
            links.push(Link {
                src: src as u32,
                dst: dst as u32,
                gbps,
                // Honest: this backend has no device-to-device path. The
                // threshold is reported so a caller can see how far a staged
                // hop is from peer-class bandwidth.
                transport: Transport::HostStaged.code(),
                loopback: u32::from(loopback),
            });
        }
    }
    Ok(links)
}

/// Whether a measured hop reaches peer-class bandwidth.
pub fn is_peer_class(gbps: f64) -> bool {
    gbps >= PEER_CLASS_GBPS
}

// ============================================================================
// Registry + C ABI
// ============================================================================

static CONTEXTS: OnceLock<Mutex<HashMap<u64, Collective>>> = OnceLock::new();
static NEXT_CONTEXT: Mutex<u64> = Mutex::new(1);

fn contexts() -> &'static Mutex<HashMap<u64, Collective>> {
    CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn with_context<R>(handle: u64, f: impl FnOnce(&Collective) -> Hc4jResult<R>) -> Hc4jResult<R> {
    let guard = contexts().lock().unwrap_or_else(|p| p.into_inner());
    let coll = guard.get(&handle).ok_or(Hc4jError::NotFound)?;
    f(coll)
}

/// Opens a collective over `ranks` devices and writes its handle to `out`.
///
/// # Safety
/// `out` must point to one writable `u64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_create(ranks: u32, out: *mut u64) -> i32 {
    crate::error::ffi_guard("hc4j_collective_create", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        let coll = Collective::create(ranks as usize)?;
        let mut ids = NEXT_CONTEXT.lock().unwrap_or_else(|p| p.into_inner());
        let handle = *ids;
        *ids += 1;
        contexts().lock().unwrap_or_else(|p| p.into_inner()).insert(handle, coll);
        unsafe { out.write(handle) };
        Ok(())
    })
}

/// Releases a collective and every buffer it owns.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_destroy(handle: u64) -> i32 {
    crate::error::ffi_guard("hc4j_collective_destroy", || {
        contexts()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&handle)
            .map(|_| ())
            .ok_or(Hc4jError::NotFound)
    })
}

/// Number of ranks, or a negative status code.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_ranks(handle: u64) -> i32 {
    match with_context(handle, |c| Ok(c.ranks() as i32)) {
        Ok(n) => n,
        Err(e) => e.code(),
    }
}

/// Number of directed links in the topology, or a negative status code.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_link_count(handle: u64) -> i32 {
    match with_context(handle, |c| Ok(c.links().len() as i32)) {
        Ok(n) => n,
        Err(e) => e.code(),
    }
}

/// Copies link `index` into `out`.
///
/// # Safety
/// `out` must point to writable memory for one [`Link`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_link(handle: u64, index: u32, out: *mut Link) -> i32 {
    crate::error::ffi_guard("hc4j_collective_link", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        with_context(handle, |c| {
            let link = *c
                .links()
                .get(index as usize)
                .ok_or(Hc4jError::InvalidParam("link index out of range"))?;
            unsafe { out.write(link) };
            Ok(())
        })
    })
}

/// Writes rank `rank`'s adapter name into `buf` as UTF-8, truncating to `cap`.
/// Returns the number of bytes written, or a negative status code.
///
/// # Safety
/// `buf` must point to `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_device_name(handle: u64, rank: u32, buf: *mut u8, cap: u32) -> i32 {
    let result = with_context(handle, |c| {
        let name = c.device_name(rank as usize)?;
        if buf.is_null() || cap == 0 {
            return Err(Hc4jError::InvalidParam("null or empty name buffer"));
        }
        let bytes = name.as_bytes();
        let len = bytes.len().min(cap as usize);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, len) };
        Ok(len as i32)
    });
    match result {
        Ok(len) => len,
        Err(e) => e.code(),
    }
}

/// Allocates `elements` f32 on `rank`, writing the buffer handle to `out`.
///
/// # Safety
/// `out` must point to one writable `u64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_alloc(handle: u64, rank: u32, elements: u64, out: *mut u64) -> i32 {
    crate::error::ffi_guard("hc4j_collective_alloc", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        with_context(handle, |c| {
            let id = c.alloc(rank as usize, elements)?;
            unsafe { out.write(id) };
            Ok(())
        })
    })
}

/// Frees a collective buffer.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_collective_free(handle: u64, buffer: u64) -> i32 {
    crate::error::ffi_guard("hc4j_collective_free", || with_context(handle, |c| c.free(buffer)))
}

/// Uploads `len` f32 from `data` into the whole buffer.
///
/// # Safety
/// `data` must point to `len` readable f32 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_write(handle: u64, buffer: u64, data: *const f32, len: u64) -> i32 {
    crate::error::ffi_guard("hc4j_collective_write", || {
        if data.is_null() {
            return Err(Hc4jError::InvalidParam("null data pointer"));
        }
        let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
        with_context(handle, |c| c.write(buffer, slice))
    })
}

/// Reads the whole buffer into `out`.
///
/// # Safety
/// `out` must point to `len` writable f32 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_read(handle: u64, buffer: u64, out: *mut f32, len: u64) -> i32 {
    crate::error::ffi_guard("hc4j_collective_read", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null out pointer"));
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(out, len as usize) };
        with_context(handle, |c| c.read(buffer, slice))
    })
}

/// Ring AllReduce (sum) in place over one buffer per rank.
///
/// # Safety
/// `buffers` must point to `n` readable `u64` handles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_collective_allreduce_sum(handle: u64, buffers: *const u64, n: u32) -> i32 {
    crate::error::ffi_guard("hc4j_collective_allreduce_sum", || {
        if buffers.is_null() {
            return Err(Hc4jError::InvalidParam("null buffer list"));
        }
        let ids = unsafe { std::slice::from_raw_parts(buffers, n as usize) };
        with_context(handle, |c| c.all_reduce_sum(ids))
    })
}

#[cfg(test)]
mod tests {
    use super::*;


    /// Only skip when the machine has no GPU at all; a real failure must fail.
    fn gpu_available() -> bool {
        crate::get_engine().is_ok()
    }

    /// Rank `r` contributes `r * n + i + 1`, so the expected sum per element
    /// is known in closed form — the same shape as the reference PoC.
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
        // Six elements over two ranks: one scatter-reduce round and one
        // allgather round, three elements per chunk.
        run_allreduce(2, 6);
    }

    #[test]
    fn ring_allreduce_sums_across_four_ranks() {
        if !gpu_available() {
            return;
        }
        // Four ranks means three rounds each way, and 4096 elements per rank
        // puts several workgroups behind every reduction.
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

        // Element count must divide by the rank count.
        let seven_a = coll.alloc(0, 7).expect("alloc");
        let seven_b = coll.alloc(1, 7).expect("alloc");
        assert!(matches!(
            coll.all_reduce_sum(&[seven_a, seven_b]),
            Err(Hc4jError::InvalidParam(_))
        ));

        // Every rank must contribute the same length.
        assert!(matches!(coll.all_reduce_sum(&[a, odd]), Err(Hc4jError::InvalidParam(_))));
        // One buffer per rank, in rank order.
        assert!(matches!(coll.all_reduce_sum(&[a]), Err(Hc4jError::InvalidParam(_))));
        assert!(matches!(coll.all_reduce_sum(&[b, a]), Err(Hc4jError::InvalidParam(_))));
        // Unknown handles are reported, not ignored.
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
            assert_ne!(link.src, link.dst);
            assert!(link.gbps > 0.0, "a measured hop should report throughput");
            assert_eq!(link.transport, Transport::HostStaged.code());
        }
        // On a single-adapter machine both ranks share it, which the report
        // has to admit so the number is not read as a PCIe measurement.
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
        // A host-staged hop on consumer hardware does not reach PCIe P2P
        // speeds; the threshold exists so the report can say by how much.
        assert!(!is_peer_class(3.9));
        assert!(is_peer_class(25.0));
    }

    #[test]
    fn the_add_kernel_binds_both_operands_read_write() {
        // Slab-style sharing and in-place accumulation both need it, and wgpu
        // rejects a buffer bound read-only and read-write in one dispatch.
        assert_eq!(ADD_WGSL.matches("storage, read_write").count(), 2);
        assert!(!ADD_WGSL.contains("storage, read>"));
    }

    /// The ring index arithmetic, checked without a GPU.
    ///
    /// Scatter-reduce gives each rank exactly one incoming add per chunk, for
    /// every chunk but its own — a chunk accumulates across ranks as it
    /// travels, rather than piling up on one of them. The chunk a rank
    /// receives in the *last* round is the one that has now seen every rank's
    /// contribution, and that is the chunk the allgather starts from.
    #[test]
    fn ring_schedule_reduces_each_chunk_exactly_once() {
        for n in 2..=8usize {
            // adds[r][c]: how many times rank r added into chunk c.
            let mut adds = vec![vec![0u32; n]; n];
            // last[r]: the chunk rank r received in the final round.
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
            // The fully reduced chunk on rank r is r + 1, which is where the
            // allgather model below begins.
            for (r, &chunk) in last.iter().enumerate() {
                assert_eq!(chunk, (r + 1) % n, "n={n}: rank {r} should finish chunk {}", (r + 1) % n);
            }
        }
    }

    #[test]
    fn ring_allgather_visits_every_rank_and_chunk() {
        for n in 2..=8usize {
            let mut seen = vec![vec![false; n]; n];
            // Each rank starts holding its own reduced chunk.
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
