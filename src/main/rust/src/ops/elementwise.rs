//! Shared execution engine for elementwise kernels (unary, binary, fused).
//!
//! **Calling convention.** Every elementwise op is caller-allocated: the
//! output tensor is passed in by handle, so Java can reuse buffers and run
//! ops in place (`a.sin(a)`, `a.add(b, a)`).
//!
//! **Contiguous ops** run through [`run_contiguous`]:
//!
//! 1. *Resident path.* Pin and page in every operand, then dispatch.
//!    Operands larger than the storage-binding limit are processed as
//!    sub-range tiles with no copies.
//! 2. *Streaming path.* If the operands cannot be co-resident, the kernel
//!    runs tile by tile over transient scratch buffers. Inputs are uploaded
//!    from host RAM or disk; output tiles are either written straight into
//!    the output's VRAM or streamed back through the double-buffered
//!    readback ring into fresh storage that replaces the output's.
//!
//! **Strided ops** run through [`run_strided`]: resident only, since strided
//! index math crosses tile boundaries.
//!
//! **Grid.** Dispatches fold into X×Y (see [`grid_2d`]) and kernels rebuild
//! the linear id from `num_workgroups`, so no tensor size hits the per-dimension
//! workgroup limit (65535 on D3D12, 65536 on some Vulkan drivers).
//!
//! **Bindings.** Every storage binding is `read_write`: wgpu rejects a buffer
//! bound both read-only and read-write in one dispatch, and slab regions (and
//! in-place ops) put inputs and outputs in the same buffer.

use std::collections::VecDeque;

use crate::error::{Hc4jError, Hc4jResult, ffi_guard};
use crate::memory::budget::Reservation;
use crate::memory::manager::try_with_capacity;
use crate::memory::spill::{self, SpillFile};
use crate::memory::transfer::TRANSFER_CHUNK;
use crate::memory::{DeviceBlock, DeviceSpan, HostBlock, Residency, Snapshot, TensorId, TieredMemoryManager, manager};
use crate::{ErrorTrap, GpuEngine, get_engine};

pub const WORKGROUP_SIZE: u32 = 256;
/// Contiguous kernels process one vec4 per invocation.
const ELEMS_PER_CONTIGUOUS_THREAD: u64 = 4;
/// Caps a tile at 2^28 elements so `tid << 2` can never overflow u32.
const MAX_TILE_ELEMS: u64 = 1 << 28;
/// Smallest scratch tile tried before the streaming path gives up.
const MIN_STREAM_TILE: u64 = 1 << 20;
const DEVICE_OUTPUT_INFLIGHT: usize = 2;

// ============================================================================
// Grid folding
// ============================================================================

/// Splits a 1D workgroup count across two dimensions so neither exceeds
/// `max_per_dim`. The split is balanced (x = ceil(n / y)) rather than
/// x = max_per_dim: just past the limit, x = 65535 would launch nearly 2x
/// the needed workgroups, all of which then exit in the bounds check.
pub fn grid_2d(workgroups: u32, max_per_dim: u32) -> (u32, u32) {
    let max_per_dim = max_per_dim.max(1);
    if workgroups <= max_per_dim {
        return (workgroups.max(1), 1);
    }
    let y = workgroups.div_ceil(max_per_dim);
    (workgroups.div_ceil(y), y)
}

/// Grid for `threads` invocations of a 256-wide 1D kernel.
pub fn grid_for_threads(engine: &GpuEngine, threads: u64) -> Hc4jResult<(u32, u32, u32)> {
    let workgroups = u32::try_from(threads.div_ceil(WORKGROUP_SIZE as u64))
        .map_err(|_| Hc4jError::Unsupported("dispatch exceeds 2^32 workgroups"))?;
    let (x, y) = grid_2d(workgroups, engine.limits.max_compute_workgroups_per_dimension);
    if y > engine.limits.max_compute_workgroups_per_dimension {
        return Err(Hc4jError::Unsupported("dispatch too large for a 2D grid"));
    }
    Ok((x, y, 1))
}

/// Largest tile that fits one storage binding, aligned so tile offsets
/// satisfy `min_storage_buffer_offset_alignment`.
pub fn max_binding_tile(engine: &GpuEngine) -> u64 {
    let align = (engine.limits.min_storage_buffer_offset_alignment as u64).max(4);
    let cap = engine.limits.max_storage_buffer_binding_size.min(MAX_TILE_ELEMS * 4);
    (cap / align).max(1) * align
}

// ============================================================================
// Kernel metadata
// ============================================================================

/// `params = [length, is_contiguous, 0, 0]`; shape and strides are
/// reverse-pruned (innermost first, size-1 dims removed) and map to
/// `array<vec4<u32>, 2>` in WGSL, sidestepping 16-byte array-stride rules.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UnaryElemDims {
    pub params: [u32; 4],
    pub shape: [u32; 8],
    pub strides_a: [u32; 8],
    pub strides_c: [u32; 8],
}

impl UnaryElemDims {
    pub fn contiguous(length: u32) -> Self {
        Self {
            params: [length, 1, 0, 0],
            shape: [1; 8],
            strides_a: [0; 8],
            strides_c: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BinaryElemDims {
    pub params: [u32; 4],
    pub shape: [u32; 8],
    pub strides_a: [u32; 8],
    pub strides_b: [u32; 8],
    pub strides_c: [u32; 8],
}

impl BinaryElemDims {
    pub fn contiguous(length: u32) -> Self {
        Self {
            params: [length, 1, 0, 0],
            shape: [1; 8],
            strides_a: [0; 8],
            strides_b: [0; 8],
            strides_c: [0; 8],
        }
    }
}

/// Reads a strided layout and reverse-prunes it: size-1 dimensions are
/// dropped and the rest reordered innermost-first, so kernels decompose the
/// linear index without knowing the rank. Returns the shape and one stride
/// array per input pointer.
///
/// # Safety
/// For `rank > 0`, `shape` and every pointer in `strides` must point to
/// `rank` readable `u32` values.
pub unsafe fn pruned_layout(rank: u32, shape: *const u32, strides: &[*const u32]) -> Hc4jResult<([u32; 8], Vec<[u32; 8]>)> {
    let mut out_shape = [1u32; 8];
    let mut out_strides = vec![[0u32; 8]; strides.len()];
    if rank == 0 {
        return Ok((out_shape, out_strides));
    }
    if rank > 8 {
        return Err(Hc4jError::Unsupported("rank above 8"));
    }
    if shape.is_null() || strides.iter().any(|p| p.is_null()) {
        return Err(Hc4jError::InvalidParam("null shape/stride pointer"));
    }
    let r = rank as usize;
    let raw_shape = unsafe { std::slice::from_raw_parts(shape, r) };
    let raw_strides: Vec<&[u32]> = strides.iter().map(|&p| unsafe { std::slice::from_raw_parts(p, r) }).collect();
    let mut effective = 0;
    for i in (0..r).rev() {
        if raw_shape[i] > 1 {
            out_shape[effective] = raw_shape[i];
            for (k, stride) in raw_strides.iter().enumerate() {
                out_strides[k][effective] = stride[i];
            }
            effective += 1;
        }
    }
    Ok((out_shape, out_strides))
}

/// Unary f32 template shared by the trigonometric and exponential suites.
pub const UNARY_F32_SHADER_TEMPLATE: &str = r#"
struct Dims {
    params: vec4<u32>, // `meta` is a reserved word in WGSL
    shape: array<vec4<u32>, 2>,
    strides_a: array<vec4<u32>, 2>,
    strides_c: array<vec4<u32>, 2>,
}

// read_write on both: slab regions and in-place ops share buffers.
@group(0) @binding(0) var<storage, read_write> arrayA: array<f32>;
@group(0) @binding(1) var<storage, read_write> arrayC: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims; // UNIFORM: Hits fast constant cache!

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    // The host folds large grids into X*Y; rebuild the linear thread id.
    let tid = gid.x + gid.y * (nwg.x * 256u);
    let length = dims.params.x;

    if (dims.params.y == 1u) {
        // --- CONTIGUOUS FAST PATH ---
        let base = tid << 2u; // base = tid * 4

        // Instant Warp Exit for out-of-bounds threads
        if (base >= length) { return; }

        if (base + 3u < length) {
            // Highly optimized contiguous execution (4-wide vectorization)
            let val_a = vec4<f32>(arrayA[base], arrayA[base+1u], arrayA[base+2u], arrayA[base+3u]);
            let res = __WGSL_OP__(val_a);
            arrayC[base] = res.x;
            arrayC[base+1u] = res.y;
            arrayC[base+2u] = res.z;
            arrayC[base+3u] = res.w;
        } else {
            // Manually unrolled tail execution (avoids 'for' loop overhead on the GPU)
            arrayC[base] = __WGSL_OP__(arrayA[base]);
            if (base + 1u < length) {
                arrayC[base+1u] = __WGSL_OP__(arrayA[base+1u]);
                if (base + 2u < length) {
                    arrayC[base+2u] = __WGSL_OP__(arrayA[base+2u]);
                }
            }
        }
    } else {
        // --- NON-CONTIGUOUS STRIDED PATH (ZERO-BRANCHING) ---
        let i = tid;

        // Instant Warp Exit
        if (i >= length) { return; }

        var remaining = i;
        var offset_a = 0u;
        var offset_c = 0u;

        // 100% UNROLLED & BRANCHLESS LOOP!
        for (var d = 0u; d < 8u; d = d + 1u) {
            let dim_size = dims.shape[d >> 2u][d & 3u];

            // OPTIMIZATION: Skip expensive modulo/division hardware instructions if dimension is 1
            if (dim_size > 1u) {
                let coord = remaining % dim_size;
                remaining = remaining / dim_size;

                offset_a = offset_a + coord * dims.strides_a[d >> 2u][d & 3u];
                offset_c = offset_c + coord * dims.strides_c[d >> 2u][d & 3u];
            }
        }

        arrayC[offset_c] = __WGSL_OP__(arrayA[offset_a]);
    }
}
"#;

pub fn generate_unary_shader(wgsl_op: &str) -> String {
    UNARY_F32_SHADER_TEMPLATE.replace("__WGSL_OP__", wgsl_op)
}

pub fn unary_pipeline(family: &str, op_name: &str, wgsl_op: &str) -> Hc4jResult<wgpu::ComputePipeline> {
    let shader_name = format!("{family}_{op_name}_f32");
    get_engine()?.get_or_compile(&shader_name, "main", &generate_unary_shader(wgsl_op))
}

/// FFI body shared by every unary f32 op (trigonometric and exponential).
///
/// # Safety
/// When `is_contiguous == 0` and `rank > 0`, the three pointers must each
/// point to `rank` readable `u32` values.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_unary(
    family: &str,
    op_name: &str,
    wgsl_op: &str,
    id_a: u64,
    id_out: u64,
    rank: u32,
    ptr_shape: *const u32,
    ptr_strides_a: *const u32,
    ptr_strides_c: *const u32,
    length: usize,
    is_contiguous: u32,
) -> i32 {
    ffi_guard(op_name, || {
        let pipeline = unary_pipeline(family, op_name, wgsl_op)?;
        if is_contiguous != 0 {
            check_contiguous_length(id_out, length)?;
            return run_contiguous(
                &pipeline,
                &|len| bytemuck::bytes_of(&UnaryElemDims::contiguous(len)).to_vec(),
                &[id_a],
                id_out,
            );
        }
        let length = u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("strided length exceeds u32"))?;
        let (shape, strides) = unsafe { pruned_layout(rank, ptr_shape, &[ptr_strides_a, ptr_strides_c])? };
        let dims = UnaryElemDims {
            params: [length, 0, 0, 0],
            shape,
            strides_a: strides[0],
            strides_c: strides[1],
        };
        run_strided(&pipeline, &[id_a, id_out], bytemuck::bytes_of(&dims), length as u64)
    })
}

/// A contiguous call must cover the whole output tensor.
pub fn check_contiguous_length(id_out: TensorId, length: usize) -> Hc4jResult<()> {
    let size = manager()?.size_of(id_out)?;
    if (length as u64).checked_mul(4) != Some(size) {
        return Err(Hc4jError::InvalidParam("length does not match the output tensor"));
    }
    Ok(())
}

// ============================================================================
// Contiguous executor
// ============================================================================

/// Builds the uniform block for a contiguous tile of `len` elements.
pub type UniformFn<'a> = dyn Fn(u32) -> Vec<u8> + 'a;

/// Runs a contiguous elementwise `pipeline` over `inputs`, writing `output`.
/// Bindings: inputs at 0..n, output at n, uniform at n + 1. All operands
/// must have the output's byte size.
pub fn run_contiguous(
    pipeline: &wgpu::ComputePipeline,
    uniform: &UniformFn<'_>,
    inputs: &[TensorId],
    output: TensorId,
) -> Hc4jResult<()> {
    let mgr = manager()?;
    let size = mgr.size_of(output)?;
    if size == 0 || !size.is_multiple_of(4) {
        return Err(Hc4jError::InvalidParam("tensor is not a whole number of 4-byte elements"));
    }
    for &input in inputs {
        if mgr.size_of(input)? != size {
            return Err(Hc4jError::InvalidParam("operand sizes differ"));
        }
    }

    let mut operands = inputs.to_vec();
    operands.push(output);
    match mgr.acquire_resident(&operands) {
        Ok(resident) => return dispatch_resident(mgr.engine(), pipeline, uniform, &resident.spans, size),
        Err(Hc4jError::OutOfMemory) => {}
        Err(err) => return Err(err),
    }
    crate::hc4j_trace!("elementwise op over {size} B exceeds VRAM; streaming");
    stream_contiguous(mgr, pipeline, uniform, inputs, output, size)
}

/// All operands resident: tile only as far as binding limits require. The
/// whole op is one recorded unit, so inside a batch it adds no submission.
fn dispatch_resident(
    engine: &GpuEngine,
    pipeline: &wgpu::ComputePipeline,
    uniform: &UniformFn<'_>,
    spans: &[DeviceSpan],
    size: u64,
) -> Hc4jResult<()> {
    let tile = max_binding_tile(engine);
    let tiles = u32::try_from(size.div_ceil(tile)).map_err(|_| Hc4jError::Unsupported("too many tiles"))?;
    let layout = pipeline.get_bind_group_layout(0);
    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(tiles, |rec| {
        let mut offset = 0;
        while offset < size {
            let len = tile.min(size - offset);
            let uniform_binding = rec.uniform(&uniform((len / 4) as u32))?;
            let mut entries: Vec<wgpu::BindGroupEntry> = spans
                .iter()
                .enumerate()
                .map(|(i, span)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: span.sub_binding(offset, len),
                })
                .collect();
            entries.push(wgpu::BindGroupEntry {
                binding: spans.len() as u32,
                resource: uniform_binding,
            });
            let bind_group = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("HC4J Elementwise"),
                layout: &layout,
                entries: &entries,
            });
            let grid = grid_for_threads(engine, (len / 4).div_ceil(ELEMS_PER_CONTIGUOUS_THREAD))?;
            rec.dispatch(pipeline, &bind_group, grid, len * spans.len() as u64);
            offset += len;
        }
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}

/// Where streamed output tiles land.
enum OutputSink {
    /// The output tensor is VRAM-resident (and pinned): write in place.
    Existing(DeviceSpan),
    /// Fresh storage that replaces the output's when the op completes.
    Device(DeviceBlock),
    Host { data: Vec<u8>, reservation: Reservation },
    Disk { file: SpillFile, written: u64 },
}

impl OutputSink {
    fn device_span(&self) -> Option<&DeviceSpan> {
        match self {
            OutputSink::Existing(span) => Some(span),
            OutputSink::Device(block) => Some(block.span()),
            _ => None,
        }
    }

    fn append(&mut self, chunk: &[u8]) -> Hc4jResult<()> {
        match self {
            OutputSink::Existing(_) | OutputSink::Device(_) => {
                Err(Hc4jError::Device("device sink does not stream".to_string()))
            }
            OutputSink::Host { data, .. } => {
                // Capacity was reserved up front; never let extend reallocate
                // (an infallible reallocation aborts on failure).
                if data.len() + chunk.len() > data.capacity() {
                    return Err(Hc4jError::Readback("streamed output overran its tensor".to_string()));
                }
                data.extend_from_slice(chunk);
                Ok(())
            }
            OutputSink::Disk { file, written } => {
                file.write_all_at(chunk, *written)?;
                *written += chunk.len() as u64;
                Ok(())
            }
        }
    }

    /// New storage to install for the output, or `None` if it was written in
    /// place.
    fn into_residency(self) -> Option<Residency> {
        match self {
            OutputSink::Existing(_) => None,
            OutputSink::Device(block) => Some(Residency::DeviceResident(block)),
            OutputSink::Host { data, reservation } => Some(Residency::HostResident(HostBlock::new(data, reservation))),
            OutputSink::Disk { file, .. } => Some(Residency::Evicted(file)),
        }
    }
}

fn open_output_sink(mgr: &TieredMemoryManager, size: u64) -> Hc4jResult<OutputSink> {
    match mgr.allocate_device(size, false) {
        Ok(block) => return Ok(OutputSink::Device(block)),
        Err(Hc4jError::OutOfMemory) => {}
        Err(err) => return Err(err),
    }
    if let Some(reservation) = mgr.reserve_host(size)?
        && let Some(data) = try_with_capacity(size)
    {
        return Ok(OutputSink::Host { data, reservation });
    }
    Ok(OutputSink::Disk {
        file: SpillFile::create(mgr.spill_dir(), size)?,
        written: 0,
    })
}

/// Allocates tile-sized scratch for `staged` off-device inputs (and the
/// output if `need_out`), halving the tile until they fit. Scratch comes out
/// of the VRAM budget like any tensor, so it can evict LRU tensors too.
fn allocate_scratch(
    mgr: &TieredMemoryManager,
    staged: usize,
    need_out: bool,
    size: u64,
) -> Hc4jResult<(u64, Vec<DeviceBlock>, Option<DeviceBlock>)> {
    let engine = mgr.engine();
    let align = (engine.limits.min_storage_buffer_offset_alignment as u64).max(4);
    let whole = size.div_ceil(align) * align;
    let mut tile = max_binding_tile(engine).min(TRANSFER_CHUNK).min(whole);
    loop {
        let attempt = (|| -> Hc4jResult<(Vec<DeviceBlock>, Option<DeviceBlock>)> {
            let mut inputs = Vec::with_capacity(staged);
            for _ in 0..staged {
                inputs.push(mgr.allocate_device(tile, false)?);
            }
            let out = if need_out { Some(mgr.allocate_device(tile, false)?) } else { None };
            Ok((inputs, out))
        })();
        match attempt {
            Ok((inputs, out)) => return Ok((tile, inputs, out)),
            Err(Hc4jError::OutOfMemory) if tile / 2 >= MIN_STREAM_TILE.min(whole) && tile > align => {
                // Halving keeps the tile a multiple of the offset alignment.
                tile = (tile / 2).div_ceil(align) * align;
            }
            Err(err) => return Err(err),
        }
    }
}

fn stream_contiguous(
    mgr: &TieredMemoryManager,
    pipeline: &wgpu::ComputePipeline,
    uniform: &UniformFn<'_>,
    inputs: &[TensorId],
    output: TensorId,
    size: u64,
) -> Hc4jResult<()> {
    let engine = mgr.engine();
    // Pinned for the whole op: device-resident operands cannot be evicted,
    // and host/disk snapshots stay readable even if the tensor migrates.
    let mut sources = Vec::with_capacity(inputs.len());
    let mut pins = Vec::with_capacity(inputs.len() + 1);
    for &id in inputs {
        let (snapshot, _, pin) = mgr.pin_snapshot(id)?;
        sources.push(snapshot);
        pins.push(pin);
    }
    let (out_snapshot, _, out_pin) = mgr.pin_snapshot(output)?;
    pins.push(out_pin);
    let mut sink = match out_snapshot {
        Snapshot::Device(span) => OutputSink::Existing(span),
        _ => open_output_sink(mgr, size)?,
    };

    let staged: Vec<usize> = sources
        .iter()
        .enumerate()
        .filter(|(_, s)| !matches!(s, Snapshot::Device(_)))
        .map(|(i, _)| i)
        .collect();
    let need_out = sink.device_span().is_none();
    let (tile, scratch_in, scratch_out) = allocate_scratch(mgr, staged.len(), need_out, size)?;

    let mut disk_buf = Vec::new();
    if sources.iter().any(|s| matches!(s, Snapshot::Disk(_))) {
        disk_buf = try_with_capacity(tile).ok_or(Hc4jError::OutOfMemory)?;
        disk_buf.resize(tile as usize, 0);
    }
    let layout = pipeline.get_bind_group_layout(0);
    let n = sources.len();

    mgr.with_ring(|ring| {
        let slots = ring.slot_count() as u64;
        let tiles = size.div_ceil(tile);
        let mut device_inflight = VecDeque::with_capacity(DEVICE_OUTPUT_INFLIGHT + 1);

        for t in 0..tiles {
            let offset = t * tile;
            let len = tile.min(size - offset);
            let slot = (t % slots) as usize;
            if need_out {
                // Retire tile t - slots before reusing its staging slot.
                ring.drain(engine, slot, &mut |chunk: &[u8]| sink.append(chunk))?;
            }

            // Stage off-device inputs into scratch (ordered after earlier work).
            for (scratch, &k) in scratch_in.iter().zip(&staged) {
                let dst = scratch.span();
                match &sources[k] {
                    Snapshot::Host(data) => {
                        let slice = data
                            .get(offset as usize..(offset + len) as usize)
                            .ok_or(Hc4jError::Readback("host tensor shorter than recorded size".to_string()))?;
                        engine.stream.write_buffer(&dst.buffer, dst.offset, slice);
                    }
                    Snapshot::Disk(file) => {
                        let buf = &mut disk_buf[..len as usize];
                        spill::read_exact_at(file, buf, offset)?;
                        engine.stream.write_buffer(&dst.buffer, dst.offset, buf);
                    }
                    Snapshot::Device(_) => {}
                }
            }

            let trap = ErrorTrap::push(&engine.device);
            let staging = ring.buffer(slot);
            let ((), submission) = engine.stream.submit_now(1, |rec| {
                let uniform_binding = rec.uniform(&uniform((len / 4) as u32))?;
                let mut entries = Vec::with_capacity(n + 2);
                let mut staged_iter = scratch_in.iter();
                for (k, source) in sources.iter().enumerate() {
                    let resource = match source {
                        Snapshot::Device(span) => span.sub_binding(offset, len),
                        _ => staged_iter
                            .next()
                            .ok_or(Hc4jError::Device("input scratch missing".to_string()))?
                            .span()
                            .sub_binding(0, len),
                    };
                    entries.push(wgpu::BindGroupEntry { binding: k as u32, resource });
                }
                let out_resource = match (sink.device_span(), scratch_out.as_ref()) {
                    (Some(span), _) => span.sub_binding(offset, len),
                    (None, Some(scratch)) => scratch.span().sub_binding(0, len),
                    (None, None) => return Err(Hc4jError::Device("output scratch missing".to_string())),
                };
                entries.push(wgpu::BindGroupEntry { binding: n as u32, resource: out_resource });
                entries.push(wgpu::BindGroupEntry {
                    binding: n as u32 + 1,
                    resource: uniform_binding,
                });
                let bind_group = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("HC4J Stream Tile"),
                    layout: &layout,
                    entries: &entries,
                });
                let grid = grid_for_threads(engine, (len / 4).div_ceil(ELEMS_PER_CONTIGUOUS_THREAD))?;
                rec.dispatch(pipeline, &bind_group, grid, len * (n as u64 + 1));
                if let Some(scratch) = scratch_out.as_ref().filter(|_| need_out) {
                    let src = scratch.span();
                    rec.encoder().copy_buffer_to_buffer(&src.buffer, src.offset, staging, 0, len);
                }
                Ok(())
            })?;
            trap.finish()?;

            if need_out {
                ring.arm(slot, len, submission);
            } else {
                // No readback throttles the loop, so bound the upload staging
                // wgpu holds for in-flight tiles explicitly.
                device_inflight.push_back(submission);
                if device_inflight.len() > DEVICE_OUTPUT_INFLIGHT
                    && let Some(oldest) = device_inflight.pop_front()
                {
                    engine.wait_for(oldest)?;
                }
            }
        }
        if need_out {
            for t in tiles.saturating_sub(slots)..tiles {
                ring.drain(engine, (t % slots) as usize, &mut |chunk: &[u8]| sink.append(chunk))?;
            }
        }
        Ok(())
    })?;

    if let Some(residency) = sink.into_residency() {
        mgr.replace_residency(output, residency)?;
    }
    drop(pins);
    mgr.record_streamed_op();
    Ok(())
}

// ============================================================================
// Strided executor
// ============================================================================

/// Runs a strided kernel over resident `operands` (inputs, then output) with
/// one invocation per logical element.
pub fn run_strided(pipeline: &wgpu::ComputePipeline, operands: &[TensorId], dims: &[u8], threads: u64) -> Hc4jResult<()> {
    if threads == 0 {
        return Ok(());
    }
    let mgr = manager()?;
    let engine = mgr.engine();
    let resident = mgr.acquire_resident(operands)?;
    let max_binding = engine.limits.max_storage_buffer_binding_size;
    if resident.spans.iter().any(|s| s.size > max_binding) {
        return Err(Hc4jError::Unsupported("strided kernel on a tensor above the storage-binding limit"));
    }
    let grid = grid_for_threads(engine, threads)?;
    let bytes: u64 = resident.spans.iter().map(|s| s.size).sum();
    let layout = pipeline.get_bind_group_layout(0);

    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine.stream.record(1, |rec| {
        let uniform_binding = rec.uniform(dims)?;
        let mut entries: Vec<wgpu::BindGroupEntry> = resident
            .spans
            .iter()
            .enumerate()
            .map(|(i, span)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: span.binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: resident.spans.len() as u32,
            resource: uniform_binding,
        });
        let bind_group = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HC4J Strided"),
            layout: &layout,
            entries: &entries,
        });
        rec.dispatch(pipeline, &bind_group, grid, bytes);
        Ok(())
    });
    let trapped = trap.finish();
    recorded.and(trapped)
}

#[cfg(test)]
mod tests {
    use super::{grid_2d, pruned_layout};

    #[test]
    fn grid_fits_in_one_dimension() {
        assert_eq!(grid_2d(1, 65535), (1, 1));
        assert_eq!(grid_2d(0, 65535), (1, 1));
        assert_eq!(grid_2d(65535, 65535), (65535, 1));
    }

    #[test]
    fn grid_folds_just_past_the_limit_without_doubling() {
        // 67.2M f32 elements = 16.8M vec4 threads = 65,625 workgroups.
        let (x, y) = grid_2d(65_625, 65_535);
        assert_eq!(y, 2);
        assert!(x as u64 * y as u64 >= 65_625);
        assert!(x as u64 * y as u64 - 65_625 < y as u64, "surplus below one row");
    }

    #[test]
    fn grid_covers_every_workgroup_at_scale() {
        for workgroups in [65_536u32, 131_071, 524_288, 4_000_000] {
            for limit in [65_535u32, 65_536] {
                let (x, y) = grid_2d(workgroups, limit);
                assert!(x <= limit && y <= limit);
                assert!(x as u64 * y as u64 >= workgroups as u64);
            }
        }
    }

    #[test]
    fn layout_is_reverse_pruned() {
        // shape [2, 1, 3] with strides a = [3, 3, 1], c = [1, 1, 2]
        let shape = [2u32, 1, 3];
        let a = [3u32, 3, 1];
        let c = [1u32, 1, 2];
        let (s, st) = unsafe { pruned_layout(3, shape.as_ptr(), &[a.as_ptr(), c.as_ptr()]).unwrap() };
        assert_eq!(&s[..3], &[3, 2, 1]);
        assert_eq!(&st[0][..2], &[1, 3]);
        assert_eq!(&st[1][..2], &[2, 1]);
    }

    #[test]
    fn layout_rejects_bad_input() {
        let shape = [2u32; 9];
        assert!(unsafe { pruned_layout(9, shape.as_ptr(), &[shape.as_ptr()]) }.is_err());
        assert!(unsafe { pruned_layout(2, std::ptr::null(), &[shape.as_ptr()]) }.is_err());
    }
}
