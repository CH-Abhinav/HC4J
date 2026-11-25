pub mod grid;
pub mod layout;
pub mod shaders;
pub mod streaming;

pub use grid::*;
pub use layout::*;
pub use shaders::*;
pub use streaming::*;

use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::engine::{ErrorTrap, GpuEngine};
use crate::memory::{DeviceSpan, TensorId, manager};

/// Contiguous kernels process one vec4 per invocation.
pub const ELEMS_PER_CONTIGUOUS_THREAD: u64 = 4;
/// Smallest scratch tile tried before the streaming path gives up.
pub const MIN_STREAM_TILE: u64 = 1 << 20;
pub const DEVICE_OUTPUT_INFLIGHT: usize = 2;

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
    let recorded = engine.stream.record(tiles, "elementwise", |rec| {
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
    let recorded = engine.stream.record(1, "elementwise (strided)", |rec| {
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

/// Dispatches any unary f32 builtin kernel (trig / exp / basic math).
pub fn run_unary(
    family: &str,
    op_name: &str,
    wgsl_op: &str,
    id_a: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<UnaryStrided<'_>>,
) -> Hc4jResult<()> {
    run_unary_kernel(
        UnaryKernel::builtin(family, op_name, wgsl_op),
        id_a,
        id_out,
        length,
        strided,
    )
}

/// Dispatches a parameterized or custom-prelude [`UnaryKernel`] (contiguous or strided).
pub fn run_unary_kernel(
    kernel: UnaryKernel<'_>,
    id_a: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<UnaryStrided<'_>>,
) -> Hc4jResult<()> {
    let pipeline = kernel.pipeline()?;
    let alpha = kernel.alpha_bits;
    let param_w = kernel.param_w_bits;
    match strided {
        None => {
            check_contiguous_length(id_out, length)?;
            run_contiguous(
                &pipeline,
                &|len| bytemuck::bytes_of(&UnaryElemDims::contiguous_with_params(len, alpha, param_w)).to_vec(),
                &[id_a],
                id_out,
            )
        }
        Some((shape, strides_a, strides_c)) => {
            let length_u32 =
                u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("strided length exceeds u32"))?;
            let (pruned_shape, pruned_strides) = prune_strided_layout(shape, &[strides_a, strides_c])?;
            let dims = UnaryElemDims {
                params: [length_u32, 0, alpha, param_w],
                shape: pruned_shape,
                strides_a: pruned_strides[0],
                strides_c: pruned_strides[1],
            };
            run_strided(&pipeline, &[id_a, id_out], bytemuck::bytes_of(&dims), length as u64)
        }
    }
}

/// Shared contiguous/strided executor for 2-input 1-output elementwise pipelines
/// (arithmetic, pow/max/min, comparisons).
pub fn run_binary_elementwise(
    pipeline: &wgpu::ComputePipeline,
    id_a: TensorId,
    id_b: TensorId,
    id_out: TensorId,
    length: usize,
    strided: Option<BinaryStrided<'_>>,
) -> Hc4jResult<()> {
    match strided {
        None => {
            check_contiguous_length(id_out, length)?;
            run_contiguous(
                pipeline,
                &|len| bytemuck::bytes_of(&BinaryElemDims::contiguous(len)).to_vec(),
                &[id_a, id_b],
                id_out,
            )
        }
        Some((shape, strides_a, strides_b, strides_c)) => {
            let length_u32 =
                u32::try_from(length).map_err(|_| Hc4jError::InvalidParam("strided length exceeds u32"))?;
            let (pruned_shape, pruned_strides) = prune_strided_layout(shape, &[strides_a, strides_b, strides_c])?;
            let dims = BinaryElemDims {
                params: [length_u32, 0, 0, 0],
                shape: pruned_shape,
                strides_a: pruned_strides[0],
                strides_b: pruned_strides[1],
                strides_c: pruned_strides[2],
            };
            run_strided(pipeline, &[id_a, id_b, id_out], bytemuck::bytes_of(&dims), length as u64)
        }
    }
}

/// Copies `len` 4-byte elements from `src` (at logical index
/// `src_offset + dot(coords, src_strides)`) into contiguous `out[0..len]`.
/// Works for both f32 and i32 without NaN canonicalization.
pub fn run_copy_strided(
    src: TensorId,
    out: TensorId,
    src_offset: u64,
    shape: &[u32],
    src_strides: &[u32],
    len: usize,
) -> Hc4jResult<()> {
    if len == 0 {
        return Ok(());
    }
    let len_u32 = u32::try_from(len).map_err(|_| Hc4jError::InvalidParam("len exceeds u32"))?;
    let offset_u32 = u32::try_from(src_offset).map_err(|_| Hc4jError::InvalidParam("src_offset exceeds u32"))?;

    let (pruned_shape, pruned_strides) = prune_strided_layout(shape, &[src_strides])?;
    let dims = CopyStridedDims {
        params: [len_u32, offset_u32, 0, 0],
        shape: pruned_shape,
        strides: pruned_strides[0],
    };
    let pipeline = crate::engine::get_engine()?.get_or_compile("copy_strided", "main", COPY_STRIDED_WGSL)?;
    run_strided(&pipeline, &[src, out], bytemuck::bytes_of(&dims), len as u64)
}

