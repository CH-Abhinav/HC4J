//! Per-device compute throughput (GFLOP/s) and VRAM capacity profiling.

use std::time::Instant;
use ash::vk;

use crate::collective::backend::DeviceProfileC;
use crate::collective::vulkan::device::{
    BufMem, GpuNode, alloc_device_local, alloc_staging, destroy_buf, one_shot_submit,
};
use crate::error::{Hc4jError, Hc4jResult};

pub(crate) fn map_device_type(ty: vk::PhysicalDeviceType) -> u32 {
    match ty {
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::DISCRETE_GPU => 2,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 3,
        vk::PhysicalDeviceType::CPU => 4,
        _ => 0,
    }
}

/// Measures VRAM bytes from DEVICE_LOCAL memory heaps.
pub(crate) fn query_vram_bytes(node: &GpuNode) -> u64 {
    let mut vram = 0u64;
    let count = node.mem_props.memory_heap_count as usize;
    for heap in &node.mem_props.memory_heaps[..count] {
        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            vram = vram.saturating_add(heap.size);
        }
    }
    if vram == 0 {
        for heap in &node.mem_props.memory_heaps[..count] {
            if heap.size > vram {
                vram = heap.size;
            }
        }
    }
    vram
}

/// Measures sustained compute throughput (GFLOP/s) using the compiled FMA benchmark shader.
pub(crate) fn measure_compute_gflops(node: &GpuNode) -> Hc4jResult<f64> {
    const WORKGROUP_SIZE: u32 = 256;
    let is_discrete = node.props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
    let workgroups: u32 = if is_discrete { 512 } else { 128 };
    let total_threads: u32 = workgroups * WORKGROUP_SIZE; // 131,072 threads on discrete, 32,768 on other
    let out_bytes: vk::DeviceSize = (total_threads as vk::DeviceSize) * 4; // 512 KiB on discrete, 128 KiB on other

    let spv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/collective_bench_fma.spv"));
    let spv_words: &[u32] = bytemuck::cast_slice(spv_bytes);

    let sm_ci = vk::ShaderModuleCreateInfo::default().code(spv_words);
    let shader_module = unsafe { node.device.create_shader_module(&sm_ci, None) }
        .map_err(|e| Hc4jError::Device(format!("create_shader_module bench_fma failed: {e}")))?;

    // Allocate resources
    let out_buf = match alloc_device_local(node, out_bytes, vk::BufferUsageFlags::STORAGE_BUFFER) {
        Ok(b) => b,
        Err(e) => {
            unsafe { node.device.destroy_shader_module(shader_module, None) };
            return Err(e);
        }
    };

    let uniform_buf = match alloc_staging(
        node,
        16,
        vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    ) {
        Ok(b) => b,
        Err(e) => {
            destroy_buf(node, &out_buf);
            unsafe { node.device.destroy_shader_module(shader_module, None) };
            return Err(e);
        }
    };

    let result = run_fma_benchmark(
        node,
        shader_module,
        &out_buf,
        &uniform_buf,
        total_threads,
        workgroups,
    );

    // Cleanup all profiling resources cleanly
    destroy_buf(node, &uniform_buf);
    destroy_buf(node, &out_buf);
    unsafe {
        node.device.destroy_shader_module(shader_module, None);
    }

    result
}

fn write_iterations(node: &GpuNode, uniform_buf: &BufMem, iterations: u32) -> Hc4jResult<()> {
    let params = [iterations, 0u32, 0u32, 0u32];
    unsafe {
        let ptr = node
            .device
            .map_memory(uniform_buf.memory, 0, 16, vk::MemoryMapFlags::empty())
            .map_err(|e| Hc4jError::Device(format!("map uniform buffer failed: {e}")))?
            as *mut u32;
        std::ptr::copy_nonoverlapping(params.as_ptr(), ptr, 4);
        node.device.unmap_memory(uniform_buf.memory);
    }
    Ok(())
}

fn run_fma_benchmark(
    node: &GpuNode,
    shader_module: vk::ShaderModule,
    out_buf: &BufMem,
    uniform_buf: &BufMem,
    total_threads: u32,
    workgroups: u32,
) -> Hc4jResult<f64> {
    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: std::ptr::null(),
            _marker: std::marker::PhantomData,
        },
        vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: std::ptr::null(),
            _marker: std::marker::PhantomData,
        },
    ];

    let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let desc_set_layout = unsafe { node.device.create_descriptor_set_layout(&dsl_ci, None) }
        .map_err(|e| Hc4jError::Device(format!("create_descriptor_set_layout failed: {e}")))?;

    let layout_ci = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&desc_set_layout));
    let pipe_layout = match unsafe { node.device.create_pipeline_layout(&layout_ci, None) } {
        Ok(l) => l,
        Err(e) => {
            unsafe { node.device.destroy_descriptor_set_layout(desc_set_layout, None) };
            return Err(Hc4jError::Device(format!("create_pipeline_layout failed: {e}")));
        }
    };

    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(c"main");

    let pipe_ci = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipe_layout);

    let pipeline = match unsafe {
        node.device.create_compute_pipelines(vk::PipelineCache::null(), &[pipe_ci], None)
    } {
        Ok(pipes) => pipes[0],
        Err((_, e)) => {
            unsafe {
                node.device.destroy_pipeline_layout(pipe_layout, None);
                node.device.destroy_descriptor_set_layout(desc_set_layout, None);
            }
            return Err(Hc4jError::Device(format!("create_compute_pipelines failed: {e:?}")));
        }
    };

    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        },
    ];

    let pool_ci = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);

    let desc_pool = match unsafe { node.device.create_descriptor_pool(&pool_ci, None) } {
        Ok(p) => p,
        Err(e) => {
            unsafe {
                node.device.destroy_pipeline(pipeline, None);
                node.device.destroy_pipeline_layout(pipe_layout, None);
                node.device.destroy_descriptor_set_layout(desc_set_layout, None);
            }
            return Err(Hc4jError::Device(format!("create_descriptor_pool failed: {e}")));
        }
    };

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(desc_pool)
        .set_layouts(std::slice::from_ref(&desc_set_layout));

    let desc_set = match unsafe { node.device.allocate_descriptor_sets(&alloc_info) } {
        Ok(sets) => sets[0],
        Err(e) => {
            unsafe {
                node.device.destroy_descriptor_pool(desc_pool, None);
                node.device.destroy_pipeline(pipeline, None);
                node.device.destroy_pipeline_layout(pipe_layout, None);
                node.device.destroy_descriptor_set_layout(desc_set_layout, None);
            }
            return Err(Hc4jError::Device(format!("allocate_descriptor_sets failed: {e}")));
        }
    };

    // Bind buffers to descriptor set
    let uniform_info = [vk::DescriptorBufferInfo {
        buffer: uniform_buf.buffer,
        offset: 0,
        range: 16,
    }];
    let storage_info = [vk::DescriptorBufferInfo {
        buffer: out_buf.buffer,
        offset: 0,
        range: out_buf.size,
    }];

    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&uniform_info),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&storage_info),
    ];

    unsafe {
        node.device.update_descriptor_sets(&writes, &[]);
    }

    let dispatch = |wg_count: u32| -> Hc4jResult<()> {
        one_shot_submit(node, |cb| unsafe {
            node.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            node.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipe_layout,
                0,
                &[desc_set],
                &[],
            );
            node.device.cmd_dispatch(cb, wg_count, 1, 1);
        })
    };

    // 1. Warm-up dispatch (64 iterations)
    write_iterations(node, uniform_buf, 64)?;
    dispatch(workgroups)?;

    // 2. Initial timed run (256 iterations)
    write_iterations(node, uniform_buf, 256)?;
    let t0 = Instant::now();
    dispatch(workgroups)?;
    let elapsed0 = t0.elapsed();

    let (total_flops, elapsed_secs) = if elapsed0.as_secs_f64() < 0.003 {
        // Fast GPU: scale up to 2048 iterations (~536 MFLOPs)
        write_iterations(node, uniform_buf, 2048)?;
        let t1 = Instant::now();
        dispatch(workgroups)?;
        let elapsed1 = t1.elapsed();
        if elapsed1.as_secs_f64() < 0.003 {
            // Ultra-fast discrete GPU: scale up to 8192 iterations
            write_iterations(node, uniform_buf, 8192)?;
            let t2 = Instant::now();
            dispatch(workgroups)?;
            let elapsed2 = t2.elapsed();
            let flops = (total_threads as f64) * 8192.0 * 8.0;
            (flops, elapsed2.as_secs_f64())
        } else {
            let flops = (total_threads as f64) * 2048.0 * 8.0;
            (flops, elapsed1.as_secs_f64())
        }
    } else {
        let flops = (total_threads as f64) * 256.0 * 8.0;
        (flops, elapsed0.as_secs_f64())
    };

    let gflops = if elapsed_secs > 0.0 {
        (total_flops / elapsed_secs / 1e9).max(0.01)
    } else {
        0.01
    };

    // Teardown pipeline resources
    unsafe {
        node.device.destroy_descriptor_pool(desc_pool, None);
        node.device.destroy_pipeline(pipeline, None);
        node.device.destroy_pipeline_layout(pipe_layout, None);
        node.device.destroy_descriptor_set_layout(desc_set_layout, None);
    }

    Ok(gflops)
}

/// Profiles a GPU node and creates a DeviceProfileC.
#[allow(dead_code)]
pub(crate) fn profile_node(rank: u32, node: &GpuNode) -> Hc4jResult<DeviceProfileC> {
    let vram_bytes = query_vram_bytes(node);
    let compute_gflops = measure_compute_gflops(node)?;

    Ok(DeviceProfileC {
        rank,
        vendor_id: node.props.vendor_id,
        device_id: node.props.device_id,
        device_type: map_device_type(node.props.device_type),
        vram_bytes,
        compute_gflops,
    })
}
