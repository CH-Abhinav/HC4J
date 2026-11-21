//! Reduction compute pipeline, persistent descriptor set management (Fix #3),
//! and persistent scratch buffers (Fix #4).

use ash::vk;

use crate::collective::vulkan::device::{BufMem, GpuNode, alloc_device_local, alloc_staging, destroy_buf, one_shot_submit};
use crate::error::{Hc4jError, Hc4jResult};

/// SPIR-V bytecode generated from `reduce.wgsl` at build time.
pub(crate) const REDUCE_SHADER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/collective_reduce.spv"));

pub(crate) const WG_SIZE: u32 = 256;

pub(crate) struct ReducePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub desc_pool: vk::DescriptorPool,
    pub desc_set: vk::DescriptorSet,
    pub shader_module: vk::ShaderModule,
    pub uniform_buf: BufMem,
    pub send_chunk_buf: Option<BufMem>,
    pub recv_chunk_buf: Option<BufMem>,
}

impl ReducePipeline {
    pub fn new(node: &GpuNode) -> Hc4jResult<Self> {
        let mut cursor = std::io::Cursor::new(REDUCE_SHADER_SPV);
        let code = ash::util::read_spv(&mut cursor)
            .map_err(|e| Hc4jError::Device(format!("failed to parse SPIR-V bytecode: {e}")))?;

        let module_ci = vk::ShaderModuleCreateInfo::default().code(&code);
        let shader_module = unsafe { node.device.create_shader_module(&module_ci, None) }
            .map_err(|e| Hc4jError::Device(format!("create_shader_module failed: {e}")))?;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_set_layout = unsafe { node.device.create_descriptor_set_layout(&dsl_ci, None) }
            .map_err(|e| {
                unsafe { node.device.destroy_shader_module(shader_module, None) };
                Hc4jError::Device(format!("create_descriptor_set_layout failed: {e}"))
            })?;

        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&desc_set_layout));
        let layout = unsafe { node.device.create_pipeline_layout(&layout_ci, None) }
            .map_err(|e| {
                unsafe {
                    node.device.destroy_descriptor_set_layout(desc_set_layout, None);
                    node.device.destroy_shader_module(shader_module, None);
                }
                Hc4jError::Device(format!("create_pipeline_layout failed: {e}"))
            })?;

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        let pipe_ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);

        let pipeline = unsafe {
            node.device.create_compute_pipelines(vk::PipelineCache::null(), &[pipe_ci], None)
        }
        .map_err(|e| {
            unsafe {
                node.device.destroy_pipeline_layout(layout, None);
                node.device.destroy_descriptor_set_layout(desc_set_layout, None);
                node.device.destroy_shader_module(shader_module, None);
            }
            Hc4jError::Device(format!("create_compute_pipelines failed: {e:?}"))
        })?[0];

        // Fix #3: Pool with 1 persistent descriptor set per pipeline
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
        ];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let desc_pool = unsafe { node.device.create_descriptor_pool(&pool_ci, None) }
            .map_err(|e| {
                unsafe {
                    node.device.destroy_pipeline(pipeline, None);
                    node.device.destroy_pipeline_layout(layout, None);
                    node.device.destroy_descriptor_set_layout(desc_set_layout, None);
                    node.device.destroy_shader_module(shader_module, None);
                }
                Hc4jError::Device(format!("create_descriptor_pool failed: {e}"))
            })?;

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(std::slice::from_ref(&desc_set_layout));
        let desc_set = unsafe { node.device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| {
                unsafe {
                    node.device.destroy_descriptor_pool(desc_pool, None);
                    node.device.destroy_pipeline(pipeline, None);
                    node.device.destroy_pipeline_layout(layout, None);
                    node.device.destroy_descriptor_set_layout(desc_set_layout, None);
                    node.device.destroy_shader_module(shader_module, None);
                }
                Hc4jError::Device(format!("allocate_descriptor_sets failed: {e}"))
            })?[0];

        // Allocate a 16-byte persistent uniform buffer for params (count, offset)
        let uniform_buf = alloc_staging(
            node,
            16,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        Ok(Self {
            pipeline,
            layout,
            desc_set_layout,
            desc_pool,
            desc_set,
            shader_module,
            uniform_buf,
            send_chunk_buf: None,
            recv_chunk_buf: None,
        })
    }

    /// Ensures persistent send/receive scratch buffers are allocated and at least `needed_bytes` (Fix #4).
    pub fn ensure_scratch(&mut self, node: &GpuNode, needed_bytes: vk::DeviceSize) -> Hc4jResult<()> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        if let Some(buf) = &self.send_chunk_buf {
            if buf.size < needed_bytes {
                destroy_buf(node, buf);
                self.send_chunk_buf = Some(alloc_device_local(node, needed_bytes, usage)?);
            }
        } else {
            self.send_chunk_buf = Some(alloc_device_local(node, needed_bytes, usage)?);
        }

        if let Some(buf) = &self.recv_chunk_buf {
            if buf.size < needed_bytes {
                destroy_buf(node, buf);
                self.recv_chunk_buf = Some(alloc_device_local(node, needed_bytes, usage)?);
            }
        } else {
            self.recv_chunk_buf = Some(alloc_device_local(node, needed_bytes, usage)?);
        }

        Ok(())
    }

    /// Dispatch element-wise addition: `accum[offset..offset+count] += incoming[0..count]`.
    pub fn dispatch_reduce(
        &self,
        node: &GpuNode,
        accum: vk::Buffer,
        accum_offset: vk::DeviceSize,
        incoming: vk::Buffer,
        count: u32,
    ) -> Hc4jResult<()> {
        if count == 0 {
            return Ok(());
        }

        let elem_offset = (accum_offset / 4) as u32;
        let params: [u32; 4] = [count, elem_offset, 0, 0];

        // Write uniform parameters
        unsafe {
            let ptr = node.device.map_memory(self.uniform_buf.memory, 0, 16, vk::MemoryMapFlags::empty())
                .map_err(|e| Hc4jError::Device(format!("map uniform_buf failed: {e}")))? as *mut u32;
            std::ptr::copy_nonoverlapping(params.as_ptr(), ptr, 4);
            node.device.unmap_memory(self.uniform_buf.memory);
        }

        // Update persistent descriptor set (Fix #3: no allocations)
        let byte_range = count as vk::DeviceSize * 4;
        let buf_info_a = vk::DescriptorBufferInfo {
            buffer: accum,
            offset: 0,
            range: vk::WHOLE_SIZE,
        };
        let buf_info_b = vk::DescriptorBufferInfo {
            buffer: incoming,
            offset: 0,
            range: byte_range,
        };
        let buf_info_p = vk::DescriptorBufferInfo {
            buffer: self.uniform_buf.buffer,
            offset: 0,
            range: 16,
        };

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info_a)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info_b)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info_p)),
        ];

        unsafe {
            node.device.update_descriptor_sets(&writes, &[]);
        }

        one_shot_submit(node, |cb| unsafe {
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);

            node.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );

            node.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            node.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.desc_set],
                &[],
            );

            // Fix #4: WG_SIZE = 256
            let groups = count.div_ceil(WG_SIZE);
            node.device.cmd_dispatch(cb, groups, 1, 1);
        })
    }

    pub fn destroy(&mut self, node: &GpuNode) {
        if let Some(buf) = self.send_chunk_buf.take() {
            destroy_buf(node, &buf);
        }
        if let Some(buf) = self.recv_chunk_buf.take() {
            destroy_buf(node, &buf);
        }
        destroy_buf(node, &self.uniform_buf);

        unsafe {
            node.device.destroy_descriptor_pool(self.desc_pool, None);
            node.device.destroy_pipeline(self.pipeline, None);
            node.device.destroy_pipeline_layout(self.layout, None);
            node.device.destroy_descriptor_set_layout(self.desc_set_layout, None);
            node.device.destroy_shader_module(self.shader_module, None);
        }
    }
}
