//! Reduction compute pipeline, persistent descriptor set management,
//! double-buffered scratch slots, persistently mapped uniforms, and reusable sync objects.

use ash::vk;

use crate::collective::vulkan::device::{BufMem, GpuNode, alloc_device_local, alloc_staging, destroy_buf, one_shot_submit};
use crate::collective::vulkan::transport::{EDGE_PIPELINE_SLOTS, SendMutPtr};
use crate::engine::error::{Hc4jError, Hc4jResult};

/// SPIR-V bytecode generated from `reduce.wgsl` at build time.
pub(crate) const REDUCE_SHADER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/collective_reduce.spv"));
/// SPIR-V bytecode generated from `elem_mul.wgsl` at build time.
pub(crate) const ELEM_MUL_SHADER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/collective_elem_mul.spv"));

pub(crate) const WG_SIZE: u32 = 256;

/// Reusable primary command buffers and fences per pipeline slot on a rank.
#[derive(Clone, Copy)]
pub(crate) struct RankSlotSync {
    pub send_cb: vk::CommandBuffer,
    pub send_fence: vk::Fence,
    pub recv_cb: vk::CommandBuffer,
    pub recv_fence: vk::Fence,
}

pub(crate) fn submit_cb_with_fence(
    node: &GpuNode,
    queue: vk::Queue,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    record_fn: impl FnOnce(vk::CommandBuffer),
) -> Hc4jResult<()> {
    let begin = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        node.device
            .reset_fences(std::slice::from_ref(&fence))
            .map_err(|e| Hc4jError::Device(format!("reset_fences failed: {e}")))?;
        node.device
            .begin_command_buffer(cb, &begin)
            .map_err(|e| Hc4jError::Device(format!("begin_command_buffer failed: {e}")))?;
        record_fn(cb);
        node.device
            .end_command_buffer(cb)
            .map_err(|e| Hc4jError::Device(format!("end_command_buffer failed: {e}")))?;

        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));
        node.device
            .queue_submit(queue, &[submit], fence)
            .map_err(|e| Hc4jError::Device(format!("queue_submit failed: {e}")))?;
    }
    Ok(())
}

pub(crate) struct ReducePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub desc_pool: vk::DescriptorPool,
    pub desc_sets: [vk::DescriptorSet; EDGE_PIPELINE_SLOTS],
    pub shader_module: vk::ShaderModule,
    pub uniform_bufs: [BufMem; EDGE_PIPELINE_SLOTS],
    pub uniform_ptrs: [SendMutPtr<u32>; EDGE_PIPELINE_SLOTS],
    pub recv_chunk_bufs: [Option<BufMem>; EDGE_PIPELINE_SLOTS],
    pub recv_capacities: [vk::DeviceSize; EDGE_PIPELINE_SLOTS],
    pub sync: [RankSlotSync; EDGE_PIPELINE_SLOTS],
    bound_descs: [Option<(vk::Buffer, vk::Buffer, vk::DeviceSize)>; EDGE_PIPELINE_SLOTS],
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

        // Pool with 2 persistent descriptor sets (one per double-buffered pipeline slot)
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: (2 * EDGE_PIPELINE_SLOTS) as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: EDGE_PIPELINE_SLOTS as u32,
            },
        ];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(EDGE_PIPELINE_SLOTS as u32)
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

        let set_layouts = [desc_set_layout; EDGE_PIPELINE_SLOTS];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&set_layouts);
        let desc_vec = unsafe { node.device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| {
                unsafe {
                    node.device.destroy_descriptor_pool(desc_pool, None);
                    node.device.destroy_pipeline(pipeline, None);
                    node.device.destroy_pipeline_layout(layout, None);
                    node.device.destroy_descriptor_set_layout(desc_set_layout, None);
                    node.device.destroy_shader_module(shader_module, None);
                }
                Hc4jError::Device(format!("allocate_descriptor_sets failed: {e}"))
            })?;
        let desc_sets = [desc_vec[0], desc_vec[1]];

        let cleanup_pipe_base = || unsafe {
            node.device.destroy_descriptor_pool(desc_pool, None);
            node.device.destroy_pipeline(pipeline, None);
            node.device.destroy_pipeline_layout(layout, None);
            node.device.destroy_descriptor_set_layout(desc_set_layout, None);
            node.device.destroy_shader_module(shader_module, None);
        };

        let uniform_usage = vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let u0 = match alloc_staging(node, 16, uniform_usage) {
            Ok(b) => b,
            Err(e) => {
                cleanup_pipe_base();
                return Err(e);
            }
        };
        let u1 = match alloc_staging(node, 16, uniform_usage) {
            Ok(b) => b,
            Err(e) => {
                destroy_buf(node, &u0);
                cleanup_pipe_base();
                return Err(e);
            }
        };
        let ptr0 = match unsafe { node.device.map_memory(u0.memory, 0, 16, vk::MemoryMapFlags::empty()) } {
            Ok(p) => SendMutPtr(p as *mut u32),
            Err(e) => {
                destroy_buf(node, &u1);
                destroy_buf(node, &u0);
                cleanup_pipe_base();
                return Err(Hc4jError::Device(format!("map uniform_buf 0 failed: {e}")));
            }
        };
        let ptr1 = match unsafe { node.device.map_memory(u1.memory, 0, 16, vk::MemoryMapFlags::empty()) } {
            Ok(p) => SendMutPtr(p as *mut u32),
            Err(e) => {
                unsafe { node.device.unmap_memory(u0.memory) };
                destroy_buf(node, &u1);
                destroy_buf(node, &u0);
                cleanup_pipe_base();
                return Err(Hc4jError::Device(format!("map uniform_buf 1 failed: {e}")));
            }
        };

        let slot_cb_count = EDGE_PIPELINE_SLOTS as u32;
        let send_cb_ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(node.send_cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(slot_cb_count);
        let send_cbs = match unsafe { node.device.allocate_command_buffers(&send_cb_ai) } {
            Ok(v) => v,
            Err(e) => {
                unsafe {
                    node.device.unmap_memory(u0.memory);
                    node.device.unmap_memory(u1.memory);
                }
                destroy_buf(node, &u1);
                destroy_buf(node, &u0);
                cleanup_pipe_base();
                return Err(Hc4jError::Device(format!("allocate ring send command buffers failed: {e}")));
            }
        };

        let recv_cb_ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(node.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(slot_cb_count);
        let recv_cbs = match unsafe { node.device.allocate_command_buffers(&recv_cb_ai) } {
            Ok(v) => v,
            Err(e) => {
                unsafe {
                    node.device.free_command_buffers(node.send_cmd_pool, &send_cbs);
                    node.device.unmap_memory(u0.memory);
                    node.device.unmap_memory(u1.memory);
                }
                destroy_buf(node, &u1);
                destroy_buf(node, &u0);
                cleanup_pipe_base();
                return Err(Hc4jError::Device(format!("allocate ring recv command buffers failed: {e}")));
            }
        };

        let fence_ci = vk::FenceCreateInfo::default();
        let mut fences = Vec::with_capacity(EDGE_PIPELINE_SLOTS * 2);
        for _ in 0..(EDGE_PIPELINE_SLOTS * 2) {
            match unsafe { node.device.create_fence(&fence_ci, None) } {
                Ok(f) => fences.push(f),
                Err(e) => {
                    for f in fences {
                        unsafe { node.device.destroy_fence(f, None) };
                    }
                    unsafe {
                        node.device.free_command_buffers(node.send_cmd_pool, &send_cbs);
                        node.device.free_command_buffers(node.cmd_pool, &recv_cbs);
                        node.device.unmap_memory(u0.memory);
                        node.device.unmap_memory(u1.memory);
                    }
                    destroy_buf(node, &u1);
                    destroy_buf(node, &u0);
                    cleanup_pipe_base();
                    return Err(Hc4jError::Device(format!("create_fence failed: {e}")));
                }
            }
        }

        let sync = [
            RankSlotSync {
                send_cb: send_cbs[0],
                send_fence: fences[0],
                recv_cb: recv_cbs[0],
                recv_fence: fences[1],
            },
            RankSlotSync {
                send_cb: send_cbs[1],
                send_fence: fences[2],
                recv_cb: recv_cbs[1],
                recv_fence: fences[3],
            },
        ];

        Ok(Self {
            pipeline,
            layout,
            desc_set_layout,
            desc_pool,
            desc_sets,
            shader_module,
            uniform_bufs: [u0, u1],
            uniform_ptrs: [ptr0, ptr1],
            recv_chunk_bufs: [None, None],
            recv_capacities: [0; EDGE_PIPELINE_SLOTS],
            sync,
            bound_descs: [None; EDGE_PIPELINE_SLOTS],
        })
    }

    #[inline]
    pub fn invalidate_desc_cache(&mut self) {
        self.bound_descs = [None; EDGE_PIPELINE_SLOTS];
    }

    /// Ensures persistent double-buffered receive scratch buffers are allocated and at least `needed_bytes`.
    pub fn ensure_recv_slots(&mut self, node: &GpuNode, needed_bytes: vk::DeviceSize) -> Hc4jResult<()> {
        let req = needed_bytes.max(4);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        let needs_grow = self
            .recv_capacities
            .iter()
            .zip(self.recv_chunk_bufs.iter())
            .any(|(&cap, opt)| opt.is_none() || cap < req);
        if !needs_grow {
            return Ok(());
        }

        unsafe {
            node.device.device_wait_idle().ok();
        }
        self.invalidate_desc_cache();
        for (i, opt) in self.recv_chunk_bufs.iter_mut().enumerate() {
            if opt.is_none() || self.recv_capacities[i] < req {
                if let Some(old) = opt.take() {
                    destroy_buf(node, &old);
                    self.recv_capacities[i] = 0;
                }
                let buf = alloc_device_local(node, req, usage)?;
                *opt = Some(buf);
                self.recv_capacities[i] = req;
            }
        }

        Ok(())
    }

    #[inline]
    pub fn recv_buf(&self, slot: usize) -> vk::Buffer {
        self.recv_chunk_bufs[slot % EDGE_PIPELINE_SLOTS]
            .as_ref()
            .expect("recv_chunk_buf not ensured before use")
            .buffer
    }

    #[inline]
    pub fn sync(&self, slot: usize) -> &RankSlotSync {
        &self.sync[slot % EDGE_PIPELINE_SLOTS]
    }

    fn update_reduce_bindings(
        &mut self,
        node: &GpuNode,
        slot: usize,
        accum: vk::Buffer,
        accum_offset: vk::DeviceSize,
        incoming: vk::Buffer,
        count: u32,
    ) {
        let s = slot % EDGE_PIPELINE_SLOTS;
        let elem_offset = (accum_offset / 4) as u32;
        let params: [u32; 4] = [count, elem_offset, 0, 0];

        unsafe {
            std::ptr::copy_nonoverlapping(params.as_ptr(), self.uniform_ptrs[s].0, 4);
        }

        let byte_range = count as vk::DeviceSize * 4;
        let key = (accum, incoming, byte_range);
        if self.bound_descs[s] == Some(key) {
            return;
        }

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
            buffer: self.uniform_bufs[s].buffer,
            offset: 0,
            range: 16,
        };

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[s])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info_a)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[s])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info_b)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[s])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info_p)),
        ];

        unsafe {
            node.device.update_descriptor_sets(&writes, &[]);
        }
        self.bound_descs[s] = Some(key);
    }

    #[allow(clippy::too_many_arguments)]
    fn record_reduce_dispatch(
        &mut self,
        node: &GpuNode,
        cb: vk::CommandBuffer,
        s: usize,
        accum: vk::Buffer,
        accum_offset: vk::DeviceSize,
        rhs_buf: vk::Buffer,
        count: u32,
    ) {
        self.update_reduce_bindings(node, s, accum, accum_offset, rhs_buf, count);

        // Each workgroup has 256 threads and each invocation reduces 4 floats (1024 floats/wg).
        let groups = count.div_ceil(WG_SIZE * 4);
        let max_x = node.props.limits.max_compute_work_group_count[0];
        let (gx, gy) = crate::kernels::elementwise::grid_2d(groups, max_x);

        unsafe {
            node.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            node.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.desc_sets[s]],
                &[],
            );
            node.device.cmd_dispatch(cb, gx, gy, 1);

            let compute_to_xfer = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            node.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[compute_to_xfer],
                &[],
                &[],
            );
        }
    }

    /// Prepares descriptors/uniform and records a single fused receive+reduce submission:
    /// `cmd_copy_buffer(edge_dst_buf -> recv_slot) + barrier + dispatch(reduce) + post-barrier`.
    #[allow(clippy::too_many_arguments)]
    pub fn record_copy_and_reduce(
        &mut self,
        node: &GpuNode,
        cb: vk::CommandBuffer,
        slot: usize,
        accum: vk::Buffer,
        accum_offset: vk::DeviceSize,
        edge_dst_buf: vk::Buffer,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        let s = slot % EDGE_PIPELINE_SLOTS;
        let recv_buf = self.recv_buf(s);
        let byte_range = count as vk::DeviceSize * 4;
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: byte_range,
        };

        unsafe {
            node.device.cmd_copy_buffer(cb, edge_dst_buf, recv_buf, &[region]);

            let xfer_to_compute = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            node.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[xfer_to_compute],
                &[],
                &[],
            );
        }

        self.record_reduce_dispatch(node, cb, s, accum, accum_offset, recv_buf, count);
    }

    /// Prepares descriptors/uniform and records a zero-bounce direct-read reduce submission:
    /// skips the VRAM copy and pre-compute barrier by binding `edge_dst_buf` directly.
    #[allow(clippy::too_many_arguments)]
    pub fn record_direct_reduce(
        &mut self,
        node: &GpuNode,
        cb: vk::CommandBuffer,
        slot: usize,
        accum: vk::Buffer,
        accum_offset: vk::DeviceSize,
        edge_dst_buf: vk::Buffer,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        let s = slot % EDGE_PIPELINE_SLOTS;
        self.record_reduce_dispatch(node, cb, s, accum, accum_offset, edge_dst_buf, count);
    }

    pub fn destroy(&mut self, node: &GpuNode) {
        let mut send_cbs = Vec::with_capacity(EDGE_PIPELINE_SLOTS);
        let mut recv_cbs = Vec::with_capacity(EDGE_PIPELINE_SLOTS);
        for sync in &self.sync {
            send_cbs.push(sync.send_cb);
            recv_cbs.push(sync.recv_cb);
            unsafe {
                node.device.destroy_fence(sync.send_fence, None);
                node.device.destroy_fence(sync.recv_fence, None);
            }
        }
        unsafe {
            node.device.free_command_buffers(node.send_cmd_pool, &send_cbs);
            node.device.free_command_buffers(node.cmd_pool, &recv_cbs);
        }

        for opt in &mut self.recv_chunk_bufs {
            if let Some(buf) = opt.take() {
                destroy_buf(node, &buf);
            }
        }
        for ubuf in &self.uniform_bufs {
            unsafe {
                node.device.unmap_memory(ubuf.memory);
            }
            destroy_buf(node, ubuf);
        }

        unsafe {
            node.device.destroy_descriptor_pool(self.desc_pool, None);
            node.device.destroy_pipeline(self.pipeline, None);
            node.device.destroy_pipeline_layout(self.layout, None);
            node.device.destroy_descriptor_set_layout(self.desc_set_layout, None);
            node.device.destroy_shader_module(self.shader_module, None);
        }
    }
}

pub(crate) const GEMM_BIAS_ACT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/collective_gemm_bias_act.spv"));
pub(crate) const SPLIT_KV_ATTN_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/collective_split_kv_attn.spv"));

pub(crate) struct GemmPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub desc_pool: vk::DescriptorPool,
    pub desc_set: vk::DescriptorSet,
    pub shader_module: vk::ShaderModule,
    pub uniform_buf: BufMem,
    pub dummy_bias: BufMem,
}

impl GemmPipeline {
    pub fn new(node: &GpuNode) -> Hc4jResult<Self> {
        let mut cursor = std::io::Cursor::new(GEMM_BIAS_ACT_SPV);
        let code = ash::util::read_spv(&mut cursor)
            .map_err(|e| Hc4jError::Device(format!("failed to parse GEMM SPIR-V bytecode: {e}")))?;

        let module_ci = vk::ShaderModuleCreateInfo::default().code(&code);
        let shader_module = unsafe { node.device.create_shader_module(&module_ci, None) }
            .map_err(|e| Hc4jError::Device(format!("create_shader_module for GEMM failed: {e}")))?;

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
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_set_layout = unsafe { node.device.create_descriptor_set_layout(&dsl_ci, None) }
            .map_err(|e| {
                unsafe { node.device.destroy_shader_module(shader_module, None) };
                Hc4jError::Device(format!("create_descriptor_set_layout for GEMM failed: {e}"))
            })?;

        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&desc_set_layout));
        let layout = unsafe { node.device.create_pipeline_layout(&layout_ci, None) }
            .map_err(|e| {
                unsafe {
                    node.device.destroy_descriptor_set_layout(desc_set_layout, None);
                    node.device.destroy_shader_module(shader_module, None);
                }
                Hc4jError::Device(format!("create_pipeline_layout for GEMM failed: {e}"))
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
            Hc4jError::Device(format!("create_compute_pipelines for GEMM failed: {e:?}"))
        })?[0];

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 4,
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
                Hc4jError::Device(format!("create_descriptor_pool for GEMM failed: {e}"))
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
                Hc4jError::Device(format!("allocate_descriptor_sets for GEMM failed: {e}"))
            })?[0];

        let uniform_buf = alloc_staging(
            node,
            16,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let dummy_bias = alloc_device_local(
            node,
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        Ok(Self {
            pipeline,
            layout,
            desc_set_layout,
            desc_pool,
            desc_set,
            shader_module,
            uniform_buf,
            dummy_bias,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        node: &GpuNode,
        buf_a: vk::Buffer,
        buf_b: vk::Buffer,
        buf_bias: Option<vk::Buffer>,
        buf_c: vk::Buffer,
        m: u32,
        n: u32,
        k: u32,
        flags: u32,
    ) -> Hc4jResult<()> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        let params: [u32; 4] = [m, n, k, flags];
        unsafe {
            let ptr = node.device.map_memory(self.uniform_buf.memory, 0, 16, vk::MemoryMapFlags::empty())
                .map_err(|e| Hc4jError::Device(format!("map uniform_buf failed: {e}")))? as *mut u32;
            std::ptr::copy_nonoverlapping(params.as_ptr(), ptr, 4);
            node.device.unmap_memory(self.uniform_buf.memory);
        }

        let bias_buf = buf_bias.unwrap_or(self.dummy_bias.buffer);

        let buf_a_info = vk::DescriptorBufferInfo { buffer: buf_a, offset: 0, range: vk::WHOLE_SIZE };
        let buf_b_info = vk::DescriptorBufferInfo { buffer: buf_b, offset: 0, range: vk::WHOLE_SIZE };
        let buf_bias_info = vk::DescriptorBufferInfo { buffer: bias_buf, offset: 0, range: vk::WHOLE_SIZE };
        let buf_c_info = vk::DescriptorBufferInfo { buffer: buf_c, offset: 0, range: vk::WHOLE_SIZE };
        let buf_p_info = vk::DescriptorBufferInfo { buffer: self.uniform_buf.buffer, offset: 0, range: 16 };

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_a_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_b_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_bias_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_c_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_p_info)),
        ];

        unsafe {
            node.device.update_descriptor_sets(&writes, &[]);
        }

        let gx = n.div_ceil(16);
        let gy = m.div_ceil(16);

        one_shot_submit(node, |cb| unsafe {
            let mem_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            node.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[mem_barrier],
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
            node.device.cmd_dispatch(cb, gx, gy, 1);

            let mem_barrier_after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ);
            node.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[mem_barrier_after],
                &[],
                &[],
            );
        })
    }

    pub fn destroy(&mut self, node: &GpuNode) {
        destroy_buf(node, &self.dummy_bias);
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

pub(crate) struct SplitKvPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub desc_pool: vk::DescriptorPool,
    pub desc_set: vk::DescriptorSet,
    pub shader_module: vk::ShaderModule,
    pub uniform_buf: BufMem,
}

impl SplitKvPipeline {
    pub fn new(node: &GpuNode) -> Hc4jResult<Self> {
        let mut cursor = std::io::Cursor::new(SPLIT_KV_ATTN_SPV);
        let code = ash::util::read_spv(&mut cursor)
            .map_err(|e| Hc4jError::Device(format!("failed to parse Split-KV SPIR-V bytecode: {e}")))?;

        let module_ci = vk::ShaderModuleCreateInfo::default().code(&code);
        let shader_module = unsafe { node.device.create_shader_module(&module_ci, None) }
            .map_err(|e| Hc4jError::Device(format!("create_shader_module for Split-KV failed: {e}")))?;

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
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_set_layout = unsafe { node.device.create_descriptor_set_layout(&dsl_ci, None) }
            .map_err(|e| {
                unsafe { node.device.destroy_shader_module(shader_module, None) };
                Hc4jError::Device(format!("create_descriptor_set_layout for Split-KV failed: {e}"))
            })?;

        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&desc_set_layout));
        let layout = unsafe { node.device.create_pipeline_layout(&layout_ci, None) }
            .map_err(|e| {
                unsafe {
                    node.device.destroy_descriptor_set_layout(desc_set_layout, None);
                    node.device.destroy_shader_module(shader_module, None);
                }
                Hc4jError::Device(format!("create_pipeline_layout for Split-KV failed: {e}"))
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
            Hc4jError::Device(format!("create_compute_pipelines for Split-KV failed: {e:?}"))
        })?[0];

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 5,
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
                Hc4jError::Device(format!("create_descriptor_pool for Split-KV failed: {e}"))
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
                Hc4jError::Device(format!("allocate_descriptor_sets for Split-KV failed: {e}"))
            })?[0];

        let uniform_buf = alloc_staging(
            node,
            32,
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
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        node: &GpuNode,
        buf_q: vk::Buffer,
        buf_k: vk::Buffer,
        buf_v: vk::Buffer,
        buf_out_o: vk::Buffer,
        buf_out_stats: vk::Buffer,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        local_tokens: u32,
        scale: f32,
    ) -> Hc4jResult<()> {
        if num_heads == 0 || head_dim == 0 {
            return Ok(());
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
        unsafe {
            let ptr = node.device.map_memory(self.uniform_buf.memory, 0, 32, vk::MemoryMapFlags::empty())
                .map_err(|e| Hc4jError::Device(format!("map uniform_buf failed: {e}")))? as *mut u32;
            std::ptr::copy_nonoverlapping(params.as_ptr(), ptr, 8);
            node.device.unmap_memory(self.uniform_buf.memory);
        }

        let buf_q_info = vk::DescriptorBufferInfo { buffer: buf_q, offset: 0, range: vk::WHOLE_SIZE };
        let buf_k_info = vk::DescriptorBufferInfo { buffer: buf_k, offset: 0, range: vk::WHOLE_SIZE };
        let buf_v_info = vk::DescriptorBufferInfo { buffer: buf_v, offset: 0, range: vk::WHOLE_SIZE };
        let buf_o_info = vk::DescriptorBufferInfo { buffer: buf_out_o, offset: 0, range: vk::WHOLE_SIZE };
        let buf_s_info = vk::DescriptorBufferInfo { buffer: buf_out_stats, offset: 0, range: vk::WHOLE_SIZE };
        let buf_p_info = vk::DescriptorBufferInfo { buffer: self.uniform_buf.buffer, offset: 0, range: 32 };

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_q_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_k_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_v_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_o_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_s_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_p_info)),
        ];

        unsafe {
            node.device.update_descriptor_sets(&writes, &[]);
        }

        one_shot_submit(node, |cb| unsafe {
            let mem_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            node.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[mem_barrier],
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
            node.device.cmd_dispatch(cb, num_heads, 1, 1);

            let mem_barrier_after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ);
            node.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[mem_barrier_after],
                &[],
                &[],
            );
        })
    }

    pub fn destroy(&mut self, node: &GpuNode) {
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

pub(crate) struct ElemMulPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub desc_pool: vk::DescriptorPool,
    pub desc_set: vk::DescriptorSet,
    pub shader_module: vk::ShaderModule,
    pub uniform_buf: BufMem,
}

impl ElemMulPipeline {
    pub fn new(node: &GpuNode) -> Hc4jResult<Self> {
        let mut cursor = std::io::Cursor::new(ELEM_MUL_SHADER_SPV);
        let code = ash::util::read_spv(&mut cursor)
            .map_err(|e| Hc4jError::Device(format!("failed to parse elem_mul SPIR-V bytecode: {e}")))?;

        let module_ci = vk::ShaderModuleCreateInfo::default().code(&code);
        let shader_module = unsafe { node.device.create_shader_module(&module_ci, None) }
            .map_err(|e| Hc4jError::Device(format!("create_shader_module for elem_mul failed: {e}")))?;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_set_layout = unsafe { node.device.create_descriptor_set_layout(&dsl_ci, None) }
            .map_err(|e| {
                unsafe { node.device.destroy_shader_module(shader_module, None) };
                Hc4jError::Device(format!("create_descriptor_set_layout for elem_mul failed: {e}"))
            })?;

        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&desc_set_layout));
        let layout = unsafe { node.device.create_pipeline_layout(&layout_ci, None) }
            .map_err(|e| {
                unsafe {
                    node.device.destroy_descriptor_set_layout(desc_set_layout, None);
                    node.device.destroy_shader_module(shader_module, None);
                }
                Hc4jError::Device(format!("create_pipeline_layout for elem_mul failed: {e}"))
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
            Hc4jError::Device(format!("create_compute_pipelines for elem_mul failed: {e:?}"))
        })?[0];

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 3,
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
                Hc4jError::Device(format!("create_descriptor_pool for elem_mul failed: {e}"))
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
                Hc4jError::Device(format!("allocate_descriptor_sets for elem_mul failed: {e}"))
            })?[0];

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
        })
    }

    pub fn dispatch(
        &self,
        node: &GpuNode,
        buf_a: vk::Buffer,
        buf_b: vk::Buffer,
        buf_out: vk::Buffer,
        count: u32,
    ) -> Hc4jResult<()> {
        if count == 0 {
            return Ok(());
        }

        let params: [u32; 4] = [count, 0, 0, 0];
        unsafe {
            let ptr = node.device.map_memory(self.uniform_buf.memory, 0, 16, vk::MemoryMapFlags::empty())
                .map_err(|e| Hc4jError::Device(format!("map uniform_buf failed: {e}")))? as *mut u32;
            std::ptr::copy_nonoverlapping(params.as_ptr(), ptr, 4);
            node.device.unmap_memory(self.uniform_buf.memory);
        }

        let buf_p_info = vk::DescriptorBufferInfo { buffer: self.uniform_buf.buffer, offset: 0, range: 16 };
        let buf_a_info = vk::DescriptorBufferInfo { buffer: buf_a, offset: 0, range: vk::WHOLE_SIZE };
        let buf_b_info = vk::DescriptorBufferInfo { buffer: buf_b, offset: 0, range: vk::WHOLE_SIZE };
        let buf_o_info = vk::DescriptorBufferInfo { buffer: buf_out, offset: 0, range: vk::WHOLE_SIZE };

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_p_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_a_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_b_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_o_info)),
        ];

        unsafe {
            node.device.update_descriptor_sets(&writes, &[]);
        }

        let wgs = count.div_ceil(WG_SIZE).clamp(1, 65_535);
        one_shot_submit(node, |cb| unsafe {
            let mem_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            node.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[mem_barrier],
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
            node.device.cmd_dispatch(cb, wgs, 1, 1);

            let mem_barrier_after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ);
            node.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[mem_barrier_after],
                &[],
                &[],
            );
        })
    }

    pub fn destroy(&mut self, node: &GpuNode) {
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

pub(crate) struct RankComputePipelines {
    pub reduce: ReducePipeline,
    pub gemm: GemmPipeline,
    pub split_kv: SplitKvPipeline,
    pub elem_mul: ElemMulPipeline,
}

impl RankComputePipelines {
    pub fn new(node: &GpuNode) -> Hc4jResult<Self> {
        Ok(Self {
            reduce: ReducePipeline::new(node)?,
            gemm: GemmPipeline::new(node)?,
            split_kv: SplitKvPipeline::new(node)?,
            elem_mul: ElemMulPipeline::new(node)?,
        })
    }

    pub fn destroy(&mut self, node: &GpuNode) {
        self.elem_mul.destroy(node);
        self.split_kv.destroy(node);
        self.gemm.destroy(node);
        self.reduce.destroy(node);
    }
}
