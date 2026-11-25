//! Vulkan 1.2 device enumeration, ranking, logical device creation, and buffer helpers.

use std::ffi::CStr;
use ash::{khr, vk};

use crate::engine::error::{Hc4jError, Hc4jResult};

/// Human-readable label for the physical device type.
pub(crate) fn device_type_str(ty: vk::PhysicalDeviceType) -> &'static str {
    match ty {
        vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual",
        vk::PhysicalDeviceType::CPU => "CPU",
        _ => "Other",
    }
}

pub(crate) fn device_rank(ty: vk::PhysicalDeviceType) -> u32 {
    match ty {
        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::OTHER => 1,
        vk::PhysicalDeviceType::CPU => 0,
        _ => 0,
    }
}

pub(crate) fn device_name(props: &vk::PhysicalDeviceProperties) -> String {
    unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
        .to_str()
        .unwrap_or("Unknown")
        .to_string()
}

/// Runtime extension support discovered from `vkEnumerateDeviceExtensionProperties`.
#[derive(Debug, Clone, Default)]
pub(crate) struct ExtSupport {
    #[cfg(target_os = "linux")]
    pub external_memory_fd: bool,
    #[cfg(target_os = "linux")]
    pub external_memory_dma_buf: bool,
    #[cfg(target_os = "linux")]
    pub external_semaphore_fd: bool,

    #[cfg(target_os = "windows")]
    pub external_memory_win32: bool,
    #[cfg(target_os = "windows")]
    pub external_memory_host: bool,

    pub timeline_semaphore: bool,
}

/// Holds a VkBuffer + its backing VkDeviceMemory and memory allocation size.
#[derive(Clone, Copy)]
pub(crate) struct BufMem {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

/// A logical GPU node in the collective ring.
pub(crate) struct GpuNode {
    #[allow(dead_code)]
    pub idx: usize,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    #[allow(dead_code)]
    pub queue_family: u32,
    pub cmd_pool: vk::CommandPool,
    pub props: vk::PhysicalDeviceProperties,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    pub ext: ExtSupport,

    #[cfg(target_os = "linux")]
    pub fd_fn: Option<khr::external_memory_fd::Device>,
    #[cfg(target_os = "linux")]
    pub sem_fd_fn: Option<khr::external_semaphore_fd::Device>,

    #[cfg(target_os = "windows")]
    pub win32_fn: Option<khr::external_memory_win32::Device>,
    #[cfg(target_os = "windows")]
    pub host_fn: Option<ash::ext::external_memory_host::Device>,
    #[cfg(target_os = "windows")]
    pub min_host_ptr_align: vk::DeviceSize,
}

impl GpuNode {
    pub fn destroy(&self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
        }
    }
}

/// Find the first memory-type index whose flags are a superset of `required`.
pub(crate) fn find_mem_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..mem_props.memory_type_count).find(|&i| {
        let supported = type_bits & (1 << i) != 0;
        let sufficient = mem_props.memory_types[i as usize]
            .property_flags
            .contains(required);
        supported && sufficient
    })
}

pub(crate) fn find_queue_family(instance: &ash::Instance, phd: vk::PhysicalDevice) -> Option<u32> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(phd) };
    families.iter().enumerate().find_map(|(i, f)| {
        let dominated = f
            .queue_flags
            .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER);
        dominated.then_some(i as u32)
    })
}

fn has_ext(exts: &[vk::ExtensionProperties], name: &CStr) -> bool {
    exts.iter().any(|e| {
        let ename = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
        ename == name
    })
}

pub(crate) fn probe_extensions(instance: &ash::Instance, phd: vk::PhysicalDevice) -> ExtSupport {
    let exts = unsafe { instance.enumerate_device_extension_properties(phd) }.unwrap_or_default();

    ExtSupport {
        #[cfg(target_os = "linux")]
        external_memory_fd: has_ext(&exts, khr::external_memory_fd::NAME),
        #[cfg(target_os = "linux")]
        external_memory_dma_buf: has_ext(&exts, c"VK_EXT_external_memory_dma_buf"),
        #[cfg(target_os = "linux")]
        external_semaphore_fd: has_ext(&exts, khr::external_semaphore_fd::NAME),

        #[cfg(target_os = "windows")]
        external_memory_win32: has_ext(&exts, khr::external_memory_win32::NAME),
        #[cfg(target_os = "windows")]
        external_memory_host: has_ext(&exts, ash::ext::external_memory_host::NAME),

        timeline_semaphore: {
            let api = unsafe { instance.get_physical_device_properties(phd) }.api_version;
            vk::api_version_minor(api) >= 2 || has_ext(&exts, khr::timeline_semaphore::NAME)
        },
    }
}

pub(crate) fn create_instance(entry: &ash::Entry) -> Hc4jResult<ash::Instance> {
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"HC4J Vulkan P2P Collective")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"hc4j")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_2);

    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    unsafe { entry.create_instance(&create_info, None) }
        .map_err(|e| Hc4jError::Device(format!("failed to create Vulkan 1.2 instance: {e}")))
}

pub(crate) fn create_device(
    instance: &ash::Instance,
    phd: vk::PhysicalDevice,
    ext: &ExtSupport,
) -> Hc4jResult<(ash::Device, vk::Queue, u32)> {
    let qf = find_queue_family(instance, phd)
        .ok_or_else(|| Hc4jError::Device("no queue family with COMPUTE + TRANSFER".to_string()))?;

    let priorities = [1.0f32];
    let queue_ci = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(qf)
        .queue_priorities(&priorities);

    let mut ext_names: Vec<*const i8> = Vec::new();
    #[cfg(target_os = "linux")]
    {
        if ext.external_memory_fd {
            ext_names.push(khr::external_memory_fd::NAME.as_ptr());
        }
        if ext.external_memory_dma_buf {
            ext_names.push(c"VK_EXT_external_memory_dma_buf".as_ptr());
        }
        if ext.external_semaphore_fd {
            ext_names.push(khr::external_semaphore_fd::NAME.as_ptr());
        }
    }
    #[cfg(target_os = "windows")]
    {
        if ext.external_memory_win32 {
            ext_names.push(khr::external_memory_win32::NAME.as_ptr());
        }
        if ext.external_memory_host {
            ext_names.push(ash::ext::external_memory_host::NAME.as_ptr());
        }
    }

    let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
        .timeline_semaphore(ext.timeline_semaphore);

    let device_ci = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_ci))
        .enabled_extension_names(&ext_names)
        .push_next(&mut features12);

    let device = unsafe { instance.create_device(phd, &device_ci, None) }
        .map_err(|e| Hc4jError::Device(format!("failed to create Vulkan logical device: {e}")))?;

    let queue = unsafe { device.get_device_queue(qf, 0) };

    Ok((device, queue, qf))
}

pub(crate) fn build_node(
    instance: &ash::Instance,
    phd: vk::PhysicalDevice,
    idx: usize,
) -> Hc4jResult<GpuNode> {
    let ext = probe_extensions(instance, phd);
    let (device, queue, qf) = create_device(instance, phd, &ext)?;

    let props = unsafe { instance.get_physical_device_properties(phd) };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(phd) };

    let pool_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(qf)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let cmd_pool = unsafe { device.create_command_pool(&pool_ci, None) }
        .map_err(|e| Hc4jError::Device(format!("failed to create command pool: {e}")))?;

    #[cfg(target_os = "linux")]
    let fd_fn = ext
        .external_memory_fd
        .then(|| khr::external_memory_fd::Device::new(instance, &device));
    #[cfg(target_os = "linux")]
    let sem_fd_fn = ext
        .external_semaphore_fd
        .then(|| khr::external_semaphore_fd::Device::new(instance, &device));

    #[cfg(target_os = "windows")]
    let win32_fn = ext
        .external_memory_win32
        .then(|| khr::external_memory_win32::Device::new(instance, &device));

    #[cfg(target_os = "windows")]
    let (host_fn, min_host_ptr_align) = if ext.external_memory_host {
        let loader = ash::ext::external_memory_host::Device::new(instance, &device);
        let mut host_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut host_props);
        unsafe { instance.get_physical_device_properties2(phd, &mut props2) };
        (Some(loader), host_props.min_imported_host_pointer_alignment)
    } else {
        (None, 0)
    };

    Ok(GpuNode {
        idx,
        physical_device: phd,
        device,
        queue,
        queue_family: qf,
        cmd_pool,
        props,
        mem_props,
        ext,
        #[cfg(target_os = "linux")]
        fd_fn,
        #[cfg(target_os = "linux")]
        sem_fd_fn,
        #[cfg(target_os = "windows")]
        win32_fn,
        #[cfg(target_os = "windows")]
        host_fn,
        #[cfg(target_os = "windows")]
        min_host_ptr_align,
    })
}

/// Enumerate, rank, and filter physical devices.
///
/// If `HC4J_COLLECTIVE_INCLUDE_CPU` is false (default), CPU adapters (e.g. Mesa Lavapipe)
/// are excluded when at least one hardware GPU is present. If true, CPU adapters are retained
/// right after hardware GPUs.
pub(crate) fn enum_physical_devices(instance: &ash::Instance) -> Hc4jResult<Vec<vk::PhysicalDevice>> {
    let phds = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| Hc4jError::Device(format!("failed to enumerate physical devices: {e}")))?;

    if phds.is_empty() {
        return Err(Hc4jError::Device("no Vulkan physical devices found".to_string()));
    }

    let include_cpu = std::env::var("HC4J_COLLECTIVE_INCLUDE_CPU")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let mut ranked: Vec<(vk::PhysicalDevice, vk::PhysicalDeviceProperties, u32)> = phds
        .into_iter()
        .filter_map(|phd| {
            find_queue_family(instance, phd)?;
            let props = unsafe { instance.get_physical_device_properties(phd) };
            let rank = device_rank(props.device_type);
            Some((phd, props, rank))
        })
        .collect();

    if ranked.is_empty() {
        return Err(Hc4jError::Device("no Vulkan devices support compute and transfer".to_string()));
    }

    // Sort by priority: DISCRETE > INTEGRATED > VIRTUAL > OTHER > CPU
    ranked.sort_by_key(|a| std::cmp::Reverse(a.2));

    let has_hardware_gpu = ranked.iter().any(|(_, props, _)| props.device_type != vk::PhysicalDeviceType::CPU);

    let filtered: Vec<vk::PhysicalDevice> = ranked
        .into_iter()
        .filter(|(_, props, _)| {
            if !include_cpu && has_hardware_gpu {
                props.device_type != vk::PhysicalDeviceType::CPU
            } else {
                true
            }
        })
        .map(|(phd, _, _)| phd)
        .collect();

    if filtered.is_empty() {
        return Err(Hc4jError::Device("no suitable Vulkan physical devices after filtering".to_string()));
    }

    Ok(filtered)
}

pub(crate) fn alloc_device_local(
    node: &GpuNode,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Hc4jResult<BufMem> {
    let buf_ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { node.device.create_buffer(&buf_ci, None) }
        .map_err(|e| Hc4jError::Device(format!("create_buffer failed: {e}")))?;
    let req = unsafe { node.device.get_buffer_memory_requirements(buffer) };

    let mti = find_mem_type(&node.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or_else(|| {
            unsafe { node.device.destroy_buffer(buffer, None) };
            Hc4jError::OutOfMemory
        })?;

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mti);
    let memory = unsafe { node.device.allocate_memory(&alloc, None) }.map_err(|e| {
        unsafe { node.device.destroy_buffer(buffer, None) };
        Hc4jError::from_wgpu(wgpu::Error::OutOfMemory { source: Box::new(e) })
    })?;

    if let Err(e) = unsafe { node.device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            node.device.destroy_buffer(buffer, None);
            node.device.free_memory(memory, None);
        }
        return Err(Hc4jError::Device(format!("bind_buffer_memory failed: {e}")));
    }

    Ok(BufMem { buffer, memory, size: req.size })
}

pub(crate) fn alloc_staging(
    node: &GpuNode,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Hc4jResult<BufMem> {
    let buf_ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { node.device.create_buffer(&buf_ci, None) }
        .map_err(|e| Hc4jError::Device(format!("create_buffer (staging) failed: {e}")))?;
    let req = unsafe { node.device.get_buffer_memory_requirements(buffer) };

    let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let mti = find_mem_type(&node.mem_props, req.memory_type_bits, flags).ok_or_else(|| {
        unsafe { node.device.destroy_buffer(buffer, None) };
        Hc4jError::OutOfMemory
    })?;

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mti);
    let memory = unsafe { node.device.allocate_memory(&alloc, None) }.map_err(|e| {
        unsafe { node.device.destroy_buffer(buffer, None) };
        Hc4jError::from_wgpu(wgpu::Error::OutOfMemory { source: Box::new(e) })
    })?;

    if let Err(e) = unsafe { node.device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            node.device.destroy_buffer(buffer, None);
            node.device.free_memory(memory, None);
        }
        return Err(Hc4jError::Device(format!("bind_buffer_memory (staging) failed: {e}")));
    }

    Ok(BufMem { buffer, memory, size: req.size })
}

pub(crate) fn destroy_buf(node: &GpuNode, buf: &BufMem) {
    unsafe {
        node.device.destroy_buffer(buf.buffer, None);
        node.device.free_memory(buf.memory, None);
    }
}

pub(crate) fn one_shot_submit(
    node: &GpuNode,
    record_fn: impl FnOnce(vk::CommandBuffer),
) -> Hc4jResult<()> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(node.cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cb = unsafe { node.device.allocate_command_buffers(&alloc_info) }
        .map_err(|e| Hc4jError::Device(format!("allocate_command_buffers failed: {e}")))? [0];

    let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        node.device.begin_command_buffer(cb, &begin)
            .map_err(|e| Hc4jError::Device(format!("begin_command_buffer failed: {e}")))?;
        record_fn(cb);
        node.device.end_command_buffer(cb)
            .map_err(|e| Hc4jError::Device(format!("end_command_buffer failed: {e}")))?;

        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));
        node.device.queue_submit(node.queue, &[submit], vk::Fence::null())
            .map_err(|e| Hc4jError::Device(format!("queue_submit failed: {e}")))?;
        node.device.queue_wait_idle(node.queue)
            .map_err(|e| Hc4jError::Device(format!("queue_wait_idle failed: {e}")))?;
        node.device.free_command_buffers(node.cmd_pool, &[cb]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_ranks() {
        assert!(device_rank(vk::PhysicalDeviceType::DISCRETE_GPU) > device_rank(vk::PhysicalDeviceType::INTEGRATED_GPU));
        assert!(device_rank(vk::PhysicalDeviceType::INTEGRATED_GPU) > device_rank(vk::PhysicalDeviceType::VIRTUAL_GPU));
        assert!(device_rank(vk::PhysicalDeviceType::VIRTUAL_GPU) > device_rank(vk::PhysicalDeviceType::CPU));
    }
}
