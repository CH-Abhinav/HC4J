//! Memory transport engine: Tier 1 (dma_buf / OPAQUE_WIN32),
//! Tier 1.5 (Windows VK_EXT_external_memory_host), and Tier 2 (staged + Timeline Semaphores).

use ash::vk;

use crate::collective::backend::{
    TRANSPORT_HOST_SHARED_DMA, TRANSPORT_HOST_STAGED, TRANSPORT_PEER_TO_PEER, Transport,
};
use crate::collective::vulkan::device::{BufMem, GpuNode, alloc_staging, destroy_buf, find_mem_type, one_shot_submit};
use crate::error::{Hc4jError, Hc4jResult};

/// Transport-tier classification for a directed GPU→GPU edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum TransportTier {
    /// Tier 1 — True peer-to-peer via native OS memory handle export/import
    /// (dma_buf on Linux, OPAQUE_WIN32 on Windows).
    Tier1P2P { bandwidth_gbps: f64 },

    /// Tier 1.5 — Host-imported pinned system memory (VK_EXT_external_memory_host).
    #[cfg(target_os = "windows")]
    Tier1_5HostMem { bandwidth_gbps: f64 },

    /// Tier 2 — Pinned HOST_VISIBLE staging + Timeline Semaphore orchestration.
    Tier2Staging { bandwidth_gbps: f64 },
}

impl TransportTier {
    #[allow(dead_code)]
    pub fn to_transport(self) -> Transport {
        match self {
            TransportTier::Tier1P2P { .. } => Transport::PeerToPeer,
            #[cfg(target_os = "windows")]
            TransportTier::Tier1_5HostMem { .. } => Transport::HostSharedDma,
            TransportTier::Tier2Staging { .. } => Transport::HostStaged,
        }
    }

    pub fn transport_code(self) -> u32 {
        match self {
            TransportTier::Tier1P2P { .. } => TRANSPORT_PEER_TO_PEER,
            #[cfg(target_os = "windows")]
            TransportTier::Tier1_5HostMem { .. } => TRANSPORT_HOST_SHARED_DMA,
            TransportTier::Tier2Staging { .. } => TRANSPORT_HOST_STAGED,
        }
    }

    #[allow(dead_code)]
    pub fn bandwidth_gbps(self) -> f64 {
        match self {
            TransportTier::Tier1P2P { bandwidth_gbps } => bandwidth_gbps,
            #[cfg(target_os = "windows")]
            TransportTier::Tier1_5HostMem { bandwidth_gbps } => bandwidth_gbps,
            TransportTier::Tier2Staging { bandwidth_gbps } => bandwidth_gbps,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            TransportTier::Tier1P2P { .. } => "PEER_TO_PEER",
            #[cfg(target_os = "windows")]
            TransportTier::Tier1_5HostMem { .. } => "HOST_SHARED_DMA",
            TransportTier::Tier2Staging { .. } => "HOST_STAGED",
        }
    }
}

// ── RAII Resource Guards (Fix #5) ───────────────────────────────────────────

#[cfg(target_os = "windows")]
pub(crate) struct Win32HandleGuard(pub vk::HANDLE);

#[cfg(target_os = "windows")]
impl Drop for Win32HandleGuard {
    fn drop(&mut self) {
        if self.0 != 0 {
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(self.0 as _);
            }
            self.0 = 0;
        }
    }
}

#[cfg(target_os = "windows")]
pub(crate) struct VirtualAllocGuard {
    pub ptr: *mut std::ffi::c_void,
}

#[cfg(target_os = "windows")]
impl Drop for VirtualAllocGuard {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                windows_sys::Win32::System::Memory::VirtualFree(
                    self.ptr,
                    0,
                    windows_sys::Win32::System::Memory::MEM_RELEASE,
                );
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) struct FdGuard(pub i32);

#[cfg(target_os = "linux")]
impl FdGuard {
    pub fn disarm(mut self) {
        self.0 = -1;
    }
}

#[cfg(target_os = "linux")]
impl Drop for FdGuard {
    fn drop(&mut self) {
        if self.0 >= 0 {
            unsafe {
                libc::close(self.0);
            }
            self.0 = -1;
        }
    }
}

// ── Linux dma_buf Transport ─────────────────────────────────────────────────

#[cfg(target_os = "linux")]
pub(crate) fn alloc_exportable_dmabuf(
    node: &GpuNode,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Option<BufMem> {
    if !node.ext.external_memory_fd || !node.ext.external_memory_dma_buf {
        return None;
    }

    let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    let buf_ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut ext_buf);

    let buffer = unsafe { node.device.create_buffer(&buf_ci, None) }.ok()?;
    let req = unsafe { node.device.get_buffer_memory_requirements(buffer) };

    let mti = find_mem_type(&node.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

    let mut export_info = vk::ExportMemoryAllocateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mti)
        .push_next(&mut export_info);

    let memory = unsafe { node.device.allocate_memory(&alloc, None) }.ok()?;
    if unsafe { node.device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        unsafe {
            node.device.destroy_buffer(buffer, None);
            node.device.free_memory(memory, None);
        }
        return None;
    }

    Some(BufMem { buffer, memory, size: req.size })
}

#[cfg(target_os = "linux")]
pub(crate) fn export_dma_buf_fd(node: &GpuNode, memory: vk::DeviceMemory) -> Option<i32> {
    let fd_fn = node.fd_fn.as_ref()?;
    let get_fd_info = vk::MemoryGetFdInfoKHR::default()
        .memory(memory)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    unsafe { fd_fn.get_memory_fd(&get_fd_info) }.ok()
}

#[cfg(target_os = "linux")]
pub(crate) fn import_dma_buf_fd(
    node: &GpuNode,
    size: vk::DeviceSize,
    fd: i32,
    usage: vk::BufferUsageFlags,
) -> Option<BufMem> {
    let guard = FdGuard(fd);

    let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    let buf_ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut ext_buf);

    let buffer = unsafe { node.device.create_buffer(&buf_ci, None) }.ok()?;
    let req = unsafe { node.device.get_buffer_memory_requirements(buffer) };

    let mti = match find_mem_type(&node.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL) {
        Some(i) => i,
        None => {
            unsafe { node.device.destroy_buffer(buffer, None) };
            return None;
        }
    };

    let mut import_fd = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(guard.0);

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mti)
        .push_next(&mut import_fd);

    let memory = match unsafe { node.device.allocate_memory(&alloc, None) } {
        Ok(m) => {
            // Per Vulkan spec, vkAllocateMemory consumes FD ownership only on success
            guard.disarm();
            m
        }
        Err(_) => {
            unsafe { node.device.destroy_buffer(buffer, None) };
            return None;
        }
    };

    if unsafe { node.device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        unsafe {
            node.device.destroy_buffer(buffer, None);
            node.device.free_memory(memory, None);
        }
        return None;
    }

    Some(BufMem { buffer, memory, size: req.size })
}

// ── Windows OPAQUE_WIN32 Transport ──────────────────────────────────────────

#[cfg(target_os = "windows")]
pub(crate) fn alloc_exportable_win32(
    node: &GpuNode,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Option<BufMem> {
    if !node.ext.external_memory_win32 {
        return None;
    }

    let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);

    let buf_ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut ext_buf);

    let buffer = unsafe { node.device.create_buffer(&buf_ci, None) }.ok()?;
    let req = unsafe { node.device.get_buffer_memory_requirements(buffer) };

    let mti = find_mem_type(&node.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

    let mut export_info = vk::ExportMemoryAllocateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mti)
        .push_next(&mut export_info)
        .push_next(&mut dedicated);

    let memory = match unsafe { node.device.allocate_memory(&alloc, None) } {
        Ok(m) => m,
        Err(_) => {
            let alloc_non_dedicated = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mti)
                .push_next(&mut export_info);
            unsafe { node.device.allocate_memory(&alloc_non_dedicated, None) }.ok()?
        }
    };

    if unsafe { node.device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        unsafe {
            node.device.destroy_buffer(buffer, None);
            node.device.free_memory(memory, None);
        }
        return None;
    }

    Some(BufMem { buffer, memory, size: req.size })
}

#[cfg(target_os = "windows")]
pub(crate) fn export_win32_handle(node: &GpuNode, memory: vk::DeviceMemory) -> Option<vk::HANDLE> {
    let win32_fn = node.win32_fn.as_ref()?;
    let get_win32_info = vk::MemoryGetWin32HandleInfoKHR::default()
        .memory(memory)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);

    unsafe { win32_fn.get_memory_win32_handle(&get_win32_info) }.ok()
}

#[cfg(target_os = "windows")]
pub(crate) fn import_win32_handle(
    node: &GpuNode,
    size: vk::DeviceSize,
    handle: vk::HANDLE,
    usage: vk::BufferUsageFlags,
) -> Option<BufMem> {
    let win32_fn = node.win32_fn.as_ref()?;

    let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);

    let buf_ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut ext_buf);

    let buffer = unsafe { node.device.create_buffer(&buf_ci, None) }.ok()?;
    let req = unsafe { node.device.get_buffer_memory_requirements(buffer) };

    let mut handle_props = vk::MemoryWin32HandlePropertiesKHR::default();
    let mti = if unsafe {
        win32_fn.get_memory_win32_handle_properties(
            vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32,
            handle,
            &mut handle_props,
        )
    }.is_ok() {
        let compatible = req.memory_type_bits & handle_props.memory_type_bits;
        find_mem_type(&node.mem_props, compatible, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|| find_mem_type(&node.mem_props, compatible, vk::MemoryPropertyFlags::empty()))
    } else {
        find_mem_type(&node.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }?;

    let mut import_win32 = vk::ImportMemoryWin32HandleInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32)
        .handle(handle);
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mti)
        .push_next(&mut import_win32)
        .push_next(&mut dedicated);

    let memory = match unsafe { node.device.allocate_memory(&alloc, None) } {
        Ok(m) => m,
        Err(_) => {
            let alloc_non_dedicated = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mti)
                .push_next(&mut import_win32);
            unsafe { node.device.allocate_memory(&alloc_non_dedicated, None) }.ok()?
        }
    };

    if unsafe { node.device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        unsafe {
            node.device.destroy_buffer(buffer, None);
            node.device.free_memory(memory, None);
        }
        return None;
    }

    Some(BufMem { buffer, memory, size: req.size })
}

// ── Windows VK_EXT_external_memory_host (Tier 1.5) ──────────────────────────

#[cfg(target_os = "windows")]
pub(crate) fn alloc_host_imported(
    node: &GpuNode,
    host_ptr: *mut std::ffi::c_void,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Option<BufMem> {
    let host_fn = node.host_fn.as_ref()?;

    let mut host_props = vk::MemoryHostPointerPropertiesEXT::default();
    let res = unsafe {
        (host_fn.fp().get_memory_host_pointer_properties_ext)(
            node.device.handle(),
            vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
            host_ptr,
            &mut host_props,
        )
    };
    if res != vk::Result::SUCCESS {
        return None;
    }

    let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);

    let buf_ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut ext_buf);

    let buffer = unsafe { node.device.create_buffer(&buf_ci, None) }.ok()?;
    let req = unsafe { node.device.get_buffer_memory_requirements(buffer) };

    let compatible_bits = req.memory_type_bits & host_props.memory_type_bits;
    if compatible_bits == 0 {
        unsafe { node.device.destroy_buffer(buffer, None) };
        return None;
    }

    let mti = (0..node.mem_props.memory_type_count).find(|&i| compatible_bits & (1 << i) != 0)?;

    let mut import_host = vk::ImportMemoryHostPointerInfoEXT::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
        .host_pointer(host_ptr);

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mti)
        .push_next(&mut import_host);

    let memory = unsafe { node.device.allocate_memory(&alloc, None) }.ok()?;
    if unsafe { node.device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        unsafe {
            node.device.destroy_buffer(buffer, None);
            node.device.free_memory(memory, None);
        }
        return None;
    }

    Some(BufMem { buffer, memory, size: req.size })
}

// ── Timeline Semaphore Creation ─────────────────────────────────────────────

pub(crate) fn create_timeline_semaphore(device: &ash::Device) -> Hc4jResult<vk::Semaphore> {
    let mut type_ci = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);

    let ci = vk::SemaphoreCreateInfo::default().push_next(&mut type_ci);

    unsafe { device.create_semaphore(&ci, None) }
        .map_err(|e| Hc4jError::Device(format!("failed to create timeline semaphore: {e}")))
}

// ── Transfers ───────────────────────────────────────────────────────────────

/// Execute a Tier 1 P2P transfer (Linux dma_buf or Windows OPAQUE_WIN32).
pub(crate) fn transfer_tier1(
    src: &GpuNode,
    dst: &GpuNode,
    src_buf: vk::Buffer,
    src_offset: vk::DeviceSize,
    dst_recv: vk::Buffer,
    dst_offset: vk::DeviceSize,
    size: vk::DeviceSize,
) -> Hc4jResult<()> {
    if size == 0 {
        return Ok(());
    }
    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    #[cfg(target_os = "linux")]
    {
        let export_buf = alloc_exportable_dmabuf(src, size, usage)
            .ok_or_else(|| Hc4jError::Device("Tier 1 export allocation failed".to_string()))?;

        one_shot_submit(src, |cb| {
            let region = vk::BufferCopy { src_offset, dst_offset: 0, size };
            unsafe { src.device.cmd_copy_buffer(cb, src_buf, export_buf.buffer, &[region]) };
        })?;

        let fd = export_dma_buf_fd(src, export_buf.memory)
            .ok_or_else(|| {
                destroy_buf(src, &export_buf);
                Hc4jError::Device("Tier 1 FD export failed".to_string())
            })?;

        let imported = import_dma_buf_fd(dst, size, fd, usage)
            .ok_or_else(|| {
                destroy_buf(src, &export_buf);
                Hc4jError::Device("Tier 1 FD import failed".to_string())
            })?;

        let res = one_shot_submit(dst, |cb| {
            let region = vk::BufferCopy { src_offset: 0, dst_offset, size };
            unsafe { dst.device.cmd_copy_buffer(cb, imported.buffer, dst_recv, &[region]) };
        });

        destroy_buf(dst, &imported);
        destroy_buf(src, &export_buf);
        res
    }

    #[cfg(target_os = "windows")]
    {
        let export_buf = alloc_exportable_win32(src, size, usage)
            .ok_or_else(|| Hc4jError::Device("Tier 1 Win32 export allocation failed".to_string()))?;

        one_shot_submit(src, |cb| {
            let region = vk::BufferCopy { src_offset, dst_offset: 0, size };
            unsafe { src.device.cmd_copy_buffer(cb, src_buf, export_buf.buffer, &[region]) };
        })?;

        let handle = export_win32_handle(src, export_buf.memory)
            .ok_or_else(|| {
                destroy_buf(src, &export_buf);
                Hc4jError::Device("Tier 1 Win32 handle export failed".to_string())
            })?;

        let _guard = Win32HandleGuard(handle);

        let imported = import_win32_handle(dst, size, handle, usage)
            .ok_or_else(|| {
                destroy_buf(src, &export_buf);
                Hc4jError::Device("Tier 1 Win32 handle import failed".to_string())
            })?;

        let res = one_shot_submit(dst, |cb| {
            let region = vk::BufferCopy { src_offset: 0, dst_offset, size };
            unsafe { dst.device.cmd_copy_buffer(cb, imported.buffer, dst_recv, &[region]) };
        });

        destroy_buf(dst, &imported);
        destroy_buf(src, &export_buf);
        res
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        Err(Hc4jError::Unsupported("Tier 1 P2P not supported on this OS"))
    }
}

/// Execute a Tier 1.5 transfer using VK_EXT_external_memory_host (Windows).
#[cfg(target_os = "windows")]
pub(crate) fn transfer_tier1_5_host(
    src: &GpuNode,
    dst: &GpuNode,
    src_buf: vk::Buffer,
    src_offset: vk::DeviceSize,
    dst_recv: vk::Buffer,
    dst_offset: vk::DeviceSize,
    size: vk::DeviceSize,
) -> Hc4jResult<()> {
    if size == 0 {
        return Ok(());
    }
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE,
    };

    let align = src.min_host_ptr_align.max(dst.min_host_ptr_align).max(4096);
    let aligned_size = (size + align - 1) & !(align - 1);

    let host_ptr = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            aligned_size as usize,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        )
    };
    if host_ptr.is_null() {
        return Err(Hc4jError::OutOfMemory);
    }
    let _guard = VirtualAllocGuard { ptr: host_ptr };

    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    let host_buf_src = alloc_host_imported(src, host_ptr, aligned_size, usage)
        .ok_or_else(|| Hc4jError::Device("Tier 1.5 import on source GPU failed".to_string()))?;
    let host_buf_dst = alloc_host_imported(dst, host_ptr, aligned_size, usage)
        .ok_or_else(|| {
            destroy_buf(src, &host_buf_src);
            Hc4jError::Device("Tier 1.5 import on destination GPU failed".to_string())
        })?;

    one_shot_submit(src, |cb| {
        let region = vk::BufferCopy { src_offset, dst_offset: 0, size };
        unsafe { src.device.cmd_copy_buffer(cb, src_buf, host_buf_src.buffer, &[region]) };
    })?;

    let res = one_shot_submit(dst, |cb| {
        let region = vk::BufferCopy { src_offset: 0, dst_offset, size };
        unsafe { dst.device.cmd_copy_buffer(cb, host_buf_dst.buffer, dst_recv, &[region]) };
    });

    destroy_buf(src, &host_buf_src);
    destroy_buf(dst, &host_buf_dst);
    res
}

/// Execute a Tier 2 staged transfer using HOST_VISIBLE staging buffers + Timeline Semaphore.
///
/// Fix #1: Each GpuNode strictly uses its OWN device's staging buffer even on loopback!
pub(crate) fn transfer_tier2(
    src: &GpuNode,
    dst: &GpuNode,
    src_buf: vk::Buffer,
    src_offset: vk::DeviceSize,
    dst_recv: vk::Buffer,
    dst_offset: vk::DeviceSize,
    size: vk::DeviceSize,
) -> Hc4jResult<()> {
    if size == 0 {
        return Ok(());
    }
    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    // Fix #1: Always allocate staging buffer on src.device for src, and staging buffer on dst.device for dst.
    // NEVER pass a buffer from src.device to dst.device!
    let staging_src = alloc_staging(src, size, usage)?;
    let staging_dst = alloc_staging(dst, size, usage)?;

    let tl_sem = create_timeline_semaphore(&src.device)?;

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(src.cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb_a = unsafe { src.device.allocate_command_buffers(&alloc_info) }
        .map_err(|e| Hc4jError::Device(format!("allocate_command_buffers failed: {e}")))? [0];

    let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        src.device.begin_command_buffer(cb_a, &begin)
            .map_err(|e| Hc4jError::Device(format!("begin_command_buffer failed: {e}")))?;
        let region = vk::BufferCopy { src_offset, dst_offset: 0, size };
        src.device.cmd_copy_buffer(cb_a, src_buf, staging_src.buffer, &[region]);
        src.device.end_command_buffer(cb_a)
            .map_err(|e| Hc4jError::Device(format!("end_command_buffer failed: {e}")))?;

        let signal_val: u64 = 1;
        let mut tl_submit_a = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(std::slice::from_ref(&signal_val));
        let submit_a = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&cb_a))
            .signal_semaphores(std::slice::from_ref(&tl_sem))
            .push_next(&mut tl_submit_a);

        src.device.queue_submit(src.queue, &[submit_a], vk::Fence::null())
            .map_err(|e| Hc4jError::Device(format!("queue_submit failed: {e}")))?;

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&tl_sem))
            .values(std::slice::from_ref(&signal_val));
        src.device.wait_semaphores(&wait_info, u64::MAX)
            .map_err(|e| Hc4jError::Device(format!("wait_semaphores failed: {e}")))?;

        let src_ptr = src.device.map_memory(staging_src.memory, 0, staging_src.size, vk::MemoryMapFlags::empty())
            .map_err(|e| Hc4jError::Device(format!("map_memory src failed: {e}")))? as *const u8;
        let dst_ptr = dst.device.map_memory(staging_dst.memory, 0, staging_dst.size, vk::MemoryMapFlags::empty())
            .map_err(|e| Hc4jError::Device(format!("map_memory dst failed: {e}")))? as *mut u8;

        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size as usize);

        src.device.unmap_memory(staging_src.memory);
        dst.device.unmap_memory(staging_dst.memory);
        src.device.free_command_buffers(src.cmd_pool, &[cb_a]);
        src.device.destroy_semaphore(tl_sem, None);
    }

    let res = one_shot_submit(dst, |cb| {
        let region = vk::BufferCopy { src_offset: 0, dst_offset, size };
        unsafe { dst.device.cmd_copy_buffer(cb, staging_dst.buffer, dst_recv, &[region]) };
    });

    destroy_buf(src, &staging_src);
    destroy_buf(dst, &staging_dst);
    res
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn transfer_route(
    src: &GpuNode,
    dst: &GpuNode,
    tier: &TransportTier,
    src_buf: vk::Buffer,
    src_offset: vk::DeviceSize,
    dst_recv: vk::Buffer,
    dst_offset: vk::DeviceSize,
    size: vk::DeviceSize,
) -> Hc4jResult<()> {
    match tier {
        TransportTier::Tier1P2P { .. } => {
            transfer_tier1(src, dst, src_buf, src_offset, dst_recv, dst_offset, size)
        }
        #[cfg(target_os = "windows")]
        TransportTier::Tier1_5HostMem { .. } => {
            transfer_tier1_5_host(src, dst, src_buf, src_offset, dst_recv, dst_offset, size)
        }
        TransportTier::Tier2Staging { .. } => {
            transfer_tier2(src, dst, src_buf, src_offset, dst_recv, dst_offset, size)
        }
    }
}
