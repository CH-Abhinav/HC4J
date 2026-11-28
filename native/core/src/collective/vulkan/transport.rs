//! Memory transport engine: Tier 1 (Linux OPAQUE_FD / dma_buf, Windows OPAQUE_WIN32),
//! Tier 1.5 (VK_EXT_external_memory_host), and Tier 2 (staged + Timeline Semaphores).

use std::collections::HashMap;
use ash::vk;

use crate::collective::backend::{
    TRANSPORT_HOST_SHARED_DMA, TRANSPORT_HOST_STAGED, TRANSPORT_PEER_TO_PEER, Transport,
};
use crate::collective::vulkan::device::{BufMem, GpuNode, alloc_staging, destroy_buf, find_mem_type, one_shot_submit};
use crate::engine::error::{Hc4jError, Hc4jResult};

/// Number of double-buffered transport slots per directed GPU->GPU edge.
pub(crate) const EDGE_PIPELINE_SLOTS: usize = 2;

/// Wrapper allowing raw mapped host/device pointers stored under synchronization
/// mutexes to satisfy `Send + Sync` cleanly across the Vulkan backend.
#[derive(Clone, Copy)]
pub(crate) struct SendMutPtr<T>(pub *mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

/// Native external memory handle mechanism backing a directed link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum P2pHandleKind {
    OpaqueWin32,
    OpaqueFd,
    DmaBuf,
    HostSharedDma,
    HostStaged,
}

impl P2pHandleKind {
    pub fn as_str(self) -> &'static str {
        match self {
            P2pHandleKind::OpaqueWin32 => "OPAQUE_WIN32",
            P2pHandleKind::OpaqueFd => "OPAQUE_FD",
            P2pHandleKind::DmaBuf => "DMA_BUF",
            P2pHandleKind::HostSharedDma => "HOST_SHARED_DMA",
            P2pHandleKind::HostStaged => "HOST_STAGED",
        }
    }

    pub fn from_cached(token: &str, transport: u32) -> Self {
        match token {
            "OPAQUE_WIN32" => P2pHandleKind::OpaqueWin32,
            "OPAQUE_FD" => P2pHandleKind::OpaqueFd,
            "DMA_BUF" => P2pHandleKind::DmaBuf,
            "HOST_SHARED_DMA" => P2pHandleKind::HostSharedDma,
            "HOST_STAGED" => P2pHandleKind::HostStaged,
            _ => match transport {
                TRANSPORT_PEER_TO_PEER => {
                    #[cfg(target_os = "windows")]
                    {
                        P2pHandleKind::OpaqueWin32
                    }
                    #[cfg(not(target_os = "windows"))]
                    {
                        P2pHandleKind::OpaqueFd
                    }
                }
                TRANSPORT_HOST_SHARED_DMA => P2pHandleKind::HostSharedDma,
                _ => P2pHandleKind::HostStaged,
            },
        }
    }

    #[cfg(target_os = "linux")]
    pub fn to_vk_fd_handle_type(self) -> vk::ExternalMemoryHandleTypeFlags {
        match self {
            P2pHandleKind::OpaqueFd => vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD,
            _ => vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
        }
    }
}

/// Transport-tier classification for a directed GPU→GPU edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum TransportTier {
    /// Tier 1 — True peer-to-peer via native OS memory handle export/import
    /// (OPAQUE_FD / dma_buf on Linux, OPAQUE_WIN32 on Windows).
    Tier1P2P {
        bandwidth_gbps: f64,
        handle_kind: P2pHandleKind,
    },

    /// Tier 1.5 — Host-imported pinned system memory (VK_EXT_external_memory_host).
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    Tier1_5HostMem { bandwidth_gbps: f64 },

    /// Tier 2 — Pinned HOST_VISIBLE staging + Timeline Semaphore orchestration.
    Tier2Staging { bandwidth_gbps: f64 },
}

impl TransportTier {
    #[allow(dead_code)]
    pub fn to_transport(self) -> Transport {
        match self {
            TransportTier::Tier1P2P { .. } => Transport::PeerToPeer,
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            TransportTier::Tier1_5HostMem { .. } => Transport::HostSharedDma,
            TransportTier::Tier2Staging { .. } => Transport::HostStaged,
        }
    }

    pub fn transport_code(self) -> u32 {
        match self {
            TransportTier::Tier1P2P { .. } => TRANSPORT_PEER_TO_PEER,
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            TransportTier::Tier1_5HostMem { .. } => TRANSPORT_HOST_SHARED_DMA,
            TransportTier::Tier2Staging { .. } => TRANSPORT_HOST_STAGED,
        }
    }

    #[allow(dead_code)]
    pub fn bandwidth_gbps(self) -> f64 {
        match self {
            TransportTier::Tier1P2P { bandwidth_gbps, .. } => bandwidth_gbps,
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            TransportTier::Tier1_5HostMem { bandwidth_gbps } => bandwidth_gbps,
            TransportTier::Tier2Staging { bandwidth_gbps } => bandwidth_gbps,
        }
    }

    pub fn handle_kind(self) -> P2pHandleKind {
        match self {
            TransportTier::Tier1P2P { handle_kind, .. } => handle_kind,
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            TransportTier::Tier1_5HostMem { .. } => P2pHandleKind::HostSharedDma,
            TransportTier::Tier2Staging { .. } => P2pHandleKind::HostStaged,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            TransportTier::Tier1P2P { .. } => "PEER_TO_PEER",
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            TransportTier::Tier1_5HostMem { .. } => "HOST_SHARED_DMA",
            TransportTier::Tier2Staging { .. } => "HOST_STAGED",
        }
    }
}

// ── RAII Resource Guards (Fix #5) ───────────────────────────────────────────

pub(crate) struct ScopedBuf<'a> {
    node: &'a GpuNode,
    buf: Option<BufMem>,
}

impl<'a> ScopedBuf<'a> {
    pub fn new(node: &'a GpuNode, buf: BufMem) -> Self {
        Self {
            node,
            buf: Some(buf),
        }
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buf.as_ref().unwrap().buffer
    }

    pub fn memory(&self) -> vk::DeviceMemory {
        self.buf.as_ref().unwrap().memory
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.buf.as_ref().unwrap().size
    }
}

impl<'a> Drop for ScopedBuf<'a> {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            destroy_buf(self.node, &buf);
        }
    }
}

pub(crate) struct ScopedSemaphore<'a> {
    node: &'a GpuNode,
    sem: vk::Semaphore,
}

impl<'a> ScopedSemaphore<'a> {
    pub fn new(node: &'a GpuNode, sem: vk::Semaphore) -> Self {
        Self { node, sem }
    }

    pub fn handle(&self) -> vk::Semaphore {
        self.sem
    }
}

impl<'a> Drop for ScopedSemaphore<'a> {
    fn drop(&mut self) {
        if self.sem != vk::Semaphore::null() {
            unsafe {
                self.node.device.destroy_semaphore(self.sem, None);
            }
            self.sem = vk::Semaphore::null();
        }
    }
}

pub(crate) struct ScopedCmdBuf<'a> {
    node: &'a GpuNode,
    cmd_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
}

impl<'a> ScopedCmdBuf<'a> {
    pub fn new(node: &'a GpuNode, cmd_pool: vk::CommandPool, cmd_buf: vk::CommandBuffer) -> Self {
        Self { node, cmd_pool, cmd_buf }
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.cmd_buf
    }
}

impl<'a> Drop for ScopedCmdBuf<'a> {
    fn drop(&mut self) {
        if self.cmd_buf != vk::CommandBuffer::null() {
            unsafe {
                self.node.device.free_command_buffers(self.cmd_pool, &[self.cmd_buf]);
            }
            self.cmd_buf = vk::CommandBuffer::null();
        }
    }
}

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
unsafe impl Send for VirtualAllocGuard {}
#[cfg(target_os = "windows")]
unsafe impl Sync for VirtualAllocGuard {}

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

#[cfg(target_os = "linux")]
pub(crate) struct LinuxHostAllocGuard {
    pub ptr: *mut std::ffi::c_void,
}

#[cfg(target_os = "linux")]
unsafe impl Send for LinuxHostAllocGuard {}
#[cfg(target_os = "linux")]
unsafe impl Sync for LinuxHostAllocGuard {}

#[cfg(target_os = "linux")]
impl Drop for LinuxHostAllocGuard {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                libc::free(self.ptr);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn alloc_aligned_host(
    size: usize,
    align: usize,
) -> Option<(*mut std::ffi::c_void, LinuxHostAllocGuard)> {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let aligned = align.max(std::mem::size_of::<*mut std::ffi::c_void>());
    let rc = unsafe { libc::posix_memalign(&mut ptr, aligned, size.max(1)) };
    if rc != 0 || ptr.is_null() {
        return None;
    }
    unsafe { std::ptr::write_bytes(ptr, 0, size.max(1)) };
    Some((ptr, LinuxHostAllocGuard { ptr }))
}

// ── Linux External Memory FD Transport (OPAQUE_FD & dma_buf) ────────────────

#[cfg(target_os = "linux")]
pub(crate) fn alloc_exportable_fd(
    node: &GpuNode,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    handle_type: vk::ExternalMemoryHandleTypeFlags,
) -> Option<BufMem> {
    if !node.ext.external_memory_fd {
        return None;
    }
    if handle_type == vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT && !node.ext.external_memory_dma_buf {
        return None;
    }

    let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);

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

    let mut export_info = vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);

    let alloc_dedicated = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mti)
        .push_next(&mut export_info)
        .push_next(&mut dedicated);

    let memory = match unsafe { node.device.allocate_memory(&alloc_dedicated, None) } {
        Ok(m) => m,
        Err(_) => {
            let mut export_fallback = vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
            let alloc_non_dedicated = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mti)
                .push_next(&mut export_fallback);
            match unsafe { node.device.allocate_memory(&alloc_non_dedicated, None) } {
                Ok(m) => m,
                Err(_) => {
                    unsafe { node.device.destroy_buffer(buffer, None) };
                    return None;
                }
            }
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

#[cfg(target_os = "linux")]
pub(crate) fn export_memory_fd(
    node: &GpuNode,
    memory: vk::DeviceMemory,
    handle_type: vk::ExternalMemoryHandleTypeFlags,
) -> Option<i32> {
    let fd_fn = node.fd_fn.as_ref()?;
    let get_fd_info = vk::MemoryGetFdInfoKHR::default()
        .memory(memory)
        .handle_type(handle_type);

    unsafe { fd_fn.get_memory_fd(&get_fd_info) }.ok()
}

#[cfg(target_os = "linux")]
pub(crate) fn import_memory_fd(
    node: &GpuNode,
    size: vk::DeviceSize,
    fd: i32,
    usage: vk::BufferUsageFlags,
    handle_type: vk::ExternalMemoryHandleTypeFlags,
) -> Option<BufMem> {
    let guard = FdGuard(fd);
    if !node.ext.external_memory_fd {
        return None;
    }
    if handle_type == vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT && !node.ext.external_memory_dma_buf {
        return None;
    }

    let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);

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
        .handle_type(handle_type)
        .fd(guard.0);
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);

    let alloc_dedicated = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mti)
        .push_next(&mut import_fd)
        .push_next(&mut dedicated);

    let memory = match unsafe { node.device.allocate_memory(&alloc_dedicated, None) } {
        Ok(m) => {
            // Per Vulkan spec, vkAllocateMemory consumes FD ownership only on success
            guard.disarm();
            m
        }
        Err(_) => {
            let mut import_fallback = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(handle_type)
                .fd(guard.0);
            let alloc_non_dedicated = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mti)
                .push_next(&mut import_fallback);
            match unsafe { node.device.allocate_memory(&alloc_non_dedicated, None) } {
                Ok(m) => {
                    guard.disarm();
                    m
                }
                Err(_) => {
                    unsafe { node.device.destroy_buffer(buffer, None) };
                    return None;
                }
            }
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

// ── VK_EXT_external_memory_host (Tier 1.5) ───────────────────────────────────

#[cfg(any(target_os = "windows", target_os = "linux"))]
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

/// Execute a Tier 1 P2P transfer (Linux OPAQUE_FD / dma_buf or Windows OPAQUE_WIN32).
#[allow(clippy::too_many_arguments)]
pub(crate) fn transfer_tier1(
    src: &GpuNode,
    dst: &GpuNode,
    handle_kind: P2pHandleKind,
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
        let vk_handle_ty = handle_kind.to_vk_fd_handle_type();
        let export_buf = ScopedBuf::new(
            src,
            alloc_exportable_fd(src, size, usage, vk_handle_ty)
                .ok_or_else(|| Hc4jError::Device("Tier 1 FD export allocation failed".to_string()))?,
        );

        one_shot_submit(src, |cb| {
            let region = vk::BufferCopy { src_offset, dst_offset: 0, size };
            unsafe { src.device.cmd_copy_buffer(cb, src_buf, export_buf.buffer(), &[region]) };
        })?;

        let fd = export_memory_fd(src, export_buf.memory(), vk_handle_ty)
            .ok_or_else(|| Hc4jError::Device("Tier 1 FD export failed".to_string()))?;

        let imported = ScopedBuf::new(
            dst,
            import_memory_fd(dst, size, fd, usage, vk_handle_ty)
                .ok_or_else(|| Hc4jError::Device("Tier 1 FD import failed".to_string()))?,
        );

        one_shot_submit(dst, |cb| {
            let region = vk::BufferCopy { src_offset: 0, dst_offset, size };
            unsafe { dst.device.cmd_copy_buffer(cb, imported.buffer(), dst_recv, &[region]) };
        })
    }

    #[cfg(target_os = "windows")]
    {
        let _ = handle_kind;
        let export_buf = ScopedBuf::new(
            src,
            alloc_exportable_win32(src, size, usage)
                .ok_or_else(|| Hc4jError::Device("Tier 1 Win32 export allocation failed".to_string()))?,
        );

        one_shot_submit(src, |cb| {
            let region = vk::BufferCopy { src_offset, dst_offset: 0, size };
            unsafe { src.device.cmd_copy_buffer(cb, src_buf, export_buf.buffer(), &[region]) };
        })?;

        let handle = export_win32_handle(src, export_buf.memory())
            .ok_or_else(|| Hc4jError::Device("Tier 1 Win32 handle export failed".to_string()))?;

        let _guard = Win32HandleGuard(handle);

        let imported = ScopedBuf::new(
            dst,
            import_win32_handle(dst, size, handle, usage)
                .ok_or_else(|| Hc4jError::Device("Tier 1 Win32 handle import failed".to_string()))?,
        );

        one_shot_submit(dst, |cb| {
            let region = vk::BufferCopy { src_offset: 0, dst_offset, size };
            unsafe { dst.device.cmd_copy_buffer(cb, imported.buffer(), dst_recv, &[region]) };
        })
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        Err(Hc4jError::Unsupported("Tier 1 P2P not supported on this OS"))
    }
}

/// Execute a Tier 1.5 transfer using VK_EXT_external_memory_host (Windows & Linux).
#[cfg(any(target_os = "windows", target_os = "linux"))]
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

    let align = src.min_host_ptr_align.max(dst.min_host_ptr_align).max(4096);
    let aligned_size = (size + align - 1) & !(align - 1);

    #[cfg(target_os = "windows")]
    let (host_ptr, _guard) = {
        use windows_sys::Win32::System::Memory::{
            VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE,
        };
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                aligned_size as usize,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        if ptr.is_null() {
            return Err(Hc4jError::OutOfMemory);
        }
        (ptr, VirtualAllocGuard { ptr })
    };

    #[cfg(target_os = "linux")]
    let (host_ptr, _guard) = alloc_aligned_host(aligned_size as usize, align as usize)
        .ok_or(Hc4jError::OutOfMemory)?;

    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

    let host_buf_src = ScopedBuf::new(
        src,
        alloc_host_imported(src, host_ptr, aligned_size, usage)
            .ok_or_else(|| Hc4jError::Device("Tier 1.5 import on source GPU failed".to_string()))?,
    );
    let host_buf_dst = ScopedBuf::new(
        dst,
        alloc_host_imported(dst, host_ptr, aligned_size, usage)
            .ok_or_else(|| Hc4jError::Device("Tier 1.5 import on destination GPU failed".to_string()))?,
    );

    one_shot_submit(src, |cb| {
        let region = vk::BufferCopy { src_offset, dst_offset: 0, size };
        unsafe { src.device.cmd_copy_buffer(cb, src_buf, host_buf_src.buffer(), &[region]) };
    })?;

    one_shot_submit(dst, |cb| {
        let region = vk::BufferCopy { src_offset: 0, dst_offset, size };
        unsafe { dst.device.cmd_copy_buffer(cb, host_buf_dst.buffer(), dst_recv, &[region]) };
    })
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
    let staging_src = ScopedBuf::new(src, alloc_staging(src, size, usage)?);
    let staging_dst = ScopedBuf::new(dst, alloc_staging(dst, size, usage)?);

    let tl_sem = ScopedSemaphore::new(src, create_timeline_semaphore(&src.device)?);

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(src.cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb_a_raw = unsafe { src.device.allocate_command_buffers(&alloc_info) }
        .map_err(|e| Hc4jError::Device(format!("allocate_command_buffers failed: {e}")))? [0];
    let cb_a = ScopedCmdBuf::new(src, src.cmd_pool, cb_a_raw);

    let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        src.device.begin_command_buffer(cb_a.handle(), &begin)
            .map_err(|e| Hc4jError::Device(format!("begin_command_buffer failed: {e}")))?;
        let region = vk::BufferCopy { src_offset, dst_offset: 0, size };
        src.device.cmd_copy_buffer(cb_a.handle(), src_buf, staging_src.buffer(), &[region]);
        src.device.end_command_buffer(cb_a.handle())
            .map_err(|e| Hc4jError::Device(format!("end_command_buffer failed: {e}")))?;

        let signal_val: u64 = 1;
        let cb_handle = cb_a.handle();
        let sem_handle = tl_sem.handle();
        let mut tl_submit_a = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(std::slice::from_ref(&signal_val));
        let submit_a = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&cb_handle))
            .signal_semaphores(std::slice::from_ref(&sem_handle))
            .push_next(&mut tl_submit_a);

        src.device.queue_submit(src.queue, &[submit_a], vk::Fence::null())
            .map_err(|e| Hc4jError::Device(format!("queue_submit failed: {e}")))?;

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&sem_handle))
            .values(std::slice::from_ref(&signal_val));
        src.device.wait_semaphores(&wait_info, u64::MAX)
            .map_err(|e| Hc4jError::Device(format!("wait_semaphores failed: {e}")))?;

        let src_ptr = src.device.map_memory(staging_src.memory(), 0, staging_src.size(), vk::MemoryMapFlags::empty())
            .map_err(|e| Hc4jError::Device(format!("map_memory src failed: {e}")))? as *const u8;
        let dst_ptr = dst.device.map_memory(staging_dst.memory(), 0, staging_dst.size(), vk::MemoryMapFlags::empty())
            .map_err(|e| {
                src.device.unmap_memory(staging_src.memory());
                Hc4jError::Device(format!("map_memory dst failed: {e}"))
            })? as *mut u8;

        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size as usize);

        src.device.unmap_memory(staging_src.memory());
        dst.device.unmap_memory(staging_dst.memory());
    }

    one_shot_submit(dst, |cb| {
        let region = vk::BufferCopy { src_offset: 0, dst_offset, size };
        unsafe { dst.device.cmd_copy_buffer(cb, staging_dst.buffer(), dst_recv, &[region]) };
    })
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
        TransportTier::Tier1P2P { handle_kind, .. } => {
            transfer_tier1(src, dst, *handle_kind, src_buf, src_offset, dst_recv, dst_offset, size)
        }
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        TransportTier::Tier1_5HostMem { .. } => {
            transfer_tier1_5_host(src, dst, src_buf, src_offset, dst_recv, dst_offset, size)
        }
        TransportTier::Tier2Staging { .. } => {
            transfer_tier2(src, dst, src_buf, src_offset, dst_recv, dst_offset, size)
        }
    }
}

// ── Persistent Per-Edge Double-Buffered Channels (Fix 1) ────────────────────

pub(crate) enum EdgeSlotBacking {
    Tier1 {
        export_buf: BufMem,
        imported_buf: BufMem,
        #[cfg(target_os = "windows")]
        _win32_guard: Win32HandleGuard,
    },
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    Tier1_5Host {
        host_buf_src: BufMem,
        host_buf_dst: BufMem,
        #[cfg(target_os = "windows")]
        _host_guard: VirtualAllocGuard,
        #[cfg(target_os = "linux")]
        _host_guard: LinuxHostAllocGuard,
    },
    Tier2Staged {
        staging_src: BufMem,
        staging_dst: BufMem,
        src_ptr: SendMutPtr<u8>,
        dst_ptr: SendMutPtr<u8>,
    },
}

impl EdgeSlotBacking {
    fn allocate(
        src: &GpuNode,
        dst: &GpuNode,
        tier: TransportTier,
        slot_bytes: vk::DeviceSize,
    ) -> Hc4jResult<Self> {
        let size = slot_bytes.max(4);
        let xfer_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        match tier {
            TransportTier::Tier1P2P { handle_kind, .. } => {
                #[cfg(target_os = "linux")]
                {
                    let vk_handle_ty = handle_kind.to_vk_fd_handle_type();
                    let export_buf = alloc_exportable_fd(src, size, xfer_usage, vk_handle_ty)
                        .ok_or_else(|| Hc4jError::Device("Persistent Tier 1 FD export alloc failed".to_string()))?;
                    let fd = match export_memory_fd(src, export_buf.memory, vk_handle_ty) {
                        Some(fd) => fd,
                        None => {
                            destroy_buf(src, &export_buf);
                            return Err(Hc4jError::Device("Persistent Tier 1 FD export failed".to_string()));
                        }
                    };
                    let imported_buf = match import_memory_fd(dst, size, fd, xfer_usage, vk_handle_ty) {
                        Some(b) => b,
                        None => {
                            destroy_buf(src, &export_buf);
                            return Err(Hc4jError::Device("Persistent Tier 1 FD import failed".to_string()));
                        }
                    };
                    Ok(EdgeSlotBacking::Tier1 { export_buf, imported_buf })
                }
                #[cfg(target_os = "windows")]
                {
                    let _ = handle_kind;
                    let export_buf = alloc_exportable_win32(src, size, xfer_usage)
                        .ok_or_else(|| Hc4jError::Device("Persistent Tier 1 Win32 export alloc failed".to_string()))?;
                    let handle = match export_win32_handle(src, export_buf.memory) {
                        Some(h) => h,
                        None => {
                            destroy_buf(src, &export_buf);
                            return Err(Hc4jError::Device("Persistent Tier 1 Win32 handle export failed".to_string()));
                        }
                    };
                    let guard = Win32HandleGuard(handle);
                    let imported_buf = match import_win32_handle(dst, size, handle, xfer_usage) {
                        Some(b) => b,
                        None => {
                            drop(guard);
                            destroy_buf(src, &export_buf);
                            return Err(Hc4jError::Device("Persistent Tier 1 Win32 handle import failed".to_string()));
                        }
                    };
                    Ok(EdgeSlotBacking::Tier1 {
                        export_buf,
                        imported_buf,
                        _win32_guard: guard,
                    })
                }
                #[cfg(not(any(target_os = "windows", target_os = "linux")))]
                {
                    let _ = (src, dst, handle_kind, size, xfer_usage);
                    Err(Hc4jError::Unsupported("Tier 1 P2P not supported on this OS"))
                }
            }
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            TransportTier::Tier1_5HostMem { .. } => {
                let align = src.min_host_ptr_align.max(dst.min_host_ptr_align).max(4096);
                let aligned_size = (size + align - 1) & !(align - 1);

                #[cfg(target_os = "windows")]
                let (host_ptr, host_guard) = {
                    use windows_sys::Win32::System::Memory::{
                        VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE,
                    };
                    let ptr = unsafe {
                        VirtualAlloc(
                            std::ptr::null(),
                            aligned_size as usize,
                            MEM_COMMIT | MEM_RESERVE,
                            PAGE_READWRITE,
                        )
                    };
                    if ptr.is_null() {
                        return Err(Hc4jError::OutOfMemory);
                    }
                    (ptr, VirtualAllocGuard { ptr })
                };

                #[cfg(target_os = "linux")]
                let (host_ptr, host_guard) = alloc_aligned_host(aligned_size as usize, align as usize)
                    .ok_or(Hc4jError::OutOfMemory)?;

                let host_buf_src = alloc_host_imported(src, host_ptr, aligned_size, xfer_usage)
                    .ok_or_else(|| Hc4jError::Device("Persistent Tier 1.5 import on src failed".to_string()))?;
                let dst_usage_storage = xfer_usage | vk::BufferUsageFlags::STORAGE_BUFFER;
                let host_buf_dst = alloc_host_imported(dst, host_ptr, aligned_size, dst_usage_storage)
                    .or_else(|| alloc_host_imported(dst, host_ptr, aligned_size, xfer_usage))
                    .ok_or_else(|| {
                        destroy_buf(src, &host_buf_src);
                        Hc4jError::Device("Persistent Tier 1.5 import on dst failed".to_string())
                    })?;

                Ok(EdgeSlotBacking::Tier1_5Host {
                    host_buf_src,
                    host_buf_dst,
                    _host_guard: host_guard,
                })
            }
            TransportTier::Tier2Staging { .. } => {
                let staging_src = alloc_staging(src, size, xfer_usage)?;
                let staging_dst = match alloc_staging(dst, size, xfer_usage) {
                    Ok(b) => b,
                    Err(e) => {
                        destroy_buf(src, &staging_src);
                        return Err(e);
                    }
                };
                let src_ptr = match unsafe {
                    src.device.map_memory(staging_src.memory, 0, staging_src.size, vk::MemoryMapFlags::empty())
                } {
                    Ok(p) => SendMutPtr(p as *mut u8),
                    Err(e) => {
                        destroy_buf(dst, &staging_dst);
                        destroy_buf(src, &staging_src);
                        return Err(Hc4jError::Device(format!("persistent map_memory src failed: {e}")));
                    }
                };
                let dst_ptr = match unsafe {
                    dst.device.map_memory(staging_dst.memory, 0, staging_dst.size, vk::MemoryMapFlags::empty())
                } {
                    Ok(p) => SendMutPtr(p as *mut u8),
                    Err(e) => {
                        unsafe { src.device.unmap_memory(staging_src.memory) };
                        destroy_buf(dst, &staging_dst);
                        destroy_buf(src, &staging_src);
                        return Err(Hc4jError::Device(format!("persistent map_memory dst failed: {e}")));
                    }
                };

                Ok(EdgeSlotBacking::Tier2Staged {
                    staging_src,
                    staging_dst,
                    src_ptr,
                    dst_ptr,
                })
            }
        }
    }

    fn src_buffer(&self) -> vk::Buffer {
        match self {
            EdgeSlotBacking::Tier1 { export_buf, .. } => export_buf.buffer,
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            EdgeSlotBacking::Tier1_5Host { host_buf_src, .. } => host_buf_src.buffer,
            EdgeSlotBacking::Tier2Staged { staging_src, .. } => staging_src.buffer,
        }
    }

    fn dst_buffer(&self) -> vk::Buffer {
        match self {
            EdgeSlotBacking::Tier1 { imported_buf, .. } => imported_buf.buffer,
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            EdgeSlotBacking::Tier1_5Host { host_buf_dst, .. } => host_buf_dst.buffer,
            EdgeSlotBacking::Tier2Staged { staging_dst, .. } => staging_dst.buffer,
        }
    }

    #[inline]
    fn copy_host_if_staged(&self, bytes: vk::DeviceSize) {
        if bytes == 0 {
            return;
        }
        if let EdgeSlotBacking::Tier2Staged { src_ptr, dst_ptr, .. } = self {
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr.0, dst_ptr.0, bytes as usize);
            }
        }
    }

    fn destroy(self, src: &GpuNode, dst: &GpuNode) {
        match self {
            EdgeSlotBacking::Tier1 { export_buf, imported_buf, .. } => {
                destroy_buf(dst, &imported_buf);
                destroy_buf(src, &export_buf);
            }
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            EdgeSlotBacking::Tier1_5Host { host_buf_src, host_buf_dst, _host_guard } => {
                destroy_buf(dst, &host_buf_dst);
                destroy_buf(src, &host_buf_src);
                drop(_host_guard);
            }
            EdgeSlotBacking::Tier2Staged { staging_src, staging_dst, .. } => {
                unsafe {
                    src.device.unmap_memory(staging_src.memory);
                    dst.device.unmap_memory(staging_dst.memory);
                }
                destroy_buf(dst, &staging_dst);
                destroy_buf(src, &staging_src);
            }
        }
    }
}

pub(crate) struct EdgeSlot {
    pub capacity: vk::DeviceSize,
    backing: Option<EdgeSlotBacking>,
}

impl EdgeSlot {
    fn new(src: &GpuNode, dst: &GpuNode, tier: TransportTier, capacity: vk::DeviceSize) -> Hc4jResult<Self> {
        let backing = EdgeSlotBacking::allocate(src, dst, tier, capacity)?;
        Ok(Self {
            capacity,
            backing: Some(backing),
        })
    }

    pub fn src_buffer(&self) -> vk::Buffer {
        self.backing.as_ref().expect("EdgeSlot destroyed").src_buffer()
    }

    pub fn dst_buffer(&self) -> vk::Buffer {
        self.backing.as_ref().expect("EdgeSlot destroyed").dst_buffer()
    }

    #[inline]
    pub fn copy_host_if_staged(&self, bytes: vk::DeviceSize) {
        if let Some(b) = &self.backing {
            b.copy_host_if_staged(bytes);
        }
    }

    fn destroy(&mut self, src: &GpuNode, dst: &GpuNode) {
        self.capacity = 0;
        if let Some(b) = self.backing.take() {
            b.destroy(src, dst);
        }
    }
}

pub(crate) struct EdgeChannel {
    pub tier: TransportTier,
    slots: Vec<EdgeSlot>,
}

impl EdgeChannel {
    fn create(
        src: &GpuNode,
        dst: &GpuNode,
        tier: TransportTier,
        slot_bytes: vk::DeviceSize,
    ) -> Hc4jResult<Self> {
        let cap = slot_bytes.max(4);
        let mut slots = Vec::with_capacity(EDGE_PIPELINE_SLOTS);
        for _ in 0..EDGE_PIPELINE_SLOTS {
            match EdgeSlot::new(src, dst, tier, cap) {
                Ok(slot) => slots.push(slot),
                Err(e) => {
                    for s in &mut slots {
                        s.destroy(src, dst);
                    }
                    return Err(e);
                }
            }
        }
        Ok(Self { tier, slots })
    }

    fn ensure_capacity(
        &mut self,
        src: &GpuNode,
        dst: &GpuNode,
        needed_bytes: vk::DeviceSize,
    ) -> Hc4jResult<()> {
        let req = needed_bytes.max(4);
        let needs_grow = self.slots.iter().any(|s| s.capacity < req);
        if !needs_grow {
            return Ok(());
        }
        unsafe {
            src.device.device_wait_idle().ok();
            dst.device.device_wait_idle().ok();
        }
        for slot in &mut self.slots {
            if slot.capacity < req {
                slot.destroy(src, dst);
                *slot = EdgeSlot::new(src, dst, self.tier, req)?;
            }
        }
        Ok(())
    }

    #[inline]
    pub fn slot(&self, idx: usize) -> &EdgeSlot {
        &self.slots[idx % EDGE_PIPELINE_SLOTS]
    }

    fn destroy(&mut self, src: &GpuNode, dst: &GpuNode) {
        for slot in &mut self.slots {
            slot.destroy(src, dst);
        }
        self.slots.clear();
    }
}

/// Cache of persistent double-buffered directed edge channels `(src_rank, dst_rank)`.
#[derive(Default)]
pub(crate) struct EdgeChannelCache {
    edges: HashMap<(usize, usize), EdgeChannel>,
}

impl EdgeChannelCache {
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    pub fn ensure_edge(
        &mut self,
        src: &GpuNode,
        dst: &GpuNode,
        src_idx: usize,
        dst_idx: usize,
        tier: TransportTier,
        slot_bytes: vk::DeviceSize,
    ) -> Hc4jResult<&EdgeChannel> {
        let key = (src_idx, dst_idx);
        if let Some(existing) = self.edges.get_mut(&key) {
            if existing.tier.transport_code() == tier.transport_code() {
                existing.ensure_capacity(src, dst, slot_bytes)?;
                return Ok(self.edges.get(&key).unwrap());
            }
            unsafe {
                src.device.device_wait_idle().ok();
                dst.device.device_wait_idle().ok();
            }
            if let Some(mut old) = self.edges.remove(&key) {
                old.destroy(src, dst);
            }
        }
        let ch = EdgeChannel::create(src, dst, tier, slot_bytes)?;
        Ok(self.edges.entry(key).or_insert(ch))
    }

    #[inline]
    pub fn edge(&self, src_idx: usize, dst_idx: usize) -> &EdgeChannel {
        self.edges
            .get(&(src_idx, dst_idx))
            .expect("BUG: EdgeChannel not ensured before use")
    }

    pub fn destroy_all(&mut self, nodes: &[GpuNode]) {
        for ((src_idx, dst_idx), mut channel) in self.edges.drain() {
            channel.destroy(&nodes[src_idx], &nodes[dst_idx]);
        }
    }
}
