use crate::engine::error::Hc4jResult;

// ============================================================================
// Adapter selection & ranking
// ============================================================================

/// Backends to consider. `HC4J_BACKEND=dx12|vulkan|metal` narrows the set;
/// the default on Windows is DX12 + Vulkan.
pub(crate) fn requested_backends() -> wgpu::Backends {
    match std::env::var("HC4J_BACKEND").map(|v| v.to_ascii_lowercase()).as_deref() {
        Ok("dx12") => wgpu::Backends::DX12,
        Ok("vulkan") => wgpu::Backends::VULKAN,
        Ok("metal") => wgpu::Backends::METAL,
        Ok("gl") => wgpu::Backends::GL,
        _ => wgpu::Backends::from_env()
            .unwrap_or(wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL),
    }
}

pub(crate) fn device_rank(kind: wgpu::DeviceType) -> u8 {
    match kind {
        wgpu::DeviceType::DiscreteGpu => 4,
        wgpu::DeviceType::IntegratedGpu => 3,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 1,
        wgpu::DeviceType::Cpu => 0,
    }
}

/// On Windows wgpu's D3D12 backend is the more mature path for the same
/// physical GPU; elsewhere Vulkan/Metal are native.
pub(crate) fn backend_rank(backend: wgpu::Backend) -> u8 {
    match backend {
        wgpu::Backend::Dx12 if cfg!(windows) => 3,
        wgpu::Backend::Vulkan | wgpu::Backend::Metal => 2,
        wgpu::Backend::Dx12 => 1,
        _ => 0,
    }
}

/// Index of the best adapter: device class first (discrete > integrated >
/// virtual > other > CPU), backend second. `discrete_only` restricts the
/// search to discrete GPUs. Pure, so the policy is unit-tested without GPUs.
pub fn pick_adapter(candidates: &[(wgpu::DeviceType, wgpu::Backend)], discrete_only: bool) -> Option<usize> {
    candidates
        .iter()
        .enumerate()
        .filter(|(_, (kind, _))| !discrete_only || *kind == wgpu::DeviceType::DiscreteGpu)
        .max_by_key(|(i, (kind, backend))| (device_rank(*kind), backend_rank(*backend), std::cmp::Reverse(*i)))
        .map(|(i, _)| i)
}

pub(crate) async fn select_adapter(instance: &wgpu::Instance, backends: wgpu::Backends) -> Result<(wgpu::Adapter, &'static str), String> {
    let mut adapters = instance.enumerate_adapters(backends).await;
    if !backends.contains(wgpu::Backends::GL) {
        adapters.retain(|a| a.get_info().backend != wgpu::Backend::Gl);
    }
    for adapter in &adapters {
        let info = adapter.get_info();
        crate::hc4j_trace!("adapter candidate: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
    }

    // HC4J_ADAPTER=<substring> pins a specific GPU by name.
    if let Ok(filter) = std::env::var("HC4J_ADAPTER") {
        let filter = filter.to_ascii_lowercase();
        let matching: Vec<wgpu::Adapter> = adapters
            .iter()
            .filter(|a| a.get_info().name.to_ascii_lowercase().contains(&filter))
            .cloned()
            .collect();
        if matching.is_empty() {
            eprintln!("[HC4J] HC4J_ADAPTER='{filter}' matched no adapter; ignoring it");
        } else {
            adapters = matching;
        }
    }

    let ranked = |discrete_only: bool| {
        let keys: Vec<_> = adapters.iter().map(|a| (a.get_info().device_type, a.get_info().backend)).collect();
        pick_adapter(&keys, discrete_only)
    };

    // 1. Any discrete GPU wins: hybrid laptops otherwise default to the iGPU.
    if let Some(i) = ranked(true) {
        return Ok((adapters.swap_remove(i), "discrete GPU"));
    }
    // 2. Let the platform's high-performance preference pick the *GPU*, then
    //    run that GPU on the best-ranked backend (DX12 on Windows): the
    //    preference alone may hand back the same device through Vulkan.
    if let Ok(preferred) = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
    {
        let hp = preferred.get_info();
        let same_gpu: Vec<usize> = adapters
            .iter()
            .enumerate()
            .filter(|(_, a)| {
                let info = a.get_info();
                info.vendor == hp.vendor && info.device == hp.device && info.device_type == hp.device_type
            })
            .map(|(i, _)| i)
            .collect();
        let keys: Vec<_> = same_gpu.iter().map(|&i| (adapters[i].get_info().device_type, adapters[i].get_info().backend)).collect();
        if let Some(best) = pick_adapter(&keys, false) {
            return Ok((adapters.swap_remove(same_gpu[best]), "high-performance preference (no discrete GPU found)"));
        }
        return Ok((preferred, "high-performance preference (no discrete GPU found)"));
    }
    // 3. Best remaining enumerated adapter, then the platform default.
    if let Some(i) = ranked(false) {
        return Ok((adapters.swap_remove(i), "best available (no discrete GPU found)"));
    }
    instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map(|a| (a, "platform default"))
        .map_err(|e| format!("failed to find a suitable GPU adapter: {e}"))
}

// ============================================================================
// Feature probing
// ============================================================================

pub(crate) fn probe_features(adapter: &wgpu::Adapter) -> (wgpu::Features, crate::engine::EngineFeatures) {
    let info = adapter.get_info();
    let available = adapter.features();
    let wanted = wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16;
    let required_features = available & wanted;
    let features = crate::engine::EngineFeatures {
        subgroups: required_features.contains(wgpu::Features::SUBGROUP),
        shader_f16: required_features.contains(wgpu::Features::SHADER_F16),
        subgroup_min_size: info.subgroup_min_size,
        subgroup_max_size: info.subgroup_max_size,
    };
    (required_features, features)
}

// ============================================================================
// AdapterInfo C-ABI export
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AdapterInfoC {
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub backend: u32,
    pub name_len: u32,
    pub driver_len: u32,
    pub driver_info_len: u32,
    pub name: [u8; 256],
    pub driver: [u8; 128],
    pub driver_info: [u8; 256],
}

fn copy_to_fixed<const N: usize>(src: &str, dst: &mut [u8; N]) -> u32 {
    let bytes = src.as_bytes();
    let len = bytes.len().min(N);
    dst[..len].copy_from_slice(&bytes[..len]);
    len as u32
}

/// Queries the active GPU adapter metadata into a C-compatible descriptor.
pub fn query_adapter_info() -> Hc4jResult<AdapterInfoC> {
    let engine = crate::engine::get_engine()?;
    let info = &engine.adapter_info;
    let mut c_info = AdapterInfoC {
        vendor_id: info.vendor,
        device_id: info.device,
        device_type: match info.device_type {
            wgpu::DeviceType::Other => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::DiscreteGpu => 2,
            wgpu::DeviceType::VirtualGpu => 3,
            wgpu::DeviceType::Cpu => 4,
        },
        backend: match info.backend {
            wgpu::Backend::Vulkan => 1,
            wgpu::Backend::Metal => 2,
            wgpu::Backend::Dx12 => 3,
            wgpu::Backend::Gl => 4,
            wgpu::Backend::BrowserWebGpu => 5,
            _ => 0,
        },
        name_len: 0,
        driver_len: 0,
        driver_info_len: 0,
        name: [0u8; 256],
        driver: [0u8; 128],
        driver_info: [0u8; 256],
    };
    c_info.name_len = copy_to_fixed(&info.name, &mut c_info.name);
    c_info.driver_len = copy_to_fixed(&info.driver, &mut c_info.driver);
    c_info.driver_info_len = copy_to_fixed(&info.driver_info, &mut c_info.driver_info);
    Ok(c_info)
}

#[cfg(test)]
mod tests {
    use super::pick_adapter;
    use wgpu::{Backend, DeviceType};

    #[test]
    fn discrete_beats_integrated_regardless_of_order() {
        let adapters = [
            (DeviceType::IntegratedGpu, Backend::Dx12),
            (DeviceType::DiscreteGpu, Backend::Vulkan),
            (DeviceType::Cpu, Backend::Dx12),
        ];
        assert_eq!(pick_adapter(&adapters, true), Some(1));
        assert_eq!(pick_adapter(&adapters, false), Some(1));
    }

    #[test]
    fn discrete_only_finds_nothing_on_an_igpu_laptop() {
        let adapters = [(DeviceType::IntegratedGpu, Backend::Vulkan), (DeviceType::Cpu, Backend::Dx12)];
        assert_eq!(pick_adapter(&adapters, true), None);
        assert_eq!(pick_adapter(&adapters, false), Some(0), "integrated beats the CPU rasterizer");
    }

    #[test]
    fn same_gpu_on_two_backends_prefers_the_platform_backend() {
        let adapters = [(DeviceType::DiscreteGpu, Backend::Vulkan), (DeviceType::DiscreteGpu, Backend::Dx12)];
        let expected = if cfg!(windows) { 1 } else { 0 };
        assert_eq!(pick_adapter(&adapters, true), Some(expected));
    }

    #[test]
    fn ties_keep_enumeration_order() {
        let adapters = [(DeviceType::DiscreteGpu, Backend::Dx12), (DeviceType::DiscreteGpu, Backend::Dx12)];
        assert_eq!(pick_adapter(&adapters, true), Some(0));
    }

    #[test]
    fn requested_backends_excludes_gl_by_default() {
        // Without HC4J_BACKEND or WGPU_BACKEND overrides, headless OpenGL/EGL must stay excluded.
        if std::env::var_os("HC4J_BACKEND").is_none() && std::env::var_os("WGPU_BACKEND").is_none() {
            let backends = super::requested_backends();
            assert!(!backends.contains(wgpu::Backends::GL));
            assert!(backends.contains(wgpu::Backends::VULKAN));
        }
    }
}
