pub mod error;
pub mod memory;
pub mod ops;
pub mod stream;

use std::borrow::Cow;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use error::{Hc4jError, Hc4jResult};
use stream::CommandStream;

// ============================================================================
// C ABI registry. Every exported symbol, in one auditable list. The
// `#[unsafe(no_mangle)]` definitions live next to their implementations.
// ============================================================================

pub use memory::{
    hc4j_gpu_alloc, hc4j_gpu_alloc_uninit, hc4j_gpu_download, hc4j_gpu_free, hc4j_gpu_write, hc4j_mem_configure,
    hc4j_mem_evict, hc4j_mem_residency, hc4j_mem_stats,
};
pub use ops::arithmetic::{
    dispatch_add, dispatch_add_f32, dispatch_div, dispatch_div_f32, dispatch_mul, dispatch_mul_f32, dispatch_sub,
    dispatch_sub_f32,
};
pub use ops::exponential::{dispatch_exp_f32, dispatch_ln_f32, dispatch_log2_f32, dispatch_log10_f32, dispatch_sqrt_f32};
pub use ops::fusion::hc4j_fused_dispatch;
pub use ops::matmul::{dispatch_matmul_f32, dispatch_matmul_f32_ex, dispatch_transpose_f32, hc4j_matmul_plan};
pub use ops::trigno::{
    dispatch_acos_f32, dispatch_acosh_f32, dispatch_asin_f32, dispatch_asinh_f32, dispatch_atan_f32,
    dispatch_atanh_f32, dispatch_cos_f32, dispatch_cosh_f32, dispatch_sin_f32, dispatch_sinh_f32, dispatch_tan_f32,
    dispatch_tanh_f32,
};

/// Upper bound on any single blocking GPU wait. A healthy device never gets
/// close; hitting it means the device hung and the caller gets an error code
/// instead of a frozen JVM thread.
pub const GPU_WAIT_TIMEOUT: Duration = Duration::from_secs(120);

/// Optional device capabilities, probed at init and requested if present.
#[derive(Clone, Copy, Debug, Default)]
pub struct EngineFeatures {
    /// `wgpu::Features::SUBGROUP`: subgroup (warp/wave) operations.
    pub subgroups: bool,
    /// `wgpu::Features::SHADER_F16`: 16-bit floats in shaders.
    pub shader_f16: bool,
    pub subgroup_min_size: u32,
    pub subgroup_max_size: u32,
}

pub struct GpuEngine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub limits: wgpu::Limits,
    pub adapter_info: wgpu::AdapterInfo,
    pub features: EngineFeatures,
    pub pipelines: Mutex<HashMap<String, wgpu::ComputePipeline>>,
    pub stream: CommandStream,
}

static ENGINE: OnceLock<Result<GpuEngine, String>> = OnceLock::new();
static DEVICE_LOST: AtomicBool = AtomicBool::new(false);

/// Returns the process-wide engine, initializing it on first use. Fails
/// (rather than panicking) when no adapter is available or the device is lost.
pub fn get_engine() -> Hc4jResult<&'static GpuEngine> {
    let engine = ENGINE.get_or_init(|| {
        error::init_trace_from_env();
        catch_unwind(AssertUnwindSafe(|| pollster::block_on(init_engine())))
            .unwrap_or_else(|_| Err("panic during GPU initialization".to_string()))
    });
    match engine {
        Ok(_) if DEVICE_LOST.load(Ordering::Acquire) => Err(Hc4jError::Device("GPU device was lost".to_string())),
        Ok(engine) => Ok(engine),
        Err(msg) => Err(Hc4jError::Device(msg.clone())),
    }
}

// ============================================================================
// Adapter selection
// ============================================================================

/// Backends to consider. `HC4J_BACKEND=dx12|vulkan|metal` narrows the set;
/// the default on Windows is DX12 + Vulkan.
fn requested_backends() -> wgpu::Backends {
    match std::env::var("HC4J_BACKEND").map(|v| v.to_ascii_lowercase()).as_deref() {
        Ok("dx12") => wgpu::Backends::DX12,
        Ok("vulkan") => wgpu::Backends::VULKAN,
        Ok("metal") => wgpu::Backends::METAL,
        _ if cfg!(windows) => wgpu::Backends::DX12 | wgpu::Backends::VULKAN,
        _ => wgpu::Backends::all(),
    }
}

fn device_rank(kind: wgpu::DeviceType) -> u8 {
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
fn backend_rank(backend: wgpu::Backend) -> u8 {
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

async fn select_adapter(instance: &wgpu::Instance, backends: wgpu::Backends) -> Result<(wgpu::Adapter, &'static str), String> {
    let mut adapters = instance.enumerate_adapters(backends).await;
    for adapter in &adapters {
        let info = adapter.get_info();
        hc4j_trace!("adapter candidate: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
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

async fn init_engine() -> Result<GpuEngine, String> {
    let backends = requested_backends();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        // D3D12 and Vulkan report their OS memory budget; without a threshold,
        // WDDM silently pages VRAM to system memory instead of failing. At 90%
        // buffer creation reports OutOfMemory, which the tiered manager traps
        // with an error scope and answers by evicting.
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds {
            for_resource_creation: Some(90),
            for_device_loss: None,
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let (adapter, reason) = select_adapter(&instance, backends).await?;
    let info = adapter.get_info();

    // Probe optional features and request the ones present.
    let available = adapter.features();
    let wanted = wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16;
    let required_features = available & wanted;
    let features = EngineFeatures {
        subgroups: required_features.contains(wgpu::Features::SUBGROUP),
        shader_f16: required_features.contains(wgpu::Features::SHADER_F16),
        subgroup_min_size: info.subgroup_min_size,
        subgroup_max_size: info.subgroup_max_size,
    };

    // Request the adapter's real limits. The WebGPU defaults cap storage
    // bindings at 128 MiB and buffers at 256 MiB, far below what GPUs support.
    let limits = adapter.limits();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("HC4J Device"),
            required_features,
            required_limits: limits.clone(),
            ..Default::default()
        })
        .await
        .map_err(|e| format!("failed to open logical GPU device: {e}"))?;

    eprintln!(
        "[HC4J] GPU: {} | {:?} | {:?} | driver: {} {} | selected by: {reason}",
        info.name, info.device_type, info.backend, info.driver, info.driver_info
    );
    eprintln!(
        "[HC4J] features: subgroups={} (size {}..{}) shader_f16={} | workgroups/dim={} storage-binding={} MiB",
        features.subgroups,
        features.subgroup_min_size,
        features.subgroup_max_size,
        features.shader_f16,
        limits.max_compute_workgroups_per_dimension,
        limits.max_storage_buffer_binding_size >> 20
    );
    if info.device_type != wgpu::DeviceType::DiscreteGpu {
        eprintln!("[HC4J] note: no discrete GPU available; running on {:?}", info.device_type);
    }

    // wgpu's default uncaptured-error handler panics, and a panic unwinding
    // into an `extern "C"` frame aborts the JVM. Errors on paths HC4J cares
    // about are captured by error scopes; anything that escapes is logged.
    device.on_uncaptured_error(Arc::new(|err: wgpu::Error| {
        eprintln!("[HC4J] uncaptured wgpu error: {err}");
    }));
    device.set_device_lost_callback(|reason, message| {
        DEVICE_LOST.store(true, Ordering::Release);
        eprintln!("[HC4J] GPU device lost ({reason:?}): {message}");
    });

    Ok(GpuEngine {
        stream: CommandStream::new(device.clone(), queue.clone()),
        device,
        queue,
        limits,
        adapter_info: info,
        features,
        pipelines: Mutex::new(HashMap::new()),
    })
}

/// Locks a mutex, recovering the data if another thread panicked while
/// holding it. Every critical section in HC4J leaves its state consistent
/// before any operation that could panic.
pub fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl GpuEngine {
    /// Compiles (or fetches from cache) a compute pipeline, keyed by shader
    /// name and entry point. Invalid pipelines never enter the cache.
    pub fn get_or_compile(&self, shader_name: &str, entry_point: &str, wgsl_code: &str) -> Hc4jResult<wgpu::ComputePipeline> {
        let key = format!("{shader_name}::{entry_point}");
        if let Some(pipeline) = lock_or_recover(&self.pipelines).get(&key) {
            return Ok(pipeline.clone());
        }

        let started = std::time::Instant::now();
        let trap = ErrorTrap::push(&self.device);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader_name),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl_code)),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(shader_name),
            layout: None,
            module: &module,
            entry_point: Some(entry_point),
            // WGSL zero-initializes workgroup memory; naga emits that as
            // initializer code that D3D12's FXC compiles pathologically (measured
            // on Iris Xe: 1056-float transpose tile 8.1 s -> 64 ms, 64x64 GEMM
            // 24 s -> 0.5 s, 128x128 GEMM 190 s). Invariant for every HC4J
            // kernel: workgroup memory is written before it is read (see the
            // kernel sources; the subgroup GEMV zeroes its atomic explicitly).
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });
        trap.finish()?;
        crate::hc4j_trace!("compiled pipeline {key} in {:?}", started.elapsed());

        let mut cache = lock_or_recover(&self.pipelines);
        Ok(cache.entry(key).or_insert(pipeline).clone())
    }

    /// Blocks until `submission` has finished on the GPU and its map
    /// callbacks have run.
    pub fn wait_for(&self, submission: wgpu::SubmissionIndex) -> Hc4jResult<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(GPU_WAIT_TIMEOUT),
            })
            .map(|_| ())
            .map_err(|e| Hc4jError::Device(format!("GPU wait failed: {e}")))
    }
}

/// Captures every wgpu error raised on this thread between `push` and
/// `finish`, so failures become `Hc4jError`s instead of reaching the
/// uncaptured-error handler.
///
/// wgpu error scopes are thread-local and must be popped in reverse push
/// order. Fields drop in declaration order, so if `finish` is never called
/// (early return), the guards still pop innermost-first.
pub struct ErrorTrap {
    validation: Option<wgpu::ErrorScopeGuard>,
    out_of_memory: Option<wgpu::ErrorScopeGuard>,
    internal: Option<wgpu::ErrorScopeGuard>,
}

impl ErrorTrap {
    pub fn push(device: &wgpu::Device) -> Self {
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        Self {
            validation: Some(validation),
            out_of_memory: Some(out_of_memory),
            internal: Some(internal),
        }
    }

    /// Pops all scopes. On native wgpu the futures resolve immediately.
    /// OutOfMemory takes priority because it is the one error callers
    /// recover from (by evicting and retrying).
    pub fn finish(mut self) -> Hc4jResult<()> {
        let validation = self.validation.take().and_then(|g| pollster::block_on(g.pop()));
        let out_of_memory = self.out_of_memory.take().and_then(|g| pollster::block_on(g.pop()));
        let internal = self.internal.take().and_then(|g| pollster::block_on(g.pop()));
        match out_of_memory.or(validation).or(internal) {
            Some(err) => Err(Hc4jError::from_wgpu(err)),
            None => Ok(()),
        }
    }
}

// ============================================================================
// Engine-level FFI
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn hc4j_init_gpu() -> i32 {
    error::ffi_guard("hc4j_init_gpu", || memory::manager().map(|_| ()))
}

/// Opens a command-batching scope: ops record into one command buffer until
/// the matching `hc4j_batch_end`. Scopes nest; only the outermost submits.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_batch_begin() -> i32 {
    error::ffi_guard("hc4j_batch_begin", || {
        get_engine()?.stream.begin_batch();
        Ok(())
    })
}

/// Closes a batching scope. Errors from submitting the batch surface here.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_batch_end() -> i32 {
    error::ffi_guard("hc4j_batch_end", || get_engine()?.stream.end_batch())
}

/// Submits all pending work and blocks until the GPU has finished it. For
/// benchmarking and explicit host/device synchronization; ordinary reads
/// synchronize on their own.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_synchronize() -> i32 {
    error::ffi_guard("hc4j_synchronize", || {
        let stream = &get_engine()?.stream;
        stream.wait_epoch(stream.retire_epoch())
    })
}

pub const FEATURE_SUBGROUPS: u64 = 1;
pub const FEATURE_SHADER_F16: u64 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct EngineStats {
    pub submissions: u64,
    pub dispatches: u64,
    pub kernel_bytes: u64,
    pub batches: u64,
    pub completed_epoch: u64,
    pub submitted_epoch: u64,
    /// Bitmask of `FEATURE_*`.
    pub features: u64,
    /// 0 other, 1 integrated, 2 discrete, 3 virtual, 4 CPU.
    pub device_type: u64,
    /// 0 noop, 1 Vulkan, 2 Metal, 3 DX12, 4 GL, 5 browser WebGPU.
    pub backend: u64,
    pub subgroup_min_size: u64,
    pub subgroup_max_size: u64,
}

/// # Safety
/// `out` must be null or point to writable memory for one `EngineStats`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_engine_stats(out: *mut EngineStats) -> i32 {
    error::ffi_guard("hc4j_engine_stats", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null stats pointer"));
        }
        let engine = get_engine()?;
        let s = engine.stream.stats();
        let stats = EngineStats {
            submissions: s.submissions,
            dispatches: s.dispatches,
            kernel_bytes: s.kernel_bytes,
            batches: s.batches,
            completed_epoch: s.completed_epoch,
            submitted_epoch: s.submitted_epoch,
            features: (if engine.features.subgroups { FEATURE_SUBGROUPS } else { 0 })
                | (if engine.features.shader_f16 { FEATURE_SHADER_F16 } else { 0 }),
            device_type: match engine.adapter_info.device_type {
                wgpu::DeviceType::Other => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                wgpu::DeviceType::DiscreteGpu => 2,
                wgpu::DeviceType::VirtualGpu => 3,
                wgpu::DeviceType::Cpu => 4,
            },
            backend: match engine.adapter_info.backend {
                wgpu::Backend::Vulkan => 1,
                wgpu::Backend::Metal => 2,
                wgpu::Backend::Dx12 => 3,
                wgpu::Backend::Gl => 4,
                wgpu::Backend::BrowserWebGpu => 5,
                _ => 0,
            },
            subgroup_min_size: engine.features.subgroup_min_size as u64,
            subgroup_max_size: engine.features.subgroup_max_size as u64,
        };
        unsafe { out.write_unaligned(stats) };
        Ok(())
    })
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
}
