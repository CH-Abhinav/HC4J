pub mod adapter;
pub mod error;
pub mod stream;

pub(crate) use adapter::{backend_rank, device_rank, requested_backends};
pub use adapter::{AdapterInfoC, pick_adapter, query_adapter_info};

use std::borrow::Cow;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use error::{Hc4jError, Hc4jResult};
use stream::CommandStream;

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

async fn init_engine() -> Result<GpuEngine, String> {
    let backends = adapter::requested_backends();
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
    let (adapter, reason) = adapter::select_adapter(&instance, backends).await?;
    let info = adapter.get_info();

    // Probe optional features and request the ones present.
    let (required_features, features) = adapter::probe_features(&adapter);

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
// Engine-level pure Rust operations
// ============================================================================

/// Initializes the global GPU engine, tiered memory manager, and background
/// matmul shader warm-up thread. Idempotent across repeated calls.
pub fn init() -> Hc4jResult<()> {
    crate::memory::manager()?;
    spawn_warmup();
    Ok(())
}

static WARMUP_STARTED: AtomicBool = AtomicBool::new(false);
static WARMUP_STOP: AtomicBool = AtomicBool::new(false);
static WARMUP_HANDLE: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);

#[cfg(target_os = "linux")]
extern "C" fn on_process_exit() {
    wait_warmup();
}

/// Signals any running background shader warm-up thread to stop before the next
/// kernel and joins it so driver static teardown never races with shader compilation.
pub fn wait_warmup() {
    WARMUP_STOP.store(true, Ordering::Release);
    if let Some(handle) = lock_or_recover(&WARMUP_HANDLE).take()
        && handle.thread().id() != std::thread::current().id()
    {
        let _ = handle.join();
    }
}

/// Compiles the expensive matmul kernels on a background thread, so the first
/// matmul does not pay for shader compilation (0.5-5 s under D3D12's FXC).
/// The pipeline cache is shared and lock-free during compilation, so a caller
/// that gets there first simply compiles it itself. `HC4J_WARMUP=0` opts out.
fn spawn_warmup() {
    if std::env::var("HC4J_WARMUP").is_ok_and(|v| v == "0") {
        return;
    }
    if WARMUP_STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    let spawned = std::thread::Builder::new().name("hc4j-warmup".to_string()).spawn(|| {
        if WARMUP_STOP.load(Ordering::Acquire) {
            return;
        }
        let Ok(engine) = get_engine() else { return };
        let started = std::time::Instant::now();
        for (name, source) in crate::kernels::linalg::matmul::warmup_kernels(engine) {
            if WARMUP_STOP.load(Ordering::Acquire) {
                return;
            }
            if let Err(err) = engine.get_or_compile(&name, "main", &source) {
                eprintln!("[HC4J] warm-up of {name} failed: {err}");
                return;
            }
        }
        crate::hc4j_trace!("warm-up compiled matmul kernels in {:?}", started.elapsed());
    });
    match spawned {
        Ok(handle) => {
            *lock_or_recover(&WARMUP_HANDLE) = Some(handle);
            #[cfg(target_os = "linux")]
            unsafe {
                libc::atexit(on_process_exit);
            }
        }
        Err(err) => {
            crate::hc4j_trace!("warm-up thread could not start: {err}");
        }
    }
}

/// Opens a command-batching scope: ops record into one command buffer until
/// the matching [`batch_end`]. Scopes nest; only the outermost submits.
pub fn batch_begin() -> Hc4jResult<()> {
    get_engine()?.stream.begin_batch();
    Ok(())
}

/// Closes a batching scope. Errors from submitting the batch surface here.
pub fn batch_end() -> Hc4jResult<()> {
    get_engine()?.stream.end_batch()
}

/// Submits all pending work and blocks until the GPU has finished it.
pub fn synchronize() -> Hc4jResult<()> {
    let stream = &get_engine()?.stream;
    stream.wait_epoch(stream.retire_epoch())
}

pub const FEATURE_SUBGROUPS: u64 = 1;
pub const FEATURE_SHADER_F16: u64 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct EngineStats {
    pub submissions: u64,
    pub dispatches: u64,
    pub kernel_bytes: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
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

/// Captures a snapshot of stream submission counters and adapter capabilities.
pub fn stats() -> Hc4jResult<EngineStats> {
    let engine = get_engine()?;
    let s = engine.stream.stats();
    Ok(EngineStats {
        submissions: s.submissions,
        dispatches: s.dispatches,
        kernel_bytes: s.kernel_bytes,
        host_to_device_bytes: s.host_to_device_bytes,
        device_to_host_bytes: s.device_to_host_bytes,
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
    })
}
