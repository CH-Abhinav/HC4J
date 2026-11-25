//! `hc4rs` — Safe Rust GPU tensor engine, WGSL kernel suite, Vulkan P2P
//! collective runtime, and Exact Zero-Parameter Theorems 1–4 Workload Optimizer.
//!
//! All modules under `hc4rs` expose pure safe Rust APIs; the panic-guarded C-ABI
//! shared library (`hc4j.dll` / `libhc4j.so` / `libhc4j.dylib`) lives cleanly
//! in the sibling `hc4x-ffi` crate.

pub mod collective;
pub mod engine;
pub mod kernels;
pub mod memory;

use std::sync::Arc;

pub use collective::backend::{
    DeviceProfileC, LinkInfo, TRANSPORT_HOST_SHARED_DMA, TRANSPORT_HOST_STAGED, TRANSPORT_PEER_TO_PEER,
};
pub use collective::distributed::merge_split_kv_f32;
pub use collective::planner::{
    AllSubsetsRingTable, DEFAULT_VRAM_UTILIZATION, DeviceRole, HOST_CPU_RANK, HostCpuProfile, IslandInfo, RolePlan,
    ShardAllocation, SplitKvAssignment, SplitKvSpec, WorkloadPlan, WorkloadSpec,
};
pub use engine::EngineStats;
pub use engine::adapter::AdapterInfoC;
pub use engine::error::{Hc4jError, Hc4jResult};
pub use memory::{MemStats, TensorId};

use kernels::linalg::{FLAG_TRANS_A, FLAG_TRANS_B};
use kernels::math::{BinaryOp, DataType};
use kernels::nn::Activation;

/// Supported element data types for [`Tensor`] buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE-754 32-bit floating point (`f32`).
    F32,
    /// Signed 32-bit two's-complement integer (`i32`).
    I32,
    /// Packed IEEE-754 16-bit floating point (`u32` word pairs; 2 elements/word).
    F16,
}

impl DType {
    /// Element byte size (`4` for `F32`/`I32`, `2` for packed `F16`).
    pub const fn byte_size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 => 2,
        }
    }

    pub(crate) fn elem_byte_len(self, numel: usize) -> Hc4jResult<u64> {
        let raw = (numel as u64)
            .checked_mul(self.byte_size() as u64)
            .ok_or(Hc4jError::InvalidParam("tensor byte size overflows"))?;
        Ok(match self {
            DType::F16 => (raw.saturating_add(3)) & !3,
            _ => raw,
        })
    }
}

/// Human-readable adapter identity queried from the active GPU engine.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub name: String,
    pub driver: String,
    pub driver_info: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub backend: u32,
}

/// Handle to the initialized global WebGPU compute engine.
#[derive(Debug, Clone, Copy)]
pub struct Engine;

impl Engine {
    /// Initializes the global WebGPU engine, tiered memory manager, and background
    /// shader warm-up (idempotent).
    pub fn init() -> Hc4jResult<Self> {
        engine::init()?;
        Ok(Self)
    }

    /// Queries the active adapter name, vendor/device IDs, driver string, and backend.
    pub fn adapter_info(&self) -> Hc4jResult<AdapterInfo> {
        let raw = engine::query_adapter_info()?;
        let str_from = |bytes: &[u8], len: u32| {
            let n = (len as usize).min(bytes.len());
            String::from_utf8_lossy(&bytes[..n]).into_owned()
        };
        Ok(AdapterInfo {
            name: str_from(&raw.name, raw.name_len),
            driver: str_from(&raw.driver, raw.driver_len),
            driver_info: str_from(&raw.driver_info, raw.driver_info_len),
            vendor_id: raw.vendor_id,
            device_id: raw.device_id,
            device_type: raw.device_type,
            backend: raw.backend,
        })
    }

    /// Returns cumulative command-stream submission counters and device feature flags.
    pub fn stats(&self) -> Hc4jResult<EngineStats> {
        engine::stats()
    }

    /// Returns tiered VRAM/host/disk allocation statistics.
    pub fn memory_stats(&self) -> Hc4jResult<MemStats> {
        memory::stats()
    }

    /// Configures VRAM and host-RAM budgets in bytes (`0` keeps existing value).
    pub fn configure_memory(&self, vram_budget_bytes: u64, host_budget_bytes: u64) -> Hc4jResult<()> {
        memory::configure(vram_budget_bytes, host_budget_bytes)
    }

    /// Executes `f` inside a nested command-batching scope, submitting on exit.
    pub fn batch<R>(&self, f: impl FnOnce() -> Hc4jResult<R>) -> Hc4jResult<R> {
        engine::batch_begin()?;
        let res = f();
        let end_res = engine::batch_end();
        match res {
            Ok(val) => {
                end_res?;
                Ok(val)
            }
            Err(err) => Err(err),
        }
    }

    /// Blocks until all previously submitted GPU work has completed.
    pub fn synchronize(&self) -> Hc4jResult<()> {
        engine::synchronize()
    }
}

#[derive(Debug)]
struct TensorGuard {
    id: TensorId,
}

impl Drop for TensorGuard {
    fn drop(&mut self) {
        let _ = memory::free(self.id);
    }
}

/// Reference-counted RAII GPU/tiered tensor handle. Automatically frees the
/// underlying [`TensorId`] when the final clone drops.
#[derive(Debug, Clone)]
pub struct Tensor {
    guard: Arc<TensorGuard>,
    shape: Vec<usize>,
    numel: usize,
    dtype: DType,
}

impl Tensor {
    fn checked_numel(shape: &[usize]) -> Hc4jResult<usize> {
        if shape.is_empty() || shape.contains(&0) {
            return Err(Hc4jError::InvalidParam("tensor shape must be non-empty with positive dimensions"));
        }
        shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(Hc4jError::InvalidParam("tensor element count overflows usize"))
    }

    fn alloc_raw(shape: &[usize], dtype: DType, zeroed: bool) -> Hc4jResult<Self> {
        let numel = Self::checked_numel(shape)?;
        let bytes = dtype.elem_byte_len(numel)?;
        let id = memory::alloc_bytes(bytes, zeroed)?;
        Ok(Self {
            guard: Arc::new(TensorGuard { id }),
            shape: shape.to_vec(),
            numel,
            dtype,
        })
    }

    /// Allocates a zero-initialized tensor of the given shape and `dtype`.
    pub fn zeros(shape: &[usize], dtype: DType) -> Hc4jResult<Self> {
        Self::alloc_raw(shape, dtype, true)
    }

    /// Allocates an uninitialized tensor (for kernels that overwrite every element).
    pub fn uninit(shape: &[usize], dtype: DType) -> Hc4jResult<Self> {
        Self::alloc_raw(shape, dtype, false)
    }

    /// Uploads `data` (`f32`) into a device/tiered tensor of the specified `shape`.
    pub fn from_f32(data: &[f32], shape: &[usize]) -> Hc4jResult<Self> {
        let numel = Self::checked_numel(shape)?;
        if data.len() != numel {
            return Err(Hc4jError::InvalidParam("data length does not match shape"));
        }
        let tensor = Self::uninit(shape, DType::F32)?;
        memory::write_f32(tensor.id(), data)?;
        Ok(tensor)
    }

    /// Uploads `data` (`i32`) into a device/tiered tensor of the specified `shape`.
    pub fn from_i32(data: &[i32], shape: &[usize]) -> Hc4jResult<Self> {
        let numel = Self::checked_numel(shape)?;
        if data.len() != numel {
            return Err(Hc4jError::InvalidParam("data length does not match shape"));
        }
        let tensor = Self::uninit(shape, DType::I32)?;
        memory::write_bytes(tensor.id(), bytemuck::cast_slice(data))?;
        Ok(tensor)
    }

    pub fn id(&self) -> TensorId {
        self.guard.id
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.numel
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Queries the current residency tier (`0 = VRAM`, `1 = Host RAM`, `2 = Disk`).
    pub fn residency(&self) -> Hc4jResult<i32> {
        memory::residency(self.id())
    }

    /// Downloads tensor contents as `Vec<f32>`.
    pub fn to_f32_vec(&self) -> Hc4jResult<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(Hc4jError::InvalidParam("tensor dtype is not F32"));
        }
        let mut out = vec![0.0f32; self.numel];
        memory::download_f32(self.id(), &mut out)?;
        Ok(out)
    }

    /// Downloads tensor contents as `Vec<i32>`.
    pub fn to_i32_vec(&self) -> Hc4jResult<Vec<i32>> {
        if self.dtype != DType::I32 {
            return Err(Hc4jError::InvalidParam("tensor dtype is not I32"));
        }
        let mut out = vec![0i32; self.numel];
        memory::download_bytes(self.id(), bytemuck::cast_slice_mut(&mut out))?;
        Ok(out)
    }

    /// Converts an `F32` tensor to packed `F16` storage (`u32` pairs).
    pub fn to_f16(&self) -> Hc4jResult<Tensor> {
        if self.dtype != DType::F32 {
            return Err(Hc4jError::InvalidParam("to_f16 requires an F32 tensor"));
        }
        let out = Tensor::uninit(&self.shape, DType::F16)?;
        kernels::math::run_cast_f32_to_f16(self.id(), out.id(), self.numel as u64)?;
        Ok(out)
    }

    /// Converts an `F16` (or `I32`) tensor to `F32`.
    pub fn to_f32(&self) -> Hc4jResult<Tensor> {
        match self.dtype {
            DType::F32 => Ok(self.clone()),
            DType::F16 => {
                let out = Tensor::uninit(&self.shape, DType::F32)?;
                kernels::math::run_cast_f16_to_f32(self.id(), out.id(), self.numel as u64)?;
                Ok(out)
            }
            DType::I32 => {
                let out = Tensor::uninit(&self.shape, DType::F32)?;
                kernels::math::run_cast(self.id(), out.id(), self.numel as u64, 0, 1)?;
                Ok(out)
            }
        }
    }

    fn binary(&self, rhs: &Tensor, op: BinaryOp) -> Hc4jResult<Tensor> {
        if self.shape != rhs.shape || self.dtype != rhs.dtype {
            return Err(Hc4jError::InvalidParam("binary operand shapes or dtypes do not match"));
        }
        let dtype_code = match self.dtype {
            DType::I32 => 0,
            DType::F32 => 1,
            DType::F16 => return Err(Hc4jError::InvalidParam("binary arithmetic supports F32 and I32")),
        };
        let out = Tensor::uninit(&self.shape, self.dtype)?;
        kernels::math::run_binary(op, dtype_code, self.id(), rhs.id(), out.id(), self.numel, None)?;
        Ok(out)
    }

    pub fn add(&self, rhs: &Tensor) -> Hc4jResult<Tensor> {
        self.binary(rhs, BinaryOp::Add)
    }

    pub fn sub(&self, rhs: &Tensor) -> Hc4jResult<Tensor> {
        self.binary(rhs, BinaryOp::Sub)
    }

    pub fn mul(&self, rhs: &Tensor) -> Hc4jResult<Tensor> {
        self.binary(rhs, BinaryOp::Mul)
    }

    pub fn div(&self, rhs: &Tensor) -> Hc4jResult<Tensor> {
        self.binary(rhs, BinaryOp::Div)
    }
}

/// Safe linear algebra operations (`GEMM`, `GEMV`, mixed `f32 x f16`, `transpose`).
pub struct Linalg;

impl Linalg {
    /// Computes 2-D matrix multiplication `C = A @ B`. Supports `F32 @ F32` and `F32 @ F16`.
    pub fn matmul(a: &Tensor, b: &Tensor) -> Hc4jResult<Tensor> {
        Self::matmul_ex(a, b, false, false)
    }

    /// Computes 2-D matrix multiplication `C = A @ B^T`. Supports `F32 @ F32` and `F32 @ F16`.
    pub fn matmul_transposed_b(a: &Tensor, b: &Tensor) -> Hc4jResult<Tensor> {
        Self::matmul_ex(a, b, false, true)
    }

    /// Computes 2-D matrix multiplication `C = op(A) @ op(B)` with explicit transpose flags.
    pub fn matmul_ex(a: &Tensor, b: &Tensor, trans_a: bool, trans_b: bool) -> Hc4jResult<Tensor> {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(Hc4jError::InvalidParam("matmul expects 2-D matrices"));
        }
        if a.dtype != DType::F32 {
            return Err(Hc4jError::InvalidParam("matmul left operand must be F32"));
        }
        let (m, ka) = if trans_a {
            (a.shape[1], a.shape[0])
        } else {
            (a.shape[0], a.shape[1])
        };
        let (kb, n) = if trans_b {
            (b.shape[1], b.shape[0])
        } else {
            (b.shape[0], b.shape[1])
        };
        if ka != kb {
            return Err(Hc4jError::InvalidParam("matmul inner K dimension mismatch"));
        }

        let m_u32 = u32::try_from(m).map_err(|_| Hc4jError::InvalidParam("M exceeds u32"))?;
        let n_u32 = u32::try_from(n).map_err(|_| Hc4jError::InvalidParam("N exceeds u32"))?;
        let k_u32 = u32::try_from(ka).map_err(|_| Hc4jError::InvalidParam("K exceeds u32"))?;
        let mut flags = 0u32;
        if trans_a {
            flags |= FLAG_TRANS_A;
        }
        if trans_b {
            flags |= FLAG_TRANS_B;
        }

        let out = Tensor::uninit(&[m, n], DType::F32)?;
        match b.dtype {
            DType::F32 => kernels::linalg::run_matmul(a.id(), b.id(), out.id(), m_u32, n_u32, k_u32, flags)?,
            DType::F16 => {
                if trans_a {
                    return Err(Hc4jError::Unsupported("f32 x f16 matmul does not support trans_a"));
                }
                kernels::linalg::run_matmul_f32_f16(a.id(), b.id(), out.id(), m_u32, n_u32, k_u32, flags)?;
            }
            DType::I32 => return Err(Hc4jError::InvalidParam("matmul does not support I32 operands")),
        }
        Ok(out)
    }

    /// Returns the transpose `A^T` of a 2-D `F32` matrix.
    pub fn transpose(a: &Tensor) -> Hc4jResult<Tensor> {
        if a.shape.len() != 2 || a.dtype != DType::F32 {
            return Err(Hc4jError::InvalidParam("transpose expects a 2-D F32 matrix"));
        }
        let rows = u32::try_from(a.shape[0]).map_err(|_| Hc4jError::InvalidParam("rows exceeds u32"))?;
        let cols = u32::try_from(a.shape[1]).map_err(|_| Hc4jError::InvalidParam("cols exceeds u32"))?;
        let out = Tensor::uninit(&[a.shape[1], a.shape[0]], DType::F32)?;
        kernels::linalg::run_transpose(a.id(), out.id(), rows, cols)?;
        Ok(out)
    }
}

/// Safe neural-network operations (`RmsNorm`, `LayerNorm`, `Softmax`, activations, `Embedding`, `RoPE`).
pub struct Nn;

impl Nn {
    fn rows_cols(x: &Tensor) -> Hc4jResult<(u32, u32)> {
        if x.dtype != DType::F32 || x.shape.len() != 2 {
            return Err(Hc4jError::InvalidParam("row operation expects a 2-D F32 tensor"));
        }
        let rows = u32::try_from(x.shape[0]).map_err(|_| Hc4jError::InvalidParam("rows exceeds u32"))?;
        let cols = u32::try_from(x.shape[1]).map_err(|_| Hc4jError::InvalidParam("cols exceeds u32"))?;
        Ok((rows, cols))
    }

    /// Computes row-wise RMS normalization over a 2-D `F32` tensor.
    pub fn rms_norm(x: &Tensor, gamma: Option<&Tensor>, eps: f32) -> Hc4jResult<Tensor> {
        let (rows, cols) = Self::rows_cols(x)?;
        if let Some(g) = gamma
            && (g.dtype != DType::F32 || g.numel != x.shape[1])
        {
            return Err(Hc4jError::InvalidParam("gamma must be F32 with length == cols"));
        }
        let out = Tensor::uninit(&x.shape, DType::F32)?;
        let g_id = gamma.map_or(0, Tensor::id);
        kernels::nn::run_rmsnorm_rows_f32(x.id(), g_id, out.id(), rows, cols, eps.to_bits())?;
        Ok(out)
    }

    /// Computes row-wise layer normalization over a 2-D `F32` tensor.
    pub fn layer_norm(
        x: &Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        eps: f32,
    ) -> Hc4jResult<Tensor> {
        let (rows, cols) = Self::rows_cols(x)?;
        let out = Tensor::uninit(&x.shape, DType::F32)?;
        let g_id = gamma.map_or(0, Tensor::id);
        let b_id = beta.map_or(0, Tensor::id);
        kernels::nn::run_layernorm_rows_f32(x.id(), g_id, b_id, out.id(), rows, cols, eps.to_bits())?;
        Ok(out)
    }

    /// Computes row-wise numerically stable softmax over a 2-D `F32` tensor.
    pub fn softmax(x: &Tensor) -> Hc4jResult<Tensor> {
        let (rows, cols) = Self::rows_cols(x)?;
        let out = Tensor::uninit(&x.shape, DType::F32)?;
        kernels::nn::run_softmax_rows_f32(x.id(), out.id(), rows, cols, false)?;
        Ok(out)
    }

    /// Computes row-wise log-softmax over a 2-D `F32` tensor.
    pub fn log_softmax(x: &Tensor) -> Hc4jResult<Tensor> {
        let (rows, cols) = Self::rows_cols(x)?;
        let out = Tensor::uninit(&x.shape, DType::F32)?;
        kernels::nn::run_softmax_rows_f32(x.id(), out.id(), rows, cols, true)?;
        Ok(out)
    }

    fn activate(x: &Tensor, act: Activation, alpha: f32) -> Hc4jResult<Tensor> {
        if x.dtype != DType::F32 {
            return Err(Hc4jError::InvalidParam("activation expects an F32 tensor"));
        }
        let out = Tensor::uninit(&x.shape, DType::F32)?;
        kernels::nn::run_activation_f32(x.id(), out.id(), x.numel, None, act as u32, alpha.to_bits())?;
        Ok(out)
    }

    pub fn relu(x: &Tensor) -> Hc4jResult<Tensor> {
        Self::activate(x, Activation::Relu, 0.0)
    }

    pub fn gelu(x: &Tensor) -> Hc4jResult<Tensor> {
        Self::activate(x, Activation::Gelu, 0.0)
    }

    pub fn silu(x: &Tensor) -> Hc4jResult<Tensor> {
        Self::activate(x, Activation::Silu, 0.0)
    }

    pub fn sigmoid(x: &Tensor) -> Hc4jResult<Tensor> {
        Self::activate(x, Activation::Sigmoid, 0.0)
    }

    /// Token embedding lookup (`table` may be `F32` or packed `F16`, `indices` is `I32`).
    pub fn embedding(table: &Tensor, indices: &Tensor) -> Hc4jResult<Tensor> {
        if table.shape.len() != 2 || indices.dtype != DType::I32 {
            return Err(Hc4jError::InvalidParam("embedding expects 2-D table and I32 indices"));
        }
        let is_f16 = match table.dtype {
            DType::F32 => false,
            DType::F16 => true,
            DType::I32 => return Err(Hc4jError::InvalidParam("embedding table cannot be I32")),
        };
        let vocab = u32::try_from(table.shape[0]).map_err(|_| Hc4jError::InvalidParam("vocab exceeds u32"))?;
        let hidden = u32::try_from(table.shape[1]).map_err(|_| Hc4jError::InvalidParam("hidden exceeds u32"))?;
        let num_tokens = u32::try_from(indices.numel).map_err(|_| Hc4jError::InvalidParam("tokens exceed u32"))?;
        let out = Tensor::uninit(&[indices.numel, table.shape[1]], DType::F32)?;
        kernels::nn::run_embedding(table.id(), indices.id(), out.id(), vocab, hidden, num_tokens, is_f16)?;
        Ok(out)
    }

    /// Applies NeoX half-split Rotary Position Embedding (RoPE) to `[batch, seq_len, num_heads, head_dim]`.
    pub fn rope(x: &Tensor, pos_offset: u32, theta: f32) -> Hc4jResult<Tensor> {
        if x.dtype != DType::F32 || x.shape.len() != 4 {
            return Err(Hc4jError::InvalidParam("rope expects 4-D F32 [batch, seq_len, num_heads, head_dim]"));
        }
        let out = Tensor::uninit(&x.shape, DType::F32)?;
        kernels::nn::run_rope_f32(
            x.id(),
            out.id(),
            x.shape[0] as u32,
            x.shape[1] as u32,
            x.shape[2] as u32,
            x.shape[3] as u32,
            pos_offset,
            theta.to_bits(),
        )?;
        Ok(out)
    }

    /// Adds 1-D bias `[cols]` onto each row of `[rows, cols]`.
    pub fn add_row_bias(x: &Tensor, bias: &Tensor) -> Hc4jResult<Tensor> {
        let (rows, cols) = Self::rows_cols(x)?;
        if bias.dtype != DType::F32 || bias.numel != x.shape[1] {
            return Err(Hc4jError::InvalidParam("bias must be F32 with length == cols"));
        }
        let out = Tensor::uninit(&x.shape, DType::F32)?;
        kernels::nn::run_row_bias(x.id(), bias.id(), out.id(), rows, cols)?;
        Ok(out)
    }
}

/// RAII wrapper around a multi-device collective runtime context.
#[derive(Debug)]
pub struct Collective {
    handle: u64,
}

impl Collective {
    /// Opens a multi-device collective session for `ranks` GPUs using `HC4J_COLLECTIVE_BACKEND`.
    pub fn open(ranks: u32) -> Hc4jResult<Self> {
        let handle = collective::open_backend(ranks)?;
        Ok(Self { handle })
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }

    pub fn ranks(&self) -> Hc4jResult<u32> {
        collective::with_backend(self.handle, |b| Ok(b.ranks() as u32))
    }

    pub fn backend_name(&self) -> Hc4jResult<String> {
        collective::with_backend(self.handle, |b| Ok(b.backend_name().to_string()))
    }

    pub fn device_name(&self, rank: u32) -> Hc4jResult<String> {
        collective::with_backend(self.handle, |b| b.device_name(rank as usize))
    }

    pub fn device_profile(&self, rank: u32) -> Hc4jResult<DeviceProfileC> {
        collective::with_backend(self.handle, |b| {
            let profiles = b.device_profiles();
            profiles
                .get(rank as usize)
                .copied()
                .ok_or(Hc4jError::InvalidParam("rank out of range"))
        })
    }

    pub fn links(&self) -> Hc4jResult<Vec<LinkInfo>> {
        collective::with_backend(self.handle, |b| Ok(b.links().to_vec()))
    }

    pub fn ring_order(&self) -> Hc4jResult<Vec<u32>> {
        collective::with_backend(self.handle, |b| Ok(b.ring_order().to_vec()))
    }

    pub fn islands(&self) -> Hc4jResult<Vec<IslandInfo>> {
        collective::with_backend(self.handle, |b| Ok(b.islands().to_vec()))
    }

    pub fn alloc(&self, rank: u32, elements: u64) -> Hc4jResult<u64> {
        collective::with_backend(self.handle, |b| b.alloc(rank as usize, elements))
    }

    pub fn free(&self, buffer_id: u64) -> Hc4jResult<()> {
        collective::with_backend(self.handle, |b| b.free(buffer_id))
    }

    pub fn write(&self, buffer_id: u64, data: &[f32]) -> Hc4jResult<()> {
        collective::with_backend(self.handle, |b| b.write(buffer_id, data))
    }

    pub fn read(&self, buffer_id: u64, out: &mut [f32]) -> Hc4jResult<()> {
        collective::with_backend(self.handle, |b| b.read(buffer_id, out))
    }

    pub fn allreduce_sum(&self, buffers: &[u64]) -> Hc4jResult<()> {
        collective::with_backend(self.handle, |b| b.allreduce_sum(buffers))
    }

    pub fn allreduce_sum_subset(&self, buffers: &[u64]) -> Hc4jResult<()> {
        collective::with_backend(self.handle, |b| b.allreduce_sum_subset(buffers))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bias_act(
        &self,
        rank: u32,
        buf_a: u64,
        buf_b: u64,
        buf_bias: u64,
        buf_c: u64,
        m: u32,
        n: u32,
        k: u32,
        flags: u32,
    ) -> Hc4jResult<()> {
        collective::with_backend(self.handle, |b| {
            b.gemm_bias_act(rank as usize, buf_a, buf_b, buf_bias, buf_c, m, n, k, flags)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn split_kv_attn(
        &self,
        rank: u32,
        buf_q: u64,
        buf_k: u64,
        buf_v: u64,
        buf_out_o: u64,
        buf_out_stats: u64,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        local_tokens: u32,
        scale: f32,
    ) -> Hc4jResult<()> {
        collective::with_backend(self.handle, |b| {
            b.split_kv_attn(
                rank as usize,
                buf_q,
                buf_k,
                buf_v,
                buf_out_o,
                buf_out_stats,
                num_heads,
                num_kv_heads,
                head_dim,
                local_tokens,
                scale,
            )
        })
    }

    /// Merges Split-KV partial attention outputs and online-softmax `(m_i, l_i)` stats across workers.
    pub fn merge_split_kv_f32(
        num_heads: u32,
        head_dim: u32,
        partials_o: &[&[f32]],
        partials_stats: &[&[f32]],
        out: &mut [f32],
    ) -> Hc4jResult<()> {
        collective::merge_split_kv_f32(num_heads, head_dim, partials_o, partials_stats, out)
    }
}

impl Drop for Collective {
    fn drop(&mut self) {
        let _ = collective::close_backend(self.handle);
    }
}

/// Exact Zero-Parameter Theorems 1–4 Workload & Split-KV Role Optimizer facade.
pub struct Optimizer;

impl Optimizer {
    /// Plans optimal Tensor Parallel shard allocations and 3-tier memory spilling across
    /// GPU subsets and optional host CPU offload.
    pub fn plan_work(
        devices: &[DeviceProfileC],
        topology: &[LinkInfo],
        spec: &WorkloadSpec,
    ) -> Hc4jResult<WorkloadPlan> {
        collective::planner::plan_work(devices, topology, spec)
    }

    /// Plans joint Tensor Parallel + Zero-Waste Split-KV role assignments.
    pub fn plan_roles(
        devices: &[DeviceProfileC],
        topology: &[LinkInfo],
        tp_spec: &WorkloadSpec,
        kv_spec: &SplitKvSpec,
    ) -> Hc4jResult<RolePlan> {
        collective::planner::plan_roles(devices, topology, tp_spec, kv_spec)
    }

    /// Precomputes exact bottleneck bandwidth & max hop latency across all `2^N - 1` subsets.
    pub fn precompute_all_subsets_ring(
        n: usize,
        bw: &[Vec<f64>],
        lat_us: &[Vec<f64>],
    ) -> AllSubsetsRingTable {
        collective::planner::precompute_all_subsets_ring(n, bw, lat_us)
    }
}

/// Backwards-compatible internal constant check helper for `DType::F32` / `I32`.
pub fn binary_dtype_code(dt: DataType) -> u32 {
    match dt {
        DataType::I32 => 0,
        DataType::F32 => 1,
    }
}
