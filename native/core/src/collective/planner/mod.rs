//! Exact Zero-Parameter Collective Workload & Role Optimizer (`Theorems 1–4`
//! plus Zero-Waste Split-KV Reclamation), matching `hc4j.collective.WorkloadPlanner`
//! mathematically bit-for-bit across all SDKs.

pub mod convex_spill;
pub mod mst_islands;
pub mod ring_tsp;
pub mod role_planner;

pub use convex_spill::plan_work;
pub use mst_islands::{IslandInfo, chunk_range, detect_islands, detect_islands_for_payload};
pub use ring_tsp::{AllSubsetsRingTable, precompute_all_subsets_ring, solve_optimal_ring};
pub use role_planner::plan_roles;

use crate::collective::backend::{DeviceProfileC, LinkInfo};
use crate::engine::error::{Hc4jError, Hc4jResult};

/// Default usable fraction of GPU VRAM reserved for workload buffers (`0.85`).
pub const DEFAULT_VRAM_UTILIZATION: f64 = 0.85;

/// Sentinel rank identifying the host CPU execution / KV-cache node.
pub const HOST_CPU_RANK: i32 = -1;

/// Role classification codes matching [`DeviceRole`] and C `HC4J_ROLE_*` constants.
pub const ROLE_TENSOR_PARALLEL: u32 = 0;
pub const ROLE_SPLIT_KV_RECLAIMED: u32 = 1;
pub const ROLE_CPU_IN_PLACE: u32 = 2;
pub const ROLE_IDLE: u32 = 3;

/// Active regime classification codes returned in [`RolePlan`] and [`RolePlanHeaderC`].
pub const REGIME_ALL_GPU_SHARD: u32 = 0;
pub const REGIME_STRAGGLER_PRUNED: u32 = 1;
pub const REGIME_PCIE_STREAMING: u32 = 2;
pub const REGIME_CPU_IN_PLACE: u32 = 3;
pub const REGIME_SPLIT_KV_RECLAIMED: u32 = 4;

/// Physical profile of the explicit Host CPU offload node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostCpuProfile {
    pub ram_bytes: u64,
    pub compute_gflops: f64,
    pub ram_bandwidth_gbps: f64,
    pub aggregate_pcie_gbps: f64,
}

impl HostCpuProfile {
    /// Returns a disabled CPU profile (0 RAM, 0 compute).
    pub const fn disabled() -> Self {
        Self {
            ram_bytes: 0,
            compute_gflops: 0.0,
            ram_bandwidth_gbps: 0.0,
            aggregate_pcie_gbps: 0.0,
        }
    }

    /// Returns `true` when both host RAM capacity and CPU compute throughput are strictly positive.
    pub fn is_enabled(&self) -> bool {
        self.ram_bytes > 0 && self.compute_gflops > 0.0
    }
}

impl Default for HostCpuProfile {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Parallel workload specification for Theorems 1–4 optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkloadSpec {
    pub total_units: u64,
    pub flops_per_unit: f64,
    pub bytes_per_unit: u64,
    pub sync_bytes: u64,
    pub io_multiplier: f64,
    pub micro_batches: u32,
    pub vram_utilization: f64,
    pub streaming_overlap_gamma: f64,
    pub default_pcie_gbps: f64,
    pub disk_bandwidth_gbps: f64,
    pub disk_capacity_bytes: u64,
    pub allow_spilling: bool,
    pub host_cpu: HostCpuProfile,
}

impl WorkloadSpec {
    pub fn new(total_units: u64, flops_per_unit: f64, bytes_per_unit: u64, sync_bytes: u64) -> Self {
        Self {
            total_units,
            flops_per_unit,
            bytes_per_unit,
            sync_bytes,
            io_multiplier: 2.0,
            micro_batches: 1,
            vram_utilization: DEFAULT_VRAM_UTILIZATION,
            streaming_overlap_gamma: 0.5,
            default_pcie_gbps: 16.0,
            disk_bandwidth_gbps: 0.0,
            disk_capacity_bytes: 0,
            allow_spilling: true,
            host_cpu: HostCpuProfile::disabled(),
        }
    }

    /// Pipeline micro-batch overlap factor `(P - 1) / P` in `[0, 1)`.
    pub fn overlap_alpha(&self) -> f64 {
        let p = self.micro_batches.max(1) as f64;
        (p - 1.0) / p
    }
}

/// Per-device work assignment produced by [`plan_work`].
#[derive(Debug, Clone, PartialEq)]
pub struct ShardAllocation {
    pub rank: i32,
    pub units: u64,
    pub share_ratio: f64,
    pub estimated_compute_ms: f64,
    pub estimated_memory_bytes: u64,
    pub excluded_as_straggler: bool,
    pub vram_units: u64,
    pub host_stream_units: u64,
    pub disk_stream_units: u64,
    pub cpu_in_place_units: u64,
}

/// Complete execution plan produced by Theorems 1–4 optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkloadPlan {
    pub total_units: u64,
    pub allocations: Vec<ShardAllocation>,
    pub active_ranks: Vec<i32>,
    pub estimated_makespan_ms: f64,
    pub estimated_compute_ms: f64,
    pub estimated_sync_ms: f64,
}

/// Multi-operator role of a device in Split-KV decode attention + Tensor Parallelism.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceRole {
    TensorParallel = ROLE_TENSOR_PARALLEL,
    SplitKvReclaimed = ROLE_SPLIT_KV_RECLAIMED,
    CpuInPlace = ROLE_CPU_IN_PLACE,
    Idle = ROLE_IDLE,
}

/// KV-cache workload specification for [`plan_roles`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitKvSpec {
    pub total_kv_tokens: u64,
    pub kv_bytes_per_token: u64,
    pub attention_flops_per_token: f64,
    pub query_and_partial_bytes: u64,
    pub reserved_weight_bytes_per_gpu: u64,
    pub vram_utilization: f64,
}

impl SplitKvSpec {
    pub fn new(
        total_kv_tokens: u64,
        kv_bytes_per_token: u64,
        attention_flops_per_token: f64,
        query_and_partial_bytes: u64,
    ) -> Self {
        Self {
            total_kv_tokens,
            kv_bytes_per_token,
            attention_flops_per_token,
            query_and_partial_bytes,
            reserved_weight_bytes_per_gpu: 0,
            vram_utilization: DEFAULT_VRAM_UTILIZATION,
        }
    }
}

/// Per-device Split-KV assignment produced by [`plan_roles`].
#[derive(Debug, Clone, PartialEq)]
pub struct SplitKvAssignment {
    pub rank: i32,
    pub role: DeviceRole,
    pub kv_tokens: u64,
    pub estimated_local_attn_ms: f64,
    pub estimated_comm_ms: f64,
    pub estimated_total_ms: f64,
    pub kv_memory_bytes: u64,
}

/// Combined Tensor Parallel + Reclaimed Split-KV role plan.
#[derive(Debug, Clone, PartialEq)]
pub struct RolePlan {
    pub tp_plan: WorkloadPlan,
    pub split_kv_assignments: Vec<SplitKvAssignment>,
    pub estimated_tp_ms: f64,
    pub estimated_attention_ms: f64,
    pub total_kv_tokens: u64,
    pub reclaimed_gpu_count: u32,
    pub cpu_reclaimed: bool,
    pub regime_code: u32,
    pub regime: &'static str,
}

pub fn regime_name(code: u32) -> &'static str {
    match code {
        REGIME_ALL_GPU_SHARD => "ALL_GPU_SHARD",
        REGIME_STRAGGLER_PRUNED => "STRAGGLER_PRUNED",
        REGIME_PCIE_STREAMING => "PCIE_STREAMING",
        REGIME_CPU_IN_PLACE => "CPU_IN_PLACE",
        REGIME_SPLIT_KV_RECLAIMED => "SPLIT_KV_RECLAIMED",
        _ => "UNKNOWN",
    }
}

// ============================================================================
// C-ABI #[repr(C)] Structures (matched 1-to-1 in native/include/hc4x.h)
// ============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HostCpuProfileC {
    pub ram_bytes: u64,
    pub compute_gflops: f64,
    pub ram_bandwidth_gbps: f64,
    pub aggregate_pcie_gbps: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkloadSpecC {
    pub total_units: u64,
    pub flops_per_unit: f64,
    pub bytes_per_unit: u64,
    pub sync_bytes: u64,
    pub io_multiplier: f64,
    pub vram_utilization: f64,
    pub streaming_overlap_gamma: f64,
    pub default_pcie_gbps: f64,
    pub disk_bandwidth_gbps: f64,
    pub disk_capacity_bytes: u64,
    pub micro_batches: u32,
    pub allow_spilling: u32,
    pub host_cpu: HostCpuProfileC,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ShardAllocationC {
    pub rank: i32,
    pub excluded_as_straggler: u32,
    pub units: u64,
    pub share_ratio: f64,
    pub estimated_compute_ms: f64,
    pub estimated_memory_bytes: u64,
    pub vram_units: u64,
    pub host_stream_units: u64,
    pub disk_stream_units: u64,
    pub cpu_in_place_units: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WorkloadPlanHeaderC {
    pub total_units: u64,
    pub estimated_makespan_ms: f64,
    pub estimated_compute_ms: f64,
    pub estimated_sync_ms: f64,
    pub allocation_count: u32,
    pub active_rank_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitKvSpecC {
    pub total_kv_tokens: u64,
    pub kv_bytes_per_token: u64,
    pub attention_flops_per_token: f64,
    pub query_and_partial_bytes: u64,
    pub reserved_weight_bytes_per_gpu: u64,
    pub vram_utilization: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SplitKvAssignmentC {
    pub rank: i32,
    pub role: u32,
    pub kv_tokens: u64,
    pub estimated_local_attn_ms: f64,
    pub estimated_comm_ms: f64,
    pub estimated_total_ms: f64,
    pub kv_memory_bytes: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RolePlanHeaderC {
    pub tp_header: WorkloadPlanHeaderC,
    pub estimated_tp_ms: f64,
    pub estimated_attention_ms: f64,
    pub total_kv_tokens: u64,
    pub split_kv_count: u32,
    pub reclaimed_gpu_count: u32,
    pub cpu_reclaimed: u32,
    pub regime_code: u32,
}

impl From<HostCpuProfileC> for HostCpuProfile {
    fn from(c: HostCpuProfileC) -> Self {
        Self {
            ram_bytes: c.ram_bytes,
            compute_gflops: c.compute_gflops,
            ram_bandwidth_gbps: c.ram_bandwidth_gbps,
            aggregate_pcie_gbps: c.aggregate_pcie_gbps,
        }
    }
}

impl From<WorkloadSpecC> for WorkloadSpec {
    fn from(c: WorkloadSpecC) -> Self {
        Self {
            total_units: c.total_units,
            flops_per_unit: c.flops_per_unit,
            bytes_per_unit: c.bytes_per_unit,
            sync_bytes: c.sync_bytes,
            io_multiplier: c.io_multiplier,
            micro_batches: c.micro_batches,
            vram_utilization: c.vram_utilization,
            streaming_overlap_gamma: c.streaming_overlap_gamma,
            default_pcie_gbps: c.default_pcie_gbps,
            disk_bandwidth_gbps: c.disk_bandwidth_gbps,
            disk_capacity_bytes: c.disk_capacity_bytes,
            allow_spilling: c.allow_spilling != 0,
            host_cpu: c.host_cpu.into(),
        }
    }
}

impl From<&ShardAllocation> for ShardAllocationC {
    fn from(a: &ShardAllocation) -> Self {
        Self {
            rank: a.rank,
            excluded_as_straggler: u32::from(a.excluded_as_straggler),
            units: a.units,
            share_ratio: a.share_ratio,
            estimated_compute_ms: a.estimated_compute_ms,
            estimated_memory_bytes: a.estimated_memory_bytes,
            vram_units: a.vram_units,
            host_stream_units: a.host_stream_units,
            disk_stream_units: a.disk_stream_units,
            cpu_in_place_units: a.cpu_in_place_units,
        }
    }
}

impl From<SplitKvSpecC> for SplitKvSpec {
    fn from(c: SplitKvSpecC) -> Self {
        Self {
            total_kv_tokens: c.total_kv_tokens,
            kv_bytes_per_token: c.kv_bytes_per_token,
            attention_flops_per_token: c.attention_flops_per_token,
            query_and_partial_bytes: c.query_and_partial_bytes,
            reserved_weight_bytes_per_gpu: c.reserved_weight_bytes_per_gpu,
            vram_utilization: c.vram_utilization,
        }
    }
}

impl From<&SplitKvAssignment> for SplitKvAssignmentC {
    fn from(a: &SplitKvAssignment) -> Self {
        Self {
            rank: a.rank,
            role: a.role as u32,
            kv_tokens: a.kv_tokens,
            estimated_local_attn_ms: a.estimated_local_attn_ms,
            estimated_comm_ms: a.estimated_comm_ms,
            estimated_total_ms: a.estimated_total_ms,
            kv_memory_bytes: a.kv_memory_bytes,
        }
    }
}

pub(crate) fn validate_work_inputs(devices: &[DeviceProfileC], spec: &WorkloadSpec) -> Hc4jResult<()> {
    if devices.is_empty() {
        return Err(Hc4jError::InvalidParam("devices must not be empty"));
    }
    if spec.flops_per_unit < 0.0 || !spec.flops_per_unit.is_finite() {
        return Err(Hc4jError::InvalidParam("flops_per_unit must be finite and non-negative"));
    }
    if spec.io_multiplier < 0.0 || !spec.io_multiplier.is_finite() {
        return Err(Hc4jError::InvalidParam("io_multiplier must be finite and non-negative"));
    }
    if spec.micro_batches < 1 {
        return Err(Hc4jError::InvalidParam("micro_batches must be at least 1"));
    }
    if spec.vram_utilization <= 0.0 || spec.vram_utilization > 1.0 || !spec.vram_utilization.is_finite() {
        return Err(Hc4jError::InvalidParam("vram_utilization must be in (0, 1]"));
    }
    if spec.streaming_overlap_gamma < 0.0 || spec.streaming_overlap_gamma > 1.0 {
        return Err(Hc4jError::InvalidParam("streaming_overlap_gamma must be in [0, 1]"));
    }
    if spec.default_pcie_gbps < 0.0 || spec.disk_bandwidth_gbps < 0.0 {
        return Err(Hc4jError::InvalidParam("bandwidths must be non-negative"));
    }
    Ok(())
}

pub(crate) struct TopologyMatrices {
    pub bw: Vec<Vec<f64>>,
    pub lat_us: Vec<Vec<f64>>,
    pub tr: Vec<Vec<u32>>,
}

pub(crate) fn build_topology_matrices(n: usize, topology: &[LinkInfo]) -> TopologyMatrices {
    let mut bw = vec![vec![0.0f64; n]; n];
    let mut lat_us = vec![vec![0.0f64; n]; n];
    let mut tr = vec![vec![crate::collective::backend::TRANSPORT_HOST_STAGED; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                bw[i][j] = 25.0;
                lat_us[i][j] = 0.0;
                tr[i][j] = crate::collective::backend::TRANSPORT_PEER_TO_PEER;
            } else {
                bw[i][j] = 1.0;
                lat_us[i][j] = 100.0;
                tr[i][j] = crate::collective::backend::TRANSPORT_HOST_STAGED;
            }
        }
    }
    for link in topology {
        let src = link.src_rank as usize;
        let dst = link.dst_rank as usize;
        if src < n && dst < n {
            bw[src][dst] = link.bandwidth_gbps;
            lat_us[src][dst] = link.latency_us;
            tr[src][dst] = link.transport;
        }
    }
    TopologyMatrices { bw, lat_us, tr }
}
