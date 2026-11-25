# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper over native Rust Exact Theorems 1-4 Optimizer (``hc4j_optimizer_plan_*``)."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from ._ffi import (
    DeviceProfileC,
    HostCpuProfileC,
    InvalidParamError,
    LinkInfo,
    RolePlanHeaderC,
    ShardAllocationC,
    SplitKvAssignmentC,
    SplitKvSpecC,
    WorkloadPlanHeaderC,
    WorkloadSpecC,
    check_status,
    get_library,
)
from .collective import DEVICE_TYPE_CODES, DeviceProfile, Link

HOST_CPU_RANK = -1

_REGIME_NAMES = {
    0: "ALL_GPU_SHARD",
    1: "STRAGGLER_PRUNED",
    2: "PCIE_STREAMING",
    3: "CPU_IN_PLACE",
    4: "SPLIT_KV_RECLAIMED",
}


class DeviceRole(Enum):
    """Assigned worker role in joint Tensor-Parallel + Split-KV execution."""

    TENSOR_PARALLEL = 0
    SPLIT_KV_RECLAIMED = 1
    CPU_IN_PLACE = 2
    IDLE = 3


@dataclass(frozen=True)
class HostCpuProfile:
    """Host CPU capacity and bandwidth profile for Tier-1 spilling & Tier-3 CPU compute."""

    name: str = "Disabled"
    ram_bytes: int = 0
    compute_gflops: float = 0.0
    ram_bandwidth_gbps: float = 0.0
    aggregate_pcie_gbps: float = 0.0

    @classmethod
    def disabled(cls) -> HostCpuProfile:
        return cls()


@dataclass(frozen=True)
class WorkloadSpec:
    """Workload specification mirroring ``WorkloadSpecC`` (120 bytes)."""

    total_units: int
    flops_per_unit: float
    bytes_per_unit: int
    sync_bytes: int = 0
    io_multiplier: float = 2.0
    micro_batches: int = 1
    vram_utilization: float = 0.85
    streaming_overlap_gamma: float = 0.5
    default_pcie_gbps: float = 16.0
    disk_bandwidth_gbps: float = 0.0
    disk_capacity_bytes: int = 0
    allow_spilling: bool = True
    host_cpu: HostCpuProfile = HostCpuProfile.disabled()


@dataclass(frozen=True)
class ShardAllocation:
    """Per-device workload allocation and memory-tier breakdown."""

    rank: int
    units: int
    share_ratio: float
    estimated_compute_ms: float
    estimated_memory_bytes: int
    excluded_as_straggler: bool
    vram_units: int
    host_stream_units: int
    disk_stream_units: int
    cpu_in_place_units: int


@dataclass(frozen=True)
class WorkloadPlan:
    """Global Theorems 1-4 workload optimization plan."""

    total_units: int
    allocations: list[ShardAllocation]
    active_ranks: list[int]
    estimated_makespan_ms: float
    estimated_compute_ms: float
    estimated_sync_ms: float


@dataclass(frozen=True)
class SplitKvSpec:
    """Decode-attention KV-cache specification mirroring ``SplitKvSpecC``."""

    total_kv_tokens: int
    kv_bytes_per_token: int
    attention_flops_per_token: float
    query_and_partial_bytes: int
    reserved_weight_bytes_per_gpu: int = 0
    vram_utilization: float = 0.85


@dataclass(frozen=True)
class SplitKvAssignment:
    """Per-device Split-KV decode attention token shard and timing breakdown."""

    rank: int
    role: DeviceRole
    kv_tokens: int
    estimated_local_attn_ms: float
    estimated_comm_ms: float
    estimated_total_ms: float
    kv_memory_bytes: int


@dataclass(frozen=True)
class RolePlan:
    """Combined Tensor Parallel + Split-KV reclamation execution plan."""

    tp_plan: WorkloadPlan
    split_kv_assignments: list[SplitKvAssignment]
    estimated_tp_ms: float
    estimated_attention_ms: float
    total_kv_tokens: int
    reclaimed_gpu_count: int
    cpu_reclaimed: bool
    active_regime: str


def _marshal_devices(devices: Sequence[DeviceProfile]) -> Any:
    if not devices:
        raise InvalidParamError("devices must not be empty")
    arr = (DeviceProfileC * len(devices))()
    for i, d in enumerate(devices):
        arr[i] = DeviceProfileC(
            rank=d.rank,
            vendor_id=d.vendor_id,
            device_id=d.device_id,
            device_type=DEVICE_TYPE_CODES.get(d.device_type, 0),
            vram_bytes=d.vram_bytes,
            compute_gflops=d.compute_gflops,
        )
    return arr


def _marshal_links(topology: Sequence[Link]) -> tuple[Any, int]:
    if not topology:
        return None, 0
    arr = (LinkInfo * len(topology))()
    for i, link in enumerate(topology):
        arr[i] = LinkInfo(
            src_rank=link.src,
            dst_rank=link.dst,
            bandwidth_gbps=link.gbps,
            transport=link.transport.value,
            loopback=1 if link.loopback else 0,
            latency_us=link.latency_us,
        )
    return arr, len(topology)


def _marshal_spec(spec: WorkloadSpec) -> WorkloadSpecC:
    cpu = HostCpuProfileC(
        ram_bytes=spec.host_cpu.ram_bytes,
        compute_gflops=spec.host_cpu.compute_gflops,
        ram_bandwidth_gbps=spec.host_cpu.ram_bandwidth_gbps,
        aggregate_pcie_gbps=spec.host_cpu.aggregate_pcie_gbps,
    )
    return WorkloadSpecC(
        total_units=spec.total_units,
        flops_per_unit=spec.flops_per_unit,
        bytes_per_unit=spec.bytes_per_unit,
        sync_bytes=spec.sync_bytes,
        io_multiplier=spec.io_multiplier,
        vram_utilization=spec.vram_utilization,
        streaming_overlap_gamma=spec.streaming_overlap_gamma,
        default_pcie_gbps=spec.default_pcie_gbps,
        disk_bandwidth_gbps=spec.disk_bandwidth_gbps,
        disk_capacity_bytes=spec.disk_capacity_bytes,
        micro_batches=spec.micro_batches,
        allow_spilling=1 if spec.allow_spilling else 0,
        host_cpu=cpu,
    )


def _unmarshal_plan(
    hdr: WorkloadPlanHeaderC,
    alloc_arr: Any,
    active_arr: Any,
    cap: int,
) -> WorkloadPlan:
    alloc_count = min(cap, int(hdr.allocation_count))
    active_count = min(cap, int(hdr.active_rank_count))
    allocations = [
        ShardAllocation(
            rank=int(alloc_arr[i].rank),
            units=int(alloc_arr[i].units),
            share_ratio=float(alloc_arr[i].share_ratio),
            estimated_compute_ms=float(alloc_arr[i].estimated_compute_ms),
            estimated_memory_bytes=int(alloc_arr[i].estimated_memory_bytes),
            excluded_as_straggler=alloc_arr[i].excluded_as_straggler != 0,
            vram_units=int(alloc_arr[i].vram_units),
            host_stream_units=int(alloc_arr[i].host_stream_units),
            disk_stream_units=int(alloc_arr[i].disk_stream_units),
            cpu_in_place_units=int(alloc_arr[i].cpu_in_place_units),
        )
        for i in range(alloc_count)
    ]
    active_ranks = [int(active_arr[i]) for i in range(active_count)]
    return WorkloadPlan(
        total_units=int(hdr.total_units),
        allocations=allocations,
        active_ranks=active_ranks,
        estimated_makespan_ms=float(hdr.estimated_makespan_ms),
        estimated_compute_ms=float(hdr.estimated_compute_ms),
        estimated_sync_ms=float(hdr.estimated_sync_ms),
    )


def plan_work(
    devices: Sequence[DeviceProfile],
    topology: Sequence[Link],
    spec: WorkloadSpec,
) -> WorkloadPlan:
    """Invokes the native Rust Theorems 1-4 optimizer directly via ``hc4j_optimizer_plan_work``."""
    lib = get_library()
    dev_arr = _marshal_devices(devices)
    link_arr, link_count = _marshal_links(topology)
    spec_c = _marshal_spec(spec)
    cap = len(devices) + 1
    hdr = WorkloadPlanHeaderC()
    alloc_arr = (ShardAllocationC * cap)()
    active_arr = (ctypes.c_int32 * cap)()

    check_status(
        lib.hc4j_optimizer_plan_work(
            len(devices),
            dev_arr,
            link_count,
            link_arr,
            ctypes.byref(spec_c),
            ctypes.byref(hdr),
            alloc_arr,
            cap,
            active_arr,
            cap,
        ),
        "hc4j_optimizer_plan_work",
        lib,
    )
    return _unmarshal_plan(hdr, alloc_arr, active_arr, cap)


def plan_roles(
    devices: Sequence[DeviceProfile],
    topology: Sequence[Link],
    tp_spec: WorkloadSpec,
    kv_spec: SplitKvSpec,
) -> RolePlan:
    """Invokes the native Rust Tensor Parallel + Split-KV optimizer via ``hc4j_optimizer_plan_roles``."""
    lib = get_library()
    dev_arr = _marshal_devices(devices)
    link_arr, link_count = _marshal_links(topology)
    tp_c = _marshal_spec(tp_spec)
    kv_c = SplitKvSpecC(
        total_kv_tokens=kv_spec.total_kv_tokens,
        kv_bytes_per_token=kv_spec.kv_bytes_per_token,
        attention_flops_per_token=kv_spec.attention_flops_per_token,
        query_and_partial_bytes=kv_spec.query_and_partial_bytes,
        reserved_weight_bytes_per_gpu=kv_spec.reserved_weight_bytes_per_gpu,
        vram_utilization=kv_spec.vram_utilization,
    )
    cap = len(devices) + 1
    hdr = RolePlanHeaderC()
    tp_alloc = (ShardAllocationC * cap)()
    tp_active = (ctypes.c_int32 * cap)()
    kv_assign = (SplitKvAssignmentC * cap)()

    check_status(
        lib.hc4j_optimizer_plan_roles(
            len(devices),
            dev_arr,
            link_count,
            link_arr,
            ctypes.byref(tp_c),
            ctypes.byref(kv_c),
            ctypes.byref(hdr),
            tp_alloc,
            cap,
            tp_active,
            cap,
            kv_assign,
            cap,
        ),
        "hc4j_optimizer_plan_roles",
        lib,
    )
    tp_plan = _unmarshal_plan(hdr.tp_header, tp_alloc, tp_active, cap)
    kv_count = min(cap, int(hdr.split_kv_count))
    assignments = [
        SplitKvAssignment(
            rank=int(kv_assign[i].rank),
            role=DeviceRole(int(kv_assign[i].role)),
            kv_tokens=int(kv_assign[i].kv_tokens),
            estimated_local_attn_ms=float(kv_assign[i].estimated_local_attn_ms),
            estimated_comm_ms=float(kv_assign[i].estimated_comm_ms),
            estimated_total_ms=float(kv_assign[i].estimated_total_ms),
            kv_memory_bytes=int(kv_assign[i].kv_memory_bytes),
        )
        for i in range(kv_count)
    ]
    return RolePlan(
        tp_plan=tp_plan,
        split_kv_assignments=assignments,
        estimated_tp_ms=float(hdr.estimated_tp_ms),
        estimated_attention_ms=float(hdr.estimated_attention_ms),
        total_kv_tokens=int(hdr.total_kv_tokens),
        reclaimed_gpu_count=int(hdr.reclaimed_gpu_count),
        cpu_reclaimed=hdr.cpu_reclaimed != 0,
        active_regime=_REGIME_NAMES.get(int(hdr.regime_code), "UNKNOWN"),
    )
