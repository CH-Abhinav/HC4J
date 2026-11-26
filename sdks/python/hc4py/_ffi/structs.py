# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Exact 1-to-1 ``ctypes.Structure`` definitions matching ``native/include/hc4x.h``."""

from __future__ import annotations

import ctypes


class MemStats(ctypes.Structure):
    """Tiered memory telemetry snapshot (168 bytes)."""

    _fields_ = [
        ("vram_budget", ctypes.c_uint64),
        ("vram_used", ctypes.c_uint64),
        ("host_budget", ctypes.c_uint64),
        ("host_used", ctypes.c_uint64),
        ("spilled_bytes", ctypes.c_uint64),
        ("tensors_device", ctypes.c_uint64),
        ("tensors_host", ctypes.c_uint64),
        ("tensors_evicted", ctypes.c_uint64),
        ("evictions", ctypes.c_uint64),
        ("page_ins", ctypes.c_uint64),
        ("spills", ctypes.c_uint64),
        ("driver_ooms", ctypes.c_uint64),
        ("streamed_ops", ctypes.c_uint64),
        ("allocations", ctypes.c_uint64),
        ("dedicated_buffers", ctypes.c_uint64),
        ("slabs_created", ctypes.c_uint64),
        ("region_allocs", ctypes.c_uint64),
        ("slab_count", ctypes.c_uint64),
        ("slab_bytes", ctypes.c_uint64),
        ("slab_regions", ctypes.c_uint64),
        ("quarantined", ctypes.c_uint64),
    ]


class EngineStats(ctypes.Structure):
    """Command stream and adapter telemetry snapshot (104 bytes)."""

    _fields_ = [
        ("submissions", ctypes.c_uint64),
        ("dispatches", ctypes.c_uint64),
        ("kernel_bytes", ctypes.c_uint64),
        ("host_to_device_bytes", ctypes.c_uint64),
        ("device_to_host_bytes", ctypes.c_uint64),
        ("batches", ctypes.c_uint64),
        ("completed_epoch", ctypes.c_uint64),
        ("submitted_epoch", ctypes.c_uint64),
        ("features", ctypes.c_uint64),
        ("device_type", ctypes.c_uint64),
        ("backend", ctypes.c_uint64),
        ("subgroup_min_size", ctypes.c_uint64),
        ("subgroup_max_size", ctypes.c_uint64),
    ]


class AdapterInfoC(ctypes.Structure):
    """Active WebGPU adapter descriptor (668 bytes)."""

    _fields_ = [
        ("vendor_id", ctypes.c_uint32),
        ("device_id", ctypes.c_uint32),
        ("device_type", ctypes.c_uint32),
        ("backend", ctypes.c_uint32),
        ("name_len", ctypes.c_uint32),
        ("driver_len", ctypes.c_uint32),
        ("driver_info_len", ctypes.c_uint32),
        ("name", ctypes.c_uint8 * 256),
        ("driver", ctypes.c_uint8 * 128),
        ("driver_info", ctypes.c_uint8 * 256),
    ]


class LinkInfo(ctypes.Structure):
    """Measured directed collective interconnect link (32 bytes)."""

    _fields_ = [
        ("src_rank", ctypes.c_uint32),
        ("dst_rank", ctypes.c_uint32),
        ("bandwidth_gbps", ctypes.c_double),
        ("transport", ctypes.c_uint32),
        ("loopback", ctypes.c_uint32),
        ("latency_us", ctypes.c_double),
    ]


class DeviceProfileC(ctypes.Structure):
    """Hardware compute/VRAM capability profile for one rank (40 bytes)."""

    _fields_ = [
        ("rank", ctypes.c_uint32),
        ("vendor_id", ctypes.c_uint32),
        ("device_id", ctypes.c_uint32),
        ("device_type", ctypes.c_uint32),
        ("vram_bytes", ctypes.c_uint64),
        ("compute_gflops", ctypes.c_double),
        ("vram_bandwidth_gbps", ctypes.c_double),
    ]


class IslandHeaderC(ctypes.Structure):
    """Header for one detected fast collective island (16 bytes)."""

    _fields_ = [
        ("id", ctypes.c_uint32),
        ("leader_rank", ctypes.c_uint32),
        ("internal_transport", ctypes.c_uint32),
        ("member_count", ctypes.c_uint32),
    ]


class HostCpuProfileC(ctypes.Structure):
    """Host CPU capabilities for Tier 3 in-place compute and Tier 1 spilling (32 bytes)."""

    _fields_ = [
        ("ram_bytes", ctypes.c_uint64),
        ("compute_gflops", ctypes.c_double),
        ("ram_bandwidth_gbps", ctypes.c_double),
        ("aggregate_pcie_gbps", ctypes.c_double),
    ]


class WorkloadSpecC(ctypes.Structure):
    """Parallel workload specification for Theorems 1-4 optimization (120 bytes)."""

    _fields_ = [
        ("total_units", ctypes.c_uint64),
        ("flops_per_unit", ctypes.c_double),
        ("bytes_per_unit", ctypes.c_uint64),
        ("sync_bytes", ctypes.c_uint64),
        ("io_multiplier", ctypes.c_double),
        ("vram_utilization", ctypes.c_double),
        ("streaming_overlap_gamma", ctypes.c_double),
        ("default_pcie_gbps", ctypes.c_double),
        ("disk_bandwidth_gbps", ctypes.c_double),
        ("disk_capacity_bytes", ctypes.c_uint64),
        ("micro_batches", ctypes.c_uint32),
        ("allow_spilling", ctypes.c_uint32),
        ("host_cpu", HostCpuProfileC),
    ]


class ShardAllocationC(ctypes.Structure):
    """Per-device work partition returned by hc4j_optimizer_plan_work (72 bytes)."""

    _fields_ = [
        ("rank", ctypes.c_int32),
        ("excluded_as_straggler", ctypes.c_uint32),
        ("units", ctypes.c_uint64),
        ("share_ratio", ctypes.c_double),
        ("estimated_compute_ms", ctypes.c_double),
        ("estimated_memory_bytes", ctypes.c_uint64),
        ("vram_units", ctypes.c_uint64),
        ("host_stream_units", ctypes.c_uint64),
        ("disk_stream_units", ctypes.c_uint64),
        ("cpu_in_place_units", ctypes.c_uint64),
    ]


class WorkloadPlanHeaderC(ctypes.Structure):
    """Summary header returned by hc4j_optimizer_plan_work (40 bytes)."""

    _fields_ = [
        ("total_units", ctypes.c_uint64),
        ("estimated_makespan_ms", ctypes.c_double),
        ("estimated_compute_ms", ctypes.c_double),
        ("estimated_sync_ms", ctypes.c_double),
        ("allocation_count", ctypes.c_uint32),
        ("active_rank_count", ctypes.c_uint32),
    ]


class SplitKvSpecC(ctypes.Structure):
    """KV-cache decode workload specification for hc4j_optimizer_plan_roles (48 bytes)."""

    _fields_ = [
        ("total_kv_tokens", ctypes.c_uint64),
        ("kv_bytes_per_token", ctypes.c_uint64),
        ("attention_flops_per_token", ctypes.c_double),
        ("query_and_partial_bytes", ctypes.c_uint64),
        ("reserved_weight_bytes_per_gpu", ctypes.c_uint64),
        ("vram_utilization", ctypes.c_double),
    ]


class SplitKvAssignmentC(ctypes.Structure):
    """Per-device Split-KV worker role assignment (48 bytes)."""

    _fields_ = [
        ("rank", ctypes.c_int32),
        ("role", ctypes.c_uint32),
        ("kv_tokens", ctypes.c_uint64),
        ("estimated_local_attn_ms", ctypes.c_double),
        ("estimated_comm_ms", ctypes.c_double),
        ("estimated_total_ms", ctypes.c_double),
        ("kv_memory_bytes", ctypes.c_uint64),
    ]


class RolePlanHeaderC(ctypes.Structure):
    """Combined Tensor-Parallel + Split-KV RolePlan header (80 bytes)."""

    _fields_ = [
        ("tp_header", WorkloadPlanHeaderC),
        ("estimated_tp_ms", ctypes.c_double),
        ("estimated_attention_ms", ctypes.c_double),
        ("total_kv_tokens", ctypes.c_uint64),
        ("split_kv_count", ctypes.c_uint32),
        ("reclaimed_gpu_count", ctypes.c_uint32),
        ("cpu_reclaimed", ctypes.c_uint32),
        ("regime_code", ctypes.c_uint32),
    ]
