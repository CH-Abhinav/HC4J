# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Explicit ctypes argtypes/restype bindings for all 107 C-ABI functions in ``native/include/hc4x.h``."""

from __future__ import annotations

import ctypes
from typing import Any

from .structs import (
    AdapterInfoC,
    DeviceProfileC,
    EngineStats,
    IslandHeaderC,
    LinkInfo,
    MemStats,
    RolePlanHeaderC,
    ShardAllocationC,
    SplitKvAssignmentC,
    SplitKvSpecC,
    WorkloadPlanHeaderC,
    WorkloadSpecC,
)

BOUND_SYMBOLS: list[str] = []


def _bind(lib: ctypes.CDLL, name: str, argtypes: list[Any], restype: Any) -> None:
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = restype


def bind_all(lib: ctypes.CDLL) -> None:
    """Sets ``argtypes`` and ``restype`` for every exported symbol in ``native/include/hc4x.h``."""
    u32 = ctypes.c_uint32
    i32 = ctypes.c_int32
    u64 = ctypes.c_uint64
    f32 = ctypes.c_float
    usz = ctypes.c_size_t
    voidp = ctypes.c_void_p

    # 4. Engine & Error Diagnostics (7 symbols)
    specs: list[tuple[str, list[Any], Any]] = [
        ("hc4j_init_gpu", [], i32),
        ("hc4j_batch_begin", [], i32),
        ("hc4j_batch_end", [], i32),
        ("hc4j_synchronize", [], i32),
        ("hc4j_engine_stats", [ctypes.POINTER(EngineStats)], i32),
        ("hc4j_adapter_info", [ctypes.POINTER(AdapterInfoC)], i32),
        ("hc4j_last_error", [voidp, u32], i32),
        # 5. Tiered Tensor Memory Management (12 symbols)
        ("hc4j_gpu_alloc", [usz], u64),
        ("hc4j_gpu_alloc_uninit", [usz], u64),
        ("hc4j_gpu_alloc_bytes", [u64, u32], u64),
        ("hc4j_gpu_write_bytes", [u64, voidp, usz], i32),
        ("hc4j_gpu_download_bytes", [u64, voidp, usz], i32),
        ("hc4j_gpu_write", [u64, voidp, usz], i32),
        ("hc4j_gpu_free", [u64], i32),
        ("hc4j_gpu_download", [u64, voidp, usz], i32),
        ("hc4j_mem_configure", [u64, u64], i32),
        ("hc4j_mem_stats", [ctypes.POINTER(MemStats)], i32),
        ("hc4j_mem_residency", [u64], i32),
        ("hc4j_mem_evict", [u64], i32),
    ]

    # 6. Math, Elementwise, Compare, Selection & Type Casts (43 symbols)
    bin_typed = [u64, u64, u64, u32, voidp, voidp, voidp, voidp, usz, u32, u32]
    bin_f32 = [u64, u64, u64, u32, voidp, voidp, voidp, voidp, usz, u32]
    un_f32 = [u64, u64, u32, voidp, voidp, voidp, usz, u32]

    for op in ("add", "sub", "mul", "div"):
        specs.append((f"dispatch_{op}", bin_typed, i32))
        specs.append((f"dispatch_{op}_f32", bin_f32, i32))

    for trig in (
        "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    ):
        specs.append((f"dispatch_{trig}_f32", un_f32, i32))

    for exp_op in ("exp", "ln", "log2", "log10", "sqrt"):
        specs.append((f"dispatch_{exp_op}_f32", un_f32, i32))

    for umath in ("abs", "neg", "rsqrt", "floor", "ceil", "round", "sign"):
        specs.append((f"dispatch_{umath}_f32", un_f32, i32))

    specs.extend([
        ("dispatch_clamp_f32", [*un_f32, f32, f32], i32),
        ("dispatch_unary_i32", [*un_f32, u32, u32, u32], i32),
        ("dispatch_pow_f32", bin_f32, i32),
        ("dispatch_maximum", bin_typed, i32),
        ("dispatch_minimum", bin_typed, i32),
        ("dispatch_compare", [*bin_typed, u32], i32),
        ("dispatch_where", [u64, u64, u64, u64, u32, voidp, voidp, voidp, voidp, voidp, usz, u32, u32, u32], i32),
        ("dispatch_cast", [u64, u64, u64, u32, u32], i32),
        ("dispatch_cast_f32_to_f16", [u64, u64, u64], i32),
        ("dispatch_cast_f16_to_f32", [u64, u64, u64], i32),
        ("dispatch_copy_strided", [u64, u64, u64, u32, voidp, voidp, usz], i32),
        # 7. Linear Algebra (6 symbols)
        ("dispatch_matmul_f32", [u64, u64, u64, u32, u32, u32], i32),
        ("dispatch_matmul_f32_ex", [u64, u64, u64, u32, u32, u32, u32], i32),
        ("dispatch_batched_matmul_f32", [u64, u64, u64, u32, u32, u32, u32, voidp, voidp, voidp, u32], i32),
        ("dispatch_matmul_f32_f16", [u64, u64, u64, u32, u32, u32, u32], i32),
        ("dispatch_transpose_f32", [u64, u64, u32, u32], i32),
        ("hc4j_matmul_plan", [u32, u32, u32, u32], i32),
        # 8. Neural Networks, Reductions & Fused Kernels (17 symbols)
        ("dispatch_activation_f32", [*un_f32, u32, u32], i32),
        ("dispatch_softmax_rows_f32", [u64, u64, u32, u32, u32], i32),
        ("dispatch_layernorm_rows_f32", [u64, u64, u64, u64, u32, u32, u32], i32),
        ("dispatch_rmsnorm_rows_f32", [u64, u64, u64, u32, u32, u32], i32),
        ("dispatch_cross_entropy_rows", [u64, u64, u64, u32, u32], i32),
        ("dispatch_add_row_bias_f32", [u64, u64, u64, u32, u32], i32),
        ("dispatch_kv_cache_append_f32", [u64, u64, u32, u32, u32, u32, u32], i32),
        ("dispatch_embedding", [u64, u64, u64, u32, u32, u32, u32], i32),
        ("dispatch_rope_f32", [u64, u64, u32, u32, u32, u32, u32, u32], i32),
        ("dispatch_reduce_f32", [u64, u64, u64, u32], i32),
        ("dispatch_reduce_square_f32", [u64, u64, u64, u32], i32),
        ("dispatch_reduce_diff2_f32", [u64, u64, u64, u64, u32], i32),
        ("dispatch_reduce_rows_f32", [u64, u64, u32, u32, u32], i32),
        ("dispatch_argmax_rows", [u64, u64, u32, u32, u32], i32),
        ("dispatch_reduce_axis_f32", [u64, u64, u32, u32, u32, u32], i32),
        ("dispatch_argmax_axis", [u64, u64, u32, u32, u32, u32], i32),
        ("hc4j_fused_dispatch", [voidp, u32, voidp, u32, u64], i32),
        # 9. Collective Runtime & Lossless Split-KV Merge (20 symbols)
        ("hc4j_collective_create", [u32, ctypes.POINTER(u64)], i32),
        ("hc4j_collective_destroy", [u64], i32),
        ("hc4j_collective_ranks", [u64], i32),
        ("hc4j_collective_link_count", [u64], i32),
        ("hc4j_collective_link", [u64, u32, ctypes.POINTER(LinkInfo)], i32),
        ("hc4j_collective_device_name", [u64, u32, voidp, u32], i32),
        ("hc4j_collective_backend_name", [u64, voidp, u32], i32),
        ("hc4j_collective_alloc", [u64, u32, u64, ctypes.POINTER(u64)], i32),
        ("hc4j_collective_free", [u64, u64], i32),
        ("hc4j_collective_write", [u64, u64, voidp, u64], i32),
        ("hc4j_collective_read", [u64, u64, voidp, u64], i32),
        ("hc4j_collective_allreduce_sum", [u64, voidp, u32], i32),
        ("hc4j_collective_allreduce_sum_subset", [u64, voidp, u32], i32),
        ("hc4j_collective_gemm_bias_act", [u64, u32, u64, u64, u64, u64, u32, u32, u32, u32], i32),
        ("hc4j_collective_split_kv_attn", [u64, u32, u64, u64, u64, u64, u64, u32, u32, u32, u32, u32], i32),
        ("hc4j_collective_device_profile", [u64, u32, ctypes.POINTER(DeviceProfileC)], i32),
        ("hc4j_collective_ring_order", [u64, voidp, u32], i32),
        ("hc4j_collective_island_count", [u64], i32),
        ("hc4j_collective_island", [u64, u32, ctypes.POINTER(IslandHeaderC), voidp, u32], i32),
        ("hc4j_collective_merge_split_kv_f32", [u32, u32, u32, voidp, voidp, voidp], i32),
        # 10. Exact Theorems 1-4 Workload & Split-KV Role Optimizer (2 symbols)
        (
            "hc4j_optimizer_plan_work",
            [
                u32, ctypes.POINTER(DeviceProfileC),
                u32, ctypes.POINTER(LinkInfo),
                ctypes.POINTER(WorkloadSpecC),
                ctypes.POINTER(WorkloadPlanHeaderC),
                ctypes.POINTER(ShardAllocationC), u32,
                voidp, u32,
            ],
            i32,
        ),
        (
            "hc4j_optimizer_plan_roles",
            [
                u32, ctypes.POINTER(DeviceProfileC),
                u32, ctypes.POINTER(LinkInfo),
                ctypes.POINTER(WorkloadSpecC), ctypes.POINTER(SplitKvSpecC),
                ctypes.POINTER(RolePlanHeaderC),
                ctypes.POINTER(ShardAllocationC), u32,
                voidp, u32,
                ctypes.POINTER(SplitKvAssignmentC), u32,
            ],
            i32,
        ),
    ])

    BOUND_SYMBOLS.clear()
    for name, argtypes, restype in specs:
        _bind(lib, name, argtypes, restype)
        BOUND_SYMBOLS.append(name)
