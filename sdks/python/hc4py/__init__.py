# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""hc4py — Thin, zero-logic Python SDK over ``hc4x-ffi``."""

from __future__ import annotations

from . import _ffi, collective, dtype, engine, fusion, linalg, math, nn, optimizer, tensor
from ._ffi import (
    DeviceError,
    EngineError,
    GpuValidationError,
    Hc4xError,
    HostIoError,
    InvalidParamError,
    NotFoundError,
    OutOfMemoryError,
    ReadbackError,
    UnsupportedError,
)
from .collective import Collective, CollectiveBuffer, DeviceProfile, IslandInfo, Link, Transport, merge_split_kv
from .dtype import DType
from .engine import (
    AdapterInfo,
    EngineTelemetry,
    MemoryTelemetry,
    adapter_info,
    batch,
    configure_memory,
    engine_stats,
    init,
    is_gpu_available,
    last_error,
    mem_stats,
    synchronize,
)
from .fusion import FusedExpr
from .linalg import Kernel, MatmulPlan, matmul, matmul_plan, matmul_transposed_b, transpose
from .nn import Activation, Reduction
from .optimizer import (
    HOST_CPU_RANK,
    DeviceRole,
    HostCpuProfile,
    RolePlan,
    ShardAllocation,
    SplitKvAssignment,
    SplitKvSpec,
    WorkloadPlan,
    WorkloadSpec,
    plan_roles,
    plan_work,
)
from .tensor import Tensor

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "HOST_CPU_RANK",
    "Activation",
    "AdapterInfo",
    "Collective",
    "CollectiveBuffer",
    "DType",
    "DeviceError",
    "DeviceProfile",
    "DeviceRole",
    "EngineError",
    "EngineTelemetry",
    "FusedExpr",
    "GpuValidationError",
    "Hc4xError",
    "HostCpuProfile",
    "HostIoError",
    "InvalidParamError",
    "IslandInfo",
    "Kernel",
    "Link",
    "MatmulPlan",
    "MemoryTelemetry",
    "NotFoundError",
    "OutOfMemoryError",
    "ReadbackError",
    "Reduction",
    "RolePlan",
    "ShardAllocation",
    "SplitKvAssignment",
    "SplitKvSpec",
    "Tensor",
    "Transport",
    "UnsupportedError",
    "WorkloadPlan",
    "WorkloadSpec",
    "_ffi",
    "adapter_info",
    "batch",
    "collective",
    "configure_memory",
    "dtype",
    "engine",
    "engine_stats",
    "fusion",
    "init",
    "is_gpu_available",
    "last_error",
    "linalg",
    "math",
    "matmul",
    "matmul_plan",
    "matmul_transposed_b",
    "mem_stats",
    "merge_split_kv",
    "nn",
    "optimizer",
    "plan_roles",
    "plan_work",
    "synchronize",
    "tensor",
    "transpose",
]
