# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Low-level ctypes bindings and struct definitions mirroring ``native/include/hc4x.h``."""

from .bindings import BOUND_SYMBOLS, bind_all
from .loader import get_library, loaded_library_path
from .status import (
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
    check_status,
    read_last_error,
)
from .structs import (
    AdapterInfoC,
    DeviceProfileC,
    EngineStats,
    HostCpuProfileC,
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

__all__ = [
    "BOUND_SYMBOLS",
    "AdapterInfoC",
    "DeviceError",
    "DeviceProfileC",
    "EngineError",
    "EngineStats",
    "GpuValidationError",
    "Hc4xError",
    "HostCpuProfileC",
    "HostIoError",
    "InvalidParamError",
    "IslandHeaderC",
    "LinkInfo",
    "MemStats",
    "NotFoundError",
    "OutOfMemoryError",
    "ReadbackError",
    "RolePlanHeaderC",
    "ShardAllocationC",
    "SplitKvAssignmentC",
    "SplitKvSpecC",
    "UnsupportedError",
    "WorkloadPlanHeaderC",
    "WorkloadSpecC",
    "bind_all",
    "check_status",
    "get_library",
    "loaded_library_path",
    "read_last_error",
]
