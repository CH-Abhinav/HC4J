# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Native FFI status codes, rich ``hc4j_last_error`` diagnostics, and typed exceptions."""

from __future__ import annotations

import ctypes

HC4J_SUCCESS = 0
HC4J_ERR_NOT_FOUND = -1
HC4J_ERR_INVALID_PARAM = -2
HC4J_ERR_GPU_READBACK = -3
HC4J_ERR_OUT_OF_MEMORY = -4
HC4J_ERR_DEVICE = -5
HC4J_ERR_GPU_VALIDATION = -6
HC4J_ERR_HOST_IO = -7
HC4J_ERR_UNSUPPORTED = -8


class Hc4xError(RuntimeError):
    """Base exception raised for all hc4x native and SDK failures."""

    def __init__(self, message: str, status_code: int | None = None, detail: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class EngineError(Hc4xError):
    """Native GPU engine, driver, or runtime handle failure."""


class NotFoundError(EngineError):
    """VRAM buffer handle was not found in the native registry."""


class InvalidParamError(Hc4xError, ValueError):
    """Invalid shape, pointer, handle, or parameter passed across the FFI boundary."""


class ReadbackError(EngineError):
    """GPU readback / buffer mapping failed."""


class OutOfMemoryError(Hc4xError, MemoryError):
    """VRAM, host RAM, and disk spill tiers are all exhausted."""


class DeviceError(EngineError):
    """GPU device unavailable, lost, or timed out."""


class GpuValidationError(EngineError):
    """WebGPU/Vulkan rejected shader compilation, pipeline creation, or submission."""


class HostIoError(EngineError, OSError):
    """Tier-2 disk spill I/O operation failed."""


class UnsupportedError(Hc4xError):
    """Valid operation or layout not supported on the active device/kernel."""


_STATUS_MAP: dict[int, tuple[type[Hc4xError], str]] = {
    HC4J_ERR_NOT_FOUND: (NotFoundError, "GPU buffer ID not found in native registry"),
    HC4J_ERR_INVALID_PARAM: (InvalidParamError, "Invalid parameters passed to native bridge"),
    HC4J_ERR_GPU_READBACK: (ReadbackError, "GPU readback failed during memory mapping"),
    HC4J_ERR_OUT_OF_MEMORY: (OutOfMemoryError, "VRAM, host RAM, and disk spill tiers are exhausted"),
    HC4J_ERR_DEVICE: (DeviceError, "GPU device unavailable, lost, or timed out"),
    HC4J_ERR_GPU_VALIDATION: (GpuValidationError, "wgpu rejected shader, pipeline, or submission"),
    HC4J_ERR_HOST_IO: (HostIoError, "Host spill-file I/O failed"),
    HC4J_ERR_UNSUPPORTED: (UnsupportedError, "Operation not supported for this tensor layout or size"),
}


def read_last_error(lib: ctypes.CDLL | None = None) -> str:
    """Fetches thread-local diagnostic message from ``hc4j_last_error``."""
    if lib is None:
        return ""
    try:
        cap = 512
        buf = (ctypes.c_uint8 * cap)()
        n = int(lib.hc4j_last_error(buf, cap))
        if n <= 0:
            return ""
        clamped = min(n, cap - 1)
        return bytes(buf[:clamped]).decode("utf-8", errors="replace")
    except Exception:
        return ""


def check_status(status_code: int, context: str, lib: ctypes.CDLL | None = None) -> int:
    """Checks an ``int32_t`` FFI return status, raising a typed exception on ``< 0``."""
    code = int(status_code)
    if code >= HC4J_SUCCESS:
        return code
    detail = read_last_error(lib)
    exc_cls, summary = _STATUS_MAP.get(code, (EngineError, f"Unknown native error code {code}"))
    suffix = f" [{detail}]" if detail else ""
    raise exc_cls(f"{context}: {summary}{suffix}", status_code=code, detail=detail)
