# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""GPU engine lifecycle, command batching scope, memory telemetry, and adapter metadata."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from ._ffi import AdapterInfoC, EngineStats, MemStats, check_status, get_library, read_last_error

_INIT_LOCK = threading.Lock()
_INITIALIZED = False

_DEVICE_TYPES = {
    0: "Other",
    1: "Integrated",
    2: "Discrete",
    3: "Virtual",
    4: "CPU",
}

_BACKENDS = {
    0: "Noop",
    1: "Vulkan",
    2: "Metal",
    3: "Dx12",
    4: "Gl",
    5: "WebGPU",
}


@dataclass(frozen=True)
class AdapterInfo:
    """Active WebGPU / Vulkan adapter metadata returned by ``hc4j_adapter_info``."""

    vendor_id: int
    device_id: int
    device_type: str
    backend: str
    name: str
    driver: str
    driver_info: str


@dataclass(frozen=True)
class EngineTelemetry:
    """Command submission and adapter capabilities snapshot from ``hc4j_engine_stats``."""

    submissions: int
    dispatches: int
    kernel_bytes: int
    host_to_device_bytes: int
    device_to_host_bytes: int
    batches: int
    completed_epoch: int
    submitted_epoch: int
    features: int
    device_type: str
    backend: str
    subgroup_min_size: int
    subgroup_max_size: int


@dataclass(frozen=True)
class MemoryTelemetry:
    """Three-tier VRAM/Host-RAM/Disk allocator snapshot from ``hc4j_mem_stats``."""

    vram_budget: int
    vram_used: int
    host_budget: int
    host_used: int
    spilled_bytes: int
    tensors_device: int
    tensors_host: int
    tensors_evicted: int
    evictions: int
    page_ins: int
    spills: int
    driver_ooms: int
    streamed_ops: int
    allocations: int
    dedicated_buffers: int
    slabs_created: int
    region_allocs: int
    slab_count: int
    slab_bytes: int
    slab_regions: int
    quarantined: int


def init() -> None:
    """Initializes the global WebGPU engine idempotently."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        lib = get_library()
        check_status(lib.hc4j_init_gpu(), "hc4j_init_gpu", lib)
        _INITIALIZED = True


def ensure_initialized() -> None:
    """Lazily initializes the GPU engine if not yet initialized."""
    if not _INITIALIZED:
        init()


def is_gpu_available() -> bool:
    """Returns ``True`` if a hardware/software WebGPU adapter initializes successfully."""
    try:
        ensure_initialized()
        return True
    except Exception:
        return False


def synchronize() -> None:
    """Flushes and blocks until all pending GPU compute work completes."""
    ensure_initialized()
    lib = get_library()
    check_status(lib.hc4j_synchronize(), "hc4j_synchronize", lib)


def last_error() -> str:
    """Returns the thread-local error diagnostic string from ``hc4j_last_error``."""
    return read_last_error(get_library())


def configure_memory(vram_budget_bytes: int = 0, host_budget_bytes: int = 0) -> None:
    """Configures soft Tier-0 VRAM and Tier-1 host RAM budgets (0 = keep current)."""
    if vram_budget_bytes < 0 or host_budget_bytes < 0:
        raise ValueError("Memory budgets must be non-negative")
    ensure_initialized()
    lib = get_library()
    check_status(lib.hc4j_mem_configure(vram_budget_bytes, host_budget_bytes), "hc4j_mem_configure", lib)


def adapter_info() -> AdapterInfo:
    """Queries the active WebGPU adapter descriptor from ``hc4j_adapter_info``."""
    ensure_initialized()
    lib = get_library()
    raw = AdapterInfoC()
    check_status(lib.hc4j_adapter_info(raw), "hc4j_adapter_info", lib)
    return AdapterInfo(
        vendor_id=raw.vendor_id,
        device_id=raw.device_id,
        device_type=_DEVICE_TYPES.get(raw.device_type, "Other"),
        backend=_BACKENDS.get(raw.backend, "Unknown"),
        name=bytes(raw.name[: min(raw.name_len, 256)]).decode("utf-8", errors="replace"),
        driver=bytes(raw.driver[: min(raw.driver_len, 128)]).decode("utf-8", errors="replace"),
        driver_info=bytes(raw.driver_info[: min(raw.driver_info_len, 256)]).decode("utf-8", errors="replace"),
    )


def engine_stats() -> EngineTelemetry:
    """Reads live command stream and adapter capability counters from ``hc4j_engine_stats``."""
    ensure_initialized()
    lib = get_library()
    raw = EngineStats()
    check_status(lib.hc4j_engine_stats(raw), "hc4j_engine_stats", lib)
    return EngineTelemetry(
        submissions=raw.submissions,
        dispatches=raw.dispatches,
        kernel_bytes=raw.kernel_bytes,
        host_to_device_bytes=raw.host_to_device_bytes,
        device_to_host_bytes=raw.device_to_host_bytes,
        batches=raw.batches,
        completed_epoch=raw.completed_epoch,
        submitted_epoch=raw.submitted_epoch,
        features=raw.features,
        device_type=_DEVICE_TYPES.get(raw.device_type, "Other"),
        backend=_BACKENDS.get(raw.backend, "Unknown"),
        subgroup_min_size=raw.subgroup_min_size,
        subgroup_max_size=raw.subgroup_max_size,
    )


def mem_stats() -> MemoryTelemetry:
    """Reads tiered memory pool and spill counters from ``hc4j_mem_stats``."""
    ensure_initialized()
    lib = get_library()
    raw = MemStats()
    check_status(lib.hc4j_mem_stats(raw), "hc4j_mem_stats", lib)
    return MemoryTelemetry(
        vram_budget=raw.vram_budget,
        vram_used=raw.vram_used,
        host_budget=raw.host_budget,
        host_used=raw.host_used,
        spilled_bytes=raw.spilled_bytes,
        tensors_device=raw.tensors_device,
        tensors_host=raw.tensors_host,
        tensors_evicted=raw.tensors_evicted,
        evictions=raw.evictions,
        page_ins=raw.page_ins,
        spills=raw.spills,
        driver_ooms=raw.driver_ooms,
        streamed_ops=raw.streamed_ops,
        allocations=raw.allocations,
        dedicated_buffers=raw.dedicated_buffers,
        slabs_created=raw.slabs_created,
        region_allocs=raw.region_allocs,
        slab_count=raw.slab_count,
        slab_bytes=raw.slab_bytes,
        slab_regions=raw.slab_regions,
        quarantined=raw.quarantined,
    )


@contextmanager
def batch() -> Iterator[None]:
    """Context manager batching GPU dispatches into a single command buffer submission."""
    ensure_initialized()
    lib = get_library()
    check_status(lib.hc4j_batch_begin(), "hc4j_batch_begin", lib)
    try:
        yield
    finally:
        check_status(lib.hc4j_batch_end(), "hc4j_batch_end", lib)
