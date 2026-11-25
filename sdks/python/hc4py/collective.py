# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU collective runtime, topology telemetry, and lossless Split-KV merge."""

from __future__ import annotations

import array
import ctypes
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from ._ffi import (
    DeviceProfileC,
    Hc4xError,
    InvalidParamError,
    IslandHeaderC,
    LinkInfo,
    check_status,
    get_library,
)

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore[assignment]

_DEVICE_TYPE_NAMES = {
    0: "Other",
    1: "Integrated",
    2: "Discrete",
    3: "Virtual",
    4: "CPU",
}

DEVICE_TYPE_CODES = {name: code for code, name in _DEVICE_TYPE_NAMES.items()}


class Transport(Enum):
    """Inter-rank transfer tier matching ``hc4x.h`` ``HC4J_TRANSPORT_*``."""

    HOST_STAGED = 0
    PEER_TO_PEER = 1
    HOST_SHARED_DMA = 2


@dataclass(frozen=True)
class Link:
    """Measured directed interconnect hop between two collective ranks."""

    src: int
    dst: int
    gbps: float
    transport: Transport
    loopback: bool
    latency_us: float


@dataclass(frozen=True)
class DeviceProfile:
    """Hardware compute/VRAM profile of a single rank or synthetic optimizer device."""

    rank: int
    name: str
    device_type: str
    vendor_id: int
    device_id: int
    vram_bytes: int
    compute_gflops: float


@dataclass(frozen=True)
class IslandInfo:
    """High-bandwidth cluster island extracted by native topology analysis."""

    id: int
    leader_rank: int
    internal_transport: Transport
    members: tuple[int, ...]


class CollectiveBuffer:
    """GPU buffer allocated on a specific rank inside a :class:`Collective`."""

    def __init__(self, collective: Collective, rank: int, buffer_id: int, elements: int) -> None:
        self._collective = collective
        self._rank = rank
        self._buffer_id = buffer_id
        self._elements = elements
        self._closed = False

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def elements(self) -> int:
        return self._elements

    @property
    def handle(self) -> int:
        self.ensure_open()
        return self._buffer_id

    def ensure_open(self) -> None:
        if self._closed or self._buffer_id == 0:
            raise Hc4xError("CollectiveBuffer has already been closed")
        self._collective.ensure_open()

    def write(self, data: Any) -> None:
        self.ensure_open()
        lib = get_library()
        if _np is not None and isinstance(data, _np.ndarray):
            arr = _np.ascontiguousarray(data, dtype=_np.float32)
            if arr.size != self._elements:
                raise InvalidParamError(f"Expected {self._elements} floats, got {arr.size}")
            check_status(
                lib.hc4j_collective_write(self._collective.handle, self._buffer_id, arr.ctypes.data, self._elements),
                "hc4j_collective_write",
                lib,
            )
            return
        flat = [float(x) for x in data]
        if len(flat) != self._elements:
            raise InvalidParamError(f"Expected {self._elements} floats, got {len(flat)}")
        arr = array.array("f", flat)
        ptr, _ = arr.buffer_info()
        check_status(
            lib.hc4j_collective_write(self._collective.handle, self._buffer_id, ptr, self._elements),
            "hc4j_collective_write",
            lib,
        )

    def read_list(self) -> list[float]:
        self.ensure_open()
        lib = get_library()
        arr = array.array("f", [0.0] * self._elements)
        ptr, _ = arr.buffer_info()
        check_status(
            lib.hc4j_collective_read(self._collective.handle, self._buffer_id, ptr, self._elements),
            "hc4j_collective_read",
            lib,
        )
        return arr.tolist()

    def read_numpy(self) -> Any:
        if _np is None:
            raise ImportError("Optional dependency 'numpy' is not installed")
        self.ensure_open()
        lib = get_library()
        arr = _np.empty((self._elements,), dtype=_np.float32)
        check_status(
            lib.hc4j_collective_read(self._collective.handle, self._buffer_id, arr.ctypes.data, self._elements),
            "hc4j_collective_read",
            lib,
        )
        return arr

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._buffer_id != 0 and not self._collective.is_closed:
                try:
                    get_library().hc4j_collective_free(self._collective.handle, self._buffer_id)
                except Exception:
                    pass
            self._buffer_id = 0

    def __enter__(self) -> CollectiveBuffer:
        self.ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Collective:
    """Multi-rank collective runtime backed by ``hc4x`` Vulkan/wgpu native P2P transport."""

    def __init__(self, ranks: int) -> None:
        if ranks <= 0:
            raise InvalidParamError(f"Collective ranks must be positive, got {ranks}")
        lib = get_library()
        out = ctypes.c_uint64(0)
        check_status(lib.hc4j_collective_create(ranks, ctypes.byref(out)), "hc4j_collective_create", lib)
        self._handle = int(out.value)
        self._ranks = int(ranks)
        self._closed = False
        self._lib = lib

    @property
    def ranks(self) -> int:
        return self._ranks

    @property
    def handle(self) -> int:
        self.ensure_open()
        return self._handle

    @property
    def is_closed(self) -> bool:
        return self._closed

    def ensure_open(self) -> None:
        if self._closed or self._handle == 0:
            raise Hc4xError("Collective has already been closed")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._handle != 0:
                try:
                    self._lib.hc4j_collective_destroy(self._handle)
                except Exception:
                    pass
                self._handle = 0

    def __enter__(self) -> Collective:
        self.ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def backend_name(self) -> str:
        self.ensure_open()
        buf = (ctypes.c_uint8 * 256)()
        n = check_status(
            self._lib.hc4j_collective_backend_name(self._handle, buf, 256),
            "hc4j_collective_backend_name",
            self._lib,
        )
        return bytes(buf[:n]).decode("utf-8", errors="replace")

    def device_name(self, rank: int) -> str:
        self.ensure_open()
        buf = (ctypes.c_uint8 * 256)()
        n = check_status(
            self._lib.hc4j_collective_device_name(self._handle, rank, buf, 256),
            "hc4j_collective_device_name",
            self._lib,
        )
        return bytes(buf[:n]).decode("utf-8", errors="replace")

    def device_profiles(self) -> list[DeviceProfile]:
        self.ensure_open()
        out: list[DeviceProfile] = []
        raw = DeviceProfileC()
        for r in range(self._ranks):
            check_status(
                self._lib.hc4j_collective_device_profile(self._handle, r, ctypes.byref(raw)),
                "hc4j_collective_device_profile",
                self._lib,
            )
            out.append(
                DeviceProfile(
                    rank=int(raw.rank),
                    name=self.device_name(r),
                    device_type=_DEVICE_TYPE_NAMES.get(raw.device_type, "Other"),
                    vendor_id=int(raw.vendor_id),
                    device_id=int(raw.device_id),
                    vram_bytes=int(raw.vram_bytes),
                    compute_gflops=float(raw.compute_gflops),
                )
            )
        return out

    def topology(self) -> list[Link]:
        self.ensure_open()
        count = check_status(self._lib.hc4j_collective_link_count(self._handle), "hc4j_collective_link_count", self._lib)
        raw = LinkInfo()
        links: list[Link] = []
        for idx in range(count):
            check_status(self._lib.hc4j_collective_link(self._handle, idx, ctypes.byref(raw)), "hc4j_collective_link", self._lib)
            links.append(
                Link(
                    src=int(raw.src_rank),
                    dst=int(raw.dst_rank),
                    gbps=float(raw.bandwidth_gbps),
                    transport=Transport(raw.transport),
                    loopback=raw.loopback != 0,
                    latency_us=float(raw.latency_us),
                )
            )
        return links

    def ring_order(self) -> list[int]:
        self.ensure_open()
        arr = (ctypes.c_uint32 * self._ranks)()
        count = check_status(
            self._lib.hc4j_collective_ring_order(self._handle, arr, self._ranks),
            "hc4j_collective_ring_order",
            self._lib,
        )
        return [int(arr[i]) for i in range(count)]

    def islands(self) -> list[IslandInfo]:
        self.ensure_open()
        count = check_status(self._lib.hc4j_collective_island_count(self._handle), "hc4j_collective_island_count", self._lib)
        hdr = IslandHeaderC()
        ranks_buf = (ctypes.c_uint32 * self._ranks)()
        out: list[IslandInfo] = []
        for idx in range(count):
            n = check_status(
                self._lib.hc4j_collective_island(self._handle, idx, ctypes.byref(hdr), ranks_buf, self._ranks),
                "hc4j_collective_island",
                self._lib,
            )
            out.append(
                IslandInfo(
                    id=int(hdr.id),
                    leader_rank=int(hdr.leader_rank),
                    internal_transport=Transport(hdr.internal_transport),
                    members=tuple(int(ranks_buf[i]) for i in range(n)),
                )
            )
        return out

    def allocate(self, rank: int, elements: int) -> CollectiveBuffer:
        self.ensure_open()
        if elements <= 0:
            raise InvalidParamError(f"elements must be positive, got {elements}")
        out = ctypes.c_uint64(0)
        check_status(
            self._lib.hc4j_collective_alloc(self._handle, rank, elements, ctypes.byref(out)),
            "hc4j_collective_alloc",
            self._lib,
        )
        return CollectiveBuffer(self, rank, int(out.value), elements)

    def allreduce_sum(self, buffers: Sequence[CollectiveBuffer]) -> None:
        self.ensure_open()
        ids = (ctypes.c_uint64 * len(buffers))(*[b.handle for b in buffers])
        fn = self._lib.hc4j_collective_allreduce_sum if len(buffers) == self._ranks else self._lib.hc4j_collective_allreduce_sum_subset
        check_status(fn(self._handle, ids, len(buffers)), "hc4j_collective_allreduce_sum", self._lib)

    def gemm_bias_act(
        self,
        rank: int,
        buf_a: CollectiveBuffer,
        buf_b: CollectiveBuffer,
        buf_c: CollectiveBuffer,
        m: int,
        n: int,
        k: int,
        buf_bias: CollectiveBuffer | None = None,
        act_code: int = 0,
    ) -> None:
        self.ensure_open()
        flags = (1 if buf_bias is not None else 0) | ((act_code & 0xF) << 1)
        bias_handle = buf_bias.handle if buf_bias is not None else 0
        check_status(
            self._lib.hc4j_collective_gemm_bias_act(
                self._handle, rank, buf_a.handle, buf_b.handle, bias_handle, buf_c.handle, m, n, k, flags
            ),
            "hc4j_collective_gemm_bias_act",
            self._lib,
        )

    def split_kv_attn(
        self,
        rank: int,
        buf_q: CollectiveBuffer,
        buf_k: CollectiveBuffer,
        buf_v: CollectiveBuffer,
        buf_out_o: CollectiveBuffer,
        buf_out_stats: CollectiveBuffer,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        local_tokens: int,
        scale: float,
    ) -> None:
        self.ensure_open()
        scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
        check_status(
            self._lib.hc4j_collective_split_kv_attn(
                self._handle,
                rank,
                buf_q.handle,
                buf_k.handle,
                buf_v.handle,
                buf_out_o.handle,
                buf_out_stats.handle,
                num_heads,
                num_kv_heads,
                head_dim,
                local_tokens,
                scale_bits,
            ),
            "hc4j_collective_split_kv_attn",
            self._lib,
        )


def merge_split_kv(
    num_heads: int,
    head_dim: int,
    partials_o: Sequence[Any],
    partials_stats: Sequence[Any],
    as_numpy: bool = False,
) -> Any:
    """Lossless online-softmax merge over Split-KV partials via ``hc4j_collective_merge_split_kv_f32``."""
    if len(partials_o) != len(partials_stats):
        raise InvalidParamError("partials_o and partials_stats lengths must match")
    wc = len(partials_o)
    if num_heads <= 0 or head_dim <= 0 or wc == 0:
        raise InvalidParamError("num_heads, head_dim, and worker_count must be positive")

    lib = get_library()
    ptr_o = (ctypes.c_void_p * wc)()
    ptr_s = (ctypes.c_void_p * wc)()
    pinned: list[Any] = []

    expected_o = num_heads * head_dim
    expected_s = num_heads * 2
    for w, (po, ps) in enumerate(zip(partials_o, partials_stats)):
        if _np is not None and isinstance(po, _np.ndarray):
            ao = _np.ascontiguousarray(po, dtype=_np.float32)
            pinned.append(ao)
            if ao.size != expected_o:
                raise InvalidParamError(f"partials_o[{w}] size {ao.size} != {expected_o}")
            ptr_o[w] = ao.ctypes.data
        else:
            arr_o = array.array("f", [float(v) for v in po])
            pinned.append(arr_o)
            if len(arr_o) != expected_o:
                raise InvalidParamError(f"partials_o[{w}] size {len(arr_o)} != {expected_o}")
            ptr_o[w] = arr_o.buffer_info()[0]

        if _np is not None and isinstance(ps, _np.ndarray):
            as_arr = _np.ascontiguousarray(ps, dtype=_np.float32)
            pinned.append(as_arr)
            if as_arr.size != expected_s:
                raise InvalidParamError(f"partials_stats[{w}] size {as_arr.size} != {expected_s}")
            ptr_s[w] = as_arr.ctypes.data
        else:
            arr_s = array.array("f", [float(v) for v in ps])
            pinned.append(arr_s)
            if len(arr_s) != expected_s:
                raise InvalidParamError(f"partials_stats[{w}] size {len(arr_s)} != {expected_s}")
            ptr_s[w] = arr_s.buffer_info()[0]

    if as_numpy:
        if _np is None:
            raise ImportError("Optional dependency 'numpy' is not installed")
        out_np = _np.empty((num_heads, head_dim), dtype=_np.float32)
        check_status(
            lib.hc4j_collective_merge_split_kv_f32(num_heads, head_dim, wc, ptr_o, ptr_s, out_np.ctypes.data),
            "hc4j_collective_merge_split_kv_f32",
            lib,
        )
        return out_np

    out_arr = array.array("f", [0.0] * expected_o)
    out_ptr, _ = out_arr.buffer_info()
    check_status(
        lib.hc4j_collective_merge_split_kv_f32(num_heads, head_dim, wc, ptr_o, ptr_s, out_ptr),
        "hc4j_collective_merge_split_kv_f32",
        lib,
    )
    return out_arr.tolist()
