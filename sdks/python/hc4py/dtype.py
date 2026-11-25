# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Element data types matching ``hc4x.h`` (``f32``, ``i32``, ``f16``)."""

from __future__ import annotations

import struct
from enum import Enum
from typing import Any

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore[assignment]


def has_numpy() -> bool:
    """Returns ``True`` if optional ``numpy`` is importable in the active Python environment."""
    return _np is not None


class DType(Enum):
    """Supported GPU tensor element types."""

    i32 = (0, 4, "i")
    f32 = (1, 4, "f")
    f16 = (2, 2, "e")

    def __init__(self, code: int, itemsize: int, struct_code: str) -> None:
        self._code = code
        self._itemsize = itemsize
        self._struct_code = struct_code

    @property
    def code(self) -> int:
        """Native C-ABI ``HC4J_DTYPE_*`` integer code."""
        return self._code

    @property
    def itemsize(self) -> int:
        """Byte width per scalar element."""
        return self._itemsize

    @property
    def struct_code(self) -> str:
        """``struct`` format specifier for a single element."""
        return self._struct_code

    @property
    def numpy_dtype(self) -> Any:
        """Corresponding NumPy dtype (requires optional ``numpy`` package)."""
        if _np is None:
            raise ImportError("Optional dependency 'numpy' is not installed")
        return {
            DType.f32: _np.dtype(_np.float32),
            DType.i32: _np.dtype(_np.int32),
            DType.f16: _np.dtype(_np.float16),
        }[self]

    @classmethod
    def from_code(cls, code: int) -> DType:
        """Resolves a native ``HC4J_DTYPE_*`` integer code back to :class:`DType`."""
        for member in cls:
            if member._code == code:
                return member
        raise ValueError(f"Unknown hc4x dtype code: {code}")

    @classmethod
    def from_numpy(cls, arr_or_dtype: Any) -> DType:
        """Maps a NumPy array or dtype object to :class:`DType`."""
        if _np is None:
            raise ImportError("Optional dependency 'numpy' is not installed")
        dt = _np.dtype(arr_or_dtype.dtype if hasattr(arr_or_dtype, "dtype") else arr_or_dtype)
        if dt == _np.float32:
            return cls.f32
        if dt == _np.int32:
            return cls.i32
        if dt == _np.float16:
            return cls.f16
        raise TypeError(f"Unsupported NumPy dtype {dt}; expected float32, int32, or float16")


def coerce_dtype(dt: DType | str | Any) -> DType:
    """Normalizes a :class:`DType`, string alias, or NumPy dtype to :class:`DType`."""
    if isinstance(dt, DType):
        return dt
    if isinstance(dt, str):
        key = dt.lower()
        aliases = {
            "f32": DType.f32,
            "float32": DType.f32,
            "float": DType.f32,
            "i32": DType.i32,
            "int32": DType.i32,
            "int": DType.i32,
            "f16": DType.f16,
            "float16": DType.f16,
            "half": DType.f16,
        }
        if key in aliases:
            return aliases[key]
        raise ValueError(f"Unknown dtype string: {dt!r}")
    return DType.from_numpy(dt)


def pack_f16(values: list[float]) -> bytes:
    """Encodes Python floats to packed little-endian IEEE-754 binary16 bytes."""
    return struct.pack(f"<{len(values)}e", *(float(v) for v in values))


def unpack_f16(raw: bytes) -> list[float]:
    """Decodes packed little-endian IEEE-754 binary16 bytes to Python floats."""
    count = len(raw) // 2
    return list(struct.unpack(f"<{count}e", raw))
