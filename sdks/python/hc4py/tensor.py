# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Ref-counted GPU tensor type, zero-copy NumPy bridge, views, and shared dispatch helpers."""

from __future__ import annotations

import array
import ctypes
import math
import threading
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Callable, Sequence

from ._ffi import Hc4xError, InvalidParamError, OutOfMemoryError, check_status, get_library
from .dtype import DType, coerce_dtype, pack_f16, unpack_f16
from .engine import ensure_initialized

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .fusion import FusedExpr


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return tuple(strides)


def _compute_size(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        if dim <= 0:
            raise InvalidParamError(f"Tensor dimensions must be positive, got shape {shape}")
        total *= dim
    return total


def _is_contiguous(shape: tuple[int, ...], strides: tuple[int, ...]) -> bool:
    expected = 1
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] != 1 and strides[i] != expected:
            return False
        expected *= shape[i]
    return True


def broadcast_shape(sa: tuple[int, ...], sb: tuple[int, ...]) -> tuple[int, ...]:
    ra, rb = len(sa), len(sb)
    r = max(ra, rb)
    out = [1] * r
    for i in range(r):
        da = sa[ra - 1 - i] if i < ra else 1
        db = sb[rb - 1 - i] if i < rb else 1
        if da != db and da != 1 and db != 1:
            raise InvalidParamError(f"Incompatible shapes for broadcasting: {sa} vs {sb}")
        out[r - 1 - i] = max(da, db)
    return tuple(out)


def broadcast_strides(shape: tuple[int, ...], strides: tuple[int, ...], target: tuple[int, ...]) -> tuple[int, ...]:
    r_src = len(shape)
    r_tgt = len(target)
    out = [0] * r_tgt
    offset = r_tgt - r_src
    for i in range(offset, r_tgt):
        dim = shape[i - offset]
        out[i] = 0 if (dim == 1 and target[i] > 1) else strides[i - offset]
    return tuple(out)


def require_contiguous_output(out: Tensor) -> None:
    if not out.is_contiguous or out.offset != 0:
        raise InvalidParamError("Output tensor must be contiguous with zero offset")


def require_same_shape(a: Tensor, b: Tensor, context: str = "") -> None:
    if a.shape != b.shape:
        prefix = f"{context}: " if context else ""
        raise InvalidParamError(f"{prefix}Shape mismatch {a.shape} vs {b.shape}")


def _flatten_nested(data: Any) -> tuple[list[float | int], tuple[int, ...]]:
    if isinstance(data, (int, float)):
        return [data], (1,)
    if not isinstance(data, (list, tuple)):
        data = list(data)
    if not data:
        raise InvalidParamError("Cannot construct empty tensor from empty sequence")

    first = data[0]
    if isinstance(first, (list, tuple)):
        sub_shape: tuple[int, ...] | None = None
        flat: list[float | int] = []
        for elem in data:
            sub_flat, s = _flatten_nested(elem)
            if sub_shape is None:
                sub_shape = s
            elif sub_shape != s:
                raise InvalidParamError("Jagged nested sequences are not supported")
            flat.extend(sub_flat)
        assert sub_shape is not None
        return flat, (len(data),) + sub_shape

    return list(data), (len(data),)


def _reshape_flat(flat: list[Any], shape: tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return flat
    step = math.prod(shape[1:])
    return [_reshape_flat(flat[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


class _VramHandle:
    """Thread-safe reference-counted owner of one native GPU allocation ID."""

    __slots__ = ("vram_id", "_refcount", "_lock", "_lib")

    def __init__(self, vram_id: int, lib: ctypes.CDLL) -> None:
        self.vram_id = vram_id
        self._refcount = 1
        self._lock = threading.Lock()
        self._lib = lib

    def retain(self) -> _VramHandle:
        with self._lock:
            if self._refcount <= 0:
                raise Hc4xError("Cannot retain a released VRAM handle")
            self._refcount += 1
            return self

    def release(self) -> None:
        should_free = False
        with self._lock:
            if self._refcount <= 0:
                return
            self._refcount -= 1
            if self._refcount == 0:
                should_free = True
        if should_free and self.vram_id != 0:
            try:
                self._lib.hc4j_gpu_free(self.vram_id)
            except Exception:
                pass
            self.vram_id = 0

    def __del__(self) -> None:
        self.release()


def _prepare_alias_and_offset(stack: ExitStack, inp: Tensor, out: Tensor) -> Tensor:
    """Returns a safe source tensor, materializing a copy on layout-mismatched aliasing or non-zero offset."""
    if inp.vram_id == out.vram_id and not inp.has_identical_layout(out):
        return stack.enter_context(inp.copy())
    if inp.offset > 0:
        return stack.enter_context(inp.contiguous())
    return inp


def _prepare_contiguous(stack: ExitStack, inp: Tensor) -> Tensor:
    if not inp.is_contiguous or inp.offset != 0:
        return stack.enter_context(inp.contiguous())
    return inp


def _u32_array(vals: Sequence[int]) -> Any:
    return (ctypes.c_uint32 * len(vals))(*vals)


def dispatch_strided_unary(
    context: str,
    inp: Tensor,
    out: Tensor,
    fn: Callable[[int, int, int, Any, Any, Any, int, int], int],
) -> Tensor:
    inp.ensure_open()
    out.ensure_open()
    require_same_shape(inp, out, context)
    require_contiguous_output(out)
    lib = get_library()
    with ExitStack() as stack:
        eff = _prepare_alias_and_offset(stack, inp, out)
        is_contig = 1 if (eff.is_contiguous and out.is_contiguous) else 0
        rank = out.ndim
        if is_contig:
            status = fn(eff.vram_id, out.vram_id, rank, None, None, None, out.size, 1)
        else:
            shape_arr = _u32_array(out.shape)
            st_in = _u32_array(eff.strides)
            st_out = _u32_array(out.strides)
            status = fn(eff.vram_id, out.vram_id, rank, shape_arr, st_in, st_out, out.size, 0)
        check_status(status, context, lib)
    return out


def dispatch_strided_binary(
    context: str,
    a: Tensor,
    b: Tensor,
    out: Tensor,
    fn: Callable[[int, int, int, int, Any, Any, Any, Any, int, int], int],
) -> Tensor:
    a.ensure_open()
    b.ensure_open()
    out.ensure_open()
    require_contiguous_output(out)
    lib = get_library()
    with ExitStack() as stack:
        eff_a = _prepare_alias_and_offset(stack, a, out)
        eff_b = _prepare_alias_and_offset(stack, b, out)
        target_shape = broadcast_shape(eff_a.shape, eff_b.shape)
        if out.shape != target_shape:
            raise InvalidParamError(f"{context}: expected output shape {target_shape}, got {out.shape}")
        b_strides_a = broadcast_strides(eff_a.shape, eff_a.strides, target_shape)
        b_strides_b = broadcast_strides(eff_b.shape, eff_b.strides, target_shape)
        is_contig = 1 if (
            eff_a.shape == target_shape
            and eff_b.shape == target_shape
            and eff_a.is_contiguous
            and eff_b.is_contiguous
            and out.is_contiguous
        ) else 0
        rank = len(target_shape)
        if is_contig:
            status = fn(eff_a.vram_id, eff_b.vram_id, out.vram_id, rank, None, None, None, None, out.size, 1)
        else:
            shape_arr = _u32_array(target_shape)
            st_a = _u32_array(b_strides_a)
            st_b = _u32_array(b_strides_b)
            st_c = _u32_array(out.strides)
            status = fn(eff_a.vram_id, eff_b.vram_id, out.vram_id, rank, shape_arr, st_a, st_b, st_c, out.size, 0)
        check_status(status, context, lib)
    return out


class Tensor:
    """GPU-resident multi-dimensional array supporting strided views, broadcasting, and zero-copy NumPy interop."""

    def __init__(
        self,
        handle: _VramHandle,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        offset: int,
        dtype: DType,
    ) -> None:
        self._handle = handle
        self._shape = tuple(int(d) for d in shape)
        self._strides = tuple(int(s) for s in strides)
        self._offset = int(offset)
        self._dtype = dtype
        self._size = _compute_size(self._shape)
        self._contiguous = _is_contiguous(self._shape, self._strides)
        self._closed = False

    # ------------------------------------------------------------------
    # Properties & lifecycle
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self._size * self._dtype.itemsize

    @property
    def is_contiguous(self) -> bool:
        return self._contiguous

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def vram_id(self) -> int:
        self.ensure_open()
        return self._handle.vram_id

    @property
    def T(self) -> Tensor:
        """Transposed view (2-D) or reversed last two dimensions."""
        self.ensure_open()
        if self.ndim < 2:
            raise InvalidParamError(f"transpose requires rank >= 2, got {self.ndim}-D")
        return self.transpose(self.ndim - 2, self.ndim - 1)

    def ensure_open(self) -> None:
        if self._closed or self._handle.vram_id == 0:
            raise Hc4xError("Tensor has already been closed")

    def has_identical_layout(self, other: Tensor) -> bool:
        return (
            self._offset == other._offset
            and self._shape == other._shape
            and self._strides == other._strides
        )

    def close(self) -> None:
        """Releases this view's reference to the backing GPU buffer idempotently."""
        if not self._closed:
            self._closed = True
            self._handle.release()

    def __enter__(self) -> Tensor:
        self.ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __hash__(self) -> int:
        return id(self)

    # ------------------------------------------------------------------
    # Constructors & host transfers
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls, shape: int | Sequence[int], dtype: DType | str = DType.f32) -> Tensor:
        return cls._alloc_impl(shape, dtype, zeroed=False)

    @classmethod
    def zeros(cls, shape: int | Sequence[int], dtype: DType | str = DType.f32) -> Tensor:
        return cls._alloc_impl(shape, dtype, zeroed=True)

    @classmethod
    def ones(cls, shape: int | Sequence[int], dtype: DType | str = DType.f32) -> Tensor:
        return cls.full(shape, 1 if coerce_dtype(dtype) == DType.i32 else 1.0, dtype=dtype)

    @classmethod
    def full(cls, shape: int | Sequence[int], fill_value: float | int, dtype: DType | str = DType.f32) -> Tensor:
        norm_shape = (int(shape),) if isinstance(shape, int) else tuple(int(d) for d in shape)
        size = _compute_size(norm_shape)
        dt = coerce_dtype(dtype)
        return cls.from_list([fill_value] * size, shape=norm_shape, dtype=dt)

    @classmethod
    def arange(cls, start: int | float, stop: int | float | None = None, step: int | float = 1,
               dtype: DType | str = DType.f32) -> Tensor:
        if stop is None:
            start_v, stop_v = 0, start
        else:
            start_v, stop_v = start, stop
        if step == 0:
            raise InvalidParamError("arange step must be non-zero")
        dt = coerce_dtype(dtype)
        vals: list[int | float] = []
        cur = start_v
        if step > 0:
            while cur < stop_v:
                vals.append(int(cur) if dt == DType.i32 else float(cur))
                cur += step
        else:
            while cur > stop_v:
                vals.append(int(cur) if dt == DType.i32 else float(cur))
                cur += step
        return cls.from_list(vals, shape=(len(vals),), dtype=dt)

    @classmethod
    def _alloc_impl(cls, shape: int | Sequence[int], dtype: DType | str, zeroed: bool) -> Tensor:
        ensure_initialized()
        norm_shape = (int(shape),) if isinstance(shape, int) else tuple(int(d) for d in shape)
        size = _compute_size(norm_shape)
        dt = coerce_dtype(dtype)
        nbytes = size * dt.itemsize
        lib = get_library()
        vram_id = int(lib.hc4j_gpu_alloc_bytes(nbytes, 1 if zeroed else 0))
        if vram_id == 0:
            raise OutOfMemoryError(f"Failed to allocate {nbytes} bytes for shape {norm_shape}")
        return cls(_VramHandle(vram_id, lib), norm_shape, contiguous_strides(norm_shape), 0, dt)

    @classmethod
    def from_numpy(cls, arr: Any) -> Tensor:
        """Uploads a NumPy ``float32`` / ``int32`` / ``float16`` array using zero-copy pointer staging."""
        if _np is None:
            raise ImportError("Optional dependency 'numpy' is not installed")
        ensure_initialized()
        np_arr = _np.ascontiguousarray(arr)
        dt = DType.from_numpy(np_arr)
        norm_shape = tuple(int(d) for d in np_arr.shape) if np_arr.ndim > 0 else (1,)
        t = cls._alloc_impl(norm_shape, dt, zeroed=False)
        try:
            lib = get_library()
            check_status(
                lib.hc4j_gpu_write_bytes(t.vram_id, np_arr.ctypes.data, np_arr.nbytes),
                "Tensor.from_numpy",
                lib,
            )
            return t
        except Exception:
            t.close()
            raise

    @classmethod
    def from_list(
        cls,
        data: Sequence[Any] | float | int,
        shape: int | Sequence[int] | None = None,
        dtype: DType | str = DType.f32,
    ) -> Tensor:
        """Creates a GPU tensor from Python lists or scalars using stdlib ``array``/``struct`` packing."""
        ensure_initialized()
        dt = coerce_dtype(dtype)
        flat, inferred_shape = _flatten_nested(data)
        norm_shape = (
            inferred_shape
            if shape is None
            else ((int(shape),) if isinstance(shape, int) else tuple(int(d) for d in shape))
        )
        expected = _compute_size(norm_shape)
        if len(flat) != expected:
            raise InvalidParamError(f"Sequence has {len(flat)} elements but shape {norm_shape} requires {expected}")

        t = cls._alloc_impl(norm_shape, dt, zeroed=False)
        try:
            lib = get_library()
            if dt == DType.f16:
                packed = pack_f16([float(x) for x in flat])
                buf = (ctypes.c_uint8 * len(packed)).from_buffer_copy(packed)
                ptr = ctypes.addressof(buf)
                nbytes = len(packed)
            elif dt == DType.i32:
                arr = array.array("i", (int(x) for x in flat))
                ptr, n_elems = arr.buffer_info()
                nbytes = n_elems * arr.itemsize
            else:
                arr = array.array("f", (float(x) for x in flat))
                ptr, n_elems = arr.buffer_info()
                nbytes = n_elems * arr.itemsize
            check_status(lib.hc4j_gpu_write_bytes(t.vram_id, ptr, nbytes), "Tensor.from_list", lib)
            return t
        except Exception:
            t.close()
            raise

    def numpy(self) -> Any:
        """Downloads this tensor into a contiguous NumPy ``ndarray`` (zero-copy C pointer download)."""
        if _np is None:
            raise ImportError("Optional dependency 'numpy' is not installed")
        self.ensure_open()
        lib = get_library()
        with self.contiguous() as contig:
            arr = _np.empty(self._shape, dtype=self._dtype.numpy_dtype)
            check_status(
                lib.hc4j_gpu_download_bytes(contig.vram_id, arr.ctypes.data, arr.nbytes),
                "Tensor.numpy",
                lib,
            )
            return arr

    def to_flat_list(self) -> list[float | int]:
        """Downloads elements into a flat Python ``list`` via stdlib ``array``/``struct``."""
        self.ensure_open()
        lib = get_library()
        with self.contiguous() as contig:
            nbytes = contig.nbytes
            if self._dtype == DType.f16:
                buf = (ctypes.c_uint8 * nbytes)()
                check_status(
                    lib.hc4j_gpu_download_bytes(contig.vram_id, ctypes.addressof(buf), nbytes),
                    "Tensor.to_flat_list",
                    lib,
                )
                return unpack_f16(bytes(buf))
            code = "i" if self._dtype == DType.i32 else "f"
            arr = array.array(code, [0] * contig.size)
            ptr, _ = arr.buffer_info()
            check_status(lib.hc4j_gpu_download_bytes(contig.vram_id, ptr, nbytes), "Tensor.to_flat_list", lib)
            return arr.tolist()

    def to_list(self) -> list[Any]:
        """Downloads elements into nested Python lists matching :attr:`shape`."""
        return _reshape_flat(self.to_flat_list(), self._shape)

    def item(self) -> float | int:
        if self._size != 1:
            raise InvalidParamError(f"item() requires a 1-element tensor, got size {self._size}")
        return self.to_flat_list()[0]

    # ------------------------------------------------------------------
    # Views, slicing, copy & cast
    # ------------------------------------------------------------------

    def view(self, *shape: int) -> Tensor:
        self.ensure_open()
        if not self._contiguous:
            raise InvalidParamError("view() requires a contiguous tensor; call contiguous() first")
        target = tuple(int(d) for d in (shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape))
        neg_idx = [i for i, d in enumerate(target) if d == -1]
        if len(neg_idx) > 1:
            raise InvalidParamError("Only one dimension can be inferred (-1)")
        if neg_idx:
            known = math.prod(d for d in target if d != -1)
            if known <= 0 or self._size % known != 0:
                raise InvalidParamError(f"Cannot reshape tensor of size {self._size} into {target}")
            target = tuple(self._size // known if d == -1 else d for d in target)
        if _compute_size(target) != self._size:
            raise InvalidParamError(f"Cannot reshape tensor of size {self._size} into {target}")
        return Tensor(self._handle.retain(), target, contiguous_strides(target), self._offset, self._dtype)

    reshape = view

    def transpose(self, dim0: int = 0, dim1: int = 1) -> Tensor:
        self.ensure_open()
        d0 = dim0 if dim0 >= 0 else self.ndim + dim0
        d1 = dim1 if dim1 >= 0 else self.ndim + dim1
        if not (0 <= d0 < self.ndim and 0 <= d1 < self.ndim):
            raise InvalidParamError(f"Transpose dims ({dim0}, {dim1}) out of bounds for rank {self.ndim}")
        new_shape = list(self._shape)
        new_strides = list(self._strides)
        new_shape[d0], new_shape[d1] = new_shape[d1], new_shape[d0]
        new_strides[d0], new_strides[d1] = new_strides[d1], new_strides[d0]
        return Tensor(self._handle.retain(), tuple(new_shape), tuple(new_strides), self._offset, self._dtype)

    def slice(self, dim: int, start: int, end: int, step: int = 1) -> Tensor:
        self.ensure_open()
        d = dim if dim >= 0 else self.ndim + dim
        if not (0 <= d < self.ndim):
            raise InvalidParamError(f"Dimension {dim} out of bounds for rank {self.ndim}")
        if step <= 0:
            raise InvalidParamError("Slice step must be positive")
        dim_size = self._shape[d]
        s = max(0, min(start if start >= 0 else dim_size + start, dim_size))
        e = max(s, min(end if end >= 0 else dim_size + end, dim_size))
        new_len = (e - s + step - 1) // step
        if new_len <= 0:
            raise InvalidParamError(f"Empty slice [{start}:{end}:{step}] on dim size {dim_size}")
        new_shape = list(self._shape)
        new_strides = list(self._strides)
        new_offset = self._offset + s * self._strides[d]
        new_shape[d] = new_len
        new_strides[d] = self._strides[d] * step
        return Tensor(self._handle.retain(), tuple(new_shape), tuple(new_strides), new_offset, self._dtype)

    def contiguous(self) -> Tensor:
        """Returns a retained view when already contiguous with offset=0, or materializes a contiguous copy."""
        self.ensure_open()
        if self._contiguous and self._offset == 0:
            return Tensor(self._handle.retain(), self._shape, self._strides, 0, self._dtype)
        return self.copy()

    def copy(self) -> Tensor:
        """Allocates an independent contiguous zero-offset copy in GPU memory."""
        self.ensure_open()
        if self._dtype == DType.f16 and (not self._contiguous or self._offset != 0):
            with self.cast(DType.f32) as as_f32:
                return as_f32.cast(DType.f16)
        lib = get_library()
        out = Tensor._alloc_impl(self._shape, self._dtype, zeroed=False)
        try:
            rank = self.ndim
            shape_arr = _u32_array(self._shape)
            strides_arr = _u32_array(self._strides)
            check_status(
                lib.dispatch_copy_strided(
                    self._handle.vram_id,
                    out.vram_id,
                    self._offset,
                    rank,
                    shape_arr,
                    strides_arr,
                    self._size,
                ),
                "Tensor.copy",
                lib,
            )
            return out
        except Exception:
            out.close()
            raise

    def cast(self, target_dtype: DType | str) -> Tensor:
        """Casts tensor elements to ``target_dtype`` (supports ``f32 <-> i32``, ``f32 <-> f16``, ``i32 <-> f16``)."""
        self.ensure_open()
        dst = coerce_dtype(target_dtype)
        if dst == self._dtype:
            return self.copy()
        lib = get_library()
        out = Tensor._alloc_impl(self._shape, dst, zeroed=False)
        try:
            with ExitStack() as stack:
                eff = _prepare_contiguous(stack, self)
                if eff.dtype == DType.f32 and dst == DType.f16:
                    status = lib.dispatch_cast_f32_to_f16(eff.vram_id, out.vram_id, eff.size)
                elif eff.dtype == DType.f16 and dst == DType.f32:
                    status = lib.dispatch_cast_f16_to_f32(eff.vram_id, out.vram_id, eff.size)
                elif (eff.dtype == DType.i32 and dst == DType.f16) or (eff.dtype == DType.f16 and dst == DType.i32):
                    with eff.cast(DType.f32) as tmp:
                        with tmp.cast(dst) as casted:
                            return casted.copy() if casted.vram_id == out.vram_id else casted.contiguous()
                else:
                    status = lib.dispatch_cast(eff.vram_id, out.vram_id, eff.size, eff.dtype.code, dst.code)
                check_status(status, "Tensor.cast", lib)
            return out
        except Exception:
            out.close()
            raise

    # ------------------------------------------------------------------
    # Ergonomic operator overloads & in-place ops
    # ------------------------------------------------------------------

    def _coerce_other(self, other: Tensor | int | float, stack: ExitStack) -> Tensor:
        if isinstance(other, Tensor):
            return other
        if isinstance(other, (int, float)):
            return stack.enter_context(Tensor.from_list([other], shape=(1,), dtype=self._dtype))
        return NotImplemented  # type: ignore[return-value]

    def __add__(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            rhs = self._coerce_other(other, stack)
            return NotImplemented if rhs is NotImplemented else _math.add(self, rhs)

    def __radd__(self, other: int | float) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            rhs = self._coerce_other(other, stack)
            return NotImplemented if rhs is NotImplemented else _math.sub(self, rhs)

    def __rsub__(self, other: int | float) -> Tensor:
        from . import math as _math
        if isinstance(other, (int, float)):
            with Tensor.from_list([other], shape=(1,), dtype=self._dtype) as lhs:
                return _math.sub(lhs, self)
        return NotImplemented  # type: ignore[return-value]

    def __mul__(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            rhs = self._coerce_other(other, stack)
            return NotImplemented if rhs is NotImplemented else _math.mul(self, rhs)

    def __rmul__(self, other: int | float) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            rhs = self._coerce_other(other, stack)
            return NotImplemented if rhs is NotImplemented else _math.div(self, rhs)

    def __rtruediv__(self, other: int | float) -> Tensor:
        from . import math as _math
        if isinstance(other, (int, float)):
            with Tensor.from_list([other], shape=(1,), dtype=self._dtype) as lhs:
                return _math.div(lhs, self)
        return NotImplemented  # type: ignore[return-value]

    def __matmul__(self, other: Tensor) -> Tensor:
        from . import linalg as _linalg
        if not isinstance(other, Tensor):
            return NotImplemented  # type: ignore[return-value]
        return _linalg.matmul(self, other)

    def __neg__(self) -> Tensor:
        from . import math as _math
        return _math.neg(self)

    def __abs__(self) -> Tensor:
        from . import math as _math
        return _math.abs(self)

    def add_(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            rhs = self._coerce_other(other, stack)
            return _math.add(self, rhs, out=self)

    def sub_(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            rhs = self._coerce_other(other, stack)
            return _math.sub(self, rhs, out=self)

    def mul_(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            rhs = self._coerce_other(other, stack)
            return _math.mul(self, rhs, out=self)

    def div_(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            rhs = self._coerce_other(other, stack)
            return _math.div(self, rhs, out=self)

    def eq(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            return _math.eq(self, self._coerce_other(other, stack))

    def ne(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            return _math.ne(self, self._coerce_other(other, stack))

    def gt(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            return _math.gt(self, self._coerce_other(other, stack))

    def ge(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            return _math.ge(self, self._coerce_other(other, stack))

    def lt(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            return _math.lt(self, self._coerce_other(other, stack))

    def le(self, other: Tensor | int | float) -> Tensor:
        from . import math as _math
        with ExitStack() as stack:
            return _math.le(self, self._coerce_other(other, stack))

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, (Tensor, int, float)):
            return self.eq(other)
        return False

    def __ne__(self, other: Any) -> Any:
        if isinstance(other, (Tensor, int, float)):
            return self.ne(other)
        return True

    def __gt__(self, other: Tensor | int | float) -> Tensor:
        return self.gt(other)

    def __ge__(self, other: Tensor | int | float) -> Tensor:
        return self.ge(other)

    def __lt__(self, other: Tensor | int | float) -> Tensor:
        return self.lt(other)

    def __le__(self, other: Tensor | int | float) -> Tensor:
        return self.le(other)

    def lazy(self) -> FusedExpr:
        """Wraps this tensor into a lazy elementwise :class:`hc4py.fusion.FusedExpr` DAG."""
        from .fusion import FusedExpr
        return FusedExpr.of(self)

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"vram_id={self._handle.vram_id}"
        return f"Tensor({state}, shape={self._shape}, strides={self._strides}, offset={self._offset}, dtype={self._dtype.name})"
