# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Linear algebra operations: matmul, batched matmul, transposed-B linear GEMM, transpose, and kernel planner."""

from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum

from ._ffi import InvalidParamError, UnsupportedError, check_status, get_library
from .dtype import DType
from .tensor import (
    Tensor,
    _prepare_contiguous,
    _u32_array,
    broadcast_shape,
    require_contiguous_output,
)

FLAG_TRANS_A = 1
FLAG_TRANS_B = 2


class Kernel(Enum):
    """Native GEMM/GEMV kernel variant returned by :func:`matmul_plan`."""

    REGISTER_64_VEC4 = 1
    REGISTER_64_SCALAR = 2
    REGISTER_128_VEC4 = 3
    REGISTER_128_SCALAR = 4
    TILED_16 = 5
    GEMV_ROWS = 6
    GEMV_ROWS_SUBGROUP = 7
    GEMV_COLS = 8


@dataclass(frozen=True)
class MatmulPlan:
    """Native kernel choice and scratch pretransposition plan for ``(m, n, k)``."""

    kernel: Kernel
    pretranspose_a: bool
    pretranspose_b: bool


def matmul_plan(m: int, n: int, k: int, transpose_a: bool = False, transpose_b: bool = False) -> MatmulPlan:
    """Queries ``hc4j_matmul_plan`` for the selected kernel without dispatching work."""
    flags = (FLAG_TRANS_A if transpose_a else 0) | (FLAG_TRANS_B if transpose_b else 0)
    lib = get_library()
    code = check_status(lib.hc4j_matmul_plan(m, n, k, flags), "hc4j_matmul_plan", lib)
    return MatmulPlan(
        kernel=Kernel(code & 0xFF),
        pretranspose_a=(code & 0x100) != 0,
        pretranspose_b=(code & 0x200) != 0,
    )


def transpose(a: Tensor, out: Tensor | None = None) -> Tensor:
    """Out-of-place 2-D GPU matrix transpose."""
    a.ensure_open()
    if a.dtype != DType.f32:
        raise InvalidParamError(f"transpose requires f32, got {a.dtype.name}")
    if a.ndim != 2:
        raise InvalidParamError(f"transpose requires a 2-D matrix, got {a.shape}")
    out_shape = (a.shape[1], a.shape[0])
    dest = out if out is not None else Tensor.empty(out_shape, dtype=DType.f32)
    try:
        require_contiguous_output(dest)
        if dest.shape != out_shape:
            raise InvalidParamError(f"Expected output shape {out_shape}, got {dest.shape}")
        lib = get_library()
        with ExitStack() as stack:
            ea = _prepare_contiguous(stack, a)
            if ea.vram_id == dest.vram_id:
                raise InvalidParamError("transpose cannot run in place")
            check_status(
                lib.dispatch_transpose_f32(ea.vram_id, dest.vram_id, ea.shape[0], ea.shape[1]),
                "dispatch_transpose_f32",
                lib,
            )
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def _resolve_2d_matmul_dims(
    sa: tuple[int, ...],
    sb: tuple[int, ...],
    transpose_a: bool,
    transpose_b: bool,
) -> tuple[int, int, int, tuple[int, ...]]:
    ra, rb = len(sa), len(sb)
    if ra == 1 and rb == 1:
        if sa[0] != sb[0]:
            raise InvalidParamError(f"Dot inner dimension mismatch: {sa[0]} vs {sb[0]}")
        return 1, 1, sa[0], (1,)
    if ra == 1 and rb == 2:
        k_b, n = (sb[1], sb[0]) if transpose_b else (sb[0], sb[1])
        if sa[0] != k_b:
            raise InvalidParamError(f"Matmul inner dimension mismatch: {sa[0]} vs {k_b}")
        return 1, n, k_b, (n,)
    if ra == 2 and rb == 1:
        m, k_a = (sa[1], sa[0]) if transpose_a else (sa[0], sa[1])
        if k_a != sb[0]:
            raise InvalidParamError(f"Matmul inner dimension mismatch: {k_a} vs {sb[0]}")
        return m, 1, k_a, (m,)
    if ra == 2 and rb == 2:
        m, k_a = (sa[1], sa[0]) if transpose_a else (sa[0], sa[1])
        k_b, n = (sb[1], sb[0]) if transpose_b else (sb[0], sb[1])
        if k_a != k_b:
            raise InvalidParamError(f"Matmul inner dimension mismatch: {k_a} vs {k_b}")
        return m, n, k_a, (m, n)
    raise InvalidParamError(f"Unsupported matmul operand shapes: {sa} and {sb}")


def _matmul_batched(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor:
    lib = get_library()
    sa, sb = a.shape, b.shape
    if len(sa) < 2 or len(sb) < 2:
        raise InvalidParamError(f"Batched matmul requires rank >= 2, got {sa} and {sb}")
    m, ka = sa[-2], sa[-1]
    kb, n = sb[-2], sb[-1]
    if ka != kb:
        raise InvalidParamError(f"Batched matmul inner mismatch: {ka} vs {kb}")
    batch_a, batch_b = sa[:-2], sb[:-2]
    batch_shape = broadcast_shape(batch_a, batch_b)
    batch_rank = len(batch_shape)
    if not (1 <= batch_rank <= 4):
        raise InvalidParamError(f"Batch rank must be 1..4, got {batch_rank}")
    out_shape = (*batch_shape, m, n)
    dest = out if out is not None else Tensor.empty(out_shape, dtype=DType.f32)
    try:
        if dest.dtype != DType.f32:
            raise InvalidParamError("Batched matmul output must be f32")
        require_contiguous_output(dest)
        if dest.shape != out_shape:
            raise InvalidParamError(f"Expected output shape {out_shape}, got {dest.shape}")
        with ExitStack() as stack:
            ea = _prepare_contiguous(stack, a)
            eb = _prepare_contiguous(stack, b)
            if ea.vram_id == dest.vram_id or eb.vram_id == dest.vram_id:
                raise InvalidParamError("Batched matmul result must not alias an operand")

            off_a = batch_rank - len(batch_a)
            off_b = batch_rank - len(batch_b)
            a_bstrides = [
                0 if (i < off_a or (batch_a[i - off_a] == 1 and batch_shape[i] > 1)) else ea.strides[i - off_a]
                for i in range(batch_rank)
            ]
            b_bstrides = [
                0 if (i < off_b or (batch_b[i - off_b] == 1 and batch_shape[i] > 1)) else eb.strides[i - off_b]
                for i in range(batch_rank)
            ]
            status = lib.dispatch_batched_matmul_f32(
                ea.vram_id, eb.vram_id, dest.vram_id,
                m, n, ka, batch_rank,
                _u32_array(batch_shape), _u32_array(a_bstrides), _u32_array(b_bstrides), 0,
            )
            check_status(status, "dispatch_batched_matmul_f32", lib)
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def matmul(
    a: Tensor,
    b: Tensor,
    out: Tensor | None = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> Tensor:
    """Matrix multiplication supporting 1-D/2-D, batched 3-D..6-D broadcasting, and ``f32 x f16``."""
    a.ensure_open()
    b.ensure_open()
    if a.dtype != DType.f32:
        raise InvalidParamError(f"Left matmul operand must be f32, got {a.dtype.name}")
    if b.dtype not in (DType.f32, DType.f16):
        raise InvalidParamError(f"Right matmul operand must be f32 or f16, got {b.dtype.name}")

    if a.ndim > 2 or b.ndim > 2:
        if transpose_a or transpose_b:
            raise UnsupportedError("Batched matmul does not accept explicit transpose flags")
        if b.dtype != DType.f32:
            raise UnsupportedError("Batched matmul requires f32 operands")
        return _matmul_batched(a, b, out=out)

    m, n, k, out_shape = _resolve_2d_matmul_dims(a.shape, b.shape, transpose_a, transpose_b)
    dest = out if out is not None else Tensor.empty(out_shape, dtype=DType.f32)
    try:
        if dest.dtype != DType.f32:
            raise InvalidParamError("matmul output must be f32")
        require_contiguous_output(dest)
        if dest.shape != out_shape:
            raise InvalidParamError(f"Expected output shape {out_shape}, got {dest.shape}")
        lib = get_library()
        with ExitStack() as stack:
            ea = _prepare_contiguous(stack, a)
            eb = _prepare_contiguous(stack, b)
            if ea.vram_id == dest.vram_id or eb.vram_id == dest.vram_id:
                raise InvalidParamError("matmul result must not alias an operand")
            if eb.dtype == DType.f16:
                if transpose_a:
                    raise UnsupportedError("transpose_a is not supported for f16 weights")
                flags = FLAG_TRANS_B if (transpose_b and eb.ndim == 2) else 0
                status = lib.dispatch_matmul_f32_f16(ea.vram_id, eb.vram_id, dest.vram_id, m, n, k, flags)
            else:
                flags = (FLAG_TRANS_A if (transpose_a and ea.ndim == 2) else 0) | (
                    FLAG_TRANS_B if (transpose_b and eb.ndim == 2) else 0
                )
                status = (
                    lib.dispatch_matmul_f32(ea.vram_id, eb.vram_id, dest.vram_id, m, n, k)
                    if flags == 0
                    else lib.dispatch_matmul_f32_ex(ea.vram_id, eb.vram_id, dest.vram_id, m, n, k, flags)
                )
            check_status(status, "matmul", lib)
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def matmul_transposed_b(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor:
    """Computes ``a @ b^T`` for ``a[..., M, K]`` (f32) and 2-D weights ``b[N, K]`` (f32 or f16)."""
    a.ensure_open()
    b.ensure_open()
    if a.dtype != DType.f32:
        raise InvalidParamError(f"matmul_transposed_b left operand must be f32, got {a.dtype.name}")
    if b.dtype not in (DType.f32, DType.f16):
        raise InvalidParamError(f"matmul_transposed_b right operand must be f32 or f16, got {b.dtype.name}")
    if b.ndim != 2:
        raise InvalidParamError(f"matmul_transposed_b requires 2-D [N, K] weight matrix, got {b.shape}")
    n, k = b.shape
    ka = a.shape[-1]
    if ka != k:
        raise InvalidParamError(f"Inner dimension mismatch: {ka} vs {k}")
    m = 1 if a.ndim == 1 else math.prod(a.shape[:-1])
    out_shape = (n,) if a.ndim == 1 else (*a.shape[:-1], n)

    dest = out if out is not None else Tensor.empty(out_shape, dtype=DType.f32)
    try:
        if dest.dtype != DType.f32:
            raise InvalidParamError("matmul_transposed_b output must be f32")
        require_contiguous_output(dest)
        if dest.shape != out_shape:
            raise InvalidParamError(f"Expected output shape {out_shape}, got {dest.shape}")
        lib = get_library()
        with ExitStack() as stack:
            ea = _prepare_contiguous(stack, a)
            eb = _prepare_contiguous(stack, b)
            if ea.vram_id == dest.vram_id or eb.vram_id == dest.vram_id:
                raise InvalidParamError("matmul_transposed_b result must not alias an operand")
            if eb.dtype == DType.f16:
                status = lib.dispatch_matmul_f32_f16(ea.vram_id, eb.vram_id, dest.vram_id, m, n, k, FLAG_TRANS_B)
            else:
                status = lib.dispatch_matmul_f32_ex(ea.vram_id, eb.vram_id, dest.vram_id, m, n, k, FLAG_TRANS_B)
            check_status(status, "matmul_transposed_b", lib)
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise
