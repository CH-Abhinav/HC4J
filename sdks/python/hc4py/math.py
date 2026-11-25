# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Elementwise arithmetic, trigonometric, exponential, math, comparison, and selection operations."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Callable

from ._ffi import InvalidParamError, check_status, get_library
from .dtype import DType
from .tensor import (
    Tensor,
    _prepare_alias_and_offset,
    _u32_array,
    broadcast_shape,
    broadcast_strides,
    dispatch_strided_binary,
    dispatch_strided_unary,
    require_contiguous_output,
)


def _binary_typed(op_attr: str, a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor:
    if a.dtype != b.dtype:
        raise InvalidParamError(f"DType mismatch: {a.dtype.name} vs {b.dtype.name}")
    if a.dtype not in (DType.f32, DType.i32):
        raise InvalidParamError(f"{op_attr} supports f32 and i32, got {a.dtype.name}")
    target_shape = broadcast_shape(a.shape, b.shape)
    dest = out if out is not None else Tensor.empty(target_shape, dtype=a.dtype)
    try:
        if dest.dtype != a.dtype:
            raise InvalidParamError(f"Result dtype must be {a.dtype.name}, got {dest.dtype.name}")
        fn = getattr(get_library(), op_attr)
        return dispatch_strided_binary(
            op_attr, a, b, dest,
            lambda id_a, id_b, id_o, rk, sh, sa, sb, sc, ln, cg: fn(
                id_a, id_b, id_o, rk, sh, sa, sb, sc, ln, cg, a.dtype.code
            ),
        )
    except Exception:
        if out is None:
            dest.close()
        raise


def add(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor: return _binary_typed("dispatch_add", a, b, out)
def sub(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor: return _binary_typed("dispatch_sub", a, b, out)
def mul(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor: return _binary_typed("dispatch_mul", a, b, out)
def div(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor: return _binary_typed("dispatch_div", a, b, out)
def maximum(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor: return _binary_typed("dispatch_maximum", a, b, out)
def minimum(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor: return _binary_typed("dispatch_minimum", a, b, out)


def pow(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor:
    if a.dtype != DType.f32 or b.dtype != DType.f32:
        raise InvalidParamError("pow is only defined for f32 tensors")
    target_shape = broadcast_shape(a.shape, b.shape)
    dest = out if out is not None else Tensor.empty(target_shape, dtype=DType.f32)
    try:
        if dest.dtype != DType.f32:
            raise InvalidParamError("Result dtype must be f32")
        fn = get_library().dispatch_pow_f32
        return dispatch_strided_binary("dispatch_pow_f32", a, b, dest, fn)
    except Exception:
        if out is None:
            dest.close()
        raise


def _unary_f32(op_attr: str, a: Tensor, out: Tensor | None = None) -> Tensor:
    a.ensure_open()
    if a.dtype != DType.f32:
        raise InvalidParamError(f"{op_attr} is only defined for f32 tensors, got {a.dtype.name}")
    dest = out if out is not None else Tensor.empty(a.shape, dtype=DType.f32)
    try:
        if dest.dtype != DType.f32:
            raise InvalidParamError("Result dtype must be f32")
        fn = getattr(get_library(), op_attr)
        return dispatch_strided_unary(op_attr, a, dest, fn)
    except Exception:
        if out is None:
            dest.close()
        raise


def _unary_f32_or_i32(f32_attr: str, i32_code: int, a: Tensor, out: Tensor | None = None) -> Tensor:
    a.ensure_open()
    if a.dtype == DType.f32:
        return _unary_f32(f32_attr, a, out)
    if a.dtype != DType.i32:
        raise InvalidParamError(f"Unsupported dtype {a.dtype.name}")
    dest = out if out is not None else Tensor.empty(a.shape, dtype=DType.i32)
    try:
        if dest.dtype != DType.i32:
            raise InvalidParamError("Result dtype must be i32")
        fn = get_library().dispatch_unary_i32
        return dispatch_strided_unary(
            f"dispatch_unary_i32({i32_code})", a, dest,
            lambda id_in, id_o, rk, sh, st_i, st_o, ln, cg: fn(id_in, id_o, rk, sh, st_i, st_o, ln, cg, i32_code, 0, 0),
        )
    except Exception:
        if out is None:
            dest.close()
        raise


# Trigonometric (12)
def sin(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_sin_f32", a, out)
def cos(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_cos_f32", a, out)
def tan(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_tan_f32", a, out)
def asin(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_asin_f32", a, out)
def acos(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_acos_f32", a, out)
def atan(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_atan_f32", a, out)
def sinh(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_sinh_f32", a, out)
def cosh(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_cosh_f32", a, out)
def tanh(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_tanh_f32", a, out)
def asinh(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_asinh_f32", a, out)
def acosh(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_acosh_f32", a, out)
def atanh(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_atanh_f32", a, out)

# Exponential & logarithmic (5)
def exp(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_exp_f32", a, out)
def log(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_ln_f32", a, out)
def log2(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_log2_f32", a, out)
def log10(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_log10_f32", a, out)
def sqrt(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_sqrt_f32", a, out)

# Unary math & clamp
def abs(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32_or_i32("dispatch_abs_f32", 0, a, out)
def neg(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32_or_i32("dispatch_neg_f32", 1, a, out)
def sign(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32_or_i32("dispatch_sign_f32", 2, a, out)
def rsqrt(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_rsqrt_f32", a, out)
def floor(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_floor_f32", a, out)
def ceil(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_ceil_f32", a, out)
def round(a: Tensor, out: Tensor | None = None) -> Tensor: return _unary_f32("dispatch_round_f32", a, out)


def clamp(a: Tensor, lo: float | int, hi: float | int, out: Tensor | None = None) -> Tensor:
    a.ensure_open()
    dest = out if out is not None else Tensor.empty(a.shape, dtype=a.dtype)
    try:
        if dest.dtype != a.dtype:
            raise InvalidParamError(f"Result dtype must match input {a.dtype.name}")
        lib = get_library()
        if a.dtype == DType.f32:
            return dispatch_strided_unary(
                "dispatch_clamp_f32", a, dest,
                lambda id_in, id_o, rk, sh, st_i, st_o, ln, cg: lib.dispatch_clamp_f32(
                    id_in, id_o, rk, sh, st_i, st_o, ln, cg, float(lo), float(hi)
                ),
            )
        if a.dtype == DType.i32:
            return dispatch_strided_unary(
                "dispatch_unary_i32(clamp)", a, dest,
                lambda id_in, id_o, rk, sh, st_i, st_o, ln, cg: lib.dispatch_unary_i32(
                    id_in, id_o, rk, sh, st_i, st_o, ln, cg, 3, int(lo) & 0xFFFFFFFF, int(hi) & 0xFFFFFFFF
                ),
            )
        raise InvalidParamError(f"clamp does not support dtype {a.dtype.name}")
    except Exception:
        if out is None:
            dest.close()
        raise


def _compare(cmp_op: int, name: str, a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor:
    if a.dtype != b.dtype:
        raise InvalidParamError(f"DType mismatch in {name}: {a.dtype.name} vs {b.dtype.name}")
    target_shape = broadcast_shape(a.shape, b.shape)
    dest = out if out is not None else Tensor.empty(target_shape, dtype=DType.i32)
    try:
        if dest.dtype != DType.i32:
            raise InvalidParamError("Comparison output dtype must be i32")
        fn = get_library().dispatch_compare
        return dispatch_strided_binary(
            f"compare.{name}", a, b, dest,
            lambda id_a, id_b, id_o, rk, sh, sa, sb, sc, ln, cg: fn(
                id_a, id_b, id_o, rk, sh, sa, sb, sc, ln, cg, a.dtype.code, cmp_op
            ),
        )
    except Exception:
        if out is None:
            dest.close()
        raise


def eq(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor: return _compare(0, "eq", a, b, out)
def ne(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor: return _compare(1, "ne", a, b, out)
def gt(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor: return _compare(2, "gt", a, b, out)
def ge(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor: return _compare(3, "ge", a, b, out)
def lt(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor: return _compare(4, "lt", a, b, out)
def le(a: Tensor, b: Tensor, out: Tensor | None = None) -> Tensor: return _compare(5, "le", a, b, out)


def where(cond: Tensor, x: Tensor, y: Tensor, out: Tensor | None = None) -> Tensor:
    cond.ensure_open()
    x.ensure_open()
    y.ensure_open()
    if x.dtype != y.dtype:
        raise InvalidParamError(f"where branch dtype mismatch: {x.dtype.name} vs {y.dtype.name}")
    target_shape = broadcast_shape(broadcast_shape(cond.shape, x.shape), y.shape)
    dest = out if out is not None else Tensor.empty(target_shape, dtype=x.dtype)
    try:
        if dest.dtype != x.dtype:
            raise InvalidParamError(f"Result dtype must be {x.dtype.name}")
        require_contiguous_output(dest)
        lib = get_library()
        with ExitStack() as stack:
            ec = _prepare_alias_and_offset(stack, cond, dest)
            ex = _prepare_alias_and_offset(stack, x, dest)
            ey = _prepare_alias_and_offset(stack, y, dest)
            is_contig = 1 if (
                ec.shape == target_shape and ex.shape == target_shape and ey.shape == target_shape
                and ec.is_contiguous and ex.is_contiguous and ey.is_contiguous and dest.is_contiguous
            ) else 0
            rank = len(target_shape)
            if is_contig:
                status = lib.dispatch_where(
                    ec.vram_id, ex.vram_id, ey.vram_id, dest.vram_id,
                    rank, None, None, None, None, None, dest.size, 1, ec.dtype.code, ex.dtype.code,
                )
            else:
                status = lib.dispatch_where(
                    ec.vram_id, ex.vram_id, ey.vram_id, dest.vram_id,
                    rank,
                    _u32_array(target_shape),
                    _u32_array(broadcast_strides(ec.shape, ec.strides, target_shape)),
                    _u32_array(broadcast_strides(ex.shape, ex.strides, target_shape)),
                    _u32_array(broadcast_strides(ey.shape, ey.strides, target_shape)),
                    _u32_array(dest.strides),
                    dest.size, 0, ec.dtype.code, ex.dtype.code,
                )
            check_status(status, "dispatch_where", lib)
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise
