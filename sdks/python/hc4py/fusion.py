# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Deferred elementwise expression DAG compiler targeting ``hc4j_fused_dispatch``."""

from __future__ import annotations

import ctypes
import math
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Union

from ._ffi import InvalidParamError, check_status, get_library
from .dtype import DType
from .tensor import Tensor, require_contiguous_output, require_same_shape

_LOAD = 0x01
_CONST = 0x02
_NEG = 0x10
_ABS = 0x11
_SIN = 0x12
_COS = 0x13
_TAN = 0x14
_ASIN = 0x15
_ACOS = 0x16
_ATAN = 0x17
_SINH = 0x18
_COSH = 0x19
_TANH = 0x1A
_ASINH = 0x1B
_ACOSH = 0x1C
_ATANH = 0x1D
_EXP = 0x1E
_LOG = 0x1F
_SQRT = 0x20
_RELU = 0x21
_SIGMOID = 0x22
_GELU = 0x23
_SILU = 0x24
_SOFTPLUS = 0x25
_ADD = 0x40
_SUB = 0x41
_MUL = 0x42
_DIV = 0x43
_MAX = 0x44
_MIN = 0x45
_POW = 0x46

MAX_PROGRAM_WORDS = 1024


@dataclass(frozen=True)
class _Load:
    tensor: Tensor


@dataclass(frozen=True)
class _Const:
    value: float


@dataclass(frozen=True)
class _Unary:
    op: int
    x: _Node


@dataclass(frozen=True)
class _Binary:
    op: int
    left: _Node
    right: _Node


_Node = Union[_Load, _Const, _Unary, _Binary]


def _word(opcode: int, operand: int) -> int:
    return ((opcode & 0xFF) << 24) | (operand & 0x00FF_FFFF)


def _float_bits(v: float) -> int:
    if not math.isfinite(v):
        raise InvalidParamError(f"Fused constants must be finite, got {v}")
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class FusedExpr:
    """Deferred single-kernel elementwise expression DAG over same-shape ``f32`` GPU tensors."""

    def __init__(self, root: _Node, shape: tuple[int, ...], words: int) -> None:
        if words > MAX_PROGRAM_WORDS:
            raise InvalidParamError(
                f"Fused expression exceeds {MAX_PROGRAM_WORDS} words; eval() an intermediate subexpression"
            )
        self._root = root
        self._shape = shape
        self._words = words

    @classmethod
    def of(cls, tensor: Tensor) -> FusedExpr:
        tensor.ensure_open()
        if tensor.dtype != DType.f32:
            raise InvalidParamError(f"fusion is only defined for f32 tensors, got {tensor.dtype.name}")
        return cls(_Load(tensor), tensor.shape, 1)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def _unary(self, op: int) -> FusedExpr:
        return FusedExpr(_Unary(op, self._root), self._shape, self._words + 1)

    def _binary(self, op: int, rhs: FusedExpr | Tensor | float | int) -> FusedExpr:
        if isinstance(rhs, (int, float)):
            c = float(rhs)
            _float_bits(c)
            return FusedExpr(_Binary(op, self._root, _Const(c)), self._shape, self._words + 3)
        other = rhs if isinstance(rhs, FusedExpr) else FusedExpr.of(rhs)
        if other._shape != self._shape:
            raise InvalidParamError(f"Fused expression shape mismatch: {self._shape} vs {other._shape}")
        return FusedExpr(_Binary(op, self._root, other._root), self._shape, self._words + other._words + 1)

    def _rbinary(self, op: int, lhs: float | int) -> FusedExpr:
        c = float(lhs)
        _float_bits(c)
        return FusedExpr(_Binary(op, _Const(c), self._root), self._shape, self._words + 3)

    # Unary ops
    def neg(self) -> FusedExpr: return self._unary(_NEG)
    def abs(self) -> FusedExpr: return self._unary(_ABS)
    def sin(self) -> FusedExpr: return self._unary(_SIN)
    def cos(self) -> FusedExpr: return self._unary(_COS)
    def tan(self) -> FusedExpr: return self._unary(_TAN)
    def asin(self) -> FusedExpr: return self._unary(_ASIN)
    def acos(self) -> FusedExpr: return self._unary(_ACOS)
    def atan(self) -> FusedExpr: return self._unary(_ATAN)
    def sinh(self) -> FusedExpr: return self._unary(_SINH)
    def cosh(self) -> FusedExpr: return self._unary(_COSH)
    def tanh(self) -> FusedExpr: return self._unary(_TANH)
    def asinh(self) -> FusedExpr: return self._unary(_ASINH)
    def acosh(self) -> FusedExpr: return self._unary(_ACOSH)
    def atanh(self) -> FusedExpr: return self._unary(_ATANH)
    def exp(self) -> FusedExpr: return self._unary(_EXP)
    def log(self) -> FusedExpr: return self._unary(_LOG)
    def sqrt(self) -> FusedExpr: return self._unary(_SQRT)
    def relu(self) -> FusedExpr: return self._unary(_RELU)
    def sigmoid(self) -> FusedExpr: return self._unary(_SIGMOID)
    def gelu(self) -> FusedExpr: return self._unary(_GELU)
    def silu(self) -> FusedExpr: return self._unary(_SILU)
    def softplus(self) -> FusedExpr: return self._unary(_SOFTPLUS)

    # Binary ops
    def add(self, other: FusedExpr | Tensor | float | int) -> FusedExpr: return self._binary(_ADD, other)
    def sub(self, other: FusedExpr | Tensor | float | int) -> FusedExpr: return self._binary(_SUB, other)
    def mul(self, other: FusedExpr | Tensor | float | int) -> FusedExpr: return self._binary(_MUL, other)
    def div(self, other: FusedExpr | Tensor | float | int) -> FusedExpr: return self._binary(_DIV, other)
    def maximum(self, other: FusedExpr | Tensor | float | int) -> FusedExpr: return self._binary(_MAX, other)
    def minimum(self, other: FusedExpr | Tensor | float | int) -> FusedExpr: return self._binary(_MIN, other)
    def pow(self, other: FusedExpr | Tensor | float | int) -> FusedExpr: return self._binary(_POW, other)

    def __neg__(self) -> FusedExpr: return self.neg()
    def __abs__(self) -> FusedExpr: return self.abs()
    def __add__(self, other: FusedExpr | Tensor | float | int) -> FusedExpr: return self._binary(_ADD, other)
    def __radd__(self, other: float | int) -> FusedExpr: return self._rbinary(_ADD, other)
    def __sub__(self, other: FusedExpr | Tensor | float | int) -> FusedExpr: return self._binary(_SUB, other)
    def __rsub__(self, other: float | int) -> FusedExpr: return self._rbinary(_SUB, other)
    def __mul__(self, other: FusedExpr | Tensor | float | int) -> FusedExpr: return self._binary(_MUL, other)
    def __rmul__(self, other: float | int) -> FusedExpr: return self._rbinary(_MUL, other)
    def __truediv__(self, other: FusedExpr | Tensor | float | int) -> FusedExpr: return self._binary(_DIV, other)
    def __rtruediv__(self, other: float | int) -> FusedExpr: return self._rbinary(_DIV, other)
    def __pow__(self, other: FusedExpr | Tensor | float | int) -> FusedExpr: return self._binary(_POW, other)

    def eval(self, out: Tensor | None = None) -> Tensor:
        """Compiles and dispatches the fused expression into a single GPU compute pass."""
        dest = out if out is not None else Tensor.empty(self._shape, dtype=DType.f32)
        try:
            dest.ensure_open()
            if dest.dtype != DType.f32:
                raise InvalidParamError(f"Fused destination must be f32, got {dest.dtype.name}")
            if dest.shape != self._shape:
                raise InvalidParamError(f"Expected output shape {self._shape}, got {dest.shape}")
            require_contiguous_output(dest)

            program: list[int] = []
            input_map: dict[int, int] = {}
            handles: list[int] = []

            with ExitStack() as stack:
                work: list[_Node | int] = [self._root]
                while work:
                    item = work.pop()
                    if isinstance(item, int):
                        program.append(item)
                        continue
                    if isinstance(item, _Load):
                        t = item.tensor
                        t.ensure_open()
                        eff = t
                        if t.vram_id == dest.vram_id and not t.has_identical_layout(dest):
                            eff = stack.enter_context(t.copy())
                        elif not t.is_contiguous or t.offset != 0:
                            eff = stack.enter_context(t.contiguous())
                        if eff.dtype != DType.f32:
                            raise InvalidParamError(f"Fused input must be f32, got {eff.dtype.name}")
                        if eff.shape != self._shape:
                            raise InvalidParamError(f"Fused input shape {eff.shape} != {self._shape}")
                        key = id(eff)
                        idx = input_map.get(key)
                        if idx is None:
                            idx = len(handles)
                            input_map[key] = idx
                            handles.append(eff.vram_id)
                        program.append(_word(_LOAD, idx))
                    elif isinstance(item, _Const):
                        program.append(_word(_CONST, 0))
                        program.append(_float_bits(item.value))
                    elif isinstance(item, _Unary):
                        work.append(_word(item.op, 0))
                        work.append(item.x)
                    else:
                        work.append(_word(item.op, 0))
                        work.append(item.right)
                        work.append(item.left)

                lib = get_library()
                prog_arr = (ctypes.c_uint32 * len(program))(*program)
                handles_arr = (ctypes.c_uint64 * len(handles))(*handles)
                check_status(
                    lib.hc4j_fused_dispatch(prog_arr, len(program), handles_arr, len(handles), dest.vram_id),
                    "hc4j_fused_dispatch",
                    lib,
                )
            return dest
        except Exception:
            if out is None:
                dest.close()
            raise
