# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Neural network kernels: activations, normalizations, softmax, cross-entropy, reductions, and Transformer ops."""

from __future__ import annotations

import math
import struct
from contextlib import ExitStack
from enum import Enum

from ._ffi import InvalidParamError, check_status, get_library
from .dtype import DType
from .linalg import matmul, matmul_transposed_b
from .tensor import Tensor, _prepare_contiguous, dispatch_strided_unary, require_contiguous_output, require_same_shape


class Activation(Enum):
    """Elementwise activation functions matching ``hc4x.h`` ``HC4J_ACT_*`` codes."""

    RELU = 0
    LEAKY_RELU = 1
    ELU = 2
    GELU = 3
    SILU = 4
    SIGMOID = 5
    SOFTPLUS = 6


class Reduction(Enum):
    """Tensor reduction operators matching ``hc4x.h`` ``HC4J_REDUCE_*`` codes."""

    SUM = 0
    MEAN = 1
    MAX = 2
    MIN = 3
    PROD = 4


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _rows_cols(t: Tensor) -> tuple[int, int, tuple[int, ...]]:
    rows = math.prod(t.shape[:-1]) if t.ndim > 1 else 1
    cols = t.shape[-1]
    reduced = t.shape[:-1] if t.ndim > 1 else (1,)
    return rows, cols, reduced


def _normalize_axis(axis: int, ndim: int) -> int:
    norm = axis if axis >= 0 else ndim + axis
    if not (0 <= norm < ndim):
        raise InvalidParamError(f"Axis {axis} out of bounds for {ndim}-D tensor")
    return norm


def _reduced_axis_shape(shape: tuple[int, ...], norm_axis: int, keep_dims: bool) -> tuple[int, ...]:
    if keep_dims:
        return tuple(1 if i == norm_axis else shape[i] for i in range(len(shape)))
    if len(shape) == 1:
        return (1,)
    return tuple(shape[i] for i in range(len(shape)) if i != norm_axis)


def activation(act: Activation, a: Tensor, alpha: float = 0.0, out: Tensor | None = None) -> Tensor:
    a.ensure_open()
    if a.dtype != DType.f32:
        raise InvalidParamError(f"{act.name} is only defined for f32 tensors, got {a.dtype.name}")
    dest = out if out is not None else Tensor.empty(a.shape, dtype=DType.f32)
    try:
        if dest.dtype != DType.f32:
            raise InvalidParamError("Result dtype must be f32")
        alpha_bits = _f32_bits(alpha)
        fn = get_library().dispatch_activation_f32
        return dispatch_strided_unary(
            f"activation({act.name})", a, dest,
            lambda id_in, id_o, rk, sh, st_i, st_o, ln, cg: fn(
                id_in, id_o, rk, sh, st_i, st_o, ln, cg, act.value, alpha_bits
            ),
        )
    except Exception:
        if out is None:
            dest.close()
        raise


def relu(a: Tensor, out: Tensor | None = None) -> Tensor: return activation(Activation.RELU, a, 0.0, out)
def gelu(a: Tensor, out: Tensor | None = None) -> Tensor: return activation(Activation.GELU, a, 0.0, out)
def silu(a: Tensor, out: Tensor | None = None) -> Tensor: return activation(Activation.SILU, a, 0.0, out)
def sigmoid(a: Tensor, out: Tensor | None = None) -> Tensor: return activation(Activation.SIGMOID, a, 0.0, out)
def softplus(a: Tensor, out: Tensor | None = None) -> Tensor: return activation(Activation.SOFTPLUS, a, 0.0, out)
def leaky_relu(a: Tensor, slope: float = 0.01, out: Tensor | None = None) -> Tensor:
    return activation(Activation.LEAKY_RELU, a, slope, out)
def elu(a: Tensor, alpha: float = 1.0, out: Tensor | None = None) -> Tensor:
    return activation(Activation.ELU, a, alpha, out)


def softmax(a: Tensor, logarithmic: bool = False, out: Tensor | None = None) -> Tensor:
    a.ensure_open()
    if a.dtype != DType.f32:
        raise InvalidParamError("softmax requires f32 input")
    dest = out if out is not None else Tensor.empty(a.shape, dtype=DType.f32)
    try:
        require_same_shape(a, dest, "softmax")
        require_contiguous_output(dest)
        lib = get_library()
        with ExitStack() as stack:
            ea = _prepare_contiguous(stack, a)
            rows, cols, _ = _rows_cols(ea)
            check_status(
                lib.dispatch_softmax_rows_f32(ea.vram_id, dest.vram_id, rows, cols, 1 if logarithmic else 0),
                "dispatch_softmax_rows_f32",
                lib,
            )
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def log_softmax(a: Tensor, out: Tensor | None = None) -> Tensor:
    return softmax(a, logarithmic=True, out=out)


def layer_norm(
    a: Tensor,
    gamma: Tensor | None = None,
    beta: Tensor | None = None,
    eps: float = 1e-5,
    out: Tensor | None = None,
) -> Tensor:
    a.ensure_open()
    if a.dtype != DType.f32:
        raise InvalidParamError("layer_norm requires f32 input")
    dest = out if out is not None else Tensor.empty(a.shape, dtype=DType.f32)
    try:
        require_same_shape(a, dest, "layer_norm")
        require_contiguous_output(dest)
        lib = get_library()
        with ExitStack() as stack:
            ea = _prepare_contiguous(stack, a)
            rows, cols, _ = _rows_cols(ea)
            gamma_id = 0
            if gamma is not None:
                gamma.ensure_open()
                eg = _prepare_contiguous(stack, gamma)
                if eg.dtype != DType.f32 or eg.ndim != 1 or eg.shape[0] != cols:
                    raise InvalidParamError(f"layer_norm gamma must be 1-D f32 of length {cols}")
                gamma_id = eg.vram_id
            beta_id = 0
            if beta is not None:
                beta.ensure_open()
                eb = _prepare_contiguous(stack, beta)
                if eb.dtype != DType.f32 or eb.ndim != 1 or eb.shape[0] != cols:
                    raise InvalidParamError(f"layer_norm beta must be 1-D f32 of length {cols}")
                beta_id = eb.vram_id
            check_status(
                lib.dispatch_layernorm_rows_f32(ea.vram_id, gamma_id, beta_id, dest.vram_id, rows, cols, _f32_bits(eps)),
                "dispatch_layernorm_rows_f32",
                lib,
            )
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def rms_norm(
    a: Tensor,
    gamma: Tensor | None = None,
    eps: float = 1e-5,
    out: Tensor | None = None,
) -> Tensor:
    a.ensure_open()
    if a.dtype != DType.f32:
        raise InvalidParamError("rms_norm requires f32 input")
    dest = out if out is not None else Tensor.empty(a.shape, dtype=DType.f32)
    try:
        require_same_shape(a, dest, "rms_norm")
        require_contiguous_output(dest)
        lib = get_library()
        with ExitStack() as stack:
            ea = _prepare_contiguous(stack, a)
            rows, cols, _ = _rows_cols(ea)
            gamma_id = 0
            if gamma is not None:
                gamma.ensure_open()
                eg = _prepare_contiguous(stack, gamma)
                if eg.dtype != DType.f32 or eg.ndim != 1 or eg.shape[0] != cols:
                    raise InvalidParamError(f"rms_norm gamma must be 1-D f32 of length {cols}")
                gamma_id = eg.vram_id
            check_status(
                lib.dispatch_rmsnorm_rows_f32(ea.vram_id, gamma_id, dest.vram_id, rows, cols, _f32_bits(eps)),
                "dispatch_rmsnorm_rows_f32",
                lib,
            )
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def cross_entropy(logits: Tensor, targets: Tensor, out: Tensor | None = None) -> Tensor:
    logits.ensure_open()
    targets.ensure_open()
    if logits.dtype != DType.f32 or targets.dtype != DType.i32:
        raise InvalidParamError("cross_entropy requires f32 logits and i32 targets")
    rows, cols, reduced = _rows_cols(logits)
    dest = out if out is not None else Tensor.empty(reduced, dtype=DType.f32)
    try:
        if dest.dtype != DType.f32 or dest.shape != reduced or targets.shape != reduced:
            raise InvalidParamError(f"Expected targets and output shape {reduced}")
        require_contiguous_output(dest)
        lib = get_library()
        with ExitStack() as stack:
            el = _prepare_contiguous(stack, logits)
            et = _prepare_contiguous(stack, targets)
            check_status(
                lib.dispatch_cross_entropy_rows(el.vram_id, et.vram_id, dest.vram_id, rows, cols),
                "dispatch_cross_entropy_rows",
                lib,
            )
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def add_row_bias(a: Tensor, bias: Tensor, out: Tensor | None = None) -> Tensor:
    a.ensure_open()
    bias.ensure_open()
    if a.dtype != DType.f32 or bias.dtype != DType.f32:
        raise InvalidParamError("add_row_bias requires f32 tensors")
    dest = out if out is not None else Tensor.empty(a.shape, dtype=DType.f32)
    try:
        require_same_shape(a, dest, "add_row_bias")
        require_contiguous_output(dest)
        lib = get_library()
        with ExitStack() as stack:
            ea = _prepare_contiguous(stack, a)
            eb = _prepare_contiguous(stack, bias)
            rows, cols, _ = _rows_cols(ea)
            if eb.ndim != 1 or eb.shape[0] != cols:
                raise InvalidParamError(f"bias must be 1-D of length {cols}, got {eb.shape}")
            check_status(
                lib.dispatch_add_row_bias_f32(ea.vram_id, eb.vram_id, dest.vram_id, rows, cols),
                "dispatch_add_row_bias_f32",
                lib,
            )
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None, out: Tensor | None = None) -> Tensor:
    res = matmul(input, weight, out=out)
    if bias is None:
        return res
    try:
        return add_row_bias(res, bias, out=res)
    except Exception:
        if out is None:
            res.close()
        raise


def linear_transposed(
    input: Tensor,
    weight_out_by_in: Tensor,
    bias: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    res = matmul_transposed_b(input, weight_out_by_in, out=out)
    if bias is None:
        return res
    try:
        res.add_(bias)
        return res
    except Exception:
        if out is None:
            res.close()
        raise


def reduce(op: Reduction, a: Tensor, out: Tensor | None = None) -> Tensor:
    a.ensure_open()
    if a.dtype != DType.f32:
        raise InvalidParamError(f"reduce requires f32, got {a.dtype.name}")
    dest = out if out is not None else Tensor.empty((1,), dtype=DType.f32)
    try:
        if dest.size != 1 or dest.dtype != DType.f32:
            raise InvalidParamError("reduce output must be 1-element f32 tensor")
        require_contiguous_output(dest)
        lib = get_library()
        with ExitStack() as stack:
            ea = _prepare_contiguous(stack, a)
            check_status(
                lib.dispatch_reduce_f32(ea.vram_id, dest.vram_id, ea.size, op.value),
                "dispatch_reduce_f32",
                lib,
            )
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def sum(a: Tensor, axis: int | None = None, keep_dims: bool = False, out: Tensor | None = None) -> Tensor:
    return reduce_axis(Reduction.SUM, a, axis, keep_dims, out) if axis is not None else reduce(Reduction.SUM, a, out)


def mean(a: Tensor, axis: int | None = None, keep_dims: bool = False, out: Tensor | None = None) -> Tensor:
    return reduce_axis(Reduction.MEAN, a, axis, keep_dims, out) if axis is not None else reduce(Reduction.MEAN, a, out)


def max_value(a: Tensor, axis: int | None = None, keep_dims: bool = False, out: Tensor | None = None) -> Tensor:
    return reduce_axis(Reduction.MAX, a, axis, keep_dims, out) if axis is not None else reduce(Reduction.MAX, a, out)


def min_value(a: Tensor, axis: int | None = None, keep_dims: bool = False, out: Tensor | None = None) -> Tensor:
    return reduce_axis(Reduction.MIN, a, axis, keep_dims, out) if axis is not None else reduce(Reduction.MIN, a, out)


def prod(a: Tensor, axis: int | None = None, keep_dims: bool = False, out: Tensor | None = None) -> Tensor:
    return reduce_axis(Reduction.PROD, a, axis, keep_dims, out) if axis is not None else reduce(Reduction.PROD, a, out)


def reduce_rows(op: Reduction, a: Tensor, out: Tensor | None = None) -> Tensor:
    a.ensure_open()
    if a.dtype != DType.f32:
        raise InvalidParamError("reduce_rows requires f32")
    rows, cols, reduced = _rows_cols(a)
    dest = out if out is not None else Tensor.empty(reduced, dtype=DType.f32)
    try:
        if dest.shape != reduced or dest.dtype != DType.f32:
            raise InvalidParamError(f"Expected reduced shape {reduced} f32")
        require_contiguous_output(dest)
        lib = get_library()
        with ExitStack() as stack:
            ea = _prepare_contiguous(stack, a)
            check_status(
                lib.dispatch_reduce_rows_f32(ea.vram_id, dest.vram_id, rows, cols, op.value),
                "dispatch_reduce_rows_f32",
                lib,
            )
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def argmax_rows(a: Tensor, largest: bool = True, out: Tensor | None = None) -> Tensor:
    a.ensure_open()
    if a.dtype != DType.f32:
        raise InvalidParamError("argmax_rows requires f32")
    rows, cols, reduced = _rows_cols(a)
    dest = out if out is not None else Tensor.empty(reduced, dtype=DType.i32)
    try:
        if dest.shape != reduced or dest.dtype != DType.i32:
            raise InvalidParamError(f"Expected reduced shape {reduced} i32")
        require_contiguous_output(dest)
        lib = get_library()
        with ExitStack() as stack:
            ea = _prepare_contiguous(stack, a)
            check_status(
                lib.dispatch_argmax_rows(ea.vram_id, dest.vram_id, rows, cols, 1 if largest else 0),
                "dispatch_argmax_rows",
                lib,
            )
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def reduce_axis(op: Reduction, a: Tensor, axis: int, keep_dims: bool = False, out: Tensor | None = None) -> Tensor:
    a.ensure_open()
    if a.dtype != DType.f32:
        raise InvalidParamError("reduce_axis requires f32")
    norm = _normalize_axis(axis, a.ndim)
    out_shape = _reduced_axis_shape(a.shape, norm, keep_dims)
    dest = out if out is not None else Tensor.empty(out_shape, dtype=DType.f32)
    try:
        if dest.shape != out_shape or dest.dtype != DType.f32:
            raise InvalidParamError(f"Expected output shape {out_shape} f32")
        require_contiguous_output(dest)
        lib = get_library()
        with ExitStack() as stack:
            ea = _prepare_contiguous(stack, a)
            outer = math.prod(ea.shape[:norm])
            axis_dim = ea.shape[norm]
            inner = math.prod(ea.shape[norm + 1 :])
            check_status(
                lib.dispatch_reduce_axis_f32(ea.vram_id, dest.vram_id, outer, axis_dim, inner, op.value),
                "dispatch_reduce_axis_f32",
                lib,
            )
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def argmax_axis(a: Tensor, axis: int, keep_dims: bool = False, largest: bool = True, out: Tensor | None = None) -> Tensor:
    a.ensure_open()
    if a.dtype != DType.f32:
        raise InvalidParamError("argmax_axis requires f32")
    norm = _normalize_axis(axis, a.ndim)
    out_shape = _reduced_axis_shape(a.shape, norm, keep_dims)
    dest = out if out is not None else Tensor.empty(out_shape, dtype=DType.i32)
    try:
        if dest.shape != out_shape or dest.dtype != DType.i32:
            raise InvalidParamError(f"Expected output shape {out_shape} i32")
        require_contiguous_output(dest)
        lib = get_library()
        with ExitStack() as stack:
            ea = _prepare_contiguous(stack, a)
            outer = math.prod(ea.shape[:norm])
            axis_dim = ea.shape[norm]
            inner = math.prod(ea.shape[norm + 1 :])
            check_status(
                lib.dispatch_argmax_axis(ea.vram_id, dest.vram_id, outer, axis_dim, inner, 1 if largest else 0),
                "dispatch_argmax_axis",
                lib,
            )
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def embedding(table: Tensor, token_ids: Tensor, out: Tensor | None = None) -> Tensor:
    table.ensure_open()
    token_ids.ensure_open()
    if table.dtype not in (DType.f32, DType.f16):
        raise InvalidParamError(f"Embedding table must be f32 or f16, got {table.dtype.name}")
    if token_ids.dtype != DType.i32:
        raise InvalidParamError(f"token_ids must be i32, got {token_ids.dtype.name}")
    if table.ndim != 2:
        raise InvalidParamError(f"Embedding table must be 2-D [vocab, hidden], got {table.shape}")
    vocab_size, hidden_dim = table.shape
    out_shape = (*token_ids.shape, hidden_dim)
    dest = out if out is not None else Tensor.empty(out_shape, dtype=DType.f32)
    try:
        if dest.dtype != DType.f32 or dest.shape != out_shape:
            raise InvalidParamError(f"Expected output shape {out_shape} f32")
        require_contiguous_output(dest)
        lib = get_library()
        with ExitStack() as stack:
            et = _prepare_contiguous(stack, table)
            ek = _prepare_contiguous(stack, token_ids)
            if dest.vram_id in (et.vram_id, ek.vram_id):
                raise InvalidParamError("embedding result must not alias operands")
            check_status(
                lib.dispatch_embedding(
                    et.vram_id, ek.vram_id, dest.vram_id,
                    vocab_size, hidden_dim, ek.size, 1 if et.dtype == DType.f16 else 0,
                ),
                "dispatch_embedding",
                lib,
            )
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def rope(x: Tensor, position_offset: int = 0, theta: float = 10000.0, out: Tensor | None = None) -> Tensor:
    x.ensure_open()
    if x.dtype != DType.f32:
        raise InvalidParamError("rope requires f32 input")
    if x.ndim == 4:
        batch, seq_len, num_heads, head_dim = x.shape
    elif x.ndim == 3:
        batch, seq_len, num_heads, head_dim = 1, x.shape[0], x.shape[1], x.shape[2]
    elif x.ndim == 2:
        batch, seq_len, num_heads, head_dim = 1, x.shape[0], 1, x.shape[1]
    else:
        raise InvalidParamError(f"rope requires 2-D..4-D tensor, got {x.shape}")
    if head_dim % 2 != 0:
        raise InvalidParamError(f"head_dim must be even for half-split RoPE, got {head_dim}")

    dest = out if out is not None else Tensor.empty(x.shape, dtype=DType.f32)
    try:
        require_same_shape(x, dest, "rope")
        require_contiguous_output(dest)
        lib = get_library()
        with ExitStack() as stack:
            ex = x.copy() if (x.vram_id == dest.vram_id and not x.has_identical_layout(dest)) else _prepare_contiguous(stack, x)
            if ex is not x:
                stack.enter_context(ex)
            check_status(
                lib.dispatch_rope_f32(
                    ex.vram_id, dest.vram_id, batch, seq_len, num_heads, head_dim, position_offset, _f32_bits(theta)
                ),
                "dispatch_rope_f32",
                lib,
            )
        return dest
    except Exception:
        if out is None:
            dest.close()
        raise


def kv_cache_append(cache: Tensor, new_values: Tensor, pos_offset: int) -> None:
    cache.ensure_open()
    new_values.ensure_open()
    if cache.vram_id == new_values.vram_id:
        raise InvalidParamError("kv_cache_append cache and new_values must not alias")
    require_contiguous_output(cache)
    if cache.dtype != DType.f32 or new_values.dtype != DType.f32:
        raise InvalidParamError("kv_cache_append requires f32 tensors")
    if cache.ndim != 3 or new_values.ndim != 3:
        raise InvalidParamError("kv_cache_append requires 3-D tensors")
    num_kv_heads, max_seq_len, head_dim = cache.shape
    lib = get_library()
    with ExitStack() as stack:
        en = _prepare_contiguous(stack, new_values)
        seq_len, nkh, hd = en.shape
        if nkh != num_kv_heads or hd != head_dim:
            raise InvalidParamError(f"KV shape mismatch: {en.shape} incompatible with cache {cache.shape}")
        if pos_offset < 0 or pos_offset + seq_len > max_seq_len:
            raise InvalidParamError(f"KV cache write out of bounds: pos_offset={pos_offset}, seq={seq_len}, max={max_seq_len}")
        check_status(
            lib.dispatch_kv_cache_append_f32(
                en.vram_id, cache.vram_id, num_kv_heads, max_seq_len, head_dim, seq_len, pos_offset
            ),
            "dispatch_kv_cache_append_f32",
            lib,
        )
