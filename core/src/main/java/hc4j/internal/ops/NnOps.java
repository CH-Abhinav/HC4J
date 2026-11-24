/*
 * Copyright © 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package hc4j.internal.ops;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.internal.ffi.NativeBindings;
import hc4j.internal.ffi.NativeEngine;
import hc4j.internal.ffi.TensorAccess;
import hc4j.internal.layout.Rows;
import hc4j.internal.layout.ValidUtil;
import hc4j.nn.Activation;
import hc4j.nn.Reduction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Neural network operations: activations, reductions, normalizations, softmax, losses.
 */
public final class NnOps {

    private NnOps() {
        throw new AssertionError("No instances of NnOps");
    }

    public static Rows rows(Tensor t) {
        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(t);
        return ValidUtil.rows(access.getShapeUnsafe(t), t.getSize());
    }

    public static Tensor activation(Activation act, Tensor a, Tensor res, float alpha) {
        Objects.requireNonNull(act, "act");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        if (access.getVramId(a) == access.getVramId(res) && !access.hasIdenticalLayout(a, res)) {
            try (Tensor copyA = a.copy()) {
                return activation(act, copyA, res, alpha);
            }
        }

        if (access.elementOffset(a) > 0) {
            try (Tensor contigA = a.contiguous()) {
                return activation(act, contigA, res, alpha);
            }
        }

        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32(act.name(), a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());

        int[] sa = access.getShapeUnsafe(a);
        int[] sr = access.getShapeUnsafe(res);
        ValidUtil.requireSameShape(act.name(), sa, sr);

        int[] sta = access.getStridesUnsafe(a);
        int[] str = access.getStridesUnsafe(res);

        boolean contiguous = ValidUtil.isContiguous(sa, sta) && ValidUtil.isContiguous(sr, str);
        int rank = res.dim();
        long length = res.getSize();
        int alphaBits = Float.floatToRawIntBits(alpha);

        try {
            int status;
            if (contiguous) {
                status = NativeBindings.dispatchActivation(
                        access.getVramId(a), access.getVramId(res), rank,
                        MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL,
                        length, 1, act.ordinal(), alphaBits);
            } else {
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment shapeSeg = arena.allocateFrom(ValueLayout.JAVA_INT, sr);
                    MemorySegment stridesASeg = arena.allocateFrom(ValueLayout.JAVA_INT, sta);
                    MemorySegment stridesCSeg = arena.allocateFrom(ValueLayout.JAVA_INT, str);
                    status = NativeBindings.dispatchActivation(
                            access.getVramId(a), access.getVramId(res), rank,
                            shapeSeg, stridesASeg, stridesCSeg,
                            length, 0, act.ordinal(), alphaBits);
                }
            }
            NativeEngine.checkStatus(status, "NnOps.activation (" + act + ")");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J GPU Error: " + act + " failed", t);
        }
    }

    public static Tensor reduce(Reduction op, Tensor a, Tensor res) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        if (!a.isContiguous() || access.elementOffset(a) != 0) {
            try (Tensor contigA = a.contiguous()) {
                return reduce(op, contigA, res);
            }
        }

        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("reduce", a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        if (res.getSize() != 1) {
            throw new IllegalArgumentException("reduce target must have 1 element, got " + res.getSize());
        }

        try {
            int status = NativeBindings.dispatchReduce(access.getVramId(a), access.getVramId(res), a.getSize(), op.ordinal());
            NativeEngine.checkStatus(status, "NnOps.reduce (" + op + ")");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J GPU Error: reduce failed", t);
        }
    }

    public static Tensor reduceSquare(Reduction op, Tensor a, Tensor res) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        if (!a.isContiguous() || access.elementOffset(a) != 0) {
            try (Tensor contigA = a.contiguous()) {
                return reduceSquare(op, contigA, res);
            }
        }

        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("reduceSquare", a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        if (res.getSize() != 1) {
            throw new IllegalArgumentException("reduceSquare target must have 1 element, got " + res.getSize());
        }

        try {
            int status = NativeBindings.dispatchReduceSquare(access.getVramId(a), access.getVramId(res), a.getSize(), op.ordinal());
            NativeEngine.checkStatus(status, "NnOps.reduceSquare (" + op + ")");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J GPU Error: reduceSquare failed", t);
        }
    }

    public static Tensor reduceDiffSquared(Reduction op, Tensor a, Tensor b, Tensor res) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        access.ensureOpen(res);

        if (!a.isContiguous() || access.elementOffset(a) != 0) {
            try (Tensor contigA = a.contiguous()) {
                return reduceDiffSquared(op, contigA, b, res);
            }
        }
        if (!b.isContiguous() || access.elementOffset(b) != 0) {
            try (Tensor contigB = b.contiguous()) {
                return reduceDiffSquared(op, a, contigB, res);
            }
        }

        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("reduceDiffSquared", a.getDType());
        ValidUtil.requireOperandF32("reduceDiffSquared", b.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireSameShape(access.getShapeUnsafe(a), access.getShapeUnsafe(b));

        if (res.getSize() != 1) {
            throw new IllegalArgumentException("reduceDiffSquared target must have 1 element, got " + res.getSize());
        }

        try {
            int status = NativeBindings.dispatchReduceDiff2(
                    access.getVramId(a), access.getVramId(b), access.getVramId(res), a.getSize(), op.ordinal());
            NativeEngine.checkStatus(status, "NnOps.reduceDiffSquared (" + op + ")");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J GPU Error: reduceDiffSquared failed", t);
        }
    }

    public static Tensor reduceRows(Reduction op, Tensor a, Tensor res) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        if (!a.isContiguous() || access.elementOffset(a) != 0) {
            try (Tensor contigA = a.contiguous()) {
                return reduceRows(op, contigA, res);
            }
        }

        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("reduceRows", a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());

        int[] sa = access.getShapeUnsafe(a);
        Rows r = ValidUtil.rows(sa, a.getSize());
        ValidUtil.requireResultShape(r.reducedShape(sa), access.getShapeUnsafe(res));

        try {
            int status = NativeBindings.dispatchReduceRows(
                    access.getVramId(a), access.getVramId(res), r.rows(), r.cols(), op.ordinal());
            NativeEngine.checkStatus(status, "NnOps.reduceRows (" + op + ")");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J GPU Error: reduceRows failed", t);
        }
    }

    public static Tensor argmaxRows(Tensor a, Tensor res, boolean largest) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        if (!a.isContiguous() || access.elementOffset(a) != 0) {
            try (Tensor contigA = a.contiguous()) {
                return argmaxRows(contigA, res, largest);
            }
        }

        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("argmaxRows", a.getDType());
        ValidUtil.requireResultDType(DType.i32, res.getDType());

        int[] sa = access.getShapeUnsafe(a);
        Rows r = ValidUtil.rows(sa, a.getSize());
        ValidUtil.requireResultShape(r.reducedShape(sa), access.getShapeUnsafe(res));

        try {
            int status = NativeBindings.dispatchArgmaxRows(
                    access.getVramId(a), access.getVramId(res), r.rows(), r.cols(), largest ? 1 : 0);
            NativeEngine.checkStatus(status, "NnOps.argmaxRows");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J GPU Error: argmaxRows failed", t);
        }
    }

    public static Tensor softmaxRows(Tensor a, Tensor res, boolean logarithmic) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        if (!a.isContiguous() || access.elementOffset(a) != 0) {
            try (Tensor contigA = a.contiguous()) {
                return softmaxRows(contigA, res, logarithmic);
            }
        }

        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("softmaxRows", a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());

        int[] sa = access.getShapeUnsafe(a);
        ValidUtil.requireResultShape(sa, access.getShapeUnsafe(res));
        Rows r = ValidUtil.rows(sa, a.getSize());

        try {
            int status = NativeBindings.dispatchSoftmaxRows(
                    access.getVramId(a), access.getVramId(res), r.rows(), r.cols(), logarithmic ? 1 : 0);
            NativeEngine.checkStatus(status, "NnOps.softmaxRows");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J GPU Error: softmaxRows failed", t);
        }
    }

    public static Tensor layerNormRows(Tensor a, Tensor gamma, Tensor beta, Tensor res, float eps) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        if (!a.isContiguous() || access.elementOffset(a) != 0) {
            try (Tensor contigA = a.contiguous()) {
                return layerNormRows(contigA, gamma, beta, res, eps);
            }
        }
        if (gamma != null && (!gamma.isContiguous() || access.elementOffset(gamma) != 0)) {
            try (Tensor contigG = gamma.contiguous()) {
                return layerNormRows(a, contigG, beta, res, eps);
            }
        }
        if (beta != null && (!beta.isContiguous() || access.elementOffset(beta) != 0)) {
            try (Tensor contigB = beta.contiguous()) {
                return layerNormRows(a, gamma, contigB, res, eps);
            }
        }

        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("layerNormRows", a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());

        int[] sa = access.getShapeUnsafe(a);
        ValidUtil.requireResultShape(sa, access.getShapeUnsafe(res));
        Rows r = ValidUtil.rows(sa, a.getSize());

        long gammaId = 0;
        if (gamma != null) {
            access.ensureOpen(gamma);
            ValidUtil.requireOperandF32("layerNorm gamma", gamma.getDType());
            if (gamma.dim() != 1 || access.getShapeUnsafe(gamma)[0] != r.cols()) {
                throw new IllegalArgumentException("layerNorm gamma must be 1-D of length " + r.cols());
            }
            gammaId = access.getVramId(gamma);
        }

        long betaId = 0;
        if (beta != null) {
            access.ensureOpen(beta);
            ValidUtil.requireOperandF32("layerNorm beta", beta.getDType());
            if (beta.dim() != 1 || access.getShapeUnsafe(beta)[0] != r.cols()) {
                throw new IllegalArgumentException("layerNorm beta must be 1-D of length " + r.cols());
            }
            betaId = access.getVramId(beta);
        }

        try {
            int status = NativeBindings.dispatchLayerNormRows(
                    access.getVramId(a), gammaId, betaId, access.getVramId(res),
                    r.rows(), r.cols(), Float.floatToRawIntBits(eps));
            NativeEngine.checkStatus(status, "NnOps.layerNormRows");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J GPU Error: layerNormRows failed", t);
        }
    }

    public static Tensor rmsNormRows(Tensor a, Tensor gamma, Tensor res, float eps) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        if (!a.isContiguous() || access.elementOffset(a) != 0) {
            try (Tensor contigA = a.contiguous()) {
                return rmsNormRows(contigA, gamma, res, eps);
            }
        }
        if (gamma != null && (!gamma.isContiguous() || access.elementOffset(gamma) != 0)) {
            try (Tensor contigG = gamma.contiguous()) {
                return rmsNormRows(a, contigG, res, eps);
            }
        }

        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("rmsNormRows", a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());

        int[] sa = access.getShapeUnsafe(a);
        ValidUtil.requireResultShape(sa, access.getShapeUnsafe(res));
        Rows r = ValidUtil.rows(sa, a.getSize());

        long gammaId = 0;
        if (gamma != null) {
            access.ensureOpen(gamma);
            ValidUtil.requireOperandF32("rmsNorm gamma", gamma.getDType());
            if (gamma.dim() != 1 || access.getShapeUnsafe(gamma)[0] != r.cols()) {
                throw new IllegalArgumentException("rmsNorm gamma must be 1-D of length " + r.cols());
            }
            gammaId = access.getVramId(gamma);
        }

        try {
            int status = NativeBindings.dispatchRmsNormRows(
                    access.getVramId(a), gammaId, access.getVramId(res),
                    r.rows(), r.cols(), Float.floatToRawIntBits(eps));
            NativeEngine.checkStatus(status, "NnOps.rmsNormRows");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J GPU Error: rmsNormRows failed", t);
        }
    }

    public static Tensor crossEntropyRows(Tensor logits, Tensor targets, Tensor res) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(targets, "targets");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(logits);
        access.ensureOpen(targets);
        access.ensureOpen(res);

        if (!logits.isContiguous() || access.elementOffset(logits) != 0) {
            try (Tensor contigL = logits.contiguous()) {
                return crossEntropyRows(contigL, targets, res);
            }
        }
        if (!targets.isContiguous() || access.elementOffset(targets) != 0) {
            try (Tensor contigT = targets.contiguous()) {
                return crossEntropyRows(logits, contigT, res);
            }
        }

        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("crossEntropy logits", logits.getDType());
        if (targets.getDType() != DType.i32) {
            throw new IllegalArgumentException("crossEntropy targets must be i32, got " + targets.getDType());
        }
        ValidUtil.requireResultDType(DType.f32, res.getDType());

        int[] sl = access.getShapeUnsafe(logits);
        Rows r = ValidUtil.rows(sl, logits.getSize());
        int[] expectedShape = r.reducedShape(sl);
        ValidUtil.requireResultShape(expectedShape, access.getShapeUnsafe(targets));
        ValidUtil.requireResultShape(expectedShape, access.getShapeUnsafe(res));

        try {
            int status = NativeBindings.dispatchCrossEntropyRows(
                    access.getVramId(logits), access.getVramId(targets), access.getVramId(res), r.rows(), r.cols());
            NativeEngine.checkStatus(status, "NnOps.crossEntropyRows");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J GPU Error: crossEntropyRows failed", t);
        }
    }

    public static Tensor addRowBias(Tensor a, Tensor bias, Tensor res) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(bias);
        access.ensureOpen(res);

        if (!a.isContiguous() || access.elementOffset(a) != 0) {
            try (Tensor contigA = a.contiguous()) {
                return addRowBias(contigA, bias, res);
            }
        }
        if (!bias.isContiguous() || access.elementOffset(bias) != 0) {
            try (Tensor contigB = bias.contiguous()) {
                return addRowBias(a, contigB, res);
            }
        }

        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("addRowBias input", a.getDType());
        ValidUtil.requireOperandF32("addRowBias bias", bias.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());

        int[] sa = access.getShapeUnsafe(a);
        ValidUtil.requireResultShape(sa, access.getShapeUnsafe(res));
        Rows r = ValidUtil.rows(sa, a.getSize());

        if (bias.dim() != 1 || access.getShapeUnsafe(bias)[0] != r.cols()) {
            throw new IllegalArgumentException("bias must be a 1-D tensor of length " + r.cols() + ", got " + bias.dim() + "-D");
        }

        try {
            int status = NativeBindings.dispatchAddRowBias(
                    access.getVramId(a), access.getVramId(bias), access.getVramId(res), r.rows(), r.cols());
            NativeEngine.checkStatus(status, "NnOps.addRowBias");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J GPU Error: addRowBias failed", t);
        }
    }
}
