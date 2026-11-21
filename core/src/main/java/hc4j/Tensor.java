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

package hc4j;

import hc4j.engine.Residency;
import hc4j.internal.ffi.NativeCleaner;
import hc4j.internal.ffi.NativeEngine;
import hc4j.internal.ffi.TensorAccess;
import hc4j.internal.layout.Dims;
import hc4j.internal.layout.Rows;
import hc4j.internal.layout.ValidUtil;
import hc4j.internal.ops.ArithmeticOps;
import hc4j.internal.ops.ExponentialOps;
import hc4j.internal.ops.MatmulOps;
import hc4j.internal.ops.NnOps;
import hc4j.internal.ops.TrigOps;
import hc4j.nn.Activation;
import hc4j.nn.Reduction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A VRAM-resident multi-dimensional array of 32-bit floats or integers.
 */
public class Tensor implements AutoCloseable {

    static {
        TensorAccess.setInstance(new TensorAccess() {
            @Override
            public long getVramId(Tensor t) {
                return t.vramId;
            }

            @Override
            public int[] getShapeUnsafe(Tensor t) {
                return t.shape;
            }

            @Override
            public int[] getStridesUnsafe(Tensor t) {
                return t.strides;
            }

            @Override
            public void ensureOpen(Tensor t) {
                t.ensureOpen();
            }

            @Override
            public Tensor create(DType dtype, int[] shape, int[] strides, long vramId) {
                return new Tensor(dtype, shape, strides, vramId);
            }
        });
    }

    private static final class CleanupAction implements Runnable {
        private final long vramId;
        private final AtomicBoolean closed;

        CleanupAction(long vramId, AtomicBoolean closed) {
            this.vramId = vramId;
            this.closed = closed;
        }

        @Override
        public void run() {
            if (closed.compareAndSet(false, true) && vramId != 0L) {
                try {
                    NativeEngine.freeVram(vramId);
                } catch (Throwable ignored) {
                }
            }
        }
    }

    private final DType dtype;
    private final int[] shape;
    private final int[] strides;
    private final long size;
    private final long vramId;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final Cleaner.Cleanable cleanable;

    private Tensor(DType dtype, int[] shape, int[] strides, long vramId) {
        this.dtype = dtype;
        this.shape = Arrays.copyOf(shape, shape.length);
        this.strides = Arrays.copyOf(strides, strides.length);
        this.size = computeSize(this.shape);
        this.vramId = vramId;
        this.cleanable = NativeCleaner.register(this, new CleanupAction(vramId, this.closed));
    }

    private Tensor(long vramId, int[] shape, DType dtype) {
        this(dtype, shape, computeContiguousStrides(shape), vramId);
    }

    // ------------------------------------------------------------------------------------------
    // Factories
    // ------------------------------------------------------------------------------------------

    public static Tensor zeros(DType dtype, int... shape) {
        long totalSize = computeSize(shape);
        long vramId = NativeEngine.allocVram(totalSize);
        return new Tensor(vramId, shape, dtype);
    }

    public static Tensor empty(DType dtype, int... shape) {
        long totalSize = computeSize(shape);
        long vramId = NativeEngine.allocVramUninit(totalSize);
        return new Tensor(vramId, shape, dtype);
    }

    public static Tensor fromArray(float[] values, int... shape) {
        long totalSize = checkLength(values.length, shape);
        long vramId = NativeEngine.allocVramUninit(totalSize);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, values);
            NativeEngine.writeVram(vramId, hostSegment, totalSize);
        } catch (RuntimeException | Error e) {
            NativeEngine.freeVram(vramId);
            throw e;
        }
        return new Tensor(vramId, shape, DType.f32);
    }

    public static Tensor fromArray(int[] values, int... shape) {
        long totalSize = checkLength(values.length, shape);
        long vramId = NativeEngine.allocVramUninit(totalSize);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocateFrom(ValueLayout.JAVA_INT, values);
            NativeEngine.writeVram(vramId, hostSegment, totalSize);
        } catch (RuntimeException | Error e) {
            NativeEngine.freeVram(vramId);
            throw e;
        }
        return new Tensor(vramId, shape, DType.i32);
    }

    // ------------------------------------------------------------------------------------------
    // Lifecycle & State
    // ------------------------------------------------------------------------------------------

    public boolean isClosed() {
        return closed.get();
    }

    public void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("Tensor has already been closed");
        }
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    public DType getDType() {
        return dtype;
    }

    public int[] getShape() {
        return shape.clone();
    }

    public long getSize() {
        return size;
    }

    public int dim() {
        return shape.length;
    }

    public Residency residency() {
        ensureOpen();
        return NativeEngine.residency(vramId);
    }

    public void evict() {
        ensureOpen();
        NativeEngine.evict(vramId);
    }

    // ------------------------------------------------------------------------------------------
    // Elementwise Arithmetic
    // ------------------------------------------------------------------------------------------

    public Tensor add(Tensor other) {
        ensureOpen();
        Tensor res = empty(dtype, shape);
        try { return add(other, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public Tensor add(Tensor other, Tensor res) { ensureOpen(); return ArithmeticOps.add(this, other, res); }

    public Tensor sub(Tensor other) {
        ensureOpen();
        Tensor res = empty(dtype, shape);
        try { return sub(other, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public Tensor sub(Tensor other, Tensor res) { ensureOpen(); return ArithmeticOps.sub(this, other, res); }

    public Tensor mul(Tensor other) {
        ensureOpen();
        Tensor res = empty(dtype, shape);
        try { return mul(other, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public Tensor mul(Tensor other, Tensor res) { ensureOpen(); return ArithmeticOps.mul(this, other, res); }

    public Tensor div(Tensor other) {
        ensureOpen();
        Tensor res = empty(dtype, shape);
        try { return div(other, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public Tensor div(Tensor other, Tensor res) { ensureOpen(); return ArithmeticOps.div(this, other, res); }

    // Scalar arithmetic
    public Tensor add(float c) { ensureOpen(); return lazy().add(c).eval(); }
    public Tensor sub(float c) { ensureOpen(); return lazy().sub(c).eval(); }
    public Tensor mul(float c) { ensureOpen(); return lazy().mul(c).eval(); }
    public Tensor div(float c) { ensureOpen(); return lazy().div(c).eval(); }

    public Tensor add(float c, Tensor res) { ensureOpen(); return lazy().add(c).eval(res); }
    public Tensor sub(float c, Tensor res) { ensureOpen(); return lazy().sub(c).eval(res); }
    public Tensor mul(float c, Tensor res) { ensureOpen(); return lazy().mul(c).eval(res); }
    public Tensor div(float c, Tensor res) { ensureOpen(); return lazy().div(c).eval(res); }

    // ------------------------------------------------------------------------------------------
    // Trigonometric
    // ------------------------------------------------------------------------------------------

    public Tensor sin()   { return trig(TrigOps.Function.SIN); }
    public Tensor cos()   { return trig(TrigOps.Function.COS); }
    public Tensor tan()   { return trig(TrigOps.Function.TAN); }
    public Tensor asin()  { return trig(TrigOps.Function.ASIN); }
    public Tensor acos()  { return trig(TrigOps.Function.ACOS); }
    public Tensor atan()  { return trig(TrigOps.Function.ATAN); }
    public Tensor sinh()  { return trig(TrigOps.Function.SINH); }
    public Tensor cosh()  { return trig(TrigOps.Function.COSH); }
    public Tensor tanh()  { return trig(TrigOps.Function.TANH); }
    public Tensor asinh() { return trig(TrigOps.Function.ASINH); }
    public Tensor acosh() { return trig(TrigOps.Function.ACOSH); }
    public Tensor atanh() { return trig(TrigOps.Function.ATANH); }

    public Tensor sin(Tensor res)   { ensureOpen(); return TrigOps.apply(TrigOps.Function.SIN, this, res); }
    public Tensor cos(Tensor res)   { ensureOpen(); return TrigOps.apply(TrigOps.Function.COS, this, res); }
    public Tensor tan(Tensor res)   { ensureOpen(); return TrigOps.apply(TrigOps.Function.TAN, this, res); }
    public Tensor asin(Tensor res)  { ensureOpen(); return TrigOps.apply(TrigOps.Function.ASIN, this, res); }
    public Tensor acos(Tensor res)  { ensureOpen(); return TrigOps.apply(TrigOps.Function.ACOS, this, res); }
    public Tensor atan(Tensor res)  { ensureOpen(); return TrigOps.apply(TrigOps.Function.ATAN, this, res); }
    public Tensor sinh(Tensor res)  { ensureOpen(); return TrigOps.apply(TrigOps.Function.SINH, this, res); }
    public Tensor cosh(Tensor res)  { ensureOpen(); return TrigOps.apply(TrigOps.Function.COSH, this, res); }
    public Tensor tanh(Tensor res)  { ensureOpen(); return TrigOps.apply(TrigOps.Function.TANH, this, res); }
    public Tensor asinh(Tensor res) { ensureOpen(); return TrigOps.apply(TrigOps.Function.ASINH, this, res); }
    public Tensor acosh(Tensor res) { ensureOpen(); return TrigOps.apply(TrigOps.Function.ACOSH, this, res); }
    public Tensor atanh(Tensor res) { ensureOpen(); return TrigOps.apply(TrigOps.Function.ATANH, this, res); }

    private Tensor trig(TrigOps.Function fn) {
        ensureOpen();
        ValidUtil.requireOperandF32(fn.opName(), dtype);
        Tensor res = empty(dtype, shape);
        try {
            return TrigOps.apply(fn, this, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    // ------------------------------------------------------------------------------------------
    // Exponential & Logarithmic
    // ------------------------------------------------------------------------------------------

    public Tensor exp()   { return exponential(ExponentialOps.Function.EXP); }
    public Tensor log()   { return exponential(ExponentialOps.Function.LOG); }
    public Tensor log2()  { return exponential(ExponentialOps.Function.LOG2); }
    public Tensor log10() { return exponential(ExponentialOps.Function.LOG10); }
    public Tensor sqrt()  { return exponential(ExponentialOps.Function.SQRT); }

    public Tensor exp(Tensor res)   { ensureOpen(); return ExponentialOps.apply(ExponentialOps.Function.EXP, this, res); }
    public Tensor log(Tensor res)   { ensureOpen(); return ExponentialOps.apply(ExponentialOps.Function.LOG, this, res); }
    public Tensor log2(Tensor res)  { ensureOpen(); return ExponentialOps.apply(ExponentialOps.Function.LOG2, this, res); }
    public Tensor log10(Tensor res) { ensureOpen(); return ExponentialOps.apply(ExponentialOps.Function.LOG10, this, res); }
    public Tensor sqrt(Tensor res)  { ensureOpen(); return ExponentialOps.apply(ExponentialOps.Function.SQRT, this, res); }

    private Tensor exponential(ExponentialOps.Function fn) {
        ensureOpen();
        ValidUtil.requireOperandF32(fn.opName(), dtype);
        Tensor res = empty(dtype, shape);
        try {
            return ExponentialOps.apply(fn, this, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    // ------------------------------------------------------------------------------------------
    // Matmul & Transpose
    // ------------------------------------------------------------------------------------------

    public Tensor matmul(Tensor other) {
        ensureOpen();
        ValidUtil.requireOperandF32("matmul", dtype);
        Dims d = ValidUtil.matmulDims(shape, other.shape, false, false);
        Tensor res = empty(DType.f32, d.resultShape());
        try {
            return MatmulOps.matmul(this, other, res, false, false);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor matmul(Tensor other, Tensor res) {
        ensureOpen();
        return MatmulOps.matmul(this, other, res, false, false);
    }

    public Tensor transpose() {
        ensureOpen();
        ValidUtil.require2D("transpose", shape);
        Tensor res = empty(dtype, shape[1], shape[0]);
        try {
            return MatmulOps.transpose(this, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor transpose(Tensor res) {
        ensureOpen();
        return MatmulOps.transpose(this, res);
    }

    // ------------------------------------------------------------------------------------------
    // Neural Network Ops
    // ------------------------------------------------------------------------------------------

    public Tensor relu() { return activation(Activation.RELU, 0f); }
    public Tensor gelu() { return activation(Activation.GELU, 0f); }
    public Tensor silu() { return activation(Activation.SILU, 0f); }
    public Tensor sigmoid() { return activation(Activation.SIGMOID, 0f); }
    public Tensor softplus() { return activation(Activation.SOFTPLUS, 0f); }
    public Tensor leakyRelu(float slope) { return activation(Activation.LEAKY_RELU, slope); }
    public Tensor elu(float alpha) { return activation(Activation.ELU, alpha); }

    public Tensor relu(Tensor res) { ensureOpen(); return NnOps.activation(Activation.RELU, this, res, 0f); }
    public Tensor gelu(Tensor res) { ensureOpen(); return NnOps.activation(Activation.GELU, this, res, 0f); }
    public Tensor silu(Tensor res) { ensureOpen(); return NnOps.activation(Activation.SILU, this, res, 0f); }
    public Tensor sigmoid(Tensor res) { ensureOpen(); return NnOps.activation(Activation.SIGMOID, this, res, 0f); }
    public Tensor softplus(Tensor res) { ensureOpen(); return NnOps.activation(Activation.SOFTPLUS, this, res, 0f); }
    public Tensor leakyRelu(float slope, Tensor res) { ensureOpen(); return NnOps.activation(Activation.LEAKY_RELU, this, res, slope); }
    public Tensor elu(float alpha, Tensor res) { ensureOpen(); return NnOps.activation(Activation.ELU, this, res, alpha); }

    private Tensor activation(Activation act, float alpha) {
        ensureOpen();
        ValidUtil.requireOperandF32(act.name(), dtype);
        Tensor res = empty(dtype, shape);
        try {
            return NnOps.activation(act, this, res, alpha);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    // Reductions
    public float sum() { return reduceToScalar(Reduction.SUM); }
    public float mean() { return reduceToScalar(Reduction.MEAN); }
    public float maxValue() { return reduceToScalar(Reduction.MAX); }
    public float minValue() { return reduceToScalar(Reduction.MIN); }

    public float sumSquares() {
        ensureOpen();
        try (Tensor r = empty(DType.f32, 1)) {
            NnOps.reduceSquare(Reduction.SUM, this, r);
            return r.toFloatArray()[0];
        }
    }

    public float mse(Tensor target) {
        ensureOpen();
        try (Tensor r = empty(DType.f32, 1)) {
            NnOps.reduceDiffSquared(Reduction.MEAN, this, target, r);
            return r.toFloatArray()[0];
        }
    }

    public Tensor sum(Tensor res) { ensureOpen(); return NnOps.reduce(Reduction.SUM, this, res); }
    public Tensor mean(Tensor res) { ensureOpen(); return NnOps.reduce(Reduction.MEAN, this, res); }
    public Tensor maxValue(Tensor res) { ensureOpen(); return NnOps.reduce(Reduction.MAX, this, res); }
    public Tensor minValue(Tensor res) { ensureOpen(); return NnOps.reduce(Reduction.MIN, this, res); }

    private float reduceToScalar(Reduction op) {
        ensureOpen();
        try (Tensor r = empty(DType.f32, 1)) {
            NnOps.reduce(op, this, r);
            return r.toFloatArray()[0];
        }
    }

    public Tensor sumRows() { return reduceRowsFresh(Reduction.SUM); }
    public Tensor meanRows() { return reduceRowsFresh(Reduction.MEAN); }
    public Tensor maxRows() { return reduceRowsFresh(Reduction.MAX); }
    public Tensor minRows() { return reduceRowsFresh(Reduction.MIN); }

    public Tensor sumRows(Tensor res) { ensureOpen(); return NnOps.reduceRows(Reduction.SUM, this, res); }
    public Tensor meanRows(Tensor res) { ensureOpen(); return NnOps.reduceRows(Reduction.MEAN, this, res); }
    public Tensor maxRows(Tensor res) { ensureOpen(); return NnOps.reduceRows(Reduction.MAX, this, res); }
    public Tensor minRows(Tensor res) { ensureOpen(); return NnOps.reduceRows(Reduction.MIN, this, res); }

    public Tensor argmaxRows() { return argRowsFresh(true); }
    public Tensor argminRows() { return argRowsFresh(false); }

    public Tensor argmaxRows(Tensor res) { ensureOpen(); return NnOps.argmaxRows(this, res, true); }
    public Tensor argminRows(Tensor res) { ensureOpen(); return NnOps.argmaxRows(this, res, false); }

    private Tensor reduceRowsFresh(Reduction op) {
        ensureOpen();
        Rows r = ValidUtil.rows(shape, size);
        Tensor res = empty(DType.f32, r.reducedShape(shape));
        try {
            return NnOps.reduceRows(op, this, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    private Tensor argRowsFresh(boolean largest) {
        ensureOpen();
        Rows r = ValidUtil.rows(shape, size);
        Tensor res = empty(DType.i32, r.reducedShape(shape));
        try {
            return NnOps.argmaxRows(this, res, largest);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor softmax() { return softmaxFresh(false); }
    public Tensor logSoftmax() { return softmaxFresh(true); }

    public Tensor softmax(Tensor res) { ensureOpen(); return NnOps.softmaxRows(this, res, false); }
    public Tensor logSoftmax(Tensor res) { ensureOpen(); return NnOps.softmaxRows(this, res, true); }

    private Tensor softmaxFresh(boolean logarithmic) {
        ensureOpen();
        ValidUtil.requireOperandF32("softmax", dtype);
        Tensor res = empty(dtype, shape);
        try {
            return NnOps.softmaxRows(this, res, logarithmic);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor layerNorm(float eps) { return layerNormFresh(null, null, eps); }
    public Tensor layerNorm(Tensor gamma, Tensor beta, float eps) { return layerNormFresh(gamma, beta, eps); }
    public Tensor layerNorm(Tensor gamma, Tensor beta, float eps, Tensor res) {
        ensureOpen();
        return NnOps.layerNormRows(this, gamma, beta, res, eps);
    }

    public Tensor rmsNorm(Tensor gamma, float eps) {
        ensureOpen();
        ValidUtil.requireOperandF32("rmsNorm", dtype);
        Tensor res = empty(dtype, shape);
        try {
            return NnOps.rmsNormRows(this, gamma, res, eps);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor rmsNorm(Tensor gamma, float eps, Tensor res) {
        ensureOpen();
        return NnOps.rmsNormRows(this, gamma, res, eps);
    }

    private Tensor layerNormFresh(Tensor gamma, Tensor beta, float eps) {
        ensureOpen();
        ValidUtil.requireOperandF32("layerNorm", dtype);
        Tensor res = empty(dtype, shape);
        try {
            return NnOps.layerNormRows(this, gamma, beta, res, eps);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor crossEntropy(Tensor targets) {
        ensureOpen();
        Rows r = ValidUtil.rows(shape, size);
        Tensor res = empty(DType.f32, r.reducedShape(shape));
        try {
            return NnOps.crossEntropyRows(this, targets, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor crossEntropy(Tensor targets, Tensor res) {
        ensureOpen();
        return NnOps.crossEntropyRows(this, targets, res);
    }

    public Tensor addRowBias(Tensor bias) {
        ensureOpen();
        ValidUtil.requireOperandF32("addRowBias", dtype);
        Tensor res = empty(dtype, shape);
        try {
            return NnOps.addRowBias(this, bias, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor addRowBias(Tensor bias, Tensor res) {
        ensureOpen();
        return NnOps.addRowBias(this, bias, res);
    }

    public Tensor linear(Tensor weights, Tensor bias) {
        ensureOpen();
        Tensor res = matmul(weights);
        try {
            return bias == null ? res : NnOps.addRowBias(res, bias, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor linear(Tensor weights, Tensor bias, Tensor res) {
        ensureOpen();
        matmul(weights, res);
        return bias == null ? res : NnOps.addRowBias(res, bias, res);
    }

    // ------------------------------------------------------------------------------------------
    // Fusion
    // ------------------------------------------------------------------------------------------

    public FusedExpr lazy() {
        ensureOpen();
        return FusedExpr.of(this);
    }

    // ------------------------------------------------------------------------------------------
    // Readback
    // ------------------------------------------------------------------------------------------

    public float[] toFloatArray() {
        ensureOpen();
        if (this.dtype != DType.f32) {
            throw new IllegalStateException("Cannot readback " + this.dtype + " as float[]");
        }
        float[] result = new float[(int) this.size];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocate(ValueLayout.JAVA_FLOAT, this.size);
            NativeEngine.downloadVram(this.vramId, hostSegment, this.size);
            MemorySegment.copy(hostSegment, ValueLayout.JAVA_FLOAT, 0, result, 0, (int) this.size);
        }
        return result;
    }

    public int[] toIntArray() {
        ensureOpen();
        if (this.dtype != DType.i32) {
            throw new IllegalStateException("Cannot readback " + this.dtype + " as int[]");
        }
        int[] result = new int[(int) this.size];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocate(ValueLayout.JAVA_INT, this.size);
            NativeEngine.downloadVram(this.vramId, hostSegment, this.size);
            MemorySegment.copy(hostSegment, ValueLayout.JAVA_INT, 0, result, 0, (int) this.size);
        }
        return result;
    }

    // ------------------------------------------------------------------------------------------
    // Internal Utilities
    // ------------------------------------------------------------------------------------------

    private static int[] computeContiguousStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int acc = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }

    private static long computeSize(int[] shape) {
        long total = 1;
        for (int dim : shape) total *= dim;
        return total;
    }

    private static long checkLength(int length, int[] shape) {
        long total = computeSize(shape);
        if (length != total) {
            throw new IllegalArgumentException("Array has " + length + " elements but shape "
                    + Arrays.toString(shape) + " needs " + total);
        }
        return total;
    }

    @Override
    public String toString() {
        return "Tensor(vramId=" + vramId + ", shape=" + Arrays.toString(shape) + ", dtype=" + dtype + ", size=" + size + ")";
    }
}
