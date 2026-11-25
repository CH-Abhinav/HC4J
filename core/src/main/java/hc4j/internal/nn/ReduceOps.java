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

package hc4j.internal.nn;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.internal.ffi.NativeBindings;
import hc4j.internal.ffi.NativeEngine;
import hc4j.internal.ffi.TensorAccess;
import hc4j.internal.layout.Rows;
import hc4j.internal.layout.ValidUtil;
import hc4j.nn.Reduction;
import java.util.Objects;

/**
 * Dispatcher for scalar, row-wise, and arbitrary-axis tensor reductions and argmax/argmin kernels.
 */
public final class ReduceOps {

    private ReduceOps() {
        throw new AssertionError("No instances of ReduceOps");
    }

    /**
     * Reduces all elements of {@code a} to a single-element {@code f32} tensor {@code res}.
     *
     * @param op reduction operator
     * @param a input {@code f32} tensor
     * @param res 1-element destination {@code f32} tensor
     * @return {@code res}
     */
    public static Tensor reduce(Reduction op, Tensor a, Tensor res) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("reduce", a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);
        if (res.getSize() != 1) {
            throw new IllegalArgumentException("reduce target must have 1 element, got " + res.getSize());
        }

        try (ValidUtil.SafeOperand safeA = ValidUtil.prepareContiguousZeroOffset(a)) {
            Tensor ea = safeA.tensor();
            int status = NativeBindings.dispatchReduce(access.getVramId(ea), access.getVramId(res), ea.getSize(), op.ordinal());
            NativeEngine.checkStatus(status, "ReduceOps.reduce (" + op + ")");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: reduce failed", t);
        }
    }

    /**
     * Reduces squared element values {@code a^2} across all elements into {@code res}.
     *
     * @param op reduction operator
     * @param a input {@code f32} tensor
     * @param res 1-element destination {@code f32} tensor
     * @return {@code res}
     */
    public static Tensor reduceSquare(Reduction op, Tensor a, Tensor res) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("reduceSquare", a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);
        if (res.getSize() != 1) {
            throw new IllegalArgumentException("reduceSquare target must have 1 element, got " + res.getSize());
        }

        try (ValidUtil.SafeOperand safeA = ValidUtil.prepareContiguousZeroOffset(a)) {
            Tensor ea = safeA.tensor();
            int status = NativeBindings.dispatchReduceSquare(
                    access.getVramId(ea), access.getVramId(res), ea.getSize(), op.ordinal());
            NativeEngine.checkStatus(status, "ReduceOps.reduceSquare (" + op + ")");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: reduceSquare failed", t);
        }
    }

    /**
     * Reduces squared elementwise differences {@code (a - b)^2} across matching-shape tensors into {@code res}.
     *
     * @param op reduction operator
     * @param a first {@code f32} tensor
     * @param b second {@code f32} tensor
     * @param res 1-element destination {@code f32} tensor
     * @return {@code res}
     */
    public static Tensor reduceDiffSquared(Reduction op, Tensor a, Tensor b, Tensor res) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("reduceDiffSquared", a.getDType());
        ValidUtil.requireOperandF32("reduceDiffSquared", b.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);
        if (res.getSize() != 1) {
            throw new IllegalArgumentException("reduceDiffSquared target must have 1 element, got " + res.getSize());
        }

        try (ValidUtil.SafeOperand safeA = ValidUtil.prepareContiguousZeroOffset(a);
             ValidUtil.SafeOperand safeB = ValidUtil.prepareContiguousZeroOffset(b)) {
            Tensor ea = safeA.tensor();
            Tensor eb = safeB.tensor();
            ValidUtil.requireSameShape(access.getShapeUnsafe(ea), access.getShapeUnsafe(eb));
            int status = NativeBindings.dispatchReduceDiff2(
                    access.getVramId(ea), access.getVramId(eb), access.getVramId(res), ea.getSize(), op.ordinal());
            NativeEngine.checkStatus(status, "ReduceOps.reduceDiffSquared (" + op + ")");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: reduceDiffSquared failed", t);
        }
    }

    /**
     * Reduces each row along the trailing dimension into {@code res}.
     *
     * @param op reduction operator
     * @param a input {@code f32} tensor
     * @param res destination {@code f32} tensor matching {@code Rows.reducedShape}
     * @return {@code res}
     */
    public static Tensor reduceRows(Reduction op, Tensor a, Tensor res) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("reduceRows", a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeA = ValidUtil.prepareContiguousZeroOffset(a)) {
            Tensor ea = safeA.tensor();
            int[] sa = access.getShapeUnsafe(ea);
            Rows r = ValidUtil.rows(sa, ea.getSize());
            ValidUtil.requireResultShape(r.reducedShape(sa), access.getShapeUnsafe(res));

            int status = NativeBindings.dispatchReduceRows(
                    access.getVramId(ea), access.getVramId(res), r.rows(), r.cols(), op.ordinal());
            NativeEngine.checkStatus(status, "ReduceOps.reduceRows (" + op + ")");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: reduceRows failed", t);
        }
    }

    /**
     * Computes argmax or argmin along each row (trailing dimension) into an {@code i32} tensor.
     *
     * @param a input {@code f32} tensor
     * @param res destination {@code i32} tensor matching {@code Rows.reducedShape}
     * @param largest true for argmax, false for argmin
     * @return {@code res}
     */
    public static Tensor argmaxRows(Tensor a, Tensor res, boolean largest) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("argmaxRows", a.getDType());
        ValidUtil.requireResultDType(DType.i32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeA = ValidUtil.prepareContiguousZeroOffset(a)) {
            Tensor ea = safeA.tensor();
            int[] sa = access.getShapeUnsafe(ea);
            Rows r = ValidUtil.rows(sa, ea.getSize());
            ValidUtil.requireResultShape(r.reducedShape(sa), access.getShapeUnsafe(res));

            int status = NativeBindings.dispatchArgmaxRows(
                    access.getVramId(ea), access.getVramId(res), r.rows(), r.cols(), largest ? 1 : 0);
            NativeEngine.checkStatus(status, "ReduceOps.argmaxRows");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: argmaxRows failed", t);
        }
    }

    /**
     * Reduces {@code in} along an arbitrary axis into {@code res}.
     *
     * @param in input {@code f32} tensor
     * @param axis axis to reduce (negative values normalized relative to rank)
     * @param reduceOp reduction ordinal (0..4)
     * @param res destination {@code f32} tensor
     * @return {@code res}
     */
    public static Tensor reduceAxis(Tensor in, int axis, int reduceOp, Tensor res) {
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(in);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("reduceAxis", in.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeIn = ValidUtil.prepareContiguousZeroOffset(in)) {
            Tensor ein = safeIn.tensor();
            int[] shape = access.getShapeUnsafe(ein);
            int normAxis = ValidUtil.normalizeAxis(axis, shape.length);

            int outer = product(shape, 0, normAxis);
            int axisDim = shape[normAxis];
            int inner = product(shape, normAxis + 1, shape.length);

            int status = NativeBindings.dispatchReduceAxis(
                    access.getVramId(ein), access.getVramId(res), outer, axisDim, inner, reduceOp);
            NativeEngine.checkStatus(status, "ReduceOps.reduceAxis");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native reduceAxis failed", t);
        }
    }

    /**
     * Computes argmax or argmin along an arbitrary axis into an {@code i32} tensor {@code res}.
     *
     * @param in input {@code f32} tensor
     * @param axis axis to reduce
     * @param largest true for argmax, false for argmin
     * @param res destination {@code i32} tensor
     * @return {@code res}
     */
    public static Tensor argmaxAxis(Tensor in, int axis, boolean largest, Tensor res) {
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(in);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("argmaxAxis", in.getDType());
        ValidUtil.requireResultDType(DType.i32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeIn = ValidUtil.prepareContiguousZeroOffset(in)) {
            Tensor ein = safeIn.tensor();
            int[] shape = access.getShapeUnsafe(ein);
            int normAxis = ValidUtil.normalizeAxis(axis, shape.length);

            int outer = product(shape, 0, normAxis);
            int axisDim = shape[normAxis];
            int inner = product(shape, normAxis + 1, shape.length);

            int status = NativeBindings.dispatchArgmaxAxis(
                    access.getVramId(ein), access.getVramId(res), outer, axisDim, inner, largest ? 1 : 0);
            NativeEngine.checkStatus(status, "ReduceOps.argmaxAxis");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native argmaxAxis failed", t);
        }
    }

    private static int product(int[] shape, int startInclusive, int endExclusive) {
        int acc = 1;
        for (int i = startInclusive; i < endExclusive; i++) {
            acc *= shape[i];
        }
        return acc;
    }
}
