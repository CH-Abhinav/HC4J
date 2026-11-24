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
import hc4j.internal.layout.ValidUtil;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Objects;

/**
 * Dispatcher for advanced mathematical, comparison, masking, casting, and axis reduction operations.
 */
public final class MathOps {

    private MathOps() {
        throw new AssertionError("No instances of MathOps");
    }

    private static int resolveDTypeCode(DType type) {
        return switch (type) {
            case i32 -> 0;
            case f32 -> 1;
        };
    }

    // ------------------------------------------------------------------------------------------
    // Unary Math Ops (f32)
    // ------------------------------------------------------------------------------------------

    public static Tensor applyUnaryF32(int opIndex, String opName, Tensor input, Tensor out) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(out, "out");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(input);
        access.ensureOpen(out);

        ValidUtil.requireOperandF32(opName, input.getDType());
        ValidUtil.requireResultDType(DType.f32, out.getDType());

        if (access.getVramId(input) == access.getVramId(out) && !access.hasIdenticalLayout(input, out)) {
            try (Tensor copy = input.copy()) {
                return applyUnaryF32(opIndex, opName, copy, out);
            }
        }

        if (access.elementOffset(input) > 0) {
            try (Tensor contig = input.contiguous()) {
                return applyUnaryF32(opIndex, opName, contig, out);
            }
        }
        if (!out.isContiguous() || access.elementOffset(out) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        int[] s = access.getShapeUnsafe(input);
        ValidUtil.requireResultShape(s, access.getShapeUnsafe(out));

        boolean contiguous = input.isContiguous() && out.isContiguous();
        int rank = s.length;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shapeSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, s);
            MemorySegment stridesASeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, access.getStridesUnsafe(input));
            MemorySegment stridesCSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, access.getStridesUnsafe(out));

            int status = NativeBindings.dispatchMathUnary(
                    opIndex, access.getVramId(input), access.getVramId(out),
                    rank, shapeSeg, stridesASeg, stridesCSeg,
                    out.getSize(), contiguous ? 1 : 0
            );
            NativeEngine.checkStatus(status, "MathOps." + opName);
            return out;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native " + opName + " failed", t);
        }
    }

    public static Tensor clampF32(Tensor input, float lo, float hi, Tensor out) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(out, "out");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(input);
        access.ensureOpen(out);

        ValidUtil.requireOperandF32("clamp", input.getDType());
        ValidUtil.requireResultDType(DType.f32, out.getDType());

        if (access.getVramId(input) == access.getVramId(out) && !access.hasIdenticalLayout(input, out)) {
            try (Tensor copy = input.copy()) {
                return clampF32(copy, lo, hi, out);
            }
        }

        if (access.elementOffset(input) > 0) {
            try (Tensor contig = input.contiguous()) {
                return clampF32(contig, lo, hi, out);
            }
        }
        if (!out.isContiguous() || access.elementOffset(out) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        int[] s = access.getShapeUnsafe(input);
        ValidUtil.requireResultShape(s, access.getShapeUnsafe(out));

        boolean contiguous = input.isContiguous() && out.isContiguous();
        int rank = s.length;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shapeSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, s);
            MemorySegment stridesASeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, access.getStridesUnsafe(input));
            MemorySegment stridesCSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, access.getStridesUnsafe(out));

            int status = NativeBindings.dispatchClampF32(
                    access.getVramId(input), access.getVramId(out),
                    rank, shapeSeg, stridesASeg, stridesCSeg,
                    out.getSize(), contiguous ? 1 : 0, lo, hi
            );
            NativeEngine.checkStatus(status, "MathOps.clampF32");
            return out;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native clamp failed", t);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Unary Math Ops (i32)
    // ------------------------------------------------------------------------------------------

    public static Tensor applyUnaryI32(int opCode, String opName, Tensor input, Tensor out, int pZ, int pW) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(out, "out");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(input);
        access.ensureOpen(out);

        if (input.getDType() != DType.i32) {
            throw new IllegalArgumentException(opName + " on i32 requires an i32 tensor, got " + input.getDType());
        }
        ValidUtil.requireResultDType(DType.i32, out.getDType());

        if (access.getVramId(input) == access.getVramId(out) && !access.hasIdenticalLayout(input, out)) {
            try (Tensor copy = input.copy()) {
                return applyUnaryI32(opCode, opName, copy, out, pZ, pW);
            }
        }

        if (access.elementOffset(input) > 0) {
            try (Tensor contig = input.contiguous()) {
                return applyUnaryI32(opCode, opName, contig, out, pZ, pW);
            }
        }
        if (!out.isContiguous() || access.elementOffset(out) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        int[] s = access.getShapeUnsafe(input);
        ValidUtil.requireResultShape(s, access.getShapeUnsafe(out));

        boolean contiguous = input.isContiguous() && out.isContiguous();
        int rank = s.length;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shapeSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, s);
            MemorySegment stridesASeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, access.getStridesUnsafe(input));
            MemorySegment stridesCSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, access.getStridesUnsafe(out));

            int status = NativeBindings.dispatchUnaryI32(
                    access.getVramId(input), access.getVramId(out),
                    rank, shapeSeg, stridesASeg, stridesCSeg,
                    out.getSize(), contiguous ? 1 : 0, opCode, pZ, pW
            );
            NativeEngine.checkStatus(status, "MathOps." + opName);
            return out;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native " + opName + " failed", t);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Functional Binary Ops: pow (f32), maximum, minimum (f32 & i32)
    // ------------------------------------------------------------------------------------------

    public static Tensor pow(Tensor a, Tensor b, Tensor res) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("pow", a.getDType());
        ValidUtil.requireOperandF32("pow", b.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());

        if (access.getVramId(a) == access.getVramId(res) && !access.hasIdenticalLayout(a, res)) {
            try (Tensor copyA = a.copy()) {
                return pow(copyA, b, res);
            }
        }
        if (access.getVramId(b) == access.getVramId(res) && !access.hasIdenticalLayout(b, res)) {
            try (Tensor copyB = b.copy()) {
                return pow(a, copyB, res);
            }
        }

        if (access.elementOffset(a) > 0) {
            try (Tensor contigA = a.contiguous()) {
                return pow(contigA, b, res);
            }
        }
        if (access.elementOffset(b) > 0) {
            try (Tensor contigB = b.contiguous()) {
                return pow(a, contigB, res);
            }
        }
        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        int[] sa = access.getShapeUnsafe(a);
        int[] sb = access.getShapeUnsafe(b);
        int[] targetShape = ValidUtil.broadcastShape(sa, sb);

        ValidUtil.requireResultShape(targetShape, access.getShapeUnsafe(res));

        int[] sta = access.getStridesUnsafe(a);
        int[] stb = access.getStridesUnsafe(b);
        int[] str = access.getStridesUnsafe(res);

        int[] bcastStridesA = ValidUtil.broadcastStrides(sa, sta, targetShape);
        int[] bcastStridesB = ValidUtil.broadcastStrides(sb, stb, targetShape);

        boolean contiguous = Arrays.equals(sa, targetShape)
                && Arrays.equals(sb, targetShape)
                && a.isContiguous()
                && b.isContiguous()
                && res.isContiguous();

        int rank = targetShape.length;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shapeSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, targetShape);
            MemorySegment stridesASeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesA);
            MemorySegment stridesBSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesB);
            MemorySegment stridesCSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, str);

            int status = NativeBindings.dispatchPowF32(
                    access.getVramId(a), access.getVramId(b), access.getVramId(res),
                    rank, shapeSeg, stridesASeg, stridesBSeg, stridesCSeg,
                    res.getSize(), contiguous ? 1 : 0
            );
            NativeEngine.checkStatus(status, "MathOps.pow");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native pow failed", t);
        }
    }

    public static Tensor maxMin(boolean isMax, Tensor a, Tensor b, Tensor res) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        access.ensureOpen(res);

        ValidUtil.requireSameDType(a.getDType(), b.getDType());
        ValidUtil.requireResultDType(a.getDType(), res.getDType());

        if (access.getVramId(a) == access.getVramId(res) && !access.hasIdenticalLayout(a, res)) {
            try (Tensor copyA = a.copy()) {
                return maxMin(isMax, copyA, b, res);
            }
        }
        if (access.getVramId(b) == access.getVramId(res) && !access.hasIdenticalLayout(b, res)) {
            try (Tensor copyB = b.copy()) {
                return maxMin(isMax, a, copyB, res);
            }
        }

        if (access.elementOffset(a) > 0) {
            try (Tensor contigA = a.contiguous()) {
                return maxMin(isMax, contigA, b, res);
            }
        }
        if (access.elementOffset(b) > 0) {
            try (Tensor contigB = b.contiguous()) {
                return maxMin(isMax, a, contigB, res);
            }
        }
        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        int[] sa = access.getShapeUnsafe(a);
        int[] sb = access.getShapeUnsafe(b);
        int[] targetShape = ValidUtil.broadcastShape(sa, sb);

        ValidUtil.requireResultShape(targetShape, access.getShapeUnsafe(res));

        int[] sta = access.getStridesUnsafe(a);
        int[] stb = access.getStridesUnsafe(b);
        int[] str = access.getStridesUnsafe(res);

        int[] bcastStridesA = ValidUtil.broadcastStrides(sa, sta, targetShape);
        int[] bcastStridesB = ValidUtil.broadcastStrides(sb, stb, targetShape);

        boolean contiguous = Arrays.equals(sa, targetShape)
                && Arrays.equals(sb, targetShape)
                && a.isContiguous()
                && b.isContiguous()
                && res.isContiguous();

        int rank = targetShape.length;
        int dtypeCode = resolveDTypeCode(a.getDType());

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shapeSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, targetShape);
            MemorySegment stridesASeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesA);
            MemorySegment stridesBSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesB);
            MemorySegment stridesCSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, str);

            int status = isMax
                    ? NativeBindings.dispatchMaximum(
                            access.getVramId(a), access.getVramId(b), access.getVramId(res),
                            rank, shapeSeg, stridesASeg, stridesBSeg, stridesCSeg,
                            res.getSize(), contiguous ? 1 : 0, dtypeCode)
                    : NativeBindings.dispatchMinimum(
                            access.getVramId(a), access.getVramId(b), access.getVramId(res),
                            rank, shapeSeg, stridesASeg, stridesBSeg, stridesCSeg,
                            res.getSize(), contiguous ? 1 : 0, dtypeCode);

            NativeEngine.checkStatus(status, isMax ? "MathOps.maximum" : "MathOps.minimum");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native " + (isMax ? "maximum" : "minimum") + " failed", t);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Comparison Ops (eq, ne, gt, ge, lt, le -> DType.i32 mask)
    // ------------------------------------------------------------------------------------------

    public static Tensor compare(int cmpOp, String opName, Tensor a, Tensor b, Tensor res) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        access.ensureOpen(res);

        ValidUtil.requireSameDType(a.getDType(), b.getDType());
        ValidUtil.requireResultDType(DType.i32, res.getDType());

        if (access.getVramId(a) == access.getVramId(res) && !access.hasIdenticalLayout(a, res)) {
            try (Tensor copyA = a.copy()) {
                return compare(cmpOp, opName, copyA, b, res);
            }
        }
        if (access.getVramId(b) == access.getVramId(res) && !access.hasIdenticalLayout(b, res)) {
            try (Tensor copyB = b.copy()) {
                return compare(cmpOp, opName, a, copyB, res);
            }
        }

        if (access.elementOffset(a) > 0) {
            try (Tensor contigA = a.contiguous()) {
                return compare(cmpOp, opName, contigA, b, res);
            }
        }
        if (access.elementOffset(b) > 0) {
            try (Tensor contigB = b.contiguous()) {
                return compare(cmpOp, opName, a, contigB, res);
            }
        }
        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        int[] sa = access.getShapeUnsafe(a);
        int[] sb = access.getShapeUnsafe(b);
        int[] targetShape = ValidUtil.broadcastShape(sa, sb);

        ValidUtil.requireResultShape(targetShape, access.getShapeUnsafe(res));

        int[] sta = access.getStridesUnsafe(a);
        int[] stb = access.getStridesUnsafe(b);
        int[] str = access.getStridesUnsafe(res);

        int[] bcastStridesA = ValidUtil.broadcastStrides(sa, sta, targetShape);
        int[] bcastStridesB = ValidUtil.broadcastStrides(sb, stb, targetShape);

        boolean contiguous = Arrays.equals(sa, targetShape)
                && Arrays.equals(sb, targetShape)
                && a.isContiguous()
                && b.isContiguous()
                && res.isContiguous();

        int rank = targetShape.length;
        int dtypeCode = resolveDTypeCode(a.getDType());

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shapeSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, targetShape);
            MemorySegment stridesASeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesA);
            MemorySegment stridesBSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesB);
            MemorySegment stridesCSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, str);

            int status = NativeBindings.dispatchCompare(
                    access.getVramId(a), access.getVramId(b), access.getVramId(res),
                    rank, shapeSeg, stridesASeg, stridesBSeg, stridesCSeg,
                    res.getSize(), contiguous ? 1 : 0, dtypeCode, cmpOp
            );
            NativeEngine.checkStatus(status, "MathOps." + opName);
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native " + opName + " failed", t);
        }
    }

    // ------------------------------------------------------------------------------------------
    // 3-Way Broadcasted Where
    // ------------------------------------------------------------------------------------------

    public static Tensor where(Tensor cond, Tensor x, Tensor y, Tensor res) {
        Objects.requireNonNull(cond, "cond");
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(cond);
        access.ensureOpen(x);
        access.ensureOpen(y);
        access.ensureOpen(res);

        ValidUtil.requireSameDType(x.getDType(), y.getDType());
        ValidUtil.requireResultDType(x.getDType(), res.getDType());

        if (access.getVramId(cond) == access.getVramId(res) && !access.hasIdenticalLayout(cond, res)) {
            try (Tensor copyCond = cond.copy()) {
                return where(copyCond, x, y, res);
            }
        }
        if (access.getVramId(x) == access.getVramId(res) && !access.hasIdenticalLayout(x, res)) {
            try (Tensor copyX = x.copy()) {
                return where(cond, copyX, y, res);
            }
        }
        if (access.getVramId(y) == access.getVramId(res) && !access.hasIdenticalLayout(y, res)) {
            try (Tensor copyY = y.copy()) {
                return where(cond, x, copyY, res);
            }
        }

        if (access.elementOffset(cond) > 0) {
            try (Tensor contigCond = cond.contiguous()) {
                return where(contigCond, x, y, res);
            }
        }
        if (access.elementOffset(x) > 0) {
            try (Tensor contigX = x.contiguous()) {
                return where(cond, contigX, y, res);
            }
        }
        if (access.elementOffset(y) > 0) {
            try (Tensor contigY = y.contiguous()) {
                return where(cond, x, contigY, res);
            }
        }
        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        int[] sc = access.getShapeUnsafe(cond);
        int[] sx = access.getShapeUnsafe(x);
        int[] sy = access.getShapeUnsafe(y);

        int[] targetShape = ValidUtil.broadcastShape(ValidUtil.broadcastShape(sc, sx), sy);
        ValidUtil.requireResultShape(targetShape, access.getShapeUnsafe(res));

        int[] stc = access.getStridesUnsafe(cond);
        int[] stx = access.getStridesUnsafe(x);
        int[] sty = access.getStridesUnsafe(y);
        int[] str = access.getStridesUnsafe(res);

        int[] bcastStridesCond = ValidUtil.broadcastStrides(sc, stc, targetShape);
        int[] bcastStridesX = ValidUtil.broadcastStrides(sx, stx, targetShape);
        int[] bcastStridesY = ValidUtil.broadcastStrides(sy, sty, targetShape);

        boolean contiguous = Arrays.equals(sc, targetShape)
                && Arrays.equals(sx, targetShape)
                && Arrays.equals(sy, targetShape)
                && cond.isContiguous()
                && x.isContiguous()
                && y.isContiguous()
                && res.isContiguous();

        int rank = targetShape.length;
        int condDtypeCode = resolveDTypeCode(cond.getDType());
        int valDtypeCode = resolveDTypeCode(x.getDType());

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shapeSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, targetShape);
            MemorySegment stridesCondSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesCond);
            MemorySegment stridesXSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesX);
            MemorySegment stridesYSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesY);
            MemorySegment stridesOutSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, str);

            int status = NativeBindings.dispatchWhere(
                    access.getVramId(cond), access.getVramId(x), access.getVramId(y), access.getVramId(res),
                    rank, shapeSeg, stridesCondSeg, stridesXSeg, stridesYSeg, stridesOutSeg,
                    res.getSize(), contiguous ? 1 : 0, condDtypeCode, valDtypeCode
            );
            NativeEngine.checkStatus(status, "MathOps.where");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native where failed", t);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Cast Ops (f32 <-> i32)
    // ------------------------------------------------------------------------------------------

    public static Tensor cast(Tensor src, DType targetDType, Tensor res) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(targetDType, "targetDType");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(src);
        access.ensureOpen(res);

        ValidUtil.requireResultDType(targetDType, res.getDType());

        if (!src.isContiguous() || access.elementOffset(src) != 0) {
            try (Tensor contig = src.contiguous()) {
                return cast(contig, targetDType, res);
            }
        }
        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireResultShape(access.getShapeUnsafe(src), access.getShapeUnsafe(res));

        int srcDtype = resolveDTypeCode(src.getDType());
        int dstDtype = resolveDTypeCode(targetDType);

        try {
            int status = NativeBindings.dispatchCast(
                    access.getVramId(src), access.getVramId(res),
                    src.getSize(), srcDtype, dstDtype
            );
            NativeEngine.checkStatus(status, "MathOps.cast");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native cast failed", t);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Axis Reductions & ArgMax / ArgMin
    // ------------------------------------------------------------------------------------------

    public static Tensor reduceAxis(Tensor in, int axis, int reduceOp, Tensor res) {
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(in);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("reduceAxis", in.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());

        if (!in.isContiguous() || access.elementOffset(in) != 0) {
            try (Tensor contig = in.contiguous()) {
                return reduceAxis(contig, axis, reduceOp, res);
            }
        }
        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        int[] shape = access.getShapeUnsafe(in);
        int normAxis = ValidUtil.normalizeAxis(axis, shape.length);

        int outer = 1;
        for (int i = 0; i < normAxis; i++) {
            outer *= shape[i];
        }
        int axisDim = shape[normAxis];
        int inner = 1;
        for (int i = normAxis + 1; i < shape.length; i++) {
            inner *= shape[i];
        }

        try {
            int status = NativeBindings.dispatchReduceAxis(
                    access.getVramId(in), access.getVramId(res),
                    outer, axisDim, inner, reduceOp
            );
            NativeEngine.checkStatus(status, "MathOps.reduceAxis");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native reduceAxis failed", t);
        }
    }

    public static Tensor argmaxAxis(Tensor in, int axis, boolean largest, Tensor res) {
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(in);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("argmaxAxis", in.getDType());
        ValidUtil.requireResultDType(DType.i32, res.getDType());

        if (!in.isContiguous() || access.elementOffset(in) != 0) {
            try (Tensor contig = in.contiguous()) {
                return argmaxAxis(contig, axis, largest, res);
            }
        }
        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        int[] shape = access.getShapeUnsafe(in);
        int normAxis = ValidUtil.normalizeAxis(axis, shape.length);

        int outer = 1;
        for (int i = 0; i < normAxis; i++) {
            outer *= shape[i];
        }
        int axisDim = shape[normAxis];
        int inner = 1;
        for (int i = normAxis + 1; i < shape.length; i++) {
            inner *= shape[i];
        }

        try {
            int status = NativeBindings.dispatchArgmaxAxis(
                    access.getVramId(in), access.getVramId(res),
                    outer, axisDim, inner, largest ? 1 : 0
            );
            NativeEngine.checkStatus(status, "MathOps.argmaxAxis");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native argmaxAxis failed", t);
        }
    }
}
