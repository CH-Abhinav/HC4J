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
import hc4j.internal.layout.Dims;
import hc4j.internal.layout.ValidUtil;
import hc4j.linalg.Linalg;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Internal matrix multiplication and transposition dispatching.
 */
public final class MatmulOps {

    private static final int FLAG_TRANS_A = 1;
    private static final int FLAG_TRANS_B = 2;

    private MatmulOps() {
        throw new AssertionError("No instances of MatmulOps");
    }

    public static Dims dims(int[] sa, int[] sb, boolean transposeA, boolean transposeB) {
        return ValidUtil.matmulDims(sa, sb, transposeA, transposeB);
    }

    public static Tensor matmul(Tensor a, Tensor b, Tensor res, boolean transposeA, boolean transposeB) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        access.ensureOpen(res);

        if (!a.isContiguous() || access.elementOffset(a) != 0) {
            try (Tensor contigA = a.contiguous()) {
                return matmul(contigA, b, res, transposeA, transposeB);
            }
        }
        if (!b.isContiguous() || access.elementOffset(b) != 0) {
            try (Tensor contigB = b.contiguous()) {
                return matmul(a, contigB, res, transposeA, transposeB);
            }
        }
        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("matmul", a.getDType());
        if (b.getDType() != DType.f32 && b.getDType() != DType.f16) {
            throw new IllegalArgumentException("Right operand must be f32 or f16, got " + b.getDType());
        }
        ValidUtil.requireResultDType(DType.f32, res.getDType());

        int[] sa = access.getShapeUnsafe(a);
        int[] sb = access.getShapeUnsafe(b);
        Dims d = ValidUtil.matmulDims(sa, sb, transposeA, transposeB);

        int[] sr = access.getShapeUnsafe(res);
        ValidUtil.requireResultShape(d.resultShape(), sr);

        long idA = access.getVramId(a);
        long idB = access.getVramId(b);
        long idRes = access.getVramId(res);

        ValidUtil.requireNoAlias(idRes, idA, "matmul result must not alias an operand");
        ValidUtil.requireNoAlias(idRes, idB, "matmul result must not alias an operand");

        if (b.getDType() == DType.f16) {
            if (transposeA) {
                throw new UnsupportedOperationException("transposeA is not supported for f16 operands");
            }
            int flags = transposeB ? FLAG_TRANS_B : 0;
            try {
                int status = NativeBindings.dispatchMatmulF32F16(idA, idB, idRes, d.m(), d.n(), d.k(), flags);
                NativeEngine.checkStatus(status, "MatmulOps.matmul f32 x f16");
                return res;
            } catch (Throwable t) {
                if (t instanceof RuntimeException re) throw re;
                throw new RuntimeException("HC4J native matmul f32 x f16 failed", t);
            }
        }

        int flags = (transposeA && a.dim() == 2 ? FLAG_TRANS_A : 0) | (transposeB && b.dim() == 2 ? FLAG_TRANS_B : 0);

        try {
            int status = flags == 0
                    ? NativeBindings.dispatchMatmul(idA, idB, idRes, d.m(), d.n(), d.k())
                    : NativeBindings.dispatchMatmulEx(idA, idB, idRes, d.m(), d.n(), d.k(), flags);
            NativeEngine.checkStatus(status, "MatmulOps.matmul");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native matmul failed", t);
        }
    }

    public static int[] transposedBOutputShape(Tensor a, Tensor b) {
        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        int[] sa = access.getShapeUnsafe(a);
        int[] sb = access.getShapeUnsafe(b);
        if (sb.length != 2) {
            throw new IllegalArgumentException("transposedB requires a 2-D right operand [N, K], got " + java.util.Arrays.toString(sb));
        }
        int n = sb[0];
        int k = sb[1];
        if (sa.length == 1) {
            if (sa[0] != k) {
                throw new IllegalArgumentException("Inner dimension mismatch: " + sa[0] + " vs " + k);
            }
            return new int[] {n};
        }
        int ka = sa[sa.length - 1];
        if (ka != k) {
            throw new IllegalArgumentException("Inner dimension mismatch: " + ka + " vs " + k);
        }
        int[] outShape = new int[sa.length];
        System.arraycopy(sa, 0, outShape, 0, sa.length - 1);
        outShape[sa.length - 1] = n;
        return outShape;
    }

    public static Tensor matmulTransposedB(Tensor a, Tensor b, Tensor res) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        access.ensureOpen(res);

        if (!a.isContiguous() || access.elementOffset(a) != 0) {
            try (Tensor contigA = a.contiguous()) {
                return matmulTransposedB(contigA, b, res);
            }
        }
        if (!b.isContiguous() || access.elementOffset(b) != 0) {
            try (Tensor contigB = b.contiguous()) {
                return matmulTransposedB(a, contigB, res);
            }
        }
        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("matmulTransposedB input", a.getDType());
        if (b.getDType() != DType.f32 && b.getDType() != DType.f16) {
            throw new IllegalArgumentException("matmulTransposedB right operand must be f32 or f16, got " + b.getDType());
        }
        ValidUtil.requireResultDType(DType.f32, res.getDType());

        int[] sa = access.getShapeUnsafe(a);
        int[] sb = access.getShapeUnsafe(b);
        if (sb.length != 2) {
            throw new IllegalArgumentException("matmulTransposedB requires a 2-D right operand [N, K], got " + java.util.Arrays.toString(sb));
        }

        int n = sb[0];
        int k = sb[1];
        int m;
        if (sa.length == 1) {
            if (sa[0] != k) {
                throw new IllegalArgumentException("Inner dimension mismatch: " + sa[0] + " vs " + k);
            }
            m = 1;
        } else {
            int ka = sa[sa.length - 1];
            if (ka != k) {
                throw new IllegalArgumentException("Inner dimension mismatch: " + ka + " vs " + k);
            }
            long mLong = 1;
            for (int i = 0; i < sa.length - 1; i++) {
                mLong = Math.multiplyExact(mLong, sa[i]);
            }
            if (mLong > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Total leading dimension exceeds Integer.MAX_VALUE: " + mLong);
            }
            m = (int) mLong;
        }

        int[] expectedOutShape = transposedBOutputShape(a, b);
        ValidUtil.requireResultShape(expectedOutShape, access.getShapeUnsafe(res));

        long idA = access.getVramId(a);
        long idB = access.getVramId(b);
        long idRes = access.getVramId(res);

        ValidUtil.requireNoAlias(idRes, idA, "matmul result must not alias an operand");
        ValidUtil.requireNoAlias(idRes, idB, "matmul result must not alias an operand");

        try {
            int status;
            if (b.getDType() == DType.f16) {
                status = NativeBindings.dispatchMatmulF32F16(idA, idB, idRes, m, n, k, FLAG_TRANS_B);
            } else {
                status = NativeBindings.dispatchMatmulEx(idA, idB, idRes, m, n, k, FLAG_TRANS_B);
            }
            NativeEngine.checkStatus(status, "MatmulOps.matmulTransposedB");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native matmulTransposedB failed", t);
        }
    }

    public static Tensor batchedMatmul(Tensor a, Tensor b, Tensor res, boolean transposeA, boolean transposeB) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        access.ensureOpen(res);

        if (access.elementOffset(a) > 0) {
            try (Tensor contigA = a.contiguous()) {
                return batchedMatmul(contigA, b, res, transposeA, transposeB);
            }
        }
        if (access.elementOffset(b) > 0) {
            try (Tensor contigB = b.contiguous()) {
                return batchedMatmul(a, contigB, res, transposeA, transposeB);
            }
        }
        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("batched matmul", a.getDType());
        ValidUtil.requireOperandF32("batched matmul", b.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());

        int[] sa = access.getShapeUnsafe(a);
        int[] sta = access.getStridesUnsafe(a);
        int[] sb = access.getShapeUnsafe(b);
        int[] stb = access.getStridesUnsafe(b);

        ValidUtil.BatchedMatmulPlan plan = ValidUtil.batchedMatmulPlan(sa, sta, sb, stb);
        ValidUtil.requireResultShape(plan.outputShape(), access.getShapeUnsafe(res));

        long idA = access.getVramId(a);
        long idB = access.getVramId(b);
        long idRes = access.getVramId(res);

        ValidUtil.requireNoAlias(idRes, idA, "batched matmul result must not alias an operand");
        ValidUtil.requireNoAlias(idRes, idB, "batched matmul result must not alias an operand");

        int batchRank = plan.batchShape().length;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment batchShapeSeg = arena.allocateFrom(ValueLayout.JAVA_INT, plan.batchShape());
            MemorySegment aBatchStridesSeg = arena.allocateFrom(ValueLayout.JAVA_INT, plan.aBatchStrides());
            MemorySegment bBatchStridesSeg = arena.allocateFrom(ValueLayout.JAVA_INT, plan.bBatchStrides());

            int status = NativeBindings.dispatchBatchedMatmul(
                    idA, idB, idRes,
                    plan.m(), plan.n(), plan.k(),
                    batchRank,
                    batchShapeSeg, aBatchStridesSeg, bBatchStridesSeg,
                    plan.flags()
            );
            NativeEngine.checkStatus(status, "MatmulOps.batchedMatmul");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native batched matmul failed", t);
        }
    }

    public static Tensor transpose(Tensor in, Tensor res) {
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(in);
        access.ensureOpen(res);

        if (!in.isContiguous() || access.elementOffset(in) != 0) {
            try (Tensor contig = in.contiguous()) {
                return transpose(contig, res);
            }
        }
        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        ValidUtil.requireOperandF32("transpose", in.getDType());
        ValidUtil.requireResultDType(in.getDType(), res.getDType());

        int[] s = access.getShapeUnsafe(in);
        ValidUtil.require2D("transpose", s);

        int[] expectedOutShape = new int[] {s[1], s[0]};
        ValidUtil.requireResultShape(expectedOutShape, access.getShapeUnsafe(res));

        long idIn = access.getVramId(in);
        long idRes = access.getVramId(res);
        ValidUtil.requireNoAlias(idRes, idIn, "transpose cannot run in place");

        try {
            int status = NativeBindings.dispatchTranspose(idIn, idRes, s[0], s[1]);
            NativeEngine.checkStatus(status, "MatmulOps.transpose");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native transpose failed", t);
        }
    }

    public static Linalg.Plan plan(int m, int n, int k, boolean transposeA, boolean transposeB) {
        int flags = (transposeA ? FLAG_TRANS_A : 0) | (transposeB ? FLAG_TRANS_B : 0);
        try {
            int code = NativeBindings.matmulPlan(m, n, k, flags);
            if (code < 0) {
                NativeEngine.checkStatus(code, "MatmulOps.plan");
            }
            return new Linalg.Plan(Linalg.Kernel.of(code & 0xFF), (code & 0x100) != 0, (code & 0x200) != 0);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J native matmul plan failed", t);
        }
    }
}
