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

package hc4j.internal.linalg;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.internal.ffi.NativeBindings;
import hc4j.internal.ffi.NativeEngine;
import hc4j.internal.ffi.TensorAccess;
import hc4j.internal.layout.ValidUtil;
import java.util.Arrays;
import java.util.Objects;

/**
 * Dispatcher for mixed-precision {@code f32 x f16} linear layers and transposed-weight
 * {@code A[M, K] @ B[N, K]^T} matrix multiplications.
 */
public final class F16MatmulOps {

    private F16MatmulOps() {
        throw new AssertionError("No instances of F16MatmulOps");
    }

    static Tensor matmulF32F16(long idA, long idB, long idRes, int m, int n, int k, int flags, Tensor res) throws Throwable {
        int status = NativeBindings.dispatchMatmulF32F16(idA, idB, idRes, m, n, k, flags);
        NativeEngine.checkStatus(status, "F16MatmulOps.matmulF32F16");
        return res;
    }

    /**
     * Computes the output shape for {@code a @ b^T} where {@code b} is 2-D {@code [N, K]}.
     *
     * @param a left operand ({@code [K]} or {@code [..., M, K]})
     * @param b 2-D right weight matrix {@code [N, K]}
     * @return expected output shape
     */
    public static int[] transposedBOutputShape(Tensor a, Tensor b) {
        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        int[] sa = access.getShapeUnsafe(a);
        int[] sb = access.getShapeUnsafe(b);
        if (sb.length != 2) {
            throw new IllegalArgumentException("transposedB requires a 2-D right operand [N, K], got " + Arrays.toString(sb));
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

    /**
     * Executes {@code res = a @ b^T} for {@code f32} or {@code f16} weights {@code b[N, K]}.
     *
     * @param a left {@code f32} activation tensor
     * @param b right {@code f32} or {@code f16} weight matrix of shape {@code [N, K]}
     * @param res destination {@code f32} tensor
     * @return {@code res}
     */
    public static Tensor matmulTransposedB(Tensor a, Tensor b, Tensor res) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("matmulTransposedB input", a.getDType());
        if (b.getDType() != DType.f32 && b.getDType() != DType.f16) {
            throw new IllegalArgumentException("matmulTransposedB right operand must be f32 or f16, got " + b.getDType());
        }
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeA = ValidUtil.prepareContiguousZeroOffset(a);
             ValidUtil.SafeOperand safeB = ValidUtil.prepareContiguousZeroOffset(b)) {

            Tensor ea = safeA.tensor();
            Tensor eb = safeB.tensor();

            int[] sa = access.getShapeUnsafe(ea);
            int[] sb = access.getShapeUnsafe(eb);
            if (sb.length != 2) {
                throw new IllegalArgumentException("matmulTransposedB requires a 2-D right operand [N, K], got " + Arrays.toString(sb));
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

            int[] expectedOutShape = transposedBOutputShape(ea, eb);
            ValidUtil.requireResultShape(expectedOutShape, access.getShapeUnsafe(res));

            long idA = access.getVramId(ea);
            long idB = access.getVramId(eb);
            long idRes = access.getVramId(res);
            ValidUtil.requireNoAlias(idRes, idA, "matmul result must not alias an operand");
            ValidUtil.requireNoAlias(idRes, idB, "matmul result must not alias an operand");

            if (eb.getDType() == DType.f16) {
                return matmulF32F16(idA, idB, idRes, m, n, k, MatmulOps.FLAG_TRANS_B, res);
            }
            int status = NativeBindings.dispatchMatmulEx(idA, idB, idRes, m, n, k, MatmulOps.FLAG_TRANS_B);
            NativeEngine.checkStatus(status, "F16MatmulOps.matmulTransposedB");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native matmulTransposedB failed", t);
        }
    }
}
