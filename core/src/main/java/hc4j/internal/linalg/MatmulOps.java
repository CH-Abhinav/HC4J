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
import hc4j.internal.layout.Dims;
import hc4j.internal.layout.ValidUtil;
import hc4j.linalg.Linalg;
import java.util.Objects;

/**
 * Dispatcher for 1-D/2-D matrix multiplication, 2-D matrix transposition, and kernel selection planning.
 */
public final class MatmulOps {

    static final int FLAG_TRANS_A = 1;
    static final int FLAG_TRANS_B = 2;

    private MatmulOps() {
        throw new AssertionError("No instances of MatmulOps");
    }

    /**
     * Resolves matrix multiplication output dimensions for operands with shapes {@code sa} and {@code sb}.
     *
     * @param sa left operand shape
     * @param sb right operand shape
     * @param transposeA whether {@code A} is transposed
     * @param transposeB whether {@code B} is transposed
     * @return validated {@link Dims} descriptor
     */
    public static Dims dims(int[] sa, int[] sb, boolean transposeA, boolean transposeB) {
        return ValidUtil.matmulDims(sa, sb, transposeA, transposeB);
    }

    /**
     * Dispatches 1-D/2-D matrix multiplication {@code res = op(a) * op(b)} (supports {@code f32} and {@code f32 x f16}).
     *
     * @param a left operand ({@code f32})
     * @param b right operand ({@code f32} or {@code f16})
     * @param res destination {@code f32} tensor
     * @param transposeA transpose flag for {@code a}
     * @param transposeB transpose flag for {@code b}
     * @return {@code res}
     */
    public static Tensor matmul(Tensor a, Tensor b, Tensor res, boolean transposeA, boolean transposeB) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("matmul", a.getDType());
        if (b.getDType() != DType.f32 && b.getDType() != DType.f16) {
            throw new IllegalArgumentException("Right operand must be f32 or f16, got " + b.getDType());
        }
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeA = ValidUtil.prepareContiguousZeroOffset(a);
             ValidUtil.SafeOperand safeB = ValidUtil.prepareContiguousZeroOffset(b)) {

            Tensor ea = safeA.tensor();
            Tensor eb = safeB.tensor();

            int[] sa = access.getShapeUnsafe(ea);
            int[] sb = access.getShapeUnsafe(eb);
            Dims d = ValidUtil.matmulDims(sa, sb, transposeA, transposeB);
            ValidUtil.requireResultShape(d.resultShape(), access.getShapeUnsafe(res));

            long idA = access.getVramId(ea);
            long idB = access.getVramId(eb);
            long idRes = access.getVramId(res);
            ValidUtil.requireNoAlias(idRes, idA, "matmul result must not alias an operand");
            ValidUtil.requireNoAlias(idRes, idB, "matmul result must not alias an operand");

            if (eb.getDType() == DType.f16) {
                if (transposeA) {
                    throw new UnsupportedOperationException("transposeA is not supported for f16 operands");
                }
                int flags = transposeB ? FLAG_TRANS_B : 0;
                return F16MatmulOps.matmulF32F16(idA, idB, idRes, d.m(), d.n(), d.k(), flags, res);
            }

            int flags = (transposeA && ea.dim() == 2 ? FLAG_TRANS_A : 0)
                    | (transposeB && eb.dim() == 2 ? FLAG_TRANS_B : 0);
            int status = flags == 0
                    ? NativeBindings.dispatchMatmul(idA, idB, idRes, d.m(), d.n(), d.k())
                    : NativeBindings.dispatchMatmulEx(idA, idB, idRes, d.m(), d.n(), d.k(), flags);
            NativeEngine.checkStatus(status, "MatmulOps.matmul");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native matmul failed", t);
        }
    }

    /**
     * Transposes a 2-D {@code f32} matrix in VRAM into pre-allocated {@code res}.
     *
     * @param in source 2-D {@code f32} tensor
     * @param res destination 2-D {@code f32} tensor
     * @return {@code res}
     */
    public static Tensor transpose(Tensor in, Tensor res) {
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(in);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("transpose", in.getDType());
        ValidUtil.requireResultDType(in.getDType(), res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeIn = ValidUtil.prepareContiguousZeroOffset(in)) {
            Tensor ein = safeIn.tensor();
            int[] s = access.getShapeUnsafe(ein);
            ValidUtil.require2D("transpose", s);
            ValidUtil.requireResultShape(new int[] {s[1], s[0]}, access.getShapeUnsafe(res));

            long idIn = access.getVramId(ein);
            long idRes = access.getVramId(res);
            ValidUtil.requireNoAlias(idRes, idIn, "transpose cannot run in place");

            int status = NativeBindings.dispatchTranspose(idIn, idRes, s[0], s[1]);
            NativeEngine.checkStatus(status, "MatmulOps.transpose");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native transpose failed", t);
        }
    }

    /**
     * Queries the native GEMM/GEMV kernel plan for dimensions {@code (m, n, k)}.
     *
     * @param m row count
     * @param n column count
     * @param k inner dimension
     * @param transposeA whether {@code A} is transposed
     * @param transposeB whether {@code B} is transposed
     * @return selected {@link Linalg.Plan}
     */
    public static Linalg.Plan plan(int m, int n, int k, boolean transposeA, boolean transposeB) {
        int flags = (transposeA ? FLAG_TRANS_A : 0) | (transposeB ? FLAG_TRANS_B : 0);
        try {
            int code = NativeBindings.matmulPlan(m, n, k, flags);
            if (code < 0) {
                NativeEngine.checkStatus(code, "MatmulOps.plan");
            }
            return new Linalg.Plan(Linalg.Kernel.of(code & 0xFF), (code & 0x100) != 0, (code & 0x200) != 0);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native matmul plan failed", t);
        }
    }
}
