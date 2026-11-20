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

        ValidUtil.requireOperandF32("matmul", a.getDType());
        ValidUtil.requireOperandF32("matmul", b.getDType());
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

    public static Tensor transpose(Tensor in, Tensor res) {
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(in);
        access.ensureOpen(res);

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
