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

package hc4j.linalg;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.internal.ffi.TensorAccess;
import hc4j.internal.layout.Dims;
import hc4j.internal.layout.ValidUtil;
import hc4j.internal.ops.MatmulOps;
import java.util.Objects;

/**
 * Public linear algebra facade exposing matrix multiplication, transposition,
 * and kernel selection plans.
 */
public final class Linalg {

    /** Kernel identifiers selected by the native matmul dispatcher. */
    public enum Kernel {
        REGISTER_64_VEC4(1),
        REGISTER_64_SCALAR(2),
        REGISTER_128_VEC4(3),
        REGISTER_128_SCALAR(4),
        TILED_16(5),
        GEMV_ROWS(6),
        GEMV_ROWS_SUBGROUP(7),
        GEMV_COLS(8);

        private final int code;

        Kernel(int code) {
            this.code = code;
        }

        public int code() {
            return code;
        }

        public static Kernel of(int code) {
            for (Kernel k : values()) {
                if (k.code == code) {
                    return k;
                }
            }
            throw new IllegalStateException("Unknown matmul kernel code " + code);
        }
    }

    /**
     * The dispatcher's kernel choice and scratch transposition plan.
     *
     * @param kernel the kernel selected for the shape and device
     * @param pretransposeA whether A is copied into transposed scratch before the GEMM
     * @param pretransposeB whether B is copied into transposed scratch before the GEMM
     */
    public record Plan(Kernel kernel, boolean pretransposeA, boolean pretransposeB) {}

    private Linalg() {
        throw new AssertionError("No instances of Linalg");
    }

    /**
     * Matrix multiplication {@code a · b}.
     */
    public static Tensor matmul(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        Dims d = ValidUtil.matmulDims(access.getShapeUnsafe(a), access.getShapeUnsafe(b), false, false);
        ValidUtil.requireOperandF32("matmul", a.getDType());
        ValidUtil.requireOperandF32("matmul", b.getDType());
        Tensor res = Tensor.empty(DType.f32, d.resultShape());
        try {
            return MatmulOps.matmul(a, b, res, false, false);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Matrix multiplication into {@code out}.
     */
    public static Tensor matmul(Tensor a, Tensor b, Tensor out) {
        return MatmulOps.matmul(a, b, out, false, false);
    }

    /**
     * In-VRAM transposition of a 2-D tensor into a new Tensor.
     */
    public static Tensor transpose(Tensor a) {
        Objects.requireNonNull(a, "a");
        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        ValidUtil.requireOperandF32("transpose", a.getDType());
        int[] shape = access.getShapeUnsafe(a);
        ValidUtil.require2D("transpose", shape);
        Tensor res = Tensor.empty(a.getDType(), shape[1], shape[0]);
        try {
            return MatmulOps.transpose(a, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * In-VRAM transposition into {@code out}.
     */
    public static Tensor transpose(Tensor a, Tensor out) {
        return MatmulOps.transpose(a, out);
    }

    /**
     * Reports the kernel and pretransposition plan selected for {@code (m, n, k)}.
     */
    public static Plan matmulPlan(int m, int n, int k, boolean transposeA, boolean transposeB) {
        return MatmulOps.plan(m, n, k, transposeA, transposeB);
    }

    /**
     * Reports the kernel plan for {@code (m, n, k)} without transposition.
     */
    public static Plan matmulPlan(int m, int n, int k) {
        return matmulPlan(m, n, k, false, false);
    }

    /**
     * Reports the kernel plan for the matrix product {@code a · b}.
     */
    public static Plan matmulPlan(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        Dims d = ValidUtil.matmulDims(access.getShapeUnsafe(a), access.getShapeUnsafe(b), false, false);
        return matmulPlan(d.m(), d.n(), d.k(), false, false);
    }
}
