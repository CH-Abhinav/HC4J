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
import hc4j.internal.linalg.F16MatmulOps;
import hc4j.internal.linalg.MatmulOps;
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
     * Matrix multiplication {@code a · b}, supporting 1-D, 2-D, and batched 3-D/4-D+ tensors with broadcasting.
     *
     * @param a left operand
     * @param b right operand
     * @return result tensor
     */
    public static Tensor matmul(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        return a.matmul(b);
    }

    /**
     * Matrix multiplication into {@code out}, supporting 1-D, 2-D, and batched 3-D/4-D+ tensors with broadcasting.
     *
     * @param a left operand
     * @param b right operand
     * @param out destination tensor
     * @return {@code out}
     */
    public static Tensor matmul(Tensor a, Tensor b, Tensor out) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(out, "out");
        return a.matmul(b, out);
    }

    /**
     * Matrix multiplication with transposed second operand {@code B^T},
     * supporting 2-D {@code [M, K] @ [N, K]^T -> [M, N]} and
     * 3-D+ {@code [..., M, K] @ [N, K]^T -> [..., M, N]}.
     * Supports both f32 and f16 right operands.
     *
     * @param a left operand
     * @param b right operand of shape [N, K]
     * @return result tensor of shape [..., M, N]
     */
    public static Tensor matmulTransposedB(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        int[] outShape = F16MatmulOps.transposedBOutputShape(a, b);
        Tensor res = Tensor.empty(DType.f32, outShape);
        try {
            return F16MatmulOps.matmulTransposedB(a, b, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Matrix multiplication with transposed second operand {@code B^T} into caller-provided {@code out}.
     *
     * @param a left operand
     * @param b right operand of shape [N, K]
     * @param out destination tensor
     * @return {@code out}
     */
    public static Tensor matmulTransposedB(Tensor a, Tensor b, Tensor out) {
        return F16MatmulOps.matmulTransposedB(a, b, out);
    }

    /**
     * Dot product between two tensors.
     *
     * @param a first operand
     * @param b second operand
     * @return scalar dot product as a 1-D tensor of shape {@code [1]}
     */
    public static Tensor dot(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        return a.dot(b);
    }

    /**
     * Computes the Frobenius (L2) Euclidean norm of a tensor.
     *
     * @param a operand tensor
     * @return norm scalar as a 1-D tensor of shape {@code [1]}
     */
    public static Tensor norm(Tensor a) {
        Objects.requireNonNull(a, "a");
        return a.norm();
    }

    /**
     * Extracts the main diagonal of a 2-D matrix.
     *
     * @param a 2-D matrix tensor
     * @return 1-D tensor view containing diagonal elements
     */
    public static Tensor diagonal(Tensor a) {
        Objects.requireNonNull(a, "a");
        return a.diagonal();
    }

    /**
     * Computes the trace (sum of diagonal elements) of a 2-D matrix.
     *
     * @param a 2-D matrix tensor
     * @return scalar trace as a 1-D tensor of shape {@code [1]}
     */
    public static Tensor trace(Tensor a) {
        Objects.requireNonNull(a, "a");
        return a.trace();
    }

    /**
     * Computes the cosine similarity between two tensors.
     *
     * @param a first operand
     * @param b second operand
     * @return scalar similarity as a 1-D tensor of shape {@code [1]}
     */
    public static Tensor cosineSimilarity(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        return a.cosineSimilarity(b);
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
