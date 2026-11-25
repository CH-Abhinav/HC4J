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
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Dispatcher for rank 3-D..6-D broadcasted batched matrix multiplication.
 */
public final class BatchedMatmulOps {

    private BatchedMatmulOps() {
        throw new AssertionError("No instances of BatchedMatmulOps");
    }

    private static ValidUtil.SafeOperand prepareInnerMatmulOperand(Tensor t, TensorAccess access) {
        if (access.elementOffset(t) > 0
                || !ValidUtil.hasValidInnerMatmulLayout(access.getShapeUnsafe(t), access.getStridesUnsafe(t))) {
            return new ValidUtil.SafeOperand(t.contiguous(), true);
        }
        return new ValidUtil.SafeOperand(t, false);
    }

    /**
     * Executes broadcasted batched matrix multiplication {@code res = a @ b}.
     *
     * @param a left batched {@code f32} tensor
     * @param b right batched {@code f32} tensor
     * @param res destination {@code f32} tensor
     * @param transposeA reserved transpose flag
     * @param transposeB reserved transpose flag
     * @return {@code res}
     */
    public static Tensor batchedMatmul(Tensor a, Tensor b, Tensor res, boolean transposeA, boolean transposeB) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(b);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("batched matmul", a.getDType());
        ValidUtil.requireOperandF32("batched matmul", b.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeA = prepareInnerMatmulOperand(a, access);
             ValidUtil.SafeOperand safeB = prepareInnerMatmulOperand(b, access)) {

            Tensor ea = safeA.tensor();
            Tensor eb = safeB.tensor();

            ValidUtil.BatchedMatmulPlan plan = ValidUtil.batchedMatmulPlan(
                    access.getShapeUnsafe(ea), access.getStridesUnsafe(ea),
                    access.getShapeUnsafe(eb), access.getStridesUnsafe(eb));
            ValidUtil.requireResultShape(plan.outputShape(), access.getShapeUnsafe(res));

            long idA = access.getVramId(ea);
            long idB = access.getVramId(eb);
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
                        plan.flags());
                NativeEngine.checkStatus(status, "BatchedMatmulOps.batchedMatmul");
                return res;
            }
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native batched matmul failed", t);
        }
    }
}
