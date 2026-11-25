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

package hc4j.internal.math;

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
 * Dispatcher for binary {@code pow}, {@code maximum}/{@code minimum}, comparisons, and 3-way {@code where} masking.
 */
public final class CompareMaskOps {

    private CompareMaskOps() {
        throw new AssertionError("No instances of CompareMaskOps");
    }

    /**
     * Raises {@code a} to {@code b} elementwise with NumPy broadcasting ({@code f32}).
     *
     * @param a base tensor
     * @param b exponent tensor
     * @param res destination f32 tensor
     * @return {@code res}
     */
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

        return ValidUtil.dispatchStridedBinary("CompareMaskOps.pow", a, b, res, NativeBindings::dispatchPowF32);
    }

    /**
     * Computes elementwise maximum or minimum of {@code a} and {@code b} with broadcasting.
     *
     * @param isMax true for maximum, false for minimum
     * @param a first operand
     * @param b second operand
     * @param res destination tensor
     * @return {@code res}
     */
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
        int dtypeCode = ValidUtil.resolveMathDTypeCode(a.getDType());

        return ValidUtil.dispatchStridedBinary(
                isMax ? "CompareMaskOps.maximum" : "CompareMaskOps.minimum", a, b, res,
                (idA, idB, idOut, rank, shape, stA, stB, stC, len, contig) -> isMax
                        ? NativeBindings.dispatchMaximum(idA, idB, idOut, rank, shape, stA, stB, stC, len, contig, dtypeCode)
                        : NativeBindings.dispatchMinimum(idA, idB, idOut, rank, shape, stA, stB, stC, len, contig, dtypeCode));
    }

    /**
     * Evaluates an elementwise comparison predicate writing a {@code DType.i32} 0/1 mask.
     *
     * @param cmpOp native comparison opcode ({@code eq=0, ne=1, gt=2, ge=3, lt=4, le=5})
     * @param opName operation name
     * @param a left operand
     * @param b right operand
     * @param res destination i32 mask tensor
     * @return {@code res}
     */
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
        int dtypeCode = ValidUtil.resolveMathDTypeCode(a.getDType());

        return ValidUtil.dispatchStridedBinary("CompareMaskOps." + opName, a, b, res,
                (idA, idB, idOut, rank, shape, stA, stB, stC, len, contig) ->
                        NativeBindings.dispatchCompare(idA, idB, idOut, rank, shape, stA, stB, stC, len, contig, dtypeCode, cmpOp));
    }

    /**
     * Selects elements from {@code x} or {@code y} according to predicate {@code cond} with 3-way broadcasting.
     *
     * @param cond condition tensor (non-zero selects {@code x})
     * @param x true-branch tensor
     * @param y false-branch tensor
     * @param res destination tensor
     * @return {@code res}
     */
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
        int condDtypeCode = ValidUtil.resolveMathDTypeCode(cond.getDType());
        int valDtypeCode = ValidUtil.resolveMathDTypeCode(x.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeCond = ValidUtil.prepareAliasAndOffset(cond, res);
             ValidUtil.SafeOperand safeX = ValidUtil.prepareAliasAndOffset(x, res);
             ValidUtil.SafeOperand safeY = ValidUtil.prepareAliasAndOffset(y, res)) {

            Tensor ec = safeCond.tensor();
            Tensor ex = safeX.tensor();
            Tensor ey = safeY.tensor();

            int[] sc = access.getShapeUnsafe(ec);
            int[] sx = access.getShapeUnsafe(ex);
            int[] sy = access.getShapeUnsafe(ey);
            int[] targetShape = ValidUtil.broadcastShape(ValidUtil.broadcastShape(sc, sx), sy);
            ValidUtil.requireResultShape(targetShape, access.getShapeUnsafe(res));

            int[] bcastStridesCond = ValidUtil.broadcastStrides(sc, access.getStridesUnsafe(ec), targetShape);
            int[] bcastStridesX = ValidUtil.broadcastStrides(sx, access.getStridesUnsafe(ex), targetShape);
            int[] bcastStridesY = ValidUtil.broadcastStrides(sy, access.getStridesUnsafe(ey), targetShape);
            int[] str = access.getStridesUnsafe(res);

            boolean contiguous = Arrays.equals(sc, targetShape)
                    && Arrays.equals(sx, targetShape)
                    && Arrays.equals(sy, targetShape)
                    && ec.isContiguous()
                    && ex.isContiguous()
                    && ey.isContiguous()
                    && res.isContiguous();
            int rank = targetShape.length;

            int status;
            if (contiguous) {
                status = NativeBindings.dispatchWhere(
                        access.getVramId(ec), access.getVramId(ex), access.getVramId(ey), access.getVramId(res),
                        rank, MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL,
                        MemorySegment.NULL, res.getSize(), 1, condDtypeCode, valDtypeCode);
            } else {
                try (Arena arena = Arena.ofConfined()) {
                    status = NativeBindings.dispatchWhere(
                            access.getVramId(ec), access.getVramId(ex), access.getVramId(ey), access.getVramId(res),
                            rank,
                            arena.allocateFrom(ValueLayout.JAVA_INT, targetShape),
                            arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesCond),
                            arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesX),
                            arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesY),
                            arena.allocateFrom(ValueLayout.JAVA_INT, str),
                            res.getSize(), 0, condDtypeCode, valDtypeCode);
                }
            }
            NativeEngine.checkStatus(status, "CompareMaskOps.where");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native where failed", t);
        }
    }
}
