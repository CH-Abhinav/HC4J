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
import java.util.Objects;

/**
 * Dispatcher for GPU elementwise type conversions ({@code f32 <-> i32}, {@code f32 <-> f16}, and {@code i32 <-> f16}).
 */
public final class CastOps {

    private CastOps() {
        throw new AssertionError("No instances of CastOps");
    }

    /**
     * Converts {@code src} to {@code targetDType}, writing elements into {@code res}.
     *
     * @param src source tensor
     * @param targetDType desired destination data type
     * @param res destination tensor matching {@code targetDType} and shape of {@code src}
     * @return {@code res}
     */
    public static Tensor cast(Tensor src, DType targetDType, Tensor res) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(targetDType, "targetDType");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(src);
        access.ensureOpen(res);

        ValidUtil.requireResultDType(targetDType, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeSrc = ValidUtil.prepareContiguousZeroOffset(src)) {
            Tensor eff = safeSrc.tensor();
            ValidUtil.requireResultShape(access.getShapeUnsafe(eff), access.getShapeUnsafe(res));

            DType srcType = eff.getDType();
            if (srcType == targetDType) {
                NativeEngine.copyStrided(
                        access.getVramId(eff), access.getVramId(res),
                        access.elementOffset(eff), eff.dim(),
                        access.getShapeUnsafe(eff), access.getStridesUnsafe(eff),
                        eff.getSize());
                return res;
            }

            if (srcType == DType.f32 && targetDType == DType.f16) {
                int status = NativeBindings.dispatchCastF32ToF16(
                        access.getVramId(eff), access.getVramId(res), eff.getSize());
                NativeEngine.checkStatus(status, "CastOps.cast(f32->f16)");
                return res;
            }
            if (srcType == DType.f16 && targetDType == DType.f32) {
                int status = NativeBindings.dispatchCastF16ToF32(
                        access.getVramId(eff), access.getVramId(res), eff.getSize());
                NativeEngine.checkStatus(status, "CastOps.cast(f16->f32)");
                return res;
            }
            if ((srcType == DType.i32 && targetDType == DType.f16)
                    || (srcType == DType.f16 && targetDType == DType.i32)) {
                try (Tensor f32Temp = eff.cast(DType.f32)) {
                    return cast(f32Temp, targetDType, res);
                }
            }

            int srcDtype = ValidUtil.resolveMathDTypeCode(srcType);
            int dstDtype = ValidUtil.resolveMathDTypeCode(targetDType);
            int status = NativeBindings.dispatchCast(
                    access.getVramId(eff), access.getVramId(res), eff.getSize(), srcDtype, dstDtype);
            NativeEngine.checkStatus(status, "CastOps.cast");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native cast failed", t);
        }
    }
}
