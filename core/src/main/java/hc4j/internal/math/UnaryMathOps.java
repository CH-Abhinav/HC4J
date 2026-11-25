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
import hc4j.internal.ffi.TensorAccess;
import hc4j.internal.layout.ValidUtil;
import java.util.Objects;

/**
 * Dispatcher for f32 and i32 elementwise unary math and clamping operations.
 */
public final class UnaryMathOps {

    private UnaryMathOps() {
        throw new AssertionError("No instances of UnaryMathOps");
    }

    /**
     * Applies an f32 unary math operation (`abs`, `neg`, `rsqrt`, `floor`, `ceil`, `round`, `sign`).
     *
     * @param opIndex native unary f32 opcode
     * @param opName operation name
     * @param input source f32 tensor
     * @param out destination f32 tensor
     * @return {@code out}
     */
    public static Tensor applyUnaryF32(int opIndex, String opName, Tensor input, Tensor out) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(out, "out");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(input);
        access.ensureOpen(out);

        ValidUtil.requireOperandF32(opName, input.getDType());
        ValidUtil.requireResultDType(DType.f32, out.getDType());

        return ValidUtil.dispatchStridedUnary("UnaryMathOps." + opName, opName, input, out,
                (idIn, idOut, rank, shape, stIn, stOut, len, contig) ->
                        NativeBindings.dispatchMathUnary(opIndex, idIn, idOut, rank, shape, stIn, stOut, len, contig));
    }

    /**
     * Clamps an f32 tensor elementwise into {@code [lo, hi]}.
     *
     * @param input source f32 tensor
     * @param lo inclusive lower bound
     * @param hi inclusive upper bound
     * @param out destination f32 tensor
     * @return {@code out}
     */
    public static Tensor clampF32(Tensor input, float lo, float hi, Tensor out) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(out, "out");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(input);
        access.ensureOpen(out);

        ValidUtil.requireOperandF32("clamp", input.getDType());
        ValidUtil.requireResultDType(DType.f32, out.getDType());

        return ValidUtil.dispatchStridedUnary("UnaryMathOps.clampF32", "clamp", input, out,
                (idIn, idOut, rank, shape, stIn, stOut, len, contig) ->
                        NativeBindings.dispatchClampF32(idIn, idOut, rank, shape, stIn, stOut, len, contig, lo, hi));
    }

    /**
     * Applies an i32 unary math operation (`abs`, `neg`, `sign`, `clamp`).
     *
     * @param opCode native unary i32 opcode
     * @param opName operation name
     * @param input source i32 tensor
     * @param out destination i32 tensor
     * @param pZ first integer parameter (e.g. clamp lower bound)
     * @param pW second integer parameter (e.g. clamp upper bound)
     * @return {@code out}
     */
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

        return ValidUtil.dispatchStridedUnary("UnaryMathOps." + opName, opName, input, out,
                (idIn, idOut, rank, shape, stIn, stOut, len, contig) ->
                        NativeBindings.dispatchUnaryI32(idIn, idOut, rank, shape, stIn, stOut, len, contig, opCode, pZ, pW));
    }
}
