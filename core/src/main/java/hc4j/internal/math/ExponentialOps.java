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
 * Exponential, logarithmic, and square-root unary operations.
 */
public final class ExponentialOps {

    /** Supported exponential/logarithmic unary functions mapped to native dispatch indices. */
    public enum Function {
        EXP("exp", 0),
        LOG("log", 1),
        LOG2("log2", 2),
        LOG10("log10", 3),
        SQRT("sqrt", 4);

        private final String opName;
        private final int opIndex;

        Function(String opName, int opIndex) {
            this.opName = opName;
            this.opIndex = opIndex;
        }

        /** Returns the lowercase operation name. */
        public String opName() { return opName; }

        /** Returns the native dispatch index. */
        public int opIndex() { return opIndex; }
    }

    private ExponentialOps() {
        throw new AssertionError("No instances of ExponentialOps");
    }

    /**
     * Applies exponential/logarithmic function {@code fn} elementwise on {@code input} into {@code out}.
     *
     * @param fn exponential or logarithmic function
     * @param input source f32 tensor
     * @param out destination f32 tensor
     * @return {@code out}
     */
    public static Tensor apply(Function fn, Tensor input, Tensor out) {
        Objects.requireNonNull(fn, "fn");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(out, "out");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(input);
        access.ensureOpen(out);

        ValidUtil.requireOperandF32(fn.opName(), input.getDType());
        ValidUtil.requireResultDType(DType.f32, out.getDType());

        return ValidUtil.dispatchStridedUnary("ExponentialOps." + fn.opName(), fn.opName(), input, out,
                (idIn, idOut, rank, shape, stIn, stOut, len, contig) ->
                        NativeBindings.dispatchExponential(fn.opIndex(), idIn, idOut, rank, shape, stIn, stOut, len, contig));
    }
}
