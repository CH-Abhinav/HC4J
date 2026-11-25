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

package hc4j.internal.nn;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.internal.ffi.NativeBindings;
import hc4j.internal.ffi.TensorAccess;
import hc4j.internal.layout.ValidUtil;
import hc4j.nn.Activation;
import java.util.Objects;

/**
 * Dispatcher for elementwise neural network activation functions.
 */
public final class ActivationOps {

    private ActivationOps() {
        throw new AssertionError("No instances of ActivationOps");
    }

    /**
     * Applies activation function {@code act} elementwise on {@code a} into {@code res}.
     *
     * @param act activation variant
     * @param a source {@code f32} tensor
     * @param res destination {@code f32} tensor
     * @param alpha scalar parameter used by {@link Activation#LEAKY_RELU} and {@link Activation#ELU}
     * @return {@code res}
     */
    public static Tensor activation(Activation act, Tensor a, Tensor res, float alpha) {
        Objects.requireNonNull(act, "act");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32(act.name(), a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        int alphaBits = Float.floatToRawIntBits(alpha);

        return ValidUtil.dispatchStridedUnary("ActivationOps." + act, act.name(), a, res,
                (idIn, idOut, rank, shape, stIn, stOut, len, contig) ->
                        NativeBindings.dispatchActivation(idIn, idOut, rank, shape, stIn, stOut, len, contig, act.ordinal(), alphaBits));
    }
}
