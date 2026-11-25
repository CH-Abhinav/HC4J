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

import hc4j.Tensor;
import hc4j.internal.ffi.NativeBindings;
import hc4j.internal.ffi.TensorAccess;
import hc4j.internal.layout.ValidUtil;
import java.util.Objects;

/**
 * Binary elementwise arithmetic operations with full NumPy broadcasting.
 */
public final class ArithmeticOps {

    private ArithmeticOps() {
        throw new AssertionError("No instances of ArithmeticOps");
    }

    /**
     * Dispatches a broadcasted binary arithmetic kernel over {@code a} and {@code b}.
     *
     * @param opIndex operation selector (0=add, 1=sub, 2=mul, 3=div)
     * @param opName operation name for error reporting
     * @param a left operand
     * @param b right operand
     * @param res destination tensor
     * @return {@code res}
     */
    public static Tensor execute(int opIndex, String opName, Tensor a, Tensor b, Tensor res) {
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

        return ValidUtil.dispatchStridedBinary("ArithmeticOps." + opName, a, b, res,
                (idA, idB, idOut, rank, shape, stA, stB, stC, len, contig) ->
                        NativeBindings.dispatchBinary(opIndex, idA, idB, idOut, rank, shape, stA, stB, stC, len, contig, dtypeCode));
    }

    /** Computes elementwise addition {@code res = a + b}. */
    public static Tensor add(Tensor a, Tensor b, Tensor res) { return execute(0, "Add", a, b, res); }

    /** Computes elementwise subtraction {@code res = a - b}. */
    public static Tensor sub(Tensor a, Tensor b, Tensor res) { return execute(1, "Subtract", a, b, res); }

    /** Computes elementwise multiplication {@code res = a * b}. */
    public static Tensor mul(Tensor a, Tensor b, Tensor res) { return execute(2, "Multiply", a, b, res); }

    /** Computes elementwise division {@code res = a / b}. */
    public static Tensor div(Tensor a, Tensor b, Tensor res) { return execute(3, "Divide", a, b, res); }
}
