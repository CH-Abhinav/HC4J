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
import hc4j.internal.layout.ValidUtil;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Exponential and logarithmic unary operations.
 */
public final class ExponentialOps {

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

        public String opName() {
            return opName;
        }

        public int opIndex() {
            return opIndex;
        }
    }

    private ExponentialOps() {
        throw new AssertionError("No instances of ExponentialOps");
    }

    public static Tensor apply(Function fn, Tensor input, Tensor out) {
        Objects.requireNonNull(fn, "fn");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(out, "out");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(input);
        access.ensureOpen(out);

        ValidUtil.requireOperandF32(fn.opName(), input.getDType());
        ValidUtil.requireResultDType(DType.f32, out.getDType());

        int[] sa = access.getShapeUnsafe(input);
        int[] sr = access.getShapeUnsafe(out);
        ValidUtil.requireSameShape(fn.opName(), sa, sr);

        int[] sta = access.getStridesUnsafe(input);
        int[] str = access.getStridesUnsafe(out);

        boolean contiguous = ValidUtil.isContiguous(sa, sta) && ValidUtil.isContiguous(sr, str);
        int rank = out.dim();
        long length = out.getSize();

        try {
            int status;
            if (contiguous) {
                status = NativeBindings.dispatchExponential(
                        fn.opIndex(), access.getVramId(input), access.getVramId(out),
                        rank, MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL, length, 1);
            } else {
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment shapeSeg = arena.allocateFrom(ValueLayout.JAVA_INT, sr);
                    MemorySegment stridesASeg = arena.allocateFrom(ValueLayout.JAVA_INT, sta);
                    MemorySegment stridesCSeg = arena.allocateFrom(ValueLayout.JAVA_INT, str);
                    status = NativeBindings.dispatchExponential(
                            fn.opIndex(), access.getVramId(input), access.getVramId(out),
                            rank, shapeSeg, stridesASeg, stridesCSeg, length, 0);
                }
            }
            NativeEngine.checkStatus(status, "ExponentialOps." + fn.opName());
            return out;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J GPU Error: " + fn.opName() + " failed", t);
        }
    }
}
