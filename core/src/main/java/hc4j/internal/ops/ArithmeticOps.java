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
 * Binary elementwise arithmetic operations.
 */
public final class ArithmeticOps {

    private ArithmeticOps() {
        throw new AssertionError("No instances of ArithmeticOps");
    }

    private static int resolveDTypeCode(DType type) {
        return switch (type) {
            case i32 -> 0;
            case f32 -> 1;
        };
    }

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

        int[] sa = access.getShapeUnsafe(a);
        int[] sb = access.getShapeUnsafe(b);
        int[] sr = access.getShapeUnsafe(res);

        ValidUtil.requireSameShape(sa, sb);
        ValidUtil.requireResultShape(sa, sr);

        int[] sta = access.getStridesUnsafe(a);
        int[] stb = access.getStridesUnsafe(b);
        int[] str = access.getStridesUnsafe(res);

        boolean contiguous = ValidUtil.isContiguous(sa, sta)
                && ValidUtil.isContiguous(sb, stb)
                && ValidUtil.isContiguous(sr, str);

        int rank = res.dim();
        int dtypeCode = resolveDTypeCode(a.getDType());

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shapeSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, sr);
            MemorySegment stridesASeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, sta);
            MemorySegment stridesBSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, stb);
            MemorySegment stridesCSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, str);

            int status = NativeBindings.dispatchBinary(
                    opIndex, access.getVramId(a), access.getVramId(b), access.getVramId(res),
                    rank, shapeSeg, stridesASeg, stridesBSeg, stridesCSeg,
                    res.getSize(), contiguous ? 1 : 0, dtypeCode);

            NativeEngine.checkStatus(status, "ArithmeticOps." + opName);
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J GPU Error: " + opName + " failed", t);
        }
    }

    public static Tensor add(Tensor a, Tensor b, Tensor res) { return execute(0, "Add", a, b, res); }
    public static Tensor sub(Tensor a, Tensor b, Tensor res) { return execute(1, "Subtract", a, b, res); }
    public static Tensor mul(Tensor a, Tensor b, Tensor res) { return execute(2, "Multiply", a, b, res); }
    public static Tensor div(Tensor a, Tensor b, Tensor res) { return execute(3, "Divide", a, b, res); }
}
