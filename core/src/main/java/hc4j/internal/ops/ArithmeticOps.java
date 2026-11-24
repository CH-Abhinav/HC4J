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
import java.util.Arrays;
import java.util.Objects;

/**
 * Binary elementwise arithmetic operations with full NumPy broadcasting.
 */
public final class ArithmeticOps {

    private ArithmeticOps() {
        throw new AssertionError("No instances of ArithmeticOps");
    }

    private static int resolveDTypeCode(DType type) {
        return switch (type) {
            case i32 -> 0;
            case f32 -> 1;
            default -> throw new UnsupportedOperationException("Operation not supported for dtype " + type);
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

        // Auto-materialize independent copies if aliased with mismatched layout
        if (access.getVramId(a) == access.getVramId(res) && !access.hasIdenticalLayout(a, res)) {
            try (Tensor copyA = a.copy()) {
                return execute(opIndex, opName, copyA, b, res);
            }
        }
        if (access.getVramId(b) == access.getVramId(res) && !access.hasIdenticalLayout(b, res)) {
            try (Tensor copyB = b.copy()) {
                return execute(opIndex, opName, a, copyB, res);
            }
        }

        // Materialize non-zero offsets
        if (access.elementOffset(a) > 0) {
            try (Tensor contigA = a.contiguous()) {
                return execute(opIndex, opName, contigA, b, res);
            }
        }
        if (access.elementOffset(b) > 0) {
            try (Tensor contigB = b.contiguous()) {
                return execute(opIndex, opName, a, contigB, res);
            }
        }

        // Validate output tensor: must be contiguous with zero offset
        if (!res.isContiguous() || access.elementOffset(res) != 0) {
            throw new IllegalArgumentException("Output tensor must be contiguous with zero offset");
        }

        int[] sa = access.getShapeUnsafe(a);
        int[] sb = access.getShapeUnsafe(b);
        int[] targetShape = ValidUtil.broadcastShape(sa, sb);

        int[] sr = access.getShapeUnsafe(res);
        ValidUtil.requireResultShape(targetShape, sr);

        int[] sta = access.getStridesUnsafe(a);
        int[] stb = access.getStridesUnsafe(b);
        int[] str = access.getStridesUnsafe(res);

        int[] bcastStridesA = ValidUtil.broadcastStrides(sa, sta, targetShape);
        int[] bcastStridesB = ValidUtil.broadcastStrides(sb, stb, targetShape);

        boolean contiguous = Arrays.equals(sa, targetShape)
                && Arrays.equals(sb, targetShape)
                && a.isContiguous()
                && b.isContiguous()
                && res.isContiguous();

        int rank = targetShape.length;
        int dtypeCode = resolveDTypeCode(a.getDType());

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shapeSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, targetShape);
            MemorySegment stridesASeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesA);
            MemorySegment stridesBSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, bcastStridesB);
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
