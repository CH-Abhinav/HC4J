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

package hc4j.internal.layout;

import hc4j.DType;
import java.util.Arrays;

/**
 * Validation utilities for tensor layouts, shapes, strides, and dtypes.
 */
public final class ValidUtil {

    private ValidUtil() {
        throw new AssertionError("No instances of ValidUtil");
    }

    public static void requireResultDType(DType expected, DType actual) {
        if (expected != actual) {
            throw new IllegalArgumentException("Result dtype must be " + expected + " but was " + actual);
        }
    }

    public static void requireOperandF32(String op, DType actual) {
        if (actual != DType.f32) {
            throw new IllegalArgumentException(op + " is only defined for f32 tensors, got " + actual);
        }
    }

    public static void requireSameDType(DType a, DType b) {
        if (a != b) {
            throw new IllegalArgumentException("DType mismatch: " + a + " vs " + b);
        }
    }

    public static void requireResultShape(int[] expected, int[] actual) {
        if (!Arrays.equals(expected, actual)) {
            throw new IllegalArgumentException("Result shape must be " + Arrays.toString(expected)
                    + " but was " + Arrays.toString(actual));
        }
    }

    public static void requireSameShape(int[] a, int[] b) {
        if (!Arrays.equals(a, b)) {
            throw new IllegalArgumentException("Shape mismatch: " + Arrays.toString(a) + " vs " + Arrays.toString(b));
        }
    }

    public static void requireSameShape(String op, int[] a, int[] b) {
        if (!Arrays.equals(a, b)) {
            throw new IllegalArgumentException(op + ": shape mismatch " + Arrays.toString(a) + " vs " + Arrays.toString(b));
        }
    }

    public static void require2D(String op, int[] shape) {
        if (shape.length != 2) {
            throw new IllegalArgumentException(op + " requires a 2-D tensor, got " + Arrays.toString(shape));
        }
    }

    public static void requireNoAlias(long vramA, long vramB, String message) {
        if (vramA == vramB) {
            throw new IllegalArgumentException(message);
        }
    }

    public static boolean isContiguous(int[] shape, int[] strides) {
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (shape[i] != 1 && strides[i] != expected) {
                return false;
            }
            expected *= shape[i];
        }
        return true;
    }

    public static Dims matmulDims(int[] sa, int[] sb, boolean transposeA, boolean transposeB) {
        if (sa.length < 1 || sa.length > 2 || sb.length < 1 || sb.length > 2) {
            throw new IllegalArgumentException("Matmul supports 1-D and 2-D operands, got "
                    + Arrays.toString(sa) + " and " + Arrays.toString(sb));
        }
        int m = sa.length == 1 ? 1 : (transposeA ? sa[1] : sa[0]);
        int ka = sa.length == 1 ? sa[0] : (transposeA ? sa[0] : sa[1]);
        int kb = sb.length == 1 ? sb[0] : (transposeB ? sb[1] : sb[0]);
        int n = sb.length == 1 ? 1 : (transposeB ? sb[0] : sb[1]);
        if (ka != kb) {
            throw new IllegalArgumentException("Matmul shape mismatch: left shape " + Arrays.toString(sa)
                    + " and right shape " + Arrays.toString(sb)
                    + " are incompatible because " + ka + " != " + kb);
        }
        int[] resShape;
        if (sa.length == 2 && sb.length == 2) {
            resShape = new int[] {m, n};
        } else if (sa.length == 2) {
            resShape = new int[] {m};
        } else if (sb.length == 2) {
            resShape = new int[] {n};
        } else {
            resShape = new int[] {1};
        }
        return new Dims(m, n, ka, resShape);
    }

    public static Rows rows(int[] shape, long totalSize) {
        if (shape.length == 0 || totalSize == 0) {
            throw new IllegalArgumentException("Row ops require a non-empty tensor, got shape " + Arrays.toString(shape));
        }
        long cols = shape[shape.length - 1];
        long rows = totalSize / cols;
        if (cols <= 0 || rows <= 0 || rows > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Tensor too large for row ops: rows=" + rows + ", cols=" + cols);
        }
        return new Rows((int) rows, (int) cols);
    }
}
