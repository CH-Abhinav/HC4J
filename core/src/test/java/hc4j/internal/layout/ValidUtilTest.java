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

import static hc4j.testutil.ExceptionTable.assertThrowsWithMessage;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import hc4j.DType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("ValidUtil Headless Validation Tests")
class ValidUtilTest {

    @Test
    @DisplayName("Result dtype validation error message")
    void testRequireResultDType() {
        ValidUtil.requireResultDType(DType.f32, DType.f32);
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.requireResultDType(DType.f32, DType.i32),
                "Result dtype must be f32 but was i32");
    }

    @Test
    @DisplayName("Operand f32 requirement error message")
    void testRequireOperandF32() {
        ValidUtil.requireOperandF32("sin", DType.f32);
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.requireOperandF32("sin", DType.i32),
                "sin is only defined for f32 tensors, got i32");
    }

    @Test
    @DisplayName("Same dtype requirement error message")
    void testRequireSameDType() {
        ValidUtil.requireSameDType(DType.f32, DType.f32);
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.requireSameDType(DType.f32, DType.i32),
                "DType mismatch: f32 vs i32");
    }

    @Test
    @DisplayName("Result shape validation error message")
    void testRequireResultShape() {
        ValidUtil.requireResultShape(new int[] {2, 3}, new int[] {2, 3});
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.requireResultShape(new int[] {2, 3}, new int[] {2, 4}),
                "Result shape must be [2, 3] but was [2, 4]");
    }

    @Test
    @DisplayName("Same shape requirement error message")
    void testRequireSameShape() {
        ValidUtil.requireSameShape(new int[] {2, 3}, new int[] {2, 3});
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.requireSameShape(new int[] {2, 3}, new int[] {3, 2}),
                "Shape mismatch: [2, 3] vs [3, 2]");

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.requireSameShape("Add", new int[] {2, 3}, new int[] {3, 2}),
                "Add: shape mismatch [2, 3] vs [3, 2]");
    }

    @Test
    @DisplayName("2-D tensor requirement error message")
    void testRequire2D() {
        ValidUtil.require2D("transpose", new int[] {3, 4});
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.require2D("transpose", new int[] {1, 2, 3}),
                "transpose requires a 2-D tensor, got [1, 2, 3]");
    }

    @Test
    @DisplayName("Aliasing requirement error message")
    void testRequireNoAlias() {
        ValidUtil.requireNoAlias(10L, 20L, "Cannot alias");
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.requireNoAlias(10L, 10L, "Cannot alias buffer"),
                "Cannot alias buffer");
    }

    @Test
    @DisplayName("Contiguity detection")
    void testIsContiguous() {
        // [2, 3] with row-major strides [3, 1]
        assertTrue(ValidUtil.isContiguous(new int[] {2, 3}, new int[] {3, 1}));
        // [2, 3] transposed strides [1, 2] -> not contiguous
        assertFalse(ValidUtil.isContiguous(new int[] {2, 3}, new int[] {1, 2}));
        // 1-D [5] strides [1]
        assertTrue(ValidUtil.isContiguous(new int[] {5}, new int[] {1}));
        // 3-D [2, 3, 4] with strides [12, 4, 1]
        assertTrue(ValidUtil.isContiguous(new int[] {2, 3, 4}, new int[] {12, 4, 1}));
    }

    @Test
    @DisplayName("Matmul dimension and shape validation error message")
    void testMatmulDims() {
        // [2, 3] x [3, 4] -> [2, 4]
        Dims d = ValidUtil.matmulDims(new int[] {2, 3}, new int[] {3, 4}, false, false);
        assertEquals(2, d.m());
        assertEquals(4, d.n());
        assertEquals(3, d.k());
        assertArrayEquals(new int[] {2, 4}, d.resultShape());

        // Incompatible inner dimension
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.matmulDims(new int[] {2, 3}, new int[] {4, 5}, false, false),
                "Matmul shape mismatch: left shape [2, 3] and right shape [4, 5] are incompatible because 3 != 4");

        // Unsupported rank (> 2)
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.matmulDims(new int[] {1, 2, 3}, new int[] {3, 4}, false, false),
                "Matmul supports 1-D and 2-D operands, got [1, 2, 3] and [3, 4]");

        // Transposition support
        Dims dt = ValidUtil.matmulDims(new int[] {3, 2}, new int[] {4, 3}, true, true);
        assertEquals(2, dt.m());
        assertEquals(4, dt.n());
        assertEquals(3, dt.k());
        assertArrayEquals(new int[] {2, 4}, dt.resultShape());

        // Vector operands: [M, K] x [K] -> [M]
        Dims dv = ValidUtil.matmulDims(new int[] {5, 4}, new int[] {4}, false, false);
        assertEquals(5, dv.m());
        assertEquals(1, dv.n());
        assertEquals(4, dv.k());
        assertArrayEquals(new int[] {5}, dv.resultShape());

        // [K] x [K] -> [1]
        Dims dot = ValidUtil.matmulDims(new int[] {4}, new int[] {4}, false, false);
        assertArrayEquals(new int[] {1}, dot.resultShape());
    }

    @Test
    @DisplayName("Row operations layout calculation and validation")
    void testRows() {
        Rows r = ValidUtil.rows(new int[] {4, 8, 128}, 4096);
        assertEquals(32, r.rows());
        assertEquals(128, r.cols());
        assertArrayEquals(new int[] {4, 8}, r.reducedShape(new int[] {4, 8, 128}));

        Rows r1d = ValidUtil.rows(new int[] {128}, 128);
        assertEquals(1, r1d.rows());
        assertEquals(128, r1d.cols());
        assertArrayEquals(new int[] {1}, r1d.reducedShape(new int[] {128}));

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.rows(new int[] {}, 0),
                "Row ops require a non-empty tensor, got shape []");
    }
}
