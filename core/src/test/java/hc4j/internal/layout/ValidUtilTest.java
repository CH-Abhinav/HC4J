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

    @Test
    @DisplayName("Valid shape requirement validation")
    void testRequireValidShape() {
        ValidUtil.requireValidShape(new int[] {2, 3, 4});
        ValidUtil.requireValidShape(new int[] {1});

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.requireValidShape(null),
                "Shape must be non-null and non-empty");

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.requireValidShape(new int[] {}),
                "Shape must be non-null and non-empty");

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.requireValidShape(new int[] {2, 0, 4}),
                "Shape dimensions must be strictly positive, got [2, 0, 4]");

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.requireValidShape(new int[] {2, -3, 4}),
                "Shape dimensions must be strictly positive, got [2, -3, 4]");
    }

    @Test
    @DisplayName("Broadcast shape calculations and errors")
    void testBroadcastShape() {
        // Equal shapes
        assertArrayEquals(new int[] {2, 3}, ValidUtil.broadcastShape(new int[] {2, 3}, new int[] {2, 3}));

        // Trailing expansion
        assertArrayEquals(new int[] {2, 3, 4}, ValidUtil.broadcastShape(new int[] {2, 3, 4}, new int[] {4}));
        assertArrayEquals(new int[] {2, 3, 4}, ValidUtil.broadcastShape(new int[] {4}, new int[] {2, 3, 4}));

        // 1-dim expansion
        assertArrayEquals(new int[] {2, 3, 4}, ValidUtil.broadcastShape(new int[] {2, 1, 4}, new int[] {1, 3, 1}));

        // Incompatible dimensions
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.broadcastShape(new int[] {3, 2}, new int[] {3, 4}),
                "Cannot broadcast shapes [3, 2] and [3, 4] at dimension 1 (2 != 4)");

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.broadcastShape(new int[] {2, 3}, new int[] {3, 3}),
                "Cannot broadcast shapes [2, 3] and [3, 3] at dimension 0 (2 != 3)");
    }

    @Test
    @DisplayName("Broadcast strides calculation")
    void testBroadcastStrides() {
        // [1, 4] with strides [4, 1] broadcast to [3, 4] -> strides [0, 1]
        int[] bStrides = ValidUtil.broadcastStrides(new int[] {1, 4}, new int[] {4, 1}, new int[] {3, 4});
        assertArrayEquals(new int[] {0, 1}, bStrides);

        // [4] with strides [1] broadcast to [2, 3, 4] -> strides [0, 0, 1]
        int[] bStrides3d = ValidUtil.broadcastStrides(new int[] {4}, new int[] {1}, new int[] {2, 3, 4});
        assertArrayEquals(new int[] {0, 0, 1}, bStrides3d);

        // Rank mismatch error
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.broadcastStrides(new int[] {2, 3, 4}, new int[] {12, 4, 1}, new int[] {4}),
                "Source rank 3 exceeds target rank 1");

        // Dimension mismatch error
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.broadcastStrides(new int[] {2, 4}, new int[] {4, 1}, new int[] {3, 4}),
                "Cannot broadcast dimension 2 to 3");
    }

    @Test
    @DisplayName("Infer reshape dimensions with -1 and validation")
    void testInferReshape() {
        // Simple shape without -1
        assertArrayEquals(new int[] {2, 3}, ValidUtil.inferReshape(new int[] {2, 3}, 6));

        // Unknown dimension inference
        assertArrayEquals(new int[] {2, 3}, ValidUtil.inferReshape(new int[] {2, -1}, 6));
        assertArrayEquals(new int[] {2, 3}, ValidUtil.inferReshape(new int[] {-1, 3}, 6));
        assertArrayEquals(new int[] {6}, ValidUtil.inferReshape(new int[] {-1}, 6));
        assertArrayEquals(new int[] {2, 3, 4}, ValidUtil.inferReshape(new int[] {2, -1, 4}, 24));

        // Null or empty
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.inferReshape(null, 6),
                "Shape must be non-null and non-empty");
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.inferReshape(new int[] {}, 6),
                "Shape must be non-null and non-empty");

        // Multiple -1
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.inferReshape(new int[] {-1, -1}, 6),
                "Can only specify at most one unknown dimension (-1)");

        // Invalid dimension (e.g. -2 or 0)
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.inferReshape(new int[] {2, -2}, 6),
                "Invalid dimension size: -2");
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.inferReshape(new int[] {2, 0}, 6),
                "Invalid dimension size: 0");

        // Incompatible element count
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.inferReshape(new int[] {2, 4}, 6),
                "Cannot reshape array of size 6 into shape [2, 4]");

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.inferReshape(new int[] {4, -1}, 6),
                "Cannot reshape array of size 6 into shape [4, -1]");
    }

    @Test
    @DisplayName("Permutation axes validation")
    void testValidatePermutation() {
        ValidUtil.validatePermutation(new int[] {1, 0}, 2);
        ValidUtil.validatePermutation(new int[] {2, 0, 1}, 3);

        // Rank mismatch
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.validatePermutation(new int[] {0, 1}, 3),
                "Permutation axes length 2 does not match rank 3");

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.validatePermutation(null, 2),
                "Permutation axes length 0 does not match rank 2");

        // Axis out of bounds
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.validatePermutation(new int[] {0, 2}, 2),
                "Axis 2 is out of bounds for rank 2");

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.validatePermutation(new int[] {-1, 0}, 2),
                "Axis -1 is out of bounds for rank 2");

        // Duplicate axis
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.validatePermutation(new int[] {1, 1}, 2),
                "Duplicate axis in permutation: 1");
    }

    @Test
    @DisplayName("Axis normalization: positive, negative, and out of bounds")
    void testNormalizeAxis() {
        assertEquals(0, ValidUtil.normalizeAxis(0, 3));
        assertEquals(1, ValidUtil.normalizeAxis(1, 3));
        assertEquals(2, ValidUtil.normalizeAxis(2, 3));
        assertEquals(2, ValidUtil.normalizeAxis(-1, 3));
        assertEquals(1, ValidUtil.normalizeAxis(-2, 3));
        assertEquals(0, ValidUtil.normalizeAxis(-3, 3));

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.normalizeAxis(3, 3),
                "Axis 3 is out of bounds for rank 3");
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.normalizeAxis(-4, 3),
                "Axis -4 is out of bounds for rank 3");
    }

    @Test
    @DisplayName("Batched matmul plan: 3D and 4D broadcasting and dimension checks")
    void testBatchedMatmulPlan() {
        // [2, 3, 4] @ [2, 4, 5] -> [2, 3, 5]
        int[] shapeA = {2, 3, 4};
        int[] stridesA = {12, 4, 1};
        int[] shapeB = {2, 4, 5};
        int[] stridesB = {20, 5, 1};
        ValidUtil.BatchedMatmulPlan plan = ValidUtil.batchedMatmulPlan(shapeA, stridesA, shapeB, stridesB);
        assertEquals(3, plan.m());
        assertEquals(5, plan.n());
        assertEquals(4, plan.k());
        assertArrayEquals(new int[]{2}, plan.batchShape());
        assertArrayEquals(new int[]{12}, plan.aBatchStrides());
        assertArrayEquals(new int[]{20}, plan.bBatchStrides());
        assertArrayEquals(new int[]{2, 3, 5}, plan.outputShape());

        // Batch broadcast: [2, 1, 3, 4] @ [1, 5, 4, 2] -> [2, 5, 3, 2]
        int[] shape4dA = {2, 1, 3, 4};
        int[] strides4dA = {12, 12, 4, 1};
        int[] shape4dB = {1, 5, 4, 2};
        int[] strides4dB = {40, 8, 2, 1};
        ValidUtil.BatchedMatmulPlan plan4d = ValidUtil.batchedMatmulPlan(shape4dA, strides4dA, shape4dB, strides4dB);
        assertArrayEquals(new int[]{2, 5}, plan4d.batchShape());
        assertArrayEquals(new int[]{12, 0}, plan4d.aBatchStrides());
        assertArrayEquals(new int[]{0, 8}, plan4d.bBatchStrides());
        assertArrayEquals(new int[]{2, 5, 3, 2}, plan4d.outputShape());

        // 2D B broadcast with 3D A: [4, 8, 12] @ [12, 6] -> [4, 8, 6]
        int[] shape3dA = {4, 8, 12};
        int[] strides3dA = {96, 12, 1};
        int[] shape2dB = {12, 6};
        int[] strides2dB = {6, 1};
        ValidUtil.BatchedMatmulPlan planB2d = ValidUtil.batchedMatmulPlan(shape3dA, strides3dA, shape2dB, strides2dB);
        assertArrayEquals(new int[]{4}, planB2d.batchShape());
        assertArrayEquals(new int[]{96}, planB2d.aBatchStrides());
        assertArrayEquals(new int[]{0}, planB2d.bBatchStrides());
        assertArrayEquals(new int[]{4, 8, 6}, planB2d.outputShape());

        // Inner dimension mismatch
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> ValidUtil.batchedMatmulPlan(new int[]{2, 3, 5}, new int[]{15, 5, 1}, new int[]{2, 4, 2}, new int[]{8, 2, 1}),
                "Matmul shape mismatch: inner dimensions 5 and 4 do not match");
    }
}
