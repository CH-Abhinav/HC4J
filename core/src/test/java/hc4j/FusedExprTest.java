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

package hc4j;

import static hc4j.testutil.ExceptionTable.assertThrowsWithMessage;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import hc4j.internal.ffi.TensorAccess;
import hc4j.testutil.GpuTestSupport;
import hc4j.testutil.TestAssertions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("FusedExpr Expression Construction & Validation Tests")
class FusedExprTest {

    @Test
    @DisplayName("FusedExpr construction and chaining without GPU dispatch")
    void testExpressionTreeConstruction() {
        Tensor dummyA = TensorAccess.getInstance().create(DType.f32, new int[] {4, 4}, new int[] {4, 1}, 0L);
        Tensor dummyB = TensorAccess.getInstance().create(DType.f32, new int[] {4, 4}, new int[] {4, 1}, 0L);

        FusedExpr expr = dummyA.lazy()
                .sin()
                .add(dummyB)
                .mul(2.5f)
                .relu()
                .sub(1.0f);

        assertNotNull(expr);
    }

    @Test
    @DisplayName("FusedExpr detects shape mismatches on binary nodes")
    void testShapeMismatch() {
        Tensor a = TensorAccess.getInstance().create(DType.f32, new int[] {2, 3}, new int[] {3, 1}, 0L);
        Tensor b = TensorAccess.getInstance().create(DType.f32, new int[] {2, 4}, new int[] {4, 1}, 0L);

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> a.lazy().add(b),
                "Shape mismatch: [2, 3] vs [2, 4]");
    }

    @Test
    @DisplayName("FusedExpr rejects non-finite scalar constants")
    void testNonFiniteConstants() {
        Tensor a = TensorAccess.getInstance().create(DType.f32, new int[] {2, 2}, new int[] {2, 1}, 0L);

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> a.lazy().add(Float.NaN),
                "Fused constants must be finite: NaN");

        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> a.lazy().mul(Float.POSITIVE_INFINITY),
                "Fused constants must be finite: Infinity");
    }

    @Test
    @DisplayName("FusedExpr rejects non-f32 tensors")
    void testNonF32Rejection() {
        Tensor i32Tensor = TensorAccess.getInstance().create(DType.i32, new int[] {2, 2}, new int[] {2, 1}, 0L);
        assertThrowsWithMessage(IllegalArgumentException.class,
                i32Tensor::lazy,
                "fusion is only defined for f32 tensors, got i32");
    }

    @Test
    @DisplayName("FusedExpr eval on real GPU")
    void testGpuEvaluation() {
        GpuTestSupport.assumeGpuAvailable();

        float[] data = {1f, 2f, 3f, 4f};
        try (Tensor a = Tensor.fromArray(data, 2, 2);
             Tensor b = Tensor.fromArray(data, 2, 2);
             Tensor r = a.lazy().sin().add(b).eval()) {
            float[] got = r.toFloatArray();
            for (int i = 0; i < data.length; i++) {
                TestAssertions.assertClose((float) Math.sin(data[i]) + data[i], got[i], 1e-4f, 1e-4f);
            }
        }
    }
}
