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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import hc4j.testutil.GpuTestSupport;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Math, Comparisons, Masked Where, Cast, and Clamping Tests")
class MathAndCompareOpsTest {

    @BeforeAll
    static void setUp() {
        GpuTestSupport.assumeGpuAvailable();
    }

    @Test
    @DisplayName("Unary f32 math: abs, neg, rsqrt, floor, ceil, round, sign")
    void testUnaryF32Math() {
        float[] in = {-4.8f, -2.5f, -0.0f, 1.2f, 4.0f, 9.0f};
        try (Tensor t = Tensor.fromArray(in, 2, 3);
             Tensor abs = t.abs();
             Tensor neg = t.neg();
             Tensor floor = t.floor();
             Tensor ceil = t.ceil();
             Tensor round = t.round();
             Tensor sign = t.sign()) {

            assertArrayEquals(new float[]{4.8f, 2.5f, 0.0f, 1.2f, 4.0f, 9.0f}, abs.toArray(), 1e-5f);
            assertArrayEquals(new float[]{4.8f, 2.5f, 0.0f, -1.2f, -4.0f, -9.0f}, neg.toArray(), 1e-5f);
            assertArrayEquals(new float[]{-5.0f, -3.0f, 0.0f, 1.0f, 4.0f, 9.0f}, floor.toArray(), 1e-5f);
            assertArrayEquals(new float[]{-4.0f, -2.0f, 0.0f, 2.0f, 4.0f, 9.0f}, ceil.toArray(), 1e-5f);
            assertArrayEquals(new float[]{-5.0f, -2.0f, 0.0f, 1.0f, 4.0f, 9.0f}, round.toArray(), 1e-5f);
            assertArrayEquals(new float[]{-1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 1.0f}, sign.toArray(), 1e-5f);
        }

        try (Tensor pos = Tensor.fromArray(new float[]{1.0f, 4.0f, 16.0f}, 3);
             Tensor rsqrt = pos.rsqrt()) {
            assertArrayEquals(new float[]{1.0f, 0.5f, 0.25f}, rsqrt.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("Unary i32 math: abs, neg, sign")
    void testUnaryI32Math() {
        int[] in = {-15, -4, 0, 7, 42};
        try (Tensor t = Tensor.fromArray(in, 5);
             Tensor abs = t.abs();
             Tensor neg = t.neg();
             Tensor sign = t.sign()) {

            assertEquals(DType.i32, abs.getDType());
            assertArrayEquals(new int[]{15, 4, 0, 7, 42}, abs.toArrayInt());
            assertArrayEquals(new int[]{15, 4, 0, -7, -42}, neg.toArrayInt());
            assertArrayEquals(new int[]{-1, -1, 0, 1, 1}, sign.toArrayInt());
        }
    }

    @Test
    @DisplayName("Clamp and clip operations on f32 and i32")
    void testClampAndClip() {
        try (Tensor f = Tensor.fromArray(new float[]{-5f, -1f, 0f, 2f, 8f}, 5);
             Tensor clampedF = f.clamp(-2f, 3f);
             Tensor clippedF = f.clip(-2f, 3f)) {

            assertArrayEquals(new float[]{-2f, -1f, 0f, 2f, 3f}, clampedF.toArray(), 1e-5f);
            assertArrayEquals(clampedF.toArray(), clippedF.toArray(), 1e-5f);
        }

        try (Tensor i = Tensor.fromArray(new int[]{-10, -3, 0, 5, 20}, 5);
             Tensor clampedI = i.clamp(-2, 7);
             Tensor clippedI = i.clip(-2, 7)) {

            assertEquals(DType.i32, clampedI.getDType());
            assertArrayEquals(new int[]{-2, -2, 0, 5, 7}, clampedI.toArrayInt());
            assertArrayEquals(clampedI.toArrayInt(), clippedI.toArrayInt());
        }
    }

    @Test
    @DisplayName("In-place unary math: absi, negi, signi, clampi")
    void testInPlaceUnary() {
        try (Tensor f = Tensor.fromArray(new float[]{-3f, 5f}, 2)) {
            f.absi();
            assertArrayEquals(new float[]{3f, 5f}, f.toArray(), 1e-5f);
            f.negi();
            assertArrayEquals(new float[]{-3f, -5f}, f.toArray(), 1e-5f);
            f.signi();
            assertArrayEquals(new float[]{-1f, -1f}, f.toArray(), 1e-5f);
        }

        try (Tensor f = Tensor.fromArray(new float[]{-10f, 2f, 15f}, 3)) {
            f.clampi(0f, 10f);
            assertArrayEquals(new float[]{0f, 2f, 10f}, f.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("Functional binary math: pow, maximum, minimum")
    void testFunctionalBinaryMath() {
        try (Tensor a = Tensor.fromArray(new float[]{2f, 3f, 4f}, 3);
             Tensor b = Tensor.fromArray(new float[]{3f, 2f, 0.5f}, 3);
             Tensor pow = a.pow(b);
             Tensor powScalar = a.pow(2f);
             Tensor max = a.maximum(b);
             Tensor min = a.minimum(b)) {

            assertArrayEquals(new float[]{8f, 9f, 2f}, pow.toArray(), 1e-4f);
            assertArrayEquals(new float[]{4f, 9f, 16f}, powScalar.toArray(), 1e-4f);
            assertArrayEquals(new float[]{3f, 3f, 4f}, max.toArray(), 1e-5f);
            assertArrayEquals(new float[]{2f, 2f, 0.5f}, min.toArray(), 1e-5f);
        }

        try (Tensor aI = Tensor.fromArray(new int[]{10, 20, 5}, 3);
             Tensor bI = Tensor.fromArray(new int[]{15, 12, 8}, 3);
             Tensor maxI = aI.maximum(bI);
             Tensor minI = aI.minimum(bI);
             Tensor maxScalar = aI.maximum(12)) {

            assertArrayEquals(new int[]{15, 20, 8}, maxI.toArrayInt());
            assertArrayEquals(new int[]{10, 12, 5}, minI.toArrayInt());
            assertArrayEquals(new int[]{12, 20, 12}, maxScalar.toArrayInt());
        }
    }

    @Test
    @DisplayName("Elementwise comparison operators returning DType.i32 mask")
    void testComparisons() {
        try (Tensor a = Tensor.fromArray(new float[]{1f, 5f, 3f, 7f}, 4);
             Tensor b = Tensor.fromArray(new float[]{2f, 5f, 1f, 9f}, 4);
             Tensor eq = a.eq(b);
             Tensor ne = a.ne(b);
             Tensor gt = a.gt(b);
             Tensor ge = a.ge(b);
             Tensor lt = a.lt(b);
             Tensor le = a.le(b)) {

            assertEquals(DType.i32, eq.getDType());
            assertArrayEquals(new int[]{0, 1, 0, 0}, eq.toArrayInt());
            assertArrayEquals(new int[]{1, 0, 1, 1}, ne.toArrayInt());
            assertArrayEquals(new int[]{0, 0, 1, 0}, gt.toArrayInt());
            assertArrayEquals(new int[]{0, 1, 1, 0}, ge.toArrayInt());
            assertArrayEquals(new int[]{1, 0, 0, 1}, lt.toArrayInt());
            assertArrayEquals(new int[]{1, 1, 0, 1}, le.toArrayInt());
        }

        // Scalar comparison
        try (Tensor x = Tensor.fromArray(new int[]{2, 5, 8}, 3);
             Tensor gtScalar = x.gt(4);
             Tensor leScalar = x.le(5)) {

            assertArrayEquals(new int[]{0, 1, 1}, gtScalar.toArrayInt());
            assertArrayEquals(new int[]{1, 1, 0}, leScalar.toArrayInt());
        }
    }

    @Test
    @DisplayName("3-way broadcasted masked where: cond ? x : y")
    void testMaskedWhere() {
        try (Tensor cond = Tensor.fromArray(new int[]{1, 0, 1, 0}, 4);
             Tensor x = Tensor.fromArray(new float[]{10f, 20f, 30f, 40f}, 4);
             Tensor y = Tensor.fromArray(new float[]{-1f, -2f, -3f, -4f}, 4);
             Tensor res = cond.where(x, y);
             Tensor resScalar = cond.where(100f, -100f)) {

            assertArrayEquals(new float[]{10f, -2f, 30f, -4f}, res.toArray(), 1e-5f);
            assertArrayEquals(new float[]{100f, -100f, 100f, -100f}, resScalar.toArray(), 1e-5f);
        }

        // 2D broadcasting where: cond [2, 1], x [1, 3], y scalar
        try (Tensor cond2D = Tensor.fromArray(new int[]{1, 0}, 2, 1);
             Tensor xRow = Tensor.fromArray(new float[]{10f, 20f, 30f}, 1, 3);
             Tensor resBcast = cond2D.where(xRow, 0f)) {

            assertEquals(2, resBcast.dim());
            assertArrayEquals(new int[]{2, 3}, resBcast.getShape());
            // Row 0 has cond=1 -> [10, 20, 30]
            // Row 1 has cond=0 -> [0, 0, 0]
            float[] expected = {
                10f, 20f, 30f,
                 0f,  0f,  0f
            };
            assertArrayEquals(expected, resBcast.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("Type casting: f32 <-> i32 and self-copy")
    void testTypeCasting() {
        try (Tensor f = Tensor.fromArray(new float[]{-3.7f, 0.0f, 2.9f}, 3);
             Tensor i = f.cast(DType.i32);
             Tensor fBack = i.cast(DType.f32);
             Tensor same = f.cast(DType.f32)) {

            assertEquals(DType.i32, i.getDType());
            assertArrayEquals(new int[]{-3, 0, 2}, i.toArrayInt());

            assertEquals(DType.f32, fBack.getDType());
            assertArrayEquals(new float[]{-3.0f, 0.0f, 2.0f}, fBack.toArray(), 1e-5f);

            assertEquals(DType.f32, same.getDType());
            assertArrayEquals(f.toArray(), same.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("FusedExpr clamp and rsqrt")
    void testFusedExprClampAndRsqrt() {
        try (Tensor x = Tensor.fromArray(new float[]{0.25f, 1.0f, 4.0f, 16.0f}, 4);
             Tensor rsqrtRes = x.lazy().rsqrt().eval();
             Tensor clampRes = x.lazy().clamp(0.5f, 5.0f).eval()) {

            assertArrayEquals(new float[]{2.0f, 1.0f, 0.5f, 0.25f}, rsqrtRes.toArray(), 1e-4f);
            assertArrayEquals(new float[]{0.5f, 1.0f, 4.0f, 5.0f}, clampRes.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("HC4J facade methods for math, comparisons, and where")
    void testHC4JFacade() {
        try (Tensor a = HC4J.fromArray(new float[]{-2f, 3f}, 2);
             Tensor b = HC4J.fromArray(new float[]{4f, -1f}, 2);
             Tensor abs = HC4J.abs(a);
             Tensor max = HC4J.maximum(a, b);
             Tensor gt = HC4J.gt(a, 0f);
             Tensor w = HC4J.where(gt, a, b)) {

            assertArrayEquals(new float[]{2f, 3f}, abs.toArray(), 1e-5f);
            assertArrayEquals(new float[]{4f, 3f}, max.toArray(), 1e-5f);
            assertArrayEquals(new int[]{0, 1}, gt.toArrayInt());
            assertArrayEquals(new float[]{4f, 3f}, w.toArray(), 1e-5f);
        }
    }
}
