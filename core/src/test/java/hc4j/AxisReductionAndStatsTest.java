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

import hc4j.linalg.Linalg;
import hc4j.testutil.GpuTestSupport;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("General Axis Reductions, Statistics & Linalg Parity Tests")
class AxisReductionAndStatsTest {

    @BeforeAll
    static void setUp() {
        GpuTestSupport.assumeGpuAvailable();
    }

    @Test
    @DisplayName("3-D tensor reductions along axis 0, 1, 2, and negative axis -1")
    void test3DAxisReductions() {
        // Shape [2, 3, 4], total 24
        float[] data = new float[24];
        for (int i = 0; i < 24; i++) data[i] = (i + 1) * 1.0f;

        try (Tensor t = Tensor.fromArray(data, 2, 3, 4);
             Tensor sum0 = t.sum(0);
             Tensor sum1 = t.sum(1);
             Tensor sum2 = t.sum(2);
             Tensor sumNeg1 = t.sum(-1);
             Tensor sum1Keep = t.sum(1, true)) {

            // sum along axis 0: [3, 4]
            assertArrayEquals(new int[]{3, 4}, sum0.getShape());
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 4; c++) {
                    float expected = data[0 * 12 + r * 4 + c] + data[1 * 12 + r * 4 + c];
                    assertEquals(expected, sum0.toArray()[r * 4 + c], 1e-5f);
                }
            }

            // sum along axis 1: [2, 4]
            assertArrayEquals(new int[]{2, 4}, sum1.getShape());
            for (int b = 0; b < 2; b++) {
                for (int c = 0; c < 4; c++) {
                    float expected = data[b * 12 + 0 * 4 + c] + data[b * 12 + 1 * 4 + c] + data[b * 12 + 2 * 4 + c];
                    assertEquals(expected, sum1.toArray()[b * 4 + c], 1e-5f);
                }
            }

            // sum along axis 2 vs axis -1: [2, 3]
            assertArrayEquals(new int[]{2, 3}, sum2.getShape());
            assertArrayEquals(new int[]{2, 3}, sumNeg1.getShape());
            assertArrayEquals(sum2.toArray(), sumNeg1.toArray(), 1e-5f);

            // keepDims = true: [2, 1, 4]
            assertArrayEquals(new int[]{2, 1, 4}, sum1Keep.getShape());
            assertArrayEquals(sum1.toArray(), sum1Keep.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("Axis mean, max, min, and prod")
    void testMeanMaxMinProdAxis() {
        // Shape [2, 2]:
        // [[1, 2],
        //  [3, 4]]
        float[] data = {1f, 2f, 3f, 4f};
        try (Tensor t = Tensor.fromArray(data, 2, 2);
             Tensor mean0 = t.mean(0);
             Tensor max1 = t.max(1);
             Tensor min1 = t.min(1);
             Tensor prod0 = t.prod(0)) {

            // mean along axis 0: [(1+3)/2, (2+4)/2] = [2.0, 3.0]
            assertArrayEquals(new float[]{2.0f, 3.0f}, mean0.toArray(), 1e-5f);

            // max along axis 1: [max(1,2), max(3,4)] = [2.0, 4.0]
            assertArrayEquals(new float[]{2.0f, 4.0f}, max1.toArray(), 1e-5f);

            // min along axis 1: [min(1,2), min(3,4)] = [1.0, 3.0]
            assertArrayEquals(new float[]{1.0f, 3.0f}, min1.toArray(), 1e-5f);

            // prod along axis 0: [1*3, 2*4] = [3.0, 8.0]
            assertArrayEquals(new float[]{3.0f, 8.0f}, prod0.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("Axis argmax and argmin returning DType.i32 indices")
    void testAxisArgmaxAndArgmin() {
        // [2, 3]:
        // [[10, 50, 30],
        //  [90, 20, 40]]
        float[] data = {
            10f, 50f, 30f,
            90f, 20f, 40f
        };
        try (Tensor t = Tensor.fromArray(data, 2, 3);
             Tensor argmax1 = t.argmax(1);
             Tensor argmin1 = t.argmin(1);
             Tensor argmax0 = t.argmax(0)) {

            assertEquals(DType.i32, argmax1.getDType());
            assertArrayEquals(new int[]{1, 0}, argmax1.toArrayInt());
            assertArrayEquals(new int[]{0, 1}, argmin1.toArrayInt());
            assertArrayEquals(new int[]{1, 0, 1}, argmax0.toArrayInt());
        }
    }

    @Test
    @DisplayName("1-D reduction with keepDims=false returns shape [1] matching HC4J convention")
    void test1DReductionKeepDimsFalse() {
        try (Tensor t = Tensor.fromArray(new float[]{1f, 2f, 3f, 4f}, 4);
             Tensor sum = t.sum(0, false);
             Tensor sumKeep = t.sum(0, true)) {

            assertArrayEquals(new int[]{1}, sum.getShape());
            assertArrayEquals(new int[]{1}, sumKeep.getShape());
            assertEquals(10f, sum.toArray()[0], 1e-5f);
            assertEquals(10f, sumKeep.toArray()[0], 1e-5f);
        }
    }

    @Test
    @DisplayName("Full tensor and row product: prod() and prodRows()")
    void testFullAndRowProd() {
        // [2, 3]:
        // [[1, 2, 3],
        //  [2, 3, 4]]
        float[] data = {
            1f, 2f, 3f,
            2f, 3f, 4f
        };
        try (Tensor t = Tensor.fromArray(data, 2, 3);
             Tensor prodRows = t.prodRows()) {

            // Global prod: 1*2*3 * 2*3*4 = 6 * 24 = 144
            assertEquals(144f, t.prod(), 1e-4f);

            // Row prod: [1*2*3, 2*3*4] = [6, 24]
            assertArrayEquals(new float[]{6f, 24f}, prodRows.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("Statistics: Frobenius norm, variance, and standard deviation")
    void testStatsParity() {
        // [3, 4]: 3^2 + 4^2 = 25 -> sqrt = 5
        try (Tensor v = Tensor.fromArray(new float[]{3f, 4f}, 2);
             Tensor norm = v.norm()) {
            assertEquals(5.0f, norm.item(), 1e-5f);
        }

        // [2, 4, 4, 4, 5, 5, 7, 9]: mean = 40/8 = 5
        // diffs from 5: [-3, -1, -1, -1, 0, 0, 2, 4]
        // sum of sq diffs: 9 + 1 + 1 + 1 + 0 + 0 + 4 + 16 = 32
        // pop var (ddof=0): 32 / 8 = 4.0, std = 2.0
        // sample var (ddof=1): 32 / 7 = 4.5714287, std = 2.1380899
        float[] d = {2f, 4f, 4f, 4f, 5f, 5f, 7f, 9f};
        try (Tensor t = Tensor.fromArray(d, 8);
             Tensor varPop = t.var();
             Tensor stdPop = t.std();
             Tensor varSample = t.var(1);
             Tensor stdSample = t.std(1)) {

            assertEquals(4.0f, varPop.item(), 1e-5f);
            assertEquals(2.0f, stdPop.item(), 1e-5f);
            assertEquals(32.0f / 7.0f, varSample.item(), 1e-4f);
            assertEquals((float) Math.sqrt(32.0 / 7.0), stdSample.item(), 1e-4f);
        }
    }

    @Test
    @DisplayName("Linear algebra parity: diagonal, trace, dot, and cosineSimilarity")
    void testLinalgParity() {
        // 3x3 matrix:
        // [[1, 2, 3],
        //  [4, 5, 6],
        //  [7, 8, 9]]
        float[] mat = {
            1f, 2f, 3f,
            4f, 5f, 6f,
            7f, 8f, 9f
        };
        try (Tensor m = Tensor.fromArray(mat, 3, 3);
             Tensor diag = m.diagonal();
             Tensor trace = m.trace();
             Tensor diagLinalg = Linalg.diagonal(m);
             Tensor traceLinalg = Linalg.trace(m)) {

            // Diagonal elements: [1, 5, 9]
            assertArrayEquals(new float[]{1f, 5f, 9f}, diag.toArray(), 1e-5f);
            assertArrayEquals(new float[]{1f, 5f, 9f}, diagLinalg.toArray(), 1e-5f);

            // Trace: 1 + 5 + 9 = 15
            assertEquals(15f, trace.item(), 1e-5f);
            assertEquals(15f, traceLinalg.item(), 1e-5f);
        }

        // Rectangular matrix 2x3 diagonal: min(2, 3) = 2 -> [1, 5]
        float[] rect = {
            1f, 2f, 3f,
            4f, 5f, 6f
        };
        try (Tensor r = Tensor.fromArray(rect, 2, 3);
             Tensor diagRect = r.diagonal()) {
            assertArrayEquals(new float[]{1f, 5f}, diagRect.toArray(), 1e-5f);
        }

        // Vector dot product & cosine similarity:
        // a = [1, 2, 3], b = [4, 5, 6]
        // dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        // ||a|| = sqrt(1+4+9) = sqrt(14) ~ 3.741657
        // ||b|| = sqrt(16+25+36) = sqrt(77) ~ 8.774964
        // cosSim = 32 / (sqrt(14) * sqrt(77)) = 32 / sqrt(1078) ~ 0.97463185
        try (Tensor a = Tensor.fromArray(new float[]{1f, 2f, 3f}, 3);
             Tensor b = Tensor.fromArray(new float[]{4f, 5f, 6f}, 3);
             Tensor dot = a.dot(b);
             Tensor cosSim = a.cosineSimilarity(b);
             Tensor dotLinalg = Linalg.dot(a, b);
             Tensor cosSimLinalg = Linalg.cosineSimilarity(a, b)) {

            assertEquals(32.0f, dot.item(), 1e-4f);
            assertEquals(32.0f, dotLinalg.item(), 1e-4f);

            float expectedCos = 32.0f / (float) Math.sqrt(14.0 * 77.0);
            assertEquals(expectedCos, cosSim.item(), 1e-4f);
            assertEquals(expectedCos, cosSimLinalg.item(), 1e-4f);
        }
    }
}
