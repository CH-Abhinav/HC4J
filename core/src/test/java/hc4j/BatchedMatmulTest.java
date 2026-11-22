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

@DisplayName("Batched 3D/4D+ Matrix Multiplication Tests")
class BatchedMatmulTest {

    @BeforeAll
    static void setUp() {
        GpuTestSupport.assumeGpuAvailable();
    }

    private static float[] cpuMatmul(float[] a, float[] b, int m, int k, int n) {
        float[] c = new float[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0f;
                for (int p = 0; p < k; p++) {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        return c;
    }

    @Test
    @DisplayName("3D Batched GEMM: [4, 8, 12] @ [4, 12, 6] -> [4, 8, 6]")
    void test3DBatchedMatmul() {
        int batches = 4, m = 8, k = 12, n = 6;
        float[] dataA = new float[batches * m * k];
        float[] dataB = new float[batches * k * n];
        float[] expected = new float[batches * m * n];

        for (int i = 0; i < dataA.length; i++) dataA[i] = (i % 17) * 0.25f - 2f;
        for (int i = 0; i < dataB.length; i++) dataB[i] = (i % 13) * 0.5f - 1f;

        for (int b = 0; b < batches; b++) {
            float[] aSlice = new float[m * k];
            float[] bSlice = new float[k * n];
            System.arraycopy(dataA, b * m * k, aSlice, 0, m * k);
            System.arraycopy(dataB, b * k * n, bSlice, 0, k * n);
            float[] cSlice = cpuMatmul(aSlice, bSlice, m, k, n);
            System.arraycopy(cSlice, 0, expected, b * m * n, m * n);
        }

        try (Tensor a = Tensor.fromArray(dataA, batches, m, k);
             Tensor b = Tensor.fromArray(dataB, batches, k, n);
             Tensor c = a.matmul(b)) {

            assertEquals(3, c.dim());
            assertArrayEquals(new int[]{batches, m, n}, c.getShape());
            assertArrayEquals(expected, c.toArray(), 1e-4f);
        }
    }

    @Test
    @DisplayName("3D Batched GEMM with 2D right operand: [4, 8, 12] @ [12, 6] -> [4, 8, 6]")
    void test3DBatchedWith2DFastPath() {
        int batches = 4, m = 8, k = 12, n = 6;
        float[] dataA = new float[batches * m * k];
        float[] dataB = new float[k * n];
        float[] expected = new float[batches * m * n];

        for (int i = 0; i < dataA.length; i++) dataA[i] = (i % 11) * 0.2f;
        for (int i = 0; i < dataB.length; i++) dataB[i] = (i % 7) * 0.3f;

        for (int b = 0; b < batches; b++) {
            float[] aSlice = new float[m * k];
            System.arraycopy(dataA, b * m * k, aSlice, 0, m * k);
            float[] cSlice = cpuMatmul(aSlice, dataB, m, k, n);
            System.arraycopy(cSlice, 0, expected, b * m * n, m * n);
        }

        try (Tensor a = Tensor.fromArray(dataA, batches, m, k);
             Tensor b = Tensor.fromArray(dataB, k, n);
             Tensor c = a.matmul(b)) {

            assertEquals(3, c.dim());
            assertArrayEquals(new int[]{batches, m, n}, c.getShape());
            assertArrayEquals(expected, c.toArray(), 1e-4f);
        }
    }

    @Test
    @DisplayName("4D Batched GEMM: [2, 3, 5, 7] @ [2, 3, 7, 4] -> [2, 3, 5, 4]")
    void test4DBatchedMatmul() {
        int b0 = 2, b1 = 3, m = 5, k = 7, n = 4;
        int totalBatches = b0 * b1;
        float[] dataA = new float[totalBatches * m * k];
        float[] dataB = new float[totalBatches * k * n];
        float[] expected = new float[totalBatches * m * n];

        for (int i = 0; i < dataA.length; i++) dataA[i] = (i % 9) * 0.1f;
        for (int i = 0; i < dataB.length; i++) dataB[i] = (i % 5) * 0.2f;

        for (int b = 0; b < totalBatches; b++) {
            float[] aSlice = new float[m * k];
            float[] bSlice = new float[k * n];
            System.arraycopy(dataA, b * m * k, aSlice, 0, m * k);
            System.arraycopy(dataB, b * k * n, bSlice, 0, k * n);
            float[] cSlice = cpuMatmul(aSlice, bSlice, m, k, n);
            System.arraycopy(cSlice, 0, expected, b * m * n, m * n);
        }

        try (Tensor a = Tensor.fromArray(dataA, b0, b1, m, k);
             Tensor b = Tensor.fromArray(dataB, b0, b1, k, n);
             Tensor c = a.matmul(b)) {

            assertEquals(4, c.dim());
            assertArrayEquals(new int[]{b0, b1, m, n}, c.getShape());
            assertArrayEquals(expected, c.toArray(), 1e-4f);
        }
    }

    @Test
    @DisplayName("4D Cross-Broadcasting: [2, 1, 5, 7] @ [1, 3, 7, 4] -> [2, 3, 5, 4]")
    void test4DCrossBroadcasting() {
        int m = 5, k = 7, n = 4;
        float[] dataA = new float[2 * 1 * m * k];
        float[] dataB = new float[1 * 3 * k * n];
        float[] expected = new float[2 * 3 * m * n];

        for (int i = 0; i < dataA.length; i++) dataA[i] = (i + 1) * 0.05f;
        for (int i = 0; i < dataB.length; i++) dataB[i] = (i + 1) * 0.02f;

        for (int i0 = 0; i0 < 2; i0++) {
            float[] aSlice = new float[m * k];
            System.arraycopy(dataA, i0 * m * k, aSlice, 0, m * k);
            for (int i1 = 0; i1 < 3; i1++) {
                float[] bSlice = new float[k * n];
                System.arraycopy(dataB, i1 * k * n, bSlice, 0, k * n);
                float[] cSlice = cpuMatmul(aSlice, bSlice, m, k, n);
                int outBatch = i0 * 3 + i1;
                System.arraycopy(cSlice, 0, expected, outBatch * m * n, m * n);
            }
        }

        try (Tensor a = Tensor.fromArray(dataA, 2, 1, m, k);
             Tensor b = Tensor.fromArray(dataB, 1, 3, k, n);
             Tensor c = a.matmul(b)) {

            assertEquals(4, c.dim());
            assertArrayEquals(new int[]{2, 3, m, n}, c.getShape());
            assertArrayEquals(expected, c.toArray(), 1e-4f);
        }
    }

    @Test
    @DisplayName("Attention-style Q @ K^T: [2, 4, 8, 16] @ [2, 4, 16, 8] -> [2, 4, 8, 8]")
    void testAttentionHeadTransposed() {
        int b = 2, h = 4, seq = 8, d = 16;
        float[] qData = new float[b * h * seq * d];
        float[] kData = new float[b * h * seq * d];

        for (int i = 0; i < qData.length; i++) qData[i] = (float) Math.sin(i * 0.1);
        for (int i = 0; i < kData.length; i++) kData[i] = (float) Math.cos(i * 0.1);

        try (Tensor q = Tensor.fromArray(qData, b, h, seq, d);
             Tensor k = Tensor.fromArray(kData, b, h, seq, d);
             Tensor kt = k.permute(0, 1, 3, 2);
             Tensor scores = q.matmul(kt);
             Tensor scoresLinalg = Linalg.matmul(q, kt);
             Tensor scoresHc4j = HC4J.matmul(q, kt)) {

            assertArrayEquals(new int[]{b, h, seq, seq}, scores.getShape());
            assertArrayEquals(new int[]{b, h, seq, seq}, scoresLinalg.getShape());
            assertArrayEquals(new int[]{b, h, seq, seq}, scoresHc4j.getShape());

            assertArrayEquals(scores.toArray(), scoresLinalg.toArray(), 1e-5f);
            assertArrayEquals(scores.toArray(), scoresHc4j.toArray(), 1e-5f);
        }
    }
}
