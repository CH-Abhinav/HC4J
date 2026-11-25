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

import hc4j.engine.Engine;
import hc4j.linalg.Linalg;
import hc4j.nn.Nn;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Step 4D: Portable f16, Embedding & RoPE Kernels Test")
class F16AndTransformerOpsTest {

    // Helper: float to IEEE 754 half-precision bits
    private static short floatToHalf(float fval) {
        int fbits = Float.floatToIntBits(fval);
        int sign = (fbits >>> 16) & 0x8000;
        int val = (fbits & 0x7fffffff) + 0x1000;

        if (val >= 0x47800000) {
            if ((fbits & 0x7fffffff) >= 0x47800000) {
                if (val < 0x7f800000) return (short) (sign | 0x7c00);
                return (short) (sign | 0x7c00 | ((fbits & 0x007fffff) >>> 13));
            }
            return (short) (sign | 0x7bff);
        }
        if (val >= 0x38800000) return (short) (sign | ((val - 0x38000000) >>> 13));
        if (val < 0x33000000) return (short) sign;
        val = (fbits & 0x7fffffff) >>> 23;
        return (short) (sign | ((((fbits & 0x7fffff) | 0x800000) + (0x800000 >>> (val - 102))) >>> (126 - val)));
    }

    // Helper: IEEE 754 half-precision bits to float
    private static float halfToFloat(short hbits) {
        int h = hbits & 0xffff;
        int s = (h & 0x8000) << 16;
        int e = (h & 0x7c00) >> 10;
        int m = h & 0x03ff;

        if (e == 0) {
            if (m == 0) return Float.intBitsToFloat(s);
            while ((m & 0x0400) == 0) {
                m <<= 1;
                e--;
            }
            e++;
            m &= ~0x0400;
        } else if (e == 31) {
            return Float.intBitsToFloat(s | 0x7f800000 | (m << 13));
        }
        e = e + (127 - 15);
        m = m << 13;
        return Float.intBitsToFloat(s | (e << 23) | m);
    }

    @Test
    @DisplayName("DType.f16 off-heap round trip and basic allocation")
    void testF16AllocationAndOffHeapRoundTrip() {
        short[] testBits = new short[1003]; // Odd element count
        for (int i = 0; i < testBits.length; i++) {
            float val = (i - 500) * 0.125f;
            testBits[i] = floatToHalf(val);
        }

        try (Tensor t = Tensor.fromHalfBits(testBits, 1003)) {
            assertEquals(DType.f16, t.getDType());
            assertEquals(1003, t.getSize());

            short[] readback = t.toHalfBitsArray();
            assertArrayEquals(testBits, readback);

            // Test MemorySegment copyTo / copyFrom
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment seg = arena.allocate(ValueLayout.JAVA_SHORT, 1003);
                t.copyTo(seg);

                try (Tensor t2 = Tensor.empty(DType.f16, 1003)) {
                    t2.copyFrom(seg);
                    assertArrayEquals(testBits, t2.toHalfBitsArray());
                }
            }
        }
    }

    @Test
    @DisplayName("GPU f32 <-> f16 cast round trip")
    void testF32ToF16CastRoundTrip() {
        int count = 1003; // Odd length to test packing boundary
        float[] original = new float[count];
        for (int i = 0; i < count; i++) {
            original[i] = (i - 500) * 0.0625f;
        }

        try (Tensor f32In = Tensor.fromArray(original, count);
             Tensor f16 = f32In.cast(DType.f16);
             Tensor f32Out = f16.cast(DType.f32)) {

            assertEquals(DType.f16, f16.getDType());
            assertEquals(DType.f32, f32Out.getDType());

            float[] downloaded = f32Out.toFloatArray();
            assertEquals(count, downloaded.length);

            for (int i = 0; i < count; i++) {
                float expectedHalf = halfToFloat(floatToHalf(original[i]));
                assertEquals(expectedHalf, downloaded[i], 1e-4f, "Mismatch at index " + i);
            }
        }
    }

    @Test
    @DisplayName("Mixed precision f32 x f16 GEMM and GEMV (both standard and transposed B)")
    void testMixedPrecisionMatmulF32F16() {
        int m = 17;
        int n = 35;
        int k = 64;

        float[] aData = new float[m * k];
        for (int i = 0; i < aData.length; i++) {
            aData[i] = (float) Math.sin(i + 1);
        }

        // B in transposed layout [N, K]
        float[] bData = new float[n * k];
        short[] bBits = new short[n * k];
        for (int i = 0; i < bData.length; i++) {
            bData[i] = (float) Math.cos(i + 1);
            bBits[i] = floatToHalf(bData[i]);
        }

        // CPU reference: C[i, j] = sum_p A[i, p] * B[j, p]
        float[] expected = new float[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int p = 0; p < k; p++) {
                    double aVal = aData[i * k + p];
                    double bVal = halfToFloat(bBits[j * k + p]);
                    sum += aVal * bVal;
                }
                expected[i * n + j] = (float) sum;
            }
        }

        try (Tensor a = Tensor.fromArray(aData, m, k);
             Tensor b = Tensor.fromHalfBits(bBits, n, k);
             Tensor c = a.matmulTransposedB(b)) {

            assertEquals(DType.f32, c.getDType());
            assertArrayEquals(new int[]{m, n}, c.getShape());

            float[] cData = c.toFloatArray();
            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i], cData[i], 1e-2f, "Mismatch at index " + i);
            }
        }

        // Also test GEMV decode case: M = 1
        float[] aVec = new float[k];
        for (int p = 0; p < k; p++) {
            aVec[p] = (float) Math.sin(p + 1);
        }
        float[] expectedVec = new float[n];
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += aVec[p] * halfToFloat(bBits[j * k + p]);
            }
            expectedVec[j] = (float) sum;
        }

        try (Tensor a = Tensor.fromArray(aVec, 1, k);
             Tensor b = Tensor.fromHalfBits(bBits, n, k);
             Tensor c = a.matmulTransposedB(b)) {

            assertEquals(DType.f32, c.getDType());
            assertArrayEquals(new int[]{1, n}, c.getShape());

            float[] cData = c.toFloatArray();
            for (int j = 0; j < n; j++) {
                assertEquals(expectedVec[j], cData[j], 1e-2f, "GEMV mismatch at col " + j);
            }
        }
    }

    @Test
    @DisplayName("Nn.linearTransposed with f16 weights and bias")
    void testLinearTransposed() {
        int m = 4;
        int n = 8;
        int k = 16;

        float[] aData = new float[m * k];
        for (int i = 0; i < aData.length; i++) aData[i] = 0.5f;

        short[] bBits = new short[n * k];
        for (int i = 0; i < bBits.length; i++) bBits[i] = floatToHalf(0.25f);

        float[] biasData = new float[n];
        for (int j = 0; j < n; j++) biasData[j] = 1.0f * (j + 1);

        // Expected: 0.5 * 0.25 * 16 + bias[j] = 2.0 + bias[j]
        try (Tensor a = Tensor.fromArray(aData, m, k);
             Tensor w = Tensor.fromHalfBits(bBits, n, k);
             Tensor bias = Tensor.fromArray(biasData, n);
             Tensor out = Nn.linearTransposed(a, w, bias)) {

            assertEquals(DType.f32, out.getDType());
            assertArrayEquals(new int[]{m, n}, out.getShape());

            float[] outData = out.toFloatArray();
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    assertEquals(2.0f + (j + 1), outData[i * n + j], 1e-3f);
                }
            }
        }
    }

    @Test
    @DisplayName("Nn.embedding with both f32 and f16 tables and out-of-bounds NaN")
    void testEmbedding() {
        int vocab = 100;
        int hidden = 32;

        float[] tableF32 = new float[vocab * hidden];
        short[] tableF16 = new short[vocab * hidden];
        for (int v = 0; v < vocab; v++) {
            for (int h = 0; h < hidden; h++) {
                float val = v * 1000.0f + h;
                tableF32[v * hidden + h] = val;
                tableF16[v * hidden + h] = floatToHalf(val);
            }
        }

        int[] tokenIds = new int[]{5, 0, 99, 150}; // 150 is out-of-bounds -> should yield NaNs!

        try (Tensor tF32 = Tensor.fromArray(tableF32, vocab, hidden);
             Tensor tF16 = Tensor.fromHalfBits(tableF16, vocab, hidden);
             Tensor tokens = Tensor.fromArray(tokenIds, 2, 2)) { // 2x2 token grid

            // 1. Embedding lookup with f32 table
            try (Tensor embF32 = tF32.embedding(tokens)) {
                assertArrayEquals(new int[]{2, 2, hidden}, embF32.getShape());
                float[] res = embF32.toFloatArray();

                // Token 5
                for (int h = 0; h < hidden; h++) {
                    assertEquals(5000.0f + h, res[h], 1e-4f);
                }
                // Token 0
                for (int h = 0; h < hidden; h++) {
                    assertEquals(h, res[hidden + h], 1e-4f);
                }
                // Token 99
                for (int h = 0; h < hidden; h++) {
                    assertEquals(99000.0f + h, res[2 * hidden + h], 1e-4f);
                }
                // Token 150 (out of bounds -> NaN)
                for (int h = 0; h < hidden; h++) {
                    assertTrue(Float.isNaN(res[3 * hidden + h]), "Expected NaN for out-of-bounds token");
                }
            }

            // 2. Embedding lookup with f16 table
            try (Tensor embF16 = tF16.embedding(tokens)) {
                assertArrayEquals(new int[]{2, 2, hidden}, embF16.getShape());
                float[] res = embF16.toFloatArray();

                // Token 5
                for (int h = 0; h < hidden; h++) {
                    assertEquals(halfToFloat(floatToHalf(5000.0f + h)), res[h], 1e-2f);
                }
                // Token 150 (out of bounds -> NaN)
                for (int h = 0; h < hidden; h++) {
                    assertTrue(Float.isNaN(res[3 * hidden + h]), "Expected NaN for out-of-bounds token in f16 table");
                }
            }
        }
    }

    @Test
    @DisplayName("Nn.rope (NeoX half-split) matches reference both in-place and out-of-place")
    void testRoPENeoxHalfSplit() {
        int batch = 1;
        int seqLen = 3;
        int numHeads = 2;
        int headDim = 4;
        int posOffset = 5;
        float theta = 10000.0f;

        int totalElements = batch * seqLen * numHeads * headDim;
        float[] input = new float[totalElements];
        for (int i = 0; i < totalElements; i++) {
            input[i] = (float) (i + 1);
        }

        // Compute pure-Java reference
        float[] expected = new float[totalElements];
        int half = headDim / 2;
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                int pos = posOffset + s;
                for (int h = 0; h < numHeads; h++) {
                    int base = ((b * seqLen + s) * numHeads + h) * headDim;
                    for (int d = 0; d < half; d++) {
                        double freq = Math.exp(-Math.log(theta) * (2.0 * d) / headDim);
                        double angle = pos * freq;
                        double cos = Math.cos(angle);
                        double sin = Math.sin(angle);

                        double x0 = input[base + d];
                        double x1 = input[base + d + half];

                        expected[base + d] = (float) (x0 * cos - x1 * sin);
                        expected[base + d + half] = (float) (x0 * sin + x1 * cos);
                    }
                }
            }
        }

        // Out-of-place test
        try (Tensor x = Tensor.fromArray(input, batch, seqLen, numHeads, headDim);
             Tensor out = x.rope(posOffset, theta)) {

            float[] res = out.toFloatArray();
            for (int i = 0; i < totalElements; i++) {
                assertEquals(expected[i], res[i], 5e-3f, "RoPE out-of-place mismatch at " + i);
            }
        }

        // In-place test: x.rope(posOffset, theta, x)
        try (Tensor x = Tensor.fromArray(input, batch, seqLen, numHeads, headDim)) {
            x.rope(posOffset, theta, x);
            float[] res = x.toFloatArray();
            for (int i = 0; i < totalElements; i++) {
                assertEquals(expected[i], res[i], 5e-3f, "RoPE in-place mismatch at " + i);
            }
        }
    }

    @Test
    @DisplayName("Verify zero VRAM leak after operations")
    void testZeroVramLeak() {
        var before = Engine.memoryStats();
        // Allocate and free multiple f16, embedding, and RoPE tensors
        try (Tensor t1 = Tensor.zeros(DType.f16, 128, 128);
             Tensor t2 = t1.cast(DType.f32);
             Tensor t3 = t2.rope(0, 10000.0f)) {
            assertNotNull(t3);
        }
        var after = Engine.memoryStats();
        assertEquals(before.tensorsDevice(), after.tensorsDevice());
        assertEquals(before.tensorsHost(), after.tensorsHost());
        assertEquals(before.tensorsEvicted(), after.tensorsEvicted());
    }

    @Test
    @DisplayName("Nn.kvCacheAppend performs Prefill + Decode transposed writes and bounds validation")
    void testKvCacheAppendPrefillDecodeAndBounds() {
        int numKvHeads = 2;
        int maxSeqLen = 6;
        int headDim = 4;

        var statsBefore = Engine.memoryStats();
        try (Tensor cache = Tensor.full(-777.0f, numKvHeads, maxSeqLen, headDim)) {
            int prefillSeq = 3;
            float[] prefillData = new float[prefillSeq * numKvHeads * headDim];
            for (int i = 0; i < prefillData.length; i++) {
                prefillData[i] = i + 1.0f;
            }
            try (Tensor prefillK = Tensor.fromArray(prefillData, prefillSeq, numKvHeads, headDim)) {
                Nn.kvCacheAppend(cache, prefillK, 0);
            }

            int decodeSeq = 1;
            float[] decodeData = new float[decodeSeq * numKvHeads * headDim];
            for (int i = 0; i < decodeData.length; i++) {
                decodeData[i] = 500.0f + i;
            }
            try (Tensor decodeK = Tensor.fromArray(decodeData, decodeSeq, numKvHeads, headDim)) {
                Nn.kvCacheAppend(cache, decodeK, 3);

                // Bounds validation
                assertThrows(IllegalArgumentException.class, () -> Nn.kvCacheAppend(cache, decodeK, 6));
                assertThrows(IllegalArgumentException.class, () -> Nn.kvCacheAppend(cache, decodeK, -1));
            }

            // Shape validation
            try (Tensor badHeads = Tensor.zeros(DType.f32, 1, numKvHeads + 1, headDim)) {
                assertThrows(IllegalArgumentException.class, () -> Nn.kvCacheAppend(cache, badHeads, 0));
            }

            // Verify exact transposed contents without corrupting untouched positions
            float[] got = cache.toFloatArray();
            for (int h = 0; h < numKvHeads; h++) {
                for (int pos = 0; pos < maxSeqLen; pos++) {
                    for (int d = 0; d < headDim; d++) {
                        int cacheIdx = (h * maxSeqLen + pos) * headDim + d;
                        float expected;
                        if (pos < 3) {
                            expected = prefillData[(pos * numKvHeads + h) * headDim + d];
                        } else if (pos == 3) {
                            expected = decodeData[h * headDim + d];
                        } else {
                            expected = -777.0f;
                        }
                        assertEquals(expected, got[cacheIdx], 1e-6f, "Mismatch at h=" + h + " pos=" + pos + " d=" + d);
                    }
                }
            }
        }
        var statsAfter = Engine.memoryStats();
        assertEquals(statsBefore.tensorsDevice(), statsAfter.tensorsDevice());
    }
}
