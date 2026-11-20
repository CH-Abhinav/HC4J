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
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import hc4j.engine.Engine;
import hc4j.engine.EngineStats;
import hc4j.engine.GpuBatch;
import hc4j.engine.MemoryStats;
import hc4j.linalg.Linalg;
import hc4j.testutil.GpuTestSupport;
import hc4j.testutil.ReferenceOps;
import hc4j.testutil.TestArrayFactory;
import hc4j.testutil.TestAssertions;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Engine Operations, Matmul, Fusion & Batching Tests")
class EngineOpsTest {

    private static final double WGSL_TOL = 1.0 / 2048;

    @BeforeEach
    void setUp() {
        GpuTestSupport.assumeGpuAvailable();
    }

    @Test
    @DisplayName("Caller-allocated and in-place elementwise ops")
    void testElementwiseCallerAllocated() {
        int n = 10_003;
        float[] host = TestArrayFactory.ramp(n, 0.1f, 3f);
        try (Tensor a = Tensor.fromArray(host, n);
             Tensor b = Tensor.fromArray(host, n);
             Tensor res = Tensor.empty(DType.f32, n)) {

            assertSame(res, a.sin(res));
            float[] r = res.toFloatArray();
            for (int i = 0; i < n; i++) {
                assertTrue(Math.abs(r[i] - (float) Math.sin(host[i])) < WGSL_TOL);
            }

            // In-place
            a.exp(a);
            float[] ea = a.toFloatArray();
            for (int i = 0; i < n; i++) {
                TestAssertions.assertClose((float) Math.exp(host[i]), ea[i], 1e-4f, 1e-4f);
            }

            a.log(a);
            float[] la = a.toFloatArray();
            for (int i = 0; i < n; i++) {
                TestAssertions.assertClose(host[i], la[i], 1e-4f, 1e-4f);
            }

            // Binary in-place: a = a + b
            a.add(b, a);
            float[] aa = a.toFloatArray();
            for (int i = 0; i < n; i++) {
                TestAssertions.assertClose(2f * host[i], aa[i], 1e-4f, 1e-4f);
            }

            // Sqrt, Log2, Log10
            try (Tensor sq = b.sqrt(); Tensor l2 = b.log2(); Tensor l10 = b.log10()) {
                float[] sqArr = sq.toFloatArray();
                float[] l2Arr = l2.toFloatArray();
                float[] l10Arr = l10.toFloatArray();
                for (int i = 0; i < n; i++) {
                    TestAssertions.assertClose((float) Math.sqrt(host[i]), sqArr[i], 1e-4f, 1e-4f);
                    TestAssertions.assertClose((float) (Math.log(host[i]) / Math.log(2)), l2Arr[i], 1e-4f, 1e-4f);
                    TestAssertions.assertClose((float) Math.log10(host[i]), l10Arr[i], 1e-4f, 1e-4f);
                }
            }
        }

        try (Tensor x = Tensor.fromArray(new float[] {1, 2, 3}, 3);
             Tensor wrong = Tensor.empty(DType.f32, 4)) {
            assertThrowsWithMessage(IllegalArgumentException.class,
                    () -> x.sin(wrong),
                    "sin: shape mismatch [3] vs [4]");
        }
    }

    @Test
    @DisplayName("Engine stats and observability")
    void testObservability() {
        EngineStats engine = Engine.engineStats();
        assertNotNull(engine);
        assertNotNull(engine.backendName());
        assertNotNull(engine.deviceTypeName());

        long submissionsBefore = engine.submissions();
        float[] data = {1f, 2f, 3f};
        try (Tensor t = Tensor.fromArray(data, 3);
             Tensor s = t.sin()) {
            s.toFloatArray();
        }
        EngineStats engineAfter = Engine.engineStats();
        assertTrue(engineAfter.submissions() >= submissionsBefore);
    }

    @Test
    @DisplayName("Matmul plans via Linalg facade")
    void testMatmulPlans() {
        Linalg.Plan planSquare = Linalg.matmulPlan(256, 256, 256);
        assertNotNull(planSquare.kernel());

        Linalg.Plan planTransposed = Linalg.matmulPlan(256, 256, 256, true, true);
        assertTrue(planTransposed.pretransposeA());
        assertTrue(planTransposed.pretransposeB());
    }

    @Test
    @DisplayName("Matmul correctness vs CPU reference across shapes and transpositions")
    void testMatmulCorrectness() {
        int[][] shapes = {
                {64, 64, 64}, {128, 64, 96}, {67, 33, 129}, {2, 3, 4}, {30, 1, 50}, {1, 51, 100}
        };
        SplittableRandom rng = new SplittableRandom(42);
        for (int[] s : shapes) {
            int m = s[0], n = s[1], k = s[2];
            for (int flags = 0; flags < 4; flags++) {
                boolean ta = (flags & 1) != 0, tb = (flags & 2) != 0;
                if ((m == 1 && ta) || (n == 1 && tb)) continue;

                float[] a = TestArrayFactory.random(rng.nextLong(), m * k, -1f, 1f);
                float[] b = TestArrayFactory.random(rng.nextLong(), k * n, -1f, 1f);
                int[] aShape = ta ? new int[] {k, m} : new int[] {m, k};
                int[] bShape = tb ? new int[] {n, k} : new int[] {k, n};

                try (Tensor taTensor = Tensor.fromArray(a, aShape);
                     Tensor tbTensor = Tensor.fromArray(b, bShape);
                     Tensor c = Tensor.empty(DType.f32, m, n)) {

                    hc4j.internal.ops.MatmulOps.matmul(taTensor, tbTensor, c, ta, tb);
                    float[] expected = ReferenceOps.matmul(a, b, m, n, k, ta, tb);
                    TestAssertions.assertArrayClose(expected, c.toFloatArray(), 1e-4f, 1e-4f);
                }
            }
        }

        // Linalg facade and NumPy 1-D semantics
        float[] av = TestArrayFactory.random(11L, 50 * 40, -1f, 1f);
        float[] bv = TestArrayFactory.random(22L, 40, -1f, 1f);
        try (Tensor A = Tensor.fromArray(av, 50, 40);
             Tensor x = Tensor.fromArray(bv, 40);
             Tensor y = Linalg.matmul(A, x);
             Tensor dot = Linalg.matmul(x, x)) {

            assertArrayEquals(new int[] {50}, y.getShape());
            float[] expectedY = ReferenceOps.matmul(av, bv, 50, 1, 40, false, false);
            TestAssertions.assertArrayClose(expectedY, y.toFloatArray(), 1e-4f, 1e-4f);

            assertArrayEquals(new int[] {1}, dot.getShape());
            float[] expectedDot = ReferenceOps.matmul(bv, bv, 1, 1, 40, false, false);
            TestAssertions.assertArrayClose(expectedDot, dot.toFloatArray(), 1e-4f, 1e-4f);
        }
    }

    @Test
    @DisplayName("Matmul validation errors")
    void testMatmulValidation() {
        try (Tensor a = Tensor.empty(DType.f32, 2, 3);
             Tensor b = Tensor.empty(DType.f32, 4, 5);
             Tensor res = Tensor.empty(DType.f32, 2, 5)) {
            assertThrowsWithMessage(IllegalArgumentException.class,
                    () -> Linalg.matmul(a, b),
                    "Matmul shape mismatch: left shape [2, 3] and right shape [4, 5] are incompatible because 3 != 4");
        }

        try (Tensor a = Tensor.empty(DType.f32, 2, 3);
             Tensor b = Tensor.empty(DType.f32, 3, 4);
             Tensor wrong = Tensor.empty(DType.f32, 2, 5)) {
            assertThrowsWithMessage(IllegalArgumentException.class,
                    () -> Linalg.matmul(a, b, wrong),
                    "Result shape must be [2, 4] but was [2, 5]");
        }

        try (Tensor a = Tensor.empty(DType.f32, 2, 2);
             Tensor b = Tensor.empty(DType.f32, 2, 2)) {
            assertThrowsWithMessage(IllegalArgumentException.class,
                    () -> Linalg.matmul(a, b, a),
                    "matmul result must not alias an operand");
        }
    }

    @Test
    @DisplayName("In-VRAM transpose kernel")
    void testTranspose() {
        float[] data = TestArrayFactory.ramp(12, 1f, 12f);
        try (Tensor a = Tensor.fromArray(data, 3, 4);
             Tensor at = Linalg.transpose(a)) {
            assertArrayEquals(new int[] {4, 3}, at.getShape());
            float[] expected = ReferenceOps.transpose(data, 3, 4);
            TestAssertions.assertArrayClose(expected, at.toFloatArray());
        }
    }

    @Test
    @DisplayName("Kernel fusion")
    void testFusion() {
        int n = 1024;
        float[] ha = TestArrayFactory.ramp(n, 0f, 1f);
        float[] hb = TestArrayFactory.ramp(n, 1f, 2f);
        float[] hc = TestArrayFactory.ramp(n, 2f, 3f);

        try (Tensor a = Tensor.fromArray(ha, n);
             Tensor b = Tensor.fromArray(hb, n);
             Tensor c = Tensor.fromArray(hc, n);
             Tensor r = a.lazy().sin().add(b).mul(c).eval()) {

            float[] got = r.toFloatArray();
            for (int i = 0; i < n; i++) {
                float want = ((float) Math.sin(ha[i]) + hb[i]) * hc[i];
                TestAssertions.assertClose(want, got[i], 1e-4f, 1e-4f);
            }
        }
    }

    @Test
    @DisplayName("Command batching with GpuBatch")
    void testBatching() {
        int n = 256;
        float[] data = TestArrayFactory.ramp(n, -1f, 1f);
        try (Tensor a = Tensor.fromArray(data, n)) {
            try (GpuBatch batch = Engine.openBatch()) {
                for (int i = 0; i < 10; i++) {
                    a.sin(a);
                }
            }
            Engine.synchronize();
            assertTrue(a.toFloatArray().length == n);
        }
    }

    @Test
    @DisplayName("Leak check on allocations")
    void testLeakCheck() {
        MemoryStats stats = Engine.memoryStats();
        assertEquals(0, stats.tensorsDevice() + stats.tensorsHost() + stats.tensorsEvicted());
    }
}
