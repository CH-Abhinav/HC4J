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
import static org.junit.jupiter.api.Assertions.assertTrue;

import hc4j.nn.Activation;
import hc4j.nn.Nn;
import hc4j.testutil.GpuTestSupport;
import hc4j.testutil.ReferenceOps;
import hc4j.testutil.TestArrayFactory;
import hc4j.testutil.TestAssertions;
import java.util.SplittableRandom;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Neural Network Operations & Reductions Tests")
class NnOpsTest {

    private static final float ALPHA = 0.125f;

    @BeforeEach
    void setUp() {
        GpuTestSupport.assumeGpuAvailable();
    }

    @Test
    @DisplayName("7 activations match CPU reference")
    void testActivations() {
        int n = 1003;
        float[] host = TestArrayFactory.random(1L, n, -5f, 5f);
        host[0] = 0f;
        host[1] = 88f;
        host[2] = -88f;
        host[3] = 1e-8f;

        try (Tensor a = Tensor.fromArray(host, n);
             Tensor out = Tensor.empty(DType.f32, n)) {

            for (Activation act : Activation.values()) {
                hc4j.internal.nn.ActivationOps.activation(act, a, out, ALPHA);
                float[] got = out.toFloatArray();
                for (int i = 0; i < n; i++) {
                    float expected = (float) ReferenceOps.activation(act, host[i], ALPHA);
                    TestAssertions.assertClose(expected, got[i], 1e-4f, 1e-4f);
                }
            }
        }

        // Test Nn facade delegates and in-place
        try (Tensor a = Tensor.fromArray(host, n)) {
            try (Tensor g = Nn.gelu(a)) {
                float[] got = g.toFloatArray();
                for (int i = 0; i < n; i++) {
                    TestAssertions.assertClose((float) ReferenceOps.activation(Activation.GELU, host[i], 0f), got[i], 1e-4f, 1e-4f);
                }
            }

            Nn.relu(a, a); // in place
            float[] reluGot = a.toFloatArray();
            for (int i = 0; i < n; i++) {
                TestAssertions.assertClose((float) ReferenceOps.activation(Activation.RELU, host[i], 0f), reluGot[i], 1e-5f, 1e-5f);
            }
        }
    }

    @Test
    @DisplayName("Full reductions: sum, mean, max, min")
    void testFullReductions() {
        float[] data = {1f, 2f, 3f, 4f, 5f};
        try (Tensor a = Tensor.fromArray(data, 5)) {
            TestAssertions.assertClose(15f, a.sum());
            TestAssertions.assertClose(3f, a.mean());
            TestAssertions.assertClose(5f, a.maxValue());
            TestAssertions.assertClose(1f, a.minValue());

            try (Tensor s = Nn.sum(a);
                 Tensor m = Nn.mean(a);
                 Tensor max = Nn.maxValue(a);
                 Tensor min = Nn.minValue(a)) {
                TestAssertions.assertClose(15f, s.toFloatArray()[0]);
                TestAssertions.assertClose(3f, m.toFloatArray()[0]);
                TestAssertions.assertClose(5f, max.toFloatArray()[0]);
                TestAssertions.assertClose(1f, min.toFloatArray()[0]);
            }
        }
    }

    @Test
    @DisplayName("Row reductions along last axis")
    void testRowReductions() {
        float[] data = {
                1f, 2f, 3f,
                4f, 5f, 6f
        };
        try (Tensor a = Tensor.fromArray(data, 2, 3);
             Tensor s = Nn.sumRows(a);
             Tensor m = Nn.meanRows(a);
             Tensor max = Nn.maxRows(a);
             Tensor min = Nn.minRows(a);
             Tensor argmax = Nn.argmaxRows(a);
             Tensor argmin = Nn.argminRows(a)) {

            assertArrayEquals(new int[] {2}, s.getShape());
            TestAssertions.assertArrayClose(new float[] {6f, 15f}, s.toFloatArray());
            TestAssertions.assertArrayClose(new float[] {2f, 5f}, m.toFloatArray());
            TestAssertions.assertArrayClose(new float[] {3f, 6f}, max.toFloatArray());
            TestAssertions.assertArrayClose(new float[] {1f, 4f}, min.toFloatArray());

            assertArrayEquals(new int[] {2, 2}, argmax.toIntArray());
            assertArrayEquals(new int[] {0, 0}, argmin.toIntArray());
        }
    }

    @Test
    @DisplayName("Softmax and LogSoftmax")
    void testSoftmax() {
        float[] data = {1f, 2f, 3f, 4f, 5f, 6f};
        try (Tensor a = Tensor.fromArray(data, 2, 3);
             Tensor sm = Nn.softmax(a);
             Tensor lsm = Nn.logSoftmax(a)) {

            float[] expectedSm = ReferenceOps.softmax(data, 2, 3, false);
            float[] expectedLsm = ReferenceOps.softmax(data, 2, 3, true);

            TestAssertions.assertArrayClose(expectedSm, sm.toFloatArray(), 1e-4f, 1e-4f);
            TestAssertions.assertArrayClose(expectedLsm, lsm.toFloatArray(), 1e-4f, 1e-4f);
        }
    }

    @Test
    @DisplayName("LayerNorm and RMSNorm")
    void testNormalizations() {
        float[] data = TestArrayFactory.ramp(12, -2f, 2f);
        float[] gamma = {1f, 1f, 1f, 1f};
        float[] beta = {0f, 0f, 0f, 0f};

        try (Tensor a = Tensor.fromArray(data, 3, 4);
             Tensor g = Tensor.fromArray(gamma, 4);
             Tensor b = Tensor.fromArray(beta, 4);
             Tensor ln = Nn.layerNorm(a, g, b, 1e-5f);
             Tensor rn = Nn.rmsNorm(a, g, 1e-5f)) {

            float[] expectedLn = ReferenceOps.layerNorm(data, gamma, beta, 3, 4, 1e-5f);
            float[] expectedRn = ReferenceOps.rmsNorm(data, gamma, 3, 4, 1e-5f);

            TestAssertions.assertArrayClose(expectedLn, ln.toFloatArray(), 1e-4f, 1e-4f);
            TestAssertions.assertArrayClose(expectedRn, rn.toFloatArray(), 1e-4f, 1e-4f);
        }
    }

    @Test
    @DisplayName("CrossEntropy loss")
    void testCrossEntropy() {
        float[] logits = {
                2.0f, 1.0f, 0.1f,
                0.2f, 3.0f, 1.5f
        };
        int[] targets = {0, 1};

        try (Tensor l = Tensor.fromArray(logits, 2, 3);
             Tensor t = Tensor.fromArray(targets, 2);
             Tensor loss = Nn.crossEntropy(l, t)) {

            assertArrayEquals(new int[] {2}, loss.getShape());
            float[] expected = ReferenceOps.crossEntropy(logits, targets, 2, 3);
            TestAssertions.assertArrayClose(expected, loss.toFloatArray(), 1e-4f, 1e-4f);
        }
    }

    @Test
    @DisplayName("Linear feed-forward layer with bias")
    void testLinearLayer() {
        float[] x = {1f, 2f, 3f, 4f};
        float[] w = {
                1f, 0f,
                0f, 1f,
                1f, 1f,
                0f, 0f
        };
        float[] b = {0.5f, -0.5f};

        try (Tensor xt = Tensor.fromArray(x, 1, 4);
             Tensor wt = Tensor.fromArray(w, 4, 2);
             Tensor bt = Tensor.fromArray(b, 2);
             Tensor out = Nn.linear(xt, wt, bt)) {

            assertArrayEquals(new int[] {1, 2}, out.getShape());
            // [1, 2, 3, 4] x W = [1*1 + 3*1, 2*1 + 3*1] = [4, 5] + [0.5, -0.5] = [4.5, 4.5]
            TestAssertions.assertArrayClose(new float[] {4.5f, 4.5f}, out.toFloatArray());
        }
    }

    @Test
    @DisplayName("Validation and error paths in NnOps")
    void testValidation() {
        try (Tensor a = Tensor.empty(DType.i32, 4)) {
            assertThrowsWithMessage(IllegalArgumentException.class,
                    () -> Nn.relu(a),
                    "RELU is only defined for f32 tensors, got i32");
        }

        try (Tensor a = Tensor.empty(DType.f32, 2, 3);
             Tensor g = Tensor.empty(DType.f32, 4)) { // cols is 3, gamma is 4
            assertThrowsWithMessage(IllegalArgumentException.class,
                    () -> Nn.layerNorm(a, g, null, 1e-5f),
                    "layerNorm gamma must be 1-D of length 3");
        }
    }
}
