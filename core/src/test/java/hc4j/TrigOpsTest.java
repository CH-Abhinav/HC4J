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
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import hc4j.engine.Engine;
import hc4j.engine.MemoryStats;
import hc4j.engine.Residency;
import hc4j.internal.ffi.TensorAccess;
import hc4j.testutil.GpuTestSupport;
import hc4j.testutil.TestArrayFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.function.UnaryOperator;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Trigonometric Operations & Tiered Memory Tests")
class TrigOpsTest {

    private static final long MiB = 1L << 20;
    private static final double WGSL_TOL = 1.0 / 2048;

    @BeforeEach
    void setUp() {
        GpuTestSupport.assumeGpuAvailable();
    }

    private record Case(String name, UnaryOperator<Tensor> gpu, DoubleUnaryOperator cpu, float lo, float hi) {}

    @Test
    @DisplayName("Trigonometric suite vs StrictMath (n = 100,003: exercises tail)")
    void testTrigonometricSuite() {
        List<Case> cases = List.of(
                new Case("sin", Tensor::sin, StrictMath::sin, -3f, 3f),
                new Case("cos", Tensor::cos, StrictMath::cos, -3f, 3f),
                new Case("tan", Tensor::tan, StrictMath::tan, -1.2f, 1.2f),
                new Case("asin", Tensor::asin, StrictMath::asin, -0.95f, 0.95f),
                new Case("acos", Tensor::acos, StrictMath::acos, -0.95f, 0.95f),
                new Case("atan", Tensor::atan, StrictMath::atan, -5f, 5f),
                new Case("sinh", Tensor::sinh, StrictMath::sinh, -5f, 5f),
                new Case("cosh", Tensor::cosh, StrictMath::cosh, -5f, 5f),
                new Case("tanh", Tensor::tanh, StrictMath::tanh, -5f, 5f),
                new Case("asinh", Tensor::asinh, x -> Math.log(x + Math.sqrt(x * x + 1)), -3f, 3f),
                new Case("acosh", Tensor::acosh, x -> Math.log(x + Math.sqrt(x * x - 1)), 1.05f, 10f),
                new Case("atanh", Tensor::atanh, x -> 0.5 * Math.log((1 + x) / (1 - x)), -0.95f, 0.95f));

        int n = 100_003;
        TensorAccess access = TensorAccess.getInstance();
        for (Case c : cases) {
            float[] host = TestArrayFactory.ramp(n, c.lo, c.hi);
            try (Tensor input = Tensor.fromArray(host, n);
                 Tensor out = c.gpu.apply(input)) {
                assertNotEquals(access.getVramId(input), access.getVramId(out), c.name + " allocates a new tensor");
                assertArrayEquals(input.getShape(), out.getShape(), c.name + " preserves shape");
                double err = maxRelError(out.toFloatArray(), host, c.cpu);
                assertTrue(err < WGSL_TOL, () -> String.format("%s max rel err %e exceeds tolerance %e", c.name, err, WGSL_TOL));
                assertArrayEquals(host, input.toFloatArray(), c.name + " leaves input untouched");
            }
        }

        // Test HC4J facade delegates
        float[] host = TestArrayFactory.ramp(1024, -1f, 1f);
        try (Tensor input = HC4J.fromArray(host, 1024);
             Tensor sinOut = HC4J.sin(input);
             Tensor asinOut = HC4J.asin(sinOut)) {
            double err = maxRelError(asinOut.toFloatArray(), host, v -> v);
            assertTrue(err < WGSL_TOL, "HC4J.asin(HC4J.sin(x)) == x");
        }

        // 3-D tensor shape round-trips
        try (Tensor m = Tensor.fromArray(TestArrayFactory.ramp(6 * 7 * 5, -2f, 2f), 6, 7, 5);
             Tensor c = m.cos()) {
            assertEquals(3, c.dim());
            assertEquals(210, c.getSize());
        }
    }

    @Test
    @DisplayName("Arithmetic ops on tiered manager")
    void testArithmeticStillWorks() {
        float[] a = TestArrayFactory.ramp(10_000, 0f, 1f);
        float[] b = TestArrayFactory.ramp(10_000, 1f, 2f);
        try (Tensor ta = Tensor.fromArray(a, 10_000);
             Tensor tb = Tensor.fromArray(b, 10_000);
             Tensor sum = ta.add(tb);
             Tensor self = ta.mul(ta)) {
            float[] s = sum.toFloatArray();
            float[] sq = self.toFloatArray();
            for (int i = 0; i < a.length; i++) {
                assertTrue(Math.abs(s[i] - (a[i] + b[i])) < 1e-6f);
                assertTrue(Math.abs(sq[i] - a[i] * a[i]) < 1e-6f);
            }
        }
    }

    @Test
    @DisplayName("Error paths surface as proper Java exceptions")
    void testErrorPaths() {
        Tensor closed = Tensor.fromArray(new float[] {1f, 2f, 3f}, 3);
        closed.close();
        assertThrowsWithMessage(IllegalStateException.class, closed::sin, "Tensor has already been closed");

        try (Tensor ints = Tensor.fromArray(new int[] {1, 2, 3}, 3)) {
            assertThrowsWithMessage(IllegalArgumentException.class, ints::cos, "cos is only defined for f32 tensors, got i32");
        }
    }

    @Test
    @DisplayName("Tiered memory: LRU eviction, disk spill, page-in and streaming")
    void testTieredMemory() {
        MemoryStats before = Engine.memoryStats();
        Engine.configureMemory(64 * MiB, 48 * MiB);

        int n = (int) (16 * MiB / Float.BYTES);
        List<Tensor> tensors = new ArrayList<>();
        List<float[]> expected = new ArrayList<>();
        try {
            for (int i = 0; i < 8; i++) { // 128 MiB of tensors against a 64 MiB VRAM budget
                float[] data = TestArrayFactory.pattern(n, i);
                expected.add(data);
                tensors.add(Tensor.fromArray(data, n));
            }
            MemoryStats s = Engine.memoryStats();
            assertTrue(s.vramUsed() <= 64 * MiB, "VRAM usage within budget");
            assertTrue(s.hostUsed() <= 48 * MiB, "Host usage within budget");
            assertTrue(s.evictions() > before.evictions(), "LRU tensors evicted from VRAM");
            assertTrue(s.spills() > before.spills(), "Host tier overflowed to disk");
            assertNotEquals(Residency.DEVICE, tensors.get(0).residency(), "Oldest tensor evicted");
            assertEquals(Residency.DEVICE, tensors.get(7).residency(), "Newest tensor resident");

            for (int i = 0; i < tensors.size(); i++) {
                assertTrue(Arrays.equals(tensors.get(i).toFloatArray(), expected.get(i)), "Tensor " + i + " intact after migration");
            }

            Tensor oldest = tensors.get(0);
            long pageInsBefore = Engine.memoryStats().pageIns();
            oldest.evict();
            try (Tensor r = oldest.sin()) {
                assertTrue(Engine.memoryStats().pageIns() > pageInsBefore, "sin() paged evicted input back in");
                assertEquals(Residency.DEVICE, oldest.residency(), "Input resident after page-in");
                assertTrue(maxRelError(r.toFloatArray(), expected.get(0), StrictMath::sin) < WGSL_TOL);
            }

            // Streaming (input larger than whole VRAM budget)
            Engine.configureMemory(24 * MiB, 0);
            int big = (int) (32 * MiB / Float.BYTES);
            float[] bigData = TestArrayFactory.ramp(big, -4f, 4f);
            try (Tensor x = Tensor.fromArray(bigData, big)) {
                assertNotEquals(Residency.DEVICE, x.residency(), "32 MiB tensor placed off-device");
                long streamedBefore = Engine.memoryStats().streamedOps();
                try (Tensor r = x.cos()) {
                    assertTrue(Engine.memoryStats().streamedOps() > streamedBefore, "cos() took streaming path");
                    assertTrue(maxRelError(r.toFloatArray(), bigData, StrictMath::cos) < WGSL_TOL);
                }
            }
        } finally {
            Engine.configureMemory(before.vramBudget(), before.hostBudget());
            for (Tensor t : tensors) {
                t.close();
            }
        }

        MemoryStats end = Engine.memoryStats();
        assertEquals(0, end.tensorsDevice() + end.tensorsHost() + end.tensorsEvicted(), "No tensors leaked");
    }

    private static double maxRelError(float[] got, float[] input, DoubleUnaryOperator ref) {
        double worst = 0;
        for (int i = 0; i < got.length; i++) {
            double want = ref.applyAsDouble(input[i]);
            worst = Math.max(worst, Math.abs(got[i] - want) / Math.max(1.0, Math.abs(want)));
        }
        return worst;
    }
}
