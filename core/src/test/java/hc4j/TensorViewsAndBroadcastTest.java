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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import hc4j.engine.GpuBatch;
import hc4j.testutil.GpuTestSupport;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Tensor Views, Ref-Counting, Broadcasting, In-Place Ops, GpuScope & Batch Tests")
class TensorViewsAndBroadcastTest {

    @BeforeAll
    static void setUp() {
        GpuTestSupport.assumeGpuAvailable();
    }

    // ------------------------------------------------------------------------------------------
    // 1. Ref-Counted VramHandle & View Lifecycles
    // ------------------------------------------------------------------------------------------

    @Test
    @DisplayName("Closing parent tensor does not invalidate child view")
    void testParentCloseDoesNotInvalidateView() {
        Tensor view;
        try (Tensor parent = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3)) {
            view = parent.T();
            assertFalse(view.isClosed());
        }
        // Parent is closed now; view must still function
        assertFalse(view.isClosed());
        assertArrayEquals(new int[]{3, 2}, view.getShape());
        assertArrayEquals(new float[]{1, 4, 2, 5, 3, 6}, view.toArray(), 1e-5f);
        view.close();
        assertTrue(view.isClosed());
    }

    @Test
    @DisplayName("Closing view does not invalidate parent tensor")
    void testViewCloseDoesNotInvalidateParent() {
        try (Tensor parent = Tensor.fromArray(new float[]{10, 20, 30, 40}, 2, 2)) {
            Tensor view = parent.T();
            view.close();
            assertTrue(view.isClosed());

            // Parent must still be fully open and accessible
            assertFalse(parent.isClosed());
            assertArrayEquals(new float[]{10, 20, 30, 40}, parent.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("Contiguous() retains handle safely even when already contiguous")
    void testContiguousRetentionSafety() {
        try (Tensor original = Tensor.fromArray(new float[]{1, 2, 3, 4}, 2, 2)) {
            assertTrue(original.isContiguous());
            try (Tensor contig = original.contiguous()) {
                assertFalse(contig.isClosed());
                assertArrayEquals(new float[]{1, 2, 3, 4}, contig.toArray(), 1e-5f);
            }
            // After contig is closed in try-with-resources, original MUST NOT be closed!
            assertFalse(original.isClosed());
            assertArrayEquals(new float[]{1, 2, 3, 4}, original.toArray(), 1e-5f);
        }
    }

    // ------------------------------------------------------------------------------------------
    // 2. Zero-Copy Strided Views & Slicing
    // ------------------------------------------------------------------------------------------

    @Test
    @DisplayName("Reshape with -1 dimension inference and flatten")
    void testReshapeAndFlatten() {
        try (Tensor t = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4)) {
            try (Tensor r1 = t.reshape(4, -1)) {
                assertArrayEquals(new int[]{4, 2}, r1.getShape());
                assertArrayEquals(new int[]{2, 1}, r1.getStrides());
                assertArrayEquals(t.toArray(), r1.toArray(), 1e-5f);
            }
            try (Tensor r2 = t.reshape(-1, 8)) {
                assertArrayEquals(new int[]{1, 8}, r2.getShape());
            }
            try (Tensor flat = t.flatten()) {
                assertArrayEquals(new int[]{8}, flat.getShape());
                assertArrayEquals(new int[]{1}, flat.getStrides());
                assertArrayEquals(t.toArray(), flat.toArray(), 1e-5f);
            }
        }
    }

    @Test
    @DisplayName("TransposeView and permute")
    void testTransposeAndPermute() {
        try (Tensor t = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3)) {
            try (Tensor transposed = t.transposeView()) {
                assertArrayEquals(new int[]{3, 2}, transposed.getShape());
                assertArrayEquals(new int[]{1, 3}, transposed.getStrides());
                assertFalse(transposed.isContiguous());
                assertArrayEquals(new float[]{1, 4, 2, 5, 3, 6}, transposed.toArray(), 1e-5f);
            }
            try (Tensor perm = t.permute(0, 1)) {
                assertArrayEquals(new int[]{2, 3}, perm.getShape());
                assertTrue(perm.isContiguous());
            }
        }
    }

    @Test
    @DisplayName("Squeeze and expandDims/unsqueeze")
    void testSqueezeAndExpandDims() {
        try (Tensor t = Tensor.fromArray(new float[]{1, 2, 3}, 1, 3, 1)) {
            try (Tensor sq = t.squeeze()) {
                assertArrayEquals(new int[]{3}, sq.getShape());
                assertArrayEquals(new float[]{1, 2, 3}, sq.toArray(), 1e-5f);
            }
            try (Tensor sq0 = t.squeeze(0)) {
                assertArrayEquals(new int[]{3, 1}, sq0.getShape());
            }
            try (Tensor unsq = t.unsqueeze(1)) {
                assertArrayEquals(new int[]{1, 1, 3, 1}, unsq.getShape());
            }
            try (Tensor exp = t.expandDims(0)) {
                assertArrayEquals(new int[]{1, 1, 3, 1}, exp.getShape());
            }
        }
    }

    @Test
    @DisplayName("BroadcastTo creates zero-strided view")
    void testBroadcastTo() {
        try (Tensor col = Tensor.fromArray(new float[]{10, 20}, 2, 1);
             Tensor b = col.broadcastTo(2, 4)) {

            assertArrayEquals(new int[]{2, 4}, b.getShape());
            assertArrayEquals(new int[]{1, 0}, b.getStrides());
            assertFalse(b.isContiguous());

            float[] expected = {
                10, 10, 10, 10,
                20, 20, 20, 20
            };
            assertArrayEquals(expected, b.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("Slice view with non-zero offset and strided stepping")
    void testSliceViews() {
        try (Tensor mat = Tensor.fromArray(new float[]{
            0,  1,  2,  3,
            10, 11, 12, 13,
            20, 21, 22, 23,
            30, 31, 32, 33
        }, 4, 4)) {

            // Slicing rows 1..3 and cols 1..4 with step 2
            try (Tensor sub = mat.slice(Slice.range(1, 3), Slice.range(1, 4, 2))) {
                assertArrayEquals(new int[]{2, 2}, sub.getShape());
                // Expected rows:
                // row 1: cols 1, 3 -> 11, 13
                // row 2: cols 1, 3 -> 21, 23
                float[] expected = {11, 13, 21, 23};
                assertArrayEquals(expected, sub.toArray(), 1e-5f);
            }

            // Slice with Slice.at(2)
            try (Tensor row2 = mat.slice(Slice.at(2), Slice.all())) {
                assertArrayEquals(new int[]{1, 4}, row2.getShape());
                assertArrayEquals(new float[]{20, 21, 22, 23}, row2.toArray(), 1e-5f);
            }
        }
    }

    @Test
    @DisplayName("Copy produces deep clone in distinct VramHandle")
    void testCopy() {
        try (Tensor orig = Tensor.fromArray(new float[]{1, 2, 3}, 3);
             Tensor cloned = orig.copy()) {

            assertNotEquals(orig, cloned);
            assertArrayEquals(orig.toArray(), cloned.toArray(), 1e-5f);

            // Mutate orig via addi
            orig.addi(10f);
            assertArrayEquals(new float[]{11, 12, 13}, orig.toArray(), 1e-5f);
            // Cloned must remain unchanged
            assertArrayEquals(new float[]{1, 2, 3}, cloned.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("Item and itemInt for scalar / 1-element tensors")
    void testItem() {
        try (Tensor f = Tensor.fromArray(new float[]{42.5f}, 1);
             Tensor i = Tensor.fromArray(new int[]{99}, 1)) {

            assertEquals(42.5f, f.item(), 1e-5f);
            assertEquals(99, i.itemInt());
        }

        try (Tensor multi = Tensor.zeros(DType.f32, 2, 2)) {
            assertThrows(IllegalStateException.class, multi::item);
            assertThrows(IllegalStateException.class, multi::itemInt);
        }
    }

    // ------------------------------------------------------------------------------------------
    // 3. Broadcasting on Arithmetic Ops
    // ------------------------------------------------------------------------------------------

    @Test
    @DisplayName("Broadcasting [4, 1] + [1, 5] -> [4, 5]")
    void testBroadcast2D() {
        float[] aData = {1f, 2f, 3f, 4f};
        float[] bData = {10f, 20f, 30f, 40f, 50f};

        try (Tensor a = Tensor.fromArray(aData, 4, 1);
             Tensor b = Tensor.fromArray(bData, 1, 5);
             Tensor c = a.add(b)) {

            assertArrayEquals(new int[]{4, 5}, c.getShape());
            float[] expected = new float[20];
            for (int r = 0; r < 4; r++) {
                for (int col = 0; col < 5; col++) {
                    expected[r * 5 + col] = aData[r] + bData[col];
                }
            }
            assertArrayEquals(expected, c.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("Broadcasting [2, 3, 4] * [4] -> [2, 3, 4]")
    void testBroadcast3D() {
        int total = 24;
        float[] aData = new float[total];
        for (int i = 0; i < total; i++) {
            aData[i] = i + 1;
        }
        float[] bData = {2f, 3f, 4f, 5f};

        try (Tensor a = Tensor.fromArray(aData, 2, 3, 4);
             Tensor b = Tensor.fromArray(bData, 4);
             Tensor c = a.mul(b)) {

            assertArrayEquals(new int[]{2, 3, 4}, c.getShape());
            float[] expected = new float[total];
            for (int i = 0; i < total; i++) {
                expected[i] = aData[i] * bData[i % 4];
            }
            assertArrayEquals(expected, c.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("Broadcasting with subtract and divide")
    void testBroadcastSubDiv() {
        try (Tensor a = Tensor.fromArray(new float[]{100, 200}, 2, 1);
             Tensor b = Tensor.fromArray(new float[]{1, 2}, 1, 2);
             Tensor sub = a.sub(b);
             Tensor div = a.div(b)) {

            // sub: [[99, 98], [199, 198]]
            assertArrayEquals(new float[]{99, 98, 199, 198}, sub.toArray(), 1e-5f);
            // div: [[100, 50], [200, 100]]
            assertArrayEquals(new float[]{100, 50, 200, 100}, div.toArray(), 1e-5f);
        }
    }

    // ------------------------------------------------------------------------------------------
    // 4. Universal View Safety across Non-Elementwise Ops
    // ------------------------------------------------------------------------------------------

    @Test
    @DisplayName("Matmul with transposed views")
    void testMatmulTransposedViews() {
        // A: 2x3, B: 3x2 -> A x B = 2x2
        // We will construct B by taking transpose of B_orig (2x3)
        try (Tensor a = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
             Tensor bOrig = Tensor.fromArray(new float[]{7, 8, 9, 10, 11, 12}, 2, 3);
             Tensor bT = bOrig.T(); // Shape 3x2
             Tensor c = a.matmul(bT)) {

            assertArrayEquals(new int[]{2, 2}, c.getShape());
            // row 0 of A (1,2,3) . col 0 of bT (row 0 of bOrig: 7,8,9) = 1*7 + 2*8 + 3*9 = 7 + 16 + 27 = 50
            // row 0 of A (1,2,3) . col 1 of bT (row 1 of bOrig: 10,11,12) = 1*10 + 2*11 + 3*12 = 10 + 22 + 36 = 68
            // row 1 of A (4,5,6) . col 0 of bT (7,8,9) = 4*7 + 5*8 + 6*9 = 28 + 40 + 54 = 122
            // row 1 of A (4,5,6) . col 1 of bT (10,11,12) = 4*10 + 5*11 + 6*12 = 40 + 55 + 72 = 167
            float[] expected = {50f, 68f, 122f, 167f};
            assertArrayEquals(expected, c.toArray(), 1e-4f);
        }
    }

    @Test
    @DisplayName("Unary and NN ops with non-contiguous views")
    void testUnaryAndNnOpsWithViews() {
        try (Tensor a = Tensor.fromArray(new float[]{0f, 1f, 2f, 3f}, 2, 2);
             Tensor aT = a.T();
             Tensor sinT = aT.sin();
             Tensor reluT = aT.relu()) {

            assertArrayEquals(new int[]{2, 2}, sinT.getShape());
            float[] expectedSin = {
                (float) Math.sin(0), (float) Math.sin(2),
                (float) Math.sin(1), (float) Math.sin(3)
            };
            assertArrayEquals(expectedSin, sinT.toArray(), 1e-5f);

            float[] expectedRelu = {0f, 2f, 1f, 3f};
            assertArrayEquals(expectedRelu, reluT.toArray(), 1e-5f);
        }
    }

    // ------------------------------------------------------------------------------------------
    // 5. In-Place Ops and Aliasing
    // ------------------------------------------------------------------------------------------

    @Test
    @DisplayName("In-place operations with tensors and scalars")
    void testInPlaceOps() {
        try (Tensor a = Tensor.fromArray(new float[]{1, 2, 3, 4}, 4);
             Tensor b = Tensor.fromArray(new float[]{10, 20, 30, 40}, 4)) {

            a.addi(b);
            assertArrayEquals(new float[]{11, 22, 33, 44}, a.toArray(), 1e-5f);

            a.subi(1f);
            assertArrayEquals(new float[]{10, 21, 32, 43}, a.toArray(), 1e-5f);

            a.muli(2f);
            assertArrayEquals(new float[]{20, 42, 64, 86}, a.toArray(), 1e-5f);

            a.divi(2f);
            assertArrayEquals(new float[]{10, 21, 32, 43}, a.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("In-place self-aliasing: a.addi(a) and a.muli(a)")
    void testInPlaceSelfAliasing() {
        try (Tensor a = Tensor.fromArray(new float[]{2, 3, 4, 5}, 4)) {
            a.addi(a);
            assertArrayEquals(new float[]{4, 6, 8, 10}, a.toArray(), 1e-5f);

            a.muli(a);
            assertArrayEquals(new float[]{16, 36, 64, 100}, a.toArray(), 1e-5f);
        }
    }

    // ------------------------------------------------------------------------------------------
    // 6. Reverse Scalar Ops
    // ------------------------------------------------------------------------------------------

    @Test
    @DisplayName("Reverse scalar ops: rsub and rdiv")
    void testReverseScalarOps() {
        try (Tensor x = Tensor.fromArray(new float[]{2, 4, 8}, 3);
             Tensor rsub = x.rsub(10f);
             Tensor rdiv = x.rdiv(16f);
             Tensor rsubHc4j = HC4J.rsub(10f, x);
             Tensor rdivHc4j = HC4J.rdiv(16f, x)) {

            // 10 - [2, 4, 8] = [8, 6, 2]
            assertArrayEquals(new float[]{8, 6, 2}, rsub.toArray(), 1e-5f);
            assertArrayEquals(new float[]{8, 6, 2}, rsubHc4j.toArray(), 1e-5f);

            // 16 / [2, 4, 8] = [8, 4, 2]
            assertArrayEquals(new float[]{8, 4, 2}, rdiv.toArray(), 1e-5f);
            assertArrayEquals(new float[]{8, 4, 2}, rdivHc4j.toArray(), 1e-5f);
        }

        try (Tensor xI32 = Tensor.fromArray(new int[]{2, 5}, 2);
             Tensor rsubI32 = xI32.rsub(10);
             Tensor rdivI32 = xI32.rdiv(20);
             Tensor rsubHc4jI32 = HC4J.rsub(10, xI32);
             Tensor rdivHc4jI32 = HC4J.rdiv(20, xI32)) {

            assertArrayEquals(new int[]{8, 5}, rsubI32.toArrayInt());
            assertArrayEquals(new int[]{8, 5}, rsubHc4jI32.toArrayInt());
            assertArrayEquals(new int[]{10, 4}, rdivI32.toArrayInt());
            assertArrayEquals(new int[]{10, 4}, rdivHc4jI32.toArrayInt());
        }
    }

    // ------------------------------------------------------------------------------------------
    // 7. JNum-Matching Factories
    // ------------------------------------------------------------------------------------------

    @Test
    @DisplayName("JNum factories: ones, full, arange, linspace, eye")
    void testJNumFactories() {
        try (Tensor ones = Tensor.ones(DType.f32, 2, 3)) {
            assertArrayEquals(new float[]{1, 1, 1, 1, 1, 1}, ones.toArray(), 1e-5f);
        }

        try (Tensor full = Tensor.full(7.5f, 2, 2)) {
            assertArrayEquals(new float[]{7.5f, 7.5f, 7.5f, 7.5f}, full.toArray(), 1e-5f);
        }

        try (Tensor fullI32 = Tensor.full(42, 3)) {
            assertArrayEquals(new int[]{42, 42, 42}, fullI32.toArrayInt());
        }

        try (Tensor arangeI = Tensor.arange(1, 6, 2)) {
            assertArrayEquals(new int[]{1, 3, 5}, arangeI.toArrayInt());
        }

        try (Tensor arangeF = Tensor.arange(0f, 1f, 0.25f)) {
            assertArrayEquals(new float[]{0f, 0.25f, 0.5f, 0.7f}, arangeF.toArray(), 0.1f);
        }

        try (Tensor ls = Tensor.linspace(0f, 10f, 5)) {
            assertArrayEquals(new float[]{0f, 2.5f, 5.0f, 7.5f, 10.0f}, ls.toArray(), 1e-5f);
        }

        try (Tensor eye = Tensor.eye(3)) {
            float[] expectedEye = {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1
            };
            assertArrayEquals(expectedEye, eye.toArray(), 1e-5f);
        }
    }

    // ------------------------------------------------------------------------------------------
    // 8. GpuScope Real GPU Workflow
    // ------------------------------------------------------------------------------------------

    @Test
    @DisplayName("GpuScope auto-closes intermediates and retains detached output")
    void testGpuScopeWorkflow() {
        Tensor output;
        try (GpuScope scope = GpuScope.open()) {
            Tensor a = scope.full(2.0f, 4);
            Tensor b = scope.full(3.0f, 4);
            Tensor c = scope.track(a.add(b)); // [5, 5, 5, 5]
            Tensor d = scope.track(c.mul(2.0f)); // [10, 10, 10, 10]

            assertEquals(4, scope.trackedCount());
            output = scope.detach(d);
            assertEquals(3, scope.trackedCount());
        }

        // Scope closed: intermediates freed, output remains valid!
        assertFalse(output.isClosed());
        assertArrayEquals(new float[]{10, 10, 10, 10}, output.toArray(), 1e-5f);
        output.close();
        assertTrue(output.isClosed());
    }

    // ------------------------------------------------------------------------------------------
    // 9. Thread-Safe Reentrant GpuBatch
    // ------------------------------------------------------------------------------------------

    @Test
    @DisplayName("Reentrant nested GpuBatch execution")
    @SuppressWarnings("try")
    void testNestedGpuBatch() {
        try (GpuBatch b1 = GpuBatch.begin()) {
            try (Tensor t1 = Tensor.ones(DType.f32, 4)) {
                try (GpuBatch b2 = GpuBatch.begin()) {
                    try (Tensor t2 = t1.add(5f)) {
                        assertArrayEquals(new float[]{6, 6, 6, 6}, t2.toArray(), 1e-5f);
                    }
                }
            }
        }
    }

    @Test
    @DisplayName("Concurrent multi-threaded GpuBatch execution")
    @SuppressWarnings("try")
    void testConcurrentGpuBatch() throws Exception {
        int threads = 4;
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        try {
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int i = 0; i < threads; i++) {
                final float val = i + 1;
                tasks.add(() -> {
                    for (int step = 0; step < 10; step++) {
                        try (GpuBatch batch = GpuBatch.begin();
                             Tensor t = Tensor.full(val, 8);
                             Tensor res = t.mul(2f)) {
                            float[] data = res.toArray();
                            assertEquals(8, data.length);
                            assertEquals(val * 2f, data[0], 1e-5f);
                        }
                    }
                    return null;
                });
            }
            List<Future<Void>> futures = executor.invokeAll(tasks);
            for (Future<Void> f : futures) {
                f.get();
            }
        } finally {
            executor.shutdown();
        }
    }
}
