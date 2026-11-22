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
import static org.junit.jupiter.api.Assertions.assertThrows;

import hc4j.testutil.GpuTestSupport;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Panama MemorySegment Zero-Copy I/O Tests")
class MemorySegmentIoTest {

    @BeforeAll
    static void setUp() {
        GpuTestSupport.assumeGpuAvailable();
    }

    @Test
    @DisplayName("Zero-copy round trip with native 64-byte aligned off-heap segment (f32)")
    void testNativeSegmentRoundTripF32() {
        try (Arena arena = Arena.ofConfined()) {
            int n = 8;
            MemorySegment src = arena.allocate(n * 4L, 64);
            for (int i = 0; i < n; i++) {
                src.setAtIndex(ValueLayout.JAVA_FLOAT, i, (i + 1) * 1.5f);
            }

            try (Tensor t = Tensor.fromSegment(src, DType.f32, 2, 4);
                 Tensor doubled = t.mul(2.0f)) {

                MemorySegment dst = arena.allocate(n * 4L, 64);
                doubled.copyTo(dst);

                for (int i = 0; i < n; i++) {
                    float expected = (i + 1) * 3.0f;
                    float actual = dst.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    assertEquals(expected, actual, 1e-5f);
                }
            }
        }
    }

    @Test
    @DisplayName("Zero-copy round trip with native 64-byte aligned off-heap segment (i32)")
    void testNativeSegmentRoundTripI32() {
        try (Arena arena = Arena.ofConfined()) {
            int n = 6;
            MemorySegment src = arena.allocate(n * 4L, 64);
            for (int i = 0; i < n; i++) {
                src.setAtIndex(ValueLayout.JAVA_INT, i, (i + 1) * 10);
            }

            try (Tensor t = Tensor.fromSegment(src, DType.i32, 2, 3);
                 Tensor shifted = t.add(5)) {

                MemorySegment dst = arena.allocate(n * 4L, 64);
                shifted.copyTo(dst);

                for (int i = 0; i < n; i++) {
                    int expected = (i + 1) * 10 + 5;
                    int actual = dst.getAtIndex(ValueLayout.JAVA_INT, i);
                    assertEquals(expected, actual);
                }
            }
        }
    }

    @Test
    @DisplayName("On-heap fallback memory segment I/O")
    void testHeapSegmentFallback() {
        float[] heapIn = {1.0f, 2.0f, 3.0f, 4.0f};
        MemorySegment src = MemorySegment.ofArray(heapIn);

        try (Tensor t = Tensor.fromSegment(src, DType.f32, 2, 2)) {
            float[] heapOut = new float[4];
            MemorySegment dst = MemorySegment.ofArray(heapOut);

            t.copyTo(dst);
            assertArrayEquals(heapIn, heapOut, 1e-5f);
        }

        int[] heapInI32 = {10, 20, 30};
        MemorySegment srcI32 = MemorySegment.ofArray(heapInI32);
        try (Tensor t = Tensor.fromSegment(srcI32, DType.i32, 3)) {
            int[] heapOutI32 = new int[3];
            MemorySegment dstI32 = MemorySegment.ofArray(heapOutI32);

            t.copyTo(dstI32);
            assertArrayEquals(heapInI32, heapOutI32);
        }
    }

    @Test
    @DisplayName("copyTo from non-contiguous transposed view materializes correctly into segment")
    void testCopyToFromTransposedView() {
        try (Arena arena = Arena.ofConfined();
             Tensor m = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
             Tensor mt = m.T()) {

            MemorySegment dst = arena.allocate(6 * 4L, 16);
            mt.copyTo(dst);

            // Shape [3, 2]: row 0 is (1, 4), row 1 is (2, 5), row 2 is (3, 6)
            float[] expected = {1f, 4f, 2f, 5f, 3f, 6f};
            for (int i = 0; i < 6; i++) {
                assertEquals(expected[i], dst.getAtIndex(ValueLayout.JAVA_FLOAT, i), 1e-5f);
            }
        }
    }

    @Test
    @DisplayName("copyTo from sliced view with offset materializes correctly into segment")
    void testCopyToFromSlicedView() {
        try (Arena arena = Arena.ofConfined();
             Tensor m = Tensor.fromArray(new float[]{0, 10, 20, 30, 40, 50, 60, 70}, 8);
             Tensor s = m.slice(Slice.range(2, 6))) {

            assertEquals(4, s.getSize());
            MemorySegment dst = arena.allocate(4 * 4L, 16);
            s.copyTo(dst);

            float[] expected = {20f, 30f, 40f, 50f};
            for (int i = 0; i < 4; i++) {
                assertEquals(expected[i], dst.getAtIndex(ValueLayout.JAVA_FLOAT, i), 1e-5f);
            }
        }
    }

    @Test
    @DisplayName("copyTo from broadcast view materializes correctly into segment")
    void testCopyToFromBroadcastView() {
        try (Arena arena = Arena.ofConfined();
             Tensor col = Tensor.fromArray(new float[]{10, 20}, 2, 1);
             Tensor b = col.broadcastTo(2, 3)) {

            MemorySegment dst = arena.allocate(6 * 4L, 16);
            b.copyTo(dst);

            float[] expected = {10f, 10f, 10f, 20f, 20f, 20f};
            for (int i = 0; i < 6; i++) {
                assertEquals(expected[i], dst.getAtIndex(ValueLayout.JAVA_FLOAT, i), 1e-5f);
            }
        }
    }

    @Test
    @DisplayName("copyFrom updates tensor in-place from native and heap segments")
    void testCopyFrom() {
        try (Arena arena = Arena.ofConfined();
             Tensor t = Tensor.zeros(DType.f32, 4)) {

            MemorySegment nativeSrc = arena.allocate(4 * 4L, 16);
            nativeSrc.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 100f);
            nativeSrc.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 200f);
            nativeSrc.setAtIndex(ValueLayout.JAVA_FLOAT, 2, 300f);
            nativeSrc.setAtIndex(ValueLayout.JAVA_FLOAT, 3, 400f);

            t.copyFrom(nativeSrc);
            assertArrayEquals(new float[]{100f, 200f, 300f, 400f}, t.toArray(), 1e-5f);

            // Also test heap copyFrom
            float[] heapSrc = {5f, 6f, 7f, 8f};
            t.copyFrom(MemorySegment.ofArray(heapSrc));
            assertArrayEquals(heapSrc, t.toArray(), 1e-5f);
        }
    }

    @Test
    @DisplayName("copyFrom rejects strided, offset, or broadcast views")
    void testCopyFromValidation() {
        try (Arena arena = Arena.ofConfined();
             Tensor m = Tensor.zeros(DType.f32, 2, 3)) {

            MemorySegment seg = arena.allocate(6 * 4L, 16);

            // Transposed (non-contiguous)
            try (Tensor t = m.T()) {
                assertThrows(IllegalArgumentException.class, () -> t.copyFrom(seg));
            }

            // Offset view
            try (Tensor s = m.slice(Slice.from(1), Slice.all())) {
                assertThrows(IllegalArgumentException.class, () -> s.copyFrom(seg));
            }

            // Broadcast view
            try (Tensor col = Tensor.zeros(DType.f32, 2, 1);
                 Tensor b = col.broadcastTo(2, 3)) {
                assertThrows(IllegalArgumentException.class, () -> b.copyFrom(seg));
            }
        }
    }

    @Test
    @DisplayName("copyTo and copyFrom validate bounds, read-only status, and closed state")
    void testBoundsAndClosedValidation() {
        try (Arena arena = Arena.ofConfined();
             Tensor t = Tensor.zeros(DType.f32, 4)) {

            MemorySegment undersized = arena.allocate(8L); // only 8 bytes, needs 16
            assertThrows(IllegalArgumentException.class, () -> t.copyTo(undersized));
            assertThrows(IllegalArgumentException.class, () -> t.copyFrom(undersized));

            MemorySegment readOnly = arena.allocate(16L).asReadOnly();
            assertThrows(IllegalArgumentException.class, () -> t.copyTo(readOnly));
        }

        // Closed tensor rejection
        Tensor closedTensor = Tensor.zeros(DType.f32, 4);
        closedTensor.close();

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(16L);
            assertThrows(IllegalStateException.class, () -> closedTensor.copyTo(seg));
            assertThrows(IllegalStateException.class, () -> closedTensor.copyFrom(seg));
        }
    }
}
