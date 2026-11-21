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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import hc4j.collective.Collective;
import hc4j.collective.CollectiveBuffer;
import hc4j.internal.ffi.TensorAccess;
import hc4j.testutil.GpuTestSupport;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Tensor and Resource Lifecycle Tests")
class TensorLifecycleTest {

    @Test
    @DisplayName("Zero leaked public internal methods via reflection")
    void testZeroLeakedInternalMethods() {
        assertThrows(NoSuchMethodException.class, () -> Tensor.class.getMethod("getVramId"));
        assertThrows(NoSuchMethodException.class, () -> Tensor.class.getMethod("internalShapeUnsafe"));
        assertThrows(NoSuchMethodException.class, () -> Tensor.class.getMethod("internalStridesUnsafe"));
    }

    @Test
    @DisplayName("getShape returns a defensive copy")
    void testGetShapeDefensiveCopy() {
        Tensor t = TensorAccess.getInstance().create(DType.f32, new int[] {2, 3}, new int[] {3, 1}, 0L);
        int[] s1 = t.getShape();
        int[] s2 = t.getShape();
        assertNotSame(s1, s2);
        assertArrayEquals(new int[] {2, 3}, s1);

        s1[0] = 999;
        assertNotEquals(999, t.getShape()[0]);
    }

    @Test
    @DisplayName("Idempotent double close and use-after-free guards on unbacked Tensor")
    void testTensorCloseAndUseAfterFree() {
        Tensor t = TensorAccess.getInstance().create(DType.f32, new int[] {2, 3}, new int[] {3, 1}, 0L);
        assertFalse(t.isClosed());

        t.close();
        assertTrue(t.isClosed());

        // Idempotent double close
        t.close();
        assertTrue(t.isClosed());

        // Calling operations on closed tensor throws IllegalStateException
        Tensor dummy = TensorAccess.getInstance().create(DType.f32, new int[] {2, 3}, new int[] {3, 1}, 0L);

        assertThrowsWithMessage(IllegalStateException.class, () -> t.add(dummy), "Tensor has already been closed");
        assertThrowsWithMessage(IllegalStateException.class, () -> t.sub(dummy), "Tensor has already been closed");
        assertThrowsWithMessage(IllegalStateException.class, () -> t.mul(dummy), "Tensor has already been closed");
        assertThrowsWithMessage(IllegalStateException.class, () -> t.div(dummy), "Tensor has already been closed");
        assertThrowsWithMessage(IllegalStateException.class, t::sin, "Tensor has already been closed");
        assertThrowsWithMessage(IllegalStateException.class, t::cos, "Tensor has already been closed");
        assertThrowsWithMessage(IllegalStateException.class, t::exp, "Tensor has already been closed");
        assertThrowsWithMessage(IllegalStateException.class, () -> t.matmul(dummy), "Tensor has already been closed");
        assertThrowsWithMessage(IllegalStateException.class, t::transpose, "Tensor has already been closed");
        assertThrowsWithMessage(IllegalStateException.class, t::toFloatArray, "Tensor has already been closed");
        assertThrowsWithMessage(IllegalStateException.class, t::toIntArray, "Tensor has already been closed");
        assertThrowsWithMessage(IllegalStateException.class, t::lazy, "Tensor has already been closed");
        assertThrowsWithMessage(IllegalStateException.class, t::residency, "Tensor has already been closed");
        assertThrowsWithMessage(IllegalStateException.class, t::evict, "Tensor has already been closed");
    }

    @Test
    @DisplayName("Real GPU Tensor lifecycle and double close")
    void testGpuTensorLifecycle() {
        GpuTestSupport.assumeGpuAvailable();

        Tensor t = Tensor.fromArray(new float[] {1f, 2f, 3f}, 3);
        assertFalse(t.isClosed());
        t.close();
        assertTrue(t.isClosed());
        t.close(); // idempotent
        assertTrue(t.isClosed());

        assertThrowsWithMessage(IllegalStateException.class, t::sin, "Tensor has already been closed");
    }

    @Test
    @DisplayName("Collective and CollectiveBuffer lifecycle safety")
    void testCollectiveLifecycle() {
        GpuTestSupport.assumeGpuAvailable();

        Collective ring = Collective.open(2);
        try {
            CollectiveBuffer buf = ring.allocate(0, 10);
            assertFalse(buf.isClosed());

            // Close collective first
            ring.close();
            assertTrue(ring.isClosed());

            // Idempotent double close on collective
            ring.close();
            assertTrue(ring.isClosed());

            // Collective operations throw
            assertThrowsWithMessage(IllegalStateException.class, () -> ring.deviceName(0), "Collective has already been closed");

            // Closing buffer after collective is closed is safe and does not crash or double-free
            buf.close();
            assertTrue(buf.isClosed());
            buf.close();
            assertTrue(buf.isClosed());

            // Buffer operations throw
            assertThrowsWithMessage(IllegalStateException.class, () -> buf.write(new float[10]), "CollectiveBuffer has already been closed");
        } finally {
            if (!ring.isClosed()) {
                ring.close();
            }
        }
    }
}
