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
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("GpuScope LIFO Resource Management Tests")
class GpuScopeTest {

    static final class MockResource implements AutoCloseable {
        private final int id;
        private final List<Integer> closeLog;
        private final boolean throwOnClose;
        private boolean closed = false;

        MockResource(int id, List<Integer> closeLog) {
            this(id, closeLog, false);
        }

        MockResource(int id, List<Integer> closeLog, boolean throwOnClose) {
            this.id = id;
            this.closeLog = closeLog;
            this.throwOnClose = throwOnClose;
        }

        @Override
        public void close() throws IOException {
            this.closed = true;
            if (closeLog != null) {
                closeLog.add(id);
            }
            if (throwOnClose) {
                throw new IOException("Failed to close resource " + id);
            }
        }

        public boolean isClosed() {
            return closed;
        }
    }

    @Test
    @DisplayName("GpuScope closes tracked resources in reverse LIFO order")
    void testLifoCloseOrder() {
        List<Integer> log = new ArrayList<>();
        GpuScope scope = GpuScope.open();
        assertFalse(scope.isClosed());
        assertEquals(0, scope.trackedCount());

        MockResource r1 = scope.track(new MockResource(1, log));
        MockResource r2 = scope.track(new MockResource(2, log));
        MockResource r3 = scope.track(new MockResource(3, log));

        assertEquals(3, scope.trackedCount());
        assertFalse(r1.isClosed());
        assertFalse(r2.isClosed());
        assertFalse(r3.isClosed());

        scope.close();

        assertTrue(scope.isClosed());
        assertEquals(0, scope.trackedCount());
        assertTrue(r1.isClosed());
        assertTrue(r2.isClosed());
        assertTrue(r3.isClosed());

        // LIFO order: 3, 2, 1
        assertEquals(List.of(3, 2, 1), log);
    }

    @Test
    @DisplayName("Detach removes resource from scope and prevents automatic closing")
    void testDetach() {
        List<Integer> log = new ArrayList<>();
        try (GpuScope scope = GpuScope.open()) {
            MockResource r1 = scope.track(new MockResource(1, log));
            MockResource r2 = scope.track(new MockResource(2, log));
            MockResource r3 = scope.track(new MockResource(3, log));

            assertEquals(3, scope.trackedCount());

            MockResource detached = scope.detach(r2);
            assertSame(r2, detached);
            assertEquals(2, scope.trackedCount());

            // Detaching an already detached or untracked resource throws IllegalArgumentException
            assertThrows(IllegalArgumentException.class, () -> scope.detach(r2));
        }

        // Only 3 and 1 were closed, 2 was escaped
        assertEquals(List.of(3, 1), log);
    }

    @Test
    @DisplayName("Double close is idempotent and does not re-close resources")
    void testDoubleClose() {
        List<Integer> log = new ArrayList<>();
        GpuScope scope = GpuScope.open();
        scope.track(new MockResource(1, log));

        scope.close();
        assertEquals(1, log.size());
        assertTrue(scope.isClosed());

        // Subsequent closes do nothing
        scope.close();
        assertEquals(1, log.size());
    }

    @Test
    @DisplayName("Track rejects new allocations after scope is closed")
    void testTrackAfterClose() {
        GpuScope scope = GpuScope.open();
        scope.close();

        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> scope.track(new MockResource(1, null)));
        assertEquals("GpuScope is already closed", ex.getMessage());

        assertThrows(NullPointerException.class, () -> {
            try (GpuScope s = GpuScope.open()) {
                s.track(null);
            }
        });
    }

    @Test
    @DisplayName("Exception suppression aggregates failures when multiple resources fail during close")
    void testExceptionSuppression() {
        GpuScope scope = GpuScope.open();
        scope.track(new MockResource(1, null, true));
        scope.track(new MockResource(2, null, true));
        scope.track(new MockResource(3, null, false));

        RuntimeException re = assertThrows(RuntimeException.class, scope::close);
        assertTrue(re.getMessage().contains("Exception during GpuScope close"));
        // Primary cause should be from resource 2 (closed first due to LIFO)
        assertTrue(re.getCause() instanceof IOException);
        assertEquals("Failed to close resource 2", re.getCause().getMessage());

        // Suppressed exception on the cause should be from resource 1
        Throwable[] suppressed = re.getCause().getSuppressed();
        assertEquals(1, suppressed.length);
        assertTrue(suppressed[0] instanceof IOException);
        assertEquals("Failed to close resource 1", suppressed[0].getMessage());
    }
}
