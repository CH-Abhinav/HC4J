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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import hc4j.engine.GpuOutOfMemoryException;
import hc4j.internal.ffi.NativeEngine;
import hc4j.testutil.GpuTestSupport;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("NativeEngine Typed Exception Preservation Regression Tests")
class NativeEngineExceptionTest {

    @BeforeEach
    void setUp() {
        GpuTestSupport.assumeGpuAvailable();
    }

    @Test
    @DisplayName("checkStatus preserves exact typed exception classes without wrapping")
    void testCheckStatusTypedExceptions() {
        assertThrows(IllegalStateException.class, () ->
                NativeEngine.checkStatus(NativeEngine.HC4J_ERR_NOT_FOUND, "testNotFound"));
        assertThrows(IllegalArgumentException.class, () ->
                NativeEngine.checkStatus(NativeEngine.HC4J_ERR_INVALID_PARAM, "testInvalidParam"));
        assertThrows(GpuOutOfMemoryException.class, () ->
                NativeEngine.checkStatus(NativeEngine.HC4J_ERR_OUT_OF_MEMORY, "testOom"));
        assertThrows(IllegalStateException.class, () ->
                NativeEngine.checkStatus(NativeEngine.HC4J_ERR_DEVICE, "testDevice"));
        assertThrows(IllegalStateException.class, () ->
                NativeEngine.checkStatus(NativeEngine.HC4J_ERR_GPU_VALIDATION, "testGpuValidation"));
        assertThrows(UnsupportedOperationException.class, () ->
                NativeEngine.checkStatus(NativeEngine.HC4J_ERR_UNSUPPORTED, "testUnsupported"));
    }

    @Test
    @DisplayName("downloadVram with invalid handle surfaces exact IllegalStateException")
    void testDownloadVramInvalidHandleSurfacesIllegalStateException() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostData = arena.allocate(ValueLayout.JAVA_FLOAT, 16);
            IllegalStateException ex = assertThrows(IllegalStateException.class, () ->
                    NativeEngine.downloadVram(999_999_999L, hostData, 16));
            assertEquals(IllegalStateException.class, ex.getClass());
        }
    }

    @Test
    @DisplayName("writeVram with invalid handle surfaces exact IllegalStateException")
    void testWriteVramInvalidHandleSurfacesIllegalStateException() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostData = arena.allocate(ValueLayout.JAVA_FLOAT, 16);
            IllegalStateException ex = assertThrows(IllegalStateException.class, () ->
                    NativeEngine.writeVram(999_999_999L, hostData, 16));
            assertEquals(IllegalStateException.class, ex.getClass());
        }
    }

    @Test
    @DisplayName("downloadVramSegment with invalid handle surfaces exact IllegalStateException")
    void testDownloadVramSegmentInvalidHandleSurfacesIllegalStateException() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostData = arena.allocate(ValueLayout.JAVA_FLOAT, 16);
            IllegalStateException ex = assertThrows(IllegalStateException.class, () ->
                    NativeEngine.downloadVramSegment(999_999_999L, hostData, 16));
            assertEquals(IllegalStateException.class, ex.getClass());
        }
    }

    @Test
    @DisplayName("writeVramSegment with invalid handle surfaces exact IllegalStateException")
    void testWriteVramSegmentInvalidHandleSurfacesIllegalStateException() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostData = arena.allocate(ValueLayout.JAVA_FLOAT, 16);
            IllegalStateException ex = assertThrows(IllegalStateException.class, () ->
                    NativeEngine.writeVramSegment(999_999_999L, hostData, 16));
            assertEquals(IllegalStateException.class, ex.getClass());
        }
    }

    @Test
    @DisplayName("configureMemory with negative budget surfaces exact IllegalArgumentException")
    void testConfigureMemoryNegativeBudgetSurfacesIllegalArgumentException() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                NativeEngine.configureMemory(-10, -10));
        assertEquals(IllegalArgumentException.class, ex.getClass());
    }
}
