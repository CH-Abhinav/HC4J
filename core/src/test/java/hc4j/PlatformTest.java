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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import hc4j.internal.ffi.Platform;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Platform OS/Arch Canonical Normalization Tests")
class PlatformTest {

    @Test
    @DisplayName("Windows platform normalization across variant names and arch aliases")
    void testWindowsNormalization() {
        Platform p1 = Platform.detect("Windows 11", "amd64");
        assertEquals("windows", p1.os());
        assertEquals("x86_64", p1.arch());
        assertEquals("windows-x86_64", p1.classifier());
        assertEquals("hc4j.dll", p1.libraryFileName());
        assertEquals("META-INF/hc4j/natives/windows-x86_64/hc4j.dll", p1.resourcePath());
        assertEquals("META-INF/hc4j/natives/windows-x86_64/hc4j.dll.sha256", p1.checksumResourcePath());

        Platform p2 = Platform.detect("Windows Server 2022", "x86_64");
        assertEquals("windows-x86_64", p2.classifier());

        Platform p3 = Platform.detect("Windows 10", "x64");
        assertEquals("windows-x86_64", p3.classifier());
    }

    @Test
    @DisplayName("Linux platform normalization across x86_64 and aarch64")
    void testLinuxNormalization() {
        Platform p1 = Platform.detect("Linux", "amd64");
        assertEquals("linux", p1.os());
        assertEquals("x86_64", p1.arch());
        assertEquals("linux-x86_64", p1.classifier());
        assertEquals("libhc4j.so", p1.libraryFileName());
        assertEquals("META-INF/hc4j/natives/linux-x86_64/libhc4j.so", p1.resourcePath());

        Platform p2 = Platform.detect("GNU/Linux", "aarch64");
        assertEquals("linux-aarch64", p2.classifier());

        Platform p3 = Platform.detect("Linux", "arm64");
        assertEquals("linux-aarch64", p3.classifier());
    }

    @Test
    @DisplayName("macOS platform normalization across Apple Silicon and Intel")
    void testMacOsNormalization() {
        Platform p1 = Platform.detect("Mac OS X", "aarch64");
        assertEquals("macos", p1.os());
        assertEquals("aarch64", p1.arch());
        assertEquals("macos-aarch64", p1.classifier());
        assertEquals("libhc4j.dylib", p1.libraryFileName());
        assertEquals("META-INF/hc4j/natives/macos-aarch64/libhc4j.dylib", p1.resourcePath());

        Platform p2 = Platform.detect("Darwin", "arm64");
        assertEquals("macos-aarch64", p2.classifier());

        Platform p3 = Platform.detect("Mac OS X", "x86_64");
        assertEquals("macos-x86_64", p3.classifier());
    }

    @Test
    @DisplayName("Unsupported OS throws UnsupportedOperationException containing raw name")
    void testUnsupportedOs() {
        UnsupportedOperationException ex = assertThrows(UnsupportedOperationException.class,
                () -> Platform.detect("Solaris", "x86_64"));
        assertTrue(ex.getMessage().contains("Solaris"), "Message must contain raw OS name: " + ex.getMessage());

        UnsupportedOperationException ex2 = assertThrows(UnsupportedOperationException.class,
                () -> Platform.detect("FreeBSD", "amd64"));
        assertTrue(ex2.getMessage().contains("FreeBSD"), "Message must contain raw OS name: " + ex2.getMessage());
    }

    @Test
    @DisplayName("Unsupported architecture throws UnsupportedOperationException containing raw arch")
    void testUnsupportedArch() {
        UnsupportedOperationException ex = assertThrows(UnsupportedOperationException.class,
                () -> Platform.detect("Linux", "riscv64"));
        assertTrue(ex.getMessage().contains("riscv64"), "Message must contain raw arch name: " + ex.getMessage());

        UnsupportedOperationException ex2 = assertThrows(UnsupportedOperationException.class,
                () -> Platform.detect("Linux", "mips64"));
        assertTrue(ex2.getMessage().contains("mips64"), "Message must contain raw arch name: " + ex2.getMessage());
    }

    @Test
    @DisplayName("Current host platform detects without exception")
    void testCurrentPlatform() {
        Platform current = Platform.current();
        assertNotNull(current);
        assertNotNull(current.os());
        assertNotNull(current.arch());
        assertNotNull(current.classifier());
        assertNotNull(current.libraryFileName());
    }
}
