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

package hc4j.internal.ffi;

import java.util.Objects;
import java.util.Set;

/**
 * Immutable platform descriptor providing canonical OS/architecture detection,
 * library naming, and resource path mapping for native binaries.
 *
 * @param os Canonical operating system: {@code "windows"}, {@code "linux"}, or {@code "macos"}
 * @param arch Canonical processor architecture: {@code "x86_64"} or {@code "aarch64"}
 */
public record Platform(String os, String arch) {

    public static final String WINDOWS_X86_64 = "windows-x86_64";
    public static final String LINUX_X86_64 = "linux-x86_64";
    public static final String LINUX_AARCH64 = "linux-aarch64";
    public static final String MACOS_AARCH64 = "macos-aarch64";
    public static final String MACOS_X86_64 = "macos-x86_64";

    private static final Set<String> SUPPORTED = Set.of(
            WINDOWS_X86_64, LINUX_X86_64, LINUX_AARCH64, MACOS_AARCH64, MACOS_X86_64);

    private static final Platform CURRENT = detect(
            System.getProperty("os.name", "unknown"),
            System.getProperty("os.arch", "unknown"));

    public Platform {
        Objects.requireNonNull(os, "os");
        Objects.requireNonNull(arch, "arch");
    }

    /**
     * Returns the host platform detected from the running JVM's system properties.
     */
    public static Platform current() {
        return CURRENT;
    }

    /**
     * Detects and normalizes raw OS and architecture name strings into a canonical {@link Platform}.
     *
     * @param osName Raw OS name (e.g. {@code "Windows 11"}, {@code "Linux"}, {@code "Mac OS X"})
     * @param osArch Raw architecture name (e.g. {@code "amd64"}, {@code "x86_64"}, {@code "aarch64"})
     * @return Normalized platform descriptor
     * @throws UnsupportedOperationException if the OS or architecture is not supported
     */
    public static Platform detect(String osName, String osArch) {
        if (osName == null) osName = "unknown";
        if (osArch == null) osArch = "unknown";

        String lowerOs = osName.toLowerCase();
        String normOs;
        if (lowerOs.contains("mac") || lowerOs.contains("darwin")) {
            normOs = "macos";
        } else if (lowerOs.contains("win")) {
            normOs = "windows";
        } else if (lowerOs.contains("linux") || lowerOs.contains("nux")) {
            normOs = "linux";
        } else {
            throw new UnsupportedOperationException("Unsupported OS: " + osName + " (arch: " + osArch + ")");
        }

        String lowerArch = osArch.toLowerCase();
        String normArch;
        if (lowerArch.equals("amd64") || lowerArch.equals("x86_64") || lowerArch.equals("x64")) {
            normArch = "x86_64";
        } else if (lowerArch.equals("aarch64") || lowerArch.equals("arm64")) {
            normArch = "aarch64";
        } else {
            throw new UnsupportedOperationException("Unsupported architecture: " + osArch + " (os: " + osName + ")");
        }

        String classifier = normOs + "-" + normArch;
        if (!SUPPORTED.contains(classifier)) {
            throw new UnsupportedOperationException("Unsupported platform: " + osName + " (" + osArch + ")");
        }

        return new Platform(normOs, normArch);
    }

    /**
     * Canonical classifier key formatted as {@code <os>-<arch>} (e.g. {@code "windows-x86_64"}).
     */
    public String classifier() {
        return os + "-" + arch;
    }

    /**
     * Shared library file name on this platform (e.g. {@code "hc4j.dll"}, {@code "libhc4j.so"}, {@code "libhc4j.dylib"}).
     */
    public String libraryFileName() {
        return switch (os) {
            case "windows" -> "hc4j.dll";
            case "macos" -> "libhc4j.dylib";
            case "linux" -> "libhc4j.so";
            default -> throw new IllegalStateException("Unknown OS: " + os);
        };
    }

    /**
     * Resource path inside a classifier JAR or fat bundle (e.g. {@code "META-INF/hc4j/natives/windows-x86_64/hc4j.dll"}).
     */
    public String resourcePath() {
        return "META-INF/hc4j/natives/" + classifier() + "/" + libraryFileName();
    }

    /**
     * Sidecar SHA-256 checksum resource path (e.g. {@code "META-INF/hc4j/natives/windows-x86_64/hc4j.dll.sha256"}).
     */
    public String checksumResourcePath() {
        return resourcePath() + ".sha256";
    }
}
