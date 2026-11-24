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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpServer;
import hc4j.internal.ffi.NativeLoader;
import hc4j.internal.ffi.Platform;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@DisplayName("NativeLoader 5-Stage Deterministic Resolution Tests")
class NativeLoaderTest {

    private static final class MockClassLoader extends ClassLoader {
        private final Map<String, byte[]> resources = new HashMap<>();

        MockClassLoader() {
            super(null); // isolated parent
        }

        void put(String path, byte[] data) {
            resources.put(path, data);
        }

        @Override
        public InputStream getResourceAsStream(String name) {
            byte[] data = resources.get(name);
            if (data != null) {
                return new ByteArrayInputStream(data);
            }
            return null;
        }
    }

    private static byte[] createJarBytes(String entryName, byte[] content) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ZipOutputStream zos = new ZipOutputStream(baos)) {
            ZipEntry entry = new ZipEntry(entryName);
            zos.putNextEntry(entry);
            zos.write(content);
            zos.closeEntry();
        }
        return baos.toByteArray();
    }

    @Test
    @DisplayName("Stage 1: Explicit path override fails fast on missing file")
    void testStage1MissingFileFailsFast(@TempDir Path tempDir) {
        Platform platform = Platform.detect("Linux", "x86_64");
        Path nonExistent = tempDir.resolve("does_not_exist.so");
        List<String> diagnostics = new ArrayList<>();

        UnsatisfiedLinkError err = assertThrows(UnsatisfiedLinkError.class, () ->
                NativeLoader.resolve(platform, null, tempDir, nonExistent.toString(), null, false, true, false, diagnostics));

        assertTrue(err.getMessage().contains("Stage 1 failed"), err.getMessage());
        assertTrue(err.getMessage().contains("does_not_exist.so"), err.getMessage());
    }

    @Test
    @DisplayName("Stage 1: Explicit path override resolves valid file directly")
    void testStage1ValidFileResolves(@TempDir Path tempDir) throws IOException {
        Platform platform = Platform.detect("Linux", "x86_64");
        Path validLib = tempDir.resolve("libhc4j.so");
        Files.writeString(validLib, "fake-elf-binary");
        List<String> diagnostics = new ArrayList<>();

        Path resolved = NativeLoader.resolve(platform, null, tempDir, validLib.toString(), null, false, true, false, diagnostics);
        assertEquals(validLib.toAbsolutePath(), resolved);
        assertTrue(diagnostics.get(0).contains("Stage 1"));
    }

    @Test
    @DisplayName("Stage 2: Classpath JAR extraction extracts to content-addressed cache with SHA-256")
    void testStage2ClasspathExtraction(@TempDir Path tempDir) throws IOException {
        Platform platform = Platform.detect("Linux", "x86_64");
        byte[] payload = "mock-native-bytes-stage-2".getBytes(StandardCharsets.UTF_8);
        String sha256 = NativeLoader.sha256Hex(payload);

        MockClassLoader cl = new MockClassLoader();
        cl.put(platform.resourcePath(), payload);
        cl.put(platform.checksumResourcePath(), (sha256 + "\n").getBytes(StandardCharsets.UTF_8));

        List<String> diagnostics = new ArrayList<>();
        Path resolved = NativeLoader.resolve(platform, cl, tempDir, null, null, false, true, false, diagnostics);

        assertNotNull(resolved);
        assertTrue(Files.isRegularFile(resolved));
        assertArrayEquals(payload, Files.readAllBytes(resolved));
        assertTrue(resolved.toString().contains(sha256), "Path should contain sha256: " + resolved);

        // Second call: cache hit fast path without re-extraction
        List<String> diag2 = new ArrayList<>();
        Path resolved2 = NativeLoader.resolve(platform, cl, tempDir, null, null, false, true, false, diag2);
        assertEquals(resolved, resolved2);
    }

    @Test
    @DisplayName("Stage 2: Classpath JAR extraction detects and rejects tampered checksum sidecar")
    void testStage2TamperedChecksumRejection(@TempDir Path tempDir) {
        Platform platform = Platform.detect("Linux", "x86_64");
        byte[] payload = "mock-native-payload".getBytes(StandardCharsets.UTF_8);

        MockClassLoader cl = new MockClassLoader();
        cl.put(platform.resourcePath(), payload);
        cl.put(platform.checksumResourcePath(), "0000000000000000000000000000000000000000000000000000000000000000\n".getBytes(StandardCharsets.UTF_8));

        List<String> diagnostics = new ArrayList<>();
        UnsatisfiedLinkError err = assertThrows(UnsatisfiedLinkError.class, () ->
                NativeLoader.resolve(platform, cl, tempDir, null, null, false, true, false, diagnostics));

        assertTrue(err.getMessage().contains("checksum mismatch"), err.getMessage());
    }

    @Test
    @DisplayName("Stage 5: Auto on-demand download happy path against local HttpServer")
    void testStage5AutoDownloadHappyPath(@TempDir Path tempDir) throws IOException {
        Platform platform = Platform.detect("Linux", "x86_64");
        byte[] nativePayload = "downloaded-native-lib-binary".getBytes(StandardCharsets.UTF_8);
        byte[] jarBytes = createJarBytes(platform.resourcePath(), nativePayload);
        String jarSha256 = NativeLoader.sha256Hex(jarBytes);

        AtomicInteger requestCount = new AtomicInteger(0);
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/release/hc4j-0.1.0-linux-x86_64.jar", exchange -> {
            requestCount.incrementAndGet();
            exchange.sendResponseHeaders(200, jarBytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(jarBytes);
            }
        });
        server.createContext("/release/hc4j-0.1.0-linux-x86_64.jar.sha256", exchange -> {
            byte[] response = (jarSha256 + "  hc4j.jar\n").getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, response.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(response);
            }
        });
        server.start();

        try {
            int port = server.getAddress().getPort();
            String urlTemplate = "http://127.0.0.1:" + port + "/release/hc4j-{version}-{classifier}.jar";

            List<String> diagnostics = new ArrayList<>();
            Path resolved = NativeLoader.resolve(platform, null, tempDir, null, urlTemplate, true, true, false, diagnostics);

            assertNotNull(resolved);
            assertTrue(Files.isRegularFile(resolved));
            assertArrayEquals(nativePayload, Files.readAllBytes(resolved));
            assertEquals(1, requestCount.get(), "Artifact should be downloaded once");

            // Second load: should hit cache with 0 additional downloads
            List<String> diag2 = new ArrayList<>();
            Path resolved2 = NativeLoader.resolve(platform, null, tempDir, null, urlTemplate, true, true, false, diag2);
            assertEquals(resolved, resolved2);
            assertEquals(1, requestCount.get(), "Cache hit must not make another HTTP request");
        } finally {
            server.stop(0);
        }
    }

    @Test
    @DisplayName("Stage 5: Auto on-demand download tamper test fails on checksum mismatch")
    void testStage5TamperTest(@TempDir Path tempDir) throws IOException {
        Platform platform = Platform.detect("Linux", "x86_64");
        byte[] nativePayload = "downloaded-native-binary".getBytes(StandardCharsets.UTF_8);
        byte[] jarBytes = createJarBytes(platform.resourcePath(), nativePayload);

        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/release/hc4j-0.1.0-linux-x86_64.jar", exchange -> {
            exchange.sendResponseHeaders(200, jarBytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(jarBytes);
            }
        });
        server.createContext("/release/hc4j-0.1.0-linux-x86_64.jar.sha256", exchange -> {
            byte[] response = "badhash00000000000000000000000000000000000000000000000000000000\n".getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, response.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(response);
            }
        });
        server.start();

        try {
            int port = server.getAddress().getPort();
            String urlTemplate = "http://127.0.0.1:" + port + "/release/hc4j-{version}-{classifier}.jar";

            List<String> diagnostics = new ArrayList<>();
            UnsatisfiedLinkError err = assertThrows(UnsatisfiedLinkError.class, () ->
                    NativeLoader.resolve(platform, null, tempDir, null, urlTemplate, true, true, false, diagnostics));

            assertTrue(err.getMessage().contains("SHA-256 mismatch") || err.getMessage().contains("Could not load HC4J native library"), err.getMessage());
        } finally {
            server.stop(0);
        }
    }

    @Test
    @DisplayName("Stage 5: Opt-out prevents HTTP requests and includes clear diagnostic")
    void testStage5OptOut(@TempDir Path tempDir) throws IOException {
        Platform platform = Platform.detect("Linux", "x86_64");
        AtomicInteger requestCount = new AtomicInteger(0);

        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            requestCount.incrementAndGet();
            exchange.sendResponseHeaders(500, 0);
            exchange.close();
        });
        server.start();

        try {
            int port = server.getAddress().getPort();
            String urlTemplate = "http://127.0.0.1:" + port + "/release/hc4j-{version}-{classifier}.jar";

            List<String> diagnostics = new ArrayList<>();
            UnsatisfiedLinkError err = assertThrows(UnsatisfiedLinkError.class, () ->
                    NativeLoader.resolve(platform, null, tempDir, null, urlTemplate, false, true, false, diagnostics));

            assertEquals(0, requestCount.get(), "Disabled download must not make HTTP requests");
            assertTrue(err.getMessage().contains("-Dhc4j.native.download=false"), err.getMessage());
        } finally {
            server.stop(0);
        }
    }

    @Test
    @DisplayName("Stage 5: Default mode without pinned checksum makes 0 HTTP requests and reports clear diagnostic")
    void testStage5DefaultUnpinnedSkipsDownload(@TempDir Path tempDir) throws IOException {
        Platform platform = Platform.detect("Linux", "x86_64");
        AtomicInteger requestCount = new AtomicInteger(0);

        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            requestCount.incrementAndGet();
            exchange.sendResponseHeaders(500, 0);
            exchange.close();
        });
        server.start();

        try {
            int port = server.getAddress().getPort();
            String urlTemplate = "http://127.0.0.1:" + port + "/release/hc4j-{version}-{classifier}.jar";

            List<String> diagnostics = new ArrayList<>();
            // downloadEnabled = null (default), no pinned checksum
            UnsatisfiedLinkError err = assertThrows(UnsatisfiedLinkError.class, () ->
                    NativeLoader.resolve(platform, null, tempDir, null, urlTemplate, null, true, false, diagnostics));

            assertEquals(0, requestCount.get(), "Default unpinned mode must not make HTTP requests");
            assertTrue(err.getMessage().contains("no pinned release checksum in JAR; set -Dhc4j.native.download=true to allow unpinned download"),
                    err.getMessage());
        } finally {
            server.stop(0);
        }
    }

    @Test
    @DisplayName("Stage 5: Default mode with pinned checksum allows download and strictly verifies against JAR pinned hash")
    void testStage5DefaultWithPinnedChecksumHappyPath(@TempDir Path tempDir) throws IOException {
        Platform platform = Platform.detect("Linux", "x86_64");
        byte[] nativePayload = "pinned-verified-native-binary".getBytes(StandardCharsets.UTF_8);
        byte[] jarBytes = createJarBytes(platform.resourcePath(), nativePayload);
        String jarSha256 = NativeLoader.sha256Hex(jarBytes);

        MockClassLoader cl = new MockClassLoader();
        String pinnedProps = platform.classifier() + "=" + jarSha256 + "\n";
        cl.put(NativeLoader.PINNED_CHECKSUMS_RESOURCE, pinnedProps.getBytes(StandardCharsets.UTF_8));

        AtomicInteger requestCount = new AtomicInteger(0);
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/release/hc4j-0.1.0-linux-x86_64.jar", exchange -> {
            requestCount.incrementAndGet();
            exchange.sendResponseHeaders(200, jarBytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(jarBytes);
            }
        });
        server.start();

        try {
            int port = server.getAddress().getPort();
            String urlTemplate = "http://127.0.0.1:" + port + "/release/hc4j-{version}-{classifier}.jar";

            List<String> diagnostics = new ArrayList<>();
            // downloadEnabled = null (default), but pinned checksum present in JAR!
            Path resolved = NativeLoader.resolve(platform, cl, tempDir, null, urlTemplate, null, true, false, diagnostics);

            assertNotNull(resolved);
            assertTrue(Files.isRegularFile(resolved));
            assertArrayEquals(nativePayload, Files.readAllBytes(resolved));
            assertEquals(1, requestCount.get());
        } finally {
            server.stop(0);
        }
    }

    @Test
    @DisplayName("Stage 5: Pinned checksum in JAR rejects tampered remote binary even if remote .sha256 matches tampered binary")
    void testStage5PinnedChecksumRejectsTamperedRemoteEvenWithMatchingRemoteSidecar(@TempDir Path tempDir) throws IOException {
        Platform platform = Platform.detect("Linux", "x86_64");
        byte[] tamperedNativePayload = "tampered-attacker-binary".getBytes(StandardCharsets.UTF_8);
        byte[] tamperedJarBytes = createJarBytes(platform.resourcePath(), tamperedNativePayload);
        String tamperedJarSha256 = NativeLoader.sha256Hex(tamperedJarBytes);

        // Honest release hash pinned in JAR
        String genuinePinnedSha256 = "1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff";
        MockClassLoader cl = new MockClassLoader();
        String pinnedProps = platform.classifier() + "=" + genuinePinnedSha256 + "\n";
        cl.put(NativeLoader.PINNED_CHECKSUMS_RESOURCE, pinnedProps.getBytes(StandardCharsets.UTF_8));

        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        // Attacker serves tampered JAR
        server.createContext("/release/hc4j-0.1.0-linux-x86_64.jar", exchange -> {
            exchange.sendResponseHeaders(200, tamperedJarBytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(tamperedJarBytes);
            }
        });
        // Attacker also serves matching tampered .sha256
        server.createContext("/release/hc4j-0.1.0-linux-x86_64.jar.sha256", exchange -> {
            byte[] response = (tamperedJarSha256 + "\n").getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, response.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(response);
            }
        });
        server.start();

        try {
            int port = server.getAddress().getPort();
            String urlTemplate = "http://127.0.0.1:" + port + "/release/hc4j-{version}-{classifier}.jar";

            List<String> diagnostics = new ArrayList<>();
            UnsatisfiedLinkError err = assertThrows(UnsatisfiedLinkError.class, () ->
                    NativeLoader.resolve(platform, cl, tempDir, null, urlTemplate, null, true, false, diagnostics));

            assertTrue(err.getMessage().contains("mismatch against JAR pinned digest") || err.getMessage().contains("Could not load HC4J native library"),
                    err.getMessage());
        } finally {
            server.stop(0);
        }
    }

    @Test
    @DisplayName("Packaged classifier JAR extraction via URLClassLoader")
    void testPackagedClassifierJarExtraction(@TempDir Path tempDir) throws IOException {
        Platform platform = Platform.current();
        Path candidateJar = Paths.get("build", "libs", "hc4j-0.1.0-" + platform.classifier() + ".jar");
        if (!Files.isRegularFile(candidateJar)) {
            // Check relative to core subproject
            candidateJar = Paths.get("..", "core", "build", "libs", "hc4j-0.1.0-" + platform.classifier() + ".jar");
            if (!Files.isRegularFile(candidateJar)) {
                candidateJar = Paths.get("core", "build", "libs", "hc4j-0.1.0-" + platform.classifier() + ".jar");
            }
        }

        Assumptions.assumeTrue(Files.isRegularFile(candidateJar),
                "Run :core:nativeJar first to test packaged classifier JAR extraction");

        try (URLClassLoader ucl = new URLClassLoader(new URL[] {candidateJar.toUri().toURL()}, null)) {
            List<String> diagnostics = new ArrayList<>();
            Path resolved = NativeLoader.resolve(platform, ucl, tempDir, null, null, false, true, false, diagnostics);
            assertNotNull(resolved);
            assertTrue(Files.isRegularFile(resolved));
            assertTrue(Files.size(resolved) > 0);
        }
    }
}
