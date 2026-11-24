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

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.SymbolLookup;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Properties;
import java.util.UUID;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Unified native library resolution and loading system supporting:
 * <ol>
 *   <li>Explicit path override ({@code -Dhc4j.native.path} or {@code HC4J_NATIVE_PATH})</li>
 *   <li>Classpath extraction for platform classifier JARs and fat bundles</li>
 *   <li>System {@code java.library.path} lookup</li>
 *   <li>Local developer source-tree build fallback</li>
 *   <li>Automatic on-demand binary download with cryptographic verification</li>
 * </ol>
 */
public final class NativeLoader {

    public static final String PROPERTY_NATIVE_PATH = "hc4j.native.path";
    public static final String ENV_NATIVE_PATH = "HC4J_NATIVE_PATH";
    public static final String PROPERTY_CACHE_DIR = "hc4j.cache.dir";
    public static final String ENV_CACHE_DIR = "HC4J_CACHE_DIR";
    public static final String PROPERTY_DOWNLOAD = "hc4j.native.download";
    public static final String ENV_DOWNLOAD = "HC4J_NATIVE_DOWNLOAD";
    public static final String PROPERTY_DOWNLOAD_URL = "hc4j.native.download.url";
    public static final String ENV_DOWNLOAD_URL = "HC4J_NATIVE_DOWNLOAD_URL";
    public static final String PROPERTY_DOWNLOAD_VERIFY = "hc4j.native.download.verify";

    public static final String DEFAULT_URL_TEMPLATE =
            "https://github.com/CH-Abhinav/HC4J/releases/download/v{version}/hc4j-{version}-{classifier}.jar";

    public static final String PINNED_CHECKSUMS_RESOURCE = "META-INF/hc4j/natives-checksums.properties";

    private NativeLoader() {
        throw new AssertionError("No instances of NativeLoader");
    }

    /**
     * Resolves and loads the native library using the default host platform and configuration.
     *
     * @return NativeLoadResult containing the active Panama SymbolLookup and metadata
     * @throws UnsatisfiedLinkError if all 5 resolution stages fail
     */
    public static NativeLoadResult load() {
        Platform platform = Platform.current();
        return load(platform, NativeLoader.class.getClassLoader(), resolveDefaultCacheDir(platform));
    }

    /**
     * Resolves and loads the native library with injected platform, classloader, and cache directory.
     */
    public static NativeLoadResult load(Platform platform, ClassLoader classLoader, Path cacheDir) {
        String explicitPath = System.getProperty(PROPERTY_NATIVE_PATH);
        if (explicitPath == null || explicitPath.isBlank()) {
            explicitPath = System.getenv(ENV_NATIVE_PATH);
        }

        String downloadUrl = System.getProperty(PROPERTY_DOWNLOAD_URL);
        if (downloadUrl == null || downloadUrl.isBlank()) {
            downloadUrl = System.getenv(ENV_DOWNLOAD_URL);
        }

        Boolean downloadEnabled = parseBoolean(System.getProperty(PROPERTY_DOWNLOAD), System.getenv(ENV_DOWNLOAD));
        Boolean verifyChecksum = parseBoolean(System.getProperty(PROPERTY_DOWNLOAD_VERIFY), null);

        List<String> diagnostics = new ArrayList<>();
        Path resolvedPath = resolve(platform, classLoader, cacheDir, explicitPath, downloadUrl,
                downloadEnabled, verifyChecksum, true, diagnostics);

        String source = diagnostics.isEmpty() ? "Unknown" : diagnostics.get(diagnostics.size() - 1);
        return loadLibrary(resolvedPath, source, platform);
    }

    /**
     * Lowers a verified on-disk library path to a Panama {@link SymbolLookup}.
     */
    @SuppressWarnings("restricted")
    public static NativeLoadResult loadLibrary(Path libraryPath, String source, Platform platform) {
        Objects.requireNonNull(libraryPath, "libraryPath");
        try {
            SymbolLookup lookup = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            return new NativeLoadResult(lookup, libraryPath.toAbsolutePath(), source, platform);
        } catch (Throwable t) {
            throw new UnsatisfiedLinkError("Failed to load native library from " + libraryPath + ": " + t.getMessage());
        }
    }

    /**
     * 5-Stage deterministic resolution returning the absolute path to the native library.
     */
    public static Path resolve(
            Platform platform,
            ClassLoader classLoader,
            Path cacheDir,
            String explicitPath,
            String downloadUrlTemplate,
            Boolean downloadEnabled,
            Boolean verifyChecksum,
            boolean enableDevSearch,
            List<String> diagnostics) {

        String version = getVersion();
        String classifier = platform.classifier();
        String libName = platform.libraryFileName();

        // Stage 1: Explicit Path Override
        if (explicitPath != null && !explicitPath.isBlank()) {
            Path p = Paths.get(explicitPath).toAbsolutePath();
            if (Files.isDirectory(p)) {
                Path candidate = p.resolve(libName);
                if (Files.isRegularFile(candidate)) {
                    diagnostics.add("Stage 1 (explicit path override): found at " + candidate);
                    return candidate;
                }
            } else if (Files.isRegularFile(p)) {
                diagnostics.add("Stage 1 (explicit path override): found at " + p);
                return p;
            }
            throw new UnsatisfiedLinkError("Stage 1 failed: Explicit path '" + explicitPath + "' does not exist or is not a regular file.");
        } else {
            diagnostics.add("Stage 1 (explicit path override): not configured");
        }

        // Stage 2: Classpath JAR Extraction
        if (classLoader != null) {
            String resPath = platform.resourcePath();
            try (InputStream in = classLoader.getResourceAsStream(resPath)) {
                if (in != null) {
                    byte[] bytes = in.readAllBytes();
                    String sha256 = sha256Hex(bytes);

                    // Check sidecar checksum if available
                    String checkPath = platform.checksumResourcePath();
                    try (InputStream checkIn = classLoader.getResourceAsStream(checkPath)) {
                        if (checkIn != null) {
                            String expectedSha = parseSha256(new String(checkIn.readAllBytes(), StandardCharsets.UTF_8));
                            if (!expectedSha.isBlank() && !sha256.equalsIgnoreCase(expectedSha)) {
                                throw new UnsatisfiedLinkError("Classpath native binary checksum mismatch for "
                                        + resPath + ": expected " + expectedSha + " but got " + sha256);
                            }
                        }
                    }

                    Path target = cacheDir.resolve(version).resolve(classifier).resolve(sha256).resolve(libName);
                    Path cached = extractAtomically(bytes, sha256, target, platform);
                    diagnostics.add("Stage 2 (classpath extraction): extracted from " + resPath + " to " + cached);
                    return cached;
                } else {
                    diagnostics.add("Stage 2 (classpath extraction): resource " + resPath + " not found on classpath");
                }
            } catch (IOException e) {
                diagnostics.add("Stage 2 (classpath extraction): failed reading resource: " + e.getMessage());
            }
        } else {
            diagnostics.add("Stage 2 (classpath extraction): no ClassLoader provided");
        }

        // Stage 3: java.library.path Search
        String javaLibPath = System.getProperty("java.library.path");
        if (javaLibPath != null && !javaLibPath.isBlank()) {
            String[] entries = javaLibPath.split(File.pathSeparator);
            for (String entry : entries) {
                if (!entry.isBlank()) {
                    Path candidate = Paths.get(entry).resolve(libName);
                    if (Files.isRegularFile(candidate)) {
                        diagnostics.add("Stage 3 (java.library.path): found at " + candidate);
                        return candidate;
                    }
                }
            }
            diagnostics.add("Stage 3 (java.library.path): " + libName + " not found in directories");
        } else {
            diagnostics.add("Stage 3 (java.library.path): property empty or unset");
        }

        // Stage 4: Local Dev-Tree Fallback (walk up to 6 parent directories)
        if (enableDevSearch) {
            Path current = Paths.get("").toAbsolutePath();
            int levels = 0;
            boolean found = false;
            while (current != null && levels <= 6) {
                Path releaseCandidate = current.resolve("native").resolve("target").resolve("release").resolve(libName);
                if (Files.isRegularFile(releaseCandidate)) {
                    diagnostics.add("Stage 4 (local dev-tree): found release binary at " + releaseCandidate);
                    return releaseCandidate;
                }
                Path debugCandidate = current.resolve("native").resolve("target").resolve("debug").resolve(libName);
                if (Files.isRegularFile(debugCandidate)) {
                    diagnostics.add("Stage 4 (local dev-tree): found debug binary at " + debugCandidate);
                    return debugCandidate;
                }
                current = current.getParent();
                levels++;
            }
            diagnostics.add("Stage 4 (local dev-tree): checked up to 6 parent levels; native/target/release/" + libName + " not found");
        } else {
            diagnostics.add("Stage 4 (local dev-tree): dev search disabled");
        }

        // Stage 5: Auto On-Demand Download (Mode 3)
        String pinnedChecksum = getPinnedChecksum(classLoader, classifier);

        if (Boolean.FALSE.equals(downloadEnabled)) {
            diagnostics.add("Stage 5 (auto-download): disabled via -D" + PROPERTY_DOWNLOAD + "=false or env");
        } else if (downloadEnabled == null && pinnedChecksum == null) {
            diagnostics.add("Stage 5 (auto-download): skipped (no pinned release checksum in JAR; set -D"
                    + PROPERTY_DOWNLOAD + "=true to allow unpinned download)");
        } else {
            Path platformCache = cacheDir.resolve(version).resolve(classifier);
            Path previouslyDownloaded = findExistingCachedBinary(platformCache, libName);
            if (previouslyDownloaded != null) {
                diagnostics.add("Stage 5 (auto-download): found previously downloaded verified binary at " + previouslyDownloaded);
                return previouslyDownloaded;
            }

            String template = (downloadUrlTemplate != null && !downloadUrlTemplate.isBlank())
                    ? downloadUrlTemplate
                    : DEFAULT_URL_TEMPLATE;

            String targetUrl = template
                    .replace("{version}", version)
                    .replace("{classifier}", classifier)
                    .replace("{filename}", libName);

            boolean shouldVerify = (verifyChecksum == null || verifyChecksum);
            try {
                Path downloaded = downloadAndExtract(targetUrl, cacheDir, version, classifier, libName, platform, shouldVerify, pinnedChecksum, diagnostics);
                diagnostics.add("Stage 5 (auto-download): successfully downloaded and cached to " + downloaded);
                return downloaded;
            } catch (Exception e) {
                diagnostics.add("Stage 5 (auto-download): failed downloading from " + targetUrl + ": " + e.getMessage());
            }
        }

        // Actionable Diagnostics Error Message
        StringBuilder sb = new StringBuilder();
        sb.append("Could not load HC4J native library for platform '").append(classifier)
                .append("' (version ").append(version).append(").\n")
                .append("Stages attempted:\n");
        for (int i = 0; i < diagnostics.size(); i++) {
            sb.append("  ").append(i + 1).append(". ").append(diagnostics.get(i)).append("\n");
        }
        sb.append("\nTo resolve this issue:\n")
                .append("  - Add the native dependency for your platform to your build:\n")
                .append("      implementation 'io.github.ch-abhinav:hc4j:").append(version).append(":").append(classifier).append("'\n")
                .append("    or use the all-in-one fat bundle:\n")
                .append("      implementation 'io.github.ch-abhinav:hc4j-all:").append(version).append("'\n")
                .append("  - Or specify an explicit path to the native library:\n")
                .append("      -D").append(PROPERTY_NATIVE_PATH).append("=/path/to/").append(libName).append("\n")
                .append("  - Or enable automatic on-demand download (default unless -D").append(PROPERTY_DOWNLOAD).append("=false is set).\n");

        throw new UnsatisfiedLinkError(sb.toString());
    }

    private static Path downloadAndExtract(
            String artifactUrl,
            Path cacheDir,
            String version,
            String classifier,
            String libName,
            Platform platform,
            boolean verifyChecksum,
            String pinnedChecksum,
            List<String> diagnostics) throws IOException {

        byte[] artifactBytes;
        HttpURLConnection conn = openConnectionWithRedirects(artifactUrl, 5);
        try (InputStream in = conn.getInputStream()) {
            artifactBytes = in.readAllBytes();
        } finally {
            conn.disconnect();
        }

        if (verifyChecksum) {
            String actualSha = sha256Hex(artifactBytes);
            if (pinnedChecksum != null && !pinnedChecksum.isBlank()) {
                if (!actualSha.equalsIgnoreCase(pinnedChecksum)) {
                    throw new IOException("Downloaded artifact SHA-256 mismatch against JAR pinned digest for "
                            + classifier + ": expected " + pinnedChecksum + " but computed " + actualSha);
                }
            } else {
                String checksumUrl = artifactUrl + ".sha256";
                String expectedSha;
                try {
                    HttpURLConnection checkConn = openConnectionWithRedirects(checksumUrl, 5);
                    try (InputStream checkIn = checkConn.getInputStream()) {
                        expectedSha = parseSha256(new String(checkIn.readAllBytes(), StandardCharsets.UTF_8));
                    } finally {
                        checkConn.disconnect();
                    }
                } catch (Exception e) {
                    throw new IOException("Failed to download SHA-256 sidecar checksum from " + checksumUrl + ": " + e.getMessage(), e);
                }

                if (expectedSha.isBlank() || !actualSha.equalsIgnoreCase(expectedSha)) {
                    throw new IOException("Downloaded artifact SHA-256 mismatch from " + artifactUrl + ": expected " + expectedSha + " but computed " + actualSha);
                }
            }
        }

        byte[] nativeBytes;
        if (artifactUrl.endsWith(".jar") || isZip(artifactBytes)) {
            String entryPath = "META-INF/hc4j/natives/" + classifier + "/" + libName;
            nativeBytes = extractFromZip(artifactBytes, entryPath);
        } else {
            nativeBytes = artifactBytes;
        }

        String nativeSha256 = sha256Hex(nativeBytes);
        Path target = cacheDir.resolve(version).resolve(classifier).resolve(nativeSha256).resolve(libName);
        return extractAtomically(nativeBytes, nativeSha256, target, platform);
    }

    public static String getPinnedChecksum(ClassLoader classLoader, String classifier) {
        if (classLoader == null || classifier == null) {
            return null;
        }
        try (InputStream in = classLoader.getResourceAsStream(PINNED_CHECKSUMS_RESOURCE)) {
            if (in != null) {
                Properties props = new Properties();
                props.load(in);
                String sha = props.getProperty(classifier);
                if (sha != null && !sha.isBlank()) {
                    return sha.trim().toLowerCase();
                }
            }
        } catch (Throwable ignored) {}
        return null;
    }

    private static boolean isZip(byte[] bytes) {
        return bytes.length >= 4 && bytes[0] == 0x50 && bytes[1] == 0x4B && bytes[2] == 0x03 && bytes[3] == 0x04;
    }

    private static byte[] extractFromZip(byte[] zipBytes, String entryName) throws IOException {
        try (ZipInputStream zis = new ZipInputStream(new ByteArrayInputStream(zipBytes))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (entry.getName().equals(entryName)) {
                    return zis.readAllBytes();
                }
            }
        }
        throw new FileNotFoundException("Entry '" + entryName + "' not found in downloaded archive.");
    }

    private static Path extractAtomically(byte[] bytes, String expectedSha, Path target, Platform platform) throws IOException {
        // Fast-path cache hit
        if (Files.isRegularFile(target)) {
            try {
                String existingSha = sha256Hex(Files.readAllBytes(target));
                if (existingSha.equalsIgnoreCase(expectedSha)) {
                    return target;
                }
            } catch (IOException ignored) {}
        }

        Path parent = target.getParent();
        Files.createDirectories(parent);

        Path tmp = parent.resolve(platform.libraryFileName() + "." + ProcessHandle.current().pid() + "." + UUID.randomUUID() + ".tmp");
        try {
            Files.write(tmp, bytes);
            if (!platform.os().equals("windows")) {
                try {
                    Files.setPosixFilePermissions(tmp, PosixFilePermissions.fromString("rwxr-xr-x"));
                } catch (UnsupportedOperationException ignored) {}
            }

            try {
                Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
            }
            return target;
        } catch (IOException e) {
            // On Windows, another process may have locked the existing target file while already holding it open
            if (Files.isRegularFile(target)) {
                try {
                    String existingSha = sha256Hex(Files.readAllBytes(target));
                    if (existingSha.equalsIgnoreCase(expectedSha)) {
                        return target;
                    }
                } catch (IOException ignored) {}
            }
            throw e;
        } finally {
            try {
                Files.deleteIfExists(tmp);
            } catch (IOException ignored) {}
        }
    }

    private static Path findExistingCachedBinary(Path platformCacheDir, String libraryFileName) {
        if (!Files.isDirectory(platformCacheDir)) {
            return null;
        }
        try (Stream<Path> subdirs = Files.list(platformCacheDir)) {
            for (Path subdir : subdirs.filter(Files::isDirectory).toList()) {
                Path candidate = subdir.resolve(libraryFileName);
                if (Files.isRegularFile(candidate)) {
                    String expectedHash = subdir.getFileName().toString();
                    try {
                        String actualHash = sha256Hex(Files.readAllBytes(candidate));
                        if (actualHash.equalsIgnoreCase(expectedHash)) {
                            return candidate;
                        }
                    } catch (IOException ignored) {}
                }
            }
        } catch (IOException ignored) {}
        return null;
    }

    private static HttpURLConnection openConnectionWithRedirects(String urlStr, int maxRedirects) throws IOException {
        String currentUrl = urlStr;
        for (int i = 0; i <= maxRedirects; i++) {
            URL url = URI.create(currentUrl).toURL();
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setInstanceFollowRedirects(false);
            conn.setConnectTimeout(10000);
            conn.setReadTimeout(30000);
            conn.setRequestProperty("User-Agent", "HC4J-NativeLoader/" + getVersion());
            int code = conn.getResponseCode();
            if (code == HttpURLConnection.HTTP_MOVED_PERM
                    || code == HttpURLConnection.HTTP_MOVED_TEMP
                    || code == 307
                    || code == 308) {
                String location = conn.getHeaderField("Location");
                conn.disconnect();
                if (location == null || location.isBlank()) {
                    throw new IOException("HTTP redirect without Location header from " + currentUrl);
                }
                currentUrl = URI.create(currentUrl).resolve(location).toString();
                continue;
            }
            if (code >= 200 && code < 300) {
                return conn;
            }
            conn.disconnect();
            throw new IOException("HTTP request to " + currentUrl + " failed with status " + code);
        }
        throw new IOException("Too many redirects (max " + maxRedirects + ") for " + urlStr);
    }

    public static String getVersion() {
        try (InputStream in = NativeLoader.class.getResourceAsStream("/META-INF/hc4j/version.properties")) {
            if (in != null) {
                Properties props = new Properties();
                props.load(in);
                String v = props.getProperty("version");
                if (v != null && !v.isBlank()) return v.trim();
            }
        } catch (Throwable ignored) {}
        Package pkg = NativeLoader.class.getPackage();
        if (pkg != null) {
            String v = pkg.getImplementationVersion();
            if (v != null && !v.isBlank()) return v.trim();
        }
        return "0.1.0-dev";
    }

    public static Path resolveDefaultCacheDir(Platform platform) {
        String custom = System.getProperty(PROPERTY_CACHE_DIR);
        if (custom == null || custom.isBlank()) {
            custom = System.getenv(ENV_CACHE_DIR);
        }
        if (custom != null && !custom.isBlank()) {
            return Paths.get(custom).toAbsolutePath();
        }
        String userHome = System.getProperty("user.home", ".");
        return switch (platform.os()) {
            case "windows" -> {
                String localAppData = System.getenv("LOCALAPPDATA");
                yield (localAppData != null && !localAppData.isBlank())
                        ? Paths.get(localAppData, "hc4j", "natives")
                        : Paths.get(userHome, "AppData", "Local", "hc4j", "natives");
            }
            case "macos" -> Paths.get(userHome, "Library", "Caches", "hc4j", "natives");
            case "linux" -> {
                String xdg = System.getenv("XDG_CACHE_HOME");
                yield (xdg != null && !xdg.isBlank())
                        ? Paths.get(xdg, "hc4j", "natives")
                        : Paths.get(userHome, ".cache", "hc4j", "natives");
            }
            default -> Paths.get(System.getProperty("java.io.tmpdir", "."), "hc4j-natives");
        };
    }

    public static String sha256Hex(byte[] bytes) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(bytes);
            return bytesToHex(digest);
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError("SHA-256 must be supported by Java runtime", e);
        }
    }

    public static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            sb.append(Character.forDigit((b >> 4) & 0xF, 16));
            sb.append(Character.forDigit(b & 0xF, 16));
        }
        return sb.toString();
    }

    private static String parseSha256(String raw) {
        if (raw == null) return "";
        String line = raw.lines().findFirst().orElse("").trim();
        int spaceIdx = line.indexOf(' ');
        if (spaceIdx > 0) {
            line = line.substring(0, spaceIdx).trim();
        }
        return line.toLowerCase();
    }

    private static Boolean parseBoolean(String propVal, String envVal) {
        String val = (propVal != null && !propVal.isBlank()) ? propVal : envVal;
        if (val == null || val.isBlank()) return null;
        return Boolean.parseBoolean(val.trim());
    }
}
