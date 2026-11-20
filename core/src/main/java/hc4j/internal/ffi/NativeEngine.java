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

import hc4j.engine.EngineStats;
import hc4j.engine.GpuOutOfMemoryException;
import hc4j.engine.MemoryStats;
import hc4j.engine.Residency;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Low-level native engine bridge: library lookup, initialization, memory lifecycle,
 * and command stream controls.
 */
public final class NativeEngine {

    public static final int HC4J_SUCCESS = 0;
    public static final int HC4J_ERR_NOT_FOUND = -1;
    public static final int HC4J_ERR_INVALID_PARAM = -2;
    public static final int HC4J_ERR_GPU_READBACK = -3;
    public static final int HC4J_ERR_OUT_OF_MEMORY = -4;
    public static final int HC4J_ERR_DEVICE = -5;
    public static final int HC4J_ERR_GPU_VALIDATION = -6;
    public static final int HC4J_ERR_HOST_IO = -7;
    public static final int HC4J_ERR_UNSUPPORTED = -8;

    private static volatile boolean initialized = false;
    private static volatile Throwable initFailure = null;

    private static MethodHandle initGpuHandle;
    private static MethodHandle gpuAllocHandle;
    private static MethodHandle gpuAllocUninitHandle;
    private static MethodHandle gpuWriteHandle;
    private static MethodHandle gpuFreeHandle;
    private static MethodHandle gpuDownloadHandle;
    private static MethodHandle memConfigureHandle;
    private static MethodHandle memStatsHandle;
    private static MethodHandle memResidencyHandle;
    private static MethodHandle memEvictHandle;
    private static MethodHandle batchBeginHandle;
    private static MethodHandle batchEndHandle;
    private static MethodHandle engineStatsHandle;
    private static MethodHandle synchronizeHandle;

    // Collective handles
    private static MethodHandle collCreateHandle;
    private static MethodHandle collDestroyHandle;
    private static MethodHandle collRanksHandle;
    private static MethodHandle collLinkCountHandle;
    private static MethodHandle collLinkHandle;
    private static MethodHandle collDeviceNameHandle;
    private static MethodHandle collAllocHandle;
    private static MethodHandle collFreeHandle;
    private static MethodHandle collWriteHandle;
    private static MethodHandle collReadHandle;
    private static MethodHandle collAllReduceHandle;

    private NativeEngine() {
        throw new AssertionError("No instances of NativeEngine");
    }

    public static synchronized boolean isGpuAvailable() {
        try {
            ensureInitialized();
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    public static synchronized void initialize() {
        ensureInitialized();
    }

    public static SymbolLookup nativeSymbols() {
        ensureInitialized();
        return SymbolLookup.loaderLookup();
    }

    public static synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }
        if (initFailure != null) {
            if (initFailure instanceof RuntimeException re) throw re;
            if (initFailure instanceof Error err) throw err;
            throw new RuntimeException("GPU initialization previously failed", initFailure);
        }
        try {
            loadLibrary();
            bindCoreHandles();
            int status = (int) initGpuHandle.invokeExact();
            checkStatus(status, "initGpu");
            initialized = true;
        } catch (Throwable t) {
            initFailure = t;
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("GPU initialization failed", t);
        }
    }

    private static void loadLibrary() {
        String os = System.getProperty("os.name").toLowerCase();
        String extension = os.contains("win") ? ".dll" : (os.contains("mac") ? ".dylib" : ".so");
        String libName = (os.contains("mac") || os.contains("linux"))
                ? "libhc4j" + extension
                : "hc4j" + extension;

        Path libPath = findNativeLibrary(libName);
        System.load(libPath.toString());
    }

    private static Path findNativeLibrary(String libName) {
        String customPath = System.getProperty("hc4j.native.path");
        if (customPath != null && !customPath.isBlank()) {
            Path p = Paths.get(customPath).toAbsolutePath();
            if (Files.isDirectory(p)) {
                Path candidate = p.resolve(libName);
                if (Files.exists(candidate)) {
                    return candidate;
                }
            } else if (Files.exists(p)) {
                return p;
            }
        }

        Path current = Paths.get("").toAbsolutePath();
        while (current != null) {
            Path releasePath = current.resolve(Paths.get("native", "target", "release", libName));
            if (Files.exists(releasePath)) {
                return releasePath;
            }
            Path debugPath = current.resolve(Paths.get("native", "target", "debug", libName));
            if (Files.exists(debugPath)) {
                return debugPath;
            }
            Path rootPath = current.resolve(libName);
            if (Files.exists(rootPath)) {
                return rootPath;
            }
            current = current.getParent();
        }

        throw new UnsatisfiedLinkError("Could not find " + libName
                + " anywhere in the project tree. Did you run 'cargo build --release' in the native folder?");
    }

    private static void bindCoreHandles() {
        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = SymbolLookup.loaderLookup();

        initGpuHandle = linker.downcallHandle(lookup.find("hc4j_init_gpu").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT));

        gpuAllocHandle = linker.downcallHandle(lookup.find("hc4j_gpu_alloc").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));

        gpuAllocUninitHandle = linker.downcallHandle(lookup.find("hc4j_gpu_alloc_uninit").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));

        gpuWriteHandle = linker.downcallHandle(lookup.find("hc4j_gpu_write").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        gpuFreeHandle = linker.downcallHandle(lookup.find("hc4j_gpu_free").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));

        gpuDownloadHandle = linker.downcallHandle(lookup.find("hc4j_gpu_download").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        memConfigureHandle = linker.downcallHandle(lookup.find("hc4j_mem_configure").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));

        memStatsHandle = linker.downcallHandle(lookup.find("hc4j_mem_stats").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        memResidencyHandle = linker.downcallHandle(lookup.find("hc4j_mem_residency").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));

        memEvictHandle = linker.downcallHandle(lookup.find("hc4j_mem_evict").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));

        batchBeginHandle = linker.downcallHandle(lookup.find("hc4j_batch_begin").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT));

        batchEndHandle = linker.downcallHandle(lookup.find("hc4j_batch_end").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT));

        engineStatsHandle = linker.downcallHandle(lookup.find("hc4j_engine_stats").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        synchronizeHandle = linker.downcallHandle(lookup.find("hc4j_synchronize").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT));

        // Collective handles
        collCreateHandle = linker.downcallHandle(lookup.find("hc4j_collective_create").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        collDestroyHandle = linker.downcallHandle(lookup.find("hc4j_collective_destroy").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));

        collRanksHandle = linker.downcallHandle(lookup.find("hc4j_collective_ranks").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));

        collLinkCountHandle = linker.downcallHandle(lookup.find("hc4j_collective_link_count").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));

        collLinkHandle = linker.downcallHandle(lookup.find("hc4j_collective_link").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        collDeviceNameHandle = linker.downcallHandle(lookup.find("hc4j_collective_device_name").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        collAllocHandle = linker.downcallHandle(lookup.find("hc4j_collective_alloc").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));

        collFreeHandle = linker.downcallHandle(lookup.find("hc4j_collective_free").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));

        collWriteHandle = linker.downcallHandle(lookup.find("hc4j_collective_write").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        collReadHandle = linker.downcallHandle(lookup.find("hc4j_collective_read").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        collAllReduceHandle = linker.downcallHandle(lookup.find("hc4j_collective_allreduce_sum").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
    }

    public static void checkStatus(int statusCode, String context) {
        if (statusCode == HC4J_SUCCESS) return;
        switch (statusCode) {
            case HC4J_ERR_NOT_FOUND -> throw new IllegalStateException(context + ": GPU Buffer ID not found in Rust Registry!");
            case HC4J_ERR_INVALID_PARAM -> throw new IllegalArgumentException(context + ": Invalid parameters passed to native bridge.");
            case HC4J_ERR_GPU_READBACK -> throw new RuntimeException(context + ": GPU Readback failed during memory mapping.");
            case HC4J_ERR_OUT_OF_MEMORY -> throw new GpuOutOfMemoryException(context + ": VRAM, host RAM and the spill tier are all exhausted.");
            case HC4J_ERR_DEVICE -> throw new IllegalStateException(context + ": GPU device unavailable, lost, or timed out.");
            case HC4J_ERR_GPU_VALIDATION -> throw new IllegalStateException(context + ": wgpu rejected the shader, pipeline or submission (see native log).");
            case HC4J_ERR_HOST_IO -> throw new RuntimeException(context + ": Spill-file I/O failed.");
            case HC4J_ERR_UNSUPPORTED -> throw new UnsupportedOperationException(context + ": Operation not supported for this tensor layout or size.");
            default -> throw new RuntimeException(context + ": Unknown native FFM error code: " + statusCode);
        }
    }

    public static long allocVram(long totalElements) {
        ensureInitialized();
        try {
            long id = (long) gpuAllocHandle.invokeExact(totalElements);
            if (id == 0) throw new OutOfMemoryError("Rust failed to allocate VRAM buffer.");
            return id;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J VRAM Allocation Exception", t);
        }
    }

    public static long allocVramUninit(long totalElements) {
        ensureInitialized();
        long id;
        try {
            id = (long) gpuAllocUninitHandle.invokeExact(totalElements);
        } catch (Throwable t) {
            throw new RuntimeException("HC4J VRAM Allocation Exception", t);
        }
        if (id == 0) {
            throw new GpuOutOfMemoryException("allocVramUninit: could not allocate " + totalElements + " elements in any tier");
        }
        return id;
    }

    public static void writeVram(long vramId, MemorySegment hostData, long totalElements) {
        ensureInitialized();
        try {
            int status = (int) gpuWriteHandle.invokeExact(vramId, hostData, totalElements);
            checkStatus(status, "writeVram");
        } catch (Throwable t) {
            throw new RuntimeException("HC4J VRAM Write Exception", t);
        }
    }

    public static void downloadVram(long vramId, MemorySegment hostData, long totalElements) {
        ensureInitialized();
        try {
            int status = (int) gpuDownloadHandle.invokeExact(vramId, hostData, totalElements);
            checkStatus(status, "downloadVram");
        } catch (Throwable t) {
            throw new RuntimeException("HC4J VRAM Download Exception", t);
        }
    }

    public static void freeVram(long vramId) {
        if (vramId == 0 || !initialized) return;
        try {
            int status = (int) gpuFreeHandle.invokeExact(vramId);
            checkStatus(status, "freeVram");
        } catch (Throwable t) {
            System.err.println("Warning: Failed to free VRAM handle " + vramId + ": " + t.getMessage());
        }
    }

    public static void configureMemory(long vramBudgetBytes, long hostBudgetBytes) {
        ensureInitialized();
        if (vramBudgetBytes < 0 || hostBudgetBytes < 0) {
            throw new IllegalArgumentException("Budgets must be >= 0 (0 = unchanged)");
        }
        int status;
        try {
            status = (int) memConfigureHandle.invokeExact(vramBudgetBytes, hostBudgetBytes);
        } catch (Throwable t) {
            throw new RuntimeException("HC4J memory configure failed", t);
        }
        checkStatus(status, "configureMemory");
    }

    public static MemoryStats memoryStats() {
        ensureInitialized();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(MemoryStats.LAYOUT);
            int status;
            try {
                status = (int) memStatsHandle.invokeExact(out);
            } catch (Throwable t) {
                throw new RuntimeException("HC4J memory stats failed", t);
            }
            checkStatus(status, "memoryStats");
            return MemoryStats.read(out);
        }
    }

    public static Residency residency(long vramId) {
        ensureInitialized();
        int code;
        try {
            code = (int) memResidencyHandle.invokeExact(vramId);
        } catch (Throwable t) {
            throw new RuntimeException("HC4J residency query failed", t);
        }
        if (code < 0) checkStatus(code, "residency");
        return switch (code) {
            case 0 -> Residency.DEVICE;
            case 1 -> Residency.HOST;
            case 2 -> Residency.DISK;
            default -> throw new IllegalStateException("Unknown residency code " + code);
        };
    }

    public static void evict(long vramId) {
        ensureInitialized();
        int status;
        try {
            status = (int) memEvictHandle.invokeExact(vramId);
        } catch (Throwable t) {
            throw new RuntimeException("HC4J evict failed", t);
        }
        checkStatus(status, "evict");
    }

    public static void batchBegin() {
        ensureInitialized();
        int status;
        try {
            status = (int) batchBeginHandle.invokeExact();
        } catch (Throwable t) {
            throw new RuntimeException("HC4J batch begin failed", t);
        }
        checkStatus(status, "batchBegin");
    }

    public static void batchEnd() {
        ensureInitialized();
        int status;
        try {
            status = (int) batchEndHandle.invokeExact();
        } catch (Throwable t) {
            throw new RuntimeException("HC4J batch end failed", t);
        }
        checkStatus(status, "batchEnd");
    }

    public static EngineStats engineStats() {
        ensureInitialized();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(EngineStats.LAYOUT);
            int status;
            try {
                status = (int) engineStatsHandle.invokeExact(out);
            } catch (Throwable t) {
                throw new RuntimeException("HC4J engine stats failed", t);
            }
            checkStatus(status, "engineStats");
            return EngineStats.read(out);
        }
    }

    public static void synchronize() {
        ensureInitialized();
        int status;
        try {
            status = (int) synchronizeHandle.invokeExact();
        } catch (Throwable t) {
            throw new RuntimeException("HC4J synchronize failed", t);
        }
        checkStatus(status, "synchronize");
    }

    // --- Collective Operations ---

    public static long createCollective(int ranks) {
        ensureInitialized();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ValueLayout.JAVA_LONG);
            int status = (int) collCreateHandle.invokeExact(ranks, out);
            checkStatus(status, "Collective.open");
            return out.get(ValueLayout.JAVA_LONG, 0);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J collective creation failed", t);
        }
    }

    public static void destroyCollective(long handle) {
        if (handle == 0 || !initialized) return;
        try {
            int status = (int) collDestroyHandle.invokeExact(handle);
            checkStatus(status, "Collective.destroy");
        } catch (Throwable t) {
            System.err.println("Warning: Failed to destroy collective handle " + handle + ": " + t.getMessage());
        }
    }

    public static int collectiveRanks(long handle) {
        ensureInitialized();
        try {
            int ranks = (int) collRanksHandle.invokeExact(handle);
            if (ranks < 0) checkStatus(ranks, "Collective.ranks");
            return ranks;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J collective ranks query failed", t);
        }
    }

    public static int collectiveLinkCount(long handle) {
        ensureInitialized();
        try {
            int count = (int) collLinkCountHandle.invokeExact(handle);
            if (count < 0) checkStatus(count, "Collective.topology");
            return count;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J collective link count query failed", t);
        }
    }

    public static void collectiveLink(long handle, int index, MemorySegment out) {
        ensureInitialized();
        try {
            int status = (int) collLinkHandle.invokeExact(handle, index, out);
            checkStatus(status, "Collective.topology");
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J collective link query failed", t);
        }
    }

    public static int collectiveDeviceName(long handle, int rank, MemorySegment buf, int cap) {
        ensureInitialized();
        try {
            int len = (int) collDeviceNameHandle.invokeExact(handle, rank, buf, cap);
            if (len < 0) checkStatus(len, "Collective.deviceName");
            return len;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J collective device name query failed", t);
        }
    }

    public static long allocCollectiveBuffer(long handle, int rank, long elements) {
        ensureInitialized();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ValueLayout.JAVA_LONG);
            int status = (int) collAllocHandle.invokeExact(handle, rank, elements, out);
            checkStatus(status, "Collective.allocate");
            return out.get(ValueLayout.JAVA_LONG, 0);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J collective alloc failed", t);
        }
    }

    public static void freeCollectiveBuffer(long handle, long buffer) {
        if (handle == 0 || buffer == 0 || !initialized) return;
        try {
            int status = (int) collFreeHandle.invokeExact(handle, buffer);
            checkStatus(status, "CollectiveBuffer.close");
        } catch (Throwable t) {
            System.err.println("Warning: Failed to free collective buffer " + buffer + ": " + t.getMessage());
        }
    }

    public static void writeCollectiveBuffer(long handle, long buffer, MemorySegment data, long elements) {
        ensureInitialized();
        try {
            int status = (int) collWriteHandle.invokeExact(handle, buffer, data, elements);
            checkStatus(status, "CollectiveBuffer.write");
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J collective write failed", t);
        }
    }

    public static void readCollectiveBuffer(long handle, long buffer, MemorySegment out, long elements) {
        ensureInitialized();
        try {
            int status = (int) collReadHandle.invokeExact(handle, buffer, out, elements);
            checkStatus(status, "CollectiveBuffer.read");
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J collective read failed", t);
        }
    }

    public static void allReduceSum(long handle, MemorySegment buffers, int count) {
        ensureInitialized();
        try {
            int status = (int) collAllReduceHandle.invokeExact(handle, buffers, count);
            checkStatus(status, "Collective.allReduceSum");
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J collective allReduceSum failed", t);
        }
    }
}
