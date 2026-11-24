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

import hc4j.engine.AdapterInfo;
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
import java.nio.charset.StandardCharsets;
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
    private static volatile NativeLoadResult loadResult = null;

    private static MethodHandle initGpuHandle;
    private static MethodHandle gpuAllocHandle;
    private static MethodHandle gpuAllocBytesHandle;
    private static MethodHandle gpuAllocUninitHandle;
    private static MethodHandle gpuWriteHandle;
    private static MethodHandle gpuWriteBytesHandle;
    private static MethodHandle gpuFreeHandle;
    private static MethodHandle gpuDownloadHandle;
    private static MethodHandle gpuDownloadBytesHandle;
    private static MethodHandle memConfigureHandle;
    private static MethodHandle memStatsHandle;
    private static MethodHandle memResidencyHandle;
    private static MethodHandle memEvictHandle;
    private static MethodHandle batchBeginHandle;
    private static MethodHandle batchEndHandle;
    private static MethodHandle engineStatsHandle;
    private static MethodHandle synchronizeHandle;
    private static MethodHandle adapterInfoHandle;

    // Collective handles
    private static MethodHandle collCreateHandle;
    private static MethodHandle collDestroyHandle;
    private static MethodHandle collRanksHandle;
    private static MethodHandle collLinkCountHandle;
    private static MethodHandle collLinkHandle;
    private static MethodHandle collDeviceNameHandle;
    private static MethodHandle collBackendNameHandle;
    private static MethodHandle collAllocHandle;
    private static MethodHandle collFreeHandle;
    private static MethodHandle collWriteHandle;
    private static MethodHandle collReadHandle;
    private static MethodHandle collAllReduceHandle;
    private static MethodHandle collDeviceProfileHandle;
    private static MethodHandle collRingOrderHandle;
    private static MethodHandle collIslandCountHandle;
    private static MethodHandle collIslandHandle;
    private static MethodHandle collAllReduceSubsetHandle;
    private static MethodHandle collGemmBiasActHandle;
    private static MethodHandle collSplitKvAttnHandle;

    private NativeEngine() {
        throw new AssertionError("No instances of NativeEngine");
    }

    public static boolean isGpuAvailable() {
        try {
            ensureInitialized();
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    public static void initialize() {
        ensureInitialized();
    }

    public static SymbolLookup nativeSymbols() {
        ensureInitialized();
        return loadResult.lookup();
    }

    public static Path loadedLibraryPath() {
        ensureInitialized();
        return loadResult.libraryPath();
    }

    public static String loadSource() {
        ensureInitialized();
        return loadResult.source();
    }

    public static Platform platform() {
        ensureInitialized();
        return loadResult.platform();
    }

    public static void ensureInitialized() {
        if (initialized) {
            return;
        }
        initializeLocked();
    }

    private static synchronized void initializeLocked() {
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
        loadResult = NativeLoader.load();
    }

    @SuppressWarnings("restricted")
    private static void bindCoreHandles() {
        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = loadResult.lookup();

        initGpuHandle = linker.downcallHandle(lookup.find("hc4j_init_gpu").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT));

        gpuAllocHandle = linker.downcallHandle(lookup.find("hc4j_gpu_alloc").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));

        gpuAllocBytesHandle = linker.downcallHandle(lookup.find("hc4j_gpu_alloc_bytes").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));

        gpuAllocUninitHandle = linker.downcallHandle(lookup.find("hc4j_gpu_alloc_uninit").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));

        gpuWriteHandle = linker.downcallHandle(lookup.find("hc4j_gpu_write").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        gpuWriteBytesHandle = linker.downcallHandle(lookup.find("hc4j_gpu_write_bytes").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        gpuFreeHandle = linker.downcallHandle(lookup.find("hc4j_gpu_free").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));

        gpuDownloadHandle = linker.downcallHandle(lookup.find("hc4j_gpu_download").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        gpuDownloadBytesHandle = linker.downcallHandle(lookup.find("hc4j_gpu_download_bytes").orElseThrow(),
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

        adapterInfoHandle = linker.downcallHandle(lookup.find("hc4j_adapter_info").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

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

        collBackendNameHandle = linker.downcallHandle(lookup.find("hc4j_collective_backend_name").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

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

        collDeviceProfileHandle = linker.downcallHandle(lookup.find("hc4j_collective_device_profile").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        collRingOrderHandle = linker.downcallHandle(lookup.find("hc4j_collective_ring_order").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        collIslandCountHandle = linker.downcallHandle(lookup.find("hc4j_collective_island_count").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));

        collIslandHandle = linker.downcallHandle(lookup.find("hc4j_collective_island").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        collAllReduceSubsetHandle = linker.downcallHandle(lookup.find("hc4j_collective_allreduce_sum_subset").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        collGemmBiasActHandle = linker.downcallHandle(lookup.find("hc4j_collective_gemm_bias_act").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        collSplitKvAttnHandle = linker.downcallHandle(lookup.find("hc4j_collective_split_kv_attn").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
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
        long id;
        try {
            id = (long) gpuAllocHandle.invokeExact(totalElements);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J VRAM Allocation Exception", t);
        }
        if (id == 0) throw new OutOfMemoryError("Rust failed to allocate VRAM buffer.");
        return id;
    }

    public static long allocVramUninit(long totalElements) {
        ensureInitialized();
        long id;
        try {
            id = (long) gpuAllocUninitHandle.invokeExact(totalElements);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J VRAM Allocation Exception", t);
        }
        if (id == 0) {
            throw new GpuOutOfMemoryException("allocVramUninit: could not allocate " + totalElements + " elements in any tier");
        }
        return id;
    }

    public static long allocVramBytes(long bytes, boolean zeroed) {
        ensureInitialized();
        if (bytes <= 0) {
            throw new IllegalArgumentException("Allocation byte size must be positive: " + bytes);
        }
        long id;
        try {
            id = (long) gpuAllocBytesHandle.invokeExact(bytes, zeroed ? 1 : 0);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("allocVramBytes failed", t);
        }
        if (id == 0) {
            throw new GpuOutOfMemoryException("allocVramBytes: could not allocate " + bytes + " bytes in any tier");
        }
        return id;
    }

    public static void writeVramBytesSegment(long id, MemorySegment segment, long bytes) {
        ensureInitialized();
        if (id == 0) {
            throw new IllegalArgumentException("Invalid VRAM handle: 0");
        }
        if (bytes < 0) {
            throw new IllegalArgumentException("Negative byte count: " + bytes);
        }
        if (bytes == 0) {
            return;
        }
        int status;
        try {
            status = (int) gpuWriteBytesHandle.invokeExact(id, segment, bytes);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("writeVramBytesSegment failed", t);
        }
        checkStatus(status, "writeVramBytesSegment");
    }

    public static void downloadVramBytesSegment(long id, MemorySegment segment, long bytes) {
        ensureInitialized();
        if (id == 0) {
            throw new IllegalArgumentException("Invalid VRAM handle: 0");
        }
        if (bytes < 0) {
            throw new IllegalArgumentException("Negative byte count: " + bytes);
        }
        if (bytes == 0) {
            return;
        }
        int status;
        try {
            status = (int) gpuDownloadBytesHandle.invokeExact(id, segment, bytes);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("downloadVramBytesSegment failed", t);
        }
        checkStatus(status, "downloadVramBytesSegment");
    }

    public static void writeVram(long vramId, MemorySegment hostData, long totalElements) {
        ensureInitialized();
        int status;
        try {
            status = (int) gpuWriteHandle.invokeExact(vramId, hostData, totalElements);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J VRAM Write Exception", t);
        }
        checkStatus(status, "writeVram");
    }

    public static void downloadVram(long vramId, MemorySegment hostData, long totalElements) {
        ensureInitialized();
        int status;
        try {
            status = (int) gpuDownloadHandle.invokeExact(vramId, hostData, totalElements);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J VRAM Download Exception", t);
        }
        checkStatus(status, "downloadVram");
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
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
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
                if (t instanceof RuntimeException re) throw re;
                if (t instanceof Error err) throw err;
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
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
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
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J evict failed", t);
        }
        checkStatus(status, "evict");
    }

    private static final Object BATCH_LOCK = new Object();
    private static int batchDepth = 0;

    public static void batchBegin() {
        synchronized (BATCH_LOCK) {
            if (batchDepth++ == 0) {
                ensureInitialized();
                int status;
                try {
                    status = (int) batchBeginHandle.invokeExact();
                } catch (Throwable t) {
                    batchDepth--;
                    if (t instanceof RuntimeException re) throw re;
                    if (t instanceof Error err) throw err;
                    throw new RuntimeException("HC4J batch begin failed", t);
                }
                checkStatus(status, "batchBegin");
            }
        }
    }

    public static void batchEnd() {
        synchronized (BATCH_LOCK) {
            if (batchDepth <= 0) {
                return;
            }
            if (--batchDepth == 0) {
                ensureInitialized();
                int status;
                try {
                    status = (int) batchEndHandle.invokeExact();
                } catch (Throwable t) {
                    if (t instanceof RuntimeException re) throw re;
                    if (t instanceof Error err) throw err;
                    throw new RuntimeException("HC4J batch end failed", t);
                }
                checkStatus(status, "batchEnd");
            }
        }
    }

    public static void writeVramSegment(long vramId, MemorySegment hostData, long totalElements) {
        ensureInitialized();
        if (hostData.isNative()) {
            int status;
            try {
                status = (int) gpuWriteHandle.invokeExact(vramId, hostData, totalElements);
            } catch (Throwable t) {
                if (t instanceof RuntimeException re) throw re;
                if (t instanceof Error err) throw err;
                throw new RuntimeException("HC4J VRAM Write Exception", t);
            }
            checkStatus(status, "writeVramSegment");
        } else {
            long bytes = totalElements * 4L;
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeBuf = arena.allocate(bytes, 16);
                MemorySegment.copy(hostData, 0, nativeBuf, 0, bytes);
                int status;
                try {
                    status = (int) gpuWriteHandle.invokeExact(vramId, nativeBuf, totalElements);
                } catch (Throwable t) {
                    if (t instanceof RuntimeException re) throw re;
                    if (t instanceof Error err) throw err;
                    throw new RuntimeException("HC4J VRAM Write Exception", t);
                }
                checkStatus(status, "writeVramSegment");
            } catch (Throwable t) {
                if (t instanceof RuntimeException re) throw re;
                if (t instanceof Error err) throw err;
                throw new RuntimeException("HC4J VRAM Write Exception", t);
            }
        }
    }

    public static void downloadVramSegment(long vramId, MemorySegment destination, long totalElements) {
        ensureInitialized();
        if (destination.isNative()) {
            int status;
            try {
                status = (int) gpuDownloadHandle.invokeExact(vramId, destination, totalElements);
            } catch (Throwable t) {
                if (t instanceof RuntimeException re) throw re;
                if (t instanceof Error err) throw err;
                throw new RuntimeException("HC4J VRAM Download Exception", t);
            }
            checkStatus(status, "downloadVramSegment");
        } else {
            long bytes = totalElements * 4L;
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeBuf = arena.allocate(bytes, 16);
                int status;
                try {
                    status = (int) gpuDownloadHandle.invokeExact(vramId, nativeBuf, totalElements);
                } catch (Throwable t) {
                    if (t instanceof RuntimeException re) throw re;
                    if (t instanceof Error err) throw err;
                    throw new RuntimeException("HC4J VRAM Download Exception", t);
                }
                checkStatus(status, "downloadVramSegment");
                MemorySegment.copy(nativeBuf, 0, destination, 0, bytes);
            } catch (Throwable t) {
                if (t instanceof RuntimeException re) throw re;
                if (t instanceof Error err) throw err;
                throw new RuntimeException("HC4J VRAM Download Exception", t);
            }
        }
    }

    public static void copyStrided(long src, long out, long srcOffset, int rank, int[] shape, int[] srcStrides, long len) {
        ensureInitialized();
        if (len == 0) return;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shapeSeg = rank > 0 ? arena.allocateFrom(ValueLayout.JAVA_INT, shape) : MemorySegment.NULL;
            MemorySegment stridesSeg = rank > 0 ? arena.allocateFrom(ValueLayout.JAVA_INT, srcStrides) : MemorySegment.NULL;
            int status = NativeBindings.dispatchCopyStrided(src, out, srcOffset, rank, shapeSeg, stridesSeg, len);
            checkStatus(status, "copyStrided");
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J copyStrided failed", t);
        }
    }

    public static EngineStats engineStats() {
        ensureInitialized();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(EngineStats.LAYOUT);
            int status;
            try {
                status = (int) engineStatsHandle.invokeExact(out);
            } catch (Throwable t) {
                if (t instanceof RuntimeException re) throw re;
                if (t instanceof Error err) throw err;
                throw new RuntimeException("HC4J engine stats failed", t);
            }
            checkStatus(status, "engineStats");
            return EngineStats.read(out);
        }
    }

    public static AdapterInfo adapterInfo() {
        ensureInitialized();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(NativeBindings.ADAPTER_INFO_LAYOUT);
            int status;
            try {
                status = (int) adapterInfoHandle.invokeExact(seg);
            } catch (Throwable t) {
                if (t instanceof RuntimeException re) throw re;
                if (t instanceof Error err) throw err;
                throw new RuntimeException("HC4J adapter info query failed", t);
            }
            checkStatus(status, "adapterInfo");

            int vendorId = seg.get(ValueLayout.JAVA_INT, NativeBindings.ADAPTER_INFO_VENDOR_ID_OFFSET);
            int deviceId = seg.get(ValueLayout.JAVA_INT, NativeBindings.ADAPTER_INFO_DEVICE_ID_OFFSET);
            int deviceTypeCode = seg.get(ValueLayout.JAVA_INT, NativeBindings.ADAPTER_INFO_DEVICE_TYPE_OFFSET);
            int backendCode = seg.get(ValueLayout.JAVA_INT, NativeBindings.ADAPTER_INFO_BACKEND_OFFSET);
            int nameLen = seg.get(ValueLayout.JAVA_INT, NativeBindings.ADAPTER_INFO_NAME_LEN_OFFSET);
            int driverLen = seg.get(ValueLayout.JAVA_INT, NativeBindings.ADAPTER_INFO_DRIVER_LEN_OFFSET);
            int driverInfoLen = seg.get(ValueLayout.JAVA_INT, NativeBindings.ADAPTER_INFO_DRIVER_INFO_LEN_OFFSET);

            byte[] nameBytes = new byte[nameLen];
            MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, NativeBindings.ADAPTER_INFO_NAME_OFFSET, nameBytes, 0, nameLen);
            String name = new String(nameBytes, StandardCharsets.UTF_8);

            byte[] driverBytes = new byte[driverLen];
            MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, NativeBindings.ADAPTER_INFO_DRIVER_OFFSET, driverBytes, 0, driverLen);
            String driver = new String(driverBytes, StandardCharsets.UTF_8);

            byte[] driverInfoBytes = new byte[driverInfoLen];
            MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, NativeBindings.ADAPTER_INFO_DRIVER_INFO_OFFSET, driverInfoBytes, 0, driverInfoLen);
            String driverInfo = new String(driverInfoBytes, StandardCharsets.UTF_8);

            String deviceType = switch (deviceTypeCode) {
                case 1 -> "IntegratedGpu";
                case 2 -> "DiscreteGpu";
                case 3 -> "VirtualGpu";
                case 4 -> "Cpu";
                default -> "Other";
            };

            String backend = switch (backendCode) {
                case 1 -> "Vulkan";
                case 2 -> "Metal";
                case 3 -> "Dx12";
                case 4 -> "Gl";
                case 5 -> "BrowserWebGpu";
                default -> "Empty";
            };

            return new AdapterInfo(name, vendorId, deviceId, deviceType, backend, driver, driverInfo);
        }
    }

    public static void synchronize() {
        ensureInitialized();
        int status;
        try {
            status = (int) synchronizeHandle.invokeExact();
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
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
            if (t instanceof Error err) throw err;
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
            if (t instanceof Error err) throw err;
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
            if (t instanceof Error err) throw err;
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
            if (t instanceof Error err) throw err;
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
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J collective device name query failed", t);
        }
    }

    public static int collectiveBackendName(long handle, MemorySegment buf, int cap) {
        ensureInitialized();
        try {
            int len = (int) collBackendNameHandle.invokeExact(handle, buf, cap);
            if (len < 0) checkStatus(len, "Collective.backendName");
            return len;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J collective backend name query failed", t);
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
            if (t instanceof Error err) throw err;
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
            if (t instanceof Error err) throw err;
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
            if (t instanceof Error err) throw err;
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
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J collective allReduceSum failed", t);
        }
    }

    public static void collectiveDeviceProfile(long handle, int rank, MemorySegment out) {
        ensureInitialized();
        try {
            int status = (int) collDeviceProfileHandle.invokeExact(handle, rank, out);
            checkStatus(status, "Collective.deviceProfiles");
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J collective device profile query failed", t);
        }
    }

    public static int collectiveRingOrder(long handle, MemorySegment outRanks, int cap) {
        ensureInitialized();
        try {
            int count = (int) collRingOrderHandle.invokeExact(handle, outRanks, cap);
            if (count < 0) checkStatus(count, "Collective.ringOrder");
            return count;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J collective ring order query failed", t);
        }
    }

    public static int collectiveIslandCount(long handle) {
        ensureInitialized();
        try {
            int count = (int) collIslandCountHandle.invokeExact(handle);
            if (count < 0) checkStatus(count, "Collective.islands");
            return count;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J collective island count query failed", t);
        }
    }

    public static int collectiveIsland(long handle, int index, MemorySegment outHeader, MemorySegment outRanks, int cap) {
        ensureInitialized();
        try {
            int written = (int) collIslandHandle.invokeExact(handle, index, outHeader, outRanks, cap);
            if (written < 0) checkStatus(written, "Collective.islands");
            return written;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J collective island query failed", t);
        }
    }

    public static void writeCollectiveBufferSegment(long handle, long buffer, MemorySegment data, long elements) {
        ensureInitialized();
        if (data.isNative()) {
            try {
                int status = (int) collWriteHandle.invokeExact(handle, buffer, data, elements);
                checkStatus(status, "CollectiveBuffer.write");
            } catch (Throwable t) {
                if (t instanceof RuntimeException re) throw re;
                if (t instanceof Error err) throw err;
                throw new RuntimeException("HC4J collective write failed", t);
            }
        } else {
            long bytes = elements * 4L;
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeBuf = arena.allocate(bytes, 16);
                MemorySegment.copy(data, 0, nativeBuf, 0, bytes);
                int status = (int) collWriteHandle.invokeExact(handle, buffer, nativeBuf, elements);
                checkStatus(status, "CollectiveBuffer.write");
            } catch (Throwable t) {
                if (t instanceof RuntimeException re) throw re;
                if (t instanceof Error err) throw err;
                throw new RuntimeException("HC4J collective write failed", t);
            }
        }
    }

    public static void readCollectiveBufferSegment(long handle, long buffer, MemorySegment destination, long elements) {
        ensureInitialized();
        if (destination.isNative()) {
            try {
                int status = (int) collReadHandle.invokeExact(handle, buffer, destination, elements);
                checkStatus(status, "CollectiveBuffer.read");
            } catch (Throwable t) {
                if (t instanceof RuntimeException re) throw re;
                if (t instanceof Error err) throw err;
                throw new RuntimeException("HC4J collective read failed", t);
            }
        } else {
            long bytes = elements * 4L;
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment nativeBuf = arena.allocate(bytes, 16);
                int status = (int) collReadHandle.invokeExact(handle, buffer, nativeBuf, elements);
                checkStatus(status, "CollectiveBuffer.read");
                MemorySegment.copy(nativeBuf, 0, destination, 0, bytes);
            } catch (Throwable t) {
                if (t instanceof RuntimeException re) throw re;
                if (t instanceof Error err) throw err;
                throw new RuntimeException("HC4J collective read failed", t);
            }
        }
    }

    public static void allReduceSumSubset(long handle, MemorySegment buffers, int count) {
        ensureInitialized();
        try {
            int status = (int) collAllReduceSubsetHandle.invokeExact(handle, buffers, count);
            checkStatus(status, "Collective.allReduceSumSubset");
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J collective allReduceSumSubset failed", t);
        }
    }

    public static void gemmBiasAct(long handle, int rank, long bufA, long bufB, long bufBias, long bufC, int m, int n, int k, int flags) {
        ensureInitialized();
        try {
            int status = (int) collGemmBiasActHandle.invokeExact(handle, rank, bufA, bufB, bufBias, bufC, m, n, k, flags);
            checkStatus(status, "Collective.gemmBiasAct");
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J collective gemmBiasAct failed", t);
        }
    }

    public static void splitKvAttn(long handle, int rank, long bufQ, long bufK, long bufV, long bufOutO, long bufOutStats,
                                   int numHeads, int numKvHeads, int headDim, int localTokens, int scaleBits) {
        ensureInitialized();
        try {
            int status = (int) collSplitKvAttnHandle.invokeExact(handle, rank, bufQ, bufK, bufV, bufOutO, bufOutStats,
                    numHeads, numKvHeads, headDim, localTokens, scaleBits);
            checkStatus(status, "Collective.splitKvAttention");
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            if (t instanceof Error err) throw err;
            throw new RuntimeException("HC4J collective splitKvAttention failed", t);
        }
    }
}
