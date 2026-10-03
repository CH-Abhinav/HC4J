package hc4j.engine;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class WgpuBackend {

    public static final int HC4J_SUCCESS = 0;
    public static final int HC4J_ERR_NOT_FOUND = -1;
    public static final int HC4J_ERR_INVALID_PARAM = -2;
    public static final int HC4J_ERR_GPU_READBACK = -3;
    public static final int HC4J_ERR_OUT_OF_MEMORY = -4;
    public static final int HC4J_ERR_DEVICE = -5;
    public static final int HC4J_ERR_GPU_VALIDATION = -6;
    public static final int HC4J_ERR_HOST_IO = -7;
    public static final int HC4J_ERR_UNSUPPORTED = -8;

    /** Where a tensor's bytes currently live, as reported by the tiered memory manager. */
    public enum Residency { DEVICE, HOST, DISK }

    private static final MethodHandle initGpuHandle;
    private static final MethodHandle gpuAllocHandle;
    private static final MethodHandle gpuWriteHandle;
    private static final MethodHandle gpuFreeHandle;
    private static final MethodHandle gpuDownloadHandle;
    private static final MethodHandle memConfigureHandle;
    private static final MethodHandle memStatsHandle;
    private static final MethodHandle memResidencyHandle;
    private static final MethodHandle memEvictHandle;
    private static final MethodHandle gpuAllocUninitHandle;
    private static final MethodHandle batchBeginHandle;
    private static final MethodHandle batchEndHandle;
    private static final MethodHandle engineStatsHandle;
    private static final MethodHandle synchronizeHandle;

    static {
        String os = System.getProperty("os.name").toLowerCase();
        String extension = os.contains("win") ? ".dll" : (os.contains("mac") ? ".dylib" : ".so");
        String libName = os.contains("mac") || os.contains("linux")
        ? "libhc4j" + extension
        : "hc4j" + extension;

        Path libPath = findNativeLibrary(libName);
        System.out.println("[HC4J] Native Engine loaded from: " + libPath);
        System.load(libPath.toString());

        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = SymbolLookup.loaderLookup();

        initGpuHandle = linker.downcallHandle(lookup.find("hc4j_init_gpu").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_INT));

        gpuAllocHandle = linker.downcallHandle(lookup.find("hc4j_gpu_alloc").orElseThrow(),
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

        gpuAllocUninitHandle = linker.downcallHandle(lookup.find("hc4j_gpu_alloc_uninit").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));

        batchBeginHandle = linker.downcallHandle(lookup.find("hc4j_batch_begin").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_INT));

        batchEndHandle = linker.downcallHandle(lookup.find("hc4j_batch_end").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_INT));

        engineStatsHandle = linker.downcallHandle(lookup.find("hc4j_engine_stats").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

        synchronizeHandle = linker.downcallHandle(lookup.find("hc4j_synchronize").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_INT));
    }

    private WgpuBackend() {}

    /**
     * Symbols exported by the HC4J native library. Calling this forces this class's static
     * initializer, so the library is guaranteed to be loaded before any lookup; op binding
     * classes must obtain their lookup here rather than calling {@code loaderLookup()} directly.
     */
    public static SymbolLookup nativeSymbols() {
        return SymbolLookup.loaderLookup();
    }

    private static Path findNativeLibrary(String libName) {
        Path current = Paths.get("").toAbsolutePath();

        while (current != null) {
            Path checkPath = current.resolve(Paths.get("src", "main", "rust", "target", "release", libName));
            if (Files.exists(checkPath)) {
                return checkPath;
            }
            current = current.getParent();
        }

        throw new UnsatisfiedLinkError("Could not find " + libName + " anywhere in the project tree. Did you run 'cargo build --release' in the rust folder?");
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

    public static void initGpu() {
        int status;
        try { status = (int) initGpuHandle.invokeExact(); }
        catch (Throwable t) { throw new RuntimeException("HC4J GPU Init Failed", t); }
        checkStatus(status, "initGpu");
    }

    public static long allocVram(long totalElements) {
        try {
            long id = (long) gpuAllocHandle.invokeExact(totalElements);
            if (id == 0) throw new OutOfMemoryError("Rust failed to allocate VRAM buffer.");
            return id;
        }
        catch (Throwable t) { throw new RuntimeException("HC4J VRAM Allocation Exception", t); }
    }

    public static void writeVram(long vramId, MemorySegment hostData, long totalElements) {
        try {
            int status = (int) gpuWriteHandle.invokeExact(vramId, hostData, totalElements);
            checkStatus(status, "writeVram");
        }
        catch (Throwable t) { throw new RuntimeException("HC4J VRAM Write Exception", t); }
    }

    public static void downloadVram(long vramId, MemorySegment hostData, long totalElements) {
        try {
            int status = (int) gpuDownloadHandle.invokeExact(vramId, hostData, totalElements);
            checkStatus(status, "downloadVram");
        }
        catch (Throwable t) { throw new RuntimeException("HC4J VRAM Download Exception", t); }
    }

    public static void freeVram(long vramId) {
        if (vramId == 0) return;
        try {
            int status = (int) gpuFreeHandle.invokeExact(vramId);
            checkStatus(status,"freeVram");
        }
        catch (Throwable t) {
            System.err.println("Warning: Failed to free VRAM handle " + vramId + ": " + t.getMessage());
        }
    }

    /**
     * Sets the tiered memory manager's budgets in bytes and immediately evicts (VRAM) or spills
     * (host RAM) down to them. Pass 0 to leave a tier's budget unchanged.
     */
    public static void configureMemory(long vramBudgetBytes, long hostBudgetBytes) {
        if (vramBudgetBytes < 0 || hostBudgetBytes < 0) {
            throw new IllegalArgumentException("Budgets must be >= 0 (0 = unchanged)");
        }
        int status;
        try { status = (int) memConfigureHandle.invokeExact(vramBudgetBytes, hostBudgetBytes); }
        catch (Throwable t) { throw new RuntimeException("HC4J memory configure failed", t); }
        checkStatus(status, "configureMemory");
    }

    public static MemoryStats memoryStats() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(MemoryStats.LAYOUT);
            int status;
            try { status = (int) memStatsHandle.invokeExact(out); }
            catch (Throwable t) { throw new RuntimeException("HC4J memory stats failed", t); }
            checkStatus(status, "memoryStats");
            return MemoryStats.read(out);
        }
    }

    public static Residency residency(long vramId) {
        int code;
        try { code = (int) memResidencyHandle.invokeExact(vramId); }
        catch (Throwable t) { throw new RuntimeException("HC4J residency query failed", t); }
        if (code < 0) checkStatus(code, "residency");
        return switch (code) {
            case 0 -> Residency.DEVICE;
            case 1 -> Residency.HOST;
            case 2 -> Residency.DISK;
            default -> throw new IllegalStateException("Unknown residency code " + code);
        };
    }

    /** Forces a tensor out of VRAM; a no-op if it is already off-device or in use. */
    public static void evict(long vramId) {
        int status;
        try { status = (int) memEvictHandle.invokeExact(vramId); }
        catch (Throwable t) { throw new RuntimeException("HC4J evict failed", t); }
        checkStatus(status, "evict");
    }

    /**
     * Allocates a tensor whose contents are unspecified (a reused slab region keeps stale bytes).
     * For op outputs that the kernel overwrites in full, this skips a zero-fill pass over the
     * whole tensor that {@link #allocVram} would pay.
     */
    public static long allocVramUninit(long totalElements) {
        long id;
        try { id = (long) gpuAllocUninitHandle.invokeExact(totalElements); }
        catch (Throwable t) { throw new RuntimeException("HC4J VRAM Allocation Exception", t); }
        if (id == 0) {
            throw new GpuOutOfMemoryException("allocVramUninit: could not allocate " + totalElements + " elements in any tier");
        }
        return id;
    }

    /** Opens a command-batching scope; prefer {@link GpuBatch#open()}. */
    public static void batchBegin() {
        int status;
        try { status = (int) batchBeginHandle.invokeExact(); }
        catch (Throwable t) { throw new RuntimeException("HC4J batch begin failed", t); }
        checkStatus(status, "batchBegin");
    }

    /** Closes a command-batching scope, submitting the batch if it is the outermost. */
    public static void batchEnd() {
        int status;
        try { status = (int) batchEndHandle.invokeExact(); }
        catch (Throwable t) { throw new RuntimeException("HC4J batch end failed", t); }
        checkStatus(status, "batchEnd");
    }

    public static EngineStats engineStats() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(EngineStats.LAYOUT);
            int status;
            try { status = (int) engineStatsHandle.invokeExact(out); }
            catch (Throwable t) { throw new RuntimeException("HC4J engine stats failed", t); }
            checkStatus(status, "engineStats");
            return EngineStats.read(out);
        }
    }

    /** Submits all pending GPU work and blocks until it has finished. */
    public static void synchronize() {
        int status;
        try { status = (int) synchronizeHandle.invokeExact(); }
        catch (Throwable t) { throw new RuntimeException("HC4J synchronize failed", t); }
        checkStatus(status, "synchronize");
    }
}
