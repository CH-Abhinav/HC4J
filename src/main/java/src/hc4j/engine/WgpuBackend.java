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

    private static final MethodHandle initGpuHandle;
    private static final MethodHandle gpuAllocHandle;
    private static final MethodHandle gpuWriteHandle;
    private static final MethodHandle gpuFreeHandle;
    private static final MethodHandle gpuDownloadHandle;

    static {
        String os = System.getProperty("os.name").toLowerCase();
        String extension = os.contains("win") ? ".dll" : (os.contains("mac") ? ".dylib" : ".so");
        String libName = "hc4j" + extension;
        
        Path libPath = findNativeLibrary(libName);
        System.out.println("[HC4J] Native Engine loaded from: " + libPath);
        System.load(libPath.toString());
        
        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = SymbolLookup.loaderLookup();

        initGpuHandle = linker.downcallHandle(lookup.find("hc4j_init_gpu").orElseThrow(), 
            FunctionDescriptor.ofVoid());
            
        gpuAllocHandle = linker.downcallHandle(lookup.find("hc4j_gpu_alloc").orElseThrow(), 
            FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
            
        gpuWriteHandle = linker.downcallHandle(lookup.find("hc4j_gpu_write").orElseThrow(), 
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
            
        gpuFreeHandle = linker.downcallHandle(lookup.find("hc4j_gpu_free").orElseThrow(), 
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            
        gpuDownloadHandle = linker.downcallHandle(lookup.find("hc4j_gpu_download").orElseThrow(), 
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
    }

    private WgpuBackend() {}

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
            default -> throw new RuntimeException(context + ": Unknown native FFM error code: " + statusCode);
        }
    }

    public static void initGpu() {
        try { initGpuHandle.invokeExact(); } 
        catch (Throwable t) { throw new RuntimeException("HC4J GPU Init Failed", t); }
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
}