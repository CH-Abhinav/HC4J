package hc4j.ops;

import hc4j.DType;
import hc4j.Tensor;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ArithmeticOps {
    public static final int HC4J_SUCCESS = 0;
    public static final int HC4J_ERR_NOT_FOUND = -1;
    public static final int HC4J_ERR_INVALID_PARAM = -2;
    public static final int HC4J_ERR_GPU_READBACK = -3;

    private static final MethodHandle initGpuHandle;
    private static final MethodHandle gpuAllocHandle;
    private static final MethodHandle gpuWriteHandle;
    private static final MethodHandle gpuFreeHandle;
    private static final MethodHandle gpuDownloadHandle;

    private static final MethodHandle dispatchAdd;
    private static final MethodHandle dispatchSub;
    private static final MethodHandle dispatchMul;
    private static final MethodHandle dispatchDiv;

    static {
        String os = System.getProperty("os.name").toLowerCase();
        String extension = os.contains("win") ? ".dll" : (os.contains("mac") ? ".dylib" : ".so");
        String libName = "hc4j" + extension;
        
        // Explicitly points to the exact folder where Cargo builds your manifest
        Path libPath = Paths.get("src", "main", "rust", "target", "release", libName).toAbsolutePath();
        System.load(libPath.toString());
        
        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = SymbolLookup.loaderLookup();

        initGpuHandle = linker.downcallHandle(lookup.find("hc4j_init_gpu").orElseThrow(), 
            FunctionDescriptor.ofVoid());
            
        gpuAllocHandle = linker.downcallHandle(lookup.find("hc4j_gpu_alloc").orElseThrow(), 
            FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG)); 
            
        gpuWriteHandle = linker.downcallHandle(lookup.find("hc4j_gpu_write").orElseThrow(), 
            FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
            
        gpuFreeHandle = linker.downcallHandle(lookup.find("hc4j_gpu_free").orElseThrow(), 
            FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG));
            
        gpuDownloadHandle = linker.downcallHandle(lookup.find("hc4j_gpu_download").orElseThrow(), 
            FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        FunctionDescriptor computeDesc = FunctionDescriptor.ofVoid(
            ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG,
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, 
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, 
            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT
        );

        dispatchAdd = linker.downcallHandle(lookup.find("dispatch_add").orElseThrow(), computeDesc);
        dispatchSub = linker.downcallHandle(lookup.find("dispatch_sub").orElseThrow(), computeDesc);
        dispatchMul = linker.downcallHandle(lookup.find("dispatch_mul").orElseThrow(), computeDesc);
        dispatchDiv = linker.downcallHandle(lookup.find("dispatch_div").orElseThrow(), computeDesc);
    }

    private ArithmeticOps() {
        throw new AssertionError("Utility class");
    }

    public static void initGpu() {
        try { initGpuHandle.invokeExact(); } 
        catch (Throwable t) { throw new RuntimeException("HC4J GPU Init Failed", t); }
    }

    public static long allocVram(long totalElements) {
        try { return (long) gpuAllocHandle.invokeExact(totalElements); } 
        catch (Throwable t) { throw new RuntimeException("HC4J VRAM Allocation Exception", t); }
    }

    public static void writeVram(long vramId, MemorySegment hostData, long totalElements) {
        try { gpuWriteHandle.invokeExact(vramId, hostData, totalElements); } 
        catch (Throwable t) { throw new RuntimeException("HC4J VRAM Write Exception", t); }
    }

    public static void downloadVram(long vramId, MemorySegment hostData, long totalElements) {
        try { gpuDownloadHandle.invokeExact(vramId, hostData, totalElements); } 
        catch (Throwable t) { throw new RuntimeException("HC4J VRAM Download Exception", t); }
    }

    public static void freeVram(long vramId) {
        try { gpuFreeHandle.invokeExact(vramId); } 
        catch (Throwable t) { throw new RuntimeException("HC4J VRAM Free Exception", t); }
    }

    private static boolean isContiguous(Tensor t) {
        int[] shape = t.internalShapeUnsafe();
        int[] strides = t.internalStridesUnsafe();
        int expectedStride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expectedStride) return false;
            expectedStride *= shape[i];
        }
        return true;
    }

    private static int resolveDTypeCode(DType type) {
        return switch(type) {
            case i32 -> 0;
            case f32 -> 1;
            case f64 -> throw new UnsupportedOperationException("F64 not supported.");
        };
    }

    private static Tensor executeCompute(MethodHandle handle, String opName, Tensor a, Tensor b, Tensor res) {
        long sizeA = a.getSize();
        long sizeB = b.getSize();
        long sizeRes = res.getSize();

        long idA = allocVram(sizeA);
        long idB = allocVram(sizeB);
        long idRes = allocVram(sizeRes);

        try (Arena arena = Arena.ofConfined()) {
            writeVram(idA, a.getData(), sizeA);
            writeVram(idB, b.getData(), sizeB);

            int rank = a.dim();
            int dtypeCode = resolveDTypeCode(a.getDType());
            boolean contiguous = isContiguous(a) && isContiguous(b) && isContiguous(res);
            
            MemorySegment shapeSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, res.internalShapeUnsafe());
            MemorySegment stridesASeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, a.internalStridesUnsafe());
            MemorySegment stridesBSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, b.internalStridesUnsafe());
            MemorySegment stridesCSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, res.internalStridesUnsafe());

            handle.invokeExact(
                idA, idB, idRes, rank, shapeSeg, stridesASeg, stridesBSeg, stridesCSeg, 
                sizeRes, contiguous ? 1 : 0, dtypeCode
            );
            
            downloadVram(idRes, res.getData(), sizeRes);
            return res;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: " + opName + " failed", t);
        } finally {
            freeVram(idA);
            freeVram(idB);
            freeVram(idRes);
        }
    }

    public static Tensor add_f32(Tensor a, Tensor b, Tensor res) { return executeCompute(dispatchAdd, "Add", a, b, res); }
    public static Tensor sub_f32(Tensor a, Tensor b, Tensor res) { return executeCompute(dispatchSub, "Subtract", a, b, res); }
    public static Tensor mul_f32(Tensor a, Tensor b, Tensor res) { return executeCompute(dispatchMul, "Multiply", a, b, res); }
    public static Tensor div_f32(Tensor a, Tensor b, Tensor res) { return executeCompute(dispatchDiv, "Divide", a, b, res); }

    public static Tensor add_i32(Tensor a, Tensor b, Tensor res) { return executeCompute(dispatchAdd, "Add", a, b, res); }
    public static Tensor sub_i32(Tensor a, Tensor b, Tensor res) { return executeCompute(dispatchSub, "Subtract", a, b, res); }
    public static Tensor mul_i32(Tensor a, Tensor b, Tensor res) { return executeCompute(dispatchMul, "Multiply", a, b, res); }
    public static Tensor div_i32(Tensor a, Tensor b, Tensor res) { return executeCompute(dispatchDiv, "Divide", a, b, res); }

    public static Tensor add_f64(Tensor a, Tensor b, Tensor res) { throw new UnsupportedOperationException("F64 WIP"); }
    public static Tensor sub_f64(Tensor a, Tensor b, Tensor res) { throw new UnsupportedOperationException("F64 WIP"); }
    public static Tensor mul_f64(Tensor a, Tensor b, Tensor res) { throw new UnsupportedOperationException("F64 WIP"); }
    public static Tensor div_f64(Tensor a, Tensor b, Tensor res) { throw new UnsupportedOperationException("F64 WIP"); }
}