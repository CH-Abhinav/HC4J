package hc4j.ops;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.engine.WgpuBackend;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

public class ArithmeticOps {

    private static final MethodHandle dispatchAdd;
    private static final MethodHandle dispatchSub;
    private static final MethodHandle dispatchMul;
    private static final MethodHandle dispatchDiv;

    static {
        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = SymbolLookup.loaderLookup();

        FunctionDescriptor computeDesc = FunctionDescriptor.of(
            ValueLayout.JAVA_INT, 
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

    private ArithmeticOps() {}

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
            case f64 -> throw new UnsupportedOperationException("F64 not supported yet.");
        };
    }

    public static Tensor execute(MethodHandle handle, String opName, Tensor a, Tensor b, Tensor res) {
        try (Arena arena = Arena.ofConfined()) {
            int rank = res.dim();
            int dtypeCode = resolveDTypeCode(a.getDType());
            boolean contiguous = isContiguous(a) && isContiguous(b) && isContiguous(res);

            MemorySegment shapeSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, res.internalShapeUnsafe());
            MemorySegment stridesASeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, a.internalStridesUnsafe());
            MemorySegment stridesBSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, b.internalStridesUnsafe());
            MemorySegment stridesCSeg = contiguous ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, res.internalStridesUnsafe());

            int status = (int) handle.invokeExact(
                a.getVramId(), b.getVramId(), res.getVramId(), 
                rank, shapeSeg, stridesASeg, stridesBSeg, stridesCSeg, 
                res.getSize(), contiguous ? 1 : 0, dtypeCode
            );
            
            WgpuBackend.checkStatus(status, "execute (" + opName + ")");

            return res; 
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: " + opName + " failed", t);
        }
    }

    public static Tensor add(Tensor a, Tensor b, Tensor res) { return execute(dispatchAdd, "Add", a, b, res); }
    public static Tensor sub(Tensor a, Tensor b, Tensor res) { return execute(dispatchSub, "Subtract", a, b, res); }
    public static Tensor mul(Tensor a, Tensor b, Tensor res) { return execute(dispatchMul, "Multiply", a, b, res); }
    public static Tensor div(Tensor a, Tensor b, Tensor res) { return execute(dispatchDiv, "Divide", a, b, res); }
}