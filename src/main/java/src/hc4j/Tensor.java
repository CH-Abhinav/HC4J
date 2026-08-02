package hc4j;

import hc4j.engine.WgpuBackend;
import hc4j.ops.ArithmeticOps;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

public class Tensor implements AutoCloseable {

    static {
        WgpuBackend.initGpu();
    }

    private final long vramId;
    private final int[] shape;
    private final int[] strides; 
    private final DType dtype;
    private final long size;

    private Tensor(long vramId, int[] shape, DType dtype) {
        this.vramId = vramId;
        this.shape = Arrays.copyOf(shape, shape.length);
        this.strides = computeContiguousStrides(this.shape);
        this.dtype = dtype;
        this.size = computeSize(this.shape);
    }

    public static Tensor zeros(DType dtype, int... shape) {
        long totalSize = computeSize(shape);
        long vramId = WgpuBackend.allocVram(totalSize);
        return new Tensor(vramId, shape, dtype);
    }

    public static Tensor fromArray(float[] values, int... shape) {
        long totalSize = computeSize(shape);
        long vramId = WgpuBackend.allocVram(totalSize);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, values);
            WgpuBackend.writeVram(vramId, hostSegment, totalSize);
        }
        return new Tensor(vramId, shape, DType.f32);
    }

    public static Tensor fromArray(int[] values, int... shape) {
        long totalSize = computeSize(shape);
        long vramId = WgpuBackend.allocVram(totalSize);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocateFrom(ValueLayout.JAVA_INT, values);
            WgpuBackend.writeVram(vramId, hostSegment, totalSize);
        }
        return new Tensor(vramId, shape, DType.i32);
    }

    //ArithmeticOps methods

    public Tensor add(Tensor other) {
        validateCompatible(other);
        Tensor res = Tensor.zeros(this.dtype, this.shape);
        return ArithmeticOps.add(this, other, res);
    }

    public Tensor sub(Tensor other) {
        validateCompatible(other);
        Tensor res = Tensor.zeros(this.dtype, this.shape);
        return ArithmeticOps.sub(this, other, res);
    }

    public Tensor mul(Tensor other) {
        validateCompatible(other);
        Tensor res = Tensor.zeros(this.dtype, this.shape);
        return ArithmeticOps.mul(this, other, res);
    }

    public Tensor div(Tensor other) {
        validateCompatible(other);
        Tensor res = Tensor.zeros(this.dtype, this.shape);
        return ArithmeticOps.div(this, other, res);
    }

    public Tensor add(Tensor other,Tensor res) {
        validateCompatible(other);
        validateCompatible(res);
        return ArithmeticOps.add(this, other, res);
    }

    public Tensor sub(Tensor other,Tensor res) {
        validateCompatible(other);
        validateCompatible(res);
        return ArithmeticOps.sub(this, other, res);
    }

    public Tensor mul(Tensor other,Tensor res) {
        validateCompatible(other);
        validateCompatible(res);
        return ArithmeticOps.mul(this, other, res);
    }

    public Tensor div(Tensor other,Tensor res) {
        validateCompatible(other);
        validateCompatible(res);
        return ArithmeticOps.div(this, other, res);
    }

    public float[] toFloatArray() {
        if (this.dtype != DType.f32) {
            throw new IllegalStateException("Cannot readback " + this.dtype + " as float[]");
        }
        float[] result = new float[(int) this.size];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocate(ValueLayout.JAVA_FLOAT, this.size);
            WgpuBackend.downloadVram(this.vramId, hostSegment, this.size);
            MemorySegment.copy(hostSegment, ValueLayout.JAVA_FLOAT, 0, result, 0, (int) this.size);
        }
        return result;
    }

    public int[] toIntArray() {
        if (this.dtype != DType.i32) {
            throw new IllegalStateException("Cannot readback " + this.dtype + " as int[]");
        }
        int[] result = new int[(int) this.size];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocate(ValueLayout.JAVA_INT, this.size);
            WgpuBackend.downloadVram(this.vramId, hostSegment, this.size);
            MemorySegment.copy(hostSegment, ValueLayout.JAVA_INT, 0, result, 0, (int) this.size);
        }
        return result;
    }

    @Override
    public void close() {
        WgpuBackend.freeVram(this.vramId);
    }

    //utilities - later in added into utility modules

    private static int[] computeContiguousStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int acc = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }

    private static long computeSize(int[] shape) {
        long total = 1;
        for (int dim : shape) total *= dim;
        return total;
    }

    private void validateCompatible(Tensor other) {
        if (this.dtype != other.dtype) {
            throw new IllegalArgumentException("DType mismatch: " + this.dtype + " vs " + other.dtype);
        }
        if (!Arrays.equals(this.shape, other.shape)) {
            throw new IllegalArgumentException("Shape mismatch: " + Arrays.toString(this.shape) + " vs " + Arrays.toString(other.shape));
        }
    }

    public long getVramId() { return vramId; }
    public int[] internalShapeUnsafe() { return shape; }
    public int[] internalStridesUnsafe() { return strides; }
    public DType getDType() { return dtype; }
    public long getSize() { return size; }
    public int dim() { return shape.length; }

    @Override
    public String toString() {
        return "Tensor(vramId=" + vramId + ", shape=" + Arrays.toString(shape) + ", dtype=" + dtype + ", size=" + size + ")";
    }
}