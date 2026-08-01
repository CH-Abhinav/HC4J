package hc4j;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

public class Tensor {
    private final MemorySegment data;
    private final int[] shape;
    private final int[] strides;
    private final long size;
    private final DType dtype;

    private Tensor(MemorySegment data, int[] shape, int[] strides, long size, DType dtype) {
        this.data = data;
        this.shape = shape;
        this.strides = strides;
        this.size = size;
        this.dtype = dtype;
    }

    public static Tensor zeroes(Arena arena, DType dtype, int... shape) {
        long size = 1;
        for (int dim : shape) size *= dim;
        MemorySegment data = arena.allocate(dtype.layout, size);
        int[] strides = calculateStrides(shape);
        return new Tensor(data, shape, strides, size, dtype);
    }

    private static int[] calculateStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int currentStride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = currentStride;
            currentStride *= shape[i];
        }
        return strides;
    }

    public int[] getShape() { return shape; }
    public DType getDType() { return dtype; }
    public int[] internalStridesUnsafe() { return strides; }
    public int[] internalShapeUnsafe() { return shape; }
    public int dim() { return shape.length; }
    public MemorySegment getData() { return data; }
    public long getSize() { return size; }

    public Tensor add(Tensor b) {
        Tensor result = zeroes(Arena.ofAuto(), dtype, shape);
        return Ops.add_f32(this, b, result);
    }

    // =========================================================================
    // CONNECTION TEST MAIN METHOD
    // =========================================================================
    public static void main(String[] args) {
        System.out.println("-> Initializing WebGPU through Rust FFI Bridge...");
        try {
            // This triggers Ops static block, loads the DLL, and boots WebGPU
            Ops.initGpu();
            System.out.println("-> SUCCESS: WebGPU Engine connected successfully!");

            // Test allocation and math
            try (Arena arena = Arena.ofConfined()) {
                Tensor a = zeroes(arena, DType.f32, 4);
                Tensor b = zeroes(arena, DType.f32, 4);
                
                System.out.println("-> SUCCESS: Tensors allocated safely in off-heap memory.");
            }

        } catch (Throwable t) {
            System.err.println("-> CONNECTION FAILED:");
            t.printStackTrace();
        }
    }
}