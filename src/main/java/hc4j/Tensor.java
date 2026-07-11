package hc4j;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.Arena;

public class Tensor{
    private final MemorySegment data;
    private final int[] shape;
    private final int[] strides;
    private final long size;
    private final DType dtype;

    private Tensor(MemorySegment data,int[] shape,int[] strides,long size,DType dtype){
        this.data = data;
        this.shape = shape;
        this.strides = strides;
        long CalcSize=1;
        for(int dim : shape) CalcSize *= dim;
        this.size = size;
        this.dtype = dtype;
    }
    public static Tensor zeroes(int ...shape){
        return zeroes(Arena.ofAuto(),DType.f32,shape);
    }
    public static Tensor zeroes(Arena arena,int... shape){
        return zeroes(arena,DType.f32,shape);
    }
    public static Tensor zeroes(DType dtype,int... shape){
        return zeroes(Arena.ofAuto(),dtype,shape);
    }
    public static Tensor zeroes(Arena arena,DType dtype,int... shape){
        long size = 1;
        for(int dim : shape) size *= dim;
        MemorySegment data = arena.allocate(dtype.layout,size);
        int[] strides = new int[shape.length];
        int stride = 1;
        for(int i = shape.length - 1; i >= 0; i--){
            strides[i] = stride;
            stride *= shape[i];
        }
        return new Tensor(data,shape,strides,size,dtype);
    }
    public static Tensor ones(Arena arena,DType dtype,int... shape){
        long size = 1;
        for(int dim : shape) size *=dim;
        MemorySegment segment = arena.allocate(dtype.layout,size);
        switch(dtype){
            case i32 -> {
                for(long i=0;i<size;i++){
                    segment.setAtIndex(ValueLayout.JAVA_INT,i,1);
                }
            }
            case f32 -> {
                for(long i=0;i<size;i++){
                    segment.setAtIndex(ValueLayout.JAVA_FLOAT,i,1.0f);
                }
            }
            case f64 -> {
                for(long i=0;i<size;i++){
                    segment.setAtIndex(ValueLayout.JAVA_DOUBLE,i,1.0);
                }
            }
            default -> throw new AssertionError();

        }
        return new Tensor(segment,shape,calculateStrides(shape),size,dtype);
    }
    private static final int[] calculateStrides(int[] shape){
        int[] strides = new int[shape.length];
        int currentStride = 1;
         for(int i = shape.length - 1; i >= 0; i--){
            strides[i] = currentStride;
            currentStride *= shape[i];
        }
        return strides;
    }
    public int[] getShape(){
        return this.internalShapeUnsafe().clone();
    }
     public DType getDType(){
        return dtype;
    }
      public int[] internalStridesUnsafe() {
        return strides;
    }
      public int[] internalShapeUnsafe() {
        return shape;
    }
    public int dim(){
        return internalShapeUnsafe().length;
    }
    public MemorySegment getData(){
        return data;
    }
      public long getSize() {
        return size;
    }

    public Tensor add(Tensor b){
        Tensor result = zeroes(Arena.ofAuto(),dtype,shape);
        return switch(dtype){
            case i32 -> ops.add_i32(b,result);
            case f32 -> ops.add_f32(b,result);
            case f64 -> ops.add_f64(b,result);

        };
    }

    public Tensor add(Tensor b,Tensor result){
        return switch(dtype){
            case i32 -> ops.add_i32(b,result);
            case f32 -> ops.add_f32(b,result);
            case f64 -> ops.add_f64(b,result);

        };
    }

    public Tensor sub(Tensor b){
        Tensor result = zeroes(Arena.ofAuto(),dtype,shape);
        return switch(dtype){
            case i32 -> ops.sub_i32(b,result);
            case f32 -> ops.sub_f32(b,result);
            case f64 -> ops.sub_f64(b,result);

        };
    }





}