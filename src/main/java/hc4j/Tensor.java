package hc4j;

import hc4j.engine.WgpuBackend;
import hc4j.ops.ArithmeticOps;
import hc4j.ops.ExponentialOps;
import hc4j.ops.MatmulOps;
import hc4j.ops.TrignoOps;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A VRAM-resident tensor, identified across the FFI by an opaque 64-bit handle.
 *
 * <p>Every op comes in two forms: a convenience form that allocates a fresh result
 * ({@code a.sin()}), and a caller-allocated form that writes into {@code res} and returns it
 * ({@code a.sin(res)}). The caller-allocated form reuses buffers, and elementwise ops accept
 * {@code res == this} for in-place execution ({@code a.sin(a)}, {@code a.add(b, a)}).
 */
public class Tensor implements AutoCloseable {

    static {
        WgpuBackend.initGpu();
    }

    private final long vramId;
    private final int[] shape;
    private final int[] strides;
    private final DType dtype;
    private final long size;
    private final AtomicBoolean closed = new AtomicBoolean();

    private Tensor(long vramId, int[] shape, DType dtype) {
        this.vramId = vramId;
        this.shape = Arrays.copyOf(shape, shape.length);
        this.strides = computeContiguousStrides(this.shape);
        this.dtype = dtype;
        this.size = computeSize(this.shape);
    }

    /** A zero-filled tensor. */
    public static Tensor zeros(DType dtype, int... shape) {
        long totalSize = computeSize(shape);
        long vramId = WgpuBackend.allocVram(totalSize);
        return new Tensor(vramId, shape, dtype);
    }

    /**
     * A tensor with unspecified contents (a reused slab region keeps stale bytes). Intended for
     * outputs that an op overwrites in full; it skips the zero-fill pass {@link #zeros} pays.
     */
    public static Tensor empty(DType dtype, int... shape) {
        long totalSize = computeSize(shape);
        long vramId = WgpuBackend.allocVramUninit(totalSize);
        return new Tensor(vramId, shape, dtype);
    }

    public static Tensor fromArray(float[] values, int... shape) {
        long totalSize = checkLength(values.length, shape);
        long vramId = WgpuBackend.allocVramUninit(totalSize);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, values);
            WgpuBackend.writeVram(vramId, hostSegment, totalSize);
        } catch (RuntimeException | Error e) {
            WgpuBackend.freeVram(vramId);
            throw e;
        }
        return new Tensor(vramId, shape, DType.f32);
    }

    public static Tensor fromArray(int[] values, int... shape) {
        long totalSize = checkLength(values.length, shape);
        long vramId = WgpuBackend.allocVramUninit(totalSize);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocateFrom(ValueLayout.JAVA_INT, values);
            WgpuBackend.writeVram(vramId, hostSegment, totalSize);
        } catch (RuntimeException | Error e) {
            WgpuBackend.freeVram(vramId);
            throw e;
        }
        return new Tensor(vramId, shape, DType.i32);
    }

    // ------------------------------------------------------------------------------------------
    // ArithmeticOps
    // ------------------------------------------------------------------------------------------

    public Tensor add(Tensor other) { validateCompatible(other); return binaryFresh(ArithmeticOps::add, other); }
    public Tensor sub(Tensor other) { validateCompatible(other); return binaryFresh(ArithmeticOps::sub, other); }
    public Tensor mul(Tensor other) { validateCompatible(other); return binaryFresh(ArithmeticOps::mul, other); }
    public Tensor div(Tensor other) { validateCompatible(other); return binaryFresh(ArithmeticOps::div, other); }

    public Tensor add(Tensor other, Tensor res) {
        validateCompatible(other);
        validateCompatible(res);
        return ArithmeticOps.add(this, other, res);
    }

    public Tensor sub(Tensor other, Tensor res) {
        validateCompatible(other);
        validateCompatible(res);
        return ArithmeticOps.sub(this, other, res);
    }

    public Tensor mul(Tensor other, Tensor res) {
        validateCompatible(other);
        validateCompatible(res);
        return ArithmeticOps.mul(this, other, res);
    }

    public Tensor div(Tensor other, Tensor res) {
        validateCompatible(other);
        validateCompatible(res);
        return ArithmeticOps.div(this, other, res);
    }

    @FunctionalInterface
    private interface BinaryOp {
        Tensor apply(Tensor a, Tensor b, Tensor res);
    }

    private Tensor binaryFresh(BinaryOp op, Tensor other) {
        Tensor res = empty(dtype, shape);
        try {
            return op.apply(this, other, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    // ------------------------------------------------------------------------------------------
    // TrignoOps
    // ------------------------------------------------------------------------------------------

    public Tensor sin()   { return trig(TrignoOps.Function.SIN); }
    public Tensor cos()   { return trig(TrignoOps.Function.COS); }
    public Tensor tan()   { return trig(TrignoOps.Function.TAN); }
    public Tensor asin()  { return trig(TrignoOps.Function.ASIN); }
    public Tensor acos()  { return trig(TrignoOps.Function.ACOS); }
    public Tensor atan()  { return trig(TrignoOps.Function.ATAN); }
    public Tensor sinh()  { return trig(TrignoOps.Function.SINH); }
    public Tensor cosh()  { return trig(TrignoOps.Function.COSH); }
    public Tensor tanh()  { return trig(TrignoOps.Function.TANH); }
    public Tensor asinh() { return trig(TrignoOps.Function.ASINH); }
    public Tensor acosh() { return trig(TrignoOps.Function.ACOSH); }
    public Tensor atanh() { return trig(TrignoOps.Function.ATANH); }

    public Tensor sin(Tensor res)   { return TrignoOps.apply(TrignoOps.Function.SIN, this, res); }
    public Tensor cos(Tensor res)   { return TrignoOps.apply(TrignoOps.Function.COS, this, res); }
    public Tensor tan(Tensor res)   { return TrignoOps.apply(TrignoOps.Function.TAN, this, res); }
    public Tensor asin(Tensor res)  { return TrignoOps.apply(TrignoOps.Function.ASIN, this, res); }
    public Tensor acos(Tensor res)  { return TrignoOps.apply(TrignoOps.Function.ACOS, this, res); }
    public Tensor atan(Tensor res)  { return TrignoOps.apply(TrignoOps.Function.ATAN, this, res); }
    public Tensor sinh(Tensor res)  { return TrignoOps.apply(TrignoOps.Function.SINH, this, res); }
    public Tensor cosh(Tensor res)  { return TrignoOps.apply(TrignoOps.Function.COSH, this, res); }
    public Tensor tanh(Tensor res)  { return TrignoOps.apply(TrignoOps.Function.TANH, this, res); }
    public Tensor asinh(Tensor res) { return TrignoOps.apply(TrignoOps.Function.ASINH, this, res); }
    public Tensor acosh(Tensor res) { return TrignoOps.apply(TrignoOps.Function.ACOSH, this, res); }
    public Tensor atanh(Tensor res) { return TrignoOps.apply(TrignoOps.Function.ATANH, this, res); }

    private Tensor trig(TrignoOps.Function fn) {
        requireF32(fn.opName());
        Tensor res = empty(dtype, shape);
        try {
            return TrignoOps.apply(fn, this, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    // ------------------------------------------------------------------------------------------
    // ExponentialOps
    // ------------------------------------------------------------------------------------------

    public Tensor exp()   { return exponential(ExponentialOps.Function.EXP); }
    /** Natural logarithm. */
    public Tensor log()   { return exponential(ExponentialOps.Function.LOG); }
    public Tensor log2()  { return exponential(ExponentialOps.Function.LOG2); }
    public Tensor log10() { return exponential(ExponentialOps.Function.LOG10); }
    public Tensor sqrt()  { return exponential(ExponentialOps.Function.SQRT); }

    public Tensor exp(Tensor res)   { return ExponentialOps.apply(ExponentialOps.Function.EXP, this, res); }
    public Tensor log(Tensor res)   { return ExponentialOps.apply(ExponentialOps.Function.LOG, this, res); }
    public Tensor log2(Tensor res)  { return ExponentialOps.apply(ExponentialOps.Function.LOG2, this, res); }
    public Tensor log10(Tensor res) { return ExponentialOps.apply(ExponentialOps.Function.LOG10, this, res); }
    public Tensor sqrt(Tensor res)  { return ExponentialOps.apply(ExponentialOps.Function.SQRT, this, res); }

    private Tensor exponential(ExponentialOps.Function fn) {
        requireF32(fn.opName());
        Tensor res = empty(dtype, shape);
        try {
            return ExponentialOps.apply(fn, this, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    // ------------------------------------------------------------------------------------------
    // MatmulOps
    // ------------------------------------------------------------------------------------------

    /**
     * Matrix product {@code this · other}. 2-D operands must satisfy {@code this.cols ==
     * other.rows}; 1-D operands act as row/column vectors (see {@link MatmulOps}).
     */
    public Tensor matmul(Tensor other) {
        MatmulOps.Dims d = MatmulOps.dims(this, other, false, false);
        requireF32("matmul");
        Tensor res = empty(DType.f32, d.resultShape());
        try {
            return MatmulOps.matmul(this, other, res, false, false);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /** Matrix product into {@code res}, which must not be {@code this} or {@code other}. */
    public Tensor matmul(Tensor other, Tensor res) {
        return MatmulOps.matmul(this, other, res, false, false);
    }

    /** A transposed copy of a 2-D tensor, computed in VRAM. */
    public Tensor transpose() {
        if (shape.length != 2) {
            throw new IllegalArgumentException("transpose needs a 2-D tensor, got " + Arrays.toString(shape));
        }
        Tensor res = empty(dtype, shape[1], shape[0]);
        try {
            return MatmulOps.transpose(this, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor transpose(Tensor res) {
        return MatmulOps.transpose(this, res);
    }

    // ------------------------------------------------------------------------------------------
    // Fusion
    // ------------------------------------------------------------------------------------------

    /**
     * Starts a deferred elementwise expression. Chained ops build a tree that {@link
     * FusedExpr#eval()} compiles into a single kernel: {@code a.lazy().sin().add(b).mul(c).eval()}.
     */
    public FusedExpr lazy() {
        return FusedExpr.of(this);
    }

    // ------------------------------------------------------------------------------------------
    // Readback and lifetime
    // ------------------------------------------------------------------------------------------

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

    /** Which memory tier currently holds this tensor's bytes. */
    public WgpuBackend.Residency residency() {
        return WgpuBackend.residency(vramId);
    }

    /**
     * Pages this tensor out of VRAM to host RAM (or disk). A no-op if it is already off-device or
     * in use by an operation that has not been submitted yet. Ops page it back in on demand.
     */
    public void evict() {
        WgpuBackend.evict(vramId);
    }

    /** Frees the tensor's storage. Idempotent. */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            WgpuBackend.freeVram(this.vramId);
        }
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

    private static long checkLength(int length, int[] shape) {
        long total = computeSize(shape);
        if (length != total) {
            throw new IllegalArgumentException("Array has " + length + " elements but shape "
                    + Arrays.toString(shape) + " needs " + total);
        }
        return total;
    }

    private void requireF32(String op) {
        if (dtype != DType.f32) {
            throw new IllegalArgumentException(op + " is only defined for f32 tensors, got " + dtype);
        }
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
