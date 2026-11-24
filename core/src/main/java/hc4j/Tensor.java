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

package hc4j;

import hc4j.engine.Residency;
import hc4j.internal.ffi.NativeCleaner;
import hc4j.internal.ffi.NativeEngine;
import hc4j.internal.ffi.TensorAccess;
import hc4j.internal.ffi.VramHandle;
import hc4j.internal.layout.Dims;
import hc4j.internal.layout.Rows;
import hc4j.internal.layout.ValidUtil;
import hc4j.internal.ops.ArithmeticOps;
import hc4j.internal.ops.ExponentialOps;
import hc4j.internal.ops.MathOps;
import hc4j.internal.ops.MatmulOps;
import hc4j.internal.ops.NnOps;
import hc4j.internal.ops.TrigOps;
import hc4j.linalg.Linalg;
import hc4j.nn.Activation;
import hc4j.nn.Nn;
import hc4j.nn.Reduction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A VRAM-resident multi-dimensional array of 32-bit floats or integers supporting zero-copy strided views,
 * automatic broadcasting, in-place ops, and Panama Foreign Function &amp; Memory (FFM) interop.
 */
public class Tensor implements AutoCloseable {

    static {
        TensorAccess.setInstance(new TensorAccess() {
            @Override
            public long getVramId(Tensor t) {
                return t.handle.vramId();
            }

            @Override
            public int[] getShapeUnsafe(Tensor t) {
                return t.shape;
            }

            @Override
            public int[] getStridesUnsafe(Tensor t) {
                return t.strides;
            }

            @Override
            public void ensureOpen(Tensor t) {
                t.ensureOpen();
            }

            @Override
            public long elementOffset(Tensor t) {
                return t.elementOffset;
            }

            @Override
            public VramHandle getVramHandle(Tensor t) {
                return t.handle;
            }

            @Override
            public Tensor create(DType dtype, int[] shape, int[] strides, long vramId) {
                return new Tensor(dtype, shape, strides, 0L, new VramHandle(vramId));
            }

            @Override
            public Tensor create(DType dtype, int[] shape, int[] strides, long elementOffset, VramHandle handle) {
                return new Tensor(dtype, shape, strides, elementOffset, handle);
            }
        });
    }

    private static final class TensorReleaseAction implements Runnable {
        private final VramHandle handle;
        private final AtomicBoolean closed;

        TensorReleaseAction(VramHandle handle, AtomicBoolean closed) {
            this.handle = handle;
            this.closed = closed;
        }

        @Override
        public void run() {
            if (closed.compareAndSet(false, true) && handle != null) {
                handle.release();
            }
        }
    }

    private final DType dtype;
    private final int[] shape;
    private final int[] strides;
    private final long elementOffset;
    private final long size;
    private final VramHandle handle;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final Cleaner.Cleanable cleanable;

    private Tensor(DType dtype, int[] shape, int[] strides, long elementOffset, VramHandle handle) {
        this.dtype = dtype;
        this.shape = Arrays.copyOf(shape, shape.length);
        this.strides = Arrays.copyOf(strides, strides.length);
        this.elementOffset = elementOffset;
        this.size = computeSize(this.shape);
        this.handle = handle;
        this.cleanable = NativeCleaner.register(this, new TensorReleaseAction(handle, this.closed));
    }

    // ------------------------------------------------------------------------------------------
    // Factories
    // ------------------------------------------------------------------------------------------

    /**
     * Allocates a zero-initialized tensor of the specified shape and data type.
     *
     * @param dtype tensor element type
     * @param shape dimension extents
     * @return newly allocated zero tensor
     */
    public static Tensor zeros(DType dtype, int... shape) {
        Objects.requireNonNull(dtype, "dtype");
        ValidUtil.requireValidShape(shape);
        long totalSize = computeSize(shape);
        long vramId = (dtype == DType.f16)
                ? NativeEngine.allocVramBytes(Math.multiplyExact(totalSize, 2L), true)
                : NativeEngine.allocVram(totalSize);
        VramHandle h = new VramHandle(vramId);
        return new Tensor(dtype, shape, computeContiguousStrides(shape), 0L, h);
    }

    /**
     * Allocates a tensor initialized with ones of the specified shape and data type.
     *
     * @param dtype tensor element type
     * @param shape dimension extents
     * @return newly allocated ones tensor
     */
    public static Tensor ones(DType dtype, int... shape) {
        Objects.requireNonNull(dtype, "dtype");
        ValidUtil.requireValidShape(shape);
        if (dtype == DType.f32) {
            return full(1.0f, shape);
        } else if (dtype == DType.i32) {
            return full(1, shape);
        } else {
            throw new IllegalArgumentException("Unsupported dtype: " + dtype);
        }
    }

    /**
     * Allocates an uninitialized tensor of the specified shape and data type.
     *
     * @param dtype tensor element type
     * @param shape dimension extents
     * @return newly allocated uninitialized tensor
     */
    public static Tensor empty(DType dtype, int... shape) {
        Objects.requireNonNull(dtype, "dtype");
        ValidUtil.requireValidShape(shape);
        long totalSize = computeSize(shape);
        long vramId = (dtype == DType.f16)
                ? NativeEngine.allocVramBytes(Math.multiplyExact(totalSize, 2L), false)
                : NativeEngine.allocVramUninit(totalSize);
        VramHandle h = new VramHandle(vramId);
        return new Tensor(dtype, shape, computeContiguousStrides(shape), 0L, h);
    }

    /**
     * Allocates a float tensor filled with {@code value}.
     *
     * @param value constant fill value
     * @param shape dimension extents
     * @return filled tensor
     */
    public static Tensor full(float value, int... shape) {
        ValidUtil.requireValidShape(shape);
        try (Tensor scalar = fromArray(new float[]{value}, 1);
             Tensor bcast = scalar.broadcastTo(shape)) {
            return bcast.copy();
        }
    }

    /**
     * Allocates an integer tensor filled with {@code value}.
     *
     * @param value constant fill value
     * @param shape dimension extents
     * @return filled tensor
     */
    public static Tensor full(int value, int... shape) {
        ValidUtil.requireValidShape(shape);
        try (Tensor scalar = fromArray(new int[]{value}, 1);
             Tensor bcast = scalar.broadcastTo(shape)) {
            return bcast.copy();
        }
    }

    /**
     * Creates a 1-D float tensor with values in {@code [start, stop)} with given step size.
     *
     * @param start start value (inclusive)
     * @param stop  stop value (exclusive)
     * @param step  step size
     * @return 1-D range tensor
     */
    public static Tensor arange(float start, float stop, float step) {
        if (step == 0f) {
            throw new IllegalArgumentException("step must not be zero");
        }
        double diff = stop - start;
        if ((step > 0 && diff <= 0) || (step < 0 && diff >= 0)) {
            throw new IllegalArgumentException("arange resulting size must be positive");
        }
        int count = (int) Math.ceil(diff / step);
        if (count <= 0) {
            throw new IllegalArgumentException("arange length must be > 0");
        }
        float[] data = new float[count];
        float current = start;
        for (int i = 0; i < count; i++) {
            data[i] = current;
            current += step;
        }
        return fromArray(data, count);
    }

    /**
     * Creates a 1-D integer tensor with values in {@code [0, stop)} with step 1.
     *
     * @param stop stop value (exclusive)
     * @return 1-D range tensor
     */
    public static Tensor arange(int stop) {
        return arange(0, stop, 1);
    }

    /**
     * Creates a 1-D integer tensor with values in {@code [start, stop)} with given step size.
     *
     * @param start start value (inclusive)
     * @param stop  stop value (exclusive)
     * @param step  step size
     * @return 1-D range tensor
     */
    public static Tensor arange(int start, int stop, int step) {
        if (step == 0) {
            throw new IllegalArgumentException("step must not be zero");
        }
        long diff = (long) stop - start;
        if ((step > 0 && diff <= 0) || (step < 0 && diff >= 0)) {
            throw new IllegalArgumentException("arange resulting size must be positive, start=" + start + ", stop=" + stop + ", step=" + step);
        }
        int count = (int) (diff > 0 ? (diff + step - 1) / step : (diff + step + 1) / step);
        if (count <= 0) {
            throw new IllegalArgumentException("arange length must be > 0");
        }
        int[] data = new int[count];
        int current = start;
        for (int i = 0; i < count; i++) {
            data[i] = current;
            current += step;
        }
        return fromArray(data, count);
    }

    /**
     * Creates a 1-D float tensor with {@code num} evenly spaced samples in {@code [start, stop]}.
     *
     * @param start start value (inclusive)
     * @param stop  stop value (inclusive)
     * @param num   number of samples (must be &gt; 0)
     * @return 1-D linspace tensor
     */
    public static Tensor linspace(float start, float stop, int num) {
        if (num <= 0) {
            throw new IllegalArgumentException("num must be strictly positive, got " + num);
        }
        if (num == 1) {
            return fromArray(new float[]{start}, 1);
        }
        float[] data = new float[num];
        float step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            if (i == num - 1) {
                data[i] = stop;
            } else {
                data[i] = start + i * step;
            }
        }
        return fromArray(data, num);
    }

    /**
     * Creates a 2-D float identity matrix of size {@code n x n}.
     *
     * @param n matrix dimension
     * @return identity matrix
     */
    public static Tensor eye(int n) {
        return eye(DType.f32, n, n);
    }

    /**
     * Creates a 2-D identity matrix of size {@code n x n} with the given data type.
     *
     * @param dtype data type
     * @param n     matrix dimension
     * @return identity matrix
     */
    public static Tensor eye(DType dtype, int n) {
        return eye(dtype, n, n);
    }

    /**
     * Creates a 2-D identity matrix of size {@code rows x cols} with the given data type.
     *
     * @param dtype data type
     * @param rows  row count
     * @param cols  column count
     * @return identity matrix
     */
    public static Tensor eye(DType dtype, int rows, int cols) {
        Objects.requireNonNull(dtype, "dtype");
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("rows and cols must be positive, got " + rows + "x" + cols);
        }
        int min = Math.min(rows, cols);
        int total = rows * cols;
        if (dtype == DType.f32) {
            float[] data = new float[total];
            for (int i = 0; i < min; i++) {
                data[i * cols + i] = 1.0f;
            }
            return fromArray(data, rows, cols);
        } else if (dtype == DType.i32) {
            int[] data = new int[total];
            for (int i = 0; i < min; i++) {
                data[i * cols + i] = 1;
            }
            return fromArray(data, rows, cols);
        } else {
            throw new IllegalArgumentException("Unsupported dtype: " + dtype);
        }
    }

    /**
     * Creates a float tensor from a host float array.
     *
     * @param values host data
     * @param shape  tensor shape
     * @return uploaded tensor
     */
    public static Tensor fromArray(float[] values, int... shape) {
        Objects.requireNonNull(values, "values");
        long totalSize = checkLength(values.length, shape);
        long vramId = NativeEngine.allocVramUninit(totalSize);
        VramHandle h = new VramHandle(vramId);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, values);
            NativeEngine.writeVram(vramId, hostSegment, totalSize);
        } catch (RuntimeException | Error e) {
            h.release();
            throw e;
        }
        return new Tensor(DType.f32, shape, computeContiguousStrides(shape), 0L, h);
    }

    /**
     * Creates an integer tensor from a host int array.
     *
     * @param values host data
     * @param shape  tensor shape
     * @return uploaded tensor
     */
    public static Tensor fromArray(int[] values, int... shape) {
        Objects.requireNonNull(values, "values");
        long totalSize = checkLength(values.length, shape);
        long vramId = NativeEngine.allocVramUninit(totalSize);
        VramHandle h = new VramHandle(vramId);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocateFrom(ValueLayout.JAVA_INT, values);
            NativeEngine.writeVram(vramId, hostSegment, totalSize);
        } catch (RuntimeException | Error e) {
            h.release();
            throw e;
        }
        return new Tensor(DType.i32, shape, computeContiguousStrides(shape), 0L, h);
    }

    /**
     * Creates a tensor directly from a native or heap Panama {@link MemorySegment}.
     * If {@code source} is off-heap native memory, uploads directly without intermediate heap arrays.
     *
     * @param source source memory segment
     * @param dtype  data type
     * @param shape  tensor shape
     * @return uploaded tensor
     */
    public static Tensor fromSegment(MemorySegment source, DType dtype, int... shape) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(dtype, "dtype");
        ValidUtil.requireValidShape(shape);
        long totalElements = computeSize(shape);
        long minBytes = totalElements * dtype.byteSize();
        if (source.byteSize() < minBytes) {
            throw new IllegalArgumentException("Segment byteSize " + source.byteSize() + " is too small for "
                    + totalElements + " elements of " + dtype + " (requires " + minBytes + " bytes)");
        }
        long vramId = (dtype == DType.f16)
                ? NativeEngine.allocVramBytes(minBytes, false)
                : NativeEngine.allocVramUninit(totalElements);
        VramHandle h = new VramHandle(vramId);
        try {
            if (dtype == DType.f16) {
                NativeEngine.writeVramBytesSegment(vramId, source, minBytes);
            } else {
                NativeEngine.writeVramSegment(vramId, source, totalElements);
            }
        } catch (RuntimeException | Error e) {
            h.release();
            throw e;
        }
        return new Tensor(dtype, shape, computeContiguousStrides(shape), 0L, h);
    }

    /**
     * Creates an f16 tensor from host short values representing raw IEEE 754 half-precision float bits.
     *
     * @param bits  raw half-precision float bit values
     * @param shape tensor shape
     * @return uploaded f16 tensor
     */
    public static Tensor fromHalfBits(short[] bits, int... shape) {
        Objects.requireNonNull(bits, "bits");
        long totalSize = checkLength(bits.length, shape);
        long byteSize = Math.multiplyExact(totalSize, 2L);
        long vramId = NativeEngine.allocVramBytes(byteSize, false);
        VramHandle h = new VramHandle(vramId);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocateFrom(ValueLayout.JAVA_SHORT, bits);
            NativeEngine.writeVramBytesSegment(vramId, hostSegment, byteSize);
        } catch (RuntimeException | Error e) {
            h.release();
            throw e;
        }
        return new Tensor(DType.f16, shape, computeContiguousStrides(shape), 0L, h);
    }

    /**
     * Downloads an f16 tensor into a host array of raw half-precision float bits.
     *
     * @return short array of raw f16 bits
     * @throws IllegalStateException if this tensor is not f16
     */
    public short[] toHalfBitsArray() {
        ensureOpen();
        if (dtype != DType.f16) {
            throw new IllegalStateException("toHalfBitsArray is only supported for f16 tensors, got " + dtype);
        }
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Tensor size exceeds Integer.MAX_VALUE: " + size);
        }
        int len = (int) size;
        short[] result = new short[len];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSegment = arena.allocate(ValueLayout.JAVA_SHORT, len);
            copyTo(hostSegment);
            MemorySegment.copy(hostSegment, ValueLayout.JAVA_SHORT, 0L, result, 0, len);
        }
        return result;
    }

    /**
     * Converts this f16 tensor to f32 on the GPU and downloads it to a host float array.
     *
     * @return float array of tensor values
     */
    public float[] toFloatArrayFromF16() {
        try (Tensor f32 = cast(DType.f32)) {
            return f32.toFloatArray();
        }
    }

    // ------------------------------------------------------------------------------------------
    // Panama FFM Segment I/O
    // ------------------------------------------------------------------------------------------

    /**
     * Downloads tensor data directly into the destination {@link MemorySegment}.
     * If this tensor is non-contiguous or has non-zero offset, materializes it prior to download.
     *
     * @param destination target memory segment
     */
    public void copyTo(MemorySegment destination) {
        ensureOpen();
        Objects.requireNonNull(destination, "destination");
        if (destination.isReadOnly()) {
            throw new IllegalArgumentException("Destination segment is read-only");
        }
        long minBytes = size * dtype.byteSize();
        if (destination.byteSize() < minBytes) {
            throw new IllegalArgumentException("Destination segment byteSize " + destination.byteSize()
                    + " is too small for " + size + " elements (requires " + minBytes + " bytes)");
        }
        if (isContiguous() && elementOffset == 0L) {
            if (dtype == DType.f16) {
                NativeEngine.downloadVramBytesSegment(handle.vramId(), destination, minBytes);
            } else {
                NativeEngine.downloadVramSegment(handle.vramId(), destination, size);
            }
        } else {
            try (Tensor contig = contiguous()) {
                contig.copyTo(destination);
            }
        }
    }

    /**
     * Uploads data directly from the source {@link MemorySegment} into this tensor.
     * Requires this tensor to be contiguous, non-offset, and non-broadcast.
     *
     * @param source source memory segment
     */
    public void copyFrom(MemorySegment source) {
        ensureOpen();
        Objects.requireNonNull(source, "source");
        if (!isContiguous() || elementOffset != 0L || hasZeroStride(shape, strides)) {
            throw new IllegalArgumentException("copyFrom requires a contiguous, non-offset, non-broadcast target tensor");
        }
        long minBytes = size * dtype.byteSize();
        if (source.byteSize() < minBytes) {
            throw new IllegalArgumentException("Source segment byteSize " + source.byteSize()
                    + " is too small for " + size + " elements (requires " + minBytes + " bytes)");
        }
        if (dtype == DType.f16) {
            NativeEngine.writeVramBytesSegment(handle.vramId(), source, minBytes);
        } else {
            NativeEngine.writeVramSegment(handle.vramId(), source, size);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Lifecycle & State
    // ------------------------------------------------------------------------------------------

    /**
     * Returns true if this tensor reference has been closed.
     *
     * @return true if closed
     */
    public boolean isClosed() {
        return closed.get();
    }

    /**
     * Asserts that this tensor reference is still open.
     *
     * @throws IllegalStateException if already closed
     */
    public void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("Tensor has already been closed");
        }
    }

    /**
     * Closes this tensor reference. If all references to the underlying VRAM buffer are closed, frees native VRAM.
     */
    @Override
    public void close() {
        cleanable.clean();
    }

    /**
     * Returns the data type of this tensor.
     *
     * @return data type
     */
    public DType getDType() {
        return dtype;
    }

    /**
     * Returns a copy of the tensor shape.
     *
     * @return shape array
     */
    public int[] getShape() {
        return shape.clone();
    }

    /**
     * Returns a copy of the tensor strides.
     *
     * @return strides array
     */
    public int[] getStrides() {
        return strides.clone();
    }

    /**
     * Returns total number of logical elements in this tensor.
     *
     * @return element count
     */
    public long getSize() {
        return size;
    }

    /**
     * Returns the number of dimensions (rank) of this tensor.
     *
     * @return rank
     */
    public int dim() {
        return shape.length;
    }

    /**
     * Returns the memory tier residency of the underlying VRAM buffer.
     *
     * @return residency tier
     */
    public Residency residency() {
        ensureOpen();
        return NativeEngine.residency(handle.vramId());
    }

    /**
     * Pushes this tensor out of VRAM to host RAM or disk.
     */
    public void evict() {
        ensureOpen();
        NativeEngine.evict(handle.vramId());
    }

    // ------------------------------------------------------------------------------------------
    // Zero-Copy Strided Views, Layout & Materialization
    // ------------------------------------------------------------------------------------------

    /**
     * Returns true if elements are tightly packed in standard C row-major order.
     *
     * @return true if contiguous
     */
    public boolean isContiguous() {
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (shape[i] == 1) continue;
            if (strides[i] != expected) {
                return false;
            }
            expected *= shape[i];
        }
        return true;
    }

    /**
     * Returns a contiguous copy or view of this tensor.
     * If this tensor is already contiguous with zero offset, returns a retained zero-copy view.
     *
     * @return contiguous tensor
     */
    public Tensor contiguous() {
        ensureOpen();
        if (isContiguous() && elementOffset == 0L) {
            handle.retain();
            return new Tensor(dtype, shape, strides, 0L, handle);
        }
        Tensor res = empty(dtype, shape);
        try {
            NativeEngine.copyStrided(handle.vramId(), res.handle.vramId(), elementOffset, dim(), shape, strides, size);
            return res;
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Returns a brand-new contiguous copy of this tensor backed by an independent VRAM buffer.
     *
     * @return copy of this tensor
     */
    public Tensor copy() {
        ensureOpen();
        Tensor res = empty(dtype, shape);
        try {
            NativeEngine.copyStrided(handle.vramId(), res.handle.vramId(), elementOffset, dim(), shape, strides, size);
            return res;
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Reshapes this tensor to the specified shape, supporting at most one {@code -1} inferred dimension.
     * If this tensor is contiguous, returns a retained zero-copy view with row-major strides.
     *
     * @param newShape new dimension extents
     * @return reshaped tensor
     */
    public Tensor reshape(int... newShape) {
        ensureOpen();
        int[] resolvedShape = ValidUtil.inferReshape(newShape, size);
        if (isContiguous()) {
            handle.retain();
            return new Tensor(dtype, resolvedShape, computeContiguousStrides(resolvedShape), elementOffset, handle);
        }
        Tensor res = empty(dtype, resolvedShape);
        try {
            NativeEngine.copyStrided(handle.vramId(), res.handle.vramId(), elementOffset, dim(), shape, strides, size);
            return res;
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Flattens this tensor into a 1-D tensor.
     *
     * @return 1-D flattened tensor
     */
    public Tensor flatten() {
        return reshape(-1);
    }

    /**
     * Returns a zero-copy transposed view reversing all dimensions.
     *
     * @return reversed axis view
     */
    public Tensor T() {
        return transposeView();
    }

    /**
     * Returns a zero-copy transposed view reversing all dimensions.
     *
     * @return reversed axis view
     */
    public Tensor transposeView() {
        ensureOpen();
        int n = shape.length;
        int[] revShape = new int[n];
        int[] revStrides = new int[n];
        for (int i = 0; i < n; i++) {
            revShape[i] = shape[n - 1 - i];
            revStrides[i] = strides[n - 1 - i];
        }
        handle.retain();
        return new Tensor(dtype, revShape, revStrides, elementOffset, handle);
    }

    /**
     * Returns a zero-copy view permuting axes according to {@code axes}.
     *
     * @param axes axis permutation
     * @return permuted view
     */
    public Tensor permute(int... axes) {
        ensureOpen();
        ValidUtil.validatePermutation(axes, dim());
        int n = shape.length;
        int[] newShape = new int[n];
        int[] newStrides = new int[n];
        for (int i = 0; i < n; i++) {
            newShape[i] = shape[axes[i]];
            newStrides[i] = strides[axes[i]];
        }
        handle.retain();
        return new Tensor(dtype, newShape, newStrides, elementOffset, handle);
    }

    /**
     * Removes all size-1 dimensions from the shape, returning a zero-copy view.
     *
     * @return squeezed view
     */
    public Tensor squeeze() {
        ensureOpen();
        int nonOnes = 0;
        for (int d : shape) if (d > 1) nonOnes++;
        if (nonOnes == 0) {
            handle.retain();
            return new Tensor(dtype, new int[]{1}, new int[]{1}, elementOffset, handle);
        }
        int[] newShape = new int[nonOnes];
        int[] newStrides = new int[nonOnes];
        int idx = 0;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] > 1) {
                newShape[idx] = shape[i];
                newStrides[idx] = strides[i];
                idx++;
            }
        }
        handle.retain();
        return new Tensor(dtype, newShape, newStrides, elementOffset, handle);
    }

    /**
     * Removes the size-1 dimension at {@code axis}, returning a zero-copy view.
     *
     * @param axis dimension axis to squeeze
     * @return squeezed view
     */
    public Tensor squeeze(int axis) {
        ensureOpen();
        if (axis < 0) axis += dim();
        if (axis < 0 || axis >= dim()) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for dim " + dim());
        }
        if (shape[axis] != 1) {
            throw new IllegalArgumentException("Cannot squeeze dimension " + axis + " with size " + shape[axis]);
        }
        if (dim() == 1) {
            handle.retain();
            return new Tensor(dtype, new int[]{1}, new int[]{1}, elementOffset, handle);
        }
        int[] newShape = new int[dim() - 1];
        int[] newStrides = new int[dim() - 1];
        int idx = 0;
        for (int i = 0; i < dim(); i++) {
            if (i != axis) {
                newShape[idx] = shape[i];
                newStrides[idx] = strides[i];
                idx++;
            }
        }
        handle.retain();
        return new Tensor(dtype, newShape, newStrides, elementOffset, handle);
    }

    /**
     * Inserts a dimension of size 1 at {@code axis}, returning a zero-copy view.
     *
     * @param axis position to insert dimension
     * @return expanded view
     */
    public Tensor expandDims(int axis) {
        ensureOpen();
        if (axis < 0) axis += (dim() + 1);
        if (axis < 0 || axis > dim()) {
            throw new IllegalArgumentException("Axis " + axis + " is out of bounds for expanding rank " + dim());
        }
        int[] newShape = new int[dim() + 1];
        int[] newStrides = new int[dim() + 1];
        int nextStride = axis < dim() ? strides[axis] : 1;
        int idx = 0;
        for (int i = 0; i <= dim(); i++) {
            if (i == axis) {
                newShape[i] = 1;
                newStrides[i] = nextStride;
            } else {
                newShape[i] = shape[idx];
                newStrides[i] = strides[idx];
                idx++;
            }
        }
        handle.retain();
        return new Tensor(dtype, newShape, newStrides, elementOffset, handle);
    }

    /**
     * Alias for {@link #expandDims(int)}.
     *
     * @param axis position to insert dimension
     * @return unsqueezed view
     */
    public Tensor unsqueeze(int axis) {
        return expandDims(axis);
    }

    /**
     * Broadcasts this tensor to {@code targetShape} using NumPy rules, returning a zero-copy view.
     *
     * @param targetShape broadcast target shape
     * @return broadcast view with stride 0 along broadcast dimensions
     */
    public Tensor broadcastTo(int... targetShape) {
        ensureOpen();
        ValidUtil.requireValidShape(targetShape);
        int[] bcastShape = ValidUtil.broadcastShape(shape, targetShape);
        if (!Arrays.equals(bcastShape, targetShape)) {
            throw new IllegalArgumentException("Cannot broadcast shape " + Arrays.toString(shape) + " to " + Arrays.toString(targetShape));
        }
        int[] bcastStrides = ValidUtil.broadcastStrides(shape, strides, targetShape);
        handle.retain();
        return new Tensor(dtype, targetShape, bcastStrides, elementOffset, handle);
    }

    /**
     * Slices this tensor along leading dimensions according to {@code slices}, returning a zero-copy view.
     *
     * @param slices dimension slices
     * @return sliced view
     */
    public Tensor slice(Slice... slices) {
        ensureOpen();
        if (slices == null) {
            slices = new Slice[0];
        }
        if (slices.length > dim()) {
            throw new IllegalArgumentException("Too many slices: " + slices.length + " for rank " + dim());
        }
        int[] newShape = new int[dim()];
        int[] newStrides = new int[dim()];
        long newOffset = elementOffset;
        for (int d = 0; d < dim(); d++) {
            Slice s = (d < slices.length && slices[d] != null) ? slices[d] : Slice.all();
            Slice.ResolvedSlice res = s.resolve(shape[d]);
            newShape[d] = res.length();
            newStrides[d] = strides[d] * res.step();
            newOffset += (long) res.start() * strides[d];
        }
        handle.retain();
        return new Tensor(dtype, newShape, newStrides, newOffset, handle);
    }

    /**
     * Reads a single float scalar from this tensor.
     *
     * @return scalar float value
     * @throws IllegalStateException if size != 1 or dtype != f32
     */
    public float item() {
        ensureOpen();
        if (size != 1) {
            throw new IllegalStateException("item() requires tensor with exactly 1 element, got size " + size);
        }
        if (dtype != DType.f32) {
            throw new IllegalStateException("item() requires f32 tensor, got " + dtype);
        }
        try (Tensor contig = contiguous()) {
            return contig.toFloatArray()[0];
        }
    }

    /**
     * Reads a single int scalar from this tensor.
     *
     * @return scalar int value
     * @throws IllegalStateException if size != 1 or dtype != i32
     */
    public int itemInt() {
        ensureOpen();
        if (size != 1) {
            throw new IllegalStateException("itemInt() requires tensor with exactly 1 element, got size " + size);
        }
        if (dtype != DType.i32) {
            throw new IllegalStateException("itemInt() requires i32 tensor, got " + dtype);
        }
        try (Tensor contig = contiguous()) {
            return contig.toIntArray()[0];
        }
    }

    // ------------------------------------------------------------------------------------------
    // Elementwise Arithmetic & NumPy Broadcasting
    // ------------------------------------------------------------------------------------------

    /**
     * Computes elementwise addition with automatic NumPy broadcasting.
     *
     * @param other right operand
     * @return newly allocated result tensor
     */
    public Tensor add(Tensor other) {
        ensureOpen();
        int[] targetShape = ValidUtil.broadcastShape(shape, other.getShape());
        Tensor res = empty(dtype, targetShape);
        try { return add(other, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }

    /**
     * Computes elementwise addition into caller-provided {@code res}.
     *
     * @param other right operand
     * @param res   destination tensor
     * @return {@code res}
     */
    public Tensor add(Tensor other, Tensor res) { ensureOpen(); return ArithmeticOps.add(this, other, res); }

    /**
     * Computes elementwise subtraction with automatic NumPy broadcasting.
     *
     * @param other right operand
     * @return newly allocated result tensor
     */
    public Tensor sub(Tensor other) {
        ensureOpen();
        int[] targetShape = ValidUtil.broadcastShape(shape, other.getShape());
        Tensor res = empty(dtype, targetShape);
        try { return sub(other, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }

    /**
     * Computes elementwise subtraction into caller-provided {@code res}.
     *
     * @param other right operand
     * @param res   destination tensor
     * @return {@code res}
     */
    public Tensor sub(Tensor other, Tensor res) { ensureOpen(); return ArithmeticOps.sub(this, other, res); }

    /**
     * Computes elementwise multiplication with automatic NumPy broadcasting.
     *
     * @param other right operand
     * @return newly allocated result tensor
     */
    public Tensor mul(Tensor other) {
        ensureOpen();
        int[] targetShape = ValidUtil.broadcastShape(shape, other.getShape());
        Tensor res = empty(dtype, targetShape);
        try { return mul(other, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }

    /**
     * Computes elementwise multiplication into caller-provided {@code res}.
     *
     * @param other right operand
     * @param res   destination tensor
     * @return {@code res}
     */
    public Tensor mul(Tensor other, Tensor res) { ensureOpen(); return ArithmeticOps.mul(this, other, res); }

    /**
     * Computes elementwise division with automatic NumPy broadcasting.
     *
     * @param other right operand
     * @return newly allocated result tensor
     */
    public Tensor div(Tensor other) {
        ensureOpen();
        int[] targetShape = ValidUtil.broadcastShape(shape, other.getShape());
        Tensor res = empty(dtype, targetShape);
        try { return div(other, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }

    /**
     * Computes elementwise division into caller-provided {@code res}.
     *
     * @param other right operand
     * @param res   destination tensor
     * @return {@code res}
     */
    public Tensor div(Tensor other, Tensor res) { ensureOpen(); return ArithmeticOps.div(this, other, res); }

    // Scalar arithmetic
    public Tensor add(float c) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            return add((int) c);
        }
        return lazy().add(c).eval();
    }
    public Tensor sub(float c) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            return sub((int) c);
        }
        return lazy().sub(c).eval();
    }
    public Tensor mul(float c) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            return mul((int) c);
        }
        return lazy().mul(c).eval();
    }
    public Tensor div(float c) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            return div((int) c);
        }
        return lazy().div(c).eval();
    }

    public Tensor add(float c, Tensor res) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            return add((int) c, res);
        }
        return lazy().add(c).eval(res);
    }
    public Tensor sub(float c, Tensor res) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            return sub((int) c, res);
        }
        return lazy().sub(c).eval(res);
    }
    public Tensor mul(float c, Tensor res) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            return mul((int) c, res);
        }
        return lazy().mul(c).eval(res);
    }
    public Tensor div(float c, Tensor res) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            return div((int) c, res);
        }
        return lazy().div(c).eval(res);
    }

    public Tensor add(int c) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            try (Tensor s = fromArray(new int[]{c}, 1)) {
                return add(s);
            }
        }
        return add((float) c);
    }
    public Tensor sub(int c) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            try (Tensor s = fromArray(new int[]{c}, 1)) {
                return sub(s);
            }
        }
        return sub((float) c);
    }
    public Tensor mul(int c) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            try (Tensor s = fromArray(new int[]{c}, 1)) {
                return mul(s);
            }
        }
        return mul((float) c);
    }
    public Tensor div(int c) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            try (Tensor s = fromArray(new int[]{c}, 1)) {
                return div(s);
            }
        }
        return div((float) c);
    }

    public Tensor add(int c, Tensor res) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            try (Tensor s = fromArray(new int[]{c}, 1)) {
                return add(s, res);
            }
        }
        return add((float) c, res);
    }
    public Tensor sub(int c, Tensor res) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            try (Tensor s = fromArray(new int[]{c}, 1)) {
                return sub(s, res);
            }
        }
        return sub((float) c, res);
    }
    public Tensor mul(int c, Tensor res) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            try (Tensor s = fromArray(new int[]{c}, 1)) {
                return mul(s, res);
            }
        }
        return mul((float) c, res);
    }
    public Tensor div(int c, Tensor res) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            try (Tensor s = fromArray(new int[]{c}, 1)) {
                return div(s, res);
            }
        }
        return div((float) c, res);
    }

    // ------------------------------------------------------------------------------------------
    // In-Place Operations
    // ------------------------------------------------------------------------------------------

    /**
     * Computes in-place addition {@code this += other}.
     *
     * @param other operand tensor
     * @return {@code this}
     */
    public Tensor addi(Tensor other) {
        ensureOpen();
        validateInPlaceTarget(other);
        return ArithmeticOps.add(this, other, this);
    }

    /**
     * Computes in-place subtraction {@code this -= other}.
     *
     * @param other operand tensor
     * @return {@code this}
     */
    public Tensor subi(Tensor other) {
        ensureOpen();
        validateInPlaceTarget(other);
        return ArithmeticOps.sub(this, other, this);
    }

    /**
     * Computes in-place multiplication {@code this *= other}.
     *
     * @param other operand tensor
     * @return {@code this}
     */
    public Tensor muli(Tensor other) {
        ensureOpen();
        validateInPlaceTarget(other);
        return ArithmeticOps.mul(this, other, this);
    }

    /**
     * Computes in-place division {@code this /= other}.
     *
     * @param other operand tensor
     * @return {@code this}
     */
    public Tensor divi(Tensor other) {
        ensureOpen();
        validateInPlaceTarget(other);
        return ArithmeticOps.div(this, other, this);
    }

    public Tensor addi(float scalar) { return add(scalar, this); }
    public Tensor subi(float scalar) { return sub(scalar, this); }
    public Tensor muli(float scalar) { return mul(scalar, this); }
    public Tensor divi(float scalar) { return div(scalar, this); }

    public Tensor addi(int scalar) {
        try (Tensor s = fromArray(new int[]{scalar}, 1)) {
            return addi(s);
        }
    }

    public Tensor subi(int scalar) {
        try (Tensor s = fromArray(new int[]{scalar}, 1)) {
            return subi(s);
        }
    }

    public Tensor muli(int scalar) {
        try (Tensor s = fromArray(new int[]{scalar}, 1)) {
            return muli(s);
        }
    }

    public Tensor divi(int scalar) {
        try (Tensor s = fromArray(new int[]{scalar}, 1)) {
            return divi(s);
        }
    }

    private void validateInPlaceTarget(Tensor other) {
        if (!isContiguous() || elementOffset != 0L || hasZeroStride(shape, strides)) {
            throw new IllegalArgumentException("In-place operations require a contiguous, non-offset, non-broadcast target tensor");
        }
        int[] bcast = ValidUtil.broadcastShape(shape, other.getShape());
        if (!Arrays.equals(bcast, shape)) {
            throw new IllegalArgumentException("Cannot broadcast operand shape " + Arrays.toString(other.getShape())
                    + " into in-place target shape " + Arrays.toString(shape));
        }
    }

    // ------------------------------------------------------------------------------------------
    // Reverse Scalar Operations
    // ------------------------------------------------------------------------------------------

    /**
     * Computes reverse scalar subtraction {@code scalar - this} on GPU.
     *
     * @param scalar left scalar operand
     * @return newly allocated result tensor
     */
    public Tensor rsub(float scalar) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            return rsub((int) scalar);
        }
        return lazy().rsub(scalar).eval();
    }

    /**
     * Computes reverse scalar subtraction {@code scalar - this} into {@code res}.
     *
     * @param scalar left scalar operand
     * @param res    destination tensor
     * @return {@code res}
     */
    public Tensor rsub(float scalar, Tensor res) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            return rsub((int) scalar, res);
        }
        return lazy().rsub(scalar).eval(res);
    }

    /**
     * Computes reverse scalar division {@code scalar / this} on GPU.
     *
     * @param scalar left scalar operand
     * @return newly allocated result tensor
     */
    public Tensor rdiv(float scalar) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            return rdiv((int) scalar);
        }
        return lazy().rdiv(scalar).eval();
    }

    /**
     * Computes reverse scalar division {@code scalar / this} into {@code res}.
     *
     * @param scalar left scalar operand
     * @param res    destination tensor
     * @return {@code res}
     */
    public Tensor rdiv(float scalar, Tensor res) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            return rdiv((int) scalar, res);
        }
        return lazy().rdiv(scalar).eval(res);
    }

    /**
     * Computes reverse scalar subtraction {@code scalar - this} on GPU for integers.
     *
     * @param scalar left scalar operand
     * @return newly allocated result tensor
     */
    public Tensor rsub(int scalar) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            try (Tensor s = fromArray(new int[]{scalar}, 1)) {
                return s.sub(this);
            }
        }
        return rsub((float) scalar);
    }

    /**
     * Computes reverse scalar subtraction {@code scalar - this} into {@code res} for integers.
     *
     * @param scalar left scalar operand
     * @param res    destination tensor
     * @return {@code res}
     */
    public Tensor rsub(int scalar, Tensor res) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            try (Tensor s = fromArray(new int[]{scalar}, 1)) {
                return s.sub(this, res);
            }
        }
        return rsub((float) scalar, res);
    }

    /**
     * Computes reverse scalar division {@code scalar / this} on GPU for integers.
     *
     * @param scalar left scalar operand
     * @return newly allocated result tensor
     */
    public Tensor rdiv(int scalar) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            try (Tensor s = fromArray(new int[]{scalar}, 1)) {
                return s.div(this);
            }
        }
        return rdiv((float) scalar);
    }

    /**
     * Computes reverse scalar division {@code scalar / this} into {@code res} for integers.
     *
     * @param scalar left scalar operand
     * @param res    destination tensor
     * @return {@code res}
     */
    public Tensor rdiv(int scalar, Tensor res) {
        ensureOpen();
        if (this.dtype == DType.i32) {
            try (Tensor s = fromArray(new int[]{scalar}, 1)) {
                return s.div(this, res);
            }
        }
        return rdiv((float) scalar, res);
    }

    // ------------------------------------------------------------------------------------------
    // Trigonometric
    // ------------------------------------------------------------------------------------------

    public Tensor sin()   { return trig(TrigOps.Function.SIN); }
    public Tensor cos()   { return trig(TrigOps.Function.COS); }
    public Tensor tan()   { return trig(TrigOps.Function.TAN); }
    public Tensor asin()  { return trig(TrigOps.Function.ASIN); }
    public Tensor acos()  { return trig(TrigOps.Function.ACOS); }
    public Tensor atan()  { return trig(TrigOps.Function.ATAN); }
    public Tensor sinh()  { return trig(TrigOps.Function.SINH); }
    public Tensor cosh()  { return trig(TrigOps.Function.COSH); }
    public Tensor tanh()  { return trig(TrigOps.Function.TANH); }
    public Tensor asinh() { return trig(TrigOps.Function.ASINH); }
    public Tensor acosh() { return trig(TrigOps.Function.ACOSH); }
    public Tensor atanh() { return trig(TrigOps.Function.ATANH); }

    public Tensor sin(Tensor res)   { ensureOpen(); return TrigOps.apply(TrigOps.Function.SIN, this, res); }
    public Tensor cos(Tensor res)   { ensureOpen(); return TrigOps.apply(TrigOps.Function.COS, this, res); }
    public Tensor tan(Tensor res)   { ensureOpen(); return TrigOps.apply(TrigOps.Function.TAN, this, res); }
    public Tensor asin(Tensor res)  { ensureOpen(); return TrigOps.apply(TrigOps.Function.ASIN, this, res); }
    public Tensor acos(Tensor res)  { ensureOpen(); return TrigOps.apply(TrigOps.Function.ACOS, this, res); }
    public Tensor atan(Tensor res)  { ensureOpen(); return TrigOps.apply(TrigOps.Function.ATAN, this, res); }
    public Tensor sinh(Tensor res)  { ensureOpen(); return TrigOps.apply(TrigOps.Function.SINH, this, res); }
    public Tensor cosh(Tensor res)  { ensureOpen(); return TrigOps.apply(TrigOps.Function.COSH, this, res); }
    public Tensor tanh(Tensor res)  { ensureOpen(); return TrigOps.apply(TrigOps.Function.TANH, this, res); }
    public Tensor asinh(Tensor res) { ensureOpen(); return TrigOps.apply(TrigOps.Function.ASINH, this, res); }
    public Tensor acosh(Tensor res) { ensureOpen(); return TrigOps.apply(TrigOps.Function.ACOSH, this, res); }
    public Tensor atanh(Tensor res) { ensureOpen(); return TrigOps.apply(TrigOps.Function.ATANH, this, res); }

    private Tensor trig(TrigOps.Function fn) {
        ensureOpen();
        ValidUtil.requireOperandF32(fn.opName(), dtype);
        Tensor res = empty(dtype, shape);
        try {
            return TrigOps.apply(fn, this, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    // ------------------------------------------------------------------------------------------
    // Exponential & Logarithmic
    // ------------------------------------------------------------------------------------------

    public Tensor exp()   { return exponential(ExponentialOps.Function.EXP); }
    public Tensor log()   { return exponential(ExponentialOps.Function.LOG); }
    public Tensor log2()  { return exponential(ExponentialOps.Function.LOG2); }
    public Tensor log10() { return exponential(ExponentialOps.Function.LOG10); }
    public Tensor sqrt()  { return exponential(ExponentialOps.Function.SQRT); }

    public Tensor exp(Tensor res)   { ensureOpen(); return ExponentialOps.apply(ExponentialOps.Function.EXP, this, res); }
    public Tensor log(Tensor res)   { ensureOpen(); return ExponentialOps.apply(ExponentialOps.Function.LOG, this, res); }
    public Tensor log2(Tensor res)  { ensureOpen(); return ExponentialOps.apply(ExponentialOps.Function.LOG2, this, res); }
    public Tensor log10(Tensor res) { ensureOpen(); return ExponentialOps.apply(ExponentialOps.Function.LOG10, this, res); }
    public Tensor sqrt(Tensor res)  { ensureOpen(); return ExponentialOps.apply(ExponentialOps.Function.SQRT, this, res); }

    private Tensor exponential(ExponentialOps.Function fn) {
        ensureOpen();
        ValidUtil.requireOperandF32(fn.opName(), dtype);
        Tensor res = empty(dtype, shape);
        try {
            return ExponentialOps.apply(fn, this, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    // ------------------------------------------------------------------------------------------
    // Extra Math & Clamping Operations
    // ------------------------------------------------------------------------------------------

    /**
     * Computes the elementwise absolute value.
     *
     * @return absolute value tensor
     */
    public Tensor abs() {
        ensureOpen();
        Tensor res = empty(dtype, shape);
        try {
            return abs(res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Computes the elementwise absolute value into {@code res}.
     *
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor abs(Tensor res) {
        ensureOpen();
        return dtype == DType.f32
                ? MathOps.applyUnaryF32(0, "abs", this, res)
                : MathOps.applyUnaryI32(0, "abs", this, res, 0, 0);
    }

    /**
     * In-place elementwise absolute value.
     *
     * @return {@code this}
     */
    public Tensor absi() {
        ensureOpen();
        validateInPlaceTarget(this);
        return abs(this);
    }

    /**
     * Computes the elementwise negation.
     *
     * @return negated tensor
     */
    public Tensor neg() {
        ensureOpen();
        Tensor res = empty(dtype, shape);
        try {
            return neg(res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Computes the elementwise negation into {@code res}.
     *
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor neg(Tensor res) {
        ensureOpen();
        return dtype == DType.f32
                ? MathOps.applyUnaryF32(1, "neg", this, res)
                : MathOps.applyUnaryI32(1, "neg", this, res, 0, 0);
    }

    /**
     * In-place elementwise negation.
     *
     * @return {@code this}
     */
    public Tensor negi() {
        ensureOpen();
        validateInPlaceTarget(this);
        return neg(this);
    }

    /**
     * Computes the elementwise reciprocal square root {@code 1 / sqrt(x)}.
     *
     * @return reciprocal square root tensor
     */
    public Tensor rsqrt() {
        ensureOpen();
        ValidUtil.requireOperandF32("rsqrt", dtype);
        Tensor res = empty(dtype, shape);
        try {
            return rsqrt(res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Computes the elementwise reciprocal square root into {@code res}.
     *
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor rsqrt(Tensor res) {
        ensureOpen();
        ValidUtil.requireOperandF32("rsqrt", dtype);
        return MathOps.applyUnaryF32(2, "rsqrt", this, res);
    }

    /**
     * Computes the elementwise floor function.
     *
     * @return floored tensor
     */
    public Tensor floor() {
        ensureOpen();
        ValidUtil.requireOperandF32("floor", dtype);
        Tensor res = empty(dtype, shape);
        try {
            return floor(res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Computes the elementwise floor function into {@code res}.
     *
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor floor(Tensor res) {
        ensureOpen();
        ValidUtil.requireOperandF32("floor", dtype);
        return MathOps.applyUnaryF32(3, "floor", this, res);
    }

    /**
     * Computes the elementwise ceiling function.
     *
     * @return ceiling tensor
     */
    public Tensor ceil() {
        ensureOpen();
        ValidUtil.requireOperandF32("ceil", dtype);
        Tensor res = empty(dtype, shape);
        try {
            return ceil(res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Computes the elementwise ceiling function into {@code res}.
     *
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor ceil(Tensor res) {
        ensureOpen();
        ValidUtil.requireOperandF32("ceil", dtype);
        return MathOps.applyUnaryF32(4, "ceil", this, res);
    }

    /**
     * Computes elementwise rounding to the nearest integer.
     *
     * @return rounded tensor
     */
    public Tensor round() {
        ensureOpen();
        ValidUtil.requireOperandF32("round", dtype);
        Tensor res = empty(dtype, shape);
        try {
            return round(res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Computes elementwise rounding to the nearest integer into {@code res}.
     *
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor round(Tensor res) {
        ensureOpen();
        ValidUtil.requireOperandF32("round", dtype);
        return MathOps.applyUnaryF32(5, "round", this, res);
    }

    /**
     * Computes the elementwise signum (-1, 0, or 1).
     *
     * @return signum tensor
     */
    public Tensor sign() {
        ensureOpen();
        Tensor res = empty(dtype, shape);
        try {
            return sign(res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Computes the elementwise signum into {@code res}.
     *
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor sign(Tensor res) {
        ensureOpen();
        return dtype == DType.f32
                ? MathOps.applyUnaryF32(6, "sign", this, res)
                : MathOps.applyUnaryI32(2, "sign", this, res, 0, 0);
    }

    /**
     * In-place elementwise signum.
     *
     * @return {@code this}
     */
    public Tensor signi() {
        ensureOpen();
        validateInPlaceTarget(this);
        return sign(this);
    }

    /**
     * Clamps values to the range {@code [min, max]}.
     *
     * @param min minimum value
     * @param max maximum value
     * @return clamped tensor
     */
    public Tensor clamp(float min, float max) {
        ensureOpen();
        ValidUtil.requireOperandF32("clamp", dtype);
        Tensor res = empty(dtype, shape);
        try {
            return clamp(min, max, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Clamps values to the range {@code [min, max]} into {@code res}.
     *
     * @param min minimum value
     * @param max maximum value
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor clamp(float min, float max, Tensor res) {
        ensureOpen();
        ValidUtil.requireOperandF32("clamp", dtype);
        return MathOps.clampF32(this, min, max, res);
    }

    /**
     * In-place clamping to the range {@code [min, max]}.
     *
     * @param min minimum value
     * @param max maximum value
     * @return {@code this}
     */
    public Tensor clampi(float min, float max) {
        ensureOpen();
        validateInPlaceTarget(this);
        return clamp(min, max, this);
    }

    /**
     * Clamps values to the integer range {@code [min, max]}.
     *
     * @param min minimum value
     * @param max maximum value
     * @return clamped tensor
     */
    public Tensor clamp(int min, int max) {
        ensureOpen();
        if (dtype != DType.i32) {
            throw new IllegalArgumentException("clamp(int, int) requires an i32 tensor, got " + dtype);
        }
        Tensor res = empty(dtype, shape);
        try {
            return clamp(min, max, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Clamps values to the integer range {@code [min, max]} into {@code res}.
     *
     * @param min minimum value
     * @param max maximum value
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor clamp(int min, int max, Tensor res) {
        ensureOpen();
        if (dtype != DType.i32) {
            throw new IllegalArgumentException("clamp(int, int) requires an i32 tensor, got " + dtype);
        }
        return MathOps.applyUnaryI32(3, "clamp", this, res, min, max);
    }

    /**
     * In-place clamping to the integer range {@code [min, max]}.
     *
     * @param min minimum value
     * @param max maximum value
     * @return {@code this}
     */
    public Tensor clampi(int min, int max) {
        ensureOpen();
        validateInPlaceTarget(this);
        return clamp(min, max, this);
    }

    /**
     * Alias for {@link #clamp(float, float)}.
     *
     * @param min minimum value
     * @param max maximum value
     * @return clipped tensor
     */
    public Tensor clip(float min, float max) { return clamp(min, max); }

    /**
     * Alias for {@link #clamp(float, float, Tensor)}.
     *
     * @param min minimum value
     * @param max maximum value
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor clip(float min, float max, Tensor res) { return clamp(min, max, res); }

    /**
     * Alias for {@link #clampi(float, float)}.
     *
     * @param min minimum value
     * @param max maximum value
     * @return {@code this}
     */
    public Tensor clipi(float min, float max) { return clampi(min, max); }

    /**
     * Alias for {@link #clamp(int, int)}.
     *
     * @param min minimum value
     * @param max maximum value
     * @return clipped tensor
     */
    public Tensor clip(int min, int max) { return clamp(min, max); }

    /**
     * Alias for {@link #clamp(int, int, Tensor)}.
     *
     * @param min minimum value
     * @param max maximum value
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor clip(int min, int max, Tensor res) { return clamp(min, max, res); }

    /**
     * Alias for {@link #clampi(int, int)}.
     *
     * @param min minimum value
     * @param max maximum value
     * @return {@code this}
     */
    public Tensor clipi(int min, int max) { return clampi(min, max); }

    // ------------------------------------------------------------------------------------------
    // Functional Binary Math: pow, maximum, minimum
    // ------------------------------------------------------------------------------------------

    /**
     * Computes the elementwise power {@code this ^ other} with broadcasting.
     *
     * @param other exponent tensor
     * @return result tensor
     */
    public Tensor pow(Tensor other) {
        ensureOpen();
        ValidUtil.requireOperandF32("pow", dtype);
        int[] outShape = ValidUtil.broadcastShape(shape, other.getShape());
        Tensor res = empty(DType.f32, outShape);
        try {
            return pow(other, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Computes the elementwise power {@code this ^ other} into {@code res}.
     *
     * @param other exponent tensor
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor pow(Tensor other, Tensor res) {
        ensureOpen();
        return MathOps.pow(this, other, res);
    }

    /**
     * Computes the elementwise power with a scalar exponent.
     *
     * @param exponent scalar exponent
     * @return result tensor
     */
    public Tensor pow(float exponent) {
        try (Tensor s = fromArray(new float[]{exponent}, 1)) {
            return pow(s);
        }
    }

    /**
     * Computes the elementwise maximum with {@code other} with broadcasting.
     *
     * @param other right operand
     * @return result tensor
     */
    public Tensor maximum(Tensor other) {
        ensureOpen();
        int[] outShape = ValidUtil.broadcastShape(shape, other.getShape());
        Tensor res = empty(dtype, outShape);
        try {
            return maximum(other, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Computes the elementwise maximum with {@code other} into {@code res}.
     *
     * @param other right operand
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor maximum(Tensor other, Tensor res) {
        ensureOpen();
        return MathOps.maxMin(true, this, other, res);
    }

    /**
     * Computes the elementwise maximum with a float scalar.
     *
     * @param scalar float scalar
     * @return result tensor
     */
    public Tensor maximum(float scalar) {
        try (Tensor s = fromArray(new float[]{scalar}, 1)) {
            return maximum(s);
        }
    }

    /**
     * Computes the elementwise maximum with an int scalar.
     *
     * @param scalar int scalar
     * @return result tensor
     */
    public Tensor maximum(int scalar) {
        try (Tensor s = fromArray(new int[]{scalar}, 1)) {
            return maximum(s);
        }
    }

    /**
     * Computes the elementwise minimum with {@code other} with broadcasting.
     *
     * @param other right operand
     * @return result tensor
     */
    public Tensor minimum(Tensor other) {
        ensureOpen();
        int[] outShape = ValidUtil.broadcastShape(shape, other.getShape());
        Tensor res = empty(dtype, outShape);
        try {
            return minimum(other, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Computes the elementwise minimum with {@code other} into {@code res}.
     *
     * @param other right operand
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor minimum(Tensor other, Tensor res) {
        ensureOpen();
        return MathOps.maxMin(false, this, other, res);
    }

    /**
     * Computes the elementwise minimum with a float scalar.
     *
     * @param scalar float scalar
     * @return result tensor
     */
    public Tensor minimum(float scalar) {
        try (Tensor s = fromArray(new float[]{scalar}, 1)) {
            return minimum(s);
        }
    }

    /**
     * Computes the elementwise minimum with an int scalar.
     *
     * @param scalar int scalar
     * @return result tensor
     */
    public Tensor minimum(int scalar) {
        try (Tensor s = fromArray(new int[]{scalar}, 1)) {
            return minimum(s);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Comparison Operations (returning DType.i32 0/1 mask)
    // ------------------------------------------------------------------------------------------

    private Tensor compareFresh(int cmpOp, String opName, Tensor other) {
        ensureOpen();
        int[] outShape = ValidUtil.broadcastShape(shape, other.getShape());
        Tensor res = empty(DType.i32, outShape);
        try {
            return MathOps.compare(cmpOp, opName, this, other, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Elementwise equal-to comparison {@code this == other}.
     *
     * @param other right operand
     * @return i32 mask tensor (1 if true, 0 if false)
     */
    public Tensor eq(Tensor other) { return compareFresh(0, "eq", other); }

    /**
     * Elementwise equal-to comparison into {@code res}.
     *
     * @param other right operand
     * @param res destination i32 tensor
     * @return {@code res}
     */
    public Tensor eq(Tensor other, Tensor res) { ensureOpen(); return MathOps.compare(0, "eq", this, other, res); }

    /**
     * Elementwise equal-to comparison with float scalar.
     *
     * @param scalar float scalar
     * @return i32 mask tensor
     */
    public Tensor eq(float scalar) {
        try (Tensor s = fromArray(new float[]{scalar}, 1)) { return eq(s); }
    }

    /**
     * Elementwise equal-to comparison with int scalar.
     *
     * @param scalar int scalar
     * @return i32 mask tensor
     */
    public Tensor eq(int scalar) {
        try (Tensor s = fromArray(new int[]{scalar}, 1)) { return eq(s); }
    }

    /**
     * Elementwise not-equal comparison {@code this != other}.
     *
     * @param other right operand
     * @return i32 mask tensor
     */
    public Tensor ne(Tensor other) { return compareFresh(1, "ne", other); }

    /**
     * Elementwise not-equal comparison into {@code res}.
     *
     * @param other right operand
     * @param res destination i32 tensor
     * @return {@code res}
     */
    public Tensor ne(Tensor other, Tensor res) { ensureOpen(); return MathOps.compare(1, "ne", this, other, res); }

    /**
     * Elementwise not-equal comparison with float scalar.
     *
     * @param scalar float scalar
     * @return i32 mask tensor
     */
    public Tensor ne(float scalar) {
        try (Tensor s = fromArray(new float[]{scalar}, 1)) { return ne(s); }
    }

    /**
     * Elementwise not-equal comparison with int scalar.
     *
     * @param scalar int scalar
     * @return i32 mask tensor
     */
    public Tensor ne(int scalar) {
        try (Tensor s = fromArray(new int[]{scalar}, 1)) { return ne(s); }
    }

    /**
     * Elementwise greater-than comparison {@code this > other}.
     *
     * @param other right operand
     * @return i32 mask tensor
     */
    public Tensor gt(Tensor other) { return compareFresh(2, "gt", other); }

    /**
     * Elementwise greater-than comparison into {@code res}.
     *
     * @param other right operand
     * @param res destination i32 tensor
     * @return {@code res}
     */
    public Tensor gt(Tensor other, Tensor res) { ensureOpen(); return MathOps.compare(2, "gt", this, other, res); }

    /**
     * Elementwise greater-than comparison with float scalar.
     *
     * @param scalar float scalar
     * @return i32 mask tensor
     */
    public Tensor gt(float scalar) {
        try (Tensor s = fromArray(new float[]{scalar}, 1)) { return gt(s); }
    }

    /**
     * Elementwise greater-than comparison with int scalar.
     *
     * @param scalar int scalar
     * @return i32 mask tensor
     */
    public Tensor gt(int scalar) {
        try (Tensor s = fromArray(new int[]{scalar}, 1)) { return gt(s); }
    }

    /**
     * Elementwise greater-or-equal comparison {@code this >= other}.
     *
     * @param other right operand
     * @return i32 mask tensor
     */
    public Tensor ge(Tensor other) { return compareFresh(3, "ge", other); }

    /**
     * Elementwise greater-or-equal comparison into {@code res}.
     *
     * @param other right operand
     * @param res destination i32 tensor
     * @return {@code res}
     */
    public Tensor ge(Tensor other, Tensor res) { ensureOpen(); return MathOps.compare(3, "ge", this, other, res); }

    /**
     * Elementwise greater-or-equal comparison with float scalar.
     *
     * @param scalar float scalar
     * @return i32 mask tensor
     */
    public Tensor ge(float scalar) {
        try (Tensor s = fromArray(new float[]{scalar}, 1)) { return ge(s); }
    }

    /**
     * Elementwise greater-or-equal comparison with int scalar.
     *
     * @param scalar int scalar
     * @return i32 mask tensor
     */
    public Tensor ge(int scalar) {
        try (Tensor s = fromArray(new int[]{scalar}, 1)) { return ge(s); }
    }

    /**
     * Elementwise less-than comparison {@code this < other}.
     *
     * @param other right operand
     * @return i32 mask tensor
     */
    public Tensor lt(Tensor other) { return compareFresh(4, "lt", other); }

    /**
     * Elementwise less-than comparison into {@code res}.
     *
     * @param other right operand
     * @param res destination i32 tensor
     * @return {@code res}
     */
    public Tensor lt(Tensor other, Tensor res) { ensureOpen(); return MathOps.compare(4, "lt", this, other, res); }

    /**
     * Elementwise less-than comparison with float scalar.
     *
     * @param scalar float scalar
     * @return i32 mask tensor
     */
    public Tensor lt(float scalar) {
        try (Tensor s = fromArray(new float[]{scalar}, 1)) { return lt(s); }
    }

    /**
     * Elementwise less-than comparison with int scalar.
     *
     * @param scalar int scalar
     * @return i32 mask tensor
     */
    public Tensor lt(int scalar) {
        try (Tensor s = fromArray(new int[]{scalar}, 1)) { return lt(s); }
    }

    /**
     * Elementwise less-or-equal comparison {@code this <= other}.
     *
     * @param other right operand
     * @return i32 mask tensor
     */
    public Tensor le(Tensor other) { return compareFresh(5, "le", other); }

    /**
     * Elementwise less-or-equal comparison into {@code res}.
     *
     * @param other right operand
     * @param res destination i32 tensor
     * @return {@code res}
     */
    public Tensor le(Tensor other, Tensor res) { ensureOpen(); return MathOps.compare(5, "le", this, other, res); }

    /**
     * Elementwise less-or-equal comparison with float scalar.
     *
     * @param scalar float scalar
     * @return i32 mask tensor
     */
    public Tensor le(float scalar) {
        try (Tensor s = fromArray(new float[]{scalar}, 1)) { return le(s); }
    }

    /**
     * Elementwise less-or-equal comparison with int scalar.
     *
     * @param scalar int scalar
     * @return i32 mask tensor
     */
    public Tensor le(int scalar) {
        try (Tensor s = fromArray(new int[]{scalar}, 1)) { return le(s); }
    }

    // ------------------------------------------------------------------------------------------
    // Masked Where (Ternary Select)
    // ------------------------------------------------------------------------------------------

    /**
     * Ternary where operation {@code this != 0 ? x : y} with 3-way broadcasting.
     *
     * @param x tensor chosen where condition is true (non-zero)
     * @param y tensor chosen where condition is false (zero)
     * @return result tensor with values selected from {@code x} or {@code y}
     */
    public Tensor where(Tensor x, Tensor y) {
        ensureOpen();
        int[] targetShape = ValidUtil.broadcastShape(ValidUtil.broadcastShape(shape, x.getShape()), y.getShape());
        Tensor res = empty(x.getDType(), targetShape);
        try {
            return where(x, y, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Ternary where operation {@code this != 0 ? x : y} into {@code res}.
     *
     * @param x tensor chosen where condition is true
     * @param y tensor chosen where condition is false
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor where(Tensor x, Tensor y, Tensor res) {
        ensureOpen();
        return MathOps.where(this, x, y, res);
    }

    /**
     * Ternary where with scalar {@code x} and tensor {@code y}.
     *
     * @param x scalar chosen where true
     * @param y tensor chosen where false
     * @return result tensor
     */
    public Tensor where(float x, Tensor y) {
        try (Tensor tx = fromArray(new float[]{x}, 1)) { return where(tx, y); }
    }

    /**
     * Ternary where with tensor {@code x} and scalar {@code y}.
     *
     * @param x tensor chosen where true
     * @param y scalar chosen where false
     * @return result tensor
     */
    public Tensor where(Tensor x, float y) {
        try (Tensor ty = fromArray(new float[]{y}, 1)) { return where(x, ty); }
    }

    /**
     * Ternary where with scalars {@code x} and {@code y}.
     *
     * @param x scalar chosen where true
     * @param y scalar chosen where false
     * @return result tensor
     */
    public Tensor where(float x, float y) {
        try (Tensor tx = fromArray(new float[]{x}, 1); Tensor ty = fromArray(new float[]{y}, 1)) { return where(tx, ty); }
    }

    /**
     * Ternary where with int scalar {@code x} and tensor {@code y}.
     *
     * @param x scalar chosen where true
     * @param y tensor chosen where false
     * @return result tensor
     */
    public Tensor where(int x, Tensor y) {
        try (Tensor tx = fromArray(new int[]{x}, 1)) { return where(tx, y); }
    }

    /**
     * Ternary where with tensor {@code x} and int scalar {@code y}.
     *
     * @param x tensor chosen where true
     * @param y scalar chosen where false
     * @return result tensor
     */
    public Tensor where(Tensor x, int y) {
        try (Tensor ty = fromArray(new int[]{y}, 1)) { return where(x, ty); }
    }

    /**
     * Ternary where with int scalars {@code x} and {@code y}.
     *
     * @param x scalar chosen where true
     * @param y scalar chosen where false
     * @return result tensor
     */
    public Tensor where(int x, int y) {
        try (Tensor tx = fromArray(new int[]{x}, 1); Tensor ty = fromArray(new int[]{y}, 1)) { return where(tx, ty); }
    }

    // ------------------------------------------------------------------------------------------
    // Type Casting
    // ------------------------------------------------------------------------------------------

    /**
     * Casts elements of this tensor to {@code targetDType}.
     *
     * @param targetDType desired target data type
     * @return cast tensor
     */
    public Tensor cast(DType targetDType) {
        ensureOpen();
        Objects.requireNonNull(targetDType, "targetDType");
        if (this.dtype == targetDType) {
            return copy();
        }
        Tensor res = empty(targetDType, shape.clone());
        try {
            return MathOps.cast(this, targetDType, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    // ------------------------------------------------------------------------------------------
    // Matmul & Transpose
    // ------------------------------------------------------------------------------------------

    /**
     * Matrix multiplication with {@code other}, supporting 1-D, 2-D, and batched 3-D/4-D+ tensors with broadcasting.
     *
     * @param other right operand
     * @return result tensor
     */
    public Tensor matmul(Tensor other) {
        ensureOpen();
        ValidUtil.requireOperandF32("matmul", dtype);
        if (other.getDType() != DType.f32 && other.getDType() != DType.f16) {
            throw new IllegalArgumentException("matmul right operand must be f32 or f16, got " + other.getDType());
        }
        int[] outShape;
        if (this.dim() <= 2 && other.dim() <= 2) {
            Dims d = ValidUtil.matmulDims(shape, other.getShape(), false, false);
            outShape = d.resultShape();
        } else {
            ValidUtil.BatchedMatmulPlan plan = ValidUtil.batchedMatmulPlan(shape, strides, other.getShape(), other.getStrides());
            outShape = plan.outputShape();
        }
        Tensor res = empty(DType.f32, outShape);
        try {
            return matmul(other, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Matrix multiplication with {@code other} into pre-allocated {@code res}.
     *
     * @param other right operand
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor matmul(Tensor other, Tensor res) {
        ensureOpen();
        if (this.dim() <= 2 && other.dim() <= 2) {
            return MatmulOps.matmul(this, other, res, false, false);
        } else {
            return MatmulOps.batchedMatmul(this, other, res, false, false);
        }
    }

    /**
     * Matrix multiplication with transposed second operand {@code B^T},
     * supporting 2-D {@code [M, K] @ [N, K]^T -> [M, N]} and
     * 3-D+ {@code [..., M, K] @ [N, K]^T -> [..., M, N]}.
     * Supports both f32 and f16 right operands.
     *
     * @param other right operand of shape [N, K]
     * @return result tensor of shape [..., M, N]
     */
    public Tensor matmulTransposedB(Tensor other) {
        return Linalg.matmulTransposedB(this, other);
    }

    /**
     * Matrix multiplication with transposed second operand {@code B^T} into caller-provided {@code res}.
     *
     * @param other right operand of shape [N, K]
     * @param res   destination tensor
     * @return {@code res}
     */
    public Tensor matmulTransposedB(Tensor other, Tensor res) {
        return Linalg.matmulTransposedB(this, other, res);
    }

    public Tensor transpose() {
        ensureOpen();
        ValidUtil.require2D("transpose", shape);
        Tensor res = empty(dtype, shape[1], shape[0]);
        try {
            return MatmulOps.transpose(this, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor transpose(Tensor res) {
        ensureOpen();
        return MatmulOps.transpose(this, res);
    }

    // ------------------------------------------------------------------------------------------
    // Neural Network Ops
    // ------------------------------------------------------------------------------------------

    public Tensor relu() { return activation(Activation.RELU, 0f); }
    public Tensor gelu() { return activation(Activation.GELU, 0f); }
    public Tensor silu() { return activation(Activation.SILU, 0f); }
    public Tensor sigmoid() { return activation(Activation.SIGMOID, 0f); }
    public Tensor softplus() { return activation(Activation.SOFTPLUS, 0f); }
    public Tensor leakyRelu(float slope) { return activation(Activation.LEAKY_RELU, slope); }
    public Tensor elu(float alpha) { return activation(Activation.ELU, alpha); }

    public Tensor relu(Tensor res) { ensureOpen(); return NnOps.activation(Activation.RELU, this, res, 0f); }
    public Tensor gelu(Tensor res) { ensureOpen(); return NnOps.activation(Activation.GELU, this, res, 0f); }
    public Tensor silu(Tensor res) { ensureOpen(); return NnOps.activation(Activation.SILU, this, res, 0f); }
    public Tensor sigmoid(Tensor res) { ensureOpen(); return NnOps.activation(Activation.SIGMOID, this, res, 0f); }
    public Tensor softplus(Tensor res) { ensureOpen(); return NnOps.activation(Activation.SOFTPLUS, this, res, 0f); }
    public Tensor leakyRelu(float slope, Tensor res) { ensureOpen(); return NnOps.activation(Activation.LEAKY_RELU, this, res, slope); }
    public Tensor elu(float alpha, Tensor res) { ensureOpen(); return NnOps.activation(Activation.ELU, this, res, alpha); }

    private Tensor activation(Activation act, float alpha) {
        ensureOpen();
        ValidUtil.requireOperandF32(act.name(), dtype);
        Tensor res = empty(dtype, shape);
        try {
            return NnOps.activation(act, this, res, alpha);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    // Reductions
    public float sum() { return reduceToScalar(Reduction.SUM); }
    public float mean() { return reduceToScalar(Reduction.MEAN); }
    public float maxValue() { return reduceToScalar(Reduction.MAX); }
    public float minValue() { return reduceToScalar(Reduction.MIN); }

    public float sumSquares() {
        ensureOpen();
        try (Tensor r = empty(DType.f32, 1)) {
            NnOps.reduceSquare(Reduction.SUM, this, r);
            return r.toFloatArray()[0];
        }
    }

    public float mse(Tensor target) {
        ensureOpen();
        try (Tensor r = empty(DType.f32, 1)) {
            NnOps.reduceDiffSquared(Reduction.MEAN, this, target, r);
            return r.toFloatArray()[0];
        }
    }

    public Tensor sum(Tensor res) { ensureOpen(); return NnOps.reduce(Reduction.SUM, this, res); }
    public Tensor mean(Tensor res) { ensureOpen(); return NnOps.reduce(Reduction.MEAN, this, res); }
    public Tensor maxValue(Tensor res) { ensureOpen(); return NnOps.reduce(Reduction.MAX, this, res); }
    public Tensor minValue(Tensor res) { ensureOpen(); return NnOps.reduce(Reduction.MIN, this, res); }

    private float reduceToScalar(Reduction op) {
        ensureOpen();
        try (Tensor r = empty(DType.f32, 1)) {
            NnOps.reduce(op, this, r);
            return r.toFloatArray()[0];
        }
    }

    public Tensor sumRows() { return reduceRowsFresh(Reduction.SUM); }
    public Tensor meanRows() { return reduceRowsFresh(Reduction.MEAN); }
    public Tensor maxRows() { return reduceRowsFresh(Reduction.MAX); }
    public Tensor minRows() { return reduceRowsFresh(Reduction.MIN); }

    public Tensor sumRows(Tensor res) { ensureOpen(); return NnOps.reduceRows(Reduction.SUM, this, res); }
    public Tensor meanRows(Tensor res) { ensureOpen(); return NnOps.reduceRows(Reduction.MEAN, this, res); }
    public Tensor maxRows(Tensor res) { ensureOpen(); return NnOps.reduceRows(Reduction.MAX, this, res); }
    public Tensor minRows(Tensor res) { ensureOpen(); return NnOps.reduceRows(Reduction.MIN, this, res); }

    public Tensor argmaxRows() { return argRowsFresh(true); }
    public Tensor argminRows() { return argRowsFresh(false); }

    public Tensor argmaxRows(Tensor res) { ensureOpen(); return NnOps.argmaxRows(this, res, true); }
    public Tensor argminRows(Tensor res) { ensureOpen(); return NnOps.argmaxRows(this, res, false); }

    private Tensor reduceRowsFresh(Reduction op) {
        ensureOpen();
        Rows r = ValidUtil.rows(shape, size);
        Tensor res = empty(DType.f32, r.reducedShape(shape));
        try {
            return NnOps.reduceRows(op, this, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    private Tensor argRowsFresh(boolean largest) {
        ensureOpen();
        Rows r = ValidUtil.rows(shape, size);
        Tensor res = empty(DType.i32, r.reducedShape(shape));
        try {
            return NnOps.argmaxRows(this, res, largest);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    // ------------------------------------------------------------------------------------------
    // General Axis Reductions & Statistics
    // ------------------------------------------------------------------------------------------

    private static int[] computeReducedShape(int[] shape, int normAxis, boolean keepDims) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        if (keepDims) {
            int[] out = shape.clone();
            out[normAxis] = 1;
            return out;
        } else {
            int[] out = new int[shape.length - 1];
            int idx = 0;
            for (int i = 0; i < shape.length; i++) {
                if (i != normAxis) {
                    out[idx++] = shape[i];
                }
            }
            return out;
        }
    }

    /**
     * Reduces along {@code axis} using summation.
     *
     * @param axis reduction axis (can be negative)
     * @param keepDims whether to retain reduced dimension with size 1
     * @return reduced tensor
     */
    public Tensor sum(int axis, boolean keepDims) {
        ensureOpen();
        int normAxis = ValidUtil.normalizeAxis(axis, shape.length);
        int[] outShape = computeReducedShape(shape, normAxis, keepDims);
        Tensor res = empty(DType.f32, outShape);
        try {
            return MathOps.reduceAxis(this, normAxis, 0, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Reduces along {@code axis} using summation with {@code keepDims = false}.
     *
     * @param axis reduction axis (can be negative)
     * @return reduced tensor
     */
    public Tensor sum(int axis) { return sum(axis, false); }

    /**
     * Reduces along {@code axis} using arithmetic mean.
     *
     * @param axis reduction axis (can be negative)
     * @param keepDims whether to retain reduced dimension with size 1
     * @return reduced tensor
     */
    public Tensor mean(int axis, boolean keepDims) {
        ensureOpen();
        int normAxis = ValidUtil.normalizeAxis(axis, shape.length);
        int[] outShape = computeReducedShape(shape, normAxis, keepDims);
        Tensor res = empty(DType.f32, outShape);
        try {
            return MathOps.reduceAxis(this, normAxis, 1, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Reduces along {@code axis} using arithmetic mean with {@code keepDims = false}.
     *
     * @param axis reduction axis (can be negative)
     * @return reduced tensor
     */
    public Tensor mean(int axis) { return mean(axis, false); }

    /**
     * Reduces along {@code axis} using maximum value.
     *
     * @param axis reduction axis (can be negative)
     * @param keepDims whether to retain reduced dimension with size 1
     * @return reduced tensor
     */
    public Tensor max(int axis, boolean keepDims) {
        ensureOpen();
        int normAxis = ValidUtil.normalizeAxis(axis, shape.length);
        int[] outShape = computeReducedShape(shape, normAxis, keepDims);
        Tensor res = empty(DType.f32, outShape);
        try {
            return MathOps.reduceAxis(this, normAxis, 2, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Reduces along {@code axis} using maximum value with {@code keepDims = false}.
     *
     * @param axis reduction axis (can be negative)
     * @return reduced tensor
     */
    public Tensor max(int axis) { return max(axis, false); }

    /**
     * Reduces along {@code axis} using minimum value.
     *
     * @param axis reduction axis (can be negative)
     * @param keepDims whether to retain reduced dimension with size 1
     * @return reduced tensor
     */
    public Tensor min(int axis, boolean keepDims) {
        ensureOpen();
        int normAxis = ValidUtil.normalizeAxis(axis, shape.length);
        int[] outShape = computeReducedShape(shape, normAxis, keepDims);
        Tensor res = empty(DType.f32, outShape);
        try {
            return MathOps.reduceAxis(this, normAxis, 3, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Reduces along {@code axis} using minimum value with {@code keepDims = false}.
     *
     * @param axis reduction axis (can be negative)
     * @return reduced tensor
     */
    public Tensor min(int axis) { return min(axis, false); }

    /**
     * Reduces along {@code axis} using multiplication product.
     *
     * @param axis reduction axis (can be negative)
     * @param keepDims whether to retain reduced dimension with size 1
     * @return reduced tensor
     */
    public Tensor prod(int axis, boolean keepDims) {
        ensureOpen();
        int normAxis = ValidUtil.normalizeAxis(axis, shape.length);
        int[] outShape = computeReducedShape(shape, normAxis, keepDims);
        Tensor res = empty(DType.f32, outShape);
        try {
            return MathOps.reduceAxis(this, normAxis, 4, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Reduces along {@code axis} using multiplication product with {@code keepDims = false}.
     *
     * @param axis reduction axis (can be negative)
     * @return reduced tensor
     */
    public Tensor prod(int axis) { return prod(axis, false); }

    /**
     * Computes the index of the maximum value along {@code axis}.
     *
     * @param axis reduction axis (can be negative)
     * @param keepDims whether to retain reduced dimension with size 1
     * @return i32 indices tensor
     */
    public Tensor argmax(int axis, boolean keepDims) {
        ensureOpen();
        int normAxis = ValidUtil.normalizeAxis(axis, shape.length);
        int[] outShape = computeReducedShape(shape, normAxis, keepDims);
        Tensor res = empty(DType.i32, outShape);
        try {
            return MathOps.argmaxAxis(this, normAxis, true, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Computes the index of the maximum value along {@code axis} with {@code keepDims = false}.
     *
     * @param axis reduction axis (can be negative)
     * @return i32 indices tensor
     */
    public Tensor argmax(int axis) { return argmax(axis, false); }

    /**
     * Computes the index of the minimum value along {@code axis}.
     *
     * @param axis reduction axis (can be negative)
     * @param keepDims whether to retain reduced dimension with size 1
     * @return i32 indices tensor
     */
    public Tensor argmin(int axis, boolean keepDims) {
        ensureOpen();
        int normAxis = ValidUtil.normalizeAxis(axis, shape.length);
        int[] outShape = computeReducedShape(shape, normAxis, keepDims);
        Tensor res = empty(DType.i32, outShape);
        try {
            return MathOps.argmaxAxis(this, normAxis, false, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Computes the index of the minimum value along {@code axis} with {@code keepDims = false}.
     *
     * @param axis reduction axis (can be negative)
     * @return i32 indices tensor
     */
    public Tensor argmin(int axis) { return argmin(axis, false); }

    /**
     * Computes the global product of all elements.
     *
     * @return product scalar value
     */
    public float prod() { return reduceToScalar(Reduction.PROD); }

    /**
     * Computes the global product of all elements into {@code res}.
     *
     * @param res destination tensor of size 1
     * @return {@code res}
     */
    public Tensor prod(Tensor res) { ensureOpen(); return NnOps.reduce(Reduction.PROD, this, res); }

    /**
     * Computes the product along the rows (axis 0 of a 2-D matrix).
     *
     * @return reduced tensor
     */
    public Tensor prodRows() { return reduceRowsFresh(Reduction.PROD); }

    /**
     * Computes the product along the rows into {@code res}.
     *
     * @param res destination tensor
     * @return {@code res}
     */
    public Tensor prodRows(Tensor res) { ensureOpen(); return NnOps.reduceRows(Reduction.PROD, this, res); }

    /**
     * Computes the Frobenius (L2) Euclidean norm of this tensor.
     *
     * @return scalar norm as a 1-D tensor of shape {@code [1]}
     */
    public Tensor norm() {
        ensureOpen();
        ValidUtil.requireOperandF32("norm", dtype);
        Tensor res = empty(DType.f32, 1);
        try {
            NnOps.reduceSquare(Reduction.SUM, this, res);
            return res.sqrt();
        } finally {
            res.close();
        }
    }

    /**
     * Computes the variance of elements with delta degrees of freedom {@code ddof}.
     *
     * @param ddof delta degrees of freedom (0 for population variance, 1 for sample variance)
     * @return scalar variance as a 1-D tensor of shape {@code [1]}
     */
    public Tensor var(int ddof) {
        ensureOpen();
        ValidUtil.requireOperandF32("var", dtype);
        if (ddof < 0 || size <= ddof) {
            throw new IllegalArgumentException("ddof must be non-negative and < total elements (" + size + "), got " + ddof);
        }
        float m = this.mean();
        try (Tensor diff = this.sub(m);
             Tensor sq = diff.mul(diff)) {
            float sumSq = sq.sum();
            float v = sumSq / (size - ddof);
            return fromArray(new float[]{v}, 1);
        }
    }

    /**
     * Computes the population variance (ddof = 0).
     *
     * @return scalar variance as a 1-D tensor of shape {@code [1]}
     */
    public Tensor var() { return var(0); }

    /**
     * Computes the standard deviation with delta degrees of freedom {@code ddof}.
     *
     * @param ddof delta degrees of freedom
     * @return scalar standard deviation as a 1-D tensor of shape {@code [1]}
     */
    public Tensor std(int ddof) {
        ensureOpen();
        try (Tensor v = var(ddof)) {
            return v.sqrt();
        }
    }

    /**
     * Computes the population standard deviation (ddof = 0).
     *
     * @return scalar standard deviation as a 1-D tensor of shape {@code [1]}
     */
    public Tensor std() { return std(0); }

    /**
     * Extracts the main diagonal of a 2-D matrix as a zero-copy strided 1-D view.
     *
     * @return 1-D diagonal view tensor
     */
    public Tensor diagonal() {
        ensureOpen();
        if (shape.length != 2) {
            throw new IllegalArgumentException("diagonal requires a 2-D tensor, got shape " + Arrays.toString(shape));
        }
        int d = Math.min(shape[0], shape[1]);
        int diagStride = strides[0] + strides[1];
        handle.retain();
        return new Tensor(dtype, new int[]{d}, new int[]{diagStride}, elementOffset, handle);
    }

    /**
     * Computes the trace (sum of diagonal elements) of a 2-D matrix.
     *
     * @return scalar trace as a 1-D tensor of shape {@code [1]}
     */
    public Tensor trace() {
        ensureOpen();
        try (Tensor diag = diagonal()) {
            Tensor res = empty(dtype, 1);
            return NnOps.reduce(Reduction.SUM, diag, res);
        }
    }

    /**
     * Computes the dot product with {@code other}.
     *
     * @param other right operand
     * @return scalar dot product as a 1-D tensor of shape {@code [1]}
     */
    public Tensor dot(Tensor other) {
        ensureOpen();
        ValidUtil.requireOperandF32("dot", dtype);
        ValidUtil.requireOperandF32("dot", other.getDType());
        if (shape.length == 1 && other.getShape().length == 1) {
            try (Tensor prod = this.mul(other)) {
                Tensor res = empty(DType.f32, 1);
                return NnOps.reduce(Reduction.SUM, prod, res);
            }
        }
        return this.matmul(other);
    }

    /**
     * Computes the cosine similarity with {@code other}.
     *
     * @param other right operand
     * @return scalar similarity as a 1-D tensor of shape {@code [1]}
     */
    public Tensor cosineSimilarity(Tensor other) {
        ensureOpen();
        ValidUtil.requireOperandF32("cosineSimilarity", dtype);
        ValidUtil.requireOperandF32("cosineSimilarity", other.getDType());
        try (Tensor dotProd = this.dot(other);
             Tensor normA = this.norm();
             Tensor normB = other.norm();
             Tensor denom = normA.mul(normB)) {
            return dotProd.div(denom);
        }
    }

    public Tensor softmax() { return softmaxFresh(false); }
    public Tensor logSoftmax() { return softmaxFresh(true); }

    public Tensor softmax(Tensor res) { ensureOpen(); return NnOps.softmaxRows(this, res, false); }
    public Tensor logSoftmax(Tensor res) { ensureOpen(); return NnOps.softmaxRows(this, res, true); }

    private Tensor softmaxFresh(boolean logarithmic) {
        ensureOpen();
        ValidUtil.requireOperandF32("softmax", dtype);
        Tensor res = empty(dtype, shape);
        try {
            return NnOps.softmaxRows(this, res, logarithmic);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor layerNorm(float eps) { return layerNormFresh(null, null, eps); }
    public Tensor layerNorm(Tensor gamma, Tensor beta, float eps) { return layerNormFresh(gamma, beta, eps); }
    public Tensor layerNorm(Tensor gamma, Tensor beta, float eps, Tensor res) {
        ensureOpen();
        return NnOps.layerNormRows(this, gamma, beta, res, eps);
    }

    public Tensor rmsNorm(Tensor gamma, float eps) {
        ensureOpen();
        ValidUtil.requireOperandF32("rmsNorm", dtype);
        Tensor res = empty(dtype, shape);
        try {
            return NnOps.rmsNormRows(this, gamma, res, eps);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor rmsNorm(Tensor gamma, float eps, Tensor res) {
        ensureOpen();
        return NnOps.rmsNormRows(this, gamma, res, eps);
    }

    private Tensor layerNormFresh(Tensor gamma, Tensor beta, float eps) {
        ensureOpen();
        ValidUtil.requireOperandF32("layerNorm", dtype);
        Tensor res = empty(dtype, shape);
        try {
            return NnOps.layerNormRows(this, gamma, beta, res, eps);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor crossEntropy(Tensor targets) {
        ensureOpen();
        Rows r = ValidUtil.rows(shape, size);
        Tensor res = empty(DType.f32, r.reducedShape(shape));
        try {
            return NnOps.crossEntropyRows(this, targets, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor crossEntropy(Tensor targets, Tensor res) {
        ensureOpen();
        return NnOps.crossEntropyRows(this, targets, res);
    }

    public Tensor addRowBias(Tensor bias) {
        ensureOpen();
        ValidUtil.requireOperandF32("addRowBias", dtype);
        Tensor res = empty(dtype, shape);
        try {
            return NnOps.addRowBias(this, bias, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor addRowBias(Tensor bias, Tensor res) {
        ensureOpen();
        return NnOps.addRowBias(this, bias, res);
    }

    public Tensor linear(Tensor weights, Tensor bias) {
        ensureOpen();
        Tensor res = matmul(weights);
        try {
            return bias == null ? res : NnOps.addRowBias(res, bias, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor linear(Tensor weights, Tensor bias, Tensor res) {
        ensureOpen();
        matmul(weights, res);
        return bias == null ? res : NnOps.addRowBias(res, bias, res);
    }

    /**
     * Look up embeddings from this embedding table for the given token IDs.
     * Supports both f32 and f16 embedding tables.
     *
     * @param tokenIds integer token tensor of any shape
     * @return newly allocated float embedding tensor
     */
    public Tensor embedding(Tensor tokenIds) {
        ensureOpen();
        return Nn.embedding(this, tokenIds);
    }

    /**
     * Look up embeddings from this embedding table into caller-provided {@code out}.
     * Supports both f32 and f16 embedding tables.
     *
     * @param tokenIds integer token tensor of any shape
     * @param out      destination tensor
     * @return {@code out}
     */
    public Tensor embedding(Tensor tokenIds, Tensor out) {
        ensureOpen();
        return Nn.embedding(this, tokenIds, out);
    }

    /**
     * Applies Rotary Position Embedding (RoPE) using the NeoX half-split convention.
     * In-place safe.
     *
     * @param positionOffset token position offset
     * @param theta          rotary embedding base theta (e.g. 10000.0f)
     * @return newly allocated tensor with rotary embeddings applied
     */
    public Tensor rope(int positionOffset, float theta) {
        ensureOpen();
        return Nn.rope(this, positionOffset, theta);
    }

    /**
     * Applies Rotary Position Embedding (RoPE) into caller-provided {@code out}.
     * In-place safe when {@code this == out}.
     *
     * @param positionOffset token position offset
     * @param theta          rotary embedding base theta (e.g. 10000.0f)
     * @param out            destination tensor (may be {@code this})
     * @return {@code out}
     */
    public Tensor rope(int positionOffset, float theta, Tensor out) {
        ensureOpen();
        return Nn.rope(this, positionOffset, theta, out);
    }

    // ------------------------------------------------------------------------------------------
    // Fusion
    // ------------------------------------------------------------------------------------------

    public FusedExpr lazy() {
        ensureOpen();
        return FusedExpr.of(this);
    }

    // ------------------------------------------------------------------------------------------
    // Readback
    // ------------------------------------------------------------------------------------------

    public float[] toFloatArray() {
        ensureOpen();
        if (this.dtype != DType.f32) {
            throw new IllegalStateException("Cannot readback " + this.dtype + " as float[]");
        }
        if (isContiguous() && elementOffset == 0L) {
            float[] result = new float[(int) this.size];
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment hostSegment = arena.allocate(ValueLayout.JAVA_FLOAT, this.size);
                NativeEngine.downloadVram(this.handle.vramId(), hostSegment, this.size);
                MemorySegment.copy(hostSegment, ValueLayout.JAVA_FLOAT, 0, result, 0, (int) this.size);
            }
            return result;
        }
        try (Tensor contig = contiguous()) {
            return contig.toFloatArray();
        }
    }

    /**
     * Convenience alias for {@link #toFloatArray()}. Downloads the tensor data to a newly allocated {@code float[]}.
     *
     * @return newly allocated float array containing tensor elements in row-major order
     * @throws IllegalStateException if this tensor is not of dtype {@link DType#f32} or has been closed
     */
    public float[] toArray() {
        return toFloatArray();
    }

    public int[] toIntArray() {
        ensureOpen();
        if (this.dtype != DType.i32) {
            throw new IllegalStateException("Cannot readback " + this.dtype + " as int[]");
        }
        if (isContiguous() && elementOffset == 0L) {
            int[] result = new int[(int) this.size];
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment hostSegment = arena.allocate(ValueLayout.JAVA_INT, this.size);
                NativeEngine.downloadVram(this.handle.vramId(), hostSegment, this.size);
                MemorySegment.copy(hostSegment, ValueLayout.JAVA_INT, 0, result, 0, (int) this.size);
            }
            return result;
        }
        try (Tensor contig = contiguous()) {
            return contig.toIntArray();
        }
    }

    /**
     * Convenience alias for {@link #toIntArray()}. Downloads the tensor data to a newly allocated {@code int[]}.
     *
     * @return newly allocated int array containing tensor elements in row-major order
     * @throws IllegalStateException if this tensor is not of dtype {@link DType#i32} or has been closed
     */
    public int[] toArrayInt() {
        return toIntArray();
    }

    // ------------------------------------------------------------------------------------------
    // Internal Utilities
    // ------------------------------------------------------------------------------------------

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

    private static boolean hasZeroStride(int[] shape, int[] strides) {
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] > 1 && strides[i] == 0) return true;
        }
        return false;
    }

    @Override
    public String toString() {
        return "Tensor(vramId=" + handle.vramId() + ", shape=" + Arrays.toString(shape)
                + ", strides=" + Arrays.toString(strides) + ", offset=" + elementOffset
                + ", dtype=" + dtype + ", size=" + size + ")";
    }
}
