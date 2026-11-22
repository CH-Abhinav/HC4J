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

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * An {@link AutoCloseable} deterministic LIFO resource scope for bulk VRAM and GPU resource management.
 * Resources tracked by this scope are automatically closed in reverse acquisition order upon scope exit.
 */
public final class GpuScope implements AutoCloseable {

    private final List<AutoCloseable> resources = new ArrayList<>();
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private GpuScope() {}

    /**
     * Opens a new GPU resource tracking scope.
     *
     * @return newly opened GPU scope
     */
    public static GpuScope open() {
        return new GpuScope();
    }

    /**
     * Registers an {@link AutoCloseable} resource with this scope for deterministic cleanup upon close.
     *
     * @param <T>      resource type
     * @param resource resource to track
     * @return the tracked resource
     * @throws IllegalStateException if this scope is already closed
     * @throws NullPointerException  if resource is null
     */
    public synchronized <T extends AutoCloseable> T track(T resource) {
        if (closed.get()) {
            throw new IllegalStateException("GpuScope is already closed");
        }
        Objects.requireNonNull(resource, "resource");
        resources.add(resource);
        return resource;
    }

    /**
     * Removes an {@link AutoCloseable} resource from tracking so that it survives when this scope closes.
     *
     * @param <T>      resource type
     * @param resource resource to detach
     * @return the detached resource
     * @throws IllegalArgumentException if the resource is not currently tracked by this scope
     * @throws NullPointerException     if resource is null
     */
    public synchronized <T extends AutoCloseable> T detach(T resource) {
        Objects.requireNonNull(resource, "resource");
        boolean removed = false;
        for (int i = resources.size() - 1; i >= 0; i--) {
            if (resources.get(i) == resource) {
                resources.remove(i);
                removed = true;
                break;
            }
        }
        if (!removed) {
            throw new IllegalArgumentException("Resource is not tracked by this GpuScope: " + resource);
        }
        return resource;
    }

    /**
     * Returns the number of resources currently tracked by this scope.
     *
     * @return tracked resource count
     */
    public synchronized int trackedCount() {
        return resources.size();
    }

    /**
     * Returns true if this scope has been closed.
     *
     * @return true if closed
     */
    public boolean isClosed() {
        return closed.get();
    }

    /**
     * Creates and tracks a zero-initialized tensor.
     *
     * @param dtype data type
     * @param shape tensor shape
     * @return tracked zero tensor
     */
    public Tensor zeros(DType dtype, int... shape) {
        return track(Tensor.zeros(dtype, shape));
    }

    /**
     * Creates and tracks a one-initialized tensor.
     *
     * @param dtype data type
     * @param shape tensor shape
     * @return tracked ones tensor
     */
    public Tensor ones(DType dtype, int... shape) {
        return track(Tensor.ones(dtype, shape));
    }

    /**
     * Creates and tracks an uninitialized tensor.
     *
     * @param dtype data type
     * @param shape tensor shape
     * @return tracked uninitialized tensor
     */
    public Tensor empty(DType dtype, int... shape) {
        return track(Tensor.empty(dtype, shape));
    }

    /**
     * Creates and tracks a float tensor filled with {@code value}.
     *
     * @param value fill value
     * @param shape tensor shape
     * @return tracked full float tensor
     */
    public Tensor full(float value, int... shape) {
        return track(Tensor.full(value, shape));
    }

    /**
     * Creates and tracks an integer tensor filled with {@code value}.
     *
     * @param value fill value
     * @param shape tensor shape
     * @return tracked full int tensor
     */
    public Tensor full(int value, int... shape) {
        return track(Tensor.full(value, shape));
    }

    /**
     * Creates and tracks a 1-D float tensor with values in {@code [start, stop)} with given step.
     *
     * @param start start value
     * @param stop  stop value
     * @param step  step size
     * @return tracked arange tensor
     */
    public Tensor arange(float start, float stop, float step) {
        return track(Tensor.arange(start, stop, step));
    }

    /**
     * Creates and tracks a 1-D integer tensor with values in {@code [0, stop)}.
     *
     * @param stop stop value
     * @return tracked arange tensor
     */
    public Tensor arange(int stop) {
        return track(Tensor.arange(stop));
    }

    /**
     * Creates and tracks a 1-D integer tensor with values in {@code [start, stop)} with given step.
     *
     * @param start start value
     * @param stop  stop value
     * @param step  step size
     * @return tracked arange tensor
     */
    public Tensor arange(int start, int stop, int step) {
        return track(Tensor.arange(start, stop, step));
    }

    /**
     * Creates and tracks a 1-D float tensor with {@code num} evenly spaced values in {@code [start, stop]}.
     *
     * @param start start value
     * @param stop  stop value (inclusive)
     * @param num   number of samples
     * @return tracked linspace tensor
     */
    public Tensor linspace(float start, float stop, int num) {
        return track(Tensor.linspace(start, stop, num));
    }

    /**
     * Creates and tracks an identity matrix of size {@code n x n} with f32 data type.
     *
     * @param n row and column count
     * @return tracked identity tensor
     */
    public Tensor eye(int n) {
        return track(Tensor.eye(n));
    }

    /**
     * Creates and tracks an identity matrix of size {@code n x n}.
     *
     * @param dtype data type
     * @param n     row and column count
     * @return tracked identity tensor
     */
    public Tensor eye(DType dtype, int n) {
        return track(Tensor.eye(dtype, n));
    }

    /**
     * Creates and tracks a 2-D identity matrix of size {@code rows x cols}.
     *
     * @param dtype data type
     * @param rows  row count
     * @param cols  column count
     * @return tracked identity tensor
     */
    public Tensor eye(DType dtype, int rows, int cols) {
        return track(Tensor.eye(dtype, rows, cols));
    }

    /**
     * Creates and tracks a tensor uploaded from host float array.
     *
     * @param values float array
     * @param shape  tensor shape
     * @return tracked tensor
     */
    public Tensor fromArray(float[] values, int... shape) {
        return track(Tensor.fromArray(values, shape));
    }

    /**
     * Creates and tracks a tensor uploaded from host int array.
     *
     * @param values int array
     * @param shape  tensor shape
     * @return tracked tensor
     */
    public Tensor fromArray(int[] values, int... shape) {
        return track(Tensor.fromArray(values, shape));
    }

    /**
     * Creates and tracks a tensor uploaded directly from an off-heap or on-heap {@link MemorySegment}.
     *
     * @param source source memory segment
     * @param dtype  data type
     * @param shape  tensor shape
     * @return tracked tensor
     */
    public Tensor fromSegment(MemorySegment source, DType dtype, int... shape) {
        return track(Tensor.fromSegment(source, dtype, shape));
    }

    /**
     * Closes all resources tracked by this scope in reverse acquisition order (LIFO).
     * Any exceptions thrown during cleanup are aggregated via suppressed exceptions.
     */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            List<AutoCloseable> toClose;
            synchronized (this) {
                toClose = new ArrayList<>(resources);
                resources.clear();
            }
            Throwable error = null;
            for (int i = toClose.size() - 1; i >= 0; i--) {
                try {
                    toClose.get(i).close();
                } catch (Throwable t) {
                    if (error == null) {
                        error = t;
                    } else {
                        error.addSuppressed(t);
                    }
                }
            }
            if (error != null) {
                if (error instanceof RuntimeException re) throw re;
                if (error instanceof Error er) throw er;
                throw new RuntimeException("Exception during GpuScope close", error);
            }
        }
    }
}
