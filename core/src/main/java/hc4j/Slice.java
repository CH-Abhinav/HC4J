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

/**
 * Multi-dimensional slice definition supporting Python-style start, stop, step slicing and negative indices.
 *
 * @param start start index (inclusive, or {@link #UNBOUNDED_START})
 * @param stop  stop index (exclusive, or {@link #UNBOUNDED_STOP})
 * @param step  step stride (strictly positive)
 */
public record Slice(int start, int stop, int step) {

    /** Sentinel representing an unbounded start index. */
    public static final int UNBOUNDED_START = Integer.MIN_VALUE;

    /** Sentinel representing an unbounded stop index. */
    public static final int UNBOUNDED_STOP = Integer.MAX_VALUE;

    /**
     * Compact constructor validating step stride.
     *
     * @param start start index
     * @param stop  stop index
     * @param step  step stride
     * @throws IllegalArgumentException if {@code step <= 0}
     */
    public Slice {
        if (step <= 0) {
            throw new IllegalArgumentException("Slice step must be strictly positive, got: " + step);
        }
    }

    /**
     * Represents a slice resolved against a concrete dimension size.
     *
     * @param start  clamped non-negative start index
     * @param stop   clamped non-negative stop index
     * @param step   step stride
     * @param length number of elements selected along this dimension
     */
    public record ResolvedSlice(int start, int stop, int step, int length) {}

    /**
     * Creates a slice that selects the entire dimension ({@code :}).
     *
     * @return unbounded slice with step 1
     */
    public static Slice all() {
        return new Slice(UNBOUNDED_START, UNBOUNDED_STOP, 1);
    }

    /**
     * Creates a slice from the beginning up to {@code stop} ({@code :stop}).
     *
     * @param stop exclusive stop index
     * @return slice from beginning to {@code stop}
     */
    public static Slice to(int stop) {
        return new Slice(UNBOUNDED_START, stop, 1);
    }

    /**
     * Creates a slice from {@code start} to the end of the dimension ({@code start:}).
     *
     * @param start inclusive start index
     * @return slice from {@code start} to end
     */
    public static Slice from(int start) {
        return new Slice(start, UNBOUNDED_STOP, 1);
    }

    /**
     * Creates a slice across {@code [start, stop)} with step 1 ({@code start:stop}).
     *
     * @param start inclusive start index
     * @param stop  exclusive stop index
     * @return slice across {@code [start, stop)}
     */
    public static Slice range(int start, int stop) {
        return new Slice(start, stop, 1);
    }

    /**
     * Creates a slice across {@code [start, stop)} with step {@code step} ({@code start:stop:step}).
     *
     * @param start inclusive start index
     * @param stop  exclusive stop index
     * @param step  step stride
     * @return slice across {@code [start, stop)} with step {@code step}
     * @throws IllegalArgumentException if {@code step <= 0}
     */
    public static Slice range(int start, int stop, int step) {
        return new Slice(start, stop, step);
    }

    /**
     * Creates a slice selecting the entire dimension with a strided step ({@code ::step}).
     *
     * @param step step stride
     * @return unbounded slice with step {@code step}
     * @throws IllegalArgumentException if {@code step <= 0}
     */
    public static Slice step(int step) {
        return new Slice(UNBOUNDED_START, UNBOUNDED_STOP, step);
    }

    /**
     * Creates a slice selecting a single element at {@code index} along the dimension.
     * Supports negative indices (e.g. -1 for the last element).
     *
     * @param index element index (supports negative values)
     * @return slice selecting the element at {@code index}
     */
    public static Slice at(int index) {
        if (index == -1) {
            return new Slice(-1, UNBOUNDED_STOP, 1);
        } else if (index < -1) {
            return new Slice(index, index + 1, 1);
        } else {
            return new Slice(index, index + 1, 1);
        }
    }

    /**
     * Resolves this slice against a concrete dimension size, converting negative indices and clamping out-of-bounds ranges.
     *
     * @param dimSize dimension size
     * @return resolved slice with non-negative bounds and selected length
     */
    public ResolvedSlice resolve(int dimSize) {
        if (dimSize < 0) {
            throw new IllegalArgumentException("Dimension size cannot be negative: " + dimSize);
        }
        int actualStart = start;
        if (actualStart == UNBOUNDED_START) {
            actualStart = 0;
        } else if (actualStart < 0) {
            actualStart += dimSize;
        }
        actualStart = Math.max(0, Math.min(dimSize, actualStart));

        int actualStop = stop;
        if (actualStop == UNBOUNDED_STOP) {
            actualStop = dimSize;
        } else if (actualStop < 0) {
            actualStop += dimSize;
        }
        actualStop = Math.max(0, Math.min(dimSize, actualStop));

        int length;
        if (actualStop > actualStart) {
            length = (actualStop - actualStart + step - 1) / step;
        } else {
            length = 0;
        }
        return new ResolvedSlice(actualStart, actualStop, step, length);
    }
}
