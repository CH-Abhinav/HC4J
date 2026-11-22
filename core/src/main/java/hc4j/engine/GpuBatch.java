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

package hc4j.engine;

import hc4j.internal.ffi.NativeEngine;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A command-batching scope. Every GPU op issued while a batch is open is recorded into one shared
 * command buffer and submitted once when the outermost scope closes.
 */
public final class GpuBatch implements AutoCloseable {

    private final AtomicBoolean closed = new AtomicBoolean(false);

    private GpuBatch() {}

    /**
     * Opens a new or nested command-batching scope.
     *
     * @return an AutoCloseable batch handle
     */
    public static GpuBatch open() {
        NativeEngine.batchBegin();
        return new GpuBatch();
    }

    /**
     * Opens a new or nested command-batching scope. Alias for {@link #open()}.
     *
     * @return an AutoCloseable batch handle
     */
    public static GpuBatch begin() {
        return open();
    }

    /**
     * Closes this batch scope. Idempotent across multiple invocations.
     */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            NativeEngine.batchEnd();
        }
    }
}
