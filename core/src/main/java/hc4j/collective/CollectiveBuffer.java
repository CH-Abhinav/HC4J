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

package hc4j.collective;

import hc4j.internal.ffi.NativeCleaner;
import hc4j.internal.ffi.NativeEngine;
import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * An f32 buffer resident on one rank of a {@link Collective}.
 */
public final class CollectiveBuffer implements AutoCloseable {

    private static final class CleanupAction implements Runnable {
        private final long collectiveHandle;
        private final long bufferHandle;
        private final AtomicBoolean collectiveClosed;
        private final AtomicBoolean closed;

        CleanupAction(long collectiveHandle, long bufferHandle, AtomicBoolean collectiveClosed, AtomicBoolean closed) {
            this.collectiveHandle = collectiveHandle;
            this.bufferHandle = bufferHandle;
            this.collectiveClosed = collectiveClosed;
            this.closed = closed;
        }

        @Override
        public void run() {
            if (closed.compareAndSet(false, true)) {
                if (!collectiveClosed.get()) {
                    try {
                        NativeEngine.freeCollectiveBuffer(collectiveHandle, bufferHandle);
                    } catch (Throwable ignored) {
                    }
                }
            }
        }
    }

    private final Collective collective;
    private final long handle;
    private final int rank;
    private final long elements;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final Cleaner.Cleanable cleanable;

    CollectiveBuffer(Collective collective, long handle, int rank, long elements) {
        this.collective = collective;
        this.handle = handle;
        this.rank = rank;
        this.elements = elements;
        this.cleanable = NativeCleaner.register(this,
                new CleanupAction(collective.handle(), handle, collective.closedFlag(), this.closed));
    }

    public int rank() {
        return rank;
    }

    public long elements() {
        return elements;
    }

    public boolean isClosed() {
        return closed.get();
    }

    public void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("CollectiveBuffer has already been closed");
        }
        collective.ensureOpen();
    }

    public void write(float[] data) {
        ensureOpen();
        if (data.length != elements) {
            throw new IllegalArgumentException(
                    "write expects " + elements + " elements, got " + data.length);
        }
        collective.writeBuffer(handle, data);
    }

    public float[] read() {
        ensureOpen();
        if (elements > Integer.MAX_VALUE) {
            throw new IllegalStateException("buffer is larger than a Java array can hold");
        }
        return collective.readBuffer(handle, elements);
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    @Override
    public String toString() {
        return "CollectiveBuffer[rank=" + rank + ", elements=" + elements + "]";
    }

    Collective collective() {
        return collective;
    }

    long handleUnsafe() {
        ensureOpen();
        return handle;
    }
}
