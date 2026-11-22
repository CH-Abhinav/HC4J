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

package hc4j.internal.ffi;

import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Reference-counted VRAM allocation handle shared among views of the same underlying buffer.
 */
public final class VramHandle {

    private static final class FreeAction implements Runnable {
        private final long vramId;
        private final AtomicBoolean freed;

        FreeAction(long vramId, AtomicBoolean freed) {
            this.vramId = vramId;
            this.freed = freed;
        }

        @Override
        public void run() {
            if (freed.compareAndSet(false, true) && vramId != 0L) {
                try {
                    NativeEngine.freeVram(vramId);
                } catch (Throwable ignored) {
                }
            }
        }
    }

    private final long vramId;
    private final AtomicInteger refCount = new AtomicInteger(1);
    private final AtomicBoolean freed = new AtomicBoolean(false);
    private final Cleaner.Cleanable cleanable;

    /**
     * Creates a new reference-counted VRAM handle wrapping the allocated VRAM ID.
     *
     * @param vramId native VRAM allocation ID
     */
    public VramHandle(long vramId) {
        this.vramId = vramId;
        this.cleanable = NativeCleaner.register(this, new FreeAction(vramId, this.freed));
    }

    /**
     * Atomically increments the reference count while the underlying buffer is still alive.
     *
     * @throws IllegalStateException if the buffer has already been freed
     */
    public void retain() {
        while (true) {
            int current = refCount.get();
            if (current <= 0) {
                throw new IllegalStateException("Underlying VRAM buffer has already been freed");
            }
            if (refCount.compareAndSet(current, current + 1)) {
                return;
            }
        }
    }

    /**
     * Decrements the reference count and cleans up the native allocation when it reaches zero.
     */
    public void release() {
        int remaining = refCount.decrementAndGet();
        if (remaining == 0) {
            cleanable.clean();
        }
    }

    /**
     * Returns the underlying native VRAM buffer ID.
     *
     * @return native VRAM buffer ID
     */
    public long vramId() {
        return vramId;
    }

    /**
     * Returns the current reference count.
     *
     * @return current active reference count
     */
    public int refCount() {
        return refCount.get();
    }
}
