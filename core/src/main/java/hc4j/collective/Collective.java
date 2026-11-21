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

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import hc4j.internal.ffi.NativeCleaner;
import hc4j.internal.ffi.NativeEngine;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.ref.Cleaner;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A ring of GPU ranks with collective communication across them: topology discovery and
 * ring AllReduce.
 */
public final class Collective implements AutoCloseable {

    static final StructLayout LINK_LAYOUT = MemoryLayout.structLayout(
            JAVA_INT.withName("src"),
            JAVA_INT.withName("dst"),
            JAVA_DOUBLE.withName("gbps"),
            JAVA_INT.withName("transport"),
            JAVA_INT.withName("loopback"));

    private static final int NAME_CAPACITY = 256;

    private static final class CleanupAction implements Runnable {
        private final long handle;
        private final AtomicBoolean closed;

        CleanupAction(long handle, AtomicBoolean closed) {
            this.handle = handle;
            this.closed = closed;
        }

        @Override
        public void run() {
            if (closed.compareAndSet(false, true)) {
                try {
                    NativeEngine.destroyCollective(handle);
                } catch (Throwable ignored) {
                }
            }
        }
    }

    private final long handle;
    private final int ranks;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final Cleaner.Cleanable cleanable;
    private List<Link> topology;

    private Collective(long handle, int ranks) {
        this.handle = handle;
        this.ranks = ranks;
        this.cleanable = NativeCleaner.register(this, new CleanupAction(handle, this.closed));
    }

    public static Collective open(int ranks) {
        if (ranks < 2) {
            throw new IllegalArgumentException("a collective needs at least 2 ranks, got " + ranks);
        }
        long handle = NativeEngine.createCollective(ranks);
        int actual = NativeEngine.collectiveRanks(handle);
        return new Collective(handle, actual);
    }

    public int ranks() {
        return ranks;
    }

    public boolean isClosed() {
        return closed.get();
    }

    public void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("Collective has already been closed");
        }
    }

    long handle() {
        return handle;
    }

    AtomicBoolean closedFlag() {
        return closed;
    }

    public String deviceName(int rank) {
        ensureOpen();
        checkRank(rank);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(NAME_CAPACITY);
            int len = NativeEngine.collectiveDeviceName(handle, rank, buf, NAME_CAPACITY);
            byte[] bytes = buf.asSlice(0, len).toArray(ValueLayout.JAVA_BYTE);
            return new String(bytes, StandardCharsets.UTF_8);
        }
    }

    /**
     * Returns the name of the active collective communication backend
     * ("vulkan-p2p" or "wgpu-staged").
     *
     * @return the backend name
     * @throws IllegalStateException if the collective has been closed
     */
    public String backendName() {
        ensureOpen();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(NAME_CAPACITY);
            int len = NativeEngine.collectiveBackendName(handle, buf, NAME_CAPACITY);
            byte[] bytes = buf.asSlice(0, len).toArray(ValueLayout.JAVA_BYTE);
            return new String(bytes, StandardCharsets.UTF_8);
        }
    }

    public List<Link> topology() {
        ensureOpen();
        if (topology != null) {
            return topology;
        }
        int count = NativeEngine.collectiveLinkCount(handle);
        List<Link> links = new ArrayList<>(count);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment slot = arena.allocate(LINK_LAYOUT);
            for (int i = 0; i < count; i++) {
                NativeEngine.collectiveLink(handle, i, slot);
                links.add(new Link(
                        slot.get(JAVA_INT, 0),
                        slot.get(JAVA_INT, 4),
                        slot.get(JAVA_DOUBLE, 8),
                        Transport.of(slot.get(JAVA_INT, 16)),
                        slot.get(JAVA_INT, 20) != 0));
            }
        }
        topology = List.copyOf(links);
        return topology;
    }

    public CollectiveBuffer allocate(int rank, long elements) {
        ensureOpen();
        checkRank(rank);
        if (elements <= 0) {
            throw new IllegalArgumentException("a collective buffer needs at least one element");
        }
        long bufHandle = NativeEngine.allocCollectiveBuffer(handle, rank, elements);
        return new CollectiveBuffer(this, bufHandle, rank, elements);
    }

    public void allReduceSum(CollectiveBuffer... perRank) {
        ensureOpen();
        Objects.requireNonNull(perRank, "perRank");
        if (perRank.length != ranks) {
            throw new IllegalArgumentException(
                    "allReduceSum needs one buffer per rank: expected " + ranks + ", got " + perRank.length);
        }
        long[] ids = new long[perRank.length];
        for (int i = 0; i < perRank.length; i++) {
            CollectiveBuffer buf = Objects.requireNonNull(perRank[i], "buffer " + i);
            if (buf.collective() != this) {
                throw new IllegalArgumentException("buffer " + i + " belongs to a different collective");
            }
            if (buf.rank() != i) {
                throw new IllegalArgumentException(
                        "buffer " + i + " is on rank " + buf.rank() + "; pass them in rank order");
            }
            ids[i] = buf.handleUnsafe();
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment list = arena.allocateFrom(JAVA_LONG, ids);
            NativeEngine.allReduceSum(handle, list, ids.length);
        }
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    void writeBuffer(long buffer, float[] data) {
        ensureOpen();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocateFrom(ValueLayout.JAVA_FLOAT, data);
            NativeEngine.writeCollectiveBuffer(handle, buffer, host, data.length);
        }
    }

    float[] readBuffer(long buffer, long elements) {
        ensureOpen();
        float[] out = new float[(int) elements];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(ValueLayout.JAVA_FLOAT, elements);
            NativeEngine.readCollectiveBuffer(handle, buffer, host, elements);
            MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0, out, 0, out.length);
        }
        return out;
    }

    void freeBuffer(long buffer) {
        if (!closed.get()) {
            NativeEngine.freeCollectiveBuffer(handle, buffer);
        }
    }

    private void checkRank(int rank) {
        if (rank < 0 || rank >= ranks) {
            throw new IllegalArgumentException("rank " + rank + " is outside 0.." + (ranks - 1));
        }
    }
}
