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
import hc4j.nn.Activation;
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
            JAVA_INT.withName("loopback"),
            JAVA_DOUBLE.withName("latency_us"));

    static final StructLayout DEVICE_PROFILE_LAYOUT = MemoryLayout.structLayout(
            JAVA_INT.withName("rank"),
            JAVA_INT.withName("vendor_id"),
            JAVA_INT.withName("device_id"),
            JAVA_INT.withName("device_type"),
            JAVA_LONG.withName("vram_bytes"),
            JAVA_DOUBLE.withName("compute_gflops"),
            JAVA_DOUBLE.withName("vram_bandwidth_gbps"));

    static final StructLayout ISLAND_HEADER_LAYOUT = MemoryLayout.structLayout(
            JAVA_INT.withName("id"),
            JAVA_INT.withName("leader_rank"),
            JAVA_INT.withName("internal_transport"),
            JAVA_INT.withName("member_count"));

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
    private List<DeviceProfile> deviceProfiles;
    private List<Integer> ringOrder;
    private List<Island> islands;

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
                        slot.get(JAVA_INT, 20) != 0,
                        slot.get(JAVA_DOUBLE, 24)));
            }
        }
        topology = List.copyOf(links);
        return topology;
    }

    private static String deviceTypeString(int type) {
        return switch (type) {
            case 1 -> "Integrated";
            case 2 -> "Discrete";
            case 3 -> "Virtual";
            case 4 -> "CPU";
            default -> "Other";
        };
    }

    /**
     * Returns hardware profiles (VRAM capacity and compute throughput) for all ranks.
     *
     * @return the immutable list of device profiles
     * @throws IllegalStateException if the collective has been closed
     */
    public List<DeviceProfile> deviceProfiles() {
        ensureOpen();
        if (deviceProfiles != null) {
            return deviceProfiles;
        }
        List<DeviceProfile> profiles = new ArrayList<>(ranks);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment slot = arena.allocate(DEVICE_PROFILE_LAYOUT);
            for (int i = 0; i < ranks; i++) {
                NativeEngine.collectiveDeviceProfile(handle, i, slot);
                int rank = slot.get(JAVA_INT, 0);
                int vendorId = slot.get(JAVA_INT, 4);
                int deviceId = slot.get(JAVA_INT, 8);
                String typeStr = deviceTypeString(slot.get(JAVA_INT, 12));
                long vram = slot.get(JAVA_LONG, 16);
                double gflops = slot.get(JAVA_DOUBLE, 24);
                double vramBw = slot.get(JAVA_DOUBLE, 32);
                String name = deviceName(i);
                profiles.add(new DeviceProfile(rank, name, typeStr, vendorId, deviceId, vram, gflops, vramBw));
            }
        }
        deviceProfiles = List.copyOf(profiles);
        return deviceProfiles;
    }

    /**
     * Returns the planned optimal ring traversal order visiting every rank.
     *
     * @return immutable list of rank IDs in ring traversal order
     * @throws IllegalStateException if the collective has been closed
     */
    public List<Integer> ringOrder() {
        ensureOpen();
        if (ringOrder != null) {
            return ringOrder;
        }
        List<Integer> order = new ArrayList<>(ranks);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate((long) ranks * 4);
            int count = NativeEngine.collectiveRingOrder(handle, buf, ranks);
            for (int i = 0; i < count; i++) {
                order.add(buf.get(JAVA_INT, (long) i * 4));
            }
        }
        ringOrder = List.copyOf(order);
        return ringOrder;
    }

    /**
     * Returns the detected high-speed clusters (islands) in the collective topology.
     *
     * @return immutable list of detected islands
     * @throws IllegalStateException if the collective has been closed
     */
    public List<Island> islands() {
        ensureOpen();
        if (islands != null) {
            return islands;
        }
        int count = NativeEngine.collectiveIslandCount(handle);
        List<Island> islandList = new ArrayList<>(count);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment headerSlot = arena.allocate(ISLAND_HEADER_LAYOUT);
            MemorySegment ranksSlot = arena.allocate((long) ranks * 4);
            for (int i = 0; i < count; i++) {
                int memberCount = NativeEngine.collectiveIsland(handle, i, headerSlot, ranksSlot, ranks);
                int id = headerSlot.get(JAVA_INT, 0);
                int leader = headerSlot.get(JAVA_INT, 4);
                Transport tr = Transport.of(headerSlot.get(JAVA_INT, 8));
                List<Integer> members = new ArrayList<>(memberCount);
                for (int m = 0; m < memberCount; m++) {
                    members.add(ranksSlot.get(JAVA_INT, (long) m * 4));
                }
                islandList.add(new Island(id, leader, members, tr));
            }
        }
        islands = List.copyOf(islandList);
        return islands;
    }

    /**
     * Generates a sharded workload execution plan using compute-proportional water filling,
     * VRAM capacity capping, and automatic straggler pruning.
     *
     * @param totalUnits total workload units to partition
     * @param flopsPerUnit floating-point operations required per unit
     * @param bytesPerUnit memory footprint in bytes required per unit
     * @param syncBytes AllReduce synchronization payload in bytes
     * @return the optimal workload plan
     * @throws IllegalStateException if the collective has been closed
     */
    public WorkloadPlan planWork(long totalUnits, double flopsPerUnit, long bytesPerUnit, long syncBytes) {
        ensureOpen();
        return WorkloadPlanner.plan(deviceProfiles(), topology(), totalUnits, flopsPerUnit, bytesPerUnit, syncBytes);
    }

    /**
     * Generates a pure compute-proportional workload plan (zero memory limits and zero synchronization).
     *
     * @param totalUnits total workload units to partition
     * @return the workload plan
     * @throws IllegalStateException if the collective has been closed
     */
    public WorkloadPlan planWork(long totalUnits) {
        return planWork(totalUnits, 1e9, 0L, 0L);
    }

    /**
     * Generates an exact zero-parameter optimized workload plan according to the given workload specification.
     *
     * @param spec detailed workload specification (compute, memory, communication, spilling, host CPU)
     * @return the globally optimal workload plan
     * @throws NullPointerException if {@code spec} is null
     * @throws IllegalStateException if the collective has been closed
     */
    public WorkloadPlan planWork(WorkloadSpec spec) {
        ensureOpen();
        Objects.requireNonNull(spec, "spec must not be null");
        return WorkloadPlanner.plan(deviceProfiles(), topology(), spec);
    }

    /**
     * Generates a multi-operator role assignment plan executing Tensor Parallel shards on primary
     * devices and reclaiming excluded straggler GPUs as lossless Split-KV attention workers.
     *
     * @param tpSpec specification for the Tensor Parallel operations
     * @param kvSpec specification for the Split-KV decode attention
     * @return the combined multi-operator role plan
     * @throws NullPointerException if either specification is null
     * @throws IllegalStateException if the collective has been closed
     */
    public RolePlan planRoles(WorkloadSpec tpSpec, SplitKvSpec kvSpec) {
        ensureOpen();
        Objects.requireNonNull(tpSpec, "tpSpec must not be null");
        Objects.requireNonNull(kvSpec, "kvSpec must not be null");
        return WorkloadPlanner.planRoles(deviceProfiles(), topology(), tpSpec, kvSpec);
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

    /**
     * Performs an active-subset ring AllReduce sum strictly across the provided collective buffers.
     * <p>
     * Participating buffers may sit on any non-empty subset of ranks. If exactly one buffer is passed,
     * this operation is a zero-cost no-op. Pruned straggler ranks not present in {@code activeBuffers}
     * are never locked, synchronized, or accessed.
     *
     * @param activeBuffers non-empty array of distinct active buffers belonging to this collective
     * @throws NullPointerException if {@code activeBuffers} or any buffer is null
     * @throws IllegalArgumentException if {@code activeBuffers} is empty, or if buffers reside on duplicate ranks
     *         or have differing element counts
     * @throws IllegalStateException if the collective is closed
     */
    public void allReduceSumSubset(CollectiveBuffer... activeBuffers) {
        Objects.requireNonNull(activeBuffers, "activeBuffers must not be null");
        if (activeBuffers.length == 0) {
            throw new IllegalArgumentException("activeBuffers must not be empty");
        }
        allReduceSumSubset(List.of(activeBuffers));
    }

    /**
     * Performs an active-subset ring AllReduce sum strictly across the provided collective buffers.
     * <p>
     * Participating buffers may sit on any non-empty subset of ranks. If exactly one buffer is passed,
     * this operation is a zero-cost no-op. Pruned straggler ranks not present in {@code activeBuffers}
     * are never locked, synchronized, or accessed.
     *
     * @param activeBuffers non-empty list of distinct active buffers belonging to this collective
     * @throws NullPointerException if {@code activeBuffers} or any buffer is null
     * @throws IllegalArgumentException if {@code activeBuffers} is empty, or if buffers reside on duplicate ranks
     *         or have differing element counts
     * @throws IllegalStateException if the collective is closed
     */
    public void allReduceSumSubset(List<CollectiveBuffer> activeBuffers) {
        ensureOpen();
        Objects.requireNonNull(activeBuffers, "activeBuffers must not be null");
        if (activeBuffers.isEmpty()) {
            throw new IllegalArgumentException("activeBuffers must not be empty");
        }
        int count = activeBuffers.size();
        long[] ids = new long[count];
        long expectedElements = -1;
        boolean[] seenRanks = new boolean[ranks];

        for (int i = 0; i < count; i++) {
            CollectiveBuffer buf = Objects.requireNonNull(activeBuffers.get(i), "buffer at index " + i + " must not be null");
            buf.ensureOpen();
            if (buf.collective() != this) {
                throw new IllegalArgumentException("buffer at index " + i + " belongs to a different collective");
            }
            int r = buf.rank();
            if (r < 0 || r >= ranks) {
                throw new IllegalArgumentException("buffer rank " + r + " is out of bounds");
            }
            if (seenRanks[r]) {
                throw new IllegalArgumentException("duplicate rank " + r + " in activeBuffers");
            }
            seenRanks[r] = true;
            if (i == 0) {
                expectedElements = buf.elements();
            } else if (buf.elements() != expectedElements) {
                throw new IllegalArgumentException("mismatched element counts: buffer 0 has " + expectedElements
                        + " but buffer " + i + " has " + buf.elements());
            }
            ids[i] = buf.handleUnsafe();
        }

        if (count == 1) {
            return;
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment list = arena.allocateFrom(JAVA_LONG, ids);
            NativeEngine.allReduceSumSubset(handle, list, ids.length);
        }
    }

    /**
     * Executes a fused matrix multiplication with optional row-vector bias and activation on a designated rank:
     * {@code C[M, N] = act(A[M, K] * B[K, N] + bias[N])}.
     *
     * @param rank target rank index (0 .. ranks - 1)
     * @param a input matrix A with shape [M, K] row-major resident on {@code rank}
     * @param b input matrix B with shape [K, N] row-major resident on {@code rank}
     * @param bias optional bias vector with length N resident on {@code rank} (may be null if {@code !applyBias})
     * @param c output matrix C with shape [M, N] row-major resident on {@code rank}
     * @param m number of rows in A and C
     * @param n number of columns in B and C, and length of bias
     * @param k number of columns in A and rows in B
     * @param applyBias whether to add {@code bias}
     * @param activation activation function to apply (null for none, or GELU, SILU, RELU)
     * @throws NullPointerException if {@code a}, {@code b}, or {@code c} is null, or if {@code applyBias} is true and {@code bias} is null
     * @throws IllegalArgumentException if {@code rank} is invalid, buffer ranks don't match, or buffer sizes are insufficient
     * @throws UnsupportedOperationException if {@code activation} is not supported by the fused kernel
     * @throws IllegalStateException if the collective is closed
     */
    public void gemmBiasAct(int rank, CollectiveBuffer a, CollectiveBuffer b, CollectiveBuffer bias, CollectiveBuffer c,
                            int m, int n, int k, boolean applyBias, Activation activation) {
        ensureOpen();
        checkRank(rank);
        Objects.requireNonNull(a, "buffer a must not be null");
        Objects.requireNonNull(b, "buffer b must not be null");
        Objects.requireNonNull(c, "buffer c must not be null");
        a.ensureOpen();
        b.ensureOpen();
        c.ensureOpen();
        if (a.collective() != this || b.collective() != this || c.collective() != this) {
            throw new IllegalArgumentException("buffers must belong to this collective");
        }
        if (a.rank() != rank || b.rank() != rank || c.rank() != rank) {
            throw new IllegalArgumentException("all buffers must reside on target rank " + rank);
        }
        if (m <= 0 || n <= 0 || k <= 0) {
            throw new IllegalArgumentException("dimensions m, n, k must be positive, got m=" + m + ", n=" + n + ", k=" + k);
        }
        if (a.elements() < (long) m * k) {
            throw new IllegalArgumentException("buffer a has insufficient elements: " + a.elements() + " < " + ((long) m * k));
        }
        if (b.elements() < (long) k * n) {
            throw new IllegalArgumentException("buffer b has insufficient elements: " + b.elements() + " < " + ((long) k * n));
        }
        if (c.elements() < (long) m * n) {
            throw new IllegalArgumentException("buffer c has insufficient elements: " + c.elements() + " < " + ((long) m * n));
        }
        long biasHandle = 0L;
        if (applyBias) {
            Objects.requireNonNull(bias, "bias buffer must not be null when applyBias is true");
            bias.ensureOpen();
            if (bias.collective() != this) {
                throw new IllegalArgumentException("bias buffer must belong to this collective");
            }
            if (bias.rank() != rank) {
                throw new IllegalArgumentException("bias buffer must reside on target rank " + rank);
            }
            if (bias.elements() < n) {
                throw new IllegalArgumentException("bias buffer has insufficient elements: " + bias.elements() + " < " + n);
            }
            biasHandle = bias.handleUnsafe();
        } else if (bias != null) {
            biasHandle = bias.handleUnsafe();
        }

        int actCode;
        if (activation == null) {
            actCode = 0;
        } else {
            actCode = switch (activation) {
                case GELU -> 1;
                case SILU -> 2;
                case RELU -> 3;
                default -> throw new UnsupportedOperationException("Unsupported collective GEMM activation: " + activation);
            };
        }
        int flags = (applyBias ? 1 : 0) | (actCode << 1);

        NativeEngine.gemmBiasAct(handle, rank, a.handleUnsafe(), b.handleUnsafe(), biasHandle, c.handleUnsafe(),
                m, n, k, flags);
    }

    /**
     * Executes a single-query local FlashDecoding-style Decode Attention pass on a designated rank:
     * computes unnormalized or normalized partial output and exact online-softmax reduction statistics
     * for a local KV cache shard.
     *
     * @param rank target rank index (0 .. ranks - 1)
     * @param q query tensor [numHeads, headDim] resident on {@code rank}
     * @param kCache local key cache shard [numKvHeads, localTokens, headDim] resident on {@code rank}
     * @param vCache local value cache shard [numKvHeads, localTokens, headDim] resident on {@code rank}
     * @param outO output attention tensor [numHeads, headDim] resident on {@code rank}
     * @param outStats output online-softmax statistics tensor [numHeads * 2] resident on {@code rank}
     * @param numHeads number of query attention heads
     * @param numKvHeads number of key/value attention heads (where {@code numHeads % numKvHeads == 0})
     * @param headDim dimension of each head
     * @param localTokens number of tokens in the local KV cache shard
     * @param scale query scaling factor (typically {@code 1.0f / Math.sqrt(headDim)})
     * @throws NullPointerException if any buffer is null
     * @throws IllegalArgumentException if dimensions are non-positive, head ratios are invalid, or buffers are insufficient
     * @throws IllegalStateException if the collective is closed
     */
    public void splitKvAttention(int rank, CollectiveBuffer q, CollectiveBuffer kCache, CollectiveBuffer vCache,
                                 CollectiveBuffer outO, CollectiveBuffer outStats,
                                 int numHeads, int numKvHeads, int headDim, int localTokens, float scale) {
        ensureOpen();
        checkRank(rank);
        Objects.requireNonNull(q, "q buffer must not be null");
        Objects.requireNonNull(kCache, "kCache buffer must not be null");
        Objects.requireNonNull(vCache, "vCache buffer must not be null");
        Objects.requireNonNull(outO, "outO buffer must not be null");
        Objects.requireNonNull(outStats, "outStats buffer must not be null");

        q.ensureOpen();
        kCache.ensureOpen();
        vCache.ensureOpen();
        outO.ensureOpen();
        outStats.ensureOpen();

        if (q.collective() != this || kCache.collective() != this || vCache.collective() != this
                || outO.collective() != this || outStats.collective() != this) {
            throw new IllegalArgumentException("all buffers must belong to this collective");
        }
        if (q.rank() != rank || kCache.rank() != rank || vCache.rank() != rank
                || outO.rank() != rank || outStats.rank() != rank) {
            throw new IllegalArgumentException("all buffers must reside on target rank " + rank);
        }
        if (numHeads <= 0 || numKvHeads <= 0 || headDim <= 0 || localTokens < 0) {
            throw new IllegalArgumentException("invalid attention dimensions: numHeads=" + numHeads
                    + ", numKvHeads=" + numKvHeads + ", headDim=" + headDim + ", localTokens=" + localTokens);
        }
        if (numHeads % numKvHeads != 0) {
            throw new IllegalArgumentException("numHeads (" + numHeads + ") must be divisible by numKvHeads (" + numKvHeads + ")");
        }

        if (q.elements() < (long) numHeads * headDim) {
            throw new IllegalArgumentException("buffer q too small: " + q.elements() + " < " + ((long) numHeads * headDim));
        }
        long kvElements = (long) numKvHeads * localTokens * headDim;
        if (kCache.elements() < kvElements) {
            throw new IllegalArgumentException("buffer kCache too small: " + kCache.elements() + " < " + kvElements);
        }
        if (vCache.elements() < kvElements) {
            throw new IllegalArgumentException("buffer vCache too small: " + vCache.elements() + " < " + kvElements);
        }
        if (outO.elements() < (long) numHeads * headDim) {
            throw new IllegalArgumentException("buffer outO too small: " + outO.elements() + " < " + ((long) numHeads * headDim));
        }
        if (outStats.elements() < (long) numHeads * 2) {
            throw new IllegalArgumentException("buffer outStats too small: " + outStats.elements() + " < " + ((long) numHeads * 2));
        }

        int scaleBits = Float.floatToRawIntBits(scale);
        NativeEngine.splitKvAttn(handle, rank, q.handleUnsafe(), kCache.handleUnsafe(), vCache.handleUnsafe(),
                outO.handleUnsafe(), outStats.handleUnsafe(), numHeads, numKvHeads, headDim, localTokens, scaleBits);
    }

    /**
     * Executes elementwise float32 multiplication {@code out[i] = a[i] * b[i]} on a specific rank.
     *
     * @param rank target rank executing the kernel
     * @param a first input buffer on {@code rank}
     * @param b second input buffer on {@code rank}
     * @param out output buffer on {@code rank}
     * @param count number of elements to multiply
     */
    public void elemMul(int rank, CollectiveBuffer a, CollectiveBuffer b, CollectiveBuffer out, int count) {
        ensureOpen();
        checkRank(rank);
        Objects.requireNonNull(a, "a buffer must not be null");
        Objects.requireNonNull(b, "b buffer must not be null");
        Objects.requireNonNull(out, "out buffer must not be null");
        a.ensureOpen();
        b.ensureOpen();
        out.ensureOpen();
        if (a.collective() != this || b.collective() != this || out.collective() != this) {
            throw new IllegalArgumentException("all buffers must belong to this collective");
        }
        if (a.rank() != rank || b.rank() != rank || out.rank() != rank) {
            throw new IllegalArgumentException("all buffers must reside on target rank " + rank);
        }
        if (count < 0) {
            throw new IllegalArgumentException("count must be non-negative: " + count);
        }
        if (count == 0) {
            return;
        }
        if (a.elements() < count || b.elements() < count || out.elements() < count) {
            throw new IllegalArgumentException("elemMul buffer capacity smaller than count=" + count);
        }
        NativeEngine.elemMul(handle, rank, a.handleUnsafe(), b.handleUnsafe(), out.handleUnsafe(), count);
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
