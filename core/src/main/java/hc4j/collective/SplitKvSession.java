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

import hc4j.DType;
import hc4j.Tensor;
import hc4j.internal.collective.CpuFallbackKernels;
import hc4j.internal.collective.NativeOptimizerBridge;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Persistent multi-device KV cache session executing lossless Split-KV Decode Attention.
 * <p>
 * Key and Value cache shards are uploaded once across assigned GPU workers (Tensor Parallel
 * and reclaimed straggler GPUs) and retained in system RAM for CPU-in-place workers.
 * Successive decode queries are evaluated with zero KV re-upload overhead using
 * online-softmax partial reduction and exact cross-worker merging.
 */
public final class SplitKvSession implements AutoCloseable {

    private static final class GpuWorker {
        final int rank;
        final int tokenStart;
        final int localTokens;
        final CollectiveBuffer kBuf;
        final CollectiveBuffer vBuf;
        final CollectiveBuffer qBuf;
        final CollectiveBuffer outOBuf;
        final CollectiveBuffer outStatsBuf;

        GpuWorker(int rank, int tokenStart, int localTokens, CollectiveBuffer kBuf, CollectiveBuffer vBuf,
                  CollectiveBuffer qBuf, CollectiveBuffer outOBuf, CollectiveBuffer outStatsBuf) {
            this.rank = rank;
            this.tokenStart = tokenStart;
            this.localTokens = localTokens;
            this.kBuf = kBuf;
            this.vBuf = vBuf;
            this.qBuf = qBuf;
            this.outOBuf = outOBuf;
            this.outStatsBuf = outStatsBuf;
        }

        void close() {
            try { kBuf.close(); } catch (Throwable ignored) {}
            try { vBuf.close(); } catch (Throwable ignored) {}
            try { qBuf.close(); } catch (Throwable ignored) {}
            try { outOBuf.close(); } catch (Throwable ignored) {}
            try { outStatsBuf.close(); } catch (Throwable ignored) {}
        }
    }

    private static final class CpuWorker {
        final int tokenStart;
        final int localTokens;
        final MemorySegment kShard;
        final MemorySegment vShard;

        CpuWorker(int tokenStart, int localTokens, MemorySegment kShard, MemorySegment vShard) {
            this.tokenStart = tokenStart;
            this.localTokens = localTokens;
            this.kShard = kShard;
            this.vShard = vShard;
        }
    }

    private record WorkerResult(int rank, int localTokens, float[] o, float[] stats) {}

    private final Collective collective;
    private final RolePlan rolePlan;
    private final int numHeads;
    private final int numKvHeads;
    private final int headDim;
    private final List<GpuWorker> gpuWorkers;
    private final CpuWorker cpuWorker;
    private final Arena cpuArena;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private SplitKvSession(Collective collective, RolePlan rolePlan, int numHeads, int numKvHeads, int headDim,
                           List<GpuWorker> gpuWorkers, CpuWorker cpuWorker, Arena cpuArena) {
        this.collective = collective;
        this.rolePlan = rolePlan;
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.gpuWorkers = gpuWorkers;
        this.cpuWorker = cpuWorker;
        this.cpuArena = cpuArena;
    }

    /**
     * Opens a persistent Split-KV session, partitioning and uploading the KV cache across
     * all assigned workers designated in the role plan.
     *
     * @param collective the active collective instance
     * @param rolePlan multi-operator role assignment plan
     * @param numHeads number of query attention heads
     * @param numKvHeads number of key/value attention heads (must divide {@code numHeads})
     * @param headDim dimension of each attention head
     * @param fullKCache full key cache tensor with shape [numKvHeads, totalTokens, headDim]
     * @param fullVCache full value cache tensor with shape [numKvHeads, totalTokens, headDim]
     * @return a new open SplitKvSession
     * @throws NullPointerException if any argument is null
     * @throws IllegalArgumentException if dimensions, shapes, or token counts do not match
     * @throws IllegalStateException if {@code collective} is closed
     */
    public static SplitKvSession open(Collective collective, RolePlan rolePlan, int numHeads, int numKvHeads,
                                      int headDim, Tensor fullKCache, Tensor fullVCache) {
        Objects.requireNonNull(collective, "collective must not be null");
        Objects.requireNonNull(rolePlan, "rolePlan must not be null");
        Objects.requireNonNull(fullKCache, "fullKCache must not be null");
        Objects.requireNonNull(fullVCache, "fullVCache must not be null");
        collective.ensureOpen();

        if (numHeads <= 0 || numKvHeads <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("dimensions must be positive: numHeads=" + numHeads
                    + ", numKvHeads=" + numKvHeads + ", headDim=" + headDim);
        }
        if (numHeads % numKvHeads != 0) {
            throw new IllegalArgumentException("numHeads (" + numHeads + ") must be divisible by numKvHeads (" + numKvHeads + ")");
        }

        int[] kShape = fullKCache.getShape();
        int[] vShape = fullVCache.getShape();
        if (kShape.length != 3 || kShape[0] != numKvHeads || kShape[2] != headDim) {
            throw new IllegalArgumentException("fullKCache expected shape [" + numKvHeads + ", "
                    + rolePlan.totalKvTokens() + ", " + headDim + "], got " + Arrays.toString(kShape));
        }
        if (vShape.length != 3 || vShape[0] != numKvHeads || vShape[2] != headDim) {
            throw new IllegalArgumentException("fullVCache expected shape [" + numKvHeads + ", "
                    + rolePlan.totalKvTokens() + ", " + headDim + "], got " + Arrays.toString(vShape));
        }
        int totalTokens = kShape[1];
        if (vShape[1] != totalTokens) {
            throw new IllegalArgumentException("fullKCache tokens (" + totalTokens + ") != fullVCache tokens (" + vShape[1] + ")");
        }
        if (totalTokens != rolePlan.totalKvTokens()) {
            throw new IllegalArgumentException("KV cache tokens (" + totalTokens + ") != rolePlan.totalKvTokens ("
                    + rolePlan.totalKvTokens() + ")");
        }

        List<GpuWorker> gpus = new ArrayList<>();
        CpuWorker cpu = null;
        Arena cpuSharedArena = Arena.ofShared();

        try (Arena stageArena = Arena.ofConfined()) {
            long totalBytes = (long) numKvHeads * totalTokens * headDim * 4L;
            MemorySegment fullKSeg = stageArena.allocate(totalBytes, 16);
            MemorySegment fullVSeg = stageArena.allocate(totalBytes, 16);
            fullKCache.copyTo(fullKSeg);
            fullVCache.copyTo(fullVSeg);

            int currentToken = 0;
            for (SplitKvAssignment assignment : rolePlan.splitKvAssignments()) {
                int tokens = (int) assignment.kvTokens();
                if (tokens <= 0) {
                    continue;
                }
                int rank = assignment.rank();
                int tokenStart = currentToken;
                currentToken += tokens;

                if (rank == ShardAllocation.HOST_CPU_RANK || assignment.role() == DeviceRole.CPU_IN_PLACE) {
                    long shardBytes = (long) numKvHeads * tokens * headDim * 4L;
                    MemorySegment kShard = cpuSharedArena.allocate(shardBytes, 16);
                    MemorySegment vShard = cpuSharedArena.allocate(shardBytes, 16);
                    for (int kvH = 0; kvH < numKvHeads; kvH++) {
                        long srcOff = ((long) kvH * totalTokens + tokenStart) * headDim * 4L;
                        long dstOff = (long) kvH * tokens * headDim * 4L;
                        long copyBytes = (long) tokens * headDim * 4L;
                        MemorySegment.copy(fullKSeg, srcOff, kShard, dstOff, copyBytes);
                        MemorySegment.copy(fullVSeg, srcOff, vShard, dstOff, copyBytes);
                    }
                    cpu = new CpuWorker(tokenStart, tokens, kShard, vShard);
                } else {
                    long shardBytes = (long) numKvHeads * tokens * headDim * 4L;
                    MemorySegment kShard = stageArena.allocate(shardBytes, 16);
                    MemorySegment vShard = stageArena.allocate(shardBytes, 16);
                    for (int kvH = 0; kvH < numKvHeads; kvH++) {
                        long srcOff = ((long) kvH * totalTokens + tokenStart) * headDim * 4L;
                        long dstOff = (long) kvH * tokens * headDim * 4L;
                        long copyBytes = (long) tokens * headDim * 4L;
                        MemorySegment.copy(fullKSeg, srcOff, kShard, dstOff, copyBytes);
                        MemorySegment.copy(fullVSeg, srcOff, vShard, dstOff, copyBytes);
                    }

                    CollectiveBuffer kBuf = collective.allocate(rank, (long) numKvHeads * tokens * headDim);
                    CollectiveBuffer vBuf = collective.allocate(rank, (long) numKvHeads * tokens * headDim);
                    CollectiveBuffer qBuf = collective.allocate(rank, (long) numHeads * headDim);
                    CollectiveBuffer outOBuf = collective.allocate(rank, (long) numHeads * headDim);
                    CollectiveBuffer outStatsBuf = collective.allocate(rank, (long) numHeads * 2);

                    kBuf.write(kShard);
                    vBuf.write(vShard);

                    gpus.add(new GpuWorker(rank, tokenStart, tokens, kBuf, vBuf, qBuf, outOBuf, outStatsBuf));
                }
            }
        } catch (Throwable t) {
            for (GpuWorker w : gpus) {
                w.close();
            }
            try { cpuSharedArena.close(); } catch (Throwable ignored) {}
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("Failed to initialize SplitKvSession", t);
        }

        return new SplitKvSession(collective, rolePlan, numHeads, numKvHeads, headDim, gpus, cpu, cpuSharedArena);
    }

    /**
     * Executes Decode Attention for a single query token across the persistent KV cache shards.
     *
     * @param q query tensor with shape [numHeads, headDim]
     * @param scale scaling factor applied to query-key dot products (typically {@code 1.0f / Math.sqrt(headDim)})
     * @return distributed result containing the output attention tensor [numHeads, headDim] and execution telemetry
     * @throws NullPointerException if {@code q} is null
     * @throws IllegalArgumentException if {@code q} shape does not match head dimensions
     * @throws IllegalStateException if this session or the collective is closed
     */
    public DistributedResult<Tensor> attend(Tensor q, float scale) {
        Objects.requireNonNull(q, "q must not be null");
        ensureOpen();

        long startWall = System.nanoTime();

        int[] qShape = q.getShape();
        if (qShape.length != 2 || qShape[0] != numHeads || qShape[1] != headDim) {
            long totalQElements = 1;
            for (int s : qShape) totalQElements *= s;
            if (totalQElements != (long) numHeads * headDim) {
                throw new IllegalArgumentException("q tensor shape " + Arrays.toString(qShape)
                        + " cannot be reshaped to [" + numHeads + ", " + headDim + "]");
            }
        }

        List<WorkerResult> results = new ArrayList<>(gpuWorkers.size() + (cpuWorker != null ? 1 : 0));
        List<Integer> executedRanks = new ArrayList<>();
        List<Integer> prunedRanks = new ArrayList<>();
        long cpuUnits = (cpuWorker != null) ? cpuWorker.localTokens : 0L;

        long computeStart = System.nanoTime();

        try (Arena qArena = Arena.ofShared();
             ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {

            long qBytes = (long) numHeads * headDim * 4L;
            MemorySegment qSeg = qArena.allocate(qBytes, 16);
            q.copyTo(qSeg);

            List<CompletableFuture<WorkerResult>> futures = new ArrayList<>();

            for (GpuWorker gw : gpuWorkers) {
                executedRanks.add(gw.rank);
                futures.add(CompletableFuture.supplyAsync(() -> {
                    gw.qBuf.write(qSeg);
                    collective.splitKvAttention(gw.rank, gw.qBuf, gw.kBuf, gw.vBuf, gw.outOBuf, gw.outStatsBuf,
                            numHeads, numKvHeads, headDim, gw.localTokens, scale);
                    float[] o = gw.outOBuf.read();
                    float[] stats = gw.outStatsBuf.read();
                    return new WorkerResult(gw.rank, gw.localTokens, o, stats);
                }, executor));
            }

            if (cpuWorker != null && cpuWorker.localTokens > 0) {
                futures.add(CompletableFuture.supplyAsync(() ->
                        computeCpuAttention(qSeg, cpuWorker, numHeads, numKvHeads, headDim, scale), executor));
            }

            for (CompletableFuture<WorkerResult> f : futures) {
                results.add(f.join());
            }
        }

        long computeEnd = System.nanoTime();
        long syncStart = System.nanoTime();

        // Exact Lossless Online-Softmax Merge across all active worker outputs
        int activeWorkers = 0;
        for (WorkerResult r : results) {
            if (r.localTokens > 0) {
                activeWorkers++;
            }
        }
        float[][] partialsO = new float[activeWorkers][];
        float[][] partialsStats = new float[activeWorkers][];
        int wIdx = 0;
        for (WorkerResult r : results) {
            if (r.localTokens > 0) {
                partialsO[wIdx] = r.o;
                partialsStats[wIdx] = r.stats;
                wIdx++;
            }
        }
        MemorySegment outSeg = Arena.ofAuto().allocate((long) numHeads * headDim * 4L, 16);
        NativeOptimizerBridge.mergeSplitKvF32(numHeads, headDim, partialsO, partialsStats, outSeg);

        long endWall = System.nanoTime();

        for (SplitKvAssignment a : rolePlan.splitKvAssignments()) {
            if (a.rank() >= 0 && a.kvTokens() == 0) {
                prunedRanks.add(a.rank());
            }
        }

        double wallMs = (endWall - startWall) / 1e6;
        double computeMs = (computeEnd - computeStart) / 1e6;
        double syncMs = (endWall - syncStart) / 1e6;

        ExecutionReport report = new ExecutionReport(
                "splitKvAttention",
                rolePlan.tpPlan(),
                rolePlan,
                wallMs,
                computeMs,
                syncMs,
                executedRanks,
                prunedRanks,
                cpuUnits
        );

        Tensor resultTensor = Tensor.fromSegment(outSeg, DType.f32, numHeads, headDim);
        return new DistributedResult<>(resultTensor, report);
    }

    private static WorkerResult computeCpuAttention(MemorySegment qSeg, CpuWorker cpu, int numHeads,
                                                    int numKvHeads, int headDim, float scale) {
        int tokens = cpu.localTokens;
        float[] outO = new float[numHeads * headDim];
        float[] outStats = new float[numHeads * 2];
        CpuFallbackKernels.computeCpuAttention(
                qSeg, cpu.kShard, cpu.vShard, tokens, numHeads, numKvHeads, headDim, scale, outO, outStats);
        return new WorkerResult(ShardAllocation.HOST_CPU_RANK, tokens, outO, outStats);
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("SplitKvSession has already been closed");
        }
        collective.ensureOpen();
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            for (GpuWorker w : gpuWorkers) {
                w.close();
            }
            if (cpuArena != null) {
                try {
                    cpuArena.close();
                } catch (Throwable ignored) {
                }
            }
        }
    }
}
