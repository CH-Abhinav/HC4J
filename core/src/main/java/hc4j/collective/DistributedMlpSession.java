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
import hc4j.GpuScope;
import hc4j.Tensor;
import hc4j.internal.collective.CpuFallbackKernels;
import hc4j.internal.collective.NativeOptimizerBridge;
import hc4j.nn.Activation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Persistent distributed Tensor-Parallel MLP session supporting both standard 2-layer
 * Megatron-LM MLPs ({@code act(X * W1 + b1) * W2 + b2}) and LLaMA / Qwen SwiGLU MLPs
 * ({@code (SiLU(X * Wgate) * (X * Wup)) * Wdown}).
 * <p>
 * Pre-shards and uploads all layer weights once at session construction (and only re-shards when
 * {@link #morph(WorkloadPlan)} changes the active device partition). During autoregressive decode
 * or repeated forward steps, only the small activation tensor {@code X} is uploaded to active GPU
 * ranks before executing local GEMMs, elementwise gating, and {@link Collective#allReduceSumSubset}.
 */
public final class DistributedMlpSession implements AutoCloseable {

    private final Collective collective;
    private final boolean swiglu;
    private final Activation activation;
    private final int dIn;
    private final int h;
    private final int dOut;

    // Canonical contiguous host F32 weights held once for re-sharding on morph (and optional CPU shards).
    private final Arena hostArena;
    private final MemorySegment w1OrGateSeg;   // [dIn, h]
    private final MemorySegment wUpSeg;        // [dIn, h] (null unless swiglu)
    private final MemorySegment w2OrDownSeg;   // [h, dOut]
    private final MemorySegment b1Seg;         // [h] (optional)
    private final float[] b2Values;            // [dOut] (optional)

    private WorkloadPlan currentPlan;
    private List<GpuShard> gpuShards = List.of();
    private CpuShard cpuShard;
    private List<Integer> prunedRanks = List.of();
    private int batchCapacity;
    private long weightUploadCount = 0L;
    private long forwardCount = 0L;
    private boolean closed = false;

    private static final class GpuShard {
        final int rank;
        final int units;
        final CollectiveBuffer bufW1OrGate;
        final CollectiveBuffer bufWUp;
        final CollectiveBuffer bufB1;
        final CollectiveBuffer bufW2OrDown;
        CollectiveBuffer bufX;
        CollectiveBuffer bufH;
        CollectiveBuffer bufUp;
        CollectiveBuffer bufY;

        GpuShard(int rank, int units, CollectiveBuffer bufW1OrGate, CollectiveBuffer bufWUp,
                 CollectiveBuffer bufB1, CollectiveBuffer bufW2OrDown,
                 CollectiveBuffer bufX, CollectiveBuffer bufH, CollectiveBuffer bufUp, CollectiveBuffer bufY) {
            this.rank = rank;
            this.units = units;
            this.bufW1OrGate = bufW1OrGate;
            this.bufWUp = bufWUp;
            this.bufB1 = bufB1;
            this.bufW2OrDown = bufW2OrDown;
            this.bufX = bufX;
            this.bufH = bufH;
            this.bufUp = bufUp;
            this.bufY = bufY;
        }

        void ensureBatchCapacity(Collective collective, int neededM, int dIn, int dOut, boolean swiglu) {
            long xElems = (long) neededM * dIn;
            if (bufX == null || bufX.elements() < xElems) {
                if (bufX != null) bufX.close();
                bufX = collective.allocate(rank, xElems);
            }
            long hElems = (long) neededM * units;
            if (bufH == null || bufH.elements() < hElems) {
                if (bufH != null) bufH.close();
                bufH = collective.allocate(rank, hElems);
            }
            if (swiglu && (bufUp == null || bufUp.elements() < hElems)) {
                if (bufUp != null) bufUp.close();
                bufUp = collective.allocate(rank, hElems);
            }
            long yElems = (long) neededM * dOut;
            if (bufY == null || bufY.elements() < yElems) {
                if (bufY != null) bufY.close();
                bufY = collective.allocate(rank, yElems);
            }
        }

        void closeAll() {
            closeQuietly(bufX);
            closeQuietly(bufH);
            closeQuietly(bufUp);
            closeQuietly(bufY);
            closeQuietly(bufW1OrGate);
            closeQuietly(bufWUp);
            closeQuietly(bufB1);
            closeQuietly(bufW2OrDown);
        }
    }

    private record CpuShard(
            int units,
            MemorySegment w1OrGateSlice,
            MemorySegment wUpSlice,
            MemorySegment b1Slice,
            MemorySegment w2OrDownSlice) {}

    private DistributedMlpSession(
            Collective collective,
            WorkloadPlan plan,
            int initialBatchCapacity,
            boolean swiglu,
            Activation activation,
            int dIn,
            int h,
            int dOut,
            Arena hostArena,
            MemorySegment w1OrGateSeg,
            MemorySegment wUpSeg,
            MemorySegment w2OrDownSeg,
            MemorySegment b1Seg,
            float[] b2Values) {
        this.collective = collective;
        this.swiglu = swiglu;
        this.activation = activation;
        this.dIn = dIn;
        this.h = h;
        this.dOut = dOut;
        this.hostArena = hostArena;
        this.w1OrGateSeg = w1OrGateSeg;
        this.wUpSeg = wUpSeg;
        this.w2OrDownSeg = w2OrDownSeg;
        this.b1Seg = b1Seg;
        this.b2Values = b2Values;
        this.batchCapacity = Math.max(1, initialBatchCapacity);
        try {
            allocateAndUploadShards(plan);
        } catch (RuntimeException | Error e) {
            releaseShards();
            hostArena.close();
            throw e;
        }
    }

    /**
     * Opens a persistent distributed 2-layer Megatron-LM MLP session pre-uploading weight shards once.
     *
     * @param collective active collective runtime
     * @param plan Tensor Parallel workload partitioning plan across hidden dimension {@code H}
     * @param w1 first projection weight tensor {@code [Din, H]}
     * @param b1 optional first projection bias {@code [H]} (may be {@code null})
     * @param w2 second projection weight tensor {@code [H, Dout]}
     * @param b2 optional output bias {@code [Dout]} (may be {@code null})
     * @param activation optional activation function between projections
     * @return opened {@link DistributedMlpSession}
     */
    public static DistributedMlpSession open(
            Collective collective,
            WorkloadPlan plan,
            Tensor w1,
            Tensor b1,
            Tensor w2,
            Tensor b2,
            Activation activation) {
        return open(collective, plan, 1, w1, b1, w2, b2, activation);
    }

    /**
     * Opens a persistent distributed 2-layer Megatron-LM MLP session with explicit initial batch scratch capacity.
     *
     * @param collective active collective runtime
     * @param plan Tensor Parallel workload partitioning plan across hidden dimension {@code H}
     * @param initialBatchCapacity initial batch row count reserved in reusable GPU scratch buffers
     * @param w1 first projection weight tensor {@code [Din, H]}
     * @param b1 optional first projection bias {@code [H]} (may be {@code null})
     * @param w2 second projection weight tensor {@code [H, Dout]}
     * @param b2 optional output bias {@code [Dout]} (may be {@code null})
     * @param activation optional activation function between projections
     * @return opened {@link DistributedMlpSession}
     */
    public static DistributedMlpSession open(
            Collective collective,
            WorkloadPlan plan,
            int initialBatchCapacity,
            Tensor w1,
            Tensor b1,
            Tensor w2,
            Tensor b2,
            Activation activation) {
        Objects.requireNonNull(collective, "collective must not be null");
        Objects.requireNonNull(plan, "plan must not be null");
        Objects.requireNonNull(w1, "w1 must not be null");
        Objects.requireNonNull(w2, "w2 must not be null");
        collective.ensureOpen();

        int[] w1Shape = w1.getShape();
        int[] w2Shape = w2.getShape();
        if (w1Shape.length != 2 || w2Shape.length != 2) {
            throw new IllegalArgumentException("W1 and W2 must be 2D tensors");
        }
        int dIn = w1Shape[0];
        int h = w1Shape[1];
        if (w2Shape[0] != h) {
            throw new IllegalArgumentException("W2 rows (" + w2Shape[0] + ") != W1 columns (" + h + ")");
        }
        int dOut = w2Shape[1];
        if (b1 != null && b1.getSize() != h) {
            throw new IllegalArgumentException("b1 elements (" + b1.getSize() + ") != H (" + h + ")");
        }
        if (b2 != null && b2.getSize() != dOut) {
            throw new IllegalArgumentException("b2 elements (" + b2.getSize() + ") != Dout (" + dOut + ")");
        }
        if (plan.totalUnits() != h) {
            throw new IllegalArgumentException("plan.totalUnits() (" + plan.totalUnits() + ") != H (" + h + ")");
        }

        Arena arena = Arena.ofShared();
        try {
            MemorySegment w1Seg = copyTensorF32(arena, w1, false);
            MemorySegment w2Seg = copyTensorF32(arena, w2, false);
            MemorySegment b1Seg = (b1 != null) ? copyTensorF32(arena, b1, false) : null;
            float[] b2Vals = (b2 != null) ? extractF32Array(b2) : null;
            return new DistributedMlpSession(
                    collective, plan, initialBatchCapacity, false, activation,
                    dIn, h, dOut, arena, w1Seg, null, w2Seg, b1Seg, b2Vals);
        } catch (RuntimeException | Error e) {
            arena.close();
            throw e;
        }
    }

    /**
     * Opens a persistent distributed SwiGLU MLP session pre-uploading gate, up, and down weight shards once.
     *
     * @param collective active collective runtime
     * @param plan Tensor Parallel workload partitioning plan across hidden dimension {@code H}
     * @param wGate gate projection weights ({@code [H, Din]} when {@code transposedWeights}, else {@code [Din, H]})
     * @param wUp up projection weights ({@code [H, Din]} when {@code transposedWeights}, else {@code [Din, H]})
     * @param wDown down projection weights ({@code [Dout, H]} when {@code transposedWeights}, else {@code [H, Dout]})
     * @param transposedWeights {@code true} when weights follow {@code [outFeatures, inFeatures]} linear layout
     * @return opened SwiGLU {@link DistributedMlpSession}
     */
    public static DistributedMlpSession openSwiglu(
            Collective collective,
            WorkloadPlan plan,
            Tensor wGate,
            Tensor wUp,
            Tensor wDown,
            boolean transposedWeights) {
        return openSwiglu(collective, plan, 1, wGate, wUp, wDown, transposedWeights);
    }

    /**
     * Opens a persistent distributed SwiGLU MLP session with explicit initial batch scratch capacity.
     *
     * @param collective active collective runtime
     * @param plan Tensor Parallel workload partitioning plan across hidden dimension {@code H}
     * @param initialBatchCapacity initial batch row count reserved in reusable GPU scratch buffers
     * @param wGate gate projection weights
     * @param wUp up projection weights
     * @param wDown down projection weights
     * @param transposedWeights {@code true} when weights follow {@code [outFeatures, inFeatures]} linear layout
     * @return opened SwiGLU {@link DistributedMlpSession}
     */
    public static DistributedMlpSession openSwiglu(
            Collective collective,
            WorkloadPlan plan,
            int initialBatchCapacity,
            Tensor wGate,
            Tensor wUp,
            Tensor wDown,
            boolean transposedWeights) {
        Objects.requireNonNull(collective, "collective must not be null");
        Objects.requireNonNull(plan, "plan must not be null");
        Objects.requireNonNull(wGate, "wGate must not be null");
        Objects.requireNonNull(wUp, "wUp must not be null");
        Objects.requireNonNull(wDown, "wDown must not be null");
        collective.ensureOpen();

        int[] gShape = wGate.getShape();
        int[] uShape = wUp.getShape();
        int[] dShape = wDown.getShape();
        if (gShape.length != 2 || uShape.length != 2 || dShape.length != 2) {
            throw new IllegalArgumentException("SwiGLU weights must be 2D tensors");
        }
        int dIn = transposedWeights ? gShape[1] : gShape[0];
        int h = transposedWeights ? gShape[0] : gShape[1];
        int upIn = transposedWeights ? uShape[1] : uShape[0];
        int upH = transposedWeights ? uShape[0] : uShape[1];
        if (upIn != dIn || upH != h) {
            throw new IllegalArgumentException("wUp shape does not match wGate shape");
        }
        int downH = transposedWeights ? dShape[1] : dShape[0];
        int dOut = transposedWeights ? dShape[0] : dShape[1];
        if (downH != h) {
            throw new IllegalArgumentException("wDown intermediate dim (" + downH + ") != H (" + h + ")");
        }
        if (plan.totalUnits() != h) {
            throw new IllegalArgumentException("plan.totalUnits() (" + plan.totalUnits() + ") != H (" + h + ")");
        }

        Arena arena = Arena.ofShared();
        try {
            MemorySegment gateSeg = copyTensorF32(arena, wGate, transposedWeights);
            MemorySegment upSeg = copyTensorF32(arena, wUp, transposedWeights);
            MemorySegment downSeg = copyTensorF32(arena, wDown, transposedWeights);
            return new DistributedMlpSession(
                    collective, plan, initialBatchCapacity, true, Activation.SILU,
                    dIn, h, dOut, arena, gateSeg, upSeg, downSeg, null, null);
        } catch (RuntimeException | Error e) {
            arena.close();
            throw e;
        }
    }

    /**
     * Executes a distributed forward pass on input activations {@code x [M, Din]} reusing cached weights.
     *
     * @param x input tensor {@code [M, Din]}
     * @return output tensor {@code [M, Dout]}
     */
    public Tensor forward(Tensor x) {
        return forwardWithReport(x).value();
    }

    /**
     * Executes a single decode/forward step on input activations {@code x [M, Din]} without re-uploading weights.
     *
     * @param x input tensor {@code [M, Din]}
     * @return output tensor {@code [M, Dout]}
     */
    public Tensor step(Tensor x) {
        return forward(x);
    }

    /**
     * Executes a distributed forward pass and returns both the output tensor and execution telemetry.
     *
     * @param x input tensor {@code [M, Din]}
     * @return distributed result carrying output {@code [M, Dout]} and {@link ExecutionReport}
     */
    public DistributedResult<Tensor> forwardWithReport(Tensor x) {
        ensureOpen();
        collective.ensureOpen();
        Objects.requireNonNull(x, "x must not be null");
        int[] xShape = x.getShape();
        if (xShape.length != 2 || xShape[1] != dIn) {
            throw new IllegalArgumentException("x shape " + Arrays.toString(xShape)
                    + " must be [M, " + dIn + "]");
        }
        int m = xShape[0];
        forwardCount++;

        long startWall = System.nanoTime();
        MemorySegment outSeg = Arena.ofAuto().allocate((long) m * dOut * 4L, 16);

        List<Integer> executedGpuRanks = new ArrayList<>(gpuShards.size());
        for (GpuShard s : gpuShards) {
            executedGpuRanks.add(s.rank);
        }
        long cpuUnits = (cpuShard != null) ? cpuShard.units : 0L;

        float[] yCpu = null;
        long computeStart = System.nanoTime();
        long syncStart;
        long syncEnd;

        try (Arena stepArena = Arena.ofShared();
             ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            MemorySegment xSeg = copyTensorF32(stepArena, x, false);
            if (m > batchCapacity) {
                batchCapacity = m;
            }
            for (GpuShard shard : gpuShards) {
                shard.ensureBatchCapacity(collective, batchCapacity, dIn, dOut, swiglu);
            }

            CompletableFuture<float[]> cpuFuture = null;
            if (cpuShard != null) {
                final CpuShard cs = cpuShard;
                cpuFuture = CompletableFuture.supplyAsync(() -> executeCpuShard(cs, xSeg, m), executor);
            }

            List<CompletableFuture<CollectiveBuffer>> gpuFutures = new ArrayList<>(gpuShards.size());
            for (GpuShard shard : gpuShards) {
                gpuFutures.add(CompletableFuture.supplyAsync(() -> {
                    shard.bufX.write(xSeg);
                    if (swiglu) {
                        collective.gemmBiasAct(shard.rank, shard.bufX, shard.bufW1OrGate, null,
                                shard.bufH, m, shard.units, dIn, false, Activation.SILU);
                        collective.gemmBiasAct(shard.rank, shard.bufX, shard.bufWUp, null,
                                shard.bufUp, m, shard.units, dIn, false, null);
                        collective.elemMul(shard.rank, shard.bufH, shard.bufUp, shard.bufH, m * shard.units);
                        collective.gemmBiasAct(shard.rank, shard.bufH, shard.bufW2OrDown, null,
                                shard.bufY, m, dOut, shard.units, false, null);
                    } else {
                        collective.gemmBiasAct(shard.rank, shard.bufX, shard.bufW1OrGate, shard.bufB1,
                                shard.bufH, m, shard.units, dIn, shard.bufB1 != null, activation);
                        collective.gemmBiasAct(shard.rank, shard.bufH, shard.bufW2OrDown, null,
                                shard.bufY, m, dOut, shard.units, false, null);
                    }
                    return shard.bufY;
                }, executor));
            }

            List<CollectiveBuffer> activeYBuffers = new ArrayList<>(gpuFutures.size());
            for (CompletableFuture<CollectiveBuffer> gf : gpuFutures) {
                activeYBuffers.add(gf.join());
            }
            if (cpuFuture != null) {
                yCpu = cpuFuture.join();
            }

            long computeEnd = System.nanoTime();
            syncStart = System.nanoTime();

            if (activeYBuffers.size() >= 2) {
                collective.allReduceSumSubset(activeYBuffers);
                activeYBuffers.getFirst().read(outSeg);
            } else if (activeYBuffers.size() == 1) {
                activeYBuffers.getFirst().read(outSeg);
            }
            syncEnd = System.nanoTime();

            if (yCpu != null) {
                boolean hasGpu = !activeYBuffers.isEmpty();
                for (int i = 0; i < m * dOut; i++) {
                    float base = hasGpu ? outSeg.get(ValueLayout.JAVA_FLOAT, (long) i * 4L) : 0.0f;
                    outSeg.set(ValueLayout.JAVA_FLOAT, (long) i * 4L, base + yCpu[i]);
                }
            }

            if (b2Values != null) {
                for (int r = 0; r < m; r++) {
                    long rOff = (long) r * dOut;
                    for (int c = 0; c < dOut; c++) {
                        float cur = outSeg.get(ValueLayout.JAVA_FLOAT, (rOff + c) * 4L);
                        outSeg.set(ValueLayout.JAVA_FLOAT, (rOff + c) * 4L, cur + b2Values[c]);
                    }
                }
            }

            long endWall = System.nanoTime();
            ExecutionReport report = new ExecutionReport(
                    swiglu ? "swigluMlp" : "transformerMlp",
                    currentPlan,
                    null,
                    (endWall - startWall) / 1e6,
                    (computeEnd - computeStart) / 1e6,
                    (syncEnd - syncStart) / 1e6,
                    executedGpuRanks,
                    prunedRanks,
                    cpuUnits);
            return new DistributedResult<>(Tensor.fromSegment(outSeg, DType.f32, m, dOut), report);
        }
    }

    /**
     * Morphs this session to a new {@link WorkloadPlan}. If {@code newPlan} has the exact same
     * rank/unit shard distribution as the current partition, this method is a zero-cost no-op.
     *
     * @param newPlan updated Tensor Parallel plan across {@code H}
     */
    public void morph(WorkloadPlan newPlan) {
        ensureOpen();
        Objects.requireNonNull(newPlan, "newPlan must not be null");
        if (newPlan.totalUnits() != h) {
            throw new IllegalArgumentException("newPlan.totalUnits() (" + newPlan.totalUnits() + ") != H (" + h + ")");
        }
        if (matchesPartition(newPlan)) {
            this.currentPlan = newPlan;
            return;
        }
        releaseShards();
        allocateAndUploadShards(newPlan);
    }

    /**
     * Returns {@code true} if {@code other} specifies the exact same per-rank non-zero unit assignment.
     *
     * @param other candidate workload plan
     * @return true when no re-sharding is required
     */
    public boolean matchesPartition(WorkloadPlan other) {
        if (other == null || currentPlan == null) {
            return false;
        }
        List<ShardAllocation> cur = currentPlan.allocations();
        List<ShardAllocation> nxt = other.allocations();
        if (cur.size() != nxt.size()) {
            return false;
        }
        for (int i = 0; i < cur.size(); i++) {
            ShardAllocation a = cur.get(i);
            ShardAllocation b = nxt.get(i);
            if (a.rank() != b.rank() || a.units() != b.units()) {
                return false;
            }
        }
        return true;
    }

    /** Returns the active {@link WorkloadPlan} for this session. */
    public WorkloadPlan plan() {
        return currentPlan;
    }

    /** Returns the number of weight-shard upload rounds executed (1 on open, +1 per partition morph). */
    public long weightUploadCount() {
        return weightUploadCount;
    }

    /** Returns the total number of {@link #forward(Tensor)} invocations completed. */
    public long forwardCount() {
        return forwardCount;
    }

    /** Returns input feature width {@code Din}. */
    public int inputDim() {
        return dIn;
    }

    /** Returns MLP intermediate width {@code H}. */
    public int hiddenWidth() {
        return h;
    }

    /** Returns output feature width {@code Dout}. */
    public int outputDim() {
        return dOut;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            releaseShards();
            hostArena.close();
        }
    }

    private float[] executeCpuShard(CpuShard cs, MemorySegment xSeg, int m) {
        if (!swiglu) {
            return CpuFallbackKernels.computeCpuMlpSlice(
                    xSeg, cs.w1OrGateSlice, cs.b1Slice, cs.w2OrDownSlice,
                    m, dIn, cs.units, dOut, 0, cs.units, activation);
        }
        float[] gateBuf = new float[m * cs.units];
        float[] upBuf = new float[m * cs.units];
        float[] outBuf = new float[m * dOut];
        MemorySegment gateSeg = MemorySegment.ofArray(gateBuf);
        MemorySegment upSeg = MemorySegment.ofArray(upBuf);
        MemorySegment outSeg = MemorySegment.ofArray(outBuf);
        CpuFallbackKernels.computeCpuGemm(xSeg, cs.w1OrGateSlice, gateSeg, 0, m, cs.units, dIn);
        CpuFallbackKernels.computeCpuGemm(xSeg, cs.wUpSlice, upSeg, 0, m, cs.units, dIn);
        for (int i = 0; i < gateBuf.length; i++) {
            float g = gateBuf[i];
            float sig = (g >= 0.0f)
                    ? (1.0f / (1.0f + (float) Math.exp(-g)))
                    : ((float) Math.exp(g) / (1.0f + (float) Math.exp(g)));
            gateBuf[i] = (g * sig) * upBuf[i];
        }
        CpuFallbackKernels.computeCpuGemm(gateSeg, cs.w2OrDownSlice, outSeg, 0, m, dOut, cs.units);
        return outBuf;
    }

    private void allocateAndUploadShards(WorkloadPlan plan) {
        this.currentPlan = plan;
        List<GpuShard> gpus = new ArrayList<>();
        List<Integer> pruned = new ArrayList<>();
        CpuShard cpu = null;

        int hOffset = 0;
        for (ShardAllocation alloc : plan.allocations()) {
            int units = (int) alloc.units();
            if (units <= 0) {
                if (alloc.rank() >= 0) {
                    pruned.add(alloc.rank());
                }
                continue;
            }
            int hStart = hOffset;
            hOffset += units;
            int rank = alloc.rank();

            if (rank == ShardAllocation.HOST_CPU_RANK) {
                MemorySegment w1Slice = extractColSlice(hostArena, w1OrGateSeg, dIn, h, hStart, units);
                MemorySegment wUpSlice = swiglu ? extractColSlice(hostArena, wUpSeg, dIn, h, hStart, units) : null;
                MemorySegment w2Slice = w2OrDownSeg.asSlice((long) hStart * dOut * 4L, (long) units * dOut * 4L);
                MemorySegment b1Slice = (b1Seg != null) ? b1Seg.asSlice((long) hStart * 4L, (long) units * 4L) : null;
                cpu = new CpuShard(units, w1Slice, wUpSlice, b1Slice, w2Slice);
            } else {
                MemorySegment w1Col = extractColSlice(hostArena, w1OrGateSeg, dIn, h, hStart, units);
                MemorySegment wUpCol = swiglu ? extractColSlice(hostArena, wUpSeg, dIn, h, hStart, units) : null;
                MemorySegment w2Row = w2OrDownSeg.asSlice((long) hStart * dOut * 4L, (long) units * dOut * 4L);

                CollectiveBuffer bufW1 = null;
                CollectiveBuffer bufWUp = null;
                CollectiveBuffer bufB1 = null;
                CollectiveBuffer bufW2 = null;
                CollectiveBuffer bufX = null;
                CollectiveBuffer bufH = null;
                CollectiveBuffer bufUp = null;
                CollectiveBuffer bufY = null;
                try {
                    bufW1 = collective.allocate(rank, (long) dIn * units);
                    bufW1.write(w1Col);
                    if (swiglu) {
                        bufWUp = collective.allocate(rank, (long) dIn * units);
                        bufWUp.write(wUpCol);
                    }
                    if (!swiglu && b1Seg != null) {
                        bufB1 = collective.allocate(rank, units);
                        bufB1.write(b1Seg.asSlice((long) hStart * 4L, (long) units * 4L));
                    }
                    bufW2 = collective.allocate(rank, (long) units * dOut);
                    bufW2.write(w2Row);

                    bufX = collective.allocate(rank, (long) batchCapacity * dIn);
                    bufH = collective.allocate(rank, (long) batchCapacity * units);
                    if (swiglu) {
                        bufUp = collective.allocate(rank, (long) batchCapacity * units);
                    }
                    bufY = collective.allocate(rank, (long) batchCapacity * dOut);
                    gpus.add(new GpuShard(rank, units, bufW1, bufWUp, bufB1, bufW2, bufX, bufH, bufUp, bufY));
                } catch (RuntimeException | Error e) {
                    closeQuietly(bufX);
                    closeQuietly(bufH);
                    closeQuietly(bufUp);
                    closeQuietly(bufY);
                    closeQuietly(bufW1);
                    closeQuietly(bufWUp);
                    closeQuietly(bufB1);
                    closeQuietly(bufW2);
                    throw e;
                }
            }
        }

        if (hOffset != h) {
            for (GpuShard gs : gpus) gs.closeAll();
            throw new IllegalArgumentException("Allocation unit sum (" + hOffset + ") != H (" + h + ")");
        }

        this.gpuShards = List.copyOf(gpus);
        this.cpuShard = cpu;
        this.prunedRanks = List.copyOf(pruned);
        this.weightUploadCount++;
    }

    private void releaseShards() {
        for (GpuShard gs : gpuShards) {
            gs.closeAll();
        }
        gpuShards = List.of();
        cpuShard = null;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("DistributedMlpSession has been closed");
        }
    }

    private static MemorySegment extractColSlice(Arena arena, MemorySegment src, int rows, int totalCols,
                                                 int colStart, int cols) {
        MemorySegment dst = arena.allocate((long) rows * cols * 4L, 16);
        for (int r = 0; r < rows; r++) {
            long srcOff = ((long) r * totalCols + colStart) * 4L;
            long dstOff = (long) r * cols * 4L;
            MemorySegment.copy(src, srcOff, dst, dstOff, (long) cols * 4L);
        }
        return dst;
    }

    private static float[] extractF32Array(Tensor t) {
        if (t.getDType() == DType.f32) {
            return t.toFloatArray();
        }
        try (GpuScope scope = GpuScope.open()) {
            Tensor cast = scope.track(t.cast(DType.f32));
            return cast.toFloatArray();
        }
    }

    private static MemorySegment copyTensorF32(Arena arena, Tensor t, boolean transpose2d) {
        int[] shape = t.getShape();
        long elements = t.getSize();
        MemorySegment raw = arena.allocate(elements * 4L, 16);
        if (t.getDType() == DType.f32) {
            Tensor contig = t.isContiguous() ? t : t.contiguous();
            try {
                contig.copyTo(raw);
            } finally {
                if (contig != t) {
                    contig.close();
                }
            }
        } else {
            try (GpuScope scope = GpuScope.open()) {
                Tensor cast = scope.track(t.cast(DType.f32));
                Tensor contig = cast.isContiguous() ? cast : scope.track(cast.contiguous());
                contig.copyTo(raw);
            }
        }

        if (!transpose2d) {
            return raw;
        }
        if (shape.length != 2) {
            throw new IllegalArgumentException("Expected 2D tensor for transpose");
        }
        int rows = shape[0];
        int cols = shape[1];
        MemorySegment transposed = arena.allocate(elements * 4L, 16);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                float val = raw.get(ValueLayout.JAVA_FLOAT, ((long) r * cols + c) * 4L);
                transposed.set(ValueLayout.JAVA_FLOAT, ((long) c * rows + r) * 4L, val);
            }
        }
        return transposed;
    }

    private static void closeQuietly(CollectiveBuffer buf) {
        if (buf != null) {
            try {
                buf.close();
            } catch (Throwable ignored) {
            }
        }
    }
}
