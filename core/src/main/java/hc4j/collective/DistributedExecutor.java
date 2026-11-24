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
import hc4j.nn.Activation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * High-performance distributed executor bridging workload planning with multi-GPU and hybrid execution.
 * <p>
 * Provides distributed execution for:
 * <ul>
 *     <li>Distributed Row-Sharded GEMM with straggler pruning and concurrent CPU-in-place compute</li>
 *     <li>Distributed Megatron-LM Tensor-Parallel Transformer MLP with active-subset AllReduce</li>
 *     <li>Lossless Zero-Waste Split-KV Decode Attention reclaiming straggler GPUs and host CPU</li>
 * </ul>
 */
public final class DistributedExecutor {

    private DistributedExecutor() {
        throw new AssertionError("No instances of DistributedExecutor");
    }

    // --- Workload 1: Distributed Row-Sharded GEMM ---

    /**
     * Executes distributed matrix multiplication {@code C[M, N] = A[M, K] * B[K, N]} according to a workload plan.
     *
     * @param collective collective instance
     * @param a input matrix A with shape [M, K]
     * @param b input matrix B with shape [K, N]
     * @param plan workload partitioning plan
     * @return distributed result containing output tensor [M, N] and execution telemetry
     * @throws NullPointerException if any argument is null
     * @throws IllegalArgumentException if input shapes or total units do not match
     * @throws IllegalStateException if collective is closed
     */
    public static DistributedResult<Tensor> matmul(Collective collective, Tensor a, Tensor b, WorkloadPlan plan) {
        Objects.requireNonNull(collective, "collective must not be null");
        Objects.requireNonNull(a, "a must not be null");
        Objects.requireNonNull(b, "b must not be null");
        Objects.requireNonNull(plan, "plan must not be null");
        collective.ensureOpen();

        int[] shapeA = a.getShape();
        int[] shapeB = b.getShape();
        if (shapeA.length != 2 || shapeB.length != 2) {
            throw new IllegalArgumentException("matmul requires 2D matrices, got A: " + Arrays.toString(shapeA)
                    + ", B: " + Arrays.toString(shapeB));
        }
        int m = shapeA[0];
        int k = shapeA[1];
        if (shapeB[0] != k) {
            throw new IllegalArgumentException("dimension mismatch: A columns (" + k + ") != B rows (" + shapeB[0] + ")");
        }
        int n = shapeB[1];
        if (plan.totalUnits() != m) {
            throw new IllegalArgumentException("plan.totalUnits() (" + plan.totalUnits() + ") != M (" + m + ")");
        }

        long startWall = System.nanoTime();

        Tensor contigA = a.isContiguous() ? a : a.contiguous();
        Tensor contigB = b.isContiguous() ? b : b.contiguous();

        List<Integer> executedGpuRanks = new ArrayList<>();
        List<Integer> prunedRanks = new ArrayList<>();
        long cpuUnits = 0L;

        List<CollectiveBuffer> activeBuffers = Collections.synchronizedList(new ArrayList<>());
        MemorySegment outSeg;

        long computeStart = System.nanoTime();

        try (Arena stageArena = Arena.ofShared();
             ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {

            MemorySegment aSeg = stageArena.allocate((long) m * k * 4L, 16);
            MemorySegment bSeg = stageArena.allocate((long) k * n * 4L, 16);
            outSeg = Arena.ofAuto().allocate((long) m * n * 4L, 16);

            contigA.copyTo(aSeg);
            contigB.copyTo(bSeg);

            List<CompletableFuture<Void>> futures = new ArrayList<>();
            int rowOffset = 0;

            for (ShardAllocation alloc : plan.allocations()) {
                int units = (int) alloc.units();
                if (units <= 0) {
                    if (alloc.rank() >= 0) {
                        prunedRanks.add(alloc.rank());
                    }
                    continue;
                }

                int rStart = rowOffset;
                rowOffset += units;
                int rank = alloc.rank();

                if (rank == ShardAllocation.HOST_CPU_RANK) {
                    cpuUnits += units;
                    futures.add(CompletableFuture.runAsync(() ->
                            computeCpuGemm(aSeg, bSeg, outSeg, rStart, units, n, k), executor));
                } else {
                    executedGpuRanks.add(rank);
                    futures.add(CompletableFuture.runAsync(() -> {
                        CollectiveBuffer bufA = null;
                        CollectiveBuffer bufB = null;
                        CollectiveBuffer bufC = null;
                        try {
                            bufA = collective.allocate(rank, (long) units * k);
                            bufB = collective.allocate(rank, (long) k * n);
                            bufC = collective.allocate(rank, (long) units * n);
                            activeBuffers.add(bufA);
                            activeBuffers.add(bufB);
                            activeBuffers.add(bufC);

                            MemorySegment aSlice = aSeg.asSlice((long) rStart * k * 4L, (long) units * k * 4L);
                            bufA.write(aSlice);
                            bufB.write(bSeg);

                            collective.gemmBiasAct(rank, bufA, bufB, null, bufC, units, n, k, false, null);

                            MemorySegment cSlice = outSeg.asSlice((long) rStart * n * 4L, (long) units * n * 4L);
                            bufC.read(cSlice);
                        } finally {
                            if (bufA != null) { bufA.close(); activeBuffers.remove(bufA); }
                            if (bufB != null) { bufB.close(); activeBuffers.remove(bufB); }
                            if (bufC != null) { bufC.close(); activeBuffers.remove(bufC); }
                        }
                    }, executor));
                }
            }

            for (CompletableFuture<Void> f : futures) {
                f.join();
            }
        } finally {
            for (CollectiveBuffer bRem : activeBuffers) {
                try { bRem.close(); } catch (Throwable ignored) {}
            }
        }

        long computeEnd = System.nanoTime();
        long endWall = System.nanoTime();

        double wallMs = (endWall - startWall) / 1e6;
        double computeMs = (computeEnd - computeStart) / 1e6;

        ExecutionReport report = new ExecutionReport(
                "matmul",
                plan,
                null,
                wallMs,
                computeMs,
                0.0,
                executedGpuRanks,
                prunedRanks,
                cpuUnits
        );

        Tensor resultTensor = Tensor.fromSegment(outSeg, DType.f32, m, n);
        return new DistributedResult<>(resultTensor, report);
    }

    /**
     * Executes distributed matrix multiplication according to a workload specification.
     *
     * @param collective collective instance
     * @param a input matrix A [M, K]
     * @param b input matrix B [K, N]
     * @param spec workload specification
     * @return distributed result
     */
    public static DistributedResult<Tensor> matmul(Collective collective, Tensor a, Tensor b, WorkloadSpec spec) {
        Objects.requireNonNull(collective, "collective must not be null");
        Objects.requireNonNull(spec, "spec must not be null");
        return matmul(collective, a, b, collective.planWork(spec));
    }

    /**
     * Convenience matrix multiplication partitioning rows with zero synchronization.
     *
     * @param collective collective instance
     * @param a input matrix A [M, K]
     * @param b input matrix B [K, N]
     * @return output matrix [M, N]
     */
    public static Tensor matmul(Collective collective, Tensor a, Tensor b) {
        Objects.requireNonNull(a, "a must not be null");
        Objects.requireNonNull(b, "b must not be null");
        int[] aShape = a.getShape();
        int m = aShape[0];
        int k = aShape[1];
        int n = b.getShape()[1];
        WorkloadSpec spec = WorkloadSpec.builder(m, 2.0 * k * n, (long) (k + n) * 4L)
                .syncBytes(0L)
                .ioMultiplier(1.0)
                .build();
        return matmul(collective, a, b, spec).value();
    }

    private static void computeCpuGemm(MemorySegment aSeg, MemorySegment bSeg, MemorySegment outSeg,
                                       int rStart, int units, int n, int k) {
        for (int i = 0; i < units; i++) {
            int globalRow = rStart + i;
            long aRowOff = (long) globalRow * k;
            long outRowOff = (long) globalRow * n;
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int p = 0; p < k; p++) {
                    float aVal = aSeg.get(ValueLayout.JAVA_FLOAT, (aRowOff + p) * 4L);
                    float bVal = bSeg.get(ValueLayout.JAVA_FLOAT, ((long) p * n + j) * 4L);
                    sum += aVal * bVal;
                }
                outSeg.set(ValueLayout.JAVA_FLOAT, (outRowOff + j) * 4L, sum);
            }
        }
    }

    // --- Workload 2: Distributed Megatron-LM Tensor-Parallel Transformer MLP ---

    /**
     * Executes a distributed 2-layer Megatron-LM Tensor Parallel Transformer MLP:
     * {@code act(X * W1 + b1) * W2 + b2} across active collective ranks with active-subset AllReduce.
     *
     * @param collective collective instance
     * @param x input tensor X [M, Din]
     * @param w1 gate/up projection weights W1 [Din, H]
     * @param b1 optional bias b1 [H] (may be null)
     * @param w2 down projection weights W2 [H, Dout]
     * @param b2 optional bias b2 [Dout] (may be null)
     * @param activation activation function (null for NONE, or GELU, SILU, RELU)
     * @param plan workload partitioning plan partitioning hidden dimension H
     * @return distributed result containing output tensor [M, Dout] and execution telemetry
     * @throws NullPointerException if mandatory arguments are null
     * @throws IllegalArgumentException if shapes, dimensions, or plan units do not match
     * @throws IllegalStateException if collective is closed
     */
    public static DistributedResult<Tensor> transformerMlp(Collective collective, Tensor x, Tensor w1, Tensor b1,
                                                           Tensor w2, Tensor b2, Activation activation, WorkloadPlan plan) {
        Objects.requireNonNull(collective, "collective must not be null");
        Objects.requireNonNull(x, "x must not be null");
        Objects.requireNonNull(w1, "w1 must not be null");
        Objects.requireNonNull(w2, "w2 must not be null");
        Objects.requireNonNull(plan, "plan must not be null");
        collective.ensureOpen();

        int[] xShape = x.getShape();
        int[] w1Shape = w1.getShape();
        int[] w2Shape = w2.getShape();
        if (xShape.length != 2 || w1Shape.length != 2 || w2Shape.length != 2) {
            throw new IllegalArgumentException("X, W1, W2 must be 2D tensors");
        }
        int m = xShape[0];
        int dIn = xShape[1];
        if (w1Shape[0] != dIn) {
            throw new IllegalArgumentException("W1 rows (" + w1Shape[0] + ") != X columns (" + dIn + ")");
        }
        int h = w1Shape[1];
        if (w2Shape[0] != h) {
            throw new IllegalArgumentException("W2 rows (" + w2Shape[0] + ") != W1 columns (" + h + ")");
        }
        int dOut = w2Shape[1];

        if (b1 != null && (b1.getSize() != h)) {
            throw new IllegalArgumentException("b1 elements (" + b1.getSize() + ") != H (" + h + ")");
        }
        if (b2 != null && (b2.getSize() != dOut)) {
            throw new IllegalArgumentException("b2 elements (" + b2.getSize() + ") != Dout (" + dOut + ")");
        }
        if (plan.totalUnits() != h) {
            throw new IllegalArgumentException("plan.totalUnits() (" + plan.totalUnits() + ") != H (" + h + ")");
        }

        long startWall = System.nanoTime();

        Tensor contigX = x.isContiguous() ? x : x.contiguous();
        Tensor contigW1 = w1.isContiguous() ? w1 : w1.contiguous();
        Tensor contigW2 = w2.isContiguous() ? w2 : w2.contiguous();
        Tensor contigB1 = (b1 != null && !b1.isContiguous()) ? b1.contiguous() : b1;

        List<Integer> executedGpuRanks = new ArrayList<>();
        List<Integer> prunedRanks = new ArrayList<>();
        long cpuUnits = 0L;

        List<CollectiveBuffer> activeYBuffers = Collections.synchronizedList(new ArrayList<>());
        List<CollectiveBuffer> allAllocatedBuffers = Collections.synchronizedList(new ArrayList<>());

        MemorySegment yGpuSeg;
        float[] yCpu = null;

        long computeStart = System.nanoTime();

        try (Arena stageArena = Arena.ofShared();
             ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {

            MemorySegment xSeg = stageArena.allocate((long) m * dIn * 4L, 16);
            MemorySegment w1Seg = stageArena.allocate((long) dIn * h * 4L, 16);
            MemorySegment w2Seg = stageArena.allocate((long) h * dOut * 4L, 16);
            MemorySegment b1Seg = (contigB1 != null) ? stageArena.allocate((long) h * 4L, 16) : null;

            contigX.copyTo(xSeg);
            contigW1.copyTo(w1Seg);
            contigW2.copyTo(w2Seg);
            if (contigB1 != null) {
                contigB1.copyTo(b1Seg);
            }

            yGpuSeg = Arena.ofAuto().allocate((long) m * dOut * 4L, 16);

            List<CompletableFuture<float[]>> cpuFutures = new ArrayList<>();
            List<CompletableFuture<CollectiveBuffer>> gpuFutures = new ArrayList<>();

            int hOffset = 0;
            for (ShardAllocation alloc : plan.allocations()) {
                int units = (int) alloc.units();
                if (units <= 0) {
                    if (alloc.rank() >= 0) {
                        prunedRanks.add(alloc.rank());
                    }
                    continue;
                }

                int hStart = hOffset;
                hOffset += units;
                int rank = alloc.rank();

                if (rank == ShardAllocation.HOST_CPU_RANK) {
                    cpuUnits += units;
                    cpuFutures.add(CompletableFuture.supplyAsync(() ->
                            computeCpuMlpSlice(xSeg, w1Seg, b1Seg, w2Seg, m, dIn, h, dOut, hStart, units, activation), executor));
                } else {
                    executedGpuRanks.add(rank);
                    gpuFutures.add(CompletableFuture.supplyAsync(() -> {
                        CollectiveBuffer bufX = null;
                        CollectiveBuffer bufW1 = null;
                        CollectiveBuffer bufB1 = null;
                        CollectiveBuffer bufH = null;
                        CollectiveBuffer bufW2 = null;
                        CollectiveBuffer bufY = null;
                        try {
                            MemorySegment w1ColSlice = stageArena.allocate((long) dIn * units * 4L, 16);
                            for (int r = 0; r < dIn; r++) {
                                long srcOff = ((long) r * h + hStart) * 4L;
                                long dstOff = (long) r * units * 4L;
                                MemorySegment.copy(w1Seg, srcOff, w1ColSlice, dstOff, (long) units * 4L);
                            }

                            MemorySegment w2RowSlice = w2Seg.asSlice((long) hStart * dOut * 4L, (long) units * dOut * 4L);

                            bufX = collective.allocate(rank, (long) m * dIn);
                            bufW1 = collective.allocate(rank, (long) dIn * units);
                            bufH = collective.allocate(rank, (long) m * units);
                            bufW2 = collective.allocate(rank, (long) units * dOut);
                            bufY = collective.allocate(rank, (long) m * dOut);

                            allAllocatedBuffers.add(bufX);
                            allAllocatedBuffers.add(bufW1);
                            allAllocatedBuffers.add(bufH);
                            allAllocatedBuffers.add(bufW2);
                            allAllocatedBuffers.add(bufY);

                            bufX.write(xSeg);
                            bufW1.write(w1ColSlice);
                            bufW2.write(w2RowSlice);

                            if (b1Seg != null) {
                                MemorySegment b1Slice = b1Seg.asSlice((long) hStart * 4L, (long) units * 4L);
                                bufB1 = collective.allocate(rank, units);
                                allAllocatedBuffers.add(bufB1);
                                bufB1.write(b1Slice);
                                collective.gemmBiasAct(rank, bufX, bufW1, bufB1, bufH, m, units, dIn, true, activation);
                            } else {
                                collective.gemmBiasAct(rank, bufX, bufW1, null, bufH, m, units, dIn, false, activation);
                            }

                            bufX.close(); allAllocatedBuffers.remove(bufX);
                            bufW1.close(); allAllocatedBuffers.remove(bufW1);
                            if (bufB1 != null) { bufB1.close(); allAllocatedBuffers.remove(bufB1); }

                            collective.gemmBiasAct(rank, bufH, bufW2, null, bufY, m, dOut, units, false, null);

                            bufH.close(); allAllocatedBuffers.remove(bufH);
                            bufW2.close(); allAllocatedBuffers.remove(bufW2);

                            return bufY;
                        } catch (Throwable t) {
                            if (bufX != null) { try { bufX.close(); } catch (Throwable ignored) {} }
                            if (bufW1 != null) { try { bufW1.close(); } catch (Throwable ignored) {} }
                            if (bufB1 != null) { try { bufB1.close(); } catch (Throwable ignored) {} }
                            if (bufH != null) { try { bufH.close(); } catch (Throwable ignored) {} }
                            if (bufW2 != null) { try { bufW2.close(); } catch (Throwable ignored) {} }
                            if (bufY != null) { try { bufY.close(); } catch (Throwable ignored) {} }
                            throw t;
                        }
                    }, executor));
                }
            }

            for (CompletableFuture<CollectiveBuffer> gf : gpuFutures) {
                activeYBuffers.add(gf.join());
            }

            for (CompletableFuture<float[]> cf : cpuFutures) {
                float[] partialCpu = cf.join();
                if (yCpu == null) {
                    yCpu = partialCpu;
                } else {
                    for (int i = 0; i < yCpu.length; i++) {
                        yCpu[i] += partialCpu[i];
                    }
                }
            }
        }

        long computeEnd = System.nanoTime();
        long syncStart = System.nanoTime();

        try {
            if (!activeYBuffers.isEmpty()) {
                collective.allReduceSumSubset(activeYBuffers);
                activeYBuffers.getFirst().read(yGpuSeg);
            }
        } finally {
            for (CollectiveBuffer b : activeYBuffers) {
                try { b.close(); } catch (Throwable ignored) {}
            }
            for (CollectiveBuffer bRem : allAllocatedBuffers) {
                try { bRem.close(); } catch (Throwable ignored) {}
            }
        }

        long syncEnd = System.nanoTime();

        // Merge CPU and optional bias b2
        if (yCpu != null) {
            for (int i = 0; i < m * dOut; i++) {
                float base = (!activeYBuffers.isEmpty()) ? yGpuSeg.get(ValueLayout.JAVA_FLOAT, (long) i * 4L) : 0.0f;
                yGpuSeg.set(ValueLayout.JAVA_FLOAT, (long) i * 4L, base + yCpu[i]);
            }
        }

        if (b2 != null) {
            float[] b2Vals = b2.toFloatArray();
            for (int r = 0; r < m; r++) {
                long rOff = (long) r * dOut;
                for (int c = 0; c < dOut; c++) {
                    float cur = yGpuSeg.get(ValueLayout.JAVA_FLOAT, (rOff + c) * 4L);
                    yGpuSeg.set(ValueLayout.JAVA_FLOAT, (rOff + c) * 4L, cur + b2Vals[c]);
                }
            }
        }

        long endWall = System.nanoTime();

        double wallMs = (endWall - startWall) / 1e6;
        double computeMs = (computeEnd - computeStart) / 1e6;
        double syncMs = (syncEnd - syncStart) / 1e6;

        ExecutionReport report = new ExecutionReport(
                "transformerMlp",
                plan,
                null,
                wallMs,
                computeMs,
                syncMs,
                executedGpuRanks,
                prunedRanks,
                cpuUnits
        );

        Tensor resultTensor = Tensor.fromSegment(yGpuSeg, DType.f32, m, dOut);
        return new DistributedResult<>(resultTensor, report);
    }

    /**
     * Executes distributed Transformer MLP according to a workload specification.
     *
     * @param collective collective instance
     * @param x input tensor X
     * @param w1 gate/up projection weights W1
     * @param b1 optional bias b1
     * @param w2 down projection weights W2
     * @param b2 optional bias b2
     * @param activation activation function
     * @param spec workload specification
     * @return distributed result
     */
    public static DistributedResult<Tensor> transformerMlp(Collective collective, Tensor x, Tensor w1, Tensor b1,
                                                           Tensor w2, Tensor b2, Activation activation, WorkloadSpec spec) {
        Objects.requireNonNull(collective, "collective must not be null");
        Objects.requireNonNull(spec, "spec must not be null");
        return transformerMlp(collective, x, w1, b1, w2, b2, activation, collective.planWork(spec));
    }

    /**
     * Convenience Transformer MLP execution auto-constructing a standard Tensor Parallel workload specification.
     *
     * @param collective collective instance
     * @param x input tensor X [M, Din]
     * @param w1 weights W1 [Din, H]
     * @param b1 optional bias b1 [H]
     * @param w2 weights W2 [H, Dout]
     * @param b2 optional bias b2 [Dout]
     * @param activation activation function
     * @return distributed result
     */
    public static DistributedResult<Tensor> transformerMlp(Collective collective, Tensor x, Tensor w1, Tensor b1,
                                                           Tensor w2, Tensor b2, Activation activation) {
        Objects.requireNonNull(x, "x must not be null");
        Objects.requireNonNull(w1, "w1 must not be null");
        Objects.requireNonNull(w2, "w2 must not be null");
        int[] xShape = x.getShape();
        int m = xShape[0];
        int dIn = xShape[1];
        int h = w1.getShape()[1];
        int dOut = w2.getShape()[1];
        WorkloadSpec spec = WorkloadSpec.builder(h, 4.0 * m * dIn, 8L * dIn + 4L * m)
                .syncBytes((long) m * dOut * 4L)
                .ioMultiplier(1.0)
                .build();
        return transformerMlp(collective, x, w1, b1, w2, b2, activation, spec);
    }

    private static float[] computeCpuMlpSlice(MemorySegment xSeg, MemorySegment w1Seg, MemorySegment b1Seg,
                                              MemorySegment w2Seg, int m, int dIn, int h, int dOut,
                                              int hStart, int units, Activation activation) {
        float[] hSlice = new float[m * units];

        for (int i = 0; i < m; i++) {
            long xRowOff = (long) i * dIn;
            int hRowOff = i * units;
            for (int j = 0; j < units; j++) {
                int globalH = hStart + j;
                float sum = 0.0f;
                for (int p = 0; p < dIn; p++) {
                    float xVal = xSeg.get(ValueLayout.JAVA_FLOAT, (xRowOff + p) * 4L);
                    float wVal = w1Seg.get(ValueLayout.JAVA_FLOAT, ((long) p * h + globalH) * 4L);
                    sum += xVal * wVal;
                }
                if (b1Seg != null) {
                    sum += b1Seg.get(ValueLayout.JAVA_FLOAT, (long) globalH * 4L);
                }
                hSlice[hRowOff + j] = applyActivation(sum, activation);
            }
        }

        float[] yCpu = new float[m * dOut];
        for (int i = 0; i < m; i++) {
            int hRowOff = i * units;
            int yRowOff = i * dOut;
            for (int j = 0; j < dOut; j++) {
                float sum = 0.0f;
                for (int p = 0; p < units; p++) {
                    int globalH = hStart + p;
                    float hVal = hSlice[hRowOff + p];
                    float wVal = w2Seg.get(ValueLayout.JAVA_FLOAT, ((long) globalH * dOut + j) * 4L);
                    sum += hVal * wVal;
                }
                yCpu[yRowOff + j] = sum;
            }
        }

        return yCpu;
    }

    private static float applyActivation(float x, Activation activation) {
        if (activation == null) {
            return x;
        }
        return switch (activation) {
            case RELU -> Math.max(x, 0.0f);
            case GELU -> {
                float cubic = 0.7978845608028654f * (x + 0.044715f * x * x * x);
                float inner = Math.clamp(cubic, -10.0f, 10.0f);
                yield 0.5f * x * (1.0f + (float) Math.tanh(inner));
            }
            case SILU -> {
                float e = (float) Math.exp(-Math.abs(x));
                float num = (x >= 0.0f) ? 1.0f : e;
                yield x * num / (1.0f + e);
            }
            default -> throw new UnsupportedOperationException("Unsupported MLP activation: " + activation);
        };
    }

    // --- Workload 3: Lossless Zero-Waste Split-KV Decode Attention ---

    /**
     * Executes lossless Split-KV Decode Attention for a single query token across a partitioned KV cache.
     *
     * @param collective collective instance
     * @param q query tensor [numHeads, headDim]
     * @param kCache full key cache [numKvHeads, totalTokens, headDim]
     * @param vCache full value cache [numKvHeads, totalTokens, headDim]
     * @param scale scaling factor applied to query-key dot products
     * @param rolePlan multi-operator role assignment plan
     * @return distributed result containing output attention tensor [numHeads, headDim] and execution telemetry
     */
    public static DistributedResult<Tensor> splitKvAttention(Collective collective, Tensor q, Tensor kCache,
                                                             Tensor vCache, float scale, RolePlan rolePlan) {
        Objects.requireNonNull(q, "q must not be null");
        Objects.requireNonNull(kCache, "kCache must not be null");
        int[] kShape = kCache.getShape();
        int numKvHeads = kShape[0];
        int headDim = kShape[2];
        int numHeads = (int) (q.getSize() / headDim);
        try (SplitKvSession session = SplitKvSession.open(collective, rolePlan, numHeads, numKvHeads, headDim, kCache, vCache)) {
            return session.attend(q, scale);
        }
    }
}
