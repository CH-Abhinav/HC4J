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

import static hc4j.testutil.ExceptionTable.assertThrowsWithMessage;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import hc4j.collective.Collective;
import hc4j.collective.CollectiveBuffer;
import hc4j.collective.DeviceRole;
import hc4j.collective.DistributedExecutor;
import hc4j.collective.DistributedResult;
import hc4j.collective.ExecutionReport;
import hc4j.collective.RolePlan;
import hc4j.collective.ShardAllocation;
import hc4j.collective.SplitKvAssignment;
import hc4j.collective.SplitKvSession;
import hc4j.collective.WorkloadPlan;
import hc4j.engine.Engine;
import hc4j.engine.MemoryStats;
import hc4j.nn.Activation;
import hc4j.testutil.GpuTestSupport;
import hc4j.testutil.TestAssertions;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("DistributedExecutor Multi-GPU Execution Bridge Tests")
class DistributedExecutorTest {

    private long initialTensors;

    @BeforeEach
    void setUp() {
        GpuTestSupport.assumeGpuAvailable();
        MemoryStats stats = Engine.memoryStats();
        initialTensors = stats.tensorsDevice() + stats.tensorsHost() + stats.tensorsEvicted();
    }

    @AfterEach
    void tearDown() {
        MemoryStats stats = Engine.memoryStats();
        long currentTensors = stats.tensorsDevice() + stats.tensorsHost() + stats.tensorsEvicted();
        assertEquals(initialTensors, currentTensors, "No single-GPU tensors should be leaked after test execution");
    }

    // --- Active-Subset AllReduce ---

    @Test
    @DisplayName("Active-Subset AllReduce: 4 ranks, active subset {0, 2}, pruned ranks {1, 3} untouched")
    void testActiveSubsetAllReduce4Rank() {
        try (Collective ring = Collective.open(4)) {
            final int count = 6;
            CollectiveBuffer b0 = ring.allocate(0, count);
            CollectiveBuffer b1 = ring.allocate(1, count);
            CollectiveBuffer b2 = ring.allocate(2, count);
            CollectiveBuffer b3 = ring.allocate(3, count);
            try {
                b0.write(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
                b1.write(new float[]{10f, 20f, 30f, 40f, 50f, 60f});
                b2.write(new float[]{7f, 8f, 9f, 10f, 11f, 12f});
                b3.write(new float[]{100f, 200f, 300f, 400f, 500f, 600f});

                ring.allReduceSumSubset(b0, b2);

                float[] expected02 = new float[]{8f, 10f, 12f, 14f, 16f, 18f};
                TestAssertions.assertArrayClose(expected02, b0.read(), 1e-4f, 1e-4f);
                TestAssertions.assertArrayClose(expected02, b2.read(), 1e-4f, 1e-4f);

                // Pruned ranks must be completely untouched
                float[] untouched1 = new float[]{10f, 20f, 30f, 40f, 50f, 60f};
                float[] untouched3 = new float[]{100f, 200f, 300f, 400f, 500f, 600f};
                TestAssertions.assertArrayClose(untouched1, b1.read(), 1e-4f, 1e-4f);
                TestAssertions.assertArrayClose(untouched3, b3.read(), 1e-4f, 1e-4f);
            } finally {
                b0.close();
                b1.close();
                b2.close();
                b3.close();
            }
        }
    }

    @Test
    @DisplayName("Active-Subset AllReduce: single rank subset is a zero-cost no-op")
    void testActiveSubsetAllReduceSingleRankNoOp() {
        try (Collective ring = Collective.open(2)) {
            CollectiveBuffer b0 = ring.allocate(0, 4);
            try {
                float[] initial = new float[]{42f, 43f, 44f, 45f};
                b0.write(initial);

                ring.allReduceSumSubset(b0);

                TestAssertions.assertArrayClose(initial, b0.read(), 1e-4f, 1e-4f);
            } finally {
                b0.close();
            }
        }
    }

    @Test
    @DisplayName("Active-Subset AllReduce: validates duplicate ranks, empty buffers, and mismatched lengths")
    void testActiveSubsetAllReduceValidation() {
        try (Collective ring = Collective.open(2)) {
            CollectiveBuffer b0 = ring.allocate(0, 4);
            CollectiveBuffer b1 = ring.allocate(1, 4);
            CollectiveBuffer b1Wrong = ring.allocate(1, 8);
            try {
                assertThrowsWithMessage(IllegalArgumentException.class,
                        () -> ring.allReduceSumSubset(new CollectiveBuffer[0]),
                        "activeBuffers must not be empty");

                assertThrowsWithMessage(IllegalArgumentException.class,
                        () -> ring.allReduceSumSubset(b0, b0),
                        "duplicate rank 0");

                assertThrowsWithMessage(IllegalArgumentException.class,
                        () -> ring.allReduceSumSubset(b0, b1Wrong),
                        "mismatched element counts");
            } finally {
                b0.close();
                b1.close();
                b1Wrong.close();
            }
        }
    }

    // --- Workload 1: Distributed Row-Sharded GEMM ---

    @Test
    @DisplayName("Distributed GEMM: 2-rank proportional GPU plan")
    void testDistributedMatmulTwoRankProportional() {
        try (Collective ring = Collective.open(2)) {
            int m = 32;
            int k = 16;
            int n = 24;

            float[] aData = generateDeterministicFloats(m * k, 1234);
            float[] bData = generateDeterministicFloats(k * n, 5678);

            try (Tensor a = Tensor.fromArray(aData, m, k);
                 Tensor b = Tensor.fromArray(bData, k, n);
                 Tensor ref = a.matmul(b)) {

                WorkloadPlan plan = new WorkloadPlan(
                        m,
                        List.of(new ShardAllocation(0, 16, 0.5, 0.0, 0L, false),
                                new ShardAllocation(1, 16, 0.5, 0.0, 0L, false)),
                        List.of(0, 1),
                        0.0, 0.0, 0.0
                );

                DistributedResult<Tensor> result = DistributedExecutor.matmul(ring, a, b, plan);
                try (Tensor out = result.value()) {
                    TestAssertions.assertArrayClose(ref.toFloatArray(), out.toFloatArray(), 1e-4f, 1e-4f);

                    ExecutionReport report = result.report();
                    assertEquals("matmul", report.operation());
                    assertEquals(List.of(0, 1), report.executedGpuRanks());
                    assertEquals(0L, report.cpuInPlaceUnits());
                    assertTrue(report.wallTimeMs() > 0.0);
                }
            }
        }
    }

    @Test
    @DisplayName("Distributed GEMM: 4-rank collective with straggler-pruned ranks {0, 2}")
    void testDistributedMatmulFourRankStragglerPruned() {
        try (Collective ring = Collective.open(4)) {
            int m = 32;
            int k = 16;
            int n = 24;

            float[] aData = generateDeterministicFloats(m * k, 2345);
            float[] bData = generateDeterministicFloats(k * n, 6789);

            try (Tensor a = Tensor.fromArray(aData, m, k);
                 Tensor b = Tensor.fromArray(bData, k, n);
                 Tensor ref = a.matmul(b)) {

                WorkloadPlan plan = new WorkloadPlan(
                        m,
                        List.of(new ShardAllocation(0, 16, 0.5, 0.0, 0L, false),
                                new ShardAllocation(1, 0, 0.0, 0.0, 0L, true),
                                new ShardAllocation(2, 16, 0.5, 0.0, 0L, false),
                                new ShardAllocation(3, 0, 0.0, 0.0, 0L, true)),
                        List.of(0, 2),
                        0.0, 0.0, 0.0
                );

                DistributedResult<Tensor> result = DistributedExecutor.matmul(ring, a, b, plan);
                try (Tensor out = result.value()) {
                    TestAssertions.assertArrayClose(ref.toFloatArray(), out.toFloatArray(), 1e-4f, 1e-4f);

                    ExecutionReport report = result.report();
                    assertEquals(List.of(0, 2), report.executedGpuRanks());
                    assertEquals(List.of(1, 3), report.prunedGpuRanks());
                    assertEquals(0L, report.cpuInPlaceUnits());
                }
            }
        }
    }

    @Test
    @DisplayName("Distributed GEMM: Hybrid GPU + HOST_CPU_RANK CPU-in-place execution")
    void testDistributedMatmulHybridGpuAndCpu() {
        try (Collective ring = Collective.open(2)) {
            int m = 32;
            int k = 16;
            int n = 24;

            float[] aData = generateDeterministicFloats(m * k, 3456);
            float[] bData = generateDeterministicFloats(k * n, 7890);

            try (Tensor a = Tensor.fromArray(aData, m, k);
                 Tensor b = Tensor.fromArray(bData, k, n);
                 Tensor ref = a.matmul(b)) {

                // 20 rows on GPU rank 0, 12 rows on Host CPU in-place
                WorkloadPlan plan = new WorkloadPlan(
                        m,
                        List.of(new ShardAllocation(0, 20, 20.0 / 32.0, 0.0, 0L, false),
                                new ShardAllocation(ShardAllocation.HOST_CPU_RANK, 12, 12.0 / 32.0, 0.0, 0L, false)),
                        List.of(0),
                        0.0, 0.0, 0.0
                );

                DistributedResult<Tensor> result = DistributedExecutor.matmul(ring, a, b, plan);
                try (Tensor out = result.value()) {
                    TestAssertions.assertArrayClose(ref.toFloatArray(), out.toFloatArray(), 1e-4f, 1e-4f);

                    ExecutionReport report = result.report();
                    assertEquals(List.of(0), report.executedGpuRanks());
                    assertEquals(12L, report.cpuInPlaceUnits());
                }
            }
        }
    }

    // --- Workload 2: Distributed Megatron-LM Tensor-Parallel Transformer MLP ---

    @Test
    @DisplayName("Distributed Transformer MLP: 2-rank TP with GELU activation and biases")
    void testDistributedTransformerMlpTwoRankGelu() {
        try (Collective ring = Collective.open(2)) {
            int m = 8;
            int dIn = 16;
            int h = 32;
            int dOut = 16;

            float[] xData = generateDeterministicFloats(m * dIn, 101);
            float[] w1Data = generateDeterministicFloats(dIn * h, 102);
            float[] b1Data = generateDeterministicFloats(h, 103);
            float[] w2Data = generateDeterministicFloats(h * dOut, 104);
            float[] b2Data = generateDeterministicFloats(dOut, 105);

            float[] refData = cpuReferenceMlp(xData, w1Data, b1Data, w2Data, b2Data, m, dIn, h, dOut, Activation.GELU);

            try (Tensor x = Tensor.fromArray(xData, m, dIn);
                 Tensor w1 = Tensor.fromArray(w1Data, dIn, h);
                 Tensor b1 = Tensor.fromArray(b1Data, h);
                 Tensor w2 = Tensor.fromArray(w2Data, h, dOut);
                 Tensor b2 = Tensor.fromArray(b2Data, dOut)) {

                WorkloadPlan plan = new WorkloadPlan(
                        h,
                        List.of(new ShardAllocation(0, 16, 0.5, 0.0, 0L, false),
                                new ShardAllocation(1, 16, 0.5, 0.0, 0L, false)),
                        List.of(0, 1),
                        0.0, 0.0, 0.0
                );

                DistributedResult<Tensor> result = DistributedExecutor.transformerMlp(
                        ring, x, w1, b1, w2, b2, Activation.GELU, plan);

                try (Tensor out = result.value()) {
                    TestAssertions.assertArrayClose(refData, out.toFloatArray(), 1e-4f, 1e-4f);

                    ExecutionReport report = result.report();
                    assertEquals("transformerMlp", report.operation());
                    assertEquals(List.of(0, 1), report.executedGpuRanks());
                    assertTrue(report.syncTimeMs() >= 0.0);
                }
            }
        }
    }

    @Test
    @DisplayName("Distributed Transformer MLP: 4-rank with pruned ranks {1, 2} and SiLU activation")
    void testDistributedTransformerMlpFourRankPrunedSilu() {
        try (Collective ring = Collective.open(4)) {
            int m = 8;
            int dIn = 16;
            int h = 32;
            int dOut = 16;

            float[] xData = generateDeterministicFloats(m * dIn, 201);
            float[] w1Data = generateDeterministicFloats(dIn * h, 202);
            float[] b1Data = generateDeterministicFloats(h, 203);
            float[] w2Data = generateDeterministicFloats(h * dOut, 204);
            float[] b2Data = generateDeterministicFloats(dOut, 205);

            float[] refData = cpuReferenceMlp(xData, w1Data, b1Data, w2Data, b2Data, m, dIn, h, dOut, Activation.SILU);

            try (Tensor x = Tensor.fromArray(xData, m, dIn);
                 Tensor w1 = Tensor.fromArray(w1Data, dIn, h);
                 Tensor b1 = Tensor.fromArray(b1Data, h);
                 Tensor w2 = Tensor.fromArray(w2Data, h, dOut);
                 Tensor b2 = Tensor.fromArray(b2Data, dOut)) {

                // Active ranks 0 and 3, pruned ranks 1 and 2
                WorkloadPlan plan = new WorkloadPlan(
                        h,
                        List.of(new ShardAllocation(0, 16, 0.5, 0.0, 0L, false),
                                new ShardAllocation(1, 0, 0.0, 0.0, 0L, true),
                                new ShardAllocation(2, 0, 0.0, 0.0, 0L, true),
                                new ShardAllocation(3, 16, 0.5, 0.0, 0L, false)),
                        List.of(0, 3),
                        0.0, 0.0, 0.0
                );

                DistributedResult<Tensor> result = DistributedExecutor.transformerMlp(
                        ring, x, w1, b1, w2, b2, Activation.SILU, plan);

                try (Tensor out = result.value()) {
                    TestAssertions.assertArrayClose(refData, out.toFloatArray(), 1e-4f, 1e-4f);

                    ExecutionReport report = result.report();
                    assertEquals(List.of(0, 3), report.executedGpuRanks());
                    assertEquals(List.of(1, 2), report.prunedGpuRanks());
                }
            }
        }
    }

    @Test
    @DisplayName("Distributed Transformer MLP: Null biases and RELU activation")
    void testDistributedTransformerMlpNullBiases() {
        try (Collective ring = Collective.open(2)) {
            int m = 8;
            int dIn = 16;
            int h = 32;
            int dOut = 16;

            float[] xData = generateDeterministicFloats(m * dIn, 301);
            float[] w1Data = generateDeterministicFloats(dIn * h, 302);
            float[] w2Data = generateDeterministicFloats(h * dOut, 304);

            float[] refData = cpuReferenceMlp(xData, w1Data, null, w2Data, null, m, dIn, h, dOut, Activation.RELU);

            try (Tensor x = Tensor.fromArray(xData, m, dIn);
                 Tensor w1 = Tensor.fromArray(w1Data, dIn, h);
                 Tensor w2 = Tensor.fromArray(w2Data, h, dOut)) {

                WorkloadPlan plan = new WorkloadPlan(
                        h,
                        List.of(new ShardAllocation(0, 16, 0.5, 0.0, 0L, false),
                                new ShardAllocation(1, 16, 0.5, 0.0, 0L, false)),
                        List.of(0, 1),
                        0.0, 0.0, 0.0
                );

                DistributedResult<Tensor> result = DistributedExecutor.transformerMlp(
                        ring, x, w1, null, w2, null, Activation.RELU, plan);

                try (Tensor out = result.value()) {
                    TestAssertions.assertArrayClose(refData, out.toFloatArray(), 1e-4f, 1e-4f);
                }
            }
        }
    }

    @Test
    @DisplayName("Distributed Transformer MLP: Hybrid GPU + CPU-in-place execution")
    void testDistributedTransformerMlpHybridCpu() {
        try (Collective ring = Collective.open(2)) {
            int m = 8;
            int dIn = 16;
            int h = 32;
            int dOut = 16;

            float[] xData = generateDeterministicFloats(m * dIn, 401);
            float[] w1Data = generateDeterministicFloats(dIn * h, 402);
            float[] b1Data = generateDeterministicFloats(h, 403);
            float[] w2Data = generateDeterministicFloats(h * dOut, 404);
            float[] b2Data = generateDeterministicFloats(dOut, 405);

            float[] refData = cpuReferenceMlp(xData, w1Data, b1Data, w2Data, b2Data, m, dIn, h, dOut, Activation.GELU);

            try (Tensor x = Tensor.fromArray(xData, m, dIn);
                 Tensor w1 = Tensor.fromArray(w1Data, dIn, h);
                 Tensor b1 = Tensor.fromArray(b1Data, h);
                 Tensor w2 = Tensor.fromArray(w2Data, h, dOut);
                 Tensor b2 = Tensor.fromArray(b2Data, dOut)) {

                // 24 hidden units on GPU rank 0, 8 hidden units on Host CPU
                WorkloadPlan plan = new WorkloadPlan(
                        h,
                        List.of(new ShardAllocation(0, 24, 0.75, 0.0, 0L, false),
                                new ShardAllocation(ShardAllocation.HOST_CPU_RANK, 8, 0.25, 0.0, 0L, false)),
                        List.of(0),
                        0.0, 0.0, 0.0
                );

                DistributedResult<Tensor> result = DistributedExecutor.transformerMlp(
                        ring, x, w1, b1, w2, b2, Activation.GELU, plan);

                try (Tensor out = result.value()) {
                    TestAssertions.assertArrayClose(refData, out.toFloatArray(), 1e-4f, 1e-4f);
                    assertEquals(8L, result.report().cpuInPlaceUnits());
                }
            }
        }
    }

    // --- Workload 3: Lossless Zero-Waste Split-KV Decode Attention ---

    @Test
    @DisplayName("Lossless Split-KV: GQA partitioned across 2 GPU ranks matches pure CPU reference (<= 1e-4f)")
    void testLosslessSplitKvAttentionGqaTwoGpus() {
        try (Collective ring = Collective.open(2)) {
            int numHeads = 8;
            int numKvHeads = 2;
            int headDim = 64;
            int totalTokens = 513;
            float scale = (float) (1.0 / Math.sqrt(headDim));

            float[] qData = generateDeterministicFloats(numHeads * headDim, 501);
            float[] kData = generateDeterministicFloats(numKvHeads * totalTokens * headDim, 502);
            float[] vData = generateDeterministicFloats(numKvHeads * totalTokens * headDim, 503);

            float[] cpuRef = cpuReferenceAttention(qData, kData, vData, numHeads, numKvHeads, headDim, totalTokens, scale);

            try (Tensor q = Tensor.fromArray(qData, numHeads, headDim);
                 Tensor kCache = Tensor.fromArray(kData, numKvHeads, totalTokens, headDim);
                 Tensor vCache = Tensor.fromArray(vData, numKvHeads, totalTokens, headDim)) {

                // Partition 513 tokens: 256 tokens on rank 0, 257 tokens on rank 1
                RolePlan rolePlan = new RolePlan(
                        new WorkloadPlan(0, List.of(), List.of(), 0.0, 0.0, 0.0),
                        List.of(new SplitKvAssignment(0, DeviceRole.TENSOR_PARALLEL, 256, 0.0, 0.0, 0.0, 0L),
                                new SplitKvAssignment(1, DeviceRole.SPLIT_KV_RECLAIMED, 257, 0.0, 0.0, 0.0, 0L)),
                        0.0, 0.0, totalTokens, 1, false, "SplitKvReclaimed"
                );

                DistributedResult<Tensor> result = DistributedExecutor.splitKvAttention(ring, q, kCache, vCache, scale, rolePlan);

                try (Tensor out = result.value()) {
                    TestAssertions.assertArrayClose(cpuRef, out.toFloatArray(), 1e-4f, 1e-4f);
                }
            }
        }
    }

    @Test
    @DisplayName("Lossless Split-KV: MHA partitioned across 2 GPU ranks matches pure CPU reference (<= 1e-4f)")
    void testLosslessSplitKvAttentionMha() {
        try (Collective ring = Collective.open(2)) {
            int numHeads = 4;
            int numKvHeads = 4;
            int headDim = 64;
            int totalTokens = 300;
            float scale = (float) (1.0 / Math.sqrt(headDim));

            float[] qData = generateDeterministicFloats(numHeads * headDim, 601);
            float[] kData = generateDeterministicFloats(numKvHeads * totalTokens * headDim, 602);
            float[] vData = generateDeterministicFloats(numKvHeads * totalTokens * headDim, 603);

            float[] cpuRef = cpuReferenceAttention(qData, kData, vData, numHeads, numKvHeads, headDim, totalTokens, scale);

            try (Tensor q = Tensor.fromArray(qData, numHeads, headDim);
                 Tensor kCache = Tensor.fromArray(kData, numKvHeads, totalTokens, headDim);
                 Tensor vCache = Tensor.fromArray(vData, numKvHeads, totalTokens, headDim)) {

                RolePlan rolePlan = new RolePlan(
                        new WorkloadPlan(0, List.of(), List.of(), 0.0, 0.0, 0.0),
                        List.of(new SplitKvAssignment(0, DeviceRole.TENSOR_PARALLEL, 150, 0.0, 0.0, 0.0, 0L),
                                new SplitKvAssignment(1, DeviceRole.SPLIT_KV_RECLAIMED, 150, 0.0, 0.0, 0.0, 0L)),
                        0.0, 0.0, totalTokens, 1, false, "SplitKvMha"
                );

                DistributedResult<Tensor> result = DistributedExecutor.splitKvAttention(ring, q, kCache, vCache, scale, rolePlan);

                try (Tensor out = result.value()) {
                    TestAssertions.assertArrayClose(cpuRef, out.toFloatArray(), 1e-4f, 1e-4f);
                }
            }
        }
    }

    @Test
    @DisplayName("Lossless Split-KV: 3-way hybrid (GPU 0 + GPU 1 reclaimed + Host CPU in-place)")
    void testLosslessSplitKvAttentionThreeWayHybrid() {
        try (Collective ring = Collective.open(2)) {
            int numHeads = 8;
            int numKvHeads = 2;
            int headDim = 64;
            int totalTokens = 513;
            float scale = (float) (1.0 / Math.sqrt(headDim));

            float[] qData = generateDeterministicFloats(numHeads * headDim, 701);
            float[] kData = generateDeterministicFloats(numKvHeads * totalTokens * headDim, 702);
            float[] vData = generateDeterministicFloats(numKvHeads * totalTokens * headDim, 703);

            float[] cpuRef = cpuReferenceAttention(qData, kData, vData, numHeads, numKvHeads, headDim, totalTokens, scale);

            try (Tensor q = Tensor.fromArray(qData, numHeads, headDim);
                 Tensor kCache = Tensor.fromArray(kData, numKvHeads, totalTokens, headDim);
                 Tensor vCache = Tensor.fromArray(vData, numKvHeads, totalTokens, headDim)) {

                // 200 on GPU 0, 200 on GPU 1, 113 on Host CPU
                RolePlan rolePlan = new RolePlan(
                        new WorkloadPlan(0, List.of(), List.of(), 0.0, 0.0, 0.0),
                        List.of(new SplitKvAssignment(0, DeviceRole.TENSOR_PARALLEL, 200, 0.0, 0.0, 0.0, 0L),
                                new SplitKvAssignment(1, DeviceRole.SPLIT_KV_RECLAIMED, 200, 0.0, 0.0, 0.0, 0L),
                                new SplitKvAssignment(ShardAllocation.HOST_CPU_RANK, DeviceRole.CPU_IN_PLACE, 113, 0.0, 0.0, 0.0, 0L)),
                        0.0, 0.0, totalTokens, 1, true, "ThreeWayHybrid"
                );

                DistributedResult<Tensor> result = DistributedExecutor.splitKvAttention(ring, q, kCache, vCache, scale, rolePlan);

                try (Tensor out = result.value()) {
                    TestAssertions.assertArrayClose(cpuRef, out.toFloatArray(), 1e-4f, 1e-4f);
                    assertEquals(113L, result.report().cpuInPlaceUnits());
                }
            }
        }
    }

    @Test
    @DisplayName("SplitKvSession: Multi-token decode query sequence without KV re-upload")
    void testSplitKvSessionMultiTokenDecode() {
        try (Collective ring = Collective.open(2)) {
            int numHeads = 8;
            int numKvHeads = 2;
            int headDim = 64;
            int totalTokens = 513;
            float scale = (float) (1.0 / Math.sqrt(headDim));

            float[] kData = generateDeterministicFloats(numKvHeads * totalTokens * headDim, 801);
            float[] vData = generateDeterministicFloats(numKvHeads * totalTokens * headDim, 802);

            try (Tensor kCache = Tensor.fromArray(kData, numKvHeads, totalTokens, headDim);
                 Tensor vCache = Tensor.fromArray(vData, numKvHeads, totalTokens, headDim)) {

                RolePlan rolePlan = new RolePlan(
                        new WorkloadPlan(0, List.of(), List.of(), 0.0, 0.0, 0.0),
                        List.of(new SplitKvAssignment(0, DeviceRole.TENSOR_PARALLEL, 256, 0.0, 0.0, 0.0, 0L),
                                new SplitKvAssignment(1, DeviceRole.SPLIT_KV_RECLAIMED, 257, 0.0, 0.0, 0.0, 0L)),
                        0.0, 0.0, totalTokens, 1, false, "MultiTokenSession"
                );

                try (SplitKvSession session = SplitKvSession.open(ring, rolePlan, numHeads, numKvHeads, headDim, kCache, vCache)) {
                    for (int tokenIdx = 0; tokenIdx < 5; tokenIdx++) {
                        float[] qData = generateDeterministicFloats(numHeads * headDim, 900 + tokenIdx);
                        float[] cpuRef = cpuReferenceAttention(qData, kData, vData, numHeads, numKvHeads, headDim, totalTokens, scale);

                        try (Tensor q = Tensor.fromArray(qData, numHeads, headDim)) {
                            DistributedResult<Tensor> result = session.attend(q, scale);
                            try (Tensor out = result.value()) {
                                TestAssertions.assertArrayClose(cpuRef, out.toFloatArray(), 1e-4f, 1e-4f);
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Helper Utilities ---

    private static float[] generateDeterministicFloats(int size, long seed) {
        Random rng = new Random(seed);
        float[] arr = new float[size];
        for (int i = 0; i < size; i++) {
            arr[i] = (rng.nextFloat() - 0.5f) * 0.2f;
        }
        return arr;
    }

    private static float[] cpuReferenceAttention(float[] q, float[] k, float[] v, int numHeads,
                                                 int numKvHeads, int headDim, int tokens, float scale) {
        int groupSize = numHeads / numKvHeads;
        float[] out = new float[numHeads * headDim];

        for (int h = 0; h < numHeads; h++) {
            int kvH = h / groupSize;
            double[] scores = new double[tokens];
            double maxScore = -Double.MAX_VALUE;

            for (int t = 0; t < tokens; t++) {
                double dot = 0.0;
                int kOff = (kvH * tokens + t) * headDim;
                int qOff = h * headDim;
                for (int d = 0; d < headDim; d++) {
                    dot += (double) q[qOff + d] * (double) k[kOff + d];
                }
                double s = dot * (double) scale;
                scores[t] = s;
                if (s > maxScore) {
                    maxScore = s;
                }
            }

            double sumExp = 0.0;
            for (int t = 0; t < tokens; t++) {
                scores[t] = Math.exp(scores[t] - maxScore);
                sumExp += scores[t];
            }

            for (int d = 0; d < headDim; d++) {
                double acc = 0.0;
                for (int t = 0; t < tokens; t++) {
                    int vOff = (kvH * tokens + t) * headDim + d;
                    acc += scores[t] * (double) v[vOff];
                }
                out[h * headDim + d] = (float) (acc / sumExp);
            }
        }
        return out;
    }

    private static float[] cpuReferenceMlp(float[] x, float[] w1, float[] b1, float[] w2, float[] b2,
                                           int m, int dIn, int h, int dOut, Activation activation) {
        float[] hidden = new float[m * h];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < h; j++) {
                float sum = 0.0f;
                for (int p = 0; p < dIn; p++) {
                    sum += x[i * dIn + p] * w1[p * h + j];
                }
                if (b1 != null) {
                    sum += b1[j];
                }
                hidden[i * h + j] = applyActivation(sum, activation);
            }
        }
        float[] out = new float[m * dOut];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < dOut; j++) {
                float sum = 0.0f;
                for (int p = 0; p < h; p++) {
                    sum += hidden[i * h + p] * w2[p * dOut + j];
                }
                if (b2 != null) {
                    sum += b2[j];
                }
                out[i * dOut + j] = sum;
            }
        }
        return out;
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
            default -> throw new UnsupportedOperationException("Unsupported activation: " + activation);
        };
    }
}
