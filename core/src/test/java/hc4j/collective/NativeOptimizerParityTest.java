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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import hc4j.internal.collective.CpuFallbackKernels;
import hc4j.internal.collective.NativeOptimizerBridge;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Cross-language Rust-vs-Java parity tests ensuring {@code hc4rs::collective::planner}
 * (via {@code hc4j_optimizer_plan_work}, {@code hc4j_optimizer_plan_roles},
 * {@code hc4j_collective_merge_split_kv_f32}, and {@code hc4j_last_error}) produces identical
 * plans and diagnostics to the Java reference fallback across all hardware regimes.
 */
@DisplayName("Cross-Language Rust-vs-Java Optimizer Parity Tests")
class NativeOptimizerParityTest {

    private static final double EPS = 1e-9;

    private static DeviceProfile gpu(int rank, String name, String type, long vramGb, double gflops) {
        return new DeviceProfile(rank, name, type, 0x10DE, 0x2684 + rank, vramGb << 30, gflops);
    }

    private static List<Link> fullMesh(int n, double bwGbps, double latUs, Transport tr) {
        List<Link> links = new ArrayList<>(n * n);
        for (int s = 0; s < n; s++) {
            for (int d = 0; d < n; d++) {
                links.add(new Link(s, d, s == d ? 25.0 : bwGbps, s == d ? Transport.PEER_TO_PEER : tr, s == d, s == d ? 0.0 : latUs));
            }
        }
        return links;
    }

    private static void assertPlansEqual(String label, WorkloadPlan expectedJava, WorkloadPlan actualNative) {
        assertEquals(expectedJava.totalUnits(), actualNative.totalUnits(), label + ": totalUnits");
        assertEquals(expectedJava.activeRanks(), actualNative.activeRanks(), label + ": activeRanks");
        assertEquals(expectedJava.estimatedMakespanMs(), actualNative.estimatedMakespanMs(), 1e-6, label + ": makespan");
        assertEquals(expectedJava.estimatedComputeMs(), actualNative.estimatedComputeMs(), 1e-6, label + ": computeMs");
        assertEquals(expectedJava.estimatedSyncMs(), actualNative.estimatedSyncMs(), 1e-6, label + ": syncMs");
        assertEquals(expectedJava.allocations().size(), actualNative.allocations().size(), label + ": allocation count");

        for (int i = 0; i < expectedJava.allocations().size(); i++) {
            ShardAllocation ej = expectedJava.allocations().get(i);
            ShardAllocation an = actualNative.allocations().get(i);
            assertEquals(ej.rank(), an.rank(), label + " rank[" + i + "]");
            assertEquals(ej.units(), an.units(), label + " units[" + i + "]");
            assertEquals(ej.excludedAsStraggler(), an.excludedAsStraggler(), label + " excluded[" + i + "]");
            assertEquals(ej.vramUnits(), an.vramUnits(), label + " vramUnits[" + i + "]");
            assertEquals(ej.hostStreamUnits(), an.hostStreamUnits(), label + " hostStreamUnits[" + i + "]");
            assertEquals(ej.diskStreamUnits(), an.diskStreamUnits(), label + " diskStreamUnits[" + i + "]");
            assertEquals(ej.cpuInPlaceUnits(), an.cpuInPlaceUnits(), label + " cpuInPlaceUnits[" + i + "]");
            assertEquals(ej.estimatedComputeMs(), an.estimatedComputeMs(), 1e-6, label + " estComputeMs[" + i + "]");
            assertEquals(ej.estimatedMemoryBytes(), an.estimatedMemoryBytes(), label + " estMemBytes[" + i + "]");
            assertEquals(ej.shareRatio(), an.shareRatio(), EPS, label + " shareRatio[" + i + "]");
        }
    }

    private static void assertRolePlansEqual(String label, RolePlan expectedJava, RolePlan actualNative) {
        assertPlansEqual(label + " [tpPlan]", expectedJava.tpPlan(), actualNative.tpPlan());
        assertEquals(expectedJava.estimatedTpMs(), actualNative.estimatedTpMs(), 1e-6, label + ": estimatedTpMs");
        assertEquals(expectedJava.estimatedAttentionMs(), actualNative.estimatedAttentionMs(), 1e-6, label + ": estimatedAttentionMs");
        assertEquals(expectedJava.totalKvTokens(), actualNative.totalKvTokens(), label + ": totalKvTokens");
        assertEquals(expectedJava.reclaimedGpuCount(), actualNative.reclaimedGpuCount(), label + ": reclaimedGpuCount");
        assertEquals(expectedJava.cpuReclaimed(), actualNative.cpuReclaimed(), label + ": cpuReclaimed");
        assertEquals(expectedJava.activeRegime(), actualNative.activeRegime(), label + ": activeRegime");
        assertEquals(expectedJava.splitKvAssignments().size(), actualNative.splitKvAssignments().size(), label + ": kv assignment count");

        for (int i = 0; i < expectedJava.splitKvAssignments().size(); i++) {
            SplitKvAssignment ej = expectedJava.splitKvAssignments().get(i);
            SplitKvAssignment an = actualNative.splitKvAssignments().get(i);
            assertEquals(ej.rank(), an.rank(), label + " kv rank[" + i + "]");
            assertEquals(ej.role(), an.role(), label + " kv role[" + i + "]");
            assertEquals(ej.kvTokens(), an.kvTokens(), label + " kvTokens[" + i + "]");
            assertEquals(ej.estimatedLocalAttnMs(), an.estimatedLocalAttnMs(), 1e-6, label + " localAttnMs[" + i + "]");
            assertEquals(ej.estimatedCommMs(), an.estimatedCommMs(), 1e-6, label + " commMs[" + i + "]");
            assertEquals(ej.estimatedTotalMs(), an.estimatedTotalMs(), 1e-6, label + " totalMs[" + i + "]");
            assertEquals(ej.kvMemoryBytes(), an.kvMemoryBytes(), label + " kvMemoryBytes[" + i + "]");
        }
    }

    @Test
    @DisplayName("Homogeneous 4-GPU cluster parity")
    void testHomogeneous4GpuParity() {
        Assumptions.assumeTrue(NativeOptimizerBridge.isNativeAvailable(), "Native optimizer required");

        List<DeviceProfile> devices = new ArrayList<>();
        for (int r = 0; r < 4; r++) {
            devices.add(gpu(r, "GPU-" + r, "Discrete", 16, 40_000.0));
        }
        List<Link> links = fullMesh(4, 24.0, 4.0, Transport.PEER_TO_PEER);
        WorkloadSpec spec = WorkloadSpec.builder(8192, 2e9, 4096).syncBytes(32L << 20).microBatches(4).build();

        assertPlansEqual("homogeneous-4gpu",
                WorkloadPlanner.planJavaFallback(devices, links, spec),
                WorkloadPlanner.planNative(devices, links, spec));
    }

    @Test
    @DisplayName("10x RTX 5090 + 1x UHD 620 straggler pruning parity")
    void testTen5090PlusUhd620StragglerParity() {
        Assumptions.assumeTrue(NativeOptimizerBridge.isNativeAvailable(), "Native optimizer required");

        List<DeviceProfile> devices = new ArrayList<>(11);
        for (int r = 0; r < 10; r++) {
            devices.add(gpu(r, "RTX-5090-" + r, "Discrete", 32, 104_000.0));
        }
        devices.add(gpu(10, "Intel UHD 620", "Integrated", 4, 250.0));

        List<Link> links = new ArrayList<>();
        for (int s = 0; s < 11; s++) {
            for (int d = 0; d < 11; d++) {
                if (s == d) {
                    links.add(new Link(s, d, 25.0, Transport.PEER_TO_PEER, true, 0.0));
                } else if (s < 10 && d < 10) {
                    links.add(new Link(s, d, 28.0, Transport.PEER_TO_PEER, false, 3.0));
                } else {
                    links.add(new Link(s, d, 1.0, Transport.HOST_STAGED, false, 150.0));
                }
            }
        }

        WorkloadSpec spec = WorkloadSpec.builder(10_240, 1.5e9, 4096).syncBytes(64L << 20).build();
        WorkloadPlan javaPlan = WorkloadPlanner.planJavaFallback(devices, links, spec);
        WorkloadPlan nativePlan = WorkloadPlanner.planNative(devices, links, spec);

        assertPlansEqual("10x5090+UHD620", javaPlan, nativePlan);
        assertTrue(nativePlan.allocations().get(10).excludedAsStraggler());
        assertFalse(nativePlan.activeRanks().contains(10));
    }

    @Test
    @DisplayName("Full N=12 multi-vendor asymmetric cluster parity")
    void testMultiVendor12GpuAsymmetricParity() {
        Assumptions.assumeTrue(NativeOptimizerBridge.isNativeAvailable(), "Native optimizer required");

        List<DeviceProfile> devices = new ArrayList<>(12);
        for (int r = 0; r < 4; r++) {
            devices.add(gpu(r, "NVIDIA-Island-" + r, "Discrete", 24, 60_000.0 + r * 1_000.0));
        }
        for (int r = 4; r < 8; r++) {
            devices.add(new DeviceProfile(r, "AMD-Island-" + r, "Discrete", 0x1002, 0x744C, 16L << 30, 35_000.0 + r * 500.0));
        }
        for (int r = 8; r < 12; r++) {
            devices.add(new DeviceProfile(r, "Intel-iGPU-" + r, "Integrated", 0x8086, 0x9A49, 8L << 30, 2_000.0 + r * 100.0));
        }

        List<Link> links = new ArrayList<>(144);
        for (int s = 0; s < 12; s++) {
            for (int d = 0; d < 12; d++) {
                boolean sameIsland = (s / 4) == (d / 4) && (s < 8);
                double bw = (s == d) ? 30.0 : (sameIsland ? 24.0 : 1.5);
                Transport tr = (s == d || sameIsland) ? Transport.PEER_TO_PEER : Transport.HOST_STAGED;
                double lat = (s == d) ? 0.0 : (sameIsland ? 4.0 : 80.0);
                links.add(new Link(s, d, bw, tr, s == d, lat));
            }
        }

        WorkloadSpec spec = WorkloadSpec.builder(12_000, 1e9, 4096).syncBytes(128L << 20).microBatches(2).build();
        assertPlansEqual("12-gpu-multi-vendor",
                WorkloadPlanner.planJavaFallback(devices, links, spec),
                WorkloadPlanner.planNative(devices, links, spec));
    }

    @Test
    @DisplayName("Four regimes (ALL_GPU_SHARD, STRAGGLER_PRUNED, PCIE_STREAMING, CPU_IN_PLACE) & VRAM-cliff parity")
    void testFourRegimesAndVramCliffParity() {
        Assumptions.assumeTrue(NativeOptimizerBridge.isNativeAvailable(), "Native optimizer required");

        // Regime 1 & VRAM-cliff integer minimax: unequal VRAM cliffs on 2 fast GPUs
        List<DeviceProfile> cliffDevs = List.of(
                gpu(0, "Cliff-GPU0", "Discrete", 1, 50_000.0),
                gpu(1, "Cliff-GPU1", "Discrete", 4, 25_000.0));
        List<Link> cliffLinks = fullMesh(2, 20.0, 5.0, Transport.PEER_TO_PEER);
        WorkloadSpec allGpuSpec = WorkloadSpec.builder(1000, 5e8, 1L << 20).syncBytes(1L << 20).build();
        assertPlansEqual("vram-cliff-all-gpu",
                WorkloadPlanner.planJavaFallback(cliffDevs, cliffLinks, allGpuSpec),
                WorkloadPlanner.planNative(cliffDevs, cliffLinks, allGpuSpec));

        // Exceed usable 0.85 GB on GPU0 -> hits VRAM cliff then pushes to GPU1!
        WorkloadSpec exceedCliff = WorkloadSpec.builder(2500, 5e8, 1L << 20).syncBytes(1L << 20).build();
        assertPlansEqual("vram-cliff-heterogeneous-shift",
                WorkloadPlanner.planJavaFallback(cliffDevs, cliffLinks, exceedCliff),
                WorkloadPlanner.planNative(cliffDevs, cliffLinks, exceedCliff));

        // PCIE_STREAMING + Host CPU in-place + NVMe disk tier
        List<DeviceProfile> singleGpu = List.of(gpu(0, "Small-GPU", "Discrete", 1, 20_000.0));
        WorkloadSpec spillCpuSpec = WorkloadSpec.builder(3200, 5e8, 1L << 20)
                .diskBandwidthGbps(4.0)
                .diskCapacityBytes(16L << 30)
                .hostCpu(HostCpuProfile.of("Ryzen-9", 8L << 30, 500.0, 80.0, 16.0))
                .build();
        WorkloadPlan javaSpill = WorkloadPlanner.planJavaFallback(singleGpu, List.of(), spillCpuSpec);
        WorkloadPlan nativeSpill = WorkloadPlanner.planNative(singleGpu, List.of(), spillCpuSpec);
        assertPlansEqual("3-tier-spill+cpu", javaSpill, nativeSpill);
        assertTrue(nativeSpill.allocations().stream().anyMatch(a -> a.hostStreamUnits() > 0));
        assertTrue(nativeSpill.allocations().stream().anyMatch(a -> a.rank() == ShardAllocation.HOST_CPU_RANK && a.cpuInPlaceUnits() > 0));

        // STRAGGLER_PRUNED via RolePlan short context vs SPLIT_KV_RECLAIMED via long context
        List<DeviceProfile> roleDevs = List.of(
                gpu(0, "Primary-GPU", "Discrete", 24, 90_000.0),
                gpu(1, "Straggler-GPU", "Integrated", 16, 400.0));
        List<Link> roleLinks = fullMesh(2, 2.0, 80.0, Transport.HOST_STAGED);
        WorkloadSpec tpSpec = WorkloadSpec.builder(4096, 2e9, 4096).syncBytes(64L << 20).build();

        SplitKvSpec shortKv = new SplitKvSpec(128, 512L << 10, 1e6, 64L << 10, 2L << 30, 0.85);
        RolePlan shortJava = WorkloadPlanner.planRolesJavaFallback(roleDevs, roleLinks, tpSpec, shortKv);
        RolePlan shortNative = WorkloadPlanner.planRolesNative(roleDevs, roleLinks, tpSpec, shortKv);
        assertRolePlansEqual("plan-roles-short-128", shortJava, shortNative);
        assertEquals("STRAGGLER_PRUNED", shortNative.activeRegime());

        SplitKvSpec longKv = new SplitKvSpec(32_768, 512L << 10, 2e7, 64L << 10, 2L << 30, 0.85);
        RolePlan longJava = WorkloadPlanner.planRolesJavaFallback(roleDevs, roleLinks, tpSpec, longKv);
        RolePlan longNative = WorkloadPlanner.planRolesNative(roleDevs, roleLinks, tpSpec, longKv);
        assertRolePlansEqual("plan-roles-long-32k", longJava, longNative);
        assertEquals("SPLIT_KV_RECLAIMED", longNative.activeRegime());
        assertEquals(1, longNative.reclaimedGpuCount());
    }

    @Test
    @DisplayName("mergeSplitKvF32 matches Java online-softmax reference on -50/+50 extreme logits")
    void testMergeSplitKvExtremeLogitsParity() {
        Assumptions.assumeTrue(NativeOptimizerBridge.isNativeAvailable(), "Native Split-KV merge required");

        int numHeads = 4;
        int headDim = 8;
        float[][] partialsO = new float[2][numHeads * headDim];
        float[][] partialsStats = new float[2][numHeads * 2];

        for (int h = 0; h < numHeads; h++) {
            partialsStats[0][2 * h] = -50.0f;
            partialsStats[0][2 * h + 1] = 1.5f;
            partialsStats[1][2 * h] = 50.0f;
            partialsStats[1][2 * h + 1] = 2.5f;
            for (int d = 0; d < headDim; d++) {
                partialsO[0][h * headDim + d] = (h + 1) * 0.25f + d;
                partialsO[1][h * headDim + d] = -((h + 1) * 0.5f + d * 0.1f);
            }
        }

        try (Arena arena = Arena.ofConfined()) {
            long bytes = (long) numHeads * headDim * 4L;
            MemorySegment nativeSeg = arena.allocate(bytes, 16);
            MemorySegment cpuSeg = arena.allocate(bytes, 16);

            NativeOptimizerBridge.mergeSplitKvF32(numHeads, headDim, partialsO, partialsStats, nativeSeg);
            CpuFallbackKernels.mergeSplitKvPartialsCpu(numHeads, headDim, partialsO, partialsStats, cpuSeg);

            for (int i = 0; i < numHeads * headDim; i++) {
                float expected = cpuSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float actual = nativeSeg.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                assertEquals(expected, actual, 1e-6f, "Mismatch at element " + i);
                assertTrue(Float.isFinite(actual), "Merged output must be finite");
            }
        }
    }

    @Test
    @DisplayName("hc4j_last_error enriches native status exceptions cleanly")
    void testNativeLastErrorDiagnosticEnrichment() {
        Assumptions.assumeTrue(NativeOptimizerBridge.isNativeAvailable(), "Native library required");

        List<DeviceProfile> devs = List.of(gpu(0, "Bad-Order", "Discrete", 1, 1000.0));
        WorkloadSpec oomSpec = WorkloadSpec.builder(5_000, 1e8, 1L << 20).syncBytes(0L).build();
        Exception ex = assertThrows(hc4j.engine.GpuOutOfMemoryException.class,
                () -> WorkloadPlanner.planNative(devs, List.of(), oomSpec));
        assertTrue(ex.getMessage().contains("["), "Expected diagnostic suffix from hc4j_last_error, got: " + ex.getMessage());
        assertFalse(ex.getMessage().contains("[]"), "Diagnostic suffix should contain descriptive Rust error message");
    }
}
