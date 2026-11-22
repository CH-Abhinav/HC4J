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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Headless unit tests for Theorems 1–4, 4-regime phase transitions, and zero-waste Split-KV reclamation.
 */
class ExactOptimizerTest {

    private static DeviceProfile makeProfile(int rank, String name, long vramBytes, double gflops) {
        return new DeviceProfile(rank, name, "Discrete", 0x10de, 0x2204, vramBytes, gflops);
    }

    private static List<Link> makeFullyConnectedLinks(int n, double defaultBw, double defaultLatUs, Transport transport) {
        List<Link> links = new ArrayList<>(n * (n - 1));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    links.add(new Link(i, j, defaultBw, transport, false, defaultLatUs));
                }
            }
        }
        return links;
    }

    @Test
    void testTheorem1BruteForceEquivalenceAndSpeed() {
        int n = 6;
        double[][] bw = new double[n][n];
        double[][] latUs = new double[n][n];
        Random rnd = new Random(42);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    bw[i][j] = 2.0 + (rnd.nextDouble() * 20.0);
                    latUs[i][j] = 10.0 + (rnd.nextDouble() * 50.0);
                }
            }
        }

        // Theorem 1 Shared-Root DP
        WorkloadPlanner.AllSubsetsRingTable table = WorkloadPlanner.precomputeAllSubsetsRing(n, bw, latUs);

        // Brute-force verification for all 2^6 subsets with size >= 3
        int numSubsets = 1 << n;
        for (int mask = 1; mask < numSubsets; mask++) {
            int popCount = Integer.bitCount(mask);
            if (popCount < 3) continue;

            List<Integer> nodes = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if ((mask & (1 << i)) != 0) nodes.add(i);
            }

            int root = nodes.getFirst();
            List<Integer> rest = new ArrayList<>(nodes.subList(1, nodes.size()));
            double bruteForceBestBw = -1.0;

            List<List<Integer>> perms = new ArrayList<>();
            generatePermutations(rest, 0, perms);

            for (List<Integer> p : perms) {
                double minBw = bw[root][p.getFirst()];
                for (int idx = 0; idx < p.size() - 1; idx++) {
                    minBw = Math.min(minBw, bw[p.get(idx)][p.get(idx + 1)]);
                }
                minBw = Math.min(minBw, bw[p.getLast()][root]);
                if (minBw > bruteForceBestBw) {
                    bruteForceBestBw = minBw;
                }
            }

            double dpBw = table.bottleneck(mask);
            assertEquals(bruteForceBestBw, dpBw, 1e-9, "DP bottleneck must match brute-force for mask " + mask);
        }

        // Speed check at N = 12
        int n12 = 12;
        List<DeviceProfile> devices12 = new ArrayList<>(n12);
        for (int i = 0; i < n12; i++) {
            devices12.add(makeProfile(i, "GPU-" + i, 16L << 30, 1000.0 + i * 200.0));
        }
        List<Link> links12 = makeFullyConnectedLinks(n12, 16.0, 15.0, Transport.HOST_SHARED_DMA);
        WorkloadSpec spec12 = WorkloadSpec.prefill(10_000, 1e9, 1024, 64 << 20);

        // Warm up
        for (int i = 0; i < 15; i++) {
            WorkloadPlanner.plan(devices12, links12, spec12);
        }

        long startNs = System.nanoTime();
        WorkloadPlan plan12 = WorkloadPlanner.plan(devices12, links12, spec12);
        long elapsedMs = (System.nanoTime() - startNs) / 1_000_000;
        System.out.println("[HC4J] N=12 exact optimizer solve time: " + elapsedMs + " ms");

        assertNotNull(plan12);
        assertTrue(elapsedMs < 50, "N=12 exact optimizer solve time must be < 50 ms, was " + elapsedMs + " ms");
    }

    private static void generatePermutations(List<Integer> list, int k, List<List<Integer>> out) {
        for (int i = k; i < list.size(); i++) {
            Collections.swap(list, i, k);
            generatePermutations(list, k + 1, out);
            Collections.swap(list, k, i);
        }
        if (k == list.size() - 1) {
            out.add(new ArrayList<>(list));
        }
    }

    @Test
    void testTheorem2HomogeneousZeroFalseSplit() {
        int n = 4;
        List<DeviceProfile> devices = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            devices.add(makeProfile(i, "GPU-" + i, 16L << 30, 2000.0));
        }

        double[] bws = {24.8, 25.0, 25.1, 25.2};
        List<Link> links = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    double bw = bws[(i + j) % bws.length];
                    links.add(new Link(i, j, bw, Transport.PEER_TO_PEER, false, 5.0));
                }
            }
        }

        WorkloadSpec spec = WorkloadSpec.prefill(20_000, 1e9, 1024, 64 << 20);
        WorkloadPlan plan = WorkloadPlanner.plan(devices, links, spec);

        assertNotNull(plan);
        assertEquals(4, plan.activeRanks().size(), "Homogeneous cluster must activate all 4 GPUs without false splits");
        for (ShardAllocation alloc : plan.allocations()) {
            assertFalse(alloc.excludedAsStraggler());
            assertTrue(alloc.units() > 0);
        }
    }

    @Test
    void testTheorem2HeterogeneousClusterMultiIsland() {
        // 5 NVIDIA (25 GB/s), 4 AMD (22 GB/s), 3 Intel (18 GB/s), cross-vendor (2.0 GB/s)
        int n = 12;
        List<DeviceProfile> devices = new ArrayList<>(n);
        for (int i = 0; i < 5; i++) devices.add(makeProfile(i, "NV-" + i, 24L << 30, 2500.0));
        for (int i = 5; i < 9; i++) devices.add(makeProfile(i, "AMD-" + i, 20L << 30, 2200.0));
        for (int i = 9; i < 12; i++) devices.add(makeProfile(i, "INTEL-" + i, 16L << 30, 1800.0));

        List<Link> links = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    int gI = (i < 5) ? 0 : (i < 9 ? 1 : 2);
                    int gJ = (j < 5) ? 0 : (j < 9 ? 1 : 2);
                    double bw;
                    Transport tr;
                    if (gI == gJ) {
                        tr = Transport.PEER_TO_PEER;
                        bw = (gI == 0) ? 25.0 : (gI == 1 ? 22.0 : 18.0);
                    } else {
                        tr = Transport.HOST_SHARED_DMA;
                        bw = 2.0;
                    }
                    links.add(new Link(i, j, bw, tr, false, 10.0));
                }
            }
        }

        WorkloadSpec spec = WorkloadSpec.prefill(24_000, 1e9, 1024, 64 << 20);
        WorkloadPlan plan = WorkloadPlanner.plan(devices, links, spec);

        assertNotNull(plan);
        assertTrue(plan.estimatedSyncMs() > 0.0);
        // Flat ring with 12 ranks at 2 GB/s takes > 60 ms. Hierarchical sync takes ~50 ms.
        assertTrue(plan.estimatedSyncMs() < 55.0, "Hierarchical sync must strictly beat flat ring (expected < 55 ms, was " + plan.estimatedSyncMs() + ")");
    }

    @Test
    void testTheorem3VramCliffExactIntegerMinimaxOptimality() {
        int n = 2;
        long bytesPerUnit = 1L << 30; // 1 GiB per unit
        long vramGpu0 = (long) Math.ceil((10L * bytesPerUnit) / 0.85);
        long vramGpu1 = (long) Math.ceil((50L * bytesPerUnit) / 0.85);

        List<DeviceProfile> devices = List.of(
                makeProfile(0, "GPU-0", vramGpu0, 1000.0),
                makeProfile(1, "GPU-1", vramGpu1, 1000.0));
        List<Link> links = makeFullyConnectedLinks(n, 25.0, 5.0, Transport.PEER_TO_PEER);

        WorkloadSpec spec = WorkloadSpec.builder(21, 1e9, bytesPerUnit)
                .vramUtilization(0.85)
                .defaultPcieGbps(1.0) // very slow PCIe to make cliff steep
                .syncBytes(0)
                .build();

        WorkloadPlan plan = WorkloadPlanner.plan(devices, links, spec);
        ShardAllocation alloc0 = plan.allocations().get(0);
        ShardAllocation alloc1 = plan.allocations().get(1);

        assertEquals(10L, alloc0.units(), "GPU 0 should be capped at exact 10 units to avoid VRAM cliff");
        assertEquals(11L, alloc1.units(), "GPU 1 should absorb the 11th unit in VRAM");
        assertEquals(0L, alloc0.hostStreamUnits(), "GPU 0 should have 0 spilled units");
    }

    @Test
    void testTheorem4ZeroDegradationMonotonicity() {
        Random rnd = new Random(12345);
        for (int run = 0; run < 10; run++) {
            int nFast = 3;
            List<DeviceProfile> fastDevices = new ArrayList<>();
            for (int i = 0; i < nFast; i++) {
                fastDevices.add(makeProfile(i, "Fast-" + i, 16L << 30, 2500.0 + rnd.nextDouble() * 500.0));
            }
            List<Link> fastLinks = makeFullyConnectedLinks(nFast, 25.0, 5.0, Transport.PEER_TO_PEER);
            WorkloadSpec spec = WorkloadSpec.prefill(10_000, 1e9, 1024, 64 << 20);

            WorkloadPlan fastPlan = WorkloadPlanner.plan(fastDevices, fastLinks, spec);

            // Add 2 paired stragglers with slow links and low compute
            List<DeviceProfile> clusterWithStragglers = new ArrayList<>(fastDevices);
            clusterWithStragglers.add(makeProfile(3, "Straggler-1", 4L << 30, 100.0));
            clusterWithStragglers.add(makeProfile(4, "Straggler-2", 4L << 30, 100.0));

            List<Link> clusterLinks = new ArrayList<>();
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    if (i != j) {
                        double bw = (i < 3 && j < 3) ? 25.0 : 0.5;
                        Transport tr = (i < 3 && j < 3) ? Transport.PEER_TO_PEER : Transport.HOST_STAGED;
                        double lat = (i < 3 && j < 3) ? 5.0 : 100.0;
                        clusterLinks.add(new Link(i, j, bw, tr, false, lat));
                    }
                }
            }

            WorkloadPlan stragglerPlan = WorkloadPlanner.plan(clusterWithStragglers, clusterLinks, spec);

            assertTrue(
                    stragglerPlan.estimatedMakespanMs() <= fastPlan.estimatedMakespanMs() + 1e-6,
                    String.format("Makespan with stragglers (%.2f ms) must not degrade baseline (%.2f ms)",
                            stragglerPlan.estimatedMakespanMs(), fastPlan.estimatedMakespanMs()));
        }
    }

    @Test
    void testFourRegimePhaseTransitions() {
        // Regime 1: ALL_GPU_SHARD
        List<DeviceProfile> dev2 = List.of(
                makeProfile(0, "G0", 16L << 30, 2000.0),
                makeProfile(1, "G1", 16L << 30, 2000.0));
        List<Link> links2 = makeFullyConnectedLinks(2, 25.0, 5.0, Transport.PEER_TO_PEER);
        WorkloadSpec spec1 = WorkloadSpec.gpuOnly(1000, 1e9, 1024, 0);
        RolePlan rolePlan1 = WorkloadPlanner.planRoles(dev2, links2, spec1, new SplitKvSpec(0, 1024, 1e6, 4096, 0, 0.85));
        assertEquals("ALL_GPU_SHARD", rolePlan1.activeRegime());

        // Regime 2: STRAGGLER_PRUNED
        List<DeviceProfile> devStraggler = List.of(
                makeProfile(0, "Fast0", 16L << 30, 2000.0),
                makeProfile(1, "Fast1", 16L << 30, 2000.0),
                makeProfile(2, "Slow2", 4L << 30, 50.0));
        List<Link> linksStraggler = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i != j) {
                    double bw = (i < 2 && j < 2) ? 25.0 : 0.5;
                    linksStraggler.add(new Link(i, j, bw, Transport.HOST_STAGED, false, 100.0));
                }
            }
        }
        WorkloadSpec spec2 = WorkloadSpec.prefill(10_000, 1e9, 1024, 128 << 20);
        RolePlan rolePlan2 = WorkloadPlanner.planRoles(devStraggler, linksStraggler, spec2, new SplitKvSpec(0, 1024, 1e6, 4096, 0, 0.85));
        assertEquals("STRAGGLER_PRUNED", rolePlan2.activeRegime());

        // Regime 3: PCIE_STREAMING (Prefill)
        long vram8G = 8L << 30;
        long bytesUnit = 1L << 28; // 256 MiB per unit
        List<DeviceProfile> devPrefill = List.of(
                makeProfile(0, "FastGPU", vram8G, 3000.0),
                makeProfile(1, "SlowGPU", 4L << 30, 100.0));
        List<Link> linksPrefill = makeFullyConnectedLinks(2, 1.0, 100.0, Transport.HOST_STAGED);
        HostCpuProfile hostCpu = HostCpuProfile.of("HostCPU", 32L << 30, 50.0, 40.0, 16.0);
        WorkloadSpec spec3 = WorkloadSpec.builder(34, 1e12, bytesUnit) // very high FLOPs per unit (prefill)
                .ioMultiplier(1.0)
                .vramUtilization(0.85)
                .defaultPcieGbps(16.0)
                .syncBytes(64 << 20)
                .hostCpu(hostCpu)
                .build();
        RolePlan rolePlan3 = WorkloadPlanner.planRoles(devPrefill, linksPrefill, spec3, new SplitKvSpec(0, 1024, 1e6, 4096, 0, 0.85));
        assertEquals("PCIE_STREAMING", rolePlan3.activeRegime());
        assertTrue(rolePlan3.tpPlan().allocations().getFirst().hostStreamUnits() > 0);

        // Regime 4: CPU_IN_PLACE (Decode)
        HostCpuProfile ddr5Cpu = HostCpuProfile.of("DDR5-CPU", 64L << 30, 400.0, 80.0, 16.0);
        WorkloadSpec spec4 = WorkloadSpec.builder(34, 2.0 * bytesUnit, bytesUnit) // 2 FLOPs per byte (Decode)
                .ioMultiplier(1.0)
                .vramUtilization(0.85)
                .defaultPcieGbps(16.0)
                .syncBytes(64 << 20)
                .hostCpu(ddr5Cpu)
                .build();
        RolePlan rolePlan4 = WorkloadPlanner.planRoles(devPrefill, linksPrefill, spec4, new SplitKvSpec(0, 1024, 1e6, 4096, 0, 0.85));
        assertEquals("CPU_IN_PLACE", rolePlan4.activeRegime());
        assertTrue(rolePlan4.cpuReclaimed());
    }

    @Test
    void testZeroWasteSplitKvReclamation() {
        // Fast GPU 0 + Pruned Slow GPU 1
        List<DeviceProfile> devices = List.of(
                makeProfile(0, "FastGPU", 16L << 30, 3000.0),
                makeProfile(1, "SlowGPU", 8L << 30, 200.0));
        List<Link> links = makeFullyConnectedLinks(2, 0.2, 50.0, Transport.HOST_STAGED);
        WorkloadSpec tpSpec = WorkloadSpec.prefill(10_000, 1e9, 1024, 128 << 20);

        // Short context: 128 tokens (comm overhead dominates, slow GPU should stay IDLE)
        SplitKvSpec shortKv = new SplitKvSpec(128, 1024, 1e6, 32768, 2L << 30, 0.85);
        RolePlan planShort = WorkloadPlanner.planRoles(devices, links, tpSpec, shortKv);
        SplitKvAssignment slowShort = planShort.splitKvAssignments().get(1);
        assertEquals(DeviceRole.IDLE, slowShort.role(), "Short context should leave pruned GPU IDLE due to RPC overhead");

        // Long context: 32,768 tokens (compute amortizes, slow GPU reclaimed as SPLIT_KV_RECLAIMED)
        SplitKvSpec longKv = new SplitKvSpec(32_768, 1024, 1e6, 32768, 2L << 30, 0.85);
        RolePlan planLong = WorkloadPlanner.planRoles(devices, links, tpSpec, longKv);
        SplitKvAssignment slowLong = planLong.splitKvAssignments().get(1);
        assertEquals(DeviceRole.SPLIT_KV_RECLAIMED, slowLong.role(), "Long context should reclaim pruned GPU for Split-KV");
        assertTrue(slowLong.kvTokens() > 0);
        assertEquals("SPLIT_KV_RECLAIMED", planLong.activeRegime());
    }

    @Test
    void testSharedHostRamAndPcieContention() {
        int n = 2;
        long bytesUnit = 1L << 30; // 1 GiB per unit
        List<DeviceProfile> devices = List.of(
                makeProfile(0, "GPU-0", 4L << 30, 1500.0),
                makeProfile(1, "GPU-1", 4L << 30, 1500.0));
        List<Link> links = makeFullyConnectedLinks(n, 25.0, 5.0, Transport.PEER_TO_PEER);

        HostCpuProfile hostCpu = HostCpuProfile.of("HostCPU", 2L * bytesUnit, 50.0, 40.0, 16.0);
        WorkloadSpec spec = WorkloadSpec.builder(12, 1e9, bytesUnit)
                .vramUtilization(0.85)
                .defaultPcieGbps(16.0)
                .diskBandwidthGbps(4.0)
                .diskCapacityBytes(10L * bytesUnit)
                .hostCpu(hostCpu)
                .build();

        WorkloadPlan plan = WorkloadPlanner.plan(devices, links, spec);
        assertNotNull(plan);

        long totalHostStream = 0;
        long totalDiskStream = 0;
        for (ShardAllocation a : plan.allocations()) {
            if (a.rank() != ShardAllocation.HOST_CPU_RANK) {
                totalHostStream += a.hostStreamUnits();
                totalDiskStream += a.diskStreamUnits();
            }
        }

        assertTrue(totalHostStream <= 2, "Host stream units must not exceed host RAM capacity of 2");
        assertTrue(totalDiskStream > 0, "Overflow units must cleanly spill into Tier 2 disk");
    }
}
