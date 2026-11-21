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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import hc4j.collective.DeviceProfile;
import hc4j.collective.Link;
import hc4j.collective.ShardAllocation;
import hc4j.collective.Transport;
import hc4j.collective.WorkloadPlan;
import hc4j.collective.WorkloadPlanner;
import hc4j.engine.GpuOutOfMemoryException;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Headless unit tests for {@link WorkloadPlanner}:
 * VRAM water-filling, compute-proportional sharding, makespan modeling,
 * straggler pruning, and edge cases.
 */
class WorkloadPlannerTest {

    private static DeviceProfile makeProfile(int rank, String name, String type, long vramBytes, double gflops) {
        return new DeviceProfile(rank, name, type, 0x10de, 0x2900, vramBytes, gflops);
    }

    private static List<Link> buildMeshTopology(int n, double defaultGbps, Transport defaultTransport) {
        List<Link> links = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    links.add(new Link(i, j, defaultGbps, defaultTransport, false));
                }
            }
        }
        return links;
    }

    @Test
    void test10NvidiaPlus1IntelHeavySyncStragglerPruning() {
        // 10x RTX 5090 (25 GB/s P2P, 100,000 GFLOP/s, 32 GiB)
        // 1x Intel UHD 620 (2.0 GB/s HOST_SHARED_DMA, 185 GFLOP/s, 4 GiB)
        List<DeviceProfile> allDevices = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            allDevices.add(makeProfile(i, "NVIDIA RTX 5090 #" + i, "Discrete", 32L * 1024 * 1024 * 1024, 100_000.0));
        }
        allDevices.add(makeProfile(10, "Intel(R) UHD Graphics 620", "Integrated", 4L * 1024 * 1024 * 1024, 185.0));

        List<Link> links = new ArrayList<>();
        for (int i = 0; i < 11; i++) {
            for (int j = 0; j < 11; j++) {
                if (i != j) {
                    if (i < 10 && j < 10) {
                        links.add(new Link(i, j, 25.0, Transport.PEER_TO_PEER, false));
                    } else {
                        links.add(new Link(i, j, 2.0, Transport.HOST_SHARED_DMA, false));
                    }
                }
            }
        }

        long totalUnits = 100_000L;
        double flopsPerUnit = 1e9;
        long bytesPerUnit = 1024L;
        long syncBytes = 100_000_000L; // 100 MB heavy AllReduce

        WorkloadPlan plan = WorkloadPlanner.plan(allDevices, links, totalUnits, flopsPerUnit, bytesPerUnit, syncBytes);

        // Verify Intel GPU (rank 10) was automatically pruned as a straggler
        ShardAllocation intelAlloc = plan.allocations().get(10);
        assertTrue(intelAlloc.excludedAsStraggler(), "Intel iGPU should be marked as excludedAsStraggler");
        assertEquals(0L, intelAlloc.units(), "Intel iGPU units should be 0 when pruned");

        // Verify all 10 NVIDIA GPUs are active
        assertEquals(10, plan.activeRanks().size());
        for (int i = 0; i < 10; i++) {
            assertTrue(plan.activeRanks().contains(i));
            assertTrue(plan.allocations().get(i).units() > 0);
            assertFalse(plan.allocations().get(i).excludedAsStraggler());
        }

        long sumUnits = plan.allocations().stream().mapToLong(ShardAllocation::units).sum();
        assertEquals(totalUnits, sumUnits);

        // Verify makespan equals pure 10x RTX 5090 cluster makespan (zero degradation guarantee!)
        List<DeviceProfile> rtxOnly = allDevices.subList(0, 10);
        List<Link> rtxLinks = new ArrayList<>();
        for (Link l : links) {
            if (l.src() < 10 && l.dst() < 10) {
                rtxLinks.add(l);
            }
        }
        WorkloadPlan purePlan = WorkloadPlanner.plan(rtxOnly, rtxLinks, totalUnits, flopsPerUnit, bytesPerUnit, syncBytes);
        assertEquals(purePlan.estimatedMakespanMs(), plan.estimatedMakespanMs(), 1e-4);
    }

    @Test
    void test10NvidiaPlus1IntelComputeDominatedZeroSync() {
        // Zero sync: Intel GPU should NOT be pruned because it contributes compute with zero sync overhead
        List<DeviceProfile> allDevices = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            allDevices.add(makeProfile(i, "NVIDIA RTX 5090 #" + i, "Discrete", 32L * 1024 * 1024 * 1024, 100_000.0));
        }
        allDevices.add(makeProfile(10, "Intel(R) UHD Graphics 620", "Integrated", 4L * 1024 * 1024 * 1024, 185.0));

        List<Link> links = buildMeshTopology(11, 20.0, Transport.PEER_TO_PEER);

        long totalUnits = 100_000_000L;
        double flopsPerUnit = 1e9;
        long bytesPerUnit = 0L;
        long syncBytes = 0L; // Zero sync

        WorkloadPlan plan = WorkloadPlanner.plan(allDevices, links, totalUnits, flopsPerUnit, bytesPerUnit, syncBytes);

        // All 11 devices participate
        assertEquals(11, plan.activeRanks().size(), "all 11 devices should participate when sync is zero");
        assertFalse(plan.allocations().get(10).excludedAsStraggler());
        assertTrue(plan.allocations().get(10).units() > 0, "Intel GPU should receive non-zero proportional slice");

        long sumUnits = plan.allocations().stream().mapToLong(ShardAllocation::units).sum();
        assertEquals(totalUnits, sumUnits, "exact sum of allocated units must equal totalUnits");

        // NVIDIA GPUs receive > 99.9% of units
        long nvidiaUnits = 0;
        for (int i = 0; i < 10; i++) {
            nvidiaUnits += plan.allocations().get(i).units();
        }
        double nvidiaShare = (double) nvidiaUnits / totalUnits;
        assertTrue(nvidiaShare > 0.999, "NVIDIA devices should receive >99.9% share, got " + nvidiaShare);
    }

    @Test
    void test5Nvidia4Amd3IntelAsymmetricVramWaterFilling() {
        // 5 NVIDIA (24 GiB, 30,000 GFLOP/s)
        // 4 AMD (24 GiB, 25,000 GFLOP/s)
        // 3 Intel (4 GiB, 5,000 GFLOP/s)
        List<DeviceProfile> devices = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            devices.add(makeProfile(i, "NVIDIA RTX 4090 #" + i, "Discrete", 24L * 1024 * 1024 * 1024, 30_000.0));
        }
        for (int i = 5; i < 9; i++) {
            devices.add(makeProfile(i, "AMD Radeon 7900 XTX #" + (i - 5), "Discrete", 24L * 1024 * 1024 * 1024, 25_000.0));
        }
        for (int i = 9; i < 12; i++) {
            devices.add(makeProfile(i, "Intel Arc A770 #" + (i - 9), "Discrete", 4L * 1024 * 1024 * 1024, 10_000.0));
        }

        List<Link> links = buildMeshTopology(12, 20.0, Transport.PEER_TO_PEER);

        long bytesPerUnit = 2_000_000L; // 2 MB per unit
        double vramUtilization = 0.85;
        long intelMaxUnits = (long) Math.floor((4L * 1024 * 1024 * 1024 * vramUtilization) / bytesPerUnit); // 1825 units

        // Request 80,000 units so Intel's proportional compute share (~2,857) exceeds its capacity cap (1,825)
        long totalUnits = 80_000L;
        WorkloadPlan plan = WorkloadPlanner.plan(devices, links, totalUnits, 1e6, bytesPerUnit, 0L, vramUtilization);

        // Verify Intel devices (ranks 9, 10, 11) are capped at exactly intelMaxUnits
        for (int i = 9; i < 12; i++) {
            long u = plan.allocations().get(i).units();
            assertEquals(intelMaxUnits, u, "Intel card " + i + " should be capped at maxUnits");
        }

        // Verify sum of units equals totalUnits exactly
        long sumUnits = plan.allocations().stream().mapToLong(ShardAllocation::units).sum();
        assertEquals(totalUnits, sumUnits, "sum of units must equal totalUnits");
    }

    @Test
    void testClusterOomThrowsGpuOutOfMemoryException() {
        List<DeviceProfile> devices = List.of(
                makeProfile(0, "GPU 0", "Discrete", 8L * 1024 * 1024 * 1024, 1000.0),
                makeProfile(1, "GPU 1", "Discrete", 8L * 1024 * 1024 * 1024, 1000.0)
        );
        List<Link> links = buildMeshTopology(2, 20.0, Transport.PEER_TO_PEER);

        // Usable VRAM = 2 * 8 GiB * 0.85 = ~14.6 GB
        // Request 20 GB footprint
        long bytesPerUnit = 1_000_000_000L; // 1 GB
        long totalUnits = 20L; // 20 GB > 14.6 GB

        assertThrows(GpuOutOfMemoryException.class, () ->
                WorkloadPlanner.plan(devices, links, totalUnits, 1e6, bytesPerUnit, 0L));
    }

    @Test
    void testSingleGpuAndZeroUnitsEdgeCases() {
        DeviceProfile singleDev = makeProfile(0, "GPU 0", "Discrete", 16L * 1024 * 1024 * 1024, 1000.0);
        List<DeviceProfile> devices = List.of(singleDev);
        List<Link> links = List.of();

        // 1. Single GPU
        WorkloadPlan planSingle = WorkloadPlanner.plan(devices, links, 5000L, 1e6, 1024L, 10_000L);
        assertEquals(1, planSingle.activeRanks().size());
        assertEquals(5000L, planSingle.allocations().getFirst().units());
        assertEquals(0.0, planSingle.estimatedSyncMs(), "single GPU has zero sync time");
        assertEquals(planSingle.estimatedComputeMs(), planSingle.estimatedMakespanMs());

        // 2. Zero Units
        WorkloadPlan planZero = WorkloadPlanner.plan(devices, links, 0L, 1e6, 1024L, 10_000L);
        assertEquals(0, planZero.activeRanks().size());
        assertEquals(0.0, planZero.estimatedMakespanMs());
        assertEquals(0L, planZero.allocations().getFirst().units());
        assertFalse(planZero.allocations().getFirst().excludedAsStraggler());
    }
}
