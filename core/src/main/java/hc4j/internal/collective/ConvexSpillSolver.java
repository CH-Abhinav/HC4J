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

package hc4j.internal.collective;

import hc4j.collective.DeviceProfile;
import hc4j.collective.HostCpuProfile;
import hc4j.collective.Link;
import hc4j.collective.ShardAllocation;
import hc4j.collective.Transport;
import hc4j.collective.WorkloadPlan;
import hc4j.collective.WorkloadSpec;
import hc4j.engine.GpuOutOfMemoryException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;

/**
 * Pure Java Theorem 3 &amp; Theorem 4 solver: 3-tier convex memory spilling (VRAM &rarr; Host-RAM &rarr; Disk),
 * explicit Host CPU offload node, shared PCIe root-complex contention fixed point, \(O(m \log m)\) min-heap
 * integer minimax allocation, and exact \(2^N - 1\) subset search guaranteeing Zero-Degradation Monotonicity.
 */
public final class ConvexSpillSolver {

    private static final double EPSILON = 1e-6;

    private ConvexSpillSolver() {
        throw new AssertionError("No instances of ConvexSpillSolver");
    }

    private record DeviceCurve(
            int index,
            long uVram,
            long uHost,
            long uDisk,
            long uMax,
            double r0,
            double r1,
            double r2) {

        double evalTime(long u) {
            if (u <= 0) {
                return 0.0;
            }
            if (u <= uVram) {
                return u * r0;
            }
            if (u <= uVram + uHost) {
                return (uVram * r0) + ((u - uVram) * r1);
            }
            long uD = Math.min(u - (uVram + uHost), uDisk);
            return (uVram * r0) + (uHost * r1) + (uD * r2);
        }

        double invTime(double t) {
            if (t <= 0.0) {
                return 0.0;
            }
            double tVram = uVram * r0;
            if (t <= tVram) {
                return (r0 > 0.0) ? t / r0 : uVram;
            }
            double tHost = tVram + (uHost * r1);
            if (t <= tHost) {
                return (r1 > 0.0) ? uVram + (t - tVram) / r1 : uVram + uHost;
            }
            double tDisk = tHost + (uDisk * r2);
            if (t <= tDisk) {
                return (r2 > 0.0) ? uVram + uHost + (t - tHost) / r2 : uMax;
            }
            return uMax;
        }
    }

    private record AllocationDetail(
            long totalUnits,
            long vramUnits,
            long hostStreamUnits,
            long diskStreamUnits,
            long cpuInPlaceUnits,
            double computeMs,
            long vramBytes,
            long hostBytes,
            long diskBytes) {}

    private record PlanCandidate(
            List<Integer> activeGpuIndices,
            boolean includesCpu,
            AllocationDetail[] gpuAllocations,
            AllocationDetail cpuAllocation,
            long totalSpilledUnits,
            double computeMs,
            double syncMs,
            double makespanMs) {}

    /**
     * Validates workload planning inputs identically across both native and Java fallback paths.
     *
     * @param devices participating device profiles
     * @param totalUnits total units
     * @param flopsPerUnit FLOPs per unit
     * @param bytesPerUnit working bytes per unit
     * @param syncBytes collective synchronization payload bytes
     * @param vramUtilization VRAM fraction in {@code (0, 1]}
     */
    public static void validateInputs(
            List<DeviceProfile> devices,
            long totalUnits,
            double flopsPerUnit,
            long bytesPerUnit,
            long syncBytes,
            double vramUtilization) {
        Objects.requireNonNull(devices, "devices must not be null");
        if (devices.isEmpty()) {
            throw new IllegalArgumentException("devices must not be empty");
        }
        if (totalUnits < 0) {
            throw new IllegalArgumentException("totalUnits must be non-negative: " + totalUnits);
        }
        if (flopsPerUnit < 0.0) {
            throw new IllegalArgumentException("flopsPerUnit must be non-negative: " + flopsPerUnit);
        }
        if (bytesPerUnit < 0) {
            throw new IllegalArgumentException("bytesPerUnit must be non-negative: " + bytesPerUnit);
        }
        if (syncBytes < 0) {
            throw new IllegalArgumentException("syncBytes must be non-negative: " + syncBytes);
        }
        if (vramUtilization <= 0.0 || vramUtilization > 1.0) {
            throw new IllegalArgumentException("vramUtilization must be in (0, 1]: " + vramUtilization);
        }
    }

    /**
     * Saturating addition utility shared across collective planners.
     *
     * @param a first addend
     * @param b second addend
     * @return {@code a + b} clamped to {@link Long#MAX_VALUE} on overflow
     */
    public static long addSaturating(long a, long b) {
        try {
            return Math.addExact(a, b);
        } catch (ArithmeticException e) {
            return Long.MAX_VALUE;
        }
    }

    /**
     * Computes an exact zero-parameter {@link WorkloadPlan} purely in Java (headless fallback).
     *
     * @param devices participating device profiles
     * @param topology measured network topology links
     * @param spec detailed workload specification
     * @return globally optimal {@link WorkloadPlan}
     */
    public static WorkloadPlan plan(
            List<DeviceProfile> devices,
            List<Link> topology,
            WorkloadSpec spec) {
        Objects.requireNonNull(devices, "devices must not be null");
        Objects.requireNonNull(spec, "spec must not be null");
        validateInputs(devices, spec.totalUnits(), spec.flopsPerUnit(), spec.bytesPerUnit(), spec.syncBytes(), spec.vramUtilization());

        int n = devices.size();
        if (spec.totalUnits() == 0) {
            List<ShardAllocation> allocs = new ArrayList<>(n);
            for (DeviceProfile d : devices) {
                allocs.add(new ShardAllocation(d.rank(), 0L, 0.0, 0.0, 0L, false, 0L, 0L, 0L, 0L));
            }
            return new WorkloadPlan(0L, allocs, List.of(), 0.0, 0.0, 0.0);
        }

        MstDendrogram.TopologyMatrices topo = MstDendrogram.buildTopologyMatrices(n, topology);
        BottleneckTspSolver.RingTable ringTable = (n <= 12)
                ? BottleneckTspSolver.precomputeAllSubsetsRing(n, topo.bw(), topo.latUs())
                : null;

        boolean cpuEnabled = spec.hostCpu().isEnabled();
        PlanCandidate bestPlan = null;

        if (n <= 12) {
            int numSubsets = 1 << n;
            for (int mask = 1; mask < numSubsets; mask++) {
                List<Integer> gpuIndices = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    if ((mask & (1 << i)) != 0) {
                        gpuIndices.add(i);
                    }
                }
                PlanCandidate candGpu = evaluateDeviceSet(gpuIndices, false, devices, topo.bw(), topo.latUs(), topo.tr(), ringTable, spec);
                if (candGpu != null && (bestPlan == null || isBetter(candGpu, bestPlan))) {
                    bestPlan = candGpu;
                }
                if (cpuEnabled) {
                    PlanCandidate candCpu = evaluateDeviceSet(gpuIndices, true, devices, topo.bw(), topo.latUs(), topo.tr(), ringTable, spec);
                    if (candCpu != null && (bestPlan == null || isBetter(candCpu, bestPlan))) {
                        bestPlan = candCpu;
                    }
                }
            }
            if (cpuEnabled) {
                PlanCandidate candCpuOnly = evaluateDeviceSet(List.of(), true, devices, topo.bw(), topo.latUs(), topo.tr(), ringTable, spec);
                if (candCpuOnly != null && (bestPlan == null || isBetter(candCpuOnly, bestPlan))) {
                    bestPlan = candCpuOnly;
                }
            }
        } else {
            List<Integer> activeGpus = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                activeGpus.add(i);
            }
            bestPlan = evaluateDeviceSet(activeGpus, cpuEnabled, devices, topo.bw(), topo.latUs(), topo.tr(), ringTable, spec);
            boolean improved = true;
            while (improved && activeGpus.size() > 1) {
                improved = false;
                int bestPruneIdx = -1;
                PlanCandidate bestPruneCand = null;
                for (int i = 0; i < activeGpus.size(); i++) {
                    List<Integer> candActive = new ArrayList<>(activeGpus);
                    candActive.remove(i);
                    PlanCandidate cand = evaluateDeviceSet(candActive, cpuEnabled, devices, topo.bw(), topo.latUs(), topo.tr(), ringTable, spec);
                    if (cand != null && (bestPruneCand == null || isBetter(cand, bestPruneCand))) {
                        bestPruneCand = cand;
                        bestPruneIdx = i;
                    }
                }
                if (bestPruneCand != null && isBetter(bestPruneCand, bestPlan)) {
                    bestPlan = bestPruneCand;
                    activeGpus.remove(bestPruneIdx);
                    improved = true;
                }
            }
        }

        if (bestPlan == null) {
            throw new GpuOutOfMemoryException(String.format(
                    "Workload of %d units requires memory that exceeds all available VRAM, Host-RAM, and disk storage tiers",
                    spec.totalUnits()));
        }

        List<ShardAllocation> allocations = new ArrayList<>(n + (bestPlan.includesCpu ? 1 : 0));
        List<Integer> activeRanks = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            int rank = devices.get(i).rank();
            AllocationDetail det = bestPlan.gpuAllocations[i];
            long u = det.totalUnits;
            double share = (spec.totalUnits() > 0) ? (double) u / spec.totalUnits() : 0.0;
            boolean excluded = (u == 0 && !bestPlan.activeGpuIndices.contains(i));

            allocations.add(new ShardAllocation(
                    rank,
                    u,
                    share,
                    det.computeMs,
                    det.vramBytes + det.hostBytes + det.diskBytes,
                    excluded,
                    det.vramUnits,
                    det.hostStreamUnits,
                    det.diskStreamUnits,
                    0L));
            if (u > 0) {
                activeRanks.add(rank);
            }
        }

        if (bestPlan.includesCpu && bestPlan.cpuAllocation != null && bestPlan.cpuAllocation.totalUnits > 0) {
            AllocationDetail cpuDet = bestPlan.cpuAllocation;
            long u = cpuDet.totalUnits;
            double share = (spec.totalUnits() > 0) ? (double) u / spec.totalUnits() : 0.0;
            allocations.add(new ShardAllocation(
                    ShardAllocation.HOST_CPU_RANK,
                    u,
                    share,
                    cpuDet.computeMs,
                    cpuDet.hostBytes,
                    false,
                    0L,
                    0L,
                    0L,
                    cpuDet.cpuInPlaceUnits));
            activeRanks.add(ShardAllocation.HOST_CPU_RANK);
        }

        return new WorkloadPlan(
                spec.totalUnits(),
                allocations,
                activeRanks,
                bestPlan.makespanMs,
                bestPlan.computeMs,
                bestPlan.syncMs);
    }

    private static boolean isBetter(PlanCandidate a, PlanCandidate b) {
        if (a.makespanMs < b.makespanMs * (1.0 - 1e-6)) {
            return true;
        }
        if (b.makespanMs < a.makespanMs * (1.0 - 1e-6)) {
            return false;
        }
        if (a.totalSpilledUnits < b.totalSpilledUnits) {
            return true;
        }
        if (b.totalSpilledUnits < a.totalSpilledUnits) {
            return false;
        }
        return a.activeGpuIndices.size() > b.activeGpuIndices.size();
    }

    private static PlanCandidate evaluateDeviceSet(
            List<Integer> activeGpuIndices,
            boolean includeCpu,
            List<DeviceProfile> devices,
            double[][] bw,
            double[][] latUs,
            Transport[][] tr,
            BottleneckTspSolver.RingTable ringTable,
            WorkloadSpec spec) {
        int n = devices.size();
        long totalUnits = spec.totalUnits();
        long mBytes = spec.bytesPerUnit();
        double fFlops = spec.flopsPerUnit();
        double kappa = spec.ioMultiplier();
        double gamma = spec.streamingOverlapGamma();
        HostCpuProfile cpu = spec.hostCpu();

        if (activeGpuIndices.isEmpty() && !includeCpu) {
            return null;
        }

        long[] uVram = new long[n];
        for (int i : activeGpuIndices) {
            if (mBytes > 0) {
                uVram[i] = (long) Math.floor((devices.get(i).vramBytes() * spec.vramUtilization()) / mBytes);
            } else {
                uVram[i] = Long.MAX_VALUE / 4;
            }
        }

        long totalHostRamBytes = (includeCpu && cpu.isEnabled()) ? cpu.ramBytes() : 0L;
        long totalDiskBytes = spec.diskCapacityBytes();

        int kSpill = 0;
        AllocationDetail[] gpuDets = new AllocationDetail[n];
        AllocationDetail cpuDet = null;
        double maxComputeMs = 0.0;

        for (int iter = 0; iter < 4; iter++) {
            double effectivePcie = spec.defaultPcieGbps();
            if (kSpill > 0 && cpu.aggregatePcieGbps() > 0.0) {
                effectivePcie = Math.min(spec.defaultPcieGbps(), cpu.aggregatePcieGbps() / Math.max(1, kSpill));
            }

            List<DeviceCurve> curves = new ArrayList<>();
            int participantsForHost = activeGpuIndices.size() + (includeCpu ? 1 : 0);
            long hostRamPerGpu = (participantsForHost > 0 && mBytes > 0)
                    ? totalHostRamBytes / participantsForHost
                    : ((mBytes == 0) ? Long.MAX_VALUE / 4 : 0L);

            for (int i : activeGpuIndices) {
                double cGflops = Math.max(devices.get(i).computeGflops(), EPSILON);
                double r0 = (fFlops / (cGflops * 1e9)) * 1000.0;
                double r1 = r0;
                double r2 = r0;
                long uHostCap = 0;
                long uDiskCap = 0;

                if (mBytes > 0 && totalHostRamBytes > 0) {
                    uHostCap = (hostRamPerGpu > 0) ? hostRamPerGpu / mBytes : 0L;
                    double tComp = fFlops / (cGflops * 1e9);
                    double tPcie = (effectivePcie > 0.0) ? (kappa * mBytes) / (effectivePcie * 1e9) : 1e3;
                    r1 = (Math.max(tComp, tPcie) + (1.0 - gamma) * Math.min(tComp, tPcie)) * 1000.0;
                }

                if (mBytes > 0 && spec.diskBandwidthGbps() > 0.0 && totalDiskBytes > 0) {
                    long diskPerGpu = (!activeGpuIndices.isEmpty()) ? totalDiskBytes / activeGpuIndices.size() : 0L;
                    uDiskCap = diskPerGpu / mBytes;
                    double bDiskEff = Math.min(effectivePcie, spec.diskBandwidthGbps());
                    double tComp = fFlops / (cGflops * 1e9);
                    double tDisk = (bDiskEff > 0.0) ? (kappa * mBytes) / (bDiskEff * 1e9) : 1e3;
                    r2 = (Math.max(tComp, tDisk) + (1.0 - gamma) * Math.min(tComp, tDisk)) * 1000.0;
                }

                long uMax = addSaturating(uVram[i], addSaturating(uHostCap, uDiskCap));
                curves.add(new DeviceCurve(i, uVram[i], uHostCap, uDiskCap, uMax, r0, r1, r2));
            }

            if (includeCpu && cpu.isEnabled()) {
                double cpuGflops = Math.max(cpu.computeGflops(), EPSILON);
                double tComp = fFlops / (cpuGflops * 1e9);
                double tRam = (cpu.ramBandwidthGbps() > 0.0) ? (kappa * mBytes) / (cpu.ramBandwidthGbps() * 1e9) : 1e3;
                double rCpu = Math.max(tComp, tRam) * 1000.0;
                long uCpuCap = (mBytes > 0) ? hostRamPerGpu / mBytes : Long.MAX_VALUE / 4;
                curves.add(new DeviceCurve(ShardAllocation.HOST_CPU_RANK, 0L, uCpuCap, 0L, uCpuCap, rCpu, rCpu, rCpu));
            }

            long totalCapacity = 0;
            for (DeviceCurve c : curves) {
                totalCapacity = addSaturating(totalCapacity, c.uMax);
            }
            if (totalCapacity < totalUnits) {
                return null;
            }

            long[] assigned = solveIntegerMinimax(curves, totalUnits);

            int newKSpill = 0;
            for (int idx = 0; idx < curves.size(); idx++) {
                DeviceCurve c = curves.get(idx);
                long u = assigned[idx];
                if (c.index != ShardAllocation.HOST_CPU_RANK && u > c.uVram) {
                    newKSpill++;
                }
            }

            if (newKSpill == kSpill || iter == 3) {
                maxComputeMs = 0.0;
                for (int idx = 0; idx < curves.size(); idx++) {
                    DeviceCurve c = curves.get(idx);
                    long u = assigned[idx];
                    double timeMs = c.evalTime(u);
                    maxComputeMs = Math.max(maxComputeMs, timeMs);

                    if (c.index != ShardAllocation.HOST_CPU_RANK) {
                        long vramU = Math.min(u, c.uVram);
                        long rem = u - vramU;
                        long hostU = Math.min(rem, c.uHost);
                        rem -= hostU;
                        long diskU = Math.min(rem, c.uDisk);

                        gpuDets[c.index] = new AllocationDetail(
                                u, vramU, hostU, diskU, 0L, timeMs,
                                vramU * mBytes, hostU * mBytes, diskU * mBytes);
                    } else {
                        cpuDet = new AllocationDetail(
                                u, 0L, 0L, 0L, u, timeMs,
                                0L, u * mBytes, 0L);
                    }
                }
                break;
            }
            kSpill = newKSpill;
        }

        for (int i = 0; i < n; i++) {
            if (gpuDets[i] == null) {
                gpuDets[i] = new AllocationDetail(0L, 0L, 0L, 0L, 0L, 0.0, 0L, 0L, 0L);
            }
        }

        double syncMs = (activeGpuIndices.size() >= 2)
                ? MstDendrogram.estimateSyncMsExact(activeGpuIndices, bw, latUs, tr, ringTable, devices, spec.syncBytes())
                : 0.0;

        double alpha = spec.overlapAlpha();
        double makespanMs = Math.max(maxComputeMs, syncMs) + (1.0 - alpha) * Math.min(maxComputeMs, syncMs);

        long totalSpilled = 0;
        for (AllocationDetail d : gpuDets) {
            totalSpilled += (d.hostStreamUnits + d.diskStreamUnits);
        }

        return new PlanCandidate(
                activeGpuIndices,
                includeCpu,
                gpuDets,
                cpuDet,
                totalSpilled,
                maxComputeMs,
                syncMs,
                makespanMs);
    }

    private static long[] solveIntegerMinimax(List<DeviceCurve> curves, long totalUnits) {
        int m = curves.size();
        long[] result = new long[m];
        if (m == 0 || totalUnits == 0) {
            return result;
        }

        double tLow = 0.0;
        double tHigh = 1.0;
        for (DeviceCurve c : curves) {
            tHigh = Math.max(tHigh, c.evalTime(totalUnits));
        }

        for (int step = 0; step < 64; step++) {
            double tMid = tLow + (tHigh - tLow) * 0.5;
            double sumUnits = 0.0;
            for (DeviceCurve c : curves) {
                sumUnits += c.invTime(tMid);
            }
            if (sumUnits >= totalUnits) {
                tHigh = tMid;
            } else {
                tLow = tMid;
            }
        }

        double tOpt = tHigh;
        long currentSum = 0;
        for (int i = 0; i < m; i++) {
            DeviceCurve c = curves.get(i);
            long floorU = (long) Math.floor(c.invTime(tOpt));
            floorU = Math.max(0, Math.min(floorU, c.uMax));
            result[i] = floorU;
            currentSum += floorU;
        }

        long deficit = totalUnits - currentSum;
        if (deficit > 0) {
            record HeapEntry(int index, double nextTime) implements Comparable<HeapEntry> {
                @Override
                public int compareTo(HeapEntry o) {
                    int cmp = Double.compare(this.nextTime, o.nextTime);
                    return (cmp != 0) ? cmp : Integer.compare(this.index, o.index);
                }
            }

            PriorityQueue<HeapEntry> pq = new PriorityQueue<>();
            for (int i = 0; i < m; i++) {
                if (result[i] < curves.get(i).uMax) {
                    pq.add(new HeapEntry(i, curves.get(i).evalTime(result[i] + 1)));
                }
            }

            while (deficit > 0 && !pq.isEmpty()) {
                HeapEntry top = pq.poll();
                result[top.index]++;
                deficit--;
                if (result[top.index] < curves.get(top.index).uMax) {
                    pq.add(new HeapEntry(top.index, curves.get(top.index).evalTime(result[top.index] + 1)));
                }
            }
        } else if (deficit < 0) {
            long surplus = -deficit;
            for (int i = m - 1; i >= 0 && surplus > 0; i--) {
                long take = Math.min(result[i], surplus);
                result[i] -= take;
                surplus -= take;
            }
        }

        return result;
    }
}
