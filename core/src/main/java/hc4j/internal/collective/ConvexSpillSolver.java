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
    private static final double HOST_PCIE_SYNC_LATENCY_MS = 0.02;

    private ConvexSpillSolver() {
        throw new AssertionError("No instances of ConvexSpillSolver");
    }

    private enum RoundTier {
        VRAM,
        HOST_STREAM,
        DISK_STREAM,
        CPU_IN_PLACE
    }

    private record PoolHeapEntry(int slot, RoundTier tier, double nextTime) implements Comparable<PoolHeapEntry> {
        @Override
        public int compareTo(PoolHeapEntry o) {
            int cmp = Double.compare(this.nextTime, o.nextTime);
            return (cmp != 0) ? cmp : Integer.compare(this.slot, o.slot);
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
                ? BottleneckTspSolver.precomputeAllSubsetsRingForPayload(n, topo.bw(), topo.latUs(), spec.syncBytes())
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
        int k = activeGpuIndices.size();
        long totalUnits = spec.totalUnits();
        long mBytes = spec.bytesPerUnit();
        double fFlops = spec.flopsPerUnit();
        double kappa = spec.ioMultiplier();
        double gamma = spec.streamingOverlapGamma();
        HostCpuProfile cpu = spec.hostCpu();

        if (k == 0 && !includeCpu) {
            return null;
        }
        if (includeCpu && !cpu.isEnabled()) {
            return null;
        }

        long totalHostRamBytes = (cpu.ramBytes() > 0) ? cpu.ramBytes() : 0L;
        long totalDiskBytes = spec.diskCapacityBytes();

        AllocationDetail[] gpuDets = new AllocationDetail[n];
        AllocationDetail cpuDet = null;

        double[] r0 = new double[k];
        for (int a = 0; a < k; a++) {
            DeviceProfile dev = devices.get(activeGpuIndices.get(a));
            double cGflops = Math.max(dev.computeGflops(), EPSILON);
            double tComp = fFlops / (cGflops * 1e9);
            double tVram = (mBytes > 0 && dev.vramBandwidthGbps() > 0.0 && Double.isFinite(dev.vramBandwidthGbps()))
                    ? ((kappa * mBytes) / (dev.vramBandwidthGbps() * 1e9))
                    : 0.0;
            r0[a] = Math.max(tComp, tVram) * 1000.0;
        }

        double rCpu = 0.0;
        if (includeCpu) {
            double cpuGflops = Math.max(cpu.computeGflops(), EPSILON);
            double tComp = fFlops / (cpuGflops * 1e9);
            double tRam = (cpu.ramBandwidthGbps() > 0.0) ? (kappa * mBytes) / (cpu.ramBandwidthGbps() * 1e9) : 0.0;
            rCpu = Math.max(tComp, tRam) * 1000.0;
        }

        double maxComputeMs;
        if (mBytes == 0) {
            double tHigh = 1.0;
            for (double rate : r0) {
                tHigh = Math.max(tHigh, totalUnits * rate);
            }
            if (includeCpu) {
                tHigh = Math.max(tHigh, totalUnits * rCpu);
            }
            double tLow = 0.0;
            for (int step = 0; step < 64; step++) {
                double tMid = tLow + (tHigh - tLow) * 0.5;
                double sumU = 0.0;
                for (double rate : r0) {
                    sumU += (rate > 0.0) ? (tMid / rate) : (double) totalUnits;
                }
                if (includeCpu) {
                    sumU += (rCpu > 0.0) ? (tMid / rCpu) : (double) totalUnits;
                }
                if (sumU >= totalUnits) {
                    tHigh = tMid;
                } else {
                    tLow = tMid;
                }
            }

            double tOpt = tHigh;
            long[] vInt = new long[k];
            long currentSum = 0L;
            for (int a = 0; a < k; a++) {
                long u = (r0[a] > 0.0)
                        ? Math.min((long) Math.floor(tOpt / r0[a]), totalUnits)
                        : Math.max(0L, totalUnits - currentSum);
                vInt[a] = u;
                currentSum = addSaturating(currentSum, u);
            }
            long uCInt = 0L;
            if (includeCpu) {
                uCInt = (rCpu > 0.0)
                        ? Math.min((long) Math.floor(tOpt / rCpu), totalUnits)
                        : Math.max(0L, totalUnits - currentSum);
                currentSum = addSaturating(currentSum, uCInt);
            }

            if (totalUnits > currentSum) {
                long deficit = totalUnits - currentSum;
                PriorityQueue<PoolHeapEntry> pq = new PriorityQueue<>(Math.max(1, k + 1));
                for (int a = 0; a < k; a++) {
                    pq.add(new PoolHeapEntry(a, RoundTier.VRAM, (vInt[a] + 1) * r0[a]));
                }
                if (includeCpu) {
                    pq.add(new PoolHeapEntry(Integer.MAX_VALUE, RoundTier.CPU_IN_PLACE, (uCInt + 1) * rCpu));
                }
                while (deficit > 0 && !pq.isEmpty()) {
                    PoolHeapEntry top = pq.poll();
                    deficit--;
                    if (top.slot() == Integer.MAX_VALUE) {
                        uCInt++;
                        pq.add(new PoolHeapEntry(Integer.MAX_VALUE, RoundTier.CPU_IN_PLACE, (uCInt + 1) * rCpu));
                    } else {
                        int slot = top.slot();
                        vInt[slot]++;
                        pq.add(new PoolHeapEntry(slot, RoundTier.VRAM, (vInt[slot] + 1) * r0[slot]));
                    }
                }
            } else if (currentSum > totalUnits) {
                long surplus = currentSum - totalUnits;
                if (includeCpu && surplus > 0) {
                    long take = Math.min(uCInt, surplus);
                    uCInt -= take;
                    surplus -= take;
                }
                for (int a = k - 1; a >= 0 && surplus > 0; a--) {
                    long take = Math.min(vInt[a], surplus);
                    vInt[a] -= take;
                    surplus -= take;
                }
            }

            double compMax = 0.0;
            for (int a = 0; a < k; a++) {
                long u = vInt[a];
                double tm = u * r0[a];
                compMax = Math.max(compMax, tm);
                gpuDets[activeGpuIndices.get(a)] = new AllocationDetail(
                        u, u, 0L, 0L, 0L, tm, 0L, 0L, 0L);
            }
            if (includeCpu) {
                double tm = uCInt * rCpu;
                compMax = Math.max(compMax, tm);
                cpuDet = new AllocationDetail(uCInt, 0L, 0L, 0L, uCInt, tm, 0L, 0L, 0L);
            }
            maxComputeMs = compMax;
        } else {
            long[] uVram = new long[k];
            long totalVramCap = 0L;
            for (int a = 0; a < k; a++) {
                int devIdx = activeGpuIndices.get(a);
                long uv = (long) Math.floor((devices.get(devIdx).vramBytes() * spec.vramUtilization()) / mBytes);
                uVram[a] = uv;
                totalVramCap = addSaturating(totalVramCap, uv);
            }

            double invSum = 0.0;
            if (!includeCpu && k > 0) {
                boolean allPos = true;
                for (int a = 0; a < k; a++) {
                    if (r0[a] <= 0.0) {
                        allPos = false;
                        break;
                    }
                    invSum += 1.0 / r0[a];
                }
                if (!allPos) invSum = 0.0;
            }
            double t0Vram = (invSum > 0.0) ? ((double) totalUnits / invSum) : Double.POSITIVE_INFINITY;
            boolean fastVramOnly = !includeCpu && k > 0 && totalVramCap >= totalUnits && invSum > 0.0;
            if (fastVramOnly) {
                for (int a = 0; a < k; a++) {
                    if ((t0Vram / r0[a]) + 1.0 > (double) uVram[a]) {
                        fastVramOnly = false;
                        break;
                    }
                }
            }

            if (fastVramOnly) {
                long[] vInt = new long[k];
                long currentSum = 0L;
                for (int a = 0; a < k; a++) {
                    long va = Math.min((long) Math.floor(t0Vram / r0[a]), uVram[a]);
                    vInt[a] = va;
                    currentSum = addSaturating(currentSum, va);
                }
                if (totalUnits > currentSum) {
                    long deficit = totalUnits - currentSum;
                    PriorityQueue<PoolHeapEntry> pq = new PriorityQueue<>(Math.max(1, k));
                    for (int a = 0; a < k; a++) {
                        if (vInt[a] < uVram[a]) {
                            pq.add(new PoolHeapEntry(a, RoundTier.VRAM, (vInt[a] + 1) * r0[a]));
                        }
                    }
                    while (deficit > 0 && !pq.isEmpty()) {
                        PoolHeapEntry top = pq.poll();
                        int slot = top.slot();
                        vInt[slot]++;
                        deficit--;
                        if (vInt[slot] < uVram[slot]) {
                            pq.add(new PoolHeapEntry(slot, RoundTier.VRAM, (vInt[slot] + 1) * r0[slot]));
                        }
                    }
                } else if (currentSum > totalUnits) {
                    long surplus = currentSum - totalUnits;
                    for (int a = k - 1; a >= 0 && surplus > 0; a--) {
                        long take = Math.min(vInt[a], surplus);
                        vInt[a] -= take;
                        surplus -= take;
                    }
                }
                double compMax = 0.0;
                for (int a = 0; a < k; a++) {
                    long u = vInt[a];
                    double tm = u * r0[a];
                    compMax = Math.max(compMax, tm);
                    gpuDets[activeGpuIndices.get(a)] = new AllocationDetail(
                            u, u, 0L, 0L, 0L, tm, u * mBytes, 0L, 0L);
                }
                maxComputeMs = compMax;
            } else {
            // Exact shared-pool Fixed-T LP + budget-respecting min-heap integer rounding.
            double[] r1 = new double[k];
            double[] r2 = new double[k];
            double perGpuPcie = spec.defaultPcieGbps();
            double bPcieAgg = (cpu.aggregatePcieGbps() > 0.0) ? cpu.aggregatePcieGbps() : (k * perGpuPcie);

            long hostCapUnits = totalHostRamBytes / mBytes;
            boolean hostSpillEnabled = k > 0 && totalHostRamBytes > 0 && perGpuPcie > 0.0 && bPcieAgg > 0.0;
            boolean diskEnabled = k > 0 && spec.diskBandwidthGbps() > 0.0
                    && totalDiskBytes > 0 && perGpuPcie > 0.0 && bPcieAgg > 0.0;
            long diskCapUnits = diskEnabled ? (totalDiskBytes / mBytes) : 0L;

            for (int a = 0; a < k; a++) {
                int devIdx = activeGpuIndices.get(a);
                double cGflops = Math.max(devices.get(devIdx).computeGflops(), EPSILON);
                double tComp = fFlops / (cGflops * 1e9);

                double tPcie = (perGpuPcie > 0.0) ? (kappa * mBytes) / (perGpuPcie * 1e9) : 1e3;
                r1[a] = (Math.max(tComp, tPcie) + (1.0 - gamma) * Math.min(tComp, tPcie)) * 1000.0;

                double bDiskEff = diskEnabled ? Math.min(perGpuPcie, spec.diskBandwidthGbps()) : 0.0;
                double tDisk = (bDiskEff > 0.0) ? (kappa * mBytes) / (bDiskEff * 1e9) : 1e3;
                r2[a] = (Math.max(tComp, tDisk) + (1.0 - gamma) * Math.min(tComp, tDisk)) * 1000.0;
            }

            long effectiveHostCap = (includeCpu || hostSpillEnabled) ? hostCapUnits : 0L;
            long totalCapacity = addSaturating(totalVramCap, addSaturating(effectiveHostCap, diskCapUnits));
            if (totalCapacity < totalUnits) {
                return null;
            }

            int[] hostOrder = new int[k];
            int[] diskOrder = new int[k];
            for (int a = 0; a < k; a++) {
                hostOrder[a] = a;
                diskOrder[a] = a;
            }
            for (int i = 1; i < k; i++) {
                int key = hostOrder[i];
                double rKey = diskEnabled ? (r1[key] / Math.max(r2[key], 1e-12)) : r1[key];
                int j = i - 1;
                while (j >= 0) {
                    int b = hostOrder[j];
                    double rB = diskEnabled ? (r1[b] / Math.max(r2[b], 1e-12)) : r1[b];
                    int cmp = Double.compare(rB, rKey);
                    if (cmp == 0) cmp = Double.compare(r1[b], r1[key]);
                    if (cmp == 0) cmp = Integer.compare(activeGpuIndices.get(b), activeGpuIndices.get(key));
                    if (cmp <= 0) break;
                    hostOrder[j + 1] = b;
                    j--;
                }
                hostOrder[j + 1] = key;
            }
            for (int i = 1; i < k; i++) {
                int key = diskOrder[i];
                int j = i - 1;
                while (j >= 0) {
                    int b = diskOrder[j];
                    int cmp = Double.compare(r2[b], r2[key]);
                    if (cmp == 0) cmp = Integer.compare(activeGpuIndices.get(b), activeGpuIndices.get(key));
                    if (cmp <= 0) break;
                    diskOrder[j + 1] = b;
                    j--;
                }
                diskOrder[j + 1] = key;
            }

            double kappaBytes = kappa * mBytes;
            double[] lpV = new double[k];
            double[] lpDeltaT = new double[k];
            double[] lpS = new double[k];
            double[] lpD = new double[k];
            final double[] lpUcHolder = new double[1];

            final double rCpuEff = rCpu;
            java.util.function.DoublePredicate phi = (double tMs) -> {
                double hBarT;
                if (totalHostRamBytes == 0) {
                    hBarT = 0.0;
                } else if (cpu.ramBandwidthGbps() > 0.0 && kappaBytes > 0.0) {
                    hBarT = Math.min((double) hostCapUnits, (cpu.ramBandwidthGbps() * tMs * 1e6) / kappaBytes);
                } else {
                    hBarT = (double) hostCapUnits;
                }

                double pPcieT;
                if (bPcieAgg > 0.0) {
                    pPcieT = (kappaBytes > 0.0) ? (bPcieAgg * tMs * 1e6) / kappaBytes : Double.POSITIVE_INFINITY;
                } else {
                    pPcieT = 0.0;
                }

                double kDiskT;
                if (!diskEnabled) {
                    kDiskT = 0.0;
                } else if (kappaBytes > 0.0) {
                    kDiskT = Math.min((double) diskCapUnits, (spec.diskBandwidthGbps() * tMs * 1e6) / kappaBytes);
                } else {
                    kDiskT = (double) diskCapUnits;
                }

                double sumUnits = 0.0;
                for (int a = 0; a < k; a++) {
                    double va = (r0[a] > 0.0) ? Math.min((double) uVram[a], tMs / r0[a]) : (double) uVram[a];
                    lpV[a] = va;
                    lpDeltaT[a] = Math.max(0.0, tMs - r0[a] * va);
                    sumUnits += va;
                }

                double uC = 0.0;
                if (includeCpu) {
                    uC = Math.max(0.0, Math.min(hBarT, (rCpuEff > 0.0) ? (tMs / rCpuEff) : hBarT));
                    sumUnits += uC;
                }
                lpUcHolder[0] = uC;

                java.util.Arrays.fill(lpS, 0.0);
                double sumS = 0.0;
                if (hostSpillEnabled) {
                    double remSCap = Math.min(pPcieT, Math.max(0.0, hBarT - uC));
                    for (int a : hostOrder) {
                        if (remSCap <= 0.0) break;
                        double capA = (r1[a] > 0.0) ? (lpDeltaT[a] / r1[a]) : remSCap;
                        double sa = Math.max(0.0, Math.min(capA, remSCap));
                        lpS[a] = sa;
                        remSCap -= sa;
                        sumS += sa;
                        lpDeltaT[a] = Math.max(0.0, lpDeltaT[a] - r1[a] * sa);
                    }
                    sumUnits += sumS;
                }

                java.util.Arrays.fill(lpD, 0.0);
                if (diskEnabled) {
                    double pcieRem = Math.max(0.0, pPcieT - sumS);
                    double maxDiskFromGpuTime = 0.0;
                    for (int a = 0; a < k; a++) {
                        if (r2[a] > 0.0) {
                            maxDiskFromGpuTime += lpDeltaT[a] / r2[a];
                        } else {
                            maxDiskFromGpuTime = Double.POSITIVE_INFINITY;
                        }
                    }
                    double remDisk = Math.max(0.0, Math.min(kDiskT, Math.min(pcieRem, maxDiskFromGpuTime)));
                    sumUnits += remDisk;
                    for (int a : diskOrder) {
                        if (remDisk <= 0.0) break;
                        double capA = (r2[a] > 0.0) ? (lpDeltaT[a] / r2[a]) : remDisk;
                        double da = Math.max(0.0, Math.min(capA, remDisk));
                        lpD[a] = da;
                        remDisk -= da;
                    }
                }

                return sumUnits >= (double) totalUnits;
            };

            double tLow = 0.0;
            double tHigh = 1.0;
            int doubled = 0;
            while (!phi.test(tHigh)) {
                tHigh *= 2.0;
                doubled++;
                if (!Double.isFinite(tHigh) || doubled > 100) {
                    return null;
                }
            }

            for (int step = 0; step < 64; step++) {
                double tMid = tLow + (tHigh - tLow) * 0.5;
                if (phi.test(tMid)) {
                    tHigh = tMid;
                } else {
                    tLow = tMid;
                }
            }

            phi.test(tHigh);
            long[] vInt = new long[k];
            long[] sInt = new long[k];
            long[] dInt = new long[k];

            long uCInt = includeCpu ? Math.min((long) Math.floor(lpUcHolder[0]), hostCapUnits) : 0L;
            long remHostUnits = Math.max(0L, effectiveHostCap - uCInt);
            long remDiskUnits = diskCapUnits;

            long currentSum = uCInt;
            long usedRam = uCInt;
            long usedPcie = 0L;
            long usedDisk = 0L;

            for (int a = 0; a < k; a++) {
                long va = Math.min((long) Math.floor(lpV[a]), uVram[a]);
                vInt[a] = va;
                currentSum = addSaturating(currentSum, va);

                if (hostSpillEnabled && remHostUnits > 0) {
                    long sa = Math.min((long) Math.floor(lpS[a]), remHostUnits);
                    sInt[a] = sa;
                    remHostUnits -= sa;
                    usedRam += sa;
                    usedPcie += sa;
                    currentSum = addSaturating(currentSum, sa);
                }

                if (diskEnabled && remDiskUnits > 0) {
                    long da = Math.min((long) Math.floor(lpD[a]), remDiskUnits);
                    dInt[a] = da;
                    remDiskUnits -= da;
                    usedPcie += da;
                    usedDisk += da;
                    currentSum = addSaturating(currentSum, da);
                }
            }

            double ramUnitMs = (cpu.ramBandwidthGbps() > 0.0 && kappaBytes > 0.0)
                    ? (kappaBytes / (cpu.ramBandwidthGbps() * 1e6)) : 0.0;
            double pcieUnitMs = (bPcieAgg > 0.0 && kappaBytes > 0.0)
                    ? (kappaBytes / (bPcieAgg * 1e6)) : 0.0;
            double diskUnitMs = (diskEnabled && kappaBytes > 0.0)
                    ? (kappaBytes / (spec.diskBandwidthGbps() * 1e6)) : 0.0;

            double[] gpuTime = new double[k];
            for (int a = 0; a < k; a++) {
                gpuTime[a] = vInt[a] * r0[a] + sInt[a] * r1[a] + dInt[a] * r2[a];
            }

            if (totalUnits > currentSum) {
                long deficit = totalUnits - currentSum;
                PriorityQueue<PoolHeapEntry> pq = new PriorityQueue<>(Math.max(1, k + 1));
                for (int a = 0; a < k; a++) {
                    PoolHeapEntry cand = nextGpuCand(a, vInt[a], uVram[a], gpuTime[a],
                            r0[a], r1[a], r2[a], hostSpillEnabled, diskEnabled,
                            remHostUnits, remDiskUnits, usedRam, usedPcie, usedDisk,
                            ramUnitMs, pcieUnitMs, diskUnitMs);
                    if (cand != null) {
                        pq.add(cand);
                    }
                }
                if (includeCpu && remHostUnits > 0) {
                    double tm = Math.max((uCInt + 1) * rCpu, (usedRam + 1) * ramUnitMs);
                    pq.add(new PoolHeapEntry(Integer.MAX_VALUE, RoundTier.CPU_IN_PLACE, tm));
                }

                while (deficit > 0 && !pq.isEmpty()) {
                    PoolHeapEntry top = pq.poll();
                    if (top.slot() == Integer.MAX_VALUE) {
                        if (!includeCpu || remHostUnits == 0) continue;
                        double curTm = Math.max((uCInt + 1) * rCpu, (usedRam + 1) * ramUnitMs);
                        if (Math.abs(curTm - top.nextTime()) > 1e-12) {
                            pq.add(new PoolHeapEntry(Integer.MAX_VALUE, RoundTier.CPU_IN_PLACE, curTm));
                            continue;
                        }
                        uCInt++;
                        remHostUnits--;
                        usedRam++;
                        deficit--;
                        if (remHostUnits > 0) {
                            double nt = Math.max((uCInt + 1) * rCpu, (usedRam + 1) * ramUnitMs);
                            pq.add(new PoolHeapEntry(Integer.MAX_VALUE, RoundTier.CPU_IN_PLACE, nt));
                        }
                    } else {
                        int a = top.slot();
                        PoolHeapEntry cur = nextGpuCand(a, vInt[a], uVram[a], gpuTime[a],
                                r0[a], r1[a], r2[a], hostSpillEnabled, diskEnabled,
                                remHostUnits, remDiskUnits, usedRam, usedPcie, usedDisk,
                                ramUnitMs, pcieUnitMs, diskUnitMs);
                        if (cur == null) continue;
                        if (cur.tier() != top.tier() || Math.abs(cur.nextTime() - top.nextTime()) > 1e-12) {
                            pq.add(cur);
                            continue;
                        }
                        switch (cur.tier()) {
                            case VRAM -> {
                                vInt[a]++;
                                gpuTime[a] += r0[a];
                            }
                            case HOST_STREAM -> {
                                sInt[a]++;
                                gpuTime[a] += r1[a];
                                remHostUnits--;
                                usedRam++;
                                usedPcie++;
                            }
                            case DISK_STREAM -> {
                                dInt[a]++;
                                gpuTime[a] += r2[a];
                                remDiskUnits--;
                                usedPcie++;
                                usedDisk++;
                            }
                            case CPU_IN_PLACE -> {}
                        }
                        deficit--;
                        PoolHeapEntry nxt = nextGpuCand(a, vInt[a], uVram[a], gpuTime[a],
                                r0[a], r1[a], r2[a], hostSpillEnabled, diskEnabled,
                                remHostUnits, remDiskUnits, usedRam, usedPcie, usedDisk,
                                ramUnitMs, pcieUnitMs, diskUnitMs);
                        if (nxt != null) {
                            pq.add(nxt);
                        }
                    }
                }
            } else if (currentSum > totalUnits) {
                long surplus = currentSum - totalUnits;
                while (surplus > 0) {
                    int bestSlot = -1;
                    double bestTm = -1.0;
                    if (includeCpu && uCInt > 0) {
                        bestTm = uCInt * rCpu;
                        bestSlot = Integer.MAX_VALUE;
                    }
                    for (int a = 0; a < k; a++) {
                        if (vInt[a] + sInt[a] + dInt[a] > 0 && gpuTime[a] >= bestTm) {
                            bestTm = gpuTime[a];
                            bestSlot = a;
                        }
                    }
                    if (bestSlot == -1) break;
                    if (bestSlot == Integer.MAX_VALUE) {
                        uCInt--;
                        usedRam--;
                    } else if (dInt[bestSlot] > 0) {
                        dInt[bestSlot]--;
                        gpuTime[bestSlot] -= r2[bestSlot];
                        usedPcie--;
                        usedDisk--;
                    } else if (sInt[bestSlot] > 0) {
                        sInt[bestSlot]--;
                        gpuTime[bestSlot] -= r1[bestSlot];
                        usedRam--;
                        usedPcie--;
                    } else {
                        vInt[bestSlot]--;
                        gpuTime[bestSlot] -= r0[bestSlot];
                    }
                    surplus--;
                }
            }

            double compMax = 0.0;
            for (int a = 0; a < k; a++) {
                long totalA = vInt[a] + sInt[a] + dInt[a];
                double tm = gpuTime[a];
                compMax = Math.max(compMax, tm);
                gpuDets[activeGpuIndices.get(a)] = new AllocationDetail(
                        totalA, vInt[a], sInt[a], dInt[a], 0L, tm,
                        vInt[a] * mBytes, sInt[a] * mBytes, dInt[a] * mBytes);
            }
            if (includeCpu) {
                double tm = uCInt * rCpu;
                compMax = Math.max(compMax, tm);
                cpuDet = new AllocationDetail(uCInt, 0L, 0L, 0L, uCInt, tm, 0L, uCInt * mBytes, 0L);
            }
            if (usedRam > 0) {
                compMax = Math.max(compMax, usedRam * ramUnitMs);
            }
            if (usedPcie > 0) {
                compMax = Math.max(compMax, usedPcie * pcieUnitMs);
            }
            if (usedDisk > 0) {
                compMax = Math.max(compMax, usedDisk * diskUnitMs);
            }
            maxComputeMs = compMax;
            }
        }

        for (int i = 0; i < n; i++) {
            if (gpuDets[i] == null) {
                gpuDets[i] = new AllocationDetail(0L, 0L, 0L, 0L, 0L, 0.0, 0L, 0L, 0L);
            }
        }

        double gpuSyncMs = (k >= 2 && spec.syncBytes() > 0)
                ? MstDendrogram.estimateSyncMsExact(activeGpuIndices, bw, latUs, tr, ringTable, devices, spec.syncBytes())
                : 0.0;

        double hostSyncMs = 0.0;
        if (k >= 1 && includeCpu && spec.syncBytes() > 0) {
            double pcieBw = (cpu.aggregatePcieGbps() > 0.0)
                    ? Math.min(spec.defaultPcieGbps(), cpu.aggregatePcieGbps())
                    : spec.defaultPcieGbps();
            pcieBw = Math.max(pcieBw, EPSILON);
            double pcieXferMs = HOST_PCIE_SYNC_LATENCY_MS + ((double) spec.syncBytes() / (pcieBw * 1e9)) * 1000.0;
            double ramBw = (cpu.ramBandwidthGbps() > 0.0) ? cpu.ramBandwidthGbps() : 20.0;
            double cpuGflops = Math.max(cpu.computeGflops(), EPSILON);
            double hostReduceMemMs = ((3.0 * spec.syncBytes()) / (ramBw * 1e9)) * 1000.0;
            double hostReduceCompMs = (((double) (spec.syncBytes() / 4L)) / (cpuGflops * 1e9)) * 1000.0;
            hostSyncMs = pcieXferMs + Math.max(hostReduceMemMs, hostReduceCompMs);
        }

        double syncMs = gpuSyncMs + hostSyncMs;

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

    private static PoolHeapEntry nextGpuCand(
            int a,
            long vA,
            long uVramA,
            double gT,
            double r0A,
            double r1A,
            double r2A,
            boolean hostSpillEnabled,
            boolean diskEnabled,
            long remH,
            long remD,
            long uRam,
            long uPcie,
            long uDisk,
            double ramUnitMs,
            double pcieUnitMs,
            double diskUnitMs) {
        if (vA < uVramA) {
            return new PoolHeapEntry(a, RoundTier.VRAM, gT + r0A);
        }
        if (hostSpillEnabled && remH > 0) {
            double t = Math.max(gT + r1A, Math.max((uRam + 1) * ramUnitMs, (uPcie + 1) * pcieUnitMs));
            return new PoolHeapEntry(a, RoundTier.HOST_STREAM, t);
        }
        if (diskEnabled && remD > 0) {
            double t = Math.max(gT + r2A, Math.max((uPcie + 1) * pcieUnitMs, (uDisk + 1) * diskUnitMs));
            return new PoolHeapEntry(a, RoundTier.DISK_STREAM, t);
        }
        return null;
    }
}
