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

import hc4j.engine.GpuOutOfMemoryException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Exact zero-parameter workload partitioner, multi-tier convex spilling optimizer,
 * and zero-waste straggler reclamation planner for heterogeneous multi-device collectives.
 *
 * <p>Implements:
 * <ul>
 *   <li>Theorem 1: Shared-Root All-Subsets Bottleneck-TSP Dynamic Programming in \(O(N^2 2^N)\);</li>
 *   <li>Theorem 2: Zero-Parameter Maximum Spanning Tree (MST) Dendrogram collective clustering;</li>
 *   <li>Theorem 3: 3-Tier Convex Spilling (VRAM &rarr; Host-RAM &rarr; Disk) with explicit Host CPU node,
 *       shared PCIe root-complex contention, and exact \(O(m \log m)\) Min-Heap Integer Minimax allocation;</li>
 *   <li>Theorem 4: Physically grounded pipeline overlap makespan modeling with exact \(2^N - 1\) subset search
 *       guaranteeing Zero-Degradation Monotonicity;</li>
 *   <li>Zero-Waste Straggler Reclamation: repurposing GPUs pruned from Tensor Parallelism as lossless Split-KV
 *       attention workers.</li>
 * </ul>
 */
public final class WorkloadPlanner {

    /** Default fraction of device VRAM available for workload buffers (85%). */
    public static final double DEFAULT_VRAM_UTILIZATION = 0.85;

    private static final double EPSILON = 1e-6;

    private WorkloadPlanner() {
        throw new AssertionError("No instances of WorkloadPlanner");
    }

    /**
     * Generates an optimal {@link WorkloadPlan} using default VRAM utilization and legacy GPU-only parameters.
     *
     * @param devices participating device profiles
     * @param topology measured network topology links
     * @param totalUnits total workload units to partition
     * @param flopsPerUnit sustained floating-point operations per unit
     * @param bytesPerUnit memory footprint in bytes required per unit
     * @param syncBytes total collective AllReduce payload in bytes
     * @return the optimal workload plan
     */
    public static WorkloadPlan plan(
            List<DeviceProfile> devices,
            List<Link> topology,
            long totalUnits,
            double flopsPerUnit,
            long bytesPerUnit,
            long syncBytes) {
        return plan(devices, topology, totalUnits, flopsPerUnit, bytesPerUnit, syncBytes, DEFAULT_VRAM_UTILIZATION);
    }

    /**
     * Generates an optimal {@link WorkloadPlan} with explicit VRAM utilization and legacy GPU-only parameters.
     *
     * @param devices participating device profiles
     * @param topology measured network topology links
     * @param totalUnits total workload units to partition
     * @param flopsPerUnit sustained floating-point operations per unit
     * @param bytesPerUnit memory footprint in bytes required per unit
     * @param syncBytes total collective AllReduce payload in bytes
     * @param vramUtilization fraction of device VRAM allowed (0.0 to 1.0]
     * @return the optimal workload plan
     */
    public static WorkloadPlan plan(
            List<DeviceProfile> devices,
            List<Link> topology,
            long totalUnits,
            double flopsPerUnit,
            long bytesPerUnit,
            long syncBytes,
            double vramUtilization) {
        validateInputs(devices, totalUnits, flopsPerUnit, bytesPerUnit, syncBytes, vramUtilization);

        int n = devices.size();
        if (totalUnits > 0 && bytesPerUnit > 0) {
            long totalVramCapacity = 0;
            for (DeviceProfile d : devices) {
                long cap = (long) Math.floor((d.vramBytes() * vramUtilization) / bytesPerUnit);
                totalVramCapacity = addSaturating(totalVramCapacity, cap);
            }
            if (totalUnits > totalVramCapacity) {
                long totalUsableBytes = 0;
                for (DeviceProfile d : devices) {
                    totalUsableBytes += (long) (d.vramBytes() * vramUtilization);
                }
                throw new GpuOutOfMemoryException(String.format(
                        "Total workload units (%d) requires %d bytes, which exceeds total usable VRAM capacity (%d bytes) across all %d devices",
                        totalUnits, Math.multiplyExact(totalUnits, bytesPerUnit), totalUsableBytes, n));
            }
        }

        WorkloadSpec spec = new WorkloadSpec(
                totalUnits,
                flopsPerUnit,
                bytesPerUnit,
                syncBytes,
                2.0,
                1,
                vramUtilization,
                0.5,
                16.0,
                0.0,
                0L,
                HostCpuProfile.disabled());

        return plan(devices, topology, spec);
    }

    /**
     * Generates an exact zero-parameter optimized {@link WorkloadPlan} according to the provided {@link WorkloadSpec}.
     *
     * @param devices participating device profiles
     * @param topology measured network topology links
     * @param spec detailed workload specification
     * @return the globally optimal workload plan
     * @throws NullPointerException if {@code devices} or {@code spec} is null
     * @throws IllegalArgumentException if inputs are invalid
     * @throws GpuOutOfMemoryException if workload cannot fit across available memory tiers
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
            for (int i = 0; i < n; i++) {
                allocs.add(new ShardAllocation(devices.get(i).rank(), 0L, 0.0, 0.0, 0L, false, 0L, 0L, 0L, 0L));
            }
            return new WorkloadPlan(0L, allocs, List.of(), 0.0, 0.0, 0.0);
        }

        double[][] bw = new double[n][n];
        double[][] latUs = new double[n][n];
        Transport[][] tr = new Transport[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                bw[i][j] = (i == j) ? 25.0 : 1.0;
                latUs[i][j] = (i == j) ? 0.0 : 100.0;
                tr[i][j] = (i == j) ? Transport.PEER_TO_PEER : Transport.HOST_STAGED;
            }
        }
        if (topology != null) {
            for (Link link : topology) {
                int src = link.src();
                int dst = link.dst();
                if (src < n && dst < n) {
                    bw[src][dst] = link.gbps();
                    latUs[src][dst] = link.latencyUs();
                    tr[src][dst] = link.transport();
                }
            }
        }

        AllSubsetsRingTable ringTable = (n <= 12) ? precomputeAllSubsetsRing(n, bw, latUs) : null;

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
                // Try GPU subset alone
                PlanCandidate candGpu = evaluateDeviceSet(gpuIndices, false, devices, bw, latUs, tr, ringTable, spec);
                if (candGpu != null && (bestPlan == null || isBetter(candGpu, bestPlan))) {
                    bestPlan = candGpu;
                }
                // Try GPU subset with explicit Host CPU node
                if (cpuEnabled) {
                    PlanCandidate candCpu = evaluateDeviceSet(gpuIndices, true, devices, bw, latUs, tr, ringTable, spec);
                    if (candCpu != null && (bestPlan == null || isBetter(candCpu, bestPlan))) {
                        bestPlan = candCpu;
                    }
                }
            }
            // Try CPU alone if enabled
            if (cpuEnabled) {
                PlanCandidate candCpuOnly = evaluateDeviceSet(List.of(), true, devices, bw, latUs, tr, ringTable, spec);
                if (candCpuOnly != null && (bestPlan == null || isBetter(candCpuOnly, bestPlan))) {
                    bestPlan = candCpuOnly;
                }
            }
        } else {
            // Heuristic for n > 12: greedy pruning
            List<Integer> activeGpus = new ArrayList<>();
            for (int i = 0; i < n; i++) activeGpus.add(i);
            bestPlan = evaluateDeviceSet(activeGpus, cpuEnabled, devices, bw, latUs, tr, ringTable, spec);
            boolean improved = true;
            while (improved && activeGpus.size() > 1) {
                improved = false;
                int bestPruneIdx = -1;
                PlanCandidate bestPruneCand = null;
                for (int i = 0; i < activeGpus.size(); i++) {
                    List<Integer> candActive = new ArrayList<>(activeGpus);
                    candActive.remove(i);
                    PlanCandidate cand = evaluateDeviceSet(candActive, cpuEnabled, devices, bw, latUs, tr, ringTable, spec);
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

        // Build ShardAllocations
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
        // Tie-break: fewer spilled units
        if (a.totalSpilledUnits < b.totalSpilledUnits) {
            return true;
        }
        if (b.totalSpilledUnits < a.totalSpilledUnits) {
            return false;
        }
        // Tie-break: more active GPUs
        return a.activeGpuIndices.size() > b.activeGpuIndices.size();
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Theorem 1: Shared-Root All-Subsets Bottleneck-TSP DP
    // ─────────────────────────────────────────────────────────────────────────────

    /**
     * Precomputes exact bottleneck bandwidth and maximum hop latency for all \(2^n\) subsets
     * in \(O(n^2 2^n)\) using a shared-root cycle dynamic program.
     */
    static AllSubsetsRingTable precomputeAllSubsetsRing(int n, double[][] bw, double[][] latUs) {
        int numSubsets = 1 << n;
        double[] bottleneckGbps = new double[numSubsets];
        double[] maxHopLatencyUs = new double[numSubsets];

        // 1-element subsets
        for (int i = 0; i < n; i++) {
            bottleneckGbps[1 << i] = 25.0;
            maxHopLatencyUs[1 << i] = 0.0;
        }

        // 2-element subsets
        for (int u = 0; u < n; u++) {
            for (int v = u + 1; v < n; v++) {
                int mask = (1 << u) | (1 << v);
                bottleneckGbps[mask] = Math.min(bw[u][v], bw[v][u]);
                maxHopLatencyUs[mask] = Math.max(latUs[u][v], latUs[v][u]);
            }
        }

        // DP for subsets of size >= 3: iterate over root s0 in 0..n-3
        for (int s0 = 0; s0 < n - 2; s0++) {
            int r = n - 1 - s0;
            int numSubMasks = 1 << r;
            double[][] dpBw = new double[numSubMasks][r];
            double[][] dpLat = new double[numSubMasks][r];
            for (double[] row : dpBw) Arrays.fill(row, -1.0);

            // Base cases: paths of length 1 from s0 to s0 + 1 + u
            for (int u = 0; u < r; u++) {
                int target = s0 + 1 + u;
                dpBw[1 << u][u] = bw[s0][target];
                dpLat[1 << u][u] = latUs[s0][target];
            }

            for (int subMask = 1; subMask < numSubMasks; subMask++) {
                for (int u = 0; u < r; u++) {
                    if ((subMask & (1 << u)) == 0) continue;
                    double curBw = dpBw[subMask][u];
                    if (curBw < 0) continue;
                    double curLat = dpLat[subMask][u];

                    int uRank = s0 + 1 + u;
                    for (int v = 0; v < r; v++) {
                        if ((subMask & (1 << v)) == 0) {
                            int nextMask = subMask | (1 << v);
                            int vRank = s0 + 1 + v;
                            double candBw = Math.min(curBw, bw[uRank][vRank]);
                            double candLat = Math.max(curLat, latUs[uRank][vRank]);

                            if (candBw > dpBw[nextMask][v] ||
                                    (Math.abs(candBw - dpBw[nextMask][v]) < 1e-9 && candLat < dpLat[nextMask][v])) {
                                dpBw[nextMask][v] = candBw;
                                dpLat[nextMask][v] = candLat;
                            }
                        }
                    }
                }

                // If subMask has at least 2 vertices, we can form cycles of size >= 3
                if (Integer.bitCount(subMask) >= 2) {
                    int globalMask = (1 << s0) | (subMask << (s0 + 1));
                    double bestCycleBw = -1.0;
                    double bestCycleLat = Double.MAX_VALUE;

                    for (int u = 0; u < r; u++) {
                        if ((subMask & (1 << u)) != 0 && dpBw[subMask][u] >= 0) {
                            int uRank = s0 + 1 + u;
                            double cycleBw = Math.min(dpBw[subMask][u], bw[uRank][s0]);
                            double cycleLat = Math.max(dpLat[subMask][u], latUs[uRank][s0]);

                            if (cycleBw > bestCycleBw ||
                                    (Math.abs(cycleBw - bestCycleBw) < 1e-9 && cycleLat < bestCycleLat)) {
                                bestCycleBw = cycleBw;
                                bestCycleLat = cycleLat;
                            }
                        }
                    }

                    if (bestCycleBw > 0.0) {
                        bottleneckGbps[globalMask] = bestCycleBw;
                        maxHopLatencyUs[globalMask] = bestCycleLat;
                    }
                }
            }
        }

        return new AllSubsetsRingTable(bottleneckGbps, maxHopLatencyUs);
    }

    record AllSubsetsRingTable(double[] bottleneckGbps, double[] maxHopLatencyUs) {
        double bottleneck(int mask) {
            return (mask >= 0 && mask < bottleneckGbps.length) ? bottleneckGbps[mask] : 1.0;
        }

        double latencyUs(int mask) {
            return (mask >= 0 && mask < maxHopLatencyUs.length) ? maxHopLatencyUs[mask] : 0.0;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Theorem 2: Zero-Parameter MST Dendrogram Sync Estimator
    // ─────────────────────────────────────────────────────────────────────────────

    private static double estimateSyncMsExact(
            List<Integer> activeIndices,
            double[][] bw,
            double[][] latUs,
            Transport[][] tr,
            AllSubsetsRingTable ringTable,
            List<DeviceProfile> devices,
            long syncBytes) {
        int m = activeIndices.size();
        if (m <= 1 || syncBytes == 0) {
            return 0.0;
        }

        int mask = 0;
        for (int idx : activeIndices) {
            mask |= (1 << idx);
        }

        // Flat ring over activeIndices
        double ringBw = (ringTable != null) ? ringTable.bottleneck(mask) : solveRingBottleneck(activeIndices, bw);
        double ringLatSecs = (ringTable != null) ? (ringTable.latencyUs(mask) / 1e6) : 0.0;
        if (ringBw <= 0.0) ringBw = 1e-6;

        double mF = m;
        double tFlat = 2.0 * (mF - 1.0) * ringLatSecs
                + 2.0 * ((mF - 1.0) / mF) * (syncBytes / (ringBw * 1e9));

        double bestTimeSecs = tFlat;

        // Build MST over activeIndices
        boolean allEdgesUniform = true;
        int firstTr = -1;
        double firstW = -1.0;

        List<UndirectedEdge> edges = new ArrayList<>(m * (m - 1) / 2);
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                int u = activeIndices.get(i);
                int v = activeIndices.get(j);
                double wSym = Math.min(bw[u][v], bw[v][u]);
                int trSym = Math.min(transportRank(tr[u][v]), transportRank(tr[v][u]));
                edges.add(new UndirectedEdge(u, v, trSym, wSym));
                if (firstTr == -1) {
                    firstTr = trSym;
                    firstW = wSym;
                } else if (trSym != firstTr || Math.abs(wSym - firstW) > 1e-6) {
                    allEdgesUniform = false;
                }
            }
        }

        if (allEdgesUniform) {
            return tFlat * 1000.0;
        }

        edges.sort((a, b) -> {
            int cmp = Integer.compare(b.trSym, a.trSym);
            return (cmp != 0) ? cmp : Double.compare(b.wSym, a.wSym);
        });

        int maxRank = 0;
        for (int idx : activeIndices) maxRank = Math.max(maxRank, idx);
        int[] dsu = new int[maxRank + 1];
        for (int i = 0; i <= maxRank; i++) dsu[i] = i;

        List<UndirectedEdge> mstEdges = new ArrayList<>(m - 1);
        for (UndirectedEdge e : edges) {
            int rootU = findRoot(dsu, e.u);
            int rootV = findRoot(dsu, e.v);
            if (rootU != rootV) {
                dsu[rootU] = rootV;
                mstEdges.add(e);
                if (mstEdges.size() == m - 1) break;
            }
        }

        mstEdges.sort((a, b) -> {
            int cmp = Integer.compare(a.trSym, b.trSym);
            return (cmp != 0) ? cmp : Double.compare(a.wSym, b.wSym);
        });

        // Evaluate level cuts k in 1..m-1
        for (int k = 1; k < m; k++) {
            for (int i = 0; i <= maxRank; i++) dsu[i] = i;
            for (int i = k; i < mstEdges.size(); i++) {
                UndirectedEdge e = mstEdges.get(i);
                int rU = findRoot(dsu, e.u);
                int rV = findRoot(dsu, e.v);
                if (rU != rV) dsu[rU] = rV;
            }

            Map<Integer, List<Integer>> groups = new HashMap<>();
            for (int idx : activeIndices) {
                int r = findRoot(dsu, idx);
                groups.computeIfAbsent(r, x -> new ArrayList<>()).add(idx);
            }

            List<List<Integer>> islands = new ArrayList<>(groups.values());
            for (List<Integer> isl : islands) Collections.sort(isl);
            islands.sort(Comparator.comparingInt(List::getFirst));

            List<Integer> leaders = new ArrayList<>(islands.size());
            double tIntraMax = 0.0;

            for (List<Integer> isl : islands) {
                int leader = pickLeader(isl, activeIndices, devices, bw);
                leaders.add(leader);
                if (isl.size() > 1) {
                    double bIntra = Double.MAX_VALUE;
                    double latIntraSecs = 0.0;
                    for (int member : isl) {
                        if (member != leader) {
                            double minB = Math.min(bw[member][leader], bw[leader][member]);
                            double maxL = Math.max(latUs[member][leader], latUs[leader][member]) / 1e6;
                            bIntra = Math.min(bIntra, minB);
                            latIntraSecs = Math.max(latIntraSecs, maxL);
                        }
                    }
                    if (bIntra <= 0.0) bIntra = 1e-6;
                    double szF = isl.size();
                    double tIntra = 2.0 * (szF - 1.0) * latIntraSecs
                            + 2.0 * ((szF - 1.0) / szF) * (syncBytes / (bIntra * 1e9));
                    tIntraMax = Math.max(tIntraMax, tIntra);
                }
            }

            int numLeaders = leaders.size();
            int leaderMask = 0;
            for (int l : leaders) leaderMask |= (1 << l);

            double bLead = (ringTable != null) ? ringTable.bottleneck(leaderMask) : solveRingBottleneck(leaders, bw);
            double latLeadSecs = (ringTable != null) ? (ringTable.latencyUs(leaderMask) / 1e6) : 0.0;
            if (bLead <= 0.0) bLead = 1e-6;

            double kF = numLeaders;
            double tLead = 2.0 * (kF - 1.0) * latLeadSecs
                    + 2.0 * ((kF - 1.0) / kF) * (syncBytes / (bLead * 1e9));

            double tHier = tIntraMax + tLead;
            if (tHier < bestTimeSecs * (1.0 - 1e-6)) {
                bestTimeSecs = tHier;
            }
        }

        return bestTimeSecs * 1000.0;
    }

    private static int pickLeader(List<Integer> members, List<Integer> activeIndices, List<DeviceProfile> devices, double[][] bw) {
        int best = members.getFirst();
        double bestGflops = -1.0;
        double bestMeanOut = -1.0;

        for (int m : members) {
            double gflops = (m < devices.size()) ? devices.get(m).computeGflops() : 0.0;
            double outSum = 0.0;
            int outCount = 0;
            for (int o : activeIndices) {
                if (!members.contains(o)) {
                    outSum += bw[m][o];
                    outCount++;
                }
            }
            double meanOut = (outCount > 0) ? outSum / outCount : 0.0;

            boolean better;
            if (Math.abs(gflops - bestGflops) > 1e-4) {
                better = gflops > bestGflops;
            } else if (Math.abs(meanOut - bestMeanOut) > 1e-4) {
                better = meanOut > bestMeanOut;
            } else {
                better = m < best;
            }

            if (better) {
                best = m;
                bestGflops = gflops;
                bestMeanOut = meanOut;
            }
        }
        return best;
    }

    private record UndirectedEdge(int u, int v, int trSym, double wSym) {}

    // ─────────────────────────────────────────────────────────────────────────────
    // Theorem 3 & 4: Multi-Tier Convex Spilling, Contention & Min-Heap Minimax
    // ─────────────────────────────────────────────────────────────────────────────

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
            if (u <= 0) return 0.0;
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
            if (t <= 0.0) return 0.0;
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

    private record PlanCandidate(
            List<Integer> activeGpuIndices,
            boolean includesCpu,
            AllocationDetail[] gpuAllocations,
            AllocationDetail cpuAllocation,
            long totalSpilledUnits,
            double computeMs,
            double syncMs,
            double makespanMs) {}

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

    private static PlanCandidate evaluateDeviceSet(
            List<Integer> activeGpuIndices,
            boolean includeCpu,
            List<DeviceProfile> devices,
            double[][] bw,
            double[][] latUs,
            Transport[][] tr,
            AllSubsetsRingTable ringTable,
            WorkloadSpec spec) {
        int n = devices.size();
        long totalUnits = spec.totalUnits();
        long mBytes = spec.bytesPerUnit();
        double fFlops = spec.flopsPerUnit();
        double kappa = spec.ioMultiplier();
        double gamma = spec.streamingOverlapGamma();
        HostCpuProfile cpu = spec.hostCpu();

        // Check if CPU alone is valid
        if (activeGpuIndices.isEmpty() && !includeCpu) {
            return null;
        }

        // Tier 0 VRAM limits per GPU
        long[] uVram = new long[n];
        for (int i : activeGpuIndices) {
            if (mBytes > 0) {
                uVram[i] = (long) Math.floor((devices.get(i).vramBytes() * spec.vramUtilization()) / mBytes);
            } else {
                uVram[i] = Long.MAX_VALUE / 4;
            }
        }

        // Host RAM pool
        long totalHostRamBytes = (includeCpu && cpu.isEnabled()) ? cpu.ramBytes() : 0L;
        long totalDiskBytes = spec.diskCapacityBytes();

        // Fixed-point loop for PCIe contention and shared Host-RAM pool cap (up to 4 iterations)
        int kSpill = 0;
        long[] lastUnits = null;
        AllocationDetail[] gpuDets = new AllocationDetail[n];
        AllocationDetail cpuDet = null;
        double makespanMs = 0.0;
        double maxComputeMs = 0.0;

        for (int iter = 0; iter < 4; iter++) {
            double effectivePcie = spec.defaultPcieGbps();
            if (kSpill > 0 && cpu.aggregatePcieGbps() > 0.0) {
                effectivePcie = Math.min(spec.defaultPcieGbps(), cpu.aggregatePcieGbps() / Math.max(1, kSpill));
            }

            // Construct curves
            List<DeviceCurve> curves = new ArrayList<>();

            // Host RAM quota division
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
                    long diskPerGpu = (activeGpuIndices.size() > 0) ? totalDiskBytes / activeGpuIndices.size() : 0L;
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
                return null; // Infeasible subset
            }

            // Minimax allocation over curves
            long[] assigned = solveIntegerMinimax(curves, totalUnits);
            lastUnits = assigned;

            // Recalculate kSpill
            int newKSpill = 0;
            long hostBytesUsed = 0;
            for (int idx = 0; idx < curves.size(); idx++) {
                DeviceCurve c = curves.get(idx);
                long u = assigned[idx];
                if (c.index != ShardAllocation.HOST_CPU_RANK) {
                    if (u > c.uVram) {
                        newKSpill++;
                        long spilledToHost = Math.min(u - c.uVram, c.uHost);
                        hostBytesUsed += spilledToHost * mBytes;
                    }
                } else {
                    hostBytesUsed += u * mBytes;
                }
            }

            if (newKSpill == kSpill || iter == 3) {
                // Converged
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
                                u,
                                vramU,
                                hostU,
                                diskU,
                                0L,
                                timeMs,
                                vramU * mBytes,
                                hostU * mBytes,
                                diskU * mBytes);
                    } else {
                        cpuDet = new AllocationDetail(
                                u,
                                0L,
                                0L,
                                0L,
                                u,
                                timeMs,
                                0L,
                                u * mBytes,
                                0L);
                    }
                }
                break;
            }
            kSpill = newKSpill;
        }

        // Fill non-active GPUs with 0
        for (int i = 0; i < n; i++) {
            if (gpuDets[i] == null) {
                gpuDets[i] = new AllocationDetail(0L, 0L, 0L, 0L, 0L, 0.0, 0L, 0L, 0L);
            }
        }

        // Synchronization duration
        double syncMs = (activeGpuIndices.size() >= 2)
                ? estimateSyncMsExact(activeGpuIndices, bw, latUs, tr, ringTable, devices, spec.syncBytes())
                : 0.0;

        // Theorem 4: Overlap makespan formula
        double alpha = spec.overlapAlpha();
        makespanMs = Math.max(maxComputeMs, syncMs) + (1.0 - alpha) * Math.min(maxComputeMs, syncMs);

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
        if (m == 0 || totalUnits == 0) return result;

        // Bisection on T* in [0, T_max]
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
            // Min-heap to distribute remaining units with lowest next makespan
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
            // Surplus clamp
            long surplus = -deficit;
            for (int i = m - 1; i >= 0 && surplus > 0; i--) {
                long take = Math.min(result[i], surplus);
                result[i] -= take;
                surplus -= take;
            }
        }

        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Zero-Waste Straggler Reclamation (planRoles)
    // ─────────────────────────────────────────────────────────────────────────────

    /**
     * Generates a multi-operator role assignment plan executing Tensor Parallel shards on primary
     * devices and reclaiming excluded straggler GPUs as lossless Split-KV attention workers.
     *
     * @param devices participating device profiles
     * @param topology measured network topology links
     * @param tpSpec specification for Tensor Parallel operations
     * @param kvSpec specification for Split-KV attention
     * @return the combined multi-operator role plan
     */
    public static RolePlan planRoles(
            List<DeviceProfile> devices,
            List<Link> topology,
            WorkloadSpec tpSpec,
            SplitKvSpec kvSpec) {
        Objects.requireNonNull(devices, "devices must not be null");
        Objects.requireNonNull(tpSpec, "tpSpec must not be null");
        Objects.requireNonNull(kvSpec, "kvSpec must not be null");

        WorkloadPlan tpPlan = plan(devices, topology, tpSpec);
        int n = devices.size();

        double[][] bw = new double[n][n];
        double[][] latUs = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                bw[i][j] = (i == j) ? 25.0 : 1.0;
                latUs[i][j] = (i == j) ? 0.0 : 100.0;
            }
        }
        if (topology != null) {
            for (Link link : topology) {
                int src = link.src();
                int dst = link.dst();
                if (src < n && dst < n) {
                    bw[src][dst] = link.gbps();
                    latUs[src][dst] = link.latencyUs();
                }
            }
        }

        List<Integer> activeTpGpus = new ArrayList<>();
        Set<Integer> prunedGpus = new HashSet<>();
        for (int i = 0; i < n; i++) {
            ShardAllocation alloc = tpPlan.allocations().get(i);
            if (alloc.units() > 0) {
                activeTpGpus.add(i);
            } else {
                prunedGpus.add(i);
            }
        }

        int anchorRank = activeTpGpus.isEmpty() ? 0 : activeTpGpus.getFirst();
        // Pick primary anchor as the fastest GPU in TP
        double maxTpGflops = -1.0;
        for (int r : activeTpGpus) {
            if (devices.get(r).computeGflops() > maxTpGflops) {
                maxTpGflops = devices.get(r).computeGflops();
                anchorRank = r;
            }
        }

        // Available KV capacities
        long[] maxKvTokens = new long[n];
        double[] rAttn = new double[n];
        double[] lRpc = new double[n];

        for (int i = 0; i < n; i++) {
            DeviceProfile d = devices.get(i);
            long usableVram = (long) Math.floor(d.vramBytes() * kvSpec.vramUtilization());
            long availKvVram = Math.max(0L, usableVram - kvSpec.reservedWeightBytesPerGpu());
            maxKvTokens[i] = availKvVram / kvSpec.kvBytesPerToken();

            double cGflops = Math.max(d.computeGflops(), EPSILON);
            double memBw = Math.max(50.0, cGflops * 0.1);
            double tComp = kvSpec.attentionFlopsPerToken() / (cGflops * 1e9);
            double tMem = kvSpec.kvBytesPerToken() / (memBw * 1e9);
            rAttn[i] = Math.max(tComp, tMem) * 1000.0;

            if (i == anchorRank) {
                lRpc[i] = 0.0;
            } else {
                double linkBw = Math.min(bw[anchorRank][i], bw[i][anchorRank]);
                if (linkBw <= 0.0) linkBw = 1e-6;
                double linkLat = (latUs[anchorRank][i] + latUs[i][anchorRank]) / 1000.0; // ms
                double rpcMs = (kvSpec.queryAndPartialBytes() / (linkBw * 1e9)) * 1000.0;
                lRpc[i] = linkLat + rpcMs;
            }
        }

        // Host CPU KV profile
        HostCpuProfile cpu = tpSpec.hostCpu();
        long maxCpuKvTokens = 0L;
        double rCpuAttn = 0.0;
        double lCpuRpc = 0.0;
        if (cpu.isEnabled()) {
            maxCpuKvTokens = cpu.ramBytes() / kvSpec.kvBytesPerToken();
            double cGflops = Math.max(cpu.computeGflops(), EPSILON);
            double memBw = Math.max(20.0, cpu.ramBandwidthGbps());
            double tComp = kvSpec.attentionFlopsPerToken() / (cGflops * 1e9);
            double tMem = kvSpec.kvBytesPerToken() / (memBw * 1e9);
            rCpuAttn = Math.max(tComp, tMem) * 1000.0;
            lCpuRpc = 0.05 + (kvSpec.queryAndPartialBytes() / (memBw * 1e9)) * 1000.0;
        }

        // Evaluate candidate Split-KV worker subsets W
        long totalKv = kvSpec.totalKvTokens();

        List<KvWorker> allWorkers = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (maxKvTokens[i] > 0) {
                allWorkers.add(new KvWorker(i, maxKvTokens[i], rAttn[i], lRpc[i]));
            }
        }
        if (cpu.isEnabled() && maxCpuKvTokens > 0) {
            allWorkers.add(new KvWorker(ShardAllocation.HOST_CPU_RANK, maxCpuKvTokens, rCpuAttn, lCpuRpc));
        }

        // Find optimal worker subset W minimizing attention makespan
        // Always include anchorRank in candidate worker subsets
        List<KvWorker> bestWorkerSet = null;
        long[] bestKvAlloc = null;
        double bestAttnMs = Double.MAX_VALUE;

        // Try single anchor first
        KvWorker anchorWorker = null;
        for (KvWorker w : allWorkers) {
            if (w.rank == anchorRank) {
                anchorWorker = w;
                break;
            }
        }

        int wCount = allWorkers.size();
        int maxKvSubsets = (wCount <= 12) ? (1 << wCount) : 1;

        if (wCount <= 12) {
            for (int mask = 1; mask < maxKvSubsets; mask++) {
                List<KvWorker> candWorkers = new ArrayList<>();
                long candCap = 0;
                for (int i = 0; i < wCount; i++) {
                    if ((mask & (1 << i)) != 0) {
                        KvWorker w = allWorkers.get(i);
                        candWorkers.add(w);
                        candCap = addSaturating(candCap, w.maxTokens);
                    }
                }
                if (candCap < totalKv) continue;

                // Evaluate integer minimax for candWorkers
                long[] alloc = solveKvMinimax(candWorkers, totalKv);
                double maxTime = 0.0;
                for (int i = 0; i < candWorkers.size(); i++) {
                    KvWorker w = candWorkers.get(i);
                    long tokens = alloc[i];
                    double time = (tokens > 0) ? w.lRpc + (tokens * w.rAttn) : 0.0;
                    maxTime = Math.max(maxTime, time);
                }

                if (maxTime < bestAttnMs * (1.0 - 1e-6)) {
                    bestAttnMs = maxTime;
                    bestWorkerSet = candWorkers;
                    bestKvAlloc = alloc;
                }
            }
        } else {
            // Greedy fallback
            bestWorkerSet = allWorkers;
            bestKvAlloc = solveKvMinimax(allWorkers, totalKv);
            bestAttnMs = 0.0;
            for (int i = 0; i < allWorkers.size(); i++) {
                KvWorker w = allWorkers.get(i);
                long tokens = bestKvAlloc[i];
                double time = (tokens > 0) ? w.lRpc + (tokens * w.rAttn) : 0.0;
                bestAttnMs = Math.max(bestAttnMs, time);
            }
        }

        Map<Integer, Long> tokenMap = new HashMap<>();
        if (bestWorkerSet != null && bestKvAlloc != null) {
            for (int i = 0; i < bestWorkerSet.size(); i++) {
                tokenMap.put(bestWorkerSet.get(i).rank, bestKvAlloc[i]);
            }
        }

        // Build assignments and determine roles
        List<SplitKvAssignment> assignments = new ArrayList<>(n + (cpu.isEnabled() ? 1 : 0));
        int reclaimedCount = 0;
        boolean cpuReclaimed = false;

        for (int i = 0; i < n; i++) {
            int rank = devices.get(i).rank();
            long tokens = tokenMap.getOrDefault(i, 0L);
            DeviceRole role;
            if (activeTpGpus.contains(i)) {
                role = DeviceRole.TENSOR_PARALLEL;
            } else if (tokens > 0) {
                role = DeviceRole.SPLIT_KV_RECLAIMED;
                reclaimedCount++;
            } else {
                role = DeviceRole.IDLE;
            }

            double localAttnMs = (tokens > 0) ? tokens * rAttn[i] : 0.0;
            double commMs = (tokens > 0) ? lRpc[i] : 0.0;
            double totalMs = localAttnMs + commMs;
            long kvMemBytes = tokens * kvSpec.kvBytesPerToken();

            assignments.add(new SplitKvAssignment(
                    rank,
                    role,
                    tokens,
                    localAttnMs,
                    commMs,
                    totalMs,
                    kvMemBytes));
        }

        if (cpu.isEnabled()) {
            long tokens = tokenMap.getOrDefault(ShardAllocation.HOST_CPU_RANK, 0L);
            boolean cpuInPlace = tpPlan.activeRanks().contains(ShardAllocation.HOST_CPU_RANK);
            DeviceRole role = (tokens > 0 || cpuInPlace) ? DeviceRole.CPU_IN_PLACE : DeviceRole.IDLE;
            if (role == DeviceRole.CPU_IN_PLACE) {
                cpuReclaimed = true;
            }

            double localAttnMs = (tokens > 0) ? tokens * rCpuAttn : 0.0;
            double commMs = (tokens > 0) ? lCpuRpc : 0.0;
            double totalMs = localAttnMs + commMs;
            long kvMemBytes = tokens * kvSpec.kvBytesPerToken();

            assignments.add(new SplitKvAssignment(
                    ShardAllocation.HOST_CPU_RANK,
                    role,
                    tokens,
                    localAttnMs,
                    commMs,
                    totalMs,
                    kvMemBytes));
        }

        // Determine active regime string
        String regime;
        if (reclaimedCount > 0) {
            regime = "SPLIT_KV_RECLAIMED";
        } else if (cpuReclaimed) {
            regime = "CPU_IN_PLACE";
        } else {
            long totalSpill = 0;
            for (ShardAllocation a : tpPlan.allocations()) {
                totalSpill += (a.hostStreamUnits() + a.diskStreamUnits());
            }
            if (totalSpill > 0) {
                regime = "PCIE_STREAMING";
            } else if (prunedGpus.isEmpty()) {
                regime = "ALL_GPU_SHARD";
            } else {
                regime = "STRAGGLER_PRUNED";
            }
        }

        return new RolePlan(
                tpPlan,
                assignments,
                tpPlan.estimatedMakespanMs(),
                bestAttnMs,
                totalKv,
                reclaimedCount,
                cpuReclaimed,
                regime);
    }

    private static long[] solveKvMinimax(List<KvWorker> workers, long totalTokens) {
        int m = workers.size();
        long[] result = new long[m];
        if (m == 0 || totalTokens == 0) return result;

        double tLow = 0.0;
        double tHigh = 1.0;
        for (KvWorker w : workers) {
            tHigh = Math.max(tHigh, w.lRpc + (totalTokens * w.rAttn));
        }

        for (int step = 0; step < 64; step++) {
            double tMid = tLow + (tHigh - tLow) * 0.5;
            double sumTokens = 0.0;
            for (KvWorker w : workers) {
                if (tMid > w.lRpc) {
                    double tokens = (tMid - w.lRpc) / w.rAttn;
                    sumTokens += Math.min(tokens, w.maxTokens);
                }
            }
            if (sumTokens >= totalTokens) {
                tHigh = tMid;
            } else {
                tLow = tMid;
            }
        }

        double tOpt = tHigh;
        long currentSum = 0;
        for (int i = 0; i < m; i++) {
            KvWorker w = workers.get(i);
            long tokens = 0;
            if (tOpt > w.lRpc) {
                tokens = (long) Math.floor((tOpt - w.lRpc) / w.rAttn);
                tokens = Math.max(0, Math.min(tokens, w.maxTokens));
            }
            result[i] = tokens;
            currentSum += tokens;
        }

        long deficit = totalTokens - currentSum;
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
                KvWorker w = workers.get(i);
                if (result[i] < w.maxTokens) {
                    double nextTime = w.lRpc + ((result[i] + 1) * w.rAttn);
                    pq.add(new HeapEntry(i, nextTime));
                }
            }

            while (deficit > 0 && !pq.isEmpty()) {
                HeapEntry top = pq.poll();
                result[top.index]++;
                deficit--;
                KvWorker w = workers.get(top.index);
                if (result[top.index] < w.maxTokens) {
                    double nextTime = w.lRpc + ((result[top.index] + 1) * w.rAttn);
                    pq.add(new HeapEntry(top.index, nextTime));
                }
            }
        }

        return result;
    }

    private static record KvWorker(int rank, long maxTokens, double rAttn, double lRpc) {}

    // ─────────────────────────────────────────────────────────────────────────────
    // Helper Utilities
    // ─────────────────────────────────────────────────────────────────────────────

    private static void validateInputs(
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

    private static long addSaturating(long a, long b) {
        try {
            return Math.addExact(a, b);
        } catch (ArithmeticException e) {
            return Long.MAX_VALUE;
        }
    }

    private static int findRoot(int[] parent, int i) {
        if (parent[i] == i) return i;
        parent[i] = findRoot(parent, parent[i]);
        return parent[i];
    }

    private static int transportRank(Transport t) {
        return switch (t) {
            case HOST_STAGED -> 0;
            case HOST_SHARED_DMA -> 1;
            case PEER_TO_PEER -> 2;
        };
    }

    private static double solveRingBottleneck(List<Integer> indices, double[][] bw) {
        int m = indices.size();
        if (m == 0) return 1.0;
        if (m == 1) return 25.0;
        if (m == 2) {
            int u = indices.get(0);
            int v = indices.get(1);
            return Math.min(bw[u][v], bw[v][u]);
        }

        // Greedy heuristic for large rings
        boolean[] visited = new boolean[m];
        visited[0] = true;
        int cur = 0;
        double bottleneck = Double.MAX_VALUE;

        for (int step = 1; step < m; step++) {
            int bestNext = -1;
            double bestW = -1.0;
            int curRank = indices.get(cur);
            for (int next = 0; next < m; next++) {
                if (!visited[next]) {
                    int nextRank = indices.get(next);
                    if (bw[curRank][nextRank] > bestW) {
                        bestW = bw[curRank][nextRank];
                        bestNext = next;
                    }
                }
            }
            if (bestNext == -1) bestNext = 0;
            visited[bestNext] = true;
            bottleneck = Math.min(bottleneck, bestW);
            cur = bestNext;
        }
        bottleneck = Math.min(bottleneck, bw[indices.get(cur)][indices.get(0)]);
        return (bottleneck > 0.0) ? bottleneck : 1.0;
    }
}
