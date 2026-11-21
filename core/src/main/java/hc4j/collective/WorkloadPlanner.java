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
import java.util.Set;

/**
 * Intelligent workload partitioner and makespan optimizer for heterogeneous multi-GPU collectives.
 *
 * <p>Implements:
 * <ul>
 *   <li>VRAM water-filling with per-device capacity capping;</li>
 *   <li>Compute-proportional unit allocation;</li>
 *   <li>Makespan modeling combining bottleneck compute time and ring / hierarchical AllReduce sync time;</li>
 *   <li>Iterative straggler pruning guaranteeing zero performance degradation when slow devices are present.</li>
 * </ul>
 */
public final class WorkloadPlanner {

    /** Default fraction of device VRAM available for workload buffers (85%). */
    public static final double DEFAULT_VRAM_UTILIZATION = 0.85;

    private static final double EPSILON = 1e-6;
    private static final double IMPROVEMENT_THRESHOLD = 0.995; // > 0.5% faster

    private WorkloadPlanner() {
        throw new AssertionError("No instances of WorkloadPlanner");
    }

    /**
     * Generates an optimal {@link WorkloadPlan} using default VRAM utilization.
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
     * Generates an optimal {@link WorkloadPlan} with explicit VRAM utilization.
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
        if (totalUnits == 0) {
            List<ShardAllocation> allocs = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                allocs.add(new ShardAllocation(devices.get(i).rank(), 0L, 0.0, 0.0, 0L, false));
            }
            return new WorkloadPlan(0L, allocs, List.of(), 0.0, 0.0, 0.0);
        }

        // 1. Capacity Cap per Device
        long[] maxUnits = new long[n];
        long totalCapacity = 0;
        for (int i = 0; i < n; i++) {
            if (bytesPerUnit > 0) {
                long cap = (long) Math.floor((devices.get(i).vramBytes() * vramUtilization) / bytesPerUnit);
                maxUnits[i] = cap;
                totalCapacity = addSaturating(totalCapacity, cap);
            } else {
                maxUnits[i] = Long.MAX_VALUE;
                totalCapacity = Long.MAX_VALUE;
            }
        }

        if (totalUnits > totalCapacity) {
            long totalUsableBytes = 0;
            for (DeviceProfile d : devices) {
                totalUsableBytes += (long) (d.vramBytes() * vramUtilization);
            }
            throw new GpuOutOfMemoryException(String.format(
                    "Total workload units (%d) requires %d bytes, which exceeds total usable VRAM capacity (%d bytes) across all %d devices",
                    totalUnits, Math.multiplyExact(totalUnits, bytesPerUnit), totalUsableBytes, n));
        }

        // Build topology lookup matrices
        double[][] bw = new double[n][n];
        Transport[][] tr = new Transport[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                bw[i][j] = (i == j) ? 25.0 : 1.0;
                tr[i][j] = (i == j) ? Transport.PEER_TO_PEER : Transport.HOST_STAGED;
            }
        }
        if (topology != null) {
            for (Link link : topology) {
                int src = link.src();
                int dst = link.dst();
                if (src < n && dst < n) {
                    bw[src][dst] = link.gbps();
                    tr[src][dst] = link.transport();
                }
            }
        }

        // 2. Initial active set: all devices with maxUnits[i] > 0
        List<Integer> active = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (maxUnits[i] > 0) {
                active.add(i);
            }
        }
        if (active.isEmpty()) {
            throw new GpuOutOfMemoryException("No device has sufficient VRAM to allocate even one workload unit");
        }

        // 3. Iterative Straggler Pruning
        SubsetState current = evaluateSubset(active, devices, maxUnits, bw, tr, totalUnits, flopsPerUnit, bytesPerUnit, syncBytes);
        Set<Integer> prunedRanks = new HashSet<>();

        while (active.size() > 1) {
            SubsetState bestCandidate = null;
            int bestPruneIndex = -1;

            for (int idx = 0; idx < active.size(); idx++) {
                int candRank = active.get(idx);
                List<Integer> candidateActive = new ArrayList<>(active);
                candidateActive.remove(idx);

                long candCapacity = 0;
                for (int c : candidateActive) {
                    candCapacity = addSaturating(candCapacity, maxUnits[c]);
                }
                if (candCapacity < totalUnits) {
                    continue; // subset cannot accommodate workload
                }

                SubsetState eval = evaluateSubset(candidateActive, devices, maxUnits, bw, tr, totalUnits, flopsPerUnit, bytesPerUnit, syncBytes);
                if (bestCandidate == null || eval.makespanMs < bestCandidate.makespanMs) {
                    bestCandidate = eval;
                    bestPruneIndex = candRank;
                }
            }

            if (bestCandidate != null && bestCandidate.makespanMs < current.makespanMs * IMPROVEMENT_THRESHOLD) {
                current = bestCandidate;
                active = new ArrayList<>(current.activeIndices);
                prunedRanks.add(bestPruneIndex);
            } else {
                break; // No further improvement possible
            }
        }

        // Build final WorkloadPlan
        List<ShardAllocation> allocations = new ArrayList<>(n);
        List<Integer> activeRanks = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            int rank = devices.get(i).rank();
            long u = current.units[i];
            double share = (totalUnits > 0) ? (double) u / totalUnits : 0.0;
            double devGflops = Math.max(devices.get(i).computeGflops(), EPSILON);
            double computeMs = (u > 0) ? ((u * flopsPerUnit) / (devGflops * 1e9)) * 1000.0 : 0.0;
            long memBytes = (bytesPerUnit > 0) ? u * bytesPerUnit : 0L;
            boolean excluded = prunedRanks.contains(i);

            allocations.add(new ShardAllocation(rank, u, share, computeMs, memBytes, excluded));
            if (u > 0) {
                activeRanks.add(rank);
            }
        }

        return new WorkloadPlan(
                totalUnits,
                allocations,
                activeRanks,
                current.makespanMs,
                current.computeMs,
                current.syncMs);
    }

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

    private record SubsetState(
            List<Integer> activeIndices,
            long[] units,
            double makespanMs,
            double computeMs,
            double syncMs) {}

    private static SubsetState evaluateSubset(
            List<Integer> activeIndices,
            List<DeviceProfile> devices,
            long[] maxUnits,
            double[][] bw,
            Transport[][] tr,
            long totalUnits,
            double flopsPerUnit,
            long bytesPerUnit,
            long syncBytes) {
        int n = devices.size();
        long[] units = allocateWaterFilling(activeIndices, devices, maxUnits, totalUnits);

        // Compute makespan
        double maxComputeMs = 0.0;
        for (int idx : activeIndices) {
            long u = units[idx];
            if (u > 0) {
                double gflops = Math.max(devices.get(idx).computeGflops(), EPSILON);
                double ms = ((u * flopsPerUnit) / (gflops * 1e9)) * 1000.0;
                if (ms > maxComputeMs) {
                    maxComputeMs = ms;
                }
            }
        }

        double syncMs = estimateSyncMs(activeIndices, bw, tr, syncBytes);
        double totalMakespanMs = maxComputeMs + syncMs;

        return new SubsetState(activeIndices, units, totalMakespanMs, maxComputeMs, syncMs);
    }

    /**
     * Water-filling compute-proportional unit allocation respecting VRAM capacity caps.
     */
    private static long[] allocateWaterFilling(
            List<Integer> activeIndices,
            List<DeviceProfile> devices,
            long[] maxUnits,
            long totalUnits) {
        int n = devices.size();
        long[] result = new long[n];
        if (activeIndices.isEmpty() || totalUnits == 0) {
            return result;
        }

        double[] alloc = new double[n];
        Set<Integer> uncapped = new HashSet<>(activeIndices);
        double remainingUnits = totalUnits;

        while (!uncapped.isEmpty()) {
            double totalFlops = 0.0;
            for (int idx : uncapped) {
                totalFlops += Math.max(devices.get(idx).computeGflops(), EPSILON);
            }

            boolean anyClamped = false;
            for (int idx : new ArrayList<>(uncapped)) {
                double gflops = Math.max(devices.get(idx).computeGflops(), EPSILON);
                double share = (totalFlops > 0.0) ? remainingUnits * (gflops / totalFlops) : remainingUnits / uncapped.size();
                if (share >= maxUnits[idx]) {
                    alloc[idx] = maxUnits[idx];
                    remainingUnits -= maxUnits[idx];
                    uncapped.remove(idx);
                    anyClamped = true;
                }
            }

            if (!anyClamped) {
                for (int idx : uncapped) {
                    double gflops = Math.max(devices.get(idx).computeGflops(), EPSILON);
                    alloc[idx] = (totalFlops > 0.0) ? remainingUnits * (gflops / totalFlops) : remainingUnits / uncapped.size();
                }
                break;
            }
        }

        // Largest-remainder integer rounding
        long currentSum = 0;
        double[] remainders = new double[n];
        for (int idx : activeIndices) {
            result[idx] = (long) Math.floor(alloc[idx]);
            remainders[idx] = alloc[idx] - result[idx];
            currentSum += result[idx];
        }

        long needed = totalUnits - currentSum;
        List<Integer> sortedIndices = new ArrayList<>(activeIndices);
        sortedIndices.sort((a, b) -> {
            int cmp = Double.compare(remainders[b], remainders[a]);
            return (cmp != 0) ? cmp : Integer.compare(a, b);
        });

        for (int idx : sortedIndices) {
            if (needed <= 0) break;
            if (result[idx] < maxUnits[idx]) {
                result[idx]++;
                needed--;
            }
        }

        return result;
    }

    /**
     * Estimates AllReduce synchronization duration over the active subset of devices.
     */
    private static double estimateSyncMs(
            List<Integer> activeIndices,
            double[][] bw,
            Transport[][] tr,
            long syncBytes) {
        int m = activeIndices.size();
        if (m <= 1 || syncBytes == 0) {
            return 0.0;
        }

        // Find optimal flat ring bottleneck bandwidth over activeIndices
        double ringBottleneckGbps = solveRingBottleneck(activeIndices, bw);
        double flatSyncMs = (2.0 * (m - 1.0) / m) * (syncBytes / (ringBottleneckGbps * 1e9)) * 1000.0;

        // Detect fast islands within activeIndices
        List<List<Integer>> islands = clusterIslands(activeIndices, bw, tr);
        if (islands.size() > 1 && islands.stream().anyMatch(isl -> isl.size() > 1)) {
            // Intra-island cost: max over islands with size > 1
            double maxIntraMs = 0.0;
            List<Integer> leaders = new ArrayList<>();

            for (List<Integer> isl : islands) {
                int leader = isl.get(0); // lowest rank leader
                leaders.add(leader);
                if (isl.size() > 1) {
                    double intraBottleneck = Double.MAX_VALUE;
                    for (int member : isl) {
                        if (member != leader) {
                            intraBottleneck = Math.min(intraBottleneck, bw[member][leader]);
                            intraBottleneck = Math.min(intraBottleneck, bw[leader][member]);
                        }
                    }
                    if (intraBottleneck <= 0.0) intraBottleneck = 1.0;
                    double intraMs = 2.0 * (syncBytes / (intraBottleneck * 1e9)) * 1000.0;
                    if (intraMs > maxIntraMs) {
                        maxIntraMs = intraMs;
                    }
                }
            }

            int kLeaders = leaders.size();
            double leaderRingBottleneck = solveRingBottleneck(leaders, bw);
            double leaderSyncMs = (2.0 * (kLeaders - 1.0) / kLeaders) * (syncBytes / (leaderRingBottleneck * 1e9)) * 1000.0;
            double hierarchicalSyncMs = maxIntraMs + leaderSyncMs;

            return Math.min(flatSyncMs, hierarchicalSyncMs);
        }

        return flatSyncMs;
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

        if (m <= 12) {
            int k = m - 1;
            int numMasks = 1 << k;
            double[][] dp = new double[numMasks][m];
            for (double[] row : dp) {
                Arrays.fill(row, -1.0);
            }
            dp[0][0] = Double.MAX_VALUE;

            for (int mask = 0; mask < numMasks; mask++) {
                for (int u = 0; u < m; u++) {
                    if (mask == 0 && u != 0) continue;
                    if (mask != 0 && (u == 0 || (mask & (1 << (u - 1))) == 0)) continue;
                    double curB = dp[mask][u];
                    if (curB < 0) continue;

                    int uRank = indices.get(u);
                    for (int v = 1; v < m; v++) {
                        if ((mask & (1 << (v - 1))) == 0) {
                            int nextMask = mask | (1 << (v - 1));
                            int vRank = indices.get(v);
                            double candB = Math.min(curB, bw[uRank][vRank]);
                            if (candB > dp[nextMask][v]) {
                                dp[nextMask][v] = candB;
                            }
                        }
                    }
                }
            }

            int fullMask = numMasks - 1;
            double bestB = 0.0;
            int startRank = indices.get(0);
            for (int last = 1; last < m; last++) {
                double b = dp[fullMask][last];
                if (b > 0) {
                    int lastRank = indices.get(last);
                    double candB = Math.min(b, bw[lastRank][startRank]);
                    if (candB > bestB) {
                        bestB = candB;
                    }
                }
            }
            return (bestB > 0.0) ? bestB : 1.0;
        } else {
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

    private static List<List<Integer>> clusterIslands(
            List<Integer> indices,
            double[][] bw,
            Transport[][] tr) {
        int m = indices.size();
        double sMin = Double.MAX_VALUE;
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                int u = indices.get(i);
                int v = indices.get(j);
                double sym = Math.min(bw[u][v], bw[v][u]);
                if (sym > 0.0 && sym < sMin) {
                    sMin = sym;
                }
            }
        }
        if (sMin == Double.MAX_VALUE) sMin = 1.0;

        boolean[] hasP2p = new boolean[m];
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                int u = indices.get(i);
                int v = indices.get(j);
                if (tr[u][v] == Transport.PEER_TO_PEER && tr[v][u] == Transport.PEER_TO_PEER) {
                    hasP2p[i] = true;
                    hasP2p[j] = true;
                }
            }
        }

        int[] parent = new int[m];
        for (int i = 0; i < m; i++) parent[i] = i;

        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                int u = indices.get(i);
                int v = indices.get(j);
                boolean rule1 = tr[u][v] == Transport.PEER_TO_PEER && tr[v][u] == Transport.PEER_TO_PEER;
                boolean rule2 = tr[u][v] == tr[v][u]
                        && tr[u][v] != Transport.PEER_TO_PEER
                        && Math.min(bw[u][v], bw[v][u]) >= 1.65 * sMin
                        && !hasP2p[i]
                        && !hasP2p[j];
                if (rule1 || rule2) {
                    int rootI = findRoot(parent, i);
                    int rootJ = findRoot(parent, j);
                    if (rootI != rootJ) {
                        parent[rootI] = rootJ;
                    }
                }
            }
        }

        Map<Integer, List<Integer>> groups = new HashMap<>();
        for (int i = 0; i < m; i++) {
            int root = findRoot(parent, i);
            groups.computeIfAbsent(root, k -> new ArrayList<>()).add(indices.get(i));
        }

        List<List<Integer>> result = new ArrayList<>(groups.values());
        for (List<Integer> isl : result) {
            Collections.sort(isl);
        }
        result.sort(Comparator.comparingInt(List::getFirst));
        return result;
    }

    private static int findRoot(int[] parent, int i) {
        if (parent[i] == i) return i;
        parent[i] = findRoot(parent, parent[i]);
        return parent[i];
    }
}
