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
import hc4j.collective.DeviceRole;
import hc4j.collective.HostCpuProfile;
import hc4j.collective.Link;
import hc4j.collective.RolePlan;
import hc4j.collective.ShardAllocation;
import hc4j.collective.SplitKvAssignment;
import hc4j.collective.SplitKvSpec;
import hc4j.collective.WorkloadPlan;
import hc4j.collective.WorkloadSpec;
import hc4j.engine.GpuOutOfMemoryException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Pure Java solver for Zero-Waste Straggler Reclamation ({@code planRoles} + KV integer minimax),
 * reclaiming GPUs pruned from Tensor Parallelism as lossless Split-KV attention workers.
 */
public final class SplitKvRoleSolver {

    private static final double EPSILON = 1e-6;

    private SplitKvRoleSolver() {
        throw new AssertionError("No instances of SplitKvRoleSolver");
    }

    private record KvWorker(int rank, long maxTokens, double rAttn, double lRpc) {}

    /**
     * Generates a multi-operator role plan purely in Java (headless fallback).
     *
     * @param devices participating device profiles
     * @param topology measured network topology links
     * @param tpSpec Tensor Parallel specification
     * @param kvSpec Split-KV attention specification
     * @return combined {@link RolePlan}
     */
    public static RolePlan planRoles(
            List<DeviceProfile> devices,
            List<Link> topology,
            WorkloadSpec tpSpec,
            SplitKvSpec kvSpec) {
        Objects.requireNonNull(devices, "devices must not be null");
        Objects.requireNonNull(tpSpec, "tpSpec must not be null");
        Objects.requireNonNull(kvSpec, "kvSpec must not be null");

        WorkloadPlan tpPlan = ConvexSpillSolver.plan(devices, topology, tpSpec);
        int n = devices.size();
        MstDendrogram.TopologyMatrices topo = MstDendrogram.buildTopologyMatrices(n, topology);
        double[][] bw = topo.bw();
        double[][] latUs = topo.latUs();

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
        double maxTpGflops = -1.0;
        for (int r : activeTpGpus) {
            if (devices.get(r).computeGflops() > maxTpGflops) {
                maxTpGflops = devices.get(r).computeGflops();
                anchorRank = r;
            }
        }

        long[] maxKvTokens = new long[n];
        double[] rAttn = new double[n];
        double[] lRpc = new double[n];

        for (int i = 0; i < n; i++) {
            DeviceProfile d = devices.get(i);
            long usableVram = (long) Math.floor(d.vramBytes() * kvSpec.vramUtilization());
            long availKvVram = Math.max(0L, usableVram - kvSpec.reservedWeightBytesPerGpu());
            maxKvTokens[i] = availKvVram / kvSpec.kvBytesPerToken();

            double cGflops = Math.max(d.computeGflops(), EPSILON);
            double memBw = (d.vramBandwidthGbps() > 0.0 && Double.isFinite(d.vramBandwidthGbps()))
                    ? d.vramBandwidthGbps()
                    : switch (d.deviceType()) {
                        case "Discrete" -> 100.0;
                        case "Integrated" -> 35.0;
                        default -> 50.0;
                    };
            double tComp = kvSpec.attentionFlopsPerToken() / (cGflops * 1e9);
            double tMem = kvSpec.kvBytesPerToken() / (memBw * 1e9);
            rAttn[i] = Math.max(tComp, tMem) * 1000.0;

            if (i == anchorRank) {
                lRpc[i] = 0.0;
            } else {
                double linkBw = Math.max(EPSILON, Math.min(bw[anchorRank][i], bw[i][anchorRank]));
                double linkLat = (latUs[anchorRank][i] + latUs[i][anchorRank]) / 1000.0;
                double rpcMs = (kvSpec.queryAndPartialBytes() / (linkBw * 1e9)) * 1000.0;
                lRpc[i] = linkLat + rpcMs;
            }
        }

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

        List<KvWorker> bestWorkerSet = null;
        long[] bestKvAlloc = null;
        double bestAttnMs = 0.0;

        if (totalKv > 0) {
            long totalCap = 0;
            for (KvWorker w : allWorkers) {
                totalCap = ConvexSpillSolver.addSaturating(totalCap, w.maxTokens);
            }
            if (totalCap >= totalKv) {
                bestWorkerSet = allWorkers;
                bestKvAlloc = solveKvBisection(allWorkers, totalKv);
                bestAttnMs = 0.0;
                for (int i = 0; i < allWorkers.size(); i++) {
                    KvWorker w = allWorkers.get(i);
                    long tokens = bestKvAlloc[i];
                    double time = (tokens > 0) ? w.lRpc + (tokens * w.rAttn) : 0.0;
                    bestAttnMs = Math.max(bestAttnMs, time);
                }
            }

            if (bestWorkerSet == null) {
                throw new GpuOutOfMemoryException("KV tokens (" + totalKv + ") exceed total KV capacity across all devices");
            }
        }

        Map<Integer, Long> tokenMap = new HashMap<>();
        if (bestWorkerSet != null && bestKvAlloc != null) {
            for (int i = 0; i < bestWorkerSet.size(); i++) {
                tokenMap.put(bestWorkerSet.get(i).rank, bestKvAlloc[i]);
            }
        }

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
            assignments.add(new SplitKvAssignment(
                    rank, role, tokens, localAttnMs, commMs, localAttnMs + commMs, tokens * kvSpec.kvBytesPerToken()));
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
            assignments.add(new SplitKvAssignment(
                    ShardAllocation.HOST_CPU_RANK, role, tokens, localAttnMs, commMs, localAttnMs + commMs,
                    tokens * kvSpec.kvBytesPerToken()));
        }

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

    private static long kvContrib(List<KvWorker> workers, double t, long[] out) {
        long sum = 0;
        for (int i = 0; i < workers.size(); i++) {
            KvWorker w = workers.get(i);
            double threshold = w.lRpc + w.rAttn;
            long p;
            if (w.maxTokens == 0 || t < threshold) {
                p = 0;
            } else if (w.rAttn <= 0.0) {
                p = w.maxTokens;
            } else {
                long fl = (long) Math.floor((t - w.lRpc) / w.rAttn);
                p = Math.min(Math.max(0L, fl), w.maxTokens);
            }
            out[i] = p;
            sum = ConvexSpillSolver.addSaturating(sum, p);
        }
        return sum;
    }

    private static long[] solveKvBisection(List<KvWorker> workers, long totalTokens) {
        int m = workers.size();
        long[] result = new long[m];
        if (m == 0 || totalTokens == 0) {
            return result;
        }

        double tLow = 0.0;
        double tHigh = 1.0;
        for (KvWorker w : workers) {
            double cap = (double) Math.min(w.maxTokens, totalTokens);
            tHigh = Math.max(tHigh, w.lRpc + cap * w.rAttn + w.rAttn);
        }
        long[] scratch = new long[m];
        for (int s = 0; s < 64; s++) {
            if (kvContrib(workers, tHigh, scratch) >= totalTokens) {
                break;
            }
            tHigh *= 2.0;
        }

        for (int step = 0; step < 72; step++) {
            double tMid = tLow + (tHigh - tLow) * 0.5;
            if (kvContrib(workers, tMid, scratch) >= totalTokens) {
                tHigh = tMid;
            } else {
                tLow = tMid;
            }
        }

        long currentSum = kvContrib(workers, tHigh, result);
        if (currentSum > totalTokens) {
            long excess = currentSum - totalTokens;
            List<Integer> active = new ArrayList<>();
            for (int i = 0; i < m; i++) {
                if (result[i] > 0) active.add(i);
            }
            while (excess > 0 && !active.isEmpty()) {
                active.sort((a, b) -> {
                    double ta = workers.get(a).lRpc + result[a] * workers.get(a).rAttn;
                    double tb = workers.get(b).lRpc + result[b] * workers.get(b).rAttn;
                    int cmp = Double.compare(tb, ta);
                    return (cmp != 0) ? cmp : Integer.compare(a, b);
                });
                for (int idx : active) {
                    if (excess == 0) break;
                    if (result[idx] > 0) {
                        result[idx]--;
                        excess--;
                    }
                }
                active.removeIf(idx -> result[idx] <= 0);
            }
        } else if (currentSum < totalTokens) {
            long deficit = totalTokens - currentSum;
            record HeapEntry(int index, double nextTime) implements Comparable<HeapEntry> {
                @Override
                public int compareTo(HeapEntry o) {
                    int cmp = Double.compare(this.nextTime, o.nextTime);
                    return (cmp != 0) ? cmp : Integer.compare(this.index, o.index);
                }
            }

            PriorityQueue<HeapEntry> pq = new PriorityQueue<>(m);
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
}
