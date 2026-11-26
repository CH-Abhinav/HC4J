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
import hc4j.collective.Link;
import hc4j.collective.Transport;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Theorem 2 solver: Zero-Parameter Maximum Spanning Tree (MST) Dendrogram clustering
 * for exact hierarchical AllReduce synchronization time estimation.
 */
public final class MstDendrogram {

    private MstDendrogram() {
        throw new AssertionError("No instances of MstDendrogram");
    }

    /**
     * Topology adjacency matrices derived from measured collective links.
     *
     * @param bw directed bandwidth matrix [n][n] in GB/s
     * @param latUs directed latency matrix [n][n] in microseconds
     * @param tr directed transport matrix [n][n]
     */
    public record TopologyMatrices(double[][] bw, double[][] latUs, Transport[][] tr) {}

    private record UndirectedEdge(int u, int v, int trSym, double wSym) {}

    /**
     * Builds dense bandwidth, latency, and transport matrices from measured topology links.
     *
     * @param n number of devices
     * @param topology measured directed links (may be null)
     * @return dense {@link TopologyMatrices}
     */
    public static TopologyMatrices buildTopologyMatrices(int n, List<Link> topology) {
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
        return new TopologyMatrices(bw, latUs, tr);
    }

    private static double ringAllreduceTimeSecs(
            List<Integer> indices,
            double[][] bw,
            double[][] latUs,
            BottleneckTspSolver.RingTable ringTable,
            long syncBytes) {
        int m = indices.size();
        if (m <= 1 || syncBytes == 0) {
            return 0.0;
        }
        double mF = m;
        if (ringTable != null) {
            int mask = 0;
            for (int idx : indices) {
                mask |= (1 << idx);
            }
            double hUs = ringTable.hockneyUs(mask);
            if (hUs > 0.0) {
                return 2.0 * (mF - 1.0) * (hUs / 1e6);
            }
            double ringBw = Math.max(ringTable.bottleneck(mask), 1e-6);
            double ringLatSecs = ringTable.latencyUs(mask) / 1e6;
            return 2.0 * (mF - 1.0) * ringLatSecs
                    + 2.0 * ((mF - 1.0) / mF) * (syncBytes / (ringBw * 1e9));
        }
        double hockneyUs = BottleneckTspSolver.solveRingHockneyGreedy(indices, bw, latUs, syncBytes);
        return 2.0 * (mF - 1.0) * (hockneyUs / 1e6);
    }

    /**
     * Estimates the minimum AllReduce synchronization duration across all MST dendrogram cuts.
     *
     * @param activeIndices active GPU indices
     * @param bw directed bandwidth matrix
     * @param latUs directed latency matrix
     * @param tr directed transport matrix
     * @param ringTable precomputed exact ring table (may be null for {@code n > 12})
     * @param devices hardware profiles
     * @param syncBytes collective synchronization payload in bytes
     * @return minimum estimated synchronization duration in milliseconds
     */
    public static double estimateSyncMsExact(
            List<Integer> activeIndices,
            double[][] bw,
            double[][] latUs,
            Transport[][] tr,
            BottleneckTspSolver.RingTable ringTable,
            List<DeviceProfile> devices,
            long syncBytes) {
        int m = activeIndices.size();
        if (m <= 1 || syncBytes == 0) {
            return 0.0;
        }

        double tFlat = ringAllreduceTimeSecs(activeIndices, bw, latUs, ringTable, syncBytes);
        double bestTimeSecs = tFlat;

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
        for (int idx : activeIndices) {
            maxRank = Math.max(maxRank, idx);
        }
        int[] dsu = new int[maxRank + 1];
        for (int i = 0; i <= maxRank; i++) {
            dsu[i] = i;
        }

        List<UndirectedEdge> mstEdges = new ArrayList<>(m - 1);
        for (UndirectedEdge e : edges) {
            int rootU = findRoot(dsu, e.u);
            int rootV = findRoot(dsu, e.v);
            if (rootU != rootV) {
                dsu[rootU] = rootV;
                mstEdges.add(e);
                if (mstEdges.size() == m - 1) {
                    break;
                }
            }
        }

        mstEdges.sort((a, b) -> {
            int cmp = Integer.compare(a.trSym, b.trSym);
            return (cmp != 0) ? cmp : Double.compare(a.wSym, b.wSym);
        });

        for (int k = 1; k < m; k++) {
            for (int i = 0; i <= maxRank; i++) {
                dsu[i] = i;
            }
            for (int i = k; i < mstEdges.size(); i++) {
                UndirectedEdge e = mstEdges.get(i);
                int rU = findRoot(dsu, e.u);
                int rV = findRoot(dsu, e.v);
                if (rU != rV) {
                    dsu[rU] = rV;
                }
            }

            Map<Integer, List<Integer>> groups = new HashMap<>();
            for (int idx : activeIndices) {
                int r = findRoot(dsu, idx);
                groups.computeIfAbsent(r, x -> new ArrayList<>()).add(idx);
            }

            List<List<Integer>> islands = new ArrayList<>(groups.values());
            for (List<Integer> isl : islands) {
                Collections.sort(isl);
            }
            islands.sort(Comparator.comparingInt(List::getFirst));

            List<Integer> leaders = new ArrayList<>(islands.size());
            double tIntraMax = 0.0;

            for (List<Integer> isl : islands) {
                int leader = pickLeader(isl, activeIndices, devices, bw, latUs, syncBytes);
                leaders.add(leader);
                if (isl.size() > 1) {
                    double tIntra = ringAllreduceTimeSecs(isl, bw, latUs, ringTable, syncBytes);
                    tIntraMax = Math.max(tIntraMax, tIntra);
                }
            }

            double tLead = ringAllreduceTimeSecs(leaders, bw, latUs, ringTable, syncBytes);
            double tHier = tIntraMax + tLead;
            if (tHier < bestTimeSecs * (1.0 - 1e-6)) {
                bestTimeSecs = tHier;
            }
        }

        return bestTimeSecs * 1000.0;
    }

    private static int pickLeader(
            List<Integer> members,
            List<Integer> activeIndices,
            List<DeviceProfile> devices,
            double[][] bw,
            double[][] latUs,
            long syncBytes) {
        int best = members.getFirst();
        double bestWorstOutT = Double.POSITIVE_INFINITY;
        double bestGflops = -1.0;

        for (int m : members) {
            double gflops = (m < devices.size()) ? devices.get(m).computeGflops() : 0.0;
            double worstOutT = 0.0;
            boolean hasOutbound = false;
            for (int o : activeIndices) {
                if (!members.contains(o)) {
                    hasOutbound = true;
                    double linkT = (latUs[m][o] / 1e6) + ((double) syncBytes / (Math.max(bw[m][o], 1e-6) * 1e9));
                    worstOutT = Math.max(worstOutT, linkT);
                }
            }
            if (!hasOutbound) {
                worstOutT = 0.0;
            }

            double scale = Math.max(worstOutT, Math.max(bestWorstOutT, 1e-9));
            double diff = worstOutT - bestWorstOutT;
            boolean better;
            if (diff < -1e-6 * scale) {
                better = true;
            } else if (diff > 1e-6 * scale) {
                better = false;
            } else if (Math.abs(gflops - bestGflops) > 1e-4) {
                better = gflops > bestGflops;
            } else {
                better = m < best;
            }

            if (better) {
                best = m;
                bestWorstOutT = worstOutT;
                bestGflops = gflops;
            }
        }
        return best;
    }

    private static int findRoot(int[] parent, int i) {
        if (parent[i] == i) {
            return i;
        }
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
}
