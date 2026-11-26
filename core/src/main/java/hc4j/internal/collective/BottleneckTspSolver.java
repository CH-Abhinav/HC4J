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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Theorem 1 solver: Shared-Root All-Subsets Bottleneck-TSP dynamic programming in
 * \(O(N^2 2^N)\) with exact Hockney per-round edge costs \(w_m(e) = \lambda_e + D / (m \cdot B_e)\)
 * plus greedy ring bottleneck fallback for clusters with \(N &gt; 12\).
 */
public final class BottleneckTspSolver {

    private BottleneckTspSolver() {
        throw new AssertionError("No instances of BottleneckTspSolver");
    }

    /**
     * Exact precomputed bottleneck bandwidth, maximum hop latency, and per-round Hockney cost across all \(2^N\) vertex subsets.
     *
     * @param bottleneckGbps per-subset ring bottleneck bandwidths in GB/s indexed by bitmask
     * @param maxHopLatencyUs per-subset maximum hop latency in microseconds indexed by bitmask
     * @param hockneyCostUs per-subset per-round Hockney bottleneck cost in microseconds indexed by bitmask
     */
    public record RingTable(double[] bottleneckGbps, double[] maxHopLatencyUs, double[] hockneyCostUs) {

        /**
         * Backwards-compatible constructor omitting explicit Hockney per-round costs.
         *
         * @param bottleneckGbps per-subset ring bottleneck bandwidths in GB/s
         * @param maxHopLatencyUs per-subset maximum hop latency in microseconds
         */
        public RingTable(double[] bottleneckGbps, double[] maxHopLatencyUs) {
            this(bottleneckGbps, maxHopLatencyUs, new double[bottleneckGbps.length]);
        }

        /**
         * Returns the bottleneck ring bandwidth in GB/s for bitmask {@code mask}.
         *
         * @param mask subset bitmask
         * @return bottleneck bandwidth in GB/s
         */
        public double bottleneck(int mask) {
            return (mask >= 0 && mask < bottleneckGbps.length) ? bottleneckGbps[mask] : 1.0;
        }

        /**
         * Returns the maximum directed hop latency in microseconds for bitmask {@code mask}.
         *
         * @param mask subset bitmask
         * @return maximum hop latency in microseconds
         */
        public double latencyUs(int mask) {
            return (mask >= 0 && mask < maxHopLatencyUs.length) ? maxHopLatencyUs[mask] : 0.0;
        }

        /**
         * Returns the precomputed per-round Hockney cost in microseconds for bitmask {@code mask}.
         *
         * @param mask subset bitmask
         * @return per-round Hockney bottleneck cost in microseconds
         */
        public double hockneyUs(int mask) {
            return (mask >= 0 && hockneyCostUs != null && mask < hockneyCostUs.length) ? hockneyCostUs[mask] : 0.0;
        }
    }

    private static double edgeHockneyUs(double bwGbps, double latUs, int m, long syncBytes) {
        double txUs = (syncBytes > 0)
                ? ((double) syncBytes) / ((double) m * Math.max(bwGbps, 1e-9) * 1000.0)
                : 0.0;
        return latUs + txUs;
    }

    /**
     * Precomputes exact bottleneck bandwidth and maximum hop latency for all \(2^n\) subsets
     * in \(O(n^2 2^n)\) using shared-root cycle dynamic programming (asymptotic bandwidth-first limit).
     *
     * @param n number of vertices ({@code 0 <= n <= 12})
     * @param bw directed bandwidth matrix [n][n] in GB/s
     * @param latUs directed latency matrix [n][n] in microseconds
     * @return precomputed {@link RingTable}
     */
    public static RingTable precomputeAllSubsetsRing(int n, double[][] bw, double[][] latUs) {
        return precomputeAllSubsetsRingImpl(n, bw, latUs, null);
    }

    /**
     * Precomputes exact per-round Hockney bottleneck cycles across all \(2^n\) subsets
     * for a specific collective payload {@code syncBytes} by partitioning cardinalities \(m \in \{2,\dots,n\}\)
     * into equivalence classes sharing the same sorted edge ranking of \(w_m(e) = \lambda_e + D/(m B_e)\).
     *
     * @param n number of vertices ({@code 0 <= n <= 12})
     * @param bw directed bandwidth matrix [n][n] in GB/s
     * @param latUs directed latency matrix [n][n] in microseconds
     * @param syncBytes collective synchronization payload in bytes
     * @return precomputed {@link RingTable}
     */
    public static RingTable precomputeAllSubsetsRingForPayload(int n, double[][] bw, double[][] latUs, long syncBytes) {
        return precomputeAllSubsetsRingImpl(n, bw, latUs, syncBytes);
    }

    private static final class EquivClass {
        final int repM;
        int cardMask;
        final int[] edgeRank;

        EquivClass(int repM, int cardMask, int[] edgeRank) {
            this.repM = repM;
            this.cardMask = cardMask;
            this.edgeRank = edgeRank;
        }
    }

    private static RingTable precomputeAllSubsetsRingImpl(int n, double[][] bw, double[][] latUs, Long payloadBytes) {
        int numSubsets = 1 << n;
        double[] bottleneckGbps = new double[numSubsets];
        double[] maxHopLatencyUs = new double[numSubsets];
        double[] hockneyCostUs = new double[numSubsets];

        for (int i = 0; i < n; i++) {
            bottleneckGbps[1 << i] = 25.0;
            maxHopLatencyUs[1 << i] = 0.0;
            hockneyCostUs[1 << i] = 0.0;
        }

        for (int u = 0; u < n; u++) {
            for (int v = u + 1; v < n; v++) {
                int mask = (1 << u) | (1 << v);
                if (payloadBytes != null) {
                    long d = payloadBytes;
                    double wUv = edgeHockneyUs(bw[u][v], latUs[u][v], 2, d);
                    double wVu = edgeHockneyUs(bw[v][u], latUs[v][u], 2, d);
                    if (wUv >= wVu) {
                        hockneyCostUs[mask] = wUv;
                        bottleneckGbps[mask] = bw[u][v];
                        maxHopLatencyUs[mask] = latUs[u][v];
                    } else {
                        hockneyCostUs[mask] = wVu;
                        bottleneckGbps[mask] = bw[v][u];
                        maxHopLatencyUs[mask] = latUs[v][u];
                    }
                } else {
                    bottleneckGbps[mask] = Math.min(bw[u][v], bw[v][u]);
                    maxHopLatencyUs[mask] = Math.max(latUs[u][v], latUs[v][u]);
                }
            }
        }

        if (n < 3) {
            return new RingTable(bottleneckGbps, maxHopLatencyUs, hockneyCostUs);
        }

        if (payloadBytes == null) {
            for (int s0 = 0; s0 < n - 2; s0++) {
                int r = n - 1 - s0;
                int numSubMasks = 1 << r;
                double[][] dpBw = new double[numSubMasks][r];
                double[][] dpLat = new double[numSubMasks][r];
                for (double[] row : dpBw) {
                    Arrays.fill(row, -1.0);
                }

                for (int u = 0; u < r; u++) {
                    int target = s0 + 1 + u;
                    dpBw[1 << u][u] = bw[s0][target];
                    dpLat[1 << u][u] = latUs[s0][target];
                }

                for (int subMask = 1; subMask < numSubMasks; subMask++) {
                    for (int u = 0; u < r; u++) {
                        if ((subMask & (1 << u)) == 0) {
                            continue;
                        }
                        double curBw = dpBw[subMask][u];
                        if (curBw < 0) {
                            continue;
                        }
                        double curLat = dpLat[subMask][u];

                        int uRank = s0 + 1 + u;
                        for (int v = 0; v < r; v++) {
                            if ((subMask & (1 << v)) == 0) {
                                int nextMask = subMask | (1 << v);
                                int vRank = s0 + 1 + v;
                                double candBw = Math.min(curBw, bw[uRank][vRank]);
                                double candLat = Math.max(curLat, latUs[uRank][vRank]);

                                if (candBw > dpBw[nextMask][v]
                                        || (Math.abs(candBw - dpBw[nextMask][v]) < 1e-9 && candLat < dpLat[nextMask][v])) {
                                    dpBw[nextMask][v] = candBw;
                                    dpLat[nextMask][v] = candLat;
                                }
                            }
                        }
                    }

                    if (Integer.bitCount(subMask) >= 2) {
                        int globalMask = (1 << s0) | (subMask << (s0 + 1));
                        double bestCycleBw = -1.0;
                        double bestCycleLat = Double.MAX_VALUE;

                        for (int u = 0; u < r; u++) {
                            if ((subMask & (1 << u)) != 0 && dpBw[subMask][u] >= 0) {
                                int uRank = s0 + 1 + u;
                                double cycleBw = Math.min(dpBw[subMask][u], bw[uRank][s0]);
                                double cycleLat = Math.max(dpLat[subMask][u], latUs[uRank][s0]);

                                if (cycleBw > bestCycleBw
                                        || (Math.abs(cycleBw - bestCycleBw) < 1e-9 && cycleLat < bestCycleLat)) {
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
            return new RingTable(bottleneckGbps, maxHopLatencyUs, hockneyCostUs);
        }

        long syncD = payloadBytes;
        List<int[]> directedEdges = new ArrayList<>(n * (n - 1));
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (u != v) {
                    directedEdges.add(new int[] {u, v});
                }
            }
        }

        List<EquivClass> classes = new ArrayList<>();
        for (int m = 2; m <= n; m++) {
            final int cardM = m;
            List<int[]> sorted = new ArrayList<>(directedEdges);
            sorted.sort((e1, e2) -> {
                int u1 = e1[0], v1 = e1[1];
                int u2 = e2[0], v2 = e2[1];
                double w1 = edgeHockneyUs(bw[u1][v1], latUs[u1][v1], cardM, syncD);
                double w2 = edgeHockneyUs(bw[u2][v2], latUs[u2][v2], cardM, syncD);
                double diff = w1 - w2;
                double scale = Math.max(Math.abs(w1), Math.max(Math.abs(w2), 1e-9));
                if (Math.abs(diff) > 1e-12 * scale) {
                    return Double.compare(w1, w2);
                }
                int cmpBw = Double.compare(bw[u2][v2], bw[u1][v1]);
                if (cmpBw != 0) return cmpBw;
                int cmpLat = Double.compare(latUs[u1][v1], latUs[u2][v2]);
                if (cmpLat != 0) return cmpLat;
                int cmpU = Integer.compare(u1, u2);
                return (cmpU != 0) ? cmpU : Integer.compare(v1, v2);
            });

            int[] rankVec = new int[n * n];
            int curRank = 0;
            boolean hasPrev = false;
            double prevW = 0.0;
            double prevBw = 0.0;
            double prevLat = 0.0;
            for (int[] e : sorted) {
                int u = e[0], v = e[1];
                double w = edgeHockneyUs(bw[u][v], latUs[u][v], cardM, syncD);
                if (hasPrev) {
                    double scale = Math.max(Math.abs(w), Math.max(Math.abs(prevW), 1e-9));
                    if (Math.abs(w - prevW) > 1e-12 * scale
                            || Math.abs(bw[u][v] - prevBw) > 1e-9
                            || Math.abs(latUs[u][v] - prevLat) > 1e-9) {
                        curRank++;
                    }
                }
                rankVec[u * n + v] = curRank;
                hasPrev = true;
                prevW = w;
                prevBw = bw[u][v];
                prevLat = latUs[u][v];
            }

            EquivClass match = null;
            for (EquivClass ec : classes) {
                if (Arrays.equals(ec.edgeRank, rankVec)) {
                    match = ec;
                    break;
                }
            }
            if (match != null) {
                match.cardMask |= (1 << m);
            } else {
                classes.add(new EquivClass(m, 1 << m, rankVec));
            }
        }

        for (EquivClass cls : classes) {
            if ((cls.cardMask & ~0b111) == 0) {
                continue;
            }
            int repM = cls.repM;
            int[] er = cls.edgeRank;

            for (int s0 = 0; s0 < n - 2; s0++) {
                int r = n - 1 - s0;
                int numSubMasks = 1 << r;
                int[] dpRank = new int[numSubMasks * r];
                double[] dpBw = new double[numSubMasks * r];
                double[] dpLat = new double[numSubMasks * r];
                Arrays.fill(dpRank, Integer.MAX_VALUE);

                for (int u = 0; u < r; u++) {
                    int target = s0 + 1 + u;
                    int idx = (1 << u) * r + u;
                    dpRank[idx] = er[s0 * n + target];
                    dpBw[idx] = bw[s0][target];
                    dpLat[idx] = latUs[s0][target];
                }

                for (int subMask = 1; subMask < numSubMasks; subMask++) {
                    int rowOff = subMask * r;
                    for (int u = 0; u < r; u++) {
                        if ((subMask & (1 << u)) == 0) {
                            continue;
                        }
                        int curRank = dpRank[rowOff + u];
                        if (curRank == Integer.MAX_VALUE) {
                            continue;
                        }
                        double curBw = dpBw[rowOff + u];
                        double curLat = dpLat[rowOff + u];
                        int uRank = s0 + 1 + u;

                        for (int v = 0; v < r; v++) {
                            if ((subMask & (1 << v)) == 0) {
                                int nextMask = subMask | (1 << v);
                                int nextIdx = nextMask * r + v;
                                int vRank = s0 + 1 + v;
                                int eRank = er[uRank * n + vRank];
                                int candRank;
                                double candBw;
                                double candLat;
                                if (eRank > curRank) {
                                    candRank = eRank;
                                    candBw = bw[uRank][vRank];
                                    candLat = latUs[uRank][vRank];
                                } else {
                                    candRank = curRank;
                                    candBw = curBw;
                                    candLat = curLat;
                                }

                                int existRank = dpRank[nextIdx];
                                boolean better;
                                if (candRank != existRank) {
                                    better = candRank < existRank;
                                } else {
                                    double cw = edgeHockneyUs(candBw, candLat, repM, syncD);
                                    double ew = edgeHockneyUs(dpBw[nextIdx], dpLat[nextIdx], repM, syncD);
                                    better = cw < ew;
                                }
                                if (better) {
                                    dpRank[nextIdx] = candRank;
                                    dpBw[nextIdx] = candBw;
                                    dpLat[nextIdx] = candLat;
                                }
                            }
                        }
                    }

                    int m = 1 + Integer.bitCount(subMask);
                    if (m >= 3 && ((cls.cardMask >> m) & 1) != 0) {
                        int globalMask = (1 << s0) | (subMask << (s0 + 1));
                        int bestRank = Integer.MAX_VALUE;
                        double bestBw = 0.0;
                        double bestLat = Double.MAX_VALUE;

                        for (int u = 0; u < r; u++) {
                            if ((subMask & (1 << u)) != 0) {
                                int curRank = dpRank[rowOff + u];
                                if (curRank == Integer.MAX_VALUE) {
                                    continue;
                                }
                                int uRank = s0 + 1 + u;
                                int backRank = er[uRank * n + s0];
                                int cycRank;
                                double cycBw;
                                double cycLat;
                                if (backRank > curRank) {
                                    cycRank = backRank;
                                    cycBw = bw[uRank][s0];
                                    cycLat = latUs[uRank][s0];
                                } else {
                                    cycRank = curRank;
                                    cycBw = dpBw[rowOff + u];
                                    cycLat = dpLat[rowOff + u];
                                }

                                boolean better;
                                if (cycRank != bestRank) {
                                    better = cycRank < bestRank;
                                } else {
                                    double cw = edgeHockneyUs(cycBw, cycLat, m, syncD);
                                    double bwCost = edgeHockneyUs(bestBw, bestLat, m, syncD);
                                    better = cw < bwCost;
                                }
                                if (better) {
                                    bestRank = cycRank;
                                    bestBw = cycBw;
                                    bestLat = cycLat;
                                }
                            }
                        }

                        if (bestRank != Integer.MAX_VALUE) {
                            bottleneckGbps[globalMask] = bestBw;
                            maxHopLatencyUs[globalMask] = bestLat;
                            hockneyCostUs[globalMask] = edgeHockneyUs(bestBw, bestLat, m, syncD);
                        }
                    }
                }
            }
        }

        return new RingTable(bottleneckGbps, maxHopLatencyUs, hockneyCostUs);
    }

    /**
     * Scalable greedy Hockney-cost bottleneck-ring estimator for subsets exceeding the exact DP threshold.
     *
     * @param indices active device indices
     * @param bw directed bandwidth matrix [n][n] in GB/s
     * @param latUs directed latency matrix [n][n] in microseconds
     * @param syncBytes collective synchronization payload in bytes
     * @return per-round bottleneck Hockney cost in microseconds along the greedy tour
     */
    public static double solveRingHockneyGreedy(
            List<Integer> indices, double[][] bw, double[][] latUs, long syncBytes) {
        int m = indices.size();
        if (m <= 1) {
            return 0.0;
        }
        if (m == 2) {
            int u = indices.get(0);
            int v = indices.get(1);
            return Math.max(
                    edgeHockneyUs(bw[u][v], latUs[u][v], 2, syncBytes),
                    edgeHockneyUs(bw[v][u], latUs[v][u], 2, syncBytes));
        }

        boolean[] visited = new boolean[m];
        visited[0] = true;
        int cur = 0;
        double worstHockney = -1.0;

        for (int step = 1; step < m; step++) {
            int bestNext = 0;
            double bestW = Double.MAX_VALUE;
            int curRank = indices.get(cur);
            for (int next = 0; next < m; next++) {
                if (!visited[next]) {
                    int nextRank = indices.get(next);
                    double w = edgeHockneyUs(bw[curRank][nextRank], latUs[curRank][nextRank], m, syncBytes);
                    if (w < bestW) {
                        bestW = w;
                        bestNext = next;
                    }
                }
            }
            visited[bestNext] = true;
            if (bestW > worstHockney) {
                worstHockney = bestW;
            }
            cur = bestNext;
        }

        double backW = edgeHockneyUs(bw[indices.get(cur)][indices.get(0)], latUs[indices.get(cur)][indices.get(0)], m, syncBytes);
        if (backW > worstHockney) {
            worstHockney = backW;
        }
        return Math.max(0.0, worstHockney);
    }

    /**
     * Greedy ring bottleneck heuristic used when vertex subset size exceeds the exact DP threshold.
     *
     * @param indices active device indices
     * @param bw directed bandwidth matrix in GB/s
     * @return bottleneck bandwidth along the greedy tour in GB/s
     */
    public static double solveRingBottleneck(List<Integer> indices, double[][] bw) {
        int m = indices.size();
        if (m == 0) {
            return 1.0;
        }
        if (m == 1) {
            return 25.0;
        }
        if (m == 2) {
            int u = indices.get(0);
            int v = indices.get(1);
            return Math.min(bw[u][v], bw[v][u]);
        }

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
            if (bestNext == -1) {
                bestNext = 0;
            }
            visited[bestNext] = true;
            bottleneck = Math.min(bottleneck, bestW);
            cur = bestNext;
        }
        bottleneck = Math.min(bottleneck, bw[indices.get(cur)][indices.get(0)]);
        return (bottleneck > 0.0) ? bottleneck : 1.0;
    }
}
