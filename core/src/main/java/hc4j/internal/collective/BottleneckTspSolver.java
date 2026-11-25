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

import java.util.Arrays;
import java.util.List;

/**
 * Theorem 1 solver: Shared-Root All-Subsets Bottleneck-TSP dynamic programming in
 * \(O(N^2 2^N)\) plus greedy ring bottleneck fallback for clusters with \(N &gt; 12\).
 */
public final class BottleneckTspSolver {

    private BottleneckTspSolver() {
        throw new AssertionError("No instances of BottleneckTspSolver");
    }

    /**
     * Exact precomputed bottleneck bandwidth and maximum hop latency across all \(2^N\) vertex subsets.
     *
     * @param bottleneckGbps per-subset ring bottleneck bandwidths in GB/s indexed by bitmask
     * @param maxHopLatencyUs per-subset maximum hop latency in microseconds indexed by bitmask
     */
    public record RingTable(double[] bottleneckGbps, double[] maxHopLatencyUs) {

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
    }

    /**
     * Precomputes exact bottleneck bandwidth and maximum hop latency for all \(2^n\) subsets
     * in \(O(n^2 2^n)\) using shared-root cycle dynamic programming.
     *
     * @param n number of vertices ({@code 0 <= n <= 12})
     * @param bw directed bandwidth matrix [n][n] in GB/s
     * @param latUs directed latency matrix [n][n] in microseconds
     * @return precomputed {@link RingTable}
     */
    public static RingTable precomputeAllSubsetsRing(int n, double[][] bw, double[][] latUs) {
        int numSubsets = 1 << n;
        double[] bottleneckGbps = new double[numSubsets];
        double[] maxHopLatencyUs = new double[numSubsets];

        for (int i = 0; i < n; i++) {
            bottleneckGbps[1 << i] = 25.0;
            maxHopLatencyUs[1 << i] = 0.0;
        }

        for (int u = 0; u < n; u++) {
            for (int v = u + 1; v < n; v++) {
                int mask = (1 << u) | (1 << v);
                bottleneckGbps[mask] = Math.min(bw[u][v], bw[v][u]);
                maxHopLatencyUs[mask] = Math.max(latUs[u][v], latUs[v][u]);
            }
        }

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

        return new RingTable(bottleneckGbps, maxHopLatencyUs);
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
