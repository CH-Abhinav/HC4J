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

/**
 * Workload unit allocation, tier breakdown, and makespan estimate for a single device in the collective.
 *
 * @param rank the rank index of the device, or {@link #HOST_CPU_RANK} for the explicit host CPU node
 * @param units the total number of workload units assigned to this device
 * @param shareRatio the fractional share of total units assigned to this device
 * @param estimatedComputeMs the estimated compute duration in milliseconds
 * @param estimatedMemoryBytes the estimated memory requirement in bytes
 * @param excludedAsStraggler true if this device was excluded to prevent degrading makespan
 * @param vramUnits units resident in high-bandwidth device VRAM (Tier 0)
 * @param hostStreamUnits units streamed over PCIe from host RAM (Tier 1)
 * @param diskStreamUnits units streamed from Tier 2 NVMe disk spill storage
 * @param cpuInPlaceUnits units computed in-place on the host CPU node
 */
public record ShardAllocation(
        int rank,
        long units,
        double shareRatio,
        double estimatedComputeMs,
        long estimatedMemoryBytes,
        boolean excludedAsStraggler,
        long vramUnits,
        long hostStreamUnits,
        long diskStreamUnits,
        long cpuInPlaceUnits) {

    /**
     * Constant representing the explicit host CPU node in collective role and workload plans.
     */
    public static final int HOST_CPU_RANK = -1;

    /**
     * Constructs and validates a {@link ShardAllocation}.
     *
     * @param rank device rank (non-negative or {@link #HOST_CPU_RANK})
     * @param units total assigned units (non-negative)
     * @param shareRatio fractional share in [0, 1]
     * @param estimatedComputeMs compute time in ms (non-negative)
     * @param estimatedMemoryBytes memory required in bytes (non-negative)
     * @param excludedAsStraggler whether excluded as straggler
     * @param vramUnits units in device VRAM (non-negative)
     * @param hostStreamUnits units streamed from host RAM (non-negative)
     * @param diskStreamUnits units streamed from disk (non-negative)
     * @param cpuInPlaceUnits units computed on CPU (non-negative)
     * @throws IllegalArgumentException if any numeric argument is out of bounds
     */
    public ShardAllocation {
        if (rank < 0 && rank != HOST_CPU_RANK) {
            throw new IllegalArgumentException("rank must be non-negative or HOST_CPU_RANK: " + rank);
        }
        if (units < 0) {
            throw new IllegalArgumentException("units must be non-negative: " + units);
        }
        if (shareRatio < 0.0 || shareRatio > 1.0 + 1e-6) {
            throw new IllegalArgumentException("shareRatio out of bounds [0, 1]: " + shareRatio);
        }
        if (estimatedComputeMs < 0.0) {
            throw new IllegalArgumentException("estimatedComputeMs must be non-negative: " + estimatedComputeMs);
        }
        if (estimatedMemoryBytes < 0) {
            throw new IllegalArgumentException("estimatedMemoryBytes must be non-negative: " + estimatedMemoryBytes);
        }
        if (vramUnits < 0) {
            throw new IllegalArgumentException("vramUnits must be non-negative: " + vramUnits);
        }
        if (hostStreamUnits < 0) {
            throw new IllegalArgumentException("hostStreamUnits must be non-negative: " + hostStreamUnits);
        }
        if (diskStreamUnits < 0) {
            throw new IllegalArgumentException("diskStreamUnits must be non-negative: " + diskStreamUnits);
        }
        if (cpuInPlaceUnits < 0) {
            throw new IllegalArgumentException("cpuInPlaceUnits must be non-negative: " + cpuInPlaceUnits);
        }
    }

    /**
     * Backwards-compatible constructor defaulting tier units to VRAM-only.
     *
     * @param rank device rank (non-negative)
     * @param units assigned units (non-negative)
     * @param shareRatio fractional share in [0, 1]
     * @param estimatedComputeMs compute time in ms (non-negative)
     * @param estimatedMemoryBytes memory required in bytes (non-negative)
     * @param excludedAsStraggler whether excluded as straggler
     */
    public ShardAllocation(
            int rank,
            long units,
            double shareRatio,
            double estimatedComputeMs,
            long estimatedMemoryBytes,
            boolean excludedAsStraggler) {
        this(
                rank,
                units,
                shareRatio,
                estimatedComputeMs,
                estimatedMemoryBytes,
                excludedAsStraggler,
                units,
                0L,
                0L,
                0L);
    }
}
