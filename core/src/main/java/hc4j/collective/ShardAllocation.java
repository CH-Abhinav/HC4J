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
 * Workload unit allocation and makespan estimate for a single device in the collective.
 *
 * @param rank the rank index of the device
 * @param units the number of workload units assigned to this device
 * @param shareRatio the fractional share of total units assigned to this device
 * @param estimatedComputeMs the estimated compute duration in milliseconds
 * @param estimatedMemoryBytes the estimated memory requirement in bytes
 * @param excludedAsStraggler true if this device was excluded to prevent degrading makespan
 */
public record ShardAllocation(
        int rank,
        long units,
        double shareRatio,
        double estimatedComputeMs,
        long estimatedMemoryBytes,
        boolean excludedAsStraggler) {

    /**
     * Constructs and validates a {@link ShardAllocation}.
     *
     * @param rank device rank (non-negative)
     * @param units assigned units (non-negative)
     * @param shareRatio fractional share in [0, 1]
     * @param estimatedComputeMs compute time in ms (non-negative)
     * @param estimatedMemoryBytes memory required in bytes (non-negative)
     * @param excludedAsStraggler whether excluded as straggler
     */
    public ShardAllocation {
        if (rank < 0) {
            throw new IllegalArgumentException("rank must be non-negative: " + rank);
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
    }
}
