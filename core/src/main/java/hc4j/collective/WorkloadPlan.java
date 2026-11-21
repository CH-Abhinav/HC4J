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

import java.util.List;
import java.util.Objects;

/**
 * A complete sharded workload plan across devices, including makespan and sync timing estimates.
 *
 * @param totalUnits the total workload units distributed across devices
 * @param allocations the per-device shard allocations in device rank order
 * @param activeRanks the ranks actively assigned non-zero workload units
 * @param estimatedMakespanMs the overall estimated makespan in milliseconds (compute + sync)
 * @param estimatedComputeMs the bottleneck compute time in milliseconds across active devices
 * @param estimatedSyncMs the estimated AllReduce synchronization time in milliseconds
 */
public record WorkloadPlan(
        long totalUnits,
        List<ShardAllocation> allocations,
        List<Integer> activeRanks,
        double estimatedMakespanMs,
        double estimatedComputeMs,
        double estimatedSyncMs) {

    /**
     * Constructs and defensively copies a {@link WorkloadPlan}.
     *
     * @param totalUnits total units (non-negative)
     * @param allocations per-device allocations (non-null)
     * @param activeRanks active rank indices (non-null)
     * @param estimatedMakespanMs makespan estimate in ms (non-negative)
     * @param estimatedComputeMs compute estimate in ms (non-negative)
     * @param estimatedSyncMs sync estimate in ms (non-negative)
     */
    public WorkloadPlan {
        if (totalUnits < 0) {
            throw new IllegalArgumentException("totalUnits must be non-negative: " + totalUnits);
        }
        Objects.requireNonNull(allocations, "allocations must not be null");
        Objects.requireNonNull(activeRanks, "activeRanks must not be null");
        allocations = List.copyOf(allocations);
        activeRanks = List.copyOf(activeRanks);
        if (estimatedMakespanMs < 0.0) {
            throw new IllegalArgumentException("estimatedMakespanMs must be non-negative: " + estimatedMakespanMs);
        }
        if (estimatedComputeMs < 0.0) {
            throw new IllegalArgumentException("estimatedComputeMs must be non-negative: " + estimatedComputeMs);
        }
        if (estimatedSyncMs < 0.0) {
            throw new IllegalArgumentException("estimatedSyncMs must be non-negative: " + estimatedSyncMs);
        }
    }
}
