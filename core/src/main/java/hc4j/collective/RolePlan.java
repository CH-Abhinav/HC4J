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
 * Combined multi-operator execution plan mapping devices to Tensor Parallel shards and
 * reclaimed Split-KV attention roles.
 *
 * @param tpPlan optimized Tensor Parallel plan for GEMM / AllReduce operators
 * @param splitKvAssignments per-device Split-KV assignments and role designations
 * @param estimatedTpMs estimated makespan of Tensor Parallel step in milliseconds
 * @param estimatedAttentionMs estimated makespan of Split-KV Attention step in milliseconds
 * @param totalKvTokens total number of KV tokens partitioned across the devices
 * @param reclaimedGpuCount number of GPUs pruned from Tensor Parallelism reclaimed as Split-KV workers
 * @param cpuReclaimed true if the host CPU is actively computing units or hosting KV cache
 * @param activeRegime description of the active optimizer regime
 */
public record RolePlan(
        WorkloadPlan tpPlan,
        List<SplitKvAssignment> splitKvAssignments,
        double estimatedTpMs,
        double estimatedAttentionMs,
        long totalKvTokens,
        int reclaimedGpuCount,
        boolean cpuReclaimed,
        String activeRegime) {

    /**
     * Constructs and validates a {@link RolePlan}.
     *
     * @param tpPlan Tensor Parallel plan (non-null)
     * @param splitKvAssignments list of Split-KV assignments (non-null)
     * @param estimatedTpMs estimated TP makespan in ms (non-negative)
     * @param estimatedAttentionMs estimated attention makespan in ms (non-negative)
     * @param totalKvTokens total KV tokens (non-negative)
     * @param reclaimedGpuCount count of reclaimed GPUs (non-negative)
     * @param cpuReclaimed whether CPU is actively executing
     * @param activeRegime regime descriptor string (non-null)
     * @throws NullPointerException if any non-null parameter is null
     * @throws IllegalArgumentException if any numeric parameter is negative
     */
    public RolePlan {
        Objects.requireNonNull(tpPlan, "tpPlan must not be null");
        Objects.requireNonNull(splitKvAssignments, "splitKvAssignments must not be null");
        Objects.requireNonNull(activeRegime, "activeRegime must not be null");
        splitKvAssignments = List.copyOf(splitKvAssignments);
        if (estimatedTpMs < 0.0) {
            throw new IllegalArgumentException("estimatedTpMs must be non-negative: " + estimatedTpMs);
        }
        if (estimatedAttentionMs < 0.0) {
            throw new IllegalArgumentException("estimatedAttentionMs must be non-negative: " + estimatedAttentionMs);
        }
        if (totalKvTokens < 0) {
            throw new IllegalArgumentException("totalKvTokens must be non-negative: " + totalKvTokens);
        }
        if (reclaimedGpuCount < 0) {
            throw new IllegalArgumentException("reclaimedGpuCount must be non-negative: " + reclaimedGpuCount);
        }
    }
}
