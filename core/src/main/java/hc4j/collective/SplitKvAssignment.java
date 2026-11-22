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

import java.util.Objects;

/**
 * Split-KV token assignment and performance breakdown for a single device node.
 *
 * @param rank device rank or {@link ShardAllocation#HOST_CPU_RANK}
 * @param role assigned functional role
 * @param kvTokens number of KV cache tokens hosted and processed on this device
 * @param estimatedLocalAttnMs estimated local attention compute/memory duration in milliseconds
 * @param estimatedCommMs estimated RPC communication duration in milliseconds (query broadcast + partial reduction)
 * @param estimatedTotalMs estimated total attention makespan in milliseconds
 * @param kvMemoryBytes memory footprint of hosted KV cache on this device in bytes
 */
public record SplitKvAssignment(
        int rank,
        DeviceRole role,
        long kvTokens,
        double estimatedLocalAttnMs,
        double estimatedCommMs,
        double estimatedTotalMs,
        long kvMemoryBytes) {

    /**
     * Constructs and validates a {@link SplitKvAssignment}.
     *
     * @param rank device rank or {@link ShardAllocation#HOST_CPU_RANK}
     * @param role assigned role (non-null)
     * @param kvTokens number of KV tokens (non-negative)
     * @param estimatedLocalAttnMs local attention duration in ms (non-negative)
     * @param estimatedCommMs communication duration in ms (non-negative)
     * @param estimatedTotalMs total duration in ms (non-negative)
     * @param kvMemoryBytes KV cache memory in bytes (non-negative)
     * @throws NullPointerException if {@code role} is null
     * @throws IllegalArgumentException if any numeric parameter is negative or invalid
     */
    public SplitKvAssignment {
        if (rank < 0 && rank != ShardAllocation.HOST_CPU_RANK) {
            throw new IllegalArgumentException("rank must be non-negative or HOST_CPU_RANK: " + rank);
        }
        Objects.requireNonNull(role, "role must not be null");
        if (kvTokens < 0) {
            throw new IllegalArgumentException("kvTokens must be non-negative: " + kvTokens);
        }
        if (estimatedLocalAttnMs < 0.0) {
            throw new IllegalArgumentException("estimatedLocalAttnMs must be non-negative: " + estimatedLocalAttnMs);
        }
        if (estimatedCommMs < 0.0) {
            throw new IllegalArgumentException("estimatedCommMs must be non-negative: " + estimatedCommMs);
        }
        if (estimatedTotalMs < 0.0) {
            throw new IllegalArgumentException("estimatedTotalMs must be non-negative: " + estimatedTotalMs);
        }
        if (kvMemoryBytes < 0) {
            throw new IllegalArgumentException("kvMemoryBytes must be non-negative: " + kvMemoryBytes);
        }
    }
}
