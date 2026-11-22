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
 * Specification for Decode Attention over split Key-Value (KV) cache pages.
 *
 * @param totalKvTokens total number of KV tokens across the context to evaluate
 * @param kvBytesPerToken memory footprint of KV cache entries per token in bytes (e.g., {@code 2 * layers * heads * d_head * 2})
 * @param attentionFlopsPerToken floating point operations per token required for attention compute
 * @param queryAndPartialBytes bytes transmitted per remote attention worker for broadcasting query {@code q} and returning partial output {@code (O_i, m_i, l_i)}
 * @param reservedWeightBytesPerGpu reserved bytes per GPU for model weights and scratch memory
 * @param vramUtilization maximum fraction of VRAM allowed for combined weights and KV cache, in {@code (0, 1]}
 */
public record SplitKvSpec(
        long totalKvTokens,
        long kvBytesPerToken,
        double attentionFlopsPerToken,
        long queryAndPartialBytes,
        long reservedWeightBytesPerGpu,
        double vramUtilization) {

    /**
     * Constructs and validates a {@link SplitKvSpec}.
     *
     * @param totalKvTokens total KV tokens (non-negative)
     * @param kvBytesPerToken bytes per token (positive)
     * @param attentionFlopsPerToken FLOPs per token (non-negative)
     * @param queryAndPartialBytes RPC communication bytes per remote worker (non-negative)
     * @param reservedWeightBytesPerGpu reserved weight bytes (non-negative)
     * @param vramUtilization fraction in (0, 1]
     * @throws IllegalArgumentException if any numeric parameter is out of bounds
     */
    public SplitKvSpec {
        if (totalKvTokens < 0) {
            throw new IllegalArgumentException("totalKvTokens must be non-negative: " + totalKvTokens);
        }
        if (kvBytesPerToken <= 0) {
            throw new IllegalArgumentException("kvBytesPerToken must be positive: " + kvBytesPerToken);
        }
        if (attentionFlopsPerToken < 0.0) {
            throw new IllegalArgumentException("attentionFlopsPerToken must be non-negative: " + attentionFlopsPerToken);
        }
        if (queryAndPartialBytes < 0) {
            throw new IllegalArgumentException("queryAndPartialBytes must be non-negative: " + queryAndPartialBytes);
        }
        if (reservedWeightBytesPerGpu < 0) {
            throw new IllegalArgumentException("reservedWeightBytesPerGpu must be non-negative: " + reservedWeightBytesPerGpu);
        }
        if (vramUtilization <= 0.0 || vramUtilization > 1.0) {
            throw new IllegalArgumentException("vramUtilization must be in (0, 1]: " + vramUtilization);
        }
    }
}
