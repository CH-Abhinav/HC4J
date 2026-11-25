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

package hc4j.examples.llm;

import hc4j.collective.ExecutionReport;
import hc4j.collective.RolePlan;
import java.util.Objects;

/**
 * Telemetry event emitted for a single transformer forward step capturing live Prefill/Decode
 * auto-morphing role plans and optional distributed Split-KV execution metrics.
 *
 * @param phase           execution phase ({@link Phase#PREFILL} or {@link Phase#DECODE})
 * @param seqLen          number of tokens processed in this step
 * @param totalContextLen total sequence length accumulated in the KV-cache after this step
 * @param rolePlan        active multi-operator role plan (null in single-GPU mode without a collective)
 * @param executionReport distributed execution report when Split-KV executed across &ge;2 workers (null otherwise)
 */
public record PhaseEvent(
        Phase phase,
        int seqLen,
        int totalContextLen,
        RolePlan rolePlan,
        ExecutionReport executionReport
) {

    /**
     * Execution phase of a transformer forward pass.
     */
    public enum Phase {
        /** Multi-token prompt ingestion phase ({@code S >= 1} at initial context or {@code S > 1}). */
        PREFILL,
        /** Single-token autoregressive decode step ({@code S == 1}). */
        DECODE
    }

    /**
     * Compact constructor validating non-null phase and positive counts.
     */
    public PhaseEvent {
        Objects.requireNonNull(phase, "phase must not be null");
        if (seqLen <= 0) {
            throw new IllegalArgumentException("seqLen must be positive: " + seqLen);
        }
        if (totalContextLen < seqLen) {
            throw new IllegalArgumentException(
                    "totalContextLen (" + totalContextLen + ") must be >= seqLen (" + seqLen + ")");
        }
    }

    /**
     * Returns a concise human-readable summary of this phase transition and any Split-KV telemetry.
     *
     * @return formatted summary string
     */
    public String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append(phase.name()).append(": S=").append(seqLen).append(" T=").append(totalContextLen);
        if (rolePlan != null) {
            sb.append(" regime=").append(rolePlan.activeRegime())
                    .append(" reclaimedGpus=").append(rolePlan.reclaimedGpuCount())
                    .append(" cpuReclaimed=").append(rolePlan.cpuReclaimed());
        }
        if (executionReport != null) {
            sb.append(" splitKv[executedGpus=").append(executionReport.executedGpuRanks())
                    .append(", prunedGpus=").append(executionReport.prunedGpuRanks())
                    .append(", cpuInPlaceUnits=").append(executionReport.cpuInPlaceUnits())
                    .append(", wallMs=").append(String.format("%.2f", executionReport.wallTimeMs()))
                    .append("]");
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return summary();
    }
}
