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
 * Execution telemetry and profiling report for a distributed multi-GPU or hybrid execution.
 *
 * @param operation name of the executed distributed operator (e.g., "matmul", "transformerMlp", "splitKvAttention")
 * @param workloadPlan the underlying workload partitioning plan, or {@code null} for standalone split-KV execution
 * @param rolePlan the multi-operator role assignment plan, or {@code null} for standard GEMM / MLP execution
 * @param wallTimeMs total elapsed end-to-end wall-clock time in milliseconds
 * @param computeTimeMs measured GPU/CPU local compute time in milliseconds
 * @param syncTimeMs measured collective synchronization / AllReduce time in milliseconds
 * @param executedGpuRanks list of physical GPU ranks that actively computed work
 * @param prunedGpuRanks list of GPU ranks pruned or excluded from this collective execution
 * @param cpuInPlaceUnits number of workload units executed concurrently on host CPU in-place
 */
public record ExecutionReport(
        String operation,
        WorkloadPlan workloadPlan,
        RolePlan rolePlan,
        double wallTimeMs,
        double computeTimeMs,
        double syncTimeMs,
        List<Integer> executedGpuRanks,
        List<Integer> prunedGpuRanks,
        long cpuInPlaceUnits
) {

    /**
     * Compact canonical constructor creating an immutable ExecutionReport.
     */
    public ExecutionReport {
        Objects.requireNonNull(operation, "operation must not be null");
        executedGpuRanks = executedGpuRanks == null ? List.of() : List.copyOf(executedGpuRanks);
        prunedGpuRanks = prunedGpuRanks == null ? List.of() : List.copyOf(prunedGpuRanks);
    }
}
