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
 * Functional role assigned to a device node within a multi-device execution plan.
 */
public enum DeviceRole {
    /**
     * Active participant in high-bandwidth Tensor Parallel collective shards (e.g. GEMM / AllReduce).
     */
    TENSOR_PARALLEL,

    /**
     * Reclaimed device (pruned from Tensor Parallel to avoid dragging collective bottleneck)
     * repurposed as a lossless Split-KV attention worker.
     */
    SPLIT_KV_RECLAIMED,

    /**
     * Host CPU executing units in-place using system RAM.
     */
    CPU_IN_PLACE,

    /**
     * Idle device excluded from computation to avoid degrading total makespan.
     */
    IDLE
}
