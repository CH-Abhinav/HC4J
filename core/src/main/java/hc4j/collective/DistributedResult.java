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
 * Result of a distributed collective execution along with its profiling {@link ExecutionReport}.
 *
 * @param <T> type of the computed value (e.g., {@link hc4j.Tensor})
 * @param value computed result value
 * @param report execution and timing report
 */
public record DistributedResult<T>(T value, ExecutionReport report) {

    /**
     * Constructs a DistributedResult validating non-null report.
     */
    public DistributedResult {
        Objects.requireNonNull(report, "report must not be null");
    }
}
