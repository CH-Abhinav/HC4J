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
 * One directed edge of a collective's topology graph, as measured by the bandwidth and latency probe.
 *
 * @param src the sending rank
 * @param dst the receiving rank
 * @param gbps measured throughput of the hop in GB/s, including the host round trip
 * @param transport how the hop moves its bytes
 * @param loopback true when both ranks sit on the same physical GPU
 * @param latencyUs measured fixed latency of the hop in microseconds
 */
public record Link(int src, int dst, double gbps, Transport transport, boolean loopback, double latencyUs) {

    /**
     * Backwards-compatible constructor defaulting fixed latency to {@code 0.0} microseconds.
     *
     * @param src the sending rank
     * @param dst the receiving rank
     * @param gbps measured throughput of the hop in GB/s
     * @param transport how the hop moves its bytes
     * @param loopback true when both ranks sit on the same physical GPU
     */
    public Link(int src, int dst, double gbps, Transport transport, boolean loopback) {
        this(src, dst, gbps, transport, loopback, 0.0);
    }

    @Override
    public String toString() {
        return String.format(
                "rank %d -> %d: %.2f GB/s (%.1f µs) %s%s",
                src, dst, gbps, latencyUs, transport, loopback ? " (loopback)" : "");
    }
}
