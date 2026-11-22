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
 * Hardware profile of the host CPU node, modeling shared system memory, aggregate PCIe throughput,
 * and CPU compute throughput for hybrid multi-device execution.
 *
 * @param name readable name or description of the host CPU
 * @param ramBytes total available host system RAM in bytes
 * @param computeGflops sustained FP32 compute throughput of the host CPU in GFLOP/s
 * @param ramBandwidthGbps memory bandwidth to host system RAM in GB/s (e.g. DDR4/DDR5)
 * @param aggregatePcieGbps aggregate throughput of the host PCIe root complex in GB/s
 */
public record HostCpuProfile(
        String name,
        long ramBytes,
        double computeGflops,
        double ramBandwidthGbps,
        double aggregatePcieGbps) {

    /**
     * Constructs and validates a {@link HostCpuProfile}.
     *
     * @param name CPU description (non-null)
     * @param ramBytes host RAM capacity in bytes (non-negative)
     * @param computeGflops compute throughput in GFLOP/s (non-negative)
     * @param ramBandwidthGbps RAM bandwidth in GB/s (non-negative)
     * @param aggregatePcieGbps aggregate PCIe throughput in GB/s (non-negative)
     * @throws NullPointerException if {@code name} is null
     * @throws IllegalArgumentException if any numeric parameter is negative
     */
    public HostCpuProfile {
        Objects.requireNonNull(name, "name must not be null");
        if (ramBytes < 0) {
            throw new IllegalArgumentException("ramBytes must be non-negative: " + ramBytes);
        }
        if (computeGflops < 0.0) {
            throw new IllegalArgumentException("computeGflops must be non-negative: " + computeGflops);
        }
        if (ramBandwidthGbps < 0.0) {
            throw new IllegalArgumentException("ramBandwidthGbps must be non-negative: " + ramBandwidthGbps);
        }
        if (aggregatePcieGbps < 0.0) {
            throw new IllegalArgumentException("aggregatePcieGbps must be non-negative: " + aggregatePcieGbps);
        }
    }

    /**
     * Returns a disabled {@link HostCpuProfile} with zero resources, indicating GPU-only execution.
     *
     * @return a disabled host CPU profile
     */
    public static HostCpuProfile disabled() {
        return new HostCpuProfile("Disabled", 0L, 0.0, 0.0, 0.0);
    }

    /**
     * Factory method creating an active {@link HostCpuProfile}.
     *
     * @param name CPU description
     * @param ramBytes available RAM capacity in bytes
     * @param computeGflops compute throughput in GFLOP/s
     * @param ramBandwidthGbps system RAM bandwidth in GB/s
     * @param aggregatePcieGbps aggregate PCIe throughput in GB/s
     * @return a new active host CPU profile
     */
    public static HostCpuProfile of(
            String name,
            long ramBytes,
            double computeGflops,
            double ramBandwidthGbps,
            double aggregatePcieGbps) {
        return new HostCpuProfile(name, ramBytes, computeGflops, ramBandwidthGbps, aggregatePcieGbps);
    }

    /**
     * Returns whether this host CPU profile has non-zero resources enabled for participation.
     *
     * @return true if compute, memory, and bandwidth are positive
     */
    public boolean isEnabled() {
        return computeGflops > 0.0 && ramBytes > 0 && ramBandwidthGbps > 0.0;
    }
}
