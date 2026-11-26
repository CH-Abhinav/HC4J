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
 * Hardware and compute capability profile of a single GPU rank.
 *
 * @param rank the zero-based rank index in the collective
 * @param name the human-readable adapter name
 * @param deviceType the device category ("Discrete", "Integrated", "Virtual", "CPU", "Other")
 * @param vendorId the PCI vendor identifier
 * @param deviceId the PCI device identifier
 * @param vramBytes the usable device-local video memory in bytes
 * @param computeGflops the measured sustained compute throughput in GFLOP/s
 * @param vramBandwidthGbps the measured sustained VRAM copy bandwidth in GB/s ({@code 0.0} when unset)
 */
public record DeviceProfile(
        int rank,
        String name,
        String deviceType,
        int vendorId,
        int deviceId,
        long vramBytes,
        double computeGflops,
        double vramBandwidthGbps) {

    /**
     * Backwards-compatible constructor defaulting {@code vramBandwidthGbps} to {@code 0.0}.
     *
     * @param rank the rank index (non-negative)
     * @param name adapter name (non-null)
     * @param deviceType device category (non-null)
     * @param vendorId vendor ID
     * @param deviceId device ID
     * @param vramBytes usable VRAM bytes (non-negative)
     * @param computeGflops compute throughput in GFLOP/s (non-negative)
     */
    public DeviceProfile(
            int rank,
            String name,
            String deviceType,
            int vendorId,
            int deviceId,
            long vramBytes,
            double computeGflops) {
        this(rank, name, deviceType, vendorId, deviceId, vramBytes, computeGflops, 0.0);
    }

    /**
     * Constructs and validates a {@link DeviceProfile}.
     *
     * @param rank the rank index (non-negative)
     * @param name adapter name (non-null)
     * @param deviceType device category (non-null)
     * @param vendorId vendor ID
     * @param deviceId device ID
     * @param vramBytes usable VRAM bytes (non-negative)
     * @param computeGflops compute throughput in GFLOP/s (non-negative)
     * @param vramBandwidthGbps sustained VRAM bandwidth in GB/s (non-negative)
     */
    public DeviceProfile {
        if (rank < 0) {
            throw new IllegalArgumentException("rank must be non-negative: " + rank);
        }
        Objects.requireNonNull(name, "name must not be null");
        Objects.requireNonNull(deviceType, "deviceType must not be null");
        if (vramBytes < 0) {
            throw new IllegalArgumentException("vramBytes must be non-negative: " + vramBytes);
        }
        if (computeGflops < 0.0) {
            throw new IllegalArgumentException("computeGflops must be non-negative: " + computeGflops);
        }
        if (vramBandwidthGbps < 0.0 || Double.isNaN(vramBandwidthGbps)) {
            throw new IllegalArgumentException("vramBandwidthGbps must be non-negative: " + vramBandwidthGbps);
        }
    }
}
