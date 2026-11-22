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
 * Detailed specification for a parallel workload, encompassing arithmetic intensity, communication
 * volume, pipeline micro-batching overlap, multi-tier memory spilling parameters, and host CPU offload.
 *
 * @param totalUnits total number of workload units to distribute
 * @param flopsPerUnit floating-point operations required per unit
 * @param bytesPerUnit working memory required per unit in bytes
 * @param syncBytes total bytes exchanged across participating devices in collective sync (e.g. AllReduce)
 * @param ioMultiplier memory I/O factor &kappa; (1.0 for read-only weights, 2.0 for read-write activations)
 * @param microBatches number of pipeline micro-batches {@code P >= 1}
 * @param vramUtilization maximum fraction of VRAM allowed for the workload in {@code (0, 1]}
 * @param streamingOverlapGamma overlap efficiency &gamma; &isin; [0, 1] for double-buffered PCIe streaming
 * @param defaultPcieGbps baseline per-GPU PCIe bandwidth in GB/s
 * @param diskBandwidthGbps Tier 2 NVMe disk spill bandwidth in GB/s (0.0 to disable disk spill)
 * @param diskCapacityBytes Tier 2 disk spill capacity limit in bytes (0 to disable disk spill)
 * @param hostCpu explicit host CPU node profile (or {@link HostCpuProfile#disabled()} for GPU-only planning)
 */
public record WorkloadSpec(
        long totalUnits,
        double flopsPerUnit,
        long bytesPerUnit,
        long syncBytes,
        double ioMultiplier,
        int microBatches,
        double vramUtilization,
        double streamingOverlapGamma,
        double defaultPcieGbps,
        double diskBandwidthGbps,
        long diskCapacityBytes,
        HostCpuProfile hostCpu) {

    /**
     * Constructs and validates a {@link WorkloadSpec}.
     *
     * @param totalUnits total workload units (non-negative)
     * @param flopsPerUnit compute FLOPs per unit (non-negative)
     * @param bytesPerUnit memory bytes per unit (non-negative)
     * @param syncBytes collective sync bytes (non-negative)
     * @param ioMultiplier I/O factor &kappa; (non-negative)
     * @param microBatches micro-batches (at least 1)
     * @param vramUtilization fraction in (0, 1]
     * @param streamingOverlapGamma overlap gamma in [0, 1]
     * @param defaultPcieGbps PCIe bandwidth in GB/s (non-negative)
     * @param diskBandwidthGbps disk bandwidth in GB/s (non-negative)
     * @param diskCapacityBytes disk budget in bytes (non-negative)
     * @param hostCpu host CPU profile (non-null)
     * @throws NullPointerException if {@code hostCpu} is null
     * @throws IllegalArgumentException if any numeric argument is out of bounds
     */
    public WorkloadSpec {
        if (totalUnits < 0) {
            throw new IllegalArgumentException("totalUnits must be non-negative: " + totalUnits);
        }
        if (flopsPerUnit < 0.0) {
            throw new IllegalArgumentException("flopsPerUnit must be non-negative: " + flopsPerUnit);
        }
        if (bytesPerUnit < 0) {
            throw new IllegalArgumentException("bytesPerUnit must be non-negative: " + bytesPerUnit);
        }
        if (syncBytes < 0) {
            throw new IllegalArgumentException("syncBytes must be non-negative: " + syncBytes);
        }
        if (ioMultiplier < 0.0) {
            throw new IllegalArgumentException("ioMultiplier must be non-negative: " + ioMultiplier);
        }
        if (microBatches < 1) {
            throw new IllegalArgumentException("microBatches must be at least 1: " + microBatches);
        }
        if (vramUtilization <= 0.0 || vramUtilization > 1.0) {
            throw new IllegalArgumentException("vramUtilization must be in (0, 1]: " + vramUtilization);
        }
        if (streamingOverlapGamma < 0.0 || streamingOverlapGamma > 1.0) {
            throw new IllegalArgumentException("streamingOverlapGamma must be in [0, 1]: " + streamingOverlapGamma);
        }
        if (defaultPcieGbps < 0.0) {
            throw new IllegalArgumentException("defaultPcieGbps must be non-negative: " + defaultPcieGbps);
        }
        if (diskBandwidthGbps < 0.0) {
            throw new IllegalArgumentException("diskBandwidthGbps must be non-negative: " + diskBandwidthGbps);
        }
        if (diskCapacityBytes < 0) {
            throw new IllegalArgumentException("diskCapacityBytes must be non-negative: " + diskCapacityBytes);
        }
        Objects.requireNonNull(hostCpu, "hostCpu must not be null");
    }

    /**
     * Returns the physically grounded pipeline overlap fraction &alpha; = (P - 1) / P.
     *
     * @return overlap fraction in [0, 1)
     */
    public double overlapAlpha() {
        return (microBatches - 1.0) / microBatches;
    }

    /**
     * Creates a {@link Builder} initialized with fundamental workload properties.
     *
     * @param totalUnits total units to partition
     * @param flopsPerUnit compute FLOPs per unit
     * @param bytesPerUnit memory bytes per unit
     * @return a new builder
     */
    public static Builder builder(long totalUnits, double flopsPerUnit, long bytesPerUnit) {
        return new Builder(totalUnits, flopsPerUnit, bytesPerUnit);
    }

    /**
     * Convenience factory for Prefill-style workloads (high arithmetic intensity, &kappa; = 1.0).
     *
     * @param totalUnits total tokens or units
     * @param flopsPerUnit FLOPs per unit
     * @param bytesPerUnit bytes per unit
     * @param syncBytes sync volume in bytes
     * @return configured WorkloadSpec
     */
    public static WorkloadSpec prefill(long totalUnits, double flopsPerUnit, long bytesPerUnit, long syncBytes) {
        return builder(totalUnits, flopsPerUnit, bytesPerUnit)
                .syncBytes(syncBytes)
                .ioMultiplier(1.0)
                .build();
    }

    /**
     * Convenience factory for Decode-style workloads (memory-bandwidth bound, low arithmetic intensity, &kappa; = 1.0).
     *
     * @param totalUnits total tokens or units
     * @param flopsPerUnit FLOPs per unit
     * @param bytesPerUnit bytes per unit
     * @param syncBytes sync volume in bytes
     * @return configured WorkloadSpec
     */
    public static WorkloadSpec decode(long totalUnits, double flopsPerUnit, long bytesPerUnit, long syncBytes) {
        return builder(totalUnits, flopsPerUnit, bytesPerUnit)
                .syncBytes(syncBytes)
                .ioMultiplier(1.0)
                .build();
    }

    /**
     * Convenience factory for a standard GPU-only workload without spilling or host CPU.
     *
     * @param totalUnits total units
     * @param flopsPerUnit compute FLOPs per unit
     * @param bytesPerUnit memory bytes per unit
     * @param syncBytes collective sync bytes
     * @return configured WorkloadSpec
     */
    public static WorkloadSpec gpuOnly(long totalUnits, double flopsPerUnit, long bytesPerUnit, long syncBytes) {
        return builder(totalUnits, flopsPerUnit, bytesPerUnit)
                .syncBytes(syncBytes)
                .build();
    }

    /**
     * Builder for constructing custom {@link WorkloadSpec} instances.
     */
    public static final class Builder {
        private final long totalUnits;
        private final double flopsPerUnit;
        private final long bytesPerUnit;
        private long syncBytes = 0L;
        private double ioMultiplier = 2.0;
        private int microBatches = 1;
        private double vramUtilization = 0.85;
        private double streamingOverlapGamma = 0.5;
        private double defaultPcieGbps = 16.0;
        private double diskBandwidthGbps = 0.0;
        private long diskCapacityBytes = 0L;
        private HostCpuProfile hostCpu = HostCpuProfile.disabled();

        private Builder(long totalUnits, double flopsPerUnit, long bytesPerUnit) {
            this.totalUnits = totalUnits;
            this.flopsPerUnit = flopsPerUnit;
            this.bytesPerUnit = bytesPerUnit;
        }

        /**
         * Sets collective synchronization volume in bytes.
         *
         * @param syncBytes bytes to sync
         * @return this builder
         */
        public Builder syncBytes(long syncBytes) {
            this.syncBytes = syncBytes;
            return this;
        }

        /**
         * Sets memory I/O multiplier &kappa;.
         *
         * @param ioMultiplier I/O factor
         * @return this builder
         */
        public Builder ioMultiplier(double ioMultiplier) {
            this.ioMultiplier = ioMultiplier;
            return this;
        }

        /**
         * Sets number of pipeline micro-batches.
         *
         * @param microBatches micro-batches (at least 1)
         * @return this builder
         */
        public Builder microBatches(int microBatches) {
            this.microBatches = microBatches;
            return this;
        }

        /**
         * Sets target VRAM utilization fraction.
         *
         * @param vramUtilization fraction in (0, 1]
         * @return this builder
         */
        public Builder vramUtilization(double vramUtilization) {
            this.vramUtilization = vramUtilization;
            return this;
        }

        /**
         * Sets double-buffered PCIe streaming overlap factor &gamma;.
         *
         * @param streamingOverlapGamma gamma in [0, 1]
         * @return this builder
         */
        public Builder streamingOverlapGamma(double streamingOverlapGamma) {
            this.streamingOverlapGamma = streamingOverlapGamma;
            return this;
        }

        /**
         * Sets default PCIe bandwidth in GB/s.
         *
         * @param defaultPcieGbps PCIe throughput
         * @return this builder
         */
        public Builder defaultPcieGbps(double defaultPcieGbps) {
            this.defaultPcieGbps = defaultPcieGbps;
            return this;
        }

        /**
         * Sets Tier 2 NVMe disk spill bandwidth in GB/s.
         *
         * @param diskBandwidthGbps disk throughput
         * @return this builder
         */
        public Builder diskBandwidthGbps(double diskBandwidthGbps) {
            this.diskBandwidthGbps = diskBandwidthGbps;
            return this;
        }

        /**
         * Sets Tier 2 NVMe disk spill capacity budget in bytes.
         *
         * @param diskCapacityBytes disk capacity
         * @return this builder
         */
        public Builder diskCapacityBytes(long diskCapacityBytes) {
            this.diskCapacityBytes = diskCapacityBytes;
            return this;
        }

        /**
         * Configures the host CPU node profile.
         *
         * @param hostCpu host CPU profile
         * @return this builder
         */
        public Builder hostCpu(HostCpuProfile hostCpu) {
            this.hostCpu = Objects.requireNonNull(hostCpu);
            return this;
        }

        /**
         * Builds the {@link WorkloadSpec}.
         *
         * @return new WorkloadSpec
         */
        public WorkloadSpec build() {
            return new WorkloadSpec(
                    totalUnits,
                    flopsPerUnit,
                    bytesPerUnit,
                    syncBytes,
                    ioMultiplier,
                    microBatches,
                    vramUtilization,
                    streamingOverlapGamma,
                    defaultPcieGbps,
                    diskBandwidthGbps,
                    diskCapacityBytes,
                    hostCpu);
        }
    }
}
