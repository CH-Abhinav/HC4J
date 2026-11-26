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

import hc4j.engine.GpuOutOfMemoryException;
import hc4j.internal.collective.BottleneckTspSolver;
import hc4j.internal.collective.ConvexSpillSolver;
import hc4j.internal.collective.NativeOptimizerBridge;
import hc4j.internal.collective.SplitKvRoleSolver;
import java.util.List;
import java.util.Objects;

/**
 * Exact zero-parameter workload partitioner, multi-tier convex spilling optimizer,
 * and zero-waste straggler reclamation planner for heterogeneous multi-device collectives.
 *
 * <p>Delegates to the native Rust optimizer ({@code hc4rs::collective::planner}) when the
 * shared native library is loaded so all SDKs share one canonical implementation, and falls
 * back to the mathematically equivalent pure-Java solver in headless mode.
 *
 * <p>Implements:
 * <ul>
 *   <li>Theorem 1: Shared-Root All-Subsets Bottleneck-TSP Dynamic Programming in \(O(N^2 2^N)\);</li>
 *   <li>Theorem 2: Zero-Parameter Maximum Spanning Tree (MST) Dendrogram collective clustering;</li>
 *   <li>Theorem 3: 3-Tier Convex Spilling (VRAM &rarr; Host-RAM &rarr; Disk) with explicit Host CPU node,
 *       shared PCIe root-complex contention, and exact \(O(m \log m)\) Min-Heap Integer Minimax allocation;</li>
 *   <li>Theorem 4: Physically grounded pipeline overlap makespan modeling with exact \(2^N - 1\) subset search
 *       guaranteeing Zero-Degradation Monotonicity;</li>
 *   <li>Zero-Waste Straggler Reclamation: repurposing GPUs pruned from Tensor Parallelism as lossless Split-KV
 *       attention workers.</li>
 * </ul>
 */
public final class WorkloadPlanner {

    /** Default fraction of device VRAM available for workload buffers (85%). */
    public static final double DEFAULT_VRAM_UTILIZATION = 0.85;

    private WorkloadPlanner() {
        throw new AssertionError("No instances of WorkloadPlanner");
    }

    /**
     * Generates an optimal {@link WorkloadPlan} using default VRAM utilization and legacy GPU-only parameters.
     *
     * @param devices participating device profiles
     * @param topology measured network topology links
     * @param totalUnits total workload units to partition
     * @param flopsPerUnit sustained floating-point operations per unit
     * @param bytesPerUnit memory footprint in bytes required per unit
     * @param syncBytes total collective AllReduce payload in bytes
     * @return the optimal workload plan
     */
    public static WorkloadPlan plan(
            List<DeviceProfile> devices,
            List<Link> topology,
            long totalUnits,
            double flopsPerUnit,
            long bytesPerUnit,
            long syncBytes) {
        return plan(devices, topology, totalUnits, flopsPerUnit, bytesPerUnit, syncBytes, DEFAULT_VRAM_UTILIZATION);
    }

    /**
     * Generates an optimal {@link WorkloadPlan} with explicit VRAM utilization and legacy GPU-only parameters.
     *
     * @param devices participating device profiles
     * @param topology measured network topology links
     * @param totalUnits total workload units to partition
     * @param flopsPerUnit sustained floating-point operations per unit
     * @param bytesPerUnit memory footprint in bytes required per unit
     * @param syncBytes total collective AllReduce payload in bytes
     * @param vramUtilization fraction of device VRAM allowed (0.0 to 1.0]
     * @return the optimal workload plan
     */
    public static WorkloadPlan plan(
            List<DeviceProfile> devices,
            List<Link> topology,
            long totalUnits,
            double flopsPerUnit,
            long bytesPerUnit,
            long syncBytes,
            double vramUtilization) {
        ConvexSpillSolver.validateInputs(devices, totalUnits, flopsPerUnit, bytesPerUnit, syncBytes, vramUtilization);

        int n = devices.size();
        if (totalUnits > 0 && bytesPerUnit > 0) {
            long totalVramCapacity = 0;
            for (DeviceProfile d : devices) {
                long cap = (long) Math.floor((d.vramBytes() * vramUtilization) / bytesPerUnit);
                totalVramCapacity = ConvexSpillSolver.addSaturating(totalVramCapacity, cap);
            }
            if (totalUnits > totalVramCapacity) {
                long totalUsableBytes = 0;
                for (DeviceProfile d : devices) {
                    totalUsableBytes += (long) (d.vramBytes() * vramUtilization);
                }
                throw new GpuOutOfMemoryException(String.format(
                        "Total workload units (%d) requires %d bytes, which exceeds total usable VRAM capacity (%d bytes) across all %d devices",
                        totalUnits, Math.multiplyExact(totalUnits, bytesPerUnit), totalUsableBytes, n));
            }
        }

        WorkloadSpec spec = new WorkloadSpec(
                totalUnits,
                flopsPerUnit,
                bytesPerUnit,
                syncBytes,
                2.0,
                1,
                vramUtilization,
                0.5,
                16.0,
                0.0,
                0L,
                HostCpuProfile.disabled());

        return plan(devices, topology, spec);
    }

    /**
     * Generates an exact zero-parameter optimized {@link WorkloadPlan} according to the provided {@link WorkloadSpec}.
     *
     * @param devices participating device profiles
     * @param topology measured network topology links
     * @param spec detailed workload specification
     * @return the globally optimal workload plan
     * @throws NullPointerException if {@code devices} or {@code spec} is null
     * @throws IllegalArgumentException if inputs are invalid
     * @throws GpuOutOfMemoryException if workload cannot fit across available memory tiers
     */
    public static WorkloadPlan plan(
            List<DeviceProfile> devices,
            List<Link> topology,
            WorkloadSpec spec) {
        Objects.requireNonNull(devices, "devices must not be null");
        Objects.requireNonNull(spec, "spec must not be null");
        ConvexSpillSolver.validateInputs(
                devices, spec.totalUnits(), spec.flopsPerUnit(), spec.bytesPerUnit(), spec.syncBytes(), spec.vramUtilization());

        if (NativeOptimizerBridge.isNativeAvailable()) {
            return NativeOptimizerBridge.planWork(devices, topology, spec);
        }
        return ConvexSpillSolver.plan(devices, topology, spec);
    }

    /**
     * Generates a multi-operator role assignment plan executing Tensor Parallel shards on primary
     * devices and reclaiming excluded straggler GPUs as lossless Split-KV attention workers.
     *
     * @param devices participating device profiles
     * @param topology measured network topology links
     * @param tpSpec specification for Tensor Parallel operations
     * @param kvSpec specification for Split-KV attention
     * @return the combined multi-operator role plan
     */
    public static RolePlan planRoles(
            List<DeviceProfile> devices,
            List<Link> topology,
            WorkloadSpec tpSpec,
            SplitKvSpec kvSpec) {
        Objects.requireNonNull(devices, "devices must not be null");
        Objects.requireNonNull(tpSpec, "tpSpec must not be null");
        Objects.requireNonNull(kvSpec, "kvSpec must not be null");
        ConvexSpillSolver.validateInputs(
                devices, tpSpec.totalUnits(), tpSpec.flopsPerUnit(), tpSpec.bytesPerUnit(), tpSpec.syncBytes(), tpSpec.vramUtilization());

        if (NativeOptimizerBridge.isNativeAvailable()) {
            return NativeOptimizerBridge.planRoles(devices, topology, tpSpec, kvSpec);
        }
        return SplitKvRoleSolver.planRoles(devices, topology, tpSpec, kvSpec);
    }

    static WorkloadPlan planNative(List<DeviceProfile> devices, List<Link> topology, WorkloadSpec spec) {
        ConvexSpillSolver.validateInputs(
                devices, spec.totalUnits(), spec.flopsPerUnit(), spec.bytesPerUnit(), spec.syncBytes(), spec.vramUtilization());
        return NativeOptimizerBridge.planWork(devices, topology, spec);
    }

    static WorkloadPlan planJavaFallback(List<DeviceProfile> devices, List<Link> topology, WorkloadSpec spec) {
        return ConvexSpillSolver.plan(devices, topology, spec);
    }

    static RolePlan planRolesNative(List<DeviceProfile> devices, List<Link> topology,
                                    WorkloadSpec tpSpec, SplitKvSpec kvSpec) {
        return NativeOptimizerBridge.planRoles(devices, topology, tpSpec, kvSpec);
    }

    static RolePlan planRolesJavaFallback(List<DeviceProfile> devices, List<Link> topology,
                                          WorkloadSpec tpSpec, SplitKvSpec kvSpec) {
        return SplitKvRoleSolver.planRoles(devices, topology, tpSpec, kvSpec);
    }

    static AllSubsetsRingTable precomputeAllSubsetsRing(int n, double[][] bw, double[][] latUs) {
        BottleneckTspSolver.RingTable t = BottleneckTspSolver.precomputeAllSubsetsRing(n, bw, latUs);
        return new AllSubsetsRingTable(t.bottleneckGbps(), t.maxHopLatencyUs(), t.hockneyCostUs());
    }

    static AllSubsetsRingTable precomputeAllSubsetsRingForPayload(
            int n, double[][] bw, double[][] latUs, long syncBytes) {
        BottleneckTspSolver.RingTable t =
                BottleneckTspSolver.precomputeAllSubsetsRingForPayload(n, bw, latUs, syncBytes);
        return new AllSubsetsRingTable(t.bottleneckGbps(), t.maxHopLatencyUs(), t.hockneyCostUs());
    }

    record AllSubsetsRingTable(double[] bottleneckGbps, double[] maxHopLatencyUs, double[] hockneyCostUs) {
        AllSubsetsRingTable(double[] bottleneckGbps, double[] maxHopLatencyUs) {
            this(bottleneckGbps, maxHopLatencyUs, new double[bottleneckGbps.length]);
        }

        double bottleneck(int mask) {
            return (mask >= 0 && mask < bottleneckGbps.length) ? bottleneckGbps[mask] : 1.0;
        }

        double latencyUs(int mask) {
            return (mask >= 0 && mask < maxHopLatencyUs.length) ? maxHopLatencyUs[mask] : 0.0;
        }

        double hockneyUs(int mask) {
            return (mask >= 0 && hockneyCostUs != null && mask < hockneyCostUs.length) ? hockneyCostUs[mask] : 0.0;
        }
    }
}
