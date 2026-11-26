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

package hc4j.internal.collective;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import hc4j.collective.DeviceProfile;
import hc4j.collective.DeviceRole;
import hc4j.collective.HostCpuProfile;
import hc4j.collective.Link;
import hc4j.collective.RolePlan;
import hc4j.collective.ShardAllocation;
import hc4j.collective.SplitKvAssignment;
import hc4j.collective.SplitKvSpec;
import hc4j.collective.WorkloadPlan;
import hc4j.collective.WorkloadSpec;
import hc4j.internal.ffi.NativeEngine;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Panama FFM bridge delegating Theorems 1–4 workload optimization, Zero-Waste Split-KV role
 * planning, and lossless online-softmax Split-KV merging to {@code hc4x-ffi} ({@code hc4j.dll} /
 * {@code libhc4j.so} / {@code libhc4j.dylib}) without requiring GPU device initialization.
 */
public final class NativeOptimizerBridge {

    /** C-ABI {@code DeviceProfileC} layout (40 bytes). */
    public static final StructLayout DEVICE_PROFILE_LAYOUT = MemoryLayout.structLayout(
            JAVA_INT.withName("rank"),
            JAVA_INT.withName("vendor_id"),
            JAVA_INT.withName("device_id"),
            JAVA_INT.withName("device_type"),
            JAVA_LONG.withName("vram_bytes"),
            JAVA_DOUBLE.withName("compute_gflops"),
            JAVA_DOUBLE.withName("vram_bandwidth_gbps"));

    /** C-ABI {@code LinkInfo} layout (32 bytes). */
    public static final StructLayout LINK_INFO_LAYOUT = MemoryLayout.structLayout(
            JAVA_INT.withName("src_rank"),
            JAVA_INT.withName("dst_rank"),
            JAVA_DOUBLE.withName("bandwidth_gbps"),
            JAVA_INT.withName("transport"),
            JAVA_INT.withName("loopback"),
            JAVA_DOUBLE.withName("latency_us"));

    /** C-ABI {@code HostCpuProfileC} layout (32 bytes). */
    public static final StructLayout HOST_CPU_PROFILE_LAYOUT = MemoryLayout.structLayout(
            JAVA_LONG.withName("ram_bytes"),
            JAVA_DOUBLE.withName("compute_gflops"),
            JAVA_DOUBLE.withName("ram_bandwidth_gbps"),
            JAVA_DOUBLE.withName("aggregate_pcie_gbps"));

    /** C-ABI {@code WorkloadSpecC} layout (120 bytes). */
    public static final StructLayout WORKLOAD_SPEC_LAYOUT = MemoryLayout.structLayout(
            JAVA_LONG.withName("total_units"),
            JAVA_DOUBLE.withName("flops_per_unit"),
            JAVA_LONG.withName("bytes_per_unit"),
            JAVA_LONG.withName("sync_bytes"),
            JAVA_DOUBLE.withName("io_multiplier"),
            JAVA_DOUBLE.withName("vram_utilization"),
            JAVA_DOUBLE.withName("streaming_overlap_gamma"),
            JAVA_DOUBLE.withName("default_pcie_gbps"),
            JAVA_DOUBLE.withName("disk_bandwidth_gbps"),
            JAVA_LONG.withName("disk_capacity_bytes"),
            JAVA_INT.withName("micro_batches"),
            JAVA_INT.withName("allow_spilling"),
            HOST_CPU_PROFILE_LAYOUT.withName("host_cpu"));

    /** C-ABI {@code ShardAllocationC} layout (72 bytes). */
    public static final StructLayout SHARD_ALLOCATION_LAYOUT = MemoryLayout.structLayout(
            JAVA_INT.withName("rank"),
            JAVA_INT.withName("excluded_as_straggler"),
            JAVA_LONG.withName("units"),
            JAVA_DOUBLE.withName("share_ratio"),
            JAVA_DOUBLE.withName("estimated_compute_ms"),
            JAVA_LONG.withName("estimated_memory_bytes"),
            JAVA_LONG.withName("vram_units"),
            JAVA_LONG.withName("host_stream_units"),
            JAVA_LONG.withName("disk_stream_units"),
            JAVA_LONG.withName("cpu_in_place_units"));

    /** C-ABI {@code WorkloadPlanHeaderC} layout (40 bytes). */
    public static final StructLayout WORKLOAD_PLAN_HEADER_LAYOUT = MemoryLayout.structLayout(
            JAVA_LONG.withName("total_units"),
            JAVA_DOUBLE.withName("estimated_makespan_ms"),
            JAVA_DOUBLE.withName("estimated_compute_ms"),
            JAVA_DOUBLE.withName("estimated_sync_ms"),
            JAVA_INT.withName("allocation_count"),
            JAVA_INT.withName("active_rank_count"));

    /** C-ABI {@code SplitKvSpecC} layout (48 bytes). */
    public static final StructLayout SPLIT_KV_SPEC_LAYOUT = MemoryLayout.structLayout(
            JAVA_LONG.withName("total_kv_tokens"),
            JAVA_LONG.withName("kv_bytes_per_token"),
            JAVA_DOUBLE.withName("attention_flops_per_token"),
            JAVA_LONG.withName("query_and_partial_bytes"),
            JAVA_LONG.withName("reserved_weight_bytes_per_gpu"),
            JAVA_DOUBLE.withName("vram_utilization"));

    /** C-ABI {@code SplitKvAssignmentC} layout (48 bytes). */
    public static final StructLayout SPLIT_KV_ASSIGNMENT_LAYOUT = MemoryLayout.structLayout(
            JAVA_INT.withName("rank"),
            JAVA_INT.withName("role"),
            JAVA_LONG.withName("kv_tokens"),
            JAVA_DOUBLE.withName("estimated_local_attn_ms"),
            JAVA_DOUBLE.withName("estimated_comm_ms"),
            JAVA_DOUBLE.withName("estimated_total_ms"),
            JAVA_LONG.withName("kv_memory_bytes"));

    /** C-ABI {@code RolePlanHeaderC} layout (80 bytes). */
    public static final StructLayout ROLE_PLAN_HEADER_LAYOUT = MemoryLayout.structLayout(
            WORKLOAD_PLAN_HEADER_LAYOUT.withName("tp_header"),
            JAVA_DOUBLE.withName("estimated_tp_ms"),
            JAVA_DOUBLE.withName("estimated_attention_ms"),
            JAVA_LONG.withName("total_kv_tokens"),
            JAVA_INT.withName("split_kv_count"),
            JAVA_INT.withName("reclaimed_gpu_count"),
            JAVA_INT.withName("cpu_reclaimed"),
            JAVA_INT.withName("regime_code"));

    private static volatile boolean bindAttempted = false;
    private static volatile boolean handlesReady = false;
    private static MethodHandle planWorkHandle;
    private static MethodHandle planRolesHandle;
    private static MethodHandle mergeSplitKvHandle;
    private static MethodHandle cpuGemmF32Handle;
    private static MethodHandle cpuGemmF32F16Handle;
    private static MethodHandle cpuMlpSliceF32Handle;
    private static MethodHandle cpuSplitKvAttnF32Handle;
    private static volatile boolean cpuKernelsReady;

    private NativeOptimizerBridge() {
        throw new AssertionError("No instances of NativeOptimizerBridge");
    }

    /**
     * Returns {@code true} when the shared native library is present and exposes
     * the optimizer and Split-KV merge C-ABI symbols.
     *
     * @return true if native optimizer entry points are callable
     */
    public static boolean isNativeAvailable() {
        if (handlesReady) {
            return true;
        }
        if (bindAttempted) {
            return false;
        }
        return ensureHandlesBound();
    }

    @SuppressWarnings("restricted")
    private static synchronized boolean ensureHandlesBound() {
        if (handlesReady) {
            return true;
        }
        if (bindAttempted) {
            return false;
        }
        bindAttempted = true;
        try {
            SymbolLookup lookup = NativeEngine.tryLoadLibrarySymbols();
            if (lookup == null) {
                return false;
            }
            var planWorkSym = lookup.find("hc4j_optimizer_plan_work");
            var planRolesSym = lookup.find("hc4j_optimizer_plan_roles");
            var mergeKvSym = lookup.find("hc4j_collective_merge_split_kv_f32");
            if (planWorkSym.isEmpty() || planRolesSym.isEmpty() || mergeKvSym.isEmpty()) {
                return false;
            }

            Linker linker = Linker.nativeLinker();
            planWorkHandle = linker.downcallHandle(planWorkSym.get(), FunctionDescriptor.of(
                    JAVA_INT,
                    JAVA_INT, ADDRESS,
                    JAVA_INT, ADDRESS,
                    ADDRESS,
                    ADDRESS,
                    ADDRESS, JAVA_INT,
                    ADDRESS, JAVA_INT));

            planRolesHandle = linker.downcallHandle(planRolesSym.get(), FunctionDescriptor.of(
                    JAVA_INT,
                    JAVA_INT, ADDRESS,
                    JAVA_INT, ADDRESS,
                    ADDRESS, ADDRESS,
                    ADDRESS,
                    ADDRESS, JAVA_INT,
                    ADDRESS, JAVA_INT,
                    ADDRESS, JAVA_INT));

            mergeSplitKvHandle = linker.downcallHandle(mergeKvSym.get(), FunctionDescriptor.of(
                    JAVA_INT,
                    JAVA_INT, JAVA_INT, JAVA_INT,
                    ADDRESS, ADDRESS, ADDRESS));

            var cpuGemmSym = lookup.find("hc4j_cpu_gemm_f32");
            var cpuGemmF16Sym = lookup.find("hc4j_cpu_gemm_f32_f16");
            var cpuMlpSym = lookup.find("hc4j_cpu_mlp_slice_f32");
            var cpuKvSym = lookup.find("hc4j_cpu_split_kv_attn_f32");
            if (cpuGemmSym.isPresent() && cpuGemmF16Sym.isPresent()
                    && cpuMlpSym.isPresent() && cpuKvSym.isPresent()) {
                cpuGemmF32Handle = linker.downcallHandle(cpuGemmSym.get(), FunctionDescriptor.of(
                        JAVA_INT,
                        ADDRESS, ADDRESS, ADDRESS,
                        JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
                cpuGemmF32F16Handle = linker.downcallHandle(cpuGemmF16Sym.get(), FunctionDescriptor.of(
                        JAVA_INT,
                        ADDRESS, ADDRESS, ADDRESS,
                        JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
                cpuMlpSliceF32Handle = linker.downcallHandle(cpuMlpSym.get(), FunctionDescriptor.of(
                        JAVA_INT,
                        ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS,
                        JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
                        JAVA_INT, JAVA_INT, JAVA_INT));
                cpuSplitKvAttnF32Handle = linker.downcallHandle(cpuKvSym.get(), FunctionDescriptor.of(
                        JAVA_INT,
                        ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS,
                        JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
                cpuKernelsReady = true;
            }

            handlesReady = true;
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    /**
     * Returns {@code true} when the native library exposes the multi-threaded SIMD CPU kernels.
     *
     * @return true if native CPU kernels are callable
     */
    public static boolean isCpuKernelsAvailable() {
        return isNativeAvailable() && cpuKernelsReady;
    }

    /**
     * Executes native multi-threaded SIMD {@code C = A * B} over float32 host segments.
     *
     * @param aSeg float32 matrix A {@code [m, k]}
     * @param bSeg float32 matrix B {@code [k, n]} (or {@code [n, k]} with transB flag)
     * @param outSeg output float32 matrix C {@code [m, n]}
     * @param m rows of A and C
     * @param n columns of C
     * @param k inner dimension
     * @param flags transpose flags ({@code 1 = TRANS_A, 2 = TRANS_B})
     */
    public static void cpuGemmF32(MemorySegment aSeg, MemorySegment bSeg, MemorySegment outSeg,
                                  int m, int n, int k, int flags) {
        if (!isCpuKernelsAvailable()) {
            throw new IllegalStateException("Native CPU GEMM kernel is not available");
        }
        try (Arena arena = Arena.ofConfined()) {
            long aBytes = (long) m * k * 4L;
            long bBytes = (long) k * n * 4L;
            long cBytes = (long) m * n * 4L;
            MemorySegment nA = aSeg.isNative() ? aSeg : copyToNative(arena, aSeg, aBytes);
            MemorySegment nB = bSeg.isNative() ? bSeg : copyToNative(arena, bSeg, bBytes);
            MemorySegment nC = outSeg.isNative() ? outSeg : arena.allocate(cBytes, 16);
            int status = (int) cpuGemmF32Handle.invokeExact(nA, nB, nC, m, n, k, flags);
            NativeEngine.checkStatus(status, "hc4j_cpu_gemm_f32");
            if (!outSeg.isNative()) {
                MemorySegment.copy(nC, 0, outSeg, 0, cBytes);
            }
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native CPU GEMM failed", t);
        }
    }

    /**
     * Executes native multi-threaded SIMD {@code C = A_f32 * B_f16} over host segments.
     *
     * @param aSeg float32 matrix A {@code [m, k]}
     * @param bF16Seg float16 matrix B {@code [k, n]} (or {@code [n, k]} with transB flag)
     * @param outSeg output float32 matrix C {@code [m, n]}
     * @param m rows of A and C
     * @param n columns of C
     * @param k inner dimension
     * @param flags transpose flags ({@code 2 = TRANS_B})
     */
    public static void cpuGemmF32F16(MemorySegment aSeg, MemorySegment bF16Seg, MemorySegment outSeg,
                                     int m, int n, int k, int flags) {
        if (!isCpuKernelsAvailable()) {
            throw new IllegalStateException("Native CPU F16 GEMM kernel is not available");
        }
        try (Arena arena = Arena.ofConfined()) {
            long aBytes = (long) m * k * 4L;
            long bBytes = (long) k * n * 2L;
            long cBytes = (long) m * n * 4L;
            MemorySegment nA = aSeg.isNative() ? aSeg : copyToNative(arena, aSeg, aBytes);
            MemorySegment nB = bF16Seg.isNative() ? bF16Seg : copyToNative(arena, bF16Seg, bBytes);
            MemorySegment nC = outSeg.isNative() ? outSeg : arena.allocate(cBytes, 16);
            int status = (int) cpuGemmF32F16Handle.invokeExact(nA, nB, nC, m, n, k, flags);
            NativeEngine.checkStatus(status, "hc4j_cpu_gemm_f32_f16");
            if (!outSeg.isNative()) {
                MemorySegment.copy(nC, 0, outSeg, 0, cBytes);
            }
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native CPU F16 GEMM failed", t);
        }
    }

    /**
     * Executes native multi-threaded SIMD CPU-in-Place MLP slice computation.
     *
     * @param xSeg input activations {@code [m, dIn]}
     * @param w1Seg first projection weights {@code [dIn, h]}
     * @param b1Seg optional bias {@code [h]} (may be {@code null})
     * @param w2Seg second projection weights {@code [h, dOut]}
     * @param outSeg output buffer {@code [m, dOut]}
     * @param m batch / token count
     * @param dIn input feature dimension
     * @param h total hidden width
     * @param dOut output feature dimension
     * @param hStart starting hidden offset
     * @param units hidden width assigned to CPU
     * @param actCode activation code ({@code -1 = NONE, 0 = RELU, 3 = GELU, 4 = SILU})
     */
    public static void cpuMlpSliceF32(MemorySegment xSeg, MemorySegment w1Seg, MemorySegment b1Seg,
                                      MemorySegment w2Seg, MemorySegment outSeg,
                                      int m, int dIn, int h, int dOut, int hStart, int units, int actCode) {
        if (!isCpuKernelsAvailable()) {
            throw new IllegalStateException("Native CPU MLP kernel is not available");
        }
        try (Arena arena = Arena.ofConfined()) {
            long xBytes = (long) m * dIn * 4L;
            long w1Bytes = (long) dIn * h * 4L;
            long b1Bytes = (long) h * 4L;
            long w2Bytes = (long) h * dOut * 4L;
            long outBytes = (long) m * dOut * 4L;
            MemorySegment nX = xSeg.isNative() ? xSeg : copyToNative(arena, xSeg, xBytes);
            MemorySegment nW1 = w1Seg.isNative() ? w1Seg : copyToNative(arena, w1Seg, w1Bytes);
            MemorySegment nB1 = (b1Seg == null) ? MemorySegment.NULL
                    : (b1Seg.isNative() ? b1Seg : copyToNative(arena, b1Seg, b1Bytes));
            MemorySegment nW2 = w2Seg.isNative() ? w2Seg : copyToNative(arena, w2Seg, w2Bytes);
            MemorySegment nOut = outSeg.isNative() ? outSeg : arena.allocate(outBytes, 16);
            int status = (int) cpuMlpSliceF32Handle.invokeExact(
                    nX, nW1, nB1, nW2, nOut, m, dIn, h, dOut, hStart, units, actCode);
            NativeEngine.checkStatus(status, "hc4j_cpu_mlp_slice_f32");
            if (!outSeg.isNative()) {
                MemorySegment.copy(nOut, 0, outSeg, 0, outBytes);
            }
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native CPU MLP slice failed", t);
        }
    }

    /**
     * Executes native multi-threaded SIMD Split-KV CPU attention over host RAM KV shards.
     *
     * @param qSeg query tensor {@code [numHeads, headDim]}
     * @param kSeg key shard {@code [numKvHeads, tokens, headDim]}
     * @param vSeg value shard {@code [numKvHeads, tokens, headDim]}
     * @param outO output partial {@code [numHeads * headDim]}
     * @param outStats output {@code (m, l)} pairs {@code [numHeads * 2]}
     * @param tokens local cached token count
     * @param numHeads query head count
     * @param numKvHeads KV head count
     * @param headDim per-head dimension
     * @param scale attention dot-product scale
     */
    public static void cpuSplitKvAttnF32(MemorySegment qSeg, MemorySegment kSeg, MemorySegment vSeg,
                                         float[] outO, float[] outStats,
                                         int tokens, int numHeads, int numKvHeads, int headDim, float scale) {
        if (!isCpuKernelsAvailable()) {
            throw new IllegalStateException("Native CPU Split-KV attention kernel is not available");
        }
        try (Arena arena = Arena.ofConfined()) {
            long qBytes = (long) numHeads * headDim * 4L;
            long kvBytes = (long) numKvHeads * tokens * headDim * 4L;
            long oBytes = qBytes;
            long sBytes = (long) numHeads * 2L * 4L;
            MemorySegment nQ = qSeg.isNative() ? qSeg : copyToNative(arena, qSeg, qBytes);
            MemorySegment nK = (kvBytes == 0) ? arena.allocate(4L, 4)
                    : (kSeg.isNative() ? kSeg : copyToNative(arena, kSeg, kvBytes));
            MemorySegment nV = (kvBytes == 0) ? arena.allocate(4L, 4)
                    : (vSeg.isNative() ? vSeg : copyToNative(arena, vSeg, kvBytes));
            MemorySegment nO = arena.allocate(oBytes, 16);
            MemorySegment nS = arena.allocate(sBytes, 16);
            int status = (int) cpuSplitKvAttnF32Handle.invokeExact(
                    nQ, nK, nV, nO, nS, tokens, numHeads, numKvHeads, headDim, Float.floatToRawIntBits(scale));
            NativeEngine.checkStatus(status, "hc4j_cpu_split_kv_attn_f32");
            MemorySegment.copy(nO, JAVA_FLOAT, 0, outO, 0, numHeads * headDim);
            MemorySegment.copy(nS, JAVA_FLOAT, 0, outStats, 0, numHeads * 2);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native CPU Split-KV attention failed", t);
        }
    }

    private static MemorySegment copyToNative(Arena arena, MemorySegment src, long byteLength) {
        if (byteLength <= 0) {
            return arena.allocate(4L, 4);
        }
        MemorySegment dst = arena.allocate(byteLength, 16);
        MemorySegment.copy(src, 0, dst, 0, byteLength);
        return dst;
    }

    private static void requireNativeHandles() {
        if (!isNativeAvailable()) {
            throw new IllegalStateException("HC4J native optimizer symbols are not available");
        }
    }

    /**
     * Executes Exact Theorems 1–4 optimization via the native {@code hc4j_optimizer_plan_work} C-ABI.
     *
     * @param devices participating device profiles
     * @param topology measured directed topology links
     * @param spec workload specification
     * @return globally optimal {@link WorkloadPlan}
     */
    public static WorkloadPlan planWork(List<DeviceProfile> devices, List<Link> topology, WorkloadSpec spec) {
        return planWork(devices, topology, spec, true);
    }

    /**
     * Executes Exact Theorems 1–4 optimization via {@code hc4j_optimizer_plan_work}.
     *
     * @param devices participating device profiles
     * @param topology measured directed topology links
     * @param spec workload specification
     * @param allowSpilling whether host RAM / disk spilling tiers are permitted
     * @return globally optimal {@link WorkloadPlan}
     */
    public static WorkloadPlan planWork(List<DeviceProfile> devices, List<Link> topology,
                                        WorkloadSpec spec, boolean allowSpilling) {
        Objects.requireNonNull(devices, "devices must not be null");
        Objects.requireNonNull(spec, "spec must not be null");
        requireNativeHandles();

        int deviceCount = devices.size();
        int linkCount = (topology == null) ? 0 : topology.size();
        int allocCap = deviceCount + 1;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment devSeg = marshalDevices(arena, devices);
            MemorySegment linkSeg = marshalLinks(arena, topology);
            MemorySegment specSeg = marshalWorkloadSpec(arena, spec, allowSpilling);

            MemorySegment headerSeg = arena.allocate(WORKLOAD_PLAN_HEADER_LAYOUT);
            MemorySegment allocSeg = arena.allocate(SHARD_ALLOCATION_LAYOUT, allocCap);
            MemorySegment activeRanksSeg = arena.allocate(JAVA_INT, allocCap);

            int status = (int) planWorkHandle.invokeExact(
                    deviceCount, devSeg,
                    linkCount, linkSeg,
                    specSeg,
                    headerSeg,
                    allocSeg, allocCap,
                    activeRanksSeg, allocCap);
            NativeEngine.checkStatus(status, "WorkloadPlanner.plan");

            return readWorkloadPlan(headerSeg, 0L, allocSeg, allocCap, activeRanksSeg, allocCap);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native optimizer planWork failed", t);
        }
    }

    /**
     * Executes joint Tensor Parallel + Zero-Waste Split-KV role optimization via {@code hc4j_optimizer_plan_roles}.
     *
     * @param devices participating device profiles
     * @param topology measured directed topology links
     * @param tpSpec Tensor Parallel specification
     * @param kvSpec Split-KV attention specification
     * @return combined {@link RolePlan}
     */
    public static RolePlan planRoles(List<DeviceProfile> devices, List<Link> topology,
                                     WorkloadSpec tpSpec, SplitKvSpec kvSpec) {
        Objects.requireNonNull(devices, "devices must not be null");
        Objects.requireNonNull(tpSpec, "tpSpec must not be null");
        Objects.requireNonNull(kvSpec, "kvSpec must not be null");
        requireNativeHandles();

        int deviceCount = devices.size();
        int linkCount = (topology == null) ? 0 : topology.size();
        int cap = deviceCount + 1;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment devSeg = marshalDevices(arena, devices);
            MemorySegment linkSeg = marshalLinks(arena, topology);
            MemorySegment tpSpecSeg = marshalWorkloadSpec(arena, tpSpec, true);
            MemorySegment kvSpecSeg = marshalSplitKvSpec(arena, kvSpec);

            MemorySegment headerSeg = arena.allocate(ROLE_PLAN_HEADER_LAYOUT);
            MemorySegment tpAllocSeg = arena.allocate(SHARD_ALLOCATION_LAYOUT, cap);
            MemorySegment tpActiveRanksSeg = arena.allocate(JAVA_INT, cap);
            MemorySegment splitKvSeg = arena.allocate(SPLIT_KV_ASSIGNMENT_LAYOUT, cap);

            int status = (int) planRolesHandle.invokeExact(
                    deviceCount, devSeg,
                    linkCount, linkSeg,
                    tpSpecSeg, kvSpecSeg,
                    headerSeg,
                    tpAllocSeg, cap,
                    tpActiveRanksSeg, cap,
                    splitKvSeg, cap);
            NativeEngine.checkStatus(status, "WorkloadPlanner.planRoles");

            WorkloadPlan tpPlan = readWorkloadPlan(headerSeg, 0L, tpAllocSeg, cap, tpActiveRanksSeg, cap);

            long baseHeaderSize = WORKLOAD_PLAN_HEADER_LAYOUT.byteSize(); // 40 bytes
            double estimatedTpMs = headerSeg.get(JAVA_DOUBLE, baseHeaderSize);
            double estimatedAttentionMs = headerSeg.get(JAVA_DOUBLE, baseHeaderSize + 8L);
            long totalKvTokens = headerSeg.get(JAVA_LONG, baseHeaderSize + 16L);
            int splitKvCount = Math.min(cap, headerSeg.get(JAVA_INT, baseHeaderSize + 24L));
            int reclaimedGpuCount = headerSeg.get(JAVA_INT, baseHeaderSize + 28L);
            boolean cpuReclaimed = headerSeg.get(JAVA_INT, baseHeaderSize + 32L) != 0;
            int regimeCode = headerSeg.get(JAVA_INT, baseHeaderSize + 36L);

            List<SplitKvAssignment> assignments = new ArrayList<>(splitKvCount);
            long kvStride = SPLIT_KV_ASSIGNMENT_LAYOUT.byteSize();
            for (int i = 0; i < splitKvCount; i++) {
                long off = i * kvStride;
                int rank = splitKvSeg.get(JAVA_INT, off);
                int roleCode = splitKvSeg.get(JAVA_INT, off + 4L);
                long kvTokens = splitKvSeg.get(JAVA_LONG, off + 8L);
                double localAttnMs = splitKvSeg.get(JAVA_DOUBLE, off + 16L);
                double commMs = splitKvSeg.get(JAVA_DOUBLE, off + 24L);
                double totalMs = splitKvSeg.get(JAVA_DOUBLE, off + 32L);
                long kvMemBytes = splitKvSeg.get(JAVA_LONG, off + 40L);

                assignments.add(new SplitKvAssignment(
                        rank, deviceRoleFromCode(roleCode), kvTokens, localAttnMs, commMs, totalMs, kvMemBytes));
            }

            return new RolePlan(
                    tpPlan,
                    assignments,
                    estimatedTpMs,
                    estimatedAttentionMs,
                    totalKvTokens,
                    reclaimedGpuCount,
                    cpuReclaimed,
                    regimeName(regimeCode));
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native optimizer planRoles failed", t);
        }
    }

    /**
     * Merges Split-KV worker partial outputs {@code partialsO[w][numHeads * headDim]} and
     * log-sum-exp statistics {@code partialsStats[w][2 * numHeads]} into {@code outSeg},
     * delegating to {@code hc4j_collective_merge_split_kv_f32} when available.
     *
     * @param numHeads query head count
     * @param headDim per-head dimension
     * @param partialsO worker partial output vectors
     * @param partialsStats worker {@code [m, l]} statistics vectors
     * @param outSeg destination {@code float} segment sized {@code numHeads * headDim * 4} bytes
     */
    public static void mergeSplitKvF32(int numHeads, int headDim, float[][] partialsO,
                                       float[][] partialsStats, MemorySegment outSeg) {
        Objects.requireNonNull(partialsO, "partialsO must not be null");
        Objects.requireNonNull(partialsStats, "partialsStats must not be null");
        Objects.requireNonNull(outSeg, "outSeg must not be null");
        if (partialsO.length != partialsStats.length) {
            throw new IllegalArgumentException("partialsO and partialsStats lengths must match");
        }
        int workerCount = partialsO.length;
        if (workerCount == 0) {
            outSeg.asSlice(0, (long) numHeads * headDim * 4L).fill((byte) 0);
            return;
        }
        if (!isNativeAvailable()) {
            CpuFallbackKernels.mergeSplitKvPartialsCpu(numHeads, headDim, partialsO, partialsStats, outSeg);
            return;
        }

        long outBytes = (long) numHeads * headDim * 4L;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment ptrPartialsO = arena.allocate(ADDRESS, workerCount);
            MemorySegment ptrPartialsStats = arena.allocate(ADDRESS, workerCount);
            for (int w = 0; w < workerCount; w++) {
                MemorySegment oSeg = arena.allocateFrom(JAVA_FLOAT, partialsO[w]);
                MemorySegment sSeg = arena.allocateFrom(JAVA_FLOAT, partialsStats[w]);
                ptrPartialsO.setAtIndex(ADDRESS, w, oSeg);
                ptrPartialsStats.setAtIndex(ADDRESS, w, sSeg);
            }

            MemorySegment nativeOut = outSeg.isNative() ? outSeg : arena.allocate(outBytes, 16);
            int status = (int) mergeSplitKvHandle.invokeExact(
                    numHeads, headDim, workerCount, ptrPartialsO, ptrPartialsStats, nativeOut);
            NativeEngine.checkStatus(status, "SplitKvSession.mergePartials");
            if (!outSeg.isNative()) {
                MemorySegment.copy(nativeOut, 0, outSeg, 0, outBytes);
            }
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J native Split-KV merge failed", t);
        }
    }

    private static MemorySegment marshalDevices(Arena arena, List<DeviceProfile> devices) {
        int n = devices.size();
        if (n == 0) {
            return MemorySegment.NULL;
        }
        long stride = DEVICE_PROFILE_LAYOUT.byteSize();
        MemorySegment seg = arena.allocate(DEVICE_PROFILE_LAYOUT, n);
        for (int i = 0; i < n; i++) {
            DeviceProfile d = devices.get(i);
            long off = i * stride;
            seg.set(JAVA_INT, off, d.rank());
            seg.set(JAVA_INT, off + 4L, d.vendorId());
            seg.set(JAVA_INT, off + 8L, d.deviceId());
            seg.set(JAVA_INT, off + 12L, deviceTypeCode(d.deviceType()));
            seg.set(JAVA_LONG, off + 16L, d.vramBytes());
            seg.set(JAVA_DOUBLE, off + 24L, d.computeGflops());
            seg.set(JAVA_DOUBLE, off + 32L, d.vramBandwidthGbps());
        }
        return seg;
    }

    private static MemorySegment marshalLinks(Arena arena, List<Link> topology) {
        if (topology == null || topology.isEmpty()) {
            return MemorySegment.NULL;
        }
        int n = topology.size();
        long stride = LINK_INFO_LAYOUT.byteSize();
        MemorySegment seg = arena.allocate(LINK_INFO_LAYOUT, n);
        for (int i = 0; i < n; i++) {
            Link link = topology.get(i);
            long off = i * stride;
            seg.set(JAVA_INT, off, link.src());
            seg.set(JAVA_INT, off + 4L, link.dst());
            seg.set(JAVA_DOUBLE, off + 8L, link.gbps());
            seg.set(JAVA_INT, off + 16L, link.transport().ordinal());
            seg.set(JAVA_INT, off + 20L, link.loopback() ? 1 : 0);
            seg.set(JAVA_DOUBLE, off + 24L, link.latencyUs());
        }
        return seg;
    }

    private static MemorySegment marshalWorkloadSpec(Arena arena, WorkloadSpec spec, boolean allowSpilling) {
        MemorySegment seg = arena.allocate(WORKLOAD_SPEC_LAYOUT);
        seg.set(JAVA_LONG, 0L, spec.totalUnits());
        seg.set(JAVA_DOUBLE, 8L, spec.flopsPerUnit());
        seg.set(JAVA_LONG, 16L, spec.bytesPerUnit());
        seg.set(JAVA_LONG, 24L, spec.syncBytes());
        seg.set(JAVA_DOUBLE, 32L, spec.ioMultiplier());
        seg.set(JAVA_DOUBLE, 40L, spec.vramUtilization());
        seg.set(JAVA_DOUBLE, 48L, spec.streamingOverlapGamma());
        seg.set(JAVA_DOUBLE, 56L, spec.defaultPcieGbps());
        seg.set(JAVA_DOUBLE, 64L, spec.diskBandwidthGbps());
        seg.set(JAVA_LONG, 72L, spec.diskCapacityBytes());
        seg.set(JAVA_INT, 80L, spec.microBatches());
        seg.set(JAVA_INT, 84L, allowSpilling ? 1 : 0);

        HostCpuProfile cpu = spec.hostCpu();
        seg.set(JAVA_LONG, 88L, cpu.ramBytes());
        seg.set(JAVA_DOUBLE, 96L, cpu.computeGflops());
        seg.set(JAVA_DOUBLE, 104L, cpu.ramBandwidthGbps());
        seg.set(JAVA_DOUBLE, 112L, cpu.aggregatePcieGbps());
        return seg;
    }

    private static MemorySegment marshalSplitKvSpec(Arena arena, SplitKvSpec spec) {
        MemorySegment seg = arena.allocate(SPLIT_KV_SPEC_LAYOUT);
        seg.set(JAVA_LONG, 0L, spec.totalKvTokens());
        seg.set(JAVA_LONG, 8L, spec.kvBytesPerToken());
        seg.set(JAVA_DOUBLE, 16L, spec.attentionFlopsPerToken());
        seg.set(JAVA_LONG, 24L, spec.queryAndPartialBytes());
        seg.set(JAVA_LONG, 32L, spec.reservedWeightBytesPerGpu());
        seg.set(JAVA_DOUBLE, 40L, spec.vramUtilization());
        return seg;
    }

    private static WorkloadPlan readWorkloadPlan(
            MemorySegment headerSeg, long headerOffset,
            MemorySegment allocSeg, int allocCap,
            MemorySegment activeRanksSeg, int activeCap) {
        long totalUnits = headerSeg.get(JAVA_LONG, headerOffset);
        double makespanMs = headerSeg.get(JAVA_DOUBLE, headerOffset + 8L);
        double computeMs = headerSeg.get(JAVA_DOUBLE, headerOffset + 16L);
        double syncMs = headerSeg.get(JAVA_DOUBLE, headerOffset + 24L);
        int allocCount = Math.min(allocCap, headerSeg.get(JAVA_INT, headerOffset + 32L));
        int activeCount = Math.min(activeCap, headerSeg.get(JAVA_INT, headerOffset + 36L));

        List<ShardAllocation> allocations = new ArrayList<>(allocCount);
        long allocStride = SHARD_ALLOCATION_LAYOUT.byteSize();
        for (int i = 0; i < allocCount; i++) {
            long off = i * allocStride;
            int rank = allocSeg.get(JAVA_INT, off);
            boolean excluded = allocSeg.get(JAVA_INT, off + 4L) != 0;
            long units = allocSeg.get(JAVA_LONG, off + 8L);
            double shareRatio = allocSeg.get(JAVA_DOUBLE, off + 16L);
            double estComputeMs = allocSeg.get(JAVA_DOUBLE, off + 24L);
            long estMemBytes = allocSeg.get(JAVA_LONG, off + 32L);
            long vramUnits = allocSeg.get(JAVA_LONG, off + 40L);
            long hostStreamUnits = allocSeg.get(JAVA_LONG, off + 48L);
            long diskStreamUnits = allocSeg.get(JAVA_LONG, off + 56L);
            long cpuInPlaceUnits = allocSeg.get(JAVA_LONG, off + 64L);

            allocations.add(new ShardAllocation(
                    rank, units, shareRatio, estComputeMs, estMemBytes, excluded,
                    vramUnits, hostStreamUnits, diskStreamUnits, cpuInPlaceUnits));
        }

        List<Integer> activeRanks = new ArrayList<>(activeCount);
        for (int i = 0; i < activeCount; i++) {
            activeRanks.add(activeRanksSeg.getAtIndex(JAVA_INT, i));
        }

        return new WorkloadPlan(totalUnits, allocations, activeRanks, makespanMs, computeMs, syncMs);
    }

    private static int deviceTypeCode(String type) {
        if (type == null) {
            return 0;
        }
        return switch (type) {
            case "Integrated" -> 1;
            case "Discrete" -> 2;
            case "Virtual" -> 3;
            case "CPU" -> 4;
            default -> 0;
        };
    }

    private static DeviceRole deviceRoleFromCode(int code) {
        return switch (code) {
            case 0 -> DeviceRole.TENSOR_PARALLEL;
            case 1 -> DeviceRole.SPLIT_KV_RECLAIMED;
            case 2 -> DeviceRole.CPU_IN_PLACE;
            default -> DeviceRole.IDLE;
        };
    }

    private static String regimeName(int code) {
        return switch (code) {
            case 0 -> "ALL_GPU_SHARD";
            case 1 -> "STRAGGLER_PRUNED";
            case 2 -> "PCIE_STREAMING";
            case 3 -> "CPU_IN_PLACE";
            case 4 -> "SPLIT_KV_RECLAIMED";
            default -> "UNKNOWN";
        };
    }
}
