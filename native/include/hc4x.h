/*
 * Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
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

/**
 * @file hc4x.h
 * @brief Complete C99 ABI contract for the hc4x shared native GPU engine
 *        (`hc4j.dll` / `libhc4j.so` / `libhc4j.dylib`), consumed by `hc4j`
 *        (Java 25 Panama FFM), `hc4py` (Python), and C/C++ bindings.
 *
 * Every symbol declared below is panic-guarded at the FFI boundary (`ffi_guard`)
 * and returns a status code (`>= 0` on success, `< 0` on error) or `0` handle
 * on allocation failure. Detailed thread-local diagnostics can be inspected
 * via `hc4j_last_error`.
 */

#ifndef HC4X_H
#define HC4X_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 1. Status Codes                                                           */
/* ========================================================================= */

#define HC4J_SUCCESS             0
#define HC4J_ERR_NOT_FOUND      (-1)
#define HC4J_ERR_INVALID_PARAM  (-2)
#define HC4J_ERR_GPU_READBACK   (-3)
#define HC4J_ERR_OUT_OF_MEMORY  (-4)
#define HC4J_ERR_DEVICE         (-5)
#define HC4J_ERR_GPU_VALIDATION (-6)
#define HC4J_ERR_HOST_IO        (-7)
#define HC4J_ERR_UNSUPPORTED    (-8)

/* ========================================================================= */
/* 2. Capabilities, Enums & Operation Flags                                  */
/* ========================================================================= */

/* EngineStats.features bitmask */
#define HC4J_FEATURE_SUBGROUPS   1ULL
#define HC4J_FEATURE_SHADER_F16  2ULL

/* Device / Adapter type codes */
#define HC4J_DEVICE_OTHER        0U
#define HC4J_DEVICE_INTEGRATED   1U
#define HC4J_DEVICE_DISCRETE     2U
#define HC4J_DEVICE_VIRTUAL      3U
#define HC4J_DEVICE_CPU          4U

/* Backend codes */
#define HC4J_BACKEND_NOOP        0U
#define HC4J_BACKEND_VULKAN      1U
#define HC4J_BACKEND_METAL       2U
#define HC4J_BACKEND_DX12        3U
#define HC4J_BACKEND_GL          4U
#define HC4J_BACKEND_WEBGPU      5U

/* Tiered residency codes returned by hc4j_mem_residency */
#define HC4J_TIER_VRAM           0
#define HC4J_TIER_HOST           1
#define HC4J_TIER_DISK           2

/* Element data types */
#define HC4J_DTYPE_I32           0U
#define HC4J_DTYPE_F32           1U
#define HC4J_DTYPE_F16           2U

/* Matrix multiplication transpose flags */
#define HC4J_FLAG_TRANS_A        1U
#define HC4J_FLAG_TRANS_B        2U

/* Unary i32 op codes (dispatch_unary_i32) */
#define HC4J_UNARY_I32_ABS       0U
#define HC4J_UNARY_I32_NEG       1U
#define HC4J_UNARY_I32_SIGN      2U
#define HC4J_UNARY_I32_CLAMP     3U

/* Elementwise comparison op codes (dispatch_compare) */
#define HC4J_CMP_EQ              0U
#define HC4J_CMP_NE              1U
#define HC4J_CMP_GT              2U
#define HC4J_CMP_GE              3U
#define HC4J_CMP_LT              4U
#define HC4J_CMP_LE              5U

/* Neural-network activation codes (dispatch_activation_f32) */
#define HC4J_ACT_RELU            0U
#define HC4J_ACT_LEAKY_RELU      1U
#define HC4J_ACT_ELU             2U
#define HC4J_ACT_GELU            3U
#define HC4J_ACT_SILU            4U
#define HC4J_ACT_SIGMOID         5U
#define HC4J_ACT_SOFTPLUS        6U

/* Reduction op codes (dispatch_reduce_*, dispatch_reduce_axis_f32) */
#define HC4J_REDUCE_SUM          0U
#define HC4J_REDUCE_MEAN         1U
#define HC4J_REDUCE_MAX          2U
#define HC4J_REDUCE_MIN          3U
#define HC4J_REDUCE_PROD         4U

/* Collective link transport tiers */
#define HC4J_TRANSPORT_HOST_STAGED     0U
#define HC4J_TRANSPORT_HOST_SHARED_DMA 1U
#define HC4J_TRANSPORT_PEER_TO_PEER    2U

/* Exact Optimizer constants */
#define HC4J_HOST_CPU_RANK             (-1)
#define HC4J_ROLE_TENSOR_PARALLEL      0U
#define HC4J_ROLE_SPLIT_KV_RECLAIMED   1U
#define HC4J_ROLE_CPU_IN_PLACE         2U
#define HC4J_ROLE_IDLE                 3U

#define HC4J_REGIME_ALL_GPU_SHARD      0U
#define HC4J_REGIME_STRAGGLER_PRUNED   1U
#define HC4J_REGIME_PCIE_STREAMING     2U
#define HC4J_REGIME_CPU_IN_PLACE       3U
#define HC4J_REGIME_SPLIT_KV_RECLAIMED 4U

/* ========================================================================= */
/* 3. C-Compatible Telemetry & Optimizer Struct Layouts (#[repr(C)])         */
/* ========================================================================= */

/** Tiered memory telemetry snapshot (168 bytes). */
typedef struct MemStats {
    uint64_t vram_budget;
    uint64_t vram_used;
    uint64_t host_budget;
    uint64_t host_used;
    uint64_t spilled_bytes;
    uint64_t tensors_device;
    uint64_t tensors_host;
    uint64_t tensors_evicted;
    uint64_t evictions;
    uint64_t page_ins;
    uint64_t spills;
    uint64_t driver_ooms;
    uint64_t streamed_ops;
    uint64_t allocations;
    uint64_t dedicated_buffers;
    uint64_t slabs_created;
    uint64_t region_allocs;
    uint64_t slab_count;
    uint64_t slab_bytes;
    uint64_t slab_regions;
    uint64_t quarantined;
} MemStats;

/** Command stream and adapter telemetry snapshot (104 bytes). */
typedef struct EngineStats {
    uint64_t submissions;
    uint64_t dispatches;
    uint64_t kernel_bytes;
    uint64_t host_to_device_bytes;
    uint64_t device_to_host_bytes;
    uint64_t batches;
    uint64_t completed_epoch;
    uint64_t submitted_epoch;
    uint64_t features;
    uint64_t device_type;
    uint64_t backend;
    uint64_t subgroup_min_size;
    uint64_t subgroup_max_size;
} EngineStats;

/** Active WebGPU adapter descriptor (668 bytes). */
typedef struct AdapterInfoC {
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t device_type;
    uint32_t backend;
    uint32_t name_len;
    uint32_t driver_len;
    uint32_t driver_info_len;
    uint8_t name[256];
    uint8_t driver[128];
    uint8_t driver_info[256];
} AdapterInfoC;

/** Measured directed collective interconnect link (32 bytes). */
typedef struct LinkInfo {
    uint32_t src_rank;
    uint32_t dst_rank;
    double bandwidth_gbps;
    uint32_t transport;
    uint32_t loopback;
    double latency_us;
} LinkInfo;

/** Hardware compute/VRAM capability profile for one rank (40 bytes). */
typedef struct DeviceProfileC {
    uint32_t rank;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t device_type;
    uint64_t vram_bytes;
    double compute_gflops;
    double vram_bandwidth_gbps;
} DeviceProfileC;

/** Header for one detected fast collective island (16 bytes). */
typedef struct IslandHeaderC {
    uint32_t id;
    uint32_t leader_rank;
    uint32_t internal_transport;
    uint32_t member_count;
} IslandHeaderC;

/** Host CPU node capabilities for Tier 3 in-place execution & Tier 1 spilling (32 bytes). */
typedef struct HostCpuProfileC {
    uint64_t ram_bytes;
    double compute_gflops;
    double ram_bandwidth_gbps;
    double aggregate_pcie_gbps;
} HostCpuProfileC;

/** Parallel workload specification for Theorems 1–4 optimization (120 bytes). */
typedef struct WorkloadSpecC {
    uint64_t total_units;
    double flops_per_unit;
    uint64_t bytes_per_unit;
    uint64_t sync_bytes;
    double io_multiplier;
    double vram_utilization;
    double streaming_overlap_gamma;
    double default_pcie_gbps;
    double disk_bandwidth_gbps;
    uint64_t disk_capacity_bytes;
    uint32_t micro_batches;
    uint32_t allow_spilling;
    HostCpuProfileC host_cpu;
} WorkloadSpecC;

/** Per-device work partition returned by hc4j_optimizer_plan_work (72 bytes). */
typedef struct ShardAllocationC {
    int32_t rank;
    uint32_t excluded_as_straggler;
    uint64_t units;
    double share_ratio;
    double estimated_compute_ms;
    uint64_t estimated_memory_bytes;
    uint64_t vram_units;
    uint64_t host_stream_units;
    uint64_t disk_stream_units;
    uint64_t cpu_in_place_units;
} ShardAllocationC;

/** Summary header returned by hc4j_optimizer_plan_work (40 bytes). */
typedef struct WorkloadPlanHeaderC {
    uint64_t total_units;
    double estimated_makespan_ms;
    double estimated_compute_ms;
    double estimated_sync_ms;
    uint32_t allocation_count;
    uint32_t active_rank_count;
} WorkloadPlanHeaderC;

/** KV-cache decode workload specification for hc4j_optimizer_plan_roles (48 bytes). */
typedef struct SplitKvSpecC {
    uint64_t total_kv_tokens;
    uint64_t kv_bytes_per_token;
    double attention_flops_per_token;
    uint64_t query_and_partial_bytes;
    uint64_t reserved_weight_bytes_per_gpu;
    double vram_utilization;
} SplitKvSpecC;

/** Per-device Split-KV worker role assignment (48 bytes). */
typedef struct SplitKvAssignmentC {
    int32_t rank;
    uint32_t role;
    uint64_t kv_tokens;
    double estimated_local_attn_ms;
    double estimated_comm_ms;
    double estimated_total_ms;
    uint64_t kv_memory_bytes;
} SplitKvAssignmentC;

/** Combined Tensor-Parallel + Split-KV RolePlan header (80 bytes). */
typedef struct RolePlanHeaderC {
    WorkloadPlanHeaderC tp_header;
    double estimated_tp_ms;
    double estimated_attention_ms;
    uint64_t total_kv_tokens;
    uint32_t split_kv_count;
    uint32_t reclaimed_gpu_count;
    uint32_t cpu_reclaimed;
    uint32_t regime_code;
} RolePlanHeaderC;

/* ========================================================================= */
/* 4. Engine & Error Diagnostics (7 symbols)                                 */
/* ========================================================================= */

int32_t hc4j_init_gpu(void);
int32_t hc4j_batch_begin(void);
int32_t hc4j_batch_end(void);
int32_t hc4j_synchronize(void);
int32_t hc4j_engine_stats(EngineStats* out);
int32_t hc4j_adapter_info(AdapterInfoC* out);
int32_t hc4j_last_error(uint8_t* buf, uint32_t cap);

/* ========================================================================= */
/* 5. Tiered Tensor Memory Management (12 symbols)                           */
/* ========================================================================= */

uint64_t hc4j_gpu_alloc(size_t length);
uint64_t hc4j_gpu_alloc_uninit(size_t length);
uint64_t hc4j_gpu_alloc_bytes(uint64_t bytes, uint32_t zeroed);
int32_t hc4j_gpu_write_bytes(uint64_t id, const uint8_t* ptr_in, size_t bytes);
int32_t hc4j_gpu_download_bytes(uint64_t id, uint8_t* ptr_out, size_t bytes);
int32_t hc4j_gpu_write(uint64_t id, const float* ptr_in, size_t length);
int32_t hc4j_gpu_free(uint64_t id);
int32_t hc4j_gpu_download(uint64_t id, float* ptr_out, size_t length);
int32_t hc4j_mem_configure(uint64_t vram_budget_bytes, uint64_t host_budget_bytes);
int32_t hc4j_mem_stats(MemStats* out);
int32_t hc4j_mem_residency(uint64_t id);
int32_t hc4j_mem_evict(uint64_t id);

/* ========================================================================= */
/* 6. Math, Elementwise, Compare, Selection & Type Casts (43 symbols)        */
/* ========================================================================= */

int32_t dispatch_add(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
                     const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                     const uint32_t* ptr_strides_b, const uint32_t* ptr_strides_c,
                     size_t length, uint32_t is_contiguous, uint32_t dtype_code);
int32_t dispatch_add_f32(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_b, const uint32_t* ptr_strides_c,
                         size_t length, uint32_t is_contiguous);
int32_t dispatch_sub(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
                     const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                     const uint32_t* ptr_strides_b, const uint32_t* ptr_strides_c,
                     size_t length, uint32_t is_contiguous, uint32_t dtype_code);
int32_t dispatch_sub_f32(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_b, const uint32_t* ptr_strides_c,
                         size_t length, uint32_t is_contiguous);
int32_t dispatch_mul(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
                     const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                     const uint32_t* ptr_strides_b, const uint32_t* ptr_strides_c,
                     size_t length, uint32_t is_contiguous, uint32_t dtype_code);
int32_t dispatch_mul_f32(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_b, const uint32_t* ptr_strides_c,
                         size_t length, uint32_t is_contiguous);
int32_t dispatch_div(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
                     const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                     const uint32_t* ptr_strides_b, const uint32_t* ptr_strides_c,
                     size_t length, uint32_t is_contiguous, uint32_t dtype_code);
int32_t dispatch_div_f32(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_b, const uint32_t* ptr_strides_c,
                         size_t length, uint32_t is_contiguous);

int32_t dispatch_sin_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_cos_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_tan_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_asin_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                          const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                          const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_acos_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                          const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                          const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_atan_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                          const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                          const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_sinh_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                          const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                          const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_cosh_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                          const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                          const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_tanh_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                          const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                          const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_asinh_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                           const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                           const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_acosh_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                           const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                           const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_atanh_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                           const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                           const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);

int32_t dispatch_exp_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_ln_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                        const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                        const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_log2_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                          const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                          const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_log10_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                           const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                           const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_sqrt_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                          const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                          const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);

int32_t dispatch_abs_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_neg_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_rsqrt_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                           const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                           const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_floor_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                           const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                           const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_ceil_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                          const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                          const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_round_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                           const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                           const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_sign_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                          const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                          const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous);
int32_t dispatch_clamp_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                           const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                           const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous,
                           float lo, float hi);
int32_t dispatch_unary_i32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                           const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                           const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous,
                           uint32_t op_code, uint32_t p_z, uint32_t p_w);

int32_t dispatch_pow_f32(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_b, const uint32_t* ptr_strides_c,
                         size_t length, uint32_t is_contiguous);
int32_t dispatch_maximum(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_b, const uint32_t* ptr_strides_c,
                         size_t length, uint32_t is_contiguous, uint32_t dtype_code);
int32_t dispatch_minimum(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_b, const uint32_t* ptr_strides_c,
                         size_t length, uint32_t is_contiguous, uint32_t dtype_code);
int32_t dispatch_compare(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t rank,
                         const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                         const uint32_t* ptr_strides_b, const uint32_t* ptr_strides_c,
                         size_t length, uint32_t is_contiguous, uint32_t dtype_code, uint32_t cmp_op);
int32_t dispatch_where(uint64_t id_cond, uint64_t id_x, uint64_t id_y, uint64_t id_out, uint32_t rank,
                       const uint32_t* ptr_shape, const uint32_t* ptr_strides_cond,
                       const uint32_t* ptr_strides_x, const uint32_t* ptr_strides_y,
                       const uint32_t* ptr_strides_out, size_t length, uint32_t is_contiguous,
                       uint32_t cond_dtype, uint32_t val_dtype);

int32_t dispatch_cast(uint64_t id_src, uint64_t id_out, uint64_t length, uint32_t src_dtype, uint32_t dst_dtype);
int32_t dispatch_cast_f32_to_f16(uint64_t id_src, uint64_t id_out, uint64_t length);
int32_t dispatch_cast_f16_to_f32(uint64_t id_src, uint64_t id_out, uint64_t length);
int32_t dispatch_copy_strided(uint64_t src, uint64_t out, uint64_t src_offset, uint32_t rank,
                              const uint32_t* shape, const uint32_t* src_strides, size_t len);

/* ========================================================================= */
/* 7. Linear Algebra (6 symbols)                                             */
/* ========================================================================= */

int32_t dispatch_matmul_f32(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t m, uint32_t n, uint32_t k);
int32_t dispatch_matmul_f32_ex(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t m, uint32_t n, uint32_t k,
                               uint32_t flags);
int32_t dispatch_batched_matmul_f32(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint32_t m, uint32_t n, uint32_t k,
                                    uint32_t batch_rank, const uint32_t* ptr_batch_shape,
                                    const uint32_t* ptr_a_batch_strides, const uint32_t* ptr_b_batch_strides,
                                    uint32_t flags);
int32_t dispatch_matmul_f32_f16(uint64_t id_a, uint64_t id_b_f16, uint64_t id_out, uint32_t m, uint32_t n, uint32_t k,
                                uint32_t flags);
int32_t dispatch_transpose_f32(uint64_t id_in, uint64_t id_out, uint32_t rows, uint32_t cols);
int32_t hc4j_matmul_plan(uint32_t m, uint32_t n, uint32_t k, uint32_t flags);

/* ========================================================================= */
/* 8. Neural Networks, Reductions & Fused Kernels (17 symbols)               */
/* ========================================================================= */

int32_t dispatch_activation_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
                                const uint32_t* ptr_shape, const uint32_t* ptr_strides_a,
                                const uint32_t* ptr_strides_c, size_t length, uint32_t is_contiguous,
                                uint32_t act, uint32_t alpha_bits);
int32_t dispatch_softmax_rows_f32(uint64_t id_a, uint64_t id_out, uint32_t rows, uint32_t cols, uint32_t log_variant);
int32_t dispatch_layernorm_rows_f32(uint64_t id_a, uint64_t id_gamma, uint64_t id_beta, uint64_t id_out,
                                    uint32_t rows, uint32_t cols, uint32_t eps_bits);
int32_t dispatch_rmsnorm_rows_f32(uint64_t id_a, uint64_t id_gamma, uint64_t id_out,
                                  uint32_t rows, uint32_t cols, uint32_t eps_bits);
int32_t dispatch_cross_entropy_rows(uint64_t id_logits, uint64_t id_targets, uint64_t id_out,
                                    uint32_t rows, uint32_t cols);
int32_t dispatch_add_row_bias_f32(uint64_t id_a, uint64_t id_bias, uint64_t id_out, uint32_t rows, uint32_t cols);
int32_t dispatch_kv_cache_append_f32(uint64_t id_src, uint64_t id_cache, uint32_t num_kv_heads, uint32_t max_seq_len,
                                     uint32_t head_dim, uint32_t seq_len, uint32_t pos_offset);
int32_t dispatch_embedding(uint64_t id_table, uint64_t id_indices, uint64_t id_out, uint32_t vocab_size,
                           uint32_t hidden_dim, uint32_t num_tokens, uint32_t table_is_f16);
int32_t dispatch_rope_f32(uint64_t id_in, uint64_t id_out, uint32_t batch, uint32_t seq_len, uint32_t num_heads,
                          uint32_t head_dim, uint32_t pos_offset, uint32_t theta_bits);

int32_t dispatch_reduce_f32(uint64_t id_a, uint64_t id_out, uint64_t length, uint32_t op);
int32_t dispatch_reduce_square_f32(uint64_t id_a, uint64_t id_out, uint64_t length, uint32_t op);
int32_t dispatch_reduce_diff2_f32(uint64_t id_a, uint64_t id_b, uint64_t id_out, uint64_t length, uint32_t op);
int32_t dispatch_reduce_rows_f32(uint64_t id_a, uint64_t id_out, uint32_t rows, uint32_t cols, uint32_t op);
int32_t dispatch_argmax_rows(uint64_t id_a, uint64_t id_out, uint32_t rows, uint32_t cols, uint32_t largest);
int32_t dispatch_reduce_axis_f32(uint64_t id_a, uint64_t id_out, uint32_t outer, uint32_t axis_size,
                                 uint32_t inner, uint32_t op);
int32_t dispatch_argmax_axis(uint64_t id_a, uint64_t id_out, uint32_t outer, uint32_t axis_size,
                             uint32_t inner, uint32_t largest);

int32_t hc4j_fused_dispatch(const uint32_t* program, uint32_t program_len, const uint64_t* inputs,
                            uint32_t n_inputs, uint64_t id_out);

/* ========================================================================= */
/* 9. Collective Runtime & Lossless Split-KV Merge (20 symbols)              */
/* ========================================================================= */

int32_t hc4j_collective_create(uint32_t ranks, uint64_t* out);
int32_t hc4j_collective_destroy(uint64_t handle);
int32_t hc4j_collective_ranks(uint64_t handle);
int32_t hc4j_collective_link_count(uint64_t handle);
int32_t hc4j_collective_link(uint64_t handle, uint32_t index, LinkInfo* out);
int32_t hc4j_collective_device_name(uint64_t handle, uint32_t rank, uint8_t* buf, uint32_t cap);
int32_t hc4j_collective_backend_name(uint64_t handle, uint8_t* buf, uint32_t cap);
int32_t hc4j_collective_alloc(uint64_t handle, uint32_t rank, uint64_t elements, uint64_t* out);
int32_t hc4j_collective_free(uint64_t handle, uint64_t buffer_id);
int32_t hc4j_collective_write(uint64_t handle, uint64_t buffer_id, const float* ptr, uint64_t elements);
int32_t hc4j_collective_read(uint64_t handle, uint64_t buffer_id, float* out, uint64_t elements);
int32_t hc4j_collective_allreduce_sum(uint64_t handle, const uint64_t* ptr_buffers, uint32_t count);
int32_t hc4j_collective_allreduce_sum_subset(uint64_t handle, const uint64_t* ptr_buffers, uint32_t count);
int32_t hc4j_collective_gemm_bias_act(uint64_t handle, uint32_t rank, uint64_t buf_a, uint64_t buf_b,
                                      uint64_t buf_bias, uint64_t buf_c, uint32_t m, uint32_t n, uint32_t k,
                                      uint32_t flags);
int32_t hc4j_collective_split_kv_attn(uint64_t handle, uint32_t rank, uint64_t buf_q, uint64_t buf_k, uint64_t buf_v,
                                      uint64_t buf_out_o, uint64_t buf_out_stats, uint32_t num_heads,
                                      uint32_t num_kv_heads, uint32_t head_dim, uint32_t local_tokens,
                                      uint32_t scale_bits);
int32_t hc4j_collective_device_profile(uint64_t handle, uint32_t rank, DeviceProfileC* out);
int32_t hc4j_collective_ring_order(uint64_t handle, uint32_t* out_ranks, uint32_t cap);
int32_t hc4j_collective_island_count(uint64_t handle);
int32_t hc4j_collective_island(uint64_t handle, uint32_t index, IslandHeaderC* out_header,
                               uint32_t* out_ranks, uint32_t cap);
int32_t hc4j_collective_merge_split_kv_f32(uint32_t num_heads, uint32_t head_dim, uint32_t worker_count,
                                           const float* const* ptr_partials_o,
                                           const float* const* ptr_partials_stats, float* ptr_out);
int32_t hc4j_collective_elem_mul(uint64_t handle, uint32_t rank, uint64_t buf_a, uint64_t buf_b,
                                 uint64_t buf_out, uint32_t count);

/* ========================================================================= */
/* 10. Multi-Threaded SIMD CPU-in-Place Kernels (4 symbols)                  */
/* ========================================================================= */

#define HC4J_CPU_ACT_NONE (-1)
#define HC4J_CPU_ACT_RELU 0
#define HC4J_CPU_ACT_GELU 3
#define HC4J_CPU_ACT_SILU 4

int32_t hc4j_cpu_gemm_f32(const float* ptr_a, const float* ptr_b, float* ptr_out,
                          uint32_t m, uint32_t n, uint32_t k, uint32_t flags);
int32_t hc4j_cpu_gemm_f32_f16(const float* ptr_a, const uint16_t* ptr_b_f16, float* ptr_out,
                              uint32_t m, uint32_t n, uint32_t k, uint32_t flags);
int32_t hc4j_cpu_mlp_slice_f32(const float* ptr_x, const float* ptr_w1, const float* ptr_b1,
                               const float* ptr_w2, float* ptr_out,
                               uint32_t m, uint32_t d_in, uint32_t h, uint32_t d_out,
                               uint32_t h_start, uint32_t units, int32_t act_code);
int32_t hc4j_cpu_split_kv_attn_f32(const float* ptr_q, const float* ptr_k_shard, const float* ptr_v_shard,
                                   float* ptr_out_o, float* ptr_out_stats,
                                   uint32_t tokens, uint32_t num_heads, uint32_t num_kv_heads,
                                   uint32_t head_dim, uint32_t scale_bits);

/* ========================================================================= */
/* 11. Exact Theorems 1–4 Workload & Split-KV Role Optimizer (2 symbols)     */
/* ========================================================================= */

int32_t hc4j_optimizer_plan_work(uint32_t device_count, const DeviceProfileC* ptr_devices,
                                 uint32_t link_count, const LinkInfo* ptr_links,
                                 const WorkloadSpecC* ptr_spec, WorkloadPlanHeaderC* out_header,
                                 ShardAllocationC* out_allocations, uint32_t allocations_cap,
                                 int32_t* out_active_ranks, uint32_t active_ranks_cap);

int32_t hc4j_optimizer_plan_roles(uint32_t device_count, const DeviceProfileC* ptr_devices,
                                  uint32_t link_count, const LinkInfo* ptr_links,
                                  const WorkloadSpecC* ptr_tp_spec, const SplitKvSpecC* ptr_kv_spec,
                                  RolePlanHeaderC* out_header, ShardAllocationC* out_tp_allocations,
                                  uint32_t tp_allocations_cap, int32_t* out_tp_active_ranks,
                                  uint32_t tp_active_ranks_cap, SplitKvAssignmentC* out_split_kv_assignments,
                                  uint32_t split_kv_cap);

#ifdef __cplusplus
}
#endif

#endif /* HC4X_H */
