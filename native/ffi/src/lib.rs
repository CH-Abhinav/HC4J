//! `hc4x-ffi` — Thin panic-guarded C-ABI shared library (`hc4j.dll` / `libhc4j.so`
//! / `libhc4j.dylib`) over [`hc4rs`], consumed by Java 25 Panama FFM, Python (`hc4py`),
//! and C/C++ callers.
//!
//! Every `#[unsafe(no_mangle)] pub [unsafe] extern "C" fn` export lives in this crate
//! and executes inside `hc4rs::engine::error::ffi_guard` (or `ffi_guard_preserve_error`).

pub mod collective_abi;
pub mod engine_abi;
pub mod linalg_abi;
pub mod math_abi;
pub mod memory_abi;
pub mod nn_abi;
pub mod optimizer_abi;

pub use collective_abi::*;
pub use engine_abi::*;
pub use linalg_abi::*;
pub use math_abi::*;
pub use memory_abi::*;
pub use nn_abi::*;
pub use optimizer_abi::*;

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of, size_of};

    use hc4rs::collective::planner::{
        DEFAULT_VRAM_UTILIZATION, HostCpuProfileC, REGIME_SPLIT_KV_RECLAIMED, ROLE_SPLIT_KV_RECLAIMED,
        ROLE_TENSOR_PARALLEL, RolePlanHeaderC, ShardAllocationC, SplitKvAssignmentC, SplitKvSpecC,
        WorkloadPlanHeaderC, WorkloadSpecC,
    };
    use hc4rs::engine::error::{HC4J_ERR_INVALID_PARAM, HC4J_SUCCESS};
    use hc4rs::{AdapterInfoC, DeviceProfileC, EngineStats, LinkInfo, MemStats, TRANSPORT_HOST_STAGED};

    use crate::collective_abi::{IslandHeaderC, hc4j_collective_merge_split_kv_f32};
    use crate::engine_abi::hc4j_last_error;
    use crate::optimizer_abi::{hc4j_optimizer_plan_roles, hc4j_optimizer_plan_work};

    #[test]
    fn repr_c_struct_layouts_match_hc4x_header_exactly() {
        // MemStats: 21 u64 fields = 168 B
        assert_eq!(size_of::<MemStats>(), 168);
        assert_eq!(align_of::<MemStats>(), 8);
        assert_eq!(offset_of!(MemStats, vram_budget), 0);
        assert_eq!(offset_of!(MemStats, vram_used), 8);
        assert_eq!(offset_of!(MemStats, host_budget), 16);
        assert_eq!(offset_of!(MemStats, host_used), 24);
        assert_eq!(offset_of!(MemStats, spilled_bytes), 32);
        assert_eq!(offset_of!(MemStats, allocations), 104);
        assert_eq!(offset_of!(MemStats, quarantined), 160);

        // EngineStats: 13 u64 fields = 104 B
        assert_eq!(size_of::<EngineStats>(), 104);
        assert_eq!(align_of::<EngineStats>(), 8);
        assert_eq!(offset_of!(EngineStats, submissions), 0);
        assert_eq!(offset_of!(EngineStats, features), 64);
        assert_eq!(offset_of!(EngineStats, subgroup_max_size), 96);

        // AdapterInfoC: 7 u32 + 256 + 128 + 256 = 668 B
        assert_eq!(size_of::<AdapterInfoC>(), 668);
        assert_eq!(align_of::<AdapterInfoC>(), 4);
        assert_eq!(offset_of!(AdapterInfoC, vendor_id), 0);
        assert_eq!(offset_of!(AdapterInfoC, driver_info_len), 24);
        assert_eq!(offset_of!(AdapterInfoC, name), 28);
        assert_eq!(offset_of!(AdapterInfoC, driver), 284);
        assert_eq!(offset_of!(AdapterInfoC, driver_info), 412);

        // LinkInfo: 32 B
        assert_eq!(size_of::<LinkInfo>(), 32);
        assert_eq!(align_of::<LinkInfo>(), 8);
        assert_eq!(offset_of!(LinkInfo, src_rank), 0);
        assert_eq!(offset_of!(LinkInfo, dst_rank), 4);
        assert_eq!(offset_of!(LinkInfo, bandwidth_gbps), 8);
        assert_eq!(offset_of!(LinkInfo, transport), 16);
        assert_eq!(offset_of!(LinkInfo, loopback), 20);
        assert_eq!(offset_of!(LinkInfo, latency_us), 24);

        // DeviceProfileC: 40 B
        assert_eq!(size_of::<DeviceProfileC>(), 40);
        assert_eq!(align_of::<DeviceProfileC>(), 8);
        assert_eq!(offset_of!(DeviceProfileC, rank), 0);
        assert_eq!(offset_of!(DeviceProfileC, device_type), 12);
        assert_eq!(offset_of!(DeviceProfileC, vram_bytes), 16);
        assert_eq!(offset_of!(DeviceProfileC, compute_gflops), 24);
        assert_eq!(offset_of!(DeviceProfileC, vram_bandwidth_gbps), 32);

        // IslandHeaderC: 16 B
        assert_eq!(size_of::<IslandHeaderC>(), 16);
        assert_eq!(align_of::<IslandHeaderC>(), 4);
        assert_eq!(offset_of!(IslandHeaderC, member_count), 12);

        // HostCpuProfileC: 32 B
        assert_eq!(size_of::<HostCpuProfileC>(), 32);
        assert_eq!(align_of::<HostCpuProfileC>(), 8);
        assert_eq!(offset_of!(HostCpuProfileC, ram_bytes), 0);
        assert_eq!(offset_of!(HostCpuProfileC, compute_gflops), 8);
        assert_eq!(offset_of!(HostCpuProfileC, ram_bandwidth_gbps), 16);
        assert_eq!(offset_of!(HostCpuProfileC, aggregate_pcie_gbps), 24);

        // WorkloadSpecC: 120 B
        assert_eq!(size_of::<WorkloadSpecC>(), 120);
        assert_eq!(align_of::<WorkloadSpecC>(), 8);
        assert_eq!(offset_of!(WorkloadSpecC, total_units), 0);
        assert_eq!(offset_of!(WorkloadSpecC, flops_per_unit), 8);
        assert_eq!(offset_of!(WorkloadSpecC, bytes_per_unit), 16);
        assert_eq!(offset_of!(WorkloadSpecC, sync_bytes), 24);
        assert_eq!(offset_of!(WorkloadSpecC, io_multiplier), 32);
        assert_eq!(offset_of!(WorkloadSpecC, vram_utilization), 40);
        assert_eq!(offset_of!(WorkloadSpecC, streaming_overlap_gamma), 48);
        assert_eq!(offset_of!(WorkloadSpecC, default_pcie_gbps), 56);
        assert_eq!(offset_of!(WorkloadSpecC, disk_bandwidth_gbps), 64);
        assert_eq!(offset_of!(WorkloadSpecC, disk_capacity_bytes), 72);
        assert_eq!(offset_of!(WorkloadSpecC, micro_batches), 80);
        assert_eq!(offset_of!(WorkloadSpecC, allow_spilling), 84);
        assert_eq!(offset_of!(WorkloadSpecC, host_cpu), 88);

        // ShardAllocationC: 72 B
        assert_eq!(size_of::<ShardAllocationC>(), 72);
        assert_eq!(align_of::<ShardAllocationC>(), 8);
        assert_eq!(offset_of!(ShardAllocationC, rank), 0);
        assert_eq!(offset_of!(ShardAllocationC, excluded_as_straggler), 4);
        assert_eq!(offset_of!(ShardAllocationC, units), 8);
        assert_eq!(offset_of!(ShardAllocationC, share_ratio), 16);
        assert_eq!(offset_of!(ShardAllocationC, estimated_compute_ms), 24);
        assert_eq!(offset_of!(ShardAllocationC, estimated_memory_bytes), 32);
        assert_eq!(offset_of!(ShardAllocationC, vram_units), 40);
        assert_eq!(offset_of!(ShardAllocationC, host_stream_units), 48);
        assert_eq!(offset_of!(ShardAllocationC, disk_stream_units), 56);
        assert_eq!(offset_of!(ShardAllocationC, cpu_in_place_units), 64);

        // WorkloadPlanHeaderC: 40 B
        assert_eq!(size_of::<WorkloadPlanHeaderC>(), 40);
        assert_eq!(align_of::<WorkloadPlanHeaderC>(), 8);
        assert_eq!(offset_of!(WorkloadPlanHeaderC, total_units), 0);
        assert_eq!(offset_of!(WorkloadPlanHeaderC, estimated_makespan_ms), 8);
        assert_eq!(offset_of!(WorkloadPlanHeaderC, estimated_compute_ms), 16);
        assert_eq!(offset_of!(WorkloadPlanHeaderC, estimated_sync_ms), 24);
        assert_eq!(offset_of!(WorkloadPlanHeaderC, allocation_count), 32);
        assert_eq!(offset_of!(WorkloadPlanHeaderC, active_rank_count), 36);

        // SplitKvSpecC: 48 B
        assert_eq!(size_of::<SplitKvSpecC>(), 48);
        assert_eq!(align_of::<SplitKvSpecC>(), 8);
        assert_eq!(offset_of!(SplitKvSpecC, total_kv_tokens), 0);
        assert_eq!(offset_of!(SplitKvSpecC, kv_bytes_per_token), 8);
        assert_eq!(offset_of!(SplitKvSpecC, attention_flops_per_token), 16);
        assert_eq!(offset_of!(SplitKvSpecC, query_and_partial_bytes), 24);
        assert_eq!(offset_of!(SplitKvSpecC, reserved_weight_bytes_per_gpu), 32);
        assert_eq!(offset_of!(SplitKvSpecC, vram_utilization), 40);

        // SplitKvAssignmentC: 48 B
        assert_eq!(size_of::<SplitKvAssignmentC>(), 48);
        assert_eq!(align_of::<SplitKvAssignmentC>(), 8);
        assert_eq!(offset_of!(SplitKvAssignmentC, rank), 0);
        assert_eq!(offset_of!(SplitKvAssignmentC, role), 4);
        assert_eq!(offset_of!(SplitKvAssignmentC, kv_tokens), 8);
        assert_eq!(offset_of!(SplitKvAssignmentC, estimated_local_attn_ms), 16);
        assert_eq!(offset_of!(SplitKvAssignmentC, estimated_comm_ms), 24);
        assert_eq!(offset_of!(SplitKvAssignmentC, estimated_total_ms), 32);
        assert_eq!(offset_of!(SplitKvAssignmentC, kv_memory_bytes), 40);

        // RolePlanHeaderC: 80 B
        assert_eq!(size_of::<RolePlanHeaderC>(), 80);
        assert_eq!(align_of::<RolePlanHeaderC>(), 8);
        assert_eq!(offset_of!(RolePlanHeaderC, tp_header), 0);
        assert_eq!(offset_of!(RolePlanHeaderC, estimated_tp_ms), 40);
        assert_eq!(offset_of!(RolePlanHeaderC, estimated_attention_ms), 48);
        assert_eq!(offset_of!(RolePlanHeaderC, total_kv_tokens), 56);
        assert_eq!(offset_of!(RolePlanHeaderC, split_kv_count), 64);
        assert_eq!(offset_of!(RolePlanHeaderC, reclaimed_gpu_count), 68);
        assert_eq!(offset_of!(RolePlanHeaderC, cpu_reclaimed), 72);
        assert_eq!(offset_of!(RolePlanHeaderC, regime_code), 76);
    }

    #[test]
    fn c_abi_optimizer_and_last_error_round_trip() {
        let devices = [
            DeviceProfileC {
                rank: 0,
                vendor_id: 0x10DE,
                device_id: 0,
                device_type: 2,
                vram_bytes: 8 << 30,
                compute_gflops: 40_000.0,
                vram_bandwidth_gbps: 900.0,
            },
            DeviceProfileC {
                rank: 1,
                vendor_id: 0x8086,
                device_id: 1,
                device_type: 1,
                vram_bytes: 8 << 30,
                compute_gflops: 120.0,
                vram_bandwidth_gbps: 40.0,
            },
        ];
        let links = [
            LinkInfo {
                src_rank: 0,
                dst_rank: 1,
                bandwidth_gbps: 8.0,
                transport: TRANSPORT_HOST_STAGED,
                loopback: 0,
                latency_us: 20.0,
            },
            LinkInfo {
                src_rank: 1,
                dst_rank: 0,
                bandwidth_gbps: 8.0,
                transport: TRANSPORT_HOST_STAGED,
                loopback: 0,
                latency_us: 20.0,
            },
        ];
        let tp_spec = WorkloadSpecC {
            total_units: 2048,
            flops_per_unit: 2e9,
            bytes_per_unit: 4096,
            sync_bytes: 128 << 20,
            io_multiplier: 2.0,
            vram_utilization: DEFAULT_VRAM_UTILIZATION,
            streaming_overlap_gamma: 0.5,
            default_pcie_gbps: 16.0,
            disk_bandwidth_gbps: 0.0,
            disk_capacity_bytes: 0,
            micro_batches: 1,
            allow_spilling: 1,
            host_cpu: HostCpuProfileC::default(),
        };

        // 1. Null-pointer call records last_error cleanly without panicking.
        let bad_rc = unsafe {
            hc4j_optimizer_plan_work(
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                &tp_spec,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(bad_rc, HC4J_ERR_INVALID_PARAM);
        let needed_len = unsafe { hc4j_last_error(std::ptr::null_mut(), 0) };
        assert!(needed_len > 0);
        let mut msg_buf = vec![0u8; needed_len as usize + 1];
        let got_len = unsafe { hc4j_last_error(msg_buf.as_mut_ptr(), msg_buf.len() as u32) };
        assert_eq!(got_len, needed_len);
        assert_eq!(msg_buf[got_len as usize], 0);

        // 2. plan_work C-ABI call
        let mut work_header = WorkloadPlanHeaderC::default();
        let mut allocs = [ShardAllocationC::default(); 2];
        let mut active = [0i32; 2];
        let rc = unsafe {
            hc4j_optimizer_plan_work(
                devices.len() as u32,
                devices.as_ptr(),
                links.len() as u32,
                links.as_ptr(),
                &tp_spec,
                &mut work_header,
                allocs.as_mut_ptr(),
                allocs.len() as u32,
                active.as_mut_ptr(),
                active.len() as u32,
            )
        };
        assert_eq!(rc, HC4J_SUCCESS);
        assert_eq!(work_header.total_units, 2048);
        assert_eq!(work_header.active_rank_count, 1);
        assert_eq!(active[0], 0);
        assert_eq!(allocs[1].excluded_as_straggler, 1);

        // Calling hc4j_last_error after success returns 0!
        assert_eq!(unsafe { hc4j_last_error(std::ptr::null_mut(), 0) }, 0);

        // 3. plan_roles C-ABI call
        let kv_spec = SplitKvSpecC {
            total_kv_tokens: 1200,
            kv_bytes_per_token: 1 << 20,
            attention_flops_per_token: 1e6,
            query_and_partial_bytes: 32768,
            reserved_weight_bytes_per_gpu: 6 << 30,
            vram_utilization: 0.85,
        };
        let mut role_header = RolePlanHeaderC::default();
        let mut kv_assigns = [SplitKvAssignmentC::default(); 2];
        let rc_roles = unsafe {
            hc4j_optimizer_plan_roles(
                devices.len() as u32,
                devices.as_ptr(),
                links.len() as u32,
                links.as_ptr(),
                &tp_spec,
                &kv_spec,
                &mut role_header,
                allocs.as_mut_ptr(),
                allocs.len() as u32,
                active.as_mut_ptr(),
                active.len() as u32,
                kv_assigns.as_mut_ptr(),
                kv_assigns.len() as u32,
            )
        };
        assert_eq!(rc_roles, HC4J_SUCCESS);
        assert_eq!(role_header.regime_code, REGIME_SPLIT_KV_RECLAIMED);
        assert_eq!(role_header.reclaimed_gpu_count, 1);
        assert_eq!(kv_assigns[0].role, ROLE_TENSOR_PARALLEL);
        assert_eq!(kv_assigns[1].role, ROLE_SPLIT_KV_RECLAIMED);
        assert_eq!(kv_assigns[0].kv_tokens + kv_assigns[1].kv_tokens, 1200);

        // 4. Split-KV merge C-ABI call
        let o0 = [1.0f32, 2.0];
        let s0 = [10.0f32, 2.0];
        let o1 = [3.0f32, 6.0];
        let s1 = [10.0f32, 2.0];
        let ptr_o = [o0.as_ptr(), o1.as_ptr()];
        let ptr_s = [s0.as_ptr(), s1.as_ptr()];
        let mut merged = [0.0f32; 2];
        let rc_merge = unsafe {
            hc4j_collective_merge_split_kv_f32(1, 2, 2, ptr_o.as_ptr(), ptr_s.as_ptr(), merged.as_mut_ptr())
        };
        assert_eq!(rc_merge, HC4J_SUCCESS);
        assert!((merged[0] - 2.0).abs() < 1e-6);
        assert!((merged[1] - 4.0).abs() < 1e-6);
    }
}
