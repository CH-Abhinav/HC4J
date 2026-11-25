//! C-ABI entry points exposing the lifted Exact Zero-Parameter Optimizer
//! (Theorems 1–4 + Zero-Waste Split-KV Reclamation `plan_roles`) (2 symbols).

use hc4rs::collective::planner::{
    RolePlanHeaderC, ShardAllocationC, SplitKvAssignmentC, SplitKvSpecC, WorkloadPlan, WorkloadPlanHeaderC,
    WorkloadSpecC, plan_roles, plan_work,
};
use hc4rs::engine::error::{Hc4jError, Hc4jResult, ffi_guard};
use hc4rs::{DeviceProfileC, LinkInfo};

unsafe fn read_devices_and_links(
    device_count: u32,
    ptr_devices: *const DeviceProfileC,
    link_count: u32,
    ptr_links: *const LinkInfo,
) -> Hc4jResult<(Vec<DeviceProfileC>, Vec<LinkInfo>)> {
    if device_count == 0 || ptr_devices.is_null() {
        return Err(Hc4jError::InvalidParam("devices pointer null or device_count == 0"));
    }
    if link_count > 0 && ptr_links.is_null() {
        return Err(Hc4jError::InvalidParam("null links pointer with positive link_count"));
    }
    // Read via copy/unaligned in case FFI caller buffers are 4-byte aligned.
    let mut devices = Vec::with_capacity(device_count as usize);
    for i in 0..device_count as usize {
        devices.push(unsafe { ptr_devices.add(i).read_unaligned() });
    }
    let mut links = Vec::with_capacity(link_count as usize);
    for i in 0..link_count as usize {
        links.push(unsafe { ptr_links.add(i).read_unaligned() });
    }
    Ok((devices, links))
}

unsafe fn write_workload_plan_arrays(
    plan: &WorkloadPlan,
    out_allocations: *mut ShardAllocationC,
    allocations_cap: u32,
    out_active_ranks: *mut i32,
    active_ranks_cap: u32,
) -> Hc4jResult<WorkloadPlanHeaderC> {
    if out_allocations.is_null() && allocations_cap > 0 {
        return Err(Hc4jError::InvalidParam("null out_allocations with non-zero cap"));
    }
    if out_active_ranks.is_null() && active_ranks_cap > 0 {
        return Err(Hc4jError::InvalidParam("null out_active_ranks with non-zero cap"));
    }

    let n_alloc = plan.allocations.len().min(allocations_cap as usize);
    for (i, a) in plan.allocations.iter().take(n_alloc).enumerate() {
        unsafe { out_allocations.add(i).write_unaligned(ShardAllocationC::from(a)) };
    }

    let n_active = plan.active_ranks.len().min(active_ranks_cap as usize);
    for (i, &r) in plan.active_ranks.iter().take(n_active).enumerate() {
        unsafe { out_active_ranks.add(i).write_unaligned(r) };
    }

    Ok(WorkloadPlanHeaderC {
        total_units: plan.total_units,
        estimated_makespan_ms: plan.estimated_makespan_ms,
        estimated_compute_ms: plan.estimated_compute_ms,
        estimated_sync_ms: plan.estimated_sync_ms,
        allocation_count: plan.allocations.len() as u32,
        active_rank_count: plan.active_ranks.len() as u32,
    })
}

/// Runs Exact Theorems 1–4 workload optimization across participating GPU
/// devices, network links, 3-tier memory spilling, and optional Host CPU node.
///
/// # Safety
/// All non-null input and output pointers must remain valid for the duration
/// of the call with capacities matching their respective `*_count`/`*_cap` arguments.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_optimizer_plan_work(
    device_count: u32,
    ptr_devices: *const DeviceProfileC,
    link_count: u32,
    ptr_links: *const LinkInfo,
    ptr_spec: *const WorkloadSpecC,
    out_header: *mut WorkloadPlanHeaderC,
    out_allocations: *mut ShardAllocationC,
    allocations_cap: u32,
    out_active_ranks: *mut i32,
    active_ranks_cap: u32,
) -> i32 {
    ffi_guard("hc4j_optimizer_plan_work", || {
        if ptr_spec.is_null() || out_header.is_null() {
            return Err(Hc4jError::InvalidParam("null spec or out_header pointer"));
        }
        let (devices, links) = unsafe { read_devices_and_links(device_count, ptr_devices, link_count, ptr_links)? };
        let spec_c = unsafe { ptr_spec.read_unaligned() };
        let plan = plan_work(&devices, &links, &spec_c.into())?;
        let header = unsafe {
            write_workload_plan_arrays(
                &plan,
                out_allocations,
                allocations_cap,
                out_active_ranks,
                active_ranks_cap,
            )?
        };
        unsafe { out_header.write_unaligned(header) };
        Ok(())
    })
}

/// Runs joint Tensor Parallel + Zero-Waste Split-KV Reclamation role planning.
///
/// # Safety
/// All non-null input and output pointers must remain valid for the duration
/// of the call with capacities matching their respective `*_count`/`*_cap` arguments.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn hc4j_optimizer_plan_roles(
    device_count: u32,
    ptr_devices: *const DeviceProfileC,
    link_count: u32,
    ptr_links: *const LinkInfo,
    ptr_tp_spec: *const WorkloadSpecC,
    ptr_kv_spec: *const SplitKvSpecC,
    out_header: *mut RolePlanHeaderC,
    out_tp_allocations: *mut ShardAllocationC,
    tp_allocations_cap: u32,
    out_tp_active_ranks: *mut i32,
    tp_active_ranks_cap: u32,
    out_split_kv_assignments: *mut SplitKvAssignmentC,
    split_kv_cap: u32,
) -> i32 {
    ffi_guard("hc4j_optimizer_plan_roles", || {
        if ptr_tp_spec.is_null() || ptr_kv_spec.is_null() || out_header.is_null() {
            return Err(Hc4jError::InvalidParam("null tp_spec, kv_spec, or out_header pointer"));
        }
        if out_split_kv_assignments.is_null() && split_kv_cap > 0 {
            return Err(Hc4jError::InvalidParam("null out_split_kv_assignments with non-zero cap"));
        }

        let (devices, links) = unsafe { read_devices_and_links(device_count, ptr_devices, link_count, ptr_links)? };
        let tp_spec_c = unsafe { ptr_tp_spec.read_unaligned() };
        let kv_spec_c = unsafe { ptr_kv_spec.read_unaligned() };
        let roles = plan_roles(&devices, &links, &tp_spec_c.into(), &kv_spec_c.into())?;

        let tp_header = unsafe {
            write_workload_plan_arrays(
                &roles.tp_plan,
                out_tp_allocations,
                tp_allocations_cap,
                out_tp_active_ranks,
                tp_active_ranks_cap,
            )?
        };

        let n_kv = roles.split_kv_assignments.len().min(split_kv_cap as usize);
        for (i, a) in roles.split_kv_assignments.iter().take(n_kv).enumerate() {
            unsafe {
                out_split_kv_assignments
                    .add(i)
                    .write_unaligned(SplitKvAssignmentC::from(a))
            };
        }

        let header = RolePlanHeaderC {
            tp_header,
            estimated_tp_ms: roles.estimated_tp_ms,
            estimated_attention_ms: roles.estimated_attention_ms,
            total_kv_tokens: roles.total_kv_tokens,
            split_kv_count: roles.split_kv_assignments.len() as u32,
            reclaimed_gpu_count: roles.reclaimed_gpu_count,
            cpu_reclaimed: u32::from(roles.cpu_reclaimed),
            regime_code: roles.regime_code,
        };
        unsafe { out_header.write_unaligned(header) };
        Ok(())
    })
}
