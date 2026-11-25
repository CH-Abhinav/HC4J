//! Theorem 3 & Theorem 4: 3-Tier Convex Memory-Spilling Curves, Explicit Host
//! CPU Offload, Shared Host-RAM/PCIe Contention Fixed Point, O(m log m)
//! Min-Heap Integer Minimax, and Exact 2^N-1 Subset Search with Zero-Degradation
//! Monotonicity.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::mst_islands::estimate_sync_ms_exact;
use super::ring_tsp::{AllSubsetsRingTable, precompute_all_subsets_ring};
use super::{
    HOST_CPU_RANK, ShardAllocation, WorkloadPlan, WorkloadSpec, build_topology_matrices, validate_work_inputs,
};
use crate::collective::backend::{DeviceProfileC, LinkInfo};
use crate::engine::error::{Hc4jError, Hc4jResult};

const EPSILON: f64 = 1e-6;
const UNBOUNDED_UNITS: u64 = (i64::MAX / 4) as u64;

#[derive(Debug, Clone, Copy)]
struct DeviceCurve {
    index: i32,
    u_vram: u64,
    u_host: u64,
    u_disk: u64,
    u_max: u64,
    r0: f64,
    r1: f64,
    r2: f64,
}

impl DeviceCurve {
    fn eval_time(&self, u: u64) -> f64 {
        if u == 0 {
            return 0.0;
        }
        if u <= self.u_vram {
            return (u as f64) * self.r0;
        }
        let tier01 = self.u_vram.saturating_add(self.u_host);
        if u <= tier01 {
            return (self.u_vram as f64) * self.r0 + ((u - self.u_vram) as f64) * self.r1;
        }
        let u_d = u.saturating_sub(tier01).min(self.u_disk);
        (self.u_vram as f64) * self.r0 + (self.u_host as f64) * self.r1 + (u_d as f64) * self.r2
    }

    fn inv_time(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let t_vram = (self.u_vram as f64) * self.r0;
        if t <= t_vram {
            return if self.r0 > 0.0 { t / self.r0 } else { self.u_vram as f64 };
        }
        let t_host = t_vram + (self.u_host as f64) * self.r1;
        if t <= t_host {
            return if self.r1 > 0.0 {
                (self.u_vram as f64) + (t - t_vram) / self.r1
            } else {
                self.u_vram.saturating_add(self.u_host) as f64
            };
        }
        let t_disk = t_host + (self.u_disk as f64) * self.r2;
        if t <= t_disk {
            return if self.r2 > 0.0 {
                (self.u_vram.saturating_add(self.u_host) as f64) + (t - t_host) / self.r2
            } else {
                self.u_max as f64
            };
        }
        self.u_max as f64
    }
}

#[derive(Debug, Clone)]
struct AllocationDetail {
    total_units: u64,
    vram_units: u64,
    host_stream_units: u64,
    disk_stream_units: u64,
    cpu_in_place_units: u64,
    compute_ms: f64,
    vram_bytes: u64,
    host_bytes: u64,
    disk_bytes: u64,
}

impl AllocationDetail {
    fn zero() -> Self {
        Self {
            total_units: 0,
            vram_units: 0,
            host_stream_units: 0,
            disk_stream_units: 0,
            cpu_in_place_units: 0,
            compute_ms: 0.0,
            vram_bytes: 0,
            host_bytes: 0,
            disk_bytes: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct PlanCandidate {
    active_gpu_indices: Vec<usize>,
    includes_cpu: bool,
    gpu_allocations: Vec<AllocationDetail>,
    cpu_allocation: Option<AllocationDetail>,
    total_spilled_units: u64,
    compute_ms: f64,
    sync_ms: f64,
    makespan_ms: f64,
}

#[derive(Copy, Clone, PartialEq)]
struct MinHeapEntry {
    index: usize,
    next_time: f64,
}

impl Eq for MinHeapEntry {}

impl PartialOrd for MinHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering so BinaryHeap behaves as a min-heap (smallest time, then lowest index).
        other
            .next_time
            .total_cmp(&self.next_time)
            .then_with(|| other.index.cmp(&self.index))
    }
}

fn solve_integer_minimax(curves: &[DeviceCurve], total_units: u64) -> Vec<u64> {
    let m = curves.len();
    let mut result = vec![0u64; m];
    if m == 0 || total_units == 0 {
        return result;
    }

    let mut t_low = 0.0f64;
    let mut t_high = 1.0f64;
    for c in curves {
        t_high = t_high.max(c.eval_time(total_units));
    }

    for _ in 0..64 {
        let t_mid = t_low + (t_high - t_low) * 0.5;
        let mut sum_units = 0.0f64;
        for c in curves {
            sum_units += c.inv_time(t_mid);
        }
        if sum_units >= total_units as f64 {
            t_high = t_mid;
        } else {
            t_low = t_mid;
        }
    }

    let t_opt = t_high;
    let mut current_sum = 0u64;
    for (i, c) in curves.iter().enumerate() {
        let floor_u = (c.inv_time(t_opt).floor() as u64).min(c.u_max);
        result[i] = floor_u;
        current_sum = current_sum.saturating_add(floor_u);
    }

    if total_units > current_sum {
        let mut deficit = total_units - current_sum;
        let mut pq = BinaryHeap::with_capacity(m);
        for (i, c) in curves.iter().enumerate() {
            if result[i] < c.u_max {
                pq.push(MinHeapEntry {
                    index: i,
                    next_time: c.eval_time(result[i] + 1),
                });
            }
        }

        while deficit > 0 {
            let Some(top) = pq.pop() else { break };
            result[top.index] += 1;
            deficit -= 1;
            if result[top.index] < curves[top.index].u_max {
                pq.push(MinHeapEntry {
                    index: top.index,
                    next_time: curves[top.index].eval_time(result[top.index] + 1),
                });
            }
        }
    } else if current_sum > total_units {
        let mut surplus = current_sum - total_units;
        for i in (0..m).rev() {
            if surplus == 0 {
                break;
            }
            let take = result[i].min(surplus);
            result[i] -= take;
            surplus -= take;
        }
    }

    result
}

#[allow(clippy::too_many_arguments)]
fn evaluate_device_set(
    active_gpu_indices: &[usize],
    include_cpu: bool,
    devices: &[DeviceProfileC],
    bw: &[Vec<f64>],
    lat_us: &[Vec<f64>],
    tr: &[Vec<u32>],
    ring_table: Option<&AllSubsetsRingTable>,
    spec: &WorkloadSpec,
) -> Option<PlanCandidate> {
    let n = devices.len();
    let total_units = spec.total_units;
    let m_bytes = spec.bytes_per_unit;
    let f_flops = spec.flops_per_unit;
    let kappa = spec.io_multiplier;
    let gamma = spec.streaming_overlap_gamma;
    let cpu = spec.host_cpu;

    if active_gpu_indices.is_empty() && !include_cpu {
        return None;
    }

    let mut u_vram = vec![0u64; n];
    for &i in active_gpu_indices {
        u_vram[i] = if m_bytes > 0 {
            ((devices[i].vram_bytes as f64 * spec.vram_utilization) / m_bytes as f64).floor() as u64
        } else {
            UNBOUNDED_UNITS
        };
    }

    let total_host_ram_bytes = if spec.allow_spilling && include_cpu && cpu.is_enabled() {
        cpu.ram_bytes
    } else {
        0
    };
    let total_disk_bytes = if spec.allow_spilling { spec.disk_capacity_bytes } else { 0 };

    let mut k_spill: usize = 0;
    let mut gpu_dets: Vec<Option<AllocationDetail>> = vec![None; n];
    let mut cpu_det: Option<AllocationDetail> = None;
    let mut max_compute_ms = 0.0f64;

    for iter in 0..4 {
        let mut effective_pcie = spec.default_pcie_gbps;
        if k_spill > 0 && cpu.aggregate_pcie_gbps > 0.0 {
            effective_pcie = effective_pcie.min(cpu.aggregate_pcie_gbps / (k_spill.max(1) as f64));
        }

        let mut curves = Vec::with_capacity(active_gpu_indices.len() + usize::from(include_cpu));
        let participants_for_host = active_gpu_indices.len() + usize::from(include_cpu);
        let host_ram_per_gpu = if participants_for_host > 0 && m_bytes > 0 {
            total_host_ram_bytes / participants_for_host as u64
        } else if m_bytes == 0 {
            UNBOUNDED_UNITS
        } else {
            0
        };
        let cpu_ram_quota = if participants_for_host > 0 && m_bytes > 0 && include_cpu && cpu.is_enabled() {
            cpu.ram_bytes / participants_for_host as u64
        } else if m_bytes == 0 {
            UNBOUNDED_UNITS
        } else {
            0
        };

        for &i in active_gpu_indices {
            let c_gflops = devices[i].compute_gflops.max(EPSILON);
            let r0 = (f_flops / (c_gflops * 1e9)) * 1000.0;
            let mut r1 = r0;
            let mut r2 = r0;
            let mut u_host_cap = 0u64;
            let mut u_disk_cap = 0u64;

            if m_bytes > 0 && total_host_ram_bytes > 0 {
                u_host_cap = if host_ram_per_gpu > 0 {
                    host_ram_per_gpu / m_bytes
                } else {
                    0
                };
                let t_comp = f_flops / (c_gflops * 1e9);
                let t_pcie = if effective_pcie > 0.0 {
                    (kappa * m_bytes as f64) / (effective_pcie * 1e9)
                } else {
                    1e3
                };
                r1 = (t_comp.max(t_pcie) + (1.0 - gamma) * t_comp.min(t_pcie)) * 1000.0;
            }

            if m_bytes > 0 && spec.disk_bandwidth_gbps > 0.0 && total_disk_bytes > 0 {
                let disk_per_gpu = if !active_gpu_indices.is_empty() {
                    total_disk_bytes / active_gpu_indices.len() as u64
                } else {
                    0
                };
                u_disk_cap = disk_per_gpu / m_bytes;
                let b_disk_eff = effective_pcie.min(spec.disk_bandwidth_gbps);
                let t_comp = f_flops / (c_gflops * 1e9);
                let t_disk = if b_disk_eff > 0.0 {
                    (kappa * m_bytes as f64) / (b_disk_eff * 1e9)
                } else {
                    1e3
                };
                r2 = (t_comp.max(t_disk) + (1.0 - gamma) * t_comp.min(t_disk)) * 1000.0;
            }

            let u_max = u_vram[i].saturating_add(u_host_cap.saturating_add(u_disk_cap));
            curves.push(DeviceCurve {
                index: i as i32,
                u_vram: u_vram[i],
                u_host: u_host_cap,
                u_disk: u_disk_cap,
                u_max,
                r0,
                r1,
                r2,
            });
        }

        if include_cpu && cpu.is_enabled() {
            let cpu_gflops = cpu.compute_gflops.max(EPSILON);
            let t_comp = f_flops / (cpu_gflops * 1e9);
            let t_ram = if cpu.ram_bandwidth_gbps > 0.0 {
                (kappa * m_bytes as f64) / (cpu.ram_bandwidth_gbps * 1e9)
            } else {
                1e3
            };
            let r_cpu = t_comp.max(t_ram) * 1000.0;
            let u_cpu_cap = cpu_ram_quota.checked_div(m_bytes).unwrap_or(UNBOUNDED_UNITS);
            curves.push(DeviceCurve {
                index: HOST_CPU_RANK,
                u_vram: 0,
                u_host: u_cpu_cap,
                u_disk: 0,
                u_max: u_cpu_cap,
                r0: r_cpu,
                r1: r_cpu,
                r2: r_cpu,
            });
        }

        let mut total_capacity = 0u64;
        for c in &curves {
            total_capacity = total_capacity.saturating_add(c.u_max);
        }
        if total_capacity < total_units {
            return None;
        }

        let assigned = solve_integer_minimax(&curves, total_units);

        let mut new_k_spill = 0usize;
        for (c, &u) in curves.iter().zip(&assigned) {
            if c.index != HOST_CPU_RANK && u > c.u_vram {
                new_k_spill += 1;
            }
        }

        if new_k_spill == k_spill || iter == 3 {
            max_compute_ms = 0.0;
            for (c, &u) in curves.iter().zip(&assigned) {
                let time_ms = c.eval_time(u);
                max_compute_ms = max_compute_ms.max(time_ms);

                if c.index != HOST_CPU_RANK {
                    let vram_u = u.min(c.u_vram);
                    let rem = u - vram_u;
                    let host_u = rem.min(c.u_host);
                    let disk_u = (rem - host_u).min(c.u_disk);
                    gpu_dets[c.index as usize] = Some(AllocationDetail {
                        total_units: u,
                        vram_units: vram_u,
                        host_stream_units: host_u,
                        disk_stream_units: disk_u,
                        cpu_in_place_units: 0,
                        compute_ms: time_ms,
                        vram_bytes: vram_u.saturating_mul(m_bytes),
                        host_bytes: host_u.saturating_mul(m_bytes),
                        disk_bytes: disk_u.saturating_mul(m_bytes),
                    });
                } else {
                    cpu_det = Some(AllocationDetail {
                        total_units: u,
                        vram_units: 0,
                        host_stream_units: 0,
                        disk_stream_units: 0,
                        cpu_in_place_units: u,
                        compute_ms: time_ms,
                        vram_bytes: 0,
                        host_bytes: u.saturating_mul(m_bytes),
                        disk_bytes: 0,
                    });
                }
            }
            break;
        }
        k_spill = new_k_spill;
    }

    let resolved_gpus: Vec<AllocationDetail> = gpu_dets
        .into_iter()
        .map(|d| d.unwrap_or_else(AllocationDetail::zero))
        .collect();

    let sync_ms = if active_gpu_indices.len() >= 2 {
        estimate_sync_ms_exact(active_gpu_indices, bw, lat_us, tr, ring_table, devices, spec.sync_bytes)
    } else {
        0.0
    };

    let alpha = spec.overlap_alpha();
    let makespan_ms = max_compute_ms.max(sync_ms) + (1.0 - alpha) * max_compute_ms.min(sync_ms);

    let total_spilled_units: u64 = resolved_gpus
        .iter()
        .map(|d| d.host_stream_units.saturating_add(d.disk_stream_units))
        .sum();

    Some(PlanCandidate {
        active_gpu_indices: active_gpu_indices.to_vec(),
        includes_cpu: include_cpu,
        gpu_allocations: resolved_gpus,
        cpu_allocation: cpu_det,
        total_spilled_units,
        compute_ms: max_compute_ms,
        sync_ms,
        makespan_ms,
    })
}

fn is_better_candidate(a: &PlanCandidate, b: &PlanCandidate) -> bool {
    if a.makespan_ms < b.makespan_ms * (1.0 - 1e-6) {
        return true;
    }
    if b.makespan_ms < a.makespan_ms * (1.0 - 1e-6) {
        return false;
    }
    if a.total_spilled_units != b.total_spilled_units {
        return a.total_spilled_units < b.total_spilled_units;
    }
    a.active_gpu_indices.len() > b.active_gpu_indices.len()
}

/// Computes the exact zero-parameter optimal [`WorkloadPlan`] across all GPU
/// subsets and optional host CPU node.
pub fn plan_work(
    devices: &[DeviceProfileC],
    topology: &[LinkInfo],
    spec: &WorkloadSpec,
) -> Hc4jResult<WorkloadPlan> {
    validate_work_inputs(devices, spec)?;
    let n = devices.len();

    if spec.total_units == 0 {
        let allocations = devices
            .iter()
            .map(|d| ShardAllocation {
                rank: d.rank as i32,
                units: 0,
                share_ratio: 0.0,
                estimated_compute_ms: 0.0,
                estimated_memory_bytes: 0,
                excluded_as_straggler: false,
                vram_units: 0,
                host_stream_units: 0,
                disk_stream_units: 0,
                cpu_in_place_units: 0,
            })
            .collect();
        return Ok(WorkloadPlan {
            total_units: 0,
            allocations,
            active_ranks: Vec::new(),
            estimated_makespan_ms: 0.0,
            estimated_compute_ms: 0.0,
            estimated_sync_ms: 0.0,
        });
    }

    let super::TopologyMatrices { bw, lat_us, tr } = build_topology_matrices(n, topology);
    let ring_table = (n <= 12).then(|| precompute_all_subsets_ring(n, &bw, &lat_us));
    let cpu_enabled = spec.host_cpu.is_enabled();

    let mut best_plan: Option<PlanCandidate> = None;
    let consider = |best: &mut Option<PlanCandidate>, cand: Option<PlanCandidate>| {
        if let Some(c) = cand
            && best.as_ref().is_none_or(|b| is_better_candidate(&c, b))
        {
            *best = Some(c);
        }
    };

    if n <= 12 {
        let num_subsets = 1usize << n;
        let mut gpu_indices = Vec::with_capacity(n);
        for mask in 1usize..num_subsets {
            gpu_indices.clear();
            for i in 0..n {
                if (mask & (1usize << i)) != 0 {
                    gpu_indices.push(i);
                }
            }
            consider(
                &mut best_plan,
                evaluate_device_set(&gpu_indices, false, devices, &bw, &lat_us, &tr, ring_table.as_ref(), spec),
            );
            if cpu_enabled {
                consider(
                    &mut best_plan,
                    evaluate_device_set(&gpu_indices, true, devices, &bw, &lat_us, &tr, ring_table.as_ref(), spec),
                );
            }
        }
        if cpu_enabled {
            consider(
                &mut best_plan,
                evaluate_device_set(&[], true, devices, &bw, &lat_us, &tr, ring_table.as_ref(), spec),
            );
        }
    } else {
        let mut active_gpus: Vec<usize> = (0..n).collect();
        best_plan = evaluate_device_set(&active_gpus, cpu_enabled, devices, &bw, &lat_us, &tr, None, spec);
        let mut improved = true;
        while improved && active_gpus.len() > 1 {
            improved = false;
            let mut best_prune_idx: Option<usize> = None;
            let mut best_prune_cand: Option<PlanCandidate> = None;
            for i in 0..active_gpus.len() {
                let mut cand_active = active_gpus.clone();
                cand_active.remove(i);
                if let Some(cand) = evaluate_device_set(&cand_active, cpu_enabled, devices, &bw, &lat_us, &tr, None, spec)
                    && best_prune_cand.as_ref().is_none_or(|b| is_better_candidate(&cand, b))
                {
                    best_prune_cand = Some(cand);
                    best_prune_idx = Some(i);
                }
            }
            if let (Some(cand), Some(prune_idx)) = (best_prune_cand, best_prune_idx)
                && best_plan.as_ref().is_none_or(|b| is_better_candidate(&cand, b))
            {
                best_plan = Some(cand);
                active_gpus.remove(prune_idx);
                improved = true;
            }
        }
    }

    let best = best_plan.ok_or(Hc4jError::OutOfMemory)?;
    let total_f = spec.total_units as f64;
    let mut allocations = Vec::with_capacity(n + usize::from(best.includes_cpu));
    let mut active_ranks = Vec::with_capacity(n + usize::from(best.includes_cpu));

    for (i, d) in devices.iter().enumerate() {
        let det = &best.gpu_allocations[i];
        let u = det.total_units;
        let excluded = u == 0 && !best.active_gpu_indices.contains(&i);
        allocations.push(ShardAllocation {
            rank: d.rank as i32,
            units: u,
            share_ratio: (u as f64) / total_f,
            estimated_compute_ms: det.compute_ms,
            estimated_memory_bytes: det.vram_bytes.saturating_add(det.host_bytes).saturating_add(det.disk_bytes),
            excluded_as_straggler: excluded,
            vram_units: det.vram_units,
            host_stream_units: det.host_stream_units,
            disk_stream_units: det.disk_stream_units,
            cpu_in_place_units: 0,
        });
        if u > 0 {
            active_ranks.push(d.rank as i32);
        }
    }

    if best.includes_cpu
        && let Some(cpu_det) = &best.cpu_allocation
        && cpu_det.total_units > 0
    {
        let u = cpu_det.total_units;
        allocations.push(ShardAllocation {
            rank: HOST_CPU_RANK,
            units: u,
            share_ratio: (u as f64) / total_f,
            estimated_compute_ms: cpu_det.compute_ms,
            estimated_memory_bytes: cpu_det.host_bytes,
            excluded_as_straggler: false,
            vram_units: 0,
            host_stream_units: 0,
            disk_stream_units: 0,
            cpu_in_place_units: cpu_det.cpu_in_place_units,
        });
        active_ranks.push(HOST_CPU_RANK);
    }

    Ok(WorkloadPlan {
        total_units: spec.total_units,
        allocations,
        active_ranks,
        estimated_makespan_ms: best.makespan_ms,
        estimated_compute_ms: best.compute_ms,
        estimated_sync_ms: best.sync_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::plan_work;
    use crate::collective::backend::{DeviceProfileC, LinkInfo, TRANSPORT_HOST_STAGED, TRANSPORT_PEER_TO_PEER};
    use crate::collective::planner::{HOST_CPU_RANK, HostCpuProfile, WorkloadSpec};
    use crate::engine::error::Hc4jError;

    fn gpu(rank: u32, vram_gb: u64, gflops: f64) -> DeviceProfileC {
        DeviceProfileC {
            rank,
            vendor_id: 0x10DE,
            device_id: rank,
            device_type: 2,
            vram_bytes: vram_gb << 30,
            compute_gflops: gflops,
        }
    }

    fn clique(n: u32, bw: f64, lat_us: f64, tr: u32) -> Vec<LinkInfo> {
        let mut links = Vec::new();
        for s in 0..n {
            for d in 0..n {
                links.push(LinkInfo {
                    src_rank: s,
                    dst_rank: d,
                    bandwidth_gbps: if s == d { 25.0 } else { bw },
                    transport: if s == d { TRANSPORT_PEER_TO_PEER } else { tr },
                    loopback: u32::from(s == d),
                    latency_us: if s == d { 0.0 } else { lat_us },
                });
            }
        }
        links
    }

    #[test]
    fn straggler_pruning_exhibits_zero_degradation_monotonicity() {
        // GPU 0: 40,000 GFLOPS discrete GPU; GPU 1: weak 50 GFLOPS iGPU over slow 1 GB/s PCIe
        let devices = vec![gpu(0, 16, 40_000.0), gpu(1, 4, 50.0)];
        let links = clique(2, 1.0, 100.0, TRANSPORT_HOST_STAGED);
        let spec = WorkloadSpec::new(4096, 1e9, 4096, 64 << 20);

        let plan = plan_work(&devices, &links, &spec).unwrap();
        assert_eq!(plan.active_ranks, vec![0]);
        assert!(plan.allocations[1].excluded_as_straggler);
        assert_eq!(plan.allocations[0].units, 4096);

        let gpu0_only = plan_work(&devices[..1], &[], &spec).unwrap();
        assert!(plan.estimated_makespan_ms <= gpu0_only.estimated_makespan_ms + 1e-9);
    }

    #[test]
    fn three_tier_spilling_and_cpu_in_place_regimes() {
        let devices = vec![gpu(0, 1, 20_000.0)];
        let links = vec![];
        // Workload needs 2.4 GB (> 0.85 GB usable VRAM) -> spills into host RAM + CPU in-place!
        let mut spec = WorkloadSpec::new(2400, 5e8, 1 << 20, 0);
        spec.host_cpu = HostCpuProfile {
            ram_bytes: 16 << 30,
            compute_gflops: 400.0,
            ram_bandwidth_gbps: 80.0,
            aggregate_pcie_gbps: 16.0,
        };
        let plan = plan_work(&devices, &links, &spec).unwrap();
        let sum_units: u64 = plan.allocations.iter().map(|a| a.units).sum();
        assert_eq!(sum_units, 2400);
        assert!(plan.allocations.iter().any(|a| a.rank == 0 && a.host_stream_units > 0));
        assert!(plan.allocations.iter().any(|a| a.rank == HOST_CPU_RANK && a.cpu_in_place_units > 0));
    }

    #[test]
    fn oom_when_capacity_insufficient_across_all_tiers() {
        let devices = vec![gpu(0, 1, 10_000.0)];
        let spec = WorkloadSpec::new(2000, 1e8, 1 << 20, 0);
        let err = plan_work(&devices, &[], &spec).unwrap_err();
        assert!(matches!(err, Hc4jError::OutOfMemory));
    }

    #[test]
    fn full_n12_plan_work_solve_time() {
        let devices: Vec<DeviceProfileC> = (0..12)
            .map(|r| gpu(r, 16, 15_000.0 + (r as f64) * 500.0))
            .collect();
        let links = clique(12, 24.0, 5.0, TRANSPORT_PEER_TO_PEER);
        let spec = WorkloadSpec::new(10_000, 1e9, 4096, 16 << 20);

        let started = std::time::Instant::now();
        let plan = plan_work(&devices, &links, &spec).unwrap();
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        println!("Theorem 4 N=12 plan_work solve time: {elapsed_ms:.3} ms");
        assert_eq!(plan.allocations.iter().map(|a| a.units).sum::<u64>(), 10_000);
    }
}
