//! Theorem 3 & Theorem 4: 3-Tier Convex Memory-Spilling Curves, Explicit Host
//! CPU Offload, Shared Host-RAM/PCIe Contention Fixed Point, O(m log m)
//! Min-Heap Integer Minimax, and Exact 2^N-1 Subset Search with Zero-Degradation
//! Monotonicity.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::mst_islands::estimate_sync_ms_exact;
use super::ring_tsp::{AllSubsetsRingTable, precompute_all_subsets_ring_for_payload};
use super::{
    HOST_CPU_RANK, ShardAllocation, WorkloadPlan, WorkloadSpec, build_topology_matrices, validate_work_inputs,
};
use crate::collective::backend::{DeviceProfileC, LinkInfo};
use crate::engine::error::{Hc4jError, Hc4jResult};

const EPSILON: f64 = 1e-6;
const HOST_PCIE_SYNC_LATENCY_MS: f64 = 0.02;

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RoundTier {
    Vram,
    HostStream,
    DiskStream,
    CpuInPlace,
}

#[derive(Copy, Clone, PartialEq)]
struct PoolHeapEntry {
    /// `0..k` for active GPU, `usize::MAX` for Host CPU.
    slot: usize,
    tier: RoundTier,
    next_time: f64,
}

impl Eq for PoolHeapEntry {}

impl PartialOrd for PoolHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PoolHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .next_time
            .total_cmp(&self.next_time)
            .then_with(|| other.slot.cmp(&self.slot))
    }
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
    let k = active_gpu_indices.len();
    let total_units = spec.total_units;
    let m_bytes = spec.bytes_per_unit;
    let f_flops = spec.flops_per_unit;
    let kappa = spec.io_multiplier;
    let gamma = spec.streaming_overlap_gamma;
    let cpu = spec.host_cpu;

    if k == 0 && !include_cpu {
        return None;
    }

    // Decoupled from `include_cpu`: physical host RAM allows GPU-only PCIe spilling whenever `cpu.ram_bytes > 0`.
    let total_host_ram_bytes = if spec.allow_spilling && cpu.ram_bytes > 0 {
        cpu.ram_bytes
    } else {
        0
    };
    let total_disk_bytes = if spec.allow_spilling { spec.disk_capacity_bytes } else { 0 };

    let mut gpu_dets: Vec<Option<AllocationDetail>> = vec![None; n];
    let mut cpu_det: Option<AllocationDetail> = None;

    let r0: Vec<f64> = active_gpu_indices
        .iter()
        .map(|&i| {
            let c_gflops = devices[i].compute_gflops.max(EPSILON);
            let t_comp = f_flops / (c_gflops * 1e9);
            let t_vram = if m_bytes > 0
                && devices[i].vram_bandwidth_gbps > 0.0
                && devices[i].vram_bandwidth_gbps.is_finite()
            {
                (kappa * m_bytes as f64) / (devices[i].vram_bandwidth_gbps * 1e9)
            } else {
                0.0
            };
            t_comp.max(t_vram) * 1000.0
        })
        .collect();

    let r_cpu = if include_cpu {
        let cpu_gflops = cpu.compute_gflops.max(EPSILON);
        let t_comp = f_flops / (cpu_gflops * 1e9);
        let t_ram = if cpu.ram_bandwidth_gbps > 0.0 {
            (kappa * m_bytes as f64) / (cpu.ram_bandwidth_gbps * 1e9)
        } else {
            0.0
        };
        t_comp.max(t_ram) * 1000.0
    } else {
        0.0
    };

    let max_compute_ms: f64 = if m_bytes == 0 {
        // Zero-memory workload: all units fit directly in Tier 0 / CPU with zero spilling.
        let mut t_high = 1.0f64;
        for &rate in &r0 {
            t_high = t_high.max((total_units as f64) * rate);
        }
        if include_cpu {
            t_high = t_high.max((total_units as f64) * r_cpu);
        }
        let mut t_low = 0.0f64;
        for _ in 0..64 {
            let t_mid = t_low + (t_high - t_low) * 0.5;
            let mut sum_u = 0.0f64;
            for &rate in &r0 {
                sum_u += if rate > 0.0 { t_mid / rate } else { total_units as f64 };
            }
            if include_cpu {
                sum_u += if r_cpu > 0.0 { t_mid / r_cpu } else { total_units as f64 };
            }
            if sum_u >= total_units as f64 {
                t_high = t_mid;
            } else {
                t_low = t_mid;
            }
        }

        let t_opt = t_high;
        let mut v_int = vec![0u64; k];
        let mut current_sum = 0u64;
        for a in 0..k {
            let u = if r0[a] > 0.0 {
                ((t_opt / r0[a]).floor() as u64).min(total_units)
            } else {
                total_units.saturating_sub(current_sum)
            };
            v_int[a] = u;
            current_sum = current_sum.saturating_add(u);
        }
        let mut u_c_int = if include_cpu {
            let u = if r_cpu > 0.0 {
                ((t_opt / r_cpu).floor() as u64).min(total_units)
            } else {
                total_units.saturating_sub(current_sum)
            };
            current_sum = current_sum.saturating_add(u);
            u
        } else {
            0
        };

        if total_units > current_sum {
            let mut deficit = total_units - current_sum;
            let mut pq = BinaryHeap::with_capacity(k + 1);
            for a in 0..k {
                pq.push(PoolHeapEntry {
                    slot: a,
                    tier: RoundTier::Vram,
                    next_time: ((v_int[a] + 1) as f64) * r0[a],
                });
            }
            if include_cpu {
                pq.push(PoolHeapEntry {
                    slot: usize::MAX,
                    tier: RoundTier::CpuInPlace,
                    next_time: ((u_c_int + 1) as f64) * r_cpu,
                });
            }
            while deficit > 0 {
                let Some(top) = pq.pop() else { break };
                deficit -= 1;
                if top.slot == usize::MAX {
                    u_c_int += 1;
                    pq.push(PoolHeapEntry {
                        slot: usize::MAX,
                        tier: RoundTier::CpuInPlace,
                        next_time: ((u_c_int + 1) as f64) * r_cpu,
                    });
                } else {
                    v_int[top.slot] += 1;
                    pq.push(PoolHeapEntry {
                        slot: top.slot,
                        tier: RoundTier::Vram,
                        next_time: ((v_int[top.slot] + 1) as f64) * r0[top.slot],
                    });
                }
            }
        } else if current_sum > total_units {
            let mut surplus = current_sum - total_units;
            if include_cpu && surplus > 0 {
                let take = u_c_int.min(surplus);
                u_c_int -= take;
                surplus -= take;
            }
            for a in (0..k).rev() {
                if surplus == 0 {
                    break;
                }
                let take = v_int[a].min(surplus);
                v_int[a] -= take;
                surplus -= take;
            }
        }

        let mut comp_max = 0.0f64;
        for a in 0..k {
            let u = v_int[a];
            let tm = (u as f64) * r0[a];
            comp_max = comp_max.max(tm);
            gpu_dets[active_gpu_indices[a]] = Some(AllocationDetail {
                total_units: u,
                vram_units: u,
                host_stream_units: 0,
                disk_stream_units: 0,
                cpu_in_place_units: 0,
                compute_ms: tm,
                vram_bytes: 0,
                host_bytes: 0,
                disk_bytes: 0,
            });
        }
        if include_cpu {
            let tm = (u_c_int as f64) * r_cpu;
            comp_max = comp_max.max(tm);
            cpu_det = Some(AllocationDetail {
                total_units: u_c_int,
                vram_units: 0,
                host_stream_units: 0,
                disk_stream_units: 0,
                cpu_in_place_units: u_c_int,
                compute_ms: tm,
                vram_bytes: 0,
                host_bytes: 0,
                disk_bytes: 0,
            });
        }
        comp_max
    } else {
        let mut u_vram = vec![0u64; k];
        let mut total_vram_cap = 0u64;
        for (a, &i) in active_gpu_indices.iter().enumerate() {
            let uv = ((devices[i].vram_bytes as f64 * spec.vram_utilization) / m_bytes as f64).floor() as u64;
            u_vram[a] = uv;
            total_vram_cap = total_vram_cap.saturating_add(uv);
        }

        let inv_sum: f64 = if !include_cpu && k > 0 && r0.iter().all(|&r| r > 0.0) {
            r0.iter().map(|&r| 1.0 / r).sum()
        } else {
            0.0
        };
        let t0_vram = if inv_sum > 0.0 {
            (total_units as f64) / inv_sum
        } else {
            f64::INFINITY
        };
        let fast_vram_only = !include_cpu
            && k > 0
            && total_vram_cap >= total_units
            && inv_sum > 0.0
            && (0..k).all(|a| (t0_vram / r0[a]) + 1.0 <= u_vram[a] as f64);

        if fast_vram_only {
            let mut v_int = vec![0u64; k];
            let mut current_sum = 0u64;
            for a in 0..k {
                let va = ((t0_vram / r0[a]).floor() as u64).min(u_vram[a]);
                v_int[a] = va;
                current_sum = current_sum.saturating_add(va);
            }
            if total_units > current_sum {
                let mut deficit = total_units - current_sum;
                let mut pq = BinaryHeap::with_capacity(k);
                for a in 0..k {
                    if v_int[a] < u_vram[a] {
                        pq.push(PoolHeapEntry {
                            slot: a,
                            tier: RoundTier::Vram,
                            next_time: ((v_int[a] + 1) as f64) * r0[a],
                        });
                    }
                }
                while deficit > 0 {
                    let Some(top) = pq.pop() else { break };
                    v_int[top.slot] += 1;
                    deficit -= 1;
                    if v_int[top.slot] < u_vram[top.slot] {
                        pq.push(PoolHeapEntry {
                            slot: top.slot,
                            tier: RoundTier::Vram,
                            next_time: ((v_int[top.slot] + 1) as f64) * r0[top.slot],
                        });
                    }
                }
            } else if current_sum > total_units {
                let mut surplus = current_sum - total_units;
                for a in (0..k).rev() {
                    if surplus == 0 {
                        break;
                    }
                    let take = v_int[a].min(surplus);
                    v_int[a] -= take;
                    surplus -= take;
                }
            }
            let mut comp_max = 0.0f64;
            for a in 0..k {
                let u = v_int[a];
                let tm = (u as f64) * r0[a];
                comp_max = comp_max.max(tm);
                gpu_dets[active_gpu_indices[a]] = Some(AllocationDetail {
                    total_units: u,
                    vram_units: u,
                    host_stream_units: 0,
                    disk_stream_units: 0,
                    cpu_in_place_units: 0,
                    compute_ms: tm,
                    vram_bytes: u.saturating_mul(m_bytes),
                    host_bytes: 0,
                    disk_bytes: 0,
                });
            }
            comp_max
        } else {
        // Shared-pool Fixed-T LP + budget-respecting min-heap integer rounding.
        let mut r1 = vec![0.0f64; k];
        let mut r2 = vec![0.0f64; k];
        let per_gpu_pcie = spec.default_pcie_gbps;
        let b_pcie_agg = if cpu.aggregate_pcie_gbps > 0.0 {
            cpu.aggregate_pcie_gbps
        } else {
            (k as f64) * per_gpu_pcie
        };

        let host_cap_units = total_host_ram_bytes.checked_div(m_bytes).unwrap_or(0);
        let host_spill_enabled = k > 0 && total_host_ram_bytes > 0 && per_gpu_pcie > 0.0 && b_pcie_agg > 0.0;
        let disk_enabled =
            k > 0 && spec.allow_spilling && spec.disk_bandwidth_gbps > 0.0 && total_disk_bytes > 0 && per_gpu_pcie > 0.0 && b_pcie_agg > 0.0;
        let disk_cap_units = if disk_enabled {
            total_disk_bytes.checked_div(m_bytes).unwrap_or(0)
        } else {
            0
        };

        for (a, &i) in active_gpu_indices.iter().enumerate() {
            let c_gflops = devices[i].compute_gflops.max(EPSILON);
            let t_comp = f_flops / (c_gflops * 1e9);

            let t_pcie = if per_gpu_pcie > 0.0 {
                (kappa * m_bytes as f64) / (per_gpu_pcie * 1e9)
            } else {
                1e3
            };
            r1[a] = (t_comp.max(t_pcie) + (1.0 - gamma) * t_comp.min(t_pcie)) * 1000.0;

            let b_disk_eff = if disk_enabled {
                per_gpu_pcie.min(spec.disk_bandwidth_gbps)
            } else {
                0.0
            };
            let t_disk = if b_disk_eff > 0.0 {
                (kappa * m_bytes as f64) / (b_disk_eff * 1e9)
            } else {
                1e3
            };
            r2[a] = (t_comp.max(t_disk) + (1.0 - gamma) * t_comp.min(t_disk)) * 1000.0;
        }

        let effective_host_cap = if include_cpu || host_spill_enabled {
            host_cap_units
        } else {
            0
        };
        let total_capacity = total_vram_cap
            .saturating_add(effective_host_cap)
            .saturating_add(disk_cap_units);
        if total_capacity < total_units {
            return None;
        }

        let mut host_order: Vec<usize> = (0..k).collect();
        host_order.sort_by(|&a, &b| {
            let ra = if disk_enabled {
                r1[a] / r2[a].max(1e-12)
            } else {
                r1[a]
            };
            let rb = if disk_enabled {
                r1[b] / r2[b].max(1e-12)
            } else {
                r1[b]
            };
            ra.total_cmp(&rb)
                .then_with(|| r1[a].total_cmp(&r1[b]))
                .then_with(|| active_gpu_indices[a].cmp(&active_gpu_indices[b]))
        });

        let mut disk_order: Vec<usize> = (0..k).collect();
        disk_order.sort_by(|&a, &b| {
            r2[a]
                .total_cmp(&r2[b])
                .then_with(|| active_gpu_indices[a].cmp(&active_gpu_indices[b]))
        });

        let kappa_bytes = kappa * (m_bytes as f64);

        let mut lp_v = vec![0.0f64; k];
        let mut lp_delta_t = vec![0.0f64; k];
        let mut lp_s = vec![0.0f64; k];
        let mut lp_d = vec![0.0f64; k];
        let mut lp_u_c = 0.0f64;

        let mut phi = |t_ms: f64| -> bool {
            let h_bar_t = if total_host_ram_bytes == 0 {
                0.0
            } else if cpu.ram_bandwidth_gbps > 0.0 && kappa_bytes > 0.0 {
                (host_cap_units as f64).min((cpu.ram_bandwidth_gbps * t_ms * 1e6) / kappa_bytes)
            } else {
                host_cap_units as f64
            };

            let p_pcie_t = if b_pcie_agg > 0.0 {
                if kappa_bytes > 0.0 {
                    (b_pcie_agg * t_ms * 1e6) / kappa_bytes
                } else {
                    f64::INFINITY
                }
            } else {
                0.0
            };

            let k_disk_t = if !disk_enabled {
                0.0
            } else if kappa_bytes > 0.0 {
                (disk_cap_units as f64).min((spec.disk_bandwidth_gbps * t_ms * 1e6) / kappa_bytes)
            } else {
                disk_cap_units as f64
            };

            // 1. Fill each active GPU's local VRAM tier first
            let mut sum_units = 0.0f64;
            for a in 0..k {
                let va = if r0[a] > 0.0 {
                    (u_vram[a] as f64).min(t_ms / r0[a])
                } else {
                    u_vram[a] as f64
                };
                lp_v[a] = va;
                lp_delta_t[a] = (t_ms - r0[a] * va).max(0.0);
                sum_units += va;
            }

            // 2. Fill CPU-in-place if include_cpu == true
            lp_u_c = if include_cpu {
                let uc = h_bar_t.min(if r_cpu > 0.0 { t_ms / r_cpu } else { h_bar_t }).max(0.0);
                sum_units += uc;
                uc
            } else {
                0.0
            };

            // 3. Fill GPU PCIe Host-RAM streaming greedily in ascending r1 / r2 order
            lp_s.fill(0.0);
            let mut sum_s = 0.0f64;
            if host_spill_enabled {
                let mut rem_s_cap = p_pcie_t.min((h_bar_t - lp_u_c).max(0.0));
                for &a in &host_order {
                    if rem_s_cap <= 0.0 {
                        break;
                    }
                    let cap_a = if r1[a] > 0.0 { lp_delta_t[a] / r1[a] } else { rem_s_cap };
                    let sa = cap_a.min(rem_s_cap).max(0.0);
                    lp_s[a] = sa;
                    rem_s_cap -= sa;
                    sum_s += sa;
                    lp_delta_t[a] = (lp_delta_t[a] - r1[a] * sa).max(0.0);
                }
                sum_units += sum_s;
            }

            // 4. Fill Tier-2 Disk streaming
            lp_d.fill(0.0);
            if disk_enabled {
                let pcie_rem = (p_pcie_t - sum_s).max(0.0);
                let mut max_disk_from_gpu_time = 0.0f64;
                for a in 0..k {
                    if r2[a] > 0.0 {
                        max_disk_from_gpu_time += lp_delta_t[a] / r2[a];
                    } else {
                        max_disk_from_gpu_time = f64::INFINITY;
                    }
                }
                let mut rem_disk = k_disk_t.min(pcie_rem).min(max_disk_from_gpu_time).max(0.0);
                sum_units += rem_disk;
                for &a in &disk_order {
                    if rem_disk <= 0.0 {
                        break;
                    }
                    let cap_a = if r2[a] > 0.0 { lp_delta_t[a] / r2[a] } else { rem_disk };
                    let da = cap_a.min(rem_disk).max(0.0);
                    lp_d[a] = da;
                    rem_disk -= da;
                }
            }

            sum_units >= total_units as f64
        };

        let mut t_low = 0.0f64;
        let mut t_high = 1.0f64;
        let mut doubled = 0;
        while !phi(t_high) {
            t_high *= 2.0;
            doubled += 1;
            if !t_high.is_finite() || doubled > 100 {
                return None;
            }
        }

        for _ in 0..64 {
            let t_mid = t_low + (t_high - t_low) * 0.5;
            if phi(t_mid) {
                t_high = t_mid;
            } else {
                t_low = t_mid;
            }
        }

        let _ = phi(t_high);
        let mut v_int = vec![0u64; k];
        let mut s_int = vec![0u64; k];
        let mut d_int = vec![0u64; k];

        let mut u_c_int = if include_cpu {
            (lp_u_c.floor() as u64).min(host_cap_units)
        } else {
            0
        };
        let mut rem_host_units = effective_host_cap.saturating_sub(u_c_int);
        let mut rem_disk_units = disk_cap_units;

        let mut current_sum = u_c_int;
        let mut used_ram = u_c_int;
        let mut used_pcie = 0u64;
        let mut used_disk = 0u64;

        for a in 0..k {
            let va = (lp_v[a].floor() as u64).min(u_vram[a]);
            v_int[a] = va;
            current_sum = current_sum.saturating_add(va);

            if host_spill_enabled && rem_host_units > 0 {
                let sa = (lp_s[a].floor() as u64).min(rem_host_units);
                s_int[a] = sa;
                rem_host_units -= sa;
                used_ram += sa;
                used_pcie += sa;
                current_sum = current_sum.saturating_add(sa);
            }

            if disk_enabled && rem_disk_units > 0 {
                let da = (lp_d[a].floor() as u64).min(rem_disk_units);
                d_int[a] = da;
                rem_disk_units -= da;
                used_pcie += da;
                used_disk += da;
                current_sum = current_sum.saturating_add(da);
            }
        }

        let ram_unit_ms = if cpu.ram_bandwidth_gbps > 0.0 && kappa_bytes > 0.0 {
            kappa_bytes / (cpu.ram_bandwidth_gbps * 1e6)
        } else {
            0.0
        };
        let pcie_unit_ms = if b_pcie_agg > 0.0 && kappa_bytes > 0.0 {
            kappa_bytes / (b_pcie_agg * 1e6)
        } else {
            0.0
        };
        let disk_unit_ms = if disk_enabled && kappa_bytes > 0.0 {
            kappa_bytes / (spec.disk_bandwidth_gbps * 1e6)
        } else {
            0.0
        };

        let mut gpu_time = vec![0.0f64; k];
        for a in 0..k {
            gpu_time[a] = (v_int[a] as f64) * r0[a] + (s_int[a] as f64) * r1[a] + (d_int[a] as f64) * r2[a];
        }

        let next_gpu_candidate = |a: usize,
                                  v_a: u64,
                                  g_t: f64,
                                  rem_h: u64,
                                  rem_d: u64,
                                  u_ram: u64,
                                  u_pcie: u64,
                                  u_disk: u64|
         -> Option<(RoundTier, f64)> {
            if v_a < u_vram[a] {
                Some((RoundTier::Vram, g_t + r0[a]))
            } else if host_spill_enabled && rem_h > 0 {
                let t = (g_t + r1[a])
                    .max(((u_ram + 1) as f64) * ram_unit_ms)
                    .max(((u_pcie + 1) as f64) * pcie_unit_ms);
                Some((RoundTier::HostStream, t))
            } else if disk_enabled && rem_d > 0 {
                let t = (g_t + r2[a])
                    .max(((u_pcie + 1) as f64) * pcie_unit_ms)
                    .max(((u_disk + 1) as f64) * disk_unit_ms);
                Some((RoundTier::DiskStream, t))
            } else {
                None
            }
        };

        let next_cpu_candidate = |uc: u64, rem_h: u64, u_ram: u64| -> Option<(RoundTier, f64)> {
            if include_cpu && rem_h > 0 {
                let t = (((uc + 1) as f64) * r_cpu).max(((u_ram + 1) as f64) * ram_unit_ms);
                Some((RoundTier::CpuInPlace, t))
            } else {
                None
            }
        };

        if total_units > current_sum {
            let mut deficit = total_units - current_sum;
            let mut pq = BinaryHeap::with_capacity(k + 1);
            for a in 0..k {
                if let Some((tier, tm)) =
                    next_gpu_candidate(a, v_int[a], gpu_time[a], rem_host_units, rem_disk_units, used_ram, used_pcie, used_disk)
                {
                    pq.push(PoolHeapEntry {
                        slot: a,
                        tier,
                        next_time: tm,
                    });
                }
            }
            if let Some((tier, tm)) = next_cpu_candidate(u_c_int, rem_host_units, used_ram) {
                pq.push(PoolHeapEntry {
                    slot: usize::MAX,
                    tier,
                    next_time: tm,
                });
            }

            while deficit > 0 {
                let Some(top) = pq.pop() else { break };
                if top.slot == usize::MAX {
                    let Some((cur_tier, cur_tm)) = next_cpu_candidate(u_c_int, rem_host_units, used_ram) else {
                        continue;
                    };
                    if cur_tier != top.tier || (cur_tm - top.next_time).abs() > 1e-12 {
                        pq.push(PoolHeapEntry {
                            slot: usize::MAX,
                            tier: cur_tier,
                            next_time: cur_tm,
                        });
                        continue;
                    }
                    u_c_int += 1;
                    rem_host_units -= 1;
                    used_ram += 1;
                    deficit -= 1;
                    if let Some((nt, ntm)) = next_cpu_candidate(u_c_int, rem_host_units, used_ram) {
                        pq.push(PoolHeapEntry {
                            slot: usize::MAX,
                            tier: nt,
                            next_time: ntm,
                        });
                    }
                } else {
                    let a = top.slot;
                    let Some((cur_tier, cur_tm)) = next_gpu_candidate(
                        a,
                        v_int[a],
                        gpu_time[a],
                        rem_host_units,
                        rem_disk_units,
                        used_ram,
                        used_pcie,
                        used_disk,
                    ) else {
                        continue;
                    };
                    if cur_tier != top.tier || (cur_tm - top.next_time).abs() > 1e-12 {
                        pq.push(PoolHeapEntry {
                            slot: a,
                            tier: cur_tier,
                            next_time: cur_tm,
                        });
                        continue;
                    }
                    match cur_tier {
                        RoundTier::Vram => {
                            v_int[a] += 1;
                            gpu_time[a] += r0[a];
                        }
                        RoundTier::HostStream => {
                            s_int[a] += 1;
                            gpu_time[a] += r1[a];
                            rem_host_units -= 1;
                            used_ram += 1;
                            used_pcie += 1;
                        }
                        RoundTier::DiskStream => {
                            d_int[a] += 1;
                            gpu_time[a] += r2[a];
                            rem_disk_units -= 1;
                            used_pcie += 1;
                            used_disk += 1;
                        }
                        RoundTier::CpuInPlace => {}
                    }
                    deficit -= 1;
                    if let Some((nt, ntm)) = next_gpu_candidate(
                        a,
                        v_int[a],
                        gpu_time[a],
                        rem_host_units,
                        rem_disk_units,
                        used_ram,
                        used_pcie,
                        used_disk,
                    ) {
                        pq.push(PoolHeapEntry {
                            slot: a,
                            tier: nt,
                            next_time: ntm,
                        });
                    }
                }
            }
        } else if current_sum > total_units {
            let mut surplus = current_sum - total_units;
            while surplus > 0 {
                // Trim from the highest-time participant first.
                let mut best_slot = None;
                let mut best_tm = -1.0f64;
                if include_cpu && u_c_int > 0 {
                    let t = (u_c_int as f64) * r_cpu;
                    best_slot = Some(usize::MAX);
                    best_tm = t;
                }
                for (a, &tm) in gpu_time.iter().enumerate() {
                    if v_int[a] + s_int[a] + d_int[a] > 0 && tm >= best_tm {
                        best_tm = tm;
                        best_slot = Some(a);
                    }
                }
                let Some(slot) = best_slot else { break };
                if slot == usize::MAX {
                    u_c_int -= 1;
                    used_ram -= 1;
                } else if d_int[slot] > 0 {
                    d_int[slot] -= 1;
                    gpu_time[slot] -= r2[slot];
                    used_pcie -= 1;
                    used_disk -= 1;
                } else if s_int[slot] > 0 {
                    s_int[slot] -= 1;
                    gpu_time[slot] -= r1[slot];
                    used_ram -= 1;
                    used_pcie -= 1;
                } else {
                    v_int[slot] -= 1;
                    gpu_time[slot] -= r0[slot];
                }
                surplus -= 1;
            }
        }

        let mut comp_max = 0.0f64;
        for a in 0..k {
            let total_a = v_int[a] + s_int[a] + d_int[a];
            let tm = gpu_time[a];
            comp_max = comp_max.max(tm);
            gpu_dets[active_gpu_indices[a]] = Some(AllocationDetail {
                total_units: total_a,
                vram_units: v_int[a],
                host_stream_units: s_int[a],
                disk_stream_units: d_int[a],
                cpu_in_place_units: 0,
                compute_ms: tm,
                vram_bytes: v_int[a].saturating_mul(m_bytes),
                host_bytes: s_int[a].saturating_mul(m_bytes),
                disk_bytes: d_int[a].saturating_mul(m_bytes),
            });
        }
        if include_cpu {
            let tm = (u_c_int as f64) * r_cpu;
            comp_max = comp_max.max(tm);
            cpu_det = Some(AllocationDetail {
                total_units: u_c_int,
                vram_units: 0,
                host_stream_units: 0,
                disk_stream_units: 0,
                cpu_in_place_units: u_c_int,
                compute_ms: tm,
                vram_bytes: 0,
                host_bytes: u_c_int.saturating_mul(m_bytes),
                disk_bytes: 0,
            });
        }
        if used_ram > 0 {
            comp_max = comp_max.max((used_ram as f64) * ram_unit_ms);
        }
        if used_pcie > 0 {
            comp_max = comp_max.max((used_pcie as f64) * pcie_unit_ms);
        }
        if used_disk > 0 {
            comp_max = comp_max.max((used_disk as f64) * disk_unit_ms);
        }
        comp_max
        }
    };

    let resolved_gpus: Vec<AllocationDetail> = gpu_dets
        .into_iter()
        .map(|d| d.unwrap_or_else(AllocationDetail::zero))
        .collect();

    let gpu_sync_ms = if k >= 2 && spec.sync_bytes > 0 {
        estimate_sync_ms_exact(active_gpu_indices, bw, lat_us, tr, ring_table, devices, spec.sync_bytes)
    } else {
        0.0
    };

    let host_sync_ms = if k >= 1 && include_cpu && spec.sync_bytes > 0 {
        let pcie_bw = if cpu.aggregate_pcie_gbps > 0.0 {
            spec.default_pcie_gbps.min(cpu.aggregate_pcie_gbps)
        } else {
            spec.default_pcie_gbps
        }
        .max(EPSILON);
        let pcie_xfer_ms =
            HOST_PCIE_SYNC_LATENCY_MS + ((spec.sync_bytes as f64) / (pcie_bw * 1e9)) * 1000.0;
        let ram_bw = if cpu.ram_bandwidth_gbps > 0.0 {
            cpu.ram_bandwidth_gbps
        } else {
            20.0
        };
        let cpu_gflops = cpu.compute_gflops.max(EPSILON);
        let host_reduce_mem_ms = ((3.0 * spec.sync_bytes as f64) / (ram_bw * 1e9)) * 1000.0;
        let host_reduce_comp_ms = (((spec.sync_bytes / 4) as f64) / (cpu_gflops * 1e9)) * 1000.0;
        pcie_xfer_ms + host_reduce_mem_ms.max(host_reduce_comp_ms)
    } else {
        0.0
    };

    let sync_ms = gpu_sync_ms + host_sync_ms;

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
    let ring_table = (n <= 12).then(|| precompute_all_subsets_ring_for_payload(n, &bw, &lat_us, spec.sync_bytes));
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
            vram_bandwidth_gbps: 200.0,
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

    #[test]
    fn gpu_only_plan_spills_to_host_ram_when_cpu_compute_disabled() {
        // GPU has 1 GiB VRAM (~870 units usable), workload requires 1500 units @ 1 MiB.
        // host_cpu has physical 8 GiB RAM, but compute_gflops = 0.0 -> include_cpu == false.
        let devices = vec![gpu(0, 1, 20_000.0)];
        let mut spec = WorkloadSpec::new(1500, 5e8, 1 << 20, 0);
        spec.host_cpu = HostCpuProfile {
            ram_bytes: 8 << 30,
            compute_gflops: 0.0,
            ram_bandwidth_gbps: 60.0,
            aggregate_pcie_gbps: 16.0,
        };
        let plan = plan_work(&devices, &[], &spec).unwrap();
        assert_eq!(plan.allocations.len(), 1);
        assert_eq!(plan.allocations[0].rank, 0);
        assert!(plan.allocations[0].host_stream_units > 0, "GPU-only plan must spill to Host RAM");
        assert_eq!(plan.allocations[0].units, 1500);
    }

    #[test]
    fn single_gpu_plus_cpu_has_positive_pcie_sync_cost() {
        let devices = vec![gpu(0, 1, 20_000.0)];
        let mut spec = WorkloadSpec::new(2400, 5e8, 1 << 20, 32 << 20);
        spec.host_cpu = HostCpuProfile {
            ram_bytes: 16 << 30,
            compute_gflops: 500.0,
            ram_bandwidth_gbps: 80.0,
            aggregate_pcie_gbps: 16.0,
        };
        let plan = plan_work(&devices, &[], &spec).unwrap();
        assert!(plan.allocations.iter().any(|a| a.rank == HOST_CPU_RANK && a.cpu_in_place_units > 0));
        assert!(plan.estimated_sync_ms > 0.0, "1-GPU + CPU hybrid plan must have positive PCIe sync_ms");
    }
}
