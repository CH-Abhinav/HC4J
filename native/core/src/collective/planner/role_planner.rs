//! Zero-Waste Split-KV Reclamation planner (`plan_roles` + `solve_kv_minimax`):
//! assigns Tensor Parallel compute across active GPUs while reclaiming
//! excluded stragglers and host CPU RAM as lossless Split-KV attention workers.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use super::convex_spill::plan_work;
use super::{
    DeviceRole, HOST_CPU_RANK, REGIME_ALL_GPU_SHARD, REGIME_CPU_IN_PLACE, REGIME_PCIE_STREAMING,
    REGIME_SPLIT_KV_RECLAIMED, REGIME_STRAGGLER_PRUNED, RolePlan, SplitKvAssignment, SplitKvSpec, WorkloadSpec,
    build_topology_matrices, regime_name,
};
use crate::collective::backend::{DeviceProfileC, LinkInfo};
use crate::engine::error::{Hc4jError, Hc4jResult};

const EPSILON: f64 = 1e-6;

#[derive(Debug, Clone, Copy)]
struct KvWorker {
    rank: i32,
    max_tokens: u64,
    r_attn: f64,
    l_rpc: f64,
}

#[derive(Copy, Clone, PartialEq)]
struct KvHeapEntry {
    index: usize,
    next_time: f64,
}

impl Eq for KvHeapEntry {}

impl PartialOrd for KvHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KvHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .next_time
            .total_cmp(&self.next_time)
            .then_with(|| other.index.cmp(&self.index))
    }
}

fn solve_kv_bisection(workers: &[KvWorker], total_tokens: u64) -> (Vec<u64>, f64) {
    let m = workers.len();
    let mut result = vec![0u64; m];
    if m == 0 || total_tokens == 0 {
        return (result, 0.0);
    }

    let contrib = |t: f64, out: &mut [u64]| -> u64 {
        let mut sum = 0u64;
        for (i, w) in workers.iter().enumerate() {
            let threshold = w.l_rpc + w.r_attn;
            let p = if w.max_tokens == 0 || t < threshold {
                0
            } else if w.r_attn <= 0.0 {
                w.max_tokens
            } else {
                (((t - w.l_rpc) / w.r_attn).floor() as u64).min(w.max_tokens)
            };
            out[i] = p;
            sum = sum.saturating_add(p);
        }
        sum
    };

    let mut t_low = 0.0f64;
    let mut t_high = 1.0f64;
    for w in workers {
        let cap = w.max_tokens.min(total_tokens) as f64;
        t_high = t_high.max(w.l_rpc + cap * w.r_attn + w.r_attn);
    }
    let mut scratch = vec![0u64; m];
    for _ in 0..64 {
        if contrib(t_high, &mut scratch) >= total_tokens {
            break;
        }
        t_high *= 2.0;
    }

    for _ in 0..72 {
        let t_mid = t_low + (t_high - t_low) * 0.5;
        if contrib(t_mid, &mut scratch) >= total_tokens {
            t_high = t_mid;
        } else {
            t_low = t_mid;
        }
    }

    let current_sum = contrib(t_high, &mut result);
    if current_sum > total_tokens {
        // Trim any exact tie overshoot starting from the highest-time workers so makespan stays <= t_high.
        let mut excess = current_sum - total_tokens;
        let mut order: Vec<(f64, usize)> = result
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t > 0)
            .map(|(i, &t)| (workers[i].l_rpc + (t as f64) * workers[i].r_attn, i))
            .collect();
        while excess > 0 && !order.is_empty() {
            order.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            for (_, i) in &mut order {
                if excess == 0 {
                    break;
                }
                if result[*i] > 0 {
                    result[*i] -= 1;
                    excess -= 1;
                }
            }
            order.retain(|&(_, i)| result[i] > 0);
            for (tm, i) in &mut order {
                *tm = workers[*i].l_rpc + (result[*i] as f64) * workers[*i].r_attn;
            }
        }
    } else if current_sum < total_tokens {
        let mut deficit = total_tokens - current_sum;
        let mut pq = BinaryHeap::with_capacity(m);
        for (i, w) in workers.iter().enumerate() {
            if result[i] < w.max_tokens {
                pq.push(KvHeapEntry {
                    index: i,
                    next_time: w.l_rpc + ((result[i] + 1) as f64 * w.r_attn),
                });
            }
        }
        while deficit > 0 {
            let Some(top) = pq.pop() else { break };
            result[top.index] += 1;
            deficit -= 1;
            let w = workers[top.index];
            if result[top.index] < w.max_tokens {
                pq.push(KvHeapEntry {
                    index: top.index,
                    next_time: w.l_rpc + ((result[top.index] + 1) as f64 * w.r_attn),
                });
            }
        }
    }

    let mut max_time = 0.0f64;
    for (w, &tokens) in workers.iter().zip(&result) {
        if tokens > 0 {
            max_time = max_time.max(w.l_rpc + (tokens as f64 * w.r_attn));
        }
    }

    (result, max_time)
}

/// Computes the combined Tensor Parallel + Zero-Waste Split-KV role assignment plan.
///
/// Mirrors Java `WorkloadPlanner.planRoles` mathematically bit-for-bit.
pub fn plan_roles(
    devices: &[DeviceProfileC],
    topology: &[LinkInfo],
    tp_spec: &WorkloadSpec,
    kv_spec: &SplitKvSpec,
) -> Hc4jResult<RolePlan> {
    if kv_spec.kv_bytes_per_token == 0 {
        return Err(Hc4jError::InvalidParam("kv_bytes_per_token must be positive"));
    }
    if kv_spec.attention_flops_per_token < 0.0 || !kv_spec.attention_flops_per_token.is_finite() {
        return Err(Hc4jError::InvalidParam("attention_flops_per_token must be non-negative"));
    }
    if kv_spec.vram_utilization <= 0.0 || kv_spec.vram_utilization > 1.0 {
        return Err(Hc4jError::InvalidParam("kv_spec.vram_utilization must be in (0, 1]"));
    }

    let tp_plan = plan_work(devices, topology, tp_spec)?;
    let n = devices.len();
    let super::TopologyMatrices { bw, lat_us, .. } = build_topology_matrices(n, topology);

    let mut active_tp_gpus = Vec::new();
    let mut pruned_gpus = HashSet::new();
    for i in 0..n {
        if tp_plan.allocations[i].units > 0 {
            active_tp_gpus.push(i);
        } else {
            pruned_gpus.insert(i);
        }
    }

    let mut anchor_rank = active_tp_gpus.first().copied().unwrap_or(0);
    let mut max_tp_gflops = -1.0f64;
    for &r in &active_tp_gpus {
        if devices[r].compute_gflops > max_tp_gflops {
            max_tp_gflops = devices[r].compute_gflops;
            anchor_rank = r;
        }
    }

    let mut max_kv_tokens = vec![0u64; n];
    let mut r_attn = vec![0.0f64; n];
    let mut l_rpc = vec![0.0f64; n];

    for i in 0..n {
        let d = devices[i];
        let usable_vram = (d.vram_bytes as f64 * kv_spec.vram_utilization).floor() as u64;
        let avail_kv_vram = usable_vram.saturating_sub(kv_spec.reserved_weight_bytes_per_gpu);
        max_kv_tokens[i] = avail_kv_vram / kv_spec.kv_bytes_per_token;

        let c_gflops = d.compute_gflops.max(EPSILON);
        let mem_bw = if d.vram_bandwidth_gbps > 0.0 && d.vram_bandwidth_gbps.is_finite() {
            d.vram_bandwidth_gbps
        } else {
            match d.device_type {
                2 => 100.0,
                1 => 35.0,
                _ => 50.0,
            }
        };
        let t_comp = kv_spec.attention_flops_per_token / (c_gflops * 1e9);
        let t_mem = (kv_spec.kv_bytes_per_token as f64) / (mem_bw * 1e9);
        r_attn[i] = t_comp.max(t_mem) * 1000.0;

        if i == anchor_rank {
            l_rpc[i] = 0.0;
        } else {
            let link_bw = bw[anchor_rank][i].min(bw[i][anchor_rank]).max(EPSILON);
            let link_lat = (lat_us[anchor_rank][i] + lat_us[i][anchor_rank]) / 1000.0;
            let rpc_ms = ((kv_spec.query_and_partial_bytes as f64) / (link_bw * 1e9)) * 1000.0;
            l_rpc[i] = link_lat + rpc_ms;
        }
    }

    let cpu = tp_spec.host_cpu;
    let mut max_cpu_kv_tokens = 0u64;
    let mut r_cpu_attn = 0.0f64;
    let mut l_cpu_rpc = 0.0f64;
    if cpu.is_enabled() {
        max_cpu_kv_tokens = cpu.ram_bytes / kv_spec.kv_bytes_per_token;
        let c_gflops = cpu.compute_gflops.max(EPSILON);
        let mem_bw = cpu.ram_bandwidth_gbps.max(20.0);
        let t_comp = kv_spec.attention_flops_per_token / (c_gflops * 1e9);
        let t_mem = (kv_spec.kv_bytes_per_token as f64) / (mem_bw * 1e9);
        r_cpu_attn = t_comp.max(t_mem) * 1000.0;
        l_cpu_rpc = 0.05 + ((kv_spec.query_and_partial_bytes as f64) / (mem_bw * 1e9)) * 1000.0;
    }

    let total_kv = kv_spec.total_kv_tokens;
    let mut all_workers = Vec::with_capacity(n + usize::from(cpu.is_enabled()));
    for i in 0..n {
        if max_kv_tokens[i] > 0 {
            all_workers.push(KvWorker {
                rank: i as i32,
                max_tokens: max_kv_tokens[i],
                r_attn: r_attn[i],
                l_rpc: l_rpc[i],
            });
        }
    }
    if cpu.is_enabled() && max_cpu_kv_tokens > 0 {
        all_workers.push(KvWorker {
            rank: HOST_CPU_RANK,
            max_tokens: max_cpu_kv_tokens,
            r_attn: r_cpu_attn,
            l_rpc: l_cpu_rpc,
        });
    }

    let mut token_map: HashMap<i32, u64> = HashMap::new();
    let best_attn_ms: f64;

    if total_kv == 0 {
        best_attn_ms = 0.0;
    } else {
        let total_cap: u64 = all_workers
            .iter()
            .fold(0u64, |acc, w| acc.saturating_add(w.max_tokens));
        if total_cap < total_kv {
            return Err(Hc4jError::OutOfMemory);
        }

        let (alloc, attn_ms) = solve_kv_bisection(&all_workers, total_kv);
        best_attn_ms = attn_ms;
        for (w, &t) in all_workers.iter().zip(&alloc) {
            if t > 0 {
                token_map.insert(w.rank, t);
            }
        }
    }

    let active_tp_set: HashSet<usize> = active_tp_gpus.iter().copied().collect();
    let mut assignments = Vec::with_capacity(n + usize::from(cpu.is_enabled()));
    let mut reclaimed_count = 0u32;
    let mut cpu_reclaimed = false;

    for (i, d) in devices.iter().enumerate() {
        let tokens = token_map.get(&(i as i32)).copied().unwrap_or(0);
        let role = if active_tp_set.contains(&i) {
            DeviceRole::TensorParallel
        } else if tokens > 0 {
            reclaimed_count += 1;
            DeviceRole::SplitKvReclaimed
        } else {
            DeviceRole::Idle
        };

        let local_attn_ms = if tokens > 0 { tokens as f64 * r_attn[i] } else { 0.0 };
        let comm_ms = if tokens > 0 { l_rpc[i] } else { 0.0 };
        assignments.push(SplitKvAssignment {
            rank: d.rank as i32,
            role,
            kv_tokens: tokens,
            estimated_local_attn_ms: local_attn_ms,
            estimated_comm_ms: comm_ms,
            estimated_total_ms: local_attn_ms + comm_ms,
            kv_memory_bytes: tokens.saturating_mul(kv_spec.kv_bytes_per_token),
        });
    }

    if cpu.is_enabled() {
        let tokens = token_map.get(&HOST_CPU_RANK).copied().unwrap_or(0);
        let cpu_in_place = tp_plan.active_ranks.contains(&HOST_CPU_RANK);
        let role = if tokens > 0 || cpu_in_place {
            cpu_reclaimed = true;
            DeviceRole::CpuInPlace
        } else {
            DeviceRole::Idle
        };

        let local_attn_ms = if tokens > 0 { tokens as f64 * r_cpu_attn } else { 0.0 };
        let comm_ms = if tokens > 0 { l_cpu_rpc } else { 0.0 };
        assignments.push(SplitKvAssignment {
            rank: HOST_CPU_RANK,
            role,
            kv_tokens: tokens,
            estimated_local_attn_ms: local_attn_ms,
            estimated_comm_ms: comm_ms,
            estimated_total_ms: local_attn_ms + comm_ms,
            kv_memory_bytes: tokens.saturating_mul(kv_spec.kv_bytes_per_token),
        });
    }

    let regime_code = if reclaimed_count > 0 {
        REGIME_SPLIT_KV_RECLAIMED
    } else if cpu_reclaimed {
        REGIME_CPU_IN_PLACE
    } else {
        let total_spill: u64 = tp_plan
            .allocations
            .iter()
            .map(|a| a.host_stream_units.saturating_add(a.disk_stream_units))
            .sum();
        if total_spill > 0 {
            REGIME_PCIE_STREAMING
        } else if pruned_gpus.is_empty() {
            REGIME_ALL_GPU_SHARD
        } else {
            REGIME_STRAGGLER_PRUNED
        }
    };

    let estimated_tp_ms = tp_plan.estimated_makespan_ms;
    Ok(RolePlan {
        tp_plan,
        split_kv_assignments: assignments,
        estimated_tp_ms,
        estimated_attention_ms: best_attn_ms,
        total_kv_tokens: total_kv,
        reclaimed_gpu_count: reclaimed_count,
        cpu_reclaimed,
        regime_code,
        regime: regime_name(regime_code),
    })
}

#[cfg(test)]
mod tests {
    use super::plan_roles;
    use crate::collective::backend::{DeviceProfileC, LinkInfo, TRANSPORT_HOST_STAGED};
    use crate::collective::planner::{
        DeviceRole, REGIME_SPLIT_KV_RECLAIMED, SplitKvSpec, WorkloadSpec,
    };

    #[test]
    fn split_kv_reclaims_excluded_straggler_when_primary_kv_overflows() {
        let devices = vec![
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
                compute_gflops: 100.0,
                vram_bandwidth_gbps: 40.0,
            },
        ];
        let topology = vec![
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

        // TP prunes GPU 1 due to huge AllReduce / slow iGPU.
        let tp_spec = WorkloadSpec::new(2048, 2e9, 4096, 128 << 20);
        // GPU 0 holds ~6.8B usable VRAM; reserving 6B leaves ~859MB (~819 tokens @ 1MB/tok)
        // for KV, forcing reclamation of excluded GPU 1 for 1200 KV tokens (total cap ~1638).
        let mut kv_spec = SplitKvSpec::new(1200, 1 << 20, 1e6, 32768);
        kv_spec.reserved_weight_bytes_per_gpu = 6 << 30;
        kv_spec.vram_utilization = 0.85;

        let roles = plan_roles(&devices, &topology, &tp_spec, &kv_spec).unwrap();
        assert_eq!(roles.regime, "SPLIT_KV_RECLAIMED");
        assert_eq!(roles.regime_code, REGIME_SPLIT_KV_RECLAIMED);
        assert_eq!(roles.reclaimed_gpu_count, 1);
        assert_eq!(roles.split_kv_assignments[0].role, DeviceRole::TensorParallel);
        assert_eq!(roles.split_kv_assignments[1].role, DeviceRole::SplitKvReclaimed);
        let sum_kv: u64 = roles.split_kv_assignments.iter().map(|a| a.kv_tokens).sum();
        assert_eq!(sum_kv, 1200);
    }

    #[test]
    fn split_kv_role_solver_scales_beyond_12_workers_without_exponential_search() {
        let count = 24u32;
        let devices: Vec<DeviceProfileC> = (0..count)
            .map(|r| DeviceProfileC {
                rank: r,
                vendor_id: 0x10DE,
                device_id: r,
                device_type: if r < 16 { 2 } else { 1 },
                vram_bytes: 16 << 30,
                compute_gflops: if r < 16 { 30_000.0 } else { 800.0 },
                vram_bandwidth_gbps: if r < 16 { 900.0 } else { 40.0 },
            })
            .collect();
        let mut links = Vec::new();
        for s in 0..count {
            for d in 0..count {
                if s != d {
                    links.push(LinkInfo {
                        src_rank: s,
                        dst_rank: d,
                        bandwidth_gbps: if s < 16 && d < 16 { 20.0 } else { 2.0 },
                        transport: TRANSPORT_HOST_STAGED,
                        loopback: 0,
                        latency_us: if s < 16 && d < 16 { 5.0 } else { 60.0 },
                    });
                }
            }
        }
        let tp_spec = WorkloadSpec::new(4096, 2e9, 4096, 32 << 20);
        let kv_spec = SplitKvSpec::new(100_000, 512 << 10, 1e6, 64 << 10);

        let started = std::time::Instant::now();
        let roles = plan_roles(&devices, &links, &tp_spec, &kv_spec).unwrap();
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let sum_kv: u64 = roles.split_kv_assignments.iter().map(|a| a.kv_tokens).sum();
        assert_eq!(sum_kv, 100_000);
        assert!(elapsed_ms < 500.0, "24-worker Split-KV bisection took {elapsed_ms:.2} ms");
    }
}
