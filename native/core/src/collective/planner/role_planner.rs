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

fn solve_kv_minimax(workers: &[KvWorker], total_tokens: u64) -> Vec<u64> {
    let m = workers.len();
    let mut result = vec![0u64; m];
    if m == 0 || total_tokens == 0 {
        return result;
    }

    let mut t_low = 0.0f64;
    let mut t_high = 1.0f64;
    for w in workers {
        t_high = t_high.max(w.l_rpc + (total_tokens as f64) * w.r_attn);
    }

    for _ in 0..64 {
        let t_mid = t_low + (t_high - t_low) * 0.5;
        let mut sum_tokens = 0.0f64;
        for w in workers {
            if t_mid > w.l_rpc {
                let tokens = (t_mid - w.l_rpc) / w.r_attn;
                sum_tokens += tokens.min(w.max_tokens as f64);
            }
        }
        if sum_tokens >= total_tokens as f64 {
            t_high = t_mid;
        } else {
            t_low = t_mid;
        }
    }

    let t_opt = t_high;
    let mut current_sum = 0u64;
    for (i, w) in workers.iter().enumerate() {
        let mut tokens = 0u64;
        if t_opt > w.l_rpc {
            let raw = ((t_opt - w.l_rpc) / w.r_attn).floor() as u64;
            tokens = raw.min(w.max_tokens);
        }
        result[i] = tokens;
        current_sum = current_sum.saturating_add(tokens);
    }

    if total_tokens > current_sum {
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

    result
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
        let mem_bw = (c_gflops * 0.1).max(50.0);
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

    let mut best_worker_set: Option<Vec<KvWorker>> = None;
    let mut best_kv_alloc: Option<Vec<u64>> = None;
    let mut best_attn_ms = f64::MAX;

    if total_kv == 0 {
        best_attn_ms = 0.0;
    } else {
        let w_count = all_workers.len();
        if w_count <= 12 {
            let max_kv_subsets = 1usize << w_count;
            let mut cand_workers = Vec::with_capacity(w_count);
            for mask in 1usize..max_kv_subsets {
                cand_workers.clear();
                let mut cand_cap = 0u64;
                for (i, &w) in all_workers.iter().enumerate() {
                    if (mask & (1usize << i)) != 0 {
                        cand_workers.push(w);
                        cand_cap = cand_cap.saturating_add(w.max_tokens);
                    }
                }
                if cand_cap < total_kv {
                    continue;
                }

                let alloc = solve_kv_minimax(&cand_workers, total_kv);
                let mut max_time = 0.0f64;
                for (w, &tokens) in cand_workers.iter().zip(&alloc) {
                    let time = if tokens > 0 { w.l_rpc + (tokens as f64 * w.r_attn) } else { 0.0 };
                    max_time = max_time.max(time);
                }

                if max_time < best_attn_ms * (1.0 - 1e-6) {
                    best_attn_ms = max_time;
                    best_worker_set = Some(cand_workers.clone());
                    best_kv_alloc = Some(alloc);
                }
            }
        } else {
            let total_cap: u64 = all_workers.iter().fold(0u64, |acc, w| acc.saturating_add(w.max_tokens));
            if total_cap >= total_kv {
                let alloc = solve_kv_minimax(&all_workers, total_kv);
                let mut max_time = 0.0f64;
                for (w, &tokens) in all_workers.iter().zip(&alloc) {
                    let time = if tokens > 0 { w.l_rpc + (tokens as f64 * w.r_attn) } else { 0.0 };
                    max_time = max_time.max(time);
                }
                best_attn_ms = max_time;
                best_worker_set = Some(all_workers);
                best_kv_alloc = Some(alloc);
            }
        }

        if best_worker_set.is_none() {
            return Err(Hc4jError::OutOfMemory);
        }
    }

    let mut token_map: HashMap<i32, u64> = HashMap::new();
    if let (Some(workers), Some(alloc)) = (&best_worker_set, &best_kv_alloc) {
        for (w, &t) in workers.iter().zip(alloc) {
            token_map.insert(w.rank, t);
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
            },
            DeviceProfileC {
                rank: 1,
                vendor_id: 0x8086,
                device_id: 1,
                device_type: 1,
                vram_bytes: 8 << 30,
                compute_gflops: 100.0,
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
}
