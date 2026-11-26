//! Theorem 1: Bottleneck-TSP ring solver (`O(N^2 2^N)` Held-Karp DP for `N <= 12`,
//! greedy + 2-opt for `N > 12`) and Shared-Root All-Subsets Bottleneck-TSP DP
//! precomputing all `2^N - 1` subset bottlenecks and max-hop latencies in one sweep.

use crate::collective::backend::LinkInfo;

/// Precomputed exact bottleneck bandwidth (GB/s), maximum hop latency (microseconds),
/// and per-round Hockney cost (microseconds) for every non-empty device subset bitmask in `0..(1 << n)`.
#[derive(Debug, Clone)]
pub struct AllSubsetsRingTable {
    pub bottleneck_gbps: Vec<f64>,
    pub max_hop_latency_us: Vec<f64>,
    pub hockney_cost_us: Vec<f64>,
}

impl AllSubsetsRingTable {
    #[inline]
    pub fn bottleneck(&self, mask: usize) -> f64 {
        self.bottleneck_gbps.get(mask).copied().unwrap_or(1.0)
    }

    #[inline]
    pub fn latency_us(&self, mask: usize) -> f64 {
        self.max_hop_latency_us.get(mask).copied().unwrap_or(0.0)
    }

    #[inline]
    pub fn hockney_us(&self, mask: usize) -> f64 {
        self.hockney_cost_us.get(mask).copied().unwrap_or(0.0)
    }
}

#[inline]
fn edge_hockney_us(bw_gbps: f64, lat_us: f64, m: usize, sync_bytes: u64) -> f64 {
    let tx_us = if sync_bytes > 0 {
        (sync_bytes as f64) / ((m as f64) * bw_gbps.max(1e-9) * 1000.0)
    } else {
        0.0
    };
    lat_us + tx_us
}

/// Precomputes exact bottleneck bandwidth and maximum hop latency for all `2^n` subsets
/// in `O(n^2 2^n)` using a shared-root cycle dynamic program (bandwidth-first asymptotic limit).
///
/// Mirrors Java `BottleneckTspSolver.precomputeAllSubsetsRing` bit-for-bit.
pub fn precompute_all_subsets_ring(n: usize, bw: &[Vec<f64>], lat_us: &[Vec<f64>]) -> AllSubsetsRingTable {
    precompute_all_subsets_ring_impl(n, bw, lat_us, None)
}

/// Precomputes exact per-round Hockney bottleneck cycles across all `2^n` subsets
/// for a specific collective payload `sync_bytes` by partitioning cardinalities `m in 2..=n`
/// into equivalence classes sharing the same sorted edge ranking of `w_m(e) = lambda_e + D / (m * B_e)`.
pub fn precompute_all_subsets_ring_for_payload(
    n: usize,
    bw: &[Vec<f64>],
    lat_us: &[Vec<f64>],
    sync_bytes: u64,
) -> AllSubsetsRingTable {
    precompute_all_subsets_ring_impl(n, bw, lat_us, Some(sync_bytes))
}

fn precompute_all_subsets_ring_impl(
    n: usize,
    bw: &[Vec<f64>],
    lat_us: &[Vec<f64>],
    payload_bytes: Option<u64>,
) -> AllSubsetsRingTable {
    let num_subsets = 1usize << n;
    let mut bottleneck_gbps = vec![0.0f64; num_subsets];
    let mut max_hop_latency_us = vec![0.0f64; num_subsets];
    let mut hockney_cost_us = vec![0.0f64; num_subsets];

    // 1-element subsets
    for i in 0..n {
        bottleneck_gbps[1usize << i] = 25.0;
        max_hop_latency_us[1usize << i] = 0.0;
        hockney_cost_us[1usize << i] = 0.0;
    }

    // 2-element subsets (unique 2-cycle u <-> v)
    for u in 0..n {
        for v in (u + 1)..n {
            let mask = (1usize << u) | (1usize << v);
            match payload_bytes {
                Some(d) => {
                    let w_uv = edge_hockney_us(bw[u][v], lat_us[u][v], 2, d);
                    let w_vu = edge_hockney_us(bw[v][u], lat_us[v][u], 2, d);
                    if w_uv >= w_vu {
                        hockney_cost_us[mask] = w_uv;
                        bottleneck_gbps[mask] = bw[u][v];
                        max_hop_latency_us[mask] = lat_us[u][v];
                    } else {
                        hockney_cost_us[mask] = w_vu;
                        bottleneck_gbps[mask] = bw[v][u];
                        max_hop_latency_us[mask] = lat_us[v][u];
                    }
                }
                None => {
                    bottleneck_gbps[mask] = bw[u][v].min(bw[v][u]);
                    max_hop_latency_us[mask] = lat_us[u][v].max(lat_us[v][u]);
                }
            }
        }
    }

    if n < 3 {
        return AllSubsetsRingTable {
            bottleneck_gbps,
            max_hop_latency_us,
            hockney_cost_us,
        };
    }

    let Some(sync_d) = payload_bytes else {
        // Asymptotic bandwidth-first DP (D -> infinity limit)
        for s0 in 0..(n - 2) {
            let r = n - 1 - s0;
            let num_sub_masks = 1usize << r;
            let mut dp_bw = vec![-1.0f64; num_sub_masks * r];
            let mut dp_lat = vec![0.0f64; num_sub_masks * r];

            for u in 0..r {
                let target = s0 + 1 + u;
                let idx = (1usize << u) * r + u;
                dp_bw[idx] = bw[s0][target];
                dp_lat[idx] = lat_us[s0][target];
            }

            for sub_mask in 1usize..num_sub_masks {
                let row_off = sub_mask * r;
                for u in 0..r {
                    if (sub_mask & (1usize << u)) == 0 {
                        continue;
                    }
                    let cur_bw = dp_bw[row_off + u];
                    if cur_bw < 0.0 {
                        continue;
                    }
                    let cur_lat = dp_lat[row_off + u];
                    let u_rank = s0 + 1 + u;

                    for v in 0..r {
                        if (sub_mask & (1usize << v)) == 0 {
                            let next_mask = sub_mask | (1usize << v);
                            let next_idx = next_mask * r + v;
                            let v_rank = s0 + 1 + v;
                            let cand_bw = cur_bw.min(bw[u_rank][v_rank]);
                            let cand_lat = cur_lat.max(lat_us[u_rank][v_rank]);

                            let exist_bw = dp_bw[next_idx];
                            if cand_bw > exist_bw
                                || ((cand_bw - exist_bw).abs() < 1e-9 && cand_lat < dp_lat[next_idx])
                            {
                                dp_bw[next_idx] = cand_bw;
                                dp_lat[next_idx] = cand_lat;
                            }
                        }
                    }
                }

                if sub_mask.count_ones() >= 2 {
                    let global_mask = (1usize << s0) | (sub_mask << (s0 + 1));
                    let mut best_cycle_bw = -1.0f64;
                    let mut best_cycle_lat = f64::MAX;

                    for u in 0..r {
                        if (sub_mask & (1usize << u)) != 0 && dp_bw[row_off + u] >= 0.0 {
                            let u_rank = s0 + 1 + u;
                            let cycle_bw = dp_bw[row_off + u].min(bw[u_rank][s0]);
                            let cycle_lat = dp_lat[row_off + u].max(lat_us[u_rank][s0]);

                            if cycle_bw > best_cycle_bw
                                || ((cycle_bw - best_cycle_bw).abs() < 1e-9 && cycle_lat < best_cycle_lat)
                            {
                                best_cycle_bw = cycle_bw;
                                best_cycle_lat = cycle_lat;
                            }
                        }
                    }

                    if best_cycle_bw > 0.0 {
                        bottleneck_gbps[global_mask] = best_cycle_bw;
                        max_hop_latency_us[global_mask] = best_cycle_lat;
                    }
                }
            }
        }

        return AllSubsetsRingTable {
            bottleneck_gbps,
            max_hop_latency_us,
            hockney_cost_us,
        };
    };

    // Exact Hockney edge cost equivalence-class partitioning over m in 2..=n.
    struct EquivClass {
        rep_m: usize,
        card_mask: u32,
        edge_rank: Vec<u16>,
    }

    let mut classes: Vec<EquivClass> = Vec::new();
    let mut directed_edges: Vec<(usize, usize)> = Vec::with_capacity(n * (n - 1));
    for u in 0..n {
        for v in 0..n {
            if u != v {
                directed_edges.push((u, v));
            }
        }
    }

    for m in 2..=n {
        let mut sorted = directed_edges.clone();
        sorted.sort_by(|&(u1, v1), &(u2, v2)| {
            let w1 = edge_hockney_us(bw[u1][v1], lat_us[u1][v1], m, sync_d);
            let w2 = edge_hockney_us(bw[u2][v2], lat_us[u2][v2], m, sync_d);
            let diff = w1 - w2;
            let scale = w1.abs().max(w2.abs()).max(1e-9);
            if diff.abs() > 1e-12 * scale {
                w1.total_cmp(&w2)
            } else {
                bw[u2][v2]
                    .total_cmp(&bw[u1][v1])
                    .then_with(|| lat_us[u1][v1].total_cmp(&lat_us[u2][v2]))
                    .then_with(|| u1.cmp(&u2))
                    .then_with(|| v1.cmp(&v2))
            }
        });

        let mut rank_vec = vec![0u16; n * n];
        let mut cur_rank = 0u16;
        let mut prev_w: Option<f64> = None;
        let mut prev_bw = 0.0f64;
        let mut prev_lat = 0.0f64;
        for (u, v) in sorted {
            let w = edge_hockney_us(bw[u][v], lat_us[u][v], m, sync_d);
            if let Some(pw) = prev_w {
                let scale = w.abs().max(pw.abs()).max(1e-9);
                if (w - pw).abs() > 1e-12 * scale
                    || (bw[u][v] - prev_bw).abs() > 1e-9
                    || (lat_us[u][v] - prev_lat).abs() > 1e-9
                {
                    cur_rank += 1;
                }
            }
            rank_vec[u * n + v] = cur_rank;
            prev_w = Some(w);
            prev_bw = bw[u][v];
            prev_lat = lat_us[u][v];
        }

        if let Some(existing) = classes.iter_mut().find(|c| c.edge_rank == rank_vec) {
            existing.card_mask |= 1u32 << m;
        } else {
            classes.push(EquivClass {
                rep_m: m,
                card_mask: 1u32 << m,
                edge_rank: rank_vec,
            });
        }
    }

    for cls in classes {
        // Subsets of size 2 are already exact; skip DP if this class only contains m == 2.
        if (cls.card_mask & !0b111u32) == 0 {
            continue;
        }
        let rep_m = cls.rep_m;
        let er = &cls.edge_rank;

        for s0 in 0..(n - 2) {
            let r = n - 1 - s0;
            let num_sub_masks = 1usize << r;
            let mut dp_rank = vec![u16::MAX; num_sub_masks * r];
            let mut dp_bw = vec![0.0f64; num_sub_masks * r];
            let mut dp_lat = vec![0.0f64; num_sub_masks * r];

            for u in 0..r {
                let target = s0 + 1 + u;
                let idx = (1usize << u) * r + u;
                dp_rank[idx] = er[s0 * n + target];
                dp_bw[idx] = bw[s0][target];
                dp_lat[idx] = lat_us[s0][target];
            }

            for sub_mask in 1usize..num_sub_masks {
                let row_off = sub_mask * r;
                for u in 0..r {
                    if (sub_mask & (1usize << u)) == 0 {
                        continue;
                    }
                    let cur_rank = dp_rank[row_off + u];
                    if cur_rank == u16::MAX {
                        continue;
                    }
                    let cur_bw = dp_bw[row_off + u];
                    let cur_lat = dp_lat[row_off + u];
                    let u_rank = s0 + 1 + u;

                    for v in 0..r {
                        if (sub_mask & (1usize << v)) == 0 {
                            let next_mask = sub_mask | (1usize << v);
                            let next_idx = next_mask * r + v;
                            let v_rank = s0 + 1 + v;
                            let e_rank = er[u_rank * n + v_rank];
                            let (cand_rank, cand_bw, cand_lat) = if e_rank > cur_rank {
                                (e_rank, bw[u_rank][v_rank], lat_us[u_rank][v_rank])
                            } else {
                                (cur_rank, cur_bw, cur_lat)
                            };

                            let exist_rank = dp_rank[next_idx];
                            let better = if cand_rank != exist_rank {
                                cand_rank < exist_rank
                            } else {
                                let cw = edge_hockney_us(cand_bw, cand_lat, rep_m, sync_d);
                                let ew = edge_hockney_us(dp_bw[next_idx], dp_lat[next_idx], rep_m, sync_d);
                                cw < ew
                            };
                            if better {
                                dp_rank[next_idx] = cand_rank;
                                dp_bw[next_idx] = cand_bw;
                                dp_lat[next_idx] = cand_lat;
                            }
                        }
                    }
                }

                let m = 1 + sub_mask.count_ones() as usize;
                if m >= 3 && ((cls.card_mask >> m) & 1) != 0 {
                    let global_mask = (1usize << s0) | (sub_mask << (s0 + 1));
                    let mut best_rank = u16::MAX;
                    let mut best_bw = 0.0f64;
                    let mut best_lat = f64::MAX;

                    for u in 0..r {
                        if (sub_mask & (1usize << u)) != 0 {
                            let cur_rank = dp_rank[row_off + u];
                            if cur_rank == u16::MAX {
                                continue;
                            }
                            let u_rank = s0 + 1 + u;
                            let back_rank = er[u_rank * n + s0];
                            let (cyc_rank, cyc_bw, cyc_lat) = if back_rank > cur_rank {
                                (back_rank, bw[u_rank][s0], lat_us[u_rank][s0])
                            } else {
                                (cur_rank, dp_bw[row_off + u], dp_lat[row_off + u])
                            };

                            let better = if cyc_rank != best_rank {
                                cyc_rank < best_rank
                            } else {
                                let cw = edge_hockney_us(cyc_bw, cyc_lat, m, sync_d);
                                let bw_cost = edge_hockney_us(best_bw, best_lat, m, sync_d);
                                cw < bw_cost
                            };
                            if better {
                                best_rank = cyc_rank;
                                best_bw = cyc_bw;
                                best_lat = cyc_lat;
                            }
                        }
                    }

                    if best_rank != u16::MAX {
                        bottleneck_gbps[global_mask] = best_bw;
                        max_hop_latency_us[global_mask] = best_lat;
                        hockney_cost_us[global_mask] = edge_hockney_us(best_bw, best_lat, m, sync_d);
                    }
                }
            }
        }
    }

    AllSubsetsRingTable {
        bottleneck_gbps,
        max_hop_latency_us,
        hockney_cost_us,
    }
}

/// Scalable greedy Hockney-cost bottleneck-ring estimator for subsets of size > 12.
pub fn solve_ring_hockney_greedy(
    indices: &[usize],
    bw: &[Vec<f64>],
    lat_us: &[Vec<f64>],
    sync_bytes: u64,
) -> (f64, f64, f64) {
    let m = indices.len();
    if m == 0 {
        return (1.0, 0.0, 0.0);
    }
    if m == 1 {
        return (25.0, 0.0, 0.0);
    }
    if m == 2 {
        let u = indices[0];
        let v = indices[1];
        let w_uv = edge_hockney_us(bw[u][v], lat_us[u][v], 2, sync_bytes);
        let w_vu = edge_hockney_us(bw[v][u], lat_us[v][u], 2, sync_bytes);
        return if w_uv >= w_vu {
            (bw[u][v], lat_us[u][v], w_uv)
        } else {
            (bw[v][u], lat_us[v][u], w_vu)
        };
    }

    let mut visited = vec![false; m];
    visited[0] = true;
    let mut cur = 0usize;
    let mut worst_bw = 0.0f64;
    let mut worst_lat = 0.0f64;
    let mut worst_hockney = -1.0f64;

    for _ in 1..m {
        let mut best_next: Option<usize> = None;
        let mut best_w = f64::MAX;
        let cur_rank = indices[cur];
        for next in 0..m {
            if !visited[next] {
                let next_rank = indices[next];
                let w = edge_hockney_us(bw[cur_rank][next_rank], lat_us[cur_rank][next_rank], m, sync_bytes);
                if w < best_w {
                    best_w = w;
                    best_next = Some(next);
                }
            }
        }
        let next_idx = best_next.unwrap_or(0);
        let next_rank = indices[next_idx];
        visited[next_idx] = true;
        if best_w > worst_hockney {
            worst_hockney = best_w;
            worst_bw = bw[cur_rank][next_rank];
            worst_lat = lat_us[cur_rank][next_rank];
        }
        cur = next_idx;
    }

    let back_w = edge_hockney_us(bw[indices[cur]][indices[0]], lat_us[indices[cur]][indices[0]], m, sync_bytes);
    if back_w > worst_hockney {
        worst_hockney = back_w;
        worst_bw = bw[indices[cur]][indices[0]];
        worst_lat = lat_us[indices[cur]][indices[0]];
    }
    let eff_bw = if worst_bw > 0.0 { worst_bw } else { 1.0 };
    (eff_bw, worst_lat, worst_hockney.max(0.0))
}

/// Scalable greedy bottleneck-ring estimator for subsets of size > 12 (bandwidth-first).
pub fn solve_ring_bottleneck_greedy(indices: &[usize], bw: &[Vec<f64>]) -> f64 {
    let m = indices.len();
    if m == 0 {
        return 1.0;
    }
    if m == 1 {
        return 25.0;
    }
    if m == 2 {
        let u = indices[0];
        let v = indices[1];
        return bw[u][v].min(bw[v][u]);
    }

    let mut visited = vec![false; m];
    visited[0] = true;
    let mut cur = 0usize;
    let mut bottleneck = f64::MAX;

    for _ in 1..m {
        let mut best_next: Option<usize> = None;
        let mut best_w = -1.0f64;
        let cur_rank = indices[cur];
        for next in 0..m {
            if !visited[next] {
                let next_rank = indices[next];
                if bw[cur_rank][next_rank] > best_w {
                    best_w = bw[cur_rank][next_rank];
                    best_next = Some(next);
                }
            }
        }
        let next_idx = best_next.unwrap_or(0);
        visited[next_idx] = true;
        bottleneck = bottleneck.min(best_w);
        cur = next_idx;
    }
    bottleneck = bottleneck.min(bw[indices[cur]][indices[0]]);
    if bottleneck > 0.0 { bottleneck } else { 1.0 }
}

/// Helper to compare two candidate ring evaluations (bottleneck, total).
///
/// Maximizes bottleneck link bandwidth; tie-broken by total ring bandwidth
/// using a relative epsilon (`1e-6`) so nearly equal bottlenecks pick highest total.
pub(crate) fn is_better(cand_b: f64, cand_tot: f64, cur_b: f64, cur_tot: f64) -> bool {
    const REL_EPS: f64 = 1e-6;
    let diff = cand_b - cur_b;
    let scale = cand_b.max(cur_b).max(1.0);
    if diff > REL_EPS * scale {
        true
    } else if diff < -REL_EPS * scale {
        false
    } else {
        cand_tot > cur_tot
    }
}

/// Solves for the optimal ring order visiting all `ranks` GPUs.
///
/// Returns a permutation of `0..ranks` starting with `0`.
pub fn solve_optimal_ring(ranks: usize, links: &[LinkInfo]) -> Vec<u32> {
    if ranks == 0 {
        return Vec::new();
    }
    if ranks == 1 {
        return vec![0];
    }
    if ranks == 2 {
        return vec![0, 1];
    }

    let mut bw = vec![vec![0.0f64; ranks]; ranks];
    for link in links {
        let src = link.src_rank as usize;
        let dst = link.dst_rank as usize;
        if src < ranks && dst < ranks {
            bw[src][dst] = link.bandwidth_gbps;
        }
    }

    if ranks <= 12 {
        solve_ring_dp(ranks, &bw)
    } else {
        solve_ring_heuristic(ranks, &bw)
    }
}

/// Exact Held-Karp dynamic programming for Hamiltonian cycle on `N <= 12`.
fn solve_ring_dp(ranks: usize, bw: &[Vec<f64>]) -> Vec<u32> {
    let k = ranks - 1;
    let num_masks = 1usize << k;

    let mut dp = vec![vec![(-1.0f64, -1.0f64, 0u8); ranks]; num_masks];
    dp[0][0] = (f64::INFINITY, 0.0, 0);

    for mask in 0..num_masks {
        for u in 0..ranks {
            if mask == 0 && u != 0 {
                continue;
            }
            if mask != 0 && (u == 0 || (mask & (1 << (u - 1))) == 0) {
                continue;
            }

            let (cur_b, cur_tot, _) = dp[mask][u];
            if cur_b < 0.0 {
                continue;
            }

            for v in 1..ranks {
                if (mask & (1 << (v - 1))) == 0 {
                    let next_mask = mask | (1 << (v - 1));
                    let edge_bw = bw[u][v];
                    let cand_b = cur_b.min(edge_bw);
                    let cand_tot = cur_tot + edge_bw;

                    let (exist_b, exist_tot, _) = dp[next_mask][v];
                    if is_better(cand_b, cand_tot, exist_b, exist_tot) {
                        dp[next_mask][v] = (cand_b, cand_tot, u as u8);
                    }
                }
            }
        }
    }

    let full_mask = num_masks - 1;
    let mut best_last = 1;
    let mut best_b = -1.0;
    let mut best_tot = -1.0;

    for last in 1..ranks {
        let (b, tot, _) = dp[full_mask][last];
        if b < 0.0 {
            continue;
        }
        let edge_bw = bw[last][0];
        let cand_b = b.min(edge_bw);
        let cand_tot = tot + edge_bw;

        if is_better(cand_b, cand_tot, best_b, best_tot) {
            best_b = cand_b;
            best_tot = cand_tot;
            best_last = last;
        }
    }

    let mut curr = best_last;
    let mut cur_mask = full_mask;
    let mut rev_path = Vec::with_capacity(ranks);

    while curr != 0 {
        rev_path.push(curr as u32);
        let prev = dp[cur_mask][curr].2 as usize;
        cur_mask &= !(1 << (curr - 1));
        curr = prev;
    }
    rev_path.push(0);
    rev_path.reverse();

    rev_path
}

/// Scalable greedy nearest-neighbor + 2-opt heuristic for `N > 12`.
fn solve_ring_heuristic(ranks: usize, bw: &[Vec<f64>]) -> Vec<u32> {
    let mut visited = vec![false; ranks];
    visited[0] = true;
    let mut order = Vec::with_capacity(ranks);
    order.push(0u32);
    let mut cur = 0;

    for _ in 1..ranks {
        let mut best_next = None;
        let mut best_bw = -1.0;
        for next in 0..ranks {
            if !visited[next] {
                let w = bw[cur][next];
                if w > best_bw {
                    best_bw = w;
                    best_next = Some(next);
                }
            }
        }
        let next = best_next.unwrap_or(0);
        visited[next] = true;
        order.push(next as u32);
        cur = next;
    }

    let eval = |ord: &[u32]| -> (f64, f64) {
        let mut bot = f64::INFINITY;
        let mut tot = 0.0;
        for i in 0..ord.len() {
            let u = ord[i] as usize;
            let v = ord[(i + 1) % ord.len()] as usize;
            let w = bw[u][v];
            bot = bot.min(w);
            tot += w;
        }
        (bot, tot)
    };

    let (mut best_b, mut best_tot) = eval(&order);
    let mut improved = true;
    let mut passes = 0;

    while improved && passes < 100 {
        improved = false;
        passes += 1;
        for i in 1..ranks - 1 {
            for j in (i + 1)..ranks {
                order[i..=j].reverse();
                let (cand_b, cand_tot) = eval(&order);
                if is_better(cand_b, cand_tot, best_b, best_tot) {
                    best_b = cand_b;
                    best_tot = cand_tot;
                    improved = true;
                } else {
                    order[i..=j].reverse();
                }
            }
        }
    }

    order
}

#[cfg(test)]
mod tests {
    use super::precompute_all_subsets_ring;

    fn brute_force_subset(nodes: &[usize], bw: &[Vec<f64>], lat: &[Vec<f64>]) -> (f64, f64) {
        struct Search<'a> {
            s0: usize,
            bw: &'a [Vec<f64>],
            lat: &'a [Vec<f64>],
            best_bw: f64,
            best_lat: f64,
        }
        fn permute(ctx: &mut Search<'_>, cur: usize, rem: &mut Vec<usize>, cur_bw: f64, cur_lat: f64) {
            if rem.is_empty() {
                let b = cur_bw.min(ctx.bw[cur][ctx.s0]);
                let l = cur_lat.max(ctx.lat[cur][ctx.s0]);
                if b > ctx.best_bw || ((b - ctx.best_bw).abs() < 1e-9 && l < ctx.best_lat) {
                    ctx.best_bw = b;
                    ctx.best_lat = l;
                }
                return;
            }
            for i in 0..rem.len() {
                let next = rem.remove(i);
                let next_bw = cur_bw.min(ctx.bw[cur][next]);
                let next_lat = cur_lat.max(ctx.lat[cur][next]);
                permute(ctx, next, rem, next_bw, next_lat);
                rem.insert(i, next);
            }
        }

        if nodes.len() == 1 {
            return (25.0, 0.0);
        }
        if nodes.len() == 2 {
            let (u, v) = (nodes[0], nodes[1]);
            return (bw[u][v].min(bw[v][u]), lat[u][v].max(lat[v][u]));
        }
        let mut rem = nodes[1..].to_vec();
        let mut ctx = Search {
            s0: nodes[0],
            bw,
            lat,
            best_bw: -1.0,
            best_lat: f64::MAX,
        };
        permute(&mut ctx, nodes[0], &mut rem, f64::MAX, 0.0);
        (ctx.best_bw, ctx.best_lat)
    }

    #[test]
    fn shared_root_dp_matches_brute_force_on_all_subsets_n6() {
        let n = 6usize;
        let mut s = 0x1234_5678_9abc_def1u64;
        let mut next_f64 = || -> f64 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 32) as f64) / (u32::MAX as f64)
        };

        let mut bw = vec![vec![25.0; n]; n];
        let mut lat = vec![vec![0.0; n]; n];
        for u in 0..n {
            for v in 0..n {
                if u != v {
                    bw[u][v] = 1.0 + next_f64() * 24.0;
                    lat[u][v] = 2.0 + next_f64() * 98.0;
                }
            }
        }

        let table = precompute_all_subsets_ring(n, &bw, &lat);
        for mask in 1usize..(1 << n) {
            let nodes: Vec<usize> = (0..n).filter(|i| (mask & (1 << i)) != 0).collect();
            let (exp_bw, exp_lat) = brute_force_subset(&nodes, &bw, &lat);
            assert!((table.bottleneck(mask) - exp_bw).abs() < 1e-6, "mask={mask:#b}");
            assert!((table.latency_us(mask) - exp_lat).abs() < 1e-6, "mask={mask:#b}");
        }
    }

    #[test]
    fn shared_root_dp_n12_performance_smoke() {
        let n = 12usize;
        let mut bw = vec![vec![25.0; n]; n];
        let mut lat = vec![vec![0.0; n]; n];
        for u in 0..n {
            for v in 0..n {
                if u != v {
                    bw[u][v] = 4.0 + ((u * 7 + v * 3) % 20) as f64;
                    lat[u][v] = 5.0 + ((u + v) % 15) as f64;
                }
            }
        }
        let started = std::time::Instant::now();
        let table = precompute_all_subsets_ring(n, &bw, &lat);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        println!("Theorem 1 N=12 precompute_all_subsets_ring solve time: {elapsed_ms:.3} ms");
        assert!(table.bottleneck((1 << n) - 1) >= 4.0);
    }

    #[test]
    fn decode_small_payload_prefers_low_latency_over_high_bandwidth() {
        // Two edge-disjoint directed Hamiltonian cycles over 4 ranks:
        // Cycle A (0->1->2->3->0): 4 GB/s, 2 us latency (optimal for small decode sync_bytes)
        // Cycle B (0->3->2->1->0): 25 GB/s, 150 us latency (optimal for large prefill sync_bytes)
        let n = 4usize;
        let mut bw = vec![vec![0.5; n]; n];
        let mut lat = vec![vec![500.0; n]; n];
        for i in 0..n {
            bw[i][i] = 25.0;
            lat[i][i] = 0.0;
        }
        for &(u, v) in &[(0, 1), (1, 2), (2, 3), (3, 0)] {
            bw[u][v] = 4.0;
            lat[u][v] = 2.0;
        }
        for &(u, v) in &[(0, 3), (3, 2), (2, 1), (1, 0)] {
            bw[u][v] = 25.0;
            lat[u][v] = 150.0;
        }

        let tiny_table = super::precompute_all_subsets_ring_for_payload(n, &bw, &lat, 16 * 1024);
        let small_table = super::precompute_all_subsets_ring_for_payload(n, &bw, &lat, 4096);
        let large_table = super::precompute_all_subsets_ring_for_payload(n, &bw, &lat, 256 << 20);
        let full_mask = 0b1111;
        assert!((tiny_table.bottleneck(full_mask) - 4.0).abs() < 1e-9);
        assert!((tiny_table.latency_us(full_mask) - 2.0).abs() < 1e-9);
        assert!((small_table.bottleneck(full_mask) - 4.0).abs() < 1e-9);
        assert!((small_table.latency_us(full_mask) - 2.0).abs() < 1e-9);
        assert!((large_table.bottleneck(full_mask) - 25.0).abs() < 1e-9);
        assert!((large_table.latency_us(full_mask) - 150.0).abs() < 1e-9);
    }
}
