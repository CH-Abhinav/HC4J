//! Theorem 2: Zero-Parameter MST Dendrogram island detector & hierarchical
//! collective sync estimator (`pick_leader`, Kruskal single-linkage level cuts,
//! star intra-island + flat/DP inter-leader sync).

use std::collections::HashMap;

use super::ring_tsp::{AllSubsetsRingTable, solve_optimal_ring, solve_ring_bottleneck_greedy};
use crate::collective::backend::{
    DeviceProfileC, LinkInfo, TRANSPORT_HOST_SHARED_DMA, TRANSPORT_HOST_STAGED, TRANSPORT_PEER_TO_PEER,
};

/// A detected high-bandwidth cluster of ranks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IslandInfo {
    pub id: u32,
    pub leader_rank: u32,
    pub internal_transport: u32,
    pub members: Vec<u32>,
}

/// Estimates exact collective AllReduce sync latency in milliseconds by evaluating
/// both flat global ring and all `m - 1` hierarchical MST dendrogram level cuts.
///
/// Mirrors Java `WorkloadPlanner.estimateSyncMsExact` bit-for-bit.
pub(crate) fn estimate_sync_ms_exact(
    active_indices: &[usize],
    bw: &[Vec<f64>],
    lat_us: &[Vec<f64>],
    tr: &[Vec<u32>],
    ring_table: Option<&AllSubsetsRingTable>,
    devices: &[DeviceProfileC],
    sync_bytes: u64,
) -> f64 {
    let m = active_indices.len();
    if m <= 1 || sync_bytes == 0 {
        return 0.0;
    }

    let mut mask = 0usize;
    for &idx in active_indices {
        mask |= 1usize << idx;
    }

    let ring_bw = match ring_table {
        Some(t) => t.bottleneck(mask),
        None => solve_ring_bottleneck_greedy(active_indices, bw),
    };
    let ring_lat_secs = match ring_table {
        Some(t) => t.latency_us(mask) / 1e6,
        None => 0.0,
    };
    let ring_bw = if ring_bw <= 0.0 { 1e-6 } else { ring_bw };

    let m_f = m as f64;
    let sync_f = sync_bytes as f64;
    let t_flat = 2.0 * (m_f - 1.0) * ring_lat_secs + 2.0 * ((m_f - 1.0) / m_f) * (sync_f / (ring_bw * 1e9));
    let mut best_time_secs = t_flat;

    #[derive(Clone, Copy)]
    struct UndirectedEdge {
        u: usize,
        v: usize,
        tr_sym: u32,
        w_sym: f64,
    }

    let mut all_edges_uniform = true;
    let mut first_tr: Option<u32> = None;
    let mut first_w = -1.0f64;

    let mut edges = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        for j in (i + 1)..m {
            let u = active_indices[i];
            let v = active_indices[j];
            let w_sym = bw[u][v].min(bw[v][u]);
            let tr_sym = transport_rank(tr[u][v]).min(transport_rank(tr[v][u]));
            edges.push(UndirectedEdge { u, v, tr_sym, w_sym });
            if let Some(ft) = first_tr {
                if tr_sym != ft || (w_sym - first_w).abs() > 1e-6 {
                    all_edges_uniform = false;
                }
            } else {
                first_tr = Some(tr_sym);
                first_w = w_sym;
            }
        }
    }

    if all_edges_uniform {
        return t_flat * 1000.0;
    }

    edges.sort_by(|a, b| {
        b.tr_sym
            .cmp(&a.tr_sym)
            .then_with(|| b.w_sym.total_cmp(&a.w_sym))
    });

    let max_idx = active_indices.iter().copied().max().unwrap_or(0);
    let mut dsu = Dsu::new(max_idx + 1);
    let mut mst_edges = Vec::with_capacity(m - 1);
    for e in &edges {
        if dsu.find(e.u) != dsu.find(e.v) {
            dsu.union(e.u, e.v);
            mst_edges.push(*e);
            if mst_edges.len() == m - 1 {
                break;
            }
        }
    }

    mst_edges.sort_by(|a, b| {
        a.tr_sym
            .cmp(&b.tr_sym)
            .then_with(|| a.w_sym.total_cmp(&b.w_sym))
    });

    for k in 1..m {
        let mut dsu_cut = Dsu::new(max_idx + 1);
        for e in &mst_edges[k..] {
            dsu_cut.union(e.u, e.v);
        }

        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for &idx in active_indices {
            let root = dsu_cut.find(idx);
            groups.entry(root).or_default().push(idx);
        }

        let mut islands: Vec<Vec<usize>> = groups.into_values().collect();
        for isl in &mut islands {
            isl.sort_unstable();
        }
        islands.sort_by_key(|isl| isl[0]);

        let mut leaders = Vec::with_capacity(islands.len());
        let mut t_intra_max = 0.0f64;

        for isl in &islands {
            let leader = pick_mst_leader(isl, active_indices, devices, bw);
            leaders.push(leader);
            if isl.len() > 1 {
                let mut b_intra = f64::MAX;
                let mut lat_intra_secs = 0.0f64;
                for &member in isl {
                    if member != leader {
                        let min_b = bw[member][leader].min(bw[leader][member]);
                        let max_l = lat_us[member][leader].max(lat_us[leader][member]) / 1e6;
                        b_intra = b_intra.min(min_b);
                        lat_intra_secs = lat_intra_secs.max(max_l);
                    }
                }
                if b_intra <= 0.0 {
                    b_intra = 1e-6;
                }
                let sz_f = isl.len() as f64;
                let t_intra =
                    2.0 * (sz_f - 1.0) * lat_intra_secs + 2.0 * ((sz_f - 1.0) / sz_f) * (sync_f / (b_intra * 1e9));
                t_intra_max = t_intra_max.max(t_intra);
            }
        }

        let mut leader_mask = 0usize;
        for &l in &leaders {
            leader_mask |= 1usize << l;
        }
        let b_lead = match ring_table {
            Some(t) => t.bottleneck(leader_mask),
            None => solve_ring_bottleneck_greedy(&leaders, bw),
        };
        let lat_lead_secs = match ring_table {
            Some(t) => t.latency_us(leader_mask) / 1e6,
            None => 0.0,
        };
        let b_lead = if b_lead <= 0.0 { 1e-6 } else { b_lead };

        let k_f = leaders.len() as f64;
        let t_lead = 2.0 * (k_f - 1.0) * lat_lead_secs + 2.0 * ((k_f - 1.0) / k_f) * (sync_f / (b_lead * 1e9));
        let t_hier = t_intra_max + t_lead;
        if t_hier < best_time_secs * (1.0 - 1e-6) {
            best_time_secs = t_hier;
        }
    }

    best_time_secs * 1000.0
}

fn pick_mst_leader(members: &[usize], active_indices: &[usize], devices: &[DeviceProfileC], bw: &[Vec<f64>]) -> usize {
    let mut best = members[0];
    let mut best_gflops = -1.0f64;
    let mut best_mean_out = -1.0f64;

    for &m in members {
        let gflops = devices.get(m).map_or(0.0, |d| d.compute_gflops);
        let mut out_sum = 0.0f64;
        let mut out_count = 0usize;
        for &o in active_indices {
            if !members.contains(&o) {
                out_sum += bw[m][o];
                out_count += 1;
            }
        }
        let mean_out = if out_count > 0 {
            out_sum / out_count as f64
        } else {
            0.0
        };

        let better = if (gflops - best_gflops).abs() > 1e-4 {
            gflops > best_gflops
        } else if (mean_out - best_mean_out).abs() > 1e-4 {
            mean_out > best_mean_out
        } else {
            m < best
        };

        if better {
            best = m;
            best_gflops = gflops;
            best_mean_out = mean_out;
        }
    }
    best
}

/// Capability ordering rank: HOST_STAGED (0) < HOST_SHARED_DMA (1) < PEER_TO_PEER (2).
fn transport_rank(t: u32) -> u32 {
    match t {
        TRANSPORT_HOST_STAGED => 0,
        TRANSPORT_HOST_SHARED_DMA => 1,
        TRANSPORT_PEER_TO_PEER => 2,
        _ => 0,
    }
}

/// Disjoint Set Union (Union-Find) for clustering.
struct Dsu {
    parent: Vec<usize>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, i: usize) -> usize {
        if self.parent[i] == i {
            i
        } else {
            let p = self.parent[i];
            self.parent[i] = self.find(p);
            self.parent[i]
        }
    }

    fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            self.parent[root_i] = root_j;
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn pick_leader(
    members: &[u32],
    profiles: &[DeviceProfileC],
    bw: &[Vec<f64>],
    ranks: usize,
) -> u32 {
    let mut best_leader = members[0];
    let mut best_gflops = -1.0;
    let mut best_mean_out = -1.0;

    for &m in members {
        let m_idx = m as usize;
        let gflops = profiles
            .iter()
            .find(|p| p.rank == m)
            .map(|p| p.compute_gflops)
            .unwrap_or(0.0);

        let mut out_bw_sum = 0.0;
        let mut out_count = 0;
        for other in 0..ranks {
            if !members.contains(&(other as u32)) {
                out_bw_sum += bw[m_idx][other];
                out_count += 1;
            }
        }
        let mean_out = if out_count > 0 {
            out_bw_sum / out_count as f64
        } else {
            0.0
        };

        let is_cand_better = if (gflops - best_gflops).abs() > 1e-4 {
            gflops > best_gflops
        } else if (mean_out - best_mean_out).abs() > 1e-4 {
            mean_out > best_mean_out
        } else {
            m < best_leader
        };

        if is_cand_better {
            best_leader = m;
            best_gflops = gflops;
            best_mean_out = mean_out;
        }
    }
    best_leader
}

/// Detects high-speed clusters (islands) purely from measured edge weights and transports
/// using Theorem 2 Zero-Parameter MST Dendrogram level cuts for a 64 MiB payload.
pub fn detect_islands(
    ranks: usize,
    profiles: &[DeviceProfileC],
    links: &[LinkInfo],
) -> Vec<IslandInfo> {
    detect_islands_for_payload(ranks, profiles, links, 64 << 20)
}

/// Detects high-speed clusters (islands) purely from measured edge weights, transports, and
/// latencies using Theorem 2 Zero-Parameter MST Dendrogram level cuts for an arbitrary AllReduce payload.
#[allow(clippy::needless_range_loop)]
pub fn detect_islands_for_payload(
    ranks: usize,
    profiles: &[DeviceProfileC],
    links: &[LinkInfo],
    sync_bytes: u64,
) -> Vec<IslandInfo> {
    if ranks == 0 {
        return Vec::new();
    }
    if ranks == 1 {
        return vec![IslandInfo {
            id: 0,
            leader_rank: 0,
            internal_transport: TRANSPORT_PEER_TO_PEER,
            members: vec![0],
        }];
    }

    // Build directed matrices
    let mut bw = vec![vec![0.0f64; ranks]; ranks];
    let mut lat_us = vec![vec![100.0f64; ranks]; ranks];
    let mut tr = vec![vec![TRANSPORT_HOST_STAGED; ranks]; ranks];
    for link in links {
        let src = link.src_rank as usize;
        let dst = link.dst_rank as usize;
        if src < ranks && dst < ranks {
            bw[src][dst] = link.bandwidth_gbps;
            lat_us[src][dst] = link.latency_us;
            tr[src][dst] = link.transport;
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct MstEdge {
        u: usize,
        v: usize,
        tr_sym: u32,
        w_sym: f64,
    }

    let mut edges = Vec::with_capacity(ranks * (ranks - 1) / 2);
    for u in 0..ranks {
        for v in (u + 1)..ranks {
            let w_sym = bw[u][v].min(bw[v][u]);
            let tr_sym = transport_rank(tr[u][v]).min(transport_rank(tr[v][u]));
            edges.push(MstEdge { u, v, tr_sym, w_sym });
        }
    }

    // Sort undirected edges descending by (tr_sym, w_sym)
    edges.sort_by(|a, b| {
        b.tr_sym
            .cmp(&a.tr_sym)
            .then_with(|| b.w_sym.partial_cmp(&a.w_sym).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut mst_dsu = Dsu::new(ranks);
    let mut mst_edges = Vec::with_capacity(ranks - 1);
    for e in edges {
        if mst_dsu.find(e.u) != mst_dsu.find(e.v) {
            mst_dsu.union(e.u, e.v);
            mst_edges.push(e);
            if mst_edges.len() == ranks - 1 {
                break;
            }
        }
    }

    // Sort the m - 1 MST edges ascending by (tr_sym, w_sym): e_0, ..., e_{m-2}
    mst_edges.sort_by(|a, b| {
        a.tr_sym
            .cmp(&b.tr_sym)
            .then_with(|| a.w_sym.partial_cmp(&b.w_sym).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Evaluate all m dendrogram cuts k in 0..ranks
    let mut best_k = 0;
    let mut best_time = f64::INFINITY;

    for k in 0..ranks {
        if k == 0 {
            // Flat ring over all ranks
            let ring = solve_optimal_ring(ranks, links);
            let mut b_ring = f64::INFINITY;
            let mut lat_ring_secs = 0.0f64;
            for i in 0..ring.len() {
                let u = ring[i] as usize;
                let v = ring[(i + 1) % ring.len()] as usize;
                b_ring = b_ring.min(bw[u][v]);
                lat_ring_secs = lat_ring_secs.max(lat_us[u][v] / 1e6);
            }
            if b_ring <= 0.0 {
                b_ring = 1e-6;
            }
            let m_f = ranks as f64;
            let t_flat = 2.0 * (m_f - 1.0) * lat_ring_secs
                + 2.0 * ((m_f - 1.0) / m_f) * (sync_bytes as f64 / (b_ring * 1e9));

            best_k = 0;
            best_time = t_flat;
        } else {
            // Cut k removes e_0..e_{k-1}, keeps e_k..e_{m-2}
            let mut dsu = Dsu::new(ranks);
            for e in &mst_edges[k..] {
                dsu.union(e.u, e.v);
            }

            use std::collections::BTreeMap;
            let mut groups: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
            for u in 0..ranks {
                let root = dsu.find(u);
                groups.entry(root).or_default().push(u as u32);
            }

            let mut island_members: Vec<Vec<u32>> = groups.into_values().collect();
            for m in &mut island_members {
                m.sort_unstable();
            }
            island_members.sort_by_key(|m| m[0]);

            let mut leaders = Vec::with_capacity(island_members.len());
            let mut t_intra_max = 0.0f64;
            for members in &island_members {
                let leader = pick_leader(members, profiles, &bw, ranks);
                leaders.push(leader);
                if members.len() > 1 {
                    let mut b_intra = f64::INFINITY;
                    let mut lat_intra_secs = 0.0f64;
                    for &m in members {
                        if m != leader {
                            let u = m as usize;
                            let l = leader as usize;
                            let min_bw = bw[u][l].min(bw[l][u]);
                            let max_lat = lat_us[u][l].max(lat_us[l][u]) / 1e6;
                            b_intra = b_intra.min(min_bw);
                            lat_intra_secs = lat_intra_secs.max(max_lat);
                        }
                    }
                    if b_intra <= 0.0 {
                        b_intra = 1e-6;
                    }
                    let sz_f = members.len() as f64;
                    let t_intra = 2.0 * (sz_f - 1.0) * lat_intra_secs
                        + 2.0 * ((sz_f - 1.0) / sz_f) * (sync_bytes as f64 / (b_intra * 1e9));
                    t_intra_max = t_intra_max.max(t_intra);
                }
            }

            let num_leaders = leaders.len();
            let mut leader_links = Vec::with_capacity(num_leaders * (num_leaders - 1));
            for (i, &li) in leaders.iter().enumerate() {
                for (j, &lj) in leaders.iter().enumerate() {
                    if i != j {
                        leader_links.push(LinkInfo {
                            src_rank: i as u32,
                            dst_rank: j as u32,
                            bandwidth_gbps: bw[li as usize][lj as usize],
                            transport: tr[li as usize][lj as usize],
                            loopback: if li == lj { 1 } else { 0 },
                            latency_us: lat_us[li as usize][lj as usize],
                        });
                    }
                }
            }
            let leader_ring_idx = solve_optimal_ring(num_leaders, &leader_links);
            let mut b_lead = f64::INFINITY;
            let mut lat_lead_secs = 0.0f64;
            for i in 0..leader_ring_idx.len() {
                let idx_u = leader_ring_idx[i] as usize;
                let idx_v = leader_ring_idx[(i + 1) % leader_ring_idx.len()] as usize;
                let u = leaders[idx_u] as usize;
                let v = leaders[idx_v] as usize;
                b_lead = b_lead.min(bw[u][v]);
                lat_lead_secs = lat_lead_secs.max(lat_us[u][v] / 1e6);
            }
            if b_lead <= 0.0 {
                b_lead = 1e-6;
            }
            let k_f = num_leaders as f64;
            let t_lead = 2.0 * (k_f - 1.0) * lat_lead_secs
                + 2.0 * ((k_f - 1.0) / k_f) * (sync_bytes as f64 / (b_lead * 1e9));

            let t_hier = t_intra_max + t_lead;
            if t_hier < best_time * (1.0 - 1e-6) {
                best_time = t_hier;
                best_k = k;
            }
        }
    }

    // Reconstruct islands for winning best_k
    let island_members: Vec<Vec<u32>> = if best_k == 0 {
        vec![(0..ranks as u32).collect()]
    } else {
        let mut dsu = Dsu::new(ranks);
        for e in &mst_edges[best_k..] {
            dsu.union(e.u, e.v);
        }
        use std::collections::BTreeMap;
        let mut groups: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for u in 0..ranks {
            let root = dsu.find(u);
            groups.entry(root).or_default().push(u as u32);
        }
        let mut members_vec: Vec<Vec<u32>> = groups.into_values().collect();
        for m in &mut members_vec {
            m.sort_unstable();
        }
        members_vec.sort_by_key(|m| m[0]);
        members_vec
    };

    island_members
        .into_iter()
        .enumerate()
        .map(|(id, members)| {
            let best_leader = pick_leader(&members, profiles, &bw, ranks);
            let internal_transport = if members.len() == 1 {
                let m = members[0] as usize;
                let mut best_out_tr = TRANSPORT_PEER_TO_PEER;
                let mut best_out_rank = 0;
                for other in 0..ranks {
                    if other != m {
                        let rank = transport_rank(tr[m][other]);
                        if rank > best_out_rank {
                            best_out_rank = rank;
                            best_out_tr = tr[m][other];
                        }
                    }
                }
                best_out_tr
            } else {
                let mut min_tr = TRANSPORT_PEER_TO_PEER;
                let mut min_rank = u32::MAX;
                for &u in &members {
                    for &v in &members {
                        if u != v {
                            let t = tr[u as usize][v as usize];
                            let rank = transport_rank(t);
                            if rank < min_rank {
                                min_rank = rank;
                                min_tr = t;
                            }
                        }
                    }
                }
                min_tr
            };

            IslandInfo {
                id: id as u32,
                leader_rank: best_leader,
                internal_transport,
                members,
            }
        })
        .collect()
}

/// Balanced chunk partitioning for arbitrary lengths and rank counts.
///
/// Divides `total` elements across `ranks` into balanced, non-overlapping contiguous slices.
/// The first `total % ranks` chunks have length `total / ranks + 1`, and the remaining chunks
/// have length `total / ranks`.
pub fn chunk_range(total: usize, ranks: usize, chunk_idx: usize) -> (usize, usize) {
    if ranks == 0 || chunk_idx >= ranks {
        return (0, 0);
    }
    let base = total / ranks;
    let rem = total % ranks;
    let len = base + usize::from(chunk_idx < rem);
    let start = chunk_idx * base + chunk_idx.min(rem);
    (start, len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_link(src: u32, dst: u32, bw: f64, tr: u32) -> LinkInfo {
        LinkInfo {
            src_rank: src,
            dst_rank: dst,
            bandwidth_gbps: bw,
            transport: tr,
            loopback: if src == dst { 1 } else { 0 },
            latency_us: 10.0,
        }
    }

    #[test]
    fn test_homogeneous_zero_false_split() {
        // 4 GPUs with slightly jittered links {24.8, 25.0, 25.1, 25.2} GB/s
        let ranks = 4;
        let profiles = vec![
            make_profile(0, 1000.0),
            make_profile(1, 1000.0),
            make_profile(2, 1000.0),
            make_profile(3, 1000.0),
        ];

        let jitter = [
            [0.0, 24.8, 25.0, 25.1],
            [24.8, 0.0, 25.2, 25.0],
            [25.0, 25.2, 0.0, 24.9],
            [25.1, 25.0, 24.9, 0.0],
        ];

        let mut links = Vec::new();
        for i in 0..ranks as u32 {
            for j in 0..ranks as u32 {
                if i != j {
                    links.push(make_link(i, j, jitter[i as usize][j as usize], TRANSPORT_PEER_TO_PEER));
                }
            }
        }

        let islands = detect_islands(ranks, &profiles, &links);
        assert_eq!(islands.len(), 1, "homogeneous cluster MUST NOT split into multiple islands! Got {:?}", islands);
        assert_eq!(islands[0].members, vec![0, 1, 2, 3]);
    }

    fn make_profile(rank: u32, gflops: f64) -> DeviceProfileC {
        DeviceProfileC {
            rank,
            vendor_id: 0,
            device_id: 0,
            device_type: 2, // discrete
            vram_bytes: 8 * 1024 * 1024 * 1024,
            compute_gflops: gflops,
        }
    }

    #[test]
    fn test_mixed_12_gpu_cluster() {
        // 5 NVIDIA (0..5), 4 AMD (5..9), 3 Intel (9..12)
        // Intra-NVIDIA: 25 GB/s P2P
        // Intra-AMD: 22 GB/s P2P
        // Intra-Intel: 18 GB/s P2P
        // Cross-vendor: 2.0 GB/s HOST_SHARED_DMA
        let ranks = 12;
        let mut links = Vec::new();
        let mut profiles = Vec::new();

        for i in 0..ranks as u32 {
            let gflops = if i < 5 { 10000.0 } else if i < 9 { 8000.0 } else { 2000.0 };
            profiles.push(make_profile(i, gflops));
        }

        let vendor_group = |r: u32| -> u32 {
            if r < 5 { 0 } else if r < 9 { 1 } else { 2 }
        };

        for i in 0..ranks as u32 {
            for j in 0..ranks as u32 {
                if i != j {
                    if vendor_group(i) == vendor_group(j) {
                        let bw = match vendor_group(i) {
                            0 => 25.0,
                            1 => 22.0,
                            _ => 18.0,
                        };
                        links.push(make_link(i, j, bw, TRANSPORT_PEER_TO_PEER));
                    } else {
                        links.push(make_link(i, j, 2.0, TRANSPORT_HOST_SHARED_DMA));
                    }
                }
            }
        }

        let islands = detect_islands(ranks, &profiles, &links);
        assert_eq!(islands.len(), 3, "expected 3 islands");
        assert_eq!(islands[0].members.len(), 5);
        assert_eq!(islands[1].members.len(), 4);
        assert_eq!(islands[2].members.len(), 3);

        let ring = solve_optimal_ring(ranks, &links);
        assert_eq!(ring.len(), ranks);

        // Count cross-vendor hops in the ring
        let mut cross_vendor_hops = 0;
        for k in 0..ranks {
            let u = ring[k];
            let v = ring[(k + 1) % ranks];
            if vendor_group(u) != vendor_group(v) {
                cross_vendor_hops += 1;
            }
        }
        // An optimal ring grouping each vendor contiguous has exactly 3 cross-vendor hops!
        assert_eq!(cross_vendor_hops, 3, "optimal ring should have exactly 3 cross-vendor hops, got {cross_vendor_hops}");
    }

    #[test]
    fn test_4_rank_heterogeneous_intel_lavapipe() {
        // [Intel0, Lvp1, Lvp2, Intel3]
        // 0 <-> 3: Tier 1 P2P (15 GB/s)
        // 1 <-> 2: 4.5 GB/s HOST_SHARED_DMA
        // Intel <-> Lavapipe: 1.8 GB/s HOST_SHARED_DMA
        let ranks = 4;
        let profiles = vec![
            make_profile(0, 185.0),
            make_profile(1, 50.0),
            make_profile(2, 50.0),
            make_profile(3, 185.0),
        ];

        let mut links = Vec::new();
        for i in 0..ranks as u32 {
            for j in 0..ranks as u32 {
                if i != j {
                    if (i == 0 && j == 3) || (i == 3 && j == 0) {
                        links.push(make_link(i, j, 15.0, TRANSPORT_PEER_TO_PEER));
                    } else if (i == 1 && j == 2) || (i == 2 && j == 1) {
                        links.push(make_link(i, j, 4.5, TRANSPORT_HOST_SHARED_DMA));
                    } else {
                        links.push(make_link(i, j, 1.8, TRANSPORT_HOST_SHARED_DMA));
                    }
                }
            }
        }

        let islands = detect_islands(ranks, &profiles, &links);
        assert_eq!(islands.len(), 2, "expected 2 islands");
        assert_eq!(islands[0].members, vec![0, 3]);
        assert_eq!(islands[0].internal_transport, TRANSPORT_PEER_TO_PEER);
        assert_eq!(islands[1].members, vec![1, 2]);
        assert_eq!(islands[1].internal_transport, TRANSPORT_HOST_SHARED_DMA);

        let ring = solve_optimal_ring(ranks, &links);
        assert_eq!(ring.len(), 4);
        // 0 must be adjacent to 3, and 1 must be adjacent to 2
        let pos = |r: u32| ring.iter().position(|&x| x == r).unwrap();
        let adj = |a: u32, b: u32| {
            let p1 = pos(a);
            let p2 = pos(b);
            (p1 + 1) % 4 == p2 || (p2 + 1) % 4 == p1
        };
        assert!(adj(0, 3), "0 and 3 must be adjacent in ring: {ring:?}");
        assert!(adj(1, 2), "1 and 2 must be adjacent in ring: {ring:?}");
    }

    #[test]
    fn test_uniform_4_rank_p2p_loopback() {
        let ranks = 4;
        let profiles = vec![
            make_profile(0, 1000.0),
            make_profile(1, 1000.0),
            make_profile(2, 1000.0),
            make_profile(3, 1000.0),
        ];

        let mut links = Vec::new();
        for i in 0..ranks as u32 {
            for j in 0..ranks as u32 {
                if i != j {
                    links.push(make_link(i, j, 28.0, TRANSPORT_PEER_TO_PEER));
                }
            }
        }

        let islands = detect_islands(ranks, &profiles, &links);
        assert_eq!(islands.len(), 1, "expected 1 uniform island");
        assert_eq!(islands[0].members, vec![0, 1, 2, 3]);
        assert_eq!(islands[0].internal_transport, TRANSPORT_PEER_TO_PEER);
    }

    #[test]
    fn test_large_cluster_n16_heuristic() {
        let ranks = 16;
        let mut links = Vec::new();
        for i in 0..ranks as u32 {
            for j in 0..ranks as u32 {
                if i != j {
                    // Non-uniform bandwidth pattern
                    let bw = if (i as i32 - j as i32).abs() == 1 { 20.0 } else { 5.0 };
                    links.push(make_link(i, j, bw, TRANSPORT_PEER_TO_PEER));
                }
            }
        }

        let ring = solve_optimal_ring(ranks, &links);
        assert_eq!(ring.len(), 16);
        let mut sorted = ring.clone();
        sorted.sort_unstable();
        let expected: Vec<u32> = (0..16).collect();
        assert_eq!(sorted, expected, "heuristic ring must be a valid permutation of 0..16");
    }

    #[test]
    fn theorem2_hierarchical_dendrogram_beats_flat_ring_across_slow_bridge() {
        use super::estimate_sync_ms_exact;
        use super::super::ring_tsp::precompute_all_subsets_ring;

        // Island A: ranks {0, 1} connected by 25 GB/s P2P; Island B: ranks {2, 3} connected by 25 GB/s P2P;
        // Cross-island bridge: 2 GB/s HOST_STAGED.
        let devices = vec![
            make_profile(0, 10_000.0),
            make_profile(1, 9_000.0),
            make_profile(2, 10_000.0),
            make_profile(3, 9_000.0),
        ];
        let mut bw = vec![vec![25.0; 4]; 4];
        let mut lat = vec![vec![0.0; 4]; 4];
        let mut tr = vec![vec![TRANSPORT_PEER_TO_PEER; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    continue;
                }
                let same_island = (i < 2 && j < 2) || (i >= 2 && j >= 2);
                if same_island {
                    bw[i][j] = 25.0;
                    lat[i][j] = 2.0;
                    tr[i][j] = TRANSPORT_PEER_TO_PEER;
                } else {
                    bw[i][j] = 2.0;
                    lat[i][j] = 100.0;
                    tr[i][j] = TRANSPORT_HOST_STAGED;
                }
            }
        }
        let table = precompute_all_subsets_ring(4, &bw, &lat);
        let active = [0, 1, 2, 3];
        let sync_bytes = 64u64 << 20;
        let ms = estimate_sync_ms_exact(&active, &bw, &lat, &tr, Some(&table), &devices, sync_bytes);
        // Flat ring across 2 GB/s bridge with 4 ranks: 2*(3/4)*(64MB / 2GB/s) = 48ms;
        // Hierarchical (intra 25 GB/s + inter 2 GB/s across 2 leaders): 2*(1/2)*(64MB/25GB/s + 64MB/2GB/s) ~= 34.56ms!
        assert!(ms < 40.0, "expected hierarchical dendrogram sync < 40ms, got {ms}");
    }
}
