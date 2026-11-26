//! Theorem 2: Zero-Parameter MST Dendrogram island detector & hierarchical
//! collective sync estimator (`pick_leader`, Kruskal single-linkage level cuts,
//! star intra-island + flat/DP inter-leader sync).

use std::collections::HashMap;

use super::ring_tsp::{
    AllSubsetsRingTable, precompute_all_subsets_ring_for_payload, solve_optimal_ring, solve_ring_hockney_greedy,
};
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

#[inline]
fn ring_allreduce_time_secs(
    indices: &[usize],
    bw: &[Vec<f64>],
    lat_us: &[Vec<f64>],
    ring_table: Option<&AllSubsetsRingTable>,
    sync_bytes: u64,
) -> f64 {
    let m = indices.len();
    if m <= 1 || sync_bytes == 0 {
        return 0.0;
    }
    let m_f = m as f64;
    if let Some(t) = ring_table {
        let mut mask = 0usize;
        for &idx in indices {
            mask |= 1usize << idx;
        }
        let h_us = t.hockney_us(mask);
        if h_us > 0.0 {
            return 2.0 * (m_f - 1.0) * (h_us / 1e6);
        }
        let ring_bw = t.bottleneck(mask).max(1e-6);
        let ring_lat_secs = t.latency_us(mask) / 1e6;
        return 2.0 * (m_f - 1.0) * ring_lat_secs
            + 2.0 * ((m_f - 1.0) / m_f) * ((sync_bytes as f64) / (ring_bw * 1e9));
    }
    let (_, _, hockney_us) = solve_ring_hockney_greedy(indices, bw, lat_us, sync_bytes);
    2.0 * (m_f - 1.0) * (hockney_us / 1e6)
}

/// Estimates exact collective AllReduce sync latency in milliseconds by evaluating
/// both flat global ring and all `m - 1` hierarchical MST dendrogram level cuts.
///
/// Mirrors Java `MstDendrogram.estimateSyncMsExact` bit-for-bit.
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

    let t_flat = ring_allreduce_time_secs(active_indices, bw, lat_us, ring_table, sync_bytes);
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
            let leader = pick_mst_leader(isl, active_indices, devices, bw, lat_us, sync_bytes);
            leaders.push(leader);
            if isl.len() > 1 {
                let t_intra = ring_allreduce_time_secs(isl, bw, lat_us, ring_table, sync_bytes);
                t_intra_max = t_intra_max.max(t_intra);
            }
        }

        let t_lead = ring_allreduce_time_secs(&leaders, bw, lat_us, ring_table, sync_bytes);
        let t_hier = t_intra_max + t_lead;
        if t_hier < best_time_secs * (1.0 - 1e-6) {
            best_time_secs = t_hier;
        }
    }

    best_time_secs * 1000.0
}

fn pick_mst_leader(
    members: &[usize],
    active_indices: &[usize],
    devices: &[DeviceProfileC],
    bw: &[Vec<f64>],
    lat_us: &[Vec<f64>],
    sync_bytes: u64,
) -> usize {
    let mut best = members[0];
    let mut best_worst_out_t = f64::INFINITY;
    let mut best_gflops = -1.0f64;

    for &m in members {
        let gflops = devices.get(m).map_or(0.0, |d| d.compute_gflops);
        let mut worst_out_t = 0.0f64;
        let mut has_outbound = false;
        for &o in active_indices {
            if !members.contains(&o) {
                has_outbound = true;
                let link_t = (lat_us[m][o] / 1e6) + (sync_bytes as f64) / (bw[m][o].max(1e-6) * 1e9);
                worst_out_t = worst_out_t.max(link_t);
            }
        }
        if !has_outbound {
            worst_out_t = 0.0;
        }

        let scale = worst_out_t.max(best_worst_out_t).max(1e-9);
        let diff = worst_out_t - best_worst_out_t;
        let better = if diff < -1e-6 * scale {
            true
        } else if diff > 1e-6 * scale {
            false
        } else if (gflops - best_gflops).abs() > 1e-4 {
            gflops > best_gflops
        } else {
            m < best
        };

        if better {
            best = m;
            best_worst_out_t = worst_out_t;
            best_gflops = gflops;
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
    lat_us: &[Vec<f64>],
    ranks: usize,
    sync_bytes: u64,
) -> u32 {
    let mut best_leader = members[0];
    let mut best_worst_out_t = f64::INFINITY;
    let mut best_gflops = -1.0;

    for &m in members {
        let m_idx = m as usize;
        let gflops = profiles
            .iter()
            .find(|p| p.rank == m)
            .map(|p| p.compute_gflops)
            .unwrap_or(0.0);

        let mut worst_out_t = 0.0f64;
        let mut has_outbound = false;
        for other in 0..ranks {
            if !members.contains(&(other as u32)) {
                has_outbound = true;
                let link_t =
                    (lat_us[m_idx][other] / 1e6) + (sync_bytes as f64) / (bw[m_idx][other].max(1e-6) * 1e9);
                worst_out_t = worst_out_t.max(link_t);
            }
        }
        if !has_outbound {
            worst_out_t = 0.0;
        }

        let scale = worst_out_t.max(best_worst_out_t).max(1e-9);
        let diff = worst_out_t - best_worst_out_t;
        let is_cand_better = if diff < -1e-6 * scale {
            true
        } else if diff > 1e-6 * scale {
            false
        } else if (gflops - best_gflops).abs() > 1e-4 {
            gflops > best_gflops
        } else {
            m < best_leader
        };

        if is_cand_better {
            best_leader = m;
            best_worst_out_t = worst_out_t;
            best_gflops = gflops;
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
    let ring_table = (ranks <= 12).then(|| precompute_all_subsets_ring_for_payload(ranks, &bw, &lat_us, sync_bytes));
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
            let t_flat = if let Some(t) = &ring_table {
                let all_idx: Vec<usize> = (0..ranks).collect();
                ring_allreduce_time_secs(&all_idx, &bw, &lat_us, Some(t), sync_bytes)
            } else {
                2.0 * (m_f - 1.0) * lat_ring_secs
                    + 2.0 * ((m_f - 1.0) / m_f) * (sync_bytes as f64 / (b_ring * 1e9))
            };

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
                let leader = pick_leader(members, profiles, &bw, &lat_us, ranks, sync_bytes);
                leaders.push(leader);
                if members.len() > 1 {
                    let isl_idx: Vec<usize> = members.iter().map(|&r| r as usize).collect();
                    let t_intra = ring_allreduce_time_secs(
                        &isl_idx,
                        &bw,
                        &lat_us,
                        ring_table.as_ref(),
                        sync_bytes,
                    );
                    t_intra_max = t_intra_max.max(t_intra);
                }
            }

            let lead_idx: Vec<usize> = leaders.iter().map(|&r| r as usize).collect();
            let t_lead = ring_allreduce_time_secs(&lead_idx, &bw, &lat_us, ring_table.as_ref(), sync_bytes);

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
            let best_leader = pick_leader(&members, profiles, &bw, &lat_us, ranks, sync_bytes);
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
            vram_bandwidth_gbps: 200.0,
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

    #[test]
    fn pick_leader_prefers_lowest_worst_case_outbound_hockney_latency() {
        let devices = vec![
            make_profile(0, 12_000.0), // higher GFLOP/s but slow outbound bridge
            make_profile(1, 10_000.0), // slightly lower GFLOP/s but fast outbound bridge
            make_profile(2, 10_000.0),
        ];
        let mut bw = vec![vec![25.0; 3]; 3];
        let mut lat = vec![vec![2.0; 3]; 3];
        // Rank 0 -> Rank 2 slow bridge; Rank 1 -> Rank 2 fast bridge.
        bw[0][2] = 1.0;
        lat[0][2] = 200.0;
        bw[1][2] = 15.0;
        lat[1][2] = 5.0;

        let leader = super::pick_mst_leader(&[0, 1], &[0, 1, 2], &devices, &bw, &lat, 64 << 20);
        assert_eq!(leader, 1, "leader must minimize worst-case outbound Hockney time");
    }
}
