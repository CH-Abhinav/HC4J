//! Distributed collective kernels and lossless Split-KV attention merge.

use crate::engine::error::{Hc4jError, Hc4jResult};

/// Merges partial Split-KV attention outputs and online-softmax `(m_i, l_i)` statistics
/// across multiple workers losslessly without overflow or intermediate allocation drift.
///
/// Mirrors `SplitKvSession`'s online-softmax reduction mathematically:
/// For each head `h`:
/// - `m_h = max_i m_{i,h}` over active workers (`l_{i,h} > 0`)
/// - `w_{i,h} = l_{i,h} * exp(m_{i,h} - m_h)` (always in `[0, l_{i,h}]`, never overflows)
/// - `out[h, d] = sum_i w_{i,h} * O_i[h, d] / sum_i w_{i,h}`
pub fn merge_split_kv_f32(
    num_heads: u32,
    head_dim: u32,
    partials_o: &[&[f32]],
    partials_stats: &[&[f32]],
    out: &mut [f32],
) -> Hc4jResult<()> {
    if num_heads == 0 || head_dim == 0 || partials_o.is_empty() {
        return Err(Hc4jError::InvalidParam("num_heads, head_dim, and worker count must be positive"));
    }
    if partials_o.len() != partials_stats.len() {
        return Err(Hc4jError::InvalidParam("partials_o and partials_stats length mismatch"));
    }

    let nh = num_heads as usize;
    let hd = head_dim as usize;
    let out_elems = nh.checked_mul(hd).ok_or(Hc4jError::InvalidParam("output elements overflow"))?;
    let stats_elems = nh.checked_mul(2).ok_or(Hc4jError::InvalidParam("stats elements overflow"))?;
    if out.len() < out_elems {
        return Err(Hc4jError::InvalidParam("output slice smaller than num_heads * head_dim"));
    }
    for (i, (po, ps)) in partials_o.iter().zip(partials_stats.iter()).enumerate() {
        if po.len() < out_elems {
            return Err(Hc4jError::InvalidParam("partial_o slice too short for head layout"));
        }
        if ps.len() < stats_elems {
            return Err(Hc4jError::InvalidParam("partial_stats slice too short for 2 * num_heads"));
        }
        let _ = i;
    }

    let mut weights = vec![0.0f32; partials_o.len()];
    for h in 0..nh {
        let mut m_h = f32::NEG_INFINITY;
        for ps in partials_stats {
            let m_i = ps[2 * h];
            let l_i = ps[2 * h + 1];
            if l_i > 0.0 && l_i.is_finite() && m_i.is_finite() && m_i > m_h {
                m_h = m_i;
            }
        }

        if !m_h.is_finite() {
            out[h * hd..(h + 1) * hd].fill(0.0);
            continue;
        }

        let mut l_h = 0.0f32;
        for (i, ps) in partials_stats.iter().enumerate() {
            let m_i = ps[2 * h];
            let l_i = ps[2 * h + 1];
            if l_i > 0.0 && l_i.is_finite() && m_i.is_finite() {
                // `m_i - m_h <= 0.0`, so exp is always in (0.0, 1.0].
                let w_i = l_i * (m_i - m_h).exp();
                weights[i] = w_i;
                l_h += w_i;
            } else {
                weights[i] = 0.0;
            }
        }

        let row_out = &mut out[h * hd..(h + 1) * hd];
        if l_h > 0.0 {
            for (d, dst) in row_out.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                let idx = h * hd + d;
                for (i, &w_i) in weights.iter().enumerate() {
                    if w_i > 0.0 {
                        acc += w_i * partials_o[i][idx];
                    }
                }
                *dst = acc / l_h;
            }
        } else {
            row_out.fill(0.0);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::merge_split_kv_f32;

    fn worker_partial(scores: &[f32], values: &[f32], head_dim: usize) -> (Vec<f32>, Vec<f32>) {
        if scores.is_empty() {
            return (vec![0.0; head_dim], vec![f32::NEG_INFINITY, 0.0]);
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut l = 0.0f32;
        let mut o = vec![0.0f32; head_dim];
        for (t, &s) in scores.iter().enumerate() {
            let w = (s - m).exp();
            l += w;
            for d in 0..head_dim {
                o[d] += w * values[t * head_dim + d];
            }
        }
        for v in &mut o {
            *v /= l;
        }
        (o, vec![m, l])
    }

    #[test]
    fn online_softmax_merge_matches_exact_full_attention_with_extreme_shifts() {
        let hd = 4usize;
        // Worker 0 has extreme negative scores (-50), Worker 1 has extreme positive scores (+50),
        // Worker 2 is empty (0 tokens), Worker 3 has uneven length (7 tokens).
        let scores_w0 = [-50.0f32, -52.5, -49.0];
        let vals_w0 = [1.0, 2.0, -3.0, 4.0, 0.5, 1.5, -1.0, 2.5, 3.0, -1.0, 2.0, 0.0];

        let scores_w1 = [50.0f32, 48.0];
        let vals_w1 = [10.0, -20.0, 30.0, -40.0, 12.0, -18.0, 28.0, -36.0];

        let scores_w3 = [49.0f32, 50.5, 47.0, 46.0, 49.5, 48.5, 45.0];
        let vals_w3: Vec<f32> = (0..7 * hd).map(|i| (i as f32 * 0.37).sin()).collect();

        let (o0, s0) = worker_partial(&scores_w0, &vals_w0, hd);
        let (o1, s1) = worker_partial(&scores_w1, &vals_w1, hd);
        let (o2, s2) = worker_partial(&[], &[], hd);
        let (o3, s3) = worker_partial(&scores_w3, &vals_w3, hd);

        let po: Vec<&[f32]> = vec![&o0, &o1, &o2, &o3];
        let ps: Vec<&[f32]> = vec![&s0, &s1, &s2, &s3];
        let mut got = vec![999.0f32; hd];
        merge_split_kv_f32(1, hd as u32, &po, &ps, &mut got).unwrap();

        // Exact reference computed over the concatenated sequence
        let mut all_scores = Vec::new();
        all_scores.extend_from_slice(&scores_w0);
        all_scores.extend_from_slice(&scores_w1);
        all_scores.extend_from_slice(&scores_w3);

        let mut all_vals = Vec::new();
        all_vals.extend_from_slice(&vals_w0);
        all_vals.extend_from_slice(&vals_w1);
        all_vals.extend_from_slice(&vals_w3);

        let (want, _) = worker_partial(&all_scores, &all_vals, hd);
        for d in 0..hd {
            assert!(got[d].is_finite(), "dim {d} non-finite");
            assert!((got[d] - want[d]).abs() < 1e-5, "dim {d}: got {} want {}", got[d], want[d]);
        }
    }
}
