struct SplitKvParams {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    local_tokens: u32,
    scale_bits: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_o: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_stats: array<f32>;
@group(0) @binding(5) var<uniform> params: SplitKvParams;

var<workgroup> sm: array<f32, 256>;
var<workgroup> sl: array<f32, 256>;
var<workgroup> weights: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let tid = lid.x;

    let group_size = params.num_heads / params.num_kv_heads;
    let kv_h = h / group_size;
    let scale = bitcast<f32>(params.scale_bits);

    // ─── Pass 1: Online Max & Sum-Exp Reduction ─────────────────────────────
    var m_local: f32 = -3.4028235e38;
    var l_local: f32 = 0.0;

    let num_tiles = (params.local_tokens + 255u) >> 8u;
    for (var tile: u32 = 0u; tile < num_tiles; tile = tile + 1u) {
        let t = tile * 256u + tid;
        let valid = t < params.local_tokens;
        let t_safe = select(0u, t, valid);

        let q_offset = h * params.head_dim;
        let k_offset = (kv_h * params.local_tokens + t_safe) * params.head_dim;

        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            dot = fma(q[q_offset + d], k_cache[k_offset + d], dot);
        }

        if (valid) {
            let score = dot * scale;
            let m_next = max(m_local, score);
            l_local = l_local * exp(m_local - m_next) + exp(score - m_next);
            m_local = m_next;
        }
    }

    sm[tid] = m_local;
    sl[tid] = l_local;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let m_a = sm[tid];
            let l_a = sl[tid];
            let m_b = sm[tid + stride];
            let l_b = sl[tid + stride];
            let m_max = max(m_a, m_b);
            sl[tid] = l_a * exp(m_a - m_max) + l_b * exp(m_b - m_max);
            sm[tid] = m_max;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        out_stats[2u * h] = sm[0];
        out_stats[2u * h + 1u] = sl[0];
    }
    workgroupBarrier();

    let m_h = sm[0];
    let l_h = sl[0];

    // ─── Pass 2: Tiled Weight-Value Accumulation ────────────────────────────
    let num_kv_tiles = (params.local_tokens + 255u) >> 8u;

    for (var tile: u32 = 0u; tile < num_kv_tiles; tile = tile + 1u) {
        let t0 = tile * 256u;
        let t_curr = t0 + tid;
        let valid = t_curr < params.local_tokens;
        let t_safe = select(0u, t_curr, valid);

        let q_offset = h * params.head_dim;
        let k_offset = (kv_h * params.local_tokens + t_safe) * params.head_dim;

        var dot: f32 = 0.0;
        for (var di: u32 = 0u; di < params.head_dim; di = di + 1u) {
            dot = fma(q[q_offset + di], k_cache[k_offset + di], dot);
        }

        if (valid) {
            let score = dot * scale;
            weights[tid] = exp(score - m_h);
        } else {
            weights[tid] = 0.0;
        }

        workgroupBarrier();

        let tile_len = min(256u, params.local_tokens - t0);
        let num_d_tiles = (params.head_dim + 255u) >> 8u;
        for (var d_tile: u32 = 0u; d_tile < num_d_tiles; d_tile = d_tile + 1u) {
            let d = d_tile * 256u + tid;
            let d_valid = d < params.head_dim;
            let d_safe = select(0u, d, d_valid);
            let v_base = (kv_h * params.local_tokens + t0) * params.head_dim + d_safe;

            var sum_v: f32 = 0.0;
            for (var j: u32 = 0u; j < tile_len; j = j + 1u) {
                sum_v = fma(weights[j], v_cache[v_base + j * params.head_dim], sum_v);
            }

            if (d_valid) {
                let out_idx = h * params.head_dim + d;
                if (tile == 0u) {
                    out_o[out_idx] = sum_v;
                } else {
                    out_o[out_idx] = out_o[out_idx] + sum_v;
                }
            }
        }

        workgroupBarrier();
    }

    // Final normalization
    let num_d_tiles = (params.head_dim + 255u) >> 8u;
    for (var d_tile: u32 = 0u; d_tile < num_d_tiles; d_tile = d_tile + 1u) {
        let d = d_tile * 256u + tid;
        if (d < params.head_dim) {
            let out_idx = h * params.head_dim + d;
            out_o[out_idx] = out_o[out_idx] / max(l_h, 1e-30);
        }
    }
}
