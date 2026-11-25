struct GemmParams {
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
}

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<uniform> params: GemmParams;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

fn apply_act(x: f32, act_code: u32) -> f32 {
    if (act_code == 1u) {
        // GELU (clamped tanh approximation matching ops/nn/activations.rs)
        let cubic = 0.7978845608028654 * (x + 0.044715 * x * x * x);
        let inner = clamp(cubic, -10.0, 10.0);
        return 0.5 * x * (1.0 + tanh(inner));
    } else if (act_code == 2u) {
        // SILU
        let e = exp(-abs(x));
        let num = select(e, 1.0, x >= 0.0);
        return x * num / (1.0 + e);
    } else if (act_code == 3u) {
        // RELU
        return max(x, 0.0);
    }
    return x;
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wid.y * 16u + lid.y;
    let col = wid.x * 16u + lid.x;

    var acc: f32 = 0.0;
    let num_tiles = (params.k + 15u) / 16u;

    for (var t_idx: u32 = 0u; t_idx < num_tiles; t_idx = t_idx + 1u) {
        let t = t_idx * 16u;
        let a_col = t + lid.x;
        let b_row = t + lid.y;

        if (row < params.m && a_col < params.k) {
            tile_a[lid.y * 16u + lid.x] = a[row * params.k + a_col];
        } else {
            tile_a[lid.y * 16u + lid.x] = 0.0;
        }

        if (b_row < params.k && col < params.n) {
            tile_b[lid.y * 16u + lid.x] = b[b_row * params.n + col];
        } else {
            tile_b[lid.y * 16u + lid.x] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            acc = fma(tile_a[lid.y * 16u + i], tile_b[i * 16u + lid.x], acc);
        }

        workgroupBarrier();
    }

    if (row < params.m && col < params.n) {
        if ((params.flags & 1u) != 0u) {
            acc = acc + bias[col];
        }
        let act = (params.flags >> 1u) & 3u;
        c[row * params.n + col] = apply_act(acc, act);
    }
}
