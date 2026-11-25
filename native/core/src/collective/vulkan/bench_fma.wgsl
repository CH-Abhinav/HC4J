struct Params {
    iterations: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;

    var a0 = f32(gid) * 0.001;
    var a1 = f32(gid) * 0.002 + 0.5;
    var a2 = f32(gid) * 0.003 + 1.0;
    var a3 = f32(gid) * 0.004 + 1.5;

    let k0 = 1.0000001;
    let c0 = 0.0000001;
    let k1 = 0.9999999;
    let c1 = 0.0000002;
    let k2 = 1.0000002;
    let c2 = 0.0000001;
    let k3 = 0.9999998;
    let c3 = 0.0000003;

    for (var i: u32 = 0u; i < params.iterations; i = i + 1u) {
        a0 = fma(a0, k0, c0);
        a1 = fma(a1, k1, c1);
        a2 = fma(a2, k2, c2);
        a3 = fma(a3, k3, c3);
    }

    out[gid] = a0 + a1 + a2 + a3;
}
