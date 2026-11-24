struct Params {
    count: u32,
    offset: u32,
}

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.x + gid.y * (nwg.x * 256u);
    if (idx < p.count) {
        a[idx + p.offset] = a[idx + p.offset] + b[idx];
    }
}
