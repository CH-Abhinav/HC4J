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
    let base = (gid.x + gid.y * (nwg.x * 256u)) * 4u;
    if (base + 3u < p.count) {
        let o = base + p.offset;
        let a0 = a[o]; let a1 = a[o + 1u]; let a2 = a[o + 2u]; let a3 = a[o + 3u];
        let b0 = b[base]; let b1 = b[base + 1u]; let b2 = b[base + 2u]; let b3 = b[base + 3u];
        a[o] = a0 + b0; a[o + 1u] = a1 + b1; a[o + 2u] = a2 + b2; a[o + 3u] = a3 + b3;
    } else {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let idx = base + j;
            if (idx < p.count) {
                a[idx + p.offset] = a[idx + p.offset] + b[idx];
            }
        }
    }
}
