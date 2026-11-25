//! WGSL shader generation for matrix multiplication and transposition.

use std::fmt::Write as _;

use crate::kernels::linalg::matmul::plan::{Caps, Kernel, TILE_64, TILE_128, TileConfig};
use crate::engine::GpuEngine;

pub const DIMS_HEADER: &str = "struct Dims {\n    p: vec4<u32>,\n}\n";

/// Register-tiled GEMM for `cfg`. `vec4` requires K % 4 == 0 and N % 4 == 0.
pub fn register_kernel_wgsl(cfg: TileConfig, vec4: bool) -> String {
    let TileConfig { bm, bn, bk, tm, tn } = cfg;
    let tnv = tn / 4; // vec4 column groups per thread
    let tx_count = bn / tn;
    let ty_count = bm / tm;
    let threads = tx_count * ty_count;
    let a_stride = bm + 1; // +1 pad: conflict-free transposed stores
    let bs_row = bn / 4; // vec4s per B tile row
    let a_loads = (bm * bk / 4) / threads;
    let b_loads = (bk * bn / 4) / threads;
    let group_span = bn / tnv; // columns between a thread's vec4 groups
    let elem = if vec4 { "vec4<f32>" } else { "f32" };

    let mut s = String::new();
    let _ = writeln!(s, "{DIMS_HEADER}");
    let _ = writeln!(s, "@group(0) @binding(0) var<storage, read_write> A: array<{elem}>;");
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read_write> B: array<{elem}>;");
    let _ = writeln!(s, "@group(0) @binding(2) var<storage, read_write> C: array<{elem}>;");
    let _ = writeln!(s, "@group(0) @binding(3) var<uniform> dims: Dims;\n");
    let _ = writeln!(s, "var<workgroup> As: array<f32, {}>;", bk * a_stride);
    let _ = writeln!(s, "var<workgroup> Bs: array<vec4<f32>, {}>;\n", bk * bs_row);
    let _ = writeln!(s, "@compute @workgroup_size({tx_count}, {ty_count})");
    let _ = writeln!(
        s,
        "fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, \
         @builtin(num_workgroups) nwg: vec3<u32>) {{"
    );
    let _ = writeln!(s, "    let M = dims.p.x; let N = dims.p.y; let K = dims.p.z; let beta = dims.p.w;");
    let _ = writeln!(s, "    let row0 = (wid.y + wid.z * nwg.y) * {bm}u;");
    let _ = writeln!(s, "    let col0 = wid.x * {bn}u;");
    let _ = writeln!(s, "    let tx = lid.x; let ty = lid.y;");
    let _ = writeln!(s, "    let li = ty * {tx_count}u + tx;");
    let _ = writeln!(s, "    var acc: array<vec4<f32>, {}>;", tm * tnv);
    let _ = writeln!(s, "    let tiles = (K + {}u) / {bk}u;", bk - 1);
    let _ = writeln!(s, "    for (var t = 0u; t < tiles; t = t + 1u) {{");
    let _ = writeln!(s, "        let k0 = t * {bk}u;");

    // A tile: BM rows × BK columns -> As[k][m] (k-major, padded).
    for slot in 0..a_loads {
        let _ = writeln!(s, "        {{");
        let _ = writeln!(s, "            let idx = li + {}u;", slot * threads);
        let _ = writeln!(s, "            let r = idx / {}u; let kq = idx % {}u;", bk / 4, bk / 4);
        let _ = writeln!(s, "            let gr = row0 + r; let gk = k0 + kq * 4u;");
        let _ = writeln!(s, "            var v = vec4<f32>(0.0);");
        if vec4 {
            let _ = writeln!(s, "            if (gr < M && gk < K) {{ v = A[(gr * K + gk) / 4u]; }}");
        } else {
            let _ = writeln!(s, "            if (gr < M) {{");
            let _ = writeln!(s, "                let base = gr * K + gk;");
            for (q, c) in ["x", "y", "z", "w"].iter().enumerate() {
                let _ = writeln!(s, "                if (gk + {q}u < K) {{ v.{c} = A[base + {q}u]; }}");
            }
            let _ = writeln!(s, "            }}");
        }
        let _ = writeln!(s, "            let a0 = kq * {}u + r;", 4 * a_stride);
        for (q, c) in ["x", "y", "z", "w"].iter().enumerate() {
            let _ = writeln!(s, "            As[a0 + {}u] = v.{c};", q as u32 * a_stride);
        }
        let _ = writeln!(s, "        }}");
    }
    // B tile: BK rows × BN columns -> Bs[k][n/4].
    for slot in 0..b_loads {
        let _ = writeln!(s, "        {{");
        let _ = writeln!(s, "            let idx = li + {}u;", slot * threads);
        let _ = writeln!(s, "            let kr = idx / {bs_row}u; let cq = idx % {bs_row}u;");
        let _ = writeln!(s, "            let gk = k0 + kr; let gc = col0 + cq * 4u;");
        let _ = writeln!(s, "            var v = vec4<f32>(0.0);");
        if vec4 {
            let _ = writeln!(s, "            if (gk < K && gc < N) {{ v = B[(gk * N + gc) / 4u]; }}");
        } else {
            let _ = writeln!(s, "            if (gk < K) {{");
            let _ = writeln!(s, "                let base = gk * N + gc;");
            for (q, c) in ["x", "y", "z", "w"].iter().enumerate() {
                let _ = writeln!(s, "                if (gc + {q}u < N) {{ v.{c} = B[base + {q}u]; }}");
            }
            let _ = writeln!(s, "            }}");
        }
        let _ = writeln!(s, "            Bs[kr * {bs_row}u + cq] = v;");
        let _ = writeln!(s, "        }}");
    }
    let _ = writeln!(s, "        workgroupBarrier();");

    // Fully unrolled over the tile's K.
    for kk in 0..bk {
        let _ = writeln!(s, "        {{");
        for i in 0..tm {
            let _ = writeln!(s, "            let a{i} = vec4<f32>(As[{}u + ty * {tm}u]);", kk * a_stride + i);
        }
        for j in 0..tnv {
            let _ = writeln!(s, "            let b{j} = Bs[{}u + tx];", kk * bs_row + j * (group_span / 4));
        }
        for i in 0..tm {
            for j in 0..tnv {
                let slot = i * tnv + j;
                let _ = writeln!(s, "            acc[{slot}] = fma(a{i}, b{j}, acc[{slot}]);");
            }
        }
        let _ = writeln!(s, "        }}");
    }
    let _ = writeln!(s, "        workgroupBarrier();");
    let _ = writeln!(s, "    }}");

    // Store the micro-tile.
    for i in 0..tm {
        for j in 0..tnv {
            let slot = i * tnv + j;
            let _ = writeln!(s, "    {{");
            let _ = writeln!(s, "        let r = row0 + ty * {tm}u + {i}u;");
            let _ = writeln!(s, "        let c = col0 + {}u + tx * 4u;", j * group_span);
            if vec4 {
                let _ = writeln!(s, "        if (r < M && c < N) {{");
                let _ = writeln!(s, "            let idx = (r * N + c) / 4u;");
                // beta == 1 accumulates, for K-blocked out-of-core matmul.
                let _ = writeln!(s, "            if (beta == 1u) {{ C[idx] = C[idx] + acc[{slot}]; }} else {{ C[idx] = acc[{slot}]; }}");
                let _ = writeln!(s, "        }}");
            } else {
                let _ = writeln!(s, "        if (r < M) {{");
                let _ = writeln!(s, "            let base = r * N + c; let v = acc[{slot}];");
                for (q, comp) in ["x", "y", "z", "w"].iter().enumerate() {
                    let _ = writeln!(s, "            if (c + {q}u < N) {{");
                    let _ = writeln!(
                        s,
                        "                if (beta == 1u) {{ C[base + {q}u] = C[base + {q}u] + v.{comp}; }} else {{ C[base + {q}u] = v.{comp}; }}"
                    );
                    let _ = writeln!(s, "            }}");
                }
                let _ = writeln!(s, "        }}");
            }
            let _ = writeln!(s, "    }}");
        }
    }
    let _ = writeln!(s, "}}");
    s
}

/// Classic 16×16 shared-memory tiled GEMM, one output element per thread.
pub const TILED16_WGSL: &str = r#"
struct Dims {
    p: vec4<u32>,
}

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let M = dims.p.x;
    let N = dims.p.y;
    let K = dims.p.z;
    let row = (wid.y + wid.z * nwg.y) * 16u + lid.y;
    let col = wid.x * 16u + lid.x;
    var acc = 0.0;
    let tiles = (K + 15u) / 16u;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let ka = t * 16u + lid.x;
        var va = 0.0;
        if (row < M && ka < K) { va = A[row * K + ka]; }
        tile_a[lid.y * 16u + lid.x] = va;
        let kb = t * 16u + lid.y;
        var vb = 0.0;
        if (kb < K && col < N) { vb = B[kb * N + col]; }
        tile_b[lid.y * 16u + lid.x] = vb;
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            acc = fma(tile_a[lid.y * 16u + kk], tile_b[kk * 16u + lid.x], acc);
        }
        workgroupBarrier();
    }
    if (row < M && col < N) {
        let idx = row * N + col;
        // beta == 1 accumulates, for K-blocked out-of-core matmul.
        if (dims.p.w == 1u) { C[idx] = C[idx] + acc; } else { C[idx] = acc; }
    }
}
"#;

/// GEMV, matrix R×K row-major times vector K. `LANES` lanes per row; rows per
/// workgroup = 256 / LANES. With `subgroup`, one row per workgroup and the
/// reduction uses `subgroupAdd` plus one shared slot per subgroup. Slots are
/// claimed with an atomic because WGSL does not tie subgroup membership to
/// `local_invocation_index`.
pub fn gemv_rows_wgsl(lanes: u32, vec4: bool, subgroup: bool) -> String {
    let elem = if vec4 { "vec4<f32>" } else { "f32" };
    let rows_per_wg = 256 / lanes;
    let accumulate = if vec4 {
        "        let K4 = K / 4u;\n        let base = row * K4;\n        for (var k = lane; k < K4; k = k + LANES) { sum = sum + dot(Mx[base + k], V[k]); }"
    } else {
        "        let base = row * K;\n        for (var k = lane; k < K; k = k + LANES) { sum = fma(Mx[base + k], V[k], sum); }"
    };
    let head = format!(
        "{DIMS_HEADER}\
@group(0) @binding(0) var<storage, read_write> Mx: array<{elem}>;\n\
@group(0) @binding(1) var<storage, read_write> V: array<{elem}>;\n\
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;\n\
@group(0) @binding(3) var<uniform> dims: Dims;\n\n\
const LANES: u32 = {lanes}u;\n\
const ROWS: u32 = {rows_per_wg}u;\n\
var<workgroup> partial: array<f32, 256>;\n"
    );
    if subgroup {
        format!(
            "{head}var<workgroup> slots: atomic<u32>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
) {{
    let R = dims.p.x;
    let K = dims.p.y;
    let lane = li;
    let row = wid.x + wid.y * nwg.x;
    // Pipelines skip WGSL's implicit workgroup zero-init, so reset the
    // slot counter explicitly before any subgroup claims a slot.
    if (li == 0u) {{ atomicStore(&slots, 0u); }}
    workgroupBarrier();
    var sum = 0.0;
    if (row < R) {{
{accumulate}
    }}
    let total = subgroupAdd(sum);
    if (sg_lane == 0u) {{
        let slot = atomicAdd(&slots, 1u);
        partial[slot] = total;
    }}
    workgroupBarrier();
    if (li == 0u && row < R) {{
        let count = atomicLoad(&slots);
        var acc = 0.0;
        for (var i = 0u; i < count; i = i + 1u) {{ acc = acc + partial[i]; }}
        if (dims.p.z == 1u) {{ Y[row] = Y[row] + acc; }} else {{ Y[row] = acc; }}
    }}
}}
"
        )
    } else {
        format!(
            "{head}
@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let R = dims.p.x;
    let K = dims.p.y;
    let group = li / LANES;
    let lane = li % LANES;
    let row = (wid.x + wid.y * nwg.x) * ROWS + group;
    var sum = 0.0;
    if (row < R) {{
{accumulate}
    }}
    partial[li] = sum;
    workgroupBarrier();
    for (var s = LANES / 2u; s > 0u; s = s >> 1u) {{
        if (lane < s) {{ partial[li] = partial[li] + partial[li + s]; }}
        workgroupBarrier();
    }}
    if (lane == 0u && row < R) {{
        if (dims.p.z == 1u) {{ Y[row] = Y[row] + partial[li]; }} else {{ Y[row] = partial[li]; }}
    }}
}}
"
        )
    }
}

/// GEMV, vector K times matrix K×C row-major: a thread per column (reads
/// coalesced across columns), K split into 256 / COLS slices reduced in
/// shared memory.
pub fn gemv_cols_wgsl(cols: u32) -> String {
    let slices = 256 / cols;
    format!(
        "{DIMS_HEADER}\
@group(0) @binding(0) var<storage, read_write> Mx: array<f32>;\n\
@group(0) @binding(1) var<storage, read_write> V: array<f32>;\n\
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;\n\
@group(0) @binding(3) var<uniform> dims: Dims;\n\n\
const COLS: u32 = {cols}u;\n\
const SLICES: u32 = {slices}u;\n\
var<workgroup> partial: array<f32, 256>;\n\n\
@compute @workgroup_size(256)\n\
fn main(\n    \
@builtin(local_invocation_index) li: u32,\n    \
@builtin(workgroup_id) wid: vec3<u32>,\n    \
@builtin(num_workgroups) nwg: vec3<u32>,\n\
) {{\n    \
let Cn = dims.p.x;\n    \
let K = dims.p.y;\n    \
let lx = li % COLS;\n    \
let ly = li / COLS;\n    \
let col = (wid.x + wid.y * nwg.x) * COLS + lx;\n    \
var sum = 0.0;\n    \
if (col < Cn) {{\n        \
for (var k = ly; k < K; k = k + SLICES) {{ sum = fma(V[k], Mx[k * Cn + col], sum); }}\n    \
}}\n    \
partial[li] = sum;\n    \
workgroupBarrier();\n    \
for (var s = SLICES / 2u; s > 0u; s = s >> 1u) {{\n        \
if (ly < s) {{ partial[li] = partial[li] + partial[li + s * COLS]; }}\n        \
workgroupBarrier();\n    \
}}\n    \
if (ly == 0u && col < Cn) {{\n        \
if (dims.p.z == 1u) {{ Y[col] = Y[col] + partial[li]; }} else {{ Y[col] = partial[li]; }}\n    \
}}\n\
}}\n"
    )
}

/// `dst[C×R] = transpose(src[R×C])` through a 32×33 tile: the +1 pad makes
/// the column-wise reads of the tile hit 32 distinct banks.
pub const TRANSPOSE_WGSL: &str = r#"
struct Dims {
    p: vec4<u32>,
}

@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

var<workgroup> tile: array<f32, 1056>; // 32 x 33

@compute @workgroup_size(32, 8)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let R = dims.p.x;
    let Cn = dims.p.y;
    let r0 = (wid.y + wid.z * nwg.y) * 32u;
    let c0 = wid.x * 32u;
    for (var j = 0u; j < 32u; j = j + 8u) {
        let r = r0 + lid.y + j;
        let c = c0 + lid.x;
        var v = 0.0;
        if (r < R && c < Cn) { v = src[r * Cn + c]; }
        tile[(lid.y + j) * 33u + lid.x] = v;
    }
    workgroupBarrier();
    for (var j = 0u; j < 32u; j = j + 8u) {
        let out_row = c0 + lid.y + j;
        let out_col = r0 + lid.x;
        if (out_row < Cn && out_col < R) {
            dst[out_row * R + out_col] = tile[lid.x * 33u + lid.y + j];
        }
    }
}
"#;

pub fn kernel_source(kernel: Kernel) -> (String, String) {
    match kernel {
        Kernel::Register { cfg, vec4 } => (
            format!("gemm_reg_{}x{}_{}x{}_{}", cfg.bm, cfg.bn, cfg.tm, cfg.tn, if vec4 { "v4" } else { "s" }),
            register_kernel_wgsl(cfg, vec4),
        ),
        Kernel::Tiled16 => ("gemm_tiled16".to_string(), TILED16_WGSL.to_string()),
        Kernel::GemvRows { lanes, vec4, subgroup } => (
            format!("gemv_rows_{lanes}_{}{}", if vec4 { "v4" } else { "s" }, if subgroup { "_sg" } else { "" }),
            gemv_rows_wgsl(lanes, vec4, subgroup),
        ),
        Kernel::GemvCols { cols } => (format!("gemv_cols_{cols}"), gemv_cols_wgsl(cols)),
    }
}

/// Kernels worth compiling before first use: on D3D12 the register-tiled GEMM
/// and the transpose kernel cost 0.5-5 s in FXC, which would otherwise land
/// on the first matmul a caller issues.
pub fn warmup_kernels(engine: &GpuEngine) -> Vec<(String, String)> {
    let caps = Caps::of(engine);
    let cfg = match caps.forced_tile {
        Some(cfg) if caps.fits(cfg) => cfg,
        _ if caps.discrete && caps.fits(TILE_128) => TILE_128,
        _ => TILE_64,
    };
    vec![
        kernel_source(Kernel::Register { cfg, vec4: true }),
        kernel_source(Kernel::Tiled16),
        ("transpose_f32".to_string(), TRANSPOSE_WGSL.to_string()),
    ]
}
