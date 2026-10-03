//! Matrix multiplication `C[M×N] = op(A)[M×K] · op(B)[K×N]`, f32, row-major.
//!
//! Three kernel families, chosen at runtime by [`plan`] from `(M, N, K)`,
//! the transpose flags and the device's capabilities:
//!
//! 1. **Register-tiled GEMM** ([`register_kernel_wgsl`]). Each workgroup
//!    computes a BM×BN tile of C; each thread a TM×TN register micro-tile.
//!    A and B tiles are staged in workgroup memory with vec4 global loads (or
//!    guarded scalar loads when K or N is not a multiple of 4). The A tile is
//!    stored k-major with a +1 row pad so its transposed stores do not
//!    serialize on shared-memory banks. The K loop over a tile is fully
//!    unrolled at generation time, and accumulation uses vec4 `fma`. Two
//!    configurations: 64×64 with 4×4 micro-tiles (portable), and 128×128
//!    with 8×8 (discrete GPUs, whose register files hold 64 accumulators
//!    without spilling).
//! 2. **16×16 tiled GEMM** ([`TILED16_WGSL`]). Classic shared-memory tiling,
//!    one output per thread, for small or thin shapes.
//! 3. **GEMV** for `M == 1` or `N == 1`. A 2-D GEMM tile would read one row
//!    or column of a whole tile to produce a single value, wasting most of
//!    its bandwidth. Row-major matrix · vector uses a group of lanes per row
//!    (coalesced reads along K) plus a shared-memory tree reduction, or a
//!    `subgroupAdd` reduction when the device supports subgroups. Vector ·
//!    matrix uses a thread per column (coalesced across columns) with K split
//!    into slices reduced in shared memory.
//!
//! **Transposed operands.** A GEMV with a transposed matrix swaps to the
//! other GEMV mapping, so no copy is needed. A GEMM with a transposed operand
//! first transposes it in VRAM with a 32×33 padded-tile kernel, recorded into
//! the same command buffer, so the GEMM streams B along K with coalesced
//! reads.
//!
//! Storage bindings are `read_write` (slab regions share buffers). Output
//! aliasing an input is rejected, since tiles of C would race with reads of A/B.

use std::sync::OnceLock;

use crate::error::{Hc4jError, Hc4jResult, ffi_guard};
use crate::memory::{DeviceBlock, DeviceSpan, TensorId, manager};
use crate::ops::elementwise::grid_2d;
use crate::stream::Recorder;
use crate::{ErrorTrap, GpuEngine};

pub const FLAG_TRANS_A: u32 = 1;
pub const FLAG_TRANS_B: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileConfig {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub tm: u32,
    pub tn: u32,
}

pub const TILE_64: TileConfig = TileConfig { bm: 64, bn: 64, bk: 16, tm: 4, tn: 4 };
pub const TILE_128: TileConfig = TileConfig { bm: 128, bn: 128, bk: 16, tm: 8, tn: 8 };

impl TileConfig {
    /// Workgroup memory: padded k-major A tile plus the vec4 B tile.
    pub fn shared_bytes(&self) -> u32 {
        self.bk * (self.bm + 1) * 4 + self.bk * (self.bn / 4) * 16
    }

    pub fn threads(&self) -> u32 {
        (self.bn / self.tn) * (self.bm / self.tm)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Register { cfg: TileConfig, vec4: bool },
    Tiled16,
    /// `y[r] = Σ_k M[r][k] · v[k]` with M row-major R×K.
    GemvRows { lanes: u32, vec4: bool, subgroup: bool },
    /// `y[c] = Σ_k v[k] · M[k][c]` with M row-major K×C.
    GemvCols { cols: u32 },
}

impl Kernel {
    /// Stable ids reported through `hc4j_matmul_plan` (mirrored in Java).
    pub fn code(&self) -> i32 {
        match self {
            Kernel::Register { cfg, vec4: true } if *cfg == TILE_64 => 1,
            Kernel::Register { cfg, vec4: false } if *cfg == TILE_64 => 2,
            Kernel::Register { vec4: true, .. } => 3,
            Kernel::Register { vec4: false, .. } => 4,
            Kernel::Tiled16 => 5,
            Kernel::GemvRows { subgroup: false, .. } => 6,
            Kernel::GemvRows { subgroup: true, .. } => 7,
            Kernel::GemvCols { .. } => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operands {
    /// A (M×K) · B (K×N), transposing stored operands into scratch first.
    Gemm { pretranspose_a: bool, pretranspose_b: bool },
    /// One operand is the matrix, the other the vector. `rows`/`cols` is the
    /// output length and `k` the reduction length.
    Gemv { matrix_is_a: bool, out_len: u32, k: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub kernel: Kernel,
    pub operands: Operands,
}

impl Plan {
    /// `kernel | pretranspose_a << 8 | pretranspose_b << 9`.
    pub fn code(&self) -> i32 {
        let (ta, tb) = match self.operands {
            Operands::Gemm { pretranspose_a, pretranspose_b } => (pretranspose_a, pretranspose_b),
            Operands::Gemv { .. } => (false, false),
        };
        self.kernel.code() | (i32::from(ta) << 8) | (i32::from(tb) << 9)
    }
}

/// What the dispatcher needs to know about the device.
#[derive(Clone, Copy, Debug)]
pub struct Caps {
    pub subgroups: bool,
    pub workgroup_storage: u32,
    pub invocations: u32,
    pub discrete: bool,
    /// `HC4J_GEMM_TILE=64|128` forces a register-tile config (tuning/tests).
    pub forced_tile: Option<TileConfig>,
}

impl Caps {
    pub fn of(engine: &GpuEngine) -> Self {
        static FORCED: OnceLock<Option<TileConfig>> = OnceLock::new();
        let forced_tile = *FORCED.get_or_init(|| match std::env::var("HC4J_GEMM_TILE").as_deref() {
            Ok("64") => Some(TILE_64),
            Ok("128") => Some(TILE_128),
            _ => None,
        });
        Self {
            subgroups: engine.features.subgroups,
            workgroup_storage: engine.limits.max_compute_workgroup_storage_size,
            invocations: engine.limits.max_compute_invocations_per_workgroup,
            discrete: engine.adapter_info.device_type == wgpu::DeviceType::DiscreteGpu,
            forced_tile,
        }
    }

    fn fits(&self, cfg: TileConfig) -> bool {
        cfg.threads() <= self.invocations && cfg.shared_bytes() <= self.workgroup_storage
    }
}

fn gemv_lanes(k: u32) -> u32 {
    if k >= 1024 { 256 } else { 32 }
}

fn gemv_cols(cols: u32) -> u32 {
    if cols >= 64 {
        64
    } else if cols >= 16 {
        16
    } else {
        4
    }
}

fn gemv_rows_kernel(caps: &Caps, k: u32) -> Kernel {
    let lanes = gemv_lanes(k);
    Kernel::GemvRows {
        lanes,
        vec4: k.is_multiple_of(4),
        // The subgroup reduction is written for one row per workgroup.
        subgroup: caps.subgroups && lanes == 256,
    }
}

/// Chooses the kernel for `C[M×N] = op(A) · op(B)`. Pure: unit-tested
/// without a GPU.
pub fn plan(caps: &Caps, m: u32, n: u32, k: u32, flags: u32) -> Hc4jResult<Plan> {
    if m == 0 || n == 0 || k == 0 {
        return Err(Hc4jError::InvalidParam("matmul dimensions must be non-zero"));
    }
    if flags & !(FLAG_TRANS_A | FLAG_TRANS_B) != 0 {
        return Err(Hc4jError::InvalidParam("unknown matmul flags"));
    }
    let (trans_a, trans_b) = (flags & FLAG_TRANS_A != 0, flags & FLAG_TRANS_B != 0);

    if n == 1 {
        // y[M] = op(A) · x. The vector's storage is the same either way.
        return Ok(if trans_a && m > 1 {
            // A stored K×M: y[m] = Σ_k A_s[k][m] x[k] is the column mapping.
            Plan {
                kernel: Kernel::GemvCols { cols: gemv_cols(m) },
                operands: Operands::Gemv { matrix_is_a: true, out_len: m, k },
            }
        } else {
            Plan {
                kernel: gemv_rows_kernel(caps, k),
                operands: Operands::Gemv { matrix_is_a: true, out_len: m, k },
            }
        });
    }
    if m == 1 {
        // y[N] = x · op(B).
        return Ok(if trans_b {
            // B stored N×K: y[n] = Σ_k B_s[n][k] a[k] is the row mapping.
            Plan {
                kernel: gemv_rows_kernel(caps, k),
                operands: Operands::Gemv { matrix_is_a: false, out_len: n, k },
            }
        } else {
            Plan {
                kernel: Kernel::GemvCols { cols: gemv_cols(n) },
                operands: Operands::Gemv { matrix_is_a: false, out_len: n, k },
            }
        });
    }

    let operands = Operands::Gemm { pretranspose_a: trans_a, pretranspose_b: trans_b };
    let vec4 = k.is_multiple_of(4) && n.is_multiple_of(4);
    let big_enough = m.min(n) >= 64 && k >= 64;
    let cfg = match caps.forced_tile {
        Some(cfg) if caps.fits(cfg) && m.min(n) >= 16 => Some(cfg),
        Some(_) => None,
        None if big_enough && caps.discrete && m >= 512 && n >= 512 && caps.fits(TILE_128) => Some(TILE_128),
        None if big_enough && caps.fits(TILE_64) => Some(TILE_64),
        None => None,
    };
    let kernel = match cfg {
        Some(cfg) => Kernel::Register { cfg, vec4 },
        None => Kernel::Tiled16,
    };
    Ok(Plan { kernel, operands })
}

// ============================================================================
// WGSL
// ============================================================================

const DIMS_HEADER: &str = "struct Dims {\n    p: vec4<u32>,\n}\n";

/// Register-tiled GEMM for `cfg`. `vec4` requires K % 4 == 0 and N % 4 == 0.
pub fn register_kernel_wgsl(cfg: TileConfig, vec4: bool) -> String {
    use std::fmt::Write as _;
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
    let _ = writeln!(s, "    let M = dims.p.x; let N = dims.p.y; let K = dims.p.z;");
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
                let _ = writeln!(s, "        if (r < M && c < N) {{ C[(r * N + c) / 4u] = acc[{slot}]; }}");
            } else {
                let _ = writeln!(s, "        if (r < M) {{");
                let _ = writeln!(s, "            let base = r * N + c; let v = acc[{slot}];");
                for (q, comp) in ["x", "y", "z", "w"].iter().enumerate() {
                    let _ = writeln!(s, "            if (c + {q}u < N) {{ C[base + {q}u] = v.{comp}; }}");
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
        C[row * N + col] = acc;
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
        "{DIMS_HEADER}
@group(0) @binding(0) var<storage, read_write> Mx: array<{elem}>;
@group(0) @binding(1) var<storage, read_write> V: array<{elem}>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const LANES: u32 = {lanes}u;
const ROWS: u32 = {rows_per_wg}u;
var<workgroup> partial: array<f32, 256>;
"
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
        Y[row] = acc;
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
    if (lane == 0u && row < R) {{ Y[row] = partial[li]; }}
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
        "{DIMS_HEADER}
@group(0) @binding(0) var<storage, read_write> Mx: array<f32>;
@group(0) @binding(1) var<storage, read_write> V: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const COLS: u32 = {cols}u;
const SLICES: u32 = {slices}u;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let Cn = dims.p.x;
    let K = dims.p.y;
    let lx = li % COLS;
    let ly = li / COLS;
    let col = (wid.x + wid.y * nwg.x) * COLS + lx;
    var sum = 0.0;
    if (col < Cn) {{
        for (var k = ly; k < K; k = k + SLICES) {{ sum = fma(V[k], Mx[k * Cn + col], sum); }}
    }}
    partial[li] = sum;
    workgroupBarrier();
    for (var s = SLICES / 2u; s > 0u; s = s >> 1u) {{
        if (ly < s) {{ partial[li] = partial[li] + partial[li + s * COLS]; }}
        workgroupBarrier();
    }}
    if (ly == 0u && col < Cn) {{ Y[col] = partial[li]; }}
}}
"
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

fn kernel_source(kernel: Kernel) -> (String, String) {
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

// ============================================================================
// Execution
// ============================================================================

fn dims_bytes(p: [u32; 4]) -> Vec<u8> {
    bytemuck::bytes_of(&p).to_vec()
}

/// Grid for a kernel tiled over (cols, rows) with the row tiles folded into
/// Y×Z when they exceed the per-dimension limit.
fn tiled_grid(engine: &GpuEngine, col_tiles: u32, row_tiles: u32) -> Hc4jResult<(u32, u32, u32)> {
    let max = engine.limits.max_compute_workgroups_per_dimension;
    if col_tiles > max {
        return Err(Hc4jError::Unsupported("matrix too wide for one dispatch"));
    }
    let (y, z) = grid_2d(row_tiles, max);
    if z > max {
        return Err(Hc4jError::Unsupported("matrix too tall for one dispatch"));
    }
    Ok((col_tiles.max(1), y, z))
}

fn bind_group(
    engine: &GpuEngine,
    pipeline: &wgpu::ComputePipeline,
    resources: Vec<wgpu::BindingResource<'_>>,
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = resources
        .into_iter()
        .enumerate()
        .map(|(i, resource)| wgpu::BindGroupEntry { binding: i as u32, resource })
        .collect();
    engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("HC4J Matmul"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

fn encode_transpose(
    rec: &mut Recorder<'_>,
    engine: &GpuEngine,
    pipeline: &wgpu::ComputePipeline,
    src: &DeviceSpan,
    dst: &DeviceSpan,
    rows: u32,
    cols: u32,
) -> Hc4jResult<()> {
    let uniform = rec.uniform(&dims_bytes([rows, cols, 0, 0]))?;
    let bg = bind_group(engine, pipeline, vec![src.binding(), dst.binding(), uniform]);
    let grid = tiled_grid(engine, cols.div_ceil(32), rows.div_ceil(32))?;
    rec.dispatch(pipeline, &bg, grid, src.size + dst.size);
    Ok(())
}

fn transpose_pipeline(engine: &GpuEngine) -> Hc4jResult<wgpu::ComputePipeline> {
    engine.get_or_compile("transpose_f32", "main", TRANSPOSE_WGSL)
}

fn checked_bytes(a: u32, b: u32) -> Hc4jResult<u64> {
    (a as u64)
        .checked_mul(b as u64)
        .and_then(|e| e.checked_mul(4))
        .ok_or(Hc4jError::InvalidParam("matrix size overflows"))
}

pub fn run_matmul(id_a: TensorId, id_b: TensorId, id_out: TensorId, m: u32, n: u32, k: u32, flags: u32) -> Hc4jResult<()> {
    let mgr = manager()?;
    let engine = mgr.engine();
    let plan = plan(&Caps::of(engine), m, n, k, flags)?;
    if mgr.size_of(id_a)? != checked_bytes(m, k)?
        || mgr.size_of(id_b)? != checked_bytes(k, n)?
        || mgr.size_of(id_out)? != checked_bytes(m, n)?
    {
        return Err(Hc4jError::InvalidParam("tensor sizes do not match (m, n, k)"));
    }
    if id_out == id_a || id_out == id_b {
        return Err(Hc4jError::InvalidParam("matmul output must not alias an input"));
    }

    // No streaming fallback: GEMM tiles need random access to whole operands.
    let resident = mgr.acquire_resident(&[id_a, id_b, id_out])?;
    let (a, b, c) = (&resident.spans[0], &resident.spans[1], &resident.spans[2]);
    if c.overlaps(a) || c.overlaps(b) {
        return Err(Hc4jError::InvalidParam("matmul output must not alias an input"));
    }
    let max_binding = engine.limits.max_storage_buffer_binding_size;
    if [a, b, c].iter().any(|s| s.size > max_binding) {
        return Err(Hc4jError::Unsupported("matmul operand exceeds the storage-binding limit"));
    }

    let (name, source) = kernel_source(plan.kernel);
    let pipeline = engine.get_or_compile(&name, "main", &source)?;

    match plan.operands {
        Operands::Gemv { matrix_is_a, out_len, k } => {
            let (matrix, vector) = if matrix_is_a { (a, b) } else { (b, a) };
            let grid = match plan.kernel {
                Kernel::GemvRows { lanes, .. } => {
                    let wgs = out_len.div_ceil(256 / lanes);
                    let (x, y) = grid_2d(wgs, engine.limits.max_compute_workgroups_per_dimension);
                    (x, y, 1)
                }
                Kernel::GemvCols { cols } => {
                    let (x, y) = grid_2d(out_len.div_ceil(cols), engine.limits.max_compute_workgroups_per_dimension);
                    (x, y, 1)
                }
                _ => return Err(Hc4jError::Device("GEMV plan with a GEMM kernel".to_string())),
            };
            let bytes = matrix.size + vector.size + c.size;
            let trap = ErrorTrap::push(&engine.device);
            let recorded = engine.stream.record(1, |rec| {
                let uniform = rec.uniform(&dims_bytes([out_len, k, 0, 0]))?;
                let bg = bind_group(engine, &pipeline, vec![matrix.binding(), vector.binding(), c.binding(), uniform]);
                rec.dispatch(&pipeline, &bg, grid, bytes);
                Ok(())
            });
            let trapped = trap.finish();
            recorded.and(trapped)
        }
        Operands::Gemm { pretranspose_a, pretranspose_b } => {
            // Scratch for in-VRAM transposition: slab regions or dedicated
            // buffers, released (quarantined) once the work is recorded.
            let scratch_a: Option<DeviceBlock> =
                if pretranspose_a { Some(mgr.allocate_device(a.size, false)?) } else { None };
            let scratch_b: Option<DeviceBlock> =
                if pretranspose_b { Some(mgr.allocate_device(b.size, false)?) } else { None };
            let transpose = if pretranspose_a || pretranspose_b { Some(transpose_pipeline(engine)?) } else { None };
            let (bm, bn) = match plan.kernel {
                Kernel::Register { cfg, .. } => (cfg.bm, cfg.bn),
                _ => (16, 16),
            };
            let grid = tiled_grid(engine, n.div_ceil(bn), m.div_ceil(bm))?;
            let slots = 1 + u32::from(pretranspose_a) + u32::from(pretranspose_b);

            // Transposes and the GEMM go into one recorded unit: one command
            // buffer, one submission (or none extra inside a batch).
            let trap = ErrorTrap::push(&engine.device);
            let recorded = engine.stream.record(slots, |rec| {
                let a_eff = match (&scratch_a, &transpose) {
                    (Some(scratch), Some(tp)) => {
                        // A stored K×M -> M×K.
                        encode_transpose(rec, engine, tp, a, scratch.span(), k, m)?;
                        scratch.span()
                    }
                    _ => a,
                };
                let b_eff = match (&scratch_b, &transpose) {
                    (Some(scratch), Some(tp)) => {
                        // B stored N×K -> K×N.
                        encode_transpose(rec, engine, tp, b, scratch.span(), n, k)?;
                        scratch.span()
                    }
                    _ => b,
                };
                let uniform = rec.uniform(&dims_bytes([m, n, k, 0]))?;
                let bg = bind_group(engine, &pipeline, vec![a_eff.binding(), b_eff.binding(), c.binding(), uniform]);
                rec.dispatch(&pipeline, &bg, grid, a_eff.size + b_eff.size + c.size);
                Ok(())
            });
            let trapped = trap.finish();
            drop((scratch_a, scratch_b));
            recorded.and(trapped)
        }
    }
}

pub fn run_transpose(id_in: TensorId, id_out: TensorId, rows: u32, cols: u32) -> Hc4jResult<()> {
    if rows == 0 || cols == 0 {
        return Err(Hc4jError::InvalidParam("transpose dimensions must be non-zero"));
    }
    let mgr = manager()?;
    let engine = mgr.engine();
    let bytes = checked_bytes(rows, cols)?;
    if mgr.size_of(id_in)? != bytes || mgr.size_of(id_out)? != bytes {
        return Err(Hc4jError::InvalidParam("tensor sizes do not match rows x cols"));
    }
    if id_in == id_out {
        return Err(Hc4jError::InvalidParam("transpose cannot run in place"));
    }
    let resident = mgr.acquire_resident(&[id_in, id_out])?;
    let (src, dst) = (&resident.spans[0], &resident.spans[1]);
    if src.size > engine.limits.max_storage_buffer_binding_size {
        return Err(Hc4jError::Unsupported("transpose operand exceeds the storage-binding limit"));
    }
    let pipeline = transpose_pipeline(engine)?;
    let trap = ErrorTrap::push(&engine.device);
    let recorded = engine
        .stream
        .record(1, |rec| encode_transpose(rec, engine, &pipeline, src, dst, rows, cols));
    let trapped = trap.finish();
    recorded.and(trapped)
}

// ============================================================================
// FFI
// ============================================================================

/// `C[M×N] = A[M×K] · B[K×N]` into the caller-allocated `id_out`.
///
/// # Safety
/// No pointer arguments; `unsafe` matches the published signature. Handles
/// are validated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dispatch_matmul_f32(id_a: u64, id_b: u64, id_out: u64, m: u32, n: u32, k: u32) -> i32 {
    ffi_guard("dispatch_matmul_f32", || run_matmul(id_a, id_b, id_out, m, n, k, 0))
}

/// As `dispatch_matmul_f32`, with `flags` bit 0: A is stored transposed
/// (K×M), bit 1: B is stored transposed (N×K).
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_matmul_f32_ex(id_a: u64, id_b: u64, id_out: u64, m: u32, n: u32, k: u32, flags: u32) -> i32 {
    ffi_guard("dispatch_matmul_f32_ex", || run_matmul(id_a, id_b, id_out, m, n, k, flags))
}

/// `out[cols×rows] = transpose(in[rows×cols])`.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_transpose_f32(id_in: u64, id_out: u64, rows: u32, cols: u32) -> i32 {
    ffi_guard("dispatch_transpose_f32", || run_transpose(id_in, id_out, rows, cols))
}

/// The dispatcher's choice for a shape, as `Plan::code()`, or a negative
/// status code. Diagnostic: lets tests and tooling see which kernel runs.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_matmul_plan(m: u32, n: u32, k: u32, flags: u32) -> i32 {
    let mut code = 0;
    let status = ffi_guard("hc4j_matmul_plan", || {
        let engine = crate::get_engine()?;
        code = plan(&Caps::of(engine), m, n, k, flags)?.code();
        Ok(())
    });
    if status == crate::error::HC4J_SUCCESS { code } else { status }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(discrete: bool, subgroups: bool) -> Caps {
        Caps {
            subgroups,
            workgroup_storage: 32768,
            invocations: 1024,
            discrete,
            forced_tile: None,
        }
    }

    #[test]
    fn gemv_shapes_pick_gemv_kernels() {
        let c = caps(false, false);
        let p = plan(&c, 777, 1, 1000, 0).unwrap();
        assert_eq!(p.kernel, Kernel::GemvRows { lanes: 32, vec4: true, subgroup: false });
        let p = plan(&c, 300, 1, 4096, 0).unwrap();
        assert_eq!(p.kernel, Kernel::GemvRows { lanes: 256, vec4: true, subgroup: false });
        let p = plan(&caps(false, true), 300, 1, 4096, 0).unwrap();
        assert_eq!(p.kernel, Kernel::GemvRows { lanes: 256, vec4: true, subgroup: true });
        let p = plan(&c, 1, 517, 1000, 0).unwrap();
        assert_eq!(p.kernel, Kernel::GemvCols { cols: 64 });
        assert_eq!(p.operands, Operands::Gemv { matrix_is_a: false, out_len: 517, k: 1000 });
    }

    #[test]
    fn transposed_gemv_swaps_mapping_instead_of_copying() {
        let c = caps(false, false);
        // y = A^T x with A stored K×M: column mapping over A.
        let p = plan(&c, 300, 1, 200, FLAG_TRANS_A).unwrap();
        assert_eq!(p.kernel, Kernel::GemvCols { cols: 64 });
        assert_eq!(p.operands, Operands::Gemv { matrix_is_a: true, out_len: 300, k: 200 });
        // y = x B^T with B stored N×K: row mapping over B.
        let p = plan(&c, 1, 300, 200, FLAG_TRANS_B).unwrap();
        assert!(matches!(p.kernel, Kernel::GemvRows { .. }));
        assert_eq!(p.operands, Operands::Gemv { matrix_is_a: false, out_len: 300, k: 200 });
        assert_eq!(p.code() & 0x300, 0, "no pre-transposition");
    }

    #[test]
    fn gemm_shapes_pick_register_or_tiled() {
        let c = caps(false, false);
        assert_eq!(plan(&c, 256, 256, 256, 0).unwrap().kernel, Kernel::Register { cfg: TILE_64, vec4: true });
        assert_eq!(plan(&c, 130, 70, 90, 0).unwrap().kernel, Kernel::Register { cfg: TILE_64, vec4: false });
        assert_eq!(plan(&c, 67, 33, 129, 0).unwrap().kernel, Kernel::Tiled16);
        assert_eq!(plan(&c, 2, 3, 4, 0).unwrap().kernel, Kernel::Tiled16);
        // Large config only on discrete GPUs.
        assert_eq!(plan(&c, 1024, 1024, 1024, 0).unwrap().kernel, Kernel::Register { cfg: TILE_64, vec4: true });
        assert_eq!(
            plan(&caps(true, false), 1024, 1024, 1024, 0).unwrap().kernel,
            Kernel::Register { cfg: TILE_128, vec4: true }
        );
    }

    #[test]
    fn transposed_gemm_pretransposes() {
        let p = plan(&caps(false, false), 128, 128, 128, FLAG_TRANS_A | FLAG_TRANS_B).unwrap();
        assert_eq!(p.operands, Operands::Gemm { pretranspose_a: true, pretranspose_b: true });
        assert_eq!(p.code(), 1 | 0x100 | 0x200);
    }

    #[test]
    fn capability_limits_are_respected() {
        let mut c = caps(true, false);
        c.workgroup_storage = 16384; // WebGPU default: TILE_128 (16448 B) does not fit
        assert_eq!(plan(&c, 1024, 1024, 1024, 0).unwrap().kernel, Kernel::Register { cfg: TILE_64, vec4: true });
    }

    #[test]
    fn bad_arguments_are_rejected() {
        let c = caps(false, false);
        assert!(plan(&c, 0, 4, 4, 0).is_err());
        assert!(plan(&c, 4, 4, 4, 8).is_err());
    }

    #[test]
    fn tile_configs_are_consistent() {
        for cfg in [TILE_64, TILE_128] {
            assert_eq!(cfg.threads(), 256);
            assert_eq!((cfg.bm * cfg.bk / 4) % cfg.threads(), 0);
            assert_eq!((cfg.bk * cfg.bn / 4) % cfg.threads(), 0);
            assert_eq!(cfg.tn % 4, 0);
        }
        assert_eq!(TILE_64.shared_bytes(), 16 * 65 * 4 + 16 * 16 * 16);
        assert!(TILE_128.shared_bytes() > 16384);
    }
}
