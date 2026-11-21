//! Kernel selection and execution planning for matrix multiplication.

use std::sync::OnceLock;

use crate::error::{Hc4jError, Hc4jResult};
use crate::GpuEngine;

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

    pub fn fits(&self, cfg: TileConfig) -> bool {
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
