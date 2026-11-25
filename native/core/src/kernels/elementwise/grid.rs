use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::engine::GpuEngine;

pub const WORKGROUP_SIZE: u32 = 256;
pub const MAX_TILE_ELEMS: u64 = 1 << 28;

/// Splits a 1D workgroup count across two dimensions so neither exceeds
/// `max_per_dim`. The split is balanced (x = ceil(n / y)) rather than
/// x = max_per_dim: just past the limit, x = 65535 would launch nearly 2x
/// the needed workgroups, all of which then exit in the bounds check.
pub fn grid_2d(workgroups: u32, max_per_dim: u32) -> (u32, u32) {
    let max_per_dim = max_per_dim.max(1);
    if workgroups <= max_per_dim {
        return (workgroups.max(1), 1);
    }
    let y = workgroups.div_ceil(max_per_dim);
    (workgroups.div_ceil(y), y)
}

/// Grid for `threads` invocations of a 256-wide 1D kernel.
pub fn grid_for_threads(engine: &GpuEngine, threads: u64) -> Hc4jResult<(u32, u32, u32)> {
    let workgroups = u32::try_from(threads.div_ceil(WORKGROUP_SIZE as u64))
        .map_err(|_| Hc4jError::Unsupported("dispatch exceeds 2^32 workgroups"))?;
    let (x, y) = grid_2d(workgroups, engine.limits.max_compute_workgroups_per_dimension);
    if y > engine.limits.max_compute_workgroups_per_dimension {
        return Err(Hc4jError::Unsupported("dispatch too large for a 2D grid"));
    }
    Ok((x, y, 1))
}

/// Largest tile that fits one storage binding, aligned so tile offsets
/// satisfy `min_storage_buffer_offset_alignment`.
pub fn max_binding_tile(engine: &GpuEngine) -> u64 {
    let align = (engine.limits.min_storage_buffer_offset_alignment as u64).max(4);
    let cap = engine.limits.max_storage_buffer_binding_size.min(MAX_TILE_ELEMS * 4);
    (cap / align).max(1) * align
}

#[cfg(test)]
mod tests {
    use super::grid_2d;

    #[test]
    fn grid_fits_in_one_dimension() {
        assert_eq!(grid_2d(1, 65535), (1, 1));
        assert_eq!(grid_2d(0, 65535), (1, 1));
        assert_eq!(grid_2d(65535, 65535), (65535, 1));
    }

    #[test]
    fn grid_folds_just_past_the_limit_without_doubling() {
        let (x, y) = grid_2d(65_625, 65_535);
        assert_eq!(y, 2);
        assert!(x as u64 * y as u64 >= 65_625);
        assert!(x as u64 * y as u64 - 65_625 < y as u64, "surplus below one row");
    }

    #[test]
    fn grid_covers_every_workgroup_at_scale() {
        for workgroups in [65_536u32, 131_071, 524_288, 4_000_000] {
            for limit in [65_535u32, 65_536] {
                let (x, y) = grid_2d(workgroups, limit);
                assert!(x <= limit && y <= limit);
                assert!(x as u64 * y as u64 >= workgroups as u64);
            }
        }
    }
}
