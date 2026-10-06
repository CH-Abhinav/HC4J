//! Integration tests against the real GPU through the C ABI. Each test takes
//! a process-wide lock (the engine, budgets and counters are global) and
//! returns early if no adapter is available.

use std::sync::{Mutex, MutexGuard};

use crate::memory::{hc4j_gpu_alloc, hc4j_gpu_alloc_uninit, hc4j_gpu_download, hc4j_gpu_free, hc4j_gpu_write, manager};
use crate::ops::arithmetic::dispatch_add_f32;
use crate::ops::exponential::dispatch_exp_f32;
use crate::ops::fusion::{hc4j_fused_dispatch, op};
use crate::ops::matmul::{FLAG_TRANS_A, FLAG_TRANS_B, dispatch_matmul_f32_ex};
use crate::ops::trigno::{dispatch_cos_f32, dispatch_sin_f32};
use crate::{get_engine, hc4j_batch_begin, hc4j_batch_end};

static GPU: Mutex<()> = Mutex::new(());

/// WGSL guarantees sin/cos only to 2^-11 absolute error on [-pi, pi]; Intel
/// hardware measures ~2e-5. Tighter tolerances would test the driver.
const TRIG_TOL: f32 = 1.0 / 2048.0;

fn gpu() -> Option<MutexGuard<'static, ()>> {
    let guard = GPU.lock().unwrap_or_else(|p| p.into_inner());
    get_engine().ok().map(|_| guard)
}

fn upload(data: &[f32]) -> u64 {
    let id = hc4j_gpu_alloc(data.len());
    assert_ne!(id, 0, "alloc failed");
    assert_eq!(unsafe { hc4j_gpu_write(id, data.as_ptr(), data.len()) }, 0);
    id
}

fn download(id: u64, len: usize) -> Vec<f32> {
    let mut out = vec![0f32; len];
    assert_eq!(unsafe { hc4j_gpu_download(id, out.as_mut_ptr(), len) }, 0);
    out
}

fn free(id: u64) {
    assert_eq!(hc4j_gpu_free(id), 0);
}

/// Deterministic values in [-1, 1).
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn contiguous_unary(f: unsafe extern "C" fn(u64, u64, u32, *const u32, *const u32, *const u32, usize, u32) -> i32, a: u64, out: u64, n: usize) -> i32 {
    unsafe { f(a, out, 0, std::ptr::null(), std::ptr::null(), std::ptr::null(), n, 1) }
}

#[test]
fn strided_add_reads_a_transposed_view() {
    let Some(_gpu) = gpu() else { return };
    let (rows, cols) = (3usize, 5usize);
    let a: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    // b is stored 5x3; read through strides [1, 3] it is its own 3x5 transpose.
    let b_stored: Vec<f32> = (0..rows * cols).map(|i| 100.0 * i as f32).collect();
    let (ia, ib) = (upload(&a), upload(&b_stored));
    let out = hc4j_gpu_alloc(rows * cols);
    let shape = [rows as u32, cols as u32];
    let strides_a = [cols as u32, 1];
    let strides_b = [1u32, rows as u32];
    let strides_c = [cols as u32, 1];
    let status = unsafe {
        dispatch_add_f32(
            ia, ib, out, 2,
            shape.as_ptr(), strides_a.as_ptr(), strides_b.as_ptr(), strides_c.as_ptr(),
            rows * cols, 0,
        )
    };
    assert_eq!(status, 0);
    let got = download(out, rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(got[i * cols + j], a[i * cols + j] + b_stored[j * rows + i], "({i},{j})");
        }
    }
    for id in [ia, ib, out] {
        free(id);
    }
}

#[test]
fn both_grid_paths_fold_past_the_67m_workgroup_limit() {
    let Some(_gpu) = gpu() else { return };
    // 67.2M elements > 65535 * 1024: the contiguous path needs 65,625
    // workgroups and the strided path (one element per thread) 262,500.
    let n = 67_200_000usize;
    let host: Vec<f32> = (0..n).map(|i| (i % 1000) as f32 * 0.001 - 0.5).collect();
    let a = upload(&host);
    let out = hc4j_gpu_alloc_uninit(n);

    assert_eq!(contiguous_unary(dispatch_exp_f32, a, out, n), 0);
    let got = download(out, n);
    let bad = (0..n).filter(|&i| (got[i] - host[i].exp()).abs() > 1e-5).count();
    assert_eq!(bad, 0, "contiguous: {bad} wrong elements (first/last must be covered)");

    let shape = [n as u32];
    let unit = [1u32];
    let status = unsafe { dispatch_exp_f32(a, out, 1, shape.as_ptr(), unit.as_ptr(), unit.as_ptr(), n, 0) };
    assert_eq!(status, 0);
    let got = download(out, n);
    let bad = (0..n).filter(|&i| (got[i] - host[i].exp()).abs() > 1e-5).count();
    assert_eq!(bad, 0, "strided: {bad} wrong elements");
    free(a);
    free(out);
}

#[test]
fn in_place_elementwise_ops() {
    let Some(_gpu) = gpu() else { return };
    let n = 10_007;
    let host = noise(n, 1);
    let a = upload(&host);
    assert_eq!(contiguous_unary(dispatch_sin_f32, a, a, n), 0);
    let got = download(a, n);
    assert!(got.iter().zip(&host).all(|(g, h)| (g - h.sin()).abs() < TRIG_TOL));

    let b = upload(&host);
    let status = unsafe {
        dispatch_add_f32(a, b, a, 0, std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(), n, 1)
    };
    assert_eq!(status, 0);
    let got = download(a, n);
    assert!(got.iter().zip(&host).all(|(g, h)| (g - (h.sin() + h)).abs() < TRIG_TOL));
    free(a);
    free(b);
}

#[test]
fn fused_dispatch_matches_cpu() {
    let Some(_gpu) = gpu() else { return };
    let n = 1003;
    let (x0, x1, x2) = (noise(n, 2), noise(n, 3), noise(n, 4));
    let ids = [upload(&x0), upload(&x1), upload(&x2)];
    let out = hc4j_gpu_alloc_uninit(n);
    let program = [
        op::word(op::LOAD, 0),
        op::word(op::SIN, 0),
        op::word(op::LOAD, 1),
        op::word(op::ADD, 0),
        op::word(op::LOAD, 2),
        op::word(op::MUL, 0),
        op::word(op::CONST, 0),
        2.0f32.to_bits(),
        op::word(op::ADD, 0),
    ];
    let status = unsafe { hc4j_fused_dispatch(program.as_ptr(), program.len() as u32, ids.as_ptr(), 3, out) };
    assert_eq!(status, 0);
    let got = download(out, n);
    for i in 0..n {
        let want = (x0[i].sin() + x1[i]) * x2[i] + 2.0;
        // sin error (2^-11) scaled by |x2| < 1, plus rounding.
        assert!((got[i] - want).abs() < TRIG_TOL + 1e-5, "i={i} got={} want={want}", got[i]);
    }
    // A malformed program is an error code, not a crash.
    let bad = [op::word(op::ADD, 0)];
    assert!(unsafe { hc4j_fused_dispatch(bad.as_ptr(), 1, ids.as_ptr(), 3, out) } < 0);
    for id in ids.into_iter().chain([out]) {
        free(id);
    }
}

/// Exact-ish reference (f64) plus, per element, the sum of |a·b| terms that
/// bounds f32 accumulation error: |fl(Σ) - Σ| <= k·u·Σ|a·b| for any order.
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, ta: bool, tb: bool) -> Vec<(f64, f64)> {
    let at = |i: usize, p: usize| if ta { a[p * m + i] } else { a[i * k + p] } as f64;
    let bt = |p: usize, j: usize| if tb { b[j * k + p] } else { b[p * n + j] } as f64;
    let mut c = vec![(0f64, 0f64); m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).fold((0.0, 0.0), |(s, abs), p| {
                let t = at(i, p) * bt(p, j);
                (s + t, abs + t.abs())
            });
        }
    }
    c
}

#[test]
fn matmul_kernels_match_cpu() {
    let Some(_gpu) = gpu() else { return };
    let shapes = [(67, 33, 129), (128, 128, 128), (130, 70, 90), (1, 300, 257), (300, 1, 1500), (1, 1, 4096)];
    for (si, &(m, n, k)) in shapes.iter().enumerate() {
        for flags in [0, FLAG_TRANS_A, FLAG_TRANS_B, FLAG_TRANS_A | FLAG_TRANS_B] {
            let a = noise(m * k, 10 + si as u64);
            let b = noise(k * n, 20 + si as u64);
            let (ia, ib) = (upload(&a), upload(&b));
            let out = hc4j_gpu_alloc_uninit(m * n);
            let status = dispatch_matmul_f32_ex(ia, ib, out, m as u32, n as u32, k as u32, flags);
            assert_eq!(status, 0, "{m}x{n}x{k} flags={flags}");
            let got = download(out, m * n);
            let want = cpu_matmul(&a, &b, m, n, k, flags & 1 != 0, flags & 2 != 0);
            let unit = f32::EPSILON as f64 / 2.0;
            for (idx, (&g, &(w, abs))) in got.iter().zip(&want).enumerate() {
                let tol = 2.0 * k as f64 * unit * abs + 1e-6;
                assert!((g as f64 - w).abs() <= tol, "{m}x{n}x{k} flags={flags} @{idx}: got {g} want {w} tol {tol}");
            }
            for id in [ia, ib, out] {
                free(id);
            }
        }
    }
}

#[test]
fn small_tensors_use_slab_regions_and_quarantine_until_retired() {
    let Some(_gpu) = gpu() else { return };
    let mgr = manager().unwrap();
    let n = 1000;
    let x = upload(&noise(n, 5));
    let before = mgr.stats();

    assert_eq!(hc4j_batch_begin(), 0);
    let first = hc4j_gpu_alloc_uninit(n);
    assert_eq!(contiguous_unary(dispatch_sin_f32, x, first, n), 0);
    // Freed while its dispatch is still only recorded: must be quarantined.
    free(first);
    assert!(mgr.stats().quarantined > before.quarantined, "freed region must wait for its epoch");
    let second = hc4j_gpu_alloc_uninit(n);
    assert_eq!(contiguous_unary(dispatch_cos_f32, x, second, n), 0);
    assert_eq!(hc4j_batch_end(), 0);

    let after = mgr.stats();
    assert!(after.region_allocs >= before.region_allocs + 2, "small outputs come from slabs");
    assert_eq!(after.dedicated_buffers, before.dedicated_buffers, "no create_buffer per op");
    let host = download(x, n);
    let got = download(second, n);
    assert!(got.iter().zip(&host).all(|(g, h)| (g - h.cos()).abs() < TRIG_TOL));

    // Once the batch retires, the quarantine drains on the next allocation.
    get_engine().unwrap().stream.wait_epoch(get_engine().unwrap().stream.retire_epoch()).unwrap();
    let probe = hc4j_gpu_alloc_uninit(n);
    assert_eq!(mgr.stats().quarantined, 0);
    for id in [x, second, probe] {
        free(id);
    }
}

#[test]
fn a_batch_is_one_submission() {
    let Some(_gpu) = gpu() else { return };
    let engine = get_engine().unwrap();
    let n = 4096;
    let x = upload(&noise(n, 6));
    let out = hc4j_gpu_alloc_uninit(n);
    engine.stream.flush();
    let before = engine.stream.stats();
    assert_eq!(hc4j_batch_begin(), 0);
    for i in 0..40 {
        let f = if i % 2 == 0 { dispatch_sin_f32 } else { dispatch_cos_f32 };
        assert_eq!(contiguous_unary(f, if i == 0 { x } else { out }, out, n), 0);
    }
    assert_eq!(hc4j_batch_end(), 0);
    let after = engine.stream.stats();
    assert_eq!(after.dispatches - before.dispatches, 40);
    assert_eq!(after.submissions - before.submissions, 1, "40 ops, one queue.submit");

    let mut want: Vec<f32> = download(x, n);
    for i in 0..40 {
        for v in &mut want {
            *v = if i % 2 == 0 { v.sin() } else { v.cos() };
        }
    }
    let got = download(out, n);
    assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 40.0 * TRIG_TOL));
    free(x);
    free(out);
}

#[test]
fn matmul_streams_when_the_working_set_exceeds_vram() {
    let Some(_gpu) = gpu() else { return };
    let mgr = manager().unwrap();
    let restore = mgr.stats().vram_budget;
    // 1 MiB of VRAM: B (256 KiB) stays resident, A and C (2 MiB each) cannot.
    mgr.configure(1 << 20, 0).unwrap();
    let before = mgr.stats().streamed_ops;

    let (m, n, k) = (2048usize, 256usize, 256usize);
    let a = noise(m * k, 31);
    let b = noise(k * n, 32);
    let (ia, ib) = (upload(&a), upload(&b));
    let out = hc4j_gpu_alloc_uninit(m * n);
    assert_ne!(crate::memory::hc4j_mem_residency(ia), 0, "A should not be VRAM-resident at this budget");

    for flags in [0, FLAG_TRANS_B] {
        let status = dispatch_matmul_f32_ex(ia, ib, out, m as u32, n as u32, k as u32, flags);
        assert_eq!(status, 0, "flags={flags}");
        let got = download(out, m * n);
        let want = cpu_matmul(&a, &b, m, n, k, false, flags & 2 != 0);
        let unit = f32::EPSILON as f64 / 2.0;
        for (idx, (&g, &(w, abs))) in got.iter().zip(&want).enumerate() {
            let tol = 2.0 * k as f64 * unit * abs + 1e-6;
            assert!((g as f64 - w).abs() <= tol, "flags={flags} @{idx}: got {g} want {w}");
        }
    }
    // A streamed GEMV: 8 MiB of A, one row block at a time.
    let (gm, gk) = (4096usize, 512usize);
    let ga = noise(gm * gk, 33);
    let gx = noise(gk, 34);
    let (iga, igx) = (upload(&ga), upload(&gx));
    let gout = hc4j_gpu_alloc_uninit(gm);
    assert_eq!(dispatch_matmul_f32_ex(iga, igx, gout, gm as u32, 1, gk as u32, 0), 0);
    let got = download(gout, gm);
    let want = cpu_matmul(&ga, &gx, gm, 1, gk, false, false);
    let unit = f32::EPSILON as f64 / 2.0;
    for (idx, (&g, &(w, abs))) in got.iter().zip(&want).enumerate() {
        assert!((g as f64 - w).abs() <= 2.0 * gk as f64 * unit * abs + 1e-6, "gemv @{idx}");
    }

    let after = mgr.stats().streamed_ops;
    assert!(after >= before + 3, "streaming path ran ({} ops)", after - before);
    // A stored-transposed A cannot be streamed: reading its logical rows would
    // need the whole matrix resident. Same A storage, read as K x M = 2048 x 256.
    let wide_b = upload(&noise(m * n, 35)); // k' = 2048 by n' = 256
    let small_out = hc4j_gpu_alloc_uninit(k * n);
    assert_eq!(
        dispatch_matmul_f32_ex(ia, wide_b, small_out, k as u32, n as u32, m as u32, FLAG_TRANS_A),
        crate::error::HC4J_ERR_UNSUPPORTED
    );

    for id in [ia, ib, out, iga, igx, gout, wide_b, small_out] {
        free(id);
    }
    mgr.configure(restore, 0).unwrap();
}

#[test]
fn matmul_blocks_over_k_when_b_does_not_fit_either() {
    let Some(_gpu) = gpu() else { return };
    let mgr = manager().unwrap();
    let restore = mgr.stats().vram_budget;
    // 256 KiB of VRAM: B alone (512 KiB) cannot be resident, so the product
    // has to be blocked over K and accumulated in VRAM.
    mgr.configure(256 << 10, 0).unwrap();
    let before = mgr.stats().streamed_ops;
    let unit = f32::EPSILON as f64 / 2.0;

    // Blocked GEMM.
    let (m, n, k) = (512usize, 256usize, 512usize);
    let a = noise(m * k, 41);
    let b = noise(k * n, 42);
    let (ia, ib) = (upload(&a), upload(&b));
    let out = hc4j_gpu_alloc_uninit(m * n);
    assert_eq!(dispatch_matmul_f32_ex(ia, ib, out, m as u32, n as u32, k as u32, 0), 0);
    let got = download(out, m * n);
    let want = cpu_matmul(&a, &b, m, n, k, false, false);
    for (idx, (&g, &(w, abs))) in got.iter().zip(&want).enumerate() {
        let tol = 2.0 * k as f64 * unit * abs + 1e-6;
        assert!((g as f64 - w).abs() <= tol, "gemm @{idx}: got {g} want {w} tol {tol}");
    }

    // Blocked GEMV: the vector itself (512 KiB) does not fit, so the row
    // kernel accumulates across K blocks.
    let (gm, gk) = (8usize, 131_072usize);
    let ga = noise(gm * gk, 43);
    let gx = noise(gk, 44);
    let (iga, igx) = (upload(&ga), upload(&gx));
    let gout = hc4j_gpu_alloc_uninit(gm);
    assert_eq!(dispatch_matmul_f32_ex(iga, igx, gout, gm as u32, 1, gk as u32, 0), 0);
    let got = download(gout, gm);
    let want = cpu_matmul(&ga, &gx, gm, 1, gk, false, false);
    for (idx, (&g, &(w, abs))) in got.iter().zip(&want).enumerate() {
        assert!((g as f64 - w).abs() <= 2.0 * gk as f64 * unit * abs + 1e-6, "gemv @{idx}: got {g} want {w}");
    }

    assert!(mgr.stats().streamed_ops >= before + 2, "blocked path ran");
    // A stored-transposed B cannot be transposed in place when it does not
    // fit: ib (512x256) serves as both A (m x k) and a stored-transposed
    // B (n x k) for m = n = 512, k = 256.
    let square_out = hc4j_gpu_alloc_uninit(m * m);
    assert_eq!(
        dispatch_matmul_f32_ex(ib, ib, square_out, m as u32, m as u32, n as u32, FLAG_TRANS_B),
        crate::error::HC4J_ERR_UNSUPPORTED
    );

    for id in [ia, ib, out, iga, igx, gout, square_out] {
        free(id);
    }
    mgr.configure(restore, 0).unwrap();
}
