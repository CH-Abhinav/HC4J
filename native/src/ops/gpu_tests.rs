//! Integration tests against the real GPU through the C ABI. Each test takes
//! a process-wide lock (the engine, budgets and counters are global) and
//! returns early if no adapter is available.

use std::sync::{Mutex, MutexGuard};

use crate::memory::{hc4j_gpu_alloc, hc4j_gpu_alloc_uninit, hc4j_gpu_download, hc4j_gpu_free, hc4j_gpu_write, manager};
use crate::ops::arithmetic::{dispatch_add_f32, dispatch_mul};
use crate::ops::elementwise::dispatch_copy_strided;
use crate::ops::exponential::dispatch_exp_f32;
use crate::ops::fusion::{hc4j_fused_dispatch, op};
use crate::ops::matmul::{FLAG_TRANS_A, FLAG_TRANS_B, dispatch_matmul_f32_ex};
use crate::ops::nn::{
    Activation, dispatch_activation_f32, dispatch_add_row_bias_f32, dispatch_cross_entropy_rows,
    dispatch_layernorm_rows_f32, dispatch_rmsnorm_rows_f32, dispatch_softmax_rows_f32,
};
use crate::ops::reduce::{
    dispatch_argmax_rows, dispatch_reduce_diff2_f32, dispatch_reduce_f32, dispatch_reduce_rows_f32,
    dispatch_reduce_square_f32,
};
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

/// f64 reference for one activation, so the GPU is compared against the
/// mathematical definition rather than against another f32 computation.
fn cpu_activation(act: Activation, x: f64, alpha: f64) -> f64 {
    match act {
        Activation::Relu => x.max(0.0),
        Activation::LeakyRelu => {
            if x > 0.0 {
                x
            } else {
                alpha * x
            }
        }
        Activation::Elu => {
            if x > 0.0 {
                x
            } else {
                alpha * (x.exp() - 1.0)
            }
        }
        Activation::Gelu => {
            let inner = 0.797_884_560_802_865_4 * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }
        Activation::Silu => x / (1.0 + (-x).exp()),
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::Softplus => x.max(0.0) + (1.0 + (-x.abs()).exp()).ln(),
    }
}

#[test]
fn activations_match_cpu_including_the_vec4_tail() {
    let Some(_guard) = gpu() else { return };
    // 1003 is not a multiple of 4, so the scalar tail path runs as well as
    // the vec4 fast path.
    let n = 1003;
    let mut host = noise(n, 11);
    // Pin down the saturating ends and the kink at zero, where the stable
    // formulations differ from the naive ones.
    host[0] = 0.0;
    host[1] = -0.0;
    host[2] = 88.0;
    host[3] = -88.0;
    host[4] = 1e-8;
    host[5] = -1e-8;
    let a = upload(&host);
    let out = hc4j_gpu_alloc_uninit(n);

    for act in [
        Activation::Relu,
        Activation::LeakyRelu,
        Activation::Elu,
        Activation::Gelu,
        Activation::Silu,
        Activation::Sigmoid,
        Activation::Softplus,
    ] {
        let alpha = 0.125f32;
        let status = unsafe {
            dispatch_activation_f32(
                a,
                out,
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                n,
                1,
                act as u32,
                alpha.to_bits(),
            )
        };
        assert_eq!(status, 0, "{} dispatch", act.name());
        let got = download(out, n);
        for (i, (&g, &x)) in got.iter().zip(host.iter()).enumerate() {
            let want = cpu_activation(act, x as f64, alpha as f64);
            let tol = 1e-5 * want.abs().max(1.0);
            assert!(
                (g as f64 - want).abs() <= tol,
                "{} at {i}: x={x} got {g} want {want}",
                act.name()
            );
            assert!(g.is_finite(), "{} at {i}: x={x} produced {g}", act.name());
        }
    }
    free(a);
    free(out);
}

#[test]
fn activations_run_in_place() {
    let Some(_guard) = gpu() else { return };
    let n = 257;
    let host = noise(n, 12);
    let a = upload(&host);
    let status = unsafe {
        dispatch_activation_f32(
            a,
            a,
            0,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            n,
            1,
            Activation::Relu as u32,
            0,
        )
    };
    assert_eq!(status, 0);
    for (i, (&g, &x)) in download(a, n).iter().zip(host.iter()).enumerate() {
        assert_eq!(g, x.max(0.0), "in-place relu at {i}");
    }
    free(a);
}

#[test]
fn full_reductions_match_cpu_at_sizes_that_need_two_passes() {
    let Some(_guard) = gpu() else { return };
    // Past 256 * 4 * 1024 elements pass 1 saturates at MAX_PARTIALS and the
    // grid-stride loop covers the rest, which is the path worth testing.
    for &n in &[1usize, 3, 256, 1024, 1_048_577] {
        let host = noise(n, 13);
        let a = upload(&host);
        let out = hc4j_gpu_alloc_uninit(1);
        for (code, name) in [(0u32, "sum"), (1, "mean"), (2, "max"), (3, "min")] {
            assert_eq!(dispatch_reduce_f32(a, out, n as u64, code), 0, "{name} dispatch");
            let got = download(out, 1)[0] as f64;
            let want: f64 = match code {
                0 => host.iter().map(|&v| v as f64).sum(),
                1 => host.iter().map(|&v| v as f64).sum::<f64>() / n as f64,
                2 => host.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64)),
                _ => host.iter().fold(f64::INFINITY, |m, &v| m.min(v as f64)),
            };
            // Tree summation error grows with log(n), not n.
            let tol = 1e-6 * (n as f64).log2().max(1.0) * want.abs().max(1.0);
            assert!((got - want).abs() <= tol, "{name} n={n}: got {got} want {want}");
        }
        free(a);
        free(out);
    }
}

#[test]
fn mse_reduces_a_difference_without_an_intermediate_tensor() {
    let Some(_guard) = gpu() else { return };
    let n = 4096;
    let (pred, target) = (noise(n, 14), noise(n, 15));
    let (a, b) = (upload(&pred), upload(&target));
    let out = hc4j_gpu_alloc_uninit(1);
    assert_eq!(dispatch_reduce_diff2_f32(a, b, out, n as u64, 1), 0);
    let got = download(out, 1)[0] as f64;
    let want = pred
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| (p as f64 - t as f64).powi(2))
        .sum::<f64>()
        / n as f64;
    assert!((got - want).abs() <= 1e-6 * want.max(1.0), "mse: got {got} want {want}");

    // Sum of squares, the other folded load.
    assert_eq!(dispatch_reduce_square_f32(a, out, n as u64, 0), 0);
    let sq = download(out, 1)[0] as f64;
    let want_sq = pred.iter().map(|&p| (p as f64).powi(2)).sum::<f64>();
    assert!((sq - want_sq).abs() <= 1e-5 * want_sq.max(1.0), "sumsq: got {sq} want {want_sq}");
    free(a);
    free(b);
    free(out);
}

#[test]
fn row_reductions_and_argmax_match_cpu() {
    let Some(_guard) = gpu() else { return };
    // cols > 256 forces the strided walk; rows > 1 exercises the grid.
    let (rows, cols) = (37usize, 613usize);
    let host = noise(rows * cols, 16);
    let a = upload(&host);
    let out = hc4j_gpu_alloc_uninit(rows);

    for (code, name) in [(0u32, "sum"), (1, "mean"), (2, "max"), (3, "min")] {
        assert_eq!(dispatch_reduce_rows_f32(a, out, rows as u32, cols as u32, code), 0, "{name}");
        let got = download(out, rows);
        for r in 0..rows {
            let row = &host[r * cols..(r + 1) * cols];
            let want: f64 = match code {
                0 => row.iter().map(|&v| v as f64).sum(),
                1 => row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64,
                2 => row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64)),
                _ => row.iter().fold(f64::INFINITY, |m, &v| m.min(v as f64)),
            };
            let tol = 1e-5 * want.abs().max(1.0);
            assert!((got[r] as f64 - want).abs() <= tol, "{name} row {r}: got {} want {want}", got[r]);
        }
    }

    let idx = hc4j_gpu_alloc_uninit(rows);
    assert_eq!(dispatch_argmax_rows(a, idx, rows as u32, cols as u32, 1), 0);
    let mut got_idx = vec![0i32; rows];
    assert_eq!(
        unsafe { hc4j_gpu_download(idx, got_idx.as_mut_ptr().cast::<f32>(), rows) },
        0
    );
    for r in 0..rows {
        let row = &host[r * cols..(r + 1) * cols];
        let want = row
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
            .0;
        assert_eq!(got_idx[r] as usize, want, "argmax row {r}");
    }
    free(a);
    free(out);
    free(idx);
}

#[test]
fn softmax_is_stable_and_sums_to_one() {
    let Some(_guard) = gpu() else { return };
    let (rows, cols) = (11usize, 500usize);
    let mut host = noise(rows * cols, 17);
    // A row whose logits would overflow exp() without the max subtraction.
    for (c, slot) in host[..cols].iter_mut().enumerate() {
        *slot = 90.0 + c as f32;
    }
    let a = upload(&host);
    let out = hc4j_gpu_alloc_uninit(rows * cols);

    assert_eq!(dispatch_softmax_rows_f32(a, out, rows as u32, cols as u32, 0), 0);
    let got = download(out, rows * cols);
    for r in 0..rows {
        let row = &host[r * cols..(r + 1) * cols];
        let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        let exps: Vec<f64> = row.iter().map(|&v| (v as f64 - m).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let mut total = 0.0f64;
        for c in 0..cols {
            let want = exps[c] / sum;
            let g = got[r * cols + c] as f64;
            assert!(g.is_finite(), "softmax row {r} col {c} is {g}");
            assert!((g - want).abs() <= 1e-6, "softmax row {r} col {c}: got {g} want {want}");
            total += g;
        }
        assert!((total - 1.0).abs() < 1e-4, "softmax row {r} sums to {total}");
    }

    // log-softmax over the same data.
    assert_eq!(dispatch_softmax_rows_f32(a, out, rows as u32, cols as u32, 1), 0);
    let got_log = download(out, rows * cols);
    for r in 0..rows {
        let row = &host[r * cols..(r + 1) * cols];
        let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        let lse = m + row.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln();
        for c in 0..cols {
            let want = row[c] as f64 - lse;
            let g = got_log[r * cols + c] as f64;
            assert!((g - want).abs() <= 1e-4, "log_softmax row {r} col {c}: got {g} want {want}");
        }
    }
    free(a);
    free(out);
}

#[test]
fn layer_norm_and_rms_norm_match_cpu() {
    let Some(_guard) = gpu() else { return };
    let (rows, cols) = (9usize, 777usize);
    let host = noise(rows * cols, 18);
    let gamma: Vec<f32> = (0..cols).map(|c| 0.5 + (c % 7) as f32 * 0.1).collect();
    let beta: Vec<f32> = (0..cols).map(|c| -0.25 + (c % 5) as f32 * 0.05).collect();
    let (a, g, b) = (upload(&host), upload(&gamma), upload(&beta));
    let out = hc4j_gpu_alloc_uninit(rows * cols);
    let eps = 1e-5f32;

    assert_eq!(
        dispatch_layernorm_rows_f32(a, g, b, out, rows as u32, cols as u32, eps.to_bits()),
        0
    );
    let got = download(out, rows * cols);
    for r in 0..rows {
        let row = &host[r * cols..(r + 1) * cols];
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / cols as f64;
        let inv = 1.0 / (var + eps as f64).sqrt();
        for c in 0..cols {
            let want = (row[c] as f64 - mean) * inv * gamma[c] as f64 + beta[c] as f64;
            let g = got[r * cols + c] as f64;
            assert!((g - want).abs() <= 1e-4, "layernorm row {r} col {c}: got {g} want {want}");
        }
    }

    // Without the affine transform each row must come out zero-mean and
    // unit-variance.
    assert_eq!(
        dispatch_layernorm_rows_f32(a, 0, 0, out, rows as u32, cols as u32, eps.to_bits()),
        0
    );
    let plain = download(out, rows * cols);
    for r in 0..rows {
        let row = &plain[r * cols..(r + 1) * cols];
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / cols as f64;
        assert!(mean.abs() < 1e-4, "layernorm row {r} mean {mean}");
        assert!((var - 1.0).abs() < 1e-3, "layernorm row {r} var {var}");
    }

    assert_eq!(
        dispatch_rmsnorm_rows_f32(a, g, out, rows as u32, cols as u32, eps.to_bits()),
        0
    );
    let rms = download(out, rows * cols);
    for r in 0..rows {
        let row = &host[r * cols..(r + 1) * cols];
        let ms = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / cols as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        for c in 0..cols {
            let want = row[c] as f64 * inv * gamma[c] as f64;
            let g = rms[r * cols + c] as f64;
            assert!((g - want).abs() <= 1e-4, "rmsnorm row {r} col {c}: got {g} want {want}");
        }
    }
    free(a);
    free(g);
    free(b);
    free(out);
}

#[test]
fn layer_norm_rejects_half_an_affine_pair() {
    let Some(_guard) = gpu() else { return };
    let (rows, cols) = (2usize, 8usize);
    let a = upload(&noise(rows * cols, 19));
    let g = upload(&vec![1.0; cols]);
    let out = hc4j_gpu_alloc_uninit(rows * cols);
    assert_eq!(
        dispatch_layernorm_rows_f32(a, g, 0, out, rows as u32, cols as u32, 0),
        crate::error::HC4J_ERR_INVALID_PARAM
    );
    free(a);
    free(g);
    free(out);
}

#[test]
fn cross_entropy_matches_cpu_and_flags_bad_classes() {
    let Some(_guard) = gpu() else { return };
    let (rows, cols) = (13usize, 101usize);
    let logits = noise(rows * cols, 20);
    let targets: Vec<i32> = (0..rows).map(|r| (r * 7 % cols) as i32).collect();
    let a = upload(&logits);
    let t = hc4j_gpu_alloc(rows);
    assert_eq!(unsafe { hc4j_gpu_write(t, targets.as_ptr().cast::<f32>(), rows) }, 0);
    let out = hc4j_gpu_alloc_uninit(rows);

    assert_eq!(dispatch_cross_entropy_rows(a, t, out, rows as u32, cols as u32), 0);
    let got = download(out, rows);
    for r in 0..rows {
        let row = &logits[r * cols..(r + 1) * cols];
        let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        let lse = m + row.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln();
        let want = lse - row[targets[r] as usize] as f64;
        assert!(
            (got[r] as f64 - want).abs() <= 1e-5,
            "cross entropy row {r}: got {} want {want}",
            got[r]
        );
    }

    // An out-of-range class must surface, not read a neighbouring row.
    let bad: Vec<i32> = vec![cols as i32; rows];
    assert_eq!(unsafe { hc4j_gpu_write(t, bad.as_ptr().cast::<f32>(), rows) }, 0);
    assert_eq!(dispatch_cross_entropy_rows(a, t, out, rows as u32, cols as u32), 0);
    assert!(download(out, rows).iter().all(|v| v.is_nan()), "bad class must give NaN");
    free(a);
    free(t);
    free(out);
}

#[test]
fn row_bias_broadcasts_and_runs_in_place() {
    let Some(_guard) = gpu() else { return };
    let (rows, cols) = (64usize, 71usize);
    let host = noise(rows * cols, 21);
    let bias = noise(cols, 22);
    let (a, b) = (upload(&host), upload(&bias));
    let out = hc4j_gpu_alloc_uninit(rows * cols);

    assert_eq!(dispatch_add_row_bias_f32(a, b, out, rows as u32, cols as u32), 0);
    for (i, &g) in download(out, rows * cols).iter().enumerate() {
        let want = host[i] + bias[i % cols];
        assert!((g - want).abs() <= 1e-6, "bias at {i}: got {g} want {want}");
    }

    // In place: the output aliases the input.
    assert_eq!(dispatch_add_row_bias_f32(a, b, a, rows as u32, cols as u32), 0);
    for (i, &g) in download(a, rows * cols).iter().enumerate() {
        let want = host[i] + bias[i % cols];
        assert!((g - want).abs() <= 1e-6, "in-place bias at {i}: got {g} want {want}");
    }
    free(a);
    free(b);
    free(out);
}

#[test]
fn an_mlp_layer_composes_gemm_bias_and_activation() {
    let Some(_guard) = gpu() else { return };
    // y = gelu(x . w + b), the shape of every feed-forward block.
    let (m, k, n) = (24usize, 40usize, 16usize);
    let (x, w, b) = (noise(m * k, 23), noise(k * n, 24), noise(n, 25));
    let (idx, idw, idb) = (upload(&x), upload(&w), upload(&b));
    let out = hc4j_gpu_alloc_uninit(m * n);

    assert_eq!(dispatch_matmul_f32_ex(idx, idw, out, m as u32, n as u32, k as u32, 0), 0);
    assert_eq!(dispatch_add_row_bias_f32(out, idb, out, m as u32, n as u32), 0);
    let status = unsafe {
        dispatch_activation_f32(
            out,
            out,
            0,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            m * n,
            1,
            Activation::Gelu as u32,
            0,
        )
    };
    assert_eq!(status, 0);

    let got = download(out, m * n);
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f64;
            for i in 0..k {
                acc += x[r * k + i] as f64 * w[i * n + c] as f64;
            }
            let want = cpu_activation(Activation::Gelu, acc + b[c] as f64, 0.0);
            let g = got[r * n + c] as f64;
            assert!((g - want).abs() <= 1e-4, "mlp at ({r},{c}): got {g} want {want}");
        }
    }
    free(idx);
    free(idw);
    free(idb);
    free(out);
}

#[test]
fn a_cancelled_zero_fill_leaves_no_phantom_epoch() {
    let Some(_gpu) = gpu() else { return };
    let engine = get_engine().unwrap();
    let mgr = manager().unwrap();
    engine.stream.flush();

    // hc4j_gpu_alloc zeroes, which queues a clear; writing the whole tensor
    // cancels it again. That used to leave `pending` set with nothing to
    // submit, so retire_epoch named an epoch no submission would claim and
    // any region quarantined against it could never be reclaimed.
    let n = 1000;
    let host = noise(n, 44);
    let id = hc4j_gpu_alloc(n);
    assert_ne!(id, 0, "alloc failed");
    assert_eq!(unsafe { hc4j_gpu_write(id, host.as_ptr(), n) }, 0);

    // The invariant: whatever retire_epoch hands out, wait_epoch can satisfy.
    let epoch = engine.stream.retire_epoch();
    engine.stream.wait_epoch(epoch).expect("retire_epoch must name a reachable epoch");

    free(id);
    let budget = mgr.stats().vram_budget;
    mgr.configure(budget, 0).expect("quarantine must drain");
    assert_eq!(mgr.stats().quarantined, 0, "freed region should be back in its slab");
}

fn upload_i32(data: &[i32]) -> u64 {
    let id = hc4j_gpu_alloc(data.len());
    assert_ne!(id, 0, "alloc failed");
    assert_eq!(unsafe { hc4j_gpu_write(id, data.as_ptr().cast::<f32>(), data.len()) }, 0);
    id
}

fn download_i32(id: u64, len: usize) -> Vec<i32> {
    let mut out = vec![0i32; len];
    assert_eq!(unsafe { hc4j_gpu_download(id, out.as_mut_ptr().cast::<f32>(), len) }, 0);
    out
}

#[test]
fn strided_copy_transpose_broadcast_and_offset() {
    let Some(_gpu) = gpu() else { return };

    // 1. 2D Transpose copy on f32 (3x4 -> 4x3)
    let src_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let src_id = upload(&src_data);
    let out_id = hc4j_gpu_alloc_uninit(12);

    let shape = [4u32, 3u32];
    let src_strides = [1u32, 4u32]; // row-major [3, 4] transposed is stride [1, 4]
    let status = unsafe {
        dispatch_copy_strided(src_id, out_id, 0, 2, shape.as_ptr(), src_strides.as_ptr(), 12)
    };
    assert_eq!(status, 0);
    let got = download(out_id, 12);
    // Transposed 3x4:
    // [ 1,  2,  3,  4 ]
    // [ 5,  6,  7,  8 ]
    // [ 9, 10, 11, 12 ]
    // -> 4x3:
    // col 0: 1, 5, 9
    // col 1: 2, 6, 10
    // col 2: 3, 7, 11
    // col 3: 4, 8, 12
    let expected = vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];
    assert_eq!(got, expected);
    free(out_id);

    // 2. Non-zero offset slice (row 1 of 3x4, offset = 4, length = 4)
    let out_slice = hc4j_gpu_alloc_uninit(4);
    let slice_shape = [4u32];
    let slice_strides = [1u32];
    let status_slice = unsafe {
        dispatch_copy_strided(src_id, out_slice, 4, 1, slice_shape.as_ptr(), slice_strides.as_ptr(), 4)
    };
    assert_eq!(status_slice, 0);
    let got_slice = download(out_slice, 4);
    assert_eq!(got_slice, vec![5.0, 6.0, 7.0, 8.0]);
    free(out_slice);

    // 3. Broadcast stride-0 copy (1x4 broadcast to 3x4)
    let out_bcast = hc4j_gpu_alloc_uninit(12);
    let bcast_shape = [3u32, 4u32];
    let bcast_strides = [0u32, 1u32]; // row 0 repeated 3 times
    let status_bcast = unsafe {
        dispatch_copy_strided(src_id, out_bcast, 0, 2, bcast_shape.as_ptr(), bcast_strides.as_ptr(), 12)
    };
    assert_eq!(status_bcast, 0);
    let got_bcast = download(out_bcast, 12);
    let expected_bcast = vec![
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0,
    ];
    assert_eq!(got_bcast, expected_bcast);
    free(out_bcast);
    free(src_id);

    // 4. i32 strided copy with negative & large values
    let src_i32: Vec<i32> = vec![-100, 200, -300, 400, -500, 600];
    let src_i32_id = upload_i32(&src_i32);
    let out_i32_id = hc4j_gpu_alloc_uninit(6);
    let i32_shape = [3u32, 2u32];
    let i32_strides = [1u32, 3u32]; // transpose 2x3 -> 3x2
    let status_i32 = unsafe {
        dispatch_copy_strided(src_i32_id, out_i32_id, 0, 2, i32_shape.as_ptr(), i32_strides.as_ptr(), 6)
    };
    assert_eq!(status_i32, 0);
    let got_i32 = download_i32(out_i32_id, 6);
    // [-100, 200, -300]
    // [-500, 600,  ---] wait, 2x3:
    // row 0: -100, 200, -300
    // row 1: 400, -500, 600
    // col 0: -100, 400
    // col 1: 200, -500
    // col 2: -300, 600
    assert_eq!(got_i32, vec![-100, 400, 200, -500, -300, 600]);
    free(src_i32_id);
    free(out_i32_id);
}

#[test]
fn strided_broadcast_binary_ops() {
    let Some(_gpu) = gpu() else { return };

    // [4, 1] + [1, 5] -> [4, 5]
    let a_data = vec![10.0f32, 20.0, 30.0, 40.0];
    let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let a_id = upload(&a_data);
    let b_id = upload(&b_data);
    let out_id = hc4j_gpu_alloc_uninit(20);

    let shape = [4u32, 5u32];
    let strides_a = [1u32, 0u32]; // [4, 1] broadcast along dim 1
    let strides_b = [0u32, 1u32]; // [1, 5] broadcast along dim 0
    let strides_c = [5u32, 1u32];

    let status = unsafe {
        dispatch_add_f32(
            a_id, b_id, out_id, 2,
            shape.as_ptr(), strides_a.as_ptr(), strides_b.as_ptr(), strides_c.as_ptr(),
            20, 0,
        )
    };
    assert_eq!(status, 0);
    let got = download(out_id, 20);
    for r in 0..4 {
        for c in 0..5 {
            let expected = a_data[r] + b_data[c];
            assert_eq!(got[r * 5 + c], expected);
        }
    }
    free(a_id);
    free(b_id);
    free(out_id);

    // [2, 3, 4] * [4] -> [2, 3, 4]
    let total = 2 * 3 * 4;
    let x_data: Vec<f32> = (1..=total).map(|v| v as f32).collect();
    let y_data = vec![2.0f32, 3.0, 4.0, 5.0];
    let x_id = upload(&x_data);
    let y_id = upload(&y_data);
    let mul_out = hc4j_gpu_alloc_uninit(total);

    let shape3d = [2u32, 3u32, 4u32];
    let strides_x = [12u32, 4u32, 1u32];
    let strides_y = [0u32, 0u32, 1u32];
    let strides_out = [12u32, 4u32, 1u32];

    let status_mul = unsafe {
        dispatch_mul(
            x_id, y_id, mul_out, 3,
            shape3d.as_ptr(), strides_x.as_ptr(), strides_y.as_ptr(), strides_out.as_ptr(),
            total, 0, 1, // dtype 1 = f32
        )
    };
    assert_eq!(status_mul, 0);
    let got_mul = download(mul_out, total);
    for i in 0..total {
        let expected = x_data[i] * y_data[i % 4];
        assert_eq!(got_mul[i], expected);
    }
    free(x_id);
    free(y_id);
    free(mul_out);
}

#[test]
fn in_place_self_aliasing_and_mutations() {
    let Some(_gpu) = gpu() else { return };
    let n = 256;
    let host: Vec<f32> = (1..=n).map(|x| x as f32).collect();
    let a = upload(&host);

    // a.addi(a) -> a = a + a
    let status = unsafe {
        dispatch_add_f32(
            a, a, a, 0,
            std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
            n, 1,
        )
    };
    assert_eq!(status, 0);
    let got = download(a, n);
    for i in 0..n {
        assert_eq!(got[i], host[i] * 2.0);
    }

    // a.muli(2.0 via a itself: a = a * a)
    let status_mul = unsafe {
        dispatch_mul(
            a, a, a, 0,
            std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(),
            n, 1, 1,
        )
    };
    assert_eq!(status_mul, 0);
    let got_mul = download(a, n);
    for i in 0..n {
        let doubled = host[i] * 2.0;
        assert_eq!(got_mul[i], doubled * doubled);
    }

    free(a);
}

