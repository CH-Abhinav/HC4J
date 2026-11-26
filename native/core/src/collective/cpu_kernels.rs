//! High-throughput multi-threaded, cache-blocked SIMD CPU-in-Place compute kernels
//! supporting f32/f16 GEMM/GEMV, Transformer MLP slices (2-layer and SwiGLU),
//! and online-softmax Split-KV Decode Attention.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::kernels::linalg::{FLAG_TRANS_A, FLAG_TRANS_B};

pub const CPU_ACT_NONE: i32 = -1;
pub const CPU_ACT_RELU: i32 = 0;
pub const CPU_ACT_GELU: i32 = 3;
pub const CPU_ACT_SILU: i32 = 4;

// ============================================================================
// Persistent Low-Latency Work-Stealing Thread Pool (zero new deps)
// ============================================================================

struct PoolJob {
    generation: u64,
    total_chunks: usize,
    next_chunk: AtomicUsize,
    finished_workers: AtomicUsize,
    active_workers: usize,
    ctx_ptr: usize,
    run_chunk: unsafe fn(usize, usize),
}

struct WorkerPool {
    submit_mutex: Mutex<()>,
    state: Mutex<PoolJob>,
    cv_work: Condvar,
    cv_done: Condvar,
    num_threads: usize,
}

unsafe impl Send for WorkerPool {}
unsafe impl Sync for WorkerPool {}

fn noop_chunk(_ctx: usize, _chunk: usize) {}

static GLOBAL_POOL: OnceLock<WorkerPool> = OnceLock::new();

fn global_pool() -> &'static WorkerPool {
    GLOBAL_POOL.get_or_init(|| {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, 64);

        let pool = WorkerPool {
            submit_mutex: Mutex::new(()),
            state: Mutex::new(PoolJob {
                generation: 0,
                total_chunks: 0,
                next_chunk: AtomicUsize::new(0),
                finished_workers: AtomicUsize::new(0),
                active_workers: num_threads,
                ctx_ptr: 0,
                run_chunk: noop_chunk,
            }),
            cv_work: Condvar::new(),
            cv_done: Condvar::new(),
            num_threads,
        };

        // Leak a raw address? No—Workers can access `GLOBAL_POOL` once initialized!
        pool
    })
}

static WORKERS_STARTED: OnceLock<()> = OnceLock::new();

fn ensure_workers_started(pool: &WorkerPool) {
    WORKERS_STARTED.get_or_init(|| {
        for _ in 1..pool.num_threads {
            let _ = std::thread::Builder::new()
                .name("hc4x-cpu-worker".to_string())
                .spawn(worker_loop);
        }
    });
}

fn worker_loop() {
    let pool = global_pool();
    let mut seen_gen = 0u64;
    loop {
        let (total_chunks, active_workers, ctx_ptr, run_chunk, cur_gen) = {
            let mut guard = pool.state.lock().unwrap_or_else(|p| p.into_inner());
            while guard.generation == seen_gen {
                guard = pool.cv_work.wait(guard).unwrap_or_else(|p| p.into_inner());
            }
            seen_gen = guard.generation;
            (
                guard.total_chunks,
                guard.active_workers,
                guard.ctx_ptr,
                guard.run_chunk,
                seen_gen,
            )
        };

        // Steal chunks until all are completed.
        loop {
            let chunk = pool.state.lock().unwrap_or_else(|p| p.into_inner()).next_chunk.fetch_add(1, Ordering::Relaxed);
            if chunk >= total_chunks {
                break;
            }
            unsafe { run_chunk(ctx_ptr, chunk) };
        }

        let guard = pool.state.lock().unwrap_or_else(|p| p.into_inner());
        if guard.generation == cur_gen {
            let prev = guard.finished_workers.fetch_add(1, Ordering::AcqRel);
            if prev + 1 >= active_workers {
                pool.cv_done.notify_all();
            }
        }
    }
}

/// Executes `f(chunk_idx)` for `chunk_idx in 0..total_chunks` across CPU cores.
/// Falls back inline when `total_chunks <= 1` or total estimated cost is small.
fn parallel_chunks<F>(total_chunks: usize, run_parallel: bool, f: F)
where
    F: Fn(usize) + Sync,
{
    if total_chunks == 0 {
        return;
    }
    if total_chunks == 1 || !run_parallel {
        for c in 0..total_chunks {
            f(c);
        }
        return;
    }

    let pool = global_pool();
    if pool.num_threads <= 1 {
        for c in 0..total_chunks {
            f(c);
        }
        return;
    }

    // Avoid deadlock if a parallel invocation somehow occurs inside another submit_mutex holder.
    let Ok(_submit_guard) = pool.submit_mutex.try_lock() else {
        for c in 0..total_chunks {
            f(c);
        }
        return;
    };

    ensure_workers_started(pool);

    unsafe fn trampoline<F: Fn(usize) + Sync>(ctx: usize, chunk: usize) {
        let f_ref = unsafe { &*(ctx as *const F) };
        f_ref(chunk);
    }

    let cur_gen: u64;
    let ctx_ptr = (&f as *const F) as usize;
    {
        let mut guard = pool.state.lock().unwrap_or_else(|p| p.into_inner());
        guard.generation = guard.generation.wrapping_add(1);
        cur_gen = guard.generation;
        guard.total_chunks = total_chunks;
        guard.next_chunk.store(0, Ordering::Release);
        guard.finished_workers.store(0, Ordering::Release);
        guard.active_workers = pool.num_threads;
        guard.ctx_ptr = ctx_ptr;
        guard.run_chunk = trampoline::<F>;
        pool.cv_work.notify_all();
    }

    // Calling thread executes chunks as worker 0.
    loop {
        let chunk = pool
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .next_chunk
            .fetch_add(1, Ordering::Relaxed);
        if chunk >= total_chunks {
            break;
        }
        f(chunk);
    }

    let mut guard = pool.state.lock().unwrap_or_else(|p| p.into_inner());
    let prev = guard.finished_workers.fetch_add(1, Ordering::AcqRel);
    if prev + 1 < pool.num_threads {
        while guard.generation == cur_gen && guard.finished_workers.load(Ordering::Acquire) < pool.num_threads {
            guard = pool.cv_done.wait(guard).unwrap_or_else(|p| p.into_inner());
        }
    }
}

// ============================================================================
// Portable IEEE-754 Half-Precision & Activations
// ============================================================================

#[inline]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mant = (bits & 0x03FF) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            let mut m = mant;
            let mut e = 127u32 - 14u32;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            f32::from_bits(sign | (e << 23) | (m << 13))
        }
    } else if exp == 31 {
        f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
    } else {
        let e = (exp as u32) + (127 - 15);
        f32::from_bits(sign | (e << 23) | (mant << 13))
    }
}

#[inline]
pub fn apply_activation_f32(x: f32, act_code: i32) -> Hc4jResult<f32> {
    match act_code {
        CPU_ACT_NONE => Ok(x),
        CPU_ACT_RELU => Ok(x.max(0.0)),
        CPU_ACT_GELU => {
            let cubic = 0.797_884_6_f32 * (x + 0.044_715_f32 * x * x * x);
            let inner = cubic.clamp(-10.0, 10.0);
            Ok(0.5 * x * (1.0 + inner.tanh()))
        }
        CPU_ACT_SILU => {
            let e = (-x.abs()).exp();
            let num = if x >= 0.0 { 1.0 } else { e };
            Ok(x * num / (1.0 + e))
        }
        _ => Err(Hc4jError::InvalidParam("unsupported CPU activation code")),
    }
}

#[inline]
fn apply_activation_in_place(slice: &mut [f32], act_code: i32) -> Hc4jResult<()> {
    match act_code {
        CPU_ACT_NONE => Ok(()),
        CPU_ACT_RELU => {
            for v in slice {
                *v = v.max(0.0);
            }
            Ok(())
        }
        CPU_ACT_GELU => {
            for v in slice {
                let x = *v;
                let cubic = 0.797_884_6_f32 * (x + 0.044_715_f32 * x * x * x);
                *v = 0.5 * x * (1.0 + cubic.clamp(-10.0, 10.0).tanh());
            }
            Ok(())
        }
        CPU_ACT_SILU => {
            for v in slice {
                let x = *v;
                let e = (-x.abs()).exp();
                let num = if x >= 0.0 { 1.0 } else { e };
                *v = x * num / (1.0 + e);
            }
            Ok(())
        }
        _ => Err(Hc4jError::InvalidParam("unsupported CPU activation code")),
    }
}

// ============================================================================
// SIMD Micro-Kernels (x86_64 AVX2+FMA+F16C + Portable Fallback)
// ============================================================================

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    dot_f32_scalar(a, b)
}

#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let n = a.len();
    let chunks = n / 4;
    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base] * b[base];
        s1 += a[base + 1] * b[base + 1];
        s2 += a[base + 2] * b[base + 2];
        s3 += a[base + 3] * b[base + 3];
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    for i in (chunks * 4)..n {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    while i + 32 <= len {
        let a0 = unsafe { _mm256_loadu_ps(pa.add(i)) };
        let b0 = unsafe { _mm256_loadu_ps(pb.add(i)) };
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);

        let a1 = unsafe { _mm256_loadu_ps(pa.add(i + 8)) };
        let b1 = unsafe { _mm256_loadu_ps(pb.add(i + 8)) };
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);

        let a2 = unsafe { _mm256_loadu_ps(pa.add(i + 16)) };
        let b2 = unsafe { _mm256_loadu_ps(pb.add(i + 16)) };
        acc2 = _mm256_fmadd_ps(a2, b2, acc2);

        let a3 = unsafe { _mm256_loadu_ps(pa.add(i + 24)) };
        let b3 = unsafe { _mm256_loadu_ps(pb.add(i + 24)) };
        acc3 = _mm256_fmadd_ps(a3, b3, acc3);
        i += 32;
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    while i + 8 <= len {
        let av = unsafe { _mm256_loadu_ps(pa.add(i)) };
        let bv = unsafe { _mm256_loadu_ps(pb.add(i)) };
        acc0 = _mm256_fmadd_ps(av, bv, acc0);
        i += 8;
    }
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sum2 = _mm_add_ps(sum4, shuf);
    let shuf2 = _mm_movehl_ps(sum2, sum2);
    let sum1 = _mm_add_ss(sum2, shuf2);
    let mut total = _mm_cvtss_f32(sum1);
    while i < len {
        total += unsafe { *pa.add(i) * *pb.add(i) };
        i += 1;
    }
    total
}

#[inline]
fn dot_f32_f16(a: &[f32], b_f16: &[u16]) -> f32 {
    debug_assert_eq!(a.len(), b_f16.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            return unsafe { dot_f32_f16_avx2_f16c(a, b_f16) };
        }
    }
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let n = a.len();
    let pairs = n / 2;
    for i in 0..pairs {
        let base = i * 2;
        s0 += a[base] * f16_bits_to_f32(b_f16[base]);
        s1 += a[base + 1] * f16_bits_to_f32(b_f16[base + 1]);
    }
    let mut total = s0 + s1;
    if (n & 1) != 0 {
        total += a[n - 1] * f16_bits_to_f32(b_f16[n - 1]);
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f32_f16_avx2_f16c(a: &[f32], b_f16: &[u16]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let pa = a.as_ptr();
    let pb = b_f16.as_ptr();
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    while i + 16 <= len {
        let a0 = unsafe { _mm256_loadu_ps(pa.add(i)) };
        let b0_h = unsafe { _mm_loadu_si128(pb.add(i) as *const __m128i) };
        let b0 = _mm256_cvtph_ps(b0_h);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);

        let a1 = unsafe { _mm256_loadu_ps(pa.add(i + 8)) };
        let b1_h = unsafe { _mm_loadu_si128(pb.add(i + 8) as *const __m128i) };
        let b1 = _mm256_cvtph_ps(b1_h);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        i += 16;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    while i + 8 <= len {
        let a0 = unsafe { _mm256_loadu_ps(pa.add(i)) };
        let b0_h = unsafe { _mm_loadu_si128(pb.add(i) as *const __m128i) };
        let b0 = _mm256_cvtph_ps(b0_h);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        i += 8;
    }
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sum2 = _mm_add_ps(sum4, shuf);
    let shuf2 = _mm_movehl_ps(sum2, sum2);
    let sum1 = _mm_add_ss(sum2, shuf2);
    let mut total = _mm_cvtss_f32(sum1);
    while i < len {
        total += unsafe { *pa.add(i) } * f16_bits_to_f32(unsafe { *pb.add(i) });
        i += 1;
    }
    total
}

/// Vector axpy: `dst[i] += alpha * src[i]`
#[inline]
fn axpy_f32(dst: &mut [f32], alpha: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { axpy_f32_avx2(dst, alpha, src) };
            return;
        }
    }
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += alpha * s;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_f32_avx2(dst: &mut [f32], alpha: f32, src: &[f32]) {
    use std::arch::x86_64::*;
    let len = dst.len();
    let pd = dst.as_mut_ptr();
    let ps = src.as_ptr();
    let va = _mm256_set1_ps(alpha);
    let mut i = 0usize;
    while i + 16 <= len {
        let d0 = unsafe { _mm256_loadu_ps(pd.add(i)) };
        let s0 = unsafe { _mm256_loadu_ps(ps.add(i)) };
        let r0 = _mm256_fmadd_ps(va, s0, d0);
        unsafe { _mm256_storeu_ps(pd.add(i), r0) };

        let d1 = unsafe { _mm256_loadu_ps(pd.add(i + 8)) };
        let s1 = unsafe { _mm256_loadu_ps(ps.add(i + 8)) };
        let r1 = _mm256_fmadd_ps(va, s1, d1);
        unsafe { _mm256_storeu_ps(pd.add(i + 8), r1) };
        i += 16;
    }
    while i + 8 <= len {
        let d0 = unsafe { _mm256_loadu_ps(pd.add(i)) };
        let s0 = unsafe { _mm256_loadu_ps(ps.add(i)) };
        let r0 = _mm256_fmadd_ps(va, s0, d0);
        unsafe { _mm256_storeu_ps(pd.add(i), r0) };
        i += 8;
    }
    while i < len {
        unsafe {
            *pd.add(i) += alpha * *ps.add(i);
        }
        i += 1;
    }
}

#[inline]
fn axpy_f32_f16(dst: &mut [f32], alpha: f32, src_f16: &[u16]) {
    debug_assert_eq!(dst.len(), src_f16.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            unsafe { axpy_f32_f16_avx2(dst, alpha, src_f16) };
            return;
        }
    }
    for (d, &s) in dst.iter_mut().zip(src_f16.iter()) {
        *d += alpha * f16_bits_to_f32(s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn axpy_f32_f16_avx2(dst: &mut [f32], alpha: f32, src_f16: &[u16]) {
    use std::arch::x86_64::*;
    let len = dst.len();
    let pd = dst.as_mut_ptr();
    let ps = src_f16.as_ptr();
    let va = _mm256_set1_ps(alpha);
    let mut i = 0usize;
    while i + 8 <= len {
        let d0 = unsafe { _mm256_loadu_ps(pd.add(i)) };
        let sh = unsafe { _mm_loadu_si128(ps.add(i) as *const __m128i) };
        let s0 = _mm256_cvtph_ps(sh);
        let r0 = _mm256_fmadd_ps(va, s0, d0);
        unsafe { _mm256_storeu_ps(pd.add(i), r0) };
        i += 8;
    }
    while i < len {
        unsafe {
            *pd.add(i) += alpha * f16_bits_to_f32(*ps.add(i));
        }
        i += 1;
    }
}

// ============================================================================
// Multi-Threaded Blocked GEMM / GEMV (f32 and f32 x f16)
// ============================================================================

/// High-throughput multi-threaded SIMD matrix multiplication `C[M, N] = op(A)[M, K] * op(B)[K, N]`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_gemm_f32(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    flags: u32,
) -> Hc4jResult<()> {
    let trans_a = (flags & FLAG_TRANS_A) != 0;
    let trans_b = (flags & FLAG_TRANS_B) != 0;

    let req_a = m.checked_mul(k).ok_or(Hc4jError::InvalidParam("A size overflow"))?;
    let req_b = k.checked_mul(n).ok_or(Hc4jError::InvalidParam("B size overflow"))?;
    let req_c = m.checked_mul(n).ok_or(Hc4jError::InvalidParam("C size overflow"))?;
    if a.len() < req_a || b.len() < req_b || out.len() < req_c {
        return Err(Hc4jError::InvalidParam("CPU GEMM slice smaller than declared dimensions"));
    }
    if m == 0 || n == 0 || k == 0 {
        out[..req_c].fill(0.0);
        return Ok(());
    }

    let work_ops = m.saturating_mul(n).saturating_mul(k);
    let run_par = work_ops >= 32_768;

    if !trans_a && !trans_b {
        // Fast NN row-major path: cache-blocked KC=128, NC=64, row-major saxpy accumulation (`B[p, j..]` contiguous!).
        out[..req_c].fill(0.0);
        let out_addr = out.as_mut_ptr() as usize;

        if m <= 4 && n >= 64 {
            // Decode / thin-M fast path: parallelize across disjoint N column blocks.
            const NC: usize = 64;
            let n_blocks = n.div_ceil(NC);
            parallel_chunks(n_blocks, run_par, |nb| {
                let jc = nb * NC;
                let nc = (n - jc).min(NC);
                for i in 0..m {
                    let c_row = unsafe { std::slice::from_raw_parts_mut((out_addr as *mut f32).add(i * n + jc), nc) };
                    let a_row = &a[i * k..(i + 1) * k];
                    for (p, &av) in a_row.iter().enumerate() {
                        if av != 0.0 {
                            let b_slice = &b[p * n + jc..p * n + jc + nc];
                            axpy_f32(c_row, av, b_slice);
                        }
                    }
                }
            });
        } else {
            const MC: usize = 16;
            const KC: usize = 128;
            let m_blocks = m.div_ceil(MC);
            parallel_chunks(m_blocks, run_par, |mb| {
                let ic = mb * MC;
                let mc = (m - ic).min(MC);
                for pc in (0..k).step_by(KC) {
                    let kc = (k - pc).min(KC);
                    for i in 0..mc {
                        let row_idx = ic + i;
                        let c_row = unsafe {
                            std::slice::from_raw_parts_mut((out_addr as *mut f32).add(row_idx * n), n)
                        };
                        let a_row = &a[row_idx * k + pc..row_idx * k + pc + kc];
                        for (p_off, &av) in a_row.iter().enumerate() {
                            if av != 0.0 {
                                let p = pc + p_off;
                                let b_row = &b[p * n..(p + 1) * n];
                                axpy_f32(c_row, av, b_row);
                            }
                        }
                    }
                }
            });
        }
        return Ok(());
    }

    if !trans_a && trans_b {
        // NT row-major path (`C[i, j] = dot(A[i, :], B[j, :])`): both rows contiguous in memory!
        let out_addr = out.as_mut_ptr() as usize;
        if m <= 4 && n >= 32 {
            const NC: usize = 32;
            let n_blocks = n.div_ceil(NC);
            parallel_chunks(n_blocks, run_par, |nb| {
                let jc = nb * NC;
                let nc = (n - jc).min(NC);
                for i in 0..m {
                    let a_row = &a[i * k..(i + 1) * k];
                    let c_row = unsafe { std::slice::from_raw_parts_mut((out_addr as *mut f32).add(i * n + jc), nc) };
                    for (j_off, dst) in c_row.iter_mut().enumerate() {
                        let j = jc + j_off;
                        let b_row = &b[j * k..(j + 1) * k];
                        *dst = dot_f32(a_row, b_row);
                    }
                }
            });
        } else {
            const MC: usize = 8;
            let m_blocks = m.div_ceil(MC);
            parallel_chunks(m_blocks, run_par, |mb| {
                let ic = mb * MC;
                let mc = (m - ic).min(MC);
                for i in 0..mc {
                    let row = ic + i;
                    let a_row = &a[row * k..(row + 1) * k];
                    let c_row = unsafe { std::slice::from_raw_parts_mut((out_addr as *mut f32).add(row * n), n) };
                    for (j, dst) in c_row.iter_mut().enumerate() {
                        let b_row = &b[j * k..(j + 1) * k];
                        *dst = dot_f32(a_row, b_row);
                    }
                }
            });
        }
        return Ok(());
    }

    // General TN / TT paths (A is [k, m]).
    out[..req_c].fill(0.0);
    let out_addr = out.as_mut_ptr() as usize;
    const MC: usize = 16;
    let m_blocks = m.div_ceil(MC);
    parallel_chunks(m_blocks, run_par, |mb| {
        let ic = mb * MC;
        let mc = (m - ic).min(MC);
        for i_off in 0..mc {
            let i = ic + i_off;
            let c_row = unsafe { std::slice::from_raw_parts_mut((out_addr as *mut f32).add(i * n), n) };
            if !trans_b {
                for p in 0..k {
                    let av = a[p * m + i];
                    if av != 0.0 {
                        axpy_f32(c_row, av, &b[p * n..(p + 1) * n]);
                    }
                }
            } else {
                for (j, dst) in c_row.iter_mut().enumerate() {
                    let b_row = &b[j * k..(j + 1) * k];
                    let mut acc = 0.0f32;
                    for (p, &bv) in b_row.iter().enumerate() {
                        acc += a[p * m + i] * bv;
                    }
                    *dst = acc;
                }
            }
        }
    });
    Ok(())
}

/// High-throughput multi-threaded SIMD mixed-precision matrix multiplication
/// `C[M, N] = A[M, K] * op(B_f16)[K, N]` where `B_f16` contains packed IEEE-754 `f16` bit patterns.
#[allow(clippy::too_many_arguments)]
pub fn cpu_gemm_f32_f16(
    a: &[f32],
    b_f16: &[u16],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    flags: u32,
) -> Hc4jResult<()> {
    if (flags & FLAG_TRANS_A) != 0 {
        return Err(Hc4jError::Unsupported("cpu_gemm_f32_f16 does not support FLAG_TRANS_A"));
    }
    let trans_b = (flags & FLAG_TRANS_B) != 0;

    let req_a = m.checked_mul(k).ok_or(Hc4jError::InvalidParam("A size overflow"))?;
    let req_b = k.checked_mul(n).ok_or(Hc4jError::InvalidParam("B size overflow"))?;
    let req_c = m.checked_mul(n).ok_or(Hc4jError::InvalidParam("C size overflow"))?;
    if a.len() < req_a || b_f16.len() < req_b || out.len() < req_c {
        return Err(Hc4jError::InvalidParam("CPU F16 GEMM slice smaller than declared dimensions"));
    }
    if m == 0 || n == 0 || k == 0 {
        out[..req_c].fill(0.0);
        return Ok(());
    }

    let work_ops = m.saturating_mul(n).saturating_mul(k);
    let run_par = work_ops >= 32_768;
    let out_addr = out.as_mut_ptr() as usize;

    if trans_b {
        if m <= 4 && n >= 32 {
            const NC: usize = 32;
            let n_blocks = n.div_ceil(NC);
            parallel_chunks(n_blocks, run_par, |nb| {
                let jc = nb * NC;
                let nc = (n - jc).min(NC);
                for i in 0..m {
                    let a_row = &a[i * k..(i + 1) * k];
                    let c_row = unsafe { std::slice::from_raw_parts_mut((out_addr as *mut f32).add(i * n + jc), nc) };
                    for (j_off, dst) in c_row.iter_mut().enumerate() {
                        let j = jc + j_off;
                        *dst = dot_f32_f16(a_row, &b_f16[j * k..(j + 1) * k]);
                    }
                }
            });
        } else {
            const MC: usize = 8;
            let m_blocks = m.div_ceil(MC);
            parallel_chunks(m_blocks, run_par, |mb| {
                let ic = mb * MC;
                let mc = (m - ic).min(MC);
                for i in 0..mc {
                    let row = ic + i;
                    let a_row = &a[row * k..(row + 1) * k];
                    let c_row = unsafe { std::slice::from_raw_parts_mut((out_addr as *mut f32).add(row * n), n) };
                    for (j, dst) in c_row.iter_mut().enumerate() {
                        *dst = dot_f32_f16(a_row, &b_f16[j * k..(j + 1) * k]);
                    }
                }
            });
        }
    } else {
        out[..req_c].fill(0.0);
        if m <= 4 && n >= 64 {
            const NC: usize = 64;
            let n_blocks = n.div_ceil(NC);
            parallel_chunks(n_blocks, run_par, |nb| {
                let jc = nb * NC;
                let nc = (n - jc).min(NC);
                for i in 0..m {
                    let c_row = unsafe { std::slice::from_raw_parts_mut((out_addr as *mut f32).add(i * n + jc), nc) };
                    let a_row = &a[i * k..(i + 1) * k];
                    for (p, &av) in a_row.iter().enumerate() {
                        if av != 0.0 {
                            axpy_f32_f16(c_row, av, &b_f16[p * n + jc..p * n + jc + nc]);
                        }
                    }
                }
            });
        } else {
            const MC: usize = 16;
            const KC: usize = 128;
            let m_blocks = m.div_ceil(MC);
            parallel_chunks(m_blocks, run_par, |mb| {
                let ic = mb * MC;
                let mc = (m - ic).min(MC);
                for pc in (0..k).step_by(KC) {
                    let kc = (k - pc).min(KC);
                    for i in 0..mc {
                        let row = ic + i;
                        let c_row = unsafe { std::slice::from_raw_parts_mut((out_addr as *mut f32).add(row * n), n) };
                        let a_row = &a[row * k + pc..row * k + pc + kc];
                        for (p_off, &av) in a_row.iter().enumerate() {
                            if av != 0.0 {
                                let p = pc + p_off;
                                axpy_f32_f16(c_row, av, &b_f16[p * n..(p + 1) * n]);
                            }
                        }
                    }
                }
            });
        }
    }
    Ok(())
}

// ============================================================================
// Multi-Threaded SIMD Tensor-Parallel MLP Slice
// ============================================================================

/// Executes a Tensor-Parallel MLP hidden slice `[h_start, h_start + units)` on Host CPU:
/// `out[M, Dout] = act(X[M, Din] * W1[:, h_start..+units] + b1[h_start..+units]) * W2[h_start..+units, :]`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_mlp_slice_f32(
    x: &[f32],
    w1: &[f32],
    b1: Option<&[f32]>,
    w2: &[f32],
    out: &mut [f32],
    m: usize,
    d_in: usize,
    h: usize,
    d_out: usize,
    h_start: usize,
    units: usize,
    act_code: i32,
) -> Hc4jResult<()> {
    let req_x = m.checked_mul(d_in).ok_or(Hc4jError::InvalidParam("X size overflow"))?;
    let req_w1 = d_in.checked_mul(h).ok_or(Hc4jError::InvalidParam("W1 size overflow"))?;
    let req_w2 = h.checked_mul(d_out).ok_or(Hc4jError::InvalidParam("W2 size overflow"))?;
    let req_out = m.checked_mul(d_out).ok_or(Hc4jError::InvalidParam("out size overflow"))?;
    if x.len() < req_x || w1.len() < req_w1 || w2.len() < req_w2 || out.len() < req_out {
        return Err(Hc4jError::InvalidParam("CPU MLP slice buffer too small"));
    }
    if h_start.saturating_add(units) > h {
        return Err(Hc4jError::InvalidParam("CPU MLP h_start + units exceeds H"));
    }
    if let Some(bias) = b1
        && bias.len() < h_start + units
    {
        return Err(Hc4jError::InvalidParam("CPU MLP b1 too small"));
    }
    if m == 0 || units == 0 || d_out == 0 {
        out[..req_out].fill(0.0);
        return Ok(());
    }

    // 1. Compute hidden slice H_local[m, units] using blocked SIMD over column tiles of W1[:, h_start..h_start+units]
    let mut h_slice = vec![0.0f32; m * units];
    let run_par_w1 = m.saturating_mul(d_in).saturating_mul(units) >= 32_768;

    let h_slice_addr = h_slice.as_mut_ptr() as usize;
    const HC: usize = 64;
    let h_blocks = units.div_ceil(HC);

    // Parallelize across hidden chunks when M is small, or M chunks when M is larger.
    if m <= 4 && h_blocks > 1 {
        parallel_chunks(h_blocks, run_par_w1, |hb| {
            let hc_off = hb * HC;
            let hc = (units - hc_off).min(HC);
            let global_c = h_start + hc_off;
            for i in 0..m {
                let dst = unsafe {
                    std::slice::from_raw_parts_mut((h_slice_addr as *mut f32).add(i * units + hc_off), hc)
                };
                let x_row = &x[i * d_in..(i + 1) * d_in];
                for (p, &xv) in x_row.iter().enumerate() {
                    if xv != 0.0 {
                        let w1_seg = &w1[p * h + global_c..p * h + global_c + hc];
                        axpy_f32(dst, xv, w1_seg);
                    }
                }
                if let Some(bias) = b1 {
                    let b_seg = &bias[global_c..global_c + hc];
                    for (d, &bv) in dst.iter_mut().zip(b_seg.iter()) {
                        *d += bv;
                    }
                }
                let _ = apply_activation_in_place(dst, act_code);
            }
        });
    } else {
        for i in 0..m {
            let dst = &mut h_slice[i * units..(i + 1) * units];
            let x_row = &x[i * d_in..(i + 1) * d_in];
            for (p, &xv) in x_row.iter().enumerate() {
                if xv != 0.0 {
                    let w1_seg = &w1[p * h + h_start..p * h + h_start + units];
                    axpy_f32(dst, xv, w1_seg);
                }
            }
            if let Some(bias) = b1 {
                let b_seg = &bias[h_start..h_start + units];
                for (d, &bv) in dst.iter_mut().zip(b_seg.iter()) {
                    *d += bv;
                }
            }
            apply_activation_in_place(dst, act_code)?;
        }
    }

    // Validate activation even when parallel path ran (fast O(1) check)
    let _ = apply_activation_f32(0.0, act_code)?;

    // 2. W2[h_start..h_start+units, 0..d_out] is already contiguous! Delegate directly to multi-threaded SIMD GEMM.
    let w2_slice = &w2[h_start * d_out..(h_start + units) * d_out];
    cpu_gemm_f32(&h_slice, w2_slice, out, m, d_out, units, 0)
}

// ============================================================================
// Multi-Threaded SIMD Online-Softmax Split-KV Decode Attention
// ============================================================================

/// Evaluates local online-softmax Decode Attention on Host CPU over KV shards in system RAM.
/// Matches `SplitKvSession` exact `[m, l]` convention across all heads.
#[allow(clippy::too_many_arguments)]
pub fn cpu_split_kv_attn_f32(
    q: &[f32],
    k_shard: &[f32],
    v_shard: &[f32],
    out_o: &mut [f32],
    out_stats: &mut [f32],
    tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Hc4jResult<()> {
    if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Err(Hc4jError::InvalidParam("attention dimensions must be positive"));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(Hc4jError::InvalidParam("num_heads must be divisible by num_kv_heads"));
    }
    let q_elems = num_heads
        .checked_mul(head_dim)
        .ok_or(Hc4jError::InvalidParam("q size overflow"))?;
    let kv_elems = num_kv_heads
        .checked_mul(tokens)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or(Hc4jError::InvalidParam("kv size overflow"))?;
    let stats_elems = num_heads * 2;
    if q.len() < q_elems || k_shard.len() < kv_elems || v_shard.len() < kv_elems || out_o.len() < q_elems || out_stats.len() < stats_elems {
        return Err(Hc4jError::InvalidParam("CPU Split-KV attention buffer too small"));
    }

    if tokens == 0 {
        out_o[..q_elems].fill(0.0);
        for h in 0..num_heads {
            out_stats[2 * h] = f32::NEG_INFINITY;
            out_stats[2 * h + 1] = 0.0;
        }
        return Ok(());
    }

    let group_size = num_heads / num_kv_heads;
    let out_o_addr = out_o.as_mut_ptr() as usize;
    let out_s_addr = out_stats.as_mut_ptr() as usize;
    let run_par = num_heads >= 2 && num_heads.saturating_mul(tokens).saturating_mul(head_dim) >= 16_384;

    parallel_chunks(num_heads, run_par, |h| {
        let kv_h = h / group_size;
        let q_vec = &q[h * head_dim..(h + 1) * head_dim];
        let kv_base = kv_h * tokens * head_dim;

        let mut scores = vec![0.0f32; tokens];
        let mut m = f32::NEG_INFINITY;
        for (t, dst_score) in scores.iter_mut().enumerate() {
            let k_off = kv_base + t * head_dim;
            let k_vec = &k_shard[k_off..k_off + head_dim];
            let s = dot_f32(q_vec, k_vec) * scale;
            *dst_score = s;
            if s > m {
                m = s;
            }
        }

        let o_row = unsafe { std::slice::from_raw_parts_mut((out_o_addr as *mut f32).add(h * head_dim), head_dim) };
        o_row.fill(0.0);

        let mut l = 0.0f32;
        for (t, &score) in scores.iter().enumerate() {
            let w = (score - m).exp();
            l += w;
            let v_off = kv_base + t * head_dim;
            let v_vec = &v_shard[v_off..v_off + head_dim];
            axpy_f32(o_row, w, v_vec);
        }

        if l > 0.0 {
            let inv_l = 1.0 / l;
            for dst in o_row.iter_mut() {
                *dst *= inv_l;
            }
        }

        unsafe {
            let st_ptr = (out_s_addr as *mut f32).add(2 * h);
            *st_ptr = m;
            *st_ptr.add(1) = l;
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_gemm_nn_nt_tn_and_f16_match_reference() {
        let m = 7usize;
        let n = 19usize;
        let k = 43usize;

        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.13).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.07).cos()).collect();
        let b_t: Vec<f32> = (0..n * k)
            .map(|jk| {
                let j = jk / k;
                let p = jk % k;
                b[p * n + j]
            })
            .collect();

        let mut ref_nn = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                ref_nn[i * n + j] = s;
            }
        }

        let mut out_nn = vec![0.0f32; m * n];
        cpu_gemm_f32(&a, &b, &mut out_nn, m, n, k, 0).unwrap();
        for (got, want) in out_nn.iter().zip(&ref_nn) {
            assert!((got - want).abs() < 1e-4, "NN got={got} want={want}");
        }

        let mut out_nt = vec![0.0f32; m * n];
        cpu_gemm_f32(&a, &b_t, &mut out_nt, m, n, k, FLAG_TRANS_B).unwrap();
        for (got, want) in out_nt.iter().zip(&ref_nn) {
            assert!((got - want).abs() < 1e-4, "NT got={got} want={want}");
        }

        // Convert B^T to IEEE-754 f16 bits (round-to-nearest simplified via exact f32->f16 helper)
        let b_t_f16: Vec<u16> = b_t.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let mut out_f16 = vec![0.0f32; m * n];
        cpu_gemm_f32_f16(&a, &b_t_f16, &mut out_f16, m, n, k, FLAG_TRANS_B).unwrap();
        for (got, want) in out_f16.iter().zip(&ref_nn) {
            assert!((got - want).abs() < 2e-2, "F16 got={got} want={want}");
        }
    }

    #[test]
    fn cpu_mlp_and_attention_match_scalar_reference() {
        let m = 5usize;
        let d_in = 17usize;
        let h = 24usize;
        let d_out = 13usize;
        let h_start = 6usize;
        let units = 12usize;

        let x: Vec<f32> = (0..m * d_in).map(|i| ((i as f32) * 0.11).sin() * 0.5).collect();
        let w1: Vec<f32> = (0..d_in * h).map(|i| ((i as f32) * 0.05).cos() * 0.3).collect();
        let b1: Vec<f32> = (0..h).map(|i| (i as f32) * 0.02 - 0.2).collect();
        let w2: Vec<f32> = (0..h * d_out).map(|i| ((i as f32) * 0.09).sin() * 0.4).collect();

        let mut out = vec![0.0f32; m * d_out];
        cpu_mlp_slice_f32(&x, &w1, Some(&b1), &w2, &mut out, m, d_in, h, d_out, h_start, units, CPU_ACT_SILU)
            .unwrap();

        let mut ref_out = vec![0.0f32; m * d_out];
        let mut h_ref = vec![0.0f32; m * units];
        for i in 0..m {
            for j in 0..units {
                let mut sum = b1[h_start + j];
                for p in 0..d_in {
                    sum += x[i * d_in + p] * w1[p * h + h_start + j];
                }
                h_ref[i * units + j] = apply_activation_f32(sum, CPU_ACT_SILU).unwrap();
            }
            for j in 0..d_out {
                let mut sum = 0.0f32;
                for p in 0..units {
                    sum += h_ref[i * units + p] * w2[(h_start + p) * d_out + j];
                }
                ref_out[i * d_out + j] = sum;
            }
        }
        for (got, want) in out.iter().zip(&ref_out) {
            assert!((got - want).abs() < 1e-4, "MLP got={got} want={want}");
        }

        // Split-KV attention check
        let num_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 40usize;
        let tokens = 37usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q: Vec<f32> = (0..num_heads * head_dim).map(|i| ((i as f32) * 0.19).sin()).collect();
        let k: Vec<f32> = (0..num_kv_heads * tokens * head_dim)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();
        let v: Vec<f32> = (0..num_kv_heads * tokens * head_dim)
            .map(|i| ((i as f32) * 0.13).sin())
            .collect();

        let mut got_o = vec![0.0f32; num_heads * head_dim];
        let mut got_stats = vec![0.0f32; num_heads * 2];
        cpu_split_kv_attn_f32(&q, &k, &v, &mut got_o, &mut got_stats, tokens, num_heads, num_kv_heads, head_dim, scale)
            .unwrap();

        for head in 0..num_heads {
            let kv_h = head / (num_heads / num_kv_heads);
            let mut scores = vec![0.0f32; tokens];
            let mut m_val = f32::NEG_INFINITY;
            for t in 0..tokens {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[head * head_dim + d] * k[(kv_h * tokens + t) * head_dim + d];
                }
                let s = dot * scale;
                scores[t] = s;
                m_val = m_val.max(s);
            }
            let l_val: f32 = scores.iter().map(|&s| (s - m_val).exp()).sum();
            assert!((got_stats[2 * head] - m_val).abs() < 1e-5);
            assert!((got_stats[2 * head + 1] - l_val).abs() < 1e-4);
            for d in 0..head_dim {
                let acc: f32 = (0..tokens)
                    .map(|t| (scores[t] - m_val).exp() * v[(kv_h * tokens + t) * head_dim + d])
                    .sum();
                assert!((got_o[head * head_dim + d] - acc / l_val).abs() < 1e-5);
            }
        }
    }

    fn f32_to_f16_bits(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x007F_FFFF;
        if exp == 255 {
            return sign | 0x7C00 | u16::from(mant != 0);
        }
        let unbiased = exp - 127 + 15;
        if unbiased >= 31 {
            return sign | 0x7C00;
        }
        if unbiased <= 0 {
            return sign;
        }
        sign | ((unbiased as u16) << 10) | ((mant >> 13) as u16)
    }
}
