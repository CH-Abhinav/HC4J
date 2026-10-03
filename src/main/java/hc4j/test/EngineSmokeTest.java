package hc4j.test;

import hc4j.DType;
import hc4j.FusedExpr;
import hc4j.Tensor;
import hc4j.engine.EngineStats;
import hc4j.engine.GpuBatch;
import hc4j.engine.MemoryStats;
import hc4j.engine.WgpuBackend;
import hc4j.ops.MatmulOps;
import hc4j.ops.MatmulOps.Kernel;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.DoubleUnaryOperator;
import java.util.function.UnaryOperator;

/**
 * End-to-end validation of the engine against the real GPU: caller-allocated and in-place
 * elementwise ops, 2-D grid dispatch past 67M elements, matmul (every kernel) against a CPU
 * reference, in-VRAM transposition, kernel fusion (with measured traffic and allocation counts),
 * slab sub-allocation, and command batching.
 *
 * <p>Run with {@code gradlew engineSmokeTest}; set {@code HC4J_BACKEND=vulkan} to exercise the
 * subgroup GEMV kernel on GPUs whose Vulkan driver exposes subgroups. Exits non-zero on the first
 * failure.
 */
public final class EngineSmokeTest {

    /** WGSL guarantees sin/cos only to 2^-11 absolute; see TrignoSmokeTest. */
    private static final double WGSL_TOL = 1.0 / 2048;
    private static final double UNIT_ROUNDOFF = Math.ulp(1.0f) / 2;
    private static int checks = 0;

    public static void main(String[] args) {
        EngineStats engine = WgpuBackend.engineStats();
        section("Engine");
        System.out.printf("      %s %s GPU | subgroups=%s (size %d..%d) | shader_f16=%s%n",
                engine.deviceTypeName(), engine.backendName(), engine.hasSubgroups(),
                engine.subgroupMinSize(), engine.subgroupMaxSize(), engine.hasShaderF16());

        elementwiseCallerAllocated();
        gridBeyond67M();
        matmulPlans(engine);
        matmulCorrectness();
        matmulValidation();
        transposeKernel();
        matmulThroughput();
        fusion();
        slabs();
        batching();
        leakCheck();
        System.out.printf("%nALL %d CHECKS PASSED%n", checks);
    }

    // ------------------------------------------------------------------------------------------

    private static void elementwiseCallerAllocated() {
        section("Caller-allocated and in-place elementwise ops");
        int n = 100_003;
        float[] host = ramp(n, 0.1f, 3f);
        try (Tensor a = Tensor.fromArray(host, n); Tensor b = Tensor.fromArray(host, n);
             Tensor res = Tensor.empty(DType.f32, n)) {
            // Out of place, into a caller-provided result.
            expect(a.sin(res) == res, "sin(res) returns the caller's tensor");
            expect(maxError(res.toFloatArray(), host, Math::sin) < WGSL_TOL, "sin(res) correct");

            // In place: output handle == input handle.
            a.exp(a);
            expect(maxError(a.toFloatArray(), host, Math::exp) < 1e-5, "a.exp(a) in place");
            a.log(a);
            expect(maxError(a.toFloatArray(), host, v -> v) < 1e-5, "a.log(a) undoes it in place");

            // Binary in place: a = a + b.
            a.add(b, a);
            expect(maxError(a.toFloatArray(), host, v -> 2 * v) < 1e-5, "a.add(b, a) in place");

            record Case(String name, UnaryOperator<Tensor> op, DoubleUnaryOperator ref) {}
            for (Case c : List.of(
                    new Case("sqrt", Tensor::sqrt, Math::sqrt),
                    new Case("log2", Tensor::log2, v -> Math.log(v) / Math.log(2)),
                    new Case("log10", Tensor::log10, Math::log10))) {
                try (Tensor r = c.op().apply(b)) {
                    expect(maxError(r.toFloatArray(), host, c.ref()) < 1e-5, c.name() + " correct");
                }
            }
        }
        try (Tensor x = Tensor.fromArray(new float[] {1, 2, 3}, 3); Tensor wrong = Tensor.empty(DType.f32, 4)) {
            expectThrows(IllegalArgumentException.class, () -> x.sin(wrong), "sin into a mismatched result");
        }
    }

    private static void gridBeyond67M() {
        section("2-D grid dispatch beyond 65535 x 1024 = 67.1M elements");
        int n = 67_200_000;
        float[] host = new float[n];
        for (int i = 0; i < n; i++) {
            host[i] = (i % 1000) * 0.001f - 0.5f;
        }
        EngineStats before = WgpuBackend.engineStats();
        try (Tensor a = Tensor.fromArray(host, n); Tensor r = Tensor.empty(DType.f32, n)) {
            a.add(a, r); // arithmetic.rs
            expect(countBad(r.toFloatArray(), host, v -> 2 * v, 0) == 0, "add: every one of 67.2M elements");
            a.exp(r); // exponential.rs
            expect(countBad(r.toFloatArray(), host, Math::exp, 1e-5) == 0, "exp: every one of 67.2M elements");
            a.sin(r); // trigno.rs
            expect(countBad(r.toFloatArray(), host, Math::sin, WGSL_TOL) == 0, "sin: every one of 67.2M elements");
        }
        EngineStats after = WgpuBackend.engineStats();
        expect(after.dispatches() - before.dispatches() == 3, "one folded dispatch per op");
    }

    // ------------------------------------------------------------------------------------------

    private static void matmulPlans(EngineStats engine) {
        section("Matmul dispatcher");
        expectPlan(777, 1, 1000, false, false, Kernel.GEMV_ROWS, "matrix x vector, K < 1024");
        expectPlan(300, 1, 4096, false, false,
                engine.hasSubgroups() ? Kernel.GEMV_ROWS_SUBGROUP : Kernel.GEMV_ROWS,
                "matrix x vector, K >= 1024 (subgroups " + (engine.hasSubgroups() ? "on" : "off") + ")");
        expectPlan(1, 517, 1000, false, false, Kernel.GEMV_COLS, "vector x matrix");
        expectPlan(300, 1, 200, true, false, Kernel.GEMV_COLS, "A^T x vector swaps to the column mapping");
        expectPlan(130, 70, 90, false, false, Kernel.REGISTER_64_SCALAR, "register-tiled, unaligned K/N");
        expectPlan(67, 33, 129, false, false, Kernel.TILED_16, "small/thin -> 16x16 tiles");
        Kernel big = engine.isDiscrete() ? Kernel.REGISTER_128_VEC4 : Kernel.REGISTER_64_VEC4;
        expectPlan(1024, 1024, 1024, false, false, big, "large square on a " + engine.deviceTypeName() + " GPU");
        MatmulOps.Plan p = MatmulOps.plan(256, 256, 256, true, true);
        expect(p.pretransposeA() && p.pretransposeB(), "A^T B^T pre-transposes both operands in VRAM");
    }

    private static void expectPlan(int m, int n, int k, boolean ta, boolean tb, Kernel want, String what) {
        Kernel got = MatmulOps.plan(m, n, k, ta, tb).kernel();
        expect(got == want, String.format("%dx%dx%d -> %s (%s)", m, n, k, got, what));
    }

    private static void matmulCorrectness() {
        section("Matmul vs CPU reference (every kernel, all transpose combinations)");
        int[][] shapes = {
                {64, 64, 64}, {256, 256, 256}, {1000, 1000, 1000}, {130, 70, 90}, {67, 33, 129},
                {2, 3, 4}, {300, 1, 1500}, {777, 1, 1000}, {1, 517, 1000}, {1, 1, 4096}, {513, 257, 129},
        };
        SplittableRandom rng = new SplittableRandom(42);
        for (int[] s : shapes) {
            int m = s[0], n = s[1], k = s[2];
            for (int flags = 0; flags < 4; flags++) {
                boolean ta = (flags & 1) != 0, tb = (flags & 2) != 0;
                if ((m == 1 && ta) || (n == 1 && tb) || (m * n * (long) k > 2e8 && flags != 0)) {
                    continue; // transposing a vector is a no-op; keep the 1000^3 CPU reference to one run
                }
                float[] a = random(rng, m * k);
                float[] b = random(rng, k * n);
                int[] aShape = ta ? new int[] {k, m} : new int[] {m, k};
                int[] bShape = tb ? new int[] {n, k} : new int[] {k, n};
                try (Tensor ta2 = Tensor.fromArray(a, aShape); Tensor tb2 = Tensor.fromArray(b, bShape);
                     Tensor c = Tensor.empty(DType.f32, m, n)) {
                    MatmulOps.matmul(ta2, tb2, c, ta, tb);
                    String bad = compareMatmul(c.toFloatArray(), a, b, m, n, k, ta, tb);
                    expect(bad == null, String.format("%4dx%4dx%4d %-9s %-19s %s", m, n, k,
                            (ta ? "A^T" : "A") + (tb ? " B^T" : " B"),
                            MatmulOps.plan(m, n, k, ta, tb).kernel(), bad == null ? "ok" : bad));
                }
            }
        }

        // Fluent API and 1-D (NumPy) semantics.
        float[] av = random(rng, 50 * 40);
        float[] bv = random(rng, 40);
        try (Tensor A = Tensor.fromArray(av, 50, 40); Tensor x = Tensor.fromArray(bv, 40);
             Tensor y = A.matmul(x); Tensor dot = x.matmul(x)) {
            expect(Arrays.equals(y.internalShapeUnsafe(), new int[] {50}), "[50,40] . [40] -> [50]");
            expect(compareMatmul(y.toFloatArray(), av, bv, 50, 1, 40, false, false) == null, "matrix . vector values");
            expect(Arrays.equals(dot.internalShapeUnsafe(), new int[] {1}), "[40] . [40] -> [1]");
            expect(compareMatmul(dot.toFloatArray(), bv, bv, 1, 1, 40, false, false) == null, "dot product value");
        }
    }

    /** Null if every element is within the f32 accumulation bound k·u·Σ|a·b|, else a message. */
    private static String compareMatmul(float[] got, float[] a, float[] b, int m, int n, int k, boolean ta, boolean tb) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                double abs = 0;
                for (int p = 0; p < k; p++) {
                    double t = (double) (ta ? a[p * m + i] : a[i * k + p]) * (tb ? b[j * k + p] : b[p * n + j]);
                    sum += t;
                    abs += Math.abs(t);
                }
                double tol = 2 * k * UNIT_ROUNDOFF * abs + 1e-6;
                double err = Math.abs(got[i * n + j] - sum);
                if (err > tol) {
                    return String.format("(%d,%d): got %.6f want %.6f err %.2e > %.2e", i, j, got[i * n + j], sum, err, tol);
                }
            }
        }
        return null;
    }

    private static void matmulValidation() {
        section("Matmul shape and aliasing validation");
        try (Tensor a = Tensor.zeros(DType.f32, 3, 4); Tensor b = Tensor.zeros(DType.f32, 5, 6);
             Tensor sq = Tensor.zeros(DType.f32, 4, 4); Tensor t3 = Tensor.zeros(DType.f32, 2, 2, 2)) {
            expectThrows(IllegalArgumentException.class, () -> a.matmul(b), "A.cols != B.rows is rejected");
            expectThrows(IllegalArgumentException.class, () -> sq.matmul(sq, sq), "result aliasing an operand is rejected");
            expectThrows(IllegalArgumentException.class, () -> t3.matmul(t3), "3-D operands are rejected");
            try (Tensor wrong = Tensor.zeros(DType.f32, 3, 3)) {
                expectThrows(IllegalArgumentException.class, () -> a.matmul(sq, wrong), "wrong result shape is rejected");
            }
        }
    }

    private static void transposeKernel() {
        section("In-VRAM transpose (32x33 padded tiles)");
        int r = 333, c = 517;
        float[] host = random(new SplittableRandom(7), r * c);
        try (Tensor t = Tensor.fromArray(host, r, c); Tensor tt = t.transpose()) {
            expect(Arrays.equals(tt.internalShapeUnsafe(), new int[] {c, r}), "shape [333,517] -> [517,333]");
            float[] got = tt.toFloatArray();
            boolean ok = true;
            for (int i = 0; i < r && ok; i++) {
                for (int j = 0; j < c; j++) {
                    if (got[j * r + i] != host[i * c + j]) {
                        ok = false;
                        break;
                    }
                }
            }
            expect(ok, "transpose is exact");
            expectThrows(IllegalArgumentException.class, () -> t.transpose(t), "in-place transpose is rejected");
        }
    }

    private static void matmulThroughput() {
        section("Matmul throughput (informational)");
        SplittableRandom rng = new SplittableRandom(3);
        for (int size : new int[] {512, 1024, 2048}) {
            try (Tensor a = Tensor.fromArray(random(rng, size * size), size, size);
                 Tensor b = Tensor.fromArray(random(rng, size * size), size, size);
                 Tensor c = Tensor.empty(DType.f32, size, size)) {
                a.matmul(b, c); // warm-up: pipeline compile
                WgpuBackend.synchronize();
                int reps = size >= 2048 ? 3 : 10;
                long t0 = System.nanoTime();
                for (int i = 0; i < reps; i++) {
                    a.matmul(b, c);
                }
                WgpuBackend.synchronize();
                double secs = (System.nanoTime() - t0) / 1e9 / reps;
                System.out.printf("      %4d^3  %-19s %7.2f ms  %7.1f GFLOP/s%n", size,
                        MatmulOps.plan(size, size, size, false, false).kernel(), secs * 1e3,
                        2.0 * size * size * size / secs / 1e9);
            }
        }
    }

    // ------------------------------------------------------------------------------------------

    private static void fusion() {
        section("Kernel fusion: sin(a).add(b).mul(c)");
        int n = 1 << 20;
        SplittableRandom rng = new SplittableRandom(11);
        float[] ha = random(rng, n), hb = random(rng, n), hc = random(rng, n);
        try (Tensor a = Tensor.fromArray(ha, n); Tensor b = Tensor.fromArray(hb, n); Tensor c = Tensor.fromArray(hc, n)) {
            EngineStats e0 = WgpuBackend.engineStats();
            MemoryStats m0 = WgpuBackend.memoryStats();
            Tensor unfused;
            try (Tensor s = a.sin(); Tensor t = s.add(b)) {
                unfused = t.mul(c);
            }
            EngineStats e1 = WgpuBackend.engineStats();
            MemoryStats m1 = WgpuBackend.memoryStats();
            Tensor fused = a.lazy().sin().add(b).mul(c).eval();
            EngineStats e2 = WgpuBackend.engineStats();
            MemoryStats m2 = WgpuBackend.memoryStats();

            long unfusedBytes = e1.kernelBytes() - e0.kernelBytes();
            long fusedBytes = e2.kernelBytes() - e1.kernelBytes();
            System.out.printf("      unfused: %d dispatches, %d allocations, %.1f MiB moved%n",
                    e1.dispatches() - e0.dispatches(), m1.allocations() - m0.allocations(), unfusedBytes / 1048576.0);
            System.out.printf("      fused:   %d dispatch,   %d allocation,  %.1f MiB moved%n",
                    e2.dispatches() - e1.dispatches(), m2.allocations() - m1.allocations(), fusedBytes / 1048576.0);
            expect(unfusedBytes == 8L * n * Float.BYTES, "unfused chain moves 8N floats");
            expect(fusedBytes == 4L * n * Float.BYTES, "fused kernel moves 4N floats (3 reads + 1 write)");
            expect(m2.allocations() - m1.allocations() == 1, "fused: no intermediate tensors (1 allocation vs 3)");
            expect(e2.dispatches() - e1.dispatches() == 1, "fused: one dispatch vs three");

            float[] f = fused.toFloatArray();
            float[] u = unfused.toFloatArray();
            expect(maxError(f, u) < 1e-6, "fused == unfused");
            double ref = 0;
            for (int i = 0; i < n; i++) {
                ref = Math.max(ref, Math.abs(f[i] - (Math.sin(ha[i]) + hb[i]) * hc[i]));
            }
            expect(ref < WGSL_TOL, String.format("fused vs CPU, max err %.2e", ref));
            unfused.close();
            fused.close();

            // Timing, synchronized so GPU time is measured.
            double tUnfused = time(20, () -> {
                try (Tensor s = a.sin(); Tensor t = s.add(b); Tensor r = t.mul(c)) {
                    // measured
                }
            });
            double tFused = time(20, () -> {
                try (Tensor r = a.lazy().sin().add(b).mul(c).eval()) {
                    // measured
                }
            });
            System.out.printf("      time/op: unfused %.3f ms, fused %.3f ms (%.2fx)%n", tUnfused, tFused, tUnfused / tFused);

            // In-place eval and common subexpressions: a = a*a + sin(a)*sin(a).
            FusedExpr e = a.lazy().mul(a).add(a.lazy().sin().mul(a.lazy().sin()));
            e.eval(a);
            float[] got = a.toFloatArray();
            double worst = 0;
            for (int i = 0; i < n; i++) {
                double s = Math.sin(ha[i]);
                worst = Math.max(worst, Math.abs(got[i] - (ha[i] * ha[i] + s * s)));
            }
            expect(worst < 2 * WGSL_TOL, String.format("in-place eval with CSE, max err %.2e", worst));
        }
    }

    private static void slabs() {
        section("Slab sub-allocation and quarantine");
        int n = 1 << 18; // 1 MiB tensors: slab regions
        float[] host = ramp(n, -2f, 2f);
        try (Tensor x = Tensor.fromArray(host, n)) {
            MemoryStats before = WgpuBackend.memoryStats();
            for (int i = 0; i < 200; i++) {
                try (Tensor y = x.sin()) {
                    // closed immediately after the dispatch is issued
                }
            }
            MemoryStats after = WgpuBackend.memoryStats();
            expect(after.dedicatedBuffers() == before.dedicatedBuffers(), "200 op outputs, 0 create_buffer calls");
            expect(after.regionAllocs() - before.regionAllocs() >= 200, "outputs are slab regions ("
                    + (after.regionAllocs() - before.regionAllocs()) + ")");
            expect(after.slabCount() <= 2, "working set stays in " + after.slabCount() + " slab(s)");

            // Close-while-in-flight inside a batch: y's region is freed before the batch is even
            // submitted, so it must stay quarantined while w and z are computed.
            Tensor z;
            Tensor w;
            try (GpuBatch batch = GpuBatch.open()) {
                Tensor y = x.cos();
                z = y.sin();
                y.close();
                expect(WgpuBackend.memoryStats().quarantined() > 0, "freed-in-flight region is quarantined");
                w = x.exp();
            }
            expect(maxError(z.toFloatArray(), host, v -> Math.sin(Math.cos(v))) < 2 * WGSL_TOL, "z = sin(cos(x)) intact");
            expect(maxError(w.toFloatArray(), host, Math::exp) < 1e-5, "w = exp(x) intact");
            z.close();
            w.close();
        }
    }

    private static void batching() {
        section("Command batching");
        int n = 4096;
        float[] host = ramp(n, -1f, 1f);
        try (Tensor x = Tensor.fromArray(host, n); Tensor acc = Tensor.empty(DType.f32, n)) {
            WgpuBackend.synchronize();
            EngineStats before = WgpuBackend.engineStats();
            try (GpuBatch batch = GpuBatch.open()) {
                x.sin(acc);
                for (int i = 1; i < 64; i++) {
                    acc.sin(acc);
                }
            }
            EngineStats after = WgpuBackend.engineStats();
            expect(after.dispatches() - before.dispatches() == 64, "64 dispatches recorded");
            expect(after.submissions() - before.submissions() == 1, "64 ops -> 1 queue.submit");
            float[] got = acc.toFloatArray();
            double worst = 0;
            for (int i = 0; i < n; i++) {
                double v = host[i];
                for (int r = 0; r < 64; r++) {
                    v = Math.sin(v);
                }
                worst = Math.max(worst, Math.abs(got[i] - v));
            }
            expect(worst < 64 * WGSL_TOL, String.format("sin^64(x) correct, max err %.2e", worst));

            double unbatched = time(5, () -> {
                for (int i = 0; i < 100; i++) {
                    acc.sin(acc);
                }
            });
            double batched = time(5, () -> {
                try (GpuBatch batch = GpuBatch.open()) {
                    for (int i = 0; i < 100; i++) {
                        acc.sin(acc);
                    }
                }
            });
            System.out.printf("      100 small ops: unbatched %.2f ms, batched %.2f ms (%.1fx)%n",
                    unbatched, batched, unbatched / batched);
        }
    }

    private static void leakCheck() {
        section("Leak check");
        MemoryStats s = WgpuBackend.memoryStats();
        expect(s.tensorsDevice() + s.tensorsHost() + s.tensorsEvicted() == 0, "no tensors leaked");
        expect(s.slabRegions() == s.quarantined(), "no live slab regions besides quarantined ones");
    }

    // ------------------------------------------------------------------------------------------

    private static double time(int reps, Runnable body) {
        body.run(); // warm-up
        WgpuBackend.synchronize();
        long t0 = System.nanoTime();
        for (int i = 0; i < reps; i++) {
            body.run();
        }
        WgpuBackend.synchronize();
        return (System.nanoTime() - t0) / 1e6 / reps;
    }

    private static float[] ramp(int n, float lo, float hi) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = lo + (hi - lo) * i / Math.max(1, n - 1);
        }
        return out;
    }

    private static float[] random(SplittableRandom rng, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) (rng.nextDouble() * 2 - 1);
        }
        return out;
    }

    private static double maxError(float[] got, float[] input, DoubleUnaryOperator ref) {
        double worst = 0;
        for (int i = 0; i < got.length; i++) {
            double want = ref.applyAsDouble(input[i]);
            worst = Math.max(worst, Math.abs(got[i] - want) / Math.max(1.0, Math.abs(want)));
        }
        return worst;
    }

    private static double maxError(float[] a, float[] b) {
        double worst = 0;
        for (int i = 0; i < a.length; i++) {
            worst = Math.max(worst, Math.abs(a[i] - b[i]));
        }
        return worst;
    }

    private static long countBad(float[] got, float[] input, DoubleUnaryOperator ref, double tol) {
        long bad = 0;
        for (int i = 0; i < got.length; i++) {
            double want = ref.applyAsDouble(input[i]);
            if (Math.abs(got[i] - want) > tol * Math.max(1.0, Math.abs(want))) {
                bad++;
            }
        }
        return bad;
    }

    private static void section(String title) {
        System.out.println("\n== " + title);
    }

    private static void expect(boolean condition, String what) {
        checks++;
        System.out.println((condition ? "  PASS " : "  FAIL ") + what);
        if (!condition) {
            System.exit(1);
        }
    }

    private static void expectThrows(Class<? extends Throwable> type, Runnable body, String what) {
        try {
            body.run();
            expect(false, what + " should throw " + type.getSimpleName());
        } catch (Throwable t) {
            expect(type.isInstance(t), what + " -> " + t.getClass().getSimpleName());
        }
    }
}
