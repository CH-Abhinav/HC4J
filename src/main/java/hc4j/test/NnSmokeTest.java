package hc4j.test;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.engine.EngineStats;
import hc4j.engine.GpuBatch;
import hc4j.engine.WgpuBackend;
import hc4j.ops.NnOps;
import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.DoubleUnaryOperator;

/**
 * End-to-end validation of the neural-network ops against the real GPU and CPU references:
 * activations (including the fused forms), full and row-wise reductions, argmax, softmax and
 * log-softmax, layer norm and RMS norm, cross entropy, bias broadcast, and a two-layer MLP.
 *
 * <p>Run with {@code gradlew nnSmokeTest}; set {@code HC4J_BACKEND=vulkan} to repeat it on the
 * other backend. Exits non-zero on the first failure.
 */
public final class NnSmokeTest {

    private static int checks = 0;

    public static void main(String[] args) {
        EngineStats engine = WgpuBackend.engineStats();
        section("Engine");
        System.out.printf("      %s %s GPU%n", engine.deviceTypeName(), engine.backendName());

        activations();
        fusedActivations();
        fullReductions();
        rowReductions();
        softmax();
        normalisation();
        crossEntropy();
        linearLayer();
        mlp();
        validation();
        leakCheck();
        System.out.printf("%nALL %d CHECKS PASSED%n", checks);
    }

    // ------------------------------------------------------------------------------------------

    private static final float ALPHA = 0.125f;

    /** The f64 reference for each activation, matching the WGSL definitions. */
    private static DoubleUnaryOperator reference(NnOps.Activation act) {
        return switch (act) {
            case RELU -> x -> Math.max(x, 0);
            case LEAKY_RELU -> x -> x > 0 ? x : ALPHA * x;
            case ELU -> x -> x > 0 ? x : ALPHA * (Math.exp(x) - 1);
            case GELU -> x -> 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
            case SILU -> x -> x / (1 + Math.exp(-x));
            case SIGMOID -> x -> 1 / (1 + Math.exp(-x));
            case SOFTPLUS -> x -> Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
        };
    }

    private static void activations() {
        section("Activations");
        // 1003 is not a multiple of 4, so the scalar tail path runs too; the
        // extremes catch formulations that overflow instead of saturating.
        int n = 1003;
        float[] host = random(new SplittableRandom(1), n);
        host[0] = 0f;
        host[1] = 88f;
        host[2] = -88f;
        host[3] = 1e-8f;

        try (Tensor a = Tensor.fromArray(host, n); Tensor out = Tensor.empty(DType.f32, n)) {
            for (NnOps.Activation act : NnOps.Activation.values()) {
                NnOps.activation(act, a, out, ALPHA);
                float[] got = out.toFloatArray();
                DoubleUnaryOperator ref = reference(act);
                expect(maxError(got, host, ref) < 1e-5,
                        act + " matches the CPU reference (max rel err " + fmt(maxError(got, host, ref)) + ")");
                expect(allFinite(got), act + " is finite at x = +-88");
            }
        }

        // The fluent forms, and in place.
        try (Tensor a = Tensor.fromArray(host, n)) {
            try (Tensor r = a.gelu()) {
                expect(maxError(r.toFloatArray(), host, reference(NnOps.Activation.GELU)) < 1e-5, "gelu() allocates");
            }
            a.relu(a);
            expect(maxError(a.toFloatArray(), host, reference(NnOps.Activation.RELU)) < 1e-6, "relu in place");
        }

        try (Tensor a = Tensor.fromArray(host, n); Tensor r = a.leakyRelu(0.01f)) {
            expect(maxError(r.toFloatArray(), host, x -> x > 0 ? x : 0.01 * x) < 1e-6, "leakyRelu(0.01)");
        }
    }

    private static void fusedActivations() {
        section("Fused activations");
        int n = 4096;
        float[] hx = random(new SplittableRandom(2), n);
        try (Tensor x = Tensor.fromArray(hx, n)) {
            EngineStats before = WgpuBackend.engineStats();
            try (Tensor r = x.lazy().gelu().mul(2f).eval()) {
                EngineStats after = WgpuBackend.engineStats();
                DoubleUnaryOperator gelu = reference(NnOps.Activation.GELU);
                expect(maxError(r.toFloatArray(), hx, v -> 2 * gelu.applyAsDouble(v)) < 1e-5,
                        "fused gelu(x) * 2 matches the CPU reference");
                expect(after.dispatches() - before.dispatches() == 1, "the whole expression is one dispatch");
            }
            // The fused and standalone forms are generated from the same WGSL
            // body, so they must agree to within shader-compiler latitude
            // (instruction selection, FMA contraction) and nothing more.
            try (Tensor fused = x.lazy().silu().eval(); Tensor direct = x.silu()) {
                expect(maxError(fused.toFloatArray(), direct.toFloatArray()) < 1e-7,
                        "fused silu agrees with the standalone kernel");
            }
            try (Tensor r = x.lazy().relu().sigmoid().softplus().eval()) {
                expect(allFinite(r.toFloatArray()), "three activations in one kernel");
            }
        }
    }

    private static void fullReductions() {
        section("Full reductions");
        // Past 256 * 4 * 1024 elements pass 1 saturates and the grid-stride
        // loop covers the rest.
        for (int n : new int[] {1, 1000, 1_048_577}) {
            float[] host = random(new SplittableRandom(3 + n), n);
            double sum = 0;
            double max = Double.NEGATIVE_INFINITY;
            double min = Double.POSITIVE_INFINITY;
            double sumSq = 0;
            for (float v : host) {
                sum += v;
                max = Math.max(max, v);
                min = Math.min(min, v);
                sumSq += (double) v * v;
            }
            try (Tensor a = Tensor.fromArray(host, n)) {
                double tol = 1e-6 * Math.max(1, Math.log(n) / Math.log(2)) * Math.max(1, Math.abs(sum));
                expect(Math.abs(a.sum() - sum) <= tol, "sum of " + n + " elements");
                expect(Math.abs(a.mean() - sum / n) <= tol, "mean of " + n + " elements");
                expect(a.maxValue() == (float) max, "max of " + n + " elements");
                expect(a.minValue() == (float) min, "min of " + n + " elements");
                expect(Math.abs(a.sumSquares() - sumSq) <= 1e-5 * Math.max(1, sumSq), "sum of squares");
            }
        }

        // MSE folds the difference into the reduction: no intermediate tensor.
        int n = 8192;
        float[] p = random(new SplittableRandom(4), n);
        float[] t = random(new SplittableRandom(5), n);
        double want = 0;
        for (int i = 0; i < n; i++) {
            want += ((double) p[i] - t[i]) * ((double) p[i] - t[i]);
        }
        want /= n;
        try (Tensor pred = Tensor.fromArray(p, n); Tensor target = Tensor.fromArray(t, n)) {
            EngineStats before = WgpuBackend.engineStats();
            double got = pred.mse(target);
            EngineStats after = WgpuBackend.engineStats();
            expect(Math.abs(got - want) <= 1e-6 * Math.max(1, want), "mse(pred, target)");
            // Two dispatches: the partial pass and the final pass.
            expect(after.dispatches() - before.dispatches() == 2, "a full reduction is exactly two dispatches");
        }
    }

    private static void rowReductions() {
        section("Row reductions along the last axis");
        // Rank 3: [4, 8, 128] is 32 rows of 128, reducing to [4, 8].
        int d0 = 4;
        int d1 = 8;
        int cols = 128;
        int rows = d0 * d1;
        float[] host = random(new SplittableRandom(6), rows * cols);

        try (Tensor a = Tensor.fromArray(host, d0, d1, cols)) {
            NnOps.Rows dims = NnOps.rows(a);
            expect(dims.rows() == rows && dims.cols() == cols, "[4,8,128] is " + rows + " rows of " + cols);

            try (Tensor sums = a.sumRows()) {
                expect(Arrays.equals(sums.internalShapeUnsafe(), new int[] {d0, d1}),
                        "the reduced shape drops the last axis: " + Arrays.toString(sums.internalShapeUnsafe()));
                float[] got = sums.toFloatArray();
                double worst = 0;
                for (int r = 0; r < rows; r++) {
                    double want = 0;
                    for (int c = 0; c < cols; c++) {
                        want += host[r * cols + c];
                    }
                    worst = Math.max(worst, Math.abs(got[r] - want));
                }
                expect(worst < 1e-4, "row sums (max abs err " + fmt(worst) + ")");
            }

            try (Tensor means = a.meanRows(); Tensor maxes = a.maxRows(); Tensor mins = a.minRows()) {
                float[] gm = means.toFloatArray();
                float[] gx = maxes.toFloatArray();
                float[] gn = mins.toFloatArray();
                boolean ok = true;
                for (int r = 0; r < rows; r++) {
                    double sum = 0;
                    float hi = Float.NEGATIVE_INFINITY;
                    float lo = Float.POSITIVE_INFINITY;
                    for (int c = 0; c < cols; c++) {
                        float v = host[r * cols + c];
                        sum += v;
                        hi = Math.max(hi, v);
                        lo = Math.min(lo, v);
                    }
                    ok &= Math.abs(gm[r] - sum / cols) < 1e-5 && gx[r] == hi && gn[r] == lo;
                }
                expect(ok, "row mean, max and min");
            }

            // A 1-D tensor is one row, reducing to [1].
            try (Tensor v = Tensor.fromArray(new float[] {1f, 2f, 3f, 4f}, 4); Tensor s = v.sumRows()) {
                expect(Arrays.equals(s.internalShapeUnsafe(), new int[] {1}) && s.toFloatArray()[0] == 10f,
                        "a 1-D tensor reduces as a single row");
            }

            try (Tensor idx = a.argmaxRows()) {
                expect(idx.getDType() == DType.i32, "argmax returns i32 indices");
                int[] got = idx.toIntArray();
                boolean ok = true;
                for (int r = 0; r < rows; r++) {
                    int best = 0;
                    for (int c = 1; c < cols; c++) {
                        if (host[r * cols + c] > host[r * cols + best]) {
                            best = c;
                        }
                    }
                    ok &= got[r] == best;
                }
                expect(ok, "argmax index per row");
            }
            try (Tensor idx = a.argminRows()) {
                int[] got = idx.toIntArray();
                boolean ok = true;
                for (int r = 0; r < rows; r++) {
                    int best = 0;
                    for (int c = 1; c < cols; c++) {
                        if (host[r * cols + c] < host[r * cols + best]) {
                            best = c;
                        }
                    }
                    ok &= got[r] == best;
                }
                expect(ok, "argmin index per row");
            }
        }
    }

    private static void softmax() {
        section("Softmax");
        int rows = 11;
        int cols = 500;
        float[] host = random(new SplittableRandom(7), rows * cols);
        // A row that would overflow exp() without the max subtraction.
        for (int c = 0; c < cols; c++) {
            host[c] = 90f + c;
        }

        try (Tensor a = Tensor.fromArray(host, rows, cols); Tensor r = a.softmax()) {
            float[] got = r.toFloatArray();
            expect(allFinite(got), "softmax is finite for logits around 590");
            double worstSum = 0;
            double worstVal = 0;
            for (int row = 0; row < rows; row++) {
                double[] want = rowSoftmax(host, row, cols);
                double total = 0;
                for (int c = 0; c < cols; c++) {
                    total += got[row * cols + c];
                    worstVal = Math.max(worstVal, Math.abs(got[row * cols + c] - want[c]));
                }
                worstSum = Math.max(worstSum, Math.abs(total - 1));
            }
            expect(worstVal < 1e-6, "softmax matches the CPU reference (max abs err " + fmt(worstVal) + ")");
            expect(worstSum < 1e-4, "every row sums to 1 (max dev " + fmt(worstSum) + ")");
        }

        try (Tensor a = Tensor.fromArray(host, rows, cols); Tensor r = a.logSoftmax()) {
            float[] got = r.toFloatArray();
            double worst = 0;
            for (int row = 0; row < rows; row++) {
                double[] want = rowSoftmax(host, row, cols);
                for (int c = 0; c < cols; c++) {
                    worst = Math.max(worst, Math.abs(got[row * cols + c] - Math.log(want[c])));
                }
            }
            expect(worst < 1e-4, "log-softmax matches log(softmax) (max abs err " + fmt(worst) + ")");
        }

        // In place, which the row kernels allow.
        try (Tensor a = Tensor.fromArray(host, rows, cols)) {
            a.softmax(a);
            double total = 0;
            float[] got = a.toFloatArray();
            for (int c = 0; c < cols; c++) {
                total += got[c];
            }
            expect(Math.abs(total - 1) < 1e-4, "softmax runs in place");
        }
    }

    private static void normalisation() {
        section("Layer norm and RMS norm");
        int rows = 9;
        int cols = 777;
        float eps = 1e-5f;
        float[] host = random(new SplittableRandom(8), rows * cols);
        float[] gammaH = new float[cols];
        float[] betaH = new float[cols];
        for (int c = 0; c < cols; c++) {
            gammaH[c] = 0.5f + (c % 7) * 0.1f;
            betaH[c] = -0.25f + (c % 5) * 0.05f;
        }

        try (Tensor a = Tensor.fromArray(host, rows, cols);
                Tensor gamma = Tensor.fromArray(gammaH, cols);
                Tensor beta = Tensor.fromArray(betaH, cols)) {

            try (Tensor r = a.layerNorm(gamma, beta, eps)) {
                float[] got = r.toFloatArray();
                double worst = 0;
                for (int row = 0; row < rows; row++) {
                    double mean = 0;
                    for (int c = 0; c < cols; c++) {
                        mean += host[row * cols + c];
                    }
                    mean /= cols;
                    double var = 0;
                    for (int c = 0; c < cols; c++) {
                        double d = host[row * cols + c] - mean;
                        var += d * d;
                    }
                    var /= cols;
                    double inv = 1 / Math.sqrt(var + eps);
                    for (int c = 0; c < cols; c++) {
                        double want = (host[row * cols + c] - mean) * inv * gammaH[c] + betaH[c];
                        worst = Math.max(worst, Math.abs(got[row * cols + c] - want));
                    }
                }
                expect(worst < 1e-4, "layer norm with affine (max abs err " + fmt(worst) + ")");
            }

            try (Tensor r = a.layerNorm(eps)) {
                float[] got = r.toFloatArray();
                double worstMean = 0;
                double worstVar = 0;
                for (int row = 0; row < rows; row++) {
                    double mean = 0;
                    for (int c = 0; c < cols; c++) {
                        mean += got[row * cols + c];
                    }
                    mean /= cols;
                    double var = 0;
                    for (int c = 0; c < cols; c++) {
                        var += (got[row * cols + c] - mean) * (got[row * cols + c] - mean);
                    }
                    var /= cols;
                    worstMean = Math.max(worstMean, Math.abs(mean));
                    worstVar = Math.max(worstVar, Math.abs(var - 1));
                }
                expect(worstMean < 1e-4, "every normalised row is zero-mean (max " + fmt(worstMean) + ")");
                expect(worstVar < 1e-3, "every normalised row is unit-variance (max dev " + fmt(worstVar) + ")");
            }

            try (Tensor r = a.rmsNorm(gamma, eps)) {
                float[] got = r.toFloatArray();
                double worst = 0;
                for (int row = 0; row < rows; row++) {
                    double ms = 0;
                    for (int c = 0; c < cols; c++) {
                        ms += (double) host[row * cols + c] * host[row * cols + c];
                    }
                    double inv = 1 / Math.sqrt(ms / cols + eps);
                    for (int c = 0; c < cols; c++) {
                        double want = host[row * cols + c] * inv * gammaH[c];
                        worst = Math.max(worst, Math.abs(got[row * cols + c] - want));
                    }
                }
                expect(worst < 1e-4, "RMS norm with gamma (max abs err " + fmt(worst) + ")");
            }

            try (Tensor r = a.rmsNorm(null, eps)) {
                expect(allFinite(r.toFloatArray()), "RMS norm without gamma");
            }
        }
    }

    private static void crossEntropy() {
        section("Cross entropy");
        int rows = 13;
        int cols = 101;
        float[] logits = random(new SplittableRandom(9), rows * cols);
        int[] targets = new int[rows];
        for (int r = 0; r < rows; r++) {
            targets[r] = (r * 7) % cols;
        }

        try (Tensor a = Tensor.fromArray(logits, rows, cols);
                Tensor t = Tensor.fromArray(targets, rows);
                Tensor loss = a.crossEntropy(t)) {
            float[] got = loss.toFloatArray();
            expect(Arrays.equals(loss.internalShapeUnsafe(), new int[] {rows}), "loss has one value per row");
            double worst = 0;
            for (int r = 0; r < rows; r++) {
                double[] soft = rowSoftmax(logits, r, cols);
                worst = Math.max(worst, Math.abs(got[r] + Math.log(soft[targets[r]])));
            }
            expect(worst < 1e-5, "cross entropy equals -log(softmax[target]) (max abs err " + fmt(worst) + ")");

            // The batch loss is the mean over rows.
            float batch = loss.mean();
            double want = 0;
            for (float v : got) {
                want += v;
            }
            expect(Math.abs(batch - want / rows) < 1e-5, "mean over rows gives the batch loss");
        }

        // A class index outside [0, cols) must surface as NaN, not read a
        // neighbouring row.
        int[] bad = new int[rows];
        Arrays.fill(bad, cols);
        try (Tensor a = Tensor.fromArray(logits, rows, cols);
                Tensor t = Tensor.fromArray(bad, rows);
                Tensor loss = a.crossEntropy(t)) {
            boolean allNan = true;
            for (float v : loss.toFloatArray()) {
                allNan &= Float.isNaN(v);
            }
            expect(allNan, "an out-of-range class yields NaN");
        }
    }

    private static void linearLayer() {
        section("Linear layer");
        int m = 24;
        int k = 40;
        int n = 16;
        SplittableRandom rng = new SplittableRandom(10);
        float[] hx = random(rng, m * k);
        float[] hw = random(rng, k * n);
        float[] hb = random(rng, n);

        try (Tensor x = Tensor.fromArray(hx, m, k);
                Tensor w = Tensor.fromArray(hw, k, n);
                Tensor b = Tensor.fromArray(hb, n);
                Tensor y = x.linear(w, b)) {
            float[] got = y.toFloatArray();
            expect(Arrays.equals(y.internalShapeUnsafe(), new int[] {m, n}), "linear gives [m, n]");
            double worst = 0;
            for (int r = 0; r < m; r++) {
                for (int c = 0; c < n; c++) {
                    double acc = 0;
                    for (int i = 0; i < k; i++) {
                        acc += (double) hx[r * k + i] * hw[i * n + c];
                    }
                    worst = Math.max(worst, Math.abs(got[r * n + c] - (acc + hb[c])));
                }
            }
            expect(worst < 1e-4, "x.w + b matches the CPU reference (max abs err " + fmt(worst) + ")");
        }

        // The bias broadcast on its own, in place.
        try (Tensor a = Tensor.fromArray(hx, m, k); Tensor bias = Tensor.fromArray(random(rng, k), k)) {
            float[] bh = bias.toFloatArray();
            a.addRowBias(bias, a);
            float[] got = a.toFloatArray();
            double worst = 0;
            for (int i = 0; i < got.length; i++) {
                worst = Math.max(worst, Math.abs(got[i] - (hx[i] + bh[i % k])));
            }
            expect(worst < 1e-6, "addRowBias broadcasts over rows, in place");
        }
    }

    private static void mlp() {
        section("Two-layer MLP, batched");
        int batch = 32;
        int in = 64;
        int hidden = 128;
        int out = 10;
        SplittableRandom rng = new SplittableRandom(11);
        float[] hx = random(rng, batch * in);
        float[] hw1 = random(rng, in * hidden);
        float[] hb1 = random(rng, hidden);
        float[] hw2 = random(rng, hidden * out);
        float[] hb2 = random(rng, out);

        try (Tensor x = Tensor.fromArray(hx, batch, in);
                Tensor w1 = Tensor.fromArray(hw1, in, hidden);
                Tensor b1 = Tensor.fromArray(hb1, hidden);
                Tensor w2 = Tensor.fromArray(hw2, hidden, out);
                Tensor b2 = Tensor.fromArray(hb2, out);
                Tensor h = Tensor.empty(DType.f32, batch, hidden);
                Tensor logits = Tensor.empty(DType.f32, batch, out);
                Tensor probs = Tensor.empty(DType.f32, batch, out)) {

            EngineStats before = WgpuBackend.engineStats();
            try (GpuBatch ignored = GpuBatch.open()) {
                x.linear(w1, b1, h);
                h.gelu(h);
                h.linear(w2, b2, logits);
                logits.softmax(probs);
            }
            EngineStats after = WgpuBackend.engineStats();
            expect(after.submissions() - before.submissions() == 1, "the whole forward pass is one submission");

            float[] got = probs.toFloatArray();
            // CPU reference for the same network.
            double[] hidRef = new double[batch * hidden];
            for (int r = 0; r < batch; r++) {
                for (int c = 0; c < hidden; c++) {
                    double acc = hb1[c];
                    for (int i = 0; i < in; i++) {
                        acc += (double) hx[r * in + i] * hw1[i * hidden + c];
                    }
                    hidRef[r * hidden + c] = reference(NnOps.Activation.GELU).applyAsDouble(acc);
                }
            }
            float[] logitRef = new float[batch * out];
            for (int r = 0; r < batch; r++) {
                for (int c = 0; c < out; c++) {
                    double acc = hb2[c];
                    for (int i = 0; i < hidden; i++) {
                        acc += hidRef[r * hidden + i] * hw2[i * out + c];
                    }
                    logitRef[r * out + c] = (float) acc;
                }
            }
            double worst = 0;
            double worstSum = 0;
            for (int r = 0; r < batch; r++) {
                double[] want = rowSoftmax(logitRef, r, out);
                double total = 0;
                for (int c = 0; c < out; c++) {
                    worst = Math.max(worst, Math.abs(got[r * out + c] - want[c]));
                    total += got[r * out + c];
                }
                worstSum = Math.max(worstSum, Math.abs(total - 1));
            }
            expect(worst < 1e-4, "softmax(gelu(x.W1 + b1).W2 + b2) matches CPU (max abs err " + fmt(worst) + ")");
            expect(worstSum < 1e-5, "every output row is a distribution");

            // argmax of the probabilities is the predicted class.
            try (Tensor pred = probs.argmaxRows()) {
                int[] cls = pred.toIntArray();
                boolean ok = true;
                for (int r = 0; r < batch; r++) {
                    int best = 0;
                    for (int c = 1; c < out; c++) {
                        if (got[r * out + c] > got[r * out + best]) {
                            best = c;
                        }
                    }
                    ok &= cls[r] == best;
                }
                expect(ok, "argmax of the output row is the prediction");
            }
        }
    }

    private static void validation() {
        section("Shape and dtype validation");
        // Every operand is allocated here, not inside a lambda: a tensor
        // created in a throwing expression would never be closed, and the
        // leak check below would report it.
        try (Tensor a = Tensor.fromArray(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
                Tensor f32x3 = Tensor.empty(DType.f32, 3);
                Tensor f32x5 = Tensor.empty(DType.f32, 5);
                Tensor i32x2 = Tensor.empty(DType.i32, 2);
                Tensor i32x5 = Tensor.empty(DType.i32, 5)) {

            expectThrows(IllegalArgumentException.class, () -> a.sumRows(f32x3),
                    "a row reduction into the wrong shape");
            expectThrows(IllegalArgumentException.class, () -> NnOps.reduce(NnOps.Reduction.SUM, a, f32x3),
                    "a full reduction into more than one element");
            expectThrows(IllegalArgumentException.class, () -> a.argmaxRows(i32x5),
                    "argmax into the wrong number of rows");
            expectThrows(IllegalArgumentException.class, () -> NnOps.argmaxRows(a, f32x5, true),
                    "argmax into an f32 tensor");
            expectThrows(IllegalArgumentException.class, () -> a.crossEntropy(f32x3),
                    "cross entropy with f32 targets");
            expectThrows(IllegalArgumentException.class, () -> a.layerNorm(f32x3, null, 1e-5f),
                    "layer norm with gamma but no beta");
            expectThrows(IllegalArgumentException.class, () -> a.rmsNorm(null, -1f),
                    "a negative epsilon");
            expectThrows(IllegalArgumentException.class, () -> a.addRowBias(f32x5, a),
                    "a bias whose length is not the last axis");
            expectThrows(IllegalArgumentException.class, i32x2::relu,
                    "an activation on an i32 tensor");
            expectThrows(IllegalArgumentException.class, () -> a.leakyRelu(Float.NaN, a),
                    "a non-finite slope");
        }

        // The row kernels index linearly, so a strided view must be refused
        // rather than silently read in the wrong order.
        try (Tensor m = Tensor.fromArray(new float[] {1, 2, 3, 4, 5, 6}, 2, 3); Tensor view = m.transpose()) {
            expect(view.toFloatArray()[1] == 4f, "transpose materialises a contiguous tensor");
        }
    }

    private static void leakCheck() {
        section("Leak check");
        var stats = WgpuBackend.memoryStats();
        expect(stats.tensorsDevice() + stats.tensorsHost() + stats.tensorsEvicted() == 0,
                "no tensors leaked (dev=" + stats.tensorsDevice() + " host=" + stats.tensorsHost()
                        + " disk=" + stats.tensorsEvicted() + ")");
    }

    // ------------------------------------------------------------------------------------------

    /** Stable row softmax in f64, the reference for both softmax and the loss. */
    private static double[] rowSoftmax(float[] data, int row, int cols) {
        double max = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < cols; c++) {
            max = Math.max(max, data[row * cols + c]);
        }
        double[] out = new double[cols];
        double sum = 0;
        for (int c = 0; c < cols; c++) {
            out[c] = Math.exp(data[row * cols + c] - max);
            sum += out[c];
        }
        for (int c = 0; c < cols; c++) {
            out[c] /= sum;
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

    private static boolean allFinite(float[] values) {
        for (float v : values) {
            if (!Float.isFinite(v)) {
                return false;
            }
        }
        return true;
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

    private static String fmt(double v) {
        return String.format("%.2e", v);
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
