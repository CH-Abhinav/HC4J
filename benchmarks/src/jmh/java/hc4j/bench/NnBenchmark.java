/*
 * Copyright © 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package hc4j.bench;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.engine.Engine;
import hc4j.engine.GpuBatch;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;

/**
 * Neural network operations benchmark.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(1)
public class NnBenchmark {

    /** Rows of {@link #cols}; 1024 x 1024 is 4 MiB, one slab. */
    @Param({"1024"})
    public int rows;

    @Param({"1024"})
    public int cols;

    private Tensor x;
    private Tensor out;
    private Tensor rowOut;
    private Tensor scalar;
    private Tensor gamma;
    private Tensor beta;
    private Tensor targets;

    // Feed-forward layer: [rows, cols] . [cols, hidden] + bias.
    private static final int HIDDEN = 256;
    private Tensor weights;
    private Tensor bias;
    private Tensor hidden;

    @Setup(Level.Trial)
    public void setUp() {
        SplittableRandom rng = new SplittableRandom(17);
        x = Tensor.fromArray(random(rng, rows * cols), rows, cols);
        out = Tensor.empty(DType.f32, rows, cols);
        rowOut = Tensor.empty(DType.f32, rows);
        scalar = Tensor.empty(DType.f32, 1);
        gamma = Tensor.fromArray(random(rng, cols), cols);
        beta = Tensor.fromArray(random(rng, cols), cols);
        int[] classes = new int[rows];
        for (int r = 0; r < rows; r++) {
            classes[r] = r % cols;
        }
        targets = Tensor.fromArray(classes, rows);
        weights = Tensor.fromArray(random(rng, cols * HIDDEN), cols, HIDDEN);
        bias = Tensor.fromArray(random(rng, HIDDEN), HIDDEN);
        hidden = Tensor.empty(DType.f32, rows, HIDDEN);
    }

    private static float[] random(SplittableRandom rng, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) (rng.nextDouble() * 2 - 1);
        }
        return out;
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        for (Tensor t : new Tensor[] {x, out, rowOut, scalar, gamma, beta, targets, weights, bias, hidden}) {
            if (t != null) {
                t.close();
            }
        }
    }

    @Benchmark
    public void relu() {
        x.relu(out);
        Engine.synchronize();
    }

    /** The transcendental one: tanh plus a cubic. */
    @Benchmark
    public void gelu() {
        x.gelu(out);
        Engine.synchronize();
    }

    /**
     * {@code (gelu(x) + x) * x} as three kernels and two intermediates. Only
     * tensor-tensor ops: the scalar forms are themselves fused, so they would
     * not be an honest baseline.
     */
    @Benchmark
    public void geluChainUnfused() {
        try (Tensor g = x.gelu(); Tensor s = g.add(x); Tensor r = s.mul(x)) {
            Engine.synchronize();
        }
    }

    /** The same expression as one fused kernel. */
    @Benchmark
    public void geluChainFused() {
        try (Tensor r = x.lazy().gelu().add(x).mul(x).eval()) {
            Engine.synchronize();
        }
    }

    /** Full reduction: two dispatches whatever the size. */
    @Benchmark
    public void sum() {
        x.sum(scalar);
        Engine.synchronize();
    }

    /** Reduction along the last axis: one workgroup per row. */
    @Benchmark
    public void sumRows() {
        x.sumRows(rowOut);
        Engine.synchronize();
    }

    /** Online max-and-sum, then a write pass: two reads of each row. */
    @Benchmark
    public void softmax() {
        x.softmax(out);
        Engine.synchronize();
    }

    /** Welford statistics, then a scale-and-shift pass. */
    @Benchmark
    public void layerNorm() {
        x.layerNorm(gamma, beta, 1e-5f, out);
        Engine.synchronize();
    }

    @Benchmark
    public void rmsNorm() {
        x.rmsNorm(gamma, 1e-5f, out);
        Engine.synchronize();
    }

    /** Log-sum-exp and a gather, with no softmax materialised. */
    @Benchmark
    public void crossEntropy() {
        x.crossEntropy(targets, rowOut);
        Engine.synchronize();
    }

    /** gelu(x . W + b): a GEMM, a bias pass and an activation, one submission. */
    @Benchmark
    public void feedForwardLayer() {
        try (GpuBatch ignored = GpuBatch.open()) {
            x.linear(weights, bias, hidden);
            hidden.gelu(hidden);
        }
        Engine.synchronize();
    }
}
