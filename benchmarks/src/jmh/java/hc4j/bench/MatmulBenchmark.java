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
import hc4j.linalg.Linalg;
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
 * Square GEMM and the two GEMV mappings benchmark.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(1)
public class MatmulBenchmark {

    @Param({"512", "1024"})
    public int size;

    private Tensor a;
    private Tensor b;
    private Tensor c;
    private Tensor vector;
    private Tensor rowOut;
    private Tensor colOut;

    @Setup(Level.Trial)
    public void setUp() {
        SplittableRandom rng = new SplittableRandom(7);
        a = Tensor.fromArray(random(rng, size * size), size, size);
        b = Tensor.fromArray(random(rng, size * size), size, size);
        c = Tensor.empty(DType.f32, size, size);
        vector = Tensor.fromArray(random(rng, size), size);
        rowOut = Tensor.empty(DType.f32, size);
        colOut = Tensor.empty(DType.f32, size);
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
        for (Tensor t : new Tensor[] {a, b, c, vector, rowOut, colOut}) {
            if (t != null) {
                t.close();
            }
        }
    }

    /** Register-tiled GEMM: 2 * size^3 FLOPs. */
    @Benchmark
    public void gemm() {
        Linalg.matmul(a, b, c);
        Engine.synchronize();
    }

    /** Matrix times vector: lanes per row plus a reduction. */
    @Benchmark
    public void gemvRows() {
        Linalg.matmul(a, vector, rowOut);
        Engine.synchronize();
    }

    /** Vector times matrix: a thread per column. */
    @Benchmark
    public void gemvCols() {
        Linalg.matmul(vector, a, colOut);
        Engine.synchronize();
    }

    /** In-VRAM transposition, the 32x33 padded-tile kernel. */
    @Benchmark
    public void transpose() {
        Linalg.transpose(a, c);
        Engine.synchronize();
    }
}
