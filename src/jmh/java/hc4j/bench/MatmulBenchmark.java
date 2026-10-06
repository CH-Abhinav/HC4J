package hc4j.bench;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.engine.WgpuBackend;
import hc4j.ops.MatmulOps;
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
 * Square GEMM and the two GEMV mappings. Divide {@code 2 * size^3} by the reported time for
 * FLOP/s; {@link MatmulOps#plan} reports which kernel the dispatcher picked for a shape.
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
        a.matmul(b, c);
        WgpuBackend.synchronize();
    }

    /** Matrix times vector: lanes per row plus a reduction. */
    @Benchmark
    public void gemvRows() {
        MatmulOps.matmul(a, vector, rowOut, false, false);
        WgpuBackend.synchronize();
    }

    /** Vector times matrix: a thread per column. */
    @Benchmark
    public void gemvCols() {
        MatmulOps.matmul(vector, a, colOut, false, false);
        WgpuBackend.synchronize();
    }

    /** In-VRAM transposition, the 32x33 padded-tile kernel. */
    @Benchmark
    public void transpose() {
        a.transpose(c);
        WgpuBackend.synchronize();
    }
}
