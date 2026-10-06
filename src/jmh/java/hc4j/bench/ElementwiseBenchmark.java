package hc4j.bench;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.engine.GpuBatch;
import hc4j.engine.WgpuBackend;
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
 * Elementwise dispatch, kernel fusion and command batching.
 *
 * <p>GPU work is asynchronous, so every benchmark ends with
 * {@link WgpuBackend#synchronize()}: without it the measurement would only cover the cost of
 * recording commands. Shader compilation happens on the first dispatch and is absorbed by the
 * warm-up iterations.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(1)
public class ElementwiseBenchmark {

    /** Elements per tensor; 1 Mi floats is 4 MiB, a slab region. */
    @Param({"1048576"})
    public int elements;

    private Tensor a;
    private Tensor b;
    private Tensor c;
    private Tensor out;

    @Setup(Level.Trial)
    public void setUp() {
        float[] host = new float[elements];
        for (int i = 0; i < elements; i++) {
            host[i] = (i % 1000) * 0.001f - 0.5f;
        }
        a = Tensor.fromArray(host, elements);
        b = Tensor.fromArray(host, elements);
        c = Tensor.fromArray(host, elements);
        out = Tensor.empty(DType.f32, elements);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        for (Tensor t : new Tensor[] {a, b, c, out}) {
            if (t != null) {
                t.close();
            }
        }
    }

    /** One unary kernel, caller-allocated output. */
    @Benchmark
    public void sin() {
        a.sin(out);
        WgpuBackend.synchronize();
    }

    /** The same kernel in place, which binds one buffer twice. */
    @Benchmark
    public void sinInPlace() {
        out.sin(out);
        WgpuBackend.synchronize();
    }

    /** sin(a) + b) * c as three kernels and two intermediate tensors. */
    @Benchmark
    public void chainUnfused() {
        try (Tensor s = a.sin(); Tensor t = s.add(b); Tensor r = t.mul(c)) {
            WgpuBackend.synchronize();
        }
    }

    /** The same expression as one fused kernel and no intermediates. */
    @Benchmark
    public void chainFused() {
        try (Tensor r = a.lazy().sin().add(b).mul(c).eval()) {
            WgpuBackend.synchronize();
        }
    }

    /** 64 small ops, one submission each. */
    @Benchmark
    public void manyOpsUnbatched() {
        for (int i = 0; i < 64; i++) {
            out.sin(out);
        }
        WgpuBackend.synchronize();
    }

    /** The same 64 ops in a single command buffer. */
    @Benchmark
    public void manyOpsBatched() {
        try (GpuBatch batch = GpuBatch.open()) {
            for (int i = 0; i < 64; i++) {
                out.sin(out);
            }
        }
        WgpuBackend.synchronize();
    }
}
