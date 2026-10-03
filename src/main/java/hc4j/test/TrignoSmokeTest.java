package hc4j.test;

import hc4j.Tensor;
import hc4j.engine.MemoryStats;
import hc4j.engine.WgpuBackend;
import hc4j.engine.WgpuBackend.Residency;
import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.function.UnaryOperator;

/**
 * End-to-end check of the trigonometric FFM bindings and the tiered memory manager. Budgets are
 * shrunk far below the device's real VRAM so eviction, disk spill, page-in and both streaming
 * paths run deterministically on any GPU.
 *
 * <p>Run with {@code gradlew trignoSmokeTest}. Exits non-zero on the first failure.
 */
public final class TrignoSmokeTest {

    private static final long MiB = 1L << 20;
    /**
     * WGSL precision is hardware-defined within spec bounds: sin/cos to 2^-11 absolute, inverse
     * trig inherited from atan2 (4096 ULP). Checking tighter would test the driver, not HC4J. A
     * wrong kernel is off by O(1), so this bound still catches real bugs.
     */
    private static final double WGSL_TOL = 1.0 / 2048;
    private static int checks = 0;

    public static void main(String[] args) {
        trigonometricSuite();
        arithmeticStillWorks();
        errorPaths();
        tieredMemory();
        System.out.printf("%nALL %d CHECKS PASSED%n", checks);
    }

    // ------------------------------------------------------------------------------------------

    private record Case(String name, UnaryOperator<Tensor> gpu, DoubleUnaryOperator cpu, float lo, float hi) {}

    private static void trigonometricSuite() {
        section("Trigonometric suite vs StrictMath (n = 1,000,003: exercises the vec4 tail)");
        List<Case> cases = List.of(
                new Case("sin", Tensor::sin, StrictMath::sin, -3f, 3f),
                new Case("cos", Tensor::cos, StrictMath::cos, -3f, 3f),
                new Case("tan", Tensor::tan, StrictMath::tan, -1.2f, 1.2f),
                new Case("asin", Tensor::asin, StrictMath::asin, -0.95f, 0.95f),
                new Case("acos", Tensor::acos, StrictMath::acos, -0.95f, 0.95f),
                new Case("atan", Tensor::atan, StrictMath::atan, -5f, 5f),
                new Case("sinh", Tensor::sinh, StrictMath::sinh, -5f, 5f),
                new Case("cosh", Tensor::cosh, StrictMath::cosh, -5f, 5f),
                new Case("tanh", Tensor::tanh, StrictMath::tanh, -5f, 5f),
                new Case("asinh", Tensor::asinh, x -> Math.log(x + Math.sqrt(x * x + 1)), -3f, 3f),
                new Case("acosh", Tensor::acosh, x -> Math.log(x + Math.sqrt(x * x - 1)), 1.05f, 10f),
                new Case("atanh", Tensor::atanh, x -> 0.5 * Math.log((1 + x) / (1 - x)), -0.95f, 0.95f));

        int n = 1_000_003;
        for (Case c : cases) {
            float[] host = ramp(n, c.lo, c.hi);
            try (Tensor input = Tensor.fromArray(host, n); Tensor out = c.gpu.apply(input)) {
                expect(out.getVramId() != input.getVramId(), c.name + " allocates a new tensor");
                expect(java.util.Arrays.equals(out.internalShapeUnsafe(), input.internalShapeUnsafe()),
                        c.name + " preserves shape");
                double err = maxRelError(out.toFloatArray(), host, c.cpu);
                expect(err < WGSL_TOL, String.format("%-5s max rel err %.2e", c.name, err));
                expect(java.util.Arrays.equals(input.toFloatArray(), host), c.name + " leaves input untouched");
            }
        }

        // Fluent chaining through several native allocations.
        float[] host = ramp(4099, -1f, 1f);
        try (Tensor x = Tensor.fromArray(host, 4099);
             Tensor s = x.sin();
             Tensor t = s.asin()) {
            double err = maxRelError(t.toFloatArray(), host, v -> v);
            expect(err < WGSL_TOL, String.format("asin(sin(x)) == x, max rel err %.2e", err));
        }

        // Multi-dimensional shape round-trips.
        try (Tensor m = Tensor.fromArray(ramp(6 * 7 * 5, -2f, 2f), 6, 7, 5); Tensor c = m.cos()) {
            expect(c.dim() == 3 && c.getSize() == 210, "3-D tensor keeps rank and size");
        }
    }

    private static void arithmeticStillWorks() {
        section("Arithmetic ops on the tiered manager");
        float[] a = ramp(10_000, 0f, 1f);
        float[] b = ramp(10_000, 1f, 2f);
        try (Tensor ta = Tensor.fromArray(a, 10_000);
             Tensor tb = Tensor.fromArray(b, 10_000);
             Tensor sum = ta.add(tb);
             Tensor self = ta.mul(ta)) {
            float[] s = sum.toFloatArray();
            float[] sq = self.toFloatArray();
            boolean ok = true;
            for (int i = 0; i < a.length; i++) {
                ok &= Math.abs(s[i] - (a[i] + b[i])) < 1e-6f && Math.abs(sq[i] - a[i] * a[i]) < 1e-6f;
            }
            expect(ok, "add and a.mul(a) (same handle twice) are exact");
        }
    }

    private static void errorPaths() {
        section("Error paths surface as Java exceptions, not native crashes");
        Tensor closed = Tensor.fromArray(new float[] {1f, 2f, 3f}, 3);
        closed.close();
        expectThrows(IllegalStateException.class, closed::sin, "sin() on a closed tensor");
        try (Tensor ints = Tensor.fromArray(new int[] {1, 2, 3}, 3)) {
            expectThrows(IllegalArgumentException.class, ints::cos, "cos() on an i32 tensor");
        }
    }

    // ------------------------------------------------------------------------------------------

    private static void tieredMemory() {
        section("Tiered memory: LRU eviction, disk spill, page-in");
        MemoryStats before = WgpuBackend.memoryStats();
        WgpuBackend.configureMemory(64 * MiB, 48 * MiB);

        int n = (int) (16 * MiB / Float.BYTES);
        List<Tensor> tensors = new ArrayList<>();
        List<float[]> expected = new ArrayList<>();
        for (int i = 0; i < 8; i++) { // 128 MiB of tensors against a 64 MiB VRAM budget
            float[] data = pattern(n, i);
            expected.add(data);
            tensors.add(Tensor.fromArray(data, n));
        }
        MemoryStats s = WgpuBackend.memoryStats();
        printStats(s);
        expect(s.vramUsed() <= 64 * MiB, "VRAM usage stays within budget: " + s.vramUsed() / MiB + " MiB");
        expect(s.hostUsed() <= 48 * MiB, "host usage stays within budget: " + s.hostUsed() / MiB + " MiB");
        expect(s.evictions() > before.evictions(), "LRU tensors were evicted from VRAM");
        expect(s.spills() > before.spills(), "host tier overflowed to disk");
        expect(WgpuBackend.residency(tensors.get(0).getVramId()) != Residency.DEVICE, "oldest tensor was evicted");
        expect(WgpuBackend.residency(tensors.get(7).getVramId()) == Residency.DEVICE, "newest tensor is resident");

        for (int i = 0; i < tensors.size(); i++) {
            expect(java.util.Arrays.equals(tensors.get(i).toFloatArray(), expected.get(i)),
                    "tensor " + i + " intact after migration (" + WgpuBackend.residency(tensors.get(i).getVramId()) + ")");
        }

        Tensor oldest = tensors.get(0);
        long pageInsBefore = WgpuBackend.memoryStats().pageIns();
        WgpuBackend.evict(oldest.getVramId());
        try (Tensor r = oldest.sin()) {
            expect(WgpuBackend.memoryStats().pageIns() > pageInsBefore, "sin() paged its evicted input back in");
            expect(WgpuBackend.residency(oldest.getVramId()) == Residency.DEVICE, "input is resident after page-in");
            expect(maxRelError(r.toFloatArray(), expected.get(0), StrictMath::sin) < WGSL_TOL, "sin of paged-in tensor");
        }

        section("Tiered memory: streaming (input larger than the whole VRAM budget)");
        WgpuBackend.configureMemory(24 * MiB, 0);
        int big = (int) (32 * MiB / Float.BYTES);
        float[] bigData = ramp(big, -4f, 4f);
        try (Tensor x = Tensor.fromArray(bigData, big)) {
            expect(WgpuBackend.residency(x.getVramId()) != Residency.DEVICE, "32 MiB tensor placed off-device");
            long streamedBefore = WgpuBackend.memoryStats().streamedOps();
            try (Tensor r = x.cos()) {
                expect(WgpuBackend.memoryStats().streamedOps() > streamedBefore, "cos() took the streaming path");
                expect(maxRelError(r.toFloatArray(), bigData, StrictMath::cos) < WGSL_TOL,
                        "streamed cos correct (output in " + WgpuBackend.residency(r.getVramId()) + ")");
            }
        }

        section("Tiered memory: disk -> disk streaming (host budget below tensor size)");
        WgpuBackend.configureMemory(24 * MiB, 16 * MiB);
        try (Tensor x = Tensor.fromArray(bigData, big)) {
            expect(WgpuBackend.residency(x.getVramId()) == Residency.DISK, "32 MiB tensor spilled straight to disk");
            try (Tensor r = x.tanh()) {
                expect(WgpuBackend.residency(r.getVramId()) == Residency.DISK, "streamed output landed on disk");
                expect(maxRelError(r.toFloatArray(), bigData, StrictMath::tanh) < WGSL_TOL, "disk-streamed tanh correct");
            }
        }

        section("Tiered memory: restore budgets, verify, leak check");
        WgpuBackend.configureMemory(before.vramBudget(), before.hostBudget());
        for (int i = 0; i < tensors.size(); i++) {
            expect(java.util.Arrays.equals(tensors.get(i).toFloatArray(), expected.get(i)), "tensor " + i + " intact");
            tensors.get(i).close();
        }
        MemoryStats end = WgpuBackend.memoryStats();
        printStats(end);
        expect(end.tensorsDevice() + end.tensorsHost() + end.tensorsEvicted() == 0, "no tensors leaked");
        // Empty slabs may be kept as a spare; everything else must be released.
        expect(end.slabRegions() == 0 && end.vramUsed() == end.slabBytes() && end.hostUsed() == 0,
                "all reservations released (spare slab capacity: " + end.slabBytes() / MiB + " MiB)");
    }

    // ------------------------------------------------------------------------------------------

    private static float[] ramp(int n, float lo, float hi) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = lo + (hi - lo) * i / Math.max(1, n - 1);
        }
        return out;
    }

    private static float[] pattern(int n, int seed) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) ((i * 31L + seed * 7919L) % 10007) / 10007f * 6f - 3f;
        }
        return out;
    }

    private static double maxRelError(float[] got, float[] input, DoubleUnaryOperator ref) {
        double worst = 0;
        for (int i = 0; i < got.length; i++) {
            double want = ref.applyAsDouble(input[i]);
            worst = Math.max(worst, Math.abs(got[i] - want) / Math.max(1.0, Math.abs(want)));
        }
        return worst;
    }

    private static void printStats(MemoryStats s) {
        System.out.printf(
                "      vram %d/%d MiB | host %d/%d MiB | spilled %d MiB | tensors dev=%d host=%d disk=%d%n"
                        + "      evictions=%d page-ins=%d spills=%d driver-OOMs=%d streamed-ops=%d%n",
                s.vramUsed() / MiB, s.vramBudget() / MiB, s.hostUsed() / MiB, s.hostBudget() / MiB,
                s.spilledBytes() / MiB, s.tensorsDevice(), s.tensorsHost(), s.tensorsEvicted(),
                s.evictions(), s.pageIns(), s.spills(), s.driverOoms(), s.streamedOps());
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
