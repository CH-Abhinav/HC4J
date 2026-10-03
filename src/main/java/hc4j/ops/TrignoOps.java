package hc4j.ops;

import static java.lang.foreign.ValueLayout.JAVA_INT;

import hc4j.Tensor;
import hc4j.engine.WgpuBackend;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Locale;
import java.util.Objects;

/**
 * Typed dispatch for the elementwise trigonometric suite, caller-allocated: the output tensor is
 * passed in, so buffers can be reused and ops can run in place ({@code apply(SIN, a, a)}).
 *
 * <p>No tensor data crosses PCIe: only handles (and, for strided layouts, shape/stride arrays) do.
 * If VRAM cannot hold the working set, the native tiered memory manager evicts or streams
 * transparently; the caller only sees a {@link hc4j.engine.GpuOutOfMemoryException} when every
 * memory tier is exhausted.
 */
public final class TrignoOps {

    /** The kernels exported by {@code ops/trigno.rs}, each as {@code dispatch_<name>_f32}. */
    public enum Function {
        SIN, COS, TAN,
        ASIN, ACOS, ATAN,
        SINH, COSH, TANH,
        ASINH, ACOSH, ATANH;

        private final String opName = name().toLowerCase(Locale.ROOT);

        public String opName() {
            return opName;
        }
    }

    private TrignoOps() {}

    /**
     * Computes {@code out = fn(input)} elementwise and returns {@code out}. {@code out} may be
     * {@code input} (in place).
     *
     * @throws IllegalArgumentException if either tensor is not f32 or the shapes differ
     * @throws hc4j.engine.GpuOutOfMemoryException if no memory tier can hold the working set
     */
    public static Tensor apply(Function fn, Tensor input, Tensor out) {
        Objects.requireNonNull(fn, "fn");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(out, "out");
        Layouts.requireF32(fn.opName(), input, out);
        Layouts.requireSameShape(fn.opName(), input, out);

        int status;
        try {
            if (Layouts.isContiguous(input) && Layouts.isContiguous(out)) {
                // Fast path: no layout arrays, no Arena.
                status = TrignoNative.dispatch(fn, input.getVramId(), out.getVramId(), out.dim(),
                        MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL, out.getSize(), 1);
            } else {
                try (Arena arena = Arena.ofConfined()) {
                    status = TrignoNative.dispatch(fn, input.getVramId(), out.getVramId(), out.dim(),
                            arena.allocateFrom(JAVA_INT, out.internalShapeUnsafe()),
                            arena.allocateFrom(JAVA_INT, input.internalStridesUnsafe()),
                            arena.allocateFrom(JAVA_INT, out.internalStridesUnsafe()),
                            out.getSize(), 0);
                }
            }
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native dispatch of " + fn.opName() + " failed", t);
        }
        WgpuBackend.checkStatus(status, "TrignoOps." + fn.opName());
        return out;
    }
}
