package hc4j.ops;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import hc4j.Tensor;
import hc4j.engine.WgpuBackend;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.Objects;

/**
 * Exponential and logarithmic suite ({@code ops/exponential.rs}), with the same caller-allocated
 * elementwise ABI and binding style as {@link TrignoOps}.
 */
public final class ExponentialOps {

    /** Kernels and their native symbols. */
    public enum Function {
        EXP("exp", "dispatch_exp_f32"),
        LOG("log", "dispatch_ln_f32"),
        LOG2("log2", "dispatch_log2_f32"),
        LOG10("log10", "dispatch_log10_f32"),
        SQRT("sqrt", "dispatch_sqrt_f32");

        private final String opName;
        private final String symbol;

        Function(String opName, String symbol) {
            this.opName = opName;
            this.symbol = symbol;
        }

        public String opName() {
            return opName;
        }
    }

    private static final FunctionDescriptor UNARY = FunctionDescriptor.of(
            JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT);

    private static final MethodHandle EXP;
    private static final MethodHandle LOG;
    private static final MethodHandle LOG2;
    private static final MethodHandle LOG10;
    private static final MethodHandle SQRT;

    static {
        Linker linker = Linker.nativeLinker();
        SymbolLookup symbols = WgpuBackend.nativeSymbols();
        EXP = bind(linker, symbols, Function.EXP);
        LOG = bind(linker, symbols, Function.LOG);
        LOG2 = bind(linker, symbols, Function.LOG2);
        LOG10 = bind(linker, symbols, Function.LOG10);
        SQRT = bind(linker, symbols, Function.SQRT);
    }

    private ExponentialOps() {}

    private static MethodHandle bind(Linker linker, SymbolLookup symbols, Function fn) {
        MemorySegment address = symbols.find(fn.symbol).orElseThrow(() -> new UnsatisfiedLinkError(
                "HC4J native symbol '" + fn.symbol + "' not found; rebuild the Rust backend."));
        return linker.downcallHandle(address, UNARY);
    }

    private static int dispatch(Function fn, long a, long out, int rank, MemorySegment shape,
            MemorySegment stridesA, MemorySegment stridesC, long length, int contiguous) throws Throwable {
        return switch (fn) {
            case EXP -> (int) EXP.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case LOG -> (int) LOG.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case LOG2 -> (int) LOG2.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case LOG10 -> (int) LOG10.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case SQRT -> (int) SQRT.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
        };
    }

    /** Computes {@code out = fn(input)} elementwise; {@code out} may be {@code input}. */
    public static Tensor apply(Function fn, Tensor input, Tensor out) {
        Objects.requireNonNull(fn, "fn");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(out, "out");
        Layouts.requireF32(fn.opName(), input, out);
        Layouts.requireSameShape(fn.opName(), input, out);

        int status;
        try {
            if (Layouts.isContiguous(input) && Layouts.isContiguous(out)) {
                status = dispatch(fn, input.getVramId(), out.getVramId(), out.dim(),
                        MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL, out.getSize(), 1);
            } else {
                try (Arena arena = Arena.ofConfined()) {
                    status = dispatch(fn, input.getVramId(), out.getVramId(), out.dim(),
                            arena.allocateFrom(JAVA_INT, out.internalShapeUnsafe()),
                            arena.allocateFrom(JAVA_INT, input.internalStridesUnsafe()),
                            arena.allocateFrom(JAVA_INT, out.internalStridesUnsafe()),
                            out.getSize(), 0);
                }
            }
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native dispatch of " + fn.opName() + " failed", t);
        }
        WgpuBackend.checkStatus(status, "ExponentialOps." + fn.opName());
        return out;
    }
}
