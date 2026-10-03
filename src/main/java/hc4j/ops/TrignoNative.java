package hc4j.ops;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import hc4j.engine.WgpuBackend;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

/**
 * Panama downcall bindings for the trigonometric suite exported by {@code ops/trigno.rs}, using
 * the unified caller-allocated elementwise ABI:
 * <pre>{@code
 * int32_t dispatch_<op>_f32(uint64_t id_a, uint64_t id_out, uint32_t rank,
 *         const uint32_t* shape, const uint32_t* strides_a, const uint32_t* strides_c,
 *         size_t length, uint32_t is_contiguous);
 * }</pre>
 * The handles are {@code static final} so HotSpot treats them as constants and inlines each
 * {@code invokeExact} into a direct native call; {@link #dispatch} selects one with an enum switch
 * instead of passing a (non-constant) handle around.
 *
 * <p>Internal: use {@link TrignoOps} or the fluent {@code Tensor} methods.
 */
final class TrignoNative {

    /** {@code (u64 a, u64 out, u32 rank, u32* shape, u32* strides_a, u32* strides_c, usize len, u32 contiguous) -> i32}. */
    static final FunctionDescriptor UNARY = FunctionDescriptor.of(
            JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT);

    static final MethodHandle SIN;
    static final MethodHandle COS;
    static final MethodHandle TAN;
    static final MethodHandle ASIN;
    static final MethodHandle ACOS;
    static final MethodHandle ATAN;
    static final MethodHandle SINH;
    static final MethodHandle COSH;
    static final MethodHandle TANH;
    static final MethodHandle ASINH;
    static final MethodHandle ACOSH;
    static final MethodHandle ATANH;

    static {
        Linker linker = Linker.nativeLinker();
        SymbolLookup symbols = WgpuBackend.nativeSymbols();
        SIN = bind(linker, symbols, "dispatch_sin_f32");
        COS = bind(linker, symbols, "dispatch_cos_f32");
        TAN = bind(linker, symbols, "dispatch_tan_f32");
        ASIN = bind(linker, symbols, "dispatch_asin_f32");
        ACOS = bind(linker, symbols, "dispatch_acos_f32");
        ATAN = bind(linker, symbols, "dispatch_atan_f32");
        SINH = bind(linker, symbols, "dispatch_sinh_f32");
        COSH = bind(linker, symbols, "dispatch_cosh_f32");
        TANH = bind(linker, symbols, "dispatch_tanh_f32");
        ASINH = bind(linker, symbols, "dispatch_asinh_f32");
        ACOSH = bind(linker, symbols, "dispatch_acosh_f32");
        ATANH = bind(linker, symbols, "dispatch_atanh_f32");
    }

    private TrignoNative() {}

    private static MethodHandle bind(Linker linker, SymbolLookup symbols, String name) {
        MemorySegment address = symbols.find(name).orElseThrow(() -> new UnsatisfiedLinkError(
                "HC4J native symbol '" + name + "' not found; the loaded hc4j library is stale. "
                        + "Rebuild it with 'cargo build --release' in src/main/rust."));
        return linker.downcallHandle(address, UNARY);
    }

    /** Invokes the native kernel for {@code fn}; returns the native status code. */
    static int dispatch(TrignoOps.Function fn, long a, long out, int rank, MemorySegment shape,
            MemorySegment stridesA, MemorySegment stridesC, long length, int contiguous) throws Throwable {
        return switch (fn) {
            case SIN -> (int) SIN.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case COS -> (int) COS.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case TAN -> (int) TAN.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case ASIN -> (int) ASIN.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case ACOS -> (int) ACOS.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case ATAN -> (int) ATAN.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case SINH -> (int) SINH.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case COSH -> (int) COSH.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case TANH -> (int) TANH.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case ASINH -> (int) ASINH.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case ACOSH -> (int) ACOSH.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
            case ATANH -> (int) ATANH.invokeExact(a, out, rank, shape, stridesA, stridesC, length, contiguous);
        };
    }
}
