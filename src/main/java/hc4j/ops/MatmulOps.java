package hc4j.ops;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import hc4j.Tensor;
import hc4j.engine.WgpuBackend;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;
import java.util.Objects;

/**
 * Matrix multiplication and transposition ({@code ops/matmul.rs}).
 *
 * <p>Operands larger than VRAM are handled natively: the product is streamed in row blocks with B
 * resident, and if B does not fit either, blocked over K as well (one row of B plus one row of the
 * result must still fit). A stored-transposed operand that has to be streamed must be materialised
 * with {@link #transpose} first; otherwise this reports {@link UnsupportedOperationException}.
 *
 * <p>{@code res = op(a) · op(b)} where {@code op} optionally transposes an operand that is
 * <em>stored</em> transposed. The native dispatcher picks a register-tiled GEMM, a 16×16 tiled
 * GEMM, or a GEMV kernel from the shape and device; {@link #plan} reports its choice.
 *
 * <p>Operands are 2-D matrices, or 1-D vectors with NumPy semantics: a 1-D left operand is a row
 * vector ({@code [K] -> 1×K}), a 1-D right operand a column vector ({@code [K] -> K×1}), and
 * that dimension is dropped from the result ({@code [M,K]·[K] -> [M]}, {@code [K]·[K] -> [1]}).
 */
public final class MatmulOps {

    /** Kernel ids reported by {@code hc4j_matmul_plan}; keep in sync with {@code Kernel::code}. */
    public enum Kernel {
        REGISTER_64_VEC4(1),
        REGISTER_64_SCALAR(2),
        REGISTER_128_VEC4(3),
        REGISTER_128_SCALAR(4),
        TILED_16(5),
        GEMV_ROWS(6),
        GEMV_ROWS_SUBGROUP(7),
        GEMV_COLS(8);

        private final int code;

        Kernel(int code) {
            this.code = code;
        }

        static Kernel of(int code) {
            for (Kernel k : values()) {
                if (k.code == code) {
                    return k;
                }
            }
            throw new IllegalStateException("Unknown matmul kernel code " + code);
        }
    }

    /** The dispatcher's choice, including in-VRAM pre-transposition of stored-transposed operands. */
    public record Plan(Kernel kernel, boolean pretransposeA, boolean pretransposeB) {}

    private static final int FLAG_TRANS_A = 1;
    private static final int FLAG_TRANS_B = 2;

    private static final MethodHandle MATMUL;
    private static final MethodHandle MATMUL_EX;
    private static final MethodHandle TRANSPOSE;
    private static final MethodHandle PLAN;

    static {
        Linker linker = Linker.nativeLinker();
        SymbolLookup symbols = WgpuBackend.nativeSymbols();
        // int32_t dispatch_matmul_f32(u64 a, u64 b, u64 out, u32 m, u32 n, u32 k)
        MATMUL = bind(linker, symbols, "dispatch_matmul_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));
        // int32_t dispatch_matmul_f32_ex(u64 a, u64 b, u64 out, u32 m, u32 n, u32 k, u32 flags)
        MATMUL_EX = bind(linker, symbols, "dispatch_matmul_f32_ex",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
        // int32_t dispatch_transpose_f32(u64 in, u64 out, u32 rows, u32 cols)
        TRANSPOSE = bind(linker, symbols, "dispatch_transpose_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT));
        // int32_t hc4j_matmul_plan(u32 m, u32 n, u32 k, u32 flags)
        PLAN = bind(linker, symbols, "hc4j_matmul_plan",
                FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
    }

    private MatmulOps() {}

    private static MethodHandle bind(Linker linker, SymbolLookup symbols, String name, FunctionDescriptor desc) {
        MemorySegment address = symbols.find(name).orElseThrow(() -> new UnsatisfiedLinkError(
                "HC4J native symbol '" + name + "' not found; rebuild the Rust backend."));
        return linker.downcallHandle(address, desc);
    }

    /** Logical GEMM dimensions of {@code op(a) · op(b)}. */
    public record Dims(int m, int n, int k, int[] resultShape) {}

    /**
     * Validates operand shapes and derives {@code (M, N, K)} and the result shape.
     *
     * @throws IllegalArgumentException if ranks are not 1 or 2 or the inner dimensions differ
     */
    public static Dims dims(Tensor a, Tensor b, boolean transposeA, boolean transposeB) {
        int[] sa = a.internalShapeUnsafe();
        int[] sb = b.internalShapeUnsafe();
        if (sa.length < 1 || sa.length > 2 || sb.length < 1 || sb.length > 2) {
            throw new IllegalArgumentException("matmul supports 1-D and 2-D operands, got "
                    + Arrays.toString(sa) + " and " + Arrays.toString(sb));
        }
        int m;
        int ka;
        if (sa.length == 1) {
            m = 1;
            ka = sa[0];
        } else {
            m = transposeA ? sa[1] : sa[0];
            ka = transposeA ? sa[0] : sa[1];
        }
        int kb;
        int n;
        if (sb.length == 1) {
            kb = sb[0];
            n = 1;
        } else {
            kb = transposeB ? sb[1] : sb[0];
            n = transposeB ? sb[0] : sb[1];
        }
        if (ka != kb) {
            throw new IllegalArgumentException("matmul inner dimensions differ: op(A) is " + m + "x" + ka
                    + " but op(B) is " + kb + "x" + n + " (A.cols must equal B.rows)");
        }
        int[] shape;
        if (sa.length == 2 && sb.length == 2) {
            shape = new int[] {m, n};
        } else if (sa.length == 2) {
            shape = new int[] {m};
        } else if (sb.length == 2) {
            shape = new int[] {n};
        } else {
            shape = new int[] {1};
        }
        return new Dims(m, n, ka, shape);
    }

    /**
     * {@code res = op(a) · op(b)}. {@code res} must be f32 with the shape from {@link #dims} and
     * must not be {@code a} or {@code b}.
     *
     * @return {@code res}
     */
    public static Tensor matmul(Tensor a, Tensor b, Tensor res, boolean transposeA, boolean transposeB) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(res, "res");
        Layouts.requireF32("matmul", a, b, res);
        Dims d = dims(a, b, transposeA, transposeB);
        if (!Arrays.equals(res.internalShapeUnsafe(), d.resultShape())) {
            throw new IllegalArgumentException("matmul result must have shape " + Arrays.toString(d.resultShape())
                    + ", got " + Arrays.toString(res.internalShapeUnsafe()));
        }
        if (res == a || res == b || res.getVramId() == a.getVramId() || res.getVramId() == b.getVramId()) {
            throw new IllegalArgumentException("matmul result must not alias an operand");
        }
        // Transposing a 1-D operand is a no-op on its storage.
        int flags = (transposeA && a.dim() == 2 ? FLAG_TRANS_A : 0) | (transposeB && b.dim() == 2 ? FLAG_TRANS_B : 0);
        int status;
        try {
            status = flags == 0
                    ? (int) MATMUL.invokeExact(a.getVramId(), b.getVramId(), res.getVramId(), d.m(), d.n(), d.k())
                    : (int) MATMUL_EX.invokeExact(a.getVramId(), b.getVramId(), res.getVramId(), d.m(), d.n(), d.k(), flags);
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native matmul failed", t);
        }
        WgpuBackend.checkStatus(status, "MatmulOps.matmul");
        return res;
    }

    /** {@code res = in^T} for a 2-D {@code in} of shape {@code [r, c]}; {@code res} is {@code [c, r]}. */
    public static Tensor transpose(Tensor in, Tensor res) {
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(res, "res");
        Layouts.requireF32("transpose", in, res);
        int[] s = in.internalShapeUnsafe();
        if (s.length != 2) {
            throw new IllegalArgumentException("transpose needs a 2-D tensor, got " + Arrays.toString(s));
        }
        if (!Arrays.equals(res.internalShapeUnsafe(), new int[] {s[1], s[0]})) {
            throw new IllegalArgumentException("transpose result must have shape [" + s[1] + ", " + s[0] + "]");
        }
        if (res == in || res.getVramId() == in.getVramId()) {
            throw new IllegalArgumentException("transpose cannot run in place");
        }
        int status;
        try {
            status = (int) TRANSPOSE.invokeExact(in.getVramId(), res.getVramId(), s[0], s[1]);
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native transpose failed", t);
        }
        WgpuBackend.checkStatus(status, "MatmulOps.transpose");
        return res;
    }

    /** The kernel the native dispatcher selects for {@code (m, n, k)} on this device. */
    public static Plan plan(int m, int n, int k, boolean transposeA, boolean transposeB) {
        int flags = (transposeA ? FLAG_TRANS_A : 0) | (transposeB ? FLAG_TRANS_B : 0);
        int code;
        try {
            code = (int) PLAN.invokeExact(m, n, k, flags);
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native matmul plan failed", t);
        }
        if (code < 0) {
            WgpuBackend.checkStatus(code, "MatmulOps.plan");
        }
        return new Plan(Kernel.of(code & 0xFF), (code & 0x100) != 0, (code & 0x200) != 0);
    }
}
