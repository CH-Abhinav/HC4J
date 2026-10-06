package hc4j;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import hc4j.engine.WgpuBackend;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.IdentityHashMap;

/**
 * A deferred elementwise expression over same-shape f32 tensors. Nothing runs until
 * {@link #eval()}, which lowers the whole tree to one postfix program and one native dispatch:
 * every input read once, the output written once, no intermediate tensors.
 *
 * <pre>{@code
 * try (Tensor r = a.lazy().sin().add(b).mul(c).eval()) { ... }
 * }</pre>
 *
 * For {@code sin(a) + b) * c} this moves 4N floats (3 reads, 1 write) instead of the 8N that three
 * separate kernels move, and allocates one tensor instead of three. The native side
 * value-numbers the program, so repeated subexpressions are computed once and the same expression
 * shape reuses one compiled pipeline across calls.
 *
 * <p>Expressions are immutable and may be evaluated repeatedly. Input tensors are referenced, not
 * copied: they must stay open until {@code eval} returns.
 */
public final class FusedExpr {

    // Opcodes: keep in sync with ops/fusion.rs `op::*`.
    private static final int LOAD = 0x01, CONST = 0x02;
    private static final int NEG = 0x10, ABS = 0x11, SIN = 0x12, COS = 0x13, TAN = 0x14;
    private static final int ASIN = 0x15, ACOS = 0x16, ATAN = 0x17, SINH = 0x18, COSH = 0x19, TANH = 0x1A;
    private static final int ASINH = 0x1B, ACOSH = 0x1C, ATANH = 0x1D, EXP = 0x1E, LOG = 0x1F, SQRT = 0x20;
    private static final int ADD = 0x40, SUB = 0x41, MUL = 0x42, DIV = 0x43, MAX = 0x44, MIN = 0x45, POW = 0x46;

    /** Mirrors fusion.rs MAX_PROGRAM_WORDS; a constant costs two words. */
    private static final int MAX_PROGRAM_WORDS = 1024;

    private sealed interface Node permits Load, Const, Unary, Binary {}
    private record Load(Tensor tensor) implements Node {}
    private record Const(float value) implements Node {}
    private record Unary(int op, Node x) implements Node {}
    private record Binary(int op, Node l, Node r) implements Node {}

    /** Bound on first eval, so building expressions never touches native code. */
    private static final class Native {
        // int32_t hc4j_fused_dispatch(const u32* program, u32 len, const u64* inputs, u32 n, u64 id_out)
        static final MethodHandle DISPATCH = Linker.nativeLinker().downcallHandle(
                WgpuBackend.nativeSymbols().find("hc4j_fused_dispatch").orElseThrow(
                        () -> new UnsatisfiedLinkError("hc4j_fused_dispatch not exported; rebuild the Rust backend")),
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG));
    }

    private final Node root;
    private final int[] shape;
    private final int words;

    private FusedExpr(Node root, int[] shape, int words) {
        if (words > MAX_PROGRAM_WORDS) {
            throw new IllegalStateException(
                    "Fused expression exceeds " + MAX_PROGRAM_WORDS + " program words; eval() a subexpression first");
        }
        this.root = root;
        this.shape = shape;
        this.words = words;
    }

    static FusedExpr of(Tensor t) {
        if (t.getDType() != DType.f32) {
            throw new IllegalArgumentException("Fusion supports f32 tensors only, got " + t.getDType());
        }
        return new FusedExpr(new Load(t), t.internalShapeUnsafe().clone(), 1);
    }

    public FusedExpr neg()   { return unary(NEG); }
    public FusedExpr abs()   { return unary(ABS); }
    public FusedExpr sin()   { return unary(SIN); }
    public FusedExpr cos()   { return unary(COS); }
    public FusedExpr tan()   { return unary(TAN); }
    public FusedExpr asin()  { return unary(ASIN); }
    public FusedExpr acos()  { return unary(ACOS); }
    public FusedExpr atan()  { return unary(ATAN); }
    public FusedExpr sinh()  { return unary(SINH); }
    public FusedExpr cosh()  { return unary(COSH); }
    public FusedExpr tanh()  { return unary(TANH); }
    public FusedExpr asinh() { return unary(ASINH); }
    public FusedExpr acosh() { return unary(ACOSH); }
    public FusedExpr atanh() { return unary(ATANH); }
    public FusedExpr exp()   { return unary(EXP); }
    public FusedExpr log()   { return unary(LOG); }
    public FusedExpr sqrt()  { return unary(SQRT); }

    public FusedExpr add(FusedExpr o) { return binary(ADD, o); }
    public FusedExpr sub(FusedExpr o) { return binary(SUB, o); }
    public FusedExpr mul(FusedExpr o) { return binary(MUL, o); }
    public FusedExpr div(FusedExpr o) { return binary(DIV, o); }
    public FusedExpr max(FusedExpr o) { return binary(MAX, o); }
    public FusedExpr min(FusedExpr o) { return binary(MIN, o); }
    public FusedExpr pow(FusedExpr o) { return binary(POW, o); }

    public FusedExpr add(Tensor t) { return binary(ADD, of(t)); }
    public FusedExpr sub(Tensor t) { return binary(SUB, of(t)); }
    public FusedExpr mul(Tensor t) { return binary(MUL, of(t)); }
    public FusedExpr div(Tensor t) { return binary(DIV, of(t)); }
    public FusedExpr max(Tensor t) { return binary(MAX, of(t)); }
    public FusedExpr min(Tensor t) { return binary(MIN, of(t)); }

    public FusedExpr add(float c) { return scalar(ADD, c); }

    /** {@code c - this}, the reverse of {@link #sub(float)}. */
    public FusedExpr rsub(float c) { return reverseScalar(SUB, c); }

    /** {@code c / this}. */
    public FusedExpr rdiv(float c) { return reverseScalar(DIV, c); }

    /** {@code c ^ this}. */
    public FusedExpr rpow(float c) { return reverseScalar(POW, c); }
    public FusedExpr sub(float c) { return scalar(SUB, c); }
    public FusedExpr mul(float c) { return scalar(MUL, c); }
    public FusedExpr div(float c) { return scalar(DIV, c); }
    public FusedExpr pow(float c) { return scalar(POW, c); }

    private FusedExpr unary(int op) {
        return new FusedExpr(new Unary(op, root), shape, words + 1);
    }

    private FusedExpr binary(int op, FusedExpr other) {
        if (!Arrays.equals(shape, other.shape)) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + Arrays.toString(shape) + " vs " + Arrays.toString(other.shape));
        }
        return new FusedExpr(new Binary(op, root, other.root), shape, words + other.words + 1);
    }

    /// Scalar on the left: the tree holds (const op this).
    private FusedExpr reverseScalar(int op, float c) {
        requireFinite(c);
        return new FusedExpr(new Binary(op, new Const(c), root), shape, words + 3);
    }

    private FusedExpr scalar(int op, float c) {
        requireFinite(c);
        return new FusedExpr(new Binary(op, root, new Const(c)), shape, words + 3);
    }

    /** Evaluates into a newly allocated tensor. */
    public Tensor eval() {
        Tensor res = Tensor.empty(DType.f32, shape);
        try {
            return eval(res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Evaluates into {@code res}, which must be f32 with this expression's shape. {@code res} may
     * be one of the inputs: each element is read before it is written.
     *
     * @return {@code res}
     */
    public Tensor eval(Tensor res) {
        if (res.getDType() != DType.f32 || !Arrays.equals(res.internalShapeUnsafe(), shape)) {
            throw new IllegalArgumentException("eval target must be f32 with shape " + Arrays.toString(shape));
        }
        // Inputs are deduplicated by identity, so x.mul(x) binds one buffer.
        IdentityHashMap<Tensor, Integer> inputs = new IdentityHashMap<>();
        int[] program = new int[words];
        int pc = 0;

        // Iterative post-order walk: a long chain cannot overflow the Java stack.
        Deque<Object> work = new ArrayDeque<>();
        work.push(root);
        while (!work.isEmpty()) {
            Object item = work.pop();
            if (item instanceof Integer opWord) {
                program[pc++] = opWord;
                continue;
            }
            switch ((Node) item) {
                case Load(Tensor t) -> {
                    int k = inputs.computeIfAbsent(t, unused -> inputs.size());
                    program[pc++] = word(LOAD, k);
                }
                case Const(float v) -> {
                    program[pc++] = word(CONST, 0);
                    program[pc++] = Float.floatToRawIntBits(v);
                }
                case Unary(int op, Node x) -> {
                    work.push(word(op, 0));
                    work.push(x);
                }
                case Binary(int op, Node l, Node r) -> {
                    work.push(word(op, 0));
                    work.push(r);
                    work.push(l);
                }
            }
        }

        long[] handles = new long[inputs.size()];
        inputs.forEach((t, k) -> handles[k] = t.getVramId());

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment prog = arena.allocateFrom(JAVA_INT, Arrays.copyOf(program, pc));
            MemorySegment ids = arena.allocateFrom(JAVA_LONG, handles);
            int status;
            try {
                status = (int) Native.DISPATCH.invokeExact(prog, pc, ids, handles.length, res.getVramId());
            } catch (Throwable t) {
                throw new IllegalStateException("HC4J fused dispatch failed", t);
            }
            WgpuBackend.checkStatus(status, "FusedExpr.eval");
            return res;
        }
    }

    private static void requireFinite(float c) {
        if (!Float.isFinite(c)) {
            throw new IllegalArgumentException("Fused constants must be finite: " + c);
        }
    }

    private static int word(int opcode, int operand) {
        return (opcode << 24) | (operand & 0x00FF_FFFF);
    }
}
