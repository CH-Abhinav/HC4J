package hc4j.ops;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.engine.WgpuBackend;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;
import java.util.Objects;

/**
 * Neural-network ops: activations, reductions, softmax, normalisation, losses and bias broadcast.
 * Caller-allocated like the rest of the engine, so outputs can be reused and the elementwise ones
 * can run in place.
 *
 * <h2>The last axis</h2>
 * The row-wise ops (softmax, layer norm, RMS norm, cross entropy, the {@code *Rows} reductions)
 * work along the <em>last</em> axis, as NumPy's {@code axis=-1} does. A tensor of any rank is
 * treated as {@code [rows, cols]} with {@code cols} its last dimension and {@code rows} the
 * product of the rest, so {@code [32, 128]} is 32 rows of 128, {@code [4, 8, 128]} is 32 rows of
 * 128, and a 1-D {@code [128]} is one row. Shape-preserving ops return the input shape; reducing
 * ops drop the last axis ({@code [4, 8, 128] -> [4, 8]}, and {@code [128] -> [1]}).
 *
 * <h2>Residency</h2>
 * Only the activations inherit the full elementwise tiering, including the host-streaming path for
 * tensors larger than VRAM. The reductions tile over the storage-binding limit but still need
 * their operands resident, and the row-wise ops need a contiguous operand within one binding
 * (2 GiB on most adapters). When those cannot be made resident the call raises
 * {@link hc4j.engine.GpuOutOfMemoryException} rather than degrading.
 */
public final class NnOps {

    /** Elementwise activations exported by {@code ops/nn.rs}; ordinals match the native codes. */
    public enum Activation {
        RELU,
        /** {@code max(x, alpha * x)}, with {@code alpha} the negative slope. */
        LEAKY_RELU,
        /** {@code x > 0 ? x : alpha * (exp(x) - 1)}. */
        ELU,
        /** The tanh approximation, as GPT-2 and BERT use; not the erf form. */
        GELU,
        SILU,
        SIGMOID,
        SOFTPLUS;

        /** Whether {@code alpha} is read; the others ignore it. */
        public boolean takesAlpha() {
            return this == LEAKY_RELU || this == ELU;
        }
    }

    /** Folds available to the reductions; ordinals match the native codes. */
    public enum Reduction {
        SUM,
        MEAN,
        MAX,
        MIN
    }

    private static final MethodHandle ACTIVATION;
    private static final MethodHandle REDUCE;
    private static final MethodHandle REDUCE_SQUARE;
    private static final MethodHandle REDUCE_DIFF2;
    private static final MethodHandle REDUCE_ROWS;
    private static final MethodHandle ARGMAX_ROWS;
    private static final MethodHandle SOFTMAX_ROWS;
    private static final MethodHandle LAYERNORM_ROWS;
    private static final MethodHandle RMSNORM_ROWS;
    private static final MethodHandle CROSS_ENTROPY_ROWS;
    private static final MethodHandle ADD_ROW_BIAS;

    static {
        Linker linker = Linker.nativeLinker();
        SymbolLookup symbols = WgpuBackend.nativeSymbols();
        // The elementwise ABI plus the activation code and its alpha bits.
        ACTIVATION = bind(linker, symbols, "dispatch_activation_f32", FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT,
                JAVA_INT, JAVA_INT));
        // (u64 a, u64 out, u64 length, u32 op) -> i32
        REDUCE = bind(linker, symbols, "dispatch_reduce_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT));
        REDUCE_SQUARE = bind(linker, symbols, "dispatch_reduce_square_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT));
        // (u64 a, u64 b, u64 out, u64 length, u32 op) -> i32
        REDUCE_DIFF2 = bind(linker, symbols, "dispatch_reduce_diff2_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT));
        // (u64 a, u64 out, u32 rows, u32 cols, u32 op) -> i32
        REDUCE_ROWS = bind(linker, symbols, "dispatch_reduce_rows_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));
        ARGMAX_ROWS = bind(linker, symbols, "dispatch_argmax_rows",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));
        SOFTMAX_ROWS = bind(linker, symbols, "dispatch_softmax_rows_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));
        // (u64 a, u64 gamma, u64 beta, u64 out, u32 rows, u32 cols, u32 eps_bits) -> i32
        LAYERNORM_ROWS = bind(linker, symbols, "dispatch_layernorm_rows_f32", FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));
        RMSNORM_ROWS = bind(linker, symbols, "dispatch_rmsnorm_rows_f32", FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));
        CROSS_ENTROPY_ROWS = bind(linker, symbols, "dispatch_cross_entropy_rows",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT));
        ADD_ROW_BIAS = bind(linker, symbols, "dispatch_add_row_bias_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT));
    }

    private NnOps() {}

    private static MethodHandle bind(Linker linker, SymbolLookup symbols, String name, FunctionDescriptor desc) {
        MemorySegment address = symbols.find(name).orElseThrow(() -> new UnsatisfiedLinkError(
                "HC4J native symbol '" + name + "' not found; rebuild the Rust backend."));
        return linker.downcallHandle(address, desc);
    }

    /**
     * A tensor of any rank seen as rows of its last axis.
     *
     * @param rows the product of every dimension but the last
     * @param cols the last dimension, the axis the row-wise ops work along
     */
    public record Rows(int rows, int cols) {

        /** The shape a reduction over the last axis produces. */
        public int[] reducedShape(Tensor t) {
            int[] shape = t.internalShapeUnsafe();
            return shape.length <= 1 ? new int[] {1} : Arrays.copyOf(shape, shape.length - 1);
        }
    }

    /**
     * Splits {@code t} into rows of its last axis.
     *
     * @throws IllegalArgumentException if {@code t} is empty or either extent exceeds an int
     */
    public static Rows rows(Tensor t) {
        Objects.requireNonNull(t, "tensor");
        int[] shape = t.internalShapeUnsafe();
        if (shape.length == 0 || t.getSize() == 0) {
            throw new IllegalArgumentException("row ops need a non-empty tensor");
        }
        long cols = shape[shape.length - 1];
        long rows = t.getSize() / cols;
        if (cols <= 0 || rows <= 0 || rows > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "cannot split " + Arrays.toString(shape) + " into rows of its last axis");
        }
        return new Rows((int) rows, (int) cols);
    }

    // ----------------------------------------------------------------------------------------
    // Activations
    // ----------------------------------------------------------------------------------------

    /**
     * {@code out = act(in)} elementwise. {@code out} may be {@code in}.
     *
     * @param alpha the negative slope (leaky ReLU) or alpha (ELU); ignored by the others
     * @return {@code out}
     * @throws IllegalArgumentException if either tensor is not f32, or the shapes differ
     */
    public static Tensor activation(Activation act, Tensor in, Tensor out, float alpha) {
        Objects.requireNonNull(act, "act");
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        String name = act.name().toLowerCase(java.util.Locale.ROOT);
        Layouts.requireF32(name, in, out);
        Layouts.requireSameShape(name, in, out);
        if (act.takesAlpha() && !Float.isFinite(alpha)) {
            throw new IllegalArgumentException(name + ": alpha must be finite, got " + alpha);
        }
        int bits = Float.floatToRawIntBits(alpha);

        int status;
        try {
            if (Layouts.isContiguous(in) && Layouts.isContiguous(out)) {
                status = (int) ACTIVATION.invokeExact(in.getVramId(), out.getVramId(), out.dim(),
                        MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL, out.getSize(), 1,
                        act.ordinal(), bits);
            } else {
                try (Arena arena = Arena.ofConfined()) {
                    status = (int) ACTIVATION.invokeExact(in.getVramId(), out.getVramId(), out.dim(),
                            arena.allocateFrom(JAVA_INT, out.internalShapeUnsafe()),
                            arena.allocateFrom(JAVA_INT, in.internalStridesUnsafe()),
                            arena.allocateFrom(JAVA_INT, out.internalStridesUnsafe()),
                            out.getSize(), 0, act.ordinal(), bits);
                }
            }
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native dispatch of " + name + " failed", t);
        }
        WgpuBackend.checkStatus(status, "NnOps." + name);
        return out;
    }

    // ----------------------------------------------------------------------------------------
    // Reductions
    // ----------------------------------------------------------------------------------------

    /**
     * {@code out[0] = reduce(in)} over every element. {@code out} must be f32 with one element.
     *
     * @return {@code out}
     */
    public static Tensor reduce(Reduction op, Tensor in, Tensor out) {
        return reduceInto(op, in, out, REDUCE, "reduce");
    }

    /**
     * {@code out[0] = reduce(in * in)}: the fold an L2 norm or an uncentred variance needs,
     * without materialising the squares.
     */
    public static Tensor reduceSquare(Reduction op, Tensor in, Tensor out) {
        return reduceInto(op, in, out, REDUCE_SQUARE, "reduceSquare");
    }

    private static Tensor reduceInto(Reduction op, Tensor in, Tensor out, MethodHandle handle, String name) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        Layouts.requireF32(name, in, out);
        requireContiguous(name, in);
        if (out.getSize() != 1) {
            throw new IllegalArgumentException(name + ": result must hold exactly one element, got " + out.getSize());
        }
        int status;
        try {
            status = (int) handle.invokeExact(in.getVramId(), out.getVramId(), in.getSize(), op.ordinal());
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native dispatch of " + name + " failed", t);
        }
        WgpuBackend.checkStatus(status, "NnOps." + name);
        return out;
    }

    /**
     * {@code out[0] = reduce((pred - target)^2)}. With {@link Reduction#MEAN} this is the mean
     * squared error, computed in one pass with no intermediate tensor.
     */
    public static Tensor reduceDiffSquared(Reduction op, Tensor pred, Tensor target, Tensor out) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(pred, "pred");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(out, "out");
        Layouts.requireF32("mse", pred, target, out);
        Layouts.requireSameShape("mse", pred, target);
        requireContiguous("mse", pred);
        requireContiguous("mse", target);
        if (out.getSize() != 1) {
            throw new IllegalArgumentException("mse: result must hold exactly one element, got " + out.getSize());
        }
        int status;
        try {
            status = (int) REDUCE_DIFF2.invokeExact(
                    pred.getVramId(), target.getVramId(), out.getVramId(), pred.getSize(), op.ordinal());
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native dispatch of mse failed", t);
        }
        WgpuBackend.checkStatus(status, "NnOps.reduceDiffSquared");
        return out;
    }

    /**
     * {@code out[r] = reduce(in[r, :])} along the last axis. {@code out} must be f32 with the
     * {@linkplain Rows#reducedShape reduced shape}.
     */
    public static Tensor reduceRows(Reduction op, Tensor in, Tensor out) {
        Objects.requireNonNull(op, "op");
        Rows dims = rows(in);
        Layouts.requireF32("reduceRows", in, out);
        requireContiguous("reduceRows", in);
        requireReducedShape("reduceRows", in, out, dims);
        int status;
        try {
            status = (int) REDUCE_ROWS.invokeExact(
                    in.getVramId(), out.getVramId(), dims.rows(), dims.cols(), op.ordinal());
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native dispatch of reduceRows failed", t);
        }
        WgpuBackend.checkStatus(status, "NnOps.reduceRows");
        return out;
    }

    /**
     * {@code out[r] = argmax(in[r, :])} along the last axis, as i32 class indices. Ties resolve to
     * the lowest index, as NumPy and PyTorch do.
     *
     * @param largest false for argmin
     */
    public static Tensor argmaxRows(Tensor in, Tensor out, boolean largest) {
        Rows dims = rows(in);
        Layouts.requireF32("argmaxRows", in);
        requireContiguous("argmaxRows", in);
        if (out.getDType() != DType.i32) {
            throw new IllegalArgumentException("argmaxRows: indices must be an i32 tensor, got " + out.getDType());
        }
        requireReducedShape("argmaxRows", in, out, dims);
        int status;
        try {
            status = (int) ARGMAX_ROWS.invokeExact(
                    in.getVramId(), out.getVramId(), dims.rows(), dims.cols(), largest ? 1 : 0);
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native dispatch of argmaxRows failed", t);
        }
        WgpuBackend.checkStatus(status, "NnOps.argmaxRows");
        return out;
    }

    // ----------------------------------------------------------------------------------------
    // Softmax and normalisation
    // ----------------------------------------------------------------------------------------

    /**
     * Row-wise softmax (or log-softmax) along the last axis, numerically stable: the row maximum
     * is subtracted, so no {@code exp} ever sees a positive argument. {@code out} may be
     * {@code in}.
     */
    public static Tensor softmaxRows(Tensor in, Tensor out, boolean logarithmic) {
        Rows dims = rows(in);
        Layouts.requireF32("softmax", in, out);
        Layouts.requireSameShape("softmax", in, out);
        requireContiguous("softmax", in);
        int status;
        try {
            status = (int) SOFTMAX_ROWS.invokeExact(
                    in.getVramId(), out.getVramId(), dims.rows(), dims.cols(), logarithmic ? 1 : 0);
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native dispatch of softmax failed", t);
        }
        WgpuBackend.checkStatus(status, "NnOps.softmaxRows");
        return out;
    }

    /**
     * Row-wise layer norm along the last axis: each row is centred and scaled to unit variance
     * (computed with Welford accumulation, biased as PyTorch does), then optionally scaled by
     * {@code gamma} and shifted by {@code beta}.
     *
     * @param gamma per-column scale, or null for no affine transform
     * @param beta per-column shift, or null for no affine transform
     * @throws IllegalArgumentException if exactly one of gamma and beta is null
     */
    public static Tensor layerNormRows(Tensor in, Tensor gamma, Tensor beta, Tensor out, float eps) {
        Rows dims = rows(in);
        Layouts.requireF32("layerNorm", in, out);
        Layouts.requireSameShape("layerNorm", in, out);
        requireContiguous("layerNorm", in);
        if ((gamma == null) != (beta == null)) {
            throw new IllegalArgumentException("layerNorm needs both gamma and beta, or neither");
        }
        requireEps("layerNorm", eps);
        if (gamma != null) {
            Layouts.requireF32("layerNorm", gamma, beta);
            requireLength("layerNorm", "gamma", gamma, dims.cols());
            requireLength("layerNorm", "beta", beta, dims.cols());
        }
        int status;
        try {
            status = (int) LAYERNORM_ROWS.invokeExact(in.getVramId(), handle(gamma), handle(beta), out.getVramId(),
                    dims.rows(), dims.cols(), Float.floatToRawIntBits(eps));
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native dispatch of layerNorm failed", t);
        }
        WgpuBackend.checkStatus(status, "NnOps.layerNormRows");
        return out;
    }

    /**
     * Row-wise RMS norm along the last axis: {@code out = in / sqrt(mean(in^2) + eps)}, optionally
     * scaled by {@code gamma}. No centring, which is what distinguishes it from layer norm.
     *
     * @param gamma per-column scale, or null for none
     */
    public static Tensor rmsNormRows(Tensor in, Tensor gamma, Tensor out, float eps) {
        Rows dims = rows(in);
        Layouts.requireF32("rmsNorm", in, out);
        Layouts.requireSameShape("rmsNorm", in, out);
        requireContiguous("rmsNorm", in);
        requireEps("rmsNorm", eps);
        if (gamma != null) {
            Layouts.requireF32("rmsNorm", gamma);
            requireLength("rmsNorm", "gamma", gamma, dims.cols());
        }
        int status;
        try {
            status = (int) RMSNORM_ROWS.invokeExact(in.getVramId(), handle(gamma), out.getVramId(),
                    dims.rows(), dims.cols(), Float.floatToRawIntBits(eps));
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native dispatch of rmsNorm failed", t);
        }
        WgpuBackend.checkStatus(status, "NnOps.rmsNormRows");
        return out;
    }

    // ----------------------------------------------------------------------------------------
    // Losses and bias
    // ----------------------------------------------------------------------------------------

    /**
     * Per-row softmax cross entropy from logits: {@code out[r] = logsumexp(logits[r, :]) -
     * logits[r, targets[r]]}. The softmax is never materialised.
     *
     * <p>{@code targets} is an i32 tensor of class indices with the {@linkplain Rows#reducedShape
     * reduced shape}. An index outside {@code [0, cols)} yields NaN for that row rather than
     * reading a neighbouring one — the kernel cannot report an error, so the value is the signal.
     * Reduce the result with {@link #reduceRows} or {@link #reduce} for a batch loss.
     */
    public static Tensor crossEntropyRows(Tensor logits, Tensor targets, Tensor out) {
        Rows dims = rows(logits);
        Layouts.requireF32("crossEntropy", logits, out);
        requireContiguous("crossEntropy", logits);
        if (targets.getDType() != DType.i32) {
            throw new IllegalArgumentException("crossEntropy: targets must be i32, got " + targets.getDType());
        }
        requireReducedShape("crossEntropy", logits, out, dims);
        requireLength("crossEntropy", "targets", targets, dims.rows());
        int status;
        try {
            status = (int) CROSS_ENTROPY_ROWS.invokeExact(
                    logits.getVramId(), targets.getVramId(), out.getVramId(), dims.rows(), dims.cols());
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native dispatch of crossEntropy failed", t);
        }
        WgpuBackend.checkStatus(status, "NnOps.crossEntropyRows");
        return out;
    }

    /**
     * {@code out[r, c] = in[r, c] + bias[c]}, broadcasting the bias over rows of the last axis.
     * {@code out} may be {@code in}. Fusion cannot express this, since it requires identical
     * shapes.
     */
    public static Tensor addRowBias(Tensor in, Tensor bias, Tensor out) {
        Rows dims = rows(in);
        Layouts.requireF32("addRowBias", in, bias, out);
        Layouts.requireSameShape("addRowBias", in, out);
        requireContiguous("addRowBias", in);
        requireLength("addRowBias", "bias", bias, dims.cols());
        int status;
        try {
            status = (int) ADD_ROW_BIAS.invokeExact(
                    in.getVramId(), bias.getVramId(), out.getVramId(), dims.rows(), dims.cols());
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native dispatch of addRowBias failed", t);
        }
        WgpuBackend.checkStatus(status, "NnOps.addRowBias");
        return out;
    }

    // ----------------------------------------------------------------------------------------
    // Validation helpers
    // ----------------------------------------------------------------------------------------

    private static long handle(Tensor t) {
        // 0 is never a valid handle, so the native side reads it as "absent".
        return t == null ? 0L : t.getVramId();
    }

    private static void requireEps(String op, float eps) {
        if (!Float.isFinite(eps) || eps < 0) {
            throw new IllegalArgumentException(op + ": eps must be finite and non-negative, got " + eps);
        }
    }

    private static void requireLength(String op, String what, Tensor t, long expected) {
        Objects.requireNonNull(t, what);
        if (t.getSize() != expected) {
            throw new IllegalArgumentException(
                    op + ": " + what + " must hold " + expected + " elements, got " + t.getSize());
        }
    }

    private static void requireReducedShape(String op, Tensor in, Tensor out, Rows dims) {
        Objects.requireNonNull(out, "out");
        int[] want = dims.reducedShape(in);
        if (out.getSize() != dims.rows()) {
            throw new IllegalArgumentException(op + ": result must have shape " + Arrays.toString(want)
                    + " (" + dims.rows() + " rows), got " + Arrays.toString(out.internalShapeUnsafe()));
        }
    }

    /**
     * The row-wise and reducing kernels index their operands linearly, so a strided view would be
     * read in the wrong order. Materialise it first (for a 2-D view, with {@code transpose}).
     */
    private static void requireContiguous(String op, Tensor t) {
        if (!Layouts.isContiguous(t)) {
            throw new IllegalArgumentException(
                    op + " needs a contiguous tensor; materialise the view first (e.g. with transpose)");
        }
    }
}
