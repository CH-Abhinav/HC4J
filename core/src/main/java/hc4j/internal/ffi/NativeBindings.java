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

package hc4j.internal.ffi;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * Lazy-initialized registry of Panama FFM downcall MethodHandles for all GPU kernels.
 */
public final class NativeBindings {

    public static final StructLayout ADAPTER_INFO_LAYOUT = MemoryLayout.structLayout(
            JAVA_INT.withName("vendorId"),
            JAVA_INT.withName("deviceId"),
            JAVA_INT.withName("deviceType"),
            JAVA_INT.withName("backend"),
            JAVA_INT.withName("nameLen"),
            JAVA_INT.withName("driverLen"),
            JAVA_INT.withName("driverInfoLen"),
            MemoryLayout.sequenceLayout(256, ValueLayout.JAVA_BYTE).withName("name"),
            MemoryLayout.sequenceLayout(128, ValueLayout.JAVA_BYTE).withName("driver"),
            MemoryLayout.sequenceLayout(256, ValueLayout.JAVA_BYTE).withName("driverInfo")
    );

    public static final long ADAPTER_INFO_VENDOR_ID_OFFSET = ADAPTER_INFO_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("vendorId"));
    public static final long ADAPTER_INFO_DEVICE_ID_OFFSET = ADAPTER_INFO_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("deviceId"));
    public static final long ADAPTER_INFO_DEVICE_TYPE_OFFSET = ADAPTER_INFO_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("deviceType"));
    public static final long ADAPTER_INFO_BACKEND_OFFSET = ADAPTER_INFO_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("backend"));
    public static final long ADAPTER_INFO_NAME_LEN_OFFSET = ADAPTER_INFO_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("nameLen"));
    public static final long ADAPTER_INFO_DRIVER_LEN_OFFSET = ADAPTER_INFO_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("driverLen"));
    public static final long ADAPTER_INFO_DRIVER_INFO_LEN_OFFSET = ADAPTER_INFO_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("driverInfoLen"));
    public static final long ADAPTER_INFO_NAME_OFFSET = ADAPTER_INFO_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("name"));
    public static final long ADAPTER_INFO_DRIVER_OFFSET = ADAPTER_INFO_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("driver"));
    public static final long ADAPTER_INFO_DRIVER_INFO_OFFSET = ADAPTER_INFO_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("driverInfo"));

    private static final class Holder {
        static final Linker LINKER = Linker.nativeLinker();
        static final SymbolLookup SYMBOLS = NativeEngine.nativeSymbols();

        // Binary elementwise (add, sub, mul, div)
        static final FunctionDescriptor BINARY_DESC = FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT,
                ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT);

        static final MethodHandle DISPATCH_ADD = bind("dispatch_add", BINARY_DESC);
        static final MethodHandle DISPATCH_SUB = bind("dispatch_sub", BINARY_DESC);
        static final MethodHandle DISPATCH_MUL = bind("dispatch_mul", BINARY_DESC);
        static final MethodHandle DISPATCH_DIV = bind("dispatch_div", BINARY_DESC);

        // Strided copy / materialization
        static final FunctionDescriptor COPY_STRIDED_DESC = FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG);
        static final MethodHandle DISPATCH_COPY_STRIDED = bind("dispatch_copy_strided", COPY_STRIDED_DESC);

        // Unary elementwise (trigonometric & exponential)
        static final FunctionDescriptor UNARY_DESC = FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT);

        static final MethodHandle DISPATCH_SIN = bind("dispatch_sin_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_COS = bind("dispatch_cos_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_TAN = bind("dispatch_tan_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_ASIN = bind("dispatch_asin_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_ACOS = bind("dispatch_acos_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_ATAN = bind("dispatch_atan_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_SINH = bind("dispatch_sinh_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_COSH = bind("dispatch_cosh_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_TANH = bind("dispatch_tanh_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_ASINH = bind("dispatch_asinh_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_ACOSH = bind("dispatch_acosh_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_ATANH = bind("dispatch_atanh_f32", UNARY_DESC);

        static final MethodHandle DISPATCH_EXP = bind("dispatch_exp_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_LOG = bind("dispatch_ln_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_LOG2 = bind("dispatch_log2_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_LOG10 = bind("dispatch_log10_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_SQRT = bind("dispatch_sqrt_f32", UNARY_DESC);

        // Matmul & Transpose
        static final MethodHandle MATMUL = bind("dispatch_matmul_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));

        static final MethodHandle MATMUL_EX = bind("dispatch_matmul_f32_ex",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));

        static final MethodHandle TRANSPOSE = bind("dispatch_transpose_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT));

        static final MethodHandle MATMUL_PLAN = bind("hc4j_matmul_plan",
                FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));

        static final FunctionDescriptor BATCHED_MATMUL_DESC = FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT);
        static final MethodHandle DISPATCH_BATCHED_MATMUL = bind("dispatch_batched_matmul_f32", BATCHED_MATMUL_DESC);

        // Extra math ops
        static final MethodHandle DISPATCH_ABS_F32 = bind("dispatch_abs_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_NEG_F32 = bind("dispatch_neg_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_RSQRT_F32 = bind("dispatch_rsqrt_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_FLOOR_F32 = bind("dispatch_floor_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_CEIL_F32 = bind("dispatch_ceil_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_ROUND_F32 = bind("dispatch_round_f32", UNARY_DESC);
        static final MethodHandle DISPATCH_SIGN_F32 = bind("dispatch_sign_f32", UNARY_DESC);

        static final FunctionDescriptor CLAMP_F32_DESC = FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT);
        static final MethodHandle DISPATCH_CLAMP_F32 = bind("dispatch_clamp_f32", CLAMP_F32_DESC);

        static final FunctionDescriptor UNARY_I32_DESC = FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT);
        static final MethodHandle DISPATCH_UNARY_I32 = bind("dispatch_unary_i32", UNARY_I32_DESC);

        static final FunctionDescriptor POW_F32_DESC = FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT);
        static final MethodHandle DISPATCH_POW_F32 = bind("dispatch_pow_f32", POW_F32_DESC);

        static final FunctionDescriptor MAX_MIN_DESC = FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT);
        static final MethodHandle DISPATCH_MAXIMUM = bind("dispatch_maximum", MAX_MIN_DESC);
        static final MethodHandle DISPATCH_MINIMUM = bind("dispatch_minimum", MAX_MIN_DESC);

        static final FunctionDescriptor COMPARE_DESC = FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT);
        static final MethodHandle DISPATCH_COMPARE = bind("dispatch_compare", COMPARE_DESC);

        static final FunctionDescriptor WHERE_DESC = FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT);
        static final MethodHandle DISPATCH_WHERE = bind("dispatch_where", WHERE_DESC);

        static final FunctionDescriptor CAST_DESC = FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT);
        static final MethodHandle DISPATCH_CAST = bind("dispatch_cast", CAST_DESC);

        static final FunctionDescriptor REDUCE_AXIS_DESC = FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT);
        static final MethodHandle DISPATCH_REDUCE_AXIS = bind("dispatch_reduce_axis_f32", REDUCE_AXIS_DESC);

        static final FunctionDescriptor ARGMAX_AXIS_DESC = FunctionDescriptor.of(
                JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT);
        static final MethodHandle DISPATCH_ARGMAX_AXIS = bind("dispatch_argmax_axis", ARGMAX_AXIS_DESC);

        // Neural network ops
        static final MethodHandle ACTIVATION = bind("dispatch_activation_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, ADDRESS,
                        JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));

        static final MethodHandle REDUCE = bind("dispatch_reduce_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT));

        static final MethodHandle REDUCE_SQUARE = bind("dispatch_reduce_square_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT));

        static final MethodHandle REDUCE_DIFF2 = bind("dispatch_reduce_diff2_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT));

        static final MethodHandle REDUCE_ROWS = bind("dispatch_reduce_rows_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));

        static final MethodHandle ARGMAX_ROWS = bind("dispatch_argmax_rows",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));

        static final MethodHandle SOFTMAX_ROWS = bind("dispatch_softmax_rows_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));

        static final MethodHandle LAYERNORM_ROWS = bind("dispatch_layernorm_rows_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));

        static final MethodHandle RMSNORM_ROWS = bind("dispatch_rmsnorm_rows_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT));

        static final MethodHandle CROSS_ENTROPY_ROWS = bind("dispatch_cross_entropy_rows",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT));

        static final MethodHandle ADD_ROW_BIAS = bind("dispatch_add_row_bias_f32",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT));

        // Fusion
        static final MethodHandle FUSED_DISPATCH = bind("hc4j_fused_dispatch",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG));

        @SuppressWarnings("restricted")
        private static MethodHandle bind(String name, FunctionDescriptor desc) {
            MemorySegment addr = SYMBOLS.find(name).orElseThrow(() -> new UnsatisfiedLinkError(
                    "HC4J native symbol '" + name + "' not found; the native library may be outdated."));
            return LINKER.downcallHandle(addr, desc);
        }
    }

    private NativeBindings() {
        throw new AssertionError("No instances of NativeBindings");
    }

    public static int dispatchBinary(int opIndex, long a, long b, long out, int rank,
                                     MemorySegment shape, MemorySegment stridesA, MemorySegment stridesB, MemorySegment stridesC,
                                     long length, int isContiguous, int dtypeCode) throws Throwable {
        MethodHandle mh = switch (opIndex) {
            case 0 -> Holder.DISPATCH_ADD;
            case 1 -> Holder.DISPATCH_SUB;
            case 2 -> Holder.DISPATCH_MUL;
            case 3 -> Holder.DISPATCH_DIV;
            default -> throw new IllegalArgumentException("Unknown binary op index: " + opIndex);
        };
        return (int) mh.invokeExact(a, b, out, rank, shape, stridesA, stridesB, stridesC, length, isContiguous, dtypeCode);
    }

    public static int dispatchTrig(int opIndex, long a, long out, int rank,
                                   MemorySegment shape, MemorySegment stridesA, MemorySegment stridesC,
                                   long length, int isContiguous) throws Throwable {
        MethodHandle mh = switch (opIndex) {
            case 0 -> Holder.DISPATCH_SIN;
            case 1 -> Holder.DISPATCH_COS;
            case 2 -> Holder.DISPATCH_TAN;
            case 3 -> Holder.DISPATCH_ASIN;
            case 4 -> Holder.DISPATCH_ACOS;
            case 5 -> Holder.DISPATCH_ATAN;
            case 6 -> Holder.DISPATCH_SINH;
            case 7 -> Holder.DISPATCH_COSH;
            case 8 -> Holder.DISPATCH_TANH;
            case 9 -> Holder.DISPATCH_ASINH;
            case 10 -> Holder.DISPATCH_ACOSH;
            case 11 -> Holder.DISPATCH_ATANH;
            default -> throw new IllegalArgumentException("Unknown trig op index: " + opIndex);
        };
        return (int) mh.invokeExact(a, out, rank, shape, stridesA, stridesC, length, isContiguous);
    }

    public static int dispatchExponential(int opIndex, long a, long out, int rank,
                                          MemorySegment shape, MemorySegment stridesA, MemorySegment stridesC,
                                          long length, int isContiguous) throws Throwable {
        MethodHandle mh = switch (opIndex) {
            case 0 -> Holder.DISPATCH_EXP;
            case 1 -> Holder.DISPATCH_LOG;
            case 2 -> Holder.DISPATCH_LOG2;
            case 3 -> Holder.DISPATCH_LOG10;
            case 4 -> Holder.DISPATCH_SQRT;
            default -> throw new IllegalArgumentException("Unknown exponential op index: " + opIndex);
        };
        return (int) mh.invokeExact(a, out, rank, shape, stridesA, stridesC, length, isContiguous);
    }

    public static int dispatchMatmul(long a, long b, long out, int m, int n, int k) throws Throwable {
        return (int) Holder.MATMUL.invokeExact(a, b, out, m, n, k);
    }

    public static int dispatchMatmulEx(long a, long b, long out, int m, int n, int k, int flags) throws Throwable {
        return (int) Holder.MATMUL_EX.invokeExact(a, b, out, m, n, k, flags);
    }

    public static int dispatchTranspose(long in, long out, int rows, int cols) throws Throwable {
        return (int) Holder.TRANSPOSE.invokeExact(in, out, rows, cols);
    }

    public static int matmulPlan(int m, int n, int k, int flags) throws Throwable {
        return (int) Holder.MATMUL_PLAN.invokeExact(m, n, k, flags);
    }

    public static int dispatchActivation(long a, long out, int rank,
                                         MemorySegment shape, MemorySegment stridesA, MemorySegment stridesC,
                                         long length, int isContiguous, int act, int alphaBits) throws Throwable {
        return (int) Holder.ACTIVATION.invokeExact(a, out, rank, shape, stridesA, stridesC, length, isContiguous, act, alphaBits);
    }

    public static int dispatchReduce(long a, long out, long length, int op) throws Throwable {
        return (int) Holder.REDUCE.invokeExact(a, out, length, op);
    }

    public static int dispatchReduceSquare(long a, long out, long length, int op) throws Throwable {
        return (int) Holder.REDUCE_SQUARE.invokeExact(a, out, length, op);
    }

    public static int dispatchReduceDiff2(long a, long b, long out, long length, int op) throws Throwable {
        return (int) Holder.REDUCE_DIFF2.invokeExact(a, b, out, length, op);
    }

    public static int dispatchReduceRows(long a, long out, int rows, int cols, int op) throws Throwable {
        return (int) Holder.REDUCE_ROWS.invokeExact(a, out, rows, cols, op);
    }

    public static int dispatchArgmaxRows(long a, long out, int rows, int cols, int largest) throws Throwable {
        return (int) Holder.ARGMAX_ROWS.invokeExact(a, out, rows, cols, largest);
    }

    public static int dispatchSoftmaxRows(long a, long out, int rows, int cols, int logVariant) throws Throwable {
        return (int) Holder.SOFTMAX_ROWS.invokeExact(a, out, rows, cols, logVariant);
    }

    public static int dispatchLayerNormRows(long a, long gamma, long beta, long out, int rows, int cols, int epsBits) throws Throwable {
        return (int) Holder.LAYERNORM_ROWS.invokeExact(a, gamma, beta, out, rows, cols, epsBits);
    }

    public static int dispatchRmsNormRows(long a, long gamma, long out, int rows, int cols, int epsBits) throws Throwable {
        return (int) Holder.RMSNORM_ROWS.invokeExact(a, gamma, out, rows, cols, epsBits);
    }

    public static int dispatchCrossEntropyRows(long logits, long targets, long out, int rows, int cols) throws Throwable {
        return (int) Holder.CROSS_ENTROPY_ROWS.invokeExact(logits, targets, out, rows, cols);
    }

    public static int dispatchAddRowBias(long a, long bias, long out, int rows, int cols) throws Throwable {
        return (int) Holder.ADD_ROW_BIAS.invokeExact(a, bias, out, rows, cols);
    }

    public static int dispatchFused(MemorySegment program, int len, MemorySegment inputs, int n, long idOut) throws Throwable {
        return (int) Holder.FUSED_DISPATCH.invokeExact(program, len, inputs, n, idOut);
    }

    public static int dispatchCopyStrided(long src, long out, long srcOffset, int rank,
            MemorySegment shape, MemorySegment srcStrides, long len) throws Throwable {
        return (int) Holder.DISPATCH_COPY_STRIDED.invokeExact(src, out, srcOffset, rank, shape, srcStrides, len);
    }

    public static int dispatchMathUnary(int opIndex, long a, long out, int rank,
                                        MemorySegment shape, MemorySegment stridesA, MemorySegment stridesC,
                                        long length, int isContiguous) throws Throwable {
        MethodHandle mh = switch (opIndex) {
            case 0 -> Holder.DISPATCH_ABS_F32;
            case 1 -> Holder.DISPATCH_NEG_F32;
            case 2 -> Holder.DISPATCH_RSQRT_F32;
            case 3 -> Holder.DISPATCH_FLOOR_F32;
            case 4 -> Holder.DISPATCH_CEIL_F32;
            case 5 -> Holder.DISPATCH_ROUND_F32;
            case 6 -> Holder.DISPATCH_SIGN_F32;
            default -> throw new IllegalArgumentException("Unknown math unary op index: " + opIndex);
        };
        return (int) mh.invokeExact(a, out, rank, shape, stridesA, stridesC, length, isContiguous);
    }

    public static int dispatchClampF32(long a, long out, int rank,
                                       MemorySegment shape, MemorySegment stridesA, MemorySegment stridesC,
                                       long length, int isContiguous, float lo, float hi) throws Throwable {
        return (int) Holder.DISPATCH_CLAMP_F32.invokeExact(a, out, rank, shape, stridesA, stridesC, length, isContiguous, lo, hi);
    }

    public static int dispatchUnaryI32(long a, long out, int rank,
                                       MemorySegment shape, MemorySegment stridesA, MemorySegment stridesC,
                                       long length, int isContiguous, int opCode, int pZ, int pW) throws Throwable {
        return (int) Holder.DISPATCH_UNARY_I32.invokeExact(a, out, rank, shape, stridesA, stridesC, length, isContiguous, opCode, pZ, pW);
    }

    public static int dispatchPowF32(long a, long b, long out, int rank,
                                     MemorySegment shape, MemorySegment stridesA, MemorySegment stridesB, MemorySegment stridesC,
                                     long length, int isContiguous) throws Throwable {
        return (int) Holder.DISPATCH_POW_F32.invokeExact(a, b, out, rank, shape, stridesA, stridesB, stridesC, length, isContiguous);
    }

    public static int dispatchMaximum(long a, long b, long out, int rank,
                                      MemorySegment shape, MemorySegment stridesA, MemorySegment stridesB, MemorySegment stridesC,
                                      long length, int isContiguous, int dtypeCode) throws Throwable {
        return (int) Holder.DISPATCH_MAXIMUM.invokeExact(a, b, out, rank, shape, stridesA, stridesB, stridesC, length, isContiguous, dtypeCode);
    }

    public static int dispatchMinimum(long a, long b, long out, int rank,
                                      MemorySegment shape, MemorySegment stridesA, MemorySegment stridesB, MemorySegment stridesC,
                                      long length, int isContiguous, int dtypeCode) throws Throwable {
        return (int) Holder.DISPATCH_MINIMUM.invokeExact(a, b, out, rank, shape, stridesA, stridesB, stridesC, length, isContiguous, dtypeCode);
    }

    public static int dispatchCompare(long a, long b, long out, int rank,
                                      MemorySegment shape, MemorySegment stridesA, MemorySegment stridesB, MemorySegment stridesC,
                                      long length, int isContiguous, int dtypeCode, int cmpOp) throws Throwable {
        return (int) Holder.DISPATCH_COMPARE.invokeExact(a, b, out, rank, shape, stridesA, stridesB, stridesC, length, isContiguous, dtypeCode, cmpOp);
    }

    public static int dispatchWhere(long cond, long x, long y, long out, int rank,
                                    MemorySegment shape, MemorySegment stridesCond, MemorySegment stridesX, MemorySegment stridesY, MemorySegment stridesOut,
                                    long length, int isContiguous, int condDtype, int valDtype) throws Throwable {
        return (int) Holder.DISPATCH_WHERE.invokeExact(cond, x, y, out, rank, shape, stridesCond, stridesX, stridesY, stridesOut, length, isContiguous, condDtype, valDtype);
    }

    public static int dispatchCast(long src, long out, long length, int srcDtype, int dstDtype) throws Throwable {
        return (int) Holder.DISPATCH_CAST.invokeExact(src, out, length, srcDtype, dstDtype);
    }

    public static int dispatchReduceAxis(long in, long out, int outer, int axisDim, int inner, int reduceOp) throws Throwable {
        return (int) Holder.DISPATCH_REDUCE_AXIS.invokeExact(in, out, outer, axisDim, inner, reduceOp);
    }

    public static int dispatchArgmaxAxis(long in, long out, int outer, int axisDim, int inner, int largest) throws Throwable {
        return (int) Holder.DISPATCH_ARGMAX_AXIS.invokeExact(in, out, outer, axisDim, inner, largest);
    }

    public static int dispatchBatchedMatmul(long a, long b, long out, int m, int n, int k, int batchRank,
                                           MemorySegment batchShape, MemorySegment aBatchStrides, MemorySegment bBatchStrides, int flags) throws Throwable {
        return (int) Holder.DISPATCH_BATCHED_MATMUL.invokeExact(a, b, out, m, n, k, batchRank, batchShape, aBatchStrides, bBatchStrides, flags);
    }
}
