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

package hc4j.internal.nn;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.internal.ffi.NativeBindings;
import hc4j.internal.ffi.NativeEngine;
import hc4j.internal.ffi.TensorAccess;
import hc4j.internal.layout.Rows;
import hc4j.internal.layout.ValidUtil;
import java.util.Objects;

/**
 * Dispatcher for row-wise neural network operations: softmax, LayerNorm, RMSNorm,
 * cross-entropy loss, and bias addition.
 */
public final class NormAndSoftmaxOps {

    private NormAndSoftmaxOps() {
        throw new AssertionError("No instances of NormAndSoftmaxOps");
    }

    /**
     * Resolves 2-D row/column descriptor over the trailing axis of {@code t}.
     *
     * @param t open tensor
     * @return validated {@link Rows} descriptor
     */
    public static Rows rows(Tensor t) {
        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(t);
        return ValidUtil.rows(access.getShapeUnsafe(t), t.getSize());
    }

    private static ValidUtil.SafeOperand prepareOptional(Tensor t) {
        return t == null ? new ValidUtil.SafeOperand(null, false) : ValidUtil.prepareContiguousZeroOffset(t);
    }

    private static long validateAffineParam(String opLabel, Tensor param, int cols, TensorAccess access) {
        if (param == null) {
            return 0L;
        }
        access.ensureOpen(param);
        ValidUtil.requireOperandF32(opLabel, param.getDType());
        if (param.dim() != 1 || access.getShapeUnsafe(param)[0] != cols) {
            throw new IllegalArgumentException(opLabel + " must be 1-D of length " + cols);
        }
        return access.getVramId(param);
    }

    /**
     * Computes row-wise softmax or log-softmax along the last dimension.
     *
     * @param a source {@code f32} tensor
     * @param res destination {@code f32} tensor
     * @param logarithmic true for log-softmax, false for standard softmax
     * @return {@code res}
     */
    public static Tensor softmaxRows(Tensor a, Tensor res, boolean logarithmic) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("softmaxRows", a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeA = ValidUtil.prepareContiguousZeroOffset(a)) {
            Tensor ea = safeA.tensor();
            int[] sa = access.getShapeUnsafe(ea);
            ValidUtil.requireResultShape(sa, access.getShapeUnsafe(res));
            Rows r = ValidUtil.rows(sa, ea.getSize());

            int status = NativeBindings.dispatchSoftmaxRows(
                    access.getVramId(ea), access.getVramId(res), r.rows(), r.cols(), logarithmic ? 1 : 0);
            NativeEngine.checkStatus(status, "NormAndSoftmaxOps.softmaxRows");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: softmaxRows failed", t);
        }
    }

    /**
     * Applies row-wise LayerNorm with optional affine scale ({@code gamma}) and shift ({@code beta}).
     *
     * @param a input {@code f32} tensor
     * @param gamma optional 1-D scale parameter matching trailing dimension
     * @param beta optional 1-D bias parameter matching trailing dimension
     * @param res destination {@code f32} tensor
     * @param eps numerical stability epsilon
     * @return {@code res}
     */
    public static Tensor layerNormRows(Tensor a, Tensor gamma, Tensor beta, Tensor res, float eps) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("layerNormRows", a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeA = ValidUtil.prepareContiguousZeroOffset(a);
             ValidUtil.SafeOperand safeG = prepareOptional(gamma);
             ValidUtil.SafeOperand safeB = prepareOptional(beta)) {

            Tensor ea = safeA.tensor();
            int[] sa = access.getShapeUnsafe(ea);
            ValidUtil.requireResultShape(sa, access.getShapeUnsafe(res));
            Rows r = ValidUtil.rows(sa, ea.getSize());

            long gammaId = validateAffineParam("layerNorm gamma", safeG.tensor(), r.cols(), access);
            long betaId = validateAffineParam("layerNorm beta", safeB.tensor(), r.cols(), access);

            int status = NativeBindings.dispatchLayerNormRows(
                    access.getVramId(ea), gammaId, betaId, access.getVramId(res),
                    r.rows(), r.cols(), Float.floatToRawIntBits(eps));
            NativeEngine.checkStatus(status, "NormAndSoftmaxOps.layerNormRows");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: layerNormRows failed", t);
        }
    }

    /**
     * Applies row-wise RMSNorm with optional affine scale ({@code gamma}).
     *
     * @param a input {@code f32} tensor
     * @param gamma optional 1-D scale parameter matching trailing dimension
     * @param res destination {@code f32} tensor
     * @param eps numerical stability epsilon
     * @return {@code res}
     */
    public static Tensor rmsNormRows(Tensor a, Tensor gamma, Tensor res, float eps) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("rmsNormRows", a.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeA = ValidUtil.prepareContiguousZeroOffset(a);
             ValidUtil.SafeOperand safeG = prepareOptional(gamma)) {

            Tensor ea = safeA.tensor();
            int[] sa = access.getShapeUnsafe(ea);
            ValidUtil.requireResultShape(sa, access.getShapeUnsafe(res));
            Rows r = ValidUtil.rows(sa, ea.getSize());
            long gammaId = validateAffineParam("rmsNorm gamma", safeG.tensor(), r.cols(), access);

            int status = NativeBindings.dispatchRmsNormRows(
                    access.getVramId(ea), gammaId, access.getVramId(res),
                    r.rows(), r.cols(), Float.floatToRawIntBits(eps));
            NativeEngine.checkStatus(status, "NormAndSoftmaxOps.rmsNormRows");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: rmsNormRows failed", t);
        }
    }

    /**
     * Computes row-wise softmax cross-entropy loss from unnormalized {@code logits} and {@code i32} class {@code targets}.
     *
     * @param logits unnormalized class logits ({@code f32})
     * @param targets target class indices ({@code i32})
     * @param res destination loss tensor ({@code f32})
     * @return {@code res}
     */
    public static Tensor crossEntropyRows(Tensor logits, Tensor targets, Tensor res) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(targets, "targets");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(logits);
        access.ensureOpen(targets);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("crossEntropy logits", logits.getDType());
        if (targets.getDType() != DType.i32) {
            throw new IllegalArgumentException("crossEntropy targets must be i32, got " + targets.getDType());
        }
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeL = ValidUtil.prepareContiguousZeroOffset(logits);
             ValidUtil.SafeOperand safeT = ValidUtil.prepareContiguousZeroOffset(targets)) {

            Tensor el = safeL.tensor();
            Tensor et = safeT.tensor();
            int[] sl = access.getShapeUnsafe(el);
            Rows r = ValidUtil.rows(sl, el.getSize());
            int[] expectedShape = r.reducedShape(sl);
            ValidUtil.requireResultShape(expectedShape, access.getShapeUnsafe(et));
            ValidUtil.requireResultShape(expectedShape, access.getShapeUnsafe(res));

            int status = NativeBindings.dispatchCrossEntropyRows(
                    access.getVramId(el), access.getVramId(et), access.getVramId(res), r.rows(), r.cols());
            NativeEngine.checkStatus(status, "NormAndSoftmaxOps.crossEntropyRows");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: crossEntropyRows failed", t);
        }
    }

    /**
     * Adds a 1-D row bias vector to each row of {@code a} along the trailing dimension.
     *
     * @param a input {@code f32} tensor
     * @param bias 1-D {@code f32} bias vector matching trailing dimension
     * @param res destination {@code f32} tensor
     * @return {@code res}
     */
    public static Tensor addRowBias(Tensor a, Tensor bias, Tensor res) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(a);
        access.ensureOpen(bias);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("addRowBias input", a.getDType());
        ValidUtil.requireOperandF32("addRowBias bias", bias.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeA = ValidUtil.prepareContiguousZeroOffset(a);
             ValidUtil.SafeOperand safeB = ValidUtil.prepareContiguousZeroOffset(bias)) {

            Tensor ea = safeA.tensor();
            Tensor eb = safeB.tensor();
            int[] sa = access.getShapeUnsafe(ea);
            ValidUtil.requireResultShape(sa, access.getShapeUnsafe(res));
            Rows r = ValidUtil.rows(sa, ea.getSize());

            if (eb.dim() != 1 || access.getShapeUnsafe(eb)[0] != r.cols()) {
                throw new IllegalArgumentException(
                        "bias must be a 1-D tensor of length " + r.cols() + ", got " + eb.dim() + "-D");
            }

            int status = NativeBindings.dispatchAddRowBias(
                    access.getVramId(ea), access.getVramId(eb), access.getVramId(res), r.rows(), r.cols());
            NativeEngine.checkStatus(status, "NormAndSoftmaxOps.addRowBias");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: addRowBias failed", t);
        }
    }
}
