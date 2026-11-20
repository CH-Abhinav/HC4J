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

package hc4j.nn;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.internal.ffi.TensorAccess;
import hc4j.internal.layout.Rows;
import hc4j.internal.layout.ValidUtil;
import hc4j.internal.ops.NnOps;
import hc4j.linalg.Linalg;
import java.util.Objects;

/**
 * Public neural network facade exposing activations, normalizations, reductions,
 * and loss operations.
 */
public final class Nn {

    private Nn() {
        throw new AssertionError("No instances of Nn");
    }

    private static Tensor createLike(Tensor t) {
        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(t);
        return Tensor.empty(t.getDType(), access.getShapeUnsafe(t).clone());
    }

    private static Tensor createReduced(Tensor t, DType dtype) {
        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(t);
        int[] shape = access.getShapeUnsafe(t);
        Rows r = ValidUtil.rows(shape, t.getSize());
        return Tensor.empty(dtype, r.reducedShape(shape));
    }

    // --- Activations ---

    public static Tensor relu(Tensor a) {
        Tensor res = createLike(a);
        try { return relu(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor relu(Tensor a, Tensor out) { return NnOps.activation(Activation.RELU, a, out, 0f); }

    public static Tensor gelu(Tensor a) {
        Tensor res = createLike(a);
        try { return gelu(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor gelu(Tensor a, Tensor out) { return NnOps.activation(Activation.GELU, a, out, 0f); }

    public static Tensor silu(Tensor a) {
        Tensor res = createLike(a);
        try { return silu(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor silu(Tensor a, Tensor out) { return NnOps.activation(Activation.SILU, a, out, 0f); }

    public static Tensor sigmoid(Tensor a) {
        Tensor res = createLike(a);
        try { return sigmoid(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor sigmoid(Tensor a, Tensor out) { return NnOps.activation(Activation.SIGMOID, a, out, 0f); }

    public static Tensor softplus(Tensor a) {
        Tensor res = createLike(a);
        try { return softplus(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor softplus(Tensor a, Tensor out) { return NnOps.activation(Activation.SOFTPLUS, a, out, 0f); }

    public static Tensor leakyRelu(Tensor a, float slope) {
        Tensor res = createLike(a);
        try { return leakyRelu(a, res, slope); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor leakyRelu(Tensor a, Tensor out, float slope) { return NnOps.activation(Activation.LEAKY_RELU, a, out, slope); }

    public static Tensor elu(Tensor a, float alpha) {
        Tensor res = createLike(a);
        try { return elu(a, res, alpha); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor elu(Tensor a, Tensor out, float alpha) { return NnOps.activation(Activation.ELU, a, out, alpha); }

    // --- Softmax ---

    public static Tensor softmax(Tensor a) {
        Tensor res = createLike(a);
        try { return softmax(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor softmax(Tensor a, Tensor out) { return NnOps.softmaxRows(a, out, false); }

    public static Tensor logSoftmax(Tensor a) {
        Tensor res = createLike(a);
        try { return logSoftmax(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor logSoftmax(Tensor a, Tensor out) { return NnOps.softmaxRows(a, out, true); }

    // --- Normalizations ---

    public static Tensor layerNorm(Tensor a, float eps) {
        return layerNorm(a, null, null, eps);
    }
    public static Tensor layerNorm(Tensor a, Tensor gamma, Tensor beta, float eps) {
        Tensor res = createLike(a);
        try { return layerNorm(a, gamma, beta, eps, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor layerNorm(Tensor a, Tensor gamma, Tensor beta, float eps, Tensor out) {
        return NnOps.layerNormRows(a, gamma, beta, out, eps);
    }

    public static Tensor rmsNorm(Tensor a, Tensor gamma, float eps) {
        Tensor res = createLike(a);
        try { return rmsNorm(a, gamma, eps, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor rmsNorm(Tensor a, Tensor gamma, float eps, Tensor out) {
        return NnOps.rmsNormRows(a, gamma, out, eps);
    }

    // --- Losses & Bias & Linear ---

    public static Tensor crossEntropy(Tensor logits, Tensor targets) {
        Tensor res = createReduced(logits, DType.f32);
        try { return crossEntropy(logits, targets, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor crossEntropy(Tensor logits, Tensor targets, Tensor out) {
        return NnOps.crossEntropyRows(logits, targets, out);
    }

    public static Tensor addRowBias(Tensor a, Tensor bias) {
        Tensor res = createLike(a);
        try { return addRowBias(a, bias, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor addRowBias(Tensor a, Tensor bias, Tensor out) {
        return NnOps.addRowBias(a, bias, out);
    }

    public static Tensor linear(Tensor input, Tensor weight, Tensor bias) {
        Tensor res = Linalg.matmul(input, weight);
        try {
            return bias == null ? res : addRowBias(res, bias, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }
    public static Tensor linear(Tensor input, Tensor weight, Tensor bias, Tensor out) {
        Linalg.matmul(input, weight, out);
        return bias == null ? out : addRowBias(out, bias, out);
    }

    // --- Full Reductions ---

    public static Tensor sum(Tensor a) {
        Tensor res = Tensor.empty(DType.f32, 1);
        try { return sum(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor sum(Tensor a, Tensor out) { return NnOps.reduce(Reduction.SUM, a, out); }

    public static Tensor mean(Tensor a) {
        Tensor res = Tensor.empty(DType.f32, 1);
        try { return mean(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor mean(Tensor a, Tensor out) { return NnOps.reduce(Reduction.MEAN, a, out); }

    public static Tensor maxValue(Tensor a) {
        Tensor res = Tensor.empty(DType.f32, 1);
        try { return maxValue(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor maxValue(Tensor a, Tensor out) { return NnOps.reduce(Reduction.MAX, a, out); }

    public static Tensor minValue(Tensor a) {
        Tensor res = Tensor.empty(DType.f32, 1);
        try { return minValue(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor minValue(Tensor a, Tensor out) { return NnOps.reduce(Reduction.MIN, a, out); }

    // --- Row Reductions ---

    public static Tensor sumRows(Tensor a) {
        Tensor res = createReduced(a, DType.f32);
        try { return sumRows(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor sumRows(Tensor a, Tensor out) { return NnOps.reduceRows(Reduction.SUM, a, out); }

    public static Tensor meanRows(Tensor a) {
        Tensor res = createReduced(a, DType.f32);
        try { return meanRows(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor meanRows(Tensor a, Tensor out) { return NnOps.reduceRows(Reduction.MEAN, a, out); }

    public static Tensor maxRows(Tensor a) {
        Tensor res = createReduced(a, DType.f32);
        try { return maxRows(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor maxRows(Tensor a, Tensor out) { return NnOps.reduceRows(Reduction.MAX, a, out); }

    public static Tensor minRows(Tensor a) {
        Tensor res = createReduced(a, DType.f32);
        try { return minRows(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor minRows(Tensor a, Tensor out) { return NnOps.reduceRows(Reduction.MIN, a, out); }

    public static Tensor argmaxRows(Tensor a) {
        Tensor res = createReduced(a, DType.i32);
        try { return argmaxRows(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor argmaxRows(Tensor a, Tensor out) { return NnOps.argmaxRows(a, out, true); }

    public static Tensor argminRows(Tensor a) {
        Tensor res = createReduced(a, DType.i32);
        try { return argminRows(a, res); } catch (RuntimeException | Error e) { res.close(); throw e; }
    }
    public static Tensor argminRows(Tensor a, Tensor out) { return NnOps.argmaxRows(a, out, false); }
}
