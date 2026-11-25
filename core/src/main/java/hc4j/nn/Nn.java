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

    // --- Linear, Embedding & RoPE ---

    /**
     * Computes a linear transformation with weights in transposed format: {@code input @ weightOutByIn^T + bias}.
     * Supports both f32 and f16 weights.
     *
     * @param input         input activations of shape {@code [..., M, K]} or {@code [M, K]}
     * @param weightOutByIn weights of shape {@code [N, K]} (f32 or f16)
     * @param bias          optional 1-D bias of shape {@code [N]} or broadcastable tensor (may be null)
     * @return transformed tensor of shape {@code [..., M, N]}
     */
    public static Tensor linearTransposed(Tensor input, Tensor weightOutByIn, Tensor bias) {
        Tensor res = Linalg.matmulTransposedB(input, weightOutByIn);
        if (bias != null) {
            try {
                res.add(bias, res);
            } catch (RuntimeException | Error e) {
                res.close();
                throw e;
            }
        }
        return res;
    }

    /**
     * Computes a linear transformation with weights in transposed format into {@code out}: {@code input @ weightOutByIn^T + bias}.
     *
     * @param input         input activations of shape {@code [..., M, K]} or {@code [M, K]}
     * @param weightOutByIn weights of shape {@code [N, K]} (f32 or f16)
     * @param bias          optional 1-D bias of shape {@code [N]} or broadcastable tensor (may be null)
     * @param out           destination tensor
     * @return {@code out}
     */
    public static Tensor linearTransposed(Tensor input, Tensor weightOutByIn, Tensor bias, Tensor out) {
        Linalg.matmulTransposedB(input, weightOutByIn, out);
        if (bias != null) {
            out.add(bias, out);
        }
        return out;
    }

    /**
     * Looks up embeddings for {@code tokenIds} in {@code table}.
     * Supports both f32 and f16 embedding tables.
     *
     * @param table    2-D embedding table of shape {@code [vocabSize, hiddenDim]}
     * @param tokenIds integer token tensor
     * @return float embedding tensor of shape {@code [..., hiddenDim]}
     */
    public static Tensor embedding(Tensor table, Tensor tokenIds) {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(tokenIds, "tokenIds");
        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(table);
        access.ensureOpen(tokenIds);
        int[] tableShape = access.getShapeUnsafe(table);
        if (tableShape.length != 2) {
            throw new IllegalArgumentException("Embedding table must be 2-D [vocabSize, hiddenDim], got " + table.dim() + "-D");
        }
        int[] tokenShape = access.getShapeUnsafe(tokenIds);
        int[] outShape = new int[tokenShape.length + 1];
        System.arraycopy(tokenShape, 0, outShape, 0, tokenShape.length);
        outShape[tokenShape.length] = tableShape[1];
        Tensor res = Tensor.empty(DType.f32, outShape);
        try {
            return NnOps.embedding(table, tokenIds, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Looks up embeddings for {@code tokenIds} into caller-provided {@code out}.
     *
     * @param table    2-D embedding table of shape {@code [vocabSize, hiddenDim]}
     * @param tokenIds integer token tensor
     * @param out      destination tensor
     * @return {@code out}
     */
    public static Tensor embedding(Tensor table, Tensor tokenIds, Tensor out) {
        return NnOps.embedding(table, tokenIds, out);
    }

    /**
     * Applies Rotary Position Embedding (RoPE) using the NeoX half-split convention.
     *
     * @param x              input tensor of shape {@code [batch, seqLen, numHeads, headDim]}, {@code [seqLen, numHeads, headDim]}, or {@code [seqLen, headDim]}
     * @param positionOffset token position offset
     * @param theta          rotary embedding base theta (e.g. 10000.0f)
     * @return newly allocated tensor with rotary embeddings applied
     */
    public static Tensor rope(Tensor x, int positionOffset, float theta) {
        Objects.requireNonNull(x, "x");
        Tensor res = createLike(x);
        try {
            return NnOps.rope(x, positionOffset, theta, res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    /**
     * Applies Rotary Position Embedding (RoPE) into caller-provided {@code out}.
     * In-place safe when {@code x == out}.
     *
     * @param x              input tensor
     * @param positionOffset token position offset
     * @param theta          rotary embedding base theta
     * @param out            destination tensor (may be {@code x})
     * @return {@code out}
     */
    public static Tensor rope(Tensor x, int positionOffset, float theta, Tensor out) {
        return NnOps.rope(x, positionOffset, theta, out);
    }

    /**
     * Appends and transposes incoming Key or Value states of shape {@code [seqLen, numKvHeads, headDim]}
     * directly into a pre-allocated VRAM KV-cache tensor of shape {@code [numKvHeads, maxSeqLen, headDim]}
     * starting at token index {@code posOffset}, without host round-trips or buffer reallocations.
     *
     * @param cache     destination KV-cache tensor of shape {@code [numKvHeads, maxSeqLen, headDim]}
     * @param newValues incoming key or value tensor of shape {@code [seqLen, numKvHeads, headDim]}
     * @param posOffset starting sequence position offset in the cache ({@code posOffset + seqLen <= maxSeqLen})
     * @throws NullPointerException     if {@code cache} or {@code newValues} is null
     * @throws IllegalArgumentException if shapes, dtypes, aliasing, or bounds are invalid
     */
    public static void kvCacheAppend(Tensor cache, Tensor newValues, int posOffset) {
        NnOps.kvCacheAppend(cache, newValues, posOffset);
    }
}
