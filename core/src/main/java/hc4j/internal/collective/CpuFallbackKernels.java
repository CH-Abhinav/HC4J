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

package hc4j.internal.collective;

import hc4j.nn.Activation;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Pure Java / FFM host CPU fallback compute kernels supporting Tier 3 CPU-in-place execution
 * and lossless online-softmax Split-KV partial merging.
 */
public final class CpuFallbackKernels {

    private CpuFallbackKernels() {
        throw new AssertionError("No instances of CpuFallbackKernels");
    }

    /**
     * Executes a contiguous row range {@code [rStart, rStart + units)} of {@code C = A * B} on host CPU.
     *
     * @param aSeg host segment containing A [M, K]
     * @param bSeg host segment containing B [K, N]
     * @param outSeg destination segment containing C [M, N]
     * @param rStart starting row index
     * @param units number of rows to compute
     * @param n columns in B and C
     * @param k inner dimension
     */
    public static void computeCpuGemm(MemorySegment aSeg, MemorySegment bSeg, MemorySegment outSeg,
                                      int rStart, int units, int n, int k) {
        if (units <= 0 || n <= 0 || k <= 0) {
            return;
        }
        if (NativeOptimizerBridge.isCpuKernelsAvailable()) {
            long aBytes = (long) units * k * 4L;
            long outBytes = (long) units * n * 4L;
            MemorySegment aSlice = aSeg.asSlice((long) rStart * k * 4L, aBytes);
            MemorySegment outSlice = outSeg.asSlice((long) rStart * n * 4L, outBytes);
            NativeOptimizerBridge.cpuGemmF32(aSlice, bSeg, outSlice, units, n, k, 0);
            return;
        }
        for (int i = 0; i < units; i++) {
            int globalRow = rStart + i;
            long aRowOff = (long) globalRow * k;
            long outRowOff = (long) globalRow * n;
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int p = 0; p < k; p++) {
                    float aVal = aSeg.get(ValueLayout.JAVA_FLOAT, (aRowOff + p) * 4L);
                    float bVal = bSeg.get(ValueLayout.JAVA_FLOAT, ((long) p * n + j) * 4L);
                    sum += aVal * bVal;
                }
                outSeg.set(ValueLayout.JAVA_FLOAT, (outRowOff + j) * 4L, sum);
            }
        }
    }

    /**
     * Executes a column-parallel / row-parallel Tensor Parallel MLP slice {@code [hStart, hStart + units)}
     * on the host CPU node.
     *
     * @param xSeg input activations X [m, dIn]
     * @param w1Seg first projection weights W1 [dIn, h]
     * @param b1Seg optional bias vector [h] (may be null)
     * @param w2Seg second projection weights W2 [h, dOut]
     * @param m batch / sequence length
     * @param dIn input hidden dimension
     * @param h total MLP intermediate width
     * @param dOut output hidden dimension
     * @param hStart starting intermediate channel index
     * @param units number of intermediate channels assigned to CPU
     * @param activation elementwise activation between projections (may be null)
     * @return partial output matrix [m * dOut]
     */
    public static float[] computeCpuMlpSlice(MemorySegment xSeg, MemorySegment w1Seg, MemorySegment b1Seg,
                                             MemorySegment w2Seg, int m, int dIn, int h, int dOut,
                                             int hStart, int units, Activation activation) {
        int actCode = activationCodeOrMinValue(activation);
        if (actCode != Integer.MIN_VALUE && m > 0 && dIn > 0 && h > 0 && dOut > 0 && units > 0
                && NativeOptimizerBridge.isCpuKernelsAvailable()) {
            float[] yCpu = new float[m * dOut];
            NativeOptimizerBridge.cpuMlpSliceF32(
                    xSeg, w1Seg, b1Seg, w2Seg, MemorySegment.ofArray(yCpu),
                    m, dIn, h, dOut, hStart, units, actCode);
            return yCpu;
        }
        float[] hSlice = new float[m * units];

        for (int i = 0; i < m; i++) {
            long xRowOff = (long) i * dIn;
            int hRowOff = i * units;
            for (int j = 0; j < units; j++) {
                int globalH = hStart + j;
                float sum = 0.0f;
                for (int p = 0; p < dIn; p++) {
                    float xVal = xSeg.get(ValueLayout.JAVA_FLOAT, (xRowOff + p) * 4L);
                    float wVal = w1Seg.get(ValueLayout.JAVA_FLOAT, ((long) p * h + globalH) * 4L);
                    sum += xVal * wVal;
                }
                if (b1Seg != null) {
                    sum += b1Seg.get(ValueLayout.JAVA_FLOAT, (long) globalH * 4L);
                }
                hSlice[hRowOff + j] = applyActivation(sum, activation);
            }
        }

        float[] yCpu = new float[m * dOut];
        for (int i = 0; i < m; i++) {
            int hRowOff = i * units;
            int yRowOff = i * dOut;
            for (int j = 0; j < dOut; j++) {
                float sum = 0.0f;
                for (int p = 0; p < units; p++) {
                    int globalH = hStart + p;
                    float hVal = hSlice[hRowOff + p];
                    float wVal = w2Seg.get(ValueLayout.JAVA_FLOAT, ((long) globalH * dOut + j) * 4L);
                    sum += hVal * wVal;
                }
                yCpu[yRowOff + j] = sum;
            }
        }

        return yCpu;
    }

    /**
     * Evaluates local online-softmax decode attention on CPU over cached KV shards in system RAM.
     *
     * @param qSeg query segment [numHeads, headDim]
     * @param kShard local key cache shard [numKvHeads, tokens, headDim]
     * @param vShard local value cache shard [numKvHeads, tokens, headDim]
     * @param tokens local token count
     * @param numHeads query head count
     * @param numKvHeads KV head count
     * @param headDim per-head dimension
     * @param scale attention dot-product scale
     * @param outO destination array [numHeads * headDim]
     * @param outStats destination online-softmax stats array [numHeads * 2] storing {@code (m, l)} pairs
     */
    public static void computeCpuAttention(MemorySegment qSeg, MemorySegment kShard, MemorySegment vShard,
                                           int tokens, int numHeads, int numKvHeads, int headDim, float scale,
                                           float[] outO, float[] outStats) {
        if (tokens > 0 && numHeads > 0 && numKvHeads > 0 && headDim > 0
                && NativeOptimizerBridge.isCpuKernelsAvailable()) {
            NativeOptimizerBridge.cpuSplitKvAttnF32(
                    qSeg, kShard, vShard, outO, outStats, tokens, numHeads, numKvHeads, headDim, scale);
            return;
        }
        int groupSize = numHeads / numKvHeads;
        float[] scores = new float[tokens];

        for (int h = 0; h < numHeads; h++) {
            int kvH = h / groupSize;
            float m = -Float.MAX_VALUE;
            float l = 0.0f;

            for (int t = 0; t < tokens; t++) {
                long kOff = ((long) kvH * tokens + t) * headDim;
                float dot = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    float qVal = qSeg.get(ValueLayout.JAVA_FLOAT, ((long) h * headDim + d) * 4L);
                    float kVal = kShard.get(ValueLayout.JAVA_FLOAT, (kOff + d) * 4L);
                    dot += qVal * kVal;
                }
                float score = dot * scale;
                scores[t] = score;
                float newM = Math.max(m, score);
                l = l * (float) Math.exp(m - newM) + (float) Math.exp(score - newM);
                m = newM;
            }

            outStats[2 * h] = m;
            outStats[2 * h + 1] = l;

            if (l > 0.0f) {
                for (int d = 0; d < headDim; d++) {
                    float acc = 0.0f;
                    for (int t = 0; t < tokens; t++) {
                        float weight = (float) Math.exp(scores[t] - m);
                        long vOff = ((long) kvH * tokens + t) * headDim + d;
                        float vVal = vShard.get(ValueLayout.JAVA_FLOAT, vOff * 4L);
                        acc += weight * vVal;
                    }
                    outO[h * headDim + d] = acc / l;
                }
            }
        }
    }

    /**
     * Pure-Java fallback performing lossless online-softmax reduction across Split-KV worker partials.
     *
     * @param numHeads query head count
     * @param headDim per-head dimension
     * @param partialsO worker partial outputs {@code [worker][numHeads * headDim]}
     * @param partialsStats worker log-sum-exp pairs {@code [worker][2 * numHeads]} ({@code [m, l]})
     * @param outSeg destination segment sized at least {@code numHeads * headDim * 4} bytes
     */
    public static void mergeSplitKvPartialsCpu(int numHeads, int headDim, float[][] partialsO,
                                               float[][] partialsStats, MemorySegment outSeg) {
        int workerCount = partialsO.length;
        float[] weights = new float[workerCount];

        for (int h = 0; h < numHeads; h++) {
            float mH = -Float.MAX_VALUE;
            boolean hasPositive = false;
            for (int w = 0; w < workerCount; w++) {
                float mI = partialsStats[w][2 * h];
                float lI = partialsStats[w][2 * h + 1];
                if (lI > 0.0f && mI > mH) {
                    mH = mI;
                    hasPositive = true;
                }
            }

            float lH = 0.0f;
            for (int w = 0; w < workerCount; w++) {
                float mI = partialsStats[w][2 * h];
                float lI = partialsStats[w][2 * h + 1];
                if (hasPositive && lI > 0.0f) {
                    float wI = lI * (float) Math.exp(mI - mH);
                    weights[w] = wI;
                    lH += wI;
                } else {
                    weights[w] = 0.0f;
                }
            }

            for (int d = 0; d < headDim; d++) {
                float acc = 0.0f;
                for (int w = 0; w < workerCount; w++) {
                    if (weights[w] > 0.0f) {
                        acc += weights[w] * partialsO[w][h * headDim + d];
                    }
                }
                float val = (lH > 0.0f) ? (acc / lH) : 0.0f;
                outSeg.set(ValueLayout.JAVA_FLOAT, ((long) h * headDim + d) * 4L, val);
            }
        }
    }

    private static int activationCodeOrMinValue(Activation activation) {
        if (activation == null) {
            return -1;
        }
        return switch (activation) {
            case RELU -> 0;
            case GELU -> 3;
            case SILU -> 4;
            default -> Integer.MIN_VALUE;
        };
    }

    private static float applyActivation(float x, Activation activation) {
        if (activation == null) {
            return x;
        }
        return switch (activation) {
            case RELU -> Math.max(x, 0.0f);
            case GELU -> {
                float cubic = 0.7978845608028654f * (x + 0.044715f * x * x * x);
                float inner = Math.clamp(cubic, -10.0f, 10.0f);
                yield 0.5f * x * (1.0f + (float) Math.tanh(inner));
            }
            case SILU -> {
                float e = (float) Math.exp(-Math.abs(x));
                float num = (x >= 0.0f) ? 1.0f : e;
                yield x * num / (1.0f + e);
            }
            default -> throw new UnsupportedOperationException("Unsupported MLP activation: " + activation);
        };
    }
}
