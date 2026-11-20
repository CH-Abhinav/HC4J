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

package hc4j.testutil;

import hc4j.nn.Activation;
import java.util.Arrays;

/**
 * Pure Java reference implementations of mathematical and neural network operations.
 */
public final class ReferenceOps {

    private ReferenceOps() {
        throw new AssertionError("No instances of ReferenceOps");
    }

    public static double activation(Activation act, double x, float alpha) {
        return switch (act) {
            case RELU -> Math.max(x, 0.0);
            case LEAKY_RELU -> x > 0 ? x : alpha * x;
            case ELU -> x > 0 ? x : alpha * (Math.exp(x) - 1);
            case GELU -> 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
            case SILU -> x / (1 + Math.exp(-x));
            case SIGMOID -> 1 / (1 + Math.exp(-x));
            case SOFTPLUS -> Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
        };
    }

    public static float[] matmul(float[] a, float[] b, int m, int n, int k, boolean ta, boolean tb) {
        float[] c = new float[m * n];
        for (int r = 0; r < m; r++) {
            for (int col = 0; col < n; col++) {
                double sum = 0;
                for (int p = 0; p < k; p++) {
                    float va = ta ? a[p * m + r] : a[r * k + p];
                    float vb = tb ? b[col * k + p] : b[p * n + col];
                    sum += (double) va * vb;
                }
                c[r * n + col] = (float) sum;
            }
        }
        return c;
    }

    public static float[] transpose(float[] in, int rows, int cols) {
        float[] out = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out[c * rows + r] = in[r * cols + c];
            }
        }
        return out;
    }

    public static float[] softmax(float[] in, int rows, int cols, boolean log) {
        float[] out = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            int off = r * cols;
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < cols; c++) {
                max = Math.max(max, in[off + c]);
            }
            double sum = 0;
            for (int c = 0; c < cols; c++) {
                sum += Math.exp(in[off + c] - max);
            }
            if (log) {
                double lse = max + Math.log(sum);
                for (int c = 0; c < cols; c++) {
                    out[off + c] = (float) (in[off + c] - lse);
                }
            } else {
                for (int c = 0; c < cols; c++) {
                    out[off + c] = (float) (Math.exp(in[off + c] - max) / sum);
                }
            }
        }
        return out;
    }

    public static float[] layerNorm(float[] in, float[] gamma, float[] beta, int rows, int cols, float eps) {
        float[] out = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            int off = r * cols;
            double mean = 0;
            for (int c = 0; c < cols; c++) mean += in[off + c];
            mean /= cols;

            double var = 0;
            for (int c = 0; c < cols; c++) {
                double diff = in[off + c] - mean;
                var += diff * diff;
            }
            var /= cols;
            double invStd = 1.0 / Math.sqrt(var + eps);

            for (int c = 0; c < cols; c++) {
                double norm = (in[off + c] - mean) * invStd;
                double g = gamma != null ? gamma[c] : 1.0;
                double b = beta != null ? beta[c] : 0.0;
                out[off + c] = (float) (norm * g + b);
            }
        }
        return out;
    }

    public static float[] rmsNorm(float[] in, float[] gamma, int rows, int cols, float eps) {
        float[] out = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            int off = r * cols;
            double sqSum = 0;
            for (int c = 0; c < cols; c++) {
                double v = in[off + c];
                sqSum += v * v;
            }
            double rms = Math.sqrt(sqSum / cols + eps);
            double invRms = 1.0 / rms;
            for (int c = 0; c < cols; c++) {
                double g = gamma != null ? gamma[c] : 1.0;
                out[off + c] = (float) (in[off + c] * invRms * g);
            }
        }
        return out;
    }

    public static float[] crossEntropy(float[] logits, int[] targets, int rows, int cols) {
        float[] out = new float[rows];
        for (int r = 0; r < rows; r++) {
            int off = r * cols;
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < cols; c++) max = Math.max(max, logits[off + c]);
            double sum = 0;
            for (int c = 0; c < cols; c++) sum += Math.exp(logits[off + c] - max);
            double lse = max + Math.log(sum);
            int target = targets[r];
            out[r] = (float) (lse - logits[off + target]);
        }
        return out;
    }
}
