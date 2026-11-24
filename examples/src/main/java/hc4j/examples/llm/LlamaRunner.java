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

package hc4j.examples.llm;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.collective.Collective;
import hc4j.collective.RolePlan;
import hc4j.collective.SplitKvSpec;
import hc4j.collective.WorkloadSpec;
import hc4j.nn.Nn;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * High-performance pure-Java Transformer decoder forward runner for Llama-3.2, Qwen2.5, and SmolLM2 models.
 * <p>
 * Supports portable f16 and f32 weights, RMSNorm, GQA (Grouped Query Attention), NeoX half-split RoPE,
 * SwiGLU MLP, KV-Caching, and live Prefill &rarr; Decode Auto-Morphing telemetry when attached to a {@link Collective}.
 */
public final class LlamaRunner implements AutoCloseable {

    private final LlamaConfig config;
    private final LlamaWeights weights;
    private final Collective collective;

    // KV Cache stored per layer: list of tokens, each token is a float[numKvHeads * headDim]
    private final List<List<float[]>> kCache;
    private final List<List<float[]>> vCache;

    // Telemetry trace recording phase transitions and role plans
    private final List<String> morphTrace = new ArrayList<>();

    private int curPos = 0;
    private boolean closed = false;

    /**
     * Creates a new runner for the given config and weights.
     *
     * @param config  model architecture configuration
     * @param weights loaded GPU weights
     */
    public LlamaRunner(LlamaConfig config, LlamaWeights weights) {
        this(config, weights, null);
    }

    /**
     * Creates a new runner attached to a {@link Collective} for distributed execution and auto-morphing.
     *
     * @param config     model architecture configuration
     * @param weights    loaded GPU weights
     * @param collective optional collective (may be null for single-device execution)
     */
    public LlamaRunner(LlamaConfig config, LlamaWeights weights, Collective collective) {
        this.config = Objects.requireNonNull(config, "config");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.collective = collective;

        this.kCache = new ArrayList<>(config.numHiddenLayers());
        this.vCache = new ArrayList<>(config.numHiddenLayers());
        for (int i = 0; i < config.numHiddenLayers(); i++) {
            kCache.add(new ArrayList<>());
            vCache.add(new ArrayList<>());
        }
    }

    /**
     * Returns the telemetry trace of live Prefill &rarr; Decode auto-morphing transitions.
     *
     * @return unmodifiable list of morphing events
     */
    public List<String> morphTrace() {
        return Collections.unmodifiableList(morphTrace);
    }

    /**
     * Resets the KV-cache and position pointer to start a new sequence.
     */
    public void reset() {
        for (int i = 0; i < config.numHiddenLayers(); i++) {
            kCache.get(i).clear();
            vCache.get(i).clear();
        }
        curPos = 0;
        morphTrace.clear();
    }

    /**
     * Current token position in the sequence.
     *
     * @return token position index
     */
    public int currentPosition() {
        return curPos;
    }

    /**
     * Runs the prefill forward pass on a prompt sequence of token IDs.
     *
     * @param promptTokens list of input token IDs
     * @return predicted next token ID
     */
    public int prefill(List<Integer> promptTokens) {
        Objects.requireNonNull(promptTokens, "promptTokens");
        if (promptTokens.isEmpty()) {
            throw new IllegalArgumentException("Prompt tokens must not be empty");
        }

        int seqLen = promptTokens.size();

        // 1. Telemetry & Live Prefill Auto-Morphing
        if (collective != null) {
            long totalUnits = (long) seqLen * config.hiddenSize();
            double flopsPerUnit = 2.0 * config.intermediateSize();
            long bytesPerUnit = 4L * config.hiddenSize();
            long syncBytes = (long) seqLen * config.hiddenSize() * 4L;
            WorkloadSpec tpSpec = WorkloadSpec.prefill(totalUnits, flopsPerUnit, bytesPerUnit, syncBytes);

            long kvBytesPerToken = 2L * config.numHiddenLayers() * config.numKeyValueHeads() * config.headDim() * 2L;
            double attnFlops = 4.0 * config.numAttentionHeads() * config.headDim();
            SplitKvSpec kvSpec = new SplitKvSpec(seqLen, kvBytesPerToken, attnFlops, 1024L, 100_000_000L, 0.85);

            RolePlan rolePlan = collective.planRoles(tpSpec, kvSpec);
            morphTrace.add("PREFILL: S=" + seqLen + " regime=" + rolePlan.activeRegime()
                    + " reclaimedGpus=" + rolePlan.reclaimedGpuCount());
        }

        int[] tokenArray = new int[seqLen];
        for (int i = 0; i < seqLen; i++) tokenArray[i] = promptTokens.get(i);

        try (Tensor tokens = Tensor.fromArray(tokenArray, seqLen);
             Tensor x = weights.embedTokens().embedding(tokens)) {

            Tensor h = x;
            boolean hOwned = false;

            for (int l = 0; l < config.numHiddenLayers(); l++) {
                Tensor nextH = forwardLayerPrefill(l, h, seqLen);
                if (hOwned) {
                    h.close();
                }
                h = nextH;
                hOwned = true;
            }

            try (Tensor finalNorm = h.rmsNorm(weights.finalNorm(), config.rmsNormEps());
                 Tensor logits = Nn.linearTransposed(finalNorm, weights.lmHead(), null)) {

                // Read last token logits: logits[seqLen - 1]
                try (Tensor lastLogits = logits.slice(hc4j.Slice.at(seqLen - 1));
                     Tensor bestToken = lastLogits.argmaxRows()) {
                    curPos += seqLen;
                    return bestToken.itemInt();
                }
            } finally {
                if (hOwned) {
                    h.close();
                }
            }
        }
    }

    /**
     * Runs a single auto-regressive decode step (seqLen = 1).
     *
     * @param tokenId token ID of the previous step
     * @return predicted next token ID
     */
    public int decodeStep(int tokenId) {
        // 1. Telemetry & Live Decode Auto-Morphing
        if (collective != null) {
            long totalUnits = config.hiddenSize();
            double flopsPerUnit = 2.0 * config.intermediateSize();
            long bytesPerUnit = 4L * config.hiddenSize();
            long syncBytes = (long) config.hiddenSize() * 4L;
            WorkloadSpec tpSpec = WorkloadSpec.decode(totalUnits, flopsPerUnit, bytesPerUnit, syncBytes);

            int totalKvTokens = curPos + 1;
            long kvBytesPerToken = 2L * config.numHiddenLayers() * config.numKeyValueHeads() * config.headDim() * 2L;
            double attnFlops = 4.0 * config.numAttentionHeads() * config.headDim();
            SplitKvSpec kvSpec = new SplitKvSpec(totalKvTokens, kvBytesPerToken, attnFlops, 1024L, 100_000_000L, 0.85);

            RolePlan rolePlan = collective.planRoles(tpSpec, kvSpec);
            morphTrace.add("DECODE: S=1 pos=" + curPos + " regime=" + rolePlan.activeRegime()
                    + " reclaimedGpus=" + rolePlan.reclaimedGpuCount());
        }

        try (Tensor token = Tensor.fromArray(new int[]{tokenId}, 1);
             Tensor x = weights.embedTokens().embedding(token)) {

            Tensor h = x;
            boolean hOwned = false;

            for (int l = 0; l < config.numHiddenLayers(); l++) {
                Tensor nextH = forwardLayerDecode(l, h);
                if (hOwned) {
                    h.close();
                }
                h = nextH;
                hOwned = true;
            }

            try (Tensor finalNorm = h.rmsNorm(weights.finalNorm(), config.rmsNormEps());
                 Tensor logits = Nn.linearTransposed(finalNorm, weights.lmHead(), null);
                 Tensor bestToken = logits.argmaxRows()) {
                curPos += 1;
                return bestToken.itemInt();
            } finally {
                if (hOwned) {
                    h.close();
                }
            }
        }
    }

    /**
     * Greedy auto-regressive text generation.
     *
     * @param promptTokens prompt token IDs
     * @param maxNewTokens maximum number of new tokens to generate
     * @return full sequence including prompt and generated tokens
     */
    public List<Integer> generate(List<Integer> promptTokens, int maxNewTokens) {
        List<Integer> result = new ArrayList<>(promptTokens);
        int nextToken = prefill(promptTokens);
        result.add(nextToken);

        for (int i = 1; i < maxNewTokens; i++) {
            nextToken = decodeStep(nextToken);
            result.add(nextToken);
        }
        return result;
    }

    // --- Internal Layer Forward Passes ---

    private Tensor forwardLayerPrefill(int layerIdx, Tensor x, int seqLen) {
        LlamaWeights.LayerWeights lw = weights.layer(layerIdx);
        float eps = config.rmsNormEps();
        int headDim = config.headDim();
        int numHeads = config.numAttentionHeads();
        int numKvHeads = config.numKeyValueHeads();
        int gqaRatio = config.gqaRatio();
        float scale = (float) (1.0 / Math.sqrt(headDim));

        try (Tensor normX = x.rmsNorm(lw.inputNorm(), eps);
             Tensor qFlat = Nn.linearTransposed(normX, lw.qProj(), lw.qBias());
             Tensor kFlat = Nn.linearTransposed(normX, lw.kProj(), lw.kBias());
             Tensor vFlat = Nn.linearTransposed(normX, lw.vProj(), lw.vBias());
             Tensor qReshaped = qFlat.reshape(seqLen, numHeads, headDim);
             Tensor kReshaped = kFlat.reshape(seqLen, numKvHeads, headDim);
             Tensor qRoPE = qReshaped.rope(curPos, config.ropeTheta());
             Tensor kRoPE = kReshaped.rope(curPos, config.ropeTheta())) {

            // Cache Key & Value
            float[] kData = kRoPE.toFloatArray();
            float[] vData = vFlat.toFloatArray();
            int kvDim = config.kvDim();
            for (int s = 0; s < seqLen; s++) {
                float[] kTok = new float[kvDim];
                float[] vTok = new float[kvDim];
                System.arraycopy(kData, s * kvDim, kTok, 0, kvDim);
                System.arraycopy(vData, s * kvDim, vTok, 0, kvDim);
                kCache.get(layerIdx).add(kTok);
                vCache.get(layerIdx).add(vTok);
            }

            // Multi-Head Causal Attention with GQA
            float[] qArr = qRoPE.toFloatArray();
            float[] attnOutArr = new float[seqLen * config.hiddenSize()];

            for (int h = 0; h < numHeads; h++) {
                int kvH = h / gqaRatio;
                for (int s = 0; s < seqLen; s++) {
                    int totalContext = curPos + s + 1;
                    float[] scores = new float[totalContext];
                    float maxScore = -Float.MAX_VALUE;

                    for (int t = 0; t < totalContext; t++) {
                        // Causal mask: only attend to tokens <= curPos + s
                        float[] kTok = kCache.get(layerIdx).get(t);
                        float dot = 0.0f;
                        int qOffset = (s * numHeads + h) * headDim;
                        int kOffset = kvH * headDim;
                        for (int d = 0; d < headDim; d++) {
                            dot += qArr[qOffset + d] * kTok[kOffset + d];
                        }
                        scores[t] = dot * scale;
                        if (scores[t] > maxScore) maxScore = scores[t];
                    }

                    // Softmax
                    float sumExp = 0.0f;
                    for (int t = 0; t < totalContext; t++) {
                        scores[t] = (float) Math.exp(scores[t] - maxScore);
                        sumExp += scores[t];
                    }
                    float invSum = 1.0f / sumExp;
                    for (int t = 0; t < totalContext; t++) {
                        scores[t] *= invSum;
                    }

                    // Weighted sum of values
                    int outOffset = s * config.hiddenSize() + h * headDim;
                    for (int t = 0; t < totalContext; t++) {
                        float weight = scores[t];
                        float[] vTok = vCache.get(layerIdx).get(t);
                        int vOffset = kvH * headDim;
                        for (int d = 0; d < headDim; d++) {
                            attnOutArr[outOffset + d] += weight * vTok[vOffset + d];
                        }
                    }
                }
            }

            // Attention output projection & residual
            Tensor outAttn;
            try (Tensor attnTensor = Tensor.fromArray(attnOutArr, seqLen, config.hiddenSize());
                 Tensor projOut = Nn.linearTransposed(attnTensor, lw.oProj(), null)) {
                outAttn = x.add(projOut);
            }

            // SwiGLU MLP & residual
            try (Tensor normX2 = outAttn.rmsNorm(lw.postAttentionNorm(), eps);
                 Tensor gateProj = Nn.linearTransposed(normX2, lw.gateProj(), null);
                 Tensor gate = gateProj.silu();
                 Tensor up = Nn.linearTransposed(normX2, lw.upProj(), null);
                 Tensor mlp = gate.mul(up);
                 Tensor down = Nn.linearTransposed(mlp, lw.downProj(), null)) {
                return outAttn.add(down);
            } finally {
                outAttn.close();
            }
        }
    }

    private Tensor forwardLayerDecode(int layerIdx, Tensor x) {
        LlamaWeights.LayerWeights lw = weights.layer(layerIdx);
        float eps = config.rmsNormEps();
        int headDim = config.headDim();
        int numHeads = config.numAttentionHeads();
        int numKvHeads = config.numKeyValueHeads();
        int gqaRatio = config.gqaRatio();
        float scale = (float) (1.0 / Math.sqrt(headDim));

        try (Tensor normX = x.rmsNorm(lw.inputNorm(), eps);
             Tensor qFlat = Nn.linearTransposed(normX, lw.qProj(), lw.qBias());
             Tensor kFlat = Nn.linearTransposed(normX, lw.kProj(), lw.kBias());
             Tensor vFlat = Nn.linearTransposed(normX, lw.vProj(), lw.vBias());
             Tensor qReshaped = qFlat.reshape(1, numHeads, headDim);
             Tensor kReshaped = kFlat.reshape(1, numKvHeads, headDim);
             Tensor qRoPE = qReshaped.rope(curPos, config.ropeTheta());
             Tensor kRoPE = kReshaped.rope(curPos, config.ropeTheta())) {

            // Store single token K & V
            float[] kData = kRoPE.toFloatArray();
            float[] vData = vFlat.toFloatArray();
            kCache.get(layerIdx).add(kData);
            vCache.get(layerIdx).add(vData);

            int totalContext = kCache.get(layerIdx).size();
            float[] qArr = qRoPE.toFloatArray();
            float[] attnOutArr = new float[config.hiddenSize()];

            for (int h = 0; h < numHeads; h++) {
                int kvH = h / gqaRatio;
                float[] scores = new float[totalContext];
                float maxScore = -Float.MAX_VALUE;

                for (int t = 0; t < totalContext; t++) {
                    float[] kTok = kCache.get(layerIdx).get(t);
                    float dot = 0.0f;
                    int qOffset = h * headDim;
                    int kOffset = kvH * headDim;
                    for (int d = 0; d < headDim; d++) {
                        dot += qArr[qOffset + d] * kTok[kOffset + d];
                    }
                    scores[t] = dot * scale;
                    if (scores[t] > maxScore) maxScore = scores[t];
                }

                // Softmax
                float sumExp = 0.0f;
                for (int t = 0; t < totalContext; t++) {
                    scores[t] = (float) Math.exp(scores[t] - maxScore);
                    sumExp += scores[t];
                }
                float invSum = 1.0f / sumExp;
                for (int t = 0; t < totalContext; t++) {
                    scores[t] *= invSum;
                }

                // Weighted sum of values
                int outOffset = h * headDim;
                for (int t = 0; t < totalContext; t++) {
                    float weight = scores[t];
                    float[] vTok = vCache.get(layerIdx).get(t);
                    int vOffset = kvH * headDim;
                    for (int d = 0; d < headDim; d++) {
                        attnOutArr[outOffset + d] += weight * vTok[vOffset + d];
                    }
                }
            }

            // Output projection & residual
            Tensor outAttn;
            try (Tensor attnTensor = Tensor.fromArray(attnOutArr, 1, config.hiddenSize());
                 Tensor projOut = Nn.linearTransposed(attnTensor, lw.oProj(), null)) {
                outAttn = x.add(projOut);
            }

            // SwiGLU MLP & residual
            try (Tensor normX2 = outAttn.rmsNorm(lw.postAttentionNorm(), eps);
                 Tensor gateProj = Nn.linearTransposed(normX2, lw.gateProj(), null);
                 Tensor gate = gateProj.silu();
                 Tensor up = Nn.linearTransposed(normX2, lw.upProj(), null);
                 Tensor mlp = gate.mul(up);
                 Tensor down = Nn.linearTransposed(mlp, lw.downProj(), null)) {
                return outAttn.add(down);
            } finally {
                outAttn.close();
            }
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            reset();
        }
    }
}
