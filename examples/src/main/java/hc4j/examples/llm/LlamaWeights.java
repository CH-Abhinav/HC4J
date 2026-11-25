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
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Container holding all GPU VRAM-resident weights for a Llama / Qwen transformer model.
 */
public final class LlamaWeights implements AutoCloseable {

    /**
     * GPU weights for a single transformer decoder layer.
     *
     * @param inputNorm          input RMSNorm weight [hiddenSize] (f32)
     * @param qProj              query projection weight [hiddenSize, hiddenSize] (f32 or f16)
     * @param kProj              key projection weight [kvDim, hiddenSize] (f32 or f16)
     * @param vProj              value projection weight [kvDim, hiddenSize] (f32 or f16)
     * @param oProj              output projection weight [hiddenSize, hiddenSize] (f32 or f16)
     * @param qBias              optional query projection bias [qDim] (f32, may be null)
     * @param kBias              optional key projection bias [kvDim] (f32, may be null)
     * @param vBias              optional value projection bias [kvDim] (f32, may be null)
     * @param oBias              optional output projection bias [hiddenSize] (f32, may be null)
     * @param postAttentionNorm  post-attention RMSNorm weight [hiddenSize] (f32)
     * @param gateProj           SwiGLU gate projection weight [intermediateSize, hiddenSize] (f32 or f16)
     * @param upProj             SwiGLU up projection weight [intermediateSize, hiddenSize] (f32 or f16)
     * @param downProj           SwiGLU down projection weight [hiddenSize, intermediateSize] (f32 or f16)
     */
    public record LayerWeights(
            Tensor inputNorm,
            Tensor qProj,
            Tensor kProj,
            Tensor vProj,
            Tensor oProj,
            Tensor qBias,
            Tensor kBias,
            Tensor vBias,
            Tensor oBias,
            Tensor postAttentionNorm,
            Tensor gateProj,
            Tensor upProj,
            Tensor downProj
    ) implements AutoCloseable {

        /**
         * Backwards-compatible constructor without {@code o_proj.bias}.
         *
         * @param inputNorm         input RMSNorm weight
         * @param qProj             query projection weight
         * @param kProj             key projection weight
         * @param vProj             value projection weight
         * @param oProj             output projection weight
         * @param qBias             optional query bias
         * @param kBias             optional key bias
         * @param vBias             optional value bias
         * @param postAttentionNorm post-attention RMSNorm weight
         * @param gateProj          SwiGLU gate weight
         * @param upProj            SwiGLU up weight
         * @param downProj          SwiGLU down weight
         */
        public LayerWeights(
                Tensor inputNorm,
                Tensor qProj,
                Tensor kProj,
                Tensor vProj,
                Tensor oProj,
                Tensor qBias,
                Tensor kBias,
                Tensor vBias,
                Tensor postAttentionNorm,
                Tensor gateProj,
                Tensor upProj,
                Tensor downProj
        ) {
            this(inputNorm, qProj, kProj, vProj, oProj, qBias, kBias, vBias, null,
                    postAttentionNorm, gateProj, upProj, downProj);
        }

        @Override
        public void close() {
            closeQuietly(inputNorm);
            closeQuietly(qProj);
            closeQuietly(kProj);
            closeQuietly(vProj);
            closeQuietly(oProj);
            closeQuietly(qBias);
            closeQuietly(kBias);
            closeQuietly(vBias);
            closeQuietly(oBias);
            closeQuietly(postAttentionNorm);
            closeQuietly(gateProj);
            closeQuietly(upProj);
            closeQuietly(downProj);
        }
    }

    private final Tensor embedTokens;
    private final List<LayerWeights> layers;
    private final Tensor finalNorm;
    private final Tensor lmHead;
    private final boolean lmHeadTied;
    private boolean closed = false;

    /**
     * Constructs a loaded {@link LlamaWeights} container.
     *
     * @param embedTokens token embedding table [vocabSize, hiddenSize]
     * @param layers      per-layer transformer weights
     * @param finalNorm   final RMSNorm weight [hiddenSize]
     * @param lmHead      output projection weight [vocabSize, hiddenSize]
     * @param lmHeadTied  whether {@code lmHead} shares the underlying tensor with {@code embedTokens}
     */
    public LlamaWeights(Tensor embedTokens, List<LayerWeights> layers, Tensor finalNorm, Tensor lmHead, boolean lmHeadTied) {
        this.embedTokens = Objects.requireNonNull(embedTokens, "embedTokens");
        this.layers = List.copyOf(layers);
        this.finalNorm = Objects.requireNonNull(finalNorm, "finalNorm");
        this.lmHead = Objects.requireNonNull(lmHead, "lmHead");
        this.lmHeadTied = lmHeadTied;
    }

    /**
     * Returns the token embedding weight tensor.
     *
     * @return embedding tensor
     */
    public Tensor embedTokens() { return embedTokens; }

    /**
     * Returns the immutable list of per-layer weights.
     *
     * @return layer weights list
     */
    public List<LayerWeights> layers() { return layers; }

    /**
     * Returns the weights for decoder layer {@code idx}.
     *
     * @param idx zero-based layer index
     * @return layer weights
     */
    public LayerWeights layer(int idx) { return layers.get(idx); }

    /**
     * Returns the final RMSNorm weight tensor.
     *
     * @return final norm tensor
     */
    public Tensor finalNorm() { return finalNorm; }

    /**
     * Returns the LM head weight tensor.
     *
     * @return LM head tensor
     */
    public Tensor lmHead() { return lmHead; }

    /**
     * Loads weights from a {@link SafetensorsReader} using the specified target precision.
     *
     * @param reader           open reader
     * @param config           model architecture config
     * @param weightPrecision  desired precision for projection weights ({@link DType#f16} or {@link DType#f32})
     * @return loaded weights
     */
    public static LlamaWeights load(SafetensorsReader reader, LlamaConfig config, DType weightPrecision) {
        Objects.requireNonNull(reader, "reader");
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(weightPrecision, "weightPrecision");

        // 1. Embedding table
        String embedName = reader.hasTensor("model.embed_tokens.weight")
                ? "model.embed_tokens.weight"
                : "transformer.wte.weight";
        Tensor embedTokens = reader.loadTensor(embedName, weightPrecision);

        // 2. Transformer layers
        List<LayerWeights> layersList = new ArrayList<>(config.numHiddenLayers());
        try {
            for (int i = 0; i < config.numHiddenLayers(); i++) {
                String prefix = "model.layers." + i + ".";

                Tensor inputNorm = reader.loadTensor(prefix + "input_layernorm.weight", DType.f32);
                Tensor qProj = reader.loadTensor(prefix + "self_attn.q_proj.weight", weightPrecision);
                Tensor kProj = reader.loadTensor(prefix + "self_attn.k_proj.weight", weightPrecision);
                Tensor vProj = reader.loadTensor(prefix + "self_attn.v_proj.weight", weightPrecision);
                Tensor oProj = reader.loadTensor(prefix + "self_attn.o_proj.weight", weightPrecision);

                // Auto-detect attention biases directly from safetensors (Qwen2.5 omits "attention_bias" in config.json)
                Tensor qBias = loadOptionalTensor(reader, prefix + "self_attn.q_proj.bias", DType.f32);
                Tensor kBias = loadOptionalTensor(reader, prefix + "self_attn.k_proj.bias", DType.f32);
                Tensor vBias = loadOptionalTensor(reader, prefix + "self_attn.v_proj.bias", DType.f32);
                Tensor oBias = loadOptionalTensor(reader, prefix + "self_attn.o_proj.bias", DType.f32);

                Tensor postAttnNorm = reader.loadTensor(prefix + "post_attention_layernorm.weight", DType.f32);
                Tensor gateProj = reader.loadTensor(prefix + "mlp.gate_proj.weight", weightPrecision);
                Tensor upProj = reader.loadTensor(prefix + "mlp.up_proj.weight", weightPrecision);
                Tensor downProj = reader.loadTensor(prefix + "mlp.down_proj.weight", weightPrecision);

                layersList.add(new LayerWeights(
                        inputNorm, qProj, kProj, vProj, oProj,
                        qBias, kBias, vBias, oBias,
                        postAttnNorm, gateProj, upProj, downProj
                ));
            }

            // 3. Final norm
            String normName = reader.hasTensor("model.norm.weight") ? "model.norm.weight" : "transformer.ln_f.weight";
            Tensor finalNorm = reader.loadTensor(normName, DType.f32);

            // 4. LM Head (tied or separate)
            Tensor lmHead;
            boolean lmHeadTied = config.tieWordEmbeddings() || !reader.hasTensor("lm_head.weight");
            if (lmHeadTied) {
                lmHead = embedTokens;
            } else {
                lmHead = reader.loadTensor("lm_head.weight", weightPrecision);
            }

            return new LlamaWeights(embedTokens, layersList, finalNorm, lmHead, lmHeadTied);
        } catch (Exception e) {
            closeQuietly(embedTokens);
            for (LayerWeights lw : layersList) lw.close();
            throw e;
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            closeQuietly(embedTokens);
            for (LayerWeights layer : layers) {
                layer.close();
            }
            closeQuietly(finalNorm);
            if (!lmHeadTied) {
                closeQuietly(lmHead);
            }
        }
    }

    private static Tensor loadOptionalTensor(SafetensorsReader reader, String name, DType targetDType) {
        return reader.hasTensor(name) ? reader.loadTensor(name, targetDType) : null;
    }

    private static void closeQuietly(Tensor t) {
        if (t != null && !t.isClosed()) {
            try {
                t.close();
            } catch (Exception ignored) {}
        }
    }
}
