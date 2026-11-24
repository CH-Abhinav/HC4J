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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

/**
 * Architectural configuration for Llama-3.2, Qwen2.5, and SmolLM2 transformer decoder models.
 *
 * @param vocabSize             vocabulary size
 * @param hiddenSize            hidden embedding dimension
 * @param intermediateSize      feed-forward intermediate dimension
 * @param numHiddenLayers       number of transformer decoder layers
 * @param numAttentionHeads     number of query attention heads
 * @param numKeyValueHeads      number of key/value heads for GQA (Grouped Query Attention)
 * @param rmsNormEps            epsilon for RMSNorm
 * @param ropeTheta             base theta for Rotary Position Embedding
 * @param maxPositionEmbeddings maximum context length
 * @param tieWordEmbeddings     whether lm_head shares weights with embed_tokens
 * @param qkvBias               whether Q, K, V projections include bias terms (e.g. Qwen2.5)
 */
public record LlamaConfig(
        int vocabSize,
        int hiddenSize,
        int intermediateSize,
        int numHiddenLayers,
        int numAttentionHeads,
        int numKeyValueHeads,
        float rmsNormEps,
        float ropeTheta,
        int maxPositionEmbeddings,
        boolean tieWordEmbeddings,
        boolean qkvBias
) {

    /**
     * Dimension of each individual attention head.
     *
     * @return head dimension
     */
    public int headDim() {
        return hiddenSize / numAttentionHeads;
    }

    /**
     * Key/value projection dimension across all KV heads.
     *
     * @return key/value dimension
     */
    public int kvDim() {
        return numKeyValueHeads * headDim();
    }

    /**
     * Number of query heads sharing each key/value head in Grouped Query Attention.
     *
     * @return GQA group size
     */
    public int gqaRatio() {
        return numAttentionHeads / numKeyValueHeads;
    }

    /**
     * Parses a {@link LlamaConfig} from a HuggingFace {@code config.json} string.
     *
     * @param json config JSON string
     * @return parsed config
     */
    public static LlamaConfig fromJson(String json) {
        Objects.requireNonNull(json, "json");
        Map<String, Object> map = MinimalJson.parseObject(json);

        int vocabSize = getInt(map, "vocab_size", 32000);
        int hiddenSize = getInt(map, "hidden_size", 4096);
        int intermediateSize = getInt(map, "intermediate_size", 11008);
        int numHiddenLayers = getInt(map, "num_hidden_layers", 32);
        int numAttentionHeads = getInt(map, "num_attention_heads", 32);
        int numKeyValueHeads = getInt(map, "num_key_value_heads", numAttentionHeads);
        float rmsNormEps = getFloat(map, "rms_norm_eps", 1e-5f);
        float ropeTheta = getFloat(map, "rope_theta", 10000.0f);
        int maxPositionEmbeddings = getInt(map, "max_position_embeddings", 2048);
        boolean tieWordEmbeddings = getBoolean(map, "tie_word_embeddings", false);

        // Qwen2.5 uses attention_bias=true or qkv_bias=true
        boolean qkvBias = getBoolean(map, "attention_bias", false) || getBoolean(map, "qkv_bias", false);

        return new LlamaConfig(
                vocabSize,
                hiddenSize,
                intermediateSize,
                numHiddenLayers,
                numAttentionHeads,
                numKeyValueHeads,
                rmsNormEps,
                ropeTheta,
                maxPositionEmbeddings,
                tieWordEmbeddings,
                qkvBias
        );
    }

    /**
     * Loads a {@link LlamaConfig} from a {@code config.json} file.
     *
     * @param path path to config.json
     * @return parsed config
     * @throws IOException if reading fails
     */
    public static LlamaConfig fromFile(Path path) throws IOException {
        String json = Files.readString(path, StandardCharsets.UTF_8);
        return fromJson(json);
    }

    private static int getInt(Map<String, Object> map, String key, int def) {
        Object val = map.get(key);
        if (val instanceof Number n) return n.intValue();
        return def;
    }

    private static float getFloat(Map<String, Object> map, String key, float def) {
        Object val = map.get(key);
        if (val instanceof Number n) return n.floatValue();
        return def;
    }

    private static boolean getBoolean(Map<String, Object> map, String key, boolean def) {
        Object val = map.get(key);
        if (val instanceof Boolean b) return b;
        return def;
    }
}
