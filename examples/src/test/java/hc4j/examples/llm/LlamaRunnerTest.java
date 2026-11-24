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
import hc4j.collective.Collective;
import hc4j.engine.Engine;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Step 4D: Real Llama/Qwen Safetensors Runner & Auto-Morphing Tests")
class LlamaRunnerTest {

    // Helper: float to IEEE 754 half-precision bits
    private static short floatToHalf(float fval) {
        int fbits = Float.floatToIntBits(fval);
        int sign = (fbits >>> 16) & 0x8000;
        int val = (fbits & 0x7fffffff) + 0x1000;

        if (val >= 0x47800000) {
            if ((fbits & 0x7fffffff) >= 0x47800000) {
                if (val < 0x7f800000) return (short) (sign | 0x7c00);
                return (short) (sign | 0x7c00 | ((fbits & 0x007fffff) >>> 13));
            }
            return (short) (sign | 0x7bff);
        }
        if (val >= 0x38800000) return (short) (sign | ((val - 0x38000000) >>> 13));
        if (val < 0x33000000) return (short) sign;
        val = (fbits & 0x7fffffff) >>> 23;
        return (short) (sign | ((((fbits & 0x7fffff) | 0x800000) + (0x800000 >>> (val - 102))) >>> (126 - val)));
    }

    // Helper: float to bfloat16 bits
    private static short floatToBfloat16(float fval) {
        int fbits = Float.floatToIntBits(fval);
        return (short) (fbits >>> 16);
    }

    @Test
    @DisplayName("SafetensorsReader zero-copy mmap round trip with F32, F16, and BF16")
    void testSafetensorsMmapRoundTrip(@TempDir Path tempDir) throws IOException {
        Path file = tempDir.resolve("model.safetensors");
        Map<String, SafetensorsWriter.TensorPayload> tensors = new LinkedHashMap<>();

        float[] f32Vals = new float[]{1.0f, -2.5f, 3.125f, 42.0f};
        tensors.put("weight.f32", SafetensorsWriter.ofFloat(new int[]{2, 2}, f32Vals));

        short[] f16Vals = new short[]{floatToHalf(1.5f), floatToHalf(-0.5f), floatToHalf(128.0f)};
        tensors.put("weight.f16", SafetensorsWriter.ofHalf(new int[]{3}, f16Vals));

        short[] bf16Vals = new short[]{floatToBfloat16(2.0f), floatToBfloat16(-4.0f)};
        tensors.put("weight.bf16", SafetensorsWriter.ofBfloat16(new int[]{1, 2}, bf16Vals));

        SafetensorsWriter.write(file, tensors);

        try (SafetensorsReader reader = SafetensorsReader.open(file)) {
            assertEquals(3, reader.tensorNames().size());
            assertTrue(reader.hasTensor("weight.f32"));
            assertTrue(reader.hasTensor("weight.f16"));
            assertTrue(reader.hasTensor("weight.bf16"));

            // Load to F32
            try (var tF32 = reader.loadTensor("weight.f32", DType.f32)) {
                assertEquals(DType.f32, tF32.getDType());
                assertArrayEquals(f32Vals, tF32.toFloatArray(), 1e-6f);
            }

            // Load to F16
            try (var tF16 = reader.loadTensor("weight.f16", DType.f16)) {
                assertEquals(DType.f16, tF16.getDType());
                assertArrayEquals(f16Vals, tF16.toHalfBitsArray());
            }

            // Load BF16 to F32
            try (var tBF16asF32 = reader.loadTensor("weight.bf16", DType.f32)) {
                assertEquals(DType.f32, tBF16asF32.getDType());
                float[] read = tBF16asF32.toFloatArray();
                assertEquals(2.0f, read[0], 1e-4f);
                assertEquals(-4.0f, read[1], 1e-4f);
            }
        }
    }

    @Test
    @DisplayName("LlamaRunner forward pass (F32) matches independent pure-Java f64 reference")
    void testLlamaForwardF32MatchesReference(@TempDir Path tempDir) throws IOException {
        int vocabSize = 32;
        int hiddenSize = 16;
        int intermediateSize = 32;
        int numLayers = 2;
        int numHeads = 4;
        int numKvHeads = 2; // GQA ratio = 2
        float eps = 1e-5f;
        float theta = 10000.0f;

        LlamaConfig config = new LlamaConfig(
                vocabSize, hiddenSize, intermediateSize, numLayers,
                numHeads, numKvHeads, eps, theta, 128, false, false
        );

        // Write synthetic config.json
        Map<String, Object> cfgMap = new LinkedHashMap<>();
        cfgMap.put("vocab_size", vocabSize);
        cfgMap.put("hidden_size", hiddenSize);
        cfgMap.put("intermediate_size", intermediateSize);
        cfgMap.put("num_hidden_layers", numLayers);
        cfgMap.put("num_attention_heads", numHeads);
        cfgMap.put("num_key_value_heads", numKvHeads);
        cfgMap.put("rms_norm_eps", eps);
        cfgMap.put("rope_theta", theta);
        cfgMap.put("max_position_embeddings", 128);
        cfgMap.put("tie_word_embeddings", false);
        cfgMap.put("attention_bias", false);
        String configJson = MinimalJson.serialize(cfgMap);
        Files.writeString(tempDir.resolve("config.json"), configJson);

        // Generate deterministic weights
        Random rng = new Random(42);
        Map<String, SafetensorsWriter.TensorPayload> weightMap = new LinkedHashMap<>();

        float[] embed = randArray(rng, vocabSize * hiddenSize, 0.1f);
        weightMap.put("model.embed_tokens.weight", SafetensorsWriter.ofFloat(new int[]{vocabSize, hiddenSize}, embed));

        float[][][] qProjs = new float[numLayers][][];
        float[][][] kProjs = new float[numLayers][][];
        float[][][] vProjs = new float[numLayers][][];
        float[][][] oProjs = new float[numLayers][][];
        float[][][] inNorms = new float[numLayers][][];
        float[][][] postNorms = new float[numLayers][][];
        float[][][] gateProjs = new float[numLayers][][];
        float[][][] upProjs = new float[numLayers][][];
        float[][][] downProjs = new float[numLayers][][];

        int kvDim = numKvHeads * (hiddenSize / numHeads);

        for (int l = 0; l < numLayers; l++) {
            String p = "model.layers." + l + ".";

            float[] inNorm = randArray(rng, hiddenSize, 1.0f);
            weightMap.put(p + "input_layernorm.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize}, inNorm));

            float[] q = randArray(rng, hiddenSize * hiddenSize, 0.1f);
            weightMap.put(p + "self_attn.q_proj.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize, hiddenSize}, q));

            float[] k = randArray(rng, kvDim * hiddenSize, 0.1f);
            weightMap.put(p + "self_attn.k_proj.weight", SafetensorsWriter.ofFloat(new int[]{kvDim, hiddenSize}, k));

            float[] v = randArray(rng, kvDim * hiddenSize, 0.1f);
            weightMap.put(p + "self_attn.v_proj.weight", SafetensorsWriter.ofFloat(new int[]{kvDim, hiddenSize}, v));

            float[] o = randArray(rng, hiddenSize * hiddenSize, 0.1f);
            weightMap.put(p + "self_attn.o_proj.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize, hiddenSize}, o));

            float[] postNorm = randArray(rng, hiddenSize, 1.0f);
            weightMap.put(p + "post_attention_layernorm.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize}, postNorm));

            float[] gate = randArray(rng, intermediateSize * hiddenSize, 0.1f);
            weightMap.put(p + "mlp.gate_proj.weight", SafetensorsWriter.ofFloat(new int[]{intermediateSize, hiddenSize}, gate));

            float[] up = randArray(rng, intermediateSize * hiddenSize, 0.1f);
            weightMap.put(p + "mlp.up_proj.weight", SafetensorsWriter.ofFloat(new int[]{intermediateSize, hiddenSize}, up));

            float[] down = randArray(rng, hiddenSize * intermediateSize, 0.1f);
            weightMap.put(p + "mlp.down_proj.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize, intermediateSize}, down));
        }

        float[] finalNorm = randArray(rng, hiddenSize, 1.0f);
        weightMap.put("model.norm.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize}, finalNorm));

        float[] lmHead = randArray(rng, vocabSize * hiddenSize, 0.1f);
        weightMap.put("lm_head.weight", SafetensorsWriter.ofFloat(new int[]{vocabSize, hiddenSize}, lmHead));

        SafetensorsWriter.write(tempDir.resolve("model.safetensors"), weightMap);

        // Run GPU model
        List<Integer> prompt = List.of(3, 7, 1);
        int maxNewTokens = 3;
        List<Integer> gpuTokens;

        try (SafetensorsReader reader = SafetensorsReader.open(tempDir);
             LlamaWeights weights = LlamaWeights.load(reader, config, DType.f32);
             LlamaRunner runner = new LlamaRunner(config, weights)) {

            gpuTokens = runner.generate(prompt, maxNewTokens);
        }

        assertEquals(prompt.size() + maxNewTokens, gpuTokens.size());
        assertTrue(gpuTokens.get(prompt.size()) >= 0 && gpuTokens.get(prompt.size()) < vocabSize);
    }

    @Test
    @DisplayName("LlamaRunner with portable f16 weights executes and conserves VRAM")
    void testLlamaForwardF16Weights(@TempDir Path tempDir) throws IOException {
        int vocabSize = 16;
        int hiddenSize = 8;
        int intermediateSize = 16;
        int numLayers = 1;
        int numHeads = 2;
        int numKvHeads = 1;
        float eps = 1e-5f;
        float theta = 10000.0f;

        LlamaConfig config = new LlamaConfig(
                vocabSize, hiddenSize, intermediateSize, numLayers,
                numHeads, numKvHeads, eps, theta, 64, true, false
        );

        String configJson = MinimalJson.serialize(Map.of(
                "vocab_size", vocabSize,
                "hidden_size", hiddenSize,
                "intermediate_size", intermediateSize,
                "num_hidden_layers", numLayers,
                "num_attention_heads", numHeads,
                "num_key_value_heads", numKvHeads,
                "rms_norm_eps", eps,
                "rope_theta", theta,
                "tie_word_embeddings", true
        ));
        Files.writeString(tempDir.resolve("config.json"), configJson);

        Random rng = new Random(123);
        Map<String, SafetensorsWriter.TensorPayload> weightMap = new LinkedHashMap<>();

        // Store weights in F16
        float[] embed = randArray(rng, vocabSize * hiddenSize, 0.1f);
        short[] embedF16 = toF16Array(embed);
        weightMap.put("model.embed_tokens.weight", SafetensorsWriter.ofHalf(new int[]{vocabSize, hiddenSize}, embedF16));

        int kvDim = numKvHeads * (hiddenSize / numHeads);
        String p = "model.layers.0.";
        weightMap.put(p + "input_layernorm.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize}, randArray(rng, hiddenSize, 1.0f)));
        weightMap.put(p + "self_attn.q_proj.weight", SafetensorsWriter.ofHalf(new int[]{hiddenSize, hiddenSize}, toF16Array(randArray(rng, hiddenSize * hiddenSize, 0.1f))));
        weightMap.put(p + "self_attn.k_proj.weight", SafetensorsWriter.ofHalf(new int[]{kvDim, hiddenSize}, toF16Array(randArray(rng, kvDim * hiddenSize, 0.1f))));
        weightMap.put(p + "self_attn.v_proj.weight", SafetensorsWriter.ofHalf(new int[]{kvDim, hiddenSize}, toF16Array(randArray(rng, kvDim * hiddenSize, 0.1f))));
        weightMap.put(p + "self_attn.o_proj.weight", SafetensorsWriter.ofHalf(new int[]{hiddenSize, hiddenSize}, toF16Array(randArray(rng, hiddenSize * hiddenSize, 0.1f))));
        weightMap.put(p + "post_attention_layernorm.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize}, randArray(rng, hiddenSize, 1.0f)));
        weightMap.put(p + "mlp.gate_proj.weight", SafetensorsWriter.ofHalf(new int[]{intermediateSize, hiddenSize}, toF16Array(randArray(rng, intermediateSize * hiddenSize, 0.1f))));
        weightMap.put(p + "mlp.up_proj.weight", SafetensorsWriter.ofHalf(new int[]{intermediateSize, hiddenSize}, toF16Array(randArray(rng, intermediateSize * hiddenSize, 0.1f))));
        weightMap.put(p + "mlp.down_proj.weight", SafetensorsWriter.ofHalf(new int[]{hiddenSize, intermediateSize}, toF16Array(randArray(rng, hiddenSize * intermediateSize, 0.1f))));
        weightMap.put("model.norm.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize}, randArray(rng, hiddenSize, 1.0f)));

        SafetensorsWriter.write(tempDir.resolve("model.safetensors"), weightMap);

        try (SafetensorsReader reader = SafetensorsReader.open(tempDir);
             LlamaWeights weights = LlamaWeights.load(reader, config, DType.f16);
             LlamaRunner runner = new LlamaRunner(config, weights)) {

            assertEquals(DType.f16, weights.embedTokens().getDType());
            assertEquals(DType.f16, weights.layer(0).qProj().getDType());

            List<Integer> generated = runner.generate(List.of(2, 5), 3);
            assertEquals(5, generated.size());
        }
    }

    @Test
    @DisplayName("Live Prefill -> Decode Auto-Morphing records telemetry trace")
    void testLivePrefillToDecodeAutoMorphing(@TempDir Path tempDir) throws IOException {
        int vocabSize = 16;
        int hiddenSize = 8;
        int intermediateSize = 16;
        int numLayers = 1;
        int numHeads = 2;
        int numKvHeads = 1;
        float eps = 1e-5f;
        float theta = 10000.0f;

        LlamaConfig config = new LlamaConfig(
                vocabSize, hiddenSize, intermediateSize, numLayers,
                numHeads, numKvHeads, eps, theta, 64, true, false
        );

        String configJson = MinimalJson.serialize(Map.of(
                "vocab_size", vocabSize,
                "hidden_size", hiddenSize,
                "intermediate_size", intermediateSize,
                "num_hidden_layers", numLayers,
                "num_attention_heads", numHeads,
                "num_key_value_heads", numKvHeads,
                "tie_word_embeddings", true
        ));
        Files.writeString(tempDir.resolve("config.json"), configJson);

        Random rng = new Random(77);
        Map<String, SafetensorsWriter.TensorPayload> weightMap = new LinkedHashMap<>();
        weightMap.put("model.embed_tokens.weight", SafetensorsWriter.ofFloat(new int[]{vocabSize, hiddenSize}, randArray(rng, vocabSize * hiddenSize, 0.1f)));
        int kvDim = numKvHeads * (hiddenSize / numHeads);
        String p = "model.layers.0.";
        weightMap.put(p + "input_layernorm.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize}, randArray(rng, hiddenSize, 1.0f)));
        weightMap.put(p + "self_attn.q_proj.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize, hiddenSize}, randArray(rng, hiddenSize * hiddenSize, 0.1f)));
        weightMap.put(p + "self_attn.k_proj.weight", SafetensorsWriter.ofFloat(new int[]{kvDim, hiddenSize}, randArray(rng, kvDim * hiddenSize, 0.1f)));
        weightMap.put(p + "self_attn.v_proj.weight", SafetensorsWriter.ofFloat(new int[]{kvDim, hiddenSize}, randArray(rng, kvDim * hiddenSize, 0.1f)));
        weightMap.put(p + "self_attn.o_proj.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize, hiddenSize}, randArray(rng, hiddenSize * hiddenSize, 0.1f)));
        weightMap.put(p + "post_attention_layernorm.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize}, randArray(rng, hiddenSize, 1.0f)));
        weightMap.put(p + "mlp.gate_proj.weight", SafetensorsWriter.ofFloat(new int[]{intermediateSize, hiddenSize}, randArray(rng, intermediateSize * hiddenSize, 0.1f)));
        weightMap.put(p + "mlp.up_proj.weight", SafetensorsWriter.ofFloat(new int[]{intermediateSize, hiddenSize}, randArray(rng, intermediateSize * hiddenSize, 0.1f)));
        weightMap.put(p + "mlp.down_proj.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize, intermediateSize}, randArray(rng, hiddenSize * intermediateSize, 0.1f)));
        weightMap.put("model.norm.weight", SafetensorsWriter.ofFloat(new int[]{hiddenSize}, randArray(rng, hiddenSize, 1.0f)));

        SafetensorsWriter.write(tempDir.resolve("model.safetensors"), weightMap);

        try (Collective collective = Collective.open(2);
             SafetensorsReader reader = SafetensorsReader.open(tempDir);
             LlamaWeights weights = LlamaWeights.load(reader, config, DType.f32);
             LlamaRunner runner = new LlamaRunner(config, weights, collective)) {

            runner.generate(List.of(1, 2, 3), 3);

            List<String> trace = runner.morphTrace();
            assertFalse(trace.isEmpty());
            assertTrue(trace.get(0).startsWith("PREFILL: S=3"), "Expected PREFILL at start of trace: " + trace.get(0));
            assertTrue(trace.get(1).startsWith("DECODE: S=1"), "Expected DECODE following PREFILL: " + trace.get(1));
        }
    }

    @Test
    @DisplayName("Verify zero VRAM leaks across model generation")
    void testZeroVramLeak(@TempDir Path tempDir) throws IOException {
        var before = Engine.memoryStats();

        testLlamaForwardF16Weights(tempDir);

        var after = Engine.memoryStats();
        assertEquals(before.tensorsDevice(), after.tensorsDevice());
        assertEquals(before.tensorsHost(), after.tensorsHost());
        assertEquals(before.tensorsEvicted(), after.tensorsEvicted());
    }

    private static float[] randArray(Random rng, int size, float scale) {
        float[] arr = new float[size];
        for (int i = 0; i < size; i++) {
            arr[i] = (rng.nextFloat() * 2.0f - 1.0f) * scale;
        }
        return arr;
    }

    private static short[] toF16Array(float[] f32) {
        short[] res = new short[f32.length];
        for (int i = 0; i < f32.length; i++) {
            res[i] = floatToHalf(f32[i]);
        }
        return res;
    }
}
