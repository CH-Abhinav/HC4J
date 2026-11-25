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
import hc4j.collective.DeviceProfile;
import hc4j.collective.DeviceRole;
import hc4j.collective.ExecutionReport;
import hc4j.collective.HostCpuProfile;
import hc4j.collective.RolePlan;
import hc4j.collective.ShardAllocation;
import hc4j.collective.SplitKvAssignment;
import hc4j.collective.SplitKvSpec;
import hc4j.collective.WorkloadPlan;
import hc4j.collective.WorkloadPlanner;
import hc4j.collective.WorkloadSpec;
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

@DisplayName("Step 4D-Fix: 100% GPU Llama/Qwen Runner, Split-KV & Tokenizer End-to-End Tests")
class LlamaRunnerTest {

    // --- Independent Pure-Java f64 Transformer Reference ---

    private static final class ReferenceTransformerF64 {
        final LlamaConfig config;
        final float[] embed;
        final float[][] inNorms, qProjs, kProjs, vProjs, oProjs, postNorms, gateProjs, upProjs, downProjs;
        final float[][] qBiases, kBiases, vBiases, oBiases;
        final float[] finalNorm;
        final float[] lmHead;
        final List<List<double[]>> kCache = new ArrayList<>();
        final List<List<double[]>> vCache = new ArrayList<>();
        int curPos = 0;

        ReferenceTransformerF64(
                LlamaConfig config, float[] embed,
                float[][] inNorms, float[][] qProjs, float[][] kProjs, float[][] vProjs, float[][] oProjs,
                float[][] qBiases, float[][] kBiases, float[][] vBiases, float[][] oBiases,
                float[][] postNorms, float[][] gateProjs, float[][] upProjs, float[][] downProjs,
                float[] finalNorm, float[] lmHead
        ) {
            this.config = config;
            this.embed = embed;
            this.inNorms = inNorms;
            this.qProjs = qProjs;
            this.kProjs = kProjs;
            this.vProjs = vProjs;
            this.oProjs = oProjs;
            this.qBiases = qBiases;
            this.kBiases = kBiases;
            this.vBiases = vBiases;
            this.oBiases = oBiases;
            this.postNorms = postNorms;
            this.gateProjs = gateProjs;
            this.upProjs = upProjs;
            this.downProjs = downProjs;
            this.finalNorm = finalNorm;
            this.lmHead = lmHead;
            for (int l = 0; l < config.numHiddenLayers(); l++) {
                kCache.add(new ArrayList<>());
                vCache.add(new ArrayList<>());
            }
        }

        float[] forward(int[] tokens) {
            int seqLen = tokens.length;
            int D = config.hiddenSize();
            int H = config.numAttentionHeads();
            int numHeads = H;
            int Hkv = config.numKeyValueHeads();
            int Dh = config.headDim();
            int kvDim = config.kvDim();
            int groupSize = H / Hkv;
            double scale = 1.0 / Math.sqrt(Dh);

            double[][] hidden = new double[seqLen][D];
            for (int s = 0; s < seqLen; s++) {
                int tok = tokens[s];
                for (int d = 0; d < D; d++) {
                    hidden[s][d] = embed[tok * D + d];
                }
            }

            for (int l = 0; l < config.numHiddenLayers(); l++) {
                double[][] qSeq = new double[seqLen][numHeads * Dh];
                for (int s = 0; s < seqLen; s++) {
                    double[] norm1 = rmsNorm(hidden[s], inNorms[l], config.rmsNormEps());
                    double[] q = matVecT(norm1, qProjs[l], qBiases != null ? qBiases[l] : null, numHeads * Dh, D);
                    double[] k = matVecT(norm1, kProjs[l], kBiases != null ? kBiases[l] : null, kvDim, D);
                    double[] v = matVecT(norm1, vProjs[l], vBiases != null ? vBiases[l] : null, kvDim, D);

                    int pos = curPos + s;
                    applyRope(q, H, Dh, pos, config.ropeTheta());
                    applyRope(k, Hkv, Dh, pos, config.ropeTheta());

                    qSeq[s] = q;
                    kCache.get(l).add(k);
                    vCache.get(l).add(v);
                }

                for (int s = 0; s < seqLen; s++) {
                    int totalContext = curPos + s + 1;
                    double[] attnOut = new double[numHeads * Dh];
                    for (int h = 0; h < H; h++) {
                        int kvH = h / groupSize;
                        double[] scores = new double[totalContext];
                        double maxS = Double.NEGATIVE_INFINITY;
                        for (int t = 0; t < totalContext; t++) {
                            double[] kt = kCache.get(l).get(t);
                            double dot = 0.0;
                            for (int d = 0; d < Dh; d++) {
                                dot += qSeq[s][h * Dh + d] * kt[kvH * Dh + d];
                            }
                            scores[t] = dot * scale;
                            if (scores[t] > maxS) maxS = scores[t];
                        }
                        double sum = 0.0;
                        for (int t = 0; t < totalContext; t++) {
                            scores[t] = Math.exp(scores[t] - maxS);
                            sum += scores[t];
                        }
                        for (int t = 0; t < totalContext; t++) {
                            double w = scores[t] / sum;
                            double[] vt = vCache.get(l).get(t);
                            for (int d = 0; d < Dh; d++) {
                                attnOut[h * Dh + d] += w * vt[kvH * Dh + d];
                            }
                        }
                    }

                    double[] oOut = matVecT(attnOut, oProjs[l], oBiases != null ? oBiases[l] : null, D, numHeads * Dh);
                    for (int d = 0; d < D; d++) {
                        hidden[s][d] += oOut[d];
                    }

                    double[] norm2 = rmsNorm(hidden[s], postNorms[l], config.rmsNormEps());
                    int inter = config.intermediateSize();
                    double[] gate = matVecT(norm2, gateProjs[l], null, inter, D);
                    double[] up = matVecT(norm2, upProjs[l], null, inter, D);
                    double[] swiglu = new double[inter];
                    for (int i = 0; i < inter; i++) {
                        double g = gate[i];
                        double silu = g / (1.0 + Math.exp(-g));
                        swiglu[i] = silu * up[i];
                    }
                    double[] down = matVecT(swiglu, downProjs[l], null, D, inter);
                    for (int d = 0; d < D; d++) {
                        hidden[s][d] += down[d];
                    }
                }
            }

            double[] lastNorm = rmsNorm(hidden[seqLen - 1], finalNorm, config.rmsNormEps());
            double[] logits64 = matVecT(lastNorm, lmHead, null, config.vocabSize(), D);
            curPos += seqLen;

            float[] logits = new float[logits64.length];
            for (int i = 0; i < logits.length; i++) {
                logits[i] = (float) logits64[i];
            }
            return logits;
        }

        private static double[] rmsNorm(double[] x, float[] weight, float eps) {
            double sq = 0.0;
            for (double v : x) sq += v * v;
            double invRms = 1.0 / Math.sqrt(sq / x.length + eps);
            double[] out = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                out[i] = x[i] * invRms * weight[i];
            }
            return out;
        }

        private static double[] matVecT(double[] x, float[] weightOutByIn, float[] bias, int outDim, int inDim) {
            double[] out = new double[outDim];
            for (int o = 0; o < outDim; o++) {
                double sum = bias != null ? bias[o] : 0.0;
                int base = o * inDim;
                for (int i = 0; i < inDim; i++) {
                    sum += x[i] * weightOutByIn[base + i];
                }
                out[o] = sum;
            }
            return out;
        }

        private static void applyRope(double[] vec, int numHeads, int headDim, int pos, float theta) {
            int half = headDim / 2;
            for (int h = 0; h < numHeads; h++) {
                int base = h * headDim;
                for (int d = 0; d < half; d++) {
                    double freq = Math.exp(-Math.log(theta) * (2.0 * d) / headDim);
                    double angle = pos * freq;
                    double c = Math.cos(angle);
                    double s = Math.sin(angle);
                    double x0 = vec[base + d];
                    double x1 = vec[base + d + half];
                    vec[base + d] = x0 * c - x1 * s;
                    vec[base + d + half] = x0 * s + x1 * c;
                }
            }
        }
    }

    private static ReferenceTransformerF64 writeSyntheticModel(
            Path dir, LlamaConfig cfg, DType weightDType, long seed
    ) throws IOException {
        return writeSyntheticModel(dir, cfg, weightDType, seed, cfg.qkvBias(), false, false, 0.05f, false);
    }

    private static ReferenceTransformerF64 writeSyntheticModel(
            Path dir,
            LlamaConfig cfg,
            DType weightDType,
            long seed,
            boolean writeQkvBiases,
            boolean omitAttentionBiasInConfig,
            boolean writeOBias,
            float biasScale,
            boolean omitLmHeadWhenUntied
    ) throws IOException {
        Map<String, Object> cfgMap = new LinkedHashMap<>();
        cfgMap.put("vocab_size", cfg.vocabSize());
        cfgMap.put("hidden_size", cfg.hiddenSize());
        cfgMap.put("intermediate_size", cfg.intermediateSize());
        cfgMap.put("num_hidden_layers", cfg.numHiddenLayers());
        cfgMap.put("num_attention_heads", cfg.numAttentionHeads());
        cfgMap.put("num_key_value_heads", cfg.numKeyValueHeads());
        if (cfg.headDim() != cfg.hiddenSize() / cfg.numAttentionHeads()) {
            cfgMap.put("head_dim", cfg.headDim());
        }
        cfgMap.put("rms_norm_eps", cfg.rmsNormEps());
        cfgMap.put("rope_theta", cfg.ropeTheta());
        cfgMap.put("max_position_embeddings", cfg.maxPositionEmbeddings());
        cfgMap.put("tie_word_embeddings", cfg.tieWordEmbeddings());
        if (!omitAttentionBiasInConfig) {
            cfgMap.put("attention_bias", cfg.qkvBias());
        }
        if (!cfg.eosTokenIds().isEmpty()) {
            cfgMap.put("eos_token_id", cfg.eosTokenIds().size() == 1
                    ? cfg.eosTokenIds().getFirst()
                    : cfg.eosTokenIds());
        }
        Files.writeString(dir.resolve("config.json"), MinimalJson.serialize(cfgMap));

        Random rng = new Random(seed);
        Map<String, SafetensorsWriter.TensorPayload> weightMap = new LinkedHashMap<>();
        boolean useHalf = (weightDType == DType.f16);

        int V = cfg.vocabSize();
        int D = cfg.hiddenSize();
        int qDim = cfg.numAttentionHeads() * cfg.headDim();
        int I = cfg.intermediateSize();
        int L = cfg.numHiddenLayers();
        int kvDim = cfg.kvDim();

        float[] embed = randArray(rng, V * D, 0.08f);
        weightMap.put("model.embed_tokens.weight", makePayload(useHalf, new int[]{V, D}, embed));

        float[][] inNorms = new float[L][];
        float[][] qProjs = new float[L][];
        float[][] kProjs = new float[L][];
        float[][] vProjs = new float[L][];
        float[][] oProjs = new float[L][];
        float[][] qBiases = writeQkvBiases ? new float[L][] : null;
        float[][] kBiases = writeQkvBiases ? new float[L][] : null;
        float[][] vBiases = writeQkvBiases ? new float[L][] : null;
        float[][] oBiases = writeOBias ? new float[L][] : null;
        float[][] postNorms = new float[L][];
        float[][] gateProjs = new float[L][];
        float[][] upProjs = new float[L][];
        float[][] downProjs = new float[L][];

        for (int l = 0; l < L; l++) {
            String p = "model.layers." + l + ".";
            inNorms[l] = positiveRandArray(rng, D, 0.8f, 1.2f);
            qProjs[l] = randArray(rng, qDim * D, 0.08f);
            kProjs[l] = randArray(rng, kvDim * D, 0.08f);
            vProjs[l] = randArray(rng, kvDim * D, 0.08f);
            oProjs[l] = randArray(rng, D * qDim, 0.08f);
            postNorms[l] = positiveRandArray(rng, D, 0.8f, 1.2f);
            gateProjs[l] = randArray(rng, I * D, 0.08f);
            upProjs[l] = randArray(rng, I * D, 0.08f);
            downProjs[l] = randArray(rng, D * I, 0.08f);

            weightMap.put(p + "input_layernorm.weight", SafetensorsWriter.ofFloat(new int[]{D}, inNorms[l]));
            weightMap.put(p + "self_attn.q_proj.weight", makePayload(useHalf, new int[]{qDim, D}, qProjs[l]));
            weightMap.put(p + "self_attn.k_proj.weight", makePayload(useHalf, new int[]{kvDim, D}, kProjs[l]));
            weightMap.put(p + "self_attn.v_proj.weight", makePayload(useHalf, new int[]{kvDim, D}, vProjs[l]));
            weightMap.put(p + "self_attn.o_proj.weight", makePayload(useHalf, new int[]{D, qDim}, oProjs[l]));

            if (writeQkvBiases) {
                qBiases[l] = randArray(rng, qDim, biasScale);
                kBiases[l] = randArray(rng, kvDim, biasScale);
                vBiases[l] = randArray(rng, kvDim, biasScale);
                weightMap.put(p + "self_attn.q_proj.bias", SafetensorsWriter.ofFloat(new int[]{qDim}, qBiases[l]));
                weightMap.put(p + "self_attn.k_proj.bias", SafetensorsWriter.ofFloat(new int[]{kvDim}, kBiases[l]));
                weightMap.put(p + "self_attn.v_proj.bias", SafetensorsWriter.ofFloat(new int[]{kvDim}, vBiases[l]));
            }
            if (writeOBias) {
                oBiases[l] = randArray(rng, D, biasScale);
                weightMap.put(p + "self_attn.o_proj.bias", SafetensorsWriter.ofFloat(new int[]{D}, oBiases[l]));
            }

            weightMap.put(p + "post_attention_layernorm.weight", SafetensorsWriter.ofFloat(new int[]{D}, postNorms[l]));
            weightMap.put(p + "mlp.gate_proj.weight", makePayload(useHalf, new int[]{I, D}, gateProjs[l]));
            weightMap.put(p + "mlp.up_proj.weight", makePayload(useHalf, new int[]{I, D}, upProjs[l]));
            weightMap.put(p + "mlp.down_proj.weight", makePayload(useHalf, new int[]{D, I}, downProjs[l]));
        }

        float[] finalNorm = positiveRandArray(rng, D, 0.8f, 1.2f);
        weightMap.put("model.norm.weight", SafetensorsWriter.ofFloat(new int[]{D}, finalNorm));

        float[] lmHead;
        if (cfg.tieWordEmbeddings() || omitLmHeadWhenUntied) {
            lmHead = embed;
        } else {
            lmHead = randArray(rng, V * D, 0.08f);
            weightMap.put("lm_head.weight", makePayload(useHalf, new int[]{V, D}, lmHead));
        }

        SafetensorsWriter.write(dir.resolve("model.safetensors"), weightMap);
        return new ReferenceTransformerF64(
                cfg, embed, inNorms, qProjs, kProjs, vProjs, oProjs,
                qBiases, kBiases, vBiases, oBiases, postNorms, gateProjs, upProjs, downProjs,
                finalNorm, lmHead
        );
    }

    private static SafetensorsWriter.TensorPayload makePayload(boolean useHalf, int[] shape, float[] vals) {
        return useHalf
                ? SafetensorsWriter.ofHalf(shape, toF16Array(vals))
                : SafetensorsWriter.ofFloat(shape, vals);
    }

    @Test
    @DisplayName("SafetensorsReader zero-copy mmap round trip with F32, F16, BF16, and multi-shard index")
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
            try (var tF32 = reader.loadTensor("weight.f32", DType.f32)) {
                assertEquals(DType.f32, tF32.getDType());
                assertArrayEquals(f32Vals, tF32.toFloatArray(), 1e-6f);
            }
            try (var tF16 = reader.loadTensor("weight.f16", DType.f16)) {
                assertEquals(DType.f16, tF16.getDType());
                assertArrayEquals(f16Vals, tF16.toHalfBitsArray());
            }
            try (var tBF16 = reader.loadTensor("weight.bf16", DType.f32)) {
                float[] read = tBF16.toFloatArray();
                assertEquals(2.0f, read[0], 1e-4f);
                assertEquals(-4.0f, read[1], 1e-4f);
            }
        }
    }

    @Test
    @DisplayName("Hardware Counter Proof: single-GPU forward() executes 0 per-layer D2H transfers")
    void testSingleGpuForwardHasZeroPerLayerReadbacks(@TempDir Path tempDir) throws IOException {
        LlamaConfig config = new LlamaConfig(32, 16, 32, 2, 4, 2, 1e-5f, 10000.0f, 131072, false, true);
        writeSyntheticModel(tempDir, config, DType.f32, 101L);

        int[] prompt = new int[]{3, 7, 1, 14, 22};

        try (SafetensorsReader reader = SafetensorsReader.open(tempDir);
             LlamaWeights weights = LlamaWeights.load(reader, config, DType.f32);
             LlamaRunner runner = new LlamaRunner(config, weights, 64)) {

            assertEquals(64, runner.maxSeqLen());

            // Prefill step: ONLY final [vocabSize] logits downloaded (zero per-layer attention readbacks!)
            long d2hBeforePrefill = Engine.engineStats().deviceToHostBytes();
            float[] prefillLogits = runner.forward(prompt);
            long d2hAfterPrefill = Engine.engineStats().deviceToHostBytes();

            assertEquals(config.vocabSize(), prefillLogits.length);
            assertEquals((long) config.vocabSize() * 4L, d2hAfterPrefill - d2hBeforePrefill,
                    "Expected ONLY the final [vocabSize] logits readback during 2-layer GQA Prefill");

            // Decode step: ONLY final [vocabSize] logits downloaded
            int nextTok = LlamaRunner.argmax(prefillLogits);
            long d2hBeforeDecode = Engine.engineStats().deviceToHostBytes();
            float[] decodeLogits = runner.forward(new int[]{nextTok});
            long d2hAfterDecode = Engine.engineStats().deviceToHostBytes();

            assertEquals(config.vocabSize(), decodeLogits.length);
            assertEquals((long) config.vocabSize() * 4L, d2hAfterDecode - d2hBeforeDecode,
                    "Expected ONLY the final [vocabSize] logits readback during 2-layer GQA Decode");
        }
    }

    @Test
    @DisplayName("LlamaRunner F32 Prefill (S=5) + 4 Decode steps match pure-Java f64 reference <= 1e-4f")
    void testLlamaForwardF32MatchesReference(@TempDir Path tempDir) throws IOException {
        LlamaConfig config = new LlamaConfig(64, 32, 64, 2, 4, 2, 1e-5f, 10000.0f, 128, false, true);
        ReferenceTransformerF64 ref = writeSyntheticModel(tempDir, config, DType.f32, 42L);

        int[] prompt = new int[]{3, 17, 41, 7, 55};

        try (SafetensorsReader reader = SafetensorsReader.open(tempDir);
             LlamaWeights weights = LlamaWeights.load(reader, config, DType.f32);
             LlamaRunner runner = new LlamaRunner(config, weights)) {

            float[] gpuLogits = runner.forward(prompt);
            float[] refLogits = ref.forward(prompt);
            assertArrayEquals(refLogits, gpuLogits, 1e-4f, "Prefill logits diverged from f64 reference");

            int nextTok = LlamaRunner.argmax(gpuLogits);
            assertEquals(LlamaRunner.argmax(refLogits), nextTok);

            for (int step = 0; step < 4; step++) {
                float[] stepGpu = runner.forward(new int[]{nextTok});
                float[] stepRef = ref.forward(new int[]{nextTok});
                assertArrayEquals(stepRef, stepGpu, 1e-4f, "Decode step " + step + " logits diverged");

                int tokGpu = LlamaRunner.argmax(stepGpu);
                int tokRef = LlamaRunner.argmax(stepRef);
                assertEquals(tokRef, tokGpu, "Predicted token diverged at decode step " + step);
                nextTok = tokGpu;
            }
        }
    }

    @Test
    @DisplayName("LlamaRunner with portable f16 weights matches f64 reference <= 2e-2f")
    void testLlamaForwardF16WeightsMatchesReference(@TempDir Path tempDir) throws IOException {
        LlamaConfig config = new LlamaConfig(32, 16, 32, 2, 4, 2, 1e-5f, 10000.0f, 64, true, false);
        ReferenceTransformerF64 ref = writeSyntheticModel(tempDir, config, DType.f16, 123L);

        int[] prompt = new int[]{2, 9, 15, 21, 4};

        try (SafetensorsReader reader = SafetensorsReader.open(tempDir);
             LlamaWeights weights = LlamaWeights.load(reader, config, DType.f16);
             LlamaRunner runner = new LlamaRunner(config, weights)) {

            assertEquals(DType.f16, weights.embedTokens().getDType());
            assertEquals(DType.f16, weights.layer(0).qProj().getDType());

            float[] gpuLogits = runner.forward(prompt);
            float[] refLogits = ref.forward(prompt);
            assertArrayEquals(refLogits, gpuLogits, 2e-2f, "F16 prefill logits exceeded tolerance");

            int nextTok = LlamaRunner.argmax(gpuLogits);
            for (int step = 0; step < 3; step++) {
                float[] stepGpu = runner.forward(new int[]{nextTok});
                float[] stepRef = ref.forward(new int[]{nextTok});
                assertArrayEquals(stepRef, stepGpu, 2e-2f, "F16 decode step " + step + " logits exceeded tolerance");
                nextTok = LlamaRunner.argmax(stepGpu);
            }
        }
    }

    @Test
    @DisplayName("Live Prefill -> Decode Auto-Morphing & DistributedExecutor.splitKvAttention execution proof")
    void testLivePrefillToDecodeAutoMorphingAndSplitKv(@TempDir Path tempDir) throws IOException {
        LlamaConfig config = new LlamaConfig(32, 16, 32, 1, 4, 2, 1e-5f, 10000.0f, 64, false, false);
        ReferenceTransformerF64 ref = writeSyntheticModel(tempDir, config, DType.f32, 77L);

        HostCpuProfile cpuProfile = HostCpuProfile.of("Test-Host-CPU", 4L * 1024 * 1024 * 1024, 150.0, 40.0, 25.0);

        try (Collective collective = Collective.open(2);
             SafetensorsReader reader = SafetensorsReader.open(tempDir);
             LlamaWeights weights = LlamaWeights.load(reader, config, DType.f32)) {

            List<DeviceProfile> devices = collective.deviceProfiles();
            try (LlamaRunner runner = new LlamaRunner(config, weights, collective, cpuProfile, 64, (tp, kv) -> {
                if (tp.totalUnits() > config.hiddenSize()) {
                    // Prefill: delegate directly to production WorkloadPlanner.planRoles(...)
                    return WorkloadPlanner.planRoles(devices, collective.topology(), tp, kv);
                }
                // Decode: force 3 active workers -> GPU 0 (TP), GPU 1 (Reclaimed Split-KV), CPU-in-Place!
                long totalTokens = kv.totalKvTokens();
                long gpu0Tokens = Math.max(1L, totalTokens / 3);
                long gpu1Tokens = Math.max(1L, totalTokens / 3);
                long cpuTokens = totalTokens - gpu0Tokens - gpu1Tokens;
                WorkloadPlan tpPlan = WorkloadPlanner.plan(devices, collective.topology(), tp);
                List<SplitKvAssignment> assignments = List.of(
                        new SplitKvAssignment(0, DeviceRole.TENSOR_PARALLEL, gpu0Tokens, 0.1, 0.0, 0.1, gpu0Tokens * 256),
                        new SplitKvAssignment(1, DeviceRole.SPLIT_KV_RECLAIMED, gpu1Tokens, 0.1, 0.05, 0.15, gpu1Tokens * 256),
                        new SplitKvAssignment(ShardAllocation.HOST_CPU_RANK, DeviceRole.CPU_IN_PLACE,
                                cpuTokens, 0.2, 0.05, 0.25, cpuTokens * 256)
                );
                return new RolePlan(tpPlan, assignments, 0.5, 0.25, totalTokens, 1, cpuTokens > 0,
                        "ReclaimedSplitKV+CPU");
            })) {
                int[] prompt = new int[]{1, 4, 8, 12, 19}; // S = 5
                float[] gpuPrefill = runner.forward(prompt);
                float[] refPrefill = ref.forward(prompt);
                assertArrayEquals(refPrefill, gpuPrefill, 1e-4f);

                int nextTok = LlamaRunner.argmax(gpuPrefill);
                float[] gpuDecode = runner.forward(new int[]{nextTok}); // S = 1, T = 6
                float[] refDecode = ref.forward(new int[]{nextTok});
                assertArrayEquals(refDecode, gpuDecode, 1e-4f, "Split-KV decode logits must match f64 reference");

                List<PhaseEvent> trace = runner.morphTrace();
                assertEquals(2, trace.size());

                PhaseEvent prefillEvent = trace.get(0);
                assertEquals(PhaseEvent.Phase.PREFILL, prefillEvent.phase());
                assertEquals(5, prefillEvent.seqLen());
                assertNull(prefillEvent.executionReport());

                PhaseEvent decodeEvent = trace.get(1);
                assertEquals(PhaseEvent.Phase.DECODE, decodeEvent.phase());
                assertEquals(1, decodeEvent.seqLen());
                assertEquals(6, decodeEvent.totalContextLen());

                ExecutionReport report = decodeEvent.executionReport();
                assertNotNull(report, "Expected live Split-KV ExecutionReport during Decode");
                assertEquals("splitKvAttention", report.operation());
                assertTrue(report.executedGpuRanks().containsAll(List.of(0, 1)),
                        "Expected both GPU ranks 0 and 1 in executedGpuRanks: " + report.executedGpuRanks());
                assertTrue(report.cpuInPlaceUnits() > 0,
                        "Expected positive CPU-in-place KV tokens: " + report.cpuInPlaceUnits());
            }
        }
    }

    @Test
    @DisplayName("End-to-End Prompt + BpeTokenizer Generation & Zero VRAM Leak Proof")
    void testEndToEndPromptGenerationWithTokenizer(@TempDir Path tempDir) throws IOException {
        var memBefore = Engine.memoryStats();

        // Write a tiny valid ByteLevel tokenizer.json alongside synthetic model files
        boolean[] keep = new boolean[256];
        for (int b = '!'; b <= '~'; b++) keep[b] = true;
        for (int b = 0xA1; b <= 0xAC; b++) keep[b] = true;
        for (int b = 0xAE; b <= 0xFF; b++) keep[b] = true;
        Map<String, Integer> vocab = new LinkedHashMap<>();
        int nextId = 0;
        int extra = 0;
        for (int b = 0; b < 256; b++) {
            vocab.put(String.valueOf(keep[b] ? (char) b : (char) (256 + extra++)), nextId++);
        }
        int bosId = nextId++;
        int eosId = nextId++;
        List<Map<String, Object>> addedTokens = List.of(
                Map.of("id", bosId, "content", "<|begin_of_text|>", "special", true),
                Map.of("id", eosId, "content", "<|end_of_text|>", "special", true)
        );
        Map<String, Object> tokRoot = new LinkedHashMap<>();
        tokRoot.put("model", Map.of("type", "BPE", "vocab", vocab, "merges", List.of("h e", "he l", "hel l", "hell o")));
        tokRoot.put("added_tokens", addedTokens);
        tokRoot.put("pre_tokenizer", Map.of("type", "ByteLevel", "add_prefix_space", false));
        tokRoot.put("decoder", Map.of("type", "ByteLevel"));
        vocab.put("he", nextId++);
        vocab.put("hel", nextId++);
        vocab.put("hell", nextId++);
        vocab.put("hello", nextId++);
        Files.writeString(tempDir.resolve("tokenizer.json"), MinimalJson.serialize(tokRoot));

        int vocabSize = nextId;
        LlamaConfig config = new LlamaConfig(vocabSize, 16, 32, 1, 2, 1, 1e-5f, 10000.0f, 64, true, false);
        writeSyntheticModel(tempDir, config, DType.f16, 999L);

        BpeTokenizer tokenizer = BpeTokenizer.load(tempDir.resolve("tokenizer.json"));
        int[] promptIds = tokenizer.encode("hello world!", true);
        assertEquals(bosId, promptIds[0]);

        try (SafetensorsReader reader = SafetensorsReader.open(tempDir);
             LlamaWeights weights = LlamaWeights.load(reader, config, DType.f16);
             LlamaRunner runner = new LlamaRunner(config, weights, 64)) {

            int[] generated = runner.generateGreedy(promptIds, 4, tokenizer.eosTokenIds());
            assertTrue(generated.length > promptIds.length);
            assertTrue(generated.length <= promptIds.length + 4);

            String decoded = tokenizer.decode(generated, true);
            assertFalse(decoded.isEmpty());
            assertTrue(decoded.startsWith("hello world!"), "Decoded output should preserve prompt prefix: " + decoded);
        }

        var memAfter = Engine.memoryStats();
        assertEquals(memBefore.tensorsDevice(), memAfter.tensorsDevice());
        assertEquals(memBefore.tensorsHost(), memAfter.tensorsHost());
        assertEquals(memBefore.tensorsEvicted(), memAfter.tensorsEvicted());

        // Exercise LlamaCli.main end-to-end with --prompt and --temperature
        assertDoesNotThrow(() -> LlamaCli.main(new String[]{
                "--model-dir", tempDir.toString(),
                "--prompt", "hello",
                "--max-new-tokens", "3",
                "--max-seq-len", "32",
                "--dtype", "f16",
                "--temperature", "0.7",
                "--seed", "42"
        }));

        // Exercise LlamaCli.main with --distributed (live auto-morphing trace)
        assertDoesNotThrow(() -> LlamaCli.main(new String[]{
                "--model-dir", tempDir.toString(),
                "--prompt", "hello",
                "--max-new-tokens", "2",
                "--max-seq-len", "32",
                "--dtype", "f32",
                "--distributed"
        }));

        var memAfterCli = Engine.memoryStats();
        assertEquals(memBefore.tensorsDevice(), memAfterCli.tensorsDevice());
        assertEquals(memBefore.tensorsHost(), memAfterCli.tensorsHost());
        assertEquals(memBefore.tensorsEvicted(), memAfterCli.tensorsEvicted());
    }

    @Test
    @DisplayName("Qwen2.5 Regression: omitted attention_bias auto-detected for Q/K/V/O, array eos_token_id, scientific config & CLI UX")
    void testQwen25OmittedAttentionBiasAndConfigFields(@TempDir Path tempDir) throws IOException {
        // Verify MinimalJson & LlamaConfig scientific notation, large rope_theta, explicit head_dim, and array eos_token_id
        String qwenConfigJson = """
                {
                  "vocab_size": 32,
                  "hidden_size": 16,
                  "intermediate_size": 32,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2,
                  "rms_norm_eps": 1e-06,
                  "rope_theta": 1e6,
                  "max_position_embeddings": 32768,
                  "tie_word_embeddings": true,
                  "eos_token_id": [15, 16]
                }
                """;
        LlamaConfig parsedCfg = LlamaConfig.fromJson(qwenConfigJson);
        assertFalse(parsedCfg.qkvBias(), "attention_bias omitted so config flag defaults to false");
        assertEquals(1e-6f, parsedCfg.rmsNormEps(), 1e-12f);
        assertEquals(1_000_000.0f, parsedCfg.ropeTheta(), 1e-3f);
        assertEquals(4, parsedCfg.headDim());
        assertEquals(List.of(15, 16), parsedCfg.eosTokenIds());

        // Also check explicit head_dim override (e.g. Qwen3-style non-hiddenSize/heads)
        LlamaConfig explicitHeadCfg = LlamaConfig.fromJson("""
                {"vocab_size":32,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,
                 "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"eos_token_id":151645}
                """);
        assertEquals(8, explicitHeadCfg.headDim());
        assertEquals(2, explicitHeadCfg.gqaRatio());
        assertEquals(List.of(151645), explicitHeadCfg.eosTokenIds());

        var memBefore = Engine.memoryStats();

        // 1. F32 model: omits attention_bias in config.json, writes +/-20.0 Q/K/V biases + EOS [15, 16]
        Path f32Dir = tempDir.resolve("qwen_f32");
        Files.createDirectories(f32Dir);
        ReferenceTransformerF64 refF32 = writeSyntheticModel(
                f32Dir, parsedCfg, DType.f32, 31415L, true, true, false, 20.0f, false);

        LlamaConfig loadedCfgF32 = LlamaConfig.fromFile(f32Dir.resolve("config.json"));
        assertFalse(loadedCfgF32.qkvBias());
        assertEquals(List.of(15, 16), loadedCfgF32.eosTokenIds());

        int[] prompt = new int[]{2, 7, 13, 4};
        try (SafetensorsReader reader = SafetensorsReader.open(f32Dir);
             LlamaWeights weights = LlamaWeights.load(reader, loadedCfgF32, DType.f32);
             LlamaRunner runner = new LlamaRunner(loadedCfgF32, weights, 64)) {

            for (LlamaWeights.LayerWeights lw : weights.layers()) {
                assertNotNull(lw.qBias(), "q_proj.bias must be auto-detected even when attention_bias is omitted");
                assertNotNull(lw.kBias(), "k_proj.bias must be auto-detected even when attention_bias is omitted");
                assertNotNull(lw.vBias(), "v_proj.bias must be auto-detected even when attention_bias is omitted");
                assertNull(lw.oBias(), "o_proj.bias absent so expected null");
            }
            assertEquals(java.util.Set.of(15, 16), runner.eosTokenIds());

            float[] gpuPrefill = runner.forward(prompt);
            float[] refPrefill = refF32.forward(prompt);
            assertArrayEquals(refPrefill, gpuPrefill, 1e-4f, "F32 Qwen2.5-style prefill with +/-20 QKV biases");

            int nextTok = LlamaRunner.argmax(gpuPrefill);
            for (int step = 0; step < 3; step++) {
                float[] gpuDec = runner.forward(new int[]{nextTok});
                float[] refDec = refF32.forward(new int[]{nextTok});
                assertArrayEquals(refDec, gpuDec, 1e-4f, "F32 Qwen2.5-style decode step " + step);
                nextTok = LlamaRunner.argmax(gpuDec);
            }
        }

        // 2. F16 model: omits attention_bias in config.json, includes Q/K/V + optional O biases (+/-20.0), tied fallback
        Path f16Dir = tempDir.resolve("qwen_f16");
        Files.createDirectories(f16Dir);
        LlamaConfig untiedFallbackCfg = new LlamaConfig(
                32, 16, 32, 2, 4, 2, 4, 1e-6f, 1_000_000.0f, 32768, false, false, List.of(15, 16));
        ReferenceTransformerF64 refF16 = writeSyntheticModel(
                f16Dir, untiedFallbackCfg, DType.f16, 27182L, true, true, true, 20.0f, true);

        LlamaConfig loadedCfgF16 = LlamaConfig.fromFile(f16Dir.resolve("config.json"));
        try (SafetensorsReader reader = SafetensorsReader.open(f16Dir);
             LlamaWeights weights = LlamaWeights.load(reader, loadedCfgF16, DType.f16);
             LlamaRunner runner = new LlamaRunner(loadedCfgF16, weights, 64)) {

            assertNotNull(weights.layer(0).qBias());
            assertNotNull(weights.layer(0).kBias());
            assertNotNull(weights.layer(0).vBias());
            assertNotNull(weights.layer(0).oBias(), "o_proj.bias should be auto-detected when present");
            assertSame(weights.embedTokens(), weights.lmHead(), "lm_head should fall back to embedTokens when absent");

            float[] gpuPrefill = runner.forward(prompt);
            float[] refPrefill = refF16.forward(prompt);
            assertArrayEquals(refPrefill, gpuPrefill, 2e-2f, "F16 Qwen2.5-style prefill with +/-20 Q/K/V/O biases");

            int nextTok = LlamaRunner.argmax(gpuPrefill);
            for (int step = 0; step < 3; step++) {
                float[] gpuDec = runner.forward(new int[]{nextTok});
                float[] refDec = refF16.forward(new int[]{nextTok});
                assertArrayEquals(refDec, gpuDec, 2e-2f, "F16 Qwen2.5-style decode step " + step);
                nextTok = LlamaRunner.argmax(gpuDec);
            }

            // Verify LlamaRunner stops early when config EOS token ID is generated
            runner.reset();
            LlamaConfig stopNowCfg = new LlamaConfig(
                    32, 16, 32, 2, 4, 2, 4, 1e-6f, 1_000_000.0f, 32768, true, false,
                    List.of(LlamaRunner.argmax(gpuPrefill)));
            try (LlamaRunner eosRunner = new LlamaRunner(stopNowCfg, weights, 64)) {
                int[] seq = eosRunner.generateGreedy(prompt, 8, java.util.Set.of(31));
                assertEquals(prompt.length + 1, seq.length,
                        "generateGreedy must stop immediately when config.eosTokenIds() triggers");
            }
        }

        // 3. Verify CLI prompt unescaping and transition-only morph trace formatting
        assertEquals("<|im_start|>user\nHi\tthere\\end\r\n<|im_start|>assistant\n",
                LlamaCli.unescapePrompt("<|im_start|>user\\nHi\\tthere\\\\end\\r\\n<|im_start|>assistant\\n"));

        WorkloadPlan wp = WorkloadPlanner.plan(
                List.of(new DeviceProfile(0, "GPU0", "Discrete", 0x10DE, 1, 8_000_000_000L, 1000.0)),
                List.of(),
                WorkloadSpec.builder(100, 10.0, 16).build());
        RolePlan prefillPlan = new RolePlan(wp, List.of(), 0.1, 0.0, 0, 0, false, "BalancedTP");
        RolePlan decodePlan = new RolePlan(wp, List.of(), 0.1, 0.0, 10, 1, true, "ReclaimedSplitKV+CPU");
        List<PhaseEvent> sampleTrace = List.of(
                new PhaseEvent(PhaseEvent.Phase.PREFILL, 20, 20, prefillPlan, null),
                new PhaseEvent(PhaseEvent.Phase.DECODE, 1, 21, decodePlan, null),
                new PhaseEvent(PhaseEvent.Phase.DECODE, 1, 22, decodePlan, null),
                new PhaseEvent(PhaseEvent.Phase.DECODE, 1, 23, decodePlan, null)
        );
        List<String> formattedTrace = LlamaCli.formatMorphTraceLines(sampleTrace);
        assertEquals(3, formattedTrace.size(), "Expected 2 transition lines + 1 summary line: " + formattedTrace);
        assertTrue(formattedTrace.get(0).contains("PREFILL:"));
        assertTrue(formattedTrace.get(1).contains("DECODE:"));
        assertTrue(formattedTrace.get(2).contains("PREFILL/BalancedTP=1 steps"));
        assertTrue(formattedTrace.get(2).contains("DECODE/ReclaimedSplitKV+CPU=3 steps"));

        var memAfter = Engine.memoryStats();
        assertEquals(memBefore.tensorsDevice(), memAfter.tensorsDevice());
        assertEquals(memBefore.tensorsHost(), memAfter.tensorsHost());
        assertEquals(memBefore.tensorsEvicted(), memAfter.tensorsEvicted());
    }

    // --- Helpers ---

    private static float[] randArray(Random rng, int size, float scale) {
        float[] arr = new float[size];
        for (int i = 0; i < size; i++) {
            arr[i] = (rng.nextFloat() * 2.0f - 1.0f) * scale;
        }
        return arr;
    }

    private static float[] positiveRandArray(Random rng, int size, float min, float max) {
        float[] arr = new float[size];
        for (int i = 0; i < size; i++) {
            arr[i] = min + rng.nextFloat() * (max - min);
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

    private static short floatToBfloat16(float fval) {
        return (short) (Float.floatToIntBits(fval) >>> 16);
    }
}
