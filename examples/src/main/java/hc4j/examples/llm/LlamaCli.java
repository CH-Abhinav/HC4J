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
import hc4j.collective.HostCpuProfile;
import hc4j.engine.Engine;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Command-line runner for executing Llama-3.2, Qwen2.5, and SmolLM2 models on the HC4J engine
 * with 100%-GPU KV-caching, pure-Java {@code tokenizer.json} BPE encoding/streaming,
 * temperature sampling, and live Prefill &rarr; Decode auto-morphing.
 */
public final class LlamaCli {

    private LlamaCli() {
        throw new AssertionError("No instances of LlamaCli");
    }

    /**
     * CLI entrypoint.
     *
     * @param args CLI flags
     * @throws Exception if model loading or inference fails
     */
    public static void main(String[] args) throws Exception {
        Path modelDir = Paths.get(".");
        String promptText = null;
        int[] explicitTokens = null;
        int maxNewTokens = 32;
        int maxSeqLen = LlamaRunner.DEFAULT_MAX_SEQ_LEN;
        DType precision = DType.f16;
        float temperature = 0.0f;
        int distributedRanks = 0;
        boolean ignoreEos = false;
        long seed = System.nanoTime();

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--model-dir" -> {
                    if (i + 1 < args.length) modelDir = Paths.get(args[++i]);
                }
                case "--prompt" -> {
                    if (i + 1 < args.length) promptText = args[++i];
                }
                case "--tokens" -> {
                    if (i + 1 < args.length) {
                        String[] parts = args[++i].split(",");
                        explicitTokens = new int[parts.length];
                        for (int j = 0; j < parts.length; j++) {
                            explicitTokens[j] = Integer.parseInt(parts[j].trim());
                        }
                    }
                }
                case "--max-new-tokens" -> {
                    if (i + 1 < args.length) maxNewTokens = Integer.parseInt(args[++i]);
                }
                case "--max-seq-len" -> {
                    if (i + 1 < args.length) maxSeqLen = Integer.parseInt(args[++i]);
                }
                case "--dtype" -> {
                    if (i + 1 < args.length) {
                        String dt = args[++i].toLowerCase();
                        precision = "f32".equals(dt) ? DType.f32 : DType.f16;
                    }
                }
                case "--temperature" -> {
                    if (i + 1 < args.length) temperature = Float.parseFloat(args[++i]);
                }
                case "--seed" -> {
                    if (i + 1 < args.length) seed = Long.parseLong(args[++i]);
                }
                case "--distributed" -> {
                    if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                        distributedRanks = Integer.parseInt(args[++i]);
                    } else {
                        distributedRanks = 2;
                    }
                }
                case "--ignore-eos" -> ignoreEos = true;
                case "--help", "-h" -> {
                    printUsage();
                    return;
                }
                default -> throw new IllegalArgumentException("Unknown CLI flag: " + args[i]);
            }
        }

        Path configPath = modelDir.resolve("config.json");
        if (!Files.exists(configPath)) {
            System.err.println("Error: config.json not found at " + configPath.toAbsolutePath());
            return;
        }

        LlamaConfig config = LlamaConfig.fromFile(configPath);
        int effectiveMaxSeqLen = config.maxPositionEmbeddings() > 0
                ? Math.min(maxSeqLen, config.maxPositionEmbeddings())
                : maxSeqLen;

        BpeTokenizer tokenizer = null;
        Path tokenizerPath = modelDir.resolve("tokenizer.json");
        if (Files.exists(tokenizerPath)) {
            tokenizer = BpeTokenizer.load(tokenizerPath);
        }

        if (promptText != null && tokenizer == null) {
            throw new IllegalArgumentException(
                    "--prompt specified, but tokenizer.json not found at " + tokenizerPath.toAbsolutePath());
        }

        int[] promptTokens;
        if (promptText != null) {
            promptTokens = tokenizer.encode(promptText, true);
        } else if (explicitTokens != null) {
            promptTokens = explicitTokens;
        } else {
            promptTokens = new int[]{1, 150, 42};
        }

        Set<Integer> eosTokenIds = ignoreEos ? Set.of() : collectEosTokenIds(tokenizer, configPath);

        System.out.println("==================================================================");
        System.out.println("HC4J Transformer LLM Runner (100% GPU)");
        System.out.println("==================================================================");
        System.out.println("Adapter: " + Engine.adapterInfo());
        System.out.println("Model Directory: " + modelDir.toAbsolutePath());
        System.out.printf("Architecture: layers=%d, hidden=%d, intermediate=%d, heads=%d, kv_heads=%d, vocab=%d%n",
                config.numHiddenLayers(), config.hiddenSize(), config.intermediateSize(),
                config.numAttentionHeads(), config.numKeyValueHeads(), config.vocabSize());
        System.out.printf("Config: weight_dtype=%s, max_seq_len=%d, temperature=%.2f%n",
                precision, effectiveMaxSeqLen, temperature);

        Collective collective = null;
        HostCpuProfile cpuProfile = HostCpuProfile.disabled();
        if (distributedRanks >= 2) {
            collective = Collective.open(distributedRanks);
            cpuProfile = HostCpuProfile.of("Host-CPU", 16L * 1024 * 1024 * 1024, 250.0, 50.0, 32.0);
            System.out.printf("Attached Collective: %d ranks + HostCpuProfile (live auto-morphing)%n", distributedRanks);
        }

        long vramBeforeLoad = Engine.memoryStats().vramUsed();

        try (SafetensorsReader reader = SafetensorsReader.open(modelDir);
             LlamaWeights weights = LlamaWeights.load(reader, config, precision);
             LlamaRunner runner = new LlamaRunner(config, weights, collective, cpuProfile, effectiveMaxSeqLen)) {

            long vramAfterLoad = Engine.memoryStats().vramUsed();
            double weightVramMb = Math.max(0L, vramAfterLoad - vramBeforeLoad) / (1024.0 * 1024.0);
            System.out.printf("Loaded %d tensors | Live VRAM footprint: %.2f MiB%n",
                    reader.tensorNames().size(), weightVramMb);

            if (promptText != null) {
                System.out.println("Prompt (" + promptTokens.length + " tokens): " + promptText);
            } else {
                System.out.println("Prompt Tokens: " + java.util.Arrays.toString(promptTokens));
            }
            System.out.println("------------------------------------------------------------------");

            Random rng = new Random(seed);
            List<Integer> generated = new ArrayList<>(maxNewTokens);
            BpeTokenizer.StreamDecoder streamDecoder = tokenizer != null ? tokenizer.newStreamDecoder() : null;

            long prefillStartNs = System.nanoTime();
            float[] firstLogits = runner.forward(promptTokens);
            long prefillNs = System.nanoTime() - prefillStartNs;

            int nextToken = selectToken(firstLogits, temperature, rng);
            if (maxNewTokens > 0) {
                if (promptText != null) {
                    System.out.print(promptText);
                }
                emitToken(streamDecoder, nextToken, eosTokenIds);
                generated.add(nextToken);
            }

            long decodeStartNs = System.nanoTime();
            int decodeSteps = 0;
            for (int step = 1; step < maxNewTokens; step++) {
                if (eosTokenIds.contains(nextToken)) {
                    break;
                }
                float[] logits = runner.forward(new int[]{nextToken});
                decodeSteps++;
                nextToken = selectToken(logits, temperature, rng);
                generated.add(nextToken);
                if (eosTokenIds.contains(nextToken)) {
                    break;
                }
                emitToken(streamDecoder, nextToken, eosTokenIds);
            }
            long decodeNs = System.nanoTime() - decodeStartNs;
            if (streamDecoder != null) {
                System.out.print(streamDecoder.flush());
            }
            if (promptText != null) {
                System.out.println();
            }

            double prefillTokPerSec = prefillNs > 0 ? promptTokens.length / (prefillNs / 1e9) : 0.0;
            double decodeTokPerSec = (decodeSteps > 0 && decodeNs > 0) ? decodeSteps / (decodeNs / 1e9) : 0.0;
            long liveVramTotalMb = Engine.memoryStats().vramUsed() / (1024L * 1024L);

            System.out.println("------------------------------------------------------------------");
            System.out.printf("Summary: weight_dtype=%s | live_vram=%d MiB | generated=%d tokens%n",
                    precision, liveVramTotalMb, generated.size());
            System.out.printf("Throughput: Prefill=%.2f tok/s (S=%d) | Decode=%.2f tok/s (%d steps)%n",
                    prefillTokPerSec, promptTokens.length, decodeTokPerSec, decodeSteps);

            if (tokenizer == null) {
                System.out.println("Generated Token IDs: " + generated);
            }

            if (collective != null && !runner.morphTrace().isEmpty()) {
                System.out.println("\nLive Prefill -> Decode Auto-Morphing Trace:");
                for (PhaseEvent event : runner.morphTrace()) {
                    System.out.println("  " + event.summary());
                }
            }
        } finally {
            if (collective != null) {
                collective.close();
            }
        }
    }

    private static void emitToken(BpeTokenizer.StreamDecoder streamDecoder, int tokenId, Set<Integer> eosIds) {
        if (eosIds.contains(tokenId)) {
            return;
        }
        if (streamDecoder != null) {
            System.out.print(streamDecoder.next(tokenId));
            System.out.flush();
        } else {
            System.out.println("  token=" + tokenId);
        }
    }

    static int selectToken(float[] logits, float temperature, Random rng) {
        if (temperature <= 0.0f) {
            return LlamaRunner.argmax(logits);
        }
        double invTemp = 1.0 / temperature;
        double maxScaled = Double.NEGATIVE_INFINITY;
        double[] scaled = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            scaled[i] = logits[i] * invTemp;
            if (scaled[i] > maxScaled) {
                maxScaled = scaled[i];
            }
        }
        double sumExp = 0.0;
        double[] probs = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = Math.exp(scaled[i] - maxScaled);
            sumExp += probs[i];
        }
        double target = rng.nextDouble() * sumExp;
        double cdf = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cdf += probs[i];
            if (cdf >= target) {
                return i;
            }
        }
        return probs.length - 1;
    }

    private static Set<Integer> collectEosTokenIds(BpeTokenizer tokenizer, Path configPath) throws IOException {
        Set<Integer> eos = new LinkedHashSet<>();
        if (tokenizer != null) {
            eos.addAll(tokenizer.eosTokenIds());
        }
        if (Files.exists(configPath)) {
            Map<String, Object> cfg = MinimalJson.parseObject(Files.readString(configPath, StandardCharsets.UTF_8));
            Object eosObj = cfg.get("eos_token_id");
            if (eosObj instanceof Number n) {
                eos.add(n.intValue());
            } else if (eosObj instanceof List<?> list) {
                for (Object item : list) {
                    if (item instanceof Number n) {
                        eos.add(n.intValue());
                    }
                }
            }
        }
        return eos;
    }

    private static void printUsage() {
        System.out.println("Usage: LlamaCli [options]");
        System.out.println("Options:");
        System.out.println("  --model-dir <dir>        Directory containing config.json, model.safetensors, and tokenizer.json");
        System.out.println("  --prompt \"<text>\"        Text prompt to encode with tokenizer.json and stream output");
        System.out.println("  --tokens <t1,t2,...>     Comma-separated prompt token IDs (fallback when tokenizer.json is absent)");
        System.out.println("  --max-new-tokens <N>     Maximum number of new tokens to generate (default: 32)");
        System.out.println("  --max-seq-len <N>        Maximum KV-cache sequence length (default: 2048)");
        System.out.println("  --dtype <f16|f32>        Weight precision (default: f16)");
        System.out.println("  --temperature <float>    Sampling temperature (0.0 = deterministic greedy, default: 0.0)");
        System.out.println("  --seed <long>            Random seed for temperature sampling");
        System.out.println("  --distributed [ranks]    Attach Collective (default 2 ranks) + HostCpuProfile for live auto-morphing");
        System.out.println("  --ignore-eos             Continue generation past EOS token IDs");
    }
}
