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
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Command-line runner for executing Llama-3.2, Qwen2.5, and SmolLM2 models on the HC4J engine.
 */
public final class LlamaCli {

    public static void main(String[] args) throws Exception {
        Path modelDir = Paths.get(".");
        List<Integer> promptTokens = new ArrayList<>(List.of(1, 150, 42));
        int maxNewTokens = 10;
        DType precision = DType.f16;
        int distributedRanks = 0;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--model-dir" -> {
                    if (i + 1 < args.length) modelDir = Paths.get(args[++i]);
                }
                case "--tokens" -> {
                    if (i + 1 < args.length) {
                        promptTokens.clear();
                        for (String part : args[++i].split(",")) {
                            promptTokens.add(Integer.parseInt(part.trim()));
                        }
                    }
                }
                case "--max-new-tokens" -> {
                    if (i + 1 < args.length) maxNewTokens = Integer.parseInt(args[++i]);
                }
                case "--dtype" -> {
                    if (i + 1 < args.length) {
                        String dt = args[++i].toLowerCase();
                        precision = "f32".equals(dt) ? DType.f32 : DType.f16;
                    }
                }
                case "--distributed" -> {
                    if (i + 1 < args.length) distributedRanks = Integer.parseInt(args[++i]);
                }
                case "--help" -> {
                    printUsage();
                    return;
                }
            }
        }

        System.out.println("==================================================================");
        System.out.println("HC4J Transformer LLM Runner (Step 4D)");
        System.out.println("==================================================================");
        System.out.println("Adapter: " + Engine.adapterInfo());
        System.out.println("Model Directory: " + modelDir.toAbsolutePath());
        System.out.println("Precision: " + precision);

        Path configPath = modelDir.resolve("config.json");
        if (!configPath.toFile().exists()) {
            System.err.println("Error: config.json not found at " + configPath);
            return;
        }

        LlamaConfig config = LlamaConfig.fromFile(configPath);
        System.out.printf("Architecture: layers=%d, hidden=%d, intermediate=%d, heads=%d, kv_heads=%d, vocab=%d%n",
                config.numHiddenLayers(), config.hiddenSize(), config.intermediateSize(),
                config.numAttentionHeads(), config.numKeyValueHeads(), config.vocabSize());

        Collective collective = null;
        if (distributedRanks >= 2) {
            collective = Collective.open(distributedRanks);
            System.out.printf("Attached Collective: %d ranks (virtual loopback mode)%n", distributedRanks);
        }

        try (SafetensorsReader reader = SafetensorsReader.open(modelDir);
             LlamaWeights weights = LlamaWeights.load(reader, config, precision);
             LlamaRunner runner = new LlamaRunner(config, weights, collective)) {

            System.out.printf("Loaded %d tensors into GPU VRAM.%n", reader.tensorNames().size());
            System.out.println("Prompt Tokens: " + promptTokens);
            System.out.println("------------------------------------------------------------------");

            long t0 = System.nanoTime();
            int firstToken = runner.prefill(promptTokens);
            long prefillTimeNs = System.nanoTime() - t0;
            System.out.printf("Prefill (S=%d): %.2f ms (First token: %d)%n",
                    promptTokens.size(), prefillTimeNs / 1_000_000.0, firstToken);

            List<Integer> generated = new ArrayList<>();
            generated.add(firstToken);
            int currentToken = firstToken;

            long decodeStartNs = System.nanoTime();
            for (int step = 1; step < maxNewTokens; step++) {
                long stepT0 = System.nanoTime();
                currentToken = runner.decodeStep(currentToken);
                long stepTimeNs = System.nanoTime() - stepT0;
                generated.add(currentToken);
                System.out.printf("Decode step %d: token=%d (%.2f ms)%n", step, currentToken, stepTimeNs / 1_000_000.0);
            }
            long totalDecodeNs = System.nanoTime() - decodeStartNs;
            double tokPerSec = (maxNewTokens - 1) / (totalDecodeNs / 1_000_000_000.0);
            System.out.println("------------------------------------------------------------------");
            System.out.printf("Total Generated: %d tokens | Decode Throughput: %.2f tok/s%n", generated.size(), tokPerSec);

            if (collective != null && !runner.morphTrace().isEmpty()) {
                System.out.println("\nLive Prefill -> Decode Auto-Morphing Trace:");
                for (String event : runner.morphTrace()) {
                    System.out.println("  " + event);
                }
            }
        } finally {
            if (collective != null) {
                collective.close();
            }
        }
    }

    private static void printUsage() {
        System.out.println("Usage: java -jar hc4j-examples.jar [options]");
        System.out.println("Options:");
        System.out.println("  --model-dir <dir>        Directory containing config.json and .safetensors files");
        System.out.println("  --tokens <t1,t2,...>     Comma-separated prompt token IDs (e.g. 1,45,99)");
        System.out.println("  --max-new-tokens <N>     Maximum number of tokens to generate (default: 10)");
        System.out.println("  --dtype <f16|f32>        Weight precision (default: f16)");
        System.out.println("  --distributed <ranks>    Number of virtual ranks to attach for Auto-Morphing");
    }
}
