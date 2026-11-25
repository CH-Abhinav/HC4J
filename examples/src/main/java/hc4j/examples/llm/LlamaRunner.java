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
import hc4j.GpuScope;
import hc4j.Slice;
import hc4j.Tensor;
import hc4j.collective.Collective;
import hc4j.collective.DistributedExecutor;
import hc4j.collective.DistributedResult;
import hc4j.collective.ExecutionReport;
import hc4j.collective.HostCpuProfile;
import hc4j.collective.RolePlan;
import hc4j.collective.SplitKvAssignment;
import hc4j.collective.SplitKvSpec;
import hc4j.collective.WorkloadPlanner;
import hc4j.collective.WorkloadSpec;
import hc4j.nn.Nn;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiFunction;

/**
 * Pure-Java 100%-GPU Transformer decoder forward runner for Llama-3.2, Qwen2.5, and SmolLM2 models.
 * <p>
 * Maintains per-layer KV-caches resident in GPU VRAM (updated in-place via {@link Nn#kvCacheAppend}),
 * executes single-device Prefill and Decode attention purely on the GPU using batched 3-D GEMM and
 * {@link Nn#softmax} with zero per-layer host round-trips, and dynamically morphs Decode attention
 * onto {@link DistributedExecutor#splitKvAttention} when attached to a {@link Collective} whose
 * {@link RolePlan} activates &ge; 2 Split-KV workers.
 */
public final class LlamaRunner implements AutoCloseable {

    /** Default upper bound on pre-allocated VRAM KV-cache sequence length. */
    public static final int DEFAULT_MAX_SEQ_LEN = 2048;

    private final LlamaConfig config;
    private final LlamaWeights weights;
    private final Collective collective;
    private final HostCpuProfile hostCpuProfile;
    private final BiFunction<WorkloadSpec, SplitKvSpec, RolePlan> rolePlanner;
    private final int maxSeqLen;

    // VRAM-resident KV caches per layer: [numKvHeads, maxSeqLen, headDim]
    private final Tensor[] kCaches;
    private final Tensor[] vCaches;

    // Telemetry trace recording phase transitions, role plans, and Split-KV execution reports
    private final List<PhaseEvent> morphTrace = new ArrayList<>();

    private int curPos = 0;
    private boolean closed = false;

    /**
     * Creates a single-GPU runner with the default bounded KV-cache length.
     *
     * @param config  model architecture configuration
     * @param weights loaded GPU weights
     */
    public LlamaRunner(LlamaConfig config, LlamaWeights weights) {
        this(config, weights, null, HostCpuProfile.disabled(), defaultCacheLen(config), null);
    }

    /**
     * Creates a single-GPU runner with an explicit maximum KV-cache sequence length.
     *
     * @param config    model architecture configuration
     * @param weights   loaded GPU weights
     * @param maxSeqLen maximum sequence length reserved in VRAM KV-cache
     */
    public LlamaRunner(LlamaConfig config, LlamaWeights weights, int maxSeqLen) {
        this(config, weights, null, HostCpuProfile.disabled(), maxSeqLen, null);
    }

    /**
     * Creates a runner attached to a {@link Collective} for live Prefill &rarr; Decode auto-morphing.
     *
     * @param config     model architecture configuration
     * @param weights    loaded GPU weights
     * @param collective optional collective (may be null for single-GPU execution)
     */
    public LlamaRunner(LlamaConfig config, LlamaWeights weights, Collective collective) {
        this(config, weights, collective, HostCpuProfile.disabled(), defaultCacheLen(config), null);
    }

    /**
     * Creates a runner attached to a {@link Collective} with an explicit maximum KV-cache sequence length.
     *
     * @param config     model architecture configuration
     * @param weights    loaded GPU weights
     * @param collective optional collective (may be null)
     * @param maxSeqLen  maximum sequence length reserved in VRAM KV-cache
     */
    public LlamaRunner(LlamaConfig config, LlamaWeights weights, Collective collective, int maxSeqLen) {
        this(config, weights, collective, HostCpuProfile.disabled(), maxSeqLen, null);
    }

    /**
     * Creates a runner attached to a {@link Collective} and optional {@link HostCpuProfile}.
     *
     * @param config         model architecture configuration
     * @param weights        loaded GPU weights
     * @param collective     optional collective (may be null)
     * @param hostCpuProfile host CPU profile for hybrid planning (defaults to disabled if null)
     * @param maxSeqLen      maximum sequence length reserved in VRAM KV-cache
     */
    public LlamaRunner(LlamaConfig config, LlamaWeights weights, Collective collective,
                       HostCpuProfile hostCpuProfile, int maxSeqLen) {
        this(config, weights, collective, hostCpuProfile, maxSeqLen, null);
    }

    /**
     * Creates a runner attached to a {@link Collective} with a custom {@link RolePlan} supplier.
     *
     * @param config      model architecture configuration
     * @param weights     loaded GPU weights
     * @param collective  optional collective (may be null)
     * @param rolePlanner optional custom role planning function (defaults to {@link WorkloadPlanner#planRoles})
     */
    public LlamaRunner(LlamaConfig config, LlamaWeights weights, Collective collective,
                       BiFunction<WorkloadSpec, SplitKvSpec, RolePlan> rolePlanner) {
        this(config, weights, collective, HostCpuProfile.disabled(), defaultCacheLen(config), rolePlanner);
    }

    /**
     * Full constructor configuring VRAM KV-cache capacity, collective handle, CPU profile, and role planner.
     *
     * @param config         model architecture configuration
     * @param weights        loaded GPU weights
     * @param collective     optional collective (may be null)
     * @param hostCpuProfile optional host CPU profile (defaults to disabled if null)
     * @param maxSeqLen      maximum sequence length reserved in VRAM KV-cache
     * @param rolePlanner    optional custom role planner (defaults to {@link WorkloadPlanner#planRoles})
     */
    public LlamaRunner(LlamaConfig config, LlamaWeights weights, Collective collective,
                       HostCpuProfile hostCpuProfile, int maxSeqLen,
                       BiFunction<WorkloadSpec, SplitKvSpec, RolePlan> rolePlanner) {
        this.config = Objects.requireNonNull(config, "config");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.collective = collective;
        this.hostCpuProfile = hostCpuProfile != null ? hostCpuProfile : HostCpuProfile.disabled();
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be strictly positive, got " + maxSeqLen);
        }
        this.maxSeqLen = maxSeqLen;
        if (rolePlanner != null) {
            this.rolePlanner = rolePlanner;
        } else if (collective != null) {
            this.rolePlanner = (tp, kv) ->
                    WorkloadPlanner.planRoles(collective.deviceProfiles(), collective.topology(), tp, kv);
        } else {
            this.rolePlanner = null;
        }

        int numLayers = config.numHiddenLayers();
        int numKvHeads = config.numKeyValueHeads();
        int headDim = config.headDim();
        this.kCaches = new Tensor[numLayers];
        this.vCaches = new Tensor[numLayers];
        try {
            for (int l = 0; l < numLayers; l++) {
                this.kCaches[l] = Tensor.empty(DType.f32, numKvHeads, maxSeqLen, headDim);
                this.vCaches[l] = Tensor.empty(DType.f32, numKvHeads, maxSeqLen, headDim);
            }
        } catch (RuntimeException | Error e) {
            releaseCaches();
            throw e;
        }
    }

    private static int defaultCacheLen(LlamaConfig config) {
        Objects.requireNonNull(config, "config");
        int maxPos = config.maxPositionEmbeddings();
        return maxPos > 0 ? Math.min(maxPos, DEFAULT_MAX_SEQ_LEN) : DEFAULT_MAX_SEQ_LEN;
    }

    /**
     * Returns the configured maximum KV-cache sequence length.
     *
     * @return maximum sequence length
     */
    public int maxSeqLen() {
        return maxSeqLen;
    }

    /**
     * Returns the telemetry trace of Prefill/Decode auto-morphing transitions and Split-KV reports.
     *
     * @return unmodifiable list of phase events
     */
    public List<PhaseEvent> morphTrace() {
        return Collections.unmodifiableList(morphTrace);
    }

    /**
     * Resets the sequence position cursor and telemetry trace to start a fresh generation sequence
     * without reallocating the VRAM KV-cache buffers.
     */
    public void reset() {
        ensureOpen();
        curPos = 0;
        morphTrace.clear();
    }

    /**
     * Current accumulated token position in the sequence.
     *
     * @return token position index
     */
    public int currentPosition() {
        return curPos;
    }

    /**
     * Runs a forward pass on the provided token IDs and returns the final token's logits {@code float[vocabSize]}.
     * Uses {@link PhaseEvent.Phase#PREFILL} when {@code tokens.length > 1 || currentPosition() == 0},
     * and {@link PhaseEvent.Phase#DECODE} for single-token autoregressive steps.
     *
     * @param tokens non-empty array of input token IDs
     * @return next-token logits of length {@link LlamaConfig#vocabSize()}
     */
    public float[] forward(int[] tokens) {
        ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("Input tokens must not be empty");
        }
        PhaseEvent.Phase phase = (tokens.length > 1 || curPos == 0)
                ? PhaseEvent.Phase.PREFILL
                : PhaseEvent.Phase.DECODE;
        return forwardWithPhase(tokens, phase);
    }

    /**
     * Runs a forward pass on a list of token IDs and returns the final token's logits.
     *
     * @param tokens non-empty list of input token IDs
     * @return next-token logits of length {@link LlamaConfig#vocabSize()}
     */
    public float[] forward(List<Integer> tokens) {
        Objects.requireNonNull(tokens, "tokens");
        int[] arr = new int[tokens.size()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = tokens.get(i);
        }
        return forward(arr);
    }

    /**
     * Runs the prefill forward pass on a prompt sequence of token IDs and returns the greedy next token.
     *
     * @param promptTokens list of input token IDs
     * @return predicted next token ID
     */
    public int prefill(List<Integer> promptTokens) {
        Objects.requireNonNull(promptTokens, "promptTokens");
        if (promptTokens.isEmpty()) {
            throw new IllegalArgumentException("Prompt tokens must not be empty");
        }
        int[] arr = new int[promptTokens.size()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = promptTokens.get(i);
        }
        return argmax(forwardWithPhase(arr, PhaseEvent.Phase.PREFILL));
    }

    /**
     * Runs a single autoregressive decode step ({@code S == 1}) and returns the greedy next token.
     *
     * @param tokenId token ID from the previous step
     * @return predicted next token ID
     */
    public int decodeStep(int tokenId) {
        ensureOpen();
        return argmax(forwardWithPhase(new int[]{tokenId}, PhaseEvent.Phase.DECODE));
    }

    /**
     * Returns the immutable set of EOS token IDs configured in {@link LlamaConfig}.
     *
     * @return set of config EOS token IDs
     */
    public Set<Integer> eosTokenIds() {
        return collectEosTokenIds(config, null);
    }

    /**
     * Collects the union of end-of-sequence token IDs declared by both {@link LlamaConfig}
     * and {@link BpeTokenizer}.
     *
     * @param config    optional model config (may be null)
     * @param tokenizer optional BPE tokenizer (may be null)
     * @return immutable union of EOS token IDs
     */
    public static Set<Integer> collectEosTokenIds(LlamaConfig config, BpeTokenizer tokenizer) {
        Set<Integer> eos = new LinkedHashSet<>();
        if (tokenizer != null) {
            eos.addAll(tokenizer.eosTokenIds());
        }
        if (config != null) {
            eos.addAll(config.eosTokenIds());
        }
        return Collections.unmodifiableSet(eos);
    }

    /**
     * Greedy autoregressive generation returning the full token sequence {@code [prompt..., generated...]}.
     * Automatically halts if any {@link LlamaConfig#eosTokenIds()} token is generated.
     *
     * @param promptTokens initial prompt token IDs
     * @param maxNewTokens maximum number of new tokens to generate
     * @return full list containing prompt and generated tokens
     */
    public List<Integer> generate(List<Integer> promptTokens, int maxNewTokens) {
        Objects.requireNonNull(promptTokens, "promptTokens");
        if (promptTokens.isEmpty()) {
            throw new IllegalArgumentException("Prompt tokens must not be empty");
        }
        int[] promptArr = new int[promptTokens.size()];
        for (int i = 0; i < promptArr.length; i++) {
            promptArr[i] = promptTokens.get(i);
        }
        int[] full = generateGreedy(promptArr, maxNewTokens, Set.of());
        List<Integer> out = new ArrayList<>(full.length);
        for (int tok : full) {
            out.add(tok);
        }
        return out;
    }

    /**
     * Greedy autoregressive generation returning the full token sequence {@code [prompt..., generated...]}.
     * Automatically halts if any {@link LlamaConfig#eosTokenIds()} token is generated.
     *
     * @param promptTokens initial prompt token IDs
     * @param maxNewTokens maximum number of new tokens to generate
     * @return full token array containing prompt and generated tokens
     */
    public int[] generateGreedy(int[] promptTokens, int maxNewTokens) {
        return generateGreedy(promptTokens, maxNewTokens, Set.of());
    }

    /**
     * Greedy autoregressive generation stopping cleanly if any token in {@code config.eosTokenIds()}
     * or {@code tokenizer.eosTokenIds()} is emitted.
     *
     * @param promptTokens initial prompt token IDs
     * @param maxNewTokens maximum number of new tokens to generate
     * @param tokenizer    optional tokenizer contributing EOS token IDs (may be null)
     * @return full token array containing prompt and generated tokens
     */
    public int[] generateGreedy(int[] promptTokens, int maxNewTokens, BpeTokenizer tokenizer) {
        return generateGreedy(promptTokens, maxNewTokens, tokenizer != null ? tokenizer.eosTokenIds() : Set.of());
    }

    /**
     * Greedy autoregressive generation stopping cleanly if any token in {@code eosTokenIds}
     * or {@link LlamaConfig#eosTokenIds()} is emitted.
     *
     * @param promptTokens initial prompt token IDs
     * @param maxNewTokens maximum number of new tokens to generate
     * @param eosTokenIds  additional end-of-sequence token IDs causing early termination (may be null or empty)
     * @return full token array containing prompt and generated tokens
     */
    public int[] generateGreedy(int[] promptTokens, int maxNewTokens, Set<Integer> eosTokenIds) {
        ensureOpen();
        Objects.requireNonNull(promptTokens, "promptTokens");
        if (promptTokens.length == 0) {
            throw new IllegalArgumentException("Prompt tokens must not be empty");
        }
        if (maxNewTokens < 0) {
            throw new IllegalArgumentException("maxNewTokens must be non-negative, got " + maxNewTokens);
        }
        Set<Integer> eos = new LinkedHashSet<>(config.eosTokenIds());
        if (eosTokenIds != null) {
            eos.addAll(eosTokenIds);
        }

        List<Integer> seq = new ArrayList<>(promptTokens.length + maxNewTokens);
        for (int t : promptTokens) {
            seq.add(t);
        }
        if (maxNewTokens == 0) {
            return seq.stream().mapToInt(Integer::intValue).toArray();
        }

        int nextToken = argmax(forwardWithPhase(promptTokens, PhaseEvent.Phase.PREFILL));
        seq.add(nextToken);

        for (int step = 1; step < maxNewTokens; step++) {
            if (eos.contains(nextToken)) {
                break;
            }
            nextToken = decodeStep(nextToken);
            seq.add(nextToken);
        }

        int[] result = new int[seq.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = seq.get(i);
        }
        return result;
    }

    /**
     * Returns the index of the maximum logit value.
     *
     * @param logits float array of logits
     * @return argmax index
     */
    public static int argmax(float[] logits) {
        Objects.requireNonNull(logits, "logits");
        if (logits.length == 0) {
            throw new IllegalArgumentException("logits must not be empty");
        }
        int bestIdx = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > bestVal) {
                bestVal = logits[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    // --- Core GPU Forward Implementation ---

    private float[] forwardWithPhase(int[] tokens, PhaseEvent.Phase phase) {
        ensureOpen();
        int seqLen = tokens.length;
        if (seqLen == 0) {
            throw new IllegalArgumentException("tokens must not be empty");
        }
        int posOffset = curPos;
        int totalLen = posOffset + seqLen;
        if (totalLen > maxSeqLen) {
            throw new IllegalArgumentException(
                    "Context length " + totalLen + " exceeds configured KV-cache capacity " + maxSeqLen);
        }

        RolePlan rolePlan = null;
        boolean useSplitKvDecode = false;
        if (collective != null && rolePlanner != null) {
            WorkloadSpec tpSpec = buildTpSpec(phase, seqLen);
            SplitKvSpec kvSpec = buildSplitKvSpec(totalLen);
            rolePlan = rolePlanner.apply(tpSpec, kvSpec);
            if (phase == PhaseEvent.Phase.DECODE && seqLen == 1) {
                int activeWorkers = 0;
                for (SplitKvAssignment a : rolePlan.splitKvAssignments()) {
                    if (a.kvTokens() > 0) {
                        activeWorkers++;
                    }
                }
                useSplitKvDecode = (activeWorkers >= 2);
            }
        }

        int headDim = config.headDim();
        int numHeads = config.numAttentionHeads();
        int numKvHeads = config.numKeyValueHeads();
        int groupSize = numHeads / numKvHeads;
        float eps = config.rmsNormEps();
        float ropeTheta = config.ropeTheta();
        float scale = (float) (1.0 / Math.sqrt(headDim));

        List<ExecutionReport> layerReports = useSplitKvDecode
                ? new ArrayList<>(config.numHiddenLayers())
                : null;

        float[] logits;
        try (GpuScope outerScope = GpuScope.open()) {
            Tensor causalMask = null;
            if (seqLen > 1) {
                float[] maskData = new float[seqLen * totalLen];
                for (int qPos = 0; qPos < seqLen; qPos++) {
                    int absQ = posOffset + qPos;
                    int rowOffset = qPos * totalLen;
                    for (int kPos = 0; kPos < totalLen; kPos++) {
                        maskData[rowOffset + kPos] = (kPos <= absQ) ? 0.0f : -1e9f;
                    }
                }
                causalMask = outerScope.fromArray(maskData, 1, seqLen, totalLen);
            }

            Tensor tokenIds = outerScope.fromArray(tokens, seqLen);
            Tensor hidden = outerScope.track(weights.embedTokens().embedding(tokenIds));

            for (int l = 0; l < config.numHiddenLayers(); l++) {
                LlamaWeights.LayerWeights lw = weights.layer(l);
                try (GpuScope layerScope = GpuScope.open()) {
                    Tensor normX = layerScope.track(hidden.rmsNorm(lw.inputNorm(), eps));
                    Tensor q = layerScope.track(Nn.linearTransposed(normX, lw.qProj(), lw.qBias()));
                    Tensor k = layerScope.track(Nn.linearTransposed(normX, lw.kProj(), lw.kBias()));
                    Tensor v = layerScope.track(Nn.linearTransposed(normX, lw.vProj(), lw.vBias()));

                    Tensor q3 = layerScope.track(q.reshape(seqLen, numHeads, headDim));
                    Tensor k3 = layerScope.track(k.reshape(seqLen, numKvHeads, headDim));
                    Tensor v3 = layerScope.track(v.reshape(seqLen, numKvHeads, headDim));

                    Nn.rope(q3, posOffset, ropeTheta, q3);
                    Nn.rope(k3, posOffset, ropeTheta, k3);

                    Nn.kvCacheAppend(kCaches[l], k3, posOffset);
                    Nn.kvCacheAppend(vCaches[l], v3, posOffset);

                    Tensor kActive = layerScope.track(kCaches[l].slice(Slice.all(), Slice.to(totalLen), Slice.all()));
                    Tensor vActive = layerScope.track(vCaches[l].slice(Slice.all(), Slice.to(totalLen), Slice.all()));

                    Tensor attnFlat;
                    if (useSplitKvDecode) {
                        Tensor qDec = layerScope.track(q3.reshape(numHeads, headDim));
                        DistributedResult<Tensor> splitRes = DistributedExecutor.splitKvAttention(
                                collective, qDec, kActive, vActive, scale, rolePlan);
                        Tensor splitOut = layerScope.track(splitRes.value());
                        attnFlat = layerScope.track(splitOut.reshape(1, numHeads * headDim));
                        layerReports.add(splitRes.report());
                    } else {
                        Tensor kExp;
                        Tensor vExp;
                        if (groupSize == 1) {
                            kExp = kActive;
                            vExp = vActive;
                        } else {
                            kExp = layerScope.track(
                                    layerScope.track(
                                            layerScope.track(kActive.unsqueeze(1))
                                                    .broadcastTo(numKvHeads, groupSize, totalLen, headDim))
                                            .reshape(numHeads, totalLen, headDim));
                            vExp = layerScope.track(
                                    layerScope.track(
                                            layerScope.track(vActive.unsqueeze(1))
                                                    .broadcastTo(numKvHeads, groupSize, totalLen, headDim))
                                            .reshape(numHeads, totalLen, headDim));
                        }

                        Tensor qH = layerScope.track(q3.permute(1, 0, 2)); // [H, S, D_h]
                        Tensor kHT = layerScope.track(kExp.permute(0, 2, 1)); // [H, D_h, totalLen]
                        Tensor scores = layerScope.track(qH.matmul(kHT)); // [H, S, totalLen]
                        scores.muli(scale);
                        if (causalMask != null) {
                            scores.addi(causalMask);
                        }
                        Tensor probs = layerScope.track(Nn.softmax(scores)); // [H, S, totalLen]
                        Tensor ctx = layerScope.track(probs.matmul(vExp)); // [H, S, D_h]
                        Tensor ctxPerm = layerScope.track(ctx.permute(1, 0, 2)); // [S, H, D_h]
                        attnFlat = layerScope.track(ctxPerm.reshape(seqLen, numHeads * headDim)); // [S, D]
                    }

                    Tensor projOut = layerScope.track(Nn.linearTransposed(attnFlat, lw.oProj(), lw.oBias()));
                    Tensor outAttn = layerScope.track(hidden.add(projOut));

                    Tensor normX2 = layerScope.track(outAttn.rmsNorm(lw.postAttentionNorm(), eps));
                    Tensor gateProj = layerScope.track(Nn.linearTransposed(normX2, lw.gateProj(), null));
                    Tensor gate = layerScope.track(gateProj.silu());
                    Tensor up = layerScope.track(Nn.linearTransposed(normX2, lw.upProj(), null));
                    Tensor mlp = layerScope.track(gate.mul(up));
                    Tensor down = layerScope.track(Nn.linearTransposed(mlp, lw.downProj(), null));

                    Tensor nextHidden = outerScope.track(outAttn.add(down));
                    hidden.close();
                    hidden = nextHidden;
                }
            }

            Tensor lastTokenHidden;
            if (seqLen == 1) {
                lastTokenHidden = hidden;
            } else {
                Tensor lastSlice = outerScope.track(hidden.slice(Slice.range(seqLen - 1, seqLen), Slice.all()));
                lastTokenHidden = outerScope.track(lastSlice.contiguous());
            }

            Tensor finalNorm = outerScope.track(lastTokenHidden.rmsNorm(weights.finalNorm(), eps));
            Tensor logitsTensor = outerScope.track(Nn.linearTransposed(finalNorm, weights.lmHead(), null));
            logits = logitsTensor.toFloatArray();
        }

        curPos = totalLen;
        ExecutionReport stepReport = mergeReports(layerReports);
        morphTrace.add(new PhaseEvent(phase, seqLen, totalLen, rolePlan, stepReport));
        return logits;
    }

    private WorkloadSpec buildTpSpec(PhaseEvent.Phase phase, int seqLen) {
        long totalUnits = (phase == PhaseEvent.Phase.PREFILL)
                ? (long) seqLen * config.hiddenSize()
                : config.hiddenSize();
        double flopsPerUnit = 2.0 * config.intermediateSize();
        long bytesPerUnit = 4L * config.hiddenSize();
        long syncBytes = (long) seqLen * config.hiddenSize() * 4L;
        return WorkloadSpec.builder(totalUnits, flopsPerUnit, bytesPerUnit)
                .syncBytes(syncBytes)
                .ioMultiplier(1.0)
                .hostCpu(hostCpuProfile)
                .build();
    }

    private SplitKvSpec buildSplitKvSpec(int totalKvTokens) {
        long kvBytesPerToken = 2L * config.numHiddenLayers() * config.numKeyValueHeads() * config.headDim() * 2L;
        double attnFlops = 4.0 * config.numAttentionHeads() * config.headDim();
        return new SplitKvSpec(totalKvTokens, kvBytesPerToken, attnFlops, 1024L, 100_000_000L, 0.85);
    }

    private static ExecutionReport mergeReports(List<ExecutionReport> reports) {
        if (reports == null || reports.isEmpty()) {
            return null;
        }
        if (reports.size() == 1) {
            return reports.getFirst();
        }
        ExecutionReport first = reports.getFirst();
        double wall = 0.0;
        double compute = 0.0;
        double sync = 0.0;
        long cpuUnits = 0L;
        Set<Integer> executed = new LinkedHashSet<>();
        Set<Integer> pruned = new LinkedHashSet<>();
        for (ExecutionReport r : reports) {
            wall += r.wallTimeMs();
            compute += r.computeTimeMs();
            sync += r.syncTimeMs();
            cpuUnits += r.cpuInPlaceUnits();
            executed.addAll(r.executedGpuRanks());
            pruned.addAll(r.prunedGpuRanks());
        }
        return new ExecutionReport(
                first.operation(),
                first.workloadPlan(),
                first.rolePlan(),
                wall,
                compute,
                sync,
                List.copyOf(executed),
                List.copyOf(pruned),
                cpuUnits
        );
    }

    private void releaseCaches() {
        for (int i = 0; i < kCaches.length; i++) {
            closeQuietly(kCaches[i]);
            kCaches[i] = null;
            closeQuietly(vCaches[i]);
            vCaches[i] = null;
        }
    }

    private static void closeQuietly(Tensor t) {
        if (t != null && !t.isClosed()) {
            try {
                t.close();
            } catch (Throwable ignored) {
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("LlamaRunner has already been closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            curPos = 0;
            morphTrace.clear();
            releaseCaches();
        }
    }
}
