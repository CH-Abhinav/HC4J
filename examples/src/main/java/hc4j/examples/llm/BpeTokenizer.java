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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Pure-Java zero-dependency Byte-Pair Encoding (BPE) tokenizer loading HuggingFace {@code tokenizer.json}.
 * <p>
 * Supports Llama-3.2, Qwen2.5, SmolLM2, GPT-2, and Mistral tokenizers, including both legacy
 * {@code "a b"} string merges and modern {@code ["a", "b"]} array merges, added special tokens,
 * lossless GPT-2 ByteLevel BPE encoding/decoding, and SentencePiece Metaspace ({@code \u2581}) fallback.
 */
public final class BpeTokenizer {

    private record MergePair(String left, String right) {}

    private enum Mode {
        BYTE_LEVEL,
        METASPACE
    }

    // Standard GPT-2 pre-tokenization pattern separating contractions, words (with leading space), numbers, punctuation, and whitespace
    private static final Pattern GPT2_PRETOKEN_PATTERN = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
            Pattern.UNICODE_CHARACTER_CLASS);

    private static final Pattern DIGITS_INDIVIDUAL_PATTERN = Pattern.compile("\\p{N}");
    private static final Pattern DIGITS_GROUPED_PATTERN = Pattern.compile("\\p{N}+");

    private sealed interface PreTokenStep permits SplitStep, ByteLevelStep, MetaspaceStep {}

    private record SplitStep(Pattern pattern, String behavior, boolean invert) implements PreTokenStep {}

    private record ByteLevelStep(boolean addPrefixSpace, boolean useRegex) implements PreTokenStep {}

    private record MetaspaceStep(char replacement, boolean addPrefixSpace) implements PreTokenStep {}

    private static final char[] BYTE_TO_CHAR = new char[256];
    private static final Map<Character, Byte> CHAR_TO_BYTE = new HashMap<>(512);

    static {
        boolean[] keep = new boolean[256];
        for (int b = '!'; b <= '~'; b++) keep[b] = true;
        for (int b = 0xA1; b <= 0xAC; b++) keep[b] = true;
        for (int b = 0xAE; b <= 0xFF; b++) keep[b] = true;
        int extra = 0;
        for (int b = 0; b < 256; b++) {
            char mapped = keep[b] ? (char) b : (char) (256 + extra++);
            BYTE_TO_CHAR[b] = mapped;
            CHAR_TO_BYTE.put(mapped, (byte) b);
        }
    }

    private static final List<String> BOS_CANDIDATES = List.of(
            "<|begin_of_text|>", "<s>", "<bos>", "<|startoftext|>");
    private static final List<String> EOS_CANDIDATES = List.of(
            "<|end_of_text|>", "<|eot_id|>", "<|im_end|>", "<|endoftext|>", "</s>", "<eos>", "<|end|>");

    private final Map<String, Integer> tokenToId;
    private final String[] idToToken;
    private final Map<MergePair, Integer> mergeRanks;
    private final Map<String, Integer> addedTokens;
    private final Set<Integer> specialTokenIds;
    private final Pattern addedTokenPattern;
    private final Mode mode;
    private final List<Normalizer.Form> normalizerForms;
    private final List<PreTokenStep> preTokenSteps;
    private final char metaspaceChar;
    private final boolean metaspaceAddPrefixSpace;
    private final boolean ignoreMerges;
    private final Integer unkTokenId;
    private final OptionalInt bosTokenId;
    private final Set<Integer> eosTokenIds;
    private final Map<String, int[]> pieceCache = new ConcurrentHashMap<>();

    private BpeTokenizer(
            Map<String, Integer> tokenToId,
            String[] idToToken,
            Map<MergePair, Integer> mergeRanks,
            Map<String, Integer> addedTokens,
            Set<Integer> specialTokenIds,
            Pattern addedTokenPattern,
            Mode mode,
            List<Normalizer.Form> normalizerForms,
            List<PreTokenStep> preTokenSteps,
            char metaspaceChar,
            boolean metaspaceAddPrefixSpace,
            boolean ignoreMerges,
            Integer unkTokenId,
            OptionalInt bosTokenId,
            Set<Integer> eosTokenIds
    ) {
        this.tokenToId = Collections.unmodifiableMap(tokenToId);
        this.idToToken = idToToken;
        this.mergeRanks = Collections.unmodifiableMap(mergeRanks);
        this.addedTokens = Collections.unmodifiableMap(addedTokens);
        this.specialTokenIds = Collections.unmodifiableSet(specialTokenIds);
        this.addedTokenPattern = addedTokenPattern;
        this.mode = mode;
        this.normalizerForms = List.copyOf(normalizerForms);
        this.preTokenSteps = List.copyOf(preTokenSteps);
        this.metaspaceChar = metaspaceChar;
        this.metaspaceAddPrefixSpace = metaspaceAddPrefixSpace;
        this.ignoreMerges = ignoreMerges;
        this.unkTokenId = unkTokenId;
        this.bosTokenId = bosTokenId;
        this.eosTokenIds = Collections.unmodifiableSet(eosTokenIds);
    }

    /**
     * Loads a {@link BpeTokenizer} from a HuggingFace {@code tokenizer.json} file.
     *
     * @param tokenizerJsonPath path to {@code tokenizer.json}
     * @return loaded tokenizer instance
     * @throws IOException if reading the file fails
     */
    public static BpeTokenizer load(Path tokenizerJsonPath) throws IOException {
        Objects.requireNonNull(tokenizerJsonPath, "tokenizerJsonPath");
        String json = Files.readString(tokenizerJsonPath, StandardCharsets.UTF_8);
        return fromJson(json);
    }

    /**
     * Parses a {@link BpeTokenizer} directly from a {@code tokenizer.json} string.
     *
     * @param json HuggingFace {@code tokenizer.json} contents
     * @return parsed tokenizer instance
     */
    public static BpeTokenizer fromJson(String json) {
        Objects.requireNonNull(json, "json");
        Map<String, Object> root = MinimalJson.parseObject(json);

        Object modelObj = root.get("model");
        if (!(modelObj instanceof Map<?, ?> model)) {
            throw new IllegalArgumentException("tokenizer.json missing 'model' object");
        }
        Object vocabObj = model.get("vocab");
        if (!(vocabObj instanceof Map<?, ?> vocabMap)) {
            throw new IllegalArgumentException("tokenizer.json missing 'model.vocab' object");
        }

        Map<String, Integer> tokenToId = new HashMap<>(vocabMap.size() * 2);
        int maxId = -1;
        for (Map.Entry<?, ?> entry : vocabMap.entrySet()) {
            String tok = String.valueOf(entry.getKey());
            int id = ((Number) entry.getValue()).intValue();
            tokenToId.put(tok, id);
            if (id > maxId) maxId = id;
        }

        Map<String, Integer> addedTokens = new HashMap<>();
        Set<Integer> specialTokenIds = new LinkedHashSet<>();
        Object addedObj = root.get("added_tokens");
        if (addedObj instanceof List<?> addedList) {
            for (Object item : addedList) {
                if (item instanceof Map<?, ?> entry) {
                    String content = String.valueOf(entry.get("content"));
                    int id = ((Number) entry.get("id")).intValue();
                    boolean special = Boolean.TRUE.equals(entry.get("special"));
                    tokenToId.put(content, id);
                    addedTokens.put(content, id);
                    if (special) {
                        specialTokenIds.add(id);
                    }
                    if (id > maxId) maxId = id;
                }
            }
        }

        String[] idToToken = new String[Math.max(0, maxId + 1)];
        for (Map.Entry<String, Integer> entry : tokenToId.entrySet()) {
            int id = entry.getValue();
            if (id >= 0 && id < idToToken.length) {
                idToToken[id] = entry.getKey();
            }
        }

        Map<MergePair, Integer> mergeRanks = new HashMap<>();
        Object mergesObj = model.get("merges");
        if (mergesObj instanceof List<?> mergesList) {
            int rank = 0;
            for (Object item : mergesList) {
                if (item instanceof String s) {
                    int space = s.indexOf(' ');
                    if (space <= 0 || space >= s.length() - 1) {
                        throw new IllegalArgumentException("Malformed string merge entry: '" + s + "'");
                    }
                    mergeRanks.put(new MergePair(s.substring(0, space), s.substring(space + 1)), rank++);
                } else if (item instanceof List<?> pair && pair.size() == 2) {
                    mergeRanks.put(new MergePair(String.valueOf(pair.get(0)), String.valueOf(pair.get(1))), rank++);
                } else {
                    throw new IllegalArgumentException("Unsupported merge entry format: " + item);
                }
            }
        }

        Pattern addedTokenPattern = compileAddedTokenPattern(addedTokens.keySet());

        List<Normalizer.Form> normalizerForms = parseNormalizerForms(root.get("normalizer"));
        TokenizerLayoutConfig layout = detectLayout(root, tokenToId);
        boolean ignoreMerges = Boolean.TRUE.equals(model.get("ignore_merges"));

        Integer unkId = null;
        Object unkObj = model.get("unk_token");
        if (unkObj instanceof String unkTok) {
            unkId = tokenToId.get(unkTok);
        }

        OptionalInt bosId = detectBosTokenId(root, addedTokens, specialTokenIds);
        Set<Integer> eosIds = detectEosTokenIds(root, addedTokens, specialTokenIds);

        return new BpeTokenizer(
                tokenToId, idToToken, mergeRanks, addedTokens, specialTokenIds, addedTokenPattern,
                layout.mode, normalizerForms, layout.preTokenSteps,
                layout.metaspaceChar, layout.metaspaceAddPrefixSpace, ignoreMerges,
                unkId, bosId, eosIds
        );
    }

    /**
     * Encodes {@code text} into an array of vocabulary token IDs without prepending BOS.
     *
     * @param text input text
     * @return encoded token IDs
     */
    public int[] encode(String text) {
        return encode(text, false);
    }

    /**
     * Encodes {@code text} into an array of vocabulary token IDs, optionally prepending BOS.
     *
     * @param text   input text
     * @param addBos whether to prepend the detected BOS token ID (if present)
     * @return encoded token IDs
     */
    public int[] encode(String text, boolean addBos) {
        Objects.requireNonNull(text, "text");
        List<Integer> out = new ArrayList<>();

        if (addedTokenPattern == null) {
            encodeNormalChunk(text, out);
        } else {
            Matcher m = addedTokenPattern.matcher(text);
            int lastEnd = 0;
            while (m.find()) {
                if (m.start() > lastEnd) {
                    encodeNormalChunk(text.substring(lastEnd, m.start()), out);
                }
                String special = m.group();
                Integer specialId = addedTokens.get(special);
                if (specialId != null) {
                    out.add(specialId);
                }
                lastEnd = m.end();
            }
            if (lastEnd < text.length()) {
                encodeNormalChunk(text.substring(lastEnd), out);
            }
        }

        if (addBos && bosTokenId.isPresent()) {
            int bos = bosTokenId.getAsInt();
            if (out.isEmpty() || out.getFirst() != bos) {
                out.addFirst(bos);
            }
        }

        int[] result = new int[out.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = out.get(i);
        }
        return result;
    }

    /**
     * Decodes a sequence of token IDs back to text, retaining all tokens.
     *
     * @param tokenIds token ID array
     * @return decoded text string
     */
    public String decode(int[] tokenIds) {
        return decode(tokenIds, false);
    }

    /**
     * Decodes a sequence of token IDs back to text, optionally omitting special tokens.
     *
     * @param tokenIds          token ID array
     * @param skipSpecialTokens true to skip tokens marked special in {@code added_tokens}
     * @return decoded text string
     */
    public String decode(int[] tokenIds, boolean skipSpecialTokens) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        if (mode == Mode.METASPACE) {
            StringBuilder sb = new StringBuilder();
            for (int id : tokenIds) {
                if (id < 0 || id >= idToToken.length || idToToken[id] == null) {
                    continue;
                }
                if (skipSpecialTokens && specialTokenIds.contains(id)) {
                    continue;
                }
                if (specialTokenIds.contains(id)) {
                    sb.append(idToToken[id]);
                } else {
                    sb.append(idToToken[id].replace(metaspaceChar, ' '));
                }
            }
            String decoded = sb.toString();
            if (metaspaceAddPrefixSpace && decoded.startsWith(" ")) {
                decoded = decoded.substring(1);
            }
            return decoded;
        }

        StringBuilder sb = new StringBuilder();
        ByteArrayOutputStream byteBuf = new ByteArrayOutputStream();
        for (int id : tokenIds) {
            if (id < 0 || id >= idToToken.length || idToToken[id] == null) {
                continue;
            }
            if (specialTokenIds.contains(id)) {
                flushUtf8Bytes(byteBuf, sb);
                if (!skipSpecialTokens) {
                    sb.append(idToToken[id]);
                }
                continue;
            }
            appendPieceBytes(idToToken[id], byteBuf, sb);
        }
        flushUtf8Bytes(byteBuf, sb);
        return sb.toString();
    }

    /**
     * Decodes a single token ID standalone, returning an empty string for special tokens.
     * For split multi-byte UTF-8 characters across consecutive tokens during live streaming,
     * prefer {@link #newStreamDecoder()}.
     *
     * @param tokenId single token ID
     * @return decoded token string (empty if out of bounds or special)
     */
    public String decodeToken(int tokenId) {
        if (tokenId < 0 || tokenId >= idToToken.length || idToToken[tokenId] == null) {
            return "";
        }
        if (specialTokenIds.contains(tokenId)) {
            return "";
        }
        String piece = idToToken[tokenId];
        if (mode == Mode.METASPACE) {
            return piece.replace(metaspaceChar, ' ');
        }
        StringBuilder sb = new StringBuilder();
        ByteArrayOutputStream byteBuf = new ByteArrayOutputStream();
        appendPieceBytes(piece, byteBuf, sb);
        flushUtf8Bytes(byteBuf, sb);
        return sb.toString();
    }

    /**
     * Creates a stateful streaming token decoder that buffers incomplete multi-byte UTF-8
     * sequences across adjacent tokens.
     *
     * @return new streaming decoder
     */
    public StreamDecoder newStreamDecoder() {
        return new StreamDecoder();
    }

    /**
     * Stateful streaming token decoder handling split multi-byte UTF-8 sequences cleanly.
     */
    public final class StreamDecoder {
        private final ByteArrayOutputStream pendingBytes = new ByteArrayOutputStream();
        private boolean emittedNonSpecial = false;

        private StreamDecoder() {}

        /**
         * Consumes the next generated token ID and returns any newly completed UTF-8 text.
         *
         * @param tokenId next generated token ID
         * @return decoded text ready to emit (may be empty if buffering partial UTF-8 bytes or special)
         */
        public String next(int tokenId) {
            if (tokenId < 0 || tokenId >= idToToken.length || idToToken[tokenId] == null) {
                return "";
            }
            if (specialTokenIds.contains(tokenId)) {
                return flush();
            }
            if (mode == Mode.METASPACE) {
                String decoded = idToToken[tokenId].replace(metaspaceChar, ' ');
                if (!emittedNonSpecial && metaspaceAddPrefixSpace && decoded.startsWith(" ")) {
                    decoded = decoded.substring(1);
                }
                emittedNonSpecial = true;
                return decoded;
            }
            emittedNonSpecial = true;
            StringBuilder out = new StringBuilder();
            appendPieceBytes(idToToken[tokenId], pendingBytes, out);
            byte[] buf = pendingBytes.toByteArray();
            int completeLen = completeUtf8PrefixLength(buf);
            if (completeLen > 0) {
                out.append(new String(buf, 0, completeLen, StandardCharsets.UTF_8));
                pendingBytes.reset();
                if (completeLen < buf.length) {
                    pendingBytes.write(buf, completeLen, buf.length - completeLen);
                }
            }
            return out.toString();
        }

        /**
         * Flushes any trailing buffered bytes at the end of generation.
         *
         * @return remaining decoded text
         */
        public String flush() {
            if (pendingBytes.size() == 0) {
                return "";
            }
            String rem = pendingBytes.toString(StandardCharsets.UTF_8);
            pendingBytes.reset();
            return rem;
        }
    }

    private static int completeUtf8PrefixLength(byte[] buf) {
        int i = 0;
        while (i < buf.length) {
            int b = buf[i] & 0xFF;
            int need;
            if ((b & 0x80) == 0) {
                need = 1;
            } else if ((b & 0xE0) == 0xC0) {
                need = 2;
            } else if ((b & 0xF0) == 0xE0) {
                need = 3;
            } else if ((b & 0xF8) == 0xF0) {
                need = 4;
            } else {
                need = 1;
            }
            if (i + need > buf.length) {
                break;
            }
            i += need;
        }
        return i;
    }

    /**
     * Returns total vocabulary size (including added tokens).
     *
     * @return vocabulary size
     */
    public int vocabSize() {
        return idToToken.length;
    }

    /**
     * Returns true if {@code id} corresponds to a special token.
     *
     * @param id token ID
     * @return true if marked special
     */
    public boolean isSpecialToken(int id) {
        return specialTokenIds.contains(id);
    }

    /**
     * Returns the detected beginning-of-sequence token ID if present.
     *
     * @return optional BOS token ID
     */
    public OptionalInt bosTokenId() {
        return bosTokenId;
    }

    /**
     * Returns the unmodifiable set of detected end-of-sequence token IDs.
     *
     * @return set of EOS token IDs
     */
    public Set<Integer> eosTokenIds() {
        return eosTokenIds;
    }

    // --- Internal BPE & Pre-Tokenization Helpers ---

    private void encodeNormalChunk(String text, List<Integer> out) {
        if (text.isEmpty()) {
            return;
        }
        String normalized = text;
        for (Normalizer.Form form : normalizerForms) {
            normalized = Normalizer.normalize(normalized, form);
        }

        List<String> chunks = List.of(normalized);
        boolean encodedToTargetAlphabet = false;
        for (PreTokenStep step : preTokenSteps) {
            switch (step) {
                case SplitStep s -> chunks = applySplit(chunks, s);
                case ByteLevelStep bl -> {
                    chunks = applyByteLevel(chunks, bl, encodedToTargetAlphabet);
                    encodedToTargetAlphabet = true;
                }
                case MetaspaceStep ms -> {
                    chunks = applyMetaspace(chunks, ms);
                    encodedToTargetAlphabet = true;
                }
            }
        }

        if (!encodedToTargetAlphabet && mode == Mode.BYTE_LEVEL) {
            List<String> converted = new ArrayList<>(chunks.size());
            for (String chunk : chunks) {
                if (!chunk.isEmpty()) {
                    converted.add(toByteLevelUnicode(chunk));
                }
            }
            chunks = converted;
        }

        for (String piece : chunks) {
            runBpeOnPiece(piece, out);
        }
    }

    private static List<String> applySplit(List<String> inputs, SplitStep step) {
        List<String> result = new ArrayList<>(inputs.size() * 2);
        for (String chunk : inputs) {
            if (chunk.isEmpty()) {
                continue;
            }
            splitChunk(chunk, step.pattern(), step.behavior(), step.invert(), result);
        }
        return result;
    }

    private static void splitChunk(String text, Pattern pattern, String behavior, boolean invert, List<String> out) {
        Matcher matcher = pattern.matcher(text);
        if ("Removed".equalsIgnoreCase(behavior)) {
            int lastEnd = 0;
            while (matcher.find()) {
                if (invert) {
                    if (matcher.start() < matcher.end()) {
                        out.add(matcher.group());
                    }
                } else if (matcher.start() > lastEnd) {
                    out.add(text.substring(lastEnd, matcher.start()));
                }
                lastEnd = matcher.end();
            }
            if (!invert && lastEnd < text.length()) {
                out.add(text.substring(lastEnd));
            }
            return;
        }

        if ("MergedWithPrevious".equalsIgnoreCase(behavior)) {
            int startOutIdx = out.size();
            int lastEnd = 0;
            while (matcher.find()) {
                String gap = text.substring(lastEnd, matcher.start());
                if (!gap.isEmpty()) {
                    out.add(gap);
                }
                String hit = matcher.group();
                if (!hit.isEmpty()) {
                    if (invert || out.size() == startOutIdx) {
                        out.add(hit);
                    } else {
                        int lastIdx = out.size() - 1;
                        out.set(lastIdx, out.get(lastIdx) + hit);
                    }
                }
                lastEnd = matcher.end();
            }
            if (lastEnd < text.length()) {
                out.add(text.substring(lastEnd));
            }
            return;
        }

        // Default ("Isolated", "Contiguous", "MergedWithNext" fallback): emit gaps and matches cleanly in order
        int lastEnd = 0;
        while (matcher.find()) {
            if (matcher.start() > lastEnd) {
                out.add(text.substring(lastEnd, matcher.start()));
            }
            if (!matcher.group().isEmpty()) {
                out.add(matcher.group());
            }
            lastEnd = matcher.end();
        }
        if (lastEnd < text.length()) {
            out.add(text.substring(lastEnd));
        }
    }

    private static List<String> applyByteLevel(List<String> inputs, ByteLevelStep step, boolean alreadyByteLevel) {
        List<String> current = inputs;
        if (step.addPrefixSpace() && !alreadyByteLevel && !current.isEmpty()) {
            List<String> prefixed = new ArrayList<>(current);
            String first = prefixed.getFirst();
            if (!first.isEmpty() && first.charAt(0) != ' ') {
                prefixed.set(0, " " + first);
            }
            current = prefixed;
        }
        if (step.useRegex()) {
            List<String> split = new ArrayList<>(current.size() * 4);
            for (String chunk : current) {
                if (!chunk.isEmpty()) {
                    splitChunk(chunk, GPT2_PRETOKEN_PATTERN, "Isolated", false, split);
                }
            }
            current = split;
        }
        if (alreadyByteLevel) {
            return current;
        }
        List<String> out = new ArrayList<>(current.size());
        for (String chunk : current) {
            if (!chunk.isEmpty()) {
                out.add(toByteLevelUnicode(chunk));
            }
        }
        return out;
    }

    private static List<String> applyMetaspace(List<String> inputs, MetaspaceStep step) {
        List<String> out = new ArrayList<>(inputs.size());
        boolean firstPiece = true;
        for (String chunk : inputs) {
            if (chunk.isEmpty()) {
                continue;
            }
            String replaced = chunk.replace(' ', step.replacement());
            if (firstPiece && step.addPrefixSpace() && !replaced.isEmpty() && replaced.charAt(0) != step.replacement()) {
                out.add(step.replacement() + replaced);
            } else {
                out.add(replaced);
            }
            firstPiece = false;
        }
        return out;
    }

    private static String toByteLevelUnicode(String raw) {
        byte[] bytes = raw.getBytes(StandardCharsets.UTF_8);
        StringBuilder sb = new StringBuilder(bytes.length);
        for (byte b : bytes) {
            sb.append(BYTE_TO_CHAR[b & 0xFF]);
        }
        return sb.toString();
    }

    private void runBpeOnPiece(String unicodePiece, List<Integer> out) {
        if (unicodePiece.isEmpty()) {
            return;
        }
        Integer wholeId = tokenToId.get(unicodePiece);
        if (wholeId != null && (ignoreMerges || Character.codePointCount(unicodePiece, 0, unicodePiece.length()) == 1)) {
            out.add(wholeId);
            return;
        }
        int[] cached = pieceCache.get(unicodePiece);
        if (cached != null) {
            for (int id : cached) {
                out.add(id);
            }
            return;
        }

        List<String> symbols = new ArrayList<>(unicodePiece.length());
        for (int i = 0; i < unicodePiece.length(); ) {
            int cp = unicodePiece.codePointAt(i);
            int charCount = Character.charCount(cp);
            symbols.add(unicodePiece.substring(i, i + charCount));
            i += charCount;
        }

        while (symbols.size() > 1) {
            MergePair bestPair = null;
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i < symbols.size() - 1; i++) {
                MergePair cand = new MergePair(symbols.get(i), symbols.get(i + 1));
                Integer rank = mergeRanks.get(cand);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestPair = cand;
                }
            }
            if (bestPair == null) {
                break;
            }

            List<String> merged = new ArrayList<>(symbols.size());
            int i = 0;
            while (i < symbols.size()) {
                if (i < symbols.size() - 1
                        && symbols.get(i).equals(bestPair.left())
                        && symbols.get(i + 1).equals(bestPair.right())) {
                    merged.add(bestPair.left() + bestPair.right());
                    i += 2;
                } else {
                    merged.add(symbols.get(i));
                    i += 1;
                }
            }
            symbols = merged;
        }

        int[] resolved = new int[symbols.size()];
        for (int idx = 0; idx < symbols.size(); idx++) {
            String sym = symbols.get(idx);
            Integer id = tokenToId.get(sym);
            if (id != null) {
                resolved[idx] = id;
            } else if (unkTokenId != null) {
                resolved[idx] = unkTokenId;
            } else {
                throw new IllegalArgumentException("Token piece not found in vocabulary: '" + sym + "'");
            }
            out.add(resolved[idx]);
        }
        if (pieceCache.size() < 16384) {
            pieceCache.putIfAbsent(unicodePiece, resolved);
        }
    }

    private static void appendPieceBytes(String piece, ByteArrayOutputStream byteBuf, StringBuilder sb) {
        for (int i = 0; i < piece.length(); i++) {
            char ch = piece.charAt(i);
            Byte b = CHAR_TO_BYTE.get(ch);
            if (b != null) {
                byteBuf.write(b);
            } else {
                flushUtf8Bytes(byteBuf, sb);
                sb.append(ch);
            }
        }
    }

    private static void flushUtf8Bytes(ByteArrayOutputStream byteBuf, StringBuilder sb) {
        if (byteBuf.size() > 0) {
            sb.append(byteBuf.toString(StandardCharsets.UTF_8));
            byteBuf.reset();
        }
    }

    private static Pattern compileAddedTokenPattern(Set<String> tokens) {
        if (tokens.isEmpty()) {
            return null;
        }
        List<String> sorted = new ArrayList<>();
        for (String t : tokens) {
            if (t != null && !t.isEmpty()) {
                sorted.add(t);
            }
        }
        if (sorted.isEmpty()) {
            return null;
        }
        sorted.sort(Comparator.comparingInt(String::length).reversed());
        StringBuilder regex = new StringBuilder();
        for (int i = 0; i < sorted.size(); i++) {
            if (i > 0) regex.append('|');
            regex.append(Pattern.quote(sorted.get(i)));
        }
        return Pattern.compile(regex.toString());
    }

    private record TokenizerLayoutConfig(
            Mode mode,
            List<PreTokenStep> preTokenSteps,
            char metaspaceChar,
            boolean metaspaceAddPrefixSpace
    ) {}

    private static List<Normalizer.Form> parseNormalizerForms(Object normalizerObj) {
        List<Normalizer.Form> forms = new ArrayList<>();
        collectNormalizerForms(normalizerObj, forms);
        return forms;
    }

    private static void collectNormalizerForms(Object node, List<Normalizer.Form> out) {
        if (!(node instanceof Map<?, ?> map)) {
            return;
        }
        Object typeObj = map.get("type");
        if (typeObj instanceof String type) {
            switch (type.toUpperCase()) {
                case "NFC" -> out.add(Normalizer.Form.NFC);
                case "NFD" -> out.add(Normalizer.Form.NFD);
                case "NFKC" -> out.add(Normalizer.Form.NFKC);
                case "NFKD" -> out.add(Normalizer.Form.NFKD);
                case "SEQUENCE" -> {
                    if (map.get("normalizers") instanceof List<?> list) {
                        for (Object child : list) {
                            collectNormalizerForms(child, out);
                        }
                    }
                }
                default -> {}
            }
        }
    }

    private static TokenizerLayoutConfig detectLayout(Map<String, Object> root, Map<String, Integer> tokenToId) {
        List<PreTokenStep> steps = new ArrayList<>();
        parsePreTokenizerSteps(root.get("pre_tokenizer"), steps);

        List<Map<?, ?>> decoderNodes = new ArrayList<>();
        collectTypedNodes(root.get("decoder"), decoderNodes);
        List<Map<?, ?>> preNodes = new ArrayList<>();
        collectTypedNodes(root.get("pre_tokenizer"), preNodes);

        boolean sawByteLevel = false;
        boolean sawMetaspace = false;
        char metaspaceChar = '\u2581';
        boolean metaspaceAddPrefix = false;

        for (Map<?, ?> node : decoderNodes) {
            if (!(node.get("type") instanceof String type)) {
                continue;
            }
            if ("ByteLevel".equalsIgnoreCase(type)) {
                sawByteLevel = true;
            } else if ("Metaspace".equalsIgnoreCase(type)) {
                sawMetaspace = true;
                metaspaceChar = parseMetaspaceChar(node, metaspaceChar);
                metaspaceAddPrefix = parseMetaspaceAddPrefix(node, metaspaceAddPrefix);
            }
        }

        for (Map<?, ?> node : preNodes) {
            if (!(node.get("type") instanceof String type)) {
                continue;
            }
            if ("ByteLevel".equalsIgnoreCase(type)) {
                sawByteLevel = true;
            } else if ("Metaspace".equalsIgnoreCase(type)) {
                sawMetaspace = true;
                metaspaceChar = parseMetaspaceChar(node, metaspaceChar);
                metaspaceAddPrefix = parseMetaspaceAddPrefix(node, metaspaceAddPrefix);
            }
        }

        Mode mode;
        if (sawByteLevel) {
            mode = Mode.BYTE_LEVEL;
        } else if (sawMetaspace) {
            mode = Mode.METASPACE;
        } else {
            boolean hasByteSpace = false;
            boolean hasMetaspaceChar = false;
            for (String tok : tokenToId.keySet()) {
                if (tok.indexOf('\u0120') >= 0) {
                    hasByteSpace = true;
                    break;
                }
                if (tok.indexOf('\u2581') >= 0) {
                    hasMetaspaceChar = true;
                }
            }
            mode = hasByteSpace ? Mode.BYTE_LEVEL : (hasMetaspaceChar ? Mode.METASPACE : Mode.BYTE_LEVEL);
        }

        if (steps.isEmpty()) {
            if (mode == Mode.METASPACE) {
                steps.add(new MetaspaceStep(metaspaceChar, metaspaceAddPrefix));
            } else {
                steps.add(new ByteLevelStep(false, true));
            }
        }

        return new TokenizerLayoutConfig(mode, steps, metaspaceChar, metaspaceAddPrefix);
    }

    private static void parsePreTokenizerSteps(Object node, List<PreTokenStep> steps) {
        if (!(node instanceof Map<?, ?> map)) {
            return;
        }
        if (!(map.get("type") instanceof String type)) {
            return;
        }
        switch (type) {
            case "Sequence" -> {
                if (map.get("pretokenizers") instanceof List<?> children) {
                    for (Object child : children) {
                        parsePreTokenizerSteps(child, steps);
                    }
                }
            }
            case "Split" -> {
                Pattern pat = parseSplitPattern(map.get("pattern"));
                if (pat != null) {
                    String behavior = (map.get("behavior") instanceof String b) ? b : "Isolated";
                    boolean invert = Boolean.TRUE.equals(map.get("invert"));
                    steps.add(new SplitStep(pat, behavior, invert));
                }
            }
            case "Digits" -> {
                boolean individual = Boolean.TRUE.equals(map.get("individual_digits"));
                steps.add(new SplitStep(individual ? DIGITS_INDIVIDUAL_PATTERN : DIGITS_GROUPED_PATTERN, "Isolated", false));
            }
            case "ByteLevel" -> {
                boolean addPrefixSpace = Boolean.TRUE.equals(map.get("add_prefix_space"));
                boolean useRegex = !(map.get("use_regex") instanceof Boolean b) || b;
                steps.add(new ByteLevelStep(addPrefixSpace, useRegex));
            }
            case "Metaspace" -> {
                char rep = parseMetaspaceChar(map, '\u2581');
                boolean addPrefix = parseMetaspaceAddPrefix(map, false);
                steps.add(new MetaspaceStep(rep, addPrefix));
            }
            default -> {}
        }
    }

    private static Pattern parseSplitPattern(Object patternObj) {
        if (!(patternObj instanceof Map<?, ?> patMap)) {
            return null;
        }
        if (patMap.get("String") instanceof String literal) {
            return Pattern.compile(Pattern.quote(literal));
        }
        if (patMap.get("Regex") instanceof String regex) {
            try {
                return Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
            } catch (PatternSyntaxException ignored) {
                try {
                    return Pattern.compile(regex);
                } catch (PatternSyntaxException fallback) {
                    return GPT2_PRETOKEN_PATTERN;
                }
            }
        }
        return null;
    }

    private static char parseMetaspaceChar(Map<?, ?> node, char fallback) {
        Object repObj = node.get("replacement");
        if (repObj instanceof String s && !s.isEmpty()) {
            return s.charAt(0);
        }
        return fallback;
    }

    private static boolean parseMetaspaceAddPrefix(Map<?, ?> node, boolean fallback) {
        Object addPrefixObj = node.get("add_prefix_space");
        if (addPrefixObj instanceof Boolean b) {
            return b;
        }
        Object schemeObj = node.get("prepend_scheme");
        if (schemeObj instanceof String scheme) {
            return "always".equalsIgnoreCase(scheme) || "first".equalsIgnoreCase(scheme);
        }
        return fallback;
    }

    private static void collectTypedNodes(Object node, List<Map<?, ?>> out) {
        if (node instanceof Map<?, ?> map) {
            out.add(map);
            for (Object val : map.values()) {
                if (val instanceof Map<?, ?> || val instanceof List<?>) {
                    collectTypedNodes(val, out);
                }
            }
        } else if (node instanceof List<?> list) {
            for (Object item : list) {
                collectTypedNodes(item, out);
            }
        }
    }

    private static OptionalInt detectBosTokenId(
            Map<String, Object> root,
            Map<String, Integer> addedTokens,
            Set<Integer> specialTokenIds
    ) {
        Object postObj = root.get("post_processor");
        if (postObj instanceof Map<?, ?> post) {
            Integer id = inspectTemplateProcessingSpecial(post, true);
            if (id != null) {
                return OptionalInt.of(id);
            }
        }
        for (String cand : BOS_CANDIDATES) {
            Integer id = addedTokens.get(cand);
            if (id != null && specialTokenIds.contains(id)) {
                return OptionalInt.of(id);
            }
        }
        return OptionalInt.empty();
    }

    private static Set<Integer> detectEosTokenIds(
            Map<String, Object> root,
            Map<String, Integer> addedTokens,
            Set<Integer> specialTokenIds
    ) {
        Set<Integer> eos = new LinkedHashSet<>();
        for (String cand : EOS_CANDIDATES) {
            Integer id = addedTokens.get(cand);
            if (id != null && specialTokenIds.contains(id)) {
                eos.add(id);
            }
        }
        Object postObj = root.get("post_processor");
        if (postObj instanceof Map<?, ?> post) {
            Integer id = inspectTemplateProcessingSpecial(post, false);
            if (id != null) {
                eos.add(id);
            }
        }
        return eos;
    }

    private static Integer inspectTemplateProcessingSpecial(Map<?, ?> post, boolean wantBos) {
        Object specialObj = post.get("special_tokens");
        Object singleObj = post.get("single");
        if (!(specialObj instanceof Map<?, ?> specialMap) || !(singleObj instanceof List<?> singleList)) {
            return null;
        }
        int seqIdx = -1;
        for (int i = 0; i < singleList.size(); i++) {
            if (singleList.get(i) instanceof Map<?, ?> item && item.containsKey("Sequence")) {
                seqIdx = i;
                break;
            }
        }
        if (seqIdx < 0) {
            return null;
        }
        for (int i = 0; i < singleList.size(); i++) {
            if (wantBos && i >= seqIdx) continue;
            if (!wantBos && i <= seqIdx) continue;
            if (singleList.get(i) instanceof Map<?, ?> item && item.get("SpecialToken") instanceof Map<?, ?> st) {
                String tokKey = String.valueOf(st.get("id"));
                if (specialMap.get(tokKey) instanceof Map<?, ?> entry && entry.get("ids") instanceof List<?> ids && !ids.isEmpty()) {
                    return ((Number) ids.getFirst()).intValue();
                }
            }
        }
        return null;
    }
}
