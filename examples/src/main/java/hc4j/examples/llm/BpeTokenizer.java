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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

    // Standard GPT-2 pre-tokenization pattern separating words (with leading space), numbers, punctuation, and runs of whitespace
    private static final Pattern GPT2_PRETOKEN_PATTERN = Pattern.compile(
            " ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

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
    private final boolean byteLevelUseRegex;
    private final char metaspaceChar;
    private final boolean metaspaceAddPrefixSpace;
    private final Integer unkTokenId;
    private final OptionalInt bosTokenId;
    private final Set<Integer> eosTokenIds;

    private BpeTokenizer(
            Map<String, Integer> tokenToId,
            String[] idToToken,
            Map<MergePair, Integer> mergeRanks,
            Map<String, Integer> addedTokens,
            Set<Integer> specialTokenIds,
            Pattern addedTokenPattern,
            Mode mode,
            boolean byteLevelUseRegex,
            char metaspaceChar,
            boolean metaspaceAddPrefixSpace,
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
        this.byteLevelUseRegex = byteLevelUseRegex;
        this.metaspaceChar = metaspaceChar;
        this.metaspaceAddPrefixSpace = metaspaceAddPrefixSpace;
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

        TokenizerLayoutConfig layout = detectLayout(root, tokenToId);

        Integer unkId = null;
        Object unkObj = model.get("unk_token");
        if (unkObj instanceof String unkTok) {
            unkId = tokenToId.get(unkTok);
        }

        OptionalInt bosId = detectBosTokenId(root, tokenToId);
        Set<Integer> eosIds = detectEosTokenIds(root, tokenToId);

        return new BpeTokenizer(
                tokenToId, idToToken, mergeRanks, addedTokens, specialTokenIds, addedTokenPattern,
                layout.mode, layout.byteLevelUseRegex, layout.metaspaceChar, layout.metaspaceAddPrefixSpace,
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
        if (addBos && bosTokenId.isPresent()) {
            out.add(bosTokenId.getAsInt());
        }

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
        if (mode == Mode.METASPACE) {
            StringBuilder mapped = new StringBuilder();
            String replaced = text.replace(' ', metaspaceChar);
            if (metaspaceAddPrefixSpace && !replaced.isEmpty() && replaced.charAt(0) != metaspaceChar) {
                mapped.append(metaspaceChar);
            }
            mapped.append(replaced);
            runBpeOnPiece(mapped.toString(), out);
            return;
        }

        if (byteLevelUseRegex) {
            Matcher matcher = GPT2_PRETOKEN_PATTERN.matcher(text);
            int lastEnd = 0;
            while (matcher.find()) {
                if (matcher.start() > lastEnd) {
                    runBpeOnPiece(toByteLevelUnicode(text.substring(lastEnd, matcher.start())), out);
                }
                runBpeOnPiece(toByteLevelUnicode(matcher.group()), out);
                lastEnd = matcher.end();
            }
            if (lastEnd < text.length()) {
                runBpeOnPiece(toByteLevelUnicode(text.substring(lastEnd)), out);
            }
        } else {
            runBpeOnPiece(toByteLevelUnicode(text), out);
        }
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
        if (wholeId != null && unicodePiece.length() == 1) {
            out.add(wholeId);
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

        for (String sym : symbols) {
            Integer id = tokenToId.get(sym);
            if (id != null) {
                out.add(id);
            } else if (unkTokenId != null) {
                out.add(unkTokenId);
            } else {
                throw new IllegalArgumentException("Token piece not found in vocabulary: '" + sym + "'");
            }
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
            boolean byteLevelUseRegex,
            char metaspaceChar,
            boolean metaspaceAddPrefixSpace
    ) {}

    private static TokenizerLayoutConfig detectLayout(Map<String, Object> root, Map<String, Integer> tokenToId) {
        List<Map<?, ?>> nodes = new ArrayList<>();
        collectTypedNodes(root.get("pre_tokenizer"), nodes);
        collectTypedNodes(root.get("decoder"), nodes);

        Boolean byteLevelRegex = null;
        boolean sawByteLevel = false;
        boolean sawMetaspace = false;
        char metaspaceChar = '\u2581';
        boolean metaspaceAddPrefix = false;

        for (Map<?, ?> node : nodes) {
            Object typeObj = node.get("type");
            if (!(typeObj instanceof String type)) {
                continue;
            }
            if ("ByteLevel".equalsIgnoreCase(type)) {
                sawByteLevel = true;
                Object useRegexObj = node.get("use_regex");
                if (useRegexObj instanceof Boolean b && byteLevelRegex == null) {
                    byteLevelRegex = b;
                }
            } else if ("Metaspace".equalsIgnoreCase(type)) {
                sawMetaspace = true;
                Object repObj = node.get("replacement");
                if (repObj instanceof String s && !s.isEmpty()) {
                    metaspaceChar = s.charAt(0);
                }
                Object addPrefixObj = node.get("add_prefix_space");
                if (addPrefixObj instanceof Boolean b) {
                    metaspaceAddPrefix = b;
                } else {
                    Object schemeObj = node.get("prepend_scheme");
                    if (schemeObj instanceof String scheme) {
                        metaspaceAddPrefix = "always".equalsIgnoreCase(scheme) || "first".equalsIgnoreCase(scheme);
                    }
                }
            }
        }

        if (sawByteLevel) {
            return new TokenizerLayoutConfig(Mode.BYTE_LEVEL, byteLevelRegex == null || byteLevelRegex, metaspaceChar, false);
        }
        if (sawMetaspace) {
            return new TokenizerLayoutConfig(Mode.METASPACE, true, metaspaceChar, metaspaceAddPrefix);
        }

        // Vocab-based fallback detection
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
        if (hasByteSpace) {
            return new TokenizerLayoutConfig(Mode.BYTE_LEVEL, true, '\u2581', false);
        }
        if (hasMetaspaceChar) {
            return new TokenizerLayoutConfig(Mode.METASPACE, true, '\u2581', false);
        }
        return new TokenizerLayoutConfig(Mode.BYTE_LEVEL, true, '\u2581', false);
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

    private static OptionalInt detectBosTokenId(Map<String, Object> root, Map<String, Integer> tokenToId) {
        for (String cand : BOS_CANDIDATES) {
            Integer id = tokenToId.get(cand);
            if (id != null) {
                return OptionalInt.of(id);
            }
        }
        Object postObj = root.get("post_processor");
        if (postObj instanceof Map<?, ?> post) {
            Integer id = inspectTemplateProcessingSpecial(post, true);
            if (id != null) {
                return OptionalInt.of(id);
            }
        }
        return OptionalInt.empty();
    }

    private static Set<Integer> detectEosTokenIds(Map<String, Object> root, Map<String, Integer> tokenToId) {
        Set<Integer> eos = new LinkedHashSet<>();
        for (String cand : EOS_CANDIDATES) {
            Integer id = tokenToId.get(cand);
            if (id != null) {
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
