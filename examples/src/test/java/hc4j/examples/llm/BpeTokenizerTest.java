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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Step 4D-Fix: Pure-Java BpeTokenizer Tests (Headless)")
class BpeTokenizerTest {

    private static Map<String, Integer> buildByteLevelVocab(List<String> extraMergedTokens) {
        boolean[] keep = new boolean[256];
        for (int b = '!'; b <= '~'; b++) keep[b] = true;
        for (int b = 0xA1; b <= 0xAC; b++) keep[b] = true;
        for (int b = 0xAE; b <= 0xFF; b++) keep[b] = true;

        Map<String, Integer> vocab = new LinkedHashMap<>();
        int nextId = 0;
        int extra = 0;
        for (int b = 0; b < 256; b++) {
            char ch = keep[b] ? (char) b : (char) (256 + extra++);
            vocab.put(String.valueOf(ch), nextId++);
        }
        for (String merged : extraMergedTokens) {
            if (!vocab.containsKey(merged)) {
                vocab.put(merged, nextId++);
            }
        }
        return vocab;
    }

    private static String buildByteLevelTokenizerJson(boolean arrayMerges) {
        List<String> mergedTokens = List.of(
                "he", "ll", "hello", "Ġw", "Ġwo", "Ġwor", "Ġworl", "Ġworld", "ca", "caf", "cafÃ"
        );
        Map<String, Integer> vocab = buildByteLevelVocab(mergedTokens);

        List<Object> merges = new ArrayList<>();
        List<String[]> pairs = List.of(
                new String[]{"h", "e"},
                new String[]{"l", "l"},
                new String[]{"he", "ll"},
                new String[]{"hell", "o"}, // not added directly unless present; let's merge exact path for "hello":
                new String[]{"Ġ", "w"},
                new String[]{"Ġw", "o"},
                new String[]{"Ġwo", "r"},
                new String[]{"Ġwor", "l"},
                new String[]{"Ġworl", "d"},
                new String[]{"c", "a"},
                new String[]{"ca", "f"},
                new String[]{"caf", "Ã"}
        );
        // Ensure "hell" is also in vocab for "hell" + "o" -> "hello"
        vocab.put("hell", vocab.size());

        for (String[] p : pairs) {
            if (arrayMerges) {
                merges.add(List.of(p[0], p[1]));
            } else {
                merges.add(p[0] + " " + p[1]);
            }
        }

        int bosId = vocab.size();
        int eosId = vocab.size() + 1;
        int imStartId = vocab.size() + 2;
        int imEndId = vocab.size() + 3;
        int endoTextId = vocab.size() + 4;

        List<Map<String, Object>> addedTokens = List.of(
                Map.of("id", bosId, "content", "<|begin_of_text|>", "special", true),
                Map.of("id", eosId, "content", "<|end_of_text|>", "special", true),
                Map.of("id", imStartId, "content", "<|im_start|>", "special", true),
                Map.of("id", imEndId, "content", "<|im_end|>", "special", true),
                Map.of("id", endoTextId, "content", "<|endoftext|>", "special", true)
        );

        Map<String, Object> model = new LinkedHashMap<>();
        model.put("type", "BPE");
        model.put("vocab", vocab);
        model.put("merges", merges);

        Map<String, Object> root = new LinkedHashMap<>();
        root.put("model", model);
        root.put("added_tokens", addedTokens);
        root.put("pre_tokenizer", Map.of("type", "ByteLevel", "add_prefix_space", false));
        root.put("decoder", Map.of("type", "ByteLevel"));

        return MinimalJson.serialize(root);
    }

    @Test
    @DisplayName("ByteLevel BPE supports both legacy string merges and modern array merges losslessly")
    void testByteLevelStringAndArrayMerges(@TempDir Path tempDir) throws IOException {
        Path legacyPath = tempDir.resolve("tokenizer_legacy.json");
        Path modernPath = tempDir.resolve("tokenizer_modern.json");
        Files.writeString(legacyPath, buildByteLevelTokenizerJson(false));
        Files.writeString(modernPath, buildByteLevelTokenizerJson(true));

        BpeTokenizer legacy = BpeTokenizer.load(legacyPath);
        BpeTokenizer modern = BpeTokenizer.load(modernPath);

        assertEquals(legacy.vocabSize(), modern.vocabSize());
        assertEquals(OptionalInt.of(legacy.vocabSize() - 5), legacy.bosTokenId());
        assertTrue(legacy.eosTokenIds().contains(legacy.vocabSize() - 4));
        assertTrue(legacy.eosTokenIds().contains(legacy.vocabSize() - 2));
        assertTrue(legacy.eosTokenIds().contains(legacy.vocabSize() - 1));

        String sample = "<|begin_of_text|>hello world!\ncafé<|endoftext|>";
        int[] legacyIds = legacy.encode(sample);
        int[] modernIds = modern.encode(sample);

        assertArrayEquals(legacyIds, modernIds, "String and array merges must produce identical token IDs");

        // Special tokens preserved intact as single tokens at endpoints
        int bosId = legacy.bosTokenId().orElseThrow();
        assertEquals(bosId, legacyIds[0]);
        assertTrue(legacy.isSpecialToken(bosId));
        int lastId = legacyIds[legacyIds.length - 1];
        assertTrue(legacy.isSpecialToken(lastId));
        assertFalse(legacy.isSpecialToken(legacyIds[1]));

        // Verified merged tokens used ("hello" -> 1 token, "Ġworld" -> 1 token)
        assertEquals(8, legacyIds.length); // [BOS, "hello", "Ġworld", "!", "Ċ", "cafÃ", "©", EOS]

        // Exact lossless round-trip (retaining vs skipping special tokens)
        assertEquals(sample, legacy.decode(legacyIds, false));
        assertEquals("hello world!\ncafé", legacy.decode(legacyIds, true));

        // Standalone single-token decode check on whole pieces
        assertEquals("hello", modern.decodeToken(legacyIds[1]));
        assertEquals(" world", modern.decodeToken(legacyIds[2]));

        // Streaming single-token decode check handling split multi-byte UTF-8 ("cafÃ" + "©" -> "café")
        BpeTokenizer.StreamDecoder stream = modern.newStreamDecoder();
        StringBuilder streamed = new StringBuilder();
        for (int id : legacyIds) {
            streamed.append(stream.next(id));
        }
        streamed.append(stream.flush());
        assertEquals("hello world!\ncafé", streamed.toString());
    }

    @Test
    @DisplayName("ByteLevel encode(..., addBos=true) prepends BOS when present")
    void testAddBosAndPunctuationRoundTrip(@TempDir Path tempDir) throws IOException {
        Path tokPath = tempDir.resolve("tokenizer.json");
        Files.writeString(tokPath, buildByteLevelTokenizerJson(true));
        BpeTokenizer tok = BpeTokenizer.load(tokPath);

        String text = "hello, world?\n123 + café!";
        int[] noBos = tok.encode(text, false);
        int[] withBos = tok.encode(text, true);

        assertEquals(noBos.length + 1, withBos.length);
        assertEquals(tok.bosTokenId().orElseThrow(), withBos[0]);
        assertEquals(text, tok.decode(withBos, true));
    }

    @Test
    @DisplayName("Metaspace (U+2581) BPE encoding & decoding round-trip")
    void testMetaspaceTokenizerRoundTrip(@TempDir Path tempDir) throws IOException {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        String[] base = {
                "<s>", "</s>", "\u2581", "h", "e", "l", "o", "w", "r", "d",
                "he", "ll", "hello", "\u2581w", "\u2581wo", "\u2581wor", "\u2581worl", "\u2581world", "hell"
        };
        for (int i = 0; i < base.length; i++) {
            vocab.put(base[i], i);
        }

        List<Object> merges = List.of(
                List.of("h", "e"),
                List.of("l", "l"),
                List.of("he", "ll"),
                List.of("hell", "o"),
                List.of("\u2581", "w"),
                List.of("\u2581w", "o"),
                List.of("\u2581wo", "r"),
                List.of("\u2581wor", "l"),
                List.of("\u2581worl", "d")
        );

        Map<String, Object> root = new LinkedHashMap<>();
        root.put("model", Map.of("type", "BPE", "vocab", vocab, "merges", merges));
        root.put("added_tokens", List.of(
                Map.of("id", 0, "content", "<s>", "special", true),
                Map.of("id", 1, "content", "</s>", "special", true)
        ));
        root.put("pre_tokenizer", Map.of("type", "Metaspace", "replacement", "\u2581", "add_prefix_space", true));
        root.put("decoder", Map.of("type", "Metaspace", "replacement", "\u2581", "add_prefix_space", true));

        Path tokPath = tempDir.resolve("metaspace_tokenizer.json");
        Files.writeString(tokPath, MinimalJson.serialize(root));

        BpeTokenizer tok = BpeTokenizer.load(tokPath);
        assertEquals(OptionalInt.of(0), tok.bosTokenId());
        assertTrue(tok.eosTokenIds().contains(1));

        int[] ids = tok.encode("hello world");
        // "\u2581hello\u2581world" -> "\u2581" (2), "hello" (12), "\u2581world" (17)
        assertArrayEquals(new int[]{2, 12, 17}, ids);
        assertEquals("hello world", tok.decode(ids, true));
    }

    @Test
    @DisplayName("Qwen2.5 Sequence(Split + ByteLevel(use_regex=false)) prevents cross-word/newline merges and honors NFC + ChatML specials")
    void testQwen25SequenceSplitPreTokenizerAndSpecials(@TempDir Path tempDir) throws IOException {
        List<String> extras = List.of(
                "user", "assistant", "What", "Ġis", "Ġthe", "Ġcapital", "?", "'s", "Ċ",
                "<s>", "</s>", // ordinary HTML tokens in Qwen2.5 vocab MUST NOT trigger synthetic BOS/EOS!
                "userĊ", "?Ċ" // cross-word/newline merged tokens that MUST NOT trigger across Split boundaries!
        );
        Map<String, Integer> vocab = buildByteLevelVocab(extras);
        // Add individual intermediate tokens so words build cleanly via merges
        for (String intermediate : List.of("us", "use", "as", "ass", "assi", "assis", "assist", "assista", "assistan",
                "Wh", "Wha", "Ġi", "Ġt", "Ġth", "Ġc", "Ġca", "Ġcap", "Ġcapi", "Ġcapit", "Ġcapita",
                "Ã©", "caf", "cafÃ", "cafÃ©")) {
            if (!vocab.containsKey(intermediate)) {
                vocab.put(intermediate, vocab.size());
            }
        }
        int endoTextId = vocab.size();
        int imStartId = vocab.size() + 1;
        int imEndId = vocab.size() + 2;
        int toolCallId = vocab.size() + 3;

        List<String> merges = List.of(
                // Tempting cross-boundary merges placed earliest in rank!
                "r Ċ", "? Ċ", "use rĊ",
                // Normal within-piece merges
                "u s", "us e", "use r",
                "a s", "as s", "ass i", "assi s", "assis t", "assist a", "assista n", "assistan t",
                "W h", "Wh a", "Wha t",
                "Ġ i", "Ġi s",
                "Ġ t", "Ġt h", "Ġth e",
                "Ġ c", "Ġc a", "Ġca p", "Ġcap i", "Ġcapi t", "Ġcapit a", "Ġcapita l",
                "' s",
                "c a", "ca f", "caf Ã", "cafÃ ©"
        );

        String qwenRegex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
        Map<String, Object> preTokenizer = Map.of(
                "type", "Sequence",
                "pretokenizers", List.of(
                        Map.of(
                                "type", "Split",
                                "pattern", Map.of("Regex", qwenRegex),
                                "behavior", "Isolated",
                                "invert", false
                        ),
                        Map.of(
                                "type", "ByteLevel",
                                "add_prefix_space", false,
                                "trim_offsets", false,
                                "use_regex", false
                        )
                )
        );

        Map<String, Object> root = new LinkedHashMap<>();
        root.put("normalizer", Map.of("type", "NFC"));
        root.put("pre_tokenizer", preTokenizer);
        root.put("decoder", Map.of("type", "ByteLevel", "add_prefix_space", false, "use_regex", false));
        root.put("added_tokens", List.of(
                Map.of("id", endoTextId, "content", "<|endoftext|>", "special", true),
                Map.of("id", imStartId, "content", "<|im_start|>", "special", true),
                Map.of("id", imEndId, "content", "<|im_end|>", "special", true),
                Map.of("id", toolCallId, "content", "<tool_call>", "special", false)
        ));
        root.put("model", Map.of("type", "BPE", "vocab", vocab, "merges", merges));

        Path tokPath = tempDir.resolve("qwen_tokenizer.json");
        Files.writeString(tokPath, MinimalJson.serialize(root));
        BpeTokenizer tok = BpeTokenizer.load(tokPath);

        assertTrue(tok.bosTokenId().isEmpty(), "ChatML tokenizer should not synthesize spurious BOS");
        assertTrue(tok.eosTokenIds().contains(imEndId));
        assertTrue(tok.eosTokenIds().contains(endoTextId));
        assertFalse(tok.eosTokenIds().contains(vocab.get("</s>")), "Non-added HTML </s> in vocab must not be EOS");

        String chatPrompt = "<|im_start|>user\nWhat's the capital?<|im_end|>\n<|im_start|>assistant\n";
        int[] ids = tok.encode(chatPrompt, true);
        int nlId = vocab.get("Ċ");
        int[] expected = {
                imStartId,
                vocab.get("user"),
                nlId,
                vocab.get("What"),
                vocab.get("'s"),
                vocab.get("Ġthe"),
                vocab.get("Ġcapital"),
                vocab.get("?"),
                imEndId,
                nlId,
                imStartId,
                vocab.get("assistant"),
                nlId
        };
        assertArrayEquals(expected, ids, "Split pre-tokenizer must keep 'user' + '\\n' separate despite 'r Ċ' merge");
        assertEquals(chatPrompt, tok.decode(ids, false));
        assertEquals("user\nWhat's the capital?\nassistant\n", tok.decode(ids, true));

        // Non-special added token (<tool_call>) retained when skipSpecialTokens=true
        int[] withTool = tok.encode("<tool_call>cafe\u0301");
        assertEquals(toolCallId, withTool[0]);
        assertEquals("<tool_call>café", tok.decode(withTool, true));
    }

    @Test
    @DisplayName("SmolLM2 Sequence(Digits + ByteLevel(use_regex=true)) splits individual digits cleanly")
    void testSmolLm2DigitsSequencePreTokenizer(@TempDir Path tempDir) throws IOException {
        Map<String, Integer> vocab = buildByteLevelVocab(List.of("12", "123", "Ġ1"));
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("pre_tokenizer", Map.of(
                "type", "Sequence",
                "pretokenizers", List.of(
                        Map.of("type", "Digits", "individual_digits", true),
                        Map.of("type", "ByteLevel", "add_prefix_space", false, "use_regex", true)
                )
        ));
        root.put("decoder", Map.of("type", "ByteLevel", "add_prefix_space", true, "use_regex", true));
        root.put("model", Map.of("type", "BPE", "vocab", vocab, "merges", List.of("1 2", "12 3", "Ġ 1")));

        Path tokPath = tempDir.resolve("smol_tokenizer.json");
        Files.writeString(tokPath, MinimalJson.serialize(root));
        BpeTokenizer tok = BpeTokenizer.load(tokPath);

        int[] ids = tok.encode(" 123");
        // Digits(individual_digits=true) -> [" ", "1", "2", "3"] -> separate tokens ['Ġ', '1', '2', '3']
        assertArrayEquals(new int[]{vocab.get("Ġ"), vocab.get("1"), vocab.get("2"), vocab.get("3")}, ids);
        assertEquals(" 123", tok.decode(ids, true));
    }
}
