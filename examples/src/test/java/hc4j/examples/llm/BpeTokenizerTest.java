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
}
