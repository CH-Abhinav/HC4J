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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Lightweight, zero-dependency recursive-descent JSON parser and serializer.
 */
public final class MinimalJson {

    private MinimalJson() {
        throw new AssertionError("No instances of MinimalJson");
    }

    /**
     * Parses a JSON string into nested Java collections ({@link Map}, {@link List}, {@link String}, {@link Number}, {@link Boolean}, or {@code null}).
     *
     * @param json JSON text to parse
     * @return parsed root object
     */
    public static Object parse(String json) {
        Objects.requireNonNull(json, "json");
        return new Parser(json).parseValue();
    }

    /**
     * Parses a JSON string as a JSON Object (Map of String to Object).
     *
     * @param json JSON text to parse
     * @return parsed Map
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> parseObject(String json) {
        Object result = parse(json);
        if (result instanceof Map<?, ?> map) {
            return (Map<String, Object>) map;
        }
        throw new IllegalArgumentException("Expected JSON Object, got: " + (result == null ? "null" : result.getClass().getName()));
    }

    /**
     * Serializes a Java object (Map, List, String, Number, Boolean, or null) to JSON format.
     *
     * @param value value to serialize
     * @return JSON string
     */
    public static String serialize(Object value) {
        StringBuilder sb = new StringBuilder();
        serializeValue(value, sb);
        return sb.toString();
    }

    private static void serializeValue(Object value, StringBuilder sb) {
        if (value == null) {
            sb.append("null");
        } else if (value instanceof Boolean || value instanceof Number) {
            sb.append(value);
        } else if (value instanceof CharSequence cs) {
            sb.append('"');
            for (int i = 0; i < cs.length(); i++) {
                char c = cs.charAt(i);
                switch (c) {
                    case '"' -> sb.append("\\\"");
                    case '\\' -> sb.append("\\\\");
                    case '\b' -> sb.append("\\b");
                    case '\f' -> sb.append("\\f");
                    case '\n' -> sb.append("\\n");
                    case '\r' -> sb.append("\\r");
                    case '\t' -> sb.append("\\t");
                    default -> {
                        if (c < ' ') {
                            sb.append(String.format("\\u%04x", (int) c));
                        } else {
                            sb.append(c);
                        }
                    }
                }
            }
            sb.append('"');
        } else if (value instanceof Map<?, ?> map) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (!first) sb.append(',');
                first = false;
                serializeValue(String.valueOf(entry.getKey()), sb);
                sb.append(':');
                serializeValue(entry.getValue(), sb);
            }
            sb.append('}');
        } else if (value instanceof Iterable<?> iter) {
            sb.append('[');
            boolean first = true;
            for (Object item : iter) {
                if (!first) sb.append(',');
                first = false;
                serializeValue(item, sb);
            }
            sb.append(']');
        } else if (value.getClass().isArray()) {
            sb.append('[');
            int len = java.lang.reflect.Array.getLength(value);
            for (int i = 0; i < len; i++) {
                if (i > 0) sb.append(',');
                serializeValue(java.lang.reflect.Array.get(value, i), sb);
            }
            sb.append(']');
        } else {
            serializeValue(value.toString(), sb);
        }
    }

    private static final class Parser {
        private final String src;
        private final int len;
        private int pos;

        Parser(String src) {
            this.src = src;
            this.len = src.length();
            this.pos = 0;
        }

        Object parseValue() {
            skipWhitespace();
            if (pos >= len) {
                throw new IllegalArgumentException("Unexpected end of JSON input");
            }
            char c = src.charAt(pos);
            if (c == '{') return parseObject();
            if (c == '[') return parseArray();
            if (c == '"') return parseString();
            if (c == 't' || c == 'f') return parseBoolean();
            if (c == 'n') return parseNull();
            if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
            throw new IllegalArgumentException("Unexpected character '" + c + "' at position " + pos);
        }

        private Map<String, Object> parseObject() {
            consume('{');
            Map<String, Object> map = new LinkedHashMap<>();
            skipWhitespace();
            if (peek() == '}') {
                consume('}');
                return map;
            }
            while (true) {
                skipWhitespace();
                String key = parseString();
                skipWhitespace();
                consume(':');
                Object val = parseValue();
                map.put(key, val);
                skipWhitespace();
                char next = peek();
                if (next == '}') {
                    consume('}');
                    break;
                } else if (next == ',') {
                    consume(',');
                } else {
                    throw new IllegalArgumentException("Expected ',' or '}' at position " + pos + ", got '" + next + "'");
                }
            }
            return map;
        }

        private List<Object> parseArray() {
            consume('[');
            List<Object> list = new ArrayList<>();
            skipWhitespace();
            if (peek() == ']') {
                consume(']');
                return list;
            }
            while (true) {
                list.add(parseValue());
                skipWhitespace();
                char next = peek();
                if (next == ']') {
                    consume(']');
                    break;
                } else if (next == ',') {
                    consume(',');
                } else {
                    throw new IllegalArgumentException("Expected ',' or ']' at position " + pos + ", got '" + next + "'");
                }
            }
            return list;
        }

        private String parseString() {
            consume('"');
            StringBuilder sb = new StringBuilder();
            while (pos < len) {
                char c = src.charAt(pos++);
                if (c == '"') {
                    return sb.toString();
                } else if (c == '\\') {
                    if (pos >= len) throw new IllegalArgumentException("Unterminated escape sequence");
                    char esc = src.charAt(pos++);
                    switch (esc) {
                        case '"' -> sb.append('"');
                        case '\\' -> sb.append('\\');
                        case '/' -> sb.append('/');
                        case 'b' -> sb.append('\b');
                        case 'f' -> sb.append('\f');
                        case 'n' -> sb.append('\n');
                        case 'r' -> sb.append('\r');
                        case 't' -> sb.append('\t');
                        case 'u' -> {
                            if (pos + 4 > len) throw new IllegalArgumentException("Invalid unicode escape");
                            String hex = src.substring(pos, pos + 4);
                            pos += 4;
                            sb.append((char) Integer.parseInt(hex, 16));
                        }
                        default -> throw new IllegalArgumentException("Unknown escape character: " + esc);
                    }
                } else {
                    sb.append(c);
                }
            }
            throw new IllegalArgumentException("Unterminated string");
        }

        private Number parseNumber() {
            int start = pos;
            if (peek() == '-') pos++;
            while (pos < len && Character.isDigit(src.charAt(pos))) pos++;
            boolean isFloat = false;
            if (pos < len && src.charAt(pos) == '.') {
                isFloat = true;
                pos++;
                while (pos < len && Character.isDigit(src.charAt(pos))) pos++;
            }
            if (pos < len && (src.charAt(pos) == 'e' || src.charAt(pos) == 'E')) {
                isFloat = true;
                pos++;
                if (pos < len && (src.charAt(pos) == '+' || src.charAt(pos) == '-')) pos++;
                while (pos < len && Character.isDigit(src.charAt(pos))) pos++;
            }
            String numStr = src.substring(start, pos);
            if (isFloat) {
                return Double.parseDouble(numStr);
            } else {
                long val = Long.parseLong(numStr);
                if (val >= Integer.MIN_VALUE && val <= Integer.MAX_VALUE) {
                    return (int) val;
                }
                return val;
            }
        }

        private Boolean parseBoolean() {
            if (src.startsWith("true", pos)) {
                pos += 4;
                return Boolean.TRUE;
            } else if (src.startsWith("false", pos)) {
                pos += 5;
                return Boolean.FALSE;
            }
            throw new IllegalArgumentException("Expected boolean at position " + pos);
        }

        private Object parseNull() {
            if (src.startsWith("null", pos)) {
                pos += 4;
                return null;
            }
            throw new IllegalArgumentException("Expected null at position " + pos);
        }

        private void skipWhitespace() {
            while (pos < len) {
                char c = src.charAt(pos);
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    pos++;
                } else {
                    break;
                }
            }
        }

        private char peek() {
            if (pos >= len) return '\0';
            return src.charAt(pos);
        }

        private void consume(char expected) {
            if (pos >= len || src.charAt(pos) != expected) {
                throw new IllegalArgumentException("Expected '" + expected + "' at position " + pos + ", got '" + peek() + "'");
            }
            pos++;
        }
    }
}
