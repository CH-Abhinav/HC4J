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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Utility for writing synthetic HuggingFace {@code .safetensors} files in tests and benchmarks.
 */
public final class SafetensorsWriter {

    /**
     * In-memory tensor payload to be written to a safetensors file.
     *
     * @param dtype data type string ("F32", "F16", "BF16", "I32")
     * @param shape tensor shape
     * @param data  raw little-endian bytes
     */
    public record TensorPayload(String dtype, int[] shape, byte[] data) {}

    private SafetensorsWriter() {
        throw new AssertionError("No instances of SafetensorsWriter");
    }

    /**
     * Creates a float tensor payload.
     *
     * @param shape  tensor shape
     * @param values float array
     * @return payload
     */
    public static TensorPayload ofFloat(int[] shape, float[] values) {
        ByteBuffer buf = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : values) buf.putFloat(v);
        return new TensorPayload("F32", shape, buf.array());
    }

    /**
     * Creates a half-precision float tensor payload.
     *
     * @param shape tensor shape
     * @param bits  raw IEEE 754 half-precision bits
     * @return payload
     */
    public static TensorPayload ofHalf(int[] shape, short[] bits) {
        ByteBuffer buf = ByteBuffer.allocate(bits.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (short b : bits) buf.putShort(b);
        return new TensorPayload("F16", shape, buf.array());
    }

    /**
     * Creates a bfloat16 tensor payload.
     *
     * @param shape tensor shape
     * @param bits  raw bfloat16 bits
     * @return payload
     */
    public static TensorPayload ofBfloat16(int[] shape, short[] bits) {
        ByteBuffer buf = ByteBuffer.allocate(bits.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (short b : bits) buf.putShort(b);
        return new TensorPayload("BF16", shape, buf.array());
    }

    /**
     * Writes tensors into a {@code .safetensors} file.
     *
     * @param targetFile destination path
     * @param tensors    map of tensor names to payloads
     * @throws IOException if writing fails
     */
    public static void write(Path targetFile, Map<String, TensorPayload> tensors) throws IOException {
        Objects.requireNonNull(targetFile, "targetFile");
        Objects.requireNonNull(tensors, "tensors");

        if (targetFile.getParent() != null) {
            Files.createDirectories(targetFile.getParent());
        }

        Map<String, Object> headerMap = new LinkedHashMap<>();
        long currentOffset = 0;

        for (Map.Entry<String, TensorPayload> entry : tensors.entrySet()) {
            String name = entry.getKey();
            TensorPayload p = entry.getValue();

            long begin = currentOffset;
            long end = begin + p.data().length;
            currentOffset = end;

            Map<String, Object> meta = new LinkedHashMap<>();
            meta.put("dtype", p.dtype());
            List<Integer> shapeList = new ArrayList<>();
            for (int dim : p.shape()) shapeList.add(dim);
            meta.put("shape", shapeList);
            meta.put("data_offsets", Arrays.asList(begin, end));

            headerMap.put(name, meta);
        }

        String json = MinimalJson.serialize(headerMap);
        byte[] jsonBytes = json.getBytes(StandardCharsets.UTF_8);
        long headerSize = jsonBytes.length;

        try (FileChannel channel = FileChannel.open(targetFile,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {

            ByteBuffer headerLenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            headerLenBuf.putLong(headerSize);
            headerLenBuf.flip();
            channel.write(headerLenBuf);

            channel.write(ByteBuffer.wrap(jsonBytes));

            for (TensorPayload p : tensors.values()) {
                channel.write(ByteBuffer.wrap(p.data()));
            }
        }
    }
}
