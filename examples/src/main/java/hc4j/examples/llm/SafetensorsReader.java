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
import hc4j.Tensor;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Pure-Java 25 zero-copy memory-mapped reader for HuggingFace {@code .safetensors} files.
 * Supports F32, F16, and BF16 tensor formats, as well as multi-shard {@code model.safetensors.index.json}.
 */
public final class SafetensorsReader implements AutoCloseable {

    /**
     * Metadata describing a tensor within a safetensors file.
     *
     * @param name         tensor name (e.g. "model.embed_tokens.weight")
     * @param dtype        safetensors data type string ("F32", "F16", "BF16", etc.)
     * @param shape        tensor dimension extents
     * @param beginOffset  byte offset relative to the binary payload start
     * @param endOffset    byte end offset relative to the binary payload start
     * @param shardPath    path of the shard containing this tensor
     */
    public record TensorInfo(String name, String dtype, int[] shape, long beginOffset, long endOffset, Path shardPath) {}

    private static final class ShardMapping {
        final Path path;
        final Arena arena;
        final MemorySegment mmap;
        final long payloadOffset;

        ShardMapping(Path path, Arena arena, MemorySegment mmap, long payloadOffset) {
            this.path = path;
            this.arena = arena;
            this.mmap = mmap;
            this.payloadOffset = payloadOffset;
        }
    }

    private final Map<String, TensorInfo> tensors = new HashMap<>();
    private final Map<Path, ShardMapping> shards = new HashMap<>();
    private final Arena sharedArena = Arena.ofShared();
    private boolean closed = false;

    private SafetensorsReader() {}

    /**
     * Opens a safetensors file or a model directory.
     * If {@code path} is a directory, inspects {@code model.safetensors.index.json} or {@code model.safetensors}.
     *
     * @param path file or directory path
     * @return open reader
     * @throws IOException if file reading or memory mapping fails
     */
    public static SafetensorsReader open(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        SafetensorsReader reader = new SafetensorsReader();
        try {
            if (Files.isDirectory(path)) {
                Path indexPath = path.resolve("model.safetensors.index.json");
                if (Files.exists(indexPath)) {
                    reader.loadIndex(indexPath, path);
                } else {
                    Path singleFile = path.resolve("model.safetensors");
                    if (Files.exists(singleFile)) {
                        reader.loadSingleShard(singleFile);
                    } else {
                        throw new IllegalArgumentException("No .safetensors files or index found in directory: " + path);
                    }
                }
            } else {
                reader.loadSingleShard(path);
            }
            return reader;
        } catch (Exception e) {
            reader.close();
            throw e;
        }
    }

    private void loadIndex(Path indexPath, Path baseDir) throws IOException {
        String json = Files.readString(indexPath, StandardCharsets.UTF_8);
        Map<String, Object> root = MinimalJson.parseObject(json);
        Object weightMapObj = root.get("weight_map");
        if (!(weightMapObj instanceof Map<?, ?> weightMap)) {
            throw new IllegalArgumentException("Invalid index JSON: missing or non-object 'weight_map'");
        }

        // Collect all distinct shard files
        Map<String, Path> shardFiles = new HashMap<>();
        for (Map.Entry<?, ?> entry : weightMap.entrySet()) {
            String shardName = String.valueOf(entry.getValue());
            shardFiles.computeIfAbsent(shardName, k -> baseDir.resolve(k));
        }

        // Open and index each shard
        for (Path shardPath : shardFiles.values()) {
            loadSingleShard(shardPath);
        }
    }

    private void loadSingleShard(Path shardPath) throws IOException {
        Path normPath = shardPath.toAbsolutePath().normalize();
        if (shards.containsKey(normPath)) {
            return;
        }

        long fileSize = Files.size(normPath);
        if (fileSize < 8) {
            throw new IllegalArgumentException("File too small to be a safetensors file: " + normPath);
        }

        try (FileChannel channel = FileChannel.open(normPath, StandardOpenOption.READ)) {
            MemorySegment mmap = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, sharedArena);

            long headerSize = mmap.get(ValueLayout.JAVA_LONG_UNALIGNED, 0L);
            if (headerSize <= 0 || headerSize > fileSize - 8) {
                throw new IllegalArgumentException("Corrupt safetensors header size: " + headerSize + " in " + normPath);
            }

            MemorySegment headerSeg = mmap.asSlice(8L, headerSize);
            byte[] headerBytes = headerSeg.toArray(ValueLayout.JAVA_BYTE);
            String headerJson = new String(headerBytes, StandardCharsets.UTF_8);

            Map<String, Object> headerMap = MinimalJson.parseObject(headerJson);
            long payloadOffset = 8L + headerSize;

            for (Map.Entry<String, Object> entry : headerMap.entrySet()) {
                String tensorName = entry.getKey();
                if ("__metadata__".equals(tensorName)) {
                    continue;
                }
                if (entry.getValue() instanceof Map<?, ?> tMeta) {
                    String dtype = String.valueOf(tMeta.get("dtype"));
                    List<?> shapeList = (List<?>) tMeta.get("shape");
                    int[] shape = new int[shapeList.size()];
                    for (int i = 0; i < shape.length; i++) {
                        shape[i] = ((Number) shapeList.get(i)).intValue();
                    }
                    List<?> offsetsList = (List<?>) tMeta.get("data_offsets");
                    long begin = ((Number) offsetsList.get(0)).longValue();
                    long end = ((Number) offsetsList.get(1)).longValue();

                    TensorInfo info = new TensorInfo(tensorName, dtype, shape, begin, end, normPath);
                    tensors.put(tensorName, info);
                }
            }

            shards.put(normPath, new ShardMapping(normPath, sharedArena, mmap, payloadOffset));
        }
    }

    /**
     * Returns all tensor names available in this reader.
     *
     * @return unmodifiable set of tensor names
     */
    public Set<String> tensorNames() {
        return Collections.unmodifiableSet(tensors.keySet());
    }

    /**
     * Returns true if the reader contains a tensor with the given name.
     *
     * @param name tensor name
     * @return true if present
     */
    public boolean hasTensor(String name) {
        return tensors.containsKey(name);
    }

    /**
     * Returns metadata for the specified tensor.
     *
     * @param name tensor name
     * @return metadata info
     */
    public TensorInfo getInfo(String name) {
        TensorInfo info = tensors.get(name);
        if (info == null) {
            throw new IllegalArgumentException("Tensor not found: " + name);
        }
        return info;
    }

    /**
     * Returns a zero-copy memory slice pointing directly to the tensor's raw binary data.
     *
     * @param name tensor name
     * @return raw off-heap memory segment
     */
    public MemorySegment getRawSegment(String name) {
        ensureOpen();
        TensorInfo info = getInfo(name);
        ShardMapping shard = shards.get(info.shardPath());
        long byteLength = info.endOffset() - info.beginOffset();
        return shard.mmap.asSlice(shard.payloadOffset + info.beginOffset(), byteLength);
    }

    /**
     * Loads the tensor into GPU VRAM preserving its natural format (or converting BF16 to F16).
     *
     * @param name tensor name
     * @return uploaded tensor
     */
    public Tensor loadTensor(String name) {
        TensorInfo info = getInfo(name);
        if ("F32".equalsIgnoreCase(info.dtype())) {
            return loadTensor(name, DType.f32);
        } else {
            return loadTensor(name, DType.f16);
        }
    }

    /**
     * Loads the tensor into GPU VRAM in the desired target precision ({@link DType#f32} or {@link DType#f16}).
     * Handles zero-copy direct off-heap uploads for native types and off-heap BF16 conversion.
     *
     * @param name        tensor name
     * @param targetDType desired GPU precision
     * @return uploaded tensor
     */
    public Tensor loadTensor(String name, DType targetDType) {
        ensureOpen();
        Objects.requireNonNull(targetDType, "targetDType");
        TensorInfo info = getInfo(name);
        MemorySegment raw = getRawSegment(name);
        String srcDtype = info.dtype().toUpperCase();

        if (targetDType == DType.f16) {
            if ("F16".equals(srcDtype)) {
                // Zero-copy off-heap upload directly into GPU VRAM!
                return Tensor.fromSegment(raw, DType.f16, info.shape());
            } else if ("BF16".equals(srcDtype)) {
                // Convert BF16 -> F16 via IEEE 754 float
                long numElements = raw.byteSize() / 2;
                try (Arena tempArena = Arena.ofConfined()) {
                    MemorySegment f16Seg = tempArena.allocate(ValueLayout.JAVA_SHORT, numElements);
                    for (long i = 0; i < numElements; i++) {
                        short bf16Bits = raw.get(ValueLayout.JAVA_SHORT_UNALIGNED, i * 2);
                        int f32Bits = (bf16Bits & 0xffff) << 16;
                        float fVal = Float.intBitsToFloat(f32Bits);
                        short f16Bits = floatToHalf(fVal);
                        f16Seg.set(ValueLayout.JAVA_SHORT, i * 2, f16Bits);
                    }
                    return Tensor.fromSegment(f16Seg, DType.f16, info.shape());
                }
            } else if ("F32".equals(srcDtype)) {
                try (Tensor f32Tensor = Tensor.fromSegment(raw, DType.f32, info.shape())) {
                    return f32Tensor.cast(DType.f16);
                }
            } else {
                throw new UnsupportedOperationException("Unsupported source dtype for f16 target: " + srcDtype);
            }
        } else if (targetDType == DType.f32) {
            if ("F32".equals(srcDtype)) {
                // Zero-copy off-heap upload directly into GPU VRAM!
                return Tensor.fromSegment(raw, DType.f32, info.shape());
            } else if ("F16".equals(srcDtype)) {
                try (Tensor f16Tensor = Tensor.fromSegment(raw, DType.f16, info.shape())) {
                    return f16Tensor.cast(DType.f32);
                }
            } else if ("BF16".equals(srcDtype)) {
                // Convert BF16 -> F32
                long numElements = raw.byteSize() / 2;
                try (Arena tempArena = Arena.ofConfined()) {
                    MemorySegment f32Seg = tempArena.allocate(ValueLayout.JAVA_FLOAT, numElements);
                    for (long i = 0; i < numElements; i++) {
                        short bf16Bits = raw.get(ValueLayout.JAVA_SHORT_UNALIGNED, i * 2);
                        int f32Bits = (bf16Bits & 0xffff) << 16;
                        f32Seg.set(ValueLayout.JAVA_FLOAT, i * 4, Float.intBitsToFloat(f32Bits));
                    }
                    return Tensor.fromSegment(f32Seg, DType.f32, info.shape());
                }
            } else {
                throw new UnsupportedOperationException("Unsupported source dtype for f32 target: " + srcDtype);
            }
        } else {
            throw new UnsupportedOperationException("SafetensorsReader only loads to f32 or f16, got " + targetDType);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("SafetensorsReader has already been closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            sharedArena.close();
            shards.clear();
            tensors.clear();
        }
    }

    // Helper: float to IEEE 754 half-precision bits
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
}
