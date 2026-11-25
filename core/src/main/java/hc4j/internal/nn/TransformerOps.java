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

package hc4j.internal.nn;

import hc4j.DType;
import hc4j.Tensor;
import hc4j.internal.ffi.NativeBindings;
import hc4j.internal.ffi.NativeEngine;
import hc4j.internal.ffi.TensorAccess;
import hc4j.internal.layout.ValidUtil;
import java.util.Arrays;
import java.util.Objects;

/**
 * Dispatcher for Transformer inference primitives: embedding table gather,
 * NeoX half-split Rotary Position Embedding (RoPE), and zero-copy KV-cache append.
 */
public final class TransformerOps {

    private TransformerOps() {
        throw new AssertionError("No instances of TransformerOps");
    }

    /**
     * Gathers rows from 2-D {@code f32} or {@code f16} embedding {@code table} for integer {@code tokenIds}.
     *
     * @param table 2-D embedding table {@code [vocabSize, hiddenDim]}
     * @param tokenIds {@code i32} token index tensor
     * @param res destination {@code f32} tensor {@code [..., hiddenDim]}
     * @return {@code res}
     */
    public static Tensor embedding(Tensor table, Tensor tokenIds, Tensor res) {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(tokenIds, "tokenIds");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(table);
        access.ensureOpen(tokenIds);
        access.ensureOpen(res);

        if (table.getDType() != DType.f32 && table.getDType() != DType.f16) {
            throw new IllegalArgumentException("Embedding table must be f32 or f16, got " + table.getDType());
        }
        if (tokenIds.getDType() != DType.i32) {
            throw new IllegalArgumentException("tokenIds must be i32, got " + tokenIds.getDType());
        }
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeTable = ValidUtil.prepareContiguousZeroOffset(table);
             ValidUtil.SafeOperand safeTokens = ValidUtil.prepareContiguousZeroOffset(tokenIds)) {

            Tensor et = safeTable.tensor();
            Tensor ek = safeTokens.tensor();

            int[] tableShape = access.getShapeUnsafe(et);
            if (tableShape.length != 2) {
                throw new IllegalArgumentException("Embedding table must be 2-D [vocabSize, hiddenDim], got " + et.dim() + "-D");
            }
            int vocabSize = tableShape[0];
            int hiddenDim = tableShape[1];

            int[] tokenShape = access.getShapeUnsafe(ek);
            int[] expectedOutShape = new int[tokenShape.length + 1];
            System.arraycopy(tokenShape, 0, expectedOutShape, 0, tokenShape.length);
            expectedOutShape[tokenShape.length] = hiddenDim;
            ValidUtil.requireResultShape(expectedOutShape, access.getShapeUnsafe(res));

            long idTable = access.getVramId(et);
            long idTokens = access.getVramId(ek);
            long idRes = access.getVramId(res);
            ValidUtil.requireNoAlias(idRes, idTable, "embedding result must not alias embedding table");
            ValidUtil.requireNoAlias(idRes, idTokens, "embedding result must not alias tokenIds");

            long numTokensLong = ek.getSize();
            if (numTokensLong > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Total token count exceeds Integer.MAX_VALUE: " + numTokensLong);
            }
            int numTokens = (int) numTokensLong;
            int isF16 = (et.getDType() == DType.f16) ? 1 : 0;

            int status = NativeBindings.dispatchEmbedding(
                    idTable, idTokens, idRes, vocabSize, hiddenDim, numTokens, isF16);
            NativeEngine.checkStatus(status, "TransformerOps.embedding");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: embedding failed", t);
        }
    }

    /**
     * Applies NeoX half-split Rotary Position Embedding (RoPE) on 2-D, 3-D, or 4-D {@code f32} tensors.
     * In-place safe when {@code x} and {@code res} share identical buffer and layout.
     *
     * @param x input {@code f32} tensor
     * @param positionOffset sequence position offset
     * @param theta rotary base frequency
     * @param res destination {@code f32} tensor
     * @return {@code res}
     */
    public static Tensor rope(Tensor x, int positionOffset, float theta, Tensor res) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(res, "res");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(x);
        access.ensureOpen(res);

        ValidUtil.requireOperandF32("rope input", x.getDType());
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireContiguousZeroOffsetOutput(res);

        try (ValidUtil.SafeOperand safeX = prepareRopeInput(x, res, access)) {
            Tensor ex = safeX.tensor();
            int[] sx = access.getShapeUnsafe(ex);
            ValidUtil.requireResultShape(sx, access.getShapeUnsafe(res));

            int batch;
            int seqLen;
            int numHeads;
            int headDim;
            if (sx.length == 4) {
                batch = sx[0];
                seqLen = sx[1];
                numHeads = sx[2];
                headDim = sx[3];
            } else if (sx.length == 3) {
                batch = 1;
                seqLen = sx[0];
                numHeads = sx[1];
                headDim = sx[2];
            } else if (sx.length == 2) {
                batch = 1;
                seqLen = sx[0];
                numHeads = 1;
                headDim = sx[1];
            } else {
                throw new IllegalArgumentException("rope requires a 2-D, 3-D, or 4-D tensor, got " + sx.length + "-D");
            }

            if (headDim % 2 != 0) {
                throw new IllegalArgumentException("head_dim must be even for half-split RoPE, got " + headDim);
            }

            int status = NativeBindings.dispatchRopeF32(
                    access.getVramId(ex), access.getVramId(res),
                    batch, seqLen, numHeads, headDim, positionOffset, Float.floatToRawIntBits(theta));
            NativeEngine.checkStatus(status, "TransformerOps.rope");
            return res;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: rope failed", t);
        }
    }

    private static ValidUtil.SafeOperand prepareRopeInput(Tensor x, Tensor res, TensorAccess access) {
        if (access.getVramId(x) == access.getVramId(res) && !access.hasIdenticalLayout(x, res)) {
            return new ValidUtil.SafeOperand(x.copy(), true);
        }
        if (access.elementOffset(x) > 0 || !x.isContiguous()) {
            return new ValidUtil.SafeOperand(x.contiguous(), true);
        }
        return new ValidUtil.SafeOperand(x, false);
    }

    /**
     * Appends and transposes {@code newValues [seqLen, numKvHeads, headDim]} into
     * {@code cache [numKvHeads, maxSeqLen, headDim]} at token index {@code posOffset}.
     *
     * @param cache destination KV-cache {@code f32} tensor
     * @param newValues incoming key or value states ({@code f32})
     * @param posOffset starting sequence position offset in {@code cache}
     */
    public static void kvCacheAppend(Tensor cache, Tensor newValues, int posOffset) {
        Objects.requireNonNull(cache, "cache");
        Objects.requireNonNull(newValues, "newValues");

        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(cache);
        access.ensureOpen(newValues);

        long idCache = access.getVramId(cache);
        ValidUtil.requireNoAlias(idCache, access.getVramId(newValues),
                "kvCacheAppend cache and newValues must not share the same buffer");

        if (!cache.isContiguous() || access.elementOffset(cache) != 0) {
            throw new IllegalArgumentException("KV cache tensor must be contiguous with zero offset");
        }
        ValidUtil.requireOperandF32("kvCacheAppend cache", cache.getDType());
        ValidUtil.requireOperandF32("kvCacheAppend newValues", newValues.getDType());

        try (ValidUtil.SafeOperand safeNew = ValidUtil.prepareContiguousZeroOffset(newValues)) {
            Tensor en = safeNew.tensor();
            int[] cacheShape = access.getShapeUnsafe(cache);
            if (cacheShape.length != 3) {
                throw new IllegalArgumentException(
                        "KV cache must be 3-D [numKvHeads, maxSeqLen, headDim], got " + cacheShape.length + "-D");
            }
            int[] newShape = access.getShapeUnsafe(en);
            if (newShape.length != 3) {
                throw new IllegalArgumentException(
                        "KV newValues must be 3-D [seqLen, numKvHeads, headDim], got " + newShape.length + "-D");
            }

            int numKvHeads = cacheShape[0];
            int maxSeqLen = cacheShape[1];
            int headDim = cacheShape[2];
            int seqLen = newShape[0];

            if (newShape[1] != numKvHeads || newShape[2] != headDim) {
                throw new IllegalArgumentException(
                        "KV shape mismatch: newValues=" + Arrays.toString(newShape)
                                + " incompatible with cache=" + Arrays.toString(cacheShape));
            }
            if (posOffset < 0 || (long) posOffset + seqLen > maxSeqLen) {
                throw new IllegalArgumentException(
                        "KV cache write out of bounds: posOffset=" + posOffset
                                + ", seqLen=" + seqLen + ", maxSeqLen=" + maxSeqLen);
            }

            int status = NativeBindings.dispatchKvCacheAppendF32(
                    access.getVramId(en), idCache, numKvHeads, maxSeqLen, headDim, seqLen, posOffset);
            NativeEngine.checkStatus(status, "TransformerOps.kvCacheAppend");
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("HC4J GPU Error: kvCacheAppend failed", t);
        }
    }
}
