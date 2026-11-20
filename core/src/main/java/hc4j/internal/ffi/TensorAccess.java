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

package hc4j.internal.ffi;

import hc4j.DType;
import hc4j.Tensor;

/**
 * SharedSecrets bridge allowing hc4j.internal.* and FusedExpr to access Tensor's
 * internal fields and constructor without exposing them in Tensor's public API.
 */
public abstract class TensorAccess {

    private static volatile TensorAccess instance;

    public static void setInstance(TensorAccess access) {
        instance = access;
    }

    public static TensorAccess getInstance() {
        TensorAccess a = instance;
        if (a == null) {
            try {
                Class.forName(Tensor.class.getName(), true, Tensor.class.getClassLoader());
            } catch (ClassNotFoundException e) {
                throw new AssertionError("Failed to load Tensor class", e);
            }
            a = instance;
        }
        return a;
    }

    public abstract long getVramId(Tensor t);

    public abstract int[] getShapeUnsafe(Tensor t);

    public abstract int[] getStridesUnsafe(Tensor t);

    public abstract void ensureOpen(Tensor t);

    public abstract Tensor create(DType dtype, int[] shape, int[] strides, long vramId);
}
