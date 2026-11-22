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

package hc4j;

import hc4j.internal.ffi.NativeEngine;
import java.lang.foreign.MemorySegment;

/**
 * Top-level static facade for HC4J providing tensor creation, arithmetic,
 * transcendental math, and engine utilities.
 */
public final class HC4J {

    private HC4J() {
        throw new AssertionError("No instances of HC4J");
    }

    // --- Factories ---

    public static Tensor zeros(DType dtype, int... shape) {
        return Tensor.zeros(dtype, shape);
    }

    public static Tensor ones(DType dtype, int... shape) {
        return Tensor.ones(dtype, shape);
    }

    public static Tensor empty(DType dtype, int... shape) {
        return Tensor.empty(dtype, shape);
    }

    public static Tensor full(float value, int... shape) {
        return Tensor.full(value, shape);
    }

    public static Tensor full(int value, int... shape) {
        return Tensor.full(value, shape);
    }

    public static Tensor arange(float start, float stop, float step) {
        return Tensor.arange(start, stop, step);
    }

    public static Tensor arange(int stop) {
        return Tensor.arange(stop);
    }

    public static Tensor arange(int start, int stop, int step) {
        return Tensor.arange(start, stop, step);
    }

    public static Tensor linspace(float start, float stop, int num) {
        return Tensor.linspace(start, stop, num);
    }

    public static Tensor eye(int n) {
        return Tensor.eye(n);
    }

    public static Tensor eye(DType dtype, int n) {
        return Tensor.eye(dtype, n);
    }

    public static Tensor eye(DType dtype, int rows, int cols) {
        return Tensor.eye(dtype, rows, cols);
    }

    public static Tensor fromArray(float[] values, int... shape) {
        return Tensor.fromArray(values, shape);
    }

    public static Tensor fromArray(int[] values, int... shape) {
        return Tensor.fromArray(values, shape);
    }

    public static Tensor fromSegment(MemorySegment source, DType dtype, int... shape) {
        return Tensor.fromSegment(source, dtype, shape);
    }

    // --- Elementwise Arithmetic ---

    public static Tensor add(Tensor a, Tensor b) { return a.add(b); }
    public static Tensor add(Tensor a, Tensor b, Tensor out) { return a.add(b, out); }
    public static Tensor add(Tensor a, float c) { return a.add(c); }
    public static Tensor add(Tensor a, float c, Tensor out) { return a.add(c, out); }

    public static Tensor sub(Tensor a, Tensor b) { return a.sub(b); }
    public static Tensor sub(Tensor a, Tensor b, Tensor out) { return a.sub(b, out); }
    public static Tensor sub(Tensor a, float c) { return a.sub(c); }
    public static Tensor sub(Tensor a, float c, Tensor out) { return a.sub(c, out); }

    public static Tensor mul(Tensor a, Tensor b) { return a.mul(b); }
    public static Tensor mul(Tensor a, Tensor b, Tensor out) { return a.mul(b, out); }
    public static Tensor mul(Tensor a, float c) { return a.mul(c); }
    public static Tensor mul(Tensor a, float c, Tensor out) { return a.mul(c, out); }

    public static Tensor div(Tensor a, Tensor b) { return a.div(b); }
    public static Tensor div(Tensor a, Tensor b, Tensor out) { return a.div(b, out); }
    public static Tensor div(Tensor a, float c) { return a.div(c); }
    public static Tensor div(Tensor a, float c, Tensor out) { return a.div(c, out); }

    public static Tensor rsub(float scalar, Tensor a) { return a.rsub(scalar); }
    public static Tensor rsub(float scalar, Tensor a, Tensor out) { return a.rsub(scalar, out); }
    public static Tensor rsub(int scalar, Tensor a) { return a.rsub(scalar); }
    public static Tensor rsub(int scalar, Tensor a, Tensor out) { return a.rsub(scalar, out); }

    public static Tensor rdiv(float scalar, Tensor a) { return a.rdiv(scalar); }
    public static Tensor rdiv(float scalar, Tensor a, Tensor out) { return a.rdiv(scalar, out); }
    public static Tensor rdiv(int scalar, Tensor a) { return a.rdiv(scalar); }
    public static Tensor rdiv(int scalar, Tensor a, Tensor out) { return a.rdiv(scalar, out); }

    // --- Trigonometric ---

    public static Tensor sin(Tensor a) { return a.sin(); }
    public static Tensor sin(Tensor a, Tensor out) { return a.sin(out); }

    public static Tensor cos(Tensor a) { return a.cos(); }
    public static Tensor cos(Tensor a, Tensor out) { return a.cos(out); }

    public static Tensor tan(Tensor a) { return a.tan(); }
    public static Tensor tan(Tensor a, Tensor out) { return a.tan(out); }

    public static Tensor asin(Tensor a) { return a.asin(); }
    public static Tensor asin(Tensor a, Tensor out) { return a.asin(out); }

    public static Tensor acos(Tensor a) { return a.acos(); }
    public static Tensor acos(Tensor a, Tensor out) { return a.acos(out); }

    public static Tensor atan(Tensor a) { return a.atan(); }
    public static Tensor atan(Tensor a, Tensor out) { return a.atan(out); }

    public static Tensor sinh(Tensor a) { return a.sinh(); }
    public static Tensor sinh(Tensor a, Tensor out) { return a.sinh(out); }

    public static Tensor cosh(Tensor a) { return a.cosh(); }
    public static Tensor cosh(Tensor a, Tensor out) { return a.cosh(out); }

    public static Tensor tanh(Tensor a) { return a.tanh(); }
    public static Tensor tanh(Tensor a, Tensor out) { return a.tanh(out); }

    public static Tensor asinh(Tensor a) { return a.asinh(); }
    public static Tensor asinh(Tensor a, Tensor out) { return a.asinh(out); }

    public static Tensor acosh(Tensor a) { return a.acosh(); }
    public static Tensor acosh(Tensor a, Tensor out) { return a.acosh(out); }

    public static Tensor atanh(Tensor a) { return a.atanh(); }
    public static Tensor atanh(Tensor a, Tensor out) { return a.atanh(out); }

    // --- Exponential & Logarithmic ---

    public static Tensor exp(Tensor a) { return a.exp(); }
    public static Tensor exp(Tensor a, Tensor out) { return a.exp(out); }

    public static Tensor log(Tensor a) { return a.log(); }
    public static Tensor log(Tensor a, Tensor out) { return a.log(out); }

    public static Tensor log2(Tensor a) { return a.log2(); }
    public static Tensor log2(Tensor a, Tensor out) { return a.log2(out); }

    public static Tensor log10(Tensor a) { return a.log10(); }
    public static Tensor log10(Tensor a, Tensor out) { return a.log10(out); }

    public static Tensor sqrt(Tensor a) { return a.sqrt(); }
    public static Tensor sqrt(Tensor a, Tensor out) { return a.sqrt(out); }

    // --- Engine Controls ---

    public static void synchronize() {
        NativeEngine.synchronize();
    }

    public static boolean isGpuAvailable() {
        return NativeEngine.isGpuAvailable();
    }
}
