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

    // --- Extra Math & Clamping ---

    public static Tensor abs(Tensor a) { return a.abs(); }
    public static Tensor abs(Tensor a, Tensor out) { return a.abs(out); }

    public static Tensor neg(Tensor a) { return a.neg(); }
    public static Tensor neg(Tensor a, Tensor out) { return a.neg(out); }

    public static Tensor rsqrt(Tensor a) { return a.rsqrt(); }
    public static Tensor rsqrt(Tensor a, Tensor out) { return a.rsqrt(out); }

    public static Tensor floor(Tensor a) { return a.floor(); }
    public static Tensor floor(Tensor a, Tensor out) { return a.floor(out); }

    public static Tensor ceil(Tensor a) { return a.ceil(); }
    public static Tensor ceil(Tensor a, Tensor out) { return a.ceil(out); }

    public static Tensor round(Tensor a) { return a.round(); }
    public static Tensor round(Tensor a, Tensor out) { return a.round(out); }

    public static Tensor sign(Tensor a) { return a.sign(); }
    public static Tensor sign(Tensor a, Tensor out) { return a.sign(out); }

    public static Tensor clamp(Tensor a, float min, float max) { return a.clamp(min, max); }
    public static Tensor clamp(Tensor a, float min, float max, Tensor out) { return a.clamp(min, max, out); }
    public static Tensor clamp(Tensor a, int min, int max) { return a.clamp(min, max); }
    public static Tensor clamp(Tensor a, int min, int max, Tensor out) { return a.clamp(min, max, out); }

    public static Tensor clip(Tensor a, float min, float max) { return a.clip(min, max); }
    public static Tensor clip(Tensor a, float min, float max, Tensor out) { return a.clip(min, max, out); }
    public static Tensor clip(Tensor a, int min, int max) { return a.clip(min, max); }
    public static Tensor clip(Tensor a, int min, int max, Tensor out) { return a.clip(min, max, out); }

    // --- Functional Binary Ops ---

    public static Tensor pow(Tensor a, Tensor b) { return a.pow(b); }
    public static Tensor pow(Tensor a, Tensor b, Tensor out) { return a.pow(b, out); }
    public static Tensor pow(Tensor a, float exp) { return a.pow(exp); }

    public static Tensor maximum(Tensor a, Tensor b) { return a.maximum(b); }
    public static Tensor maximum(Tensor a, Tensor b, Tensor out) { return a.maximum(b, out); }
    public static Tensor maximum(Tensor a, float s) { return a.maximum(s); }
    public static Tensor maximum(Tensor a, int s) { return a.maximum(s); }

    public static Tensor minimum(Tensor a, Tensor b) { return a.minimum(b); }
    public static Tensor minimum(Tensor a, Tensor b, Tensor out) { return a.minimum(b, out); }
    public static Tensor minimum(Tensor a, float s) { return a.minimum(s); }
    public static Tensor minimum(Tensor a, int s) { return a.minimum(s); }

    // --- Comparison Ops ---

    public static Tensor eq(Tensor a, Tensor b) { return a.eq(b); }
    public static Tensor eq(Tensor a, Tensor b, Tensor out) { return a.eq(b, out); }
    public static Tensor eq(Tensor a, float s) { return a.eq(s); }
    public static Tensor eq(Tensor a, int s) { return a.eq(s); }

    public static Tensor ne(Tensor a, Tensor b) { return a.ne(b); }
    public static Tensor ne(Tensor a, Tensor b, Tensor out) { return a.ne(b, out); }
    public static Tensor ne(Tensor a, float s) { return a.ne(s); }
    public static Tensor ne(Tensor a, int s) { return a.ne(s); }

    public static Tensor gt(Tensor a, Tensor b) { return a.gt(b); }
    public static Tensor gt(Tensor a, Tensor b, Tensor out) { return a.gt(b, out); }
    public static Tensor gt(Tensor a, float s) { return a.gt(s); }
    public static Tensor gt(Tensor a, int s) { return a.gt(s); }

    public static Tensor ge(Tensor a, Tensor b) { return a.ge(b); }
    public static Tensor ge(Tensor a, Tensor b, Tensor out) { return a.ge(b, out); }
    public static Tensor ge(Tensor a, float s) { return a.ge(s); }
    public static Tensor ge(Tensor a, int s) { return a.ge(s); }

    public static Tensor lt(Tensor a, Tensor b) { return a.lt(b); }
    public static Tensor lt(Tensor a, Tensor b, Tensor out) { return a.lt(b, out); }
    public static Tensor lt(Tensor a, float s) { return a.lt(s); }
    public static Tensor lt(Tensor a, int s) { return a.lt(s); }

    public static Tensor le(Tensor a, Tensor b) { return a.le(b); }
    public static Tensor le(Tensor a, Tensor b, Tensor out) { return a.le(b, out); }
    public static Tensor le(Tensor a, float s) { return a.le(s); }
    public static Tensor le(Tensor a, int s) { return a.le(s); }

    // --- Masked Where ---

    public static Tensor where(Tensor cond, Tensor x, Tensor y) { return cond.where(x, y); }
    public static Tensor where(Tensor cond, Tensor x, Tensor y, Tensor out) { return cond.where(x, y, out); }
    public static Tensor where(Tensor cond, float x, Tensor y) { return cond.where(x, y); }
    public static Tensor where(Tensor cond, Tensor x, float y) { return cond.where(x, y); }
    public static Tensor where(Tensor cond, float x, float y) { return cond.where(x, y); }
    public static Tensor where(Tensor cond, int x, Tensor y) { return cond.where(x, y); }
    public static Tensor where(Tensor cond, Tensor x, int y) { return cond.where(x, y); }
    public static Tensor where(Tensor cond, int x, int y) { return cond.where(x, y); }

    // --- Type Casting ---

    public static Tensor cast(Tensor a, DType targetDType) { return a.cast(targetDType); }

    // --- Linear Algebra & Matmul ---

    public static Tensor matmul(Tensor a, Tensor b) { return a.matmul(b); }
    public static Tensor matmul(Tensor a, Tensor b, Tensor out) { return a.matmul(b, out); }
    public static Tensor dot(Tensor a, Tensor b) { return a.dot(b); }
    public static Tensor norm(Tensor a) { return a.norm(); }
    public static Tensor diagonal(Tensor a) { return a.diagonal(); }
    public static Tensor trace(Tensor a) { return a.trace(); }
    public static Tensor cosineSimilarity(Tensor a, Tensor b) { return a.cosineSimilarity(b); }

    // --- General Axis Reductions & Statistics ---

    public static float sum(Tensor a) { return a.sum(); }
    public static Tensor sum(Tensor a, int axis) { return a.sum(axis); }
    public static Tensor sum(Tensor a, int axis, boolean keepDims) { return a.sum(axis, keepDims); }

    public static float mean(Tensor a) { return a.mean(); }
    public static Tensor mean(Tensor a, int axis) { return a.mean(axis); }
    public static Tensor mean(Tensor a, int axis, boolean keepDims) { return a.mean(axis, keepDims); }

    public static Tensor max(Tensor a, int axis) { return a.max(axis); }
    public static Tensor max(Tensor a, int axis, boolean keepDims) { return a.max(axis, keepDims); }

    public static Tensor min(Tensor a, int axis) { return a.min(axis); }
    public static Tensor min(Tensor a, int axis, boolean keepDims) { return a.min(axis, keepDims); }

    public static float prod(Tensor a) { return a.prod(); }
    public static Tensor prod(Tensor a, int axis) { return a.prod(axis); }
    public static Tensor prod(Tensor a, int axis, boolean keepDims) { return a.prod(axis, keepDims); }

    public static Tensor argmax(Tensor a, int axis) { return a.argmax(axis); }
    public static Tensor argmax(Tensor a, int axis, boolean keepDims) { return a.argmax(axis, keepDims); }

    public static Tensor argmin(Tensor a, int axis) { return a.argmin(axis); }
    public static Tensor argmin(Tensor a, int axis, boolean keepDims) { return a.argmin(axis, keepDims); }

    public static Tensor var(Tensor a) { return a.var(); }
    public static Tensor var(Tensor a, int ddof) { return a.var(ddof); }

    public static Tensor std(Tensor a) { return a.std(); }
    public static Tensor std(Tensor a, int ddof) { return a.std(ddof); }

    // --- Engine Controls ---

    public static void synchronize() {
        NativeEngine.synchronize();
    }

    public static boolean isGpuAvailable() {
        return NativeEngine.isGpuAvailable();
    }
}
