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

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import hc4j.internal.ffi.NativeBindings;
import hc4j.internal.ffi.NativeEngine;
import hc4j.internal.ffi.TensorAccess;
import hc4j.internal.layout.ValidUtil;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.IdentityHashMap;
import java.util.Objects;

/**
 * A deferred elementwise expression over same-shape f32 tensors, evaluated in a single fused kernel.
 */
public final class FusedExpr {

    private static final int LOAD = 0x01, CONST = 0x02;
    private static final int NEG = 0x10, ABS = 0x11, SIN = 0x12, COS = 0x13, TAN = 0x14;
    private static final int ASIN = 0x15, ACOS = 0x16, ATAN = 0x17, SINH = 0x18, COSH = 0x19, TANH = 0x1A;
    private static final int ASINH = 0x1B, ACOSH = 0x1C, ATANH = 0x1D, EXP = 0x1E, LOG = 0x1F, SQRT = 0x20;
    private static final int RELU = 0x21, SIGMOID = 0x22, GELU = 0x23, SILU = 0x24, SOFTPLUS = 0x25;
    private static final int ADD = 0x40, SUB = 0x41, MUL = 0x42, DIV = 0x43, MAX = 0x44, MIN = 0x45, POW = 0x46;

    private static final int MAX_PROGRAM_WORDS = 1024;

    private sealed interface Node permits Load, Const, Unary, Binary {}
    private record Load(Tensor tensor) implements Node {}
    private record Const(float value) implements Node {}
    private record Unary(int op, Node x) implements Node {}
    private record Binary(int op, Node l, Node r) implements Node {}

    private final Node root;
    private final int[] shape;
    private final int words;

    private FusedExpr(Node root, int[] shape, int words) {
        if (words > MAX_PROGRAM_WORDS) {
            throw new IllegalStateException(
                    "Fused expression exceeds " + MAX_PROGRAM_WORDS + " program words; eval() a subexpression first");
        }
        this.root = root;
        this.shape = shape;
        this.words = words;
    }

    static FusedExpr of(Tensor t) {
        Objects.requireNonNull(t, "tensor");
        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(t);
        ValidUtil.requireOperandF32("fusion", t.getDType());
        return new FusedExpr(new Load(t), access.getShapeUnsafe(t).clone(), 1);
    }

    public FusedExpr neg()   { return unary(NEG); }
    public FusedExpr abs()   { return unary(ABS); }
    public FusedExpr sin()   { return unary(SIN); }
    public FusedExpr cos()   { return unary(COS); }
    public FusedExpr tan()   { return unary(TAN); }
    public FusedExpr asin()  { return unary(ASIN); }
    public FusedExpr acos()  { return unary(ACOS); }
    public FusedExpr atan()  { return unary(ATAN); }
    public FusedExpr sinh()  { return unary(SINH); }
    public FusedExpr cosh()  { return unary(COSH); }
    public FusedExpr tanh()  { return unary(TANH); }
    public FusedExpr asinh() { return unary(ASINH); }
    public FusedExpr acosh() { return unary(ACOSH); }
    public FusedExpr atanh() { return unary(ATANH); }
    public FusedExpr exp()   { return unary(EXP); }
    public FusedExpr log()   { return unary(LOG); }
    public FusedExpr sqrt()  { return unary(SQRT); }

    public FusedExpr relu()     { return unary(RELU); }
    public FusedExpr sigmoid()  { return unary(SIGMOID); }
    public FusedExpr gelu()     { return unary(GELU); }
    public FusedExpr silu()     { return unary(SILU); }
    public FusedExpr softplus() { return unary(SOFTPLUS); }

    public FusedExpr add(FusedExpr o) { return binary(ADD, o); }
    public FusedExpr sub(FusedExpr o) { return binary(SUB, o); }
    public FusedExpr mul(FusedExpr o) { return binary(MUL, o); }
    public FusedExpr div(FusedExpr o) { return binary(DIV, o); }
    public FusedExpr max(FusedExpr o) { return binary(MAX, o); }
    public FusedExpr min(FusedExpr o) { return binary(MIN, o); }
    public FusedExpr pow(FusedExpr o) { return binary(POW, o); }

    public FusedExpr add(Tensor t) { return binary(ADD, of(t)); }
    public FusedExpr sub(Tensor t) { return binary(SUB, of(t)); }
    public FusedExpr mul(Tensor t) { return binary(MUL, of(t)); }
    public FusedExpr div(Tensor t) { return binary(DIV, of(t)); }
    public FusedExpr max(Tensor t) { return binary(MAX, of(t)); }
    public FusedExpr min(Tensor t) { return binary(MIN, of(t)); }

    public FusedExpr add(float c) { return scalar(ADD, c); }
    public FusedExpr rsub(float c) { return reverseScalar(SUB, c); }
    public FusedExpr rdiv(float c) { return reverseScalar(DIV, c); }
    public FusedExpr rpow(float c) { return reverseScalar(POW, c); }
    public FusedExpr sub(float c) { return scalar(SUB, c); }
    public FusedExpr mul(float c) { return scalar(MUL, c); }
    public FusedExpr div(float c) { return scalar(DIV, c); }
    public FusedExpr pow(float c) { return scalar(POW, c); }

    private FusedExpr unary(int op) {
        return new FusedExpr(new Unary(op, root), shape, words + 1);
    }

    private FusedExpr binary(int op, FusedExpr other) {
        Objects.requireNonNull(other, "other");
        ValidUtil.requireSameShape(shape, other.shape);
        return new FusedExpr(new Binary(op, root, other.root), shape, words + other.words + 1);
    }

    private FusedExpr reverseScalar(int op, float c) {
        requireFinite(c);
        return new FusedExpr(new Binary(op, new Const(c), root), shape, words + 3);
    }

    private FusedExpr scalar(int op, float c) {
        requireFinite(c);
        return new FusedExpr(new Binary(op, root, new Const(c)), shape, words + 3);
    }

    public Tensor eval() {
        Tensor res = Tensor.empty(DType.f32, shape.clone());
        try {
            return eval(res);
        } catch (RuntimeException | Error e) {
            res.close();
            throw e;
        }
    }

    public Tensor eval(Tensor res) {
        Objects.requireNonNull(res, "res");
        TensorAccess access = TensorAccess.getInstance();
        access.ensureOpen(res);
        ValidUtil.requireResultDType(DType.f32, res.getDType());
        ValidUtil.requireResultShape(shape, access.getShapeUnsafe(res));

        IdentityHashMap<Tensor, Integer> inputs = new IdentityHashMap<>();
        int[] program = new int[words];
        int pc = 0;

        Deque<Object> work = new ArrayDeque<>();
        work.push(root);
        while (!work.isEmpty()) {
            Object item = work.pop();
            if (item instanceof Integer opWord) {
                program[pc++] = opWord;
                continue;
            }
            switch ((Node) item) {
                case Load(Tensor t) -> {
                    access.ensureOpen(t);
                    ValidUtil.requireOperandF32("fusion input", t.getDType());
                    ValidUtil.requireSameShape(shape, access.getShapeUnsafe(t));
                    int k = inputs.computeIfAbsent(t, unused -> inputs.size());
                    program[pc++] = word(LOAD, k);
                }
                case Const(float v) -> {
                    program[pc++] = word(CONST, 0);
                    program[pc++] = Float.floatToRawIntBits(v);
                }
                case Unary(int op, Node x) -> {
                    work.push(word(op, 0));
                    work.push(x);
                }
                case Binary(int op, Node l, Node r) -> {
                    work.push(word(op, 0));
                    work.push(r);
                    work.push(l);
                }
            }
        }

        long[] handles = new long[inputs.size()];
        inputs.forEach((t, k) -> {
            access.ensureOpen(t);
            handles[k] = access.getVramId(t);
        });

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment prog = arena.allocateFrom(JAVA_INT, Arrays.copyOf(program, pc));
            MemorySegment ids = arena.allocateFrom(JAVA_LONG, handles);
            int status = NativeBindings.dispatchFused(prog, pc, ids, handles.length, access.getVramId(res));
            NativeEngine.checkStatus(status, "FusedExpr.eval");
            return res;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("HC4J fused dispatch failed", t);
        }
    }

    private static void requireFinite(float c) {
        if (!Float.isFinite(c)) {
            throw new IllegalArgumentException("Fused constants must be finite: " + c);
        }
    }

    private static int word(int opcode, int operand) {
        return (opcode << 24) | (operand & 0x00FF_FFFF);
    }
}
