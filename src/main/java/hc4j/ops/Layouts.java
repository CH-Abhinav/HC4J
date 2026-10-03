package hc4j.ops;

import hc4j.DType;
import hc4j.Tensor;
import java.util.Arrays;

/** Shared argument checks for the elementwise bindings. */
final class Layouts {

    private Layouts() {}

    static boolean isContiguous(Tensor t) {
        int[] shape = t.internalShapeUnsafe();
        int[] strides = t.internalStridesUnsafe();
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (shape[i] != 1 && strides[i] != expected) {
                return false;
            }
            expected *= shape[i];
        }
        return true;
    }

    static void requireF32(String op, Tensor... tensors) {
        for (Tensor t : tensors) {
            if (t.getDType() != DType.f32) {
                throw new IllegalArgumentException(op + " is only defined for f32 tensors, got " + t.getDType());
            }
        }
    }

    static void requireSameShape(String op, Tensor a, Tensor b) {
        if (!Arrays.equals(a.internalShapeUnsafe(), b.internalShapeUnsafe())) {
            throw new IllegalArgumentException(op + ": shape mismatch "
                    + Arrays.toString(a.internalShapeUnsafe()) + " vs " + Arrays.toString(b.internalShapeUnsafe()));
        }
    }
}
