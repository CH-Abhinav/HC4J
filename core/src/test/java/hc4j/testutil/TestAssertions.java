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

package hc4j.testutil;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Numerical assertion utilities handling tolerance, NaN, and Inf.
 */
public final class TestAssertions {

    public static final float DEFAULT_ABS_TOL = 1e-5f;
    public static final float DEFAULT_REL_TOL = 1e-5f;

    private TestAssertions() {
        throw new AssertionError("No instances of TestAssertions");
    }

    public static void assertClose(float expected, float actual) {
        assertClose(expected, actual, DEFAULT_ABS_TOL, DEFAULT_REL_TOL);
    }

    public static void assertClose(float expected, float actual, float absTol, float relTol) {
        if (Float.isNaN(expected)) {
            assertTrue(Float.isNaN(actual), () -> "Expected NaN but was " + actual);
            return;
        }
        if (Float.isInfinite(expected)) {
            assertEquals(expected, actual, () -> "Infinite value mismatch");
            return;
        }
        float diff = Math.abs(expected - actual);
        float tol = Math.max(absTol, relTol * Math.abs(expected));
        assertTrue(diff <= tol,
                () -> String.format("Expected %f but was %f (diff %e > tol %e)", expected, actual, diff, tol));
    }

    public static void assertArrayClose(float[] expected, float[] actual) {
        assertArrayClose(expected, actual, DEFAULT_ABS_TOL, DEFAULT_REL_TOL);
    }

    public static void assertArrayClose(float[] expected, float[] actual, float absTol, float relTol) {
        assertEquals(expected.length, actual.length, "Array lengths must match");
        for (int i = 0; i < expected.length; i++) {
            final int index = i;
            float e = expected[i];
            float a = actual[i];
            if (Float.isNaN(e)) {
                assertTrue(Float.isNaN(a), () -> "At index " + index + ": expected NaN but was " + a);
            } else if (Float.isInfinite(e)) {
                assertEquals(e, a, () -> "At index " + index + ": infinite mismatch");
            } else {
                float diff = Math.abs(e - a);
                float tol = Math.max(absTol, relTol * Math.abs(e));
                assertTrue(diff <= tol,
                        () -> String.format("At index %d: expected %f but was %f (diff %e > tol %e)",
                                index, e, a, diff, tol));
            }
        }
    }
}
