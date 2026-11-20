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

import java.util.SplittableRandom;

/**
 * Deterministic test array generators.
 */
public final class TestArrayFactory {

    private TestArrayFactory() {
        throw new AssertionError("No instances of TestArrayFactory");
    }

    public static float[] ramp(int n, float lo, float hi) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = lo + (hi - lo) * i / Math.max(1, n - 1);
        }
        return out;
    }

    public static float[] random(long seed, int n, float lo, float hi) {
        SplittableRandom rng = new SplittableRandom(seed);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) (lo + (hi - lo) * rng.nextDouble());
        }
        return out;
    }

    public static int[] randomInts(long seed, int n, int bound) {
        SplittableRandom rng = new SplittableRandom(seed);
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = rng.nextInt(bound);
        }
        return out;
    }

    public static float[] pattern(int n, int seed) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) ((i * 31L + seed * 7919L) % 10007) / 10007f * 6f - 3f;
        }
        return out;
    }
}
