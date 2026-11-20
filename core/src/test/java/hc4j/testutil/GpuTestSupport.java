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

import hc4j.engine.Engine;
import org.junit.jupiter.api.Assumptions;

/**
 * GPU test integration helper for checking GPU availability and skipping when running in headless environments.
 */
public final class GpuTestSupport {

    private GpuTestSupport() {
        throw new AssertionError("No instances of GpuTestSupport");
    }

    public static boolean isGpuAvailable() {
        return Engine.isGpuAvailable();
    }

    public static void assumeGpuAvailable() {
        Assumptions.assumeTrue(isGpuAvailable(),
                "GPU or native library is not available; skipping GPU integration test.");
    }
}
