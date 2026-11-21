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

package hc4j.engine;

import hc4j.internal.ffi.NativeEngine;

/**
 * Public hardware engine facade providing device initialization, residency query,
 * memory configuration, stats, batching, and synchronization.
 */
public final class Engine {

    private Engine() {
        throw new AssertionError("No instances of Engine");
    }

    /** Initializes the GPU compute engine. */
    public static void initialize() {
        NativeEngine.initialize();
    }

    /** Checks whether a functional GPU device and native backend are available. */
    public static boolean isGpuAvailable() {
        return NativeEngine.isGpuAvailable();
    }

    /** Submits all pending GPU work and blocks until execution is complete. */
    public static void synchronize() {
        NativeEngine.synchronize();
    }

    /** Returns engine command stream and device capability statistics. */
    public static EngineStats engineStats() {
        return NativeEngine.engineStats();
    }

    /** Returns tiered memory manager statistics. */
    public static MemoryStats memoryStats() {
        return NativeEngine.memoryStats();
    }

    /**
     * Configures the VRAM and host RAM memory budgets in bytes.
     * Pass 0 to leave a budget unchanged.
     */
    public static void configureMemory(long vramBudget, long hostBudget) {
        NativeEngine.configureMemory(vramBudget, hostBudget);
    }

    /** Opens a command-batching scope. */
    public static GpuBatch openBatch() {
        return GpuBatch.open();
    }

    /** Returns information about the active GPU adapter and driver. */
    public static AdapterInfo adapterInfo() {
        return NativeEngine.adapterInfo();
    }

    /** Returns the canonical platform identifier string (e.g. "windows-x86_64"). */
    public static String nativePlatform() {
        return NativeEngine.platform().classifier();
    }
}
