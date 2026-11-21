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

/**
 * Hardware and driver information for the active GPU adapter.
 *
 * @param name GPU device name reported by the driver (e.g. "Intel(R) UHD Graphics 620", "NVIDIA GeForce RTX 4090")
 * @param vendorId PCI vendor identifier (e.g. 0x8086 for Intel, 0x10DE for NVIDIA, 0x1002 for AMD)
 * @param deviceId PCI device identifier
 * @param deviceType Hardware category: {@code "DiscreteGpu"}, {@code "IntegratedGpu"}, {@code "VirtualGpu"}, {@code "Cpu"}, or {@code "Other"}
 * @param backend Active graphics API backend: {@code "Vulkan"}, {@code "Metal"}, {@code "Dx12"}, {@code "Gl"}, {@code "BrowserWebGpu"}, or {@code "Empty"}
 * @param driver Driver description or filename reported by the OS
 * @param driverInfo Additional driver version or build information
 */
public record AdapterInfo(
        String name,
        int vendorId,
        int deviceId,
        String deviceType,
        String backend,
        String driver,
        String driverInfo) {
}
