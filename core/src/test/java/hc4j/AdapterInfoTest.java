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

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import hc4j.engine.AdapterInfo;
import hc4j.engine.Engine;
import hc4j.testutil.GpuTestSupport;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("GPU Adapter & Hardware Information Tests")
class AdapterInfoTest {

    @Test
    @DisplayName("Engine.adapterInfo() returns valid GPU adapter metadata")
    void testAdapterInfo() {
        GpuTestSupport.assumeGpuAvailable();

        AdapterInfo info = Engine.adapterInfo();
        assertNotNull(info, "AdapterInfo must not be null");

        assertNotNull(info.name(), "GPU name must not be null");
        assertFalse(info.name().isBlank(), "GPU name must not be blank");

        assertNotEquals(0, info.vendorId(), "PCI vendor ID should be non-zero");

        assertNotNull(info.deviceType(), "deviceType must not be null");
        assertFalse(info.deviceType().isBlank(), "deviceType must not be blank");
        assertTrue(List.of("DiscreteGpu", "IntegratedGpu", "VirtualGpu", "Cpu", "Other").contains(info.deviceType()),
                "deviceType must be a recognized category: " + info.deviceType());

        assertNotNull(info.backend(), "backend must not be null");
        assertFalse(info.backend().isBlank(), "backend must not be blank");
        assertNotEquals("Empty", info.backend(), "Active backend must not be Empty");

        assertNotNull(info.driver(), "driver must not be null");
        assertNotNull(info.driverInfo(), "driverInfo must not be null");

        String platform = Engine.nativePlatform();
        assertNotNull(platform);
        assertFalse(platform.isBlank());

        System.out.println("Active GPU: " + info.name() + " | " + info.deviceType() + " | " + info.backend()
                + " | Vendor=0x" + Integer.toHexString(info.vendorId()) + " Device=0x" + Integer.toHexString(info.deviceId())
                + " | Driver=" + info.driver() + " (" + info.driverInfo() + ")");
    }
}
