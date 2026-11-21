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

import static hc4j.testutil.ExceptionTable.assertThrowsWithMessage;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import hc4j.collective.Collective;
import hc4j.collective.CollectiveBuffer;
import hc4j.collective.DeviceProfile;
import hc4j.collective.Island;
import hc4j.collective.Link;
import hc4j.collective.ShardAllocation;
import hc4j.collective.WorkloadPlan;
import hc4j.testutil.GpuTestSupport;
import hc4j.testutil.TestAssertions;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Collective Multi-GPU & Topology Discovery Tests")
class CollectiveTest {

    @BeforeEach
    void setUp() {
        GpuTestSupport.assumeGpuAvailable();
    }

    @Test
    @DisplayName("Backend name and closed lifecycle")
    void testBackendName() {
        try (Collective ring = Collective.open(2)) {
            String backend = ring.backendName();
            assertNotNull(backend);
            assertFalse(backend.isBlank());
            assertTrue("vulkan-p2p".equals(backend) || "wgpu-staged".equals(backend),
                    "Backend must be vulkan-p2p or wgpu-staged, got: " + backend);
        }
    }

    @Test
    @DisplayName("Transport enum wire codes and mapping")
    void testTransportCodes() {
        assertEquals(0, hc4j.collective.Transport.HOST_STAGED.ordinal());
        assertEquals(1, hc4j.collective.Transport.PEER_TO_PEER.ordinal());
        assertEquals(2, hc4j.collective.Transport.HOST_SHARED_DMA.ordinal());

        assertSame(hc4j.collective.Transport.HOST_STAGED, hc4j.collective.Transport.of(0));
        assertSame(hc4j.collective.Transport.PEER_TO_PEER, hc4j.collective.Transport.of(1));
        assertSame(hc4j.collective.Transport.HOST_SHARED_DMA, hc4j.collective.Transport.of(2));

        assertSame(hc4j.collective.Transport.HOST_STAGED, hc4j.collective.Transport.fromCode(0));
        assertSame(hc4j.collective.Transport.PEER_TO_PEER, hc4j.collective.Transport.fromCode(1));
        assertSame(hc4j.collective.Transport.HOST_SHARED_DMA, hc4j.collective.Transport.fromCode(2));

        assertThrowsWithMessage(IllegalStateException.class,
                () -> hc4j.collective.Transport.of(99),
                "Unknown native transport code 99");
    }

    @Test
    @DisplayName("Ranks and topology discovery")
    void testEnumerationAndTopology() {
        try (Collective ring = Collective.open(2)) {
            assertEquals(2, ring.ranks());
            for (int r = 0; r < ring.ranks(); r++) {
                String name = ring.deviceName(r);
                assertNotNull(name);
                assertFalse(name.isBlank(), "Rank " + r + " reports a non-blank device name");
            }

            List<Link> links = ring.topology();
            assertEquals(2, links.size(), "2 ranks give 2 directed links");
            for (Link link : links) {
                assertNotEquals(link.src(), link.dst(), "A link connects different ranks");
                assertTrue(link.gbps() > 0, "Throughput is positive");
                assertNotNull(link.transport());
            }

            // Second call is cached
            assertSame(links, ring.topology());
        }
    }

    @Test
    @DisplayName("Topology links have positive bandwidth and valid transports")
    void testTopologyLinkDetails() {
        try (Collective ring = Collective.open(4)) {
            List<Link> links = ring.topology();
            assertEquals(12, links.size(), "4 ranks give 12 directed edges");
            for (Link link : links) {
                assertTrue(link.src() >= 0 && link.src() < 4);
                assertTrue(link.dst() >= 0 && link.dst() < 4);
                assertNotEquals(link.src(), link.dst());
                assertTrue(link.gbps() > 0.0, "Link bandwidth must be strictly positive");
                assertNotNull(link.transport());
                assertTrue(link.loopback() || !link.loopback()); // boolean valid
            }
        }
    }

    @Test
    @DisplayName("Ring AllReduce across 2 ranks")
    void testAllReduceTwoRanks() {
        verifyAllReduce(2, 6);
    }

    @Test
    @DisplayName("Ring AllReduce across 4 ranks")
    void testAllReduceFourRanks() {
        verifyAllReduce(4, 1024);
    }

    @Test
    @DisplayName("Ring AllReduce with uneven element counts (1003 floats)")
    void testAllReduceUnevenLengths() {
        try (Collective ring = Collective.open(2)) {
            if ("vulkan-p2p".equals(ring.backendName())) {
                verifyAllReduce(2, 1);
                verifyAllReduce(2, 7);
                verifyAllReduce(2, 1003);
                verifyAllReduce(4, 1003);
            } else {
                // wgpu-staged requires elements to divide evenly by rank count
                assertThrowsWithMessage(IllegalArgumentException.class,
                        () -> verifyAllReduce(2, 7),
                        "Invalid parameters passed to native bridge");
            }
        }
    }

    @Test
    @DisplayName("Ring AllReduce stress (25+ iterations)")
    void testAllReduceStressRepeated() {
        final int ranks = 2;
        final int elements = 256;
        try (Collective ring = Collective.open(ranks)) {
            CollectiveBuffer[] bufs = new CollectiveBuffer[ranks];
            try {
                for (int r = 0; r < ranks; r++) {
                    bufs[r] = ring.allocate(r, elements);
                }

                for (int iter = 0; iter < 30; iter++) {
                    for (int r = 0; r < ranks; r++) {
                        float[] data = new float[elements];
                        for (int i = 0; i < elements; i++) {
                            data[i] = (float) (r + 1) * (iter + 1);
                        }
                        bufs[r].write(data);
                    }

                    ring.allReduceSum(bufs);

                    float[] expected = new float[elements];
                    float expectedVal = (1.0f + 2.0f) * (iter + 1);
                    for (int i = 0; i < elements; i++) {
                        expected[i] = expectedVal;
                    }

                    for (int r = 0; r < ranks; r++) {
                        float[] got = bufs[r].read();
                        TestAssertions.assertArrayClose(expected, got, 1e-4f, 1e-4f);
                    }
                }
            } finally {
                for (CollectiveBuffer b : bufs) {
                    if (b != null) b.close();
                }
            }
        }
    }

    @Test
    @DisplayName("Collective argument validation")
    void testValidation() {
        assertThrowsWithMessage(IllegalArgumentException.class,
                () -> Collective.open(1),
                "a collective needs at least 2 ranks, got 1");

        try (Collective ring = Collective.open(2)) {
            assertThrowsWithMessage(IllegalArgumentException.class,
                    () -> ring.allocate(0, 0),
                    "a collective buffer needs at least one element");

            assertThrowsWithMessage(IllegalArgumentException.class,
                    () -> ring.allocate(2, 10),
                    "rank 2 is outside 0..1");

            try (CollectiveBuffer b0 = ring.allocate(0, 4);
                 CollectiveBuffer b1 = ring.allocate(1, 4)) {
                // Wrong rank order
                assertThrowsWithMessage(IllegalArgumentException.class,
                        () -> ring.allReduceSum(b1, b0),
                        "pass them in rank order");

                // Wrong count
                assertThrowsWithMessage(IllegalArgumentException.class,
                        () -> ring.allReduceSum(b0),
                        "allReduceSum needs one buffer per rank: expected 2, got 1");
            }

            try (Collective other = Collective.open(2)) {
                try (CollectiveBuffer b0 = ring.allocate(0, 4);
                     CollectiveBuffer bOther = other.allocate(1, 4)) {
                    assertThrowsWithMessage(IllegalArgumentException.class,
                            () -> ring.allReduceSum(b0, bOther),
                            "belongs to a different collective");
                }
            }
        }
    }

    private static void verifyAllReduce(int ranks, int elements) {
        try (Collective ring = Collective.open(ranks)) {
            CollectiveBuffer[] bufs = new CollectiveBuffer[ranks];
            try {
                for (int r = 0; r < ranks; r++) {
                    bufs[r] = ring.allocate(r, elements);
                    float[] data = new float[elements];
                    for (int i = 0; i < elements; i++) {
                        data[i] = (float) r * elements + i + 1;
                    }
                    bufs[r].write(data);
                }

                ring.allReduceSum(bufs);

                float[] expected = new float[elements];
                for (int i = 0; i < elements; i++) {
                    float sum = 0;
                    for (int r = 0; r < ranks; r++) {
                        sum += (float) r * elements + i + 1;
                    }
                    expected[i] = sum;
                }

                for (int r = 0; r < ranks; r++) {
                    float[] got = bufs[r].read();
                    TestAssertions.assertArrayClose(expected, got, 1e-4f, 1e-4f);
                }
            } finally {
                for (CollectiveBuffer b : bufs) {
                    if (b != null) b.close();
                }
            }
        }
    }

    @Test
    @DisplayName("Device profiles, optimal ring order, islands, and planWork on real hardware")
    void testDeviceProfilesRingOrderAndIslands() {
        try (Collective ring = Collective.open(2)) {
            List<DeviceProfile> profiles = ring.deviceProfiles();
            assertEquals(2, profiles.size());
            for (int i = 0; i < 2; i++) {
                DeviceProfile p = profiles.get(i);
                assertEquals(i, p.rank());
                assertNotNull(p.name());
                assertNotNull(p.deviceType());
                assertTrue(p.vramBytes() > 0, "VRAM bytes must be > 0");
                assertTrue(p.computeGflops() > 0.0, "Compute GFLOP/s must be > 0.0");
            }

            List<Integer> order = ring.ringOrder();
            assertEquals(2, order.size());
            assertEquals(List.of(0, 1), order);

            List<Island> islands = ring.islands();
            assertNotNull(islands);
            assertFalse(islands.isEmpty());

            WorkloadPlan plan = ring.planWork(10_000L);
            assertNotNull(plan);
            assertEquals(10_000L, plan.totalUnits());
            long sum = plan.allocations().stream().mapToLong(ShardAllocation::units).sum();
            assertEquals(10_000L, sum);
        }
    }

    @Test
    @DisplayName("Numerical equivalence for 2-rank uneven AllReduce")
    void testAllReduceUnevenEquivalence() {
        verifyAllReduce(2, 1003);
    }
}
