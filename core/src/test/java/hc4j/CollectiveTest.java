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
import hc4j.collective.Link;
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
}
