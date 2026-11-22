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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("Slice Construction and Resolution Tests")
class SliceTest {

    @Test
    @DisplayName("Slice factories and sentinels")
    void testFactories() {
        Slice all = Slice.all();
        assertEquals(Slice.UNBOUNDED_START, all.start());
        assertEquals(Slice.UNBOUNDED_STOP, all.stop());
        assertEquals(1, all.step());

        Slice to = Slice.to(5);
        assertEquals(Slice.UNBOUNDED_START, to.start());
        assertEquals(5, to.stop());
        assertEquals(1, to.step());

        Slice from = Slice.from(2);
        assertEquals(2, from.start());
        assertEquals(Slice.UNBOUNDED_STOP, from.stop());
        assertEquals(1, from.step());

        Slice range = Slice.range(1, 10);
        assertEquals(1, range.start());
        assertEquals(10, range.stop());
        assertEquals(1, range.step());

        Slice rangeStep = Slice.range(1, 10, 2);
        assertEquals(1, rangeStep.start());
        assertEquals(10, rangeStep.stop());
        assertEquals(2, rangeStep.step());

        Slice step = Slice.step(3);
        assertEquals(Slice.UNBOUNDED_START, step.start());
        assertEquals(Slice.UNBOUNDED_STOP, step.stop());
        assertEquals(3, step.step());

        Slice atPositive = Slice.at(3);
        assertEquals(3, atPositive.start());
        assertEquals(4, atPositive.stop());
        assertEquals(1, atPositive.step());
    }

    @Test
    @DisplayName("Non-positive step is rejected")
    void testInvalidStep() {
        assertThrows(IllegalArgumentException.class, () -> new Slice(0, 10, 0));
        assertThrows(IllegalArgumentException.class, () -> new Slice(0, 10, -1));
        assertThrows(IllegalArgumentException.class, () -> Slice.range(0, 10, 0));
        assertThrows(IllegalArgumentException.class, () -> Slice.range(0, 10, -2));
        assertThrows(IllegalArgumentException.class, () -> Slice.step(0));
    }

    @Test
    @DisplayName("Slice resolution on dimension sizes")
    void testResolution() {
        Slice.ResolvedSlice r1 = Slice.all().resolve(10);
        assertEquals(0, r1.start());
        assertEquals(10, r1.stop());
        assertEquals(1, r1.step());
        assertEquals(10, r1.length());

        Slice.ResolvedSlice r2 = Slice.to(5).resolve(10);
        assertEquals(0, r2.start());
        assertEquals(5, r2.stop());
        assertEquals(1, r2.step());
        assertEquals(5, r2.length());

        Slice.ResolvedSlice r3 = Slice.from(3).resolve(10);
        assertEquals(3, r3.start());
        assertEquals(10, r3.stop());
        assertEquals(1, r3.step());
        assertEquals(7, r3.length());

        Slice.ResolvedSlice r4 = Slice.range(2, 8, 2).resolve(10);
        assertEquals(2, r4.start());
        assertEquals(8, r4.stop());
        assertEquals(2, r4.step());
        assertEquals(3, r4.length()); // 2, 4, 6

        // Negative indices
        Slice.ResolvedSlice rNeg = Slice.range(-5, -1).resolve(10);
        assertEquals(5, rNeg.start());
        assertEquals(9, rNeg.stop());
        assertEquals(1, rNeg.step());
        assertEquals(4, rNeg.length()); // 5, 6, 7, 8

        // Slice.at with negative index
        Slice.ResolvedSlice rAtLast = Slice.at(-1).resolve(10);
        assertEquals(9, rAtLast.start());
        assertEquals(10, rAtLast.stop());
        assertEquals(1, rAtLast.length());

        Slice.ResolvedSlice rAtThirdFromEnd = Slice.at(-3).resolve(10);
        assertEquals(7, rAtThirdFromEnd.start());
        assertEquals(8, rAtThirdFromEnd.stop());
        assertEquals(1, rAtThirdFromEnd.length());

        // Out-of-bounds clamping
        Slice.ResolvedSlice rClamp = Slice.range(-20, 25).resolve(10);
        assertEquals(0, rClamp.start());
        assertEquals(10, rClamp.stop());
        assertEquals(10, rClamp.length());

        // Inverted range produces zero length
        Slice.ResolvedSlice rInverted = Slice.range(8, 2).resolve(10);
        assertEquals(8, rInverted.start());
        assertEquals(2, rInverted.stop());
        assertEquals(0, rInverted.length());
    }
}
