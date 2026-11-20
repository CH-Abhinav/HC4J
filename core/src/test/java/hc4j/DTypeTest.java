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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("DType Headless Unit Tests")
class DTypeTest {

    @Test
    @DisplayName("DType constants expose correct layouts and byte sizes")
    void testLayoutsAndByteSizes() {
        assertSame(ValueLayout.JAVA_FLOAT, DType.f32.layout);
        assertEquals(4, DType.f32.byteSize());

        assertSame(ValueLayout.JAVA_INT, DType.i32.layout);
        assertEquals(4, DType.i32.byteSize());
    }

    @Test
    @DisplayName("DType enum valueOf and values consistency")
    void testEnumValues() {
        DType[] values = DType.values();
        assertEquals(2, values.length);
        assertSame(DType.i32, DType.valueOf("i32"));
        assertSame(DType.f32, DType.valueOf("f32"));
        assertNotNull(DType.f32.toString());
    }
}
