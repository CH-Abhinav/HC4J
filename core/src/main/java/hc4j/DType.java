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

import java.lang.foreign.ValueLayout;

/**
 * Data types supported by HC4J tensors.
 */
public enum DType {
    i32(ValueLayout.JAVA_INT),
    f32(ValueLayout.JAVA_FLOAT),
    f16(ValueLayout.JAVA_SHORT);

    public final ValueLayout layout;

    DType(ValueLayout layout) {
        this.layout = layout;
    }

    public long byteSize() {
        return layout.byteSize();
    }
}
