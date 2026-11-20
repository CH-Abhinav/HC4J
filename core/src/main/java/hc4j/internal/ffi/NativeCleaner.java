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

package hc4j.internal.ffi;

import java.lang.ref.Cleaner;

/**
 * Single shared Cleaner instance for native resource safety nets across HC4J.
 */
public final class NativeCleaner {

    public static final Cleaner CLEANER = Cleaner.create();

    private NativeCleaner() {
        throw new AssertionError("No instances of NativeCleaner");
    }

    public static Cleaner.Cleanable register(Object obj, Runnable action) {
        return CLEANER.register(obj, action);
    }
}
