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
 * Thrown when a request could not be satisfied by any memory tier: VRAM (after LRU eviction),
 * host RAM, and the disk spill tier were all exhausted. Unlike {@link OutOfMemoryError}, this
 * says nothing about the Java heap and is safe to catch, e.g. to retry with a smaller batch.
 */
public final class GpuOutOfMemoryException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public GpuOutOfMemoryException(String message) {
        super(message);
    }
}
