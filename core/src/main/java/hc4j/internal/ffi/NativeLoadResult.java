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

import java.lang.foreign.SymbolLookup;
import java.nio.file.Path;

/**
 * Result of loading the HC4J native library.
 *
 * @param lookup Panama FFM symbol lookup for downcalls
 * @param libraryPath Absolute path of the loaded native library binary on the local filesystem
 * @param source Description of the resolution stage that supplied the binary
 * @param platform Detected platform metadata
 */
public record NativeLoadResult(
        SymbolLookup lookup,
        Path libraryPath,
        String source,
        Platform platform) {
}
