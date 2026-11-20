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

package hc4j.collective;

/**
 * How a directed rank-to-rank hop moves its bytes. Ordinals match native codes.
 */
public enum Transport {

    /**
     * Pinned host staging: the source rank copies its chunk into a mappable buffer,
     * the host reads it, and the destination rank uploads it.
     */
    HOST_STAGED,

    /**
     * True peer-to-peer device memory transfer.
     */
    PEER_TO_PEER;

    public static Transport of(int code) {
        Transport[] values = values();
        if (code < 0 || code >= values.length) {
            throw new IllegalStateException("Unknown native transport code " + code);
        }
        return values[code];
    }
}
