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

import java.util.List;
import java.util.Objects;

/**
 * A cluster of ranks connected by high-speed intra-island links.
 *
 * @param id the zero-based island identifier
 * @param leaderRank the chosen leader rank that represents the island in inter-island collectives
 * @param ranks the immutable list of ranks participating in this island
 * @param internalTransport the transport tier used for internal transfers within this island
 */
public record Island(
        int id,
        int leaderRank,
        List<Integer> ranks,
        Transport internalTransport) {

    /**
     * Constructs and defensively copies an {@link Island}.
     *
     * @param id island ID (non-negative)
     * @param leaderRank leader rank ID (non-negative)
     * @param ranks member ranks (non-null, non-empty)
     * @param internalTransport internal transport tier (non-null)
     */
    public Island {
        if (id < 0) {
            throw new IllegalArgumentException("id must be non-negative: " + id);
        }
        if (leaderRank < 0) {
            throw new IllegalArgumentException("leaderRank must be non-negative: " + leaderRank);
        }
        Objects.requireNonNull(ranks, "ranks must not be null");
        Objects.requireNonNull(internalTransport, "internalTransport must not be null");
        ranks = List.copyOf(ranks);
        if (ranks.isEmpty()) {
            throw new IllegalArgumentException("ranks must not be empty");
        }
        if (!ranks.contains(leaderRank)) {
            throw new IllegalArgumentException("leaderRank " + leaderRank + " must belong to ranks " + ranks);
        }
    }
}
