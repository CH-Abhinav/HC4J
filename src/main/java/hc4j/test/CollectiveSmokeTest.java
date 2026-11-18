package hc4j.test;

import hc4j.collective.Collective;
import hc4j.collective.CollectiveBuffer;
import hc4j.collective.Link;
import hc4j.collective.Transport;
import java.util.List;

/**
 * End-to-end validation of the collective layer against the real GPU, in the three phases of the
 * reference design: adapter enumeration, topology discovery, and ring AllReduce verified against
 * the closed-form sum.
 *
 * <p>Run with {@code gradlew collectiveSmokeTest}. On a single-GPU machine the ring runs in
 * loopback mode, which the topology report marks. Exits non-zero on the first failure.
 */
public final class CollectiveSmokeTest {

    private static int checks = 0;

    public static void main(String[] args) {
        enumerationAndTopology();
        allReduceTwoRanks();
        allReduceFourRanks();
        allReduceLargeBuffer();
        validation();
        System.out.printf("%nALL %d CHECKS PASSED%n", checks);
    }

    // ------------------------------------------------------------------------------------------

    private static void enumerationAndTopology() {
        section("Phase 1: ranks and topology discovery");
        try (Collective ring = Collective.open(2)) {
            expect(ring.ranks() == 2, "a 2-rank ring opened");
            for (int r = 0; r < ring.ranks(); r++) {
                System.out.printf("      rank %d: %s%n", r, ring.deviceName(r));
                expect(!ring.deviceName(r).isBlank(), "rank " + r + " reports an adapter name");
            }

            System.out.println("      probing every directed hop (this moves a few hundred MiB)...");
            List<Link> links = ring.topology();
            expect(links.size() == 2, "two ranks give two directed links");
            for (Link link : links) {
                System.out.println("      " + link);
                expect(link.src() != link.dst(), "a link connects two different ranks");
                expect(link.gbps() > 0, "link " + link.src() + "->" + link.dst() + " measured throughput");
                expect(link.transport() == Transport.HOST_STAGED,
                        "wgpu has no peer-to-peer path, so the hop is host-staged");
            }
            printMatrix(ring, links);
            boolean sameGpu = ring.deviceName(0).equals(ring.deviceName(1));
            if (sameGpu) {
                expect(links.stream().allMatch(Link::loopback),
                        "both ranks are on one GPU, and every link says so");
            }
            // Cached: a second call must not re-probe.
            long t0 = System.nanoTime();
            ring.topology();
            double ms = (System.nanoTime() - t0) / 1e6;
            expect(ms < 50, String.format("the topology is cached (re-read in %.2f ms)", ms));
        }
    }

    private static void allReduceTwoRanks() {
        section("Phase 3: ring AllReduce, 2 ranks");
        verifyAllReduce(2, 6);
    }

    private static void allReduceFourRanks() {
        section("Phase 3: ring AllReduce, 4 ranks");
        // Four ranks means three scatter-reduce rounds and three allgather rounds.
        verifyAllReduce(4, 4096);
    }

    private static void allReduceLargeBuffer() {
        section("Phase 3: ring AllReduce, 1 Mi elements");
        // 4 MiB per rank: the reduction kernel runs over many workgroups and
        // each hop stages 2 MiB through the host.
        verifyAllReduce(2, 1 << 20);
    }

    /**
     * Rank {@code r} contributes {@code r * n + i + 1}, so the expected sum is known in closed
     * form and a wrong chunk or a missed round shows up as a specific mismatch.
     */
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

                long t0 = System.nanoTime();
                ring.allReduceSum(bufs);
                double ms = (System.nanoTime() - t0) / 1e6;

                float[] want = new float[elements];
                for (int i = 0; i < elements; i++) {
                    double sum = 0;
                    for (int r = 0; r < ranks; r++) {
                        sum += (double) r * elements + i + 1;
                    }
                    want[i] = (float) sum;
                }

                double worst = 0;
                int worstRank = -1;
                for (int r = 0; r < ranks; r++) {
                    float[] got = bufs[r].read();
                    for (int i = 0; i < elements; i++) {
                        double err = Math.abs(got[i] - want[i]) / Math.max(1.0, Math.abs(want[i]));
                        if (err > worst) {
                            worst = err;
                            worstRank = r;
                        }
                    }
                }
                expect(worst < 1e-6, String.format(
                        "every rank holds the sum (%d ranks x %d elements, max rel err %.2e%s)",
                        ranks, elements, worst, worst > 0 ? " on rank " + worstRank : ""));
                double moved = 2.0 * (ranks - 1) / ranks * elements * 4 / (1 << 20);
                System.out.printf("      %.1f ms, %.2f MiB staged per rank%n", ms, moved);
            } finally {
                for (CollectiveBuffer buf : bufs) {
                    if (buf != null) {
                        buf.close();
                    }
                }
            }
        }
    }

    private static void validation() {
        section("Validation");
        expectThrows(IllegalArgumentException.class, () -> Collective.open(1),
                "a ring of one rank");

        try (Collective ring = Collective.open(2)) {
            CollectiveBuffer a = ring.allocate(0, 8);
            CollectiveBuffer b = ring.allocate(1, 8);
            CollectiveBuffer odd = ring.allocate(1, 4);
            CollectiveBuffer sevenA = ring.allocate(0, 7);
            CollectiveBuffer sevenB = ring.allocate(1, 7);

            expectThrows(IllegalArgumentException.class, () -> ring.allocate(2, 8),
                    "allocating on a rank that does not exist");
            expectThrows(IllegalArgumentException.class, () -> ring.allReduceSum(a),
                    "fewer buffers than ranks");
            expectThrows(IllegalArgumentException.class, () -> ring.allReduceSum(b, a),
                    "buffers out of rank order");
            expectThrows(IllegalArgumentException.class, () -> ring.allReduceSum(a, odd),
                    "ranks contributing different lengths");
            expectThrows(IllegalArgumentException.class, () -> ring.allReduceSum(sevenA, sevenB),
                    "a length that does not divide by the rank count");
            expectThrows(IllegalArgumentException.class, () -> a.write(new float[3]),
                    "writing the wrong number of elements");

            a.close();
            expectThrows(IllegalStateException.class, a::read, "reading a closed buffer");

            ring.close();
            expectThrows(IllegalStateException.class, () -> ring.allocate(0, 8),
                    "allocating on a closed collective");
            // Buffers outlive nothing: the collective already freed them, and
            // closing them again must stay quiet.
            b.close();
            odd.close();
            sevenA.close();
            sevenB.close();
            expect(true, "closing buffers after their collective is a no-op");
        }
    }

    // ------------------------------------------------------------------------------------------

    /** The topology as an adjacency matrix of measured GB/s, as the reference design reports it. */
    private static void printMatrix(Collective ring, List<Link> links) {
        int n = ring.ranks();
        StringBuilder header = new StringBuilder("        ");
        for (int j = 0; j < n; j++) {
            header.append(String.format("rank%-6d", j));
        }
        System.out.println("      adjacency matrix (GB/s, [L] = loopback):");
        System.out.println("      " + header);
        for (int i = 0; i < n; i++) {
            StringBuilder row = new StringBuilder(String.format("  rank%d ", i));
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    row.append("   ---    ");
                    continue;
                }
                final int src = i;
                final int dst = j;
                Link link = links.stream()
                        .filter(l -> l.src() == src && l.dst() == dst)
                        .findFirst()
                        .orElseThrow(() -> new IllegalStateException("no link for " + src + "->" + dst));
                row.append(String.format("%6.2f%-4s", link.gbps(), link.loopback() ? " [L]" : ""));
            }
            System.out.println("      " + row);
        }
    }

    private static void section(String title) {
        System.out.println("\n== " + title);
    }

    private static void expect(boolean condition, String what) {
        checks++;
        System.out.println((condition ? "  PASS " : "  FAIL ") + what);
        if (!condition) {
            System.exit(1);
        }
    }

    private static void expectThrows(Class<? extends Throwable> type, Runnable body, String what) {
        try {
            body.run();
            expect(false, what + " should throw " + type.getSimpleName());
        } catch (Throwable t) {
            expect(type.isInstance(t), what + " -> " + t.getClass().getSimpleName());
        }
    }
}
