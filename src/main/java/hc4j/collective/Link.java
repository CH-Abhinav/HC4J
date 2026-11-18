package hc4j.collective;

/**
 * One directed edge of a collective's topology graph, as measured by the bandwidth probe.
 *
 * @param src the sending rank
 * @param dst the receiving rank
 * @param gbps measured throughput of the hop in GB/s, including the host round trip
 * @param transport how the hop moves its bytes
 * @param loopback true when both ranks sit on the same physical GPU, so {@code gbps} is an
 *     intra-GPU copy rather than a link between two devices — do not read it as a PCIe figure
 */
public record Link(int src, int dst, double gbps, Transport transport, boolean loopback) {

    @Override
    public String toString() {
        return String.format(
                "rank %d -> %d: %.2f GB/s %s%s", src, dst, gbps, transport, loopback ? " (loopback)" : "");
    }
}
