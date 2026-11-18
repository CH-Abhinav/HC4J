package hc4j.collective;

/**
 * How a directed rank-to-rank hop moves its bytes. Ordinals match the native codes.
 */
public enum Transport {

    /**
     * Pinned host staging: the source rank copies its chunk into a mappable buffer, the host reads
     * it, and the destination rank uploads it. This is what the wgpu backend always uses.
     */
    HOST_STAGED,

    /**
     * A true device-to-device path through an exported OS memory handle ({@code dma_buf} on Linux,
     * {@code OPAQUE_WIN32} on Windows). Reserved for a raw-Vulkan transport: wgpu exposes no
     * external-memory export/import, so the current backend never reports it.
     */
    PEER_TO_PEER;

    static Transport of(int code) {
        Transport[] values = values();
        if (code < 0 || code >= values.length) {
            throw new IllegalStateException("Unknown native transport code " + code);
        }
        return values[code];
    }
}
