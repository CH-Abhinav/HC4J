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
