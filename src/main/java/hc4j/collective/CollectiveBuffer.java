package hc4j.collective;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * An f32 buffer resident on one rank of a {@link Collective}.
 *
 * <p>Separate from {@link hc4j.Tensor}: a collective owns its own devices, so data crosses in and
 * out through {@link #write} and {@link #read}. Closing the owning collective releases every
 * buffer it handed out, so closing a buffer afterwards is a no-op.
 */
public final class CollectiveBuffer implements AutoCloseable {

    private final Collective collective;
    private final long handle;
    private final int rank;
    private final long elements;
    private final AtomicBoolean closed = new AtomicBoolean();

    CollectiveBuffer(Collective collective, long handle, int rank, long elements) {
        this.collective = collective;
        this.handle = handle;
        this.rank = rank;
        this.elements = elements;
    }

    /** The rank this buffer lives on. */
    public int rank() {
        return rank;
    }

    /** Number of f32 elements. */
    public long elements() {
        return elements;
    }

    /**
     * Uploads {@code data} over the whole buffer.
     *
     * @throws IllegalArgumentException if the length does not match {@link #elements()}
     */
    public void write(float[] data) {
        checkOpen();
        if (data.length != elements) {
            throw new IllegalArgumentException(
                    "write expects " + elements + " elements, got " + data.length);
        }
        collective.writeBuffer(handle, data);
    }

    /** Reads the whole buffer back to the host. */
    public float[] read() {
        checkOpen();
        if (elements > Integer.MAX_VALUE) {
            throw new IllegalStateException("buffer is larger than a Java array can hold");
        }
        return collective.readBuffer(handle, elements);
    }

    /** Frees the buffer. Idempotent. */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            collective.freeBuffer(handle);
        }
    }

    @Override
    public String toString() {
        return "CollectiveBuffer[rank=" + rank + ", elements=" + elements + "]";
    }

    Collective collective() {
        return collective;
    }

    long handleUnsafe() {
        checkOpen();
        return handle;
    }

    private void checkOpen() {
        if (closed.get()) {
            throw new IllegalStateException("this CollectiveBuffer is closed");
        }
    }
}
