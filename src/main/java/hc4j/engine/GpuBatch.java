package hc4j.engine;

/**
 * A command-batching scope. Every GPU op issued while a batch is open is recorded into one shared
 * command buffer and submitted once when the outermost scope closes, eliminating a
 * {@code queue.submit} per op for loops of small operations.
 *
 * <pre>{@code
 * try (GpuBatch batch = GpuBatch.open()) {
 *     for (int i = 0; i < 100; i++) x.sin(x);
 * } // one submission
 * }</pre>
 *
 * <p>Semantics are unchanged by batching: anything that needs results on the host (downloads,
 * writes, evictions) flushes the batch first. Errors raised while submitting the batch surface
 * from {@link #close()}. Scopes nest, and the batch is engine-wide: ops from other threads issued
 * while a scope is open join the same command buffer.
 */
public final class GpuBatch implements AutoCloseable {

    private boolean open = true;

    private GpuBatch() {}

    public static GpuBatch open() {
        WgpuBackend.batchBegin();
        return new GpuBatch();
    }

    @Override
    public void close() {
        if (!open) {
            return;
        }
        open = false;
        WgpuBackend.batchEnd();
    }
}
