package hc4j.collective;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import hc4j.engine.WgpuBackend;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A ring of GPU ranks with collective communication across them: topology discovery and
 * {@linkplain #allReduceSum ring AllReduce}.
 *
 * <pre>{@code
 * try (Collective ring = Collective.open(2)) {
 *     ring.topology().forEach(System.out::println);
 *
 *     try (CollectiveBuffer a = ring.allocate(0, 1024);
 *          CollectiveBuffer b = ring.allocate(1, 1024)) {
 *         a.write(gradientsFromRank0);
 *         b.write(gradientsFromRank1);
 *         ring.allReduceSum(a, b);          // both now hold the element-wise sum
 *         float[] summed = a.read();
 *     }
 * }
 * }</pre>
 *
 * <h2>Ranks and loopback</h2>
 * A rank is a logical device. Ranks are spread across the distinct physical GPUs on the machine;
 * when there are fewer GPUs than ranks the remainder open additional logical devices on a GPU
 * already in use — <em>loopback mode</em> — so the ring, the transport and the reduction all run
 * end to end on a single-GPU machine. One physical GPU enumerated under two backends counts once.
 *
 * <h2>Buffers</h2>
 * A collective owns its devices and its buffers outright, independently of the process-wide engine
 * behind {@link hc4j.Tensor}. Data moves in and out through {@link CollectiveBuffer#write} and
 * {@link CollectiveBuffer#read}; there is no zero-copy bridge to a {@code Tensor} yet.
 *
 * <h2>Cost</h2>
 * Every hop is host-staged (see {@link Transport}) and synchronous: a transfer completes before the
 * reduction that consumes it is recorded. AllReduce moves {@code 2(n-1)/n} of each buffer per rank
 * rather than the {@code n-1} copies an all-to-all would, but the rounds are not yet pipelined.
 *
 * <p>Not thread-safe for concurrent AllReduce calls on the same instance.
 */
public final class Collective implements AutoCloseable {

    /** Mirrors {@code #[repr(C)] struct Link}. */
    static final StructLayout LINK_LAYOUT = MemoryLayout.structLayout(
            JAVA_INT.withName("src"),
            JAVA_INT.withName("dst"),
            JAVA_DOUBLE.withName("gbps"),
            JAVA_INT.withName("transport"),
            JAVA_INT.withName("loopback"));

    private static final int NAME_CAPACITY = 256;

    private static final MethodHandle CREATE;
    private static final MethodHandle DESTROY;
    private static final MethodHandle RANKS;
    private static final MethodHandle LINK_COUNT;
    private static final MethodHandle LINK;
    private static final MethodHandle DEVICE_NAME;
    private static final MethodHandle ALLOC;
    private static final MethodHandle FREE;
    private static final MethodHandle WRITE;
    private static final MethodHandle READ;
    private static final MethodHandle ALLREDUCE;

    static {
        Linker linker = Linker.nativeLinker();
        SymbolLookup symbols = WgpuBackend.nativeSymbols();
        CREATE = bind(linker, symbols, "hc4j_collective_create",
                FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS));
        DESTROY = bind(linker, symbols, "hc4j_collective_destroy",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        RANKS = bind(linker, symbols, "hc4j_collective_ranks",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        LINK_COUNT = bind(linker, symbols, "hc4j_collective_link_count",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        LINK = bind(linker, symbols, "hc4j_collective_link",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, ADDRESS));
        DEVICE_NAME = bind(linker, symbols, "hc4j_collective_device_name",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_INT));
        ALLOC = bind(linker, symbols, "hc4j_collective_alloc",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_LONG, ADDRESS));
        FREE = bind(linker, symbols, "hc4j_collective_free",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG));
        WRITE = bind(linker, symbols, "hc4j_collective_write",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG));
        READ = bind(linker, symbols, "hc4j_collective_read",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG));
        ALLREDUCE = bind(linker, symbols, "hc4j_collective_allreduce_sum",
                FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_INT));
    }

    private static MethodHandle bind(Linker linker, SymbolLookup symbols, String name, FunctionDescriptor desc) {
        MemorySegment address = symbols.find(name).orElseThrow(() -> new UnsatisfiedLinkError(
                "HC4J native symbol '" + name + "' not found; rebuild the Rust backend."));
        return linker.downcallHandle(address, desc);
    }

    private final long handle;
    private final int ranks;
    private final AtomicBoolean closed = new AtomicBoolean();
    private List<Link> topology;

    private Collective(long handle, int ranks) {
        this.handle = handle;
        this.ranks = ranks;
    }

    /**
     * Opens a collective over {@code ranks} logical devices.
     *
     * @throws IllegalArgumentException if {@code ranks < 2}
     * @throws IllegalStateException if no GPU is available or a device cannot be opened
     */
    public static Collective open(int ranks) {
        if (ranks < 2) {
            throw new IllegalArgumentException("a collective needs at least 2 ranks, got " + ranks);
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(JAVA_LONG);
            int status = call(() -> (int) CREATE.invokeExact(ranks, out), "Collective.open");
            WgpuBackend.checkStatus(status, "Collective.open");
            long handle = out.get(JAVA_LONG, 0);
            int actual = call(() -> (int) RANKS.invokeExact(handle), "Collective.ranks");
            if (actual < 0) {
                WgpuBackend.checkStatus(actual, "Collective.ranks");
            }
            return new Collective(handle, actual);
        }
    }

    /** Number of ranks in the ring. */
    public int ranks() {
        return ranks;
    }

    /** The adapter name backing {@code rank}. */
    public String deviceName(int rank) {
        checkOpen();
        checkRank(rank);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(NAME_CAPACITY);
            int len = call(() -> (int) DEVICE_NAME.invokeExact(handle, rank, buf, NAME_CAPACITY),
                    "Collective.deviceName");
            if (len < 0) {
                WgpuBackend.checkStatus(len, "Collective.deviceName");
            }
            byte[] bytes = buf.asSlice(0, len).toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
            return new String(bytes, StandardCharsets.UTF_8);
        }
    }

    /**
     * The topology graph: one {@link Link} per directed rank pair, with measured throughput.
     *
     * <p>The first call runs the probe, which moves a few hundred MiB per pair and takes a moment;
     * the result is cached for the life of this instance.
     */
    public List<Link> topology() {
        checkOpen();
        if (topology != null) {
            return topology;
        }
        int count = call(() -> (int) LINK_COUNT.invokeExact(handle), "Collective.topology");
        if (count < 0) {
            WgpuBackend.checkStatus(count, "Collective.topology");
        }
        List<Link> links = new ArrayList<>(count);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment slot = arena.allocate(LINK_LAYOUT);
            for (int i = 0; i < count; i++) {
                int index = i;
                int status = call(() -> (int) LINK.invokeExact(handle, index, slot), "Collective.topology");
                WgpuBackend.checkStatus(status, "Collective.topology");
                links.add(new Link(
                        slot.get(JAVA_INT, 0),
                        slot.get(JAVA_INT, 4),
                        slot.get(JAVA_DOUBLE, 8),
                        Transport.of(slot.get(JAVA_INT, 16)),
                        slot.get(JAVA_INT, 20) != 0));
            }
        }
        topology = List.copyOf(links);
        return topology;
    }

    /**
     * Allocates {@code elements} f32 on {@code rank}.
     *
     * @throws hc4j.engine.GpuOutOfMemoryException if the rank cannot back the allocation
     */
    public CollectiveBuffer allocate(int rank, long elements) {
        checkOpen();
        checkRank(rank);
        if (elements <= 0) {
            throw new IllegalArgumentException("a collective buffer needs at least one element");
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(JAVA_LONG);
            int status = call(() -> (int) ALLOC.invokeExact(handle, rank, elements, out), "Collective.allocate");
            WgpuBackend.checkStatus(status, "Collective.allocate");
            return new CollectiveBuffer(this, out.get(JAVA_LONG, 0), rank, elements);
        }
    }

    /**
     * Ring AllReduce (sum) in place: every buffer ends up holding the element-wise sum of all of
     * them.
     *
     * @param perRank exactly one buffer per rank, in rank order, all the same length, with that
     *     length divisible by the rank count
     * @throws IllegalArgumentException if the buffers do not satisfy that contract
     */
    public void allReduceSum(CollectiveBuffer... perRank) {
        checkOpen();
        Objects.requireNonNull(perRank, "perRank");
        if (perRank.length != ranks) {
            throw new IllegalArgumentException(
                    "allReduceSum needs one buffer per rank: expected " + ranks + ", got " + perRank.length);
        }
        long[] ids = new long[perRank.length];
        for (int i = 0; i < perRank.length; i++) {
            CollectiveBuffer buf = Objects.requireNonNull(perRank[i], "buffer " + i);
            if (buf.collective() != this) {
                throw new IllegalArgumentException("buffer " + i + " belongs to a different collective");
            }
            if (buf.rank() != i) {
                throw new IllegalArgumentException(
                        "buffer " + i + " is on rank " + buf.rank() + "; pass them in rank order");
            }
            ids[i] = buf.handleUnsafe();
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment list = arena.allocateFrom(JAVA_LONG, ids);
            int n = ids.length;
            int status = call(() -> (int) ALLREDUCE.invokeExact(handle, list, n), "Collective.allReduceSum");
            WgpuBackend.checkStatus(status, "Collective.allReduceSum");
        }
    }

    /** Releases every device and buffer this collective owns. Idempotent. */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            int status = call(() -> (int) DESTROY.invokeExact(handle), "Collective.close");
            WgpuBackend.checkStatus(status, "Collective.close");
        }
    }

    // ------------------------------------------------------------------------------------------
    // Package-private plumbing used by CollectiveBuffer
    // ------------------------------------------------------------------------------------------

    void writeBuffer(long buffer, float[] data) {
        checkOpen();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocateFrom(java.lang.foreign.ValueLayout.JAVA_FLOAT, data);
            long len = data.length;
            int status = call(() -> (int) WRITE.invokeExact(handle, buffer, host, len), "CollectiveBuffer.write");
            WgpuBackend.checkStatus(status, "CollectiveBuffer.write");
        }
    }

    float[] readBuffer(long buffer, long elements) {
        checkOpen();
        float[] out = new float[(int) elements];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(java.lang.foreign.ValueLayout.JAVA_FLOAT, elements);
            int status = call(() -> (int) READ.invokeExact(handle, buffer, host, elements), "CollectiveBuffer.read");
            WgpuBackend.checkStatus(status, "CollectiveBuffer.read");
            MemorySegment.copy(host, java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, out, 0, out.length);
        }
        return out;
    }

    void freeBuffer(long buffer) {
        if (closed.get()) {
            // Destroying the collective already released its buffers.
            return;
        }
        int status = call(() -> (int) FREE.invokeExact(handle, buffer), "CollectiveBuffer.close");
        WgpuBackend.checkStatus(status, "CollectiveBuffer.close");
    }

    private void checkOpen() {
        if (closed.get()) {
            throw new IllegalStateException("this Collective is closed");
        }
    }

    private void checkRank(int rank) {
        if (rank < 0 || rank >= ranks) {
            throw new IllegalArgumentException("rank " + rank + " is outside 0.." + (ranks - 1));
        }
    }

    /** Wraps the checked {@code Throwable} of {@code invokeExact} without hiding failures. */
    private interface NativeCall {
        int invoke() throws Throwable;
    }

    private static int call(NativeCall body, String context) {
        try {
            return body.invoke();
        } catch (Throwable t) {
            throw new IllegalStateException("HC4J native dispatch of " + context + " failed", t);
        }
    }
}
