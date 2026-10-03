package hc4j.engine;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;

/**
 * Snapshot of the Rust tiered memory manager, mirroring {@code #[repr(C)] struct MemStats}
 * field for field.
 *
 * @param allocations tensors allocated through the FFI
 * @param dedicatedBuffers {@code create_buffer} calls for dedicated tensor or scratch storage
 * @param slabsCreated slab buffers created (each hosts many small tensors)
 * @param regionAllocs tensors or scratch buffers served as slab sub-regions
 * @param slabCount live slabs; each reserves its full capacity in {@link #vramUsed}
 * @param slabRegions regions handed out and not yet reclaimed
 * @param quarantined freed regions waiting for the GPU to retire their submission epoch
 */
public record MemoryStats(
        long vramBudget,
        long vramUsed,
        long hostBudget,
        long hostUsed,
        long spilledBytes,
        long tensorsDevice,
        long tensorsHost,
        long tensorsEvicted,
        long evictions,
        long pageIns,
        long spills,
        long driverOoms,
        long streamedOps,
        long allocations,
        long dedicatedBuffers,
        long slabsCreated,
        long regionAllocs,
        long slabCount,
        long slabBytes,
        long slabRegions,
        long quarantined) {

    private static final int FIELDS = 21;

    static final StructLayout LAYOUT = MemoryLayout.structLayout(
            MemoryLayout.sequenceLayout(FIELDS, ValueLayout.JAVA_LONG).withName("fields"));

    static MemoryStats read(MemorySegment s) {
        long[] v = s.toArray(ValueLayout.JAVA_LONG);
        return new MemoryStats(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10],
                v[11], v[12], v[13], v[14], v[15], v[16], v[17], v[18], v[19], v[20]);
    }
}
