package hc4j.engine;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;

/**
 * Snapshot of the Rust tiered memory manager, mirroring {@code #[repr(C)] struct MemStats}
 * field for field.
 *
 * @param vramBudget the VRAM budget in bytes, from {@code HC4J_VRAM_BUDGET_MB} or a device-class
 *     default, clamped down by the first driver OOM
 * @param vramUsed bytes reserved in VRAM, counting each slab's full capacity
 * @param hostBudget the host-RAM tier budget in bytes
 * @param hostUsed bytes held in the host-RAM tier
 * @param spilledBytes bytes resident in the disk tier
 * @param tensorsDevice tensors whose bytes are in VRAM
 * @param tensorsHost tensors paged out to host RAM
 * @param tensorsEvicted tensors spilled to disk
 * @param evictions VRAM to host-RAM demotions
 * @param pageIns off-device tensors paged back into VRAM to satisfy an op
 * @param spills host-RAM to disk demotions
 * @param driverOoms times the driver reported OutOfMemory while the software budget still had
 *     room, each one clamping {@link #vramBudget} to what is actually committed
 * @param streamedOps ops that ran tile by tile because their operands could not be co-resident
 * @param allocations tensors allocated through the FFI
 * @param dedicatedBuffers {@code create_buffer} calls for dedicated tensor or scratch storage
 * @param slabsCreated slab buffers created (each hosts many small tensors)
 * @param regionAllocs tensors or scratch buffers served as slab sub-regions
 * @param slabCount live slabs; each reserves its full capacity in {@link #vramUsed}
 * @param slabBytes total capacity of the live slabs, in bytes
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
