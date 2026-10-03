package hc4j.engine;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;

/**
 * Snapshot of the engine's command stream and device capabilities, mirroring
 * {@code #[repr(C)] struct EngineStats}.
 *
 * @param submissions {@code queue.submit} calls
 * @param dispatches compute dispatches recorded
 * @param kernelBytes bytes bound to kernels as inputs plus outputs: the logical memory traffic
 * @param batches command-batch scopes closed
 * @param features bitmask of {@link #FEATURE_SUBGROUPS} and {@link #FEATURE_SHADER_F16}
 * @param deviceType 0 other, 1 integrated, 2 discrete, 3 virtual, 4 CPU
 * @param backend 0 none, 1 Vulkan, 2 Metal, 3 DX12, 4 GL, 5 browser WebGPU
 */
public record EngineStats(
        long submissions,
        long dispatches,
        long kernelBytes,
        long batches,
        long completedEpoch,
        long submittedEpoch,
        long features,
        long deviceType,
        long backend,
        long subgroupMinSize,
        long subgroupMaxSize) {

    public static final long FEATURE_SUBGROUPS = 1;
    public static final long FEATURE_SHADER_F16 = 2;

    private static final int FIELDS = 11;

    static final StructLayout LAYOUT = MemoryLayout.structLayout(
            MemoryLayout.sequenceLayout(FIELDS, ValueLayout.JAVA_LONG).withName("fields"));

    static EngineStats read(MemorySegment s) {
        long[] v = s.toArray(ValueLayout.JAVA_LONG);
        return new EngineStats(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]);
    }

    public boolean hasSubgroups() {
        return (features & FEATURE_SUBGROUPS) != 0;
    }

    public boolean hasShaderF16() {
        return (features & FEATURE_SHADER_F16) != 0;
    }

    public boolean isDiscrete() {
        return deviceType == 2;
    }

    public String backendName() {
        return switch ((int) backend) {
            case 1 -> "Vulkan";
            case 2 -> "Metal";
            case 3 -> "DX12";
            case 4 -> "GL";
            case 5 -> "WebGPU";
            default -> "unknown";
        };
    }

    public String deviceTypeName() {
        return switch ((int) deviceType) {
            case 1 -> "integrated";
            case 2 -> "discrete";
            case 3 -> "virtual";
            case 4 -> "cpu";
            default -> "other";
        };
    }
}
