package hc4j.test;

import hc4j.Tensor;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
        System.out.println("====================================================================================================");
        printlnCentered("HC4J GPU TENSOR PERFORMANCE BENCHMARK (WEBGPU + PANAMA FFM)");
        System.out.println("====================================================================================================\n");

        // --------------------------------------------------------------------------------
        // 1. WARMUP RUN
        // --------------------------------------------------------------------------------
        System.out.println("-> Running Warmup Phase (Compiling WGSL Shaders & Warming JIT)...");
        runBenchmark("WARMUP", false, 1000);
        System.out.println("-> Warmup Complete. Starting Benchmarks...\n");

        System.out.printf("%-15s | %-15s | %-12s | %-12s | %-12s | %-12s | %-8s%n",
                "Tensor Shape", "Total Elements", "Upload (ms)", "Math (ms)", "Readback (ms)", "Total (ms)", "Check");
        System.out.println("----------------------------------------------------------------------------------------------------");

        // --------------------------------------------------------------------------------
        // 2. 1D TENSOR BENCHMARKS
        // --------------------------------------------------------------------------------
        runBenchmark("1D - 100K", true, 100_000);
        runBenchmark("1D - 1M", true, 1_000_000);
        runBenchmark("1D - 10M", true, 10_000_000);
        runBenchmark("1D- 100M",true, 100_000_000);

        System.out.println("----------------------------------------------------------------------------------------------------");

        // --------------------------------------------------------------------------------
        // 3. 2D TENSOR BENCHMARKS
        // --------------------------------------------------------------------------------
        runBenchmark("2D - 100K", true, 500, 200);         // 100,000 elements
        runBenchmark("2D - 1M", true, 1000, 1000);        // 1,000,000 elements
        runBenchmark("2D - 10M", true, 5000, 2000);       // 10,000,000 elements
        runBenchmark("2D - 100M", true, 10000, 10000);
        System.out.println("====================================================================================================\n");
    }

    private static void runBenchmark(String label, boolean printResult, int... shape) {
        long totalElements = 1;
        for (int dim : shape) totalElements *= dim;

        // Populate host CPU arrays
        float[] hostA = new float[(int) totalElements];
        float[] hostB = new float[(int) totalElements];
        Arrays.fill(hostA, 1.5f);
        Arrays.fill(hostB, 2.5f);

        // ----------------------------------------------------------------------------
        // PHASE 1: VRAM Allocation & Host-to-Device Copy
        // ----------------------------------------------------------------------------
        long startUpload = System.nanoTime();
        Tensor a = Tensor.fromArray(hostA, shape);
        Tensor b = Tensor.fromArray(hostB, shape);
        long uploadTimeNs = System.nanoTime() - startUpload;

        // ----------------------------------------------------------------------------
        // PHASE 2: Pure GPU Kernel Execution
        // ----------------------------------------------------------------------------
        long startMath = System.nanoTime();
        // Dispatches WGSL kernel asynchronously
        Tensor res = a.add(b); 
        long mathTimeNs = System.nanoTime() - startMath;

        // ----------------------------------------------------------------------------
        // PHASE 3: Device-to-Host Readback (PCIe Download)
        // ----------------------------------------------------------------------------
        long startReadback = System.nanoTime();
        float[] output = res.toFloatArray();
        long readbackTimeNs = System.nanoTime() - startReadback;

        // ----------------------------------------------------------------------------
        // INTEGRITY VERIFICATION & CLEANUP
        // ----------------------------------------------------------------------------
        boolean passed = (output[0] == 4.0f) && (output[output.length - 1] == 4.0f);

        // Free physical VRAM handles
        a.close();
        b.close();
        res.close();

        if (printResult) {
            double uploadMs = uploadTimeNs / 1_000_000.0;
            double mathMs = mathTimeNs / 1_000_000.0;
            double readbackMs = readbackTimeNs / 1_000_000.0;
            double totalMs = uploadMs + mathMs + readbackMs;

            System.out.printf("%-15s | %-15s | %-12.3f | %-12.3f | %-12.3f | %-12.3f | %-8s%n",
                    Arrays.toString(shape),
                    String.format("%,d", totalElements),
                    uploadMs,
                    mathMs,
                    readbackMs,
                    totalMs,
                    passed ? "PASSED" : "FAILED"
            );
        }
    }

    private static void printlnCentered(String text) {
        int width = 100;
        int padding = (width - text.length()) / 2;
        System.out.println(" ".repeat(Math.max(0, padding)) + text);
    }
}