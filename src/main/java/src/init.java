import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Random;

public class init {
    public static void main(String[] args) throws Throwable {
        System.out.println("[Java]: Initializing HC4J Cross-Platform Off-Heap Engine...");

        Linker linker = Linker.nativeLinker();
        
        String os = System.getProperty("os.name").toLowerCase();
        String libName = os.contains("win") ? "hc4j.dll" : (os.contains("mac") ? "libhc4j.dylib" : "libhc4j.so");
        
        // Use the absolute path to the root of your project, then append the OS-specific library name
        Path libPath = Path.of("C:\\Users\\DELL\\Desktop\\HC4J\\src\\main\\rust\\target\\release\\" + libName);
        System.out.println("[Java]: Loading native library from: " + libPath);
        
        try (Arena arena = Arena.ofConfined()) {
            SymbolLookup hc4jLib = SymbolLookup.libraryLookup(libPath, arena);

            MemorySegment addFunctionAddress = hc4jLib.find("hc4j_matrix_add").orElseThrow(
                () -> new RuntimeException("Could not find 'hc4j_matrix_add' function")
            );

            // Updated: The function now returns a 64-bit integer (JAVA_LONG)
            FunctionDescriptor addDescriptor = FunctionDescriptor.of(
                ValueLayout.JAVA_LONG, // The returned Rust math time
                ValueLayout.ADDRESS,   
                ValueLayout.ADDRESS,   
                ValueLayout.ADDRESS,   
                ValueLayout.JAVA_LONG  
            );

            MethodHandle addHandle = linker.downcallHandle(addFunctionAddress, addDescriptor);

            // 1024 x 1024 flattened matrix = 1,048,576 elements
            long arrayLength = 1024L * 1024L; 
            Random rand = new Random();

            System.out.println("[Java]: Allocating 1,048,576 elements per matrix to native RAM...");
            MemorySegment segmentA = arena.allocate(ValueLayout.JAVA_INT, arrayLength);
            MemorySegment segmentB = arena.allocate(ValueLayout.JAVA_INT, arrayLength);
            MemorySegment segmentOut = arena.allocate(ValueLayout.JAVA_INT, arrayLength);

            for (long i = 0; i < arrayLength; i++) {
                segmentA.setAtIndex(ValueLayout.JAVA_INT, i, rand.nextInt(100));
                segmentB.setAtIndex(ValueLayout.JAVA_INT, i, rand.nextInt(100));
            }

            System.out.println("[Java]: Executing 10 Warmup runs to trigger JIT optimization...");
            for (int i = 0; i < 2; i++) {
                // We MUST explicitly cast to (long) to satisfy invokeExact
                long ignoredResult = (long) addHandle.invokeExact(segmentA, segmentB, segmentOut, arrayLength);
            }

            System.out.println("---------------------------------------------------------");
            System.out.println("[Java]: Starting 1024x1024 FFM Benchmark");
            
            for (int iteration = 1; iteration <= 10; iteration++) {
                // Java starts the total round-trip stopwatch
                long totalStart = System.nanoTime();
                
                // Execute and catch the internal math time calculated by Rust
                long rustMathNs = (long) addHandle.invokeExact(segmentA, segmentB, segmentOut, arrayLength);
                
                // Java stops the round-trip stopwatch
                long totalEnd = System.nanoTime();
                
                long totalNs = totalEnd - totalStart;
                long bridgeOverheadNs = totalNs - rustMathNs;
                
                System.out.printf("Run #%d: Total: %.4f ms | Math: %.4f ms | FFM Bridge Overhead: %.4f ms\n", 
                    iteration, 
                    totalNs / 1_000_000.0, 
                    rustMathNs / 1_000_000.0, 
                    bridgeOverheadNs / 1_000_000.0
                );
            }
            System.out.println("---------------------------------------------------------");

            System.out.println("\n--- Verification (First 3 elements of final run) ---");
            for (long i = 0; i < 3; i++) {
                int valA = segmentA.getAtIndex(ValueLayout.JAVA_INT, i);
                int valB = segmentB.getAtIndex(ValueLayout.JAVA_INT, i);
                int valOut = segmentOut.getAtIndex(ValueLayout.JAVA_INT, i);
                System.out.println("Index " + i + ": " + valA + " + " + valB + " = " + valOut);
            }
        } 
    }
}