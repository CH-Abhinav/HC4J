#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::slice;
    use std::time::Instant;
    use wgpu::util::DeviceExt;

    // 1. Define the Rust equivalent of the WGSL Dimensions struct.
    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    struct MatMulDims {
        m: u32,
        k: u32,
        n: u32,
        _padding: u32, // Matches the WGSL padding exactly
    }

    // --- WGSL Shared Memory Tiled Kernel ---
    const SHADER_SRC: &str = r#"
    struct Dimensions {
        M: u32,
        K: u32,
        N: u32,
        _padding: u32,
    }

    @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
    @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
    @group(0) @binding(2) var<storage, read_write> matrixC: array<f32>;
    @group(0) @binding(3) var<uniform> dims: Dimensions;

    const TILE_SIZE: u32 = 16u;

    var<workgroup> tileA: array<array<f32, 16>, 16>;
    var<workgroup> tileB: array<array<f32, 16>, 16>;

    @compute @workgroup_size(16, 16)
    fn main(
        @builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>
    ) {
        let row = global_id.y;
        let col = global_id.x;
        
        let local_row = local_id.y;
        let local_col = local_id.x;

        var sum = 0.0;
        let num_tiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

        for (var t = 0u; t < num_tiles; t = t + 1u) {
            
            let k_a = t * TILE_SIZE + local_col;
            if (row < dims.M && k_a < dims.K) {
                tileA[local_row][local_col] = matrixA[row * dims.K + k_a];
            } else {
                tileA[local_row][local_col] = 0.0;
            }

            let k_b = t * TILE_SIZE + local_row;
            if (k_b < dims.K && col < dims.N) {
                tileB[local_row][local_col] = matrixB[k_b * dims.N + col];
            } else {
                tileB[local_row][local_col] = 0.0;
            }

            workgroupBarrier();

            for (var i = 0u; i < TILE_SIZE; i = i + 1u) {
                sum = sum + tileA[local_row][i] * tileB[i][local_col];
            }

            workgroupBarrier();
        }

        if (row < dims.M && col < dims.N) {
            matrixC[row * dims.N + col] = sum;
        }
    }
    "#;

    // --- Optimized CPU Baseline Function ---
    fn cpu_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        c.fill(0.0);
        for row in 0..m {
            let out_offset = row * n;
            for i in 0..k {
                let val_a = a[row * k + i];
                let b_offset = i * n;
                for col in 0..n {
                    c[out_offset + col] += val_a * b[b_offset + col];
                }
            }
        }
    }

    #[test]
    fn test_gpu_vs_cpu_matrix() {
        let m = 2048;
        let k = 2048;
        let n = 2048;

        println!("\n=======================================================");
        println!("   TILED SHARE-MEMORY SHOOTOUT: {} x {}", m, n);
        println!("=======================================================\n");

        let matrix_a = vec![0.01f32; m * k];
        let matrix_b = vec![0.01f32; k * n];
        let mut cpu_result = vec![0.0f32; m * n];

        // --------------------------------------------------------------------
        // 1. CPU BENCHMARK RUN
        // --------------------------------------------------------------------
        let start_cpu = Instant::now();
        cpu_matmul(&matrix_a, &matrix_b, &mut cpu_result, m, k, n);
        let cpu_time = start_cpu.elapsed().as_secs_f64() * 1000.0;
        println!("-> CPU Core Execution Time: {:.4} ms", cpu_time);

        // --------------------------------------------------------------------
        // 2. HEADLESS GPU COMPUTE PIPELINE
        // --------------------------------------------------------------------
        let start_overall = Instant::now();

        pollster::block_on(async {
            // Using Instance::default() automatically bypasses InstanceDescriptor initialization bugs
            let instance = wgpu::Instance::default();

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .expect("Failed to get GPU");

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .expect("Failed to create device");

            let size_c_bytes = (m * n * 4) as u64;

            let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Matrix A"),
                contents: bytemuck::cast_slice(&matrix_a),
                usage: wgpu::BufferUsages::STORAGE,
            });
            
            let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Matrix B"),
                contents: bytemuck::cast_slice(&matrix_b),
                usage: wgpu::BufferUsages::STORAGE,
            });
            
            let buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Matrix C"),
                size: size_c_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let dims = MatMulDims { m: m as u32, k: k as u32, n: n as u32, _padding: 0 };
            let meta_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dimensions Metadata"),
                contents: bytemuck::cast_slice(&[dims]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging"),
                size: size_c_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MatMul Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SRC)),
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MatMul Pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MatMul Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buffer_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buffer_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buffer_c.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: meta_buffer.as_entire_binding() },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                
                let workgroups_x = (n as u32 + 15) / 16;
                let workgroups_y = (m as u32 + 15) / 16;
                cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            
            encoder.copy_buffer_to_buffer(&buffer_c, 0, &staging_buffer, 0, size_c_bytes);

            let start_gpu_math = Instant::now();
            queue.submit(std::iter::once(encoder.finish()));

            let slice = staging_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
            
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            
            if let Ok(Ok(())) = receiver.recv() {
                let gpu_time = start_gpu_math.elapsed().as_secs_f64() * 1000.0;
                let overall_time = start_overall.elapsed().as_secs_f64() * 1000.0;

                let data = slice.get_mapped_range();
                let gpu_result: &[f32] = bytemuck::cast_slice(&data[..]);
                
                println!("-> GPU Pure Math Execution Time: {:.4} ms", gpu_time);
                println!("-> GPU Overall Framework Cycle Time: {:.4} ms\n", overall_time);
                
                let mut passed = true;
                for i in 0..(m * n) {
                    if (cpu_result[i] - gpu_result[i]).abs() > 1e-3 {
                        passed = false;
                        println!("Mismatch at index {}: CPU={} GPU={}", i, cpu_result[i], gpu_result[i]);
                        break;
                    }
                }

                if passed {
                    println!("-> Integrity Check: Passed (Matrix entries match perfectly).");
                } else {
                    println!("-> Integrity Check: FAILED (Data mismatch detected).");
                }
                
                drop(data);
                staging_buffer.unmap();
            }
        });
    }
}