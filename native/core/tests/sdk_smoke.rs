//! Integration smoke test for the safe `hc4rs` Rust SDK facade.
//! Exercises Engine, RAII Tensor, Linalg (including F32 x F16), Nn, Optimizer, and Collective.

use hc4rs::{
    Collective, DType, DeviceProfileC, DeviceRole, Engine, Linalg, LinkInfo, Nn, Optimizer,
    SplitKvSpec, TRANSPORT_HOST_STAGED, Tensor, WorkloadSpec,
};

#[test]
fn sdk_end_to_end_smoke_test() {
    let engine = match Engine::init() {
        Ok(e) => e,
        Err(err) => {
            println!("Skipping GPU portion of sdk_smoke (no WebGPU adapter available: {err})");
            optimizer_and_merge_smoke();
            return;
        }
    };

    let info = engine.adapter_info().expect("adapter_info");
    println!(
        "hc4rs sdk_smoke active adapter: {} (backend={}, device_type={})",
        info.name, info.backend, info.device_type
    );

    // 1. Tensor + arithmetic + RAII Drop
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("tensor a");
    let b = Tensor::from_f32(&[6.0, 5.0, 4.0, 3.0, 2.0, 1.0], &[2, 3]).expect("tensor b");
    let sum = a.add(&b).expect("add");
    assert_eq!(sum.to_f32_vec().expect("download sum"), vec![7.0; 6]);

    // 2. Linalg f32 @ f16^T: [2, 3] x [4, 3]^T -> [2, 4] (and GEMV m=1 fast path)
    let w_f32 = Tensor::from_f32(
        &[
            1.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, // row 1
            0.0, 0.0, 1.0, // row 2
            1.0, 1.0, 1.0, // row 3
        ],
        &[4, 3],
    )
    .expect("w_f32");
    let w_f16 = w_f32.to_f16().expect("to_f16");
    assert_eq!(w_f16.dtype(), DType::F16);

    let y = Linalg::matmul_transposed_b(&a, &w_f16).expect("f32 x f16 matmul");
    assert_eq!(y.shape(), &[2, 4]);
    let y_host = y.to_f32_vec().expect("download y");
    let expected = [1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0];
    for (g, w) in y_host.iter().zip(expected) {
        assert!((g - w).abs() < 1e-2, "matmul got {g} want {w}");
    }

    // 3. Nn: RMSNorm -> Softmax
    let gamma = Tensor::from_f32(&[1.0, 1.0, 1.0, 1.0], &[4]).expect("gamma");
    let normed = Nn::rms_norm(&y, Some(&gamma), 1e-5).expect("rms_norm");
    let probs = Nn::softmax(&normed).expect("softmax");
    let p_host = probs.to_f32_vec().expect("download probs");
    let row0_sum: f32 = p_host[0..4].iter().sum();
    let row1_sum: f32 = p_host[4..8].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-4, "row0 softmax sum={row0_sum}");
    assert!((row1_sum - 1.0).abs() < 1e-4, "row1 softmax sum={row1_sum}");

    // 4. Optimizer: plan_roles + Split-KV reclamation + Collective::open(2)
    optimizer_and_merge_smoke();

    let coll = Collective::open(2).expect("Collective::open(2)");
    assert_eq!(coll.ranks().expect("ranks"), 2);
    let backend_name = coll.backend_name().expect("backend_name");
    let p0 = coll.device_profile(0).expect("profile 0");
    println!("hc4rs Collective::open(2) -> {backend_name}, rank 0 vram={} MB", p0.vram_bytes >> 20);

    engine.synchronize().expect("sync");
    println!("hc4rs sdk_smoke stats: submissions={}", engine.stats().expect("stats").submissions);
}

fn optimizer_and_merge_smoke() {
    let devices = vec![
        DeviceProfileC {
            rank: 0,
            vendor_id: 0x10DE,
            device_id: 0,
            device_type: 2,
            vram_bytes: 8 << 30,
            compute_gflops: 40_000.0,
        },
        DeviceProfileC {
            rank: 1,
            vendor_id: 0x8086,
            device_id: 1,
            device_type: 1,
            vram_bytes: 8 << 30,
            compute_gflops: 120.0,
        },
    ];
    let links = vec![
        LinkInfo {
            src_rank: 0,
            dst_rank: 1,
            bandwidth_gbps: 8.0,
            transport: TRANSPORT_HOST_STAGED,
            loopback: 0,
            latency_us: 20.0,
        },
        LinkInfo {
            src_rank: 1,
            dst_rank: 0,
            bandwidth_gbps: 8.0,
            transport: TRANSPORT_HOST_STAGED,
            loopback: 0,
            latency_us: 20.0,
        },
    ];
    let tp_spec = WorkloadSpec::new(2048, 2e9, 4096, 128 << 20);
    let mut kv_spec = SplitKvSpec::new(1200, 1 << 20, 1e6, 32768);
    kv_spec.reserved_weight_bytes_per_gpu = 6 << 30;

    let plan = Optimizer::plan_roles(&devices, &links, &tp_spec, &kv_spec).expect("plan_roles");
    assert_eq!(plan.regime, "SPLIT_KV_RECLAIMED");
    assert_eq!(plan.split_kv_assignments[0].role, DeviceRole::TensorParallel);
    assert_eq!(plan.split_kv_assignments[1].role, DeviceRole::SplitKvReclaimed);
    assert_eq!(
        plan.split_kv_assignments.iter().map(|a| a.kv_tokens).sum::<u64>(),
        1200
    );
}
