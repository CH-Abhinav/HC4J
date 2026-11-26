use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn compile_wgsl_to_spv(src_path: &str, out_path: &Path) {
    println!("cargo:rerun-if-changed={src_path}");
    let wgsl_source = fs::read_to_string(src_path)
        .unwrap_or_else(|e| panic!("Failed to read {src_path}: {e}"));

    let module = naga::front::wgsl::parse_str(&wgsl_source)
        .unwrap_or_else(|e| panic!("Failed to parse {src_path}: {e}"));

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let module_info = validator.validate(&module)
        .unwrap_or_else(|e| panic!("Failed to validate {src_path}: {e}"));

    let options = naga::back::spv::Options {
        lang_version: (1, 2),
        ..Default::default()
    };

    let spv = naga::back::spv::write_vec(&module, &module_info, &options, None)
        .unwrap_or_else(|e| panic!("Failed to generate SPIR-V from {src_path}: {e}"));

    let spv_bytes: &[u8] = bytemuck::cast_slice(&spv);
    fs::write(out_path, spv_bytes)
        .unwrap_or_else(|e| panic!("Failed to write {}: {e}", out_path.display()));
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    compile_wgsl_to_spv("src/collective/vulkan/reduce.wgsl", &out_dir.join("collective_reduce.spv"));
    compile_wgsl_to_spv("src/collective/vulkan/bench_fma.wgsl", &out_dir.join("collective_bench_fma.spv"));
    compile_wgsl_to_spv("src/collective/vulkan/bench_vram_bw.wgsl", &out_dir.join("collective_bench_vram_bw.spv"));
    compile_wgsl_to_spv("src/collective/vulkan/gemm_bias_act.wgsl", &out_dir.join("collective_gemm_bias_act.spv"));
    compile_wgsl_to_spv("src/collective/vulkan/split_kv_attn.wgsl", &out_dir.join("collective_split_kv_attn.spv"));
    compile_wgsl_to_spv("src/collective/vulkan/elem_mul.wgsl", &out_dir.join("collective_elem_mul.spv"));
}
