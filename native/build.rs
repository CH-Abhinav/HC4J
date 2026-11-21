use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/collective/vulkan/reduce.wgsl");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    let wgsl_source = include_str!("src/collective/vulkan/reduce.wgsl");

    let module = naga::front::wgsl::parse_str(wgsl_source)
        .expect("Failed to parse reduce.wgsl");

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let module_info = validator.validate(&module)
        .expect("Failed to validate reduce.wgsl");

    let options = naga::back::spv::Options {
        lang_version: (1, 2),
        ..Default::default()
    };

    let spv = naga::back::spv::write_vec(&module, &module_info, &options, None)
        .expect("Failed to generate SPIR-V from reduce.wgsl");

    let spv_bytes: &[u8] = bytemuck::cast_slice(&spv);
    fs::write(out_dir.join("collective_reduce.spv"), spv_bytes)
        .expect("Failed to write collective_reduce.spv");
}
