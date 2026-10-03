//! Every WGSL source the engine can generate, parsed and validated by naga.
//! Runs without a GPU and catches shader bugs at `cargo test` time instead of
//! as runtime validation errors.

use naga::valid::{Capabilities, ValidationFlags, Validator};

use super::arithmetic::{DataType, generate_elem_shader};
use super::elementwise::generate_unary_shader;
use super::fusion::{self, op};
use super::matmul::{
    TILE_64, TILE_128, TILED16_WGSL, TRANSPOSE_WGSL, gemv_cols_wgsl, gemv_rows_wgsl, register_kernel_wgsl,
};

fn validate_with(name: &str, wgsl: &str, capabilities: Capabilities) {
    let module = naga::front::wgsl::parse_str(wgsl)
        .unwrap_or_else(|e| panic!("{name}: parse error\n{}\n{wgsl}", e.emit_to_string(wgsl)));
    if let Err(e) = Validator::new(ValidationFlags::all(), capabilities).validate(&module) {
        panic!("{name}: validation error {e:?}\n{wgsl}");
    }
}

/// Portable kernels must validate with no optional capabilities at all.
fn validate(name: &str, wgsl: &str) {
    validate_with(name, wgsl, Capabilities::empty());
}

#[test]
fn unary_suites() {
    let ops = [
        "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log",
        "log2", "0.4342944819032518 * log", "sqrt",
    ];
    for op in ops {
        validate(op, &generate_unary_shader(op));
    }
}

#[test]
fn binary_arithmetic() {
    for dtype in [DataType::I32, DataType::F32] {
        for op in ["+", "-", "*", "/"] {
            validate(&format!("{op} {dtype:?}"), &generate_elem_shader(dtype, op));
        }
    }
}

#[test]
fn fused_kernels() {
    let sample = [
        op::word(op::LOAD, 0),
        op::word(op::SIN, 0),
        op::word(op::LOAD, 1),
        op::word(op::ADD, 0),
        op::word(op::LOAD, 2),
        op::word(op::MUL, 0),
        op::word(op::CONST, 0),
        2.0f32.to_bits(),
        op::word(op::ADD, 0),
    ];
    validate("fused sample", &fusion::compile(&sample, 3).unwrap().wgsl);

    let unary = [
        op::NEG, op::ABS, op::SIN, op::COS, op::TAN, op::ASIN, op::ACOS, op::ATAN, op::SINH, op::COSH, op::TANH,
        op::ASINH, op::ACOSH, op::ATANH, op::EXP, op::LOG, op::SQRT,
    ];
    let binary = [op::ADD, op::SUB, op::MUL, op::DIV, op::MAX, op::MIN, op::POW];
    let mut program = vec![op::word(op::LOAD, 0)];
    program.extend(unary.iter().map(|&u| op::word(u, 0)));
    for b in binary {
        program.push(op::word(op::LOAD, 1));
        program.push(op::word(b, 0));
    }
    validate("fused all opcodes", &fusion::compile(&program, 2).unwrap().wgsl);
}

#[test]
fn register_tiled_gemm_variants() {
    for cfg in [TILE_64, TILE_128] {
        for vec4 in [true, false] {
            validate(&format!("gemm {cfg:?} vec4={vec4}"), &register_kernel_wgsl(cfg, vec4));
        }
    }
}

#[test]
fn small_gemm_gemv_and_transpose() {
    validate("tiled16", TILED16_WGSL);
    validate("transpose", TRANSPOSE_WGSL);
    for lanes in [32, 256] {
        for vec4 in [true, false] {
            validate(&format!("gemv rows {lanes} vec4={vec4}"), &gemv_rows_wgsl(lanes, vec4, false));
        }
    }
    for cols in [4, 16, 64] {
        validate(&format!("gemv cols {cols}"), &gemv_cols_wgsl(cols));
    }
}

#[test]
fn subgroup_gemv_needs_and_passes_with_the_subgroup_capability() {
    for vec4 in [true, false] {
        let wgsl = gemv_rows_wgsl(256, vec4, true);
        validate_with("gemv rows subgroup", &wgsl, Capabilities::SUBGROUP);
        let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
        assert!(
            Validator::new(ValidationFlags::all(), Capabilities::empty()).validate(&module).is_err(),
            "subgroup kernel must not validate without the capability"
        );
    }
}
