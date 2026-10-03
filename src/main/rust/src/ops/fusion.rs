//! Fused elementwise kernels generated from a postfix program.
//!
//! Wire format: one u32 per instruction, `opcode << 24 | operand`. `LOAD k`
//! pushes input k; `CONST` is followed by one word of f32 bits. The Java side
//! (`hc4j.FusedExpr`) emits a post-order walk of its expression tree; this
//! module validates it and lowers it to SSA WGSL with local value numbering,
//! so repeated subexpressions are computed once. The whole expression runs
//! as one dispatch: n inputs read once, the output written once, and no
//! intermediate tensors.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::error::{Hc4jError, ffi_guard};
use crate::get_engine;
use crate::memory::TensorId;
use crate::ops::elementwise::run_contiguous;

pub mod op {
    pub const LOAD: u32 = 0x01;
    pub const CONST: u32 = 0x02;

    pub const NEG: u32 = 0x10;
    pub const ABS: u32 = 0x11;
    pub const SIN: u32 = 0x12;
    pub const COS: u32 = 0x13;
    pub const TAN: u32 = 0x14;
    pub const ASIN: u32 = 0x15;
    pub const ACOS: u32 = 0x16;
    pub const ATAN: u32 = 0x17;
    pub const SINH: u32 = 0x18;
    pub const COSH: u32 = 0x19;
    pub const TANH: u32 = 0x1A;
    pub const ASINH: u32 = 0x1B;
    pub const ACOSH: u32 = 0x1C;
    pub const ATANH: u32 = 0x1D;
    pub const EXP: u32 = 0x1E;
    pub const LOG: u32 = 0x1F;
    pub const SQRT: u32 = 0x20;

    pub const ADD: u32 = 0x40;
    pub const SUB: u32 = 0x41;
    pub const MUL: u32 = 0x42;
    pub const DIV: u32 = 0x43;
    pub const MAX: u32 = 0x44;
    pub const MIN: u32 = 0x45;
    pub const POW: u32 = 0x46;

    pub const fn word(opcode: u32, operand: u32) -> u32 {
        (opcode << 24) | (operand & 0x00FF_FFFF)
    }
}

pub const MAX_PROGRAM_WORDS: usize = 1024;
const MAX_STACK: usize = 64;

#[derive(Debug, PartialEq, Eq)]
pub enum FusionError {
    Empty,
    TooLong,
    Truncated,
    UnknownOpcode(u32),
    InputOutOfRange(u32),
    UnusedInput(u32),
    StackUnderflow,
    StackOverflow,
    Unbalanced(usize),
    NonFiniteConstant,
}

pub struct FusedKernel {
    pub wgsl: String,
    /// Exact structural key for the pipeline cache. The same expression shape
    /// over different tensors reuses one compiled pipeline.
    pub cache_key: String,
    pub n_inputs: u32,
}

fn unary_fn(opcode: u32) -> Option<&'static str> {
    use op::*;
    Some(match opcode {
        ABS => "abs",
        SIN => "sin",
        COS => "cos",
        TAN => "tan",
        ASIN => "asin",
        ACOS => "acos",
        ATAN => "atan",
        SINH => "sinh",
        COSH => "cosh",
        TANH => "tanh",
        ASINH => "asinh",
        ACOSH => "acosh",
        ATANH => "atanh",
        EXP => "exp",
        LOG => "log",
        SQRT => "sqrt",
        _ => return None,
    })
}

fn binary_expr(opcode: u32, a: usize, b: usize) -> Option<String> {
    use op::*;
    Some(match opcode {
        ADD => format!("v{a} + v{b}"),
        SUB => format!("v{a} - v{b}"),
        MUL => format!("v{a} * v{b}"),
        DIV => format!("v{a} / v{b}"),
        MAX => format!("max(v{a}, v{b})"),
        MIN => format!("min(v{a}, v{b})"),
        POW => format!("pow(v{a}, v{b})"),
        _ => return None,
    })
}

fn commutative(opcode: u32) -> bool {
    matches!(opcode, op::ADD | op::MUL | op::MAX | op::MIN)
}

pub fn compile(program: &[u32], n_inputs: u32) -> Result<FusedKernel, FusionError> {
    if program.is_empty() || n_inputs == 0 {
        return Err(FusionError::Empty);
    }
    if program.len() > MAX_PROGRAM_WORDS {
        return Err(FusionError::TooLong);
    }

    let mut body = String::new();
    let mut stack: Vec<usize> = Vec::new();
    // Local value numbering: (opcode, operand a, operand b) -> SSA id.
    let mut numbering: HashMap<(u32, u64, u64), usize> = HashMap::new();
    let mut used_inputs = vec![false; n_inputs as usize];
    let mut next_id = 0usize;
    let mut pc = 0usize;

    while pc < program.len() {
        let (opcode, operand) = (program[pc] >> 24, program[pc] & 0x00FF_FFFF);
        pc += 1;
        let (key, expr) = match opcode {
            op::LOAD => {
                let slot = used_inputs.get_mut(operand as usize).ok_or(FusionError::InputOutOfRange(operand))?;
                *slot = true;
                ((op::LOAD, operand as u64, 0), format!("x{operand}"))
            }
            op::CONST => {
                let bits = *program.get(pc).ok_or(FusionError::Truncated)?;
                pc += 1;
                if !f32::from_bits(bits).is_finite() {
                    return Err(FusionError::NonFiniteConstant);
                }
                // bitcast keeps the exact bit pattern; no float formatting.
                ((op::CONST, bits as u64, 0), format!("vec4<f32>(bitcast<f32>({bits}u))"))
            }
            op::NEG => {
                let a = stack.pop().ok_or(FusionError::StackUnderflow)?;
                ((op::NEG, a as u64, 0), format!("-v{a}"))
            }
            _ if let Some(name) = unary_fn(opcode) => {
                let a = stack.pop().ok_or(FusionError::StackUnderflow)?;
                ((opcode, a as u64, 0), format!("{name}(v{a})"))
            }
            _ => {
                let b = stack.pop().ok_or(FusionError::StackUnderflow)?;
                let a = stack.pop().ok_or(FusionError::StackUnderflow)?;
                let expr = binary_expr(opcode, a, b).ok_or(FusionError::UnknownOpcode(opcode))?;
                let (ka, kb) = if commutative(opcode) { (a.min(b), a.max(b)) } else { (a, b) };
                ((opcode, ka as u64, kb as u64), expr)
            }
        };
        let id = *numbering.entry(key).or_insert_with(|| {
            let _ = writeln!(body, "    let v{next_id} = {expr};");
            next_id += 1;
            next_id - 1
        });
        stack.push(id);
        if stack.len() > MAX_STACK {
            return Err(FusionError::StackOverflow);
        }
    }

    if stack.len() != 1 {
        return Err(FusionError::Unbalanced(stack.len()));
    }
    if let Some(unused) = used_inputs.iter().position(|used| !used) {
        return Err(FusionError::UnusedInput(unused as u32));
    }

    let result = stack[0];
    let n = n_inputs as usize;
    let mut wgsl = String::from("struct Dims {\n    params: vec4<u32>,\n}\n\n");
    // read_write on inputs too: slab regions and in-place outputs share buffers.
    for k in 0..n {
        let _ = writeln!(wgsl, "@group(0) @binding({k}) var<storage, read_write> in{k}: array<f32>;");
    }
    let _ = writeln!(wgsl, "@group(0) @binding({n}) var<storage, read_write> out: array<f32>;");
    let _ = writeln!(wgsl, "@group(0) @binding({}) var<uniform> dims: Dims;\n", n + 1);

    let params: Vec<String> = (0..n).map(|k| format!("x{k}: vec4<f32>")).collect();
    let _ = writeln!(wgsl, "fn fused({}) -> vec4<f32> {{\n{body}    return v{result};\n}}\n", params.join(", "));

    let vec_args: Vec<String> = (0..n)
        .map(|k| format!("vec4<f32>(in{k}[base], in{k}[base + 1u], in{k}[base + 2u], in{k}[base + 3u])"))
        .collect();
    let tail_args: Vec<String> = (0..n).map(|k| format!("vec4<f32>(in{k}[i])")).collect();
    let _ = write!(
        wgsl,
        r#"@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {{
    let tid = gid.x + gid.y * (nwg.x * 256u);
    let length = dims.params.x;
    let base = tid << 2u;
    if (base >= length) {{ return; }}
    if (base + 3u < length) {{
        let r = fused({vec});
        out[base] = r.x;
        out[base + 1u] = r.y;
        out[base + 2u] = r.z;
        out[base + 3u] = r.w;
    }} else {{
        for (var i = base; i < length; i = i + 1u) {{
            out[i] = fused({tail}).x;
        }}
    }}
}}
"#,
        vec = vec_args.join(", "),
        tail = tail_args.join(", "),
    );

    let mut cache_key = format!("fused_{n_inputs}_");
    for word in program {
        let _ = write!(cache_key, "{word:08x}");
    }
    Ok(FusedKernel { wgsl, cache_key, n_inputs })
}

/// Runs a fused expression over `n_inputs` same-size f32 tensors into the
/// caller-allocated `id_out` (which may be one of the inputs).
///
/// # Safety
/// `program` must point to `program_len` readable `u32`s and `inputs` to
/// `n_inputs` readable `u64` handles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_fused_dispatch(
    program: *const u32,
    program_len: u32,
    inputs: *const u64,
    n_inputs: u32,
    id_out: u64,
) -> i32 {
    ffi_guard("hc4j_fused_dispatch", || {
        if program.is_null() || inputs.is_null() || program_len == 0 || n_inputs == 0 {
            return Err(Hc4jError::InvalidParam("null or empty fused program"));
        }
        if program_len as usize > MAX_PROGRAM_WORDS {
            return Err(Hc4jError::InvalidParam("fused program too long"));
        }
        let program = unsafe { std::slice::from_raw_parts(program, program_len as usize) };
        let inputs: Vec<TensorId> = unsafe { std::slice::from_raw_parts(inputs, n_inputs as usize) }.to_vec();
        let engine = get_engine()?;
        // n input bindings + 1 output binding in one shader stage.
        if n_inputs + 1 > engine.limits.max_storage_buffers_per_shader_stage {
            return Err(Hc4jError::Unsupported("too many fused inputs; split the expression"));
        }
        let kernel = compile(program, n_inputs).map_err(|e| {
            crate::hc4j_trace!("rejected fused program: {e:?}");
            Hc4jError::InvalidParam("malformed fused program")
        })?;
        let pipeline = engine.get_or_compile(&kernel.cache_key, "main", &kernel.wgsl)?;
        // The executor checks that every input matches the output's size.
        run_contiguous(&pipeline, &|len| bytemuck::bytes_of(&[len, 1u32, 0, 0]).to_vec(), &inputs, id_out)
    })
}

#[cfg(test)]
mod tests {
    use super::{FusionError, compile, op};

    /// ((sin(x0) + x1) * x2) + 2.0
    pub(crate) fn sample_program() -> Vec<u32> {
        vec![
            op::word(op::LOAD, 0),
            op::word(op::SIN, 0),
            op::word(op::LOAD, 1),
            op::word(op::ADD, 0),
            op::word(op::LOAD, 2),
            op::word(op::MUL, 0),
            op::word(op::CONST, 0),
            2.0f32.to_bits(),
            op::word(op::ADD, 0),
        ]
    }

    #[test]
    fn value_numbering_dedupes_common_subexpressions() {
        // sin(x0) * sin(x0) + (x1 + x0) * (x0 + x1)
        let program = [
            op::word(op::LOAD, 0),
            op::word(op::SIN, 0),
            op::word(op::LOAD, 0),
            op::word(op::SIN, 0),
            op::word(op::MUL, 0),
            op::word(op::LOAD, 1),
            op::word(op::LOAD, 0),
            op::word(op::ADD, 0),
            op::word(op::LOAD, 0),
            op::word(op::LOAD, 1),
            op::word(op::ADD, 0),
            op::word(op::MUL, 0),
            op::word(op::ADD, 0),
        ];
        let k = compile(&program, 2).unwrap();
        let body = k.wgsl.split("fn main").next().unwrap();
        assert_eq!(body.matches("sin(").count(), 1, "{body}");
        assert_eq!(body.matches(" + v").count(), 2, "one commuted add, one final add: {body}");
    }

    #[test]
    fn malformed_programs_are_rejected_not_panicking() {
        let l = |k| op::word(op::LOAD, k);
        assert_eq!(compile(&[], 1).err(), Some(FusionError::Empty));
        assert_eq!(compile(&[op::word(op::ADD, 0)], 1).err(), Some(FusionError::StackUnderflow));
        assert_eq!(compile(&[l(0), l(0)], 1).err(), Some(FusionError::Unbalanced(2)));
        assert_eq!(compile(&[l(5)], 1).err(), Some(FusionError::InputOutOfRange(5)));
        assert_eq!(compile(&[l(0)], 2).err(), Some(FusionError::UnusedInput(1)));
        assert_eq!(compile(&[op::word(op::CONST, 0)], 1).err(), Some(FusionError::Truncated));
        assert_eq!(
            compile(&[l(0), op::word(op::CONST, 0), f32::NAN.to_bits(), op::word(op::ADD, 0)], 1).err(),
            Some(FusionError::NonFiniteConstant)
        );
        assert_eq!(compile(&[l(0), l(0), op::word(0x7F, 0)], 1).err(), Some(FusionError::UnknownOpcode(0x7F)));
        assert_eq!(compile(&vec![l(0); 2000], 1).err(), Some(FusionError::TooLong));
    }

    #[test]
    fn cache_key_is_structural() {
        let a = compile(&sample_program(), 3).unwrap();
        let b = compile(&sample_program(), 3).unwrap();
        assert_eq!(a.cache_key, b.cache_key);
        assert!(a.cache_key.starts_with("fused_3_"));
    }
}
