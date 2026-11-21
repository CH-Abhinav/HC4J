//! Elementwise neural network activations.

use crate::error::{Hc4jError, Hc4jResult};
use crate::ops::elementwise::UnaryKernel;

/// Elementwise activations. `alpha` is only read by the parameterised ones.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    LeakyRelu,
    Elu,
    /// The tanh approximation, as used by GPT-2/BERT — not the erf form,
    /// which WGSL has no built-in for.
    Gelu,
    Silu,
    Sigmoid,
    Softplus,
}

impl Activation {
    pub fn name(self) -> &'static str {
        match self {
            Activation::Relu => "relu",
            Activation::LeakyRelu => "leaky_relu",
            Activation::Elu => "elu",
            Activation::Gelu => "gelu",
            Activation::Silu => "silu",
            Activation::Sigmoid => "sigmoid",
            Activation::Softplus => "softplus",
        }
    }

    /// Codes mirrored by `hc4j.ops.NnOps.Activation`.
    pub fn from_code(code: u32) -> Hc4jResult<Self> {
        match code {
            0 => Ok(Activation::Relu),
            1 => Ok(Activation::LeakyRelu),
            2 => Ok(Activation::Elu),
            3 => Ok(Activation::Gelu),
            4 => Ok(Activation::Silu),
            5 => Ok(Activation::Sigmoid),
            6 => Ok(Activation::Softplus),
            _ => Err(Hc4jError::InvalidParam("unknown activation")),
        }
    }

    /// Whether the op reads `dims.params.z` as an f32 parameter.
    pub fn takes_alpha(self) -> bool {
        matches!(self, Activation::LeakyRelu | Activation::Elu)
    }

    /// The function body, with `__T__` for the operand type. `__T__(c)` is a
    /// splat for the vec4 width and a no-op conversion for the scalar one, so
    /// one body serves both.
    fn body(self) -> &'static str {
        match self {
            Activation::Relu => "return max(x, __T__(0.0));",
            // select, not max: a slope above 1 would make max() wrong.
            Activation::LeakyRelu => {
                "let a = bitcast<f32>(dims.params.z);\n    \
                 return select(__T__(a) * x, x, x > __T__(0.0));"
            }
            // min(x, 0) keeps the unselected arm from overflowing to inf.
            Activation::Elu => {
                "let a = bitcast<f32>(dims.params.z);\n    \
                 let neg = __T__(a) * (exp(min(x, __T__(0.0))) - __T__(1.0));\n    \
                 return select(neg, x, x > __T__(0.0));"
            }
            // tanh saturates by |a| = 10 (within 1 ulp of f32 1.0), and the
            // cubic reaches a ~= 24000 at x = 88. Unclamped, a driver that
            // evaluates tanh as (e^2a - 1)/(e^2a + 1) returns inf/inf = NaN,
            // so gelu(88) came out NaN instead of 88.
            Activation::Gelu => {
                "let cubic = __T__(0.7978845608028654) * (x + __T__(0.044715) * x * x * x);\n    \
                 let inner = clamp(cubic, __T__(-10.0), __T__(10.0));\n    \
                 return __T__(0.5) * x * (__T__(1.0) + tanh(inner));"
            }
            // e = exp(-|x|) is in (0, 1], so neither arm can overflow.
            Activation::Silu => {
                "let e = exp(-abs(x));\n    \
                 let num = select(e, __T__(1.0), x >= __T__(0.0));\n    \
                 return x * num / (__T__(1.0) + e);"
            }
            Activation::Sigmoid => {
                "let e = exp(-abs(x));\n    \
                 let num = select(e, __T__(1.0), x >= __T__(0.0));\n    \
                 return num / (__T__(1.0) + e);"
            }
            Activation::Softplus => {
                "return max(x, __T__(0.0)) + log(__T__(1.0) + exp(-abs(x)));"
            }
        }
    }

    /// WGSL prelude: the same body at both widths.
    pub fn prelude(self) -> String {
        format!("{}\n{}", self.vec4_fn("hc4j_act4"), self.scalar_fn("hc4j_act1"))
    }

    /// The vec4 form under a chosen name, for shaders that need several
    /// activations at once — the fused kernels work only in `vec4<f32>`.
    pub fn vec4_fn(self, name: &str) -> String {
        format!(
            "fn {name}(x: vec4<f32>) -> vec4<f32> {{\n    {}\n}}\n",
            self.body().replace("__T__", "vec4<f32>")
        )
    }

    fn scalar_fn(self, name: &str) -> String {
        format!(
            "fn {name}(x: f32) -> f32 {{\n    {}\n}}\n",
            self.body().replace("__T__", "f32")
        )
    }

    pub fn kernel(self, prelude: &str, alpha: f32) -> UnaryKernel<'_> {
        UnaryKernel {
            family: "nn",
            op_name: self.name(),
            prelude,
            op4: "hc4j_act4",
            op1: "hc4j_act1",
            alpha_bits: if self.takes_alpha() { alpha.to_bits() } else { 0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_codes_round_trip() {
        let all = [
            Activation::Relu,
            Activation::LeakyRelu,
            Activation::Elu,
            Activation::Gelu,
            Activation::Silu,
            Activation::Sigmoid,
            Activation::Softplus,
        ];
        for (code, expected) in all.iter().enumerate() {
            assert_eq!(Activation::from_code(code as u32).expect("valid code"), *expected);
        }
        assert!(Activation::from_code(all.len() as u32).is_err());
    }

    #[test]
    fn both_widths_are_generated_from_one_body() {
        // The vec4 and scalar paths of the unary template must not drift, so
        // they come from the same body with __T__ substituted.
        for act in [Activation::Relu, Activation::Gelu, Activation::Silu] {
            let prelude = act.prelude();
            assert!(prelude.contains("fn hc4j_act4(x: vec4<f32>) -> vec4<f32>"));
            assert!(prelude.contains("fn hc4j_act1(x: f32) -> f32"));
            assert!(!prelude.contains("__T__"), "{}: unsubstituted type", act.name());
        }
    }

    #[test]
    fn only_parameterised_activations_read_alpha() {
        let prelude = Activation::LeakyRelu.prelude();
        assert!(prelude.contains("dims.params.z"));
        assert!(!Activation::Relu.prelude().contains("dims.params.z"));
        // An alpha passed to an op that ignores it must not reach the uniform,
        // where it would look like a parameter to a cached pipeline.
        assert_eq!(Activation::Relu.kernel("", 0.1).alpha_bits, 0);
        assert_eq!(Activation::LeakyRelu.kernel("", 0.1).alpha_bits, 0.1f32.to_bits());
    }

    #[test]
    fn every_exp_argument_is_non_positive() {
        // What keeps sigmoid, SiLU and softplus finite for large |x|.
        for act in [Activation::Sigmoid, Activation::Silu, Activation::Softplus] {
            let body = act.body();
            assert!(body.contains("exp(-abs(x))"), "{} must clamp its exp", act.name());
            assert!(!body.contains("exp(x)"), "{} has an unclamped exp", act.name());
        }
    }
}
