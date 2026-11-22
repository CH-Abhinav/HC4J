use crate::error::{Hc4jError, Hc4jResult};

pub const MAX_PARTIALS: u32 = 1024;
pub const ELEMS_PER_THREAD: u64 = 4;

/// Which fold a reduction applies.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    /// Sum, divided by the element count in the final pass.
    Mean,
    Max,
    Min,
    Prod,
}

impl ReduceOp {
    /// Identity element: the value a reduction of nothing returns.
    pub fn identity(self) -> &'static str {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => "0.0",
            ReduceOp::Max => "-3.4028235e38",
            ReduceOp::Min => "3.4028235e38",
            ReduceOp::Prod => "1.0",
        }
    }

    /// WGSL expression combining accumulators `a` and `b`.
    pub fn combine(self) -> &'static str {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => "a + b",
            ReduceOp::Max => "max(a, b)",
            ReduceOp::Min => "min(a, b)",
            ReduceOp::Prod => "a * b",
        }
    }

    /// Applied once, by the thread that writes the result.
    pub fn finalize(self) -> &'static str {
        match self {
            ReduceOp::Mean => "select(v, v / f32(p.params.y), p.params.y > 0u)",
            _ => "v",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Mean => "mean",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
            ReduceOp::Prod => "prod",
        }
    }

    /// Code for the FFI, mirrored by `hc4j.ops.NnOps.Reduction`.
    pub fn from_code(code: u32) -> Hc4jResult<Self> {
        match code {
            0 => Ok(ReduceOp::Sum),
            1 => Ok(ReduceOp::Mean),
            2 => Ok(ReduceOp::Max),
            3 => Ok(ReduceOp::Min),
            4 => Ok(ReduceOp::Prod),
            _ => Err(Hc4jError::InvalidParam("unknown reduction op")),
        }
    }
}

/// What a reduction reads at each index.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Load {
    /// `A[i]`
    Value,
    /// `A[i] * A[i]` — sum of squares without an intermediate tensor.
    Square,
    /// `(A[i] - B[i])^2` — mean squared error in one pass.
    Diff2,
}

impl Load {
    pub fn inputs(self) -> usize {
        match self {
            Load::Value | Load::Square => 1,
            Load::Diff2 => 2,
        }
    }

    pub fn expr(self) -> &'static str {
        match self {
            Load::Value => "A[i]",
            Load::Square => "A[i] * A[i]",
            Load::Diff2 => "(A[i] - B[i]) * (A[i] - B[i])",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Load::Value => "val",
            Load::Square => "sq",
            Load::Diff2 => "diff2",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_are_finite_so_pairwise_combines_cannot_produce_nan() {
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min, ReduceOp::Prod] {
            let ident: f32 = op.identity().parse().expect("identity parses as f32");
            assert!(ident.is_finite(), "{} identity must be finite", op.name());
        }
    }

    #[test]
    fn op_codes_round_trip_and_reject_garbage() {
        for (code, expected) in [
            (0, ReduceOp::Sum),
            (1, ReduceOp::Mean),
            (2, ReduceOp::Max),
            (3, ReduceOp::Min),
            (4, ReduceOp::Prod),
        ] {
            assert_eq!(ReduceOp::from_code(code).expect("valid code"), expected);
        }
        assert!(ReduceOp::from_code(5).is_err());
    }

    #[test]
    fn mean_divides_only_when_a_count_is_supplied() {
        assert!(ReduceOp::Mean.finalize().contains("p.params.y > 0u"));
        assert_eq!(ReduceOp::Sum.finalize(), "v");
    }

    #[test]
    fn two_input_loads_declare_two_bindings() {
        assert_eq!(Load::Diff2.inputs(), 2);
    }
}
