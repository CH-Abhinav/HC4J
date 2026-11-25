//! Linear algebra kernels: `f32` GEMM/GEMV/transpose, batched matmul, and mixed `f32 x f16` GEMM/GEMV.

pub mod batched;
pub mod f16_matmul;
pub mod matmul;

pub use batched::run_batched_matmul;
pub use f16_matmul::run_matmul_f32_f16;
pub use matmul::{Caps, FLAG_TRANS_A, FLAG_TRANS_B, plan as matmul_plan, run_matmul, run_transpose, warmup_kernels};
