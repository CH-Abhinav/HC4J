//! Compute kernels. All modules use strictly lowercase paths so the crate
//! builds on case-sensitive filesystems.

pub mod arithmetic;
pub mod elementwise;
pub mod exponential;
pub mod fusion;
pub mod matmul;
pub mod trigno;

#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod shader_validation;
