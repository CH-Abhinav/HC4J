//! GPU compute kernel domains: elementwise primitives, math, linear algebra,
//! neural-network ops, reductions, and fused expression compiler.

pub mod elementwise;
pub mod fusion;
pub mod linalg;
pub mod math;
pub mod nn;
pub mod reduce;

#[cfg(test)]
pub mod validation;
