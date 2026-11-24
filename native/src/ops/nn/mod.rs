//! Neural-network ops: activations, softmax, normalisation, losses and bias
//! broadcast.
//!
//! **Activations** are ordinary elementwise ops and reuse the unary executor,
//! so they inherit tiling, the 2-D grid fold, in-place output and the
//! host-streaming fallback for tensors larger than VRAM. They are not WGSL
//! built-ins, so each one supplies a prelude defining a `vec4` and a scalar
//! function — WGSL has neither overloading nor generics. One body is written
//! per activation with `__T__` standing for the width, and both functions are
//! generated from it, so the two paths cannot drift apart.
//!
//! **Row-wise ops** (softmax, layer norm, RMS norm, cross entropy) run one
//! workgroup per row: a reduction over the row in workgroup memory, then a
//! write pass over the same row. Each is a single dispatch.
//!
//! Numerics follow the stable formulations:
//!
//! * softmax uses the *online* max-and-sum recurrence (Milakov & Gimelshein),
//!   so a row is read twice rather than three times, and no `exp` ever sees a
//!   positive argument.
//! * layer norm accumulates Welford triples `(n, mean, M2)` and combines them
//!   pairwise in the tree, so the variance never comes from `E[x²] - E[x]²`.
//! * sigmoid, SiLU and softplus are written so that every `exp` argument is
//!   non-positive, which is what makes them finite for large `|x|`.
//!
//! Row-wise operands must be VRAM-resident and within one storage binding.

pub mod abi;
pub mod activations;
pub mod bias;
pub mod embedding;
pub mod rope;
pub mod row_ops;

pub use abi::*;
pub use activations::*;
pub use bias::*;
pub use embedding::*;
pub use rope::*;
pub use row_ops::*;
