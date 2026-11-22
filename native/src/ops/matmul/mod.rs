//! Matrix multiplication `C[M×N] = op(A)[M×K] · op(B)[K×N]`, f32, row-major.
//!
//! Three kernel families, chosen at runtime by [`plan`] from `(M, N, K)`,
//! the transpose flags and the device's capabilities:
//!
//! 1. **Register-tiled GEMM** ([`register_kernel_wgsl`]). Each workgroup
//!    computes a BM×BN tile of C; each thread a TM×TN register micro-tile.
//!    A and B tiles are staged in workgroup memory with vec4 global loads (or
//!    guarded scalar loads when K or N is not a multiple of 4). The A tile is
//!    stored k-major with a +1 row pad so its transposed stores do not
//!    serialize on shared-memory banks. The K loop over a tile is fully
//!    unrolled at generation time, and accumulation uses vec4 `fma`. Two
//!    configurations: 64×64 with 4×4 micro-tiles (portable), and 128×128
//!    with 8×8 (discrete GPUs, whose register files hold 64 accumulators
//!    without spilling).
//! 2. **16×16 tiled GEMM** ([`TILED16_WGSL`]). Classic shared-memory tiling,
//!    one output per thread, for small or thin shapes.
//! 3. **GEMV** for `M == 1` or `N == 1`. A 2-D GEMM tile would read one row
//!    or column of a whole tile to produce a single value, wasting most of
//!    its bandwidth. Row-major matrix · vector uses a group of lanes per row
//!    (coalesced reads along K) plus a shared-memory tree reduction, or a
//!    `subgroupAdd` reduction when the device supports subgroups. Vector ·
//!    matrix uses a thread per column (coalesced across columns) with K split
//!    into slices reduced in shared memory.
//!
//! **Transposed operands.** A GEMV with a transposed matrix swaps to the
//! other GEMV mapping, so no copy is needed. A GEMM with a transposed operand
//! first transposes it in VRAM with a 32×33 padded-tile kernel, recorded into
//! the same command buffer, so the GEMM streams B along K with coalesced
//! reads.
//!
//! **Working sets larger than VRAM.** When the operands cannot be made
//! co-resident, matmul degrades in two steps:
//!
//! * [`run_matmul_streaming`] computes C in row blocks
//!   (`C[i..i+rb] = A[i..i+rb] · B`) with B held resident. Row blocks of a
//!   row-major matrix are contiguous, so each is one copy in and one copy out,
//!   and K is never split.
//! * [`run_matmul_blocked`], when even B does not fit, additionally blocks
//!   over K and accumulates partial products in a resident C block (`beta = 1`
//!   after the first K block). N stays whole, so one row of B plus one row of
//!   C must fit in VRAM; B is re-read once per row block.
//!
//! A stored-transposed operand that has to be streamed must be transposed into
//! its own tensor first ([`run_transpose`]), since transposing in place would
//! need the whole matrix resident.
//!
//! Storage bindings are `read_write` (slab regions share buffers). Output
//! aliasing an input is rejected, since tiles of C would race with reads of A/B.

pub mod abi;
pub mod batched;
pub mod dispatch;
pub mod plan;
pub mod shaders;
pub mod streaming;

pub use abi::*;
pub use batched::*;
pub use dispatch::*;
pub use plan::*;
pub use shaders::*;
pub use streaming::*;
