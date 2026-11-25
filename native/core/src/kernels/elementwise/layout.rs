use crate::engine::error::{Hc4jError, Hc4jResult};
use crate::memory::{manager, TensorId};

pub type UnaryStrided<'a> = (&'a [u32], &'a [u32], &'a [u32]);
pub type BinaryStrided<'a> = (&'a [u32], &'a [u32], &'a [u32], &'a [u32]);
pub type TernaryStrided<'a> = (&'a [u32], &'a [u32], &'a [u32], &'a [u32], &'a [u32]);

/// `params = [length, is_contiguous, 0, 0]`; shape and strides are
/// reverse-pruned (innermost first, size-1 dims removed) and map to
/// `array<vec4<u32>, 2>` in WGSL, sidestepping 16-byte array-stride rules.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UnaryElemDims {
    pub params: [u32; 4],
    pub shape: [u32; 8],
    pub strides_a: [u32; 8],
    pub strides_c: [u32; 8],
}

impl UnaryElemDims {
    pub fn contiguous(length: u32) -> Self {
        Self::contiguous_with(length, 0)
    }

    /// `alpha_bits` is an f32 op parameter (a leaky-ReLU slope, an ELU alpha)
    /// passed as raw bits in `params.z` and read back with `bitcast`.
    pub fn contiguous_with(length: u32, alpha_bits: u32) -> Self {
        Self {
            params: [length, 1, alpha_bits, 0],
            shape: [1; 8],
            strides_a: [0; 8],
            strides_c: [0; 8],
        }
    }

    pub fn contiguous_with_params(length: u32, param_z: u32, param_w: u32) -> Self {
        Self {
            params: [length, 1, param_z, param_w],
            shape: [1; 8],
            strides_a: [0; 8],
            strides_c: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TernaryElemDims {
    pub params: [u32; 4],
    pub shape: [u32; 8],
    pub strides_cond: [u32; 8],
    pub strides_x: [u32; 8],
    pub strides_y: [u32; 8],
    pub strides_out: [u32; 8],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BinaryElemDims {
    pub params: [u32; 4],
    pub shape: [u32; 8],
    pub strides_a: [u32; 8],
    pub strides_b: [u32; 8],
    pub strides_c: [u32; 8],
}

impl BinaryElemDims {
    pub fn contiguous(length: u32) -> Self {
        Self {
            params: [length, 1, 0, 0],
            shape: [1; 8],
            strides_a: [0; 8],
            strides_b: [0; 8],
            strides_c: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CopyStridedDims {
    pub params: [u32; 4],
    pub shape: [u32; 8],
    pub strides: [u32; 8],
}

/// Reads a strided layout and reverse-prunes it: size-1 dimensions are
/// dropped and the rest reordered innermost-first, so kernels decompose the
/// linear index without knowing the rank. Returns the shape and one stride
/// array per input stride slice.
pub fn prune_strided_layout(shape: &[u32], strides: &[&[u32]]) -> Hc4jResult<([u32; 8], Vec<[u32; 8]>)> {
    let mut out_shape = [1u32; 8];
    let mut out_strides = vec![[0u32; 8]; strides.len()];
    if shape.is_empty() {
        return Ok((out_shape, out_strides));
    }
    if shape.len() > 8 {
        return Err(Hc4jError::Unsupported("rank above 8"));
    }
    if strides.iter().any(|s| s.len() != shape.len()) {
        return Err(Hc4jError::InvalidParam("stride rank mismatch"));
    }
    let mut effective = 0;
    for i in (0..shape.len()).rev() {
        if shape[i] > 1 {
            out_shape[effective] = shape[i];
            for (k, stride) in strides.iter().enumerate() {
                out_strides[k][effective] = stride[i];
            }
            effective += 1;
        }
    }
    Ok((out_shape, out_strides))
}

/// A contiguous call must cover the whole output tensor.
pub fn check_contiguous_length(id_out: TensorId, length: usize) -> Hc4jResult<()> {
    let size = manager()?.size_of(id_out)?;
    if (length as u64).checked_mul(4) != Some(size) {
        return Err(Hc4jError::InvalidParam("length does not match the output tensor"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::prune_strided_layout;

    #[test]
    fn layout_is_reverse_pruned() {
        let shape = [2u32, 1, 3];
        let a = [3u32, 3, 1];
        let c = [1u32, 1, 2];
        let (s, st) = prune_strided_layout(&shape, &[&a, &c]).unwrap();
        assert_eq!(&s[..3], &[3, 2, 1]);
        assert_eq!(&st[0][..2], &[1, 3]);
        assert_eq!(&st[1][..2], &[2, 1]);
    }

    #[test]
    fn layout_rejects_bad_input() {
        let shape = [2u32; 9];
        assert!(prune_strided_layout(&shape, &[&shape]).is_err());
        let bad_stride = [1u32; 2];
        assert!(prune_strided_layout(&[2u32, 3, 4], &[&bad_stride]).is_err());
    }
}
