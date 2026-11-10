//! Qwen3-VL RoPE variants: the text stack's mRoPE, and the vision tower's 2D
//! rotation as it lands.
//!
//! Semantics are cuda-new's (see `kernels/rope.hip`).

use apxinf_core::{Error, Result, Tensor};

use super::elementwise::HipDeviceBytes;
use super::{bf16_ptr, int, same_shape};
use crate::ffi::{self, check};
use crate::runtime::HipContext;

/// Qwen3-VL text mRoPE over a `[tokens, heads, head_dim]` BF16 tensor.
///
/// `positions` holds `tokens * 3` `u32` values as `(t, h, w)` per token.
/// Frequency pair `p` rotates the half-split pair `(p, head_dim/2 + p)` by
/// `position[axis(p)] * theta^(-2p/head_dim)`, where `sections` `[t, h, w]`
/// selects the axis in Qwen3-VL's interleaved layout. `out` may alias `input`.
pub fn apply_mrope(
    ctx: &HipContext,
    input: &Tensor,
    positions: &HipDeviceBytes,
    heads: usize,
    head_dim: usize,
    theta: f32,
    sections: [usize; 3],
    out: &Tensor,
) -> Result<()> {
    let dims = input.shape().dims();
    if dims.len() != 3 || dims[1] != heads || dims[2] != head_dim {
        return Err(Error::ShapeMismatch {
            expected: format!("[tokens, {heads}, {head_dim}]"),
            got: input.shape().to_string(),
        });
    }
    same_shape(input, out)?;
    let tokens = dims[0];
    if head_dim % 2 != 0 {
        return Err(Error::Contract(
            "apxinf-hip mRoPE: head_dim must be even",
        ));
    }
    // The three sections share the half-split pairs, so H and W together
    // cannot claim more than half the head. Same bound cuda-new enforces.
    let section_hw = sections[1]
        .checked_add(sections[2])
        .ok_or(Error::Contract("apxinf-hip mRoPE: section overflow"))?;
    if section_hw > head_dim / 2 {
        return Err(Error::Other(format!(
            "apxinf-hip mRoPE: sections h+w {section_hw} exceed head_dim/2 {}",
            head_dim / 2
        )));
    }
    let needed = tokens
        .checked_mul(3)
        .and_then(|value| value.checked_mul(std::mem::size_of::<u32>()))
        .ok_or(Error::Contract("apxinf-hip mRoPE: position size overflow"))?;
    if positions.len() < needed {
        return Err(Error::DataLengthMismatch {
            expected: needed,
            got: positions.len(),
        });
    }
    let (pi, po) = (bf16_ptr(ctx, input)?, bf16_ptr(ctx, out)?);
    check("apply_mrope", unsafe {
        ffi::apxinf_hip_native_apply_mrope(
            pi.cast_const(),
            po,
            positions.ptr(),
            int(tokens)?,
            int(heads)?,
            int(head_dim)?,
            int(head_dim)?,
            int(sections[1])?,
            int(sections[2])?,
            theta,
            ctx.stream(),
        )
    })
}

/// Qwen3-VL vision-tower 2D RoPE over a `[tokens, heads, head_dim]` BF16
/// tensor.
///
/// `positions` holds `tokens * 2` `u32` values as `(h, w)` per token. The
/// first `head_dim/4` half-split pairs rotate by the height position and the
/// rest by the width one. `out` may alias `input`.
///
/// Pair this with [`super::split_qkv_bias`] to reproduce cuda-new's fused
/// `split_qkv_axial_rope`: that kernel rounds the biased projection to BF16
/// before rotating, which is exactly what the split writes out.
pub fn apply_vision_2d(
    ctx: &HipContext,
    input: &Tensor,
    positions: &HipDeviceBytes,
    heads: usize,
    head_dim: usize,
    theta: f32,
    out: &Tensor,
) -> Result<()> {
    let dims = input.shape().dims();
    if dims.len() != 3 || dims[1] != heads || dims[2] != head_dim {
        return Err(Error::ShapeMismatch {
            expected: format!("[tokens, {heads}, {head_dim}]"),
            got: input.shape().to_string(),
        });
    }
    same_shape(input, out)?;
    let tokens = dims[0];
    if head_dim % 4 != 0 {
        return Err(Error::Contract(
            "apxinf-hip vision 2D RoPE: head_dim must be a multiple of 4",
        ));
    }
    let needed = tokens
        .checked_mul(2)
        .and_then(|value| value.checked_mul(std::mem::size_of::<u32>()))
        .ok_or(Error::Contract(
            "apxinf-hip vision 2D RoPE: position size overflow",
        ))?;
    if positions.len() < needed {
        return Err(Error::DataLengthMismatch {
            expected: needed,
            got: positions.len(),
        });
    }
    let (pi, po) = (bf16_ptr(ctx, input)?, bf16_ptr(ctx, out)?);
    check("apply_vision_2d", unsafe {
        ffi::apxinf_hip_native_apply_vision_2d(
            pi.cast_const(),
            po,
            positions.ptr(),
            int(tokens)?,
            int(heads)?,
            int(head_dim)?,
            theta,
            ctx.stream(),
        )
    })
}
