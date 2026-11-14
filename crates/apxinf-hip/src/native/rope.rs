//! Qwen3-VL RoPE variants: the text stack's mRoPE, and the vision tower's 2D
//! rotation as it lands.
//!
//! Semantics are cuda-new's (see `kernels/rope.hip`).

use apxinf_core::{Error, Result, Tensor};

use super::elementwise::HipDeviceBytes;
use super::{bf16_ptr, int, require_numel, rows_cols, same_shape};
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

/// Qwen3-VL text QK-norm and mRoPE in one pass -- cuda-new's fused
/// `rms_norm_apply_mrope_qk`.
///
/// `query` and `key` hold flat `[tokens * heads, head_dim]` projections; the
/// per-head RMSNorm (weight `[head_dim]`, eps) runs first and its result is
/// rounded to BF16 -- the rounding the separate [`rms_norm`] store performed
/// -- and then the half-split pairs rotate with the same interleaved-section
/// layout [`apply_mrope`] uses. `query_out` and `key_out` receive the rotated
/// values; the result is bit-identical to `rms_norm` followed by
/// [`apply_mrope`], which is the acceptance contract between the two forms.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_apply_mrope_qk(
    ctx: &HipContext,
    query: &Tensor,
    query_weight: &Tensor,
    key: &Tensor,
    key_weight: &Tensor,
    positions: &HipDeviceBytes,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    theta: f32,
    sections: [usize; 3],
    query_out: &Tensor,
    key_out: &Tensor,
) -> Result<()> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(Error::Contract(
            "apxinf-hip QK rms+mRoPE: head_dim must be an even, non-zero width",
        ));
    }
    // The three sections share the half-split pairs, so H and W together
    // cannot claim more than half the head. Same bound apply_mrope enforces.
    let section_hw = sections[1]
        .checked_add(sections[2])
        .ok_or(Error::Contract("apxinf-hip QK rms+mRoPE: section overflow"))?;
    if section_hw > head_dim / 2 {
        return Err(Error::Other(format!(
            "apxinf-hip QK rms+mRoPE: sections h+w {section_hw} exceed head_dim/2 {}",
            head_dim / 2
        )));
    }
    let (q_rows, cols) = rows_cols(query)?;
    let (k_rows, k_cols) = rows_cols(key)?;
    if cols != head_dim || k_cols != head_dim {
        return Err(Error::ShapeMismatch {
            expected: format!("[tokens * heads, {head_dim}]"),
            got: format!("{} columns and {}", cols, k_cols),
        });
    }
    if q_rows % q_heads != 0 || k_rows % kv_heads != 0 {
        return Err(Error::ShapeMismatch {
            expected: format!("[{q_heads}]- and [{kv_heads}]-divisible row counts"),
            got: format!("{q_rows} and {k_rows} rows"),
        });
    }
    let tokens = q_rows / q_heads;
    if tokens == 0 || k_rows / kv_heads != tokens {
        return Err(Error::ShapeMismatch {
            expected: format!("the same token count, {tokens}"),
            got: format!("{} q rows, {} k rows", q_rows / q_heads, k_rows / kv_heads),
        });
    }
    for (weight, what) in [(query_weight, "query"), (key_weight, "key")] {
        require_numel(weight, head_dim, || format!("[{head_dim}] {what} norm weight"))?;
    }
    require_numel(query_out, q_rows * head_dim, || format!("[{tokens}, {q_heads}, {head_dim}]"))?;
    require_numel(key_out, k_rows * head_dim, || {
        format!("[{tokens}, {kv_heads}, {head_dim}]")
    })?;
    let needed = tokens
        .checked_mul(3)
        .and_then(|value| value.checked_mul(std::mem::size_of::<u32>()))
        .ok_or(Error::Contract("apxinf-hip QK rms+mRoPE: position size overflow"))?;
    if positions.len() < needed {
        return Err(Error::DataLengthMismatch {
            expected: needed,
            got: positions.len(),
        });
    }
    let (pq, pqw, pk, pkw, po, pok) = (
        bf16_ptr(ctx, query)?,
        bf16_ptr(ctx, query_weight)?,
        bf16_ptr(ctx, key)?,
        bf16_ptr(ctx, key_weight)?,
        bf16_ptr(ctx, query_out)?,
        bf16_ptr(ctx, key_out)?,
    );
    check("rms_norm_apply_mrope_qk", unsafe {
        ffi::apxinf_hip_native_rms_norm_apply_mrope_qk(
            pq.cast_const(),
            pqw.cast_const(),
            po,
            pk.cast_const(),
            pkw.cast_const(),
            pok,
            positions.ptr(),
            int(tokens)?,
            int(q_heads)?,
            int(kv_heads)?,
            int(head_dim)?,
            eps,
            theta,
            int(sections[1])?,
            int(sections[2])?,
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
/// The tower's QKV path runs the fused [`super::split_qkv_axial_rope`]; this
/// rotation-only op is that fusion's acceptance reference: paired with
/// [`super::split_qkv_bias`] it reproduces the fused kernel bit for bit,
/// because both round the biased projection to BF16 before rotating.
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
