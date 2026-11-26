//! GDN operand chain in one kernel: the causal depthwise convolution, SiLU,
//! the q/k/v split and the per-head q/k l2norm, over one read of the raw
//! `in_proj_qkv` row.
//!
//! Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; the
//! convolution is this repository's causal-convolution contract (FP32 tap
//! order over zero history). See NOTICE.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

/// `silu(conv(x))` split into q, k and v, with the official GDN q/k l2norm
/// applied in the same kernel.
///
/// Inputs: BF16 `x [1, T, C]` (the raw packed projection) and the BF16
/// depthwise taps `[C, S, 1]`; `C = 2·Hk·Dk + Hv·Dv` and the q, k and v
/// channels are contiguous in that order. Returns BF16 `q`, `k`
/// `[1, T, Hk, Dk]` and `v` `[1, T, Hv, Dv]`.
///
/// The chain stays in FP32 between the convolution and the write: SiLU and
/// the q/k norm are applied to the accumulated value, and each output is
/// rounded to BF16 once. A separately composed chain (the causal-conv kernel,
/// then SiLU, then the split and two RMSNorm nodes) rounds between those
/// steps, so this kernel is the same chain with less rounding error, not a
/// different arithmetic contract.
pub struct GdnFusedPrep {
    kernel: MetalKernel,
    key_heads: usize,
    key_dim: usize,
    value_heads: usize,
    value_dim: usize,
    taps: usize,
    channels: usize,
}

/// The f32 literal form the reference substitutes (`format!("{:.12e}f")` of
/// the f32 value), so the kernel sees the same rounded constant.
fn f32_literal(value: f32) -> String {
    format!("{value:.12e}f")
}

impl GdnFusedPrep {
    /// Binds one geometry. `channels` must equal `2·key_heads·key_dim +
    /// value_heads·value_dim`; `key_dim` and `value_dim` must be multiples of
    /// 32 (the kernel's lane split) and `value_heads` a multiple of
    /// `key_heads`. Every extent the kernel indexes is checked to fit `i32`.
    pub fn new(
        stream: &Stream,
        key_heads: usize,
        key_dim: usize,
        value_heads: usize,
        value_dim: usize,
        taps: usize,
        channels: usize,
    ) -> Result<Self> {
        let expected = 2 * key_heads * key_dim + value_heads * value_dim;
        if key_heads == 0
            || value_heads == 0
            || key_dim == 0
            || value_dim == 0
            || taps == 0
            || value_heads % key_heads != 0
            || key_dim % 32 != 0
            || value_dim % 32 != 0
            || channels != expected
            || channels
                .checked_mul(taps)
                .is_none_or(|extent| extent > i32::MAX as usize)
            || channels
                .checked_mul(key_heads.max(value_heads))
                .is_none_or(|extent| extent > i32::MAX as usize)
        {
            return Err(Error::Other(format!(
                "fused GDN prep requires C = 2·Hk·Dk + Hv·Dv = {expected} = {channels}, Dk and Dv \
                 multiples of 32, Hv a multiple of Hk, and extents within i32"
            )));
        }
        let source = include_str!("../native/gdn_fused_prep.metal")
            .replace("EPSILON", &f32_literal(1e-6))
            .replace("QSCALE", &f32_literal((key_dim as f32).powf(-0.5)))
            .replace("KSCALE", &f32_literal(1.0));
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                &format!(
                    "apxinf_gdn_fused_prep_c{channels}_dk{key_dim}_dv{value_dim}_hk{key_heads}_hv{value_heads}_s{taps}"
                ),
                &["x", "weight", "length"],
                &["q", "k", "v"],
                &source,
                "",
            )?,
            key_heads,
            key_dim,
            value_heads,
            value_dim,
            taps,
            channels,
        })
    }

    pub fn call(&self, x: &Array, weight: &Array) -> Result<(Array, Array, Array)> {
        let [1, length, channels] = *x.shape() else {
            return Err(Error::Contract("fused GDN prep input must be [1, T, C]"));
        };
        let [wc, taps, 1] = *weight.shape() else {
            return Err(Error::Contract("fused GDN prep weight must be [C, S, 1]"));
        };
        if length == 0
            || channels != self.channels
            || wc != channels
            || taps != self.taps
            || [x, weight].iter().any(|a| a.dtype() != MlxDType::BF16)
            || length
                .checked_mul(channels)
                .is_none_or(|extent| extent > i32::MAX as usize)
        {
            return Err(Error::Contract(
                "fused GDN prep requires BF16 [1, T, C] qkv and [C, S, 1] taps with C = \
                 2·Hk·Dk + Hv·Dv and extents within i32",
            ));
        }
        let scalar = Array::from_i32(x.stream(), &[], &[length as i32])?;
        let q_shape = [1, length, self.key_heads, self.key_dim];
        let v_shape = [1, length, self.value_heads, self.value_dim];
        let time = length.min(8);
        // SAFETY: the grid is (32, T, Hv) and every thread with t >= T returns.
        // Each lane reads `x[row * C + channel]` for `0 <= row <= t < T` and
        // `weight[channel * S + s]` for `s < S`, and writes exactly its own
        // `Dk / 32` q/k columns of `(t, hk)` and `Dv / 32` v columns of
        // `(t, hv)`; the templates name the same extents the buffers were
        // checked against.
        let outputs = unsafe {
            self.kernel.call(
                &[x.clone(), weight.clone(), scalar],
                &[
                    (q_shape.as_slice(), MlxDType::BF16),
                    (q_shape.as_slice(), MlxDType::BF16),
                    (v_shape.as_slice(), MlxDType::BF16),
                ],
                [32, length, self.value_heads],
                [32, time, 1],
                &[
                    ("InT", TemplateArg::DType(MlxDType::BF16)),
                    ("C", TemplateArg::Int(channels as i32)),
                    ("Dk", TemplateArg::Int(self.key_dim as i32)),
                    ("Dv", TemplateArg::Int(self.value_dim as i32)),
                    ("Hk", TemplateArg::Int(self.key_heads as i32)),
                    ("Hv", TemplateArg::Int(self.value_heads as i32)),
                    ("S", TemplateArg::Int(self.taps as i32)),
                ],
            )?
        };
        let mut outputs = outputs.into_iter();
        match (
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
        ) {
            (Some(q), Some(k), Some(v), None) => Ok((q, k, v)),
            _ => Err(Error::Contract(
                "fused GDN prep must return exactly q, k and v",
            )),
        }
    }
}
