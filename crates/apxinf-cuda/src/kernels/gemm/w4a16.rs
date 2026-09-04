//! AutoAWQ INT4 weight-only (W4A16) contracts.
//!
//! Weights stay in the on-disk AutoAWQ `gemm` packing (see
//! `kernels/custom/quantization.cuh`), optionally with several experts stored
//! back to back in one allocation. Two execution modes are offered:
//!
//! * [`dequant_bf16_into`] unpacks weights into a caller-owned BF16 `[K, N]`
//!   buffer so any dense GEMM can consume them (prefill scaffold).
//! * [`gemv_partial_into`] + [`partial_sum_into`] / [`partial_silu_mul_into`]
//!   run a split-K, multi-expert GEMV directly on the packed weights for
//!   `M = 1` decode; expert selection is read on device so the sequence is
//!   CUDA-graph capturable.

use apxinf_core::{DType, Error, Result};

use super::super::contracts::{check_cuda, require_buffers};
use crate::buffer::{CudaBuffer, CudaDeviceAddress};
use crate::context::CudaContext;
use crate::ffi;

/// Group size used by every checkpoint this contract currently accepts.
pub const AWQ_GROUP_SIZE: usize = 128;

/// Borrowed view of one or more AutoAWQ linears stored expert-major.
///
/// For `experts == 1` the strides are ignored. Element strides are in units of
/// `i32` words for `qweight`/`qzeros` and `f16` values for `scales`.
#[derive(Clone, Copy)]
pub struct AwqWeightView<'a> {
    pub qweight: &'a CudaBuffer,
    pub qzeros: &'a CudaBuffer,
    pub scales: &'a CudaBuffer,
    /// Input features `K`.
    pub in_dim: usize,
    /// Output features `N` (multiple of 8).
    pub out_dim: usize,
    pub group_size: usize,
    pub experts: usize,
    pub stride_q: usize,
    pub stride_z: usize,
    pub stride_s: usize,
}

impl<'a> AwqWeightView<'a> {
    /// Single linear (no expert dimension).
    pub fn single(
        qweight: &'a CudaBuffer,
        qzeros: &'a CudaBuffer,
        scales: &'a CudaBuffer,
        in_dim: usize,
        out_dim: usize,
        group_size: usize,
    ) -> Self {
        Self {
            qweight,
            qzeros,
            scales,
            in_dim,
            out_dim,
            group_size,
            experts: 1,
            stride_q: 0,
            stride_z: 0,
            stride_s: 0,
        }
    }

    pub fn packed_cols(&self) -> usize {
        self.out_dim / 8
    }

    pub fn groups(&self) -> usize {
        self.in_dim / self.group_size
    }

    /// Word count of one expert's `qweight`.
    pub fn qweight_words(&self) -> usize {
        self.in_dim * self.packed_cols()
    }

    /// Word count of one expert's `qzeros`.
    pub fn qzeros_words(&self) -> usize {
        self.groups() * self.packed_cols()
    }

    /// Element count of one expert's `scales`.
    pub fn scales_elements(&self) -> usize {
        self.groups() * self.out_dim
    }

    fn validate(&self, ctx: &CudaContext, operation: &str) -> Result<()> {
        if self.in_dim == 0
            || self.out_dim == 0
            || self.out_dim % 8 != 0
            || self.group_size == 0
            || self.in_dim % self.group_size != 0
            || self.experts == 0
        {
            return Err(Error::Other(format!(
                "{operation}: invalid AWQ geometry K={} N={} group={} experts={}",
                self.in_dim, self.out_dim, self.group_size, self.experts
            )));
        }
        if self.experts > 1
            && (self.stride_q < self.qweight_words()
                || self.stride_z < self.qzeros_words()
                || self.stride_s < self.scales_elements())
        {
            return Err(Error::Other(format!(
                "{operation}: AWQ expert strides are smaller than one expert"
            )));
        }
        let last = self.experts - 1;
        require_buffers(
            ctx,
            operation,
            &[
                (
                    "qweight",
                    self.qweight,
                    (last * self.stride_q + self.qweight_words()) * 4,
                ),
                (
                    "qzeros",
                    self.qzeros,
                    (last * self.stride_z + self.qzeros_words()) * 4,
                ),
                (
                    "scales",
                    self.scales,
                    (last * self.stride_s + self.scales_elements()) * 2,
                ),
            ],
        )
    }
}

/// Dequantize experts `[0, experts)` into `output` as BF16 `[experts, K, N]`
/// (expert stride `K * N` elements).
pub fn dequant_bf16_into(
    ctx: &CudaContext,
    weight: AwqWeightView<'_>,
    output: &CudaBuffer,
) -> Result<()> {
    dequant_range_bf16_into(ctx, weight, 0, weight.experts, output)
}

/// Dequantize experts `[first, first + count)` into `output` as BF16
/// `[count, K, N]`.
pub fn dequant_range_bf16_into(
    ctx: &CudaContext,
    weight: AwqWeightView<'_>,
    first: usize,
    count: usize,
    output: &CudaBuffer,
) -> Result<()> {
    weight.validate(ctx, "AWQ dequant")?;
    if count == 0 || first + count > weight.experts {
        return Err(Error::Other(format!(
            "AWQ dequant: expert range {first}..{} exceeds {}",
            first + count,
            weight.experts
        )));
    }
    let per_expert = weight.in_dim * weight.out_dim;
    require_buffers(
        ctx,
        "AWQ dequant",
        &[("output", output, count * per_expert * DType::BF16.size_in_bytes())],
    )?;
    let q = weight.qweight.view(first * weight.stride_q * 4, weight.qweight.len() - first * weight.stride_q * 4).map_err(Error::Cuda)?;
    let z = weight.qzeros.view(first * weight.stride_z * 4, weight.qzeros.len() - first * weight.stride_z * 4).map_err(Error::Cuda)?;
    let s = weight.scales.view(first * weight.stride_s * 2, weight.scales.len() - first * weight.stride_s * 2).map_err(Error::Cuda)?;
    unsafe {
        check_cuda(ffi::apxinf_awq_dequant_bf16(
            q.ptr(),
            z.ptr(),
            s.ptr(),
            output.ptr(),
            weight.in_dim as i32,
            weight.packed_cols() as i32,
            weight.group_size as i32,
            count as i32,
            weight.stride_q as i64,
            weight.stride_z as i64,
            weight.stride_s as i64,
            per_expert as i64,
            ctx.stream().handle(),
        ))
    }
}

/// How the GEMV maps `slots` onto experts and activations.
#[derive(Clone, Copy, Debug)]
pub struct GemvSlots {
    /// Number of (expert, activation) pairs evaluated by one launch.
    pub slots: usize,
    /// `Some(addr)`: device `i32[slots]` expert ids; `None`: slot `s` uses
    /// expert `s` (a dense linear uses `slots = 1`).
    pub expert_ids: Option<CudaDeviceAddress>,
    /// `true` when `x` holds `[slots, K]` rows; `false` when all slots share
    /// one `[K]` row.
    pub per_slot_activation: bool,
    /// Optional device `f32[slots]` multiplier folded into the partial sums.
    pub slot_scale: Option<CudaDeviceAddress>,
}

impl GemvSlots {
    pub fn dense() -> Self {
        Self {
            slots: 1,
            expert_ids: None,
            per_slot_activation: false,
            slot_scale: None,
        }
    }
}

/// Bytes required for the FP32 partial buffer of [`gemv_partial_into`].
pub fn partial_bytes(out_dim: usize, splits: usize, slots: usize) -> usize {
    out_dim * splits * slots * std::mem::size_of::<f32>()
}

/// Split-K W4A16 GEMV. Writes FP32 partial sums `[slots, splits, N]`.
pub fn gemv_partial_into(
    ctx: &CudaContext,
    x: &CudaBuffer,
    weight: AwqWeightView<'_>,
    slots: GemvSlots,
    splits: usize,
    partial: &CudaBuffer,
) -> Result<()> {
    weight.validate(ctx, "W4A16 GEMV")?;
    if splits == 0 || slots.slots == 0 {
        return Err(Error::Other("W4A16 GEMV needs splits >= 1 and slots >= 1".into()));
    }
    if slots.expert_ids.is_none() && slots.slots > weight.experts {
        return Err(Error::Other(format!(
            "W4A16 GEMV: {} slots but only {} experts",
            slots.slots, weight.experts
        )));
    }
    let x_rows = if slots.per_slot_activation { slots.slots } else { 1 };
    require_buffers(
        ctx,
        "W4A16 GEMV",
        &[
            ("x", x, x_rows * weight.in_dim * DType::BF16.size_in_bytes()),
            ("partial", partial, partial_bytes(weight.out_dim, splits, slots.slots)),
        ],
    )?;
    if let Some(ids) = slots.expert_ids {
        super::super::contracts::require_address(ctx, "W4A16 GEMV", "expert_ids", ids, slots.slots * 4)?;
    }
    if let Some(scale) = slots.slot_scale {
        super::super::contracts::require_address(ctx, "W4A16 GEMV", "slot_scale", scale, slots.slots * 4)?;
    }
    unsafe {
        check_cuda(ffi::apxinf_w4a16_gemv_partial_bf16(
            x.ptr(),
            if slots.per_slot_activation { weight.in_dim as i64 } else { 0 },
            weight.qweight.ptr(),
            weight.qzeros.ptr(),
            weight.scales.ptr(),
            slots.expert_ids.map_or(std::ptr::null(), |a| a.ptr() as *const _),
            weight.stride_q as i64,
            weight.stride_z as i64,
            weight.stride_s as i64,
            slots.slot_scale.map_or(std::ptr::null(), |a| a.ptr() as *const _),
            partial.ptr(),
            weight.in_dim as i32,
            weight.packed_cols() as i32,
            weight.group_size as i32,
            splits as i32,
            slots.slots as i32,
            ctx.stream().handle(),
        ))
    }
}

/// `output[n] = sum_{c < count} partial[c][n]` as BF16 `[N]`.
pub fn partial_sum_into(
    ctx: &CudaContext,
    partial: &CudaBuffer,
    count: usize,
    out_dim: usize,
    output: &CudaBuffer,
) -> Result<()> {
    if count == 0 || out_dim == 0 {
        return Err(Error::Other("partial sum needs count >= 1 and N >= 1".into()));
    }
    require_buffers(
        ctx,
        "partial sum",
        &[
            ("partial", partial, partial_bytes(out_dim, count, 1)),
            ("output", output, out_dim * DType::BF16.size_in_bytes()),
        ],
    )?;
    unsafe {
        check_cuda(ffi::apxinf_partial_sum_bf16(
            partial.ptr(),
            output.ptr(),
            out_dim as i32,
            count as i32,
            ctx.stream().handle(),
        ))
    }
}

/// Reduce `[slots, splits, 2*inter]` gate/up partials into BF16
/// `[slots, inter]` SwiGLU activations.
pub fn partial_silu_mul_into(
    ctx: &CudaContext,
    partial: &CudaBuffer,
    slots: usize,
    splits: usize,
    inter: usize,
    output: &CudaBuffer,
) -> Result<()> {
    if slots == 0 || splits == 0 || inter == 0 {
        return Err(Error::Other("partial SwiGLU needs non-zero dimensions".into()));
    }
    require_buffers(
        ctx,
        "partial SwiGLU",
        &[
            ("partial", partial, partial_bytes(2 * inter, splits, slots)),
            ("output", output, slots * inter * DType::BF16.size_in_bytes()),
        ],
    )?;
    unsafe {
        check_cuda(ffi::apxinf_partial_silu_mul_bf16(
            partial.ptr(),
            output.ptr(),
            inter as i32,
            splits as i32,
            slots as i32,
            ctx.stream().handle(),
        ))
    }
}
