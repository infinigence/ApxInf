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

/// Validated host-routed row schedule, shared by gate/up and down projections.
pub struct GroupedRows<'a> {
    tiles: &'a CudaBuffer,
    count: usize,
    rows: usize,
    experts: usize,
}

impl<'a> GroupedRows<'a> {
    pub fn new(ctx: &CudaContext, offsets: &[usize], tiles: &'a CudaBuffer) -> Result<Self> {
        if offsets.len() < 2
            || offsets[0] != 0
            || offsets.windows(2).any(|p| p[0] > p[1])
            || offsets.iter().any(|&x| x > i32::MAX as usize)
        {
            return Err(Error::Other("grouped GEMM: invalid expert offsets".into()));
        }
        let mut values = Vec::<i32>::new();
        for (expert, pair) in offsets.windows(2).enumerate() {
            for row in (pair[0]..pair[1]).step_by(64) {
                values.extend_from_slice(&[expert as i32, row as i32, pair[1] as i32]);
            }
        }
        if values.is_empty() || values.len() / 3 > i32::MAX as usize {
            return Err(Error::Other(
                "grouped GEMM: empty or oversized schedule".into(),
            ));
        }
        require_buffers(
            ctx,
            "grouped GEMM schedule",
            &[("tiles", tiles, values.len() * 4)],
        )?;
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_ne_bytes()).collect();
        tiles.copy_from_host(&bytes).map_err(Error::Cuda)?;
        Ok(Self {
            tiles,
            count: values.len() / 3,
            rows: *offsets.last().unwrap(),
            experts: offsets.len() - 1,
        })
    }
}

/// Grouped BF16 activation × AutoAWQ INT4 weights, without global dequant scratch.
pub fn grouped_bf16_into(
    ctx: &CudaContext,
    x: &CudaBuffer,
    weight: AwqWeightView<'_>,
    schedule: &GroupedRows<'_>,
    output: &CudaBuffer,
) -> Result<()> {
    weight.validate(ctx, "grouped W4A16 GEMM")?;
    if ctx.caps().compute_major < 8 {
        return Err(Error::Other(
            "grouped W4A16 GEMM requires SM80 or newer".into(),
        ));
    }
    if weight.group_size != 128
        || weight.experts != schedule.experts
        || weight.in_dim > i32::MAX as usize
        || weight.out_dim > 65535 * 128
        || weight.group_size > i32::MAX as usize
    {
        return Err(Error::Other("grouped GEMM: unsupported dimensions".into()));
    }
    let bytes = |cols: usize| {
        schedule
            .rows
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| Error::Other("grouped GEMM: byte count overflow".into()))
    };
    require_buffers(
        ctx,
        "grouped W4A16 GEMM",
        &[
            ("x", x, bytes(weight.in_dim)?),
            ("output", output, bytes(weight.out_dim)?),
            ("tiles", schedule.tiles, schedule.count * 12),
        ],
    )?;
    unsafe {
        check_cuda(ffi::apxinf_w4a16_grouped_bf16(
            x.ptr(),
            weight.qweight.ptr(),
            weight.qzeros.ptr(),
            weight.scales.ptr(),
            schedule.tiles.ptr(),
            output.ptr(),
            schedule.count as i32,
            weight.in_dim as i32,
            weight.out_dim as i32,
            weight.group_size as i32,
            weight.stride_q as i64,
            weight.stride_z as i64,
            weight.stride_s as i64,
            ctx.stream().handle(),
        ))
    }
}

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

    pub(super) fn validate(&self, ctx: &CudaContext, operation: &str) -> Result<()> {
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
        &[(
            "output",
            output,
            count * per_expert * DType::BF16.size_in_bytes(),
        )],
    )?;
    let q = weight
        .qweight
        .view(
            first * weight.stride_q * 4,
            weight.qweight.len() - first * weight.stride_q * 4,
        )
        .map_err(Error::Cuda)?;
    let z = weight
        .qzeros
        .view(
            first * weight.stride_z * 4,
            weight.qzeros.len() - first * weight.stride_z * 4,
        )
        .map_err(Error::Cuda)?;
    let s = weight
        .scales
        .view(
            first * weight.stride_s * 2,
            weight.scales.len() - first * weight.stride_s * 2,
        )
        .map_err(Error::Cuda)?;
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

/// Load-time qweight copy in `[expert, N/256, K/4, 32, 4]` word order.
/// Scales and zeros retain their checkpoint format. The original source must
/// remain alive and unchanged when this copy is used.
pub struct BlockedWeights {
    q: CudaBuffer,
    source_address: usize,
    k: usize,
    n: usize,
    experts: usize,
}
impl BlockedWeights {
    fn validate_source(ctx: &CudaContext, source: AwqWeightView<'_>) -> Result<usize> {
        source.validate(ctx, "blocked AWQ repack")?;
        if source.group_size != 128
            || source.out_dim % 256 != 0
            || source.in_dim > i32::MAX as usize
            || source.out_dim > i32::MAX as usize
            || source.experts > i32::MAX as usize
            || (source.experts > 1 && source.stride_q != source.qweight_words())
        {
            return Err(Error::Other(
                "blocked AWQ: requires contiguous group-128 weights and N divisible by 256".into(),
            ));
        }
        source
            .qweight_words()
            .checked_mul(source.experts)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| Error::Other("blocked AWQ size overflow".into()))
    }

    pub fn repack(ctx: &CudaContext, source: AwqWeightView<'_>, mapped: bool) -> Result<Self> {
        let bytes = Self::validate_source(ctx, source)?;
        let q = if mapped {
            CudaBuffer::alloc_mapped(bytes, ctx.device_id())
        } else {
            CudaBuffer::alloc(bytes, ctx.device_id())
        }
        .map_err(Error::Cuda)?;
        check_cuda(unsafe {
            ffi::apxinf_w4a16_blocked_repack(
                source.qweight.ptr(),
                q.ptr(),
                source.in_dim as i32,
                source.packed_cols() as i32,
                source.experts as i32,
                ctx.stream().handle(),
            )
        })?;
        Ok(Self {
            q,
            source_address: source.qweight.ptr() as usize,
            k: source.in_dim,
            n: source.out_dim,
            experts: source.experts,
        })
    }

    /// Move checkpoint-format words to mapped storage and reuse their original
    /// allocation for decode. This overwrites `source.qweight`: callers must
    /// replace that field with the returned backup before any further AWQ use.
    /// Intended only for exclusive model construction before weights are exposed.
    /// Work is enqueued on `ctx`'s stream; retain both returned buffers until it
    /// completes. Decode keeps the loader's original device/mapped placement.
    pub fn repack_reusing_storage(
        ctx: &CudaContext,
        source: AwqWeightView<'_>,
    ) -> Result<(Self, CudaBuffer)> {
        let bytes = Self::validate_source(ctx, source)?;
        let backup = CudaBuffer::alloc_mapped(bytes, ctx.device_id()).map_err(Error::Cuda)?;
        unsafe {
            check_cuda(ffi::cudaMemcpyAsync(
                backup.ptr(),
                source.qweight.ptr(),
                bytes,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                ctx.stream().handle(),
            ))?;
            check_cuda(ffi::apxinf_w4a16_blocked_repack(
                backup.ptr(),
                source.qweight.ptr(),
                source.in_dim as i32,
                source.packed_cols() as i32,
                source.experts as i32,
                ctx.stream().handle(),
            ))?;
        }
        let repacked = Self {
            q: source.qweight.clone(),
            source_address: backup.ptr() as usize,
            k: source.in_dim,
            n: source.out_dim,
            experts: source.experts,
        };
        Ok((repacked, backup))
    }
}

enum NibbleConversion {
    Default,
    Magic,
    Pair,
}

/// Select a repacked GEMV without changing its partial reduction order.
pub fn gemv_partial_repacked_into(
    ctx: &CudaContext,
    x: &CudaBuffer,
    weight: AwqWeightView<'_>,
    blocked: Option<&BlockedWeights>,
    slots: GemvSlots,
    splits: usize,
    partial: &CudaBuffer,
) -> Result<()> {
    gemv_partial_impl(
        ctx,
        x,
        weight,
        blocked,
        if std::env::var("APXINF_QWEN3MOE_GEMV_PAIR").as_deref() == Ok("1") {
            NibbleConversion::Pair
        } else if std::env::var("APXINF_QWEN3MOE_GEMV_MAGIC").as_deref() == Ok("1") {
            NibbleConversion::Magic
        } else {
            NibbleConversion::Default
        },
        slots,
        splits,
        partial,
    )
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
    gemv_partial_impl(
        ctx,
        x,
        weight,
        None,
        NibbleConversion::Default,
        slots,
        splits,
        partial,
    )
}

fn gemv_partial_impl(
    ctx: &CudaContext,
    x: &CudaBuffer,
    weight: AwqWeightView<'_>,
    blocked: Option<&BlockedWeights>,
    conversion: NibbleConversion,
    slots: GemvSlots,
    splits: usize,
    partial: &CudaBuffer,
) -> Result<()> {
    weight.validate(ctx, "W4A16 GEMV")?;
    if let Some(copy) = blocked {
        if copy.source_address != weight.qweight.ptr() as usize
            || copy.k != weight.in_dim
            || copy.n != weight.out_dim
            || copy.experts != weight.experts
            || weight.group_size != 128
        {
            return Err(Error::Other("blocked AWQ source mismatch".into()));
        }
        require_buffers(
            ctx,
            "blocked AWQ",
            &[(
                "qweight",
                &copy.q,
                weight.qweight_words() * weight.experts * 4,
            )],
        )?;
    }
    if splits == 0 || slots.slots == 0 {
        return Err(Error::Other(
            "W4A16 GEMV needs splits >= 1 and slots >= 1".into(),
        ));
    }
    if slots.expert_ids.is_none() && slots.slots > weight.experts {
        return Err(Error::Other(format!(
            "W4A16 GEMV: {} slots but only {} experts",
            slots.slots, weight.experts
        )));
    }
    let x_rows = if slots.per_slot_activation {
        slots.slots
    } else {
        1
    };
    require_buffers(
        ctx,
        "W4A16 GEMV",
        &[
            ("x", x, x_rows * weight.in_dim * DType::BF16.size_in_bytes()),
            (
                "partial",
                partial,
                partial_bytes(weight.out_dim, splits, slots.slots),
            ),
        ],
    )?;
    if let Some(ids) = slots.expert_ids {
        super::super::contracts::require_address(
            ctx,
            "W4A16 GEMV",
            "expert_ids",
            ids,
            slots.slots * 4,
        )?;
    }
    if let Some(scale) = slots.slot_scale {
        super::super::contracts::require_address(
            ctx,
            "W4A16 GEMV",
            "slot_scale",
            scale,
            slots.slots * 4,
        )?;
    }
    unsafe {
        let launch = match (blocked.is_some(), conversion) {
            (true, NibbleConversion::Pair) => ffi::apxinf_w4a16_gemv_pair_blocked_partial_bf16,
            (false, NibbleConversion::Pair) => {
                return Err(Error::Other("GEMV_PAIR requires blocked weights".into()))
            }
            (true, NibbleConversion::Magic) => ffi::apxinf_w4a16_gemv_magic_blocked_partial_bf16,
            (false, NibbleConversion::Magic) => ffi::apxinf_w4a16_gemv_magic_partial_bf16,
            (true, NibbleConversion::Default) => ffi::apxinf_w4a16_gemv_blocked_partial_bf16,
            (false, NibbleConversion::Default) => ffi::apxinf_w4a16_gemv_partial_bf16,
        };
        check_cuda(launch(
            x.ptr(),
            if slots.per_slot_activation {
                weight.in_dim as i64
            } else {
                0
            },
            blocked.map_or(weight.qweight.ptr(), |copy| copy.q.ptr()),
            weight.qzeros.ptr(),
            weight.scales.ptr(),
            slots
                .expert_ids
                .map_or(std::ptr::null(), |a| a.ptr() as *const _),
            weight.stride_q as i64,
            weight.stride_z as i64,
            weight.stride_s as i64,
            slots
                .slot_scale
                .map_or(std::ptr::null(), |a| a.ptr() as *const _),
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
        return Err(Error::Other(
            "partial sum needs count >= 1 and N >= 1".into(),
        ));
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
        return Err(Error::Other(
            "partial SwiGLU needs non-zero dimensions".into(),
        ));
    }
    require_buffers(
        ctx,
        "partial SwiGLU",
        &[
            ("partial", partial, partial_bytes(2 * inter, splits, slots)),
            (
                "output",
                output,
                slots * inter * DType::BF16.size_in_bytes(),
            ),
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

/// Split/slot reduction followed by residual add and RMSNorm, one decode row.
///
/// See [`partial_residual_rms_rows_into`] for several independent token rows.
#[allow(clippy::too_many_arguments)]
pub fn partial_residual_rms_into(
    ctx: &CudaContext,
    x: &CudaBuffer,
    partial: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    cols: usize,
    count: usize,
    eps: f32,
) -> Result<()> {
    if cols == 0
        || cols > 8192
        || count == 0
        || count > i32::MAX as usize
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other(
            "partial residual RMSNorm: invalid geometry or epsilon".into(),
        ));
    }
    let size = count
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Other("partial residual RMSNorm size overflow".into()))?;
    require_buffers(
        ctx,
        "partial residual RMSNorm",
        &[
            ("x", x, cols * 2),
            ("partial", partial, size),
            ("weight", weight, cols * 2),
            ("output", output, cols * 2),
        ],
    )?;
    check_cuda(unsafe {
        ffi::apxinf_partial_residual_rms_bf16(
            x.ptr(),
            partial.ptr(),
            weight.ptr(),
            output.ptr(),
            cols as i32,
            count as i32,
            eps,
            ctx.stream().handle(),
        )
    })
}

/// Reduce `[rows, count, cols]` partials before residual add and RMSNorm.
/// Every row follows the same FP32 sum and BF16 boundary as the decode path.
#[allow(clippy::too_many_arguments)]
pub fn partial_residual_rms_rows_into(
    ctx: &CudaContext,
    x: &CudaBuffer,
    partial: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    cols: usize,
    rows: usize,
    count: usize,
    eps: f32,
) -> Result<()> {
    if rows == 1 {
        return partial_residual_rms_into(ctx, x, partial, weight, output, cols, count, eps);
    }
    let elements = rows
        .checked_mul(cols)
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or_else(|| Error::Other("partial residual RMSNorm row size overflow".into()))?;
    if rows == 0
        || rows > i32::MAX as usize
        || cols == 0
        || cols > 8192
        || count == 0
        || count > i32::MAX as usize
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other(
            "partial residual RMSNorm: invalid row geometry or epsilon".into(),
        ));
    }
    let partial_bytes = elements
        .checked_mul(count)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Other("partial residual RMSNorm partial size overflow".into()))?;
    require_buffers(
        ctx,
        "partial residual RMSNorm rows",
        &[
            ("x", x, elements * 2),
            ("partial", partial, partial_bytes),
            ("weight", weight, cols * 2),
            ("output", output, elements * 2),
        ],
    )?;
    check_cuda(unsafe {
        ffi::apxinf_partial_residual_rms_rows_bf16(
            x.ptr(),
            partial.ptr(),
            weight.ptr(),
            output.ptr(),
            cols as i32,
            rows as i32,
            count as i32,
            eps,
            ctx.stream().handle(),
        )
    })
}

/// BF16 activations with checkpoint FP16 RMSNorm weights.
#[allow(clippy::too_many_arguments)]
pub fn partial_residual_rms_f16_weight_into(
    ctx: &CudaContext,
    x: &CudaBuffer,
    partial: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    cols: usize,
    count: usize,
    eps: f32,
) -> Result<()> {
    if cols == 0
        || cols > 8192
        || count == 0
        || count > i32::MAX as usize
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other(
            "partial residual RMSNorm: invalid geometry or epsilon".into(),
        ));
    }
    let size = count
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Other("partial residual RMSNorm size overflow".into()))?;
    require_buffers(
        ctx,
        "partial residual RMSNorm",
        &[
            ("x", x, cols * 2),
            ("partial", partial, size),
            ("weight", weight, cols * 2),
            ("output", output, cols * 2),
        ],
    )?;
    check_cuda(unsafe {
        ffi::apxinf_partial_residual_rms_bf16_f16_weight(
            x.ptr(),
            partial.ptr(),
            weight.ptr(),
            output.ptr(),
            cols as i32,
            count as i32,
            eps,
            ctx.stream().handle(),
        )
    })
}

/// BF16 activations with checkpoint FP16 RMSNorm weights.
#[allow(clippy::too_many_arguments)]
pub fn partial_residual_rms_rows_f16_weight_into(
    ctx: &CudaContext,
    x: &CudaBuffer,
    partial: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    cols: usize,
    rows: usize,
    count: usize,
    eps: f32,
) -> Result<()> {
    if rows == 1 {
        return partial_residual_rms_f16_weight_into(
            ctx, x, partial, weight, output, cols, count, eps,
        );
    }
    let elements = rows
        .checked_mul(cols)
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or_else(|| Error::Other("partial residual RMSNorm row size overflow".into()))?;
    if rows == 0
        || rows > i32::MAX as usize
        || cols == 0
        || cols > 8192
        || count == 0
        || count > i32::MAX as usize
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other(
            "partial residual RMSNorm: invalid row geometry or epsilon".into(),
        ));
    }
    let partial_bytes = elements
        .checked_mul(count)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Other("partial residual RMSNorm partial size overflow".into()))?;
    require_buffers(
        ctx,
        "partial residual RMSNorm rows",
        &[
            ("x", x, elements * 2),
            ("partial", partial, partial_bytes),
            ("weight", weight, cols * 2),
            ("output", output, elements * 2),
        ],
    )?;
    check_cuda(unsafe {
        ffi::apxinf_partial_residual_rms_rows_bf16_f16_weight(
            x.ptr(),
            partial.ptr(),
            weight.ptr(),
            output.ptr(),
            cols as i32,
            rows as i32,
            count as i32,
            eps,
            ctx.stream().handle(),
        )
    })
}
