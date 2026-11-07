//! W8A8 (INT8) operator contracts.
//!
//! Native: the rendered `s8mma` GEMM tiles in `kernels/asm/qint8_*.asm`, driven
//! through `adapters/qint8_gemm_adapter.mu`, together with the row quantizer and
//! the widening pass in `kernels/custom/quantization.cuh`. The operand contract
//! is the CUDA backend's: an activation quantized per row, output-major INT8
//! weights carrying one f32 dequantization scale per output channel, and a BF16
//! result.
//!
//! One structural difference follows from the MUSA templates, documented at
//! the kernels that carry it:
//!
//! * the tiles read their scales as **FP16**, so the quantizer writes the
//!   per-row activation scale half-precision and the weights' f32 column
//!   scales are converted **once at pack time**: the model stores an FP16
//!   copy with [`SCALE_PAD`] trailing zeros and [`W8A8WeightView::scales_f16`]
//!   borrows it, so no conversion kernel runs on the GEMM path. Both scale
//!   vectors carry the 512-entry zero pad the guarded scale load may read.
//!
//! The tiles write the BF16 result directly: the vendored `s8mma` templates
//! carry a local patch that narrows the f32 accumulator to BF16 in the
//! epilogue, so the widening pass the upstream FP16-only templates needed is
//! gone. That changes the rounding from two steps (f32->f16 in the tile, then
//! f16->bf16 in the widening pass) to one (f32->bf16 in the tile) -- closer to
//! the exact product, not bitwise-identical to the old chain.
//!
//! Only a shape whose `n` is a positive multiple of a tile width (64, 128 or
//! 256) and whose `k` is a positive multiple of 64 has a tile. A shape outside
//! that
//! -- the ViT's `k = 588` and `n = 4304`, for example -- reports the gap rather
//! than running a different arithmetic. The model may instead pad such a shape
//! to a tile multiple at pack time (the MUSA INT8 weights pad `4304` to
//! `4352`); that is a model-side choice made before the operator is called, not
//! part of this contract. The fused-quantization entry points that
//! would need operators this port does not have stay [`not_ported`].
//!
//! Signatures mirror `apxinf-cuda/src/kernels/gemm/w8a8.rs`, including the
//! private fields on [`W8A8Activation`]: a caller cannot fabricate one, which
//! is what keeps the quantized buffer, its row scales and its dimensions
//! agreeing with each other.

use apxinf_core::{Result, Tensor};
#[cfg(apxinf_musa_available)]
use apxinf_core::{DType, Device, Error, Shape};
#[cfg(apxinf_musa_available)]
use std::ffi::c_void;

use crate::buffer::MusaBuffer;
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::error::musa_error;
#[cfg(apxinf_musa_available)]
use crate::ffi;
#[cfg(apxinf_musa_available)]
use crate::kernels::check_native;
#[cfg(apxinf_musa_available)]
use crate::kernels::contracts::checked_i32;
#[cfg(not(apxinf_musa_available))]
use crate::kernels::native_unavailable;
use crate::kernels::not_ported;

thread_local! {
    static DECODE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Marks quantized decode without changing BF16 fallback dispatch.
///
/// The decode images below serve short rows, and only the action traversal has
/// them; marking the traversal is what lets the dispatcher pick them without
/// the model naming a kernel. A nested scope restores on drop, including
/// through an unwind, and cannot cross threads. In particular an INT8 model's
/// BF16 action heads have no assembly registration, so this must not be
/// confused with the BF16 decode scope — hence a counter of its own.
#[must_use]
pub struct W8A8DecodeScope {
    _scope: super::ThreadScope,
}

#[cfg(test)]
mod scope_tests {
    use super::*;

    #[test]
    fn decode_scope_is_nested_thread_local_and_independent_of_bf16() {
        assert_eq!(DECODE_DEPTH.with(|depth| depth.get()), 0);
        {
            let _outer = decode_scope();
            assert!(!super::super::in_decode_scope());
            let _ = std::panic::catch_unwind(|| {
                let _inner = decode_scope();
                assert_eq!(DECODE_DEPTH.with(|depth| depth.get()), 2);
                std::thread::spawn(|| assert_eq!(DECODE_DEPTH.with(|depth| depth.get()), 0))
                    .join()
                    .unwrap();
                panic!("scope unwind probe");
            });
            assert_eq!(DECODE_DEPTH.with(|depth| depth.get()), 1);
        }
        assert_eq!(DECODE_DEPTH.with(|depth| depth.get()), 0);
    }
}

pub fn decode_scope() -> W8A8DecodeScope {
    W8A8DecodeScope {
        _scope: super::ThreadScope::enter(&DECODE_DEPTH),
    }
}

#[cfg(apxinf_musa_available)]
fn use_decode_tiles(rows: usize, n: usize, k: usize) -> bool {
    DECODE_DEPTH.with(|depth| depth.get() != 0)
        && decode_tiles_supported()
        && rows > 0
        && rows <= 32
        && n > 0
        && n % 128 == 0
        && k > 0
        && k % 64 == 0
}

/// Whether this build targets the architecture supported by the NN/NT decode images.
pub fn decode_tiles_supported() -> bool {
    crate::MUSA_ARCH == crate::ASSEMBLY_ARCH
}

/// How the activation's scales are produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W8A8ScaleMode {
    DynamicRowPerOutputChannel,
}

/// Physical weight layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W8A8Layout {
    OutputMajor,
}

/// Borrowed W8A8 weight view: keeps values, scales and layout in one value so
/// they cannot be mismatched at a call site.
#[derive(Clone, Copy)]
pub struct W8A8WeightView<'a> {
    pub values_i8: &'a MusaBuffer,
    /// The output-channel dequantization scales as the tile reads them: FP16,
    /// with [`SCALE_PAD`] trailing zeros. The model renders its f32 scales to
    /// this form once at pack time; the f32 -> FP16 rounding is
    /// round-to-nearest-even on either side, so the packed copy is
    /// bitwise-identical to converting the scales per GEMM.
    pub scales_f16: &'a MusaBuffer,
    pub input_dim: usize,
    pub output_dim: usize,
    pub scale_mode: W8A8ScaleMode,
    pub layout: W8A8Layout,
    /// Per-channel equalization factors the activation is multiplied by while
    /// it is quantized: `input_dim` f32, `1/c` in the SmoothQuant identity
    /// `x @ W^T == (x ./ c) @ (W .* c)^T`, paired with a weight whose
    /// contraction rows were folded by `c` at pack time. Fusing the multiply
    /// into the quantizer costs one cached vector load per lane instead of a
    /// separate elementwise pass. `None` quantizes the activation as-is.
    pub equalization_inv_c: Option<&'a Tensor>,
    /// KN copy of [`Self::values_i8`] for short-row decode: the decode image
    /// reads gate/up weights contiguous in the input dimension, and packing
    /// that copy once at load is what keeps a transpose off the GEMM path. The
    /// NK original stays for encode and for row counts the decode images do not
    /// serve.
    pub decode_nn_values_i8: Option<&'a MusaBuffer>,
}

impl<'a> W8A8WeightView<'a> {
    /// Build the view. `scales_f32` is the cross-backend contract's f32 scale
    /// vector: CUDA consumes it, while this backend's tiles read FP16 scales
    /// and consume `scales_f16`. It is part of the signature so model code
    /// constructs the view identically on either backend.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        values_i8: &'a MusaBuffer,
        _scales_f32: &'a Tensor,
        scales_f16: &'a MusaBuffer,
        input_dim: usize,
        output_dim: usize,
        scale_mode: W8A8ScaleMode,
        layout: W8A8Layout,
        equalization_inv_c: Option<&'a Tensor>,
    ) -> Self {
        Self {
            values_i8,
            scales_f16,
            input_dim,
            output_dim,
            scale_mode,
            layout,
            equalization_inv_c,
            decode_nn_values_i8: None,
        }
    }

    /// Offer the KN copy the short-row decode image reads.
    pub fn with_decode_nn_weight(mut self, values: Option<&'a MusaBuffer>) -> Self {
        self.decode_nn_values_i8 = values;
        self
    }
}

/// A dynamically row-quantized activation, shareable by projections that
/// consume the same BF16 input (a transformer's gate and up, for example).
///
/// The fields are private so that only this module can build the triple and the
/// quantized buffer, its scales and its dimensions cannot disagree. `row_scales`
/// is **FP16**, not f32 as on CUDA, because the `s8mma` tile reads it that way,
/// and it carries [`SCALE_PAD`] trailing zeros; `quantized` is `rows *
/// input_dim` INT8.
#[cfg_attr(not(apxinf_musa_available), allow(dead_code))]
pub struct W8A8Activation {
    quantized: MusaBuffer,
    row_scales: MusaBuffer,
    rows: usize,
    input_dim: usize,
}

/// The FP16 zero pad both scale vectors carry past their length.
///
/// Must match `kScalePad` in `adapters/qint8_gemm_adapter.mu` and
/// `kW8A8ScalePad` in `kernels/custom/quantization.cuh`. Public and
/// toolkit-independent because the model's packer sizes the FP16 weight-scale
/// copy by it (`musa_pi05/weights/int8_dynamic.rs`).
pub const SCALE_PAD: usize = 512;

/// Bytes of an FP16 scale vector holding `length` scales plus the pad.
#[cfg(apxinf_musa_available)]
fn padded_scale_bytes(length: usize, operation: &str) -> Result<usize> {
    length
        .checked_add(SCALE_PAD)
        .and_then(|entries| entries.checked_mul(2))
        .ok_or_else(|| Error::Other(format!("MUSA {operation} scale vector size overflow")))
}

/// Require a non-empty 2-D BF16 activation on this context's device.
#[cfg(apxinf_musa_available)]
fn activation_matrix(
    ctx: &MusaContext,
    activation: &Tensor,
    operation: &str,
) -> Result<(usize, usize)> {
    if activation.dtype() != DType::BF16 || activation.device() != Device::Musa(ctx.device_id()) {
        return Err(Error::Other(format!(
            "MUSA {operation} expects a BF16 activation on {}, got {} on {}",
            Device::Musa(ctx.device_id()),
            activation.dtype(),
            activation.device()
        )));
    }
    let dims = activation.shape().dims();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(Error::Other(format!(
            "MUSA {operation} expects a non-empty 2-D activation, got {dims:?}"
        )));
    }
    Ok((dims[0], dims[1]))
}

/// INT8 GEMM over a BF16 activation, quantizing the activation on the way.
pub fn gemm_w8a8(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: W8A8WeightView<'_>,
) -> Result<Tensor> {
    let quantized = quantize_w8a8_activation(ctx, activation, weight.equalization_inv_c)?;
    gemm_quantized_w8a8(ctx, &quantized, weight)
}

/// The equalization factors as a raw pointer, or the null pointer when this
/// projection carries no fold.
///
/// Both quantizers take the same optional `1/c` vector under the same
/// contract, so the check lives here rather than in two near-copies whose
/// messages could drift apart.
#[cfg(apxinf_musa_available)]
fn equalization_factors_ptr(
    ctx: &MusaContext,
    operation: &str,
    inv_c: Option<&Tensor>,
    input_dim: usize,
) -> Result<*const c_void> {
    let Some(factors) = inv_c else {
        return Ok(std::ptr::null());
    };
    if factors.dtype() != DType::F32
        || factors.device() != Device::Musa(ctx.device_id())
        || factors.shape().dims() != [input_dim]
    {
        return Err(Error::Other(format!(
            "MUSA {operation} expects the equalization factors to be {} f32 on {}, got {} {:?} on {}",
            input_dim,
            Device::Musa(ctx.device_id()),
            factors.dtype(),
            factors.shape().dims(),
            factors.device()
        )));
    }
    Ok(MusaBuffer::from_tensor(factors).map_err(musa_error)?.ptr() as *const c_void)
}

/// Row-quantize a BF16 activation to INT8.
///
/// `inv_c` is the activation half of a per-channel equalization fold:
/// `input_dim` f32 factors the row is multiplied by before its amax is
/// measured, or `None` to quantize as-is.
#[cfg(apxinf_musa_available)]
pub fn quantize_w8a8_activation(
    ctx: &MusaContext,
    activation: &Tensor,
    inv_c: Option<&Tensor>,
) -> Result<W8A8Activation> {
    let operation = "gemm::quantize_w8a8_activation";
    let (rows, input_dim) = activation_matrix(ctx, activation, operation)?;
    let source = MusaBuffer::from_tensor(activation).map_err(musa_error)?;
    let factors = equalization_factors_ptr(ctx, operation, inv_c, input_dim)?;
    let elements = rows
        .checked_mul(input_dim)
        .ok_or_else(|| Error::Other(format!("MUSA {operation} size overflow")))?;
    let quantized = crate::workspace::output_buffer(ctx, elements)?;
    let row_scales = crate::workspace::output_buffer(ctx, padded_scale_bytes(rows, operation)?)?;
    let rows_i32 = checked_i32(operation, "rows", rows)?;
    let cols_i32 = checked_i32(operation, "cols", input_dim)?;
    // SAFETY: `activation_matrix` established that the activation is a
    // contiguous 2-D BF16 tensor on this context's device, so `source` borrows
    // `rows * input_dim` readable bytes; `factors` is null or borrows the
    // validated `input_dim` f32 vector; `quantized` and `row_scales` are
    // allocations on the same device sized for the INT8 values and the padded
    // FP16 scales the kernel writes. The launch is asynchronous on
    // `ctx.stream()`, and every buffer it touches outlives the call.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_quantize_rows_bf16_int8(
                source.ptr() as *const c_void,
                quantized.ptr(),
                row_scales.ptr(),
                rows_i32,
                cols_i32,
                factors,
                ctx.stream().handle(),
            ),
            operation,
        )?;
    }
    Ok(W8A8Activation {
        quantized,
        row_scales,
        rows,
        input_dim,
    })
}

/// No toolkit: report the gap, do not fall back.
#[cfg(not(apxinf_musa_available))]
pub fn quantize_w8a8_activation(
    _ctx: &MusaContext,
    _activation: &Tensor,
    _inv_c: Option<&Tensor>,
) -> Result<W8A8Activation> {
    Err(native_unavailable("gemm::quantize_w8a8_activation"))
}

/// GeGLU followed by row quantization, preserving BF16 intermediate rounding.
///
/// The down projection consumes `geglu(gate_up)` and then quantizes it; running
/// the activation and the quantizer as one kernel keeps the two-pass version's
/// arithmetic exactly (the product is rounded back to BF16 in registers before
/// its row is measured) while dropping the intermediate matrix. Widths or
/// alignments the fused kernel does not take keep the materialized path, which
/// is why this returns the same type either way.
#[cfg(apxinf_musa_available)]
pub fn quantize_w8a8_geglu_activation(
    ctx: &MusaContext,
    activation: &Tensor,
    inv_c: Option<&Tensor>,
) -> Result<W8A8Activation> {
    let operation = "gemm::quantize_w8a8_geglu_activation";
    let (rows, packed_dim) = activation_matrix(ctx, activation, operation)?;
    if packed_dim % 2 != 0 {
        return Err(Error::Other(format!(
            "MUSA {operation} requires an even gate/up width"
        )));
    }
    let input_dim = packed_dim / 2;
    let source = MusaBuffer::from_tensor(activation).map_err(musa_error)?;
    let factors = equalization_factors_ptr(ctx, operation, inv_c, input_dim)?;
    if input_dim % 8 != 0
        || input_dim > 16384
        || source.ptr() as usize % 16 != 0
        || factors as usize % 16 != 0
    {
        let activated = crate::kernels::activation::geglu_bf16(ctx, activation)?;
        return quantize_w8a8_activation(ctx, &activated, inv_c);
    }
    let elements = rows
        .checked_mul(input_dim)
        .ok_or_else(|| Error::Other(format!("MUSA {operation} size overflow")))?;
    let quantized = crate::workspace::output_buffer(ctx, elements)?;
    let row_scales = crate::workspace::output_buffer(ctx, padded_scale_bytes(rows, operation)?)?;
    let rows_i32 = checked_i32(operation, "rows", rows)?;
    let cols_i32 = checked_i32(operation, "cols", input_dim)?;
    // SAFETY: `activation_matrix` established that the activation is a
    // contiguous 2-D BF16 tensor on this context's device, so `source` borrows
    // `rows * 2 * input_dim` readable BF16 elements; `factors` is null or borrows
    // the validated `input_dim` f32 vector; `quantized` and `row_scales` are
    // allocations on the same device sized for the INT8 values and the padded
    // FP16 scales the kernel writes. The launch is asynchronous on
    // `ctx.stream()`, and every buffer it touches outlives the call.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_geglu_quantize_rows_bf16_int8(
                source.ptr() as *const c_void,
                quantized.ptr(),
                row_scales.ptr(),
                rows_i32,
                cols_i32,
                factors,
                ctx.stream().handle(),
            ),
            operation,
        )?;
    }
    Ok(W8A8Activation {
        quantized,
        row_scales,
        rows,
        input_dim,
    })
}

/// No toolkit: report the gap, do not fall back.
#[cfg(not(apxinf_musa_available))]
pub fn quantize_w8a8_geglu_activation(
    _ctx: &MusaContext,
    _activation: &Tensor,
    _inv_c: Option<&Tensor>,
) -> Result<W8A8Activation> {
    Err(native_unavailable("gemm::quantize_w8a8_geglu_activation"))
}

/// INT8 GEMM over an already-quantized activation.
#[cfg(apxinf_musa_available)]
pub fn gemm_quantized_w8a8(
    ctx: &MusaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
) -> Result<Tensor> {
    let plan = plan_quantized(ctx, activation, weight)?;
    let output = crate::workspace::output_buffer(ctx, plan.output_bytes()?)?;
    launch_quantized(ctx, &plan, activation, weight, output.ptr())?;
    Ok(output.into_tensor(Shape::new(vec![plan.rows, plan.n]), DType::BF16))
}

/// INT8 GEMM into a caller-provided BF16 `[rows, output_dim]` output.
///
/// The allocating entry point above is for a caller that owns the result's
/// lifetime. The typed operator contract hands every operator a caller-owned
/// `&mut Tensor` instead, because a captured graph keeps the addresses it baked
/// and the arena owns them for the traversal.
#[cfg(apxinf_musa_available)]
pub fn gemm_quantized_w8a8_into(
    ctx: &MusaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
    output: &mut Tensor,
) -> Result<()> {
    let plan = plan_quantized(ctx, activation, weight)?;
    super::super::contracts::require_output_matrix(
        ctx,
        output,
        plan.rows,
        plan.n,
        "gemm::w8a8",
        "out",
    )?;
    let target = MusaBuffer::from_tensor(output).map_err(musa_error)?;
    launch_quantized(ctx, &plan, activation, weight, target.ptr())
}

/// What operand validation established, in the form the launch needs.
#[cfg(apxinf_musa_available)]
struct QuantizedPlan {
    rows: usize,
    n: usize,
    k: usize,
    tile: u8,
    /// Whether this call runs the short-row decode images rather than the
    /// convolution tiles.
    decode: bool,
    a_scale_bytes: u32,
    b_scale_bytes: u32,
}

#[cfg(apxinf_musa_available)]
impl QuantizedPlan {
    fn output_bytes(&self) -> Result<usize> {
        self.rows
            .checked_mul(self.n)
            .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("MUSA gemm::w8a8 output size overflow".into()))
    }
}

#[cfg(apxinf_musa_available)]
fn plan_quantized(
    ctx: &MusaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
) -> Result<QuantizedPlan> {
    let operation = "gemm::w8a8";
    if weight.scale_mode != W8A8ScaleMode::DynamicRowPerOutputChannel
        || weight.layout != W8A8Layout::OutputMajor
    {
        return Err(Error::Other(
            "MUSA gemm::w8a8 received an unsupported scale mode or layout".into(),
        ));
    }
    if activation.input_dim != weight.input_dim {
        return Err(Error::Other(format!(
            "MUSA gemm::w8a8 activation width mismatch: expected {}, got {}",
            weight.input_dim, activation.input_dim
        )));
    }
    let device = ctx.device_id();
    let (rows, k, n) = (activation.rows, weight.input_dim, weight.output_dim);
    let activation_bytes = rows
        .checked_mul(k)
        .ok_or_else(|| Error::Other(format!("MUSA {operation} activation size overflow")))?;
    if activation.quantized.device() != device || activation.quantized.len() != activation_bytes {
        return Err(Error::Other(format!(
            "MUSA {operation} activation buffer is {} bytes on MUSA{}, expected {activation_bytes} on MUSA{device}",
            activation.quantized.len(),
            activation.quantized.device()
        )));
    }
    let row_scale_bytes = padded_scale_bytes(rows, operation)?;
    if activation.row_scales.device() != device || activation.row_scales.len() < row_scale_bytes {
        return Err(Error::Other(format!(
            "MUSA {operation} row scales are {} bytes on MUSA{}, expected at least {row_scale_bytes} on MUSA{device}",
            activation.row_scales.len(),
            activation.row_scales.device()
        )));
    }
    let weight_bytes = k
        .checked_mul(n)
        .ok_or_else(|| Error::Other(format!("MUSA {operation} weight size overflow")))?;
    if weight.values_i8.device() != device || weight.values_i8.len() != weight_bytes {
        return Err(Error::Other(format!(
            "MUSA {operation} weight buffer is {} bytes on MUSA{}, expected {weight_bytes} on MUSA{device}",
            weight.values_i8.len(),
            weight.values_i8.device()
        )));
    }
    let weight_scale_bytes = padded_scale_bytes(n, operation)?;
    if weight.scales_f16.device() != device || weight.scales_f16.len() < weight_scale_bytes {
        return Err(Error::Other(format!(
            "MUSA {operation} weight scales must be at least {weight_scale_bytes} bytes (FP16 [{n}] plus pad) on MUSA{device}, got {} on MUSA{}",
            weight.scales_f16.len(),
            weight.scales_f16.device()
        )));
    }
    let tile = super::qint8_tiles::select(rows, n, k).ok_or_else(|| {
        Error::Other(format!(
            "MUSA {operation} has no INT8 tile for [{rows}x{k}] @ [{k}x{n}]: the s8mma tiles mask \
             a partial row tile but neither a partial column tile nor a partial contraction, so n \
             must be a positive multiple of a tile width (64, 128 or 256) and k a positive multiple of 64"
        ))
    })?;

    let decode = use_decode_tiles(rows, n, k);
    if decode {
        if let Some(values) = weight.decode_nn_values_i8 {
            if values.device() != device || values.len() != weight_bytes {
                return Err(Error::Other(format!(
                    "MUSA {operation} NN weight must be {weight_bytes} bytes on MUSA{device}"
                )));
            }
        }
    }
    let a_scale_bytes = u32::try_from(row_scale_bytes).map_err(|_| {
        Error::Other(format!(
            "MUSA {operation} activation scale size {row_scale_bytes} does not fit in u32"
        ))
    })?;
    let b_scale_bytes = u32::try_from(weight_scale_bytes).map_err(|_| {
        Error::Other(format!(
            "MUSA {operation} weight scale size does not fit in u32"
        ))
    })?;
    Ok(QuantizedPlan {
        rows,
        n,
        k,
        tile,
        decode,
        a_scale_bytes,
        b_scale_bytes,
    })
}

#[cfg(apxinf_musa_available)]
fn launch_quantized(
    ctx: &MusaContext,
    plan: &QuantizedPlan,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
    output: *mut c_void,
) -> Result<()> {
    let operation = "gemm::w8a8";
    // A captured graph may only replay shapes the prepare traversal already
    // ran, and these tiles have no descriptor state of their own to prepare;
    // recording them keeps the session's shape ledger describing the workload.
    crate::workspace::note_shape(crate::workspace::OperationShape::Gemm {
        m: plan.rows,
        n: plan.n,
        k: plan.k,
    })?;
    let m_i32 = checked_i32(operation, "rows", plan.rows)?;
    let n_i32 = checked_i32(operation, "output_dim", plan.n)?;
    let k_i32 = checked_i32(operation, "input_dim", plan.k)?;

    // SAFETY: `plan_quantized` established that every operand is a live
    // allocation on this context's device with the exact byte length its role
    // needs: the INT8 activation `[rows,k]` and its padded FP16 row scales, the
    // output-major INT8 weight `[n,k]` and its padded FP16 column scales.
    // `output` is a BF16 `rows * n` allocation on the same device -- the arena
    // slice the caller owns, or the buffer this module just allocated -- that
    // the tile writes directly. The launch is asynchronous on `ctx.stream()`,
    // and every buffer it touches outlives the call.
    unsafe {
        let nn_values = if plan.decode {
            weight.decode_nn_values_i8
        } else {
            None
        };
        let status = if plan.decode {
            ffi::qint8_gemm::apxinf_musa_asm_qint8_decode(
                activation.quantized.ptr() as *const c_void,
                nn_values.unwrap_or(weight.values_i8).ptr() as *const c_void,
                output,
                activation.row_scales.ptr() as *const c_void,
                weight.scales_f16.ptr() as *const c_void,
                plan.a_scale_bytes,
                plan.b_scale_bytes,
                m_i32,
                n_i32,
                k_i32,
                i32::from(nn_values.is_some()),
                ctx.stream().handle(),
            )
        } else {
            ffi::qint8_gemm::apxinf_musa_asm_qint8_gemm(
                activation.quantized.ptr() as *const c_void,
                weight.values_i8.ptr() as *const c_void,
                output,
                activation.row_scales.ptr() as *const c_void,
                weight.scales_f16.ptr() as *const c_void,
                plan.a_scale_bytes,
                plan.b_scale_bytes,
                m_i32,
                n_i32,
                k_i32,
                i32::from(plan.tile),
                ctx.stream().handle(),
            )
        };
        ffi::qint8_gemm::check_qint8_gemm(status, || {
            if plan.decode {
                format!("{operation} decode nn={}", nn_values.is_some())
            } else {
                format!("{operation} tile {}", plan.tile)
            }
        })
        .map_err(musa_error)?;
    }
    Ok(())
}

/// No toolkit: report the gap, do not fall back.
#[cfg(not(apxinf_musa_available))]
pub fn gemm_quantized_w8a8(
    _ctx: &MusaContext,
    _activation: &W8A8Activation,
    _weight: W8A8WeightView<'_>,
) -> Result<Tensor> {
    Err(native_unavailable("gemm::gemm_quantized_w8a8"))
}

/// No toolkit: report the gap, do not fall back.
#[cfg(not(apxinf_musa_available))]
pub fn gemm_quantized_w8a8_into(
    _ctx: &MusaContext,
    _activation: &W8A8Activation,
    _weight: W8A8WeightView<'_>,
    _output: &mut Tensor,
) -> Result<()> {
    Err(native_unavailable("gemm::gemm_quantized_w8a8_into"))
}

/// Fused `silu(gate) * up`, then row quantization.
pub fn quantize_w8a8_silu_mul_activation(
    _ctx: &MusaContext,
    _gate: &Tensor,
    _up: &Tensor,
) -> Result<W8A8Activation> {
    Err(not_ported("gemm::quantize_w8a8_silu_mul_activation"))
}

/// Adaptive layer norm, then row quantization.
pub fn adaptive_layer_norm_quantize_w8a8_activation(
    _ctx: &MusaContext,
    _input: &Tensor,
    _modulation: &Tensor,
    _eps: f32,
) -> Result<(Tensor, W8A8Activation)> {
    Err(not_ported(
        "gemm::adaptive_layer_norm_quantize_w8a8_activation",
    ))
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    //! Device tests. They need a MUSA device, and `apxinf_musa_available` only
    //! records that the *toolkit* was found when this crate was built, so they
    //! are `#[ignore]`d, exactly as `gemm::bf16`'s are.
    //!
    //! ```text
    //! cargo test -p apxinf-musa --release --lib -- --ignored --nocapture
    //! ```

    use std::time::Instant;

    use apxinf_core::{DType, Shape, Tensor};

    use crate::buffer::MusaBuffer;
    use crate::context::MusaContext;
    use crate::kernels::test_support::{
        from_bf16_bits, gate_for, read_back, to_bf16_bits, upload, values,
    };

    use super::{
        gemm_quantized_w8a8, gemm_w8a8, quantize_w8a8_activation, quantize_w8a8_geglu_activation,
        W8A8Activation, W8A8Layout, W8A8ScaleMode, W8A8WeightView,
    };

    /// A device f32 vector: the equalization factors, which are f32 by contract.
    fn upload_f32(ctx: &MusaContext, values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        let buffer = MusaBuffer::alloc_on(ctx, bytes.len()).expect("allocate factors");
        buffer.copy_from_host(&bytes).expect("upload factors");
        buffer
            .as_tensor(Shape::new(vec![values.len()]), DType::F32)
            .expect("view as f32")
    }

    /// The fused producer and the materialized chain must agree byte for byte:
    /// the fused kernel reproduces the intermediate BF16 rounding in registers,
    /// and the validated equalization alphas only hold for that arithmetic.
    #[test]
    #[ignore = "needs a MUSA device"]
    fn geglu_quantization_matches_the_materialized_chain_byte_for_byte() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        for (rows, width) in [(10, 4096), (522, 16384), (3, 4104), (3, 8192), (3, 150), (3, 16392)] {
            for case in 0..4 {
                let mut data = values(rows * width * 2, 41);
                if case == 1 {
                    data.fill(0.0);
                }
                if case == 2 {
                    data[0] = 128.0;
                    data[width] = 64.0;
                }
                let input = upload(&ctx, &data, Shape::new(vec![rows, width * 2]));
                let factors = upload_f32(
                    &ctx,
                    &(0..width)
                        .map(|i| 0.25 + (i % 13) as f32 * 0.2)
                        .collect::<Vec<_>>(),
                );
                let inv = if case == 3 { Some(&factors) } else { None };
                let materialized = crate::kernels::activation::geglu_bf16(&ctx, &input).unwrap();
                let expected = quantize_w8a8_activation(&ctx, &materialized, inv).unwrap();
                let actual = quantize_w8a8_geglu_activation(&ctx, &input, inv).unwrap();
                ctx.synchronize().unwrap();
                for (got, want, bytes) in [
                    (&actual.quantized, &expected.quantized, rows * width),
                    (
                        &actual.row_scales,
                        &expected.row_scales,
                        (rows + super::SCALE_PAD) * 2,
                    ),
                ] {
                    let mut got_bytes = vec![0; bytes];
                    let mut want_bytes = vec![0; bytes];
                    got.copy_to_host(&mut got_bytes).unwrap();
                    want.copy_to_host(&mut want_bytes).unwrap();
                    assert_eq!(
                        got_bytes, want_bytes,
                        "rows={rows} width={width} case={case}"
                    );
                }
            }
        }
    }

    /// Views the fused kernel cannot take must fall back, not refuse.
    #[test]
    #[ignore = "needs a MUSA device"]
    fn geglu_quantization_accepts_unaligned_views() {
        let ctx = MusaContext::new(0).unwrap();
        for (input_offset, factor_offset) in [(4, 0), (0, 4), (4, 4)] {
            let input_bytes: Vec<u8> = values(4 * 128, 41)
                .iter()
                .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
                .collect();
            let input_storage =
                MusaBuffer::alloc_on(&ctx, input_bytes.len() + input_offset).unwrap();
            let input_view = input_storage.view(input_offset, input_bytes.len()).unwrap();
            input_view.copy_from_host(&input_bytes).unwrap();
            let input = input_view
                .as_tensor(Shape::new(vec![4, 128]), DType::BF16)
                .unwrap();
            let factor_bytes: Vec<u8> = (0..64)
                .flat_map(|i| (0.5f32 + (i % 7) as f32 * 0.25).to_le_bytes())
                .collect();
            let factor_storage =
                MusaBuffer::alloc_on(&ctx, factor_bytes.len() + factor_offset).unwrap();
            let factor_view = factor_storage.view(factor_offset, factor_bytes.len()).unwrap();
            factor_view.copy_from_host(&factor_bytes).unwrap();
            let factors = factor_view
                .as_tensor(Shape::new(vec![64]), DType::F32)
                .unwrap();
            for inv in [None, Some(&factors)] {
                let materialized = crate::kernels::activation::geglu_bf16(&ctx, &input).unwrap();
                let expected = quantize_w8a8_activation(&ctx, &materialized, inv).unwrap();
                let actual = quantize_w8a8_geglu_activation(&ctx, &input, inv).unwrap();
                ctx.synchronize().unwrap();
                for (got, want, bytes) in [
                    (&actual.quantized, &expected.quantized, 4 * 64),
                    (
                        &actual.row_scales,
                        &expected.row_scales,
                        (4 + super::SCALE_PAD) * 2,
                    ),
                ] {
                    let mut got_bytes = vec![0; bytes];
                    let mut want_bytes = vec![0; bytes];
                    got.copy_to_host(&mut got_bytes).unwrap();
                    want.copy_to_host(&mut want_bytes).unwrap();
                    assert_eq!(
                        got_bytes, want_bytes,
                        "input_offset={input_offset} factor_offset={factor_offset}"
                    );
                }
            }
        }
    }

    /// Whether this build's assembly is the generation the device runs. The five
    /// INT8 images are [`crate::ASSEMBLY_ARCH`] assembly, so on another
    /// generation they are not assembled and these cases skip rather than
    /// reporting a device-less build as a failure.
    fn asm_tiles_built() -> bool {
        crate::MUSA_ARCH == crate::ASSEMBLY_ARCH
    }

    /// Deterministic INT8 values spread over `[-127, 127]`.
    fn int8_values(count: usize, mut seed: u32) -> Vec<i8> {
        (0..count)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (((seed >> 9) % 255) as i32 - 127) as i8
            })
            .collect()
    }

    fn f16_bits(value: f32) -> u16 {
        half::f16::from_f32(value).to_bits()
    }

    fn f16_value(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    /// An INT8 operand buffer: the GEMM ABI takes a raw pointer, not a view.
    fn upload_int8(ctx: &MusaContext, values: &[i8]) -> MusaBuffer {
        let bytes: Vec<u8> = values.iter().map(|value| *value as u8).collect();
        let buffer = MusaBuffer::alloc_on(ctx, bytes.len()).expect("allocate INT8 operand");
        buffer.copy_from_host(&bytes).expect("upload INT8 operand");
        buffer
    }

    /// An FP16 scale vector with the 512-entry zero pad the tile's guarded load
    /// may read.
    fn upload_f16_scales(ctx: &MusaContext, scales: &[f32]) -> MusaBuffer {
        let mut bytes: Vec<u8> = scales
            .iter()
            .flat_map(|value| f16_bits(*value).to_le_bytes())
            .collect();
        bytes.resize(bytes.len() + 512 * 2, 0);
        let buffer = MusaBuffer::alloc_on(ctx, bytes.len()).expect("allocate scales");
        buffer.copy_from_host(&bytes).expect("upload scales");
        buffer
    }

    /// An f32 vector, the shape the weights' column scales arrive in.
    fn upload_f32_scales(ctx: &MusaContext, scales: &[f32]) -> Tensor {
        let bytes: Vec<u8> = scales
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let buffer = MusaBuffer::alloc_on(ctx, bytes.len()).expect("allocate f32 scales");
        buffer.copy_from_host(&bytes).expect("upload f32 scales");
        buffer
            .as_tensor(Shape::new(vec![scales.len()]), DType::F32)
            .expect("view as f32")
    }

    /// Per-row scales with enough spread that a swapped or dropped row shows.
    fn row_scales(rows: usize) -> Vec<f32> {
        (0..rows)
            .map(|row| 0.0039 + 0.0007 * (row % 7) as f32)
            .collect()
    }

    fn column_scales(columns: usize) -> Vec<f32> {
        (0..columns)
            .map(|column| 0.0021 + 0.0004 * (column % 5) as f32)
            .collect()
    }

    /// The leading and trailing rows: the adapter splits a shape that ends
    /// mid-tile (`M = 522` into `512 + 10`), so the tail carries as much signal
    /// as the head.
    fn sampled_rows(m: usize) -> Vec<usize> {
        let head = m.min(8);
        let mut sampled: Vec<usize> = (0..head).collect();
        for row in m.saturating_sub(8)..m {
            if !sampled.contains(&row) {
                sampled.push(row);
            }
        }
        sampled
    }

    /// The GEMM and its dequantization epilogue against an integer reference.
    ///
    /// Both operands and both scale vectors are supplied exactly, so the only
    /// error this can report is the tile's own -- the activation quantizer is
    /// not on this path.
    fn check_quantized_gemm(ctx: &MusaContext, m: usize, k: usize, n: usize) {
        let a_values = int8_values(m * k, 0x1234_5678);
        let b_values = int8_values(n * k, 0x0bad_c0de);
        let a_scales = row_scales(m);
        let b_scales = column_scales(n);

        let activation = W8A8Activation {
            quantized: upload_int8(ctx, &a_values),
            row_scales: upload_f16_scales(ctx, &a_scales),
            rows: m,
            input_dim: k,
        };
        let b_buffer = upload_int8(ctx, &b_values);
        let b_scale_tensor = upload_f32_scales(ctx, &b_scales);
        let b_scales_f16 = upload_f16_scales(ctx, &b_scales);
        let weight = W8A8WeightView::new(
            &b_buffer,
            &b_scale_tensor,
            &b_scales_f16,
            k,
            n,
            W8A8ScaleMode::DynamicRowPerOutputChannel,
            W8A8Layout::OutputMajor,
            None,
        );

        let out = gemm_quantized_w8a8(ctx, &activation, weight).expect("INT8 gemm");
        ctx.stream().synchronize().expect("synchronise");
        assert_eq!(out.shape().dims(), &[m, n]);
        assert_eq!(out.dtype(), DType::BF16);
        let got = read_back(&out);

        let tolerance = gate_for(&got).max(1.0e-3);
        let mut worst = 0.0f32;
        for &row in &sampled_rows(m) {
            let a_scale = f16_value(f16_bits(a_scales[row]));
            for column in 0..n.min(8) {
                let b_scale = f16_value(f16_bits(b_scales[column]));
                let mut sum = 0i64;
                for inner in 0..k {
                    sum += i64::from(a_values[row * k + inner])
                        * i64::from(b_values[column * k + inner]);
                }
                let expected = f64::from(a_scale) * f64::from(b_scale) * sum as f64;
                let error = (f64::from(got[row * n + column]) - expected).abs() as f32;
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "[{m}x{k}]@[{k}x{n}] out[{row}][{column}]: {} vs {expected}, \
                     abs {error:.3e} > {tolerance:.3e}",
                    got[row * n + column]
                );
            }
        }
        println!(
            "  [{m:>4} x {k:>5}] @ [{k:>5} x {n:>5}]  worst abs {worst:.3e} (gate {tolerance:.3e})"
        );
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn w8a8_quantized_gemm_matches_an_integer_reference() {
        if !asm_tiles_built() {
            eprintln!(
                "skipping: this build's tiles are {} assembly, the device is {}",
                crate::ASSEMBLY_ARCH,
                crate::MUSA_ARCH
            );
            return;
        }
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        // The last two are the `(n, k)` pairs `qint8_gemm_adapter.mu`'s
        // `kTailOverrides` names, at the `m = 522` row split that reaches the
        // remainder they apply to. The first-but-one is the action expert's
        // bare `m = 10` qkv, which `qint8_tiles`' `MEASURED` routes to 32x128.
        for (m, k, n) in [
            (10, 2560, 2560),
            (128, 1152, 1152),
            (256, 2048, 2048),
            (522, 2048, 2048),
            (522, 2048, 16384),
            (522, 2560, 2048),
            (10, 1024, 2560),
            (522, 2048, 2560),
            (522, 1024, 2560),
        ] {
            check_quantized_gemm(&ctx, m, k, n);
        }
    }

    /// The whole operator, quantizer included: the activation arrives BF16 and
    /// the kernel row-quantizes it, so the reference quantizes it the same way.
    /// `equalization` is the per-channel `1/c` fold the quantizer applies, or
    /// `None`.
    fn check_full_gemm(ctx: &MusaContext, m: usize, k: usize, n: usize, equalization: Option<&[f32]>) {
        let activation_values = values(m * k, 0x5eed_1234);
        let activation = upload(ctx, &activation_values, Shape::new(vec![m, k]));
        let b_values = int8_values(n * k, 0x0bad_c0de);
        let b_scales = column_scales(n);
        let b_buffer = upload_int8(ctx, &b_values);
        let b_scale_tensor = upload_f32_scales(ctx, &b_scales);
        let b_scales_f16 = upload_f16_scales(ctx, &b_scales);
        let inv_c_tensor =
            equalization.map(|factors| upload_f32_scales(ctx, factors));
        let weight = W8A8WeightView::new(
            &b_buffer,
            &b_scale_tensor,
            &b_scales_f16,
            k,
            n,
            W8A8ScaleMode::DynamicRowPerOutputChannel,
            W8A8Layout::OutputMajor,
            inv_c_tensor.as_ref(),
        );

        let out = gemm_w8a8(ctx, &activation, weight).expect("INT8 gemm");
        ctx.stream().synchronize().expect("synchronise");
        assert_eq!(out.shape().dims(), &[m, n]);
        let got = read_back(&out);

        // `quantize_rows_bf16_int8_kernel` replicated: per-row amax over the
        // bf16 input (times the equalization factors when they are set, the
        // product rounded back to bf16 exactly as the kernel does), scale
        // `max(amax / 127, 1e-12)`, `round(x / scale)` clamped
        // to `[-128, 127]`. The quantizer divides by the f32 scale but stores its
        // FP16 rounding, which is what the epilogue multiplies back.
        let scaled = |row: usize, column: usize| {
            from_bf16_bits(to_bf16_bits(
                from_bf16_bits(to_bf16_bits(activation_values[row * k + column]))
                    * equalization.map_or(1.0, |factors| factors[column]),
            ))
        };
        let mut quantized = vec![0i8; m * k];
        let mut scales = vec![0.0f32; m];
        for row in 0..m {
            let mut amax = 0.0f32;
            for inner in 0..k {
                amax = amax.max(scaled(row, inner).abs());
            }
            let scale = (amax / 127.0).max(1.0e-12);
            scales[row] = scale;
            for inner in 0..k {
                let quantized_value = (scaled(row, inner) / scale).round().clamp(-128.0, 127.0);
                quantized[row * k + inner] = quantized_value as i8;
            }
        }

        let tolerance = gate_for(&got).max(1.0e-3);
        let mut worst = 0.0f32;
        for &row in &sampled_rows(m) {
            let a_scale = f16_value(f16_bits(scales[row]));
            for column in 0..n.min(8) {
                let b_scale = f16_value(f16_bits(b_scales[column]));
                let mut sum = 0i64;
                for inner in 0..k {
                    sum += i64::from(quantized[row * k + inner])
                        * i64::from(b_values[column * k + inner]);
                }
                let expected = f64::from(a_scale) * f64::from(b_scale) * sum as f64;
                let error = (f64::from(got[row * n + column]) - expected).abs() as f32;
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "[{m}x{k}]@[{k}x{n}] out[{row}][{column}]: {} vs {expected}, \
                     abs {error:.3e} > {tolerance:.3e}",
                    got[row * n + column]
                );
            }
        }
        println!(
            "  [{m:>4} x {k:>5}] @ [{k:>5} x {n:>5}]  worst abs {worst:.3e} (gate {tolerance:.3e})"
        );
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn w8a8_gemm_quantizes_the_activation_like_the_reference() {
        if !asm_tiles_built() {
            eprintln!(
                "skipping: this build's tiles are {} assembly, the device is {}",
                crate::ASSEMBLY_ARCH,
                crate::MUSA_ARCH
            );
            return;
        }
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        for (m, k, n) in [
            (10, 2560, 2560),
            (256, 2048, 2048),
            (522, 2048, 2048),
            (522, 2048, 16384),
        ] {
            check_full_gemm(&ctx, m, k, n, None);
        }
    }

    /// The equalization fold: a non-uniform `1/c` vector must change both the
    /// measured amax and the quantized values exactly as the reference's
    /// pre-multiplied copy does. Includes the down projection's `k = 16384`,
    /// the shape the fold runs on in the model.
    #[test]
    #[ignore = "needs a MUSA device"]
    fn w8a8_gemm_applies_the_equalization_factors() {
        if !asm_tiles_built() {
            eprintln!(
                "skipping: this build's tiles are {} assembly, the device is {}",
                crate::ASSEMBLY_ARCH,
                crate::MUSA_ARCH
            );
            return;
        }
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        for (m, k, n) in [(522, 2048, 2048), (522, 2048, 16384)] {
            // Positive, non-uniform, and not a power of two per channel: a
            // factor that only rescales the row uniformly would leave the
            // quantized values unchanged and test nothing.
            let factors: Vec<f32> = (0..k)
                .map(|column| 0.3 + (column % 13) as f32 * 0.31)
                .collect();
            check_full_gemm(&ctx, m, k, n, Some(&factors));
        }
    }

    /// Timing for the four GEMMs one pi0.5 L1 encode layer runs, through the
    /// operator the model calls -- quantizer and GEMM included.
    #[test]
    #[ignore = "needs a MUSA device"]
    fn w8a8_gemm_encode_shapes_throughput() {
        if !asm_tiles_built() {
            eprintln!(
                "skipping: this build's tiles are {} assembly, the device is {}",
                crate::ASSEMBLY_ARCH,
                crate::MUSA_ARCH
            );
            return;
        }
        let ctx = &MusaContext::new(0).expect("MUSA device 0");
        // (m, n, k) for qkv, o_proj, gate_up and down of one encode layer.
        for (m, n, k) in [
            (522, 2560, 2048),
            (522, 2048, 2048),
            (522, 32768, 2048),
            (522, 2048, 16384),
        ] {
            let activation_values = values(m * k, 0x5eed_1234);
            let activation = upload(ctx, &activation_values, Shape::new(vec![m, k]));
            let b_values = int8_values(n * k, 0x0bad_c0de);
            let b_scales = column_scales(n);
            let b_buffer = upload_int8(ctx, &b_values);
            let b_scale_tensor = upload_f32_scales(ctx, &b_scales);
            let b_scales_f16 = upload_f16_scales(ctx, &b_scales);
            let weight = W8A8WeightView::new(
                &b_buffer,
                &b_scale_tensor,
                &b_scales_f16,
                k,
                n,
                W8A8ScaleMode::DynamicRowPerOutputChannel,
                W8A8Layout::OutputMajor,
                None,
            );

            for _ in 0..5 {
                let _ = gemm_w8a8(ctx, &activation, weight).expect("warmup");
            }
            ctx.stream().synchronize().expect("synchronise");

            let iterations = 20;
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = gemm_w8a8(ctx, &activation, weight).expect("timed");
            }
            ctx.stream().synchronize().expect("synchronise");
            let elapsed = start.elapsed().as_secs_f64() / f64::from(iterations);
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            println!(
                "  [{m:>4} x {k:>5}] @ [{k:>5} x {n:>5}]  {:.4} ms  {:.2} TOPS  \
                 (tile mode {:?})",
                elapsed * 1.0e3,
                flops / elapsed / 1.0e12,
                super::super::qint8_tiles::select(m, n, k)
            );
        }
    }
}

