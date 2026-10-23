//! Fixed and direct-launch GEMM operators.

use std::cell::RefCell;
use std::rc::Rc;

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::{ops, CudaContext};

/// Calibration hook: observes every BF16 activation/weight pair entering a
/// GEMM while installed. Thread-local, so normal inference pays one
/// empty-cell check and concurrent model threads cannot see each other's
/// activations. Ported from the legacy crate with identical semantics.
pub trait Bf16ActivationObserver {
    fn observe(&self, activation: &Tensor, weight: &Tensor) -> Result<()>;
}
thread_local! {
    static BF16_OBSERVER: RefCell<Option<Rc<dyn Bf16ActivationObserver>>> =
        const { RefCell::new(None) };
}

/// Uninstalls the observer when dropped.
pub struct Bf16ObserverGuard;

impl Drop for Bf16ObserverGuard {
    fn drop(&mut self) {
        BF16_OBSERVER.with(|slot| *slot.borrow_mut() = None);
    }
}

pub fn install_bf16_observer(
    observer: Rc<dyn Bf16ActivationObserver>,
) -> Result<Bf16ObserverGuard> {
    BF16_OBSERVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(Error::Other(
                "a BF16 activation observer is already installed".into(),
            ));
        }
        *slot = Some(observer);
        Ok(Bf16ObserverGuard)
    })
}

fn observe_bf16(activation: &Tensor, weight: &Tensor) -> Result<()> {
    BF16_OBSERVER.with(|slot| {
        if let Some(observer) = slot.borrow().as_ref() {
            observer.observe(activation, weight)?;
        }
        Ok(())
    })
}

fn output_for(ctx: &CudaContext, a: &Tensor, b: &Tensor, what: &str) -> Result<Tensor> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] {
        return Err(Error::Other(format!(
            "{what} shape mismatch: {a_dims:?} @ {b_dims:?}"
        )));
    }
    ctx.allocate_output(Shape::new(vec![a_dims[0], b_dims[1]]), DType::BF16)
}

/// `bf16`: plain BF16 GEMM, `[m, k] @ [k, n] -> [m, n]`.
pub fn bf16(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    observe_bf16(activation, weight)?;
    let mut output = output_for(ctx, activation, weight, "BF16 GEMM")?;
    ops::gemm(ctx, ops::GemmArgs::new(activation, weight, &mut output))?;
    Ok(output)
}

/// BF16 GEMM through the context's shared cuBLAS handle.
///
/// This preserves the legacy static-inference contract for models whose
/// numerical baselines were produced by one process-wide handle.  It remains
/// graph-safe after the model's eager preflight, but deliberately bypasses L3
/// candidate selection; new model paths should use [`bf16`] unless they need
/// this established rounding behavior.
pub fn bf16_direct(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    observe_bf16(activation, weight)?;
    let output = output_for(ctx, activation, weight, "direct BF16 GEMM")?;
    let a_dims = activation.shape().dims();
    let b_dims = weight.shape().dims();
    let activation = crate::CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight = crate::CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let output_buffer = crate::CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    ctx.cublas()
        .gemm(
            DType::BF16,
            a_dims[0],
            b_dims[1],
            a_dims[1],
            1.0,
            &activation,
            &weight,
            0.0,
            &output_buffer,
        )
        .map_err(Error::Cuda)?;
    Ok(output)
}

/// `matmul`: the legacy generic GEMM. BF16 routes to the tuned operator; any
/// other dtype (pi0fast's FP32 SigLIP patch projection) runs through raw
/// cuBLAS at the operand dtype, as the legacy kernel did — the tuned GEMM
/// operator is BF16-only and rounding the projection to BF16 changes the
/// autoregressive action tokens.
pub fn matmul(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if activation.dtype() == DType::BF16 {
        return bf16(ctx, activation, weight);
    }
    if activation.dtype() != weight.dtype() {
        return Err(Error::DTypeMismatch {
            expected: activation.dtype(),
            got: weight.dtype(),
        });
    }
    let a_dims = activation.shape().dims();
    let w_dims = weight.shape().dims();
    if a_dims.len() != 2 || w_dims.len() != 2 || a_dims[1] != w_dims[0] {
        return Err(Error::Other(format!(
            "matmul expects [m,k] @ [k,n], got {a_dims:?} and {w_dims:?}"
        )));
    }
    let (m, k, n) = (a_dims[0], a_dims[1], w_dims[1]);
    let output = ctx.allocate_output(Shape::new(vec![m, n]), activation.dtype())?;
    let activation_buffer = crate::CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_buffer = crate::CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let output_buffer = crate::CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    ctx.cublas()
        .gemm(
            activation.dtype(),
            m,
            n,
            k,
            1.0,
            &activation_buffer,
            &weight_buffer,
            0.0,
            &output_buffer,
        )
        .map_err(Error::Cuda)?;
    Ok(output)
}

/// `bf16_bias`: BF16 GEMM with a fused `[n]` bias epilogue.
pub fn bf16_bias(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let mut output = output_for(ctx, activation, weight, "BF16 bias GEMM")?;
    let gemm = ops::GemmArgs::new(activation, weight, &mut output);
    ops::gemm_bias(ctx, ops::GemmBiasArgs { gemm, bias })?;
    Ok(output)
}

/// Pre-quantized E4M3 weight with one per-tensor scale, mirroring the legacy
/// `Fp8WeightView`. The dual-GeGLU interleaved layouts are a legacy-runtime
/// concept and are intentionally absent.
#[derive(Clone, Copy)]
pub struct Fp8WeightView<'a> {
    pub values_e4m3: &'a Tensor,
    pub scale: f32,
    /// Exact dual-GeGLU `[gate256,up256]` physical layout. cuda-new's GEMM
    /// operator does not consume this layout, so a plain FP8 GEMM rejects it.
    pub dual_geglu_interleaved: bool,
    /// Optional auto-mode physical `[gate256,up256]` matrix. The primary tensor
    /// remains plain and is used by every non-dual route.
    pub dual_geglu_auto_interleaved: Option<&'a Tensor>,
}

/// Explicit schedule attributes for the opt-in FP8-to-BF16 custom-tile GEMM.
#[derive(Clone, Copy, Debug)]
pub struct Fp8Bf16CustomConfig {
    pub tile_id: i32,
    pub custom_option: i32,
    pub stages_id: i32,
    pub cluster_shape_id: i32,
}

/// `fp8_bf16_custom`: opt-in FP8 GEMM with an explicit custom-tile schedule.
///
/// cuda-new has no custom-tile FP8 kernel; the schedule only selects among
/// equivalent candidates, so this routes to the generic tuned `fp8_bf16`
/// operator, which is numerically the same projection.
pub fn fp8_bf16_custom(
    ctx: &CudaContext,
    activation: &Tensor,
    activation_scale: f32,
    weight: Fp8WeightView<'_>,
    _config: Fp8Bf16CustomConfig,
) -> Result<Tensor> {
    if weight.dual_geglu_interleaved {
        return Err(Error::Other(
            "FP8 dual GeGLU interleaved weight cannot be used by plain FP8 GEMM".into(),
        ));
    }
    fp8_bf16(ctx, activation, activation_scale, weight)
}

/// `fp8_bf16`: static per-tensor FP8 GEMM with BF16 output.
///
/// The legacy helper consumed unit-scaled E4M3 operands and applied
/// `activation_scale * weight_scale` as alpha; cuda-new's `Fp8UnitScale`
/// quantization is the same contract.
pub fn fp8_bf16(
    ctx: &CudaContext,
    activation: &Tensor,
    activation_scale: f32,
    weight: Fp8WeightView<'_>,
) -> Result<Tensor> {
    let a_dims = activation.shape().dims();
    let b_dims = weight.values_e4m3.shape().dims();
    if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] {
        return Err(Error::Other(format!(
            "FP8 GEMM shape mismatch: {a_dims:?} @ {b_dims:?}"
        )));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![a_dims[0], b_dims[1]]), DType::BF16)?;
    let mut gemm = ops::GemmArgs::new(activation, weight.values_e4m3, &mut output);
    gemm.quantization = ops::GemmQuantization::Fp8UnitScale;
    gemm.alpha = activation_scale * weight.scale;
    ops::gemm(ctx, gemm)?;
    Ok(output)
}

/// `bf16_geglu_fused`: fused gate/up GEMM + GeGLU over a packed
/// `[k, 2*cols]` weight, producing `[rows, cols]`.
///
/// The legacy interleaved dual-GeGLU weight layouts are autotune candidates of
/// the legacy runtime; cuda-new selects its own candidates from the plain
/// layout, so only the plain weight is accepted.
pub fn bf16_geglu_fused(
    ctx: &CudaContext,
    activation: &Tensor,
    packed_weight: &Tensor,
) -> Result<Tensor> {
    observe_bf16(activation, packed_weight)?;
    let a_dims = activation.shape().dims();
    let b_dims = packed_weight.shape().dims();
    if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] || b_dims[1] % 2 != 0 {
        return Err(Error::Other(format!(
            "fused GeGLU shape mismatch: {a_dims:?} @ {b_dims:?}"
        )));
    }
    let mut output =
        ctx.allocate_output(Shape::new(vec![a_dims[0], b_dims[1] / 2]), DType::BF16)?;
    let gemm = ops::GemmArgs::new(activation, packed_weight, &mut output);
    ops::gemm_geglu(ctx, ops::GemmGegluArgs { gemm })?;
    Ok(output)
}

/// Pre-quantized rowwise-dynamic FP8 weight. Unlike the legacy view, the
/// value matrix is stored in cuda-new's canonical `[K, N]` orientation; the
/// `[N, K]` legacy layout must be transposed at load time (a pure byte
/// permutation with no numerical effect).
#[derive(Clone, Copy)]
pub struct DynamicFp8WeightView<'a> {
    /// Contiguous `[K, N]` E4M3 matrix.
    pub values_e4m3: &'a Tensor,
    /// FP32 scale for each output channel, shape `[N]`.
    pub channel_scales: &'a Tensor,
}

/// `gemm_fp8_dynamic_bf16`: rowwise-dynamic FP8 GEMM — per-row activation
/// scales, per-channel weight scales, BF16 output.
pub fn gemm_fp8_dynamic_bf16(
    ctx: &CudaContext,
    activation: &Tensor,
    activation_scales: &Tensor,
    weight: DynamicFp8WeightView<'_>,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let a_dims = activation.shape().dims();
    let b_dims = weight.values_e4m3.shape().dims();
    if activation.dtype() != DType::F8E4M3
        || weight.values_e4m3.dtype() != DType::F8E4M3
        || a_dims.len() != 2
        || b_dims.len() != 2
        || a_dims[1] != b_dims[0]
    {
        return Err(Error::Other(format!(
            "dynamic FP8 GEMM shape mismatch: {a_dims:?} @ KN {b_dims:?}"
        )));
    }
    let (m, n) = (a_dims[0], b_dims[1]);
    if activation_scales.dtype() != DType::F32
        || weight.channel_scales.dtype() != DType::F32
        || activation_scales.shape().dims() != [m]
        || weight.channel_scales.shape().dims() != [n]
    {
        return Err(Error::Other(
            "dynamic FP8 GEMM scale vectors must be F32 [M] and [N]".into(),
        ));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![m, n]), DType::BF16)?;
    let gemm = ops::GemmArgs::fp8(
        activation,
        activation_scales,
        weight.values_e4m3,
        weight.channel_scales,
        &mut output,
    );
    match bias {
        Some(bias) => ops::gemm_bias(ctx, ops::GemmBiasArgs { gemm, bias })?,
        None => ops::gemm(ctx, gemm)?,
    }
    Ok(output)
}

// ── qwen_drive direct-launch GEMM helpers over raw cuBLAS ──────────────────
//
// These predate the tuned GEMM operator and keep their legacy arithmetic:
// FP32 accumulators held until the bias lands, one final BF16 rounding.

use crate::ops::validation::{checked_bytes, require_buffers, require_finite};
use crate::cublas::CublasTranspose;
use crate::ffi::abi::hybrid as la_abi;
use crate::ffi::raw::cuda_runtime as raw;
use crate::CudaBuffer;
use apxinf_core::Device;

/// BF16 `bias + weight @ vector`, with checkpoint-row-major weight `[N,K]`.
pub fn bf16_addmv(
    ctx: &CudaContext,
    weight: &Tensor,
    vector: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let w = weight.shape().dims();
    if w.len() != 2
        || w.contains(&0)
        || vector.shape().dims() != [w[1]]
        || bias.shape().dims() != [w[0]]
    {
        return Err(Error::Other(
            "BF16 addmv expects weight[N,K], vector[K], bias[N]".into(),
        ));
    }
    for tensor in [weight, vector, bias] {
        if tensor.dtype() != DType::BF16 || tensor.device() != Device::Cuda(ctx.device_id()) {
            return Err(Error::Other(
                "BF16 addmv requires inputs on the context device".into(),
            ));
        }
        checked_bytes(DType::BF16, tensor.shape().dims(), "BF16 addmv")?;
    }
    let k = i32::try_from(w[1]).map_err(|_| Error::Other("addmv input width overflow".into()))?;
    let n = i32::try_from(w[0]).map_err(|_| Error::Other("addmv output width overflow".into()))?;
    // Some cuBLAS BF16-output GEMV paths round the dot product before
    // applying beta * C, even with FP32 compute. Keep C in FP32 until the
    // bias has been added, then round once to satisfy the addmv contract.
    let accumulator = crate::workspace::output_buffer(
        ctx,
        checked_bytes(DType::F32, &[w[0]], "BF16 addmv accumulator")?,
    )?
    .into_tensor(Shape::new(vec![w[0]]), DType::F32);
    crate::ops::linear_attention::cast_bf16_to_f32(ctx, bias, &accumulator)?;
    let wp = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let xp = CudaBuffer::from_tensor(vector).map_err(Error::Cuda)?;
    let cp = CudaBuffer::from_tensor(&accumulator).map_err(Error::Cuda)?;
    ctx.cublas()
        .gemm_bf16_f32_ex(
            CublasTranspose::None,
            CublasTranspose::Transpose,
            1,
            w[0],
            w[1],
            1.0,
            &xp,
            k,
            &wp,
            k,
            1.0,
            &cp,
            n,
        )
        .map_err(Error::Cuda)?;
    let output = crate::workspace::output_buffer(ctx, bias.size_in_bytes())?
        .into_tensor(Shape::new(vec![w[0]]), DType::BF16);
    crate::ops::linear_attention::cast_f32_to_bf16(ctx, &accumulator, &output)?;
    Ok(output)
}

/// BF16 `input @ weight.T + bias` for checkpoint-row-major weight `[N,K]`.
/// Bias is broadcast into an FP32 accumulator, with one final BF16 rounding
/// after the GEMM.
pub fn bf16_addmm_checkpoint(
    ctx: &CudaContext,
    weight: &Tensor,
    input: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let w = weight.shape().dims();
    let x = input.shape().dims();
    if w.len() != 2
        || x.len() != 2
        || w.contains(&0)
        || x.contains(&0)
        || x[1] != w[1]
        || bias.shape().dims() != [w[0]]
    {
        return Err(Error::Other(
            "BF16 checkpoint addmm expects weight[N,K], input[M,K], bias[N]".into(),
        ));
    }
    for tensor in [weight, input, bias] {
        if tensor.dtype() != DType::BF16 || tensor.device() != Device::Cuda(ctx.device_id()) {
            return Err(Error::Other(
                "BF16 checkpoint addmm requires BF16 inputs on the context device".into(),
            ));
        }
    }
    let m = i32::try_from(x[0]).map_err(|_| Error::Other("addmm row count overflow".into()))?;
    let n = i32::try_from(w[0]).map_err(|_| Error::Other("addmm output width overflow".into()))?;
    let k = w[1];
    let shape = [x[0], w[0]];
    let accumulator_bytes = checked_bytes(DType::F32, &shape, "BF16 addmm accumulator")?;
    let output_bytes = checked_bytes(DType::BF16, &shape, "BF16 addmm output")?;
    let elements = accumulator_bytes / DType::F32.size_in_bytes();
    i32::try_from((elements - 1) / 256 + 1)
        .map_err(|_| Error::Other("addmm output exceeds cast launch extent".into()))?;
    let wp = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let xp = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let bp = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let accumulator = crate::workspace::output_buffer(ctx, accumulator_bytes)?
        .into_tensor(Shape::new(shape.to_vec()), DType::F32);
    let cp = CudaBuffer::from_tensor(&accumulator).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(la_abi::apxinf_hybrid_broadcast_bf16_f32_rows(
            bp.ptr(),
            cp.ptr(),
            m,
            n,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    ctx.cublas()
        .gemm_bf16_f32_ex(
            CublasTranspose::None,
            CublasTranspose::Transpose,
            x[0],
            w[0],
            k,
            1.0,
            &xp,
            k as i32,
            &wp,
            k as i32,
            1.0,
            &cp,
            n,
        )
        .map_err(Error::Cuda)?;
    let output = crate::workspace::output_buffer(ctx, output_bytes)?
        .into_tensor(Shape::new(shape.to_vec()), DType::BF16);
    crate::ops::linear_attention::cast_f32_to_bf16(ctx, &accumulator, &output)?;
    Ok(output)
}

/// BF16 projection with FP32 accumulation, bias, and tanh GELU before the
/// final BF16 rounding. cuda-new routes this through the tuned biased-GELU
/// GEMM operator, which owns the same contract.
pub fn bf16_bias_gelu_tanh(
    ctx: &CudaContext,
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let a = x.shape().dims();
    let b = weight.shape().dims();
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] || bias.shape().dims() != [b[1]] {
        return Err(Error::Other(
            "BF16 biased GEMM expects [M,K] @ [K,N] + [N]".into(),
        ));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![a[0], b[1]]), DType::BF16)?;
    let gemm = ops::GemmArgs::new(x, weight, &mut output);
    ops::gemm_bias_gelu(ctx, ops::GemmBiasGeluArgs { gemm, bias })?;
    Ok(output)
}

/// Raw strided GEMM into a caller-owned buffer: `output = alpha * op(a) @
/// op(b) + beta * output` with explicit leading dimensions.
#[allow(clippy::too_many_arguments)]
pub fn write_ex(
    ctx: &CudaContext,
    dtype: DType,
    trans_a: CublasTranspose,
    trans_b: CublasTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &CudaBuffer,
    lda: i32,
    b: &CudaBuffer,
    ldb: i32,
    beta: f32,
    output: &CudaBuffer,
    ldc: i32,
) -> Result<()> {
    require_finite("GEMM_EX", &[alpha, beta])?;
    let (a_rows, a_cols) = match trans_a {
        CublasTranspose::None => (m, k),
        CublasTranspose::Transpose => (k, m),
    };
    let (b_rows, b_cols) = match trans_b {
        CublasTranspose::None => (k, n),
        CublasTranspose::Transpose => (n, k),
    };
    if lda <= 0
        || ldb <= 0
        || ldc <= 0
        || (lda as usize) < a_cols
        || (ldb as usize) < b_cols
        || (ldc as usize) < n
    {
        return Err(Error::Other(format!(
            "GEMM_EX invalid row strides lda={lda}, ldb={ldb}, ldc={ldc}"
        )));
    }
    let strided_bytes = |rows: usize, stride: i32, cols: usize| -> Result<usize> {
        let elements = rows
            .saturating_sub(1)
            .checked_mul(stride as usize)
            .and_then(|offset| offset.checked_add(cols))
            .ok_or_else(|| Error::Other("GEMM_EX buffer size overflow".into()))?;
        checked_bytes(dtype, &[elements], "GEMM_EX")
    };
    require_buffers(
        ctx,
        "GEMM_EX",
        &[
            ("A", a, strided_bytes(a_rows, lda, a_cols)?),
            ("B", b, strided_bytes(b_rows, ldb, b_cols)?),
            ("output", output, strided_bytes(m, ldc, n)?),
        ],
    )?;
    ctx.cublas()
        .gemm_ex(
            dtype, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, output, ldc,
        )
        .map_err(Error::Cuda)
}

// ── gr00t W8A8 (INT8) surface ────────────────────────────────────────────
//
// The legacy W8A8 path stores the weight output-major `[N, K]` and runs raw
// cuBLAS INT8 GEMM (`OP_T`, i32 accumulate) plus a fused dequantize. cuda-new
// exposes both pieces (`CublasHandle::gemm_int8_i32` and the gr00t
// dequantize launcher), so the shim reproduces the arithmetic exactly.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W8A8ScaleMode {
    DynamicRowPerOutputChannel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W8A8Layout {
    OutputMajor,
}

/// Borrowed W8A8 weight view. `values_i8` is physical output-major `[N, K]`.
#[derive(Clone, Copy)]
pub struct W8A8WeightView<'a> {
    pub values_i8: &'a CudaBuffer,
    pub scales_f32: &'a Tensor,
    pub input_dim: usize,
    pub output_dim: usize,
    pub scale_mode: W8A8ScaleMode,
    pub layout: W8A8Layout,
}

/// Dynamically row-quantized activation (I8 values + F32 row scales).
pub struct W8A8Activation {
    pub(crate) quantized: CudaBuffer,
    pub(crate) row_scales: CudaBuffer,
    pub(crate) rows: usize,
    pub(crate) input_dim: usize,
}

impl W8A8Activation {
    /// I8 activation buffer, `[rows, input_dim]`.
    pub(crate) fn quantized(&self) -> &CudaBuffer {
        &self.quantized
    }
    /// F32 per-row scales.
    pub(crate) fn row_scales(&self) -> &CudaBuffer {
        &self.row_scales
    }
}

fn w8a8_activation_from(
    ctx: &CudaContext,
    rows: usize,
    input_dim: usize,
) -> Result<W8A8Activation> {
    let quantized = crate::workspace::output_buffer(ctx, rows * input_dim)?;
    let row_scales = crate::workspace::output_buffer(ctx, rows * std::mem::size_of::<f32>())?;
    Ok(W8A8Activation {
        quantized,
        row_scales,
        rows,
        input_dim,
    })
}

/// `quantize_w8a8_activation`: dynamically row-quantize a BF16 activation to
/// I8 with per-row F32 scales.
pub fn quantize_w8a8_activation(ctx: &CudaContext, activation: &Tensor) -> Result<W8A8Activation> {
    use crate::ffi::abi::transformer_bf16 as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    let dims = activation.shape().dims();
    if activation.dtype() != DType::BF16 || dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(Error::Other(
            "W8A8 quantization expects a non-empty rank-2 BF16 activation".into(),
        ));
    }
    let (rows, input_dim) = (dims[0], dims[1]);
    let mut result = w8a8_activation_from(ctx, rows, input_dim)?;
    let input_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_transformer_bf16_quantize_rows_bf16_int8(
            input_buffer.ptr(),
            result.quantized.ptr(),
            result.row_scales.ptr(),
            i32::try_from(rows).map_err(|_| Error::Other("rows exceed i32".into()))?,
            i32::try_from(input_dim).map_err(|_| Error::Other("cols exceed i32".into()))?,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(result)
}

/// `bias_gelu_quantize_w8a8_activation`: bias + tanh-GELU then rowwise I8
/// quantization.
pub fn bias_gelu_quantize_w8a8_activation(
    ctx: &CudaContext,
    activation: &Tensor,
    bias: &Tensor,
) -> Result<W8A8Activation> {
    use crate::ffi::abi::transformer_bf16 as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    let dims = activation.shape().dims();
    if activation.dtype() != DType::BF16
        || bias.dtype() != DType::BF16
        || dims.len() != 2
        || dims[0] == 0
        || dims[1] == 0
        || bias.shape().dims() != [dims[1]]
    {
        return Err(Error::Other(
            "W8A8 fused bias-GELU quantization expects [rows,cols] and [cols]".into(),
        ));
    }
    let (rows, input_dim) = (dims[0], dims[1]);
    let mut result = w8a8_activation_from(ctx, rows, input_dim)?;
    let input_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_transformer_bf16_bias_gelu_quantize_rows_bf16_int8(
            input_buffer.ptr(),
            bias_buffer.ptr(),
            result.quantized.ptr(),
            result.row_scales.ptr(),
            i32::try_from(rows).map_err(|_| Error::Other("rows exceed i32".into()))?,
            i32::try_from(input_dim).map_err(|_| Error::Other("cols exceed i32".into()))?,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(result)
}

/// `adaptive_layer_norm_quantize_w8a8_activation`: adaptive LayerNorm producing
/// a BF16 view and a rowwise-quantized I8 view.
pub fn adaptive_layer_norm_quantize_w8a8_activation(
    ctx: &CudaContext,
    input: &Tensor,
    modulation: &Tensor,
    eps: f32,
) -> Result<(Tensor, W8A8Activation)> {
    use crate::ffi::abi::transformer_bf16 as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    let dims = input.shape().dims();
    if input.dtype() != DType::BF16
        || modulation.dtype() != DType::BF16
        || dims.len() != 2
        || dims[0] == 0
        || dims[1] == 0
        || modulation.shape().dims() != [2 * dims[1]]
        || !(eps > 0.0)
    {
        return Err(Error::Other(
            "W8A8 adaptive LayerNorm quantization expects BF16 input and [2*cols] modulation"
                .into(),
        ));
    }
    let (rows, input_dim) = (dims[0], dims[1]);
    let output = ctx.allocate_output(Shape::new(vec![rows, input_dim]), DType::BF16)?;
    let mut quantized = w8a8_activation_from(ctx, rows, input_dim)?;
    let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let modulation_buffer = CudaBuffer::from_tensor(modulation).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(
            abi::apxinf_transformer_bf16_adaptive_layer_norm_quantize_rows_bf16_int8(
                input_buffer.ptr(),
                modulation_buffer.ptr(),
                output_buffer.ptr(),
                quantized.quantized.ptr(),
                quantized.row_scales.ptr(),
                i32::try_from(rows).map_err(|_| Error::Other("rows exceed i32".into()))?,
                i32::try_from(input_dim).map_err(|_| Error::Other("cols exceed i32".into()))?,
                eps,
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)?;
    }
    Ok((output, quantized))
}

/// `layer_norm_quantize_w8a8_activation`: LayerNorm producing a BF16 view and
/// a rowwise-quantized I8 view.
pub fn layer_norm_quantize_w8a8_activation(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<(Tensor, W8A8Activation)> {
    use crate::ffi::abi::transformer_bf16 as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    let dims = input.shape().dims();
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || bias.dtype() != DType::BF16
        || dims.len() != 2
        || dims[0] == 0
        || dims[1] == 0
        || weight.shape().dims() != [dims[1]]
        || bias.shape().dims() != [dims[1]]
        || !(eps > 0.0)
    {
        return Err(Error::Other(
            "W8A8 LayerNorm quantization expects BF16 input and [cols] affine tensors".into(),
        ));
    }
    let (rows, input_dim) = (dims[0], dims[1]);
    let output = ctx.allocate_output(Shape::new(vec![rows, input_dim]), DType::BF16)?;
    let mut quantized = w8a8_activation_from(ctx, rows, input_dim)?;
    let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_transformer_bf16_layer_norm_quantize_rows_bf16_int8(
            input_buffer.ptr(),
            weight_buffer.ptr(),
            bias_buffer.ptr(),
            output_buffer.ptr(),
            quantized.quantized.ptr(),
            quantized.row_scales.ptr(),
            i32::try_from(rows).map_err(|_| Error::Other("rows exceed i32".into()))?,
            i32::try_from(input_dim).map_err(|_| Error::Other("cols exceed i32".into()))?,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok((output, quantized))
}

/// `quantize_w8a8_silu_mul_activation`: `silu(gate) * up` then rowwise I8
/// quantization.
pub fn quantize_w8a8_silu_mul_activation(
    ctx: &CudaContext,
    gate: &Tensor,
    up: &Tensor,
) -> Result<W8A8Activation> {
    quantize_w8a8_silu_mul_impl(ctx, gate, up, false)
}

/// `quantize_w8a8_silu_mul_activation_packed4`: packed-4 companion; identical
/// arithmetic, wider accesses.
pub fn quantize_w8a8_silu_mul_activation_packed4(
    ctx: &CudaContext,
    gate: &Tensor,
    up: &Tensor,
) -> Result<W8A8Activation> {
    quantize_w8a8_silu_mul_impl(ctx, gate, up, true)
}

fn quantize_w8a8_silu_mul_impl(
    ctx: &CudaContext,
    gate: &Tensor,
    up: &Tensor,
    packed4: bool,
) -> Result<W8A8Activation> {
    use crate::ffi::abi::transformer_bf16 as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    if gate.dtype() != DType::BF16 || up.dtype() != DType::BF16 || gate.shape() != up.shape() {
        return Err(Error::Other(
            "W8A8 fused SiLU-mul quantization expects equal BF16 tensors".into(),
        ));
    }
    let dims = gate.shape().dims();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(Error::Other(
            "W8A8 fused SiLU-mul quantization expects a non-empty matrix".into(),
        ));
    }
    let (rows, input_dim) = (dims[0], dims[1]);
    let mut result = w8a8_activation_from(ctx, rows, input_dim)?;
    let gate_buffer = CudaBuffer::from_tensor(gate).map_err(Error::Cuda)?;
    let up_buffer = CudaBuffer::from_tensor(up).map_err(Error::Cuda)?;
    unsafe {
        let code = if packed4 {
            abi::apxinf_transformer_bf16_silu_mul_quantize_rows_bf16_int8_packed4(
                gate_buffer.ptr(),
                up_buffer.ptr(),
                result.quantized.ptr(),
                result.row_scales.ptr(),
                i32::try_from(rows).map_err(|_| Error::Other("rows exceed i32".into()))?,
                i32::try_from(input_dim).map_err(|_| Error::Other("cols exceed i32".into()))?,
                ctx.stream().handle(),
            )
        } else {
            abi::apxinf_transformer_bf16_silu_mul_quantize_rows_bf16_int8(
                gate_buffer.ptr(),
                up_buffer.ptr(),
                result.quantized.ptr(),
                result.row_scales.ptr(),
                i32::try_from(rows).map_err(|_| Error::Other("rows exceed i32".into()))?,
                i32::try_from(input_dim).map_err(|_| Error::Other("cols exceed i32".into()))?,
                ctx.stream().handle(),
            )
        };
        raw::check_cuda(code).map_err(Error::Cuda)?;
    }
    Ok(result)
}

/// `gemm_quantized_w8a8`: INT8 GEMM over a pre-quantized activation and weight
/// with F32 row/column scales, dequantized to BF16.
pub fn gemm_quantized_w8a8(
    ctx: &CudaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
) -> Result<Tensor> {
    w8a8_gemm(ctx, activation, weight, None)
}

/// `w8a8`: BF16-activation W8A8 GEMM — quantize the activation rowwise, then
/// run the quantized GEMM. Mirrors the legacy `gemm_w8a8` entry.
pub fn w8a8(ctx: &CudaContext, activation: &Tensor, weight: W8A8WeightView<'_>) -> Result<Tensor> {
    let quantized = quantize_w8a8_activation(ctx, activation)?;
    gemm_quantized_w8a8(ctx, &quantized, weight)
}

fn w8a8_gemm(
    ctx: &CudaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    use crate::ffi::abi::transformer_bf16 as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    if activation.input_dim != weight.input_dim
        || weight.scale_mode != W8A8ScaleMode::DynamicRowPerOutputChannel
        || weight.layout != W8A8Layout::OutputMajor
        || weight.values_i8.len() != weight.input_dim * weight.output_dim
        || weight.scales_f32.dtype() != DType::F32
        || weight.scales_f32.shape().dims() != [weight.output_dim]
    {
        return Err(Error::Other(
            "W8A8 GEMM operand mismatch (row-quantized I8 activation, output-major I8 weight, F32 scales)"
                .into(),
        ));
    }
    let (rows, n, k) = (activation.rows, weight.output_dim, weight.input_dim);
    let output = ctx.allocate_output(Shape::new(vec![rows, n]), DType::BF16)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    let weight_scales = CudaBuffer::from_tensor(weight.scales_f32).map_err(Error::Cuda)?;

    // Preserve the legacy Orin path: the checkpoint weight is already in the
    // output-major layout consumed by the fused CUTLASS epilogue.  The generic
    // L3 contract is canonical [K,N] and would otherwise transpose/copy this
    // immutable weight before selecting the same kernel.  This compatibility
    // fast path stays in C++ and introduces no Rust-side tuning state.
    if ctx.caps().sm == 87 && k % 16 == 0 && n % 8 == 0 {
        let cutlass_result = unsafe {
            raw::check_cuda(abi::apxinf_transformer_bf16_cutlass_int8_gemm_bf16(
                activation.quantized.ptr(),
                weight.values_i8.ptr(),
                activation.row_scales.ptr(),
                weight_scales.ptr(),
                output_buffer.ptr(),
                i32::try_from(rows).map_err(|_| Error::Other("rows exceed i32".into()))?,
                i32::try_from(n).map_err(|_| Error::Other("cols exceed i32".into()))?,
                i32::try_from(k).map_err(|_| Error::Other("inner dimension exceeds i32".into()))?,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        };
        match cutlass_result {
            Ok(()) => return Ok(output),
            Err(error) => eprintln!(
                "[apxinf] fused Orin W8A8 GEMM failed for M{rows} N{n} K{k}: {error}; using vendor fallback"
            ),
        }
    }

    // Portable fallback: INT32 accumulators followed by row/channel dequant.
    let accumulators = crate::workspace::output_buffer(
        ctx,
        rows.checked_mul(n)
            .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
            .ok_or_else(|| Error::Other("W8A8 accumulator size overflow".into()))?,
    )?;
    ctx.cublas()
        .gemm_int8_i32(
            rows,
            n,
            k,
            &activation.quantized,
            weight.values_i8,
            &accumulators,
        )
        .map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_transformer_bf16_dequantize_int32_bf16(
            accumulators.ptr(),
            activation.row_scales.ptr(),
            weight_scales.ptr(),
            output_buffer.ptr(),
            i32::try_from(rows).map_err(|_| Error::Other("rows exceed i32".into()))?,
            i32::try_from(n).map_err(|_| Error::Other("cols exceed i32".into()))?,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    let _ = bias;
    Ok(output)
}

/// `try_fp8_bias_then_residual_bf16`: fixed-shape sm110 FP8 fusion. Not
/// vendored; callers fall back to the generic route.
#[allow(clippy::too_many_arguments)]
pub fn try_fp8_bias_then_residual_bf16(
    _ctx: &CudaContext,
    _activation: &Tensor,
    _activation_scale: f32,
    _weight: Fp8WeightView<'_>,
    _bias: &Tensor,
    _residual: &Tensor,
) -> Result<Option<Tensor>> {
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
pub fn try_gemm_w8a8_m41_n6144_k1536(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W8A8WeightView<'_>,
) -> Result<Option<Tensor>> {
    if ctx.caps().sm != 87
        || activation.shape().dims() != [41, 1536]
        || weight.input_dim != 1536
        || weight.output_dim != 6144
    {
        return Ok(None);
    }
    let activation = quantize_w8a8_activation(ctx, activation)?;
    try_gemm_quantized_w8a8_m41_n6144_k1536(ctx, &activation, weight)
}

#[allow(clippy::too_many_arguments)]
pub fn try_gemm_quantized_w8a8_m41_n6144_k1536(
    ctx: &CudaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
) -> Result<Option<Tensor>> {
    if ctx.caps().sm != 87
        || activation.rows != 41
        || activation.input_dim != 1536
        || weight.input_dim != 1536
        || weight.output_dim != 6144
    {
        return Ok(None);
    }
    validate_exact_w8a8_operands(ctx, activation, weight)?;
    let weight_scales = CudaBuffer::from_tensor(weight.scales_f32).map_err(Error::Cuda)?;
    let output = ctx.allocate_output(Shape::new(vec![41, 6144]), DType::BF16)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        crate::ffi::raw::cuda_runtime::check_cuda(
            crate::ffi::abi::transformer_bf16::apxinf_transformer_bf16_cutlass_int8_gemm_bf16_m41_n6144_k1536(
                activation.quantized.ptr(),
                weight.values_i8.ptr(),
                activation.row_scales.ptr(),
                weight_scales.ptr(),
                output_buffer.ptr(),
                41,
                6144,
                1536,
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)?;
    }
    Ok(Some(output))
}

#[allow(clippy::too_many_arguments)]
pub fn try_gemm_quantized_w8a8_bias(
    ctx: &CudaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
    bias: &Tensor,
) -> Result<Option<Tensor>> {
    if ctx.caps().sm != 87
        || activation.rows != 41
        || activation.input_dim != 1536
        || weight.input_dim != 1536
        || weight.output_dim != 4608
    {
        return Ok(None);
    }
    validate_exact_w8a8_operands(ctx, activation, weight)?;
    if bias.dtype() != DType::BF16
        || bias.device() != Device::Cuda(ctx.device_id())
        || bias.shape().dims() != [4608]
    {
        return Err(Error::Other(
            "W8A8 fused bias expects SM87 M41 N4608 K1536 and BF16 bias".into(),
        ));
    }
    let weight_scales = CudaBuffer::from_tensor(weight.scales_f32).map_err(Error::Cuda)?;
    let bias = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let output = ctx.allocate_output(Shape::new(vec![41, 4608]), DType::BF16)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        crate::ffi::raw::cuda_runtime::check_cuda(
            crate::ffi::abi::transformer_bf16::apxinf_transformer_bf16_cutlass_int8_gemm_bias_bf16_m41_n4608_k1536(
                activation.quantized.ptr(),
                weight.values_i8.ptr(),
                activation.row_scales.ptr(),
                weight_scales.ptr(),
                bias.ptr(),
                output_buffer.ptr(),
                41,
                4608,
                1536,
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)?;
    }
    Ok(Some(output))
}

#[allow(clippy::too_many_arguments)]
pub fn try_gemm_quantized_w8a8_bias_gelu_quantized(
    ctx: &CudaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
    bias: &Tensor,
) -> Result<Option<(Tensor, W8A8Activation)>> {
    const ROWS: usize = 41;
    const INPUT_DIM: usize = 1536;
    const OUTPUT_DIM: usize = 6144;
    if ctx.caps().sm != 87
        || activation.rows != ROWS
        || activation.input_dim != INPUT_DIM
        || weight.input_dim != INPUT_DIM
        || weight.output_dim != OUTPUT_DIM
    {
        return Ok(None);
    }
    validate_exact_w8a8_operands(ctx, activation, weight)?;
    if bias.dtype() != DType::BF16
        || bias.device() != Device::Cuda(ctx.device_id())
        || bias.shape().dims() != [OUTPUT_DIM]
    {
        return Err(Error::Other(
            "W8A8 bias-GELU fusion expects SM87 M41 N6144 K1536 and BF16 bias".into(),
        ));
    }

    let weight_scales = CudaBuffer::from_tensor(weight.scales_f32).map_err(Error::Cuda)?;
    let bias = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let activated = ctx.allocate_output(Shape::new(vec![ROWS, OUTPUT_DIM]), DType::BF16)?;
    let activated_buffer = CudaBuffer::from_tensor(&activated).map_err(Error::Cuda)?;
    unsafe {
        crate::ffi::raw::cuda_runtime::check_cuda(
            crate::ffi::abi::transformer_bf16::apxinf_transformer_bf16_cutlass_int8_gemm_bias_gelu_bf16_m41_n6144_k1536(
                activation.quantized.ptr(),
                weight.values_i8.ptr(),
                activation.row_scales.ptr(),
                weight_scales.ptr(),
                bias.ptr(),
                activated_buffer.ptr(),
                ROWS as i32,
                OUTPUT_DIM as i32,
                INPUT_DIM as i32,
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)?;
    }
    let quantized = quantize_w8a8_activation(ctx, &activated)?;
    Ok(Some((activated, quantized)))
}

fn validate_exact_w8a8_operands(
    ctx: &CudaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
) -> Result<()> {
    if activation.quantized.device() != ctx.device_id()
        || activation.quantized.len() != activation.rows * activation.input_dim
        || activation.row_scales.device() != ctx.device_id()
        || activation.row_scales.len() != activation.rows * std::mem::size_of::<f32>()
        || weight.scale_mode != W8A8ScaleMode::DynamicRowPerOutputChannel
        || weight.layout != W8A8Layout::OutputMajor
        || activation.input_dim != weight.input_dim
        || weight.values_i8.device() != ctx.device_id()
        || weight.values_i8.len() != weight.input_dim * weight.output_dim
        || weight.scales_f32.dtype() != DType::F32
        || weight.scales_f32.device() != Device::Cuda(ctx.device_id())
        || weight.scales_f32.shape().dims() != [weight.output_dim]
    {
        return Err(Error::Other(
            "exact W8A8 GEMM operand/device/layout mismatch".into(),
        ));
    }
    Ok(())
}

// ── Decode fast-path `write` surface (caller-owned buffers, graph-safe) ───

/// Row-major `A[M,K] @ B[K,N]` into caller-owned storage through raw cuBLAS.
/// No allocation, no tuned-operator session requirement — this is the same
/// direct `ctx.cublas().gemm(..)` dispatch the legacy decode graph used.
#[allow(clippy::too_many_arguments)]
pub fn write(
    ctx: &CudaContext,
    dtype: DType,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &CudaBuffer,
    b: &CudaBuffer,
    beta: f32,
    output: &CudaBuffer,
) -> Result<()> {
    require_finite("GEMM", &[alpha, beta])?;
    require_buffers(
        ctx,
        "GEMM",
        &[
            ("A", a, checked_bytes(dtype, &[m, k], "GEMM")?),
            ("B", b, checked_bytes(dtype, &[k, n], "GEMM")?),
            ("output", output, checked_bytes(dtype, &[m, n], "GEMM")?),
        ],
    )?;
    ctx.cublas()
        .gemm(dtype, m, n, k, alpha, a, b, beta, output)
        .map_err(Error::Cuda)
}
