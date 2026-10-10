//! Legacy `kernels::gemm` names over the cuda-new GEMM operator.

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

/// `matmul`: alias of [`bf16`] under the legacy generic name.
pub fn matmul(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    bf16(ctx, activation, weight)
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
    let mut output =
        ctx.allocate_output(Shape::new(vec![a_dims[0], b_dims[1]]), DType::BF16)?;
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

use super::contracts::{checked_bytes, require_buffers, require_finite};
use crate::cublas::CublasTranspose;
use crate::ffi::abi::vla_la as la_abi;
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
    super::linear_attention::cast_bf16_to_f32(ctx, bias, &accumulator)?;
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
    super::linear_attention::cast_f32_to_bf16(ctx, &accumulator, &output)?;
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
        raw::check_cuda(la_abi::apxinf_cn_broadcast_bf16_f32_rows(
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
    super::linear_attention::cast_f32_to_bf16(ctx, &accumulator, &output)?;
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

/// Compute SwiGLU from input `[M,K]` and gate-then-up weight `[2N,K]`.
/// The generic route: one BF16 projection, then the rounded SwiGLU kernel —
/// the same arithmetic as the legacy non-AOT path.
pub fn bf16_swiglu_checkpoint(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    let x = input.shape().dims();
    let w = weight.shape().dims();
    if x.len() != 2
        || w.len() != 2
        || x.contains(&0)
        || w.contains(&0)
        || x[1] != w[1]
        || w[0] % 2 != 0
    {
        return Err(Error::Other(
            "BF16 SwiGLU expects input[M,K], weight[2N,K]".into(),
        ));
    }
    for tensor in [input, weight] {
        if tensor.dtype() != DType::BF16 || tensor.device() != Device::Cuda(ctx.device_id()) {
            return Err(Error::Other(
                "BF16 SwiGLU requires BF16 inputs on the context device".into(),
            ));
        }
    }
    let k = i32::try_from(x[1]).map_err(|_| Error::Other("SwiGLU K exceeds i32".into()))?;
    let n2 = i32::try_from(w[0]).map_err(|_| Error::Other("SwiGLU width exceeds i32".into()))?;
    let rows = i32::try_from(x[0]).map_err(|_| Error::Other("SwiGLU rows exceed i32".into()))?;
    let inner = i32::try_from(w[0] / 2).map_err(|_| Error::Other("SwiGLU inner exceeds i32".into()))?;
    let xp = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let wp = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let projection = crate::workspace::output_buffer(
        ctx,
        checked_bytes(DType::BF16, &[x[0], w[0]], "SwiGLU projection")?,
    )?;
    ctx.cublas()
        .gemm_ex(
            DType::BF16,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            x[0],
            w[0],
            x[1],
            1.0,
            &xp,
            k,
            &wp,
            k,
            0.0,
            &projection,
            n2,
        )
        .map_err(Error::Cuda)?;
    let output = crate::workspace::output_buffer(
        ctx,
        checked_bytes(DType::BF16, &[x[0], w[0] / 2], "SwiGLU output")?,
    )?;
    unsafe {
        raw::check_cuda(la_abi::apxinf_cn_swiglu_bf16_rounded(
            projection.ptr(),
            output.ptr(),
            rows,
            inner,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output.into_tensor(Shape::new(vec![x[0], w[0] / 2]), DType::BF16))
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
