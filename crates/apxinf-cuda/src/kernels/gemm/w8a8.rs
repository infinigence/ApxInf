use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use half::bf16;

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::device_caps::CudaArchFamily;
use crate::ffi;
use crate::tuning::{
    AutoTuneConfig, AutoTuneEngine, CandidateMeasurement, DeviceFingerprint, Epilogue, GemmLayout,
    GemmOp, GemmTuningKey, ScaleMode, TacticBackend, TacticId, TuningDType, TuningOutcome,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W8A8ScaleMode {
    DynamicRowPerOutputChannel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W8A8Layout {
    OutputMajor,
}

/// Borrowed W8A8 weight view prevents values/scales/layout mismatches.
#[derive(Clone, Copy)]
pub struct W8A8WeightView<'a> {
    pub values_i8: &'a CudaBuffer,
    pub scales_f32: &'a Tensor,
    pub input_dim: usize,
    pub output_dim: usize,
    pub scale_mode: W8A8ScaleMode,
    pub layout: W8A8Layout,
}

/// Dynamically row-quantized activation that can be shared by projections
/// consuming the same BF16 input (for example, transformer gate/up).
pub struct W8A8Activation {
    quantized: CudaBuffer,
    row_scales: CudaBuffer,
    rows: usize,
    input_dim: usize,
}

#[cfg(test)]
impl W8A8Activation {
    pub(crate) fn quantized_bytes(&self) -> Result<Vec<u8>> {
        let mut values = vec![0u8; self.rows * self.input_dim];
        self.quantized
            .copy_to_host(&mut values)
            .map_err(Error::Cuda)?;
        Ok(values)
    }

    pub(crate) fn row_scale_bits(&self) -> Result<Vec<u32>> {
        let mut bytes = vec![0u8; self.rows * std::mem::size_of::<f32>()];
        self.row_scales
            .copy_to_host(&mut bytes)
            .map_err(Error::Cuda)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|value| u32::from_ne_bytes(value.try_into().unwrap()))
            .collect())
    }
}

pub fn quantize_w8a8_activation(ctx: &CudaContext, activation: &Tensor) -> Result<W8A8Activation> {
    if activation.dtype() != DType::BF16 || activation.device() != Device::Cuda(ctx.device_id()) {
        return Err(Error::Other(format!(
            "W8A8 quantization expects a BF16 activation on CUDA {}, got {} on {}",
            ctx.device_id(),
            activation.dtype(),
            activation.device()
        )));
    }
    let dims = activation.shape().dims();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(Error::Other(format!(
            "W8A8 quantization expects a non-empty matrix, got {dims:?}"
        )));
    }
    let (rows, input_dim) = (dims[0], dims[1]);
    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let quantized = crate::workspace::output_buffer(ctx, rows * input_dim)?;
    let row_scales = crate::workspace::output_buffer(ctx, rows * std::mem::size_of::<f32>())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_quantize_rows_bf16_int8(
            activation.ptr(),
            quantized.ptr(),
            row_scales.ptr(),
            rows as i32,
            input_dim as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(W8A8Activation {
        quantized,
        row_scales,
        rows,
        input_dim,
    })
}

pub fn bias_gelu_quantize_w8a8_activation(
    ctx: &CudaContext,
    activation: &Tensor,
    bias: &Tensor,
) -> Result<W8A8Activation> {
    if activation.dtype() != DType::BF16
        || bias.dtype() != DType::BF16
        || activation.device() != Device::Cuda(ctx.device_id())
        || bias.device() != Device::Cuda(ctx.device_id())
    {
        return Err(Error::Other(
            "W8A8 fused bias-GELU quantization expects BF16 CUDA tensors".into(),
        ));
    }
    let dims = activation.shape().dims();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 || bias.shape().dims() != [dims[1]] {
        return Err(Error::Other(format!(
            "W8A8 fused bias-GELU quantization expects [rows,cols] input and [cols] bias, got {dims:?} and {:?}",
            bias.shape().dims()
        )));
    }
    let (rows, input_dim) = (dims[0], dims[1]);
    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let bias = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let quantized = crate::workspace::output_buffer(ctx, rows * input_dim)?;
    let row_scales = crate::workspace::output_buffer(ctx, rows * std::mem::size_of::<f32>())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_gelu_quantize_rows_bf16_int8(
            activation.ptr(),
            bias.ptr(),
            quantized.ptr(),
            row_scales.ptr(),
            rows as i32,
            input_dim as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(W8A8Activation {
        quantized,
        row_scales,
        rows,
        input_dim,
    })
}

pub fn adaptive_layer_norm_quantize_w8a8_activation(
    ctx: &CudaContext,
    input: &Tensor,
    modulation: &Tensor,
    eps: f32,
) -> Result<(Tensor, W8A8Activation)> {
    if input.dtype() != DType::BF16
        || modulation.dtype() != DType::BF16
        || input.device() != Device::Cuda(ctx.device_id())
        || modulation.device() != Device::Cuda(ctx.device_id())
        || !(eps > 0.0)
    {
        return Err(Error::Other(
            "W8A8 adaptive LayerNorm quantization expects BF16 CUDA tensors and positive epsilon"
                .into(),
        ));
    }
    let dims = input.shape().dims();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 || modulation.shape().dims() != [2 * dims[1]]
    {
        return Err(Error::Other(format!(
            "W8A8 adaptive LayerNorm quantization expects [rows,cols] input and [2*cols] modulation, got {dims:?} and {:?}",
            modulation.shape().dims()
        )));
    }
    let (rows, input_dim) = (dims[0], dims[1]);
    let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let modulation_buffer = CudaBuffer::from_tensor(modulation).map_err(Error::Cuda)?;
    let output =
        crate::workspace::output_buffer(ctx, rows * input_dim * DType::BF16.size_in_bytes())?;
    let quantized = crate::workspace::output_buffer(ctx, rows * input_dim)?;
    let row_scales = crate::workspace::output_buffer(ctx, rows * std::mem::size_of::<f32>())?;
    unsafe {
        ffi::check_cuda(
            ffi::apxinf_static_adaptive_layer_norm_quantize_rows_bf16_int8(
                input_buffer.ptr(),
                modulation_buffer.ptr(),
                output.ptr(),
                quantized.ptr(),
                row_scales.ptr(),
                rows as i32,
                input_dim as i32,
                eps,
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)?;
    }
    Ok((
        output.into_tensor(Shape::new(vec![rows, input_dim]), DType::BF16),
        W8A8Activation {
            quantized,
            row_scales,
            rows,
            input_dim,
        },
    ))
}

pub fn layer_norm_quantize_w8a8_activation(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<(Tensor, W8A8Activation)> {
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || bias.dtype() != DType::BF16
        || input.device() != Device::Cuda(ctx.device_id())
        || weight.device() != Device::Cuda(ctx.device_id())
        || bias.device() != Device::Cuda(ctx.device_id())
        || !(eps > 0.0)
    {
        return Err(Error::Other(
            "W8A8 LayerNorm quantization expects BF16 CUDA tensors and positive epsilon".into(),
        ));
    }
    let dims = input.shape().dims();
    if dims.len() != 2
        || dims[0] == 0
        || dims[1] == 0
        || weight.shape().dims() != [dims[1]]
        || bias.shape().dims() != [dims[1]]
    {
        return Err(Error::Other(format!(
            "W8A8 LayerNorm quantization expects [rows,cols] input and [cols] affine tensors, got {dims:?}, {:?}, and {:?}",
            weight.shape().dims(),
            bias.shape().dims()
        )));
    }
    let (rows, input_dim) = (dims[0], dims[1]);
    let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let output =
        crate::workspace::output_buffer(ctx, rows * input_dim * DType::BF16.size_in_bytes())?;
    let quantized = crate::workspace::output_buffer(ctx, rows * input_dim)?;
    let row_scales = crate::workspace::output_buffer(ctx, rows * std::mem::size_of::<f32>())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_layer_norm_quantize_rows_bf16_int8(
            input_buffer.ptr(),
            weight_buffer.ptr(),
            bias_buffer.ptr(),
            output.ptr(),
            quantized.ptr(),
            row_scales.ptr(),
            rows as i32,
            input_dim as i32,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok((
        output.into_tensor(Shape::new(vec![rows, input_dim]), DType::BF16),
        W8A8Activation {
            quantized,
            row_scales,
            rows,
            input_dim,
        },
    ))
}

pub fn quantize_w8a8_silu_mul_activation(
    ctx: &CudaContext,
    gate: &Tensor,
    up: &Tensor,
) -> Result<W8A8Activation> {
    if gate.dtype() != DType::BF16
        || up.dtype() != DType::BF16
        || gate.device() != Device::Cuda(ctx.device_id())
        || up.device() != Device::Cuda(ctx.device_id())
        || gate.shape() != up.shape()
    {
        return Err(Error::Other(format!(
            "W8A8 fused SiLU-mul quantization expects equal BF16 CUDA tensors, got {} {:?} and {} {:?}",
            gate.dtype(),
            gate.shape().dims(),
            up.dtype(),
            up.shape().dims()
        )));
    }
    let dims = gate.shape().dims();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(Error::Other(format!(
            "W8A8 fused SiLU-mul quantization expects a non-empty matrix, got {dims:?}"
        )));
    }
    let (rows, input_dim) = (dims[0], dims[1]);
    let gate = CudaBuffer::from_tensor(gate).map_err(Error::Cuda)?;
    let up = CudaBuffer::from_tensor(up).map_err(Error::Cuda)?;
    let quantized = crate::workspace::output_buffer(ctx, rows * input_dim)?;
    let row_scales = crate::workspace::output_buffer(ctx, rows * std::mem::size_of::<f32>())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_silu_mul_quantize_rows_bf16_int8(
            gate.ptr(),
            up.ptr(),
            quantized.ptr(),
            row_scales.ptr(),
            rows as i32,
            input_dim as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(W8A8Activation {
        quantized,
        row_scales,
        rows,
        input_dim,
    })
}

/// Opt-in packed implementation of the bit-compatible BF16 SiLU×up dynamic
/// INT8 quantizer. Public/default dispatch deliberately remains on
/// [`quantize_w8a8_silu_mul_activation`].
pub fn quantize_w8a8_silu_mul_activation_packed4(
    ctx: &CudaContext,
    gate: &Tensor,
    up: &Tensor,
) -> Result<W8A8Activation> {
    if gate.dtype() != DType::BF16
        || up.dtype() != DType::BF16
        || gate.device() != Device::Cuda(ctx.device_id())
        || up.device() != Device::Cuda(ctx.device_id())
        || gate.shape() != up.shape()
    {
        return Err(Error::Other(format!(
            "W8A8 packed SiLU-mul quantization expects equal BF16 CUDA tensors, got {} {:?} and {} {:?}",
            gate.dtype(),
            gate.shape().dims(),
            up.dtype(),
            up.shape().dims()
        )));
    }
    let dims = gate.shape().dims();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 || dims[1] % 4 != 0 {
        return Err(Error::Other(format!(
            "W8A8 packed SiLU-mul quantization expects a non-empty [rows,cols] matrix with cols divisible by four, got {dims:?}"
        )));
    }
    let (rows, input_dim) = (dims[0], dims[1]);
    let gate = CudaBuffer::from_tensor(gate).map_err(Error::Cuda)?;
    let up = CudaBuffer::from_tensor(up).map_err(Error::Cuda)?;
    let quantized = crate::workspace::output_buffer(ctx, rows * input_dim)?;
    let row_scales = crate::workspace::output_buffer(ctx, rows * std::mem::size_of::<f32>())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_silu_mul_quantize_rows_bf16_int8_packed4(
            gate.ptr(),
            up.ptr(),
            quantized.ptr(),
            row_scales.ptr(),
            rows as i32,
            input_dim as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(W8A8Activation {
        quantized,
        row_scales,
        rows,
        input_dim,
    })
}

struct CudaEventPair {
    start: ffi::cudaEvent_t,
    stop: ffi::cudaEvent_t,
}

impl CudaEventPair {
    fn new() -> Result<Self> {
        let mut events = Self {
            start: std::ptr::null_mut(),
            stop: std::ptr::null_mut(),
        };
        unsafe {
            ffi::check_cuda(ffi::cudaEventCreate(&mut events.start)).map_err(Error::Cuda)?;
            if let Err(error) = ffi::check_cuda(ffi::cudaEventCreate(&mut events.stop)) {
                let _ = ffi::cudaEventDestroy(events.start);
                return Err(Error::Cuda(error));
            }
        }
        Ok(events)
    }

    fn measure(&self, ctx: &CudaContext, launch: impl FnOnce() -> Result<()>) -> Result<f64> {
        unsafe {
            ffi::check_cuda(ffi::cudaEventRecord(self.start, ctx.stream().handle()))
                .map_err(Error::Cuda)?;
        }
        launch()?;
        let mut milliseconds = 0.0f32;
        unsafe {
            ffi::check_cuda(ffi::cudaEventRecord(self.stop, ctx.stream().handle()))
                .map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventSynchronize(self.stop)).map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventElapsedTime(
                &mut milliseconds,
                self.start,
                self.stop,
            ))
            .map_err(Error::Cuda)?;
        }
        Ok(f64::from(milliseconds))
    }
}

impl Drop for CudaEventPair {
    fn drop(&mut self) {
        unsafe {
            if !self.start.is_null() {
                let _ = ffi::cudaEventDestroy(self.start);
            }
            if !self.stop.is_null() {
                let _ = ffi::cudaEventDestroy(self.stop);
            }
        }
    }
}

fn tuning_key(ctx: &CudaContext, m: usize, n: usize, k: usize) -> GemmTuningKey {
    GemmTuningKey {
        op: GemmOp::W8A8,
        device: DeviceFingerprint::from(ctx.caps()),
        m,
        n,
        k,
        activation_dtype: TuningDType::I8,
        weight_dtype: TuningDType::I8,
        output_dtype: TuningDType::Bf16,
        layout: GemmLayout::WeightOutputMajor,
        scale_mode: ScaleMode::DynamicRowPerOutputChannel,
        epilogue: Epilogue::None,
        workspace_limit: usize::MAX,
    }
}

#[cfg(test)]
pub(crate) fn w8a8_tuning_key_for_test(
    ctx: &CudaContext,
    m: usize,
    n: usize,
    k: usize,
) -> GemmTuningKey {
    tuning_key(ctx, m, n, k)
}

fn copy_bf16_output(output: &CudaBuffer, elements: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; elements * DType::BF16.size_in_bytes()];
    output.copy_to_host(&mut bytes).map_err(Error::Cuda)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|value| bf16::from_bits(u16::from_ne_bytes([value[0], value[1]])).to_f32())
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn launch_w8a8_tactic(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    quantized: &CudaBuffer,
    weights: &CudaBuffer,
    row_scales: &CudaBuffer,
    weight_scales: &CudaBuffer,
    accumulators: &CudaBuffer,
    output: &CudaBuffer,
    tactic: TacticId,
) -> Result<()> {
    super::providers::prepare(key, tactic)?;
    match tactic.backend {
        TacticBackend::Vendor => {
            ctx.cublas()
                .gemm_int8_i32(key.m, key.n, key.k, quantized, weights, accumulators)
                .map_err(Error::Cuda)?;
            unsafe {
                ffi::check_cuda(ffi::apxinf_static_dequantize_int32_bf16(
                    accumulators.ptr(),
                    row_scales.ptr(),
                    weight_scales.ptr(),
                    output.ptr(),
                    key.m as i32,
                    key.n as i32,
                    ctx.stream().handle(),
                ))
                .map_err(Error::Cuda)
            }
        }
        TacticBackend::Cutlass => {
            #[cfg(apxinf_cutlass_int8_sm80)]
            unsafe {
                ffi::check_cuda(ffi::apxinf_static_cutlass_int8_gemm_bf16(
                    quantized.ptr(),
                    weights.ptr(),
                    row_scales.ptr(),
                    weight_scales.ptr(),
                    output.ptr(),
                    key.m as i32,
                    key.n as i32,
                    key.k as i32,
                    ctx.stream().handle(),
                ))
                .map_err(Error::Cuda)
            }
            #[cfg(not(apxinf_cutlass_int8_sm80))]
            {
                Err(Error::Other(
                    "CUTLASS W8A8 autotune requires an SM80-family build".into(),
                ))
            }
        }
        _ => Err(Error::Other(format!(
            "W8A8 online autotune cannot execute {tactic:?}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn autotune_request_w8a8(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    quantized: &CudaBuffer,
    weights: &CudaBuffer,
    row_scales: &CudaBuffer,
    weight_scales: &CudaBuffer,
    preferred: Option<TacticId>,
) -> Result<TuningOutcome> {
    let elements = key
        .m
        .checked_mul(key.n)
        .ok_or_else(|| Error::Other("W8A8 autotune output size overflow".into()))?;
    let output_bytes = elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("W8A8 autotune output size overflow".into()))?;
    let accumulator_bytes = elements
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| Error::Other("W8A8 autotune accumulator size overflow".into()))?;
    let reference_output =
        CudaBuffer::alloc_zeros(output_bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let reference_accumulators =
        CudaBuffer::alloc_zeros(accumulator_bytes, ctx.device_id()).map_err(Error::Cuda)?;
    launch_w8a8_tactic(
        ctx,
        key,
        quantized,
        weights,
        row_scales,
        weight_scales,
        &reference_accumulators,
        &reference_output,
        TacticId {
            backend: TacticBackend::Vendor,
            value: 0,
        },
    )?;
    ctx.synchronize().map_err(Error::Cuda)?;
    let reference = copy_bf16_output(&reference_output, elements)?;
    drop((reference_output, reference_accumulators));

    let output = CudaBuffer::alloc_zeros(output_bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let accumulators =
        CudaBuffer::alloc_zeros(accumulator_bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let events = CudaEventPair::new()?;
    let engine = AutoTuneEngine::new(AutoTuneConfig::default())?;
    let candidates = super::providers::candidates(key, 0);
    engine.tune_with_preferred(key, preferred, candidates, |candidate, config| {
        launch_w8a8_tactic(
            ctx,
            key,
            quantized,
            weights,
            row_scales,
            weight_scales,
            &accumulators,
            &output,
            candidate.tactic,
        )?;
        ctx.synchronize().map_err(Error::Cuda)?;
        let actual = copy_bf16_output(&output, elements)?;
        let correct = crate::tuning::outputs_are_close(&reference, &actual, 0.01, 0.9999);
        if !correct {
            return Ok(CandidateMeasurement {
                tactic: candidate.tactic,
                milliseconds: None,
                correct: false,
            });
        }
        for _ in 0..config.warmup_iterations {
            launch_w8a8_tactic(
                ctx,
                key,
                quantized,
                weights,
                row_scales,
                weight_scales,
                &accumulators,
                &output,
                candidate.tactic,
            )?;
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        let mut milliseconds = 0.0;
        for _ in 0..config.benchmark_iterations {
            milliseconds += events.measure(ctx, || {
                launch_w8a8_tactic(
                    ctx,
                    key,
                    quantized,
                    weights,
                    row_scales,
                    weight_scales,
                    &accumulators,
                    &output,
                    candidate.tactic,
                )
            })?;
        }
        Ok(CandidateMeasurement {
            tactic: candidate.tactic,
            milliseconds: Some(milliseconds / config.benchmark_iterations as f64),
            correct: true,
        })
    })
}

/// Dynamic-row-quantized W8A8 GEMM with BF16 output.
pub fn gemm_w8a8(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W8A8WeightView<'_>,
) -> Result<Tensor> {
    // The plan resolver validates the default before the launch-time fallback.
    // Match the quantized-input path's eligibility, including patch K=588.
    let prefer_cutlass = cfg!(apxinf_cutlass_int8_sm80)
        && matches!(ctx.caps().arch_family, CudaArchFamily::Sm80)
        && weight.input_dim % 16 == 0
        && weight.output_dim % 8 == 0;
    gemm_w8a8_impl(ctx, activation, weight, Some(prefer_cutlass), false)
}

#[cfg(test)]
pub(crate) fn gemm_w8a8_with_preference(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W8A8WeightView<'_>,
    prefer_cutlass: bool,
) -> Result<Tensor> {
    gemm_w8a8_impl(ctx, activation, weight, Some(prefer_cutlass), true)
}

fn gemm_w8a8_impl(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W8A8WeightView<'_>,
    default_cutlass: Option<bool>,
    force_preference: bool,
) -> Result<Tensor> {
    if activation.dtype() != DType::BF16 || activation.device() != Device::Cuda(ctx.device_id()) {
        return Err(Error::Other(format!(
            "gemm_w8a8 expects a BF16 activation on CUDA {}, got {} on {}",
            ctx.device_id(),
            activation.dtype(),
            activation.device()
        )));
    }
    if weight.scale_mode != W8A8ScaleMode::DynamicRowPerOutputChannel
        || weight.layout != W8A8Layout::OutputMajor
    {
        return Err(Error::Other(
            "gemm_w8a8 received an unsupported scale mode or layout".into(),
        ));
    }
    let dims = activation.shape().dims();
    if dims.len() != 2 || dims[1] != weight.input_dim {
        return Err(Error::Other(format!(
            "gemm_w8a8 activation shape mismatch: expected [M,{}], got {dims:?}",
            weight.input_dim
        )));
    }
    if weight.values_i8.device() != ctx.device_id()
        || weight.values_i8.len() != weight.input_dim * weight.output_dim
        || weight.scales_f32.dtype() != DType::F32
        || weight.scales_f32.device() != Device::Cuda(ctx.device_id())
        || weight.scales_f32.shape().dims() != [weight.output_dim]
    {
        return Err(Error::Other(format!(
            "gemm_w8a8 weight contract mismatch: bytes {}, scales {} {:?}, expected [{},{}] on CUDA {}",
            weight.values_i8.len(),
            weight.scales_f32.dtype(),
            weight.scales_f32.shape().dims(),
            weight.output_dim,
            weight.input_dim,
            ctx.device_id()
        )));
    }

    let rows = dims[0];
    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_scales = CudaBuffer::from_tensor(weight.scales_f32).map_err(Error::Cuda)?;
    let quantized = crate::workspace::output_buffer(ctx, rows * weight.input_dim)?;
    let row_scales = crate::workspace::output_buffer(ctx, rows * std::mem::size_of::<f32>())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_quantize_rows_bf16_int8(
            activation.ptr(),
            quantized.ptr(),
            row_scales.ptr(),
            rows as i32,
            weight.input_dim as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }

    let output = crate::workspace::output_buffer(
        ctx,
        rows * weight.output_dim * DType::BF16.size_in_bytes(),
    )?;
    let key = tuning_key(ctx, rows, weight.output_dim, weight.input_dim);
    let default = TacticId {
        backend: if default_cutlass.unwrap_or(false) {
            TacticBackend::Cutlass
        } else {
            TacticBackend::Vendor
        },
        value: 0,
    };
    let selected = if force_preference {
        default
    } else {
        ctx.gemm_plans()
            .resolve_or_tune(ctx, &key, default, |preferred| {
                autotune_request_w8a8(
                    ctx,
                    &key,
                    &quantized,
                    weight.values_i8,
                    &row_scales,
                    &weight_scales,
                    preferred,
                )
            })?
            .tactic
    };
    #[cfg(not(apxinf_cutlass_int8_sm80))]
    let _ = selected;
    #[cfg(apxinf_cutlass_int8_sm80)]
    if selected.backend == TacticBackend::Cutlass
        && weight.input_dim % 16 == 0
        && weight.output_dim % 8 == 0
    {
        let cutlass_result = unsafe {
            ffi::check_cuda(ffi::apxinf_static_cutlass_int8_gemm_bf16(
                quantized.ptr(),
                weight.values_i8.ptr(),
                row_scales.ptr(),
                weight_scales.ptr(),
                output.ptr(),
                rows as i32,
                weight.output_dim as i32,
                weight.input_dim as i32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        };
        match cutlass_result {
            Ok(()) => {
                return Ok(
                    output.into_tensor(Shape::new(vec![rows, weight.output_dim]), DType::BF16)
                )
            }
            Err(error) => {
                eprintln!(
                    "[apxinf] W8A8 CUTLASS tactic failed for {key:?}: {error}; using vendor fallback"
                );
                ctx.gemm_plans().fallback(ctx, &key)?;
            }
        }
    }

    let accumulators = crate::workspace::output_buffer(
        ctx,
        rows * weight.output_dim * std::mem::size_of::<i32>(),
    )?;
    ctx.cublas()
        .gemm_int8_i32(
            rows,
            weight.output_dim,
            weight.input_dim,
            &quantized,
            weight.values_i8,
            &accumulators,
        )
        .map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_dequantize_int32_bf16(
            accumulators.ptr(),
            row_scales.ptr(),
            weight_scales.ptr(),
            output.ptr(),
            rows as i32,
            weight.output_dim as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output.into_tensor(Shape::new(vec![rows, weight.output_dim]), DType::BF16))
}

pub fn gemm_quantized_w8a8(
    ctx: &CudaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
) -> Result<Tensor> {
    let prefer_cutlass = cfg!(apxinf_cutlass_int8_sm80)
        && matches!(ctx.caps().arch_family, CudaArchFamily::Sm80)
        && weight.input_dim % 16 == 0
        && weight.output_dim % 8 == 0;
    gemm_quantized_w8a8_impl(ctx, activation, weight, Some(prefer_cutlass), false)
}

fn supports_sm87_m41_n6144_k1536(sm: u32, m: usize, n: usize, k: usize) -> bool {
    cfg!(apxinf_cutlass_int8_sm80) && sm == 87 && (m, n, k) == (41, 6144, 1536)
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
        return Err(Error::Other(format!(
            "exact W8A8 GEMM expects row-quantized I8 activation, output-major I8 weights and F32 row/column scales on CUDA {}",
            ctx.device_id()
        )));
    }
    Ok(())
}

/// Explicit SM87 M41 N6144 K1536 schedule for a BF16 activation.
/// Unsupported shapes/builds return `None`; operand and execution errors
/// propagate. This does not participate in generic GEMM tactic selection.
pub fn try_gemm_w8a8_m41_n6144_k1536(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W8A8WeightView<'_>,
) -> Result<Option<Tensor>> {
    if !supports_sm87_m41_n6144_k1536(ctx.caps().sm, 41, weight.output_dim, weight.input_dim)
        || activation.shape().dims() != [41, 1536]
    {
        return Ok(None);
    }
    let activation = quantize_w8a8_activation(ctx, activation)?;
    try_gemm_quantized_w8a8_m41_n6144_k1536(ctx, &activation, weight)
}

/// Explicit SM87 M41 N6144 K1536 schedule for an already quantized activation.
/// Its 64x128x128 tile, 32x64x64 warp and three stages preserve the schedule
/// formerly identified as tactic six in an external database. The caller
/// opts in directly; the shared provider and plan cache retain their defaults.
pub fn try_gemm_quantized_w8a8_m41_n6144_k1536(
    ctx: &CudaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
) -> Result<Option<Tensor>> {
    if !supports_sm87_m41_n6144_k1536(
        ctx.caps().sm,
        activation.rows,
        weight.output_dim,
        weight.input_dim,
    ) || activation.input_dim != 1536
    {
        return Ok(None);
    }
    validate_exact_w8a8_operands(ctx, activation, weight)?;
    let weight_scales = CudaBuffer::from_tensor(weight.scales_f32).map_err(Error::Cuda)?;
    let output = crate::workspace::output_buffer(ctx, 41 * 6144 * DType::BF16.size_in_bytes())?;
    #[cfg(apxinf_cutlass_int8_sm80)]
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_cutlass_int8_gemm_bf16_m41_n6144_k1536(
            activation.quantized.ptr(),
            weight.values_i8.ptr(),
            activation.row_scales.ptr(),
            weight_scales.ptr(),
            output.ptr(),
            41,
            6144,
            1536,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(Some(
        output.into_tensor(Shape::new(vec![41, 6144]), DType::BF16),
    ))
}

/// Try the fixed-shape W8A8 projection with separately rounded BF16 bias.
///
/// Returns `None` when the optional INT8 CUTLASS backend was not compiled.
/// Otherwise the strict SM87 M41 N4608 K1536 operand contract and execution
/// errors are preserved.
pub fn try_gemm_quantized_w8a8_bias(
    ctx: &CudaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
    bias: &Tensor,
) -> Result<Option<Tensor>> {
    if !cfg!(apxinf_cutlass_int8_sm80) {
        return Ok(None);
    }
    gemm_quantized_w8a8_bias_exact_qkv(ctx, activation, weight, bias).map(Some)
}

/// Explicit SM87 M41 N4608 K1536 projection with a BF16 rounding boundary
/// before bias addition.
pub fn gemm_quantized_w8a8_bias_exact_qkv(
    ctx: &CudaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
    bias: &Tensor,
) -> Result<Tensor> {
    if ctx.caps().sm != 87
        || activation.rows != 41
        || activation.input_dim != 1536
        || weight.input_dim != 1536
        || weight.output_dim != 4608
        || bias.dtype() != DType::BF16
        || bias.device() != Device::Cuda(ctx.device_id())
        || bias.shape().dims() != [4608]
    {
        return Err(Error::Other(format!(
            "W8A8 fused bias expects SM87 M41 N4608 K1536 and BF16 bias, got M{} N{} K{} and {} {:?}",
            activation.rows,
            weight.output_dim,
            weight.input_dim,
            bias.dtype(),
            bias.shape().dims()
        )));
    }
    validate_exact_w8a8_operands(ctx, activation, weight)?;
    let weight_scales = CudaBuffer::from_tensor(weight.scales_f32).map_err(Error::Cuda)?;
    let bias = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let output = crate::workspace::output_buffer(
        ctx,
        activation.rows * weight.output_dim * DType::BF16.size_in_bytes(),
    )?;
    #[cfg(apxinf_cutlass_int8_sm80)]
    unsafe {
        ffi::check_cuda(
            ffi::apxinf_static_cutlass_int8_gemm_bias_bf16_m41_n4608_k1536(
                activation.quantized.ptr(),
                weight.values_i8.ptr(),
                activation.row_scales.ptr(),
                weight_scales.ptr(),
                bias.ptr(),
                output.ptr(),
                activation.rows as i32,
                weight.output_dim as i32,
                weight.input_dim as i32,
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)?;
    }
    #[cfg(not(apxinf_cutlass_int8_sm80))]
    return Err(Error::Other(
        "W8A8 fused bias requires an SM80-family CUTLASS build".into(),
    ));
    #[cfg(apxinf_cutlass_int8_sm80)]
    Ok(output.into_tensor(
        Shape::new(vec![activation.rows, weight.output_dim]),
        DType::BF16,
    ))
}

/// Explicit SM87 M41 N6144 K1536 GEMM, bias, GELU and row quantization.
/// Selection belongs to the caller. This fused operator uses its own fixed
/// schedule and does not resolve or populate the plain GEMM tactic cache.
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
    if !cfg!(apxinf_cutlass_int8_sm80) {
        return Ok(None);
    }
    validate_exact_w8a8_operands(ctx, activation, weight)?;
    if bias.dtype() != DType::BF16
        || bias.device() != Device::Cuda(ctx.device_id())
        || bias.shape().dims() != [OUTPUT_DIM]
    {
        return Err(Error::Other(format!(
            "W8A8 bias-GELU fusion expects M41 N6144 K1536 output-major weights, F32 column scales, and BF16 bias on CUDA {}, got weight bytes {}, scales {} {:?}, and bias {} {:?}",
            ctx.device_id(),
            weight.values_i8.len(),
            weight.scales_f32.dtype(),
            weight.scales_f32.shape().dims(),
            bias.dtype(),
            bias.shape().dims()
        )));
    }

    let weight_scales = CudaBuffer::from_tensor(weight.scales_f32).map_err(Error::Cuda)?;
    let bias = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let activated =
        crate::workspace::output_buffer(ctx, ROWS * OUTPUT_DIM * DType::BF16.size_in_bytes())?;
    let quantized = crate::workspace::output_buffer(ctx, ROWS * OUTPUT_DIM)?;
    let row_scales = crate::workspace::output_buffer(ctx, ROWS * std::mem::size_of::<f32>())?;

    #[cfg(apxinf_cutlass_int8_sm80)]
    unsafe {
        ffi::check_cuda(
            ffi::apxinf_static_cutlass_int8_gemm_bias_gelu_bf16_m41_n6144_k1536(
                activation.quantized.ptr(),
                weight.values_i8.ptr(),
                activation.row_scales.ptr(),
                weight_scales.ptr(),
                bias.ptr(),
                activated.ptr(),
                ROWS as i32,
                OUTPUT_DIM as i32,
                INPUT_DIM as i32,
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)?;
        ffi::check_cuda(ffi::apxinf_static_quantize_rows_bf16_int8_packed4(
            activated.ptr(),
            quantized.ptr(),
            row_scales.ptr(),
            ROWS as i32,
            OUTPUT_DIM as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }

    Ok(Some((
        activated.into_tensor(Shape::new(vec![ROWS, OUTPUT_DIM]), DType::BF16),
        W8A8Activation {
            quantized,
            row_scales,
            rows: ROWS,
            input_dim: OUTPUT_DIM,
        },
    )))
}

fn gemm_quantized_w8a8_impl(
    ctx: &CudaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
    default_cutlass: Option<bool>,
    force_preference: bool,
) -> Result<Tensor> {
    if weight.scale_mode != W8A8ScaleMode::DynamicRowPerOutputChannel
        || weight.layout != W8A8Layout::OutputMajor
    {
        return Err(Error::Other(
            "gemm_w8a8 received an unsupported scale mode or layout".into(),
        ));
    }
    if activation.input_dim != weight.input_dim {
        return Err(Error::Other(format!(
            "gemm_w8a8 activation width mismatch: expected {}, got {}",
            weight.input_dim, activation.input_dim
        )));
    }
    if weight.values_i8.device() != ctx.device_id()
        || weight.values_i8.len() != weight.input_dim * weight.output_dim
        || weight.scales_f32.dtype() != DType::F32
        || weight.scales_f32.device() != Device::Cuda(ctx.device_id())
        || weight.scales_f32.shape().dims() != [weight.output_dim]
    {
        return Err(Error::Other(format!(
            "gemm_w8a8 weight contract mismatch: bytes {}, scales {} {:?}, expected [{},{}] on CUDA {}",
            weight.values_i8.len(),
            weight.scales_f32.dtype(),
            weight.scales_f32.shape().dims(),
            weight.output_dim,
            weight.input_dim,
            ctx.device_id()
        )));
    }

    let rows = activation.rows;
    let weight_scales = CudaBuffer::from_tensor(weight.scales_f32).map_err(Error::Cuda)?;
    let output = crate::workspace::output_buffer(
        ctx,
        rows * weight.output_dim * DType::BF16.size_in_bytes(),
    )?;
    let key = tuning_key(ctx, rows, weight.output_dim, weight.input_dim);
    let default = TacticId {
        backend: if default_cutlass.unwrap_or(false) {
            TacticBackend::Cutlass
        } else {
            TacticBackend::Vendor
        },
        value: 0,
    };
    let selected = if force_preference {
        default
    } else {
        ctx.gemm_plans()
            .resolve_or_tune(ctx, &key, default, |preferred| {
                autotune_request_w8a8(
                    ctx,
                    &key,
                    &activation.quantized,
                    weight.values_i8,
                    &activation.row_scales,
                    &weight_scales,
                    preferred,
                )
            })?
            .tactic
    };
    #[cfg(not(apxinf_cutlass_int8_sm80))]
    let _ = selected;
    #[cfg(apxinf_cutlass_int8_sm80)]
    if selected.backend == TacticBackend::Cutlass
        && weight.input_dim % 16 == 0
        && weight.output_dim % 8 == 0
    {
        let cutlass_result = unsafe {
            ffi::check_cuda(ffi::apxinf_static_cutlass_int8_gemm_bf16(
                activation.quantized.ptr(),
                weight.values_i8.ptr(),
                activation.row_scales.ptr(),
                weight_scales.ptr(),
                output.ptr(),
                rows as i32,
                weight.output_dim as i32,
                weight.input_dim as i32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        };
        match cutlass_result {
            Ok(()) => {
                return Ok(
                    output.into_tensor(Shape::new(vec![rows, weight.output_dim]), DType::BF16)
                )
            }
            Err(error) => {
                eprintln!(
                    "[apxinf] W8A8 CUTLASS tactic failed for {key:?}: {error}; using vendor fallback"
                );
                ctx.gemm_plans().fallback(ctx, &key)?;
            }
        }
    }

    let accumulators = crate::workspace::output_buffer(
        ctx,
        rows * weight.output_dim * std::mem::size_of::<i32>(),
    )?;
    ctx.cublas()
        .gemm_int8_i32(
            rows,
            weight.output_dim,
            weight.input_dim,
            &activation.quantized,
            weight.values_i8,
            &accumulators,
        )
        .map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_dequantize_int32_bf16(
            accumulators.ptr(),
            activation.row_scales.ptr(),
            weight_scales.ptr(),
            output.ptr(),
            rows as i32,
            weight.output_dim as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output.into_tensor(Shape::new(vec![rows, weight.output_dim]), DType::BF16))
}
