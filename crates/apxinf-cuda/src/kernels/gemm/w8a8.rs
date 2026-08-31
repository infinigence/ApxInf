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
    let prefer_cutlass =
        cfg!(apxinf_cutlass_int8_sm80) && matches!(ctx.caps().arch_family, CudaArchFamily::Sm80);
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
