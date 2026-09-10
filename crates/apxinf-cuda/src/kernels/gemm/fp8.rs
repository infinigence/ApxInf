use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use half::f16;

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::tuning::{
    AutoTuneConfig, AutoTuneEngine, CandidateMeasurement, DeviceFingerprint, Epilogue, GemmLayout,
    GemmOp, GemmTuningKey, ScaleMode, TacticBackend, TacticId, TuningDType, TuningOutcome,
};

#[derive(Clone, Copy, Debug)]
struct ColdL2TuningMetadata {
    eviction_buffer_bytes: usize,
}

#[cfg(apxinf_cutlass_gemm)]
fn dynamic_fp8_tactic(m: usize, n: usize, k: usize) -> i32 {
    match (m, n, k) {
        (10, 1024, 2048) => 1,
        (10, 2560, 1024) => 11,
        (10, 4096, 1024) => 13,
        (217, 22016, 2048) => 12,
        (217, 2048, 11008) => 6,
        (217, 2560, 2048) => 8,
        (217, 2048, 2048) => 10,
        (648, 1280, 1280) => 8,
        (648, 1280, 3424) => 9,
        (648, 3840, 1280) => 9,
        (648, 6848, 1280) => 6,
        _ if m <= 64 => 1,
        _ if m <= 256 && n >= 10_000 => 0,
        _ if m <= 256 && k >= 10_000 => 6,
        _ if m <= 256 && n >= 2_500 => 8,
        _ if m <= 256 => 3,
        _ if n >= 5_000 => 6,
        _ if k >= 3_000 => 8,
        _ if n >= 2_500 => 6,
        _ => 8,
    }
}

fn cold_l2_tuning_metadata(ctx: &CudaContext) -> Result<ColdL2TuningMetadata> {
    let mut l2_cache_bytes = 0i32;
    unsafe {
        ffi::check_cuda(ffi::cudaDeviceGetAttribute(
            &mut l2_cache_bytes,
            ffi::CUDA_DEV_ATTR_L2_CACHE_SIZE,
            ctx.device_id() as i32,
        ))
        .map_err(Error::Cuda)?;
    }
    let l2_cache_bytes = usize::try_from(l2_cache_bytes)
        .ok()
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| Error::Other("CUDA reported an empty L2 cache".into()))?;
    let eviction_buffer_bytes = l2_cache_bytes
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(255))
        .map(|bytes| bytes & !255usize)
        .ok_or_else(|| Error::Other("cold-L2 eviction buffer size overflow".into()))?;
    Ok(ColdL2TuningMetadata {
        eviction_buffer_bytes,
    })
}

struct ColdL2Evictor {
    buffer: CudaBuffer,
    metadata: ColdL2TuningMetadata,
    seed: u32,
}

impl ColdL2Evictor {
    fn new(ctx: &CudaContext) -> Result<Self> {
        let metadata = cold_l2_tuning_metadata(ctx)?;
        let buffer = CudaBuffer::alloc_zeros(metadata.eviction_buffer_bytes, ctx.device_id())
            .map_err(Error::Cuda)?;
        Ok(Self {
            buffer,
            metadata,
            seed: 0,
        })
    }

    fn evict(&mut self, ctx: &CudaContext) -> Result<()> {
        self.seed = self.seed.wrapping_add(1);
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_evict_l2(
                self.buffer.ptr(),
                self.metadata.eviction_buffer_bytes,
                self.seed,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        }
    }
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

    fn measure(
        &self,
        ctx: &CudaContext,
        evictor: &mut ColdL2Evictor,
        launch: impl FnOnce() -> Result<()>,
    ) -> Result<f64> {
        evictor.evict(ctx)?;
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

fn validate_fp8_dual_geglu_record(
    op: GemmOp,
    m: usize,
    n: usize,
    k: usize,
    tactic: i32,
) -> Result<()> {
    if op != GemmOp::Fp8F16 || !matches!(m, 522 | 533) || (n, k) != (32768, 2048) || tactic != 0 {
        return Err(Error::Other(format!(
            "FP8 dual GeGLU backend requires M522 or M533, N32768/K2048, tactic 0; got M{m}/N{n}/K{k} tactic {tactic}"
        )));
    }
    Ok(())
}

#[cfg(test)]
fn validate_bf16_dual_geglu_record(
    op: GemmOp,
    m: usize,
    n: usize,
    k: usize,
    tactic: i32,
    expected_m: usize,
    experiment: &str,
) -> Result<()> {
    if op != GemmOp::Bf16 || (m, n, k) != (expected_m, 32768, 2048) || tactic != 0 {
        return Err(Error::Other(format!(
            "{experiment} backend requires exact BF16 M{expected_m}/N32768/K2048 tactic 0"
        )));
    }
    Ok(())
}

/// Borrowed static-per-tensor FP8 weight contract.
#[derive(Clone, Copy)]
pub struct Fp8WeightView<'a> {
    pub values_e4m3: &'a Tensor,
    pub scale: f32,
    /// Exact dual-GeGLU [gate256,up256] physical layout. Plain GEMM must
    /// reject this layout; only the exact dual-GeGLU backend may consume it.
    pub dual_geglu_interleaved: bool,
    /// Optional auto-mode physical [gate256,up256] matrix. The primary tensor
    /// remains plain and is used by every non-dual route.
    pub dual_geglu_auto_interleaved: Option<&'a Tensor>,
}

#[derive(Clone, Copy)]
pub struct DynamicFp8WeightView<'a> {
    /// Contiguous output-major physical `[N, K]` E4M3 matrix.
    pub values_e4m3: &'a Tensor,
    /// FP32 scale for each output channel, shape `[N]`.
    pub channel_scales: &'a Tensor,
}

fn tuning_key(ctx: &CudaContext, m: usize, n: usize, k: usize) -> GemmTuningKey {
    GemmTuningKey {
        op: GemmOp::Fp8F16,
        device: DeviceFingerprint::from(ctx.caps()),
        m,
        n,
        k,
        activation_dtype: TuningDType::F8E4M3,
        weight_dtype: TuningDType::F8E4M3,
        output_dtype: TuningDType::F16,
        layout: GemmLayout::RowMajor,
        scale_mode: ScaleMode::PerTensor,
        epilogue: Epilogue::None,
        workspace_limit: usize::MAX,
    }
}

fn bf16_output_tuning_key(ctx: &CudaContext, m: usize, n: usize, k: usize) -> GemmTuningKey {
    GemmTuningKey {
        op: GemmOp::Fp8Bf16,
        device: DeviceFingerprint::from(ctx.caps()),
        m,
        n,
        k,
        activation_dtype: TuningDType::F8E4M3,
        weight_dtype: TuningDType::F8E4M3,
        output_dtype: TuningDType::Bf16,
        layout: GemmLayout::RowMajor,
        scale_mode: ScaleMode::PerTensor,
        epilogue: Epilogue::None,
        workspace_limit: usize::MAX,
    }
}

pub fn exact_fp8_tactic(
    ctx: &CudaContext,
    m: usize,
    n: usize,
    k: usize,
) -> Option<crate::tuning::TacticId> {
    ctx.tuning()
        .lookup_gemm(&tuning_key(ctx, m, n, k))
        .filter(|resolved| resolved.source == crate::tuning::TacticMatch::Exact)
        .map(|resolved| resolved.tactic)
}

fn resolve_fp8_plan(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    alpha: f32,
) -> Result<super::PreparedGemmPlan> {
    ctx.gemm_plans().resolve_or_tune(
        ctx,
        key,
        super::plan::default_fp8_tactic(key.m, key.n, key.k),
        |preferred| autotune_request_fp8(ctx, key, activation, weight, alpha, preferred),
    )
}

/// Physical static FP8 GEMM with FP16 output.
pub fn gemm_fp8(
    ctx: &CudaContext,
    activation: &Tensor,
    activation_scale: f32,
    weight: Fp8WeightView<'_>,
) -> Result<Tensor> {
    if weight.dual_geglu_interleaved {
        return Err(Error::Other(
            "FP8 dual GeGLU interleaved Gate/Up weight cannot be used by plain FP8 GEMM".into(),
        ));
    }
    if activation.dtype() != DType::F8E4M3 || weight.values_e4m3.dtype() != DType::F8E4M3 {
        return Err(Error::Other(format!(
            "gemm_fp8 expects E4M3 operands, got {} and {}",
            activation.dtype(),
            weight.values_e4m3.dtype()
        )));
    }
    if !activation_scale.is_finite()
        || activation_scale <= 0.0
        || !weight.scale.is_finite()
        || weight.scale <= 0.0
    {
        return Err(Error::Other(format!(
            "gemm_fp8 scales must be finite and positive, got activation={activation_scale}, weight={}",
            weight.scale
        )));
    }
    let a = activation.shape().dims();
    let b = weight.values_e4m3.shape().dims();
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
        return Err(Error::Other(format!(
            "gemm_fp8 shape mismatch: {a:?} @ {b:?}"
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device || weight.values_e4m3.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: if activation.device() != expected_device {
                activation.device()
            } else {
                weight.values_e4m3.device()
            },
        });
    }

    let (m, k, n) = (a[0], a[1], b[1]);
    let output = crate::workspace::output_buffer(ctx, m * n * DType::F16.size_in_bytes())?;
    let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight.values_e4m3).map_err(Error::Cuda)?;
    if crate::workspace::fp8_emulation_required(ctx)? {
        let activation_bytes = m
            .checked_mul(k)
            .and_then(|elements| elements.checked_mul(DType::F16.size_in_bytes()))
            .ok_or_else(|| Error::Other("FP8 activation decode size overflow".into()))?;
        let weight_bytes = k
            .checked_mul(n)
            .and_then(|elements| elements.checked_mul(DType::F16.size_in_bytes()))
            .ok_or_else(|| Error::Other("FP8 weight decode size overflow".into()))?;
        let (activation_f16, weight_f16) =
            crate::workspace::fp8_emulation_buffers(ctx, activation_bytes, weight_bytes)?;
        dequantize_e4m3_f16(
            ctx,
            &activation_buffer,
            &activation_f16,
            m * k,
            activation_scale,
        )?;
        dequantize_e4m3_f16(ctx, &weight_buffer, &weight_f16, k * n, weight.scale)?;
        ctx.cublas()
            .gemm(
                DType::F16,
                m,
                n,
                k,
                1.0,
                &activation_f16,
                &weight_f16,
                0.0,
                &output,
            )
            .map_err(Error::Cuda)?;
        return Ok(output.into_tensor(Shape::new(vec![m, n]), DType::F16));
    }

    let key = tuning_key(ctx, m, n, k);
    let alpha = activation_scale * weight.scale;
    let plan = resolve_fp8_plan(ctx, &key, &activation_buffer, &weight_buffer, alpha)?;
    let selected_tactic = plan.tactic;
    let use_split_serial = matches!(
        selected_tactic.backend,
        TacticBackend::CublasLtCustomSplitSerial
            | TacticBackend::CublasLtCustomSplitGeGluCutlass
            | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto
            | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3
            | TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm
    );
    let selected_result = (|| -> Result<()> {
        if selected_tactic.backend == TacticBackend::Cutlass {
            #[cfg(apxinf_cutlass_gemm)]
            {
                return cutlass_fp8_gemm_f16(
                    ctx,
                    &activation_buffer,
                    &weight_buffer,
                    &output,
                    m,
                    n,
                    k,
                    alpha,
                    selected_tactic.value,
                )?
                .then_some(())
                .ok_or_else(|| {
                    Error::Other(format!(
                        "CUTLASS tactic {} rejected [{m},{n},{k}]",
                        selected_tactic.value
                    ))
                });
            }
            #[cfg(not(apxinf_cutlass_gemm))]
            return Err(Error::Other(
                "CUTLASS FP8 tactic requires an SM100-family build".into(),
            ));
        }
        if use_split_serial {
            cublaslt_fp8_gemm_split_f16(
                ctx,
                &activation_buffer,
                &weight_buffer,
                &output,
                m,
                n,
                k,
                alpha,
            )
        } else {
            cublaslt_fp8_gemm_f16(
                ctx,
                &activation_buffer,
                &weight_buffer,
                &output,
                m,
                n,
                k,
                alpha,
            )
        }
    })();
    if let Err(error) = selected_result {
        if selected_tactic.backend == TacticBackend::Vendor {
            return Err(error);
        }
        eprintln!(
            "[apxinf] FP8 tactic {selected_tactic:?} failed for {key:?}: {error}; using vendor fallback"
        );
        ctx.gemm_plans().fallback(ctx, &key)?;
        cublaslt_fp8_gemm_f16(
            ctx,
            &activation_buffer,
            &weight_buffer,
            &output,
            m,
            n,
            k,
            alpha,
        )?;
    }
    Ok(output.into_tensor(Shape::new(vec![m, n]), DType::F16))
}

fn geglu_tuning_key(ctx: &CudaContext, m: usize, full_n: usize, k: usize) -> GemmTuningKey {
    let mut key = tuning_key(ctx, m, full_n, k);
    key.output_dtype = TuningDType::F8E4M3;
    key.epilogue = Epilogue::GeGlu;
    key
}

const fn decomposed_geglu_tactic() -> TacticId {
    TacticId {
        backend: TacticBackend::GemmThenGeGlu,
        value: 0,
    }
}

/// Dynamic FP8 GEMM with one activation scale per row and one weight scale
/// per output channel. The native backend applies both vectors and an optional
/// BF16 bias before returning the final BF16 matrix.
pub fn gemm_fp8_dynamic_bf16(
    ctx: &CudaContext,
    activation: &Tensor,
    activation_scales: &Tensor,
    weight: DynamicFp8WeightView<'_>,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    if activation.dtype() != DType::F8E4M3 || weight.values_e4m3.dtype() != DType::F8E4M3 {
        return Err(Error::Other(format!(
            "dynamic FP8 GEMM expects E4M3 operands, got {} and {}",
            activation.dtype(),
            weight.values_e4m3.dtype()
        )));
    }
    if activation_scales.dtype() != DType::F32 || weight.channel_scales.dtype() != DType::F32 {
        return Err(Error::Other(format!(
            "dynamic FP8 GEMM expects FP32 scale vectors, got {} and {}",
            activation_scales.dtype(),
            weight.channel_scales.dtype()
        )));
    }
    let a = activation.shape().dims();
    let b = weight.values_e4m3.shape().dims();
    if a.len() != 2 || b.len() != 2 || a[1] != b[1] {
        return Err(Error::Other(format!(
            "dynamic FP8 GEMM shape mismatch: activation {a:?}, NK weight {b:?}"
        )));
    }
    let (m, k, n) = (a[0], a[1], b[0]);
    if activation_scales.shape().dims() != [m] || weight.channel_scales.shape().dims() != [n] {
        return Err(Error::Other(format!(
            "dynamic FP8 GEMM scale mismatch: activation {:?}, weight {:?}, expected [{m}] and [{n}]",
            activation_scales.shape().dims(),
            weight.channel_scales.shape().dims()
        )));
    }
    if let Some(bias) = bias {
        if bias.dtype() != DType::BF16 || bias.shape().dims() != [n] {
            return Err(Error::Other(format!(
                "dynamic FP8 GEMM bias must be BF16 [{n}], got {} {:?}",
                bias.dtype(),
                bias.shape().dims()
            )));
        }
    }
    let expected_device = Device::Cuda(ctx.device_id());
    for tensor in [
        activation,
        activation_scales,
        weight.values_e4m3,
        weight.channel_scales,
    ] {
        if tensor.device() != expected_device {
            return Err(Error::DeviceMismatch {
                expected: expected_device,
                got: tensor.device(),
            });
        }
    }
    if let Some(bias) = bias {
        if bias.device() != expected_device {
            return Err(Error::DeviceMismatch {
                expected: expected_device,
                got: bias.device(),
            });
        }
    }
    if crate::workspace::fp8_emulation_required(ctx)? {
        return Err(Error::Other(
            "dynamic rowwise FP8 GEMM requires native FP8 Tensor Cores".into(),
        ));
    }
    if n % 16 != 0 || k % 16 != 0 {
        return Err(Error::Other(format!(
            "dynamic rowwise FP8 GEMM requires N and K divisible by 16, got N={n}, K={k}"
        )));
    }

    let output = crate::workspace::output_buffer(
        ctx,
        m.checked_mul(n)
            .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("dynamic FP8 GEMM output size overflow".into()))?,
    )?;
    let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight.values_e4m3).map_err(Error::Cuda)?;
    let activation_scale_buffer =
        CudaBuffer::from_tensor(activation_scales).map_err(Error::Cuda)?;
    let weight_scale_buffer =
        CudaBuffer::from_tensor(weight.channel_scales).map_err(Error::Cuda)?;
    let bias_buffer = bias
        .map(CudaBuffer::from_tensor)
        .transpose()
        .map_err(Error::Cuda)?;
    let bias_pointer = bias_buffer.as_ref().map_or(std::ptr::null(), |buffer| {
        buffer.ptr() as *const std::ffi::c_void
    });

    #[cfg(not(apxinf_cutlass_gemm))]
    {
        return Err(Error::Other(
            "dynamic rowwise FP8 GEMM requires an SM100-family native backend".into(),
        ));
    }

    #[cfg(apxinf_cutlass_gemm)]
    {
        let tactic = dynamic_fp8_tactic(m, n, k);
        let status = unsafe {
            ffi::apxinf_dynamic_cutlass_fp8_gemm_bf16(
                activation_buffer.ptr(),
                weight_buffer.ptr(),
                activation_scale_buffer.ptr().cast::<f32>(),
                weight_scale_buffer.ptr().cast::<f32>(),
                bias_pointer,
                output.ptr(),
                m as i32,
                n as i32,
                k as i32,
                tactic,
                ctx.stream().handle(),
            )
        };
        if status != 0 {
            return Err(Error::Cuda(format!(
                "dynamic rowwise FP8 GEMM rejected [{m},{n},{k}] tactic {tactic} ({status})"
            )));
        }
        Ok(output.into_tensor(Shape::new(vec![m, n]), DType::BF16))
    }
}

/// Resolve and execute the complete FP8 gate/up projection plus GeGLU
/// operator. A tactic may be one fused kernel or a prepared GEMM followed by
/// the standalone GeGLU kernel; all candidates have identical final output.
pub fn gemm_fp8_geglu_fused(
    ctx: &CudaContext,
    activation: &Tensor,
    activation_scale: f32,
    packed_weight: Fp8WeightView<'_>,
    output_scale: f32,
) -> Result<Option<Tensor>> {
    if activation.dtype() != DType::F8E4M3 || packed_weight.values_e4m3.dtype() != DType::F8E4M3 {
        return Err(Error::Other(format!(
            "FP8 fused GeGLU expects E4M3 operands, got {} and {}",
            activation.dtype(),
            packed_weight.values_e4m3.dtype()
        )));
    }
    if !activation_scale.is_finite()
        || activation_scale <= 0.0
        || !packed_weight.scale.is_finite()
        || packed_weight.scale <= 0.0
        || !output_scale.is_finite()
        || output_scale <= 0.0
    {
        return Err(Error::Other(format!(
            "FP8 fused GeGLU scales must be finite and positive, got activation={activation_scale}, weight={}, output={output_scale}",
            packed_weight.scale
        )));
    }
    let a = activation.shape().dims();
    let b = packed_weight.values_e4m3.shape().dims();
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] || b[1] % 2 != 0 {
        return Err(Error::Other(format!(
            "FP8 fused GeGLU shape mismatch: {a:?} @ {b:?}"
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device
        || packed_weight.values_e4m3.device() != expected_device
    {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: if activation.device() != expected_device {
                activation.device()
            } else {
                packed_weight.values_e4m3.device()
            },
        });
    }
    if let Some(weight) = packed_weight.dual_geglu_auto_interleaved {
        super::validate_geglu_weight(
            "FP8 dual GeGLU auto-interleaved weight",
            weight,
            DType::F8E4M3,
            b,
            expected_device,
        )?;
    }

    if crate::workspace::fp8_emulation_required(ctx)? {
        return Ok(None);
    }
    let (m, k, full_n) = (a[0], a[1], b[1]);
    let n = full_n / 2;
    let alpha = activation_scale * packed_weight.scale;
    let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let primary_weight = CudaBuffer::from_tensor(packed_weight.values_e4m3).map_err(Error::Cuda)?;
    let automatic_interleaved = packed_weight
        .dual_geglu_auto_interleaved
        .map(CudaBuffer::from_tensor)
        .transpose()
        .map_err(Error::Cuda)?;
    let (plain_weight, interleaved_weight) = if packed_weight.dual_geglu_interleaved {
        (None, Some(&primary_weight))
    } else {
        (Some(&primary_weight), automatic_interleaved.as_ref())
    };

    let key = geglu_tuning_key(ctx, m, full_n, k);
    let plain_key = tuning_key(ctx, m, full_n, k);
    // Resolve a cold key's decomposed GEMM dependency before GeGLU tuning
    // acquires the session-wide, non-reentrant tuning lock.
    let mut plain_plan = if ctx.tuning().lookup_gemm_exact(&key).is_none() {
        plain_weight
            .map(|weight| resolve_fp8_plan(ctx, &plain_key, &activation_buffer, weight, alpha))
            .transpose()?
    } else {
        None
    };
    let default = default_fp8_geglu_tactic(
        ctx,
        &key,
        plain_weight.is_some(),
        interleaved_weight.is_some(),
    )?;
    let plan = ctx
        .gemm_plans()
        .resolve_or_tune(ctx, &key, default, |preferred| {
            autotune_request_fp8_geglu(
                ctx,
                &key,
                &plain_key,
                plain_plan.as_ref(),
                &activation_buffer,
                plain_weight,
                interleaved_weight,
                alpha,
                output_scale,
                preferred,
            )
        })?;

    if plan.tactic.backend == TacticBackend::GemmThenGeGlu && plain_plan.is_none() {
        plain_plan = plain_weight
            .map(|weight| resolve_fp8_plan(ctx, &plain_key, &activation_buffer, weight, alpha))
            .transpose()?;
    }
    let mut gate = if fp8_geglu_tactic_uses_gate(plan.tactic) {
        Some(crate::workspace::output_buffer(
            ctx,
            fp8_geglu_gate_bytes(m, full_n)?,
        )?)
    } else {
        None
    };
    let output = crate::workspace::output_buffer(
        ctx,
        m.checked_mul(n)
            .and_then(|elements| elements.checked_mul(DType::F8E4M3.size_in_bytes()))
            .ok_or_else(|| Error::Other("FP8 GeGLU output size overflow".into()))?,
    )?;
    if crate::workspace::may_prepare_native_resources() {
        prepare_fp8_geglu_tactic(&key, &plain_key, plain_plan.as_ref(), plan.tactic)?;
    }
    if let Err(error) = launch_fp8_geglu_tactic(
        ctx,
        &key,
        plain_plan.as_ref(),
        &activation_buffer,
        plain_weight,
        interleaved_weight,
        gate.as_ref(),
        &output,
        alpha,
        output_scale,
        plan.tactic,
    ) {
        if plan.tactic == default || plain_weight.is_none() {
            return Err(error);
        }
        eprintln!(
            "[apxinf] FP8 GeGLU tactic {:?} failed for {key:?}: {error}; using decomposed fallback",
            plan.tactic
        );
        ctx.gemm_plans()
            .fallback_to(ctx, &key, decomposed_geglu_tactic())?;
        if plain_plan.is_none() {
            plain_plan = plain_weight
                .map(|weight| resolve_fp8_plan(ctx, &plain_key, &activation_buffer, weight, alpha))
                .transpose()?;
        }
        if gate.is_none() {
            gate = Some(crate::workspace::output_buffer(
                ctx,
                fp8_geglu_gate_bytes(m, full_n)?,
            )?);
        }
        prepare_fp8_geglu_tactic(
            &key,
            &plain_key,
            plain_plan.as_ref(),
            decomposed_geglu_tactic(),
        )?;
        launch_fp8_geglu_tactic(
            ctx,
            &key,
            plain_plan.as_ref(),
            &activation_buffer,
            plain_weight,
            interleaved_weight,
            gate.as_ref(),
            &output,
            alpha,
            output_scale,
            decomposed_geglu_tactic(),
        )?;
    }
    Ok(Some(
        output.into_tensor(Shape::new(vec![m, n]), DType::F8E4M3),
    ))
}

fn fp8_geglu_tactic_uses_gate(tactic: TacticId) -> bool {
    matches!(
        tactic.backend,
        TacticBackend::GemmThenGeGlu
            | TacticBackend::CublasLtCustomSplitGeGluCutlass
            | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto
            | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3
            | TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm
    )
}

fn fp8_geglu_gate_bytes(m: usize, full_n: usize) -> Result<usize> {
    m.checked_mul(full_n)
        .and_then(|elements| elements.checked_mul(DType::F16.size_in_bytes()))
        .ok_or_else(|| Error::Other("FP8 GeGLU gate size overflow".into()))
}

fn default_fp8_geglu_tactic(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    has_plain_weight: bool,
    has_interleaved_weight: bool,
) -> Result<TacticId> {
    if has_plain_weight {
        return Ok(decomposed_geglu_tactic());
    }
    validate_fp8_interleaved_default(ctx.caps().sm, key, has_interleaved_weight)?;
    #[cfg(apxinf_cutlass_gemm)]
    {
        Ok(TacticId {
            backend: TacticBackend::CutlassFp8DualGeGlu,
            value: 0,
        })
    }
    #[cfg(not(apxinf_cutlass_gemm))]
    {
        Err(Error::Other(
            "FP8 interleaved-only GeGLU requires the SM100-family CUTLASS build".into(),
        ))
    }
}

fn validate_fp8_interleaved_default(
    sm: u32,
    key: &GemmTuningKey,
    has_interleaved_weight: bool,
) -> Result<()> {
    if sm != 110 || !has_interleaved_weight {
        return Err(Error::Other(format!(
            "FP8 interleaved-only GeGLU requires an SM110 dual-GEMM weight and kernel for {key:?}"
        )));
    }
    validate_fp8_dual_geglu_record(key.op, key.m, key.n, key.k, 0)
}

fn prepare_fp8_geglu_tactic(
    key: &GemmTuningKey,
    plain_key: &GemmTuningKey,
    plain_plan: Option<&super::PreparedGemmPlan>,
    tactic: TacticId,
) -> Result<()> {
    if tactic.backend == TacticBackend::GemmThenGeGlu {
        let plan = plain_plan.ok_or_else(|| {
            Error::Other("decomposed FP8 GeGLU has no plain-weight GEMM plan".into())
        })?;
        return super::providers::prepare(plain_key, plan.tactic);
    }
    super::providers::prepare(key, tactic)
}

#[allow(clippy::too_many_arguments)]
fn launch_fp8_geglu_tactic(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    plain_plan: Option<&super::PreparedGemmPlan>,
    activation: &CudaBuffer,
    plain_weight: Option<&CudaBuffer>,
    interleaved_weight: Option<&CudaBuffer>,
    gate: Option<&CudaBuffer>,
    output: &CudaBuffer,
    alpha: f32,
    output_scale: f32,
    tactic: TacticId,
) -> Result<()> {
    let n = key.n / 2;
    match tactic.backend {
        TacticBackend::GemmThenGeGlu => {
            let gate =
                gate.ok_or_else(|| Error::Other("decomposed FP8 GeGLU has no gate buffer".into()))?;
            let plain_plan = plain_plan.ok_or_else(|| {
                Error::Other("decomposed FP8 GeGLU has no prepared GEMM plan".into())
            })?;
            let plain_weight = plain_weight.ok_or_else(|| {
                Error::Other("decomposed FP8 GeGLU has no plain-layout weight".into())
            })?;
            launch_tactic_fp8(
                ctx,
                &plain_plan.key,
                activation,
                plain_weight,
                gate,
                alpha,
                plain_plan.tactic,
            )?;
            unsafe {
                ffi::check_cuda(ffi::apxinf_static_geglu_quant_f16_e4m3(
                    gate.ptr(),
                    output.ptr(),
                    key.m as i32,
                    n as i32,
                    output_scale,
                    ctx.stream().handle(),
                ))
                .map_err(Error::Cuda)
            }
        }
        TacticBackend::CutlassFp8DualGeGlu => {
            let weight = interleaved_weight.ok_or_else(|| {
                Error::Other("FP8 dual-GEMM GeGLU has no interleaved weight".into())
            })?;
            #[cfg(apxinf_cutlass_gemm)]
            {
                let status = unsafe {
                    ffi::apxinf_static_cutlass_fp8_dual_gemm_geglu_e4m3(
                        activation.ptr(),
                        weight.ptr(),
                        output.ptr(),
                        key.m as i32,
                        n as i32,
                        key.k as i32,
                        key.n as i32,
                        alpha,
                        output_scale,
                        ctx.stream().handle(),
                    )
                };
                if status == 0 {
                    Ok(())
                } else {
                    Err(Error::Cuda(format!(
                        "FP8 dual-GEMM GeGLU rejected [{},{},{}] ({status})",
                        key.m, n, key.k
                    )))
                }
            }
            #[cfg(not(apxinf_cutlass_gemm))]
            {
                let _ = weight;
                Err(Error::Other(
                    "FP8 dual-GEMM GeGLU requires the SM100-family CUTLASS build".into(),
                ))
            }
        }
        TacticBackend::CublasLtCustomSplitGeGluCutlass
        | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto
        | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3
        | TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm => {
            let gate =
                gate.ok_or_else(|| Error::Other("split FP8 GeGLU has no gate buffer".into()))?;
            let weight = plain_weight
                .ok_or_else(|| Error::Other("split FP8 GeGLU has no plain-layout weight".into()))?;
            let (cutlass_tactic, expected_m) = match tactic.backend {
                TacticBackend::CublasLtCustomSplitGeGluCutlass => (0, 778),
                TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto => (1, 778),
                TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3 => (2, 778),
                TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm => (3, 522),
                _ => unreachable!(),
            };
            if (key.m, key.n, key.k) != (expected_m, 32768, 2048) {
                return Err(Error::Other(format!(
                    "split FP8 GeGLU tactic requires [{expected_m},2048] @ [2048,32768], got [{},{}] @ [{},{}]",
                    key.m, key.k, key.k, key.n
                )));
            }
            #[cfg(apxinf_cutlass_gemm)]
            {
                cublaslt_fp8_gemm_split_first_f16(
                    ctx, activation, weight, gate, key.m, key.n, key.k, alpha,
                )?;
                let status = unsafe {
                    ffi::apxinf_static_cutlass_fp8_gemm_geglu_e4m3(
                        activation.ptr(),
                        weight.ptr(),
                        gate.ptr(),
                        output.ptr(),
                        key.m as i32,
                        n as i32,
                        key.k as i32,
                        key.n as i32,
                        alpha,
                        output_scale,
                        cutlass_tactic,
                        ctx.stream().handle(),
                    )
                };
                if status == 0 {
                    Ok(())
                } else {
                    Err(Error::Cuda(format!(
                        "split FP8 GeGLU tactic {:?} rejected {:?} ({status})",
                        tactic, key
                    )))
                }
            }
            #[cfg(not(apxinf_cutlass_gemm))]
            {
                let _ = (weight, cutlass_tactic);
                Err(Error::Other(
                    "split FP8 GeGLU requires the SM100-family CUTLASS build".into(),
                ))
            }
        }
        _ => Err(Error::Other(format!(
            "FP8 GeGLU online autotune cannot execute {tactic:?}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn autotune_request_fp8_geglu(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    plain_key: &GemmTuningKey,
    plain_plan: Option<&super::PreparedGemmPlan>,
    activation: &CudaBuffer,
    plain_weight: Option<&CudaBuffer>,
    interleaved_weight: Option<&CudaBuffer>,
    alpha: f32,
    output_scale: f32,
    preferred: Option<TacticId>,
) -> Result<TuningOutcome> {
    let output_elements = key
        .m
        .checked_mul(key.n / 2)
        .ok_or_else(|| Error::Other("FP8 GeGLU autotune output size overflow".into()))?;
    let gate = CudaBuffer::alloc_zeros(
        key.m
            .checked_mul(key.n)
            .and_then(|elements| elements.checked_mul(DType::F16.size_in_bytes()))
            .ok_or_else(|| Error::Other("FP8 GeGLU autotune gate size overflow".into()))?,
        ctx.device_id(),
    )
    .map_err(Error::Cuda)?;
    let reference_output =
        CudaBuffer::alloc_zeros(output_elements, ctx.device_id()).map_err(Error::Cuda)?;
    let candidate_output =
        CudaBuffer::alloc_zeros(output_elements, ctx.device_id()).map_err(Error::Cuda)?;
    let decoded_output = CudaBuffer::alloc_zeros(
        output_elements * DType::F16.size_in_bytes(),
        ctx.device_id(),
    )
    .map_err(Error::Cuda)?;

    let reference_tactic = if plain_plan.is_some() && plain_weight.is_some() {
        decomposed_geglu_tactic()
    } else {
        TacticId {
            backend: TacticBackend::CutlassFp8DualGeGlu,
            value: 0,
        }
    };
    prepare_fp8_geglu_tactic(key, plain_key, plain_plan, reference_tactic)?;
    launch_fp8_geglu_tactic(
        ctx,
        key,
        plain_plan,
        activation,
        plain_weight,
        interleaved_weight,
        Some(&gate),
        &reference_output,
        alpha,
        output_scale,
        reference_tactic,
    )?;
    ctx.synchronize().map_err(Error::Cuda)?;
    let reference = copy_fused_output(
        ctx,
        &reference_output,
        Some(&decoded_output),
        output_elements,
        DType::F8E4M3,
        output_scale,
    )?;

    let candidates = super::providers::geglu_candidates(key)
        .into_iter()
        .filter(|candidate| {
            candidate.tactic.backend != TacticBackend::GemmThenGeGlu
                || (plain_plan.is_some() && plain_weight.is_some())
        })
        .filter(|candidate| {
            candidate.tactic.backend != TacticBackend::CutlassFp8DualGeGlu
                || interleaved_weight.is_some()
        });
    let events = CudaEventPair::new()?;
    let mut evictor = ColdL2Evictor::new(ctx)?;
    let engine = AutoTuneEngine::new(AutoTuneConfig::default())?;
    engine.tune_with_preferred(key, preferred, candidates, |candidate, config| {
        prepare_fp8_geglu_tactic(key, plain_key, plain_plan, candidate.tactic)?;
        launch_fp8_geglu_tactic(
            ctx,
            key,
            plain_plan,
            activation,
            plain_weight,
            interleaved_weight,
            Some(&gate),
            &candidate_output,
            alpha,
            output_scale,
            candidate.tactic,
        )?;
        ctx.synchronize().map_err(Error::Cuda)?;
        let actual = copy_fused_output(
            ctx,
            &candidate_output,
            Some(&decoded_output),
            output_elements,
            DType::F8E4M3,
            output_scale,
        )?;
        let correct = crate::tuning::outputs_are_close(&reference, &actual, 0.03, 0.998);
        if !correct {
            return Ok(CandidateMeasurement {
                tactic: candidate.tactic,
                milliseconds: None,
                correct: false,
            });
        }
        for _ in 0..config.warmup_iterations {
            evictor.evict(ctx)?;
            launch_fp8_geglu_tactic(
                ctx,
                key,
                plain_plan,
                activation,
                plain_weight,
                interleaved_weight,
                Some(&gate),
                &candidate_output,
                alpha,
                output_scale,
                candidate.tactic,
            )?;
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        let mut milliseconds = 0.0;
        for _ in 0..config.benchmark_iterations {
            milliseconds += events.measure(ctx, &mut evictor, || {
                launch_fp8_geglu_tactic(
                    ctx,
                    key,
                    plain_plan,
                    activation,
                    plain_weight,
                    interleaved_weight,
                    Some(&gate),
                    &candidate_output,
                    alpha,
                    output_scale,
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

pub fn native_fp8_gemm_supported_for_device(device: usize) -> Result<bool> {
    let mut supported = 0i32;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_native_fp8_supported(
            device as i32,
            &mut supported,
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(supported != 0)
}

/// Whether this CUDA device can execute E4M3 GEMMs directly on Tensor Cores.
pub fn native_fp8_gemm_supported(ctx: &CudaContext) -> Result<bool> {
    native_fp8_gemm_supported_for_device(ctx.device_id())
}

fn copy_f16_output(output: &CudaBuffer, elements: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; elements * DType::F16.size_in_bytes()];
    output.copy_to_host(&mut bytes).map_err(Error::Cuda)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|value| f16::from_bits(u16::from_ne_bytes([value[0], value[1]])).to_f32())
        .collect())
}

fn copy_fused_output(
    ctx: &CudaContext,
    output: &CudaBuffer,
    decoded: Option<&CudaBuffer>,
    elements: usize,
    dtype: DType,
    dequantization_scale: f32,
) -> Result<Vec<f32>> {
    match dtype {
        DType::F16 => copy_f16_output(output, elements),
        DType::F8E4M3 => {
            let decoded = decoded
                .ok_or_else(|| Error::Other("FP8 fused autotune has no decode buffer".into()))?;
            dequantize_e4m3_f16(ctx, output, decoded, elements, dequantization_scale)?;
            ctx.synchronize().map_err(Error::Cuda)?;
            copy_f16_output(decoded, elements)
        }
        dtype => Err(Error::Other(format!(
            "unsupported FP8 fused autotune output dtype {dtype}"
        ))),
    }
}

/// Resolve a pointer-independent plan for a fused FP8 GEMM. Native resources
/// containing the real bias/residual pointers are prepared by the supplied
/// callback and stay outside the persistent tactic identity.
pub(crate) fn resolve_fused_plan(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    output_dtype: DType,
    dequantization_scale: f32,
    mut prepare_native: impl FnMut() -> Result<()>,
    mut launch: impl FnMut(&CudaBuffer) -> Result<()>,
) -> Result<super::PreparedGemmPlan> {
    let default = TacticId {
        backend: TacticBackend::Vendor,
        value: 0,
    };
    ctx.gemm_plans()
        .resolve_or_tune(ctx, key, default, |preferred| {
            autotune_request_fp8_fused(
                ctx,
                key,
                output_dtype,
                dequantization_scale,
                preferred,
                &mut prepare_native,
                &mut launch,
            )
        })
}

#[allow(clippy::too_many_arguments)]
fn autotune_request_fp8_fused(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    output_dtype: DType,
    dequantization_scale: f32,
    preferred: Option<TacticId>,
    prepare_native: &mut impl FnMut() -> Result<()>,
    launch: &mut impl FnMut(&CudaBuffer) -> Result<()>,
) -> Result<TuningOutcome> {
    let output_elements = key
        .m
        .checked_mul(key.n)
        .ok_or_else(|| Error::Other("FP8 fused autotune output size overflow".into()))?;
    let output_bytes = output_elements
        .checked_mul(output_dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("FP8 fused autotune output size overflow".into()))?;
    let reference_output =
        CudaBuffer::alloc_zeros(output_bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let candidate_output =
        CudaBuffer::alloc_zeros(output_bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let decoded_output = if output_dtype == DType::F8E4M3 {
        Some(
            CudaBuffer::alloc_zeros(
                output_elements * DType::F16.size_in_bytes(),
                ctx.device_id(),
            )
            .map_err(Error::Cuda)?,
        )
    } else {
        None
    };

    let default = TacticId {
        backend: TacticBackend::Vendor,
        value: 0,
    };
    super::providers::prepare(key, default)?;
    prepare_native()?;
    launch(&reference_output)?;
    ctx.synchronize().map_err(Error::Cuda)?;
    let reference = copy_fused_output(
        ctx,
        &reference_output,
        decoded_output.as_ref(),
        output_elements,
        output_dtype,
        dequantization_scale,
    )?;

    let events = CudaEventPair::new()?;
    let mut evictor = ColdL2Evictor::new(ctx)?;
    let engine = AutoTuneEngine::new(AutoTuneConfig::default())?;
    let candidates = super::providers::candidates(key, 32).into_iter();
    engine.tune_with_preferred(key, preferred, candidates, |candidate, config| {
        super::providers::prepare(key, candidate.tactic)?;
        prepare_native()?;
        launch(&candidate_output)?;
        ctx.synchronize().map_err(Error::Cuda)?;
        let actual = copy_fused_output(
            ctx,
            &candidate_output,
            decoded_output.as_ref(),
            output_elements,
            output_dtype,
            dequantization_scale,
        )?;
        let correct = crate::tuning::outputs_are_close(&reference, &actual, 0.03, 0.998);
        if !correct {
            return Ok(CandidateMeasurement {
                tactic: candidate.tactic,
                milliseconds: None,
                correct: false,
            });
        }
        for _ in 0..config.warmup_iterations {
            evictor.evict(ctx)?;
            launch(&candidate_output)?;
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        let mut milliseconds = 0.0;
        for _ in 0..config.benchmark_iterations {
            milliseconds += events.measure(ctx, &mut evictor, || launch(&candidate_output))?;
        }
        Ok(CandidateMeasurement {
            tactic: candidate.tactic,
            milliseconds: Some(milliseconds / config.benchmark_iterations as f64),
            correct: true,
        })
    })
}

fn prepare_tactic_fp8(key: &GemmTuningKey, tactic: TacticId) -> Result<()> {
    super::providers::prepare(key, tactic)
}

#[allow(clippy::too_many_arguments)]
fn launch_tactic_fp8(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    alpha: f32,
    tactic: TacticId,
) -> Result<()> {
    match tactic.backend {
        TacticBackend::Vendor | TacticBackend::CublasLt | TacticBackend::CublasLtCustom => {
            cublaslt_fp8_gemm_f16(ctx, activation, weight, output, key.m, key.n, key.k, alpha)
        }
        TacticBackend::CublasLtCustomSplitSerial => {
            cublaslt_fp8_gemm_split_f16(ctx, activation, weight, output, key.m, key.n, key.k, alpha)
        }
        TacticBackend::Cutlass => {
            #[cfg(apxinf_cutlass_gemm)]
            {
                if cutlass_fp8_gemm_f16(
                    ctx,
                    activation,
                    weight,
                    output,
                    key.m,
                    key.n,
                    key.k,
                    alpha,
                    tactic.value,
                )? {
                    Ok(())
                } else {
                    Err(Error::Other(format!(
                        "CUTLASS tactic {} rejected {:?}",
                        tactic.value, key
                    )))
                }
            }
            #[cfg(not(apxinf_cutlass_gemm))]
            {
                Err(Error::Other(
                    "CUTLASS FP8 autotune requires an SM100-family build".into(),
                ))
            }
        }
        _ => Err(Error::Other(format!(
            "FP8 online autotune cannot execute {tactic:?}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn autotune_request_fp8(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    alpha: f32,
    preferred: Option<TacticId>,
) -> Result<TuningOutcome> {
    let output_elements = key
        .m
        .checked_mul(key.n)
        .ok_or_else(|| Error::Other("FP8 autotune output size overflow".into()))?;
    let output_bytes = output_elements
        .checked_mul(DType::F16.size_in_bytes())
        .ok_or_else(|| Error::Other("FP8 autotune output size overflow".into()))?;

    // The safe reference is independent of both tuned providers: dequantize
    // the real E4M3 operands and execute the existing FP16 cuBLAS path.
    let activation_f16 = CudaBuffer::alloc_zeros(
        key.m
            .checked_mul(key.k)
            .and_then(|elements| elements.checked_mul(DType::F16.size_in_bytes()))
            .ok_or_else(|| Error::Other("FP8 autotune activation size overflow".into()))?,
        ctx.device_id(),
    )
    .map_err(Error::Cuda)?;
    let weight_f16 = CudaBuffer::alloc_zeros(
        key.k
            .checked_mul(key.n)
            .and_then(|elements| elements.checked_mul(DType::F16.size_in_bytes()))
            .ok_or_else(|| Error::Other("FP8 autotune weight size overflow".into()))?,
        ctx.device_id(),
    )
    .map_err(Error::Cuda)?;
    dequantize_e4m3_f16(ctx, activation, &activation_f16, key.m * key.k, 1.0)?;
    dequantize_e4m3_f16(ctx, weight, &weight_f16, key.k * key.n, alpha)?;
    let reference_output =
        CudaBuffer::alloc_zeros(output_bytes, ctx.device_id()).map_err(Error::Cuda)?;
    ctx.cublas()
        .gemm(
            DType::F16,
            key.m,
            key.n,
            key.k,
            1.0,
            &activation_f16,
            &weight_f16,
            0.0,
            &reference_output,
        )
        .map_err(Error::Cuda)?;
    ctx.synchronize().map_err(Error::Cuda)?;
    let reference = copy_f16_output(&reference_output, output_elements)?;
    drop((activation_f16, weight_f16, reference_output));

    let output = CudaBuffer::alloc_zeros(output_bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let events = CudaEventPair::new()?;
    let mut evictor = ColdL2Evictor::new(ctx)?;
    let engine = AutoTuneEngine::new(AutoTuneConfig::default())?;
    let candidates = super::providers::candidates(key, 32).into_iter();
    engine.tune_with_preferred(key, preferred, candidates, |candidate, config| {
        prepare_tactic_fp8(key, candidate.tactic)?;
        launch_tactic_fp8(
            ctx,
            key,
            activation,
            weight,
            &output,
            alpha,
            candidate.tactic,
        )?;
        ctx.synchronize().map_err(Error::Cuda)?;
        let actual = copy_f16_output(&output, output_elements)?;
        let correct = crate::tuning::outputs_are_close(&reference, &actual, 0.02, 0.999);
        if !correct {
            return Ok(CandidateMeasurement {
                tactic: candidate.tactic,
                milliseconds: None,
                correct: false,
            });
        }
        for _ in 0..config.warmup_iterations {
            evictor.evict(ctx)?;
            launch_tactic_fp8(
                ctx,
                key,
                activation,
                weight,
                &output,
                alpha,
                candidate.tactic,
            )?;
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        let mut milliseconds = 0.0;
        for _ in 0..config.benchmark_iterations {
            milliseconds += events.measure(ctx, &mut evictor, || {
                launch_tactic_fp8(
                    ctx,
                    key,
                    activation,
                    weight,
                    &output,
                    alpha,
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

pub fn set_cublaslt_gemm_heuristic(
    m: usize,
    n: usize,
    k: usize,
    heuristic_rank: i32,
) -> Result<()> {
    if !(0..64).contains(&heuristic_rank) {
        return Err(Error::Other(format!(
            "invalid static inference cuBLASLt heuristic rank {heuristic_rank}"
        )));
    }
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_gemm_heuristic(m as i32, n as i32, k as i32, heuristic_rank)
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

pub fn set_cublaslt_fused_gemm_heuristic(
    m: usize,
    n: usize,
    k: usize,
    epilogue: Epilogue,
    heuristic_rank: i32,
) -> Result<()> {
    if !(0..64).contains(&heuristic_rank) {
        return Err(Error::Other(format!(
            "invalid static inference cuBLASLt heuristic rank {heuristic_rank}"
        )));
    }
    let epilogue = fused_epilogue_id(epilogue)?;
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_fp8_fused_heuristic(
            m as i32,
            n as i32,
            k as i32,
            epilogue,
            heuristic_rank,
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

pub fn set_cublaslt_gemm_custom(m: usize, n: usize, k: usize, tactic: i32) -> Result<()> {
    let config = crate::tuning::decode_cublaslt_custom_tactic(tactic).ok_or_else(|| {
        Error::Other(format!(
            "invalid static inference cuBLASLt custom tactic {tactic}"
        ))
    })?;
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_fp8_gemm_custom(
            m as i32,
            n as i32,
            k as i32,
            config.tile_id,
            config.custom_option,
            config.stages_id,
            config.cluster_shape_id,
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

pub fn set_cublaslt_gemm_bias_custom(
    m: usize,
    n: usize,
    k: usize,
    epilogue: Epilogue,
    tactic: i32,
) -> Result<()> {
    let config = crate::tuning::decode_cublaslt_custom_tactic(tactic).ok_or_else(|| {
        Error::Other(format!(
            "invalid static inference cuBLASLt fused-bias custom tactic {tactic}"
        ))
    })?;
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_fp8_gemm_bias_custom(
            m as i32,
            n as i32,
            k as i32,
            fused_epilogue_id(epilogue)?,
            config.tile_id,
            config.custom_option,
            config.stages_id,
            config.cluster_shape_id,
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

fn fused_epilogue_id(epilogue: Epilogue) -> Result<i32> {
    match epilogue {
        Epilogue::Bias => Ok(1),
        Epilogue::BiasGelu => Ok(2),
        Epilogue::BiasResidual => Ok(3),
        Epilogue::None | Epilogue::GeGlu => Err(Error::Other(
            "this operator cannot use a fused cuBLASLt bias epilogue configuration".into(),
        )),
    }
}

pub fn set_cublaslt_gemm_split_custom(m: usize, n: usize, k: usize, tactic: i32) -> Result<()> {
    let config = crate::tuning::decode_cublaslt_custom_tactic(tactic).ok_or_else(|| {
        Error::Other(format!(
            "invalid static inference cuBLASLt split-serial custom tactic {tactic}"
        ))
    })?;
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_fp8_gemm_split_custom(
            m as i32,
            n as i32,
            k as i32,
            config.tile_id,
            config.custom_option,
            config.stages_id,
            config.cluster_shape_id,
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

pub fn dequantize_e4m3_f16(
    ctx: &CudaContext,
    input: &CudaBuffer,
    output: &CudaBuffer,
    elements: usize,
    scale: f32,
) -> Result<()> {
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_dequantize_e4m3_f16(
            input.ptr(),
            output.ptr(),
            elements as i64,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(())
}

#[cfg(apxinf_cutlass_gemm)]
#[allow(clippy::too_many_arguments)]
pub fn cutlass_fp8_gemm_f16(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    tactic: i32,
) -> Result<bool> {
    let status = unsafe {
        ffi::apxinf_static_cutlass_fp8_gemm_f16(
            activation.ptr(),
            weight.ptr(),
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            alpha,
            tactic,
            ctx.stream().handle(),
        )
    };
    Ok(status == 0)
}

pub fn prepare_cublaslt_fp8_gemm(m: usize, n: usize, k: usize) -> Result<()> {
    let status = unsafe { ffi::apxinf_static_prepare_fp8_gemm_f16(m as i32, n as i32, k as i32) };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

pub fn prepare_cublaslt_fp8_gemm_bf16(m: usize, n: usize, k: usize) -> Result<()> {
    let status = unsafe { ffi::apxinf_static_prepare_fp8_gemm_bf16(m as i32, n as i32, k as i32) };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

fn parse_fp8_bf16_heuristic(spec: &str, m: usize, n: usize, k: usize) -> Result<Option<i32>> {
    for entry in spec.split(';').filter(|entry| !entry.is_empty()) {
        let (shape, rank) = entry
            .split_once('=')
            .ok_or_else(|| Error::Other(format!("invalid FP8 BF16 tactic entry {entry:?}")))?;
        let dims = shape
            .split(',')
            .map(str::parse::<usize>)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| Error::Other(format!("invalid FP8 BF16 tactic shape {shape:?}")))?;
        if dims.len() != 3 {
            return Err(Error::Other(format!(
                "FP8 BF16 tactic shape must be M,N,K, got {shape:?}"
            )));
        }
        let rank = rank
            .parse::<i32>()
            .map_err(|_| Error::Other(format!("invalid FP8 BF16 tactic rank {rank:?}")))?;
        if !(0..64).contains(&rank) {
            return Err(Error::Other(format!(
                "FP8 BF16 tactic rank must be in 0..64, got {rank}"
            )));
        }
        if dims == [m, n, k] {
            return Ok(Some(rank));
        }
    }
    Ok(None)
}

fn configured_fp8_bf16_heuristic(m: usize, n: usize, k: usize) -> Result<Option<i32>> {
    let Some(spec) = std::env::var_os("APXINF_FP8_BF16_CUBLASLT_TACTICS") else {
        return Ok(None);
    };
    let spec = spec
        .into_string()
        .map_err(|_| Error::Other("APXINF_FP8_BF16_CUBLASLT_TACTICS must be valid UTF-8".into()))?;
    parse_fp8_bf16_heuristic(&spec, m, n, k)
}

pub(super) fn set_cublaslt_fp8_bf16_gemm_heuristic(
    m: usize,
    n: usize,
    k: usize,
    heuristic_rank: i32,
) -> Result<()> {
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_fp8_gemm_bf16_heuristic(
            m as i32,
            n as i32,
            k as i32,
            heuristic_rank,
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

/// Static native E4M3 x E4M3 GEMM with a BF16 output. This is the physical
/// primitive used by BF16-residual model graphs such as GR00T; callers remain
/// responsible for activation quantization and for applying any BF16 bias.
pub fn gemm_fp8_bf16(
    ctx: &CudaContext,
    activation: &Tensor,
    activation_scale: f32,
    weight: Fp8WeightView<'_>,
) -> Result<Tensor> {
    if activation.dtype() != DType::F8E4M3 || weight.values_e4m3.dtype() != DType::F8E4M3 {
        return Err(Error::Other(format!(
            "gemm_fp8_bf16 expects E4M3 operands, got {} and {}",
            activation.dtype(),
            weight.values_e4m3.dtype()
        )));
    }
    if weight.dual_geglu_interleaved {
        return Err(Error::Other(
            "FP8 dual GeGLU interleaved weight cannot be used by plain FP8 GEMM".into(),
        ));
    }
    if !activation_scale.is_finite()
        || activation_scale <= 0.0
        || !weight.scale.is_finite()
        || weight.scale <= 0.0
    {
        return Err(Error::Other(
            "FP8 GEMM scales must be finite and positive".into(),
        ));
    }
    let a = activation.shape().dims();
    let b = weight.values_e4m3.shape().dims();
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
        return Err(Error::Other(format!(
            "gemm_fp8_bf16 shape mismatch: {a:?} @ {b:?}"
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device || weight.values_e4m3.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: if activation.device() != expected_device {
                activation.device()
            } else {
                weight.values_e4m3.device()
            },
        });
    }
    if !native_fp8_gemm_supported(ctx)? {
        return Err(Error::Other(
            "FP8-to-BF16 GEMM requires native E4M3 Tensor Core support".into(),
        ));
    }

    let (m, k, n) = (a[0], a[1], b[1]);
    if std::env::var_os("APXINF_GR00T_LOG_FP8_GEMM_SHAPES").is_some() {
        eprintln!("APXINF_FP8_BF16_GEMM_SHAPE m={m} n={n} k={k}");
    }
    let output = crate::workspace::output_buffer(ctx, m * n * DType::BF16.size_in_bytes())?;
    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight.values_e4m3).map_err(Error::Cuda)?;
    if crate::workspace::may_prepare_native_resources() {
        let configured_rank = configured_fp8_bf16_heuristic(m, n, k)?;
        let persisted_rank = ctx
            .tuning()
            .lookup_gemm_exact(&bf16_output_tuning_key(ctx, m, n, k))
            .filter(|tactic| tactic.backend == TacticBackend::CublasLt)
            .map(|tactic| tactic.value);
        if let Some(rank) = configured_rank.or(persisted_rank) {
            set_cublaslt_fp8_bf16_gemm_heuristic(m, n, k, rank)?;
        }
        prepare_cublaslt_fp8_gemm_bf16(m, n, k)?;
    }
    let status = unsafe {
        ffi::apxinf_static_fp8_gemm_bf16(
            activation.ptr(),
            weight_buffer.ptr(),
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            activation_scale * weight.scale,
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)?;
    Ok(output.into_tensor(Shape::new(vec![m, n]), DType::BF16))
}

/// Static native E4M3 x E4M3 GEMM with a fused BF16 per-column bias and a
/// BF16 output.  Resource preparation remains outside CUDA Graph capture.
pub fn gemm_fp8_bias_bf16(
    ctx: &CudaContext,
    activation: &Tensor,
    activation_scale: f32,
    weight: Fp8WeightView<'_>,
    bias: &Tensor,
) -> Result<Tensor> {
    if activation.dtype() != DType::F8E4M3
        || weight.values_e4m3.dtype() != DType::F8E4M3
        || bias.dtype() != DType::BF16
    {
        return Err(Error::Other(format!(
            "FP8 fused-bias GEMM expects E4M3/E4M3/BF16, got {}/{}/{}",
            activation.dtype(),
            weight.values_e4m3.dtype(),
            bias.dtype()
        )));
    }
    if weight.dual_geglu_interleaved {
        return Err(Error::Other(
            "FP8 fused-bias GEMM cannot consume dual-GeGLU interleaved weights".into(),
        ));
    }
    if !activation_scale.is_finite()
        || activation_scale <= 0.0
        || !weight.scale.is_finite()
        || weight.scale <= 0.0
    {
        return Err(Error::Other(
            "FP8 fused-bias GEMM scales must be finite and positive".into(),
        ));
    }
    let a = activation.shape().dims();
    let b = weight.values_e4m3.shape().dims();
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] || bias.shape().dims() != [b[1]] {
        return Err(Error::Other(format!(
            "FP8 fused-bias GEMM shape mismatch: {a:?} @ {b:?} + {:?}",
            bias.shape().dims()
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    for tensor in [activation, weight.values_e4m3, bias] {
        if tensor.device() != expected_device {
            return Err(Error::DeviceMismatch {
                expected: expected_device,
                got: tensor.device(),
            });
        }
    }
    if !native_fp8_gemm_supported(ctx)? {
        return Err(Error::Other(
            "FP8 fused-bias GEMM requires native E4M3 Tensor Core support".into(),
        ));
    }

    let (m, k, n) = (a[0], a[1], b[1]);
    let output = crate::workspace::output_buffer(ctx, m * n * DType::BF16.size_in_bytes())?;
    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight.values_e4m3).map_err(Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    if crate::workspace::may_prepare_native_resources() {
        let configured_rank = configured_fp8_bf16_heuristic(m, n, k)?;
        let persisted_rank = ctx
            .tuning()
            .lookup_gemm_exact(&bf16_output_tuning_key(ctx, m, n, k))
            .filter(|tactic| tactic.backend == TacticBackend::CublasLt)
            .map(|tactic| tactic.value);
        if let Some(rank) = configured_rank.or(persisted_rank) {
            set_cublaslt_fp8_bf16_gemm_heuristic(m, n, k, rank)?;
        }
        let status = unsafe {
            ffi::apxinf_static_prepare_fp8_gemm_bias_bf16(
                bias_buffer.ptr(),
                m as i32,
                n as i32,
                k as i32,
            )
        };
        ffi::check_cublas(status).map_err(Error::Cuda)?;
    }
    let status = unsafe {
        ffi::apxinf_static_fp8_gemm_bias_bf16(
            activation.ptr(),
            weight_buffer.ptr(),
            bias_buffer.ptr(),
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            activation_scale * weight.scale,
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)?;
    Ok(output.into_tensor(Shape::new(vec![m, n]), DType::BF16))
}

pub fn prepare_cublaslt_fp8_gemm_split(m: usize, n: usize, k: usize) -> Result<()> {
    let status =
        unsafe { ffi::apxinf_static_prepare_fp8_gemm_split_f16(m as i32, n as i32, k as i32) };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
pub fn cublaslt_fp8_gemm_f16(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) -> Result<()> {
    let status = unsafe {
        ffi::apxinf_static_fp8_gemm_f16(
            activation.ptr(),
            weight.ptr(),
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            alpha,
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
pub fn cublaslt_fp8_gemm_split_f16(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) -> Result<()> {
    let status = unsafe {
        ffi::apxinf_static_fp8_gemm_split_f16(
            activation.ptr(),
            weight.ptr(),
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            alpha,
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
pub fn cublaslt_fp8_gemm_split_first_f16(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) -> Result<()> {
    let status = unsafe {
        ffi::apxinf_static_fp8_gemm_split_first_f16(
            activation.ptr(),
            weight.ptr(),
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            alpha,
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

#[cfg(test)]
mod fp8_dual_geglu_tests {
    use super::*;

    fn geglu_key(sm: u32, m: usize) -> GemmTuningKey {
        GemmTuningKey {
            op: GemmOp::Fp8F16,
            device: DeviceFingerprint {
                sm,
                multiprocessor_count: 20,
            },
            m,
            n: 32768,
            k: 2048,
            activation_dtype: TuningDType::F8E4M3,
            weight_dtype: TuningDType::F8E4M3,
            output_dtype: TuningDType::F8E4M3,
            layout: GemmLayout::RowMajor,
            scale_mode: ScaleMode::PerTensor,
            epilogue: Epilogue::GeGlu,
            workspace_limit: usize::MAX,
        }
    }

    #[test]
    fn bf16_output_tactic_map_is_exact_and_strict() {
        let spec = "41,1536,1536=3;1,3072,1536=4";
        assert_eq!(
            parse_fp8_bf16_heuristic(spec, 41, 1536, 1536).unwrap(),
            Some(3)
        );
        assert_eq!(
            parse_fp8_bf16_heuristic(spec, 41, 1536, 6144).unwrap(),
            None
        );
        assert!(parse_fp8_bf16_heuristic("41,1536=3", 41, 1536, 1536).is_err());
        assert!(parse_fp8_bf16_heuristic("41,1536,1536=64", 41, 1536, 1536).is_err());
    }

    #[test]
    fn dual_backend_accepts_only_validated_m_values_and_tactic_zero() {
        for m in [522, 533] {
            assert!(validate_fp8_dual_geglu_record(GemmOp::Fp8F16, m, 32768, 2048, 0).is_ok());
        }

        for (op, m, n, k, tactic) in [
            (GemmOp::Bf16, 533, 32768, 2048, 0),
            (GemmOp::Fp8F16, 521, 32768, 2048, 0),
            (GemmOp::Fp8F16, 534, 32768, 2048, 0),
            (GemmOp::Fp8F16, 533, 16384, 2048, 0),
            (GemmOp::Fp8F16, 533, 32768, 1024, 0),
            (GemmOp::Fp8F16, 533, 32768, 2048, 1),
        ] {
            assert!(validate_fp8_dual_geglu_record(op, m, n, k, tactic).is_err());
        }
    }

    #[test]
    fn interleaved_only_default_requires_sm110_and_exact_shape() {
        assert!(validate_fp8_interleaved_default(110, &geglu_key(110, 522), true).is_ok());
        assert!(validate_fp8_interleaved_default(89, &geglu_key(89, 522), true).is_err());
        assert!(validate_fp8_interleaved_default(110, &geglu_key(110, 522), false).is_err());
        assert!(validate_fp8_interleaved_default(110, &geglu_key(110, 778), true).is_err());
    }

    #[test]
    fn dual_fused_tactic_does_not_require_gate_buffer() {
        assert!(!fp8_geglu_tactic_uses_gate(TacticId {
            backend: TacticBackend::CutlassFp8DualGeGlu,
            value: 0,
        }));
        assert!(fp8_geglu_tactic_uses_gate(decomposed_geglu_tactic()));
    }

    #[test]
    fn bf16_dual_geglu_backend_is_exact_m533_tactic_zero() {
        assert!(validate_bf16_dual_geglu_record(
            GemmOp::Bf16,
            522,
            32768,
            2048,
            0,
            522,
            "BF16 dual GeGLU",
        )
        .is_ok());
        assert!(validate_bf16_dual_geglu_record(
            GemmOp::Bf16,
            533,
            32768,
            2048,
            0,
            533,
            "BF16 dual GeGLU",
        )
        .is_ok());
        assert!(validate_bf16_dual_geglu_record(
            GemmOp::Bf16,
            533,
            32768,
            2048,
            0,
            522,
            "BF16 dual GeGLU",
        )
        .is_err());

        for (op, m, n, k, tactic) in [
            (GemmOp::Fp8F16, 533, 32768, 2048, 0),
            (GemmOp::Bf16, 522, 32768, 2048, 0),
            (GemmOp::Bf16, 534, 32768, 2048, 0),
            (GemmOp::Bf16, 533, 16384, 2048, 0),
            (GemmOp::Bf16, 533, 32768, 1024, 0),
            (GemmOp::Bf16, 533, 32768, 2048, 1),
        ] {
            assert!(
                validate_bf16_dual_geglu_record(op, m, n, k, tactic, 533, "BF16 dual GeGLU",)
                    .is_err()
            );
        }
    }
}
