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

#[cfg(test)]
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fp8DualGeGluMode {
    Auto,
    Off,
    On,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fp8DualGeGluWeightRoute {
    Plain,
    InterleavedPrimary,
    InterleavedAuto,
}

fn parse_fp8_dual_geglu_mode(value: Option<&str>) -> Result<Fp8DualGeGluMode> {
    match value {
        None | Some("auto") => Ok(Fp8DualGeGluMode::Auto),
        Some("0" | "off") => Ok(Fp8DualGeGluMode::Off),
        Some("1" | "on") => Ok(Fp8DualGeGluMode::On),
        Some(value) => Err(Error::Other(format!(
            "APXINF_PI05_FP8_DUAL_GEGLU must be auto, 0/off, or 1/on; got {value}"
        ))),
    }
}

fn fp8_dual_geglu_mode() -> Result<Fp8DualGeGluMode> {
    const NAME: &str = "APXINF_PI05_FP8_DUAL_GEGLU";
    match std::env::var(NAME) {
        Err(std::env::VarError::NotPresent) => parse_fp8_dual_geglu_mode(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(Error::Other(format!("{NAME} must be valid Unicode")))
        }
        Ok(value) => parse_fp8_dual_geglu_mode(Some(&value)),
    }
}

fn fp8_dual_geglu_weight_route(
    mode: Fp8DualGeGluMode,
    dual_mega: bool,
    primary_interleaved: bool,
    auto_interleaved_available: bool,
) -> Result<Fp8DualGeGluWeightRoute> {
    match (mode, dual_mega, primary_interleaved, auto_interleaved_available) {
        (Fp8DualGeGluMode::Off, false, false, false) => Ok(Fp8DualGeGluWeightRoute::Plain),
        (Fp8DualGeGluMode::On, true, true, false) => Ok(Fp8DualGeGluWeightRoute::InterleavedPrimary),
        (Fp8DualGeGluMode::Auto, false, false, false) => Ok(Fp8DualGeGluWeightRoute::Plain),
        (Fp8DualGeGluMode::Auto, false, false, true) => Ok(Fp8DualGeGluWeightRoute::Plain),
        (Fp8DualGeGluMode::Auto, true, false, true) => Ok(Fp8DualGeGluWeightRoute::InterleavedAuto),
        _ => Err(Error::Other(format!(
            "FP8 dual GeGLU config/layout mismatch: mode={mode:?}, backend_dual={dual_mega}, primary_interleaved={primary_interleaved}, auto_interleaved={auto_interleaved_available}"
        ))),
    }
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
    let plan = ctx.gemm_plans().resolve_or_tune(
        ctx,
        &key,
        super::plan::default_fp8_tactic(m, n, k),
        |preferred| {
            autotune_request_fp8(
                ctx,
                &key,
                &activation_buffer,
                &weight_buffer,
                alpha,
                preferred,
            )
        },
    )?;
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

/// Run the configured cuBLASLt gate + CUTLASS up/GeGLU/E4M3 fused tactic.
/// Returns `None` unless the exact physical GEMM record selects this backend;
/// a selected but unsupported record fails closed instead of falling back.
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

    let (m, k, full_n) = (a[0], a[1], b[1]);
    let key = tuning_key(ctx, m, full_n, k);
    // A missing/non-fused record must reach the plain GEMM path without
    // caching a Bucket/Default plan first; otherwise AUTO_TUNE would observe
    // that cached plan and skip this exact key.
    let Some(exact_tactic) = ctx.tuning().lookup_gemm_exact(&key) else {
        return Ok(None);
    };
    let fused_backend = matches!(
        exact_tactic.backend,
        TacticBackend::CublasLtCustomSplitGeGluCutlass
            | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto
            | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3
            | TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm
            | TacticBackend::CutlassFp8DualGeGlu
    );
    if !fused_backend {
        return Ok(None);
    }
    let plan =
        ctx.gemm_plans()
            .resolve(ctx, &key, super::plan::default_fp8_tactic(m, full_n, k))?;
    let fused_tactic = (plan.source == super::plan::PlanSource::Exact).then_some(plan.tactic);
    let (cutlass_geglu_tactic, tuned_m, dual_mega) = match fused_tactic.map(|tactic| tactic.backend)
    {
        Some(TacticBackend::CublasLtCustomSplitGeGluCutlass) => (0, 778, false),
        Some(TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto) => (1, 778, false),
        Some(TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3) => (2, 778, false),
        Some(TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm) => (3, 522, false),
        Some(TacticBackend::CutlassFp8DualGeGlu) => (0, m, true),
        _ => return Ok(None),
    };
    if (m, full_n, k) != (tuned_m, 32768, 2048) {
        return Err(Error::Other(format!(
            "fused FP8 GeGLU backend is tuned only for [{tuned_m},2048] @ [2048,32768], got [{m},{k}] @ [{k},{full_n}]"
        )));
    }
    if crate::workspace::fp8_emulation_required(ctx)? {
        return Err(Error::Other(
            "FP8 fused GeGLU requires native FP8 Tensor Cores".into(),
        ));
    }
    let weight_route = fp8_dual_geglu_weight_route(
        fp8_dual_geglu_mode()?,
        dual_mega,
        packed_weight.dual_geglu_interleaved,
        packed_weight.dual_geglu_auto_interleaved.is_some(),
    )?;
    let selected_weight = match weight_route {
        Fp8DualGeGluWeightRoute::Plain | Fp8DualGeGluWeightRoute::InterleavedPrimary => {
            packed_weight.values_e4m3
        }
        Fp8DualGeGluWeightRoute::InterleavedAuto => {
            packed_weight.dual_geglu_auto_interleaved.unwrap()
        }
    };
    if selected_weight.dtype() != DType::F8E4M3 || selected_weight.shape().dims() != b {
        return Err(Error::Other(format!(
            "FP8 dual GeGLU selected weight must be E4M3 {b:?}, got {} {:?}",
            selected_weight.dtype(),
            selected_weight.shape().dims()
        )));
    }
    if selected_weight.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: selected_weight.device(),
        });
    }

    #[cfg(not(apxinf_cutlass_gemm))]
    {
        let _ = (ctx, activation_scale, output_scale);
        return Err(Error::Other(
            "FP8 fused GeGLU requires the SM100-family CUTLASS build".into(),
        ));
    }

    #[cfg(apxinf_cutlass_gemm)]
    {
        let n = full_n / 2;
        let output = crate::workspace::output_buffer(
            ctx,
            m.checked_mul(n)
                .and_then(|elements| elements.checked_mul(DType::F8E4M3.size_in_bytes()))
                .ok_or_else(|| Error::Other("FP8 fused GeGLU output size overflow".into()))?,
        )?;
        let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
        let weight_buffer = CudaBuffer::from_tensor(selected_weight).map_err(Error::Cuda)?;
        if dual_mega {
            let status = unsafe {
                ffi::apxinf_static_cutlass_fp8_dual_gemm_geglu_e4m3(
                    activation_buffer.ptr(),
                    weight_buffer.ptr(),
                    output.ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                    full_n as i32,
                    activation_scale * packed_weight.scale,
                    output_scale,
                    ctx.stream().handle(),
                )
            };
            if status != 0 {
                return Err(Error::Cuda(format!(
                    "FP8 dual-GEMM GeGLU rejected [{m},{n},{k}] ({status})"
                )));
            }
            return Ok(Some(
                output.into_tensor(Shape::new(vec![m, n]), DType::F8E4M3),
            ));
        }
        let gate = crate::workspace::output_buffer(
            ctx,
            m.checked_mul(full_n)
                .and_then(|elements| elements.checked_mul(DType::F16.size_in_bytes()))
                .ok_or_else(|| Error::Other("FP8 fused GeGLU gate size overflow".into()))?,
        )?;
        cublaslt_fp8_gemm_split_first_f16(
            ctx,
            &activation_buffer,
            &weight_buffer,
            &gate,
            m,
            full_n,
            k,
            activation_scale * packed_weight.scale,
        )?;
        let status = unsafe {
            ffi::apxinf_static_cutlass_fp8_gemm_geglu_e4m3(
                activation_buffer.ptr(),
                weight_buffer.ptr(),
                gate.ptr(),
                output.ptr(),
                m as i32,
                n as i32,
                k as i32,
                full_n as i32,
                activation_scale * packed_weight.scale,
                output_scale,
                cutlass_geglu_tactic,
                ctx.stream().handle(),
            )
        };
        if status != 0 {
            return Err(Error::Cuda(format!(
                "FP8 fused GeGLU CUTLASS fused GeGLU rejected [{m},{n},{k}] ({status})"
            )));
        }
        Ok(Some(
            output.into_tensor(Shape::new(vec![m, n]), DType::F8E4M3),
        ))
    }
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
        TacticBackend::Vendor | TacticBackend::CublasLt => {
            cublaslt_fp8_gemm_f16(ctx, activation, weight, output, key.m, key.n, key.k, alpha)
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
        Epilogue::None => Err(Error::Other(
            "plain GEMM cannot use a fused cuBLASLt epilogue configuration".into(),
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

    #[test]
    fn mode_parser_is_tri_state_and_defaults_auto() {
        assert_eq!(
            parse_fp8_dual_geglu_mode(None).unwrap(),
            Fp8DualGeGluMode::Auto
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("auto")).unwrap(),
            Fp8DualGeGluMode::Auto
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("0")).unwrap(),
            Fp8DualGeGluMode::Off
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("off")).unwrap(),
            Fp8DualGeGluMode::Off
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("1")).unwrap(),
            Fp8DualGeGluMode::On
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("on")).unwrap(),
            Fp8DualGeGluMode::On
        );
        assert!(parse_fp8_dual_geglu_mode(Some("invalid")).is_err());
    }

    #[test]
    fn auto_routes_dual_to_copy_and_other_backends_to_plain() {
        assert_eq!(
            fp8_dual_geglu_weight_route(Fp8DualGeGluMode::Auto, true, false, true).unwrap(),
            Fp8DualGeGluWeightRoute::InterleavedAuto
        );
        assert_eq!(
            fp8_dual_geglu_weight_route(Fp8DualGeGluMode::Auto, false, false, true).unwrap(),
            Fp8DualGeGluWeightRoute::Plain
        );
        assert_eq!(
            fp8_dual_geglu_weight_route(Fp8DualGeGluMode::Auto, false, false, false).unwrap(),
            Fp8DualGeGluWeightRoute::Plain
        );
        assert_eq!(
            fp8_dual_geglu_weight_route(Fp8DualGeGluMode::Off, false, false, false).unwrap(),
            Fp8DualGeGluWeightRoute::Plain
        );
        assert_eq!(
            fp8_dual_geglu_weight_route(Fp8DualGeGluMode::On, true, true, false).unwrap(),
            Fp8DualGeGluWeightRoute::InterleavedPrimary
        );
        assert!(fp8_dual_geglu_weight_route(Fp8DualGeGluMode::Off, true, false, false).is_err());
        assert!(fp8_dual_geglu_weight_route(Fp8DualGeGluMode::On, false, true, false).is_err());
        assert!(fp8_dual_geglu_weight_route(Fp8DualGeGluMode::Auto, true, false, false).is_err());
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
