use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use half::bf16;

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::tuning::{
    AutoTuneConfig, AutoTuneEngine, CandidateMeasurement, DeviceFingerprint, Epilogue, GemmLayout,
    GemmOp, GemmTuningKey, ScaleMode, TacticBackend, TacticId, TuningDType, TuningOutcome,
};
use crate::workspace::output_buffer;

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

struct ColdL2Evictor {
    buffer: CudaBuffer,
    bytes: usize,
    seed: u32,
}

impl ColdL2Evictor {
    fn new(ctx: &CudaContext) -> Result<Self> {
        const CUDA_DEV_ATTR_L2_CACHE_SIZE: i32 = 38;
        let mut l2_cache_bytes = 0i32;
        unsafe {
            ffi::check_cuda(ffi::cudaDeviceGetAttribute(
                &mut l2_cache_bytes,
                CUDA_DEV_ATTR_L2_CACHE_SIZE,
                ctx.device_id() as i32,
            ))
            .map_err(Error::Cuda)?;
        }
        let l2_cache_bytes = usize::try_from(l2_cache_bytes)
            .ok()
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| Error::Other("CUDA reported an empty L2 cache".into()))?;
        let bytes = l2_cache_bytes
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(255))
            .map(|bytes| bytes & !255usize)
            .ok_or_else(|| Error::Other("cold-L2 eviction buffer size overflow".into()))?;
        Ok(Self {
            buffer: CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)?,
            bytes,
            seed: 0,
        })
    }

    fn evict(&mut self, ctx: &CudaContext) -> Result<()> {
        self.seed = self.seed.wrapping_add(1);
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_evict_l2(
                self.buffer.ptr(),
                self.bytes,
                self.seed,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        }
    }
}

fn tuning_key(ctx: &CudaContext, m: usize, n: usize, k: usize) -> GemmTuningKey {
    GemmTuningKey {
        op: GemmOp::Bf16,
        device: DeviceFingerprint::from(ctx.caps()),
        m,
        n,
        k,
        activation_dtype: TuningDType::Bf16,
        weight_dtype: TuningDType::Bf16,
        output_dtype: TuningDType::Bf16,
        layout: GemmLayout::RowMajor,
        scale_mode: ScaleMode::None,
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
fn launch_tactic_bf16(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    tactic: TacticId,
) -> Result<()> {
    match tactic.backend {
        TacticBackend::Vendor => ctx
            .cublas()
            .gemm(
                DType::BF16,
                key.m,
                key.n,
                key.k,
                1.0,
                activation,
                weight,
                0.0,
                output,
            )
            .map_err(Error::Cuda),
        TacticBackend::CublasLt | TacticBackend::CublasLtCustom => unsafe {
            ffi::check_cublas(ffi::apxinf_static_bf16_gemm(
                activation.ptr(),
                weight.ptr(),
                output.ptr(),
                key.m as i32,
                key.n as i32,
                key.k as i32,
                1.0,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        },
        TacticBackend::CublasLtCustomSplitSerial => unsafe {
            ffi::check_cublas(ffi::apxinf_static_bf16_gemm_split(
                activation.ptr(),
                weight.ptr(),
                output.ptr(),
                key.m as i32,
                key.n as i32,
                key.k as i32,
                1.0,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        },
        _ => Err(Error::Other(format!(
            "BF16 online autotune cannot execute {tactic:?}"
        ))),
    }
}

fn prepare_tactic_bf16(key: &GemmTuningKey, tactic: TacticId) -> Result<()> {
    super::providers::prepare(key, tactic)
}

fn resolve_bf16_plan(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
) -> Result<super::PreparedGemmPlan> {
    ctx.gemm_plans()
        .resolve_or_tune(ctx, key, super::plan::default_bf16_tactic(), |preferred| {
            autotune_request_bf16(ctx, key, activation, weight, preferred)
        })
}

fn autotune_request_bf16(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    preferred: Option<TacticId>,
) -> Result<TuningOutcome> {
    let elements = key
        .m
        .checked_mul(key.n)
        .ok_or_else(|| Error::Other("BF16 autotune output size overflow".into()))?;
    let bytes = elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("BF16 autotune output size overflow".into()))?;
    let reference_output = CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)?;
    prepare_tactic_bf16(
        key,
        TacticId {
            backend: TacticBackend::Vendor,
            value: 0,
        },
    )?;
    launch_tactic_bf16(
        ctx,
        key,
        activation,
        weight,
        &reference_output,
        TacticId {
            backend: TacticBackend::Vendor,
            value: 0,
        },
    )?;
    ctx.synchronize().map_err(Error::Cuda)?;
    let reference = copy_bf16_output(&reference_output, elements)?;
    drop(reference_output);

    let output = CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let events = CudaEventPair::new()?;
    let mut evictor = ColdL2Evictor::new(ctx)?;
    let engine = AutoTuneEngine::new(AutoTuneConfig::default())?;
    let candidates = super::providers::candidates(key, 64).into_iter();
    engine.tune_with_preferred(key, preferred, candidates, |candidate, config| {
        prepare_tactic_bf16(key, candidate.tactic)?;
        launch_tactic_bf16(ctx, key, activation, weight, &output, candidate.tactic)?;
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
            evictor.evict(ctx)?;
            launch_tactic_bf16(ctx, key, activation, weight, &output, candidate.tactic)?;
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        let mut milliseconds = 0.0;
        for _ in 0..config.benchmark_iterations {
            milliseconds += events.measure(ctx, &mut evictor, || {
                launch_tactic_bf16(ctx, key, activation, weight, &output, candidate.tactic)
            })?;
        }
        Ok(CandidateMeasurement {
            tactic: candidate.tactic,
            milliseconds: Some(milliseconds / config.benchmark_iterations as f64),
            correct: true,
        })
    })
}

pub(crate) fn set_cublaslt_gemm_heuristic(
    m: usize,
    n: usize,
    k: usize,
    heuristic_rank: i32,
) -> Result<()> {
    if !(0..64).contains(&heuristic_rank) {
        return Err(Error::Other(format!(
            "invalid BF16 cuBLASLt heuristic rank {heuristic_rank}"
        )));
    }
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_bf16_gemm_heuristic(
            m as i32,
            n as i32,
            k as i32,
            heuristic_rank,
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

pub(crate) fn set_cublaslt_gemm_custom(m: usize, n: usize, k: usize, tactic: i32) -> Result<()> {
    let config = crate::tuning::decode_cublaslt_custom_tactic(tactic)
        .ok_or_else(|| Error::Other(format!("invalid BF16 cuBLASLt custom tactic {tactic}")))?;
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_bf16_gemm_custom(
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

pub(crate) fn set_cublaslt_gemm_split_custom(
    m: usize,
    n: usize,
    k: usize,
    tactic: i32,
) -> Result<()> {
    let config = crate::tuning::decode_cublaslt_custom_tactic(tactic).ok_or_else(|| {
        Error::Other(format!(
            "invalid BF16 cuBLASLt split-serial custom tactic {tactic}"
        ))
    })?;
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_bf16_gemm_split_custom(
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

pub(crate) fn prepare_cublaslt_gemm(m: usize, n: usize, k: usize, split: bool) -> Result<()> {
    let status = unsafe {
        if split {
            ffi::apxinf_static_prepare_bf16_gemm_split(m as i32, n as i32, k as i32)
        } else {
            ffi::apxinf_static_prepare_bf16_gemm(m as i32, n as i32, k as i32)
        }
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

/// Physical BF16 GEMM contract: `[M,K] @ [K,N] -> [M,N]`.
pub fn gemm_bf16(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if activation.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "gemm_bf16 expects BF16 operands, got {} and {}",
            activation.dtype(),
            weight.dtype()
        )));
    }
    let activation_shape = activation.shape().dims();
    let weight_shape = weight.shape().dims();
    if activation_shape.len() != 2
        || weight_shape.len() != 2
        || activation_shape[1] != weight_shape[0]
    {
        return Err(Error::Other(format!(
            "gemm_bf16 shape mismatch: {activation_shape:?} @ {weight_shape:?}"
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device || weight.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: if activation.device() != expected_device {
                activation.device()
            } else {
                weight.device()
            },
        });
    }
    super::observe_bf16(activation, weight)?;

    let (m, k, n) = (activation_shape[0], activation_shape[1], weight_shape[1]);
    let output = output_buffer(ctx, m * n * DType::BF16.size_in_bytes())?;
    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let key = tuning_key(ctx, m, n, k);
    let plan = resolve_bf16_plan(ctx, &key, &activation, &weight)?;
    let use_split_serial = plan.tactic.backend == TacticBackend::CublasLtCustomSplitSerial;
    let use_persisted_cublaslt = matches!(
        plan.tactic.backend,
        TacticBackend::CublasLt
            | TacticBackend::CublasLtCustom
            | TacticBackend::CublasLtCustomSplitSerial
    );
    if use_persisted_cublaslt {
        let tuned_result = (|| -> Result<()> {
            unsafe {
                let status = if use_split_serial {
                    crate::ffi::apxinf_static_bf16_gemm_split(
                        activation.ptr(),
                        weight.ptr(),
                        output.ptr(),
                        m as i32,
                        n as i32,
                        k as i32,
                        1.0,
                        ctx.stream().handle(),
                    )
                } else {
                    crate::ffi::apxinf_static_bf16_gemm(
                        activation.ptr(),
                        weight.ptr(),
                        output.ptr(),
                        m as i32,
                        n as i32,
                        k as i32,
                        1.0,
                        ctx.stream().handle(),
                    )
                };
                ffi::check_cublas(status).map_err(Error::Cuda)?;
            }
            Ok(())
        })();
        match tuned_result {
            Ok(()) => return Ok(output.into_tensor(Shape::new(vec![m, n]), DType::BF16)),
            Err(error) => {
                eprintln!(
                    "[apxinf] BF16 tactic {:?} failed for {key:?}: {error}; using vendor fallback",
                    plan.tactic
                );
                ctx.gemm_plans().fallback(ctx, &key)?;
            }
        }
    }
    ctx.cublas()
        .gemm(
            DType::BF16,
            m,
            n,
            k,
            1.0,
            &activation,
            &weight,
            0.0,
            &output,
        )
        .map_err(Error::Cuda)?;
    Ok(output.into_tensor(Shape::new(vec![m, n]), DType::BF16))
}

fn geglu_tuning_key(ctx: &CudaContext, m: usize, full_n: usize, k: usize) -> GemmTuningKey {
    let mut key = tuning_key(ctx, m, full_n, k);
    key.epilogue = Epilogue::GeGlu;
    key
}

const fn decomposed_geglu_tactic() -> TacticId {
    TacticId {
        backend: TacticBackend::GemmThenGeGlu,
        value: 0,
    }
}

/// Resolve and execute the complete BF16 gate/up projection plus GeGLU
/// operator. The selected tactic may be fused or a prepared GEMM followed by
/// the standalone activation kernel.
pub fn gemm_bf16_geglu_fused(
    ctx: &CudaContext,
    activation: &Tensor,
    packed_weight: &Tensor,
    bf16_dual_geglu_interleaved: bool,
    bf16_dual_geglu_auto_interleaved: Option<&Tensor>,
    bf16_sm89_geglu_interleaved: Option<&Tensor>,
) -> Result<Tensor> {
    if activation.dtype() != DType::BF16 || packed_weight.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "BF16 fused GeGLU expects BF16 operands, got {} and {}",
            activation.dtype(),
            packed_weight.dtype()
        )));
    }
    let a = activation.shape().dims();
    let b = packed_weight.shape().dims();
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] || b[1] % 2 != 0 {
        return Err(Error::Other(format!(
            "BF16 fused GeGLU shape mismatch: {a:?} @ {b:?}"
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device || packed_weight.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: if activation.device() != expected_device {
                activation.device()
            } else {
                packed_weight.device()
            },
        });
    }
    if let Some(weight) = bf16_dual_geglu_auto_interleaved {
        super::validate_geglu_weight(
            "BF16 dual GeGLU auto-interleaved weight",
            weight,
            DType::BF16,
            b,
            expected_device,
        )?;
    }
    if let Some(weight) = bf16_sm89_geglu_interleaved {
        super::validate_geglu_weight(
            "BF16 SM89 GeGLU interleaved weight",
            weight,
            DType::BF16,
            b,
            expected_device,
        )?;
    }
    super::observe_bf16(activation, packed_weight)?;

    let (m, k, full_n) = (a[0], a[1], b[1]);
    let n = full_n / 2;
    let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let primary_weight = CudaBuffer::from_tensor(packed_weight).map_err(Error::Cuda)?;
    let automatic_interleaved = bf16_dual_geglu_auto_interleaved
        .map(CudaBuffer::from_tensor)
        .transpose()
        .map_err(Error::Cuda)?;
    let sm89_interleaved = bf16_sm89_geglu_interleaved
        .map(CudaBuffer::from_tensor)
        .transpose()
        .map_err(Error::Cuda)?;
    let (plain_weight, dual_weight) = if bf16_dual_geglu_interleaved {
        (None, Some(&primary_weight))
    } else {
        (Some(&primary_weight), automatic_interleaved.as_ref())
    };

    let key = geglu_tuning_key(ctx, m, full_n, k);
    let plain_key = tuning_key(ctx, m, full_n, k);
    // A cold GeGLU key may tune both the complete operator and its decomposed
    // GEMM candidate. Resolve that dependency before the outer tuning session
    // takes its non-reentrant lock.
    let mut plain_plan = if ctx.tuning().lookup_gemm_exact(&key).is_none() {
        plain_weight
            .map(|weight| resolve_bf16_plan(ctx, &plain_key, &activation_buffer, weight))
            .transpose()?
    } else {
        None
    };
    let default = default_bf16_geglu_tactic(
        ctx,
        &key,
        plain_weight.is_some(),
        dual_weight.is_some(),
        sm89_interleaved.is_some(),
    )?;
    let plan = ctx
        .gemm_plans()
        .resolve_or_tune(ctx, &key, default, |preferred| {
            autotune_request_bf16_geglu(
                ctx,
                &key,
                &plain_key,
                plain_plan.as_ref(),
                &activation_buffer,
                plain_weight,
                dual_weight,
                sm89_interleaved.as_ref(),
                preferred,
            )
        })?;

    if plan.tactic.backend == TacticBackend::GemmThenGeGlu && plain_plan.is_none() {
        plain_plan = plain_weight
            .map(|weight| resolve_bf16_plan(ctx, &plain_key, &activation_buffer, weight))
            .transpose()?;
    }
    let mut gate = if bf16_geglu_tactic_uses_gate(plan.tactic) {
        Some(output_buffer(ctx, bf16_geglu_gate_bytes(m, full_n)?)?)
    } else {
        None
    };
    let output = output_buffer(
        ctx,
        m.checked_mul(n)
            .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("BF16 GeGLU output size overflow".into()))?,
    )?;
    if crate::workspace::may_prepare_native_resources() {
        prepare_bf16_geglu_tactic(&key, &plain_key, plain_plan.as_ref(), plan.tactic)?;
    }
    if let Err(error) = launch_bf16_geglu_tactic(
        ctx,
        &key,
        plain_plan.as_ref(),
        &activation_buffer,
        plain_weight,
        dual_weight,
        sm89_interleaved.as_ref(),
        gate.as_ref(),
        &output,
        plan.tactic,
    ) {
        if plan.tactic == default || plain_weight.is_none() {
            return Err(error);
        }
        eprintln!(
            "[apxinf] BF16 GeGLU tactic {:?} failed for {key:?}: {error}; using decomposed fallback",
            plan.tactic
        );
        ctx.gemm_plans()
            .fallback_to(ctx, &key, decomposed_geglu_tactic())?;
        if plain_plan.is_none() {
            plain_plan = plain_weight
                .map(|weight| resolve_bf16_plan(ctx, &plain_key, &activation_buffer, weight))
                .transpose()?;
        }
        if gate.is_none() {
            gate = Some(output_buffer(ctx, bf16_geglu_gate_bytes(m, full_n)?)?);
        }
        prepare_bf16_geglu_tactic(
            &key,
            &plain_key,
            plain_plan.as_ref(),
            decomposed_geglu_tactic(),
        )?;
        launch_bf16_geglu_tactic(
            ctx,
            &key,
            plain_plan.as_ref(),
            &activation_buffer,
            plain_weight,
            dual_weight,
            sm89_interleaved.as_ref(),
            gate.as_ref(),
            &output,
            decomposed_geglu_tactic(),
        )?;
    }
    Ok(output.into_tensor(Shape::new(vec![m, n]), DType::BF16))
}

fn bf16_geglu_tactic_uses_gate(tactic: TacticId) -> bool {
    matches!(
        tactic.backend,
        TacticBackend::GemmThenGeGlu | TacticBackend::CublasLtCustomSplitGeGluCutlassBf16
    )
}

fn bf16_geglu_gate_bytes(m: usize, full_n: usize) -> Result<usize> {
    m.checked_mul(full_n)
        .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("BF16 GeGLU gate size overflow".into()))
}

fn default_bf16_geglu_tactic(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    has_plain_weight: bool,
    has_dual_weight: bool,
    has_sm89_weight: bool,
) -> Result<TacticId> {
    if has_plain_weight {
        return Ok(decomposed_geglu_tactic());
    }
    if ctx.caps().sm == 110 && has_dual_weight {
        let backend = match key.m {
            522 => TacticBackend::CutlassBf16DualGeGluM522,
            533 => TacticBackend::CutlassBf16DualGeGluM533,
            _ => {
                return Err(Error::Other(format!(
                    "BF16 interleaved-only GeGLU has no implementation for {key:?}"
                )))
            }
        };
        return Ok(TacticId { backend, value: 0 });
    }
    if ctx.caps().sm == 89 && has_sm89_weight {
        return Ok(TacticId {
            backend: TacticBackend::CutlassBf16GeGluSm89,
            value: 0,
        });
    }
    Err(Error::Other(format!(
        "BF16 GeGLU has no safe implementation for {key:?}"
    )))
}

fn prepare_bf16_geglu_tactic(
    key: &GemmTuningKey,
    plain_key: &GemmTuningKey,
    plain_plan: Option<&super::PreparedGemmPlan>,
    tactic: TacticId,
) -> Result<()> {
    if tactic.backend == TacticBackend::GemmThenGeGlu {
        let plan = plain_plan.ok_or_else(|| {
            Error::Other("decomposed BF16 GeGLU has no plain-weight GEMM plan".into())
        })?;
        return super::providers::prepare(plain_key, plan.tactic);
    }
    super::providers::prepare(key, tactic)
}

#[allow(clippy::too_many_arguments)]
fn launch_bf16_geglu_tactic(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    plain_plan: Option<&super::PreparedGemmPlan>,
    activation: &CudaBuffer,
    plain_weight: Option<&CudaBuffer>,
    dual_weight: Option<&CudaBuffer>,
    sm89_weight: Option<&CudaBuffer>,
    gate: Option<&CudaBuffer>,
    output: &CudaBuffer,
    tactic: TacticId,
) -> Result<()> {
    let n = key.n / 2;
    match tactic.backend {
        TacticBackend::GemmThenGeGlu => {
            let gate = gate
                .ok_or_else(|| Error::Other("decomposed BF16 GeGLU has no gate buffer".into()))?;
            let plain_plan = plain_plan.ok_or_else(|| {
                Error::Other("decomposed BF16 GeGLU has no prepared GEMM plan".into())
            })?;
            let plain_weight = plain_weight.ok_or_else(|| {
                Error::Other("decomposed BF16 GeGLU has no plain-layout weight".into())
            })?;
            launch_tactic_bf16(
                ctx,
                &plain_plan.key,
                activation,
                plain_weight,
                gate,
                plain_plan.tactic,
            )?;
            unsafe {
                ffi::check_cuda(ffi::apxinf_static_geglu_bf16(
                    gate.ptr(),
                    output.ptr(),
                    key.m as i32,
                    n as i32,
                    ctx.stream().handle(),
                ))
                .map_err(Error::Cuda)
            }
        }
        TacticBackend::CutlassBf16DualGeGluM522 | TacticBackend::CutlassBf16DualGeGluM533 => {
            let expected_m = if tactic.backend == TacticBackend::CutlassBf16DualGeGluM522 {
                522
            } else {
                533
            };
            validate_bf16_dual_geglu_shape(key.m, key.n, key.k, expected_m)?;
            let weight = dual_weight.ok_or_else(|| {
                Error::Other("BF16 dual-GEMM GeGLU has no interleaved weight".into())
            })?;
            #[cfg(apxinf_cutlass_gemm)]
            {
                let status = unsafe {
                    ffi::apxinf_static_cutlass_bf16_dual_gemm_geglu(
                        activation.ptr(),
                        weight.ptr(),
                        output.ptr(),
                        key.m as i32,
                        n as i32,
                        key.k as i32,
                        key.n as i32,
                        ctx.stream().handle(),
                    )
                };
                if status == 0 {
                    Ok(())
                } else {
                    Err(Error::Cuda(format!(
                        "BF16 dual-GEMM GeGLU rejected [{},{},{}] ({status})",
                        key.m, n, key.k
                    )))
                }
            }
            #[cfg(not(apxinf_cutlass_gemm))]
            {
                let _ = weight;
                Err(Error::Other(
                    "BF16 dual GeGLU requires the SM100-family CUTLASS build".into(),
                ))
            }
        }
        TacticBackend::CutlassBf16GeGluSm89 => {
            let weight = sm89_weight
                .ok_or_else(|| Error::Other("BF16 SM89 GeGLU has no interleaved weight".into()))?;
            #[cfg(apxinf_cutlass_bf16_sm89)]
            {
                let status = unsafe {
                    ffi::apxinf_static_cutlass_bf16_interleaved_geglu_sm89(
                        activation.ptr(),
                        weight.ptr(),
                        output.ptr(),
                        key.m as i32,
                        n as i32,
                        key.k as i32,
                        key.n as i32,
                        tactic.value,
                        ctx.stream().handle(),
                    )
                };
                if status == 0 {
                    Ok(())
                } else {
                    Err(Error::Cuda(format!(
                        "BF16 SM89 GeGLU rejected [{},{},{}] ({status})",
                        key.m, n, key.k
                    )))
                }
            }
            #[cfg(not(apxinf_cutlass_bf16_sm89))]
            {
                let _ = weight;
                Err(Error::Other(
                    "BF16 SM89 GeGLU requires the SM89 CUTLASS build".into(),
                ))
            }
        }
        TacticBackend::CublasLtCustomSplitGeGluCutlassBf16 => {
            let gate =
                gate.ok_or_else(|| Error::Other("split BF16 GeGLU has no gate buffer".into()))?;
            let weight = plain_weight.ok_or_else(|| {
                Error::Other("split BF16 GeGLU has no plain-layout weight".into())
            })?;
            if (key.m, key.n, key.k) != (789, 32768, 2048) {
                return Err(Error::Other(format!(
                    "split BF16 GeGLU tactic requires [789,2048] @ [2048,32768], got [{},{}] @ [{},{}]",
                    key.m, key.k, key.k, key.n
                )));
            }
            #[cfg(apxinf_cutlass_gemm)]
            {
                unsafe {
                    ffi::check_cublas(ffi::apxinf_static_bf16_gemm_split_first(
                        activation.ptr(),
                        weight.ptr(),
                        gate.ptr(),
                        key.m as i32,
                        key.n as i32,
                        key.k as i32,
                        1.0,
                        ctx.stream().handle(),
                    ))
                    .map_err(Error::Cuda)?;
                }
                let status = unsafe {
                    ffi::apxinf_static_cutlass_bf16_gemm_geglu(
                        activation.ptr(),
                        weight.ptr(),
                        gate.ptr(),
                        output.ptr(),
                        key.m as i32,
                        n as i32,
                        key.k as i32,
                        key.n as i32,
                        0,
                        ctx.stream().handle(),
                    )
                };
                if status == 0 {
                    Ok(())
                } else {
                    Err(Error::Cuda(format!(
                        "split BF16 GeGLU tactic {:?} rejected {:?} ({status})",
                        tactic, key
                    )))
                }
            }
            #[cfg(not(apxinf_cutlass_gemm))]
            {
                let _ = weight;
                Err(Error::Other(
                    "split BF16 GeGLU requires the SM100-family CUTLASS build".into(),
                ))
            }
        }
        _ => Err(Error::Other(format!(
            "BF16 GeGLU online autotune cannot execute {tactic:?}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn autotune_request_bf16_geglu(
    ctx: &CudaContext,
    key: &GemmTuningKey,
    plain_key: &GemmTuningKey,
    plain_plan: Option<&super::PreparedGemmPlan>,
    activation: &CudaBuffer,
    plain_weight: Option<&CudaBuffer>,
    dual_weight: Option<&CudaBuffer>,
    sm89_weight: Option<&CudaBuffer>,
    preferred: Option<TacticId>,
) -> Result<TuningOutcome> {
    let output_elements = key
        .m
        .checked_mul(key.n / 2)
        .ok_or_else(|| Error::Other("BF16 GeGLU autotune output size overflow".into()))?;
    let gate = CudaBuffer::alloc_zeros(
        key.m
            .checked_mul(key.n)
            .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("BF16 GeGLU autotune gate size overflow".into()))?,
        ctx.device_id(),
    )
    .map_err(Error::Cuda)?;
    let bytes = output_elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("BF16 GeGLU autotune output size overflow".into()))?;
    let reference_output = CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let candidate_output = CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let reference_tactic = default_bf16_geglu_tactic(
        ctx,
        key,
        plain_plan.is_some() && plain_weight.is_some(),
        dual_weight.is_some(),
        sm89_weight.is_some(),
    )?;
    prepare_bf16_geglu_tactic(key, plain_key, plain_plan, reference_tactic)?;
    launch_bf16_geglu_tactic(
        ctx,
        key,
        plain_plan,
        activation,
        plain_weight,
        dual_weight,
        sm89_weight,
        Some(&gate),
        &reference_output,
        reference_tactic,
    )?;
    ctx.synchronize().map_err(Error::Cuda)?;
    let reference = copy_bf16_output(&reference_output, output_elements)?;

    let candidates = super::providers::geglu_candidates(key)
        .into_iter()
        .filter(|candidate| {
            candidate.tactic.backend != TacticBackend::GemmThenGeGlu
                || (plain_plan.is_some() && plain_weight.is_some())
        })
        .filter(|candidate| {
            !matches!(
                candidate.tactic.backend,
                TacticBackend::CutlassBf16DualGeGluM522 | TacticBackend::CutlassBf16DualGeGluM533
            ) || dual_weight.is_some()
        })
        .filter(|candidate| {
            candidate.tactic.backend != TacticBackend::CutlassBf16GeGluSm89 || sm89_weight.is_some()
        });
    let events = CudaEventPair::new()?;
    let mut evictor = ColdL2Evictor::new(ctx)?;
    let engine = AutoTuneEngine::new(AutoTuneConfig::default())?;
    engine.tune_with_preferred(key, preferred, candidates, |candidate, config| {
        prepare_bf16_geglu_tactic(key, plain_key, plain_plan, candidate.tactic)?;
        launch_bf16_geglu_tactic(
            ctx,
            key,
            plain_plan,
            activation,
            plain_weight,
            dual_weight,
            sm89_weight,
            Some(&gate),
            &candidate_output,
            candidate.tactic,
        )?;
        ctx.synchronize().map_err(Error::Cuda)?;
        let actual = copy_bf16_output(&candidate_output, output_elements)?;
        let correct = crate::tuning::outputs_are_close(&reference, &actual, 0.01, 0.9999);
        if !correct {
            return Ok(CandidateMeasurement {
                tactic: candidate.tactic,
                milliseconds: None,
                correct: false,
            });
        }
        for _ in 0..config.warmup_iterations {
            evictor.evict(ctx)?;
            launch_bf16_geglu_tactic(
                ctx,
                key,
                plain_plan,
                activation,
                plain_weight,
                dual_weight,
                sm89_weight,
                Some(&gate),
                &candidate_output,
                candidate.tactic,
            )?;
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        let mut milliseconds = 0.0;
        for _ in 0..config.benchmark_iterations {
            milliseconds += events.measure(ctx, &mut evictor, || {
                launch_bf16_geglu_tactic(
                    ctx,
                    key,
                    plain_plan,
                    activation,
                    plain_weight,
                    dual_weight,
                    sm89_weight,
                    Some(&gate),
                    &candidate_output,
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

fn validate_bf16_dual_geglu_shape(
    m: usize,
    full_n: usize,
    k: usize,
    expected_m: usize,
) -> Result<()> {
    if (m, full_n, k) == (expected_m, 32768, 2048) {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "BF16 dual GeGLU backend requires exact [M{expected_m},2048] @ [2048,32768], got [{m},{k}] @ [{k},{full_n}]"
        )))
    }
}

#[cfg(test)]
mod bf16_dual_geglu_tests {
    use super::*;

    #[test]
    fn dual_fused_tactic_does_not_require_gate_buffer() {
        assert!(!bf16_geglu_tactic_uses_gate(TacticId {
            backend: TacticBackend::CutlassBf16DualGeGluM522,
            value: 0,
        }));
        assert!(bf16_geglu_tactic_uses_gate(decomposed_geglu_tactic()));
    }

    #[test]
    fn bf16_dual_geglu_shape_is_exact_only() {
        assert!(validate_bf16_dual_geglu_shape(522, 32768, 2048, 522).is_ok());
        assert!(validate_bf16_dual_geglu_shape(533, 32768, 2048, 533).is_ok());
        assert!(validate_bf16_dual_geglu_shape(533, 32768, 2048, 522).is_err());
        assert!(validate_bf16_dual_geglu_shape(522, 32768, 2048, 533).is_err());
        assert!(validate_bf16_dual_geglu_shape(521, 32768, 2048, 522).is_err());
        assert!(validate_bf16_dual_geglu_shape(534, 32768, 2048, 533).is_err());
        assert!(validate_bf16_dual_geglu_shape(522, 32752, 2048, 522).is_err());
        assert!(validate_bf16_dual_geglu_shape(522, 32768, 1024, 522).is_err());
    }
}
