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
        TacticBackend::CublasLt => unsafe {
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
        _ => Err(Error::Other(format!(
            "BF16 online autotune cannot execute {tactic:?}"
        ))),
    }
}

fn prepare_tactic_bf16(key: &GemmTuningKey, tactic: TacticId) -> Result<()> {
    super::providers::prepare(key, tactic)
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
    let candidates = super::providers::candidates(key, 64)
        .into_iter()
        .filter(|candidate| {
            matches!(
                candidate.tactic.backend,
                TacticBackend::Vendor | TacticBackend::CublasLt
            )
        });
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

    let (m, k, n) = (activation_shape[0], activation_shape[1], weight_shape[1]);
    let output = output_buffer(ctx, m * n * DType::BF16.size_in_bytes())?;
    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let key = tuning_key(ctx, m, n, k);
    let plan = ctx.gemm_plans().resolve_or_tune(
        ctx,
        &key,
        super::plan::default_bf16_tactic(),
        |preferred| autotune_request_bf16(ctx, &key, &activation, &weight, preferred),
    )?;
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

/// Run the configured cuBLASLt BF16 gate + native SM100 CUTLASS up/GeGLU EVT.
/// Returns `None` unless the exact physical record selects BF16 fused GeGLU;
/// selected but unsupported records fail closed instead of falling back.
pub fn gemm_bf16_geglu_fused(
    ctx: &CudaContext,
    activation: &Tensor,
    packed_weight: &Tensor,
    bf16_dual_geglu_interleaved: bool,
    bf16_dual_geglu_auto_interleaved: Option<&Tensor>,
    bf16_sm89_geglu_interleaved: Option<&Tensor>,
) -> Result<Option<Tensor>> {
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

    let (m, k, full_n) = (a[0], a[1], b[1]);
    let key = tuning_key(ctx, m, full_n, k);
    // Do not cache Bucket/Default from this optional fused probe. On an exact
    // miss the caller falls back to the plain GEMM API, which must retain the
    // opportunity to run AUTO_TUNE for this physical key.
    let Some(exact_tactic) = ctx.tuning().lookup_gemm_exact(&key) else {
        return Ok(None);
    };
    let fused_backend = matches!(
        exact_tactic.backend,
        TacticBackend::CublasLtCustomSplitGeGluCutlassBf16
            | TacticBackend::CutlassBf16DualGeGluM522
            | TacticBackend::CutlassBf16DualGeGluM533
    );
    if !fused_backend {
        return Ok(None);
    }
    let plan = ctx
        .gemm_plans()
        .resolve(ctx, &key, super::plan::default_bf16_tactic())?;
    let fused_tactic = (plan.source == super::plan::PlanSource::Exact).then_some(plan.tactic);
    let bf16_split_evt = fused_tactic
        .is_some_and(|tactic| tactic.backend == TacticBackend::CublasLtCustomSplitGeGluCutlassBf16);
    let bf16_dual_geglu_expected_m = fused_tactic.and_then(|tactic| match tactic.backend {
        TacticBackend::CutlassBf16DualGeGluM522 => Some(522),
        TacticBackend::CutlassBf16DualGeGluM533 => Some(533),
        _ => None,
    });
    let bf16_dual_geglu = bf16_dual_geglu_expected_m.is_some();
    let weight_route = bf16_dual_geglu_weight_route(
        bf16_dual_geglu_mode()?,
        bf16_dual_geglu,
        bf16_dual_geglu_interleaved,
        bf16_dual_geglu_auto_interleaved.is_some(),
    )?;
    let selected_weight = match weight_route {
        Bf16DualGeGluWeightRoute::Plain | Bf16DualGeGluWeightRoute::InterleavedPrimary => {
            packed_weight
        }
        Bf16DualGeGluWeightRoute::InterleavedAuto => bf16_dual_geglu_auto_interleaved.unwrap(),
    };
    if selected_weight.dtype() != DType::BF16 || selected_weight.shape().dims() != b {
        return Err(Error::Other(format!(
            "BF16 dual GeGLU selected weight must be BF16 {b:?}, got {} {:?}",
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
    if ctx.caps().sm == 89 && ((full_n, k) == (8192, 1024) || (full_n, k) == (32768, 2048)) {
        let Some(sm89_weight) = bf16_sm89_geglu_interleaved else {
            return Ok(None);
        };
        if sm89_weight.dtype() != DType::BF16 || sm89_weight.shape().dims() != b {
            return Err(Error::Other(format!(
                "BF16 SM89 GeGLU selected weight must be BF16 {b:?}, got {} {:?}",
                sm89_weight.dtype(),
                sm89_weight.shape().dims()
            )));
        }
        if sm89_weight.device() != expected_device {
            return Err(Error::DeviceMismatch {
                expected: expected_device,
                got: sm89_weight.device(),
            });
        }

        #[cfg(not(apxinf_cutlass_bf16_sm89))]
        return Err(Error::Other(
            "BF16 SM89 fused GeGLU requires the SM89 CUTLASS adapter build".into(),
        ));

        #[cfg(apxinf_cutlass_bf16_sm89)]
        {
            let n = full_n / 2;
            let output = output_buffer(
                ctx,
                m.checked_mul(n)
                    .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
                    .ok_or_else(|| {
                        Error::Other("BF16 SM89 fused GeGLU output size overflow".into())
                    })?,
            )?;
            let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
            let weight_buffer = CudaBuffer::from_tensor(sm89_weight).map_err(Error::Cuda)?;
            let status = unsafe {
                ffi::apxinf_static_cutlass_bf16_interleaved_geglu_sm89(
                    activation_buffer.ptr(),
                    weight_buffer.ptr(),
                    output.ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                    full_n as i32,
                    0,
                    ctx.stream().handle(),
                )
            };
            if status != 0 {
                return Err(Error::Cuda(format!(
                    "BF16 SM89 interleaved GeGLU rejected [{m},{n},{k}] ({status})"
                )));
            }
            return Ok(Some(
                output.into_tensor(Shape::new(vec![m, n]), DType::BF16),
            ));
        }
    }

    if !bf16_split_evt && !bf16_dual_geglu {
        return Ok(None);
    }
    if let Some(expected_m) = bf16_dual_geglu_expected_m {
        validate_bf16_dual_geglu_shape(m, full_n, k, expected_m)?;
        #[cfg(not(apxinf_cutlass_gemm))]
        return Err(Error::Other(
            "BF16 dual GeGLU requires the SM100-family CUTLASS build".into(),
        ));
        #[cfg(apxinf_cutlass_gemm)]
        {
            let n = full_n / 2;
            let output = output_buffer(
                ctx,
                m.checked_mul(n)
                    .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
                    .ok_or_else(|| Error::Other("BF16 dual GeGLU output size overflow".into()))?,
            )?;
            let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
            let weight_buffer = CudaBuffer::from_tensor(selected_weight).map_err(Error::Cuda)?;
            let status = unsafe {
                ffi::apxinf_static_cutlass_bf16_dual_gemm_geglu(
                    activation_buffer.ptr(),
                    weight_buffer.ptr(),
                    output.ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                    full_n as i32,
                    ctx.stream().handle(),
                )
            };
            if status != 0 {
                return Err(Error::Cuda(format!(
                    "BF16 dual-GEMM GeGLU rejected [{m},{n},{k}] ({status})"
                )));
            }
            return Ok(Some(
                output.into_tensor(Shape::new(vec![m, n]), DType::BF16),
            ));
        }
    }
    if (m, full_n, k) != (789, 32768, 2048) {
        return Err(Error::Other(format!(
            "fused BF16 GeGLU is tuned only for [789,2048] @ [2048,32768], got [{m},{k}] @ [{k},{full_n}]"
        )));
    }

    #[cfg(not(apxinf_cutlass_gemm))]
    {
        let _ = ctx;
        return Err(Error::Other(
            "BF16 fused GeGLU requires the SM100-family CUTLASS build".into(),
        ));
    }

    #[cfg(apxinf_cutlass_gemm)]
    {
        let n = full_n / 2;
        let gate = output_buffer(
            ctx,
            m.checked_mul(full_n)
                .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
                .ok_or_else(|| Error::Other("BF16 fused GeGLU gate size overflow".into()))?,
        )?;
        let output = output_buffer(
            ctx,
            m.checked_mul(n)
                .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
                .ok_or_else(|| Error::Other("BF16 fused GeGLU output size overflow".into()))?,
        )?;
        let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
        let weight_buffer = CudaBuffer::from_tensor(selected_weight).map_err(Error::Cuda)?;
        unsafe {
            ffi::check_cublas(ffi::apxinf_static_bf16_gemm_split_first(
                activation_buffer.ptr(),
                weight_buffer.ptr(),
                gate.ptr(),
                m as i32,
                full_n as i32,
                k as i32,
                1.0,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        // Tactic 0 is the bitwise-exact 128x256x64, c1x2, explicit-1SM winner
        // selected by the five-run alternating direct comparison.
        let cutlass_tactic = 0;
        let status = unsafe {
            ffi::apxinf_static_cutlass_bf16_gemm_geglu(
                activation_buffer.ptr(),
                weight_buffer.ptr(),
                gate.ptr(),
                output.ptr(),
                m as i32,
                n as i32,
                k as i32,
                full_n as i32,
                cutlass_tactic,
                ctx.stream().handle(),
            )
        };
        if status != 0 {
            return Err(Error::Cuda(format!(
                "BF16 fused GeGLU CUTLASS fused GeGLU rejected [{m},{n},{k}] ({status})"
            )));
        }
        Ok(Some(
            output.into_tensor(Shape::new(vec![m, n]), DType::BF16),
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Bf16DualGeGluMode {
    Auto,
    Off,
    On,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Bf16DualGeGluWeightRoute {
    Plain,
    InterleavedPrimary,
    InterleavedAuto,
}

fn parse_bf16_dual_geglu_mode(value: Option<&str>) -> Result<Bf16DualGeGluMode> {
    match value {
        None | Some("auto") => Ok(Bf16DualGeGluMode::Auto),
        Some("0" | "off") => Ok(Bf16DualGeGluMode::Off),
        Some("1" | "on") => Ok(Bf16DualGeGluMode::On),
        Some(value) => Err(Error::Other(format!(
            "APXINF_PI05_BF16_DUAL_GEGLU must be auto, 0/off, or 1/on; got {value}"
        ))),
    }
}

fn bf16_dual_geglu_mode() -> Result<Bf16DualGeGluMode> {
    const NAME: &str = "APXINF_PI05_BF16_DUAL_GEGLU";
    match std::env::var(NAME) {
        Err(std::env::VarError::NotPresent) => parse_bf16_dual_geglu_mode(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(Error::Other(format!("{NAME} must be valid Unicode")))
        }
        Ok(value) => parse_bf16_dual_geglu_mode(Some(&value)),
    }
}

fn bf16_dual_geglu_weight_route(
    mode: Bf16DualGeGluMode,
    dual: bool,
    primary_interleaved: bool,
    auto_interleaved_available: bool,
) -> Result<Bf16DualGeGluWeightRoute> {
    match (
        mode,
        dual,
        primary_interleaved,
        auto_interleaved_available,
    ) {
        (Bf16DualGeGluMode::Off, false, false, false) => Ok(Bf16DualGeGluWeightRoute::Plain),
        (Bf16DualGeGluMode::On, true, true, false) => Ok(Bf16DualGeGluWeightRoute::InterleavedPrimary),
        (Bf16DualGeGluMode::Auto, false, false, false) => Ok(Bf16DualGeGluWeightRoute::Plain),
        (Bf16DualGeGluMode::Auto, false, false, true) => Ok(Bf16DualGeGluWeightRoute::Plain),
        (Bf16DualGeGluMode::Auto, true, false, true) => Ok(Bf16DualGeGluWeightRoute::InterleavedAuto),
        _ => Err(Error::Other(format!(
            "BF16 dual GeGLU config/layout mismatch: mode={mode:?}, backend_dual={dual}, primary_interleaved={primary_interleaved}, auto_interleaved={auto_interleaved_available}"
        ))),
    }
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
    fn bf16_dual_geglu_mode_parser_is_strict_and_defaults_auto() {
        assert_eq!(
            parse_bf16_dual_geglu_mode(None).unwrap(),
            Bf16DualGeGluMode::Auto
        );
        assert_eq!(
            parse_bf16_dual_geglu_mode(Some("auto")).unwrap(),
            Bf16DualGeGluMode::Auto
        );
        assert_eq!(
            parse_bf16_dual_geglu_mode(Some("0")).unwrap(),
            Bf16DualGeGluMode::Off
        );
        assert_eq!(
            parse_bf16_dual_geglu_mode(Some("off")).unwrap(),
            Bf16DualGeGluMode::Off
        );
        assert_eq!(
            parse_bf16_dual_geglu_mode(Some("1")).unwrap(),
            Bf16DualGeGluMode::On
        );
        assert_eq!(
            parse_bf16_dual_geglu_mode(Some("on")).unwrap(),
            Bf16DualGeGluMode::On
        );
        assert!(parse_bf16_dual_geglu_mode(Some("invalid")).is_err());
    }

    #[test]
    fn bf16_dual_geglu_route_truth_table_covers_all_twenty_four_states() {
        for mode in [
            Bf16DualGeGluMode::Auto,
            Bf16DualGeGluMode::Off,
            Bf16DualGeGluMode::On,
        ] {
            for dual in [false, true] {
                for primary in [false, true] {
                    for automatic in [false, true] {
                        let expected = match (mode, dual, primary, automatic) {
                            (Bf16DualGeGluMode::Off, false, false, false) => {
                                Some(Bf16DualGeGluWeightRoute::Plain)
                            }
                            (Bf16DualGeGluMode::On, true, true, false) => {
                                Some(Bf16DualGeGluWeightRoute::InterleavedPrimary)
                            }
                            (Bf16DualGeGluMode::Auto, false, false, false) => {
                                Some(Bf16DualGeGluWeightRoute::Plain)
                            }
                            (Bf16DualGeGluMode::Auto, false, false, true) => {
                                Some(Bf16DualGeGluWeightRoute::Plain)
                            }
                            (Bf16DualGeGluMode::Auto, true, false, true) => {
                                Some(Bf16DualGeGluWeightRoute::InterleavedAuto)
                            }
                            _ => None,
                        };
                        let actual = bf16_dual_geglu_weight_route(mode, dual, primary, automatic);
                        match expected {
                            Some(route) => assert_eq!(actual.unwrap(), route),
                            None => assert!(actual.is_err()),
                        }
                    }
                }
            }
        }
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
