use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::kernels::contracts::gpu_ptr;
use crate::tuning::{
    DeviceFingerprint, Epilogue, GemmLayout, GemmOp, GemmTuningKey, ScaleMode, TacticBackend,
    TacticStore, TuningDType, TuningDb,
};

#[derive(Clone, Copy, Debug)]
pub struct CutlassTacticTiming {
    pub tactic: i32,
    pub milliseconds: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct CublasLtAlgorithmTiming {
    pub heuristic_rank: i32,
    pub milliseconds: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ColdL2TuningMetadata {
    pub l2_cache_bytes: usize,
    pub eviction_buffer_bytes: usize,
}

pub fn cold_l2_tuning_metadata(ctx: &CudaContext) -> Result<ColdL2TuningMetadata> {
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
        l2_cache_bytes,
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

/// Validate and install a read-only tactic database before graph capture.
pub fn install_tuning_db(ctx: &CudaContext, database: &TuningDb) -> Result<()> {
    install_tuning_dbs(ctx, std::slice::from_ref(database))
}

/// Validate and merge all databases before publishing one immutable store.
/// This is the service-startup path when tactics are split across files.
pub fn install_tuning_dbs(ctx: &CudaContext, databases: &[TuningDb]) -> Result<()> {
    let stores = databases
        .iter()
        .map(|database| database.build_store(ctx.caps(), ctx.library_versions()))
        .collect::<Result<Vec<_>>>()?;
    let store = TacticStore::merge(stores)?;
    if let Some(installed) = crate::tuning::installed() {
        return if installed == &store {
            Ok(())
        } else {
            Err(Error::Other(
                "a different CUDA tactic store is already installed".into(),
            ))
        };
    }
    // The current cuBLASLt C ABI caches plans internally. Seed its immutable
    // startup configuration before publishing the Rust store so inference
    // never mutates tactic state.
    for record in store.gemm_records() {
        match record.tactic.backend {
            TacticBackend::Cutlass => {}
            TacticBackend::CublasLt => match record.key.op {
                GemmOp::Bf16 => super::bf16::set_cublaslt_gemm_heuristic(
                    record.key.m,
                    record.key.n,
                    record.key.k,
                    record.tactic.value,
                )?,
                GemmOp::Fp8F16 => set_cublaslt_gemm_heuristic(
                    record.key.m,
                    record.key.n,
                    record.key.k,
                    record.tactic.value,
                )?,
                _ => {}
            },
            TacticBackend::Vendor => {}
        }
    }
    crate::tuning::install(store)
}

/// Borrowed static-per-tensor FP8 weight contract.
#[derive(Clone, Copy)]
pub struct Fp8WeightView<'a> {
    pub values_e4m3: &'a Tensor,
    pub scale: f32,
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
    crate::tuning::lookup_gemm_exact(&tuning_key(ctx, m, n, k))
}

#[cfg(apxinf_cutlass_gemm)]
fn selected_cutlass_tactic(ctx: &CudaContext, m: usize, n: usize, k: usize) -> i32 {
    let key = tuning_key(ctx, m, n, k);
    crate::tuning::lookup_gemm_exact(&key)
        .or_else(|| crate::tuning::lookup_gemm(&key))
        .filter(|tactic| tactic.backend == TacticBackend::Cutlass)
        .map(|tactic| tactic.value)
        .unwrap_or_else(|| {
            if m <= 16 {
                0
            } else if m <= 64 {
                1
            } else if m <= 256 {
                2
            } else {
                3
            }
        })
}

/// Physical static FP8 GEMM with FP16 output.
pub fn gemm_fp8(
    ctx: &CudaContext,
    activation: &Tensor,
    activation_scale: f32,
    weight: Fp8WeightView<'_>,
) -> Result<Tensor> {
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

    let use_cublaslt = crate::tuning::lookup_gemm_exact(&tuning_key(ctx, m, n, k))
        .is_some_and(|tactic| tactic.backend == TacticBackend::CublasLt);
    #[cfg(not(apxinf_cutlass_gemm))]
    let _ = use_cublaslt;
    #[cfg(apxinf_cutlass_gemm)]
    if n >= 1024 && n % 16 == 0 && k % 16 == 0 && !use_cublaslt {
        let tactic = selected_cutlass_tactic(ctx, m, n, k);
        if cutlass_fp8_gemm_f16(
            ctx,
            &activation_buffer,
            &weight_buffer,
            &output,
            m,
            n,
            k,
            activation_scale * weight.scale,
            tactic,
        )? {
            return Ok(output.into_tensor(Shape::new(vec![m, n]), DType::F16));
        }
    }
    if crate::workspace::may_prepare_native_resources() {
        prepare_cublaslt_fp8_gemm(m, n, k)?;
    }
    cublaslt_fp8_gemm_f16(
        ctx,
        &activation_buffer,
        &weight_buffer,
        &output,
        m,
        n,
        k,
        activation_scale * weight.scale,
    )?;
    Ok(output.into_tensor(Shape::new(vec![m, n]), DType::F16))
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

#[cfg(apxinf_cutlass_gemm)]
pub fn autotune_cutlass_gemm_f16(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
    warmup: usize,
    iterations: usize,
) -> Result<Vec<CutlassTacticTiming>> {
    if iterations == 0 {
        return Err(Error::Other(
            "CUTLASS autotune iterations must be non-zero".into(),
        ));
    }
    let a = activation.shape().dims();
    let b = weight.shape().dims();
    if activation.dtype() != DType::F8E4M3
        || weight.dtype() != DType::F8E4M3
        || a.len() != 2
        || b.len() != 2
        || a[1] != b[0]
        || b[1] % 16 != 0
        || a[1] % 16 != 0
    {
        return Err(Error::Other(
            "CUTLASS autotune expects aligned FP8 [M,K] @ [K,N]".into(),
        ));
    }
    let (m, k, n) = (a[0], a[1], b[1]);
    let output = CudaBuffer::alloc_zeros(m * n * 2, ctx.device_id()).map_err(Error::Cuda)?;
    let mut evictor = ColdL2Evictor::new(ctx)?;
    let events = CudaEventPair::new()?;
    let mut timings = Vec::new();
    // All exposed candidates are ordinary auto-scheduled one-SM kernels.
    // Explicit two-SM schedules are intentionally not compiled because they
    // can wedge CUDA graph replay on the current Thor-U software stack.
    for tactic in 0..=7 {
        let launch = || -> Result<()> {
            let status = unsafe {
                ffi::apxinf_static_cutlass_fp8_gemm_f16(
                    gpu_ptr(activation)?,
                    gpu_ptr(weight)?,
                    output.ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                    activation_scale * weight_scale,
                    tactic,
                    ctx.stream().handle(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(Error::Cuda(format!(
                    "CUTLASS tactic {tactic} rejected shape [{m},{n},{k}] ({status})"
                )))
            }
        };
        if (0..warmup)
            .try_for_each(|_| {
                evictor.evict(ctx)?;
                launch()
            })
            .is_err()
        {
            continue;
        }
        ctx.stream().synchronize().map_err(Error::Cuda)?;
        let mut milliseconds = 0.0f64;
        for _ in 0..iterations {
            milliseconds += events.measure(ctx, &mut evictor, &launch)?;
        }
        timings.push(CutlassTacticTiming {
            tactic,
            milliseconds: milliseconds / iterations as f64,
        });
    }
    Ok(timings)
}

pub fn autotune_cublaslt_gemm_f16(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
    max_algorithms: usize,
    warmup: usize,
    iterations: usize,
) -> Result<Vec<CublasLtAlgorithmTiming>> {
    if max_algorithms == 0 || max_algorithms > 64 || iterations == 0 {
        return Err(Error::Other(
            "cuBLASLt autotune expects 1..=64 algorithms and non-zero iterations".into(),
        ));
    }
    let a = activation.shape().dims();
    let b = weight.shape().dims();
    if activation.dtype() != DType::F8E4M3
        || weight.dtype() != DType::F8E4M3
        || a.len() != 2
        || b.len() != 2
        || a[1] != b[0]
    {
        return Err(Error::Other(
            "cuBLASLt autotune expects FP8 [M,K] @ [K,N]".into(),
        ));
    }
    let (m, k, n) = (a[0], a[1], b[1]);
    let output = CudaBuffer::alloc_zeros(m * n * 2, ctx.device_id()).map_err(Error::Cuda)?;
    let evictor = ColdL2Evictor::new(ctx)?;
    let mut returned = 0i32;
    let mut milliseconds = vec![-1.0f32; max_algorithms];
    let status = unsafe {
        ffi::apxinf_static_autotune_cublaslt_fp8_gemm_f16(
            gpu_ptr(activation)?,
            gpu_ptr(weight)?,
            output.ptr(),
            evictor.buffer.ptr(),
            evictor.metadata.eviction_buffer_bytes,
            m as i32,
            n as i32,
            k as i32,
            activation_scale * weight_scale,
            max_algorithms as i32,
            warmup as i32,
            iterations as i32,
            &mut returned,
            milliseconds.as_mut_ptr(),
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)?;
    Ok(milliseconds
        .into_iter()
        .take(returned.max(0) as usize)
        .enumerate()
        .filter(|(_, milliseconds)| *milliseconds >= 0.0)
        .map(|(heuristic_rank, milliseconds)| CublasLtAlgorithmTiming {
            heuristic_rank: heuristic_rank as i32,
            milliseconds: milliseconds as f64,
        })
        .collect())
}
