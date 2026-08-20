use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::tuning::{
    DeviceFingerprint, Epilogue, GemmLayout, GemmOp, GemmTuningKey, ScaleMode, TacticBackend,
    TuningDType,
};
use crate::workspace::output_buffer;

#[derive(Clone, Copy, Debug)]
pub struct Bf16AutotuneResult {
    pub heuristic_rank: i32,
    pub returned_algorithms: i32,
    pub vendor_ms: f64,
    pub cublaslt_default_ms: f64,
    pub cublaslt_best_ms: f64,
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

#[allow(clippy::too_many_arguments)]
fn benchmark_vendor_bf16(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
    warmup_iterations: usize,
    benchmark_iterations: usize,
) -> Result<f64> {
    let mut evictor = ColdL2Evictor::new(ctx)?;
    for _ in 0..warmup_iterations {
        evictor.evict(ctx)?;
        ctx.cublas()
            .gemm(DType::BF16, m, n, k, 1.0, activation, weight, 0.0, output)
            .map_err(Error::Cuda)?;
    }
    unsafe {
        ffi::check_cuda(ffi::cudaStreamSynchronize(ctx.stream().handle())).map_err(Error::Cuda)?;
    }
    let events = CudaEventPair::new()?;
    let mut milliseconds = 0.0;
    for _ in 0..benchmark_iterations {
        milliseconds += events.measure(ctx, &mut evictor, || {
            ctx.cublas()
                .gemm(DType::BF16, m, n, k, 1.0, activation, weight, 0.0, output)
                .map_err(Error::Cuda)
        })?;
    }
    Ok(milliseconds / benchmark_iterations as f64)
}

/// Cold-L2 exact-shape comparison of the production cuBLAS path and
/// cuBLASLt heuristic candidates. Autotuning must run before graph capture.
pub fn autotune_cublaslt_bf16(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    max_algorithms: i32,
    warmup_iterations: usize,
    benchmark_iterations: usize,
) -> Result<Bf16AutotuneResult> {
    if max_algorithms <= 0 || max_algorithms > 64 {
        return Err(Error::Other(format!(
            "BF16 cuBLASLt max_algorithms must be in 1..=64, got {max_algorithms}"
        )));
    }
    if benchmark_iterations == 0 {
        return Err(Error::Other(
            "BF16 autotune benchmark_iterations must be positive".into(),
        ));
    }
    if activation.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "BF16 autotune expects BF16 operands, got {} and {}",
            activation.dtype(),
            weight.dtype()
        )));
    }
    let a = activation.shape().dims();
    let b = weight.shape().dims();
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
        return Err(Error::Other(format!(
            "BF16 autotune shape mismatch: {a:?} @ {b:?}"
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

    let (m, k, n) = (a[0], a[1], b[1]);
    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let output = CudaBuffer::alloc_zeros(
        m.checked_mul(n)
            .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("BF16 autotune output size overflow".into()))?,
        ctx.device_id(),
    )
    .map_err(Error::Cuda)?;
    let vendor_ms = benchmark_vendor_bf16(
        ctx,
        &activation,
        &weight,
        &output,
        m,
        n,
        k,
        warmup_iterations,
        benchmark_iterations,
    )?;
    let mut did_tune = 0i32;
    let mut returned_algorithms = 0i32;
    let mut best_rank = -1i32;
    let mut default_ms = 0.0f32;
    let mut best_ms = 0.0f32;
    unsafe {
        ffi::check_cublas(ffi::apxinf_static_autotune_cublaslt_bf16_gemm(
            activation.ptr(),
            weight.ptr(),
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            1.0,
            max_algorithms,
            warmup_iterations as i32,
            benchmark_iterations as i32,
            &mut did_tune,
            &mut returned_algorithms,
            &mut best_rank,
            &mut default_ms,
            &mut best_ms,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    if best_rank < 0 || returned_algorithms <= 0 {
        return Err(Error::Other(
            "BF16 cuBLASLt autotune returned no usable algorithm".into(),
        ));
    }
    let _ = did_tune;
    Ok(Bf16AutotuneResult {
        heuristic_rank: best_rank,
        returned_algorithms,
        vendor_ms,
        cublaslt_default_ms: f64::from(default_ms),
        cublaslt_best_ms: f64::from(best_ms),
    })
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
    let use_persisted_cublaslt = crate::tuning::lookup_gemm_exact(&tuning_key(ctx, m, n, k))
        .is_some_and(|tactic| tactic.backend == TacticBackend::CublasLt);
    if use_persisted_cublaslt {
        if crate::workspace::may_prepare_native_resources() {
            unsafe {
                ffi::check_cublas(ffi::apxinf_static_prepare_bf16_gemm(
                    m as i32, n as i32, k as i32,
                ))
                .map_err(Error::Cuda)?;
            }
        }
        unsafe {
            crate::ffi::check_cublas(crate::ffi::apxinf_static_bf16_gemm(
                activation.ptr(),
                weight.ptr(),
                output.ptr(),
                m as i32,
                n as i32,
                k as i32,
                1.0,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        return Ok(output.into_tensor(Shape::new(vec![m, n]), DType::BF16));
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
