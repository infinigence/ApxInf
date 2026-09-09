mod bf16;
mod fp8;
mod plan;
mod providers;
pub mod w4a16;
pub mod w4a16_marlin;
mod w8a8;

use std::cell::RefCell;
use std::rc::Rc;

use apxinf_core::{DType, Device, Error, Result, Tensor};

use super::contracts::{checked_bytes, require_buffers, require_finite};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::cublas::CublasTranspose;
use crate::tuning::{TacticStore, TuningDb, TuningMode, TuningPaths, TuningSession};

pub(crate) use fp8::resolve_fused_plan as resolve_fused_fp8_plan;
pub(crate) use plan::GemmPlanCache;
pub use plan::{PlanSource, PreparedGemmPlan};

pub use bf16::{gemm_bf16 as bf16, gemm_bf16_geglu_fused as bf16_geglu_fused};
#[cfg(test)]
pub(crate) use fp8::prepare_cublaslt_fp8_gemm;
pub use fp8::{
    exact_fp8_tactic, gemm_fp8 as fp8, gemm_fp8_geglu_fused as fp8_geglu_fused,
    native_fp8_gemm_supported as native_fp8_supported, Fp8WeightView,
};
#[cfg(test)]
pub(crate) use w8a8::gemm_w8a8_with_preference;
pub use w8a8::{gemm_w8a8 as w8a8, W8A8Layout, W8A8ScaleMode, W8A8WeightView};

/// Validate and install a read-only tactic database before graph capture.
pub fn install_tuning_db(ctx: &CudaContext, database: &TuningDb) -> Result<()> {
    install_tuning_dbs(ctx, std::slice::from_ref(database))
}

/// Validate and merge databases before installing one runtime-owned session.
pub fn install_tuning_dbs(ctx: &CudaContext, databases: &[TuningDb]) -> Result<()> {
    configure_tuning(ctx, TuningMode::Inference, databases, None)
}

/// Configure tuning before model preparation. Provider-native plans are
/// created lazily only for keys reached by the real workload, then retained by
/// `GemmPlanCache`; a growing hardware database adds no unrelated startup work.
pub fn configure_tuning(
    ctx: &CudaContext,
    mode: TuningMode,
    databases: &[TuningDb],
    paths: Option<TuningPaths>,
) -> Result<()> {
    let stores = databases
        .iter()
        .map(|database| database.build_store(ctx.caps(), ctx.library_versions()))
        .collect::<Result<Vec<_>>>()?;
    let store = TacticStore::merge(stores)?;
    ctx.install_tuning(TuningSession::new(mode, store, paths))
        .map_err(Error::Other)
}

/// Internal observer used by model calibration to inspect BF16 GEMM inputs.
/// It is thread-local so normal inference pays only one empty-cell check and
/// concurrent model threads cannot observe each other's activations.
pub trait Bf16ActivationObserver {
    fn observe(&self, activation: &Tensor, weight: &Tensor) -> Result<()>;
}

thread_local! {
    static BF16_OBSERVER: RefCell<Option<Rc<dyn Bf16ActivationObserver>>> = RefCell::new(None);
}

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

pub(super) fn observe_bf16(activation: &Tensor, weight: &Tensor) -> Result<()> {
    BF16_OBSERVER.with(|slot| {
        if let Some(observer) = slot.borrow().as_ref() {
            observer.observe(activation, weight)?;
        }
        Ok(())
    })
}

pub fn matmul(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if activation.dtype() != weight.dtype() {
        return Err(Error::DTypeMismatch {
            expected: activation.dtype(),
            got: weight.dtype(),
        });
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
    let output_shape = activation.shape().matmul_shape(weight.shape())?;
    let m = activation.shape().dims()[activation.ndim() - 2];
    let k = activation.shape().dims()[activation.ndim() - 1];
    let n = weight.shape().dims()[weight.ndim() - 1];
    let output = CudaBuffer::alloc_zeros(
        output_shape.numel() * activation.dtype().size_in_bytes(),
        ctx.device_id(),
    )
    .map_err(Error::Cuda)?;
    let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
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
            &output,
        )
        .map_err(Error::Cuda)?;
    Ok(output.into_tensor(output_shape, activation.dtype()))
}

/// Row-major `A[M,K] @ B[K,N]` into caller-owned storage.
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
        .map_err(apxinf_core::Error::Cuda)
}

/// Contiguous BF16 `A[M,K] @ weight[N,K]^T` with FP32 accumulation and output.
/// Retains checkpoint BF16 weights without rounding the result back to BF16.
pub fn bf16_f32_output_into(
    ctx: &CudaContext,
    a: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use crate::ffi;
    use std::ffi::c_void;
    if [m, n, k].iter().any(|&x| x > i32::MAX as usize) {
        return Err(Error::Other(
            "BF16_F32_GEMM dimensions exceed cuBLAS i32 range".into(),
        ));
    }
    require_buffers(
        ctx,
        "BF16_F32_GEMM",
        &[
            (
                "A",
                a,
                checked_bytes(DType::BF16, &[m, k], "BF16_F32_GEMM")?,
            ),
            (
                "weight",
                weight,
                checked_bytes(DType::BF16, &[n, k], "BF16_F32_GEMM")?,
            ),
            (
                "output",
                output,
                checked_bytes(DType::F32, &[m, n], "BF16_F32_GEMM")?,
            ),
        ],
    )?;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // Row-major transpose identity: C^T[N,M] = weight[N,K] @ A^T[K,M].
    unsafe {
        ffi::check_cublas(ffi::cublasGemmEx(
            ctx.cublas().raw(),
            ffi::cublasOperation_t::CUBLAS_OP_T,
            ffi::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            &alpha as *const f32 as *const c_void,
            weight.ptr(),
            ffi::cudaDataType_t::CUDA_R_16BF,
            k as i32,
            a.ptr(),
            ffi::cudaDataType_t::CUDA_R_16BF,
            k as i32,
            &beta as *const f32 as *const c_void,
            output.ptr(),
            ffi::cudaDataType_t::CUDA_R_32F,
            n as i32,
            ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            -1,
        ))
    }
    .map_err(Error::Cuda)
}

/// GEMM with explicit transpose and row-stride contracts.
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
        return Err(apxinf_core::Error::Other(format!(
            "GEMM_EX invalid row strides lda={lda}, ldb={ldb}, ldc={ldc}"
        )));
    }
    let strided_bytes = |rows: usize, stride: i32, cols: usize| -> Result<usize> {
        let elements = rows
            .saturating_sub(1)
            .checked_mul(stride as usize)
            .and_then(|offset| offset.checked_add(cols))
            .ok_or_else(|| apxinf_core::Error::Other("GEMM_EX buffer size overflow".into()))?;
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
        .map_err(apxinf_core::Error::Cuda)
}
