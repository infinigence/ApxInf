//! BF16 NT GEMM on Metal 4 tensors and MetalPerformancePrimitives `matmul2d`.
//! Adapted from EngineTailor (MIT); see NOTICE.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

/// BF16 `[1, M, K] x [N, K]^T -> [1, M, N]` with FP32 accumulation in
/// 64 x 64 x 64 `matmul2d` tiles, rounded once to BF16. M, N and K must be
/// multiples of 64. MLX compiles the kernel at its first evaluation, so a
/// device or OS without Metal 4 tensor support fails there, not in `new`.
pub struct MppBf16GemmNt {
    kernel: MetalKernel,
}

impl MppBf16GemmNt {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_mpp_bf16_gemm_nt_64",
                &["a", "w"],
                &["out"],
                include_str!("../native/mpp_bf16_gemm_nt.metal"),
                include_str!("../native/metal_tensor_mpp.h"),
            )?,
        })
    }

    pub fn call(&self, a: &Array, w: &Array) -> Result<Array> {
        let [1, m, k] = *a.shape() else {
            return Err(Error::Contract("MPP GEMM input must be [1, M, K]"));
        };
        let [n, wk] = *w.shape() else {
            return Err(Error::Contract("MPP GEMM weight must be [N, K]"));
        };
        if a.dtype() != MlxDType::BF16
            || w.dtype() != MlxDType::BF16
            || wk != k
            || [m, n, k]
                .iter()
                .any(|&v| v == 0 || v % 64 != 0 || i32::try_from(v).is_err())
        {
            return Err(Error::Contract(
                "MPP GEMM requires BF16 [1, M, K] and [N, K] with M, N and K multiples of 64",
            ));
        }
        // The kernel addresses both inputs and the output with 32-bit indices.
        for (rows, columns) in [(m, k), (n, k), (m, n)] {
            rows.checked_mul(columns)
                .and_then(|extent| extent.checked_sub(1))
                .filter(|&last| last <= i32::MAX as usize)
                .ok_or(Error::Contract("MPP GEMM index exceeds i32"))?;
        }
        // SAFETY: inputs are row-contiguous, and every flattened index of
        // `a`, `w` and the output fits i32 (checked above). Each 128-thread
        // group owns the 64 x 64 output tile at rows `64 * group.y < M` and
        // columns `64 * group.x < N`; `matmul2d` reads rows < M of `a` and
        // rows < N of `w` over K in 64-wide slices with K a multiple of 64.
        // The group stores only its own tile of `[1, M, N]`; tiles do not
        // overlap.
        let out = unsafe {
            self.kernel.call(
                &[a.clone(), w.clone()],
                &[(&[1, m, n], MlxDType::BF16)],
                [128 * (n / 64), m / 64, 1],
                [128, 1, 1],
                &[
                    ("K", TemplateArg::Int(k as i32)),
                    ("N", TemplateArg::Int(n as i32)),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}
