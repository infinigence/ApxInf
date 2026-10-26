//! Row-major BF16 GEMM through hipBLAS.

use apxinf_core::{Error, Result, Tensor};

use super::{bf16_ptr, int, rows_cols};
use crate::ffi::{self, check, DTYPE_BF16};
use crate::runtime::HipContext;

/// `out[M, N] = a[M, K] @ b[K, N]` in BF16 with F32 accumulation. Leading
/// dimensions of `a` and `out` fold into M; `b` is a 2-D weight.
///
/// One provider for now. Choosing between hipBLAS and hipBLASLt per shape —
/// neither wins everywhere on gfx1151 — is milestone B.
pub fn gemm(ctx: &HipContext, a: &Tensor, b: &Tensor, out: &Tensor) -> Result<()> {
    let (m, k) = rows_cols(a)?;
    let weight = b.shape().dims();
    let (n_out_rows, n) = rows_cols(out)?;
    if weight.len() != 2 || weight[0] != k || weight[1] != n || n_out_rows != m {
        return Err(Error::Other(format!(
            "apxinf-hip gemm: {} @ {} -> {} is not [M, K] @ [K, N] -> [M, N]",
            a.shape(),
            b.shape(),
            out.shape()
        )));
    }
    let (pa, pb, pc) = (bf16_ptr(ctx, a)?, bf16_ptr(ctx, b)?, bf16_ptr(ctx, out)?);
    check("gemm", unsafe {
        ffi::apxinf_hip_gemm(
            ctx.blas(),
            DTYPE_BF16,
            int(m)?,
            int(n)?,
            int(k)?,
            pa,
            pb,
            pc,
        )
    })
}
