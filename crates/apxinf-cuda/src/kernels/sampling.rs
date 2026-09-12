//! Greedy token selection over BF16 logits.
//!
//! π0-FAST decodes action tokens by taking the argmax of each step's LM head
//! output. The winning index must stay on device so a captured graph can feed it
//! straight into `embedding::lookup_into`; nothing in the steady-state loop may
//! round-trip through the host.

use apxinf_core::{DType, Error, Result, Tensor};

use super::contracts::{check_cuda, gpu_ptr, require_address};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;

/// Write the index of the maximum element of 1-D BF16 `logits` into `out`.
///
/// `out` must hold at least one `u32`. Callers that need the index on the host
/// can pass a host-mapped buffer and read it after synchronizing the stream.
pub fn argmax_bf16_into(ctx: &CudaContext, logits: &Tensor, out: &CudaBuffer) -> Result<()> {
    if logits.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "CUDA argmax supports BF16 logits, got {}",
            logits.dtype()
        )));
    }
    let n = logits.numel();
    if n == 0 {
        return Err(Error::Other("CUDA argmax needs at least one logit".into()));
    }
    if n > u32::MAX as usize {
        return Err(Error::Other(format!(
            "CUDA argmax supports at most {} logits, got {n}",
            u32::MAX
        )));
    }
    require_address(ctx, "argmax", "out", out.address(), 4)?;
    unsafe {
        check_cuda(ffi::apxinf_argmax_bf16(
            gpu_ptr(logits)?,
            n as u32,
            out.address().ptr(),
            ctx.stream().handle(),
        ))
    }
}
