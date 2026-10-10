//! Legacy `kernels::sampling` names over cuda-new device selection kernels.

use apxinf_core::{DType, Error, Result, Tensor};

use crate::ffi::abi::{elementwise as abi, status};
use crate::{CudaBuffer, CudaContext};

/// `argmax_bf16_remapped_into`: device argmax over BF16 logits, mapping the
/// winning index through a u32 remap table into `out`. Ported bit-identically
/// from the legacy selection kernel, including its tie-break.
pub fn argmax_bf16_remapped_into(
    ctx: &CudaContext,
    logits: &Tensor,
    remap: &CudaBuffer,
    out: &CudaBuffer,
) -> Result<()> {
    if logits.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "argmax expects BF16 logits, got {}",
            logits.dtype()
        )));
    }
    let n = logits.shape().numel();
    let n = u32::try_from(n).map_err(|_| Error::Other("argmax logits exceed u32".into()))?;
    if n == 0 {
        return Err(Error::Other("argmax needs at least one logit".into()));
    }
    let logits_buffer = CudaBuffer::from_tensor(logits).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_elementwise_argmax_remap_bf16(
            logits_buffer.ptr(),
            n,
            remap.ptr(),
            out.ptr(),
            ctx.stream().handle(),
        ))
    }
}
