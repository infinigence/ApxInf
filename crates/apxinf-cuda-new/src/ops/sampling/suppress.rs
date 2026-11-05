use apxinf_core::{DType, Error, Result, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi::abi::sampling as sampling_abi;
use crate::ops::validation::{check_cuda, expect_bf16, gpu_ptr, matrix_shape};

/// Set selected logits to negative infinity.
pub fn suppress_logits(
    ctx: &CudaContext,
    logits: &Tensor,
    row: usize,
    ids: &CudaBuffer,
) -> Result<()> {
    let (rows, cols) = matrix_shape(logits, "logits suppression")?;
    if row >= rows {
        return Err(Error::Other("logits suppression row out of range".into()));
    }
    expect_bf16(logits, "logits suppression")?;
    let count = ids.len() / std::mem::size_of::<u32>();
    if count == 0 {
        return Ok(());
    }
    let row_ptr = gpu_ptr(logits)?
        .cast::<u8>()
        .wrapping_add(row * cols * DType::BF16.size_in_bytes())
        .cast();
    unsafe {
        check_cuda(sampling_abi::apxinf_sampling_suppress_logits_bf16(
            row_ptr,
            ids.ptr().cast(),
            count as i32,
            ctx.stream().handle(),
        ))
    }
}
