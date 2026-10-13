//! Greedy token selection.
//!
//! Moved here from `ops/model.rs`, which was named after a call site rather
//! than a mathematical family (`doc/cuda-new-design.md` section 1.3).
//!
//! Distinct from [`crate::sampling`], which is one level up: that module owns
//! the *stateful* samplers — a sampler object with a workspace and device RNG,
//! built through `create_token_sampler`. Argmax is the degenerate case with no
//! state, no workspace and no RNG, so it stays a single-shot operator here
//! (section 5).

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{sampling as abi, status};
use crate::ops::gemm::contracts::tensor_storage;
use crate::CudaContext;

/// Index of the largest logit, reduced on device.
///
/// Reducing on device avoids copying 248320 BF16 logits (~486 KiB) to the host
/// per token just to produce one integer.
pub fn argmax(ctx: &CudaContext, logits: &Tensor, index: &Tensor) -> Result<()> {
    let dims = logits.shape().dims().to_vec();
    let count = dims.iter().product::<usize>();
    let logits_buffer = tensor_storage(ctx, logits, DType::BF16, &dims)?;
    let index_buffer = tensor_storage(ctx, index, DType::I32, &[1])?;
    unsafe {
        status::check(abi::apxinf_sampling_argmax(
            logits_buffer.ptr(),
            index_buffer.ptr(),
            count as i64,
            ctx.stream().handle(),
        ))
    }
}
