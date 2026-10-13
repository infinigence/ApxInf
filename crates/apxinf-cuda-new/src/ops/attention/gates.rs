//! Qwen3.5's `attn_output_gate: true` pair: the query/gate split out of
//! `q_proj`, and the sigmoid the attention output passes through before
//! `o_proj`.
//!
//! Both belong to the attention block — the split is a strided copy out of
//! `q_proj`'s doubled head slots, and the gate acts on the attention output —
//! but neither is a member of the *tunable* attention shape: one
//! implementation each and nothing whose relative speed needs measuring, so
//! they skip `prepare`/`enqueue` and keep flat parameters
//! (`doc/cuda-new-design.md` section 5).
//!
//! Moved here from `ops/attn_ops.rs`, which was named after a call site rather
//! than a mathematical family (section 1.3).

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{attention as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// Split q_proj's `[tokens, heads, 2*head_dim]` output into query and gate.
pub fn split_query_and_gate(
    ctx: &CudaContext,
    fused: &Tensor,
    query: &Tensor,
    gate: &Tensor,
) -> Result<()> {
    let dims = query.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(invalid("query must be [tokens, heads, head_dim]"));
    }
    let (tokens, heads, head_dim) = (dims[0], dims[1], dims[2]);
    let fused_buffer =
        tensor_storage(ctx, fused, DType::BF16, &[tokens, heads, 2 * head_dim])?;
    let query_buffer = tensor_storage(ctx, query, DType::BF16, &dims)?;
    let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_attention_split_query_and_gate(
            fused_buffer.ptr(),
            query_buffer.ptr(),
            gate_buffer.ptr(),
            tokens as i64,
            heads as i64,
            head_dim as i64,
            ctx.stream().handle(),
        ))
    }
}

/// `data *= sigmoid(gate)`, elementwise and in place.
///
/// Sigmoid, not silu, even though `config.json` carries
/// `output_gate_type: "swish"`. That key is never read by the reference
/// implementation; `Qwen3_5Attention.forward` applies a bare
/// `torch.sigmoid(gate)`. The silu-shaped gate in this model is the GDN one
/// (`gdn_gated_norm`), which is separate and unaffected.
///
/// This still has no quantization epilogue, so it costs a full extra pass over
/// the tensor before `o_proj` — recorded as debt in section 4.3.
pub fn apply_output_gate(ctx: &CudaContext, data: &Tensor, gate: &Tensor) -> Result<()> {
    let dims = data.shape().dims().to_vec();
    if dims != gate.shape().dims() {
        return Err(invalid("output gate requires matching shapes"));
    }
    let data_buffer = tensor_storage(ctx, data, DType::BF16, &dims)?;
    let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_attention_apply_output_gate(
            data_buffer.ptr(),
            gate_buffer.ptr(),
            dims.iter().product::<usize>() as i64,
            ctx.stream().handle(),
        ))
    }
}
