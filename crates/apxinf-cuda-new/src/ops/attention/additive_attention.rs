use apxinf_core::{DType, Result, Tensor};

use super::contracts::{
    alignment, invalid, normalize, range, required_bytes, tensor_storage, AttentionArgs,
    Normalized, Semantic,
};
use super::execution;
use crate::CudaContext;

/// Arithmetic at the score/probability boundaries, independent of the provider.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttentionPrecision {
    /// Accumulate QK, add the bias, softmax, and accumulate PV in F32.
    #[default]
    F32 = 0,
    /// Round scaled QK to the input dtype; add bias and round again; compute
    /// softmax in F32 and round its probabilities to the input dtype before PV.
    /// This is the eager low-precision matmul/add/softmax/matmul contract.
    Input = 1,
}

/// Dense attention with an additive F32 bias in `[B,Hq,Tq,Tk]` order.
/// A leading dimension of one broadcasts the bias across the batch. Negative
/// infinity masks a key; an entirely masked row produces zero output.
/// Q/K/V and output have the same F16, BF16, or F32 dtype.
pub struct AttentionBiasArgs<'a> {
    pub attention: AttentionArgs<'a>,
    pub bias: &'a Tensor,
    pub precision: AttentionPrecision,
}

impl<'a> AttentionBiasArgs<'a> {
    pub fn new(attention: AttentionArgs<'a>, bias: &'a Tensor) -> Self {
        Self {
            attention,
            bias,
            precision: AttentionPrecision::F32,
        }
    }
}

pub(crate) fn normalize_additive(
    ctx: &CudaContext,
    args: AttentionBiasArgs<'_>,
) -> Result<Normalized> {
    if args.attention.out.dtype() != args.attention.query.dtype() {
        return Err(invalid(
            "additive Attention requires input/output dtype equality",
        ));
    }
    let out_bytes = required_bytes(
        args.attention.out.dtype(),
        args.attention.out.shape().dims(),
    )?;
    let mut normalized = normalize(ctx, args.attention)?;
    let shape = args.bias.shape().dims();
    let spec = &mut normalized.spec;
    if shape.len() != 4
        || (shape[0] != 1 && shape[0] != spec.batch as usize)
        || shape[1..]
            != [
                spec.query_heads as usize,
                spec.query_tokens as usize,
                spec.key_tokens as usize,
            ]
    {
        return Err(invalid(
            "Attention bias must be [1 or batch, query_heads, query_tokens, key_tokens]",
        ));
    }
    let bias = tensor_storage(ctx, args.bias, DType::F32, shape)?;
    let bias_range = range(&bias, required_bytes(DType::F32, shape)?)?;
    let out = normalized
        .storage
        .last()
        .expect("dense normalization retains output");
    let out_range = range(out, out_bytes)?;
    if out_range.start < bias_range.end && bias_range.start < out_range.end {
        return Err(invalid("Attention output overlaps read-only bias storage"));
    }
    spec.semantic = Semantic::Additive.abi_value();
    spec.bias_alignment = alignment(bias.ptr());
    spec.bias_broadcast_batch = u32::from(shape[0] == 1);
    spec.input_precision = args.precision as u32;
    normalized.bindings.bias = bias.ptr();
    normalized.storage.push(bias);
    Ok(normalized)
}

pub fn additive_attention(ctx: &CudaContext, args: AttentionBiasArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize_additive(ctx, args)?)
}
