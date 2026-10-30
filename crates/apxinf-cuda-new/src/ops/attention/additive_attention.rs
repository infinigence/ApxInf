use apxinf_core::{DType, Result, Tensor};

use super::contracts::{
    alignment, invalid, normalize, range, required_bytes, tensor_storage, AttentionArgs,
    Normalized, Semantic,
};
use super::execution;
use crate::CudaContext;

/// Dense attention with an additive F32 bias in `[B,Hq,Tq,Tk]` order.
/// A leading dimension of one broadcasts the bias across the batch. Negative
/// infinity masks a key; an entirely masked row produces zero output.
/// Q/K/V and output have the same F16, BF16, or F32 dtype.
pub struct AttentionBiasArgs<'a> {
    /// F32 baseline, or BF16 materialization of QK, biased scores and probabilities. Accumulation remains F32.
    pub intermediate_dtype: DType,
    pub attention: AttentionArgs<'a>,
    pub bias: &'a Tensor,
}

impl<'a> AttentionBiasArgs<'a> {
    pub fn new(attention: AttentionArgs<'a>, bias: &'a Tensor) -> Self {
        Self {
            attention,
            bias,
            intermediate_dtype: DType::F32,
        }
    }
}

pub(crate) fn normalize_additive(
    ctx: &CudaContext,
    args: AttentionBiasArgs<'_>,
) -> Result<Normalized> {
    let query = match args.attention.input {
        super::contracts::AttentionInput::Separate { query, .. } => query,
        _ => return Err(invalid("additive Attention requires separate Q/K/V")),
    };
    if args.attention.numerics != super::contracts::AttentionNumerics::Flexible {
        return Err(invalid("additive Attention requires flexible numerics"));
    }
    if args.attention.out.dtype() != query.dtype() {
        return Err(invalid(
            "additive Attention requires input/output dtype equality",
        ));
    }
    if args.intermediate_dtype != DType::F32
        && (args.intermediate_dtype != DType::BF16 || query.dtype() != DType::BF16)
    {
        return Err(invalid(
            "additive Attention intermediate dtype must be F32 or BF16 with BF16 operands",
        ));
    }
    let input_precision = u32::from(args.intermediate_dtype == DType::BF16);
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
    spec.input_precision = input_precision;
    normalized.bindings.bias = bias.ptr();
    normalized.storage.push(bias);
    Ok(normalized)
}

pub fn additive_attention(ctx: &CudaContext, args: AttentionBiasArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize_additive(ctx, args)?)
}
