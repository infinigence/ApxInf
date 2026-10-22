use super::{contracts, execution};
use crate::CudaContext;
use apxinf_core::{Result, Tensor};
use contracts::{invalid, AttentionArgs, AttentionMask, Normalized, Semantic};

/// Dense MHA with materialized input-dtype logits and probabilities:
/// `L = round(round(scale * QKᵀ) + bias)`;
/// `P = round(softmax(float(L)))`; `Y = round(PV)`.
/// Bias has shape `[batch, heads, query_tokens, key_tokens]` and the Q dtype.
/// All output rounding uses that same F16/BF16 dtype. Bias may encode masking.
pub struct RoundedAttentionBiasArgs<'a> {
    pub attention: AttentionArgs<'a>,
    pub bias: &'a Tensor,
}

pub(crate) fn normalize(
    ctx: &CudaContext,
    args: RoundedAttentionBiasArgs<'_>,
) -> Result<Normalized> {
    if args.attention.mask != AttentionMask::None
        || args.attention.out.dtype() != args.attention.query.dtype()
    {
        return Err(invalid(
            "Rounded additive attention requires no implicit mask and matching output dtype",
        ));
    }
    let mut normalized = contracts::normalize(ctx, args.attention)?;
    let spec = &mut normalized.spec;
    if spec.query_heads != spec.kv_heads {
        return Err(invalid(
            "Rounded additive attention requires equal query and KV head counts",
        ));
    }
    let shape = [
        spec.batch as usize,
        spec.query_heads as usize,
        spec.query_tokens as usize,
        spec.key_tokens as usize,
    ];
    let dtype = args.bias.dtype();
    if contracts::dtype_code(dtype)? != spec.dtype {
        return Err(invalid(
            "Rounded additive attention bias dtype must match Q/K/V",
        ));
    }
    let bias = contracts::tensor_storage(ctx, args.bias, dtype, &shape)?;
    let bias_range = contracts::range(&bias, contracts::required_bytes(dtype, &shape)?)?;
    let output = normalized.storage.last().expect("dense output binding");
    let output_range = contracts::range(output, output.len())?;
    if bias_range.start < output_range.end && output_range.start < bias_range.end {
        return Err(invalid("Rounded additive attention output overlaps bias"));
    }
    spec.semantic = Semantic::RoundedAdditive.abi_value();
    spec.bias_alignment = contracts::alignment(bias.ptr());
    normalized.bindings.bias = bias.ptr();
    normalized.storage.push(bias);
    Ok(normalized)
}

pub fn rounded_additive_attention(
    ctx: &CudaContext,
    args: RoundedAttentionBiasArgs<'_>,
) -> Result<()> {
    execution::execute(ctx, normalize(ctx, args)?)
}
