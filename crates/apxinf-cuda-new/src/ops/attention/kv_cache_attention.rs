use apxinf_core::{DType, Result, Tensor};

use super::contracts::{
    alignment, dtype_code, invalid, range, required_bytes, tensor_storage, Normalized, Semantic,
};
use super::{execution, AttentionMask, AttentionPolicy};
use crate::ffi::abi::attention as abi;
use crate::{CudaBuffer, CudaContext};

/// Contiguous KV-cache attention.
///
/// Q is `[batch, query_tokens, query_heads, head_dim]`; the cache tensors are
/// `[batch, key_capacity, kv_heads, head_dim]`. Only their leading
/// `valid_key_tokens` rows participate. For a causal mask, query token `i`
/// occupies cache position `query_start + i`.
pub struct KvCacheAttentionArgs<'a> {
    pub query: &'a Tensor,
    pub key_cache: &'a Tensor,
    pub value_cache: &'a Tensor,
    pub out: &'a mut Tensor,
    pub valid_key_tokens: usize,
    pub query_start: usize,
    pub mask: AttentionMask,
    pub scale: f32,
    pub policy: AttentionPolicy,
}

impl<'a> KvCacheAttentionArgs<'a> {
    pub fn new(
        query: &'a Tensor,
        key_cache: &'a Tensor,
        value_cache: &'a Tensor,
        out: &'a mut Tensor,
    ) -> Self {
        let query_shape = query.shape().dims();
        let head_dim = query_shape.get(3).copied().unwrap_or(1);
        let query_tokens = query_shape.get(1).copied().unwrap_or(0);
        let valid_key_tokens = key_cache.shape().dims().get(1).copied().unwrap_or(0);
        Self {
            query,
            key_cache,
            value_cache,
            out,
            valid_key_tokens,
            query_start: valid_key_tokens.saturating_sub(query_tokens),
            mask: AttentionMask::Causal,
            scale: 1.0 / (head_dim as f32).sqrt(),
            policy: AttentionPolicy::default(),
        }
    }

    pub fn non_causal(mut self) -> Self {
        self.mask = AttentionMask::None;
        self.query_start = 0;
        self
    }
}

fn reject_overlap(
    output: &CudaBuffer,
    output_bytes: usize,
    input: &CudaBuffer,
    input_bytes: usize,
    name: &str,
) -> Result<()> {
    let output = range(output, output_bytes)?;
    let input = range(input, input_bytes)?;
    if output.start < input.end && input.start < output.end {
        return Err(invalid(format!(
            "KV-cache Attention output overlaps read-only {name} storage"
        )));
    }
    Ok(())
}

pub(crate) fn normalize(ctx: &CudaContext, args: KvCacheAttentionArgs<'_>) -> Result<Normalized> {
    let q_shape = args.query.shape().dims();
    let k_shape = args.key_cache.shape().dims();
    let v_shape = args.value_cache.shape().dims();
    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
        return Err(invalid("KV-cache Attention requires rank-4 tensors"));
    }
    let (batch, query_tokens, query_heads, head_dim) =
        (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let (key_batch, key_capacity, kv_heads, key_head_dim) =
        (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
    if [
        batch,
        query_tokens,
        query_heads,
        head_dim,
        key_capacity,
        kv_heads,
    ]
    .contains(&0)
        || batch != key_batch
        || v_shape != k_shape
        || head_dim != key_head_dim
        || query_heads % kv_heads != 0
        || args.valid_key_tokens == 0
        || args.valid_key_tokens > key_capacity
        || args.query_start > args.valid_key_tokens
        || (args.mask == AttentionMask::Causal
            && match args.query_start.checked_add(query_tokens) {
                Some(end) => end > args.valid_key_tokens,
                None => true,
            })
        || args.out.shape().dims() != q_shape
        || args.query.dtype() != args.key_cache.dtype()
        || args.query.dtype() != args.value_cache.dtype()
        || args.query.dtype() != args.out.dtype()
        || !matches!(args.query.dtype(), DType::F16 | DType::BF16)
    {
        return Err(invalid("invalid KV-cache Attention contract"));
    }
    if !args.scale.is_finite() || args.scale <= 0.0 {
        return Err(invalid(
            "KV-cache Attention scale must be finite and positive",
        ));
    }
    if [
        batch,
        query_tokens,
        args.valid_key_tokens,
        key_capacity,
        query_heads,
        kv_heads,
        head_dim,
        args.query_start,
    ]
    .iter()
    .any(|value| *value > i32::MAX as usize)
    {
        return Err(invalid(
            "KV-cache Attention dimension exceeds native limits",
        ));
    }

    let dtype = args.query.dtype();
    let q = tensor_storage(ctx, args.query, dtype, q_shape)?;
    let k = tensor_storage(ctx, args.key_cache, dtype, k_shape)?;
    let v = tensor_storage(ctx, args.value_cache, dtype, v_shape)?;
    let out = tensor_storage(ctx, args.out, dtype, q_shape)?;
    let output_bytes = required_bytes(dtype, q_shape)?;
    for (name, input, shape) in [
        ("query", &q, q_shape),
        ("key cache", &k, k_shape),
        ("value cache", &v, v_shape),
    ] {
        reject_overlap(
            &out,
            output_bytes,
            input,
            required_bytes(dtype, shape)?,
            name,
        )?;
    }

    let default_scale = 1.0 / (head_dim as f32).sqrt();
    let bindings = abi::Bindings {
        query: q.ptr(),
        key: k.ptr(),
        value: v.ptr(),
        offsets: std::ptr::null(),
        output: out.ptr(),
        stream: ctx.stream().handle(),
        scale: args.scale,
        output_scale: 1.0,
    };
    Ok(Normalized {
        spec: abi::Spec {
            version: abi::SPEC_VERSION,
            semantic: Semantic::KvCache.abi_value(),
            dtype: dtype_code(dtype)?,
            output_dtype: dtype_code(dtype)?,
            mask: args.mask as u32,
            q_alignment: alignment(bindings.query),
            k_alignment: alignment(bindings.key),
            v_alignment: alignment(bindings.value),
            output_alignment: alignment(bindings.output.cast_const()),
            offsets_alignment: 0,
            batch: batch as i64,
            query_tokens: query_tokens as i64,
            key_tokens: args.valid_key_tokens as i64,
            key_capacity: key_capacity as i64,
            query_heads: query_heads as i64,
            kv_heads: kv_heads as i64,
            head_dim: head_dim as i64,
            query_start: args.query_start as i64,
            segments: 0,
            max_segment_tokens: 0,
            offsets_hash: 0,
            scale_is_default: u32::from(args.scale == default_scale),
        },
        policy: args.policy,
        bindings,
        storage: vec![q, k, v, out],
    })
}

pub fn kv_cache_attention(ctx: &CudaContext, args: KvCacheAttentionArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize(ctx, args)?)
}
