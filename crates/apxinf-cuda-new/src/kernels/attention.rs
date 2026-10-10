//! Legacy `kernels::attention` names over cuda-new attention operators.
//!
//! The legacy helpers take `[tokens, heads, head_dim]` tensors; the cuda-new
//! Attention contract is rank-4 `[batch, tokens, heads, head_dim]`. These
//! adapters reshape at the boundary — reshape on a contiguous tensor is a
//! metadata change, not a copy.
//!
//! Causality follows the legacy helper it replaces: `mqa_bf16` and `mha_bf16`
//! are non-causal over the valid keys; `causal_gqa_bf16` is causal with the
//! query block at the end of the key range.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::{ops, CudaContext};

/// Q/K/V triple produced by a packed-QKV split. Mirrors the legacy struct.
pub struct QkvTensors {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

fn rank3(tensor: &Tensor, what: &str) -> Result<[usize; 3]> {
    let dims = tensor.shape().dims();
    if dims.len() != 3 {
        return Err(Error::Other(format!("{what} requires rank-3 [tokens, heads, dim]")));
    }
    Ok([dims[0], dims[1], dims[2]])
}

fn dense_attention(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
    kv_heads: usize,
    causal: bool,
) -> Result<Tensor> {
    let [query_tokens, query_heads, head_dim] = rank3(q, "attention query")?;
    let q4 = q.reshape(vec![1, query_tokens, query_heads, head_dim])?;
    // K/V may be larger caches; present exactly the valid prefix.
    let k4 = k.reshape(vec![1, k.shape().numel() / (kv_heads * head_dim), kv_heads, head_dim])?;
    let v4 = v.reshape(vec![1, v.shape().numel() / (kv_heads * head_dim), kv_heads, head_dim])?;
    let mut out = ctx.allocate_output(Shape::new(vec![1, query_tokens, query_heads, head_dim]), DType::BF16)?;
    let mut args = ops::KvCacheAttentionArgs::new(&q4, &k4, &v4, &mut out);
    args.valid_key_tokens = key_tokens;
    if causal {
        args.mask = ops::AttentionMask::Causal;
        args.query_start = key_tokens - query_tokens;
    } else {
        args.mask = ops::AttentionMask::None;
        args.query_start = 0;
    }
    args.scale = 1.0 / (head_dim as f32).sqrt();
    ops::kv_cache_attention(ctx, args)?;
    out.reshape(vec![query_tokens, query_heads, head_dim])
}

/// `mqa_bf16`: multi-query attention — one shared K/V head, non-causal over
/// the leading `key_tokens` rows of flat `[*, head_dim]` K/V storage.
pub fn mqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    dense_attention(ctx, q, k, v, key_tokens, 1, false)
}

/// `mha_bf16`: dense multi-head attention over equal-shaped Q/K/V, non-causal,
/// batched by `tokens_per_batch`.
pub fn mha_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    tokens_per_batch: usize,
) -> Result<Tensor> {
    let [tokens, heads, head_dim] = rank3(q, "MHA query")?;
    if tokens_per_batch == 0 || tokens % tokens_per_batch != 0 {
        return Err(Error::Other("MHA tokens_per_batch mismatch".into()));
    }
    let batches = tokens / tokens_per_batch;
    let q4 = q.reshape(vec![batches, tokens_per_batch, heads, head_dim])?;
    let k4 = k.reshape(vec![batches, tokens_per_batch, heads, head_dim])?;
    let v4 = v.reshape(vec![batches, tokens_per_batch, heads, head_dim])?;
    let mut out =
        ctx.allocate_output(Shape::new(vec![batches, tokens_per_batch, heads, head_dim]), DType::BF16)?;
    let args = ops::AttentionArgs::new(&q4, &k4, &v4, &mut out);
    ops::attention(ctx, args)?;
    out.reshape(vec![tokens, heads, head_dim])
}

/// `causal_gqa_bf16`: grouped-query causal attention with the query block at
/// the end of `key_tokens` keys.
pub fn causal_gqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let [_, kv_heads, _] = rank3(k, "GQA keys")?;
    dense_attention(ctx, q, k, v, key_tokens, kv_heads, true)
}

/// `split_qkv_bias_bf16`: split a `[tokens, 3*heads*dim]` packed projection
/// into Q/K/V with an optional packed bias, no rotation (the vision layout).
pub fn split_qkv_bias_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    heads: usize,
    head_dim: usize,
) -> Result<QkvTensors> {
    let dims = qkv.shape().dims();
    if dims.len() != 2 || dims[1] != 3 * heads * head_dim {
        return Err(Error::Other(
            "packed QKV split expects [tokens, 3*heads*dim]".into(),
        ));
    }
    let tokens = dims[0];
    let shape = Shape::new(vec![tokens, heads, head_dim]);
    let mut q = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let mut k = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let mut v = ctx.allocate_output(shape, DType::BF16)?;
    ops::rope(
        ctx,
        ops::RopeArgs {
            semantic: ops::RopeSemantic::SplitQkvBias,
            qkv,
            bias,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads: heads,
            kv_heads: heads,
            head_dim,
            theta: 0.0,
            position_offset: 0,
            kv_output_offset: 0,
        },
    )?;
    Ok(QkvTensors { q, k, v })
}

/// `split_gqa_qkv_mrope_cache_bf16`: split a packed GQA projection, apply
/// mRoPE to Q/K (three position ids per token), and write K/V at
/// `cache_tokens`-long caches. With `caches` the write lands at the given
/// offset in the supplied tensors; without, fresh caches are allocated and
/// the tokens land at offset zero.
#[allow(clippy::too_many_arguments)]
pub fn split_gqa_qkv_mrope_cache_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    position_ids: &crate::CudaBuffer,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    sections: [usize; 3],
    cache_tokens: usize,
    caches: Option<(&Tensor, &Tensor, usize)>,
) -> Result<QkvTensors> {
    use crate::ffi::abi::{status, vla_attn as abi};
    use crate::CudaBuffer;
    let dims = qkv.shape().dims();
    let q_width = q_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    if dims.len() != 2
        || qkv.dtype() != DType::BF16
        || dims[1] != q_width + 2 * kv_width
        || q_heads == 0
        || kv_heads == 0
        || q_heads % kv_heads != 0
        || head_dim == 0
        || head_dim > 256
        || head_dim % 2 != 0
        || !theta.is_finite()
        || theta <= 0.0
        || sections[1] + sections[2] > head_dim / 2
    {
        return Err(Error::Other("GQA QKV mRoPE shape mismatch".into()));
    }
    let tokens = dims[0];
    if position_ids.len() < tokens * 3 * std::mem::size_of::<u32>() {
        return Err(Error::Other("GQA QKV mRoPE position ids too short".into()));
    }
    let cache_shape = [cache_tokens, kv_heads, head_dim];
    let (k, v, cache_offset) = match caches {
        Some((k, v, offset)) => {
            if k.dtype() != DType::BF16
                || v.dtype() != DType::BF16
                || k.shape().dims() != cache_shape
                || v.shape().dims() != cache_shape
                || offset + tokens > cache_tokens
            {
                return Err(Error::Other("GQA QKV mRoPE cache shape mismatch".into()));
            }
            (k.clone(), v.clone(), offset)
        }
        None => {
            if tokens > cache_tokens {
                return Err(Error::Other("GQA QKV mRoPE cache is too short".into()));
            }
            let k = ctx.allocate_output(Shape::new(cache_shape.to_vec()), DType::BF16)?;
            let v = ctx.allocate_output(Shape::new(cache_shape.to_vec()), DType::BF16)?;
            (k, v, 0)
        }
    };
    let q = ctx.allocate_output(Shape::new(vec![tokens, q_heads, head_dim]), DType::BF16)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let qkv_buffer = CudaBuffer::from_tensor(qkv).map_err(Error::Cuda)?;
    let bias_buffer = bias.map(CudaBuffer::from_tensor).transpose().map_err(Error::Cuda)?;
    let q_buffer = CudaBuffer::from_tensor(&q).map_err(Error::Cuda)?;
    let k_buffer = CudaBuffer::from_tensor(&k).map_err(Error::Cuda)?;
    let v_buffer = CudaBuffer::from_tensor(&v).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_vla_gqa_qkv_mrope_cache_bf16(
            qkv_buffer.ptr(),
            bias_buffer
                .as_ref()
                .map_or(std::ptr::null(), |buffer| buffer.ptr() as *const _),
            position_ids.ptr(),
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            to_i32(tokens, "tokens")?,
            to_i32(q_heads, "query heads")?,
            to_i32(kv_heads, "KV heads")?,
            to_i32(head_dim, "head dim")?,
            theta,
            to_i32(sections[1], "mRoPE height section")?,
            to_i32(sections[2], "mRoPE width section")?,
            to_i32(cache_offset, "cache offset")?,
            ctx.stream().handle(),
        ))?;
    }
    Ok(QkvTensors { q, k, v })
}

/// `split_vision_qkv_rope_bf16`: split a packed vision projection and apply
/// 2D RoPE to Q/K (two position ids per token). The biased value rounds to
/// BF16 before rotation, matching the legacy kernel.
pub fn split_vision_qkv_rope_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    position_ids: &crate::CudaBuffer,
    heads: usize,
    head_dim: usize,
    theta: f32,
) -> Result<QkvTensors> {
    use crate::ffi::abi::{status, vla_attn as abi};
    use crate::CudaBuffer;
    let dims = qkv.shape().dims();
    let projection_width = heads * head_dim;
    if dims.len() != 2
        || qkv.dtype() != DType::BF16
        || dims[1] != 3 * projection_width
        || heads == 0
        || head_dim == 0
        || head_dim > 256
        || head_dim % 4 != 0
        || !theta.is_finite()
        || theta <= 0.0
    {
        return Err(Error::Other("vision QKV RoPE shape mismatch".into()));
    }
    let tokens = dims[0];
    if position_ids.len() < tokens * 2 * std::mem::size_of::<u32>() {
        return Err(Error::Other("vision QKV RoPE position ids too short".into()));
    }
    let shape = Shape::new(vec![tokens, heads, head_dim]);
    let q = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let k = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let v = ctx.allocate_output(shape, DType::BF16)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let qkv_buffer = CudaBuffer::from_tensor(qkv).map_err(Error::Cuda)?;
    let bias_buffer = bias.map(CudaBuffer::from_tensor).transpose().map_err(Error::Cuda)?;
    let q_buffer = CudaBuffer::from_tensor(&q).map_err(Error::Cuda)?;
    let k_buffer = CudaBuffer::from_tensor(&k).map_err(Error::Cuda)?;
    let v_buffer = CudaBuffer::from_tensor(&v).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_vla_vision_qkv_rope_bf16(
            qkv_buffer.ptr(),
            bias_buffer
                .as_ref()
                .map_or(std::ptr::null(), |buffer| buffer.ptr() as *const _),
            position_ids.ptr(),
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            to_i32(tokens, "tokens")?,
            to_i32(heads, "heads")?,
            to_i32(head_dim, "head dim")?,
            theta,
            ctx.stream().handle(),
        ))?;
    }
    Ok(QkvTensors { q, k, v })
}

/// `segmented_mha_bf16`: dense non-causal MHA confined to segments delimited
/// by `offsets` (`segments + 1` u32 entries, mirrored on the host for
/// validation).
#[allow(clippy::too_many_arguments)]
pub fn segmented_mha_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    offsets: &crate::CudaBuffer,
    host_offsets: &[u32],
    segments: usize,
    max_tokens: usize,
) -> Result<Tensor> {
    use crate::ffi::abi::{status, vla_attn as abi};
    use crate::CudaBuffer;
    let shape = rank3(q, "segmented MHA query")?;
    if [q, k, v].into_iter().any(|t| t.dtype() != DType::BF16)
        || k.shape() != q.shape()
        || v.shape() != q.shape()
        || segments == 0
        || host_offsets.len() != segments + 1
        || max_tokens == 0
        || shape[2] > 256
    {
        return Err(Error::Other(
            "segmented BF16 MHA requires matching [tokens,heads,head_dim] tensors".into(),
        ));
    }
    if offsets.len() < (segments + 1) * std::mem::size_of::<u32>() {
        return Err(Error::Other("segmented MHA offsets buffer too short".into()));
    }
    if host_offsets.first() != Some(&0)
        || host_offsets.last().map(|&value| value as usize) != Some(shape[0])
        || host_offsets
            .windows(2)
            .any(|pair| pair[1] < pair[0] || (pair[1] - pair[0]) as usize > max_tokens)
    {
        return Err(Error::Other("segmented MHA offsets are inconsistent".into()));
    }
    let output = ctx.allocate_output(Shape::new(shape.to_vec()), DType::BF16)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let q_buffer = CudaBuffer::from_tensor(q).map_err(Error::Cuda)?;
    let k_buffer = CudaBuffer::from_tensor(k).map_err(Error::Cuda)?;
    let v_buffer = CudaBuffer::from_tensor(v).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_vla_segmented_mha_bf16(
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            offsets.ptr(),
            output_buffer.ptr(),
            to_i32(segments, "segments")?,
            to_i32(max_tokens, "max tokens")?,
            to_i32(shape[1], "heads")?,
            to_i32(shape[2], "head dim")?,
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}

/// `noncausal_gqa_bf16`: grouped-query attention over the leading
/// `key_tokens` keys with no causal mask (the joint vision/text form used by
/// qwen_drive's full-attention layers).
pub fn noncausal_gqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let [_, kv_heads, _] = rank3(k, "GQA keys")?;
    dense_attention(ctx, q, k, v, key_tokens, kv_heads, false)
}

// ── sm110 AOT fast-path stubs ───────────────────────────────────────────────
//
// The legacy crate carries hand-built FA4/AOT adapters for one fixed sm_110
// scene shape. Those objects are not vendored into cuda-new; every call site
// treats `Ok(None)` as "take the generic route", which is the tuned cuda-new
// operator.

/// FA4 D256 causal GQA, sm_110 fixed-shape AOT route. Not vendored: always
/// defers to the generic attention operator.
pub fn try_gqa_bf16_fa4_d256_sm110(
    _ctx: &CudaContext,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _key_tokens: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}

/// FA4 D256 split-batch variant of the above. Not vendored.
pub fn try_gqa_bf16_fa4_d256_splitbatch_sm110(
    _ctx: &CudaContext,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _key_tokens: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}

/// Fused vision QKV RoPE + segmented FA4 with V passthrough, sm_110
/// fixed-shape AOT route. Not vendored.
#[allow(clippy::too_many_arguments)]
pub fn try_vision_qkv_rope_segmented_fa4_skip_v(
    _ctx: &CudaContext,
    _qkv: &Tensor,
    _position_ids: &crate::CudaBuffer,
    _heads: usize,
    _head_dim: usize,
    _theta: f32,
    _offsets: &crate::CudaBuffer,
    _host_offsets: &[u32],
    _segments: usize,
    _fixed_groups: bool,
) -> Result<Option<Tensor>> {
    Ok(None)
}
