use apxinf_core::{Error, Result, Tensor};

use crate::context::CudaContext;
use crate::ffi::abi::rope as rope_abi;
use crate::ops::validation::{check_cuda, expect_bf16, gpu_ptr, matrix_shape};

/// Packed projection split, Q/K normalization, rotary embedding, and
/// token-major cache write for a full-attention layer.
pub struct QkvCacheRopeArgs<'a> {
    pub projection: &'a Tensor,
    pub query_norm: &'a Tensor,
    pub key_norm: &'a Tensor,
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
    pub query: &'a Tensor,
    pub key_cache: &'a Tensor,
    pub value_cache: &'a Tensor,
    pub cache_offset: usize,
    pub eps: f32,
}

pub fn split_qkv_cache(ctx: &CudaContext, args: QkvCacheRopeArgs<'_>) -> Result<()> {
    let (seq, width) = matrix_shape(args.projection, "QKV cache RoPE")?;
    let q_dims = args.query.shape().dims();
    let cache_dims = args.key_cache.shape().dims();
    let (q_heads, head_dim) = match q_dims {
        [query_seq, heads, dim] if *query_seq == seq => (*heads, *dim),
        _ => return Err(Error::Other("QKV cache RoPE query shape mismatch".into())),
    };
    let kv_heads = match cache_dims {
        [_, heads, dim] if *dim == head_dim => *heads,
        _ => return Err(Error::Other("QKV cache RoPE cache shape mismatch".into())),
    };
    let rotary_dim = args.cos.shape().dims().get(1).copied().unwrap_or(0);
    let expected = q_heads * 2 * head_dim + 2 * kv_heads * head_dim;
    if seq == 0
        || width < expected
        || args.value_cache.shape().dims() != cache_dims
        || args.cache_offset + seq > cache_dims[0]
        || args.query_norm.shape().dims() != [head_dim]
        || args.key_norm.shape().dims() != [head_dim]
        || args.cos.shape().dims() != [seq, rotary_dim]
        || args.sin.shape().dims() != [seq, rotary_dim]
        || rotary_dim == 0
        || rotary_dim % 2 != 0
        || rotary_dim > head_dim
    {
        return Err(Error::Other("QKV cache RoPE shape mismatch".into()));
    }
    for tensor in [
        args.projection,
        args.query_norm,
        args.key_norm,
        args.cos,
        args.sin,
        args.query,
        args.key_cache,
        args.value_cache,
    ] {
        expect_bf16(tensor, "QKV cache RoPE")?;
    }
    unsafe {
        check_cuda(rope_abi::apxinf_rope_split_qkv_cache_bf16(
            gpu_ptr(args.projection)?,
            gpu_ptr(args.query_norm)?,
            gpu_ptr(args.key_norm)?,
            gpu_ptr(args.cos)?,
            gpu_ptr(args.sin)?,
            gpu_ptr(args.query)?,
            gpu_ptr(args.key_cache)?,
            gpu_ptr(args.value_cache)?,
            seq as i32,
            args.cache_offset as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            width as i64,
            (kv_heads * head_dim) as i64,
            args.eps,
            ctx.stream().handle(),
        ))
    }
}

/// Group-major projection split, Q/K normalization, and rotary embedding.
pub struct GroupedQkvRopeArgs<'a> {
    pub projection: &'a Tensor,
    pub query_norm: &'a Tensor,
    pub key_norm: &'a Tensor,
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
    pub query: &'a Tensor,
    pub gate: &'a Tensor,
    pub key: &'a Tensor,
    pub value: &'a Tensor,
    pub eps: f32,
}

pub fn split_grouped_qkv(ctx: &CudaContext, args: GroupedQkvRopeArgs<'_>) -> Result<()> {
    let (seq, width) = matrix_shape(args.projection, "grouped QKV RoPE")?;
    let q_dims = args.query.shape().dims();
    let k_dims = args.key.shape().dims();
    let (q_heads, head_dim) = match q_dims {
        [query_seq, heads, dim] if *query_seq == seq => (*heads, *dim),
        _ => return Err(Error::Other("grouped QKV query shape mismatch".into())),
    };
    let kv_heads = match k_dims {
        [key_seq, heads, dim] if *key_seq == seq && *dim == head_dim => *heads,
        _ => return Err(Error::Other("grouped QKV key shape mismatch".into())),
    };
    let rotary_dim = args.cos.shape().dims().get(1).copied().unwrap_or(0);
    let heads_per_group = q_heads.checked_div(kv_heads).unwrap_or(0);
    let expected = kv_heads * (2 * heads_per_group + 2) * head_dim;
    if seq == 0
        || kv_heads == 0
        || q_heads % kv_heads != 0
        || width < expected
        || args.gate.shape().dims() != [seq, q_heads, head_dim]
        || args.value.shape().dims() != [seq, kv_heads, head_dim]
        || args.query_norm.shape().dims() != [head_dim]
        || args.key_norm.shape().dims() != [head_dim]
        || args.cos.shape().dims() != [seq, rotary_dim]
        || args.sin.shape().dims() != [seq, rotary_dim]
        || rotary_dim == 0
        || rotary_dim % 2 != 0
        || rotary_dim > head_dim
    {
        return Err(Error::Other("grouped QKV RoPE shape mismatch".into()));
    }
    for tensor in [
        args.projection,
        args.query_norm,
        args.key_norm,
        args.cos,
        args.sin,
        args.query,
        args.gate,
        args.key,
        args.value,
    ] {
        expect_bf16(tensor, "grouped QKV RoPE")?;
    }
    unsafe {
        check_cuda(rope_abi::apxinf_rope_split_grouped_qkv_bf16(
            gpu_ptr(args.projection)?,
            gpu_ptr(args.query_norm)?,
            gpu_ptr(args.key_norm)?,
            gpu_ptr(args.cos)?,
            gpu_ptr(args.sin)?,
            gpu_ptr(args.query)?,
            gpu_ptr(args.gate)?,
            gpu_ptr(args.key)?,
            gpu_ptr(args.value)?,
            seq as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            width as i64,
            args.eps,
            ctx.stream().handle(),
        ))
    }
}
