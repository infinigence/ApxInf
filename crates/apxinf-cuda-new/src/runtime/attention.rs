//! Runtime-only head-major KV-cache and decode launchers.
//!
//! Formal model operators use `attention`, `kv_cache_attention`, or
//! `segmented_attention`; this module only preserves backend/decode storage
//! contracts that are not L3 model semantics.

use apxinf_core::{DType, Device, Error, KvCache, Result, Shape, Tensor};

use crate::cublas::CublasTranspose;
use crate::{ops, CudaBuffer, CudaContext, CudaKvCache};

fn tensor_slice(
    tensor: &Tensor,
    byte_offset: usize,
    len: usize,
    device_id: usize,
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(device_id) {
        return Err(Error::DeviceMismatch {
            expected: Device::Cuda(device_id),
            got: tensor.device(),
        });
    }
    CudaBuffer::from_tensor(tensor)
        .and_then(|buffer| buffer.view(byte_offset, len))
        .map_err(Error::Cuda)
}

fn buffer_slice(buffer: &CudaBuffer, byte_offset: usize, len: usize) -> Result<CudaBuffer> {
    buffer.view(byte_offset, len).map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
fn gqa_scores(
    ctx: &CudaContext,
    dtype: DType,
    query: &Tensor,
    query_offset: usize,
    key_cache: &CudaBuffer,
    key_offset: usize,
    scores: &CudaBuffer,
    scores_offset: usize,
    gqa_ratio: usize,
    kv_len: usize,
    head_dim: usize,
) -> Result<()> {
    let element_bytes = dtype.size_in_bytes();
    let query = tensor_slice(
        query,
        query_offset * element_bytes,
        gqa_ratio * head_dim * element_bytes,
        ctx.device_id(),
    )?;
    let key = buffer_slice(
        key_cache,
        key_offset * element_bytes,
        kv_len * head_dim * element_bytes,
    )?;
    let output = buffer_slice(
        scores,
        scores_offset * element_bytes,
        gqa_ratio * kv_len * element_bytes,
    )?;
    ctx.cublas()
        .gemm_ex(
            dtype,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            gqa_ratio,
            kv_len,
            head_dim,
            1.0,
            &query,
            head_dim as i32,
            &key,
            head_dim as i32,
            0.0,
            &output,
            kv_len as i32,
        )
        .map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
fn gqa_values(
    ctx: &CudaContext,
    dtype: DType,
    attention: &Tensor,
    attention_offset: usize,
    value_cache: &CudaBuffer,
    value_offset: usize,
    output: &CudaBuffer,
    output_offset: usize,
    gqa_ratio: usize,
    kv_len: usize,
    head_dim: usize,
) -> Result<()> {
    let element_bytes = dtype.size_in_bytes();
    let attention = tensor_slice(
        attention,
        attention_offset * element_bytes,
        gqa_ratio * kv_len * element_bytes,
        ctx.device_id(),
    )?;
    let value = buffer_slice(
        value_cache,
        value_offset * element_bytes,
        kv_len * head_dim * element_bytes,
    )?;
    let output = buffer_slice(
        output,
        output_offset * element_bytes,
        gqa_ratio * head_dim * element_bytes,
    )?;
    ctx.cublas()
        .gemm_ex(
            dtype,
            CublasTranspose::None,
            CublasTranspose::None,
            gqa_ratio,
            head_dim,
            kv_len,
            1.0,
            &attention,
            kv_len as i32,
            &value,
            head_dim as i32,
            0.0,
            &output,
            head_dim as i32,
        )
        .map_err(Error::Cuda)
}

fn scale_legacy(ctx: &CudaContext, input: Tensor, factor: f32) -> Result<Tensor> {
    use crate::ffi::abi::elementwise as abi;
    let dtype = input.dtype();
    let input_buffer = CudaBuffer::from_tensor(&input).map_err(Error::Cuda)?;
    let output = CudaBuffer::alloc(input.size_in_bytes(), ctx.device_id()).map_err(Error::Cuda)?;
    let count = u32::try_from(input.numel())
        .map_err(|_| Error::Other("attention score count exceeds u32".into()))?;
    unsafe {
        crate::ops::validation::check_cuda(match dtype {
            DType::F32 => abi::apxinf_elementwise_fixed_decode_scale_f32(
                input_buffer.ptr(),
                output.ptr(),
                count,
                factor,
                ctx.stream().handle(),
            ),
            DType::BF16 => abi::apxinf_elementwise_fixed_decode_scale_bf16(
                input_buffer.ptr(),
                output.ptr(),
                count,
                factor,
                ctx.stream().handle(),
            ),
            _ => {
                return Err(Error::Other(format!(
                    "attention scale does not support {dtype}"
                )))
            }
        })?;
    }
    Ok(output.into_tensor(input.shape().clone(), dtype))
}

fn softmax_causal_legacy(
    ctx: &CudaContext,
    input: &Tensor,
    kv_offset: u32,
    n_heads: u32,
) -> Result<Tensor> {
    use crate::ffi::abi::attention as abi;
    let dims = input.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other("attention softmax expects a matrix".into()));
    }
    let output = CudaBuffer::alloc(input.size_in_bytes(), ctx.device_id()).map_err(Error::Cuda)?;
    let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let cols =
        u32::try_from(dims[1]).map_err(|_| Error::Other("attention columns exceed u32".into()))?;
    let rows =
        u32::try_from(dims[0]).map_err(|_| Error::Other("attention rows exceed u32".into()))?;
    unsafe {
        crate::ops::validation::check_cuda(match input.dtype() {
            DType::F32 => abi::apxinf_attention_fixed_decode_softmax_f32(
                input_buffer.ptr(),
                output.ptr(),
                cols,
                rows,
                kv_offset,
                n_heads,
                ctx.stream().handle(),
            ),
            DType::BF16 => abi::apxinf_attention_fixed_decode_softmax_bf16(
                input_buffer.ptr(),
                output.ptr(),
                cols,
                rows,
                kv_offset,
                n_heads,
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(Error::Other(format!(
                    "attention softmax does not support {dtype}"
                )))
            }
        })?;
    }
    Ok(output.into_tensor(input.shape().clone(), input.dtype()))
}

fn causal_prefill(
    ctx: &CudaContext,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    let q = query.shape().dims();
    let k = key.shape().dims();
    if q.len() != 3
        || k.len() != 3
        || value.shape() != key.shape()
        || q.contains(&0)
        || k.contains(&0)
        || q[2] != k[2]
        || q[1] % k[1] != 0
        || k[0] < q[0]
    {
        return Err(Error::Other("causal attention shape mismatch".into()));
    }
    let query = query.reshape(vec![1, q[0], q[1], q[2]])?;
    let key = key.reshape(vec![1, k[0], k[1], k[2]])?;
    let value = value.reshape(vec![1, k[0], k[1], k[2]])?;
    let mut output = ctx.allocate_output(query.shape().clone(), query.dtype())?;
    let mut args = ops::KvCacheAttentionArgs::new(&query, &key, &value, &mut output);
    args.valid_key_tokens = k[0];
    args.query_start = k[0] - q[0];
    args.numerics = ops::AttentionNumerics::Flexible;
    ops::kv_cache_attention(ctx, args)?;
    output.reshape(q.to_vec())
}

pub fn sdpa(
    ctx: &CudaContext,
    query: &Tensor,
    kv: &dyn KvCache,
    layer_idx: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    max_seq_len: usize,
    kv_offset: u32,
) -> Result<Tensor> {
    if n_kv_heads == 0 || n_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(Error::Other(format!(
            "attention requires non-zero divisible head counts, got {n_heads}/{n_kv_heads}"
        )));
    }
    let query_dims = query.shape().dims();
    if query.device() != Device::Cuda(ctx.device_id())
        || query_dims.len() != 3
        || query_dims[1] != n_heads
        || query_dims[2] != head_dim
    {
        return Err(Error::Other(format!(
            "attention query must be CUDA{} [seq,{n_heads},{head_dim}], got {} {:?}",
            ctx.device_id(),
            query.device(),
            query_dims
        )));
    }
    if kv_len == 0 || kv_len > max_seq_len {
        return Err(Error::Other(format!(
            "attention kv_len {kv_len} is outside 1..={max_seq_len}"
        )));
    }
    let cache = kv
        .as_any()
        .downcast_ref::<CudaKvCache>()
        .ok_or_else(|| Error::Other("expected a cuda-new KV cache".into()))?;
    let seq_len = query_dims[0];
    if seq_len == 0 || kv_len < seq_len {
        return Err(Error::Other(
            "attention kv_len is smaller than the query".into(),
        ));
    }
    let gqa_ratio = n_heads / n_kv_heads;
    let dtype = query.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16) {
        return Err(Error::Other(format!(
            "portable cache attention does not support {dtype}"
        )));
    }
    let element_bytes = dtype.size_in_bytes();

    // Preserve the legacy Qwen3-VL BF16 prefill fast path: gather the valid
    // head-major prefix into dense token-major K/V solely for the FA2 call.
    #[cfg(apxinf_fa2_decode)]
    if dtype == DType::BF16
        && head_dim == 128
        && kv_offset == 0
        && kv_len == seq_len
        && std::env::var_os("APXINF_PREFILL_SDPA_LEGACY").is_none()
    {
        use crate::ffi::abi::cache as abi;
        let gather = |src: &CudaBuffer| -> Result<CudaBuffer> {
            let dst = CudaBuffer::alloc(
                seq_len * n_kv_heads * head_dim * element_bytes,
                ctx.device_id(),
            )
            .map_err(Error::Cuda)?;
            unsafe {
                crate::ops::validation::check_cuda(abi::apxinf_cache_fixed_decode_gather_bf16(
                    src.ptr(),
                    dst.ptr(),
                    seq_len as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    max_seq_len as u32,
                    kv_offset,
                    ctx.stream().handle(),
                ))?;
            }
            Ok(dst)
        };
        let key = gather(cache.k_buffer(layer_idx))?
            .into_tensor(Shape::new(vec![seq_len, n_kv_heads, head_dim]), DType::BF16);
        let value = gather(cache.v_buffer(layer_idx))?
            .into_tensor(Shape::new(vec![seq_len, n_kv_heads, head_dim]), DType::BF16);
        return causal_prefill(ctx, query, &key, &value)?
            .reshape(vec![seq_len, n_heads * head_dim]);
    }

    let scores = CudaBuffer::alloc(seq_len * n_heads * kv_len * element_bytes, ctx.device_id())
        .map_err(Error::Cuda)?;
    for kv_head in 0..n_kv_heads {
        for sequence in 0..seq_len {
            gqa_scores(
                ctx,
                dtype,
                query,
                (sequence * n_heads + kv_head * gqa_ratio) * head_dim,
                cache.k_buffer(layer_idx),
                kv_head * max_seq_len * head_dim,
                &scores,
                (sequence * n_heads + kv_head * gqa_ratio) * kv_len,
                gqa_ratio,
                kv_len,
                head_dim,
            )?;
        }
    }
    let scores = scores.into_tensor(Shape::new(vec![seq_len * n_heads, kv_len]), dtype);
    let scores = scale_legacy(ctx, scores, 1.0 / (head_dim as f32).sqrt())?;
    let attention = softmax_causal_legacy(ctx, &scores, kv_offset, n_heads as u32)?;

    let output = CudaBuffer::alloc(
        seq_len * n_heads * head_dim * element_bytes,
        ctx.device_id(),
    )
    .map_err(Error::Cuda)?;
    for kv_head in 0..n_kv_heads {
        for sequence in 0..seq_len {
            gqa_values(
                ctx,
                dtype,
                &attention,
                (sequence * n_heads + kv_head * gqa_ratio) * kv_len,
                cache.v_buffer(layer_idx),
                kv_head * max_seq_len * head_dim,
                &output,
                (sequence * n_heads + kv_head * gqa_ratio) * head_dim,
                gqa_ratio,
                kv_len,
                head_dim,
            )?;
        }
    }
    Ok(output.into_tensor(Shape::new(vec![seq_len, n_heads * head_dim]), dtype))
}

// ── Decode fast-path `_into` surface (caller-owned buffers, graph-safe) ───

/// F32 attention softmax into caller-owned storage using a device position.
pub fn softmax_into(
    ctx: &CudaContext,
    scores: &CudaBuffer,
    output: &CudaBuffer,
    cols: usize,
    heads: usize,
    position: crate::CudaDeviceAddress,
) -> Result<()> {
    use crate::ffi::abi::attention as abi;
    use crate::ops::validation::{check_cuda, checked_bytes, require_address, require_buffers};
    let bytes = checked_bytes(DType::F32, &[heads, cols], "attention softmax")?;
    require_buffers(
        ctx,
        "attention softmax",
        &[("scores", scores, bytes), ("output", output, bytes)],
    )?;
    require_address(ctx, "attention softmax", "position", position, 4)?;
    check_cuda(unsafe {
        abi::apxinf_attention_fixed_decode_softmax_decode_f32(
            scores.ptr(),
            output.ptr(),
            cols as u32,
            heads as u32,
            position.ptr(),
            ctx.stream().handle(),
        )
    })
}

/// Decode-time BF16 flash attention into caller-owned storage. KV caches are
/// head-major `[kv_heads, max_seq_len, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn flash_into(
    ctx: &CudaContext,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &CudaBuffer,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    bucket_kv_len: usize,
    max_seq_len: usize,
    scale: f32,
    position: crate::CudaDeviceAddress,
) -> Result<()> {
    use crate::ffi::abi::attention as abi;
    use crate::ops::validation::{
        check_cuda, checked_bytes, require_address, require_buffers, require_finite,
    };
    require_finite("flash attention", &[scale])?;
    if heads == 0 || kv_heads == 0 || heads % kv_heads != 0 || bucket_kv_len > max_seq_len {
        return Err(Error::Other(
            "flash attention received invalid head or sequence dimensions".into(),
        ));
    }
    let query_size = checked_bytes(DType::BF16, &[heads, head_dim], "flash attention")?;
    let cache_size = checked_bytes(
        DType::BF16,
        &[kv_heads, max_seq_len, head_dim],
        "flash attention",
    )?;
    require_buffers(
        ctx,
        "flash attention",
        &[
            ("query", query, query_size),
            ("key cache", key_cache, cache_size),
            ("value cache", value_cache, cache_size),
            ("output", output, query_size),
        ],
    )?;
    require_address(ctx, "flash attention", "position", position, 4)?;
    check_cuda(unsafe {
        abi::apxinf_attention_fixed_decode_flash_attn_decode_bf16(
            query.ptr(),
            key_cache.ptr(),
            value_cache.ptr(),
            output.ptr(),
            heads as u32,
            kv_heads as u32,
            head_dim as u32,
            bucket_kv_len as u32,
            max_seq_len as u32,
            scale,
            position.ptr(),
            ctx.stream().handle(),
        )
    })
}
