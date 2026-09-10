//! Model-neutral attention contracts and workspace orchestration.

use apxinf_core::{DType, Device, Error, KvCache, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, check_cuda, checked_bytes, f16_output, gpu_ptr, make_gpu_tensor, matrix_shape,
    optional_ptr, require_address, require_buffers, require_finite, unsupported_dtype,
};
use super::elementwise::{bias_f16, concat_rows_f16};
use crate::buffer::{CudaBuffer, CudaDeviceAddress};
use crate::context::CudaContext;
use crate::cublas::CublasTranspose;
use crate::ffi;
use crate::workspace::{may_prepare_native_resources, output_buffer};
use crate::CudaKVCache;

pub struct QkvTensors {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

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

/// GQA scaled-dot-product attention over an existing CUDA KV cache.
#[allow(clippy::too_many_arguments)]
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
        .downcast_ref::<CudaKVCache>()
        .ok_or_else(|| Error::Other("expected CudaKVCache".into()))?;

    let seq_len = query_dims[0];
    let gqa_ratio = n_heads / n_kv_heads;
    let dtype = query.dtype();
    let element_bytes = dtype.size_in_bytes();
    let scores = CudaBuffer::alloc(seq_len * n_heads * kv_len * element_bytes, ctx.device_id())
        .map_err(Error::Cuda)?;
    let key_cache = cache.k_buffer(layer_idx);

    for kv_head in 0..n_kv_heads {
        for sequence in 0..seq_len {
            gqa_scores(
                ctx,
                dtype,
                query,
                (sequence * n_heads + kv_head * gqa_ratio) * head_dim,
                key_cache,
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
    let scores = super::elementwise::scale(ctx, &scores, 1.0 / (head_dim as f32).sqrt())?;
    let attention = softmax_causal(ctx, &scores, kv_offset, n_heads as u32)?;

    let output = CudaBuffer::alloc(
        seq_len * n_heads * head_dim * element_bytes,
        ctx.device_id(),
    )
    .map_err(Error::Cuda)?;
    let value_cache = cache.v_buffer(layer_idx);
    for kv_head in 0..n_kv_heads {
        for sequence in 0..seq_len {
            gqa_values(
                ctx,
                dtype,
                &attention,
                (sequence * n_heads + kv_head * gqa_ratio) * kv_len,
                value_cache,
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

/// F32 attention softmax into caller-owned storage using a device position.
pub fn softmax_f32_into(
    ctx: &CudaContext,
    scores: &CudaBuffer,
    output: &CudaBuffer,
    cols: usize,
    heads: usize,
    position: CudaDeviceAddress,
) -> Result<()> {
    let bytes = checked_bytes(DType::F32, &[heads, cols], "attention softmax")?;
    require_buffers(
        ctx,
        "attention softmax",
        &[("scores", scores, bytes), ("output", output, bytes)],
    )?;
    require_address(ctx, "attention softmax", "position", position, 4)?;
    check_cuda(unsafe {
        ffi::apxinf_attention_softmax_decode_f32(
            scores.ptr(),
            output.ptr(),
            cols as u32,
            heads as u32,
            position.ptr(),
            ctx.stream().handle(),
        )
    })
}

/// Decode-time BF16 flash attention into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn flash_bf16_into(
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
    position: CudaDeviceAddress,
) -> Result<()> {
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
        ffi::apxinf_flash_attn_decode_bf16(
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

/// Softmax on CUDA. Dispatches on dtype.
pub fn softmax(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let rows = dims[dims.len() - 2];
    let cols = *dims.last().unwrap();

    let out_bytes = input.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    unsafe {
        let res = match input.dtype() {
            DType::F32 => ffi::apxinf_softmax_f32(
                gpu_ptr(input)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_softmax_bf16(
                gpu_ptr(input)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("softmax", dtype),
        };
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }

    Ok(make_gpu_tensor(
        input.shape().clone(),
        input.dtype(),
        device_id,
        out_buf,
    ))
}

/// Non-causal full attention for the vision tower. Q/K/V each
/// `[seq, n_heads, head_dim]` bf16; returns `[seq, n_heads * head_dim]`.
/// head_dim must be 64 (Qwen3-VL-2B vision).
pub fn vision(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        return Err(Error::Other("vision_sdpa: only BF16 supported".into()));
    }
    if head_dim != 64 {
        return Err(Error::Other("vision_sdpa: head_dim must be 64".into()));
    }
    let device_id = ctx.device_id();
    let out_bytes = seq_len * n_heads * head_dim * DType::BF16.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    unsafe {
        let res = ffi::apxinf_vision_sdpa_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            out_buf.ptr(),
            seq_len as u32,
            n_heads as u32,
            head_dim as u32,
            scale,
            ctx.stream().handle(),
        );
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![seq_len, n_heads * head_dim]),
        DType::BF16,
        device_id,
        out_buf,
    ))
}

/// Causal attention mask on CUDA. Dispatches on dtype.
pub fn causal_mask(ctx: &CudaContext, input: &Tensor, kv_offset: u32) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let rows = dims[dims.len() - 2];
    let cols = *dims.last().unwrap();

    let out_bytes = input.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    unsafe {
        let res = match input.dtype() {
            DType::F32 => ffi::apxinf_causal_mask_f32(
                gpu_ptr(input)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                kv_offset,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_causal_mask_bf16(
                gpu_ptr(input)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                kv_offset,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("causal_mask", dtype),
        };
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }

    Ok(make_gpu_tensor(
        input.shape().clone(),
        input.dtype(),
        device_id,
        out_buf,
    ))
}

/// Fused causal mask + softmax. Dispatches on dtype.
pub fn softmax_causal(
    ctx: &CudaContext,
    input: &Tensor,
    kv_offset: u32,
    n_heads: u32,
) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let rows = dims[dims.len() - 2];
    let cols = *dims.last().unwrap();

    let out_bytes = input.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    unsafe {
        let res = match input.dtype() {
            DType::F32 => ffi::apxinf_attention_softmax_f32(
                gpu_ptr(input)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                kv_offset,
                n_heads,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_attention_softmax_bf16(
                gpu_ptr(input)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                kv_offset,
                n_heads,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("attention_softmax", dtype),
        };
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }

    Ok(make_gpu_tensor(
        input.shape().clone(),
        input.dtype(),
        device_id,
        out_buf,
    ))
}
pub fn split_qkv_bias_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    heads: usize,
    head_dim: usize,
) -> Result<QkvTensors> {
    let (tokens, width) = matrix_shape(qkv, "vision QKV split")?;
    let projection_width = heads * head_dim;
    if qkv.dtype() != DType::BF16
        || width != 3 * projection_width
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [width])
    {
        return Err(Error::Other(
            "static inference BF16 vision QKV shape mismatch".into(),
        ));
    }
    let q = bf16_output(ctx, tokens, projection_width)?;
    let k = bf16_output(ctx, tokens, projection_width)?;
    let v = bf16_output(ctx, tokens, projection_width)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qkv_split_bias_bf16(
            gpu_ptr(qkv)?,
            optional_ptr(bias)?,
            q.ptr(),
            k.ptr(),
            v.ptr(),
            tokens as i32,
            projection_width as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    let shape = Shape::new(vec![tokens, heads, head_dim]);
    Ok(QkvTensors {
        q: make_gpu_tensor(shape.clone(), DType::BF16, ctx.device_id(), q),
        k: make_gpu_tensor(shape.clone(), DType::BF16, ctx.device_id(), k),
        v: make_gpu_tensor(shape, DType::BF16, ctx.device_id(), v),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn split_gqa_qkv_mrope_cache_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    position_ids: &CudaBuffer,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    sections: [usize; 3],
    cache_tokens: usize,
    caches: Option<(&Tensor, &Tensor, usize)>,
) -> Result<QkvTensors> {
    let (tokens, width) = matrix_shape(qkv, "GQA QKV mRoPE")?;
    let q_width = q_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let expected_width = q_width + 2 * kv_width;
    if !matches!(qkv.dtype(), DType::BF16 | DType::F16)
        || width != expected_width
        || q_heads == 0
        || kv_heads == 0
        || q_heads % kv_heads != 0
        || head_dim == 0
        || head_dim > 256
        || head_dim % 2 != 0
        || !theta.is_finite()
        || theta <= 0.0
        || sections[1] + sections[2] > head_dim / 2
        || position_ids.len() < tokens * 3 * std::mem::size_of::<u32>()
        || bias.is_some_and(|value| {
            value.dtype() != DType::BF16 || value.shape().dims() != [expected_width]
        })
    {
        return Err(Error::Other(
            "static inference BF16 GQA QKV mRoPE shape mismatch".into(),
        ));
    }
    let q = bf16_output(ctx, tokens, q_width)?;
    let owned_k = caches
        .is_none()
        .then(|| bf16_output(ctx, cache_tokens * kv_heads, head_dim))
        .transpose()?;
    let owned_v = caches
        .is_none()
        .then(|| bf16_output(ctx, cache_tokens * kv_heads, head_dim))
        .transpose()?;
    let (k_ptr, v_ptr, cache_offset) = if let Some((k, v, offset)) = caches {
        let expected_shape = [cache_tokens, kv_heads, head_dim];
        if k.dtype() != DType::BF16
            || v.dtype() != DType::BF16
            || k.shape().dims() != expected_shape
            || v.shape().dims() != expected_shape
            || offset + tokens > cache_tokens
        {
            return Err(Error::Other(
                "static inference BF16 GQA QKV cache shape mismatch".into(),
            ));
        }
        (gpu_ptr(k)?, gpu_ptr(v)?, offset)
    } else {
        if tokens > cache_tokens {
            return Err(Error::Other(
                "static inference BF16 GQA QKV cache is too short".into(),
            ));
        }
        (
            owned_k.as_ref().unwrap().ptr(),
            owned_v.as_ref().unwrap().ptr(),
            0,
        )
    };
    let status = unsafe {
        let launch = if qkv.dtype() == DType::F16 {
            ffi::apxinf_static_gqa_qkv_mrope_cache_f16
        } else {
            ffi::apxinf_static_gqa_qkv_mrope_cache_bf16
        };
        launch(
            gpu_ptr(qkv)?,
            optional_ptr(bias)?,
            position_ids.ptr().cast(),
            q.ptr(),
            k_ptr,
            v_ptr,
            tokens as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            theta,
            sections[1] as i32,
            sections[2] as i32,
            cache_offset as i32,
            ctx.stream().handle(),
        )
    };
    ffi::check_cuda(status).map_err(Error::Cuda)?;
    let q = make_gpu_tensor(
        Shape::new(vec![tokens, q_heads, head_dim]),
        DType::BF16,
        ctx.device_id(),
        q,
    );
    if let (Some(k), Some(v)) = (owned_k, owned_v) {
        Ok(QkvTensors {
            q,
            k: make_gpu_tensor(
                Shape::new(vec![cache_tokens, kv_heads, head_dim]),
                DType::BF16,
                ctx.device_id(),
                k,
            ),
            v: make_gpu_tensor(
                Shape::new(vec![cache_tokens, kv_heads, head_dim]),
                DType::BF16,
                ctx.device_id(),
                v,
            ),
        })
    } else {
        Ok(QkvTensors {
            q,
            k: caches.unwrap().0.clone(),
            v: caches.unwrap().1.clone(),
        })
    }
}

pub fn split_vision_qkv_rope_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    position_ids: &CudaBuffer,
    heads: usize,
    head_dim: usize,
    theta: f32,
) -> Result<QkvTensors> {
    let (tokens, width) = matrix_shape(qkv, "vision QKV RoPE")?;
    let projection_width = heads * head_dim;
    let expected_width = 3 * projection_width;
    if !matches!(qkv.dtype(), DType::BF16 | DType::F16)
        || width != expected_width
        || heads == 0
        || head_dim == 0
        || head_dim > 256
        || head_dim % 4 != 0
        || !theta.is_finite()
        || theta <= 0.0
        || position_ids.len() < tokens * 2 * std::mem::size_of::<u32>()
        || bias.is_some_and(|value| {
            value.dtype() != DType::BF16 || value.shape().dims() != [expected_width]
        })
    {
        return Err(Error::Other(
            "static inference BF16 vision QKV RoPE shape mismatch".into(),
        ));
    }
    let q = bf16_output(ctx, tokens, projection_width)?;
    let k = bf16_output(ctx, tokens, projection_width)?;
    let v = bf16_output(ctx, tokens, projection_width)?;
    let status = unsafe {
        let launch = if qkv.dtype() == DType::F16 {
            ffi::apxinf_static_vision_qkv_rope_f16
        } else {
            ffi::apxinf_static_vision_qkv_rope_bf16
        };
        launch(
            gpu_ptr(qkv)?,
            optional_ptr(bias)?,
            position_ids.ptr().cast(),
            q.ptr(),
            k.ptr(),
            v.ptr(),
            tokens as i32,
            heads as i32,
            head_dim as i32,
            theta,
            ctx.stream().handle(),
        )
    };
    ffi::check_cuda(status).map_err(Error::Cuda)?;
    let shape = Shape::new(vec![tokens, heads, head_dim]);
    Ok(QkvTensors {
        q: make_gpu_tensor(shape.clone(), DType::BF16, ctx.device_id(), q),
        k: make_gpu_tensor(shape.clone(), DType::BF16, ctx.device_id(), k),
        v: make_gpu_tensor(shape, DType::BF16, ctx.device_id(), v),
    })
}

#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
#[allow(clippy::too_many_arguments)]
fn fa2_attention(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batches: usize,
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let output = output_buffer(ctx, q.size_in_bytes())?;
    let lse_elements = batches
        .checked_mul(query_heads)
        .and_then(|value| value.checked_mul(query_tokens))
        .ok_or_else(|| Error::Other("static inference BF16 FA2 LSE size overflow".into()))?;
    let softmax_lse = output_buffer(
        ctx,
        lse_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                Error::Other("static inference BF16 FA2 LSE byte size overflow".into())
            })?,
    )?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_fa2_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            softmax_lse.ptr(),
            batches as i32,
            query_tokens as i32,
            key_tokens as i32,
            query_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            (head_dim as f32).sqrt().recip(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fa2_attention_causal(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let output = output_buffer(ctx, q.size_in_bytes())?;
    let softmax_lse = output_buffer(ctx, query_heads * query_tokens * std::mem::size_of::<f32>())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_fa2_bf16_causal(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            softmax_lse.ptr(),
            1,
            query_tokens as i32,
            key_tokens as i32,
            query_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            (head_dim as f32).sqrt().recip(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
fn fa2_splitkv_enabled(
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> bool {
    if std::env::var_os("APXINF_DISABLE_FA2_SPLITKV").is_some() {
        return false;
    }
    query_tokens <= 64
        && key_tokens > query_tokens
        && query_heads > kv_heads
        && matches!(head_dim, 128 | 256)
}

#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fa2_attention_splitkv(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batches: usize,
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Result<Tensor> {
    let output = output_buffer(ctx, q.size_in_bytes())?;
    let lse_elements = batches
        .checked_mul(query_heads)
        .and_then(|value| value.checked_mul(query_tokens))
        .ok_or_else(|| Error::Other("static inference BF16 split-KV LSE size overflow".into()))?;
    let softmax_lse = output_buffer(
        ctx,
        lse_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::Other("static inference BF16 split-KV LSE overflow".into()))?,
    )?;
    let block_n = if head_dim <= 64 {
        256
    } else if head_dim <= 128 {
        128
    } else {
        64
    };
    let max_splits = key_tokens.div_ceil(block_n).min(128);
    let softmax_lse_accum = output_buffer(
        ctx,
        max_splits
            .checked_mul(lse_elements)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                Error::Other("static inference BF16 split-KV LSE accum overflow".into())
            })?,
    )?;
    let o_accum_elements = max_splits
        .checked_mul(batches)
        .and_then(|value| value.checked_mul(query_tokens))
        .and_then(|value| value.checked_mul(query_heads))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::Other("static inference BF16 split-KV O accum overflow".into()))?;
    let o_accum = output_buffer(
        ctx,
        o_accum_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                Error::Other("static inference BF16 split-KV O accum byte overflow".into())
            })?,
    )?;
    unsafe {
        let status = if causal {
            ffi::apxinf_static_fa2_bf16_causal_splitkv(
                gpu_ptr(q)?,
                gpu_ptr(k)?,
                gpu_ptr(v)?,
                output.ptr(),
                softmax_lse.ptr(),
                softmax_lse_accum.ptr(),
                o_accum.ptr(),
                batches as i32,
                query_tokens as i32,
                key_tokens as i32,
                query_heads as i32,
                kv_heads as i32,
                head_dim as i32,
                (head_dim as f32).sqrt().recip(),
                ctx.caps().multiprocessor_count as i32,
                ctx.stream().handle(),
            )
        } else {
            ffi::apxinf_static_fa2_bf16_splitkv(
                gpu_ptr(q)?,
                gpu_ptr(k)?,
                gpu_ptr(v)?,
                output.ptr(),
                softmax_lse.ptr(),
                softmax_lse_accum.ptr(),
                o_accum.ptr(),
                batches as i32,
                query_tokens as i32,
                key_tokens as i32,
                query_heads as i32,
                kv_heads as i32,
                head_dim as i32,
                (head_dim as f32).sqrt().recip(),
                ctx.caps().multiprocessor_count as i32,
                ctx.stream().handle(),
            )
        };
        ffi::check_cuda(status).map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

#[cfg(apxinf_cutlass_fmha)]
fn cublas_mqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let output = output_buffer(ctx, q.size_in_bytes())?;
    let status = unsafe {
        ffi::apxinf_static_cublas_mqa_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            q_shape[0] as i32,
            key_tokens as i32,
            q_shape[1] as i32,
            q_shape[2] as i32,
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)?;
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

pub fn mqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if [q, k, v]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || q_shape.len() != 3
        || k_shape.len() < 2
        || v.shape() != k.shape()
        || k_shape[k_shape.len() - 1] != q_shape[2]
        || key_tokens == 0
        || key_tokens > k.numel() / q_shape[2]
    {
        return Err(Error::Other(
            "static inference BF16 MQA shape mismatch".into(),
        ));
    }
    #[cfg(apxinf_fa2_sm80)]
    {
        if fa2_splitkv_enabled(q_shape[0], key_tokens, q_shape[1], 1, q_shape[2]) {
            return fa2_attention_splitkv(
                ctx, q, k, v, 1, q_shape[0], key_tokens, q_shape[1], 1, q_shape[2], false,
            );
        }
        return fa2_attention(
            ctx, q, k, v, 1, q_shape[0], key_tokens, q_shape[1], 1, q_shape[2],
        );
    }
    #[cfg(apxinf_cutlass_fmha)]
    {
        return cublas_mqa_bf16(ctx, q, k, v, key_tokens);
    }
    #[cfg(all(not(apxinf_fa2_sm80), not(apxinf_cutlass_fmha)))]
    let output = output_buffer(ctx, q.size_in_bytes())?;
    #[cfg(all(not(apxinf_fa2_sm80), not(apxinf_cutlass_fmha)))]
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_mqa_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            q_shape[0] as i32,
            key_tokens as i32,
            q_shape[1] as i32,
            q_shape[2] as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    #[cfg(all(not(apxinf_fa2_sm80), not(apxinf_cutlass_fmha)))]
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

pub fn causal_gqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if [q, k, v]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || q_shape.len() != 3
        || k_shape.len() != 3
        || v.shape() != k.shape()
        || k_shape[0] < key_tokens
        || k_shape[2] != q_shape[2]
        || k_shape[1] == 0
        || q_shape[1] % k_shape[1] != 0
        || key_tokens < q_shape[0]
    {
        return Err(Error::Other(
            "static inference causal BF16 GQA shape mismatch".into(),
        ));
    }
    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    {
        if fa2_splitkv_enabled(q_shape[0], key_tokens, q_shape[1], k_shape[1], q_shape[2]) {
            return fa2_attention_splitkv(
                ctx, q, k, v, 1, q_shape[0], key_tokens, q_shape[1], k_shape[1], q_shape[2], true,
            );
        }
        return fa2_attention_causal(
            ctx, q, k, v, q_shape[0], key_tokens, q_shape[1], k_shape[1], q_shape[2],
        );
    }
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    Err(Error::Other(
        "causal BF16 GQA requires the FA2 backend".into(),
    ))
}

pub fn mha_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    tokens_per_batch: usize,
) -> Result<Tensor> {
    let shape = q.shape().dims();
    if [q, k, v]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || shape.len() != 3
        || k.shape() != q.shape()
        || v.shape() != q.shape()
        || shape[2] > 256
        || tokens_per_batch == 0
        || shape[0] % tokens_per_batch != 0
    {
        return Err(Error::Other(
            "static inference BF16 MHA shape mismatch".into(),
        ));
    }
    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    {
        return fa2_attention(
            ctx,
            q,
            k,
            v,
            shape[0] / tokens_per_batch,
            tokens_per_batch,
            tokens_per_batch,
            shape[1],
            shape[1],
            shape[2],
        );
    }
    #[cfg(all(apxinf_cutlass_fmha, not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))))]
    if tokens_per_batch == 256 && shape[1] == 16 && shape[2] == 72 {
        let output = output_buffer(ctx, q.size_in_bytes())?;
        unsafe {
            let batches = shape[0] / tokens_per_batch;
            if may_prepare_native_resources() {
                let status = ffi::apxinf_static_prepare_cutlass_mha_bf16(
                    gpu_ptr(q)?,
                    gpu_ptr(k)?,
                    gpu_ptr(v)?,
                    output.ptr(),
                    batches as i32,
                    tokens_per_batch as i32,
                    tokens_per_batch as i32,
                    shape[1] as i32,
                    shape[1] as i32,
                    shape[2] as i32,
                    ctx.stream().handle(),
                );
                if status != 0 {
                    return Err(Error::Cuda(format!(
                        "CUTLASS BF16 FMHA resource preparation failed with status {status}"
                    )));
                }
            }
            let status = ffi::apxinf_static_cutlass_mha_bf16(
                gpu_ptr(q)?,
                gpu_ptr(k)?,
                gpu_ptr(v)?,
                output.ptr(),
                batches as i32,
                tokens_per_batch as i32,
                tokens_per_batch as i32,
                shape[1] as i32,
                shape[1] as i32,
                shape[2] as i32,
                ctx.stream().handle(),
            );
            if status != 0 {
                return Err(Error::Cuda(format!(
                    "CUTLASS BF16 FMHA execution failed with status {status}"
                )));
            }
            return Ok(make_gpu_tensor(
                q.shape().clone(),
                DType::BF16,
                ctx.device_id(),
                output,
            ));
        }
    }
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    let output = output_buffer(ctx, q.size_in_bytes())?;
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_mha_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            tokens_per_batch as i32,
            (shape[0] / tokens_per_batch) as i32,
            shape[1] as i32,
            shape[2] as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

pub fn segmented_mha_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    offsets: &crate::buffer::CudaBuffer,
    host_offsets: &[u32],
    segments: usize,
    max_tokens: usize,
) -> Result<Tensor> {
    let shape = q.shape().dims();
    if [q, k, v]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || shape.len() != 3
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
    super::contracts::require_buffers(
        ctx,
        "segmented BF16 MHA",
        &[(
            "offsets",
            offsets,
            (segments + 1) * std::mem::size_of::<u32>(),
        )],
    )?;
    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    {
        // TEMP-DIAG (implement_r4): per-call gate + clock; revert in the acceptance-bound revision.
        static MHA_DIAG_CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let mha_diag_call = MHA_DIAG_CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mha_diag = mha_diag_call < 2; // blocks 0 and 1 are the receipt-implicated calls
        let mha_t0 = std::time::Instant::now();
        // FIX (implement_r8): route-3 composed gemm_ex + tiled-softmax per-segment attention
        // replaces the r6/r7 pad route. The vendored FA2 fwd family is measured pathological
        // on sm_89 across hdim96 IsEvenK=false (r4 ~19.3s/launch), hdim96 IsEvenK=true
        // (r6 ~19.2s) and hdim128 IsEvenK=true (r7 ~28.2s), so the head_dim==64 vision arm
        // runs on measured-healthy cuBLAS GEMMs plus the tiled row-softmax kernel; scores
        // stay fp32 end-to-end and P is rounded to bf16 for the PV MMA exactly as FA2 does.
        // Revert/replace in the acceptance-bound revision per the prevailing marker policy.
        let orig_head_dim = shape[2]; // FIX (implement_r6)
        if orig_head_dim == 64 { // FIX (implement_r8): composed per-segment attention at the true dim
            let qf32_buf = CudaBuffer::alloc(shape[0] * shape[1] * orig_head_dim * 4, ctx.device_id())
                .map_err(Error::Cuda)?; // FIX (implement_r8)
            let qf32_t = qf32_buf.as_tensor(q.shape().clone(), DType::F32).map_err(Error::Cuda)?; // FIX (implement_r8)
            super::linear_attention::cast_bf16_to_f32(ctx, q, &qf32_t)?; // FIX (implement_r8)
            let kf32_buf = CudaBuffer::alloc(shape[0] * shape[1] * orig_head_dim * 4, ctx.device_id())
                .map_err(Error::Cuda)?; // FIX (implement_r8)
            let kf32_t = kf32_buf.as_tensor(k.shape().clone(), DType::F32).map_err(Error::Cuda)?; // FIX (implement_r8)
            super::linear_attention::cast_bf16_to_f32(ctx, k, &kf32_t)?; // FIX (implement_r8)
            let output = output_buffer(ctx, q.size_in_bytes())?; // FIX (implement_r8): packed [T,16,64]; out_bytes=85,327,872 signs the composed route
            // TEMP-DIAG (implement_r4): entry-alloc bracket; revert in the acceptance-bound revision.
            if mha_diag {
                eprintln!("[qwen_drive] mha_alloc_ok call={} out_bytes={} lse_bytes={} ms={}",
                          mha_diag_call, q.size_in_bytes(),
                          0, mha_t0.elapsed().as_millis());
            }
            for (seg_idx, bounds) in host_offsets.windows(2).enumerate() {
                let start = bounds[0] as usize;
                let tokens = (bounds[1] - bounds[0]) as usize;
                if tokens == 0 { continue; } // FIX (implement_r8): degenerate-segment guard
                // TEMP-DIAG (implement_r4): pre-launch segment marker; revert in the acceptance-bound revision.
                if mha_diag {
                    eprintln!("[qwen_drive] mha_seg call={} i={} start={} tokens={} ms={}",
                              mha_diag_call, seg_idx, start, tokens, mha_t0.elapsed().as_millis());
                }
                let scores = CudaBuffer::alloc(shape[1] * tokens * tokens * 4, ctx.device_id())
                    .map_err(Error::Cuda)?; // FIX (implement_r8): fp32 [heads,tokens,tokens]
                let probs = CudaBuffer::alloc(shape[1] * tokens * tokens * 2, ctx.device_id())
                    .map_err(Error::Cuda)?; // FIX (implement_r8): bf16 [heads,tokens,tokens]
                for head in 0..shape[1] {
                    let q_head = buffer_slice(
                        &qf32_buf,
                        (start * shape[1] * orig_head_dim + head * orig_head_dim) * 4,
                        ((tokens - 1) * shape[1] * orig_head_dim + orig_head_dim) * 4,
                    )?; // FIX (implement_r8)
                    let k_head = buffer_slice(
                        &kf32_buf,
                        (start * shape[1] * orig_head_dim + head * orig_head_dim) * 4,
                        ((tokens - 1) * shape[1] * orig_head_dim + orig_head_dim) * 4,
                    )?; // FIX (implement_r8)
                    let scores_head = buffer_slice(&scores, head * tokens * tokens * 4, tokens * tokens * 4)?; // FIX (implement_r8)
                    ctx.cublas()
                        .gemm_ex(
                            DType::F32,
                            CublasTranspose::None,
                            CublasTranspose::Transpose,
                            tokens,
                            tokens,
                            orig_head_dim,
                            (orig_head_dim as f32).sqrt().recip(),
                            &q_head,
                            (shape[1] * orig_head_dim) as i32,
                            &k_head,
                            (shape[1] * orig_head_dim) as i32,
                            0.0,
                            &scores_head,
                            tokens as i32,
                        )
                        .map_err(Error::Cuda)?; // FIX (implement_r8): Q*K^T; alpha folds the softmax scale bound to the true dim 64
                }
                unsafe {
                    ffi::check_cuda(ffi::apxinf_static_row_softmax_f32_bf16(
                        scores.ptr() as *const std::ffi::c_void,
                        probs.ptr(),
                        tokens as u32,
                        (shape[1] * tokens) as u32,
                        ctx.stream().handle(),
                    ))
                    .map_err(Error::Cuda)?; // FIX (implement_r8): tiled block-per-row softmax, fp32 in / bf16 out
                }
                for head in 0..shape[1] {
                    let probs_head = buffer_slice(&probs, head * tokens * tokens * 2, tokens * tokens * 2)?; // FIX (implement_r8)
                    let v_head = tensor_slice(
                        v,
                        (start * shape[1] * orig_head_dim + head * orig_head_dim) * 2,
                        ((tokens - 1) * shape[1] * orig_head_dim + orig_head_dim) * 2,
                        ctx.device_id(),
                    )?; // FIX (implement_r8)
                    let out_head = buffer_slice(
                        &output,
                        (start * shape[1] * orig_head_dim + head * orig_head_dim) * 2,
                        ((tokens - 1) * shape[1] * orig_head_dim + orig_head_dim) * 2,
                    )?; // FIX (implement_r8)
                    ctx.cublas()
                        .gemm_ex(
                            DType::BF16,
                            CublasTranspose::None,
                            CublasTranspose::None,
                            tokens,
                            orig_head_dim,
                            tokens,
                            1.0,
                            &probs_head,
                            tokens as i32,
                            &v_head,
                            (shape[1] * orig_head_dim) as i32,
                            0.0,
                            &out_head,
                            (shape[1] * orig_head_dim) as i32,
                        )
                        .map_err(Error::Cuda)?; // FIX (implement_r8): P*V straight into the packed [T,16,64] output
                }
                // TEMP-DIAG (implement_r4): post-launch retirement probe; converts an async fault into a loud Err; revert in the acceptance-bound revision.
                if mha_diag {
                    ctx.synchronize().map_err(Error::Cuda)?;
                    eprintln!("[qwen_drive] mha_seg_ok call={} i={} ms={}",
                              mha_diag_call, seg_idx, mha_t0.elapsed().as_millis());
                }
            }
            // TEMP-DIAG (implement_r4): separates "12 launches retired" from "epilogue cudaFree wedged"; revert in the acceptance-bound revision.
            if mha_diag {
                eprintln!("[qwen_drive] mha_loop_done call={} ms={}", mha_diag_call, mha_t0.elapsed().as_millis());
            }
            let result = make_gpu_tensor(
                q.shape().clone(),
                DType::BF16,
                ctx.device_id(),
                output,
            );
            return Ok(result);
        }
        let output = output_buffer(ctx, q.size_in_bytes())?;
        let softmax_lse = output_buffer(ctx, shape[0] * shape[1] * std::mem::size_of::<f32>())?;
        // TEMP-DIAG (implement_r4): entry-alloc bracket; revert in the acceptance-bound revision.
        if mha_diag {
            eprintln!("[qwen_drive] mha_alloc_ok call={} out_bytes={} lse_bytes={} ms={}",
                      mha_diag_call, q.size_in_bytes(),
                      shape[0] * shape[1] * std::mem::size_of::<f32>(), mha_t0.elapsed().as_millis());
        }
        let row_bytes = shape[1] * shape[2] * DType::BF16.size_in_bytes();
        let lse_row_bytes = shape[1] * std::mem::size_of::<f32>();
        for (seg_idx, bounds) in host_offsets.windows(2).enumerate() {
            let start = bounds[0] as usize;
            let tokens = (bounds[1] - bounds[0]) as usize;
            // TEMP-DIAG (implement_r4): pre-launch segment marker; revert in the acceptance-bound revision.
            if mha_diag {
                eprintln!("[qwen_drive] mha_seg call={} i={} start={} tokens={} ms={}",
                          mha_diag_call, seg_idx, start, tokens, mha_t0.elapsed().as_millis());
            }
            unsafe {
                ffi::check_cuda(ffi::apxinf_static_fa2_bf16(
                    gpu_ptr(q)?.cast::<u8>().add(start * row_bytes).cast(),
                    gpu_ptr(k)?.cast::<u8>().add(start * row_bytes).cast(),
                    gpu_ptr(v)?.cast::<u8>().add(start * row_bytes).cast(),
                    output.ptr().cast::<u8>().add(start * row_bytes).cast(),
                    softmax_lse
                        .ptr()
                        .cast::<u8>()
                        .add(start * lse_row_bytes)
                        .cast(),
                    1,
                    tokens as i32,
                    tokens as i32,
                    shape[1] as i32,
                    shape[1] as i32,
                    shape[2] as i32,
                    (shape[2] as f32).sqrt().recip(),
                    ctx.stream().handle(),
                ))
                .map_err(Error::Cuda)?;
            }
            // TEMP-DIAG (implement_r4): post-launch retirement probe; converts an async fault into a loud Err; revert in the acceptance-bound revision.
            if mha_diag {
                ctx.synchronize().map_err(Error::Cuda)?;
                eprintln!("[qwen_drive] mha_seg_ok call={} i={} ms={}",
                          mha_diag_call, seg_idx, mha_t0.elapsed().as_millis());
            }
        }
        // TEMP-DIAG (implement_r4): separates "12 launches retired" from "epilogue cudaFree wedged"; revert in the acceptance-bound revision.
        if mha_diag {
            eprintln!("[qwen_drive] mha_loop_done call={} ms={}", mha_diag_call, mha_t0.elapsed().as_millis());
        }
        let result = make_gpu_tensor(
            q.shape().clone(),
            DType::BF16,
            ctx.device_id(),
            output,
        );
        return Ok(result);
    }
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    let output = output_buffer(ctx, q.size_in_bytes())?;
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_segmented_mha_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            offsets.ptr(),
            output.ptr(),
            segments as i32,
            max_tokens as i32,
            shape[1] as i32,
            shape[2] as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}
pub(crate) fn cublas_mqa_f16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let output = output_buffer(ctx, q.size_in_bytes())?;
    let status = unsafe {
        ffi::apxinf_static_cublas_mqa_f16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            q_shape[0] as i32,
            key_tokens as i32,
            q_shape[1] as i32,
            q_shape[2] as i32,
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)?;
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

#[cfg(apxinf_fa2_f16_sm100)]
pub(crate) fn fa2_mqa_f16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let output = output_buffer(ctx, q.size_in_bytes())?;
    let lse_elements = q_shape[0]
        .checked_mul(q_shape[1])
        .ok_or_else(|| Error::Other("FA2 FP16 LSE size overflow".into()))?;
    let softmax_lse = output_buffer(
        ctx,
        lse_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::Other("FA2 FP16 LSE byte size overflow".into()))?,
    )?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_fa2_f16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            softmax_lse.ptr(),
            1,
            q_shape[0] as i32,
            key_tokens as i32,
            q_shape[1] as i32,
            1,
            q_shape[2] as i32,
            (q_shape[2] as f32).sqrt().recip(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

/// Exact MQA for the static inference action expert. Prefix and suffix
/// K/V are `[tokens, head_dim]`; Q/output are `[suffix, heads, head_dim]`.
pub fn mqa_prefix_suffix_f16(
    ctx: &CudaContext,
    q: &Tensor,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    suffix_k: &Tensor,
    suffix_v: &Tensor,
) -> Result<Tensor> {
    for tensor in [q, prefix_k, prefix_v, suffix_k, suffix_v] {
        if tensor.dtype() != DType::F16 {
            return Err(Error::DTypeMismatch {
                expected: DType::F16,
                got: tensor.dtype(),
            });
        }
    }
    let q_shape = q.shape().dims();
    let prefix_shape = prefix_k.shape().dims();
    let suffix_shape = suffix_k.shape().dims();
    if q_shape.len() != 3
        || prefix_shape.len() != 2
        || suffix_shape.len() != 2
        || prefix_v.shape().dims() != prefix_shape
        || suffix_v.shape().dims() != suffix_shape
        || q_shape[0] != suffix_shape[0]
        || q_shape[2] != prefix_shape[1]
        || q_shape[2] != suffix_shape[1]
    {
        return Err(Error::Other(format!(
            "static inference MQA shape mismatch: q={q_shape:?}, prefix={prefix_shape:?}, suffix={suffix_shape:?}"
        )));
    }
    let combined_k = concat_rows_f16(ctx, prefix_k, suffix_k)?;
    let combined_v = concat_rows_f16(ctx, prefix_v, suffix_v)?;
    cublas_mqa_f16(
        ctx,
        q,
        &combined_k,
        &combined_v,
        prefix_shape[0] + suffix_shape[0],
    )
}

/// MQA over an already contiguous K/V cache. This avoids rebuilding
/// `[prefix, suffix]` buffers at every action-expert layer and flow step.
pub fn mqa_cached_f16(
    ctx: &CudaContext,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    for tensor in [q, k_cache, v_cache] {
        if tensor.dtype() != DType::F16 {
            return Err(Error::DTypeMismatch {
                expected: DType::F16,
                got: tensor.dtype(),
            });
        }
    }
    let q_shape = q.shape().dims();
    let cache_shape = k_cache.shape().dims();
    if q_shape.len() != 3
        || cache_shape.len() != 2
        || v_cache.shape().dims() != cache_shape
        || cache_shape[0] != key_tokens
        || q_shape[2] != cache_shape[1]
    {
        return Err(Error::Other(format!(
            "static inference cached MQA shape mismatch: q={q_shape:?}, cache={cache_shape:?}, key_tokens={key_tokens}"
        )));
    }

    cublas_mqa_f16(ctx, q, k_cache, v_cache, key_tokens)
}

/// Split a biased dense QKV projection into `[tokens, heads, head_dim]`
/// tensors without RoPE. This is the SigLIP attention layout.
pub fn split_qkv_bias_f16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    heads: usize,
    head_dim: usize,
) -> Result<QkvTensors> {
    let (tokens, width) = matrix_shape(qkv, "vision QKV split")?;
    let projection_width = heads * head_dim;
    if qkv.dtype() != DType::F16
        || width != 3 * projection_width
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [width])
    {
        return Err(Error::Other(format!(
            "static inference vision QKV expected FP16 [tokens,{}], got {:?}",
            3 * projection_width,
            qkv.shape().dims()
        )));
    }
    let q_buffer = f16_output(ctx, tokens, projection_width)?;
    let k_buffer = f16_output(ctx, tokens, projection_width)?;
    let v_buffer = f16_output(ctx, tokens, projection_width)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qkv_split_bias_f16(
            gpu_ptr(qkv)?,
            bias.map(gpu_ptr)
                .transpose()?
                .unwrap_or(std::ptr::null_mut()),
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            tokens as i32,
            projection_width as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    let shape = Shape::new(vec![tokens, heads, head_dim]);
    Ok(QkvTensors {
        q: make_gpu_tensor(shape.clone(), DType::F16, ctx.device_id(), q_buffer),
        k: make_gpu_tensor(shape.clone(), DType::F16, ctx.device_id(), k_buffer),
        v: make_gpu_tensor(shape, DType::F16, ctx.device_id(), v_buffer),
    })
}

/// Apply QKV bias without splitting the projection, then let SM100-family
/// FMHA consume Q/K/V through row-strided views of `[tokens, 3 * heads * dim]`.
/// This avoids materializing three layout copies for SigLIP attention.
pub fn mha_packed_qkv_bias_f16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    tokens_per_batch: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let (tokens, width) = matrix_shape(qkv, "packed vision QKV")?;
    let projection_width = heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("packed vision QKV width overflow".into()))?;
    let packed_width = projection_width
        .checked_mul(3)
        .ok_or_else(|| Error::Other("packed vision QKV width overflow".into()))?;
    if qkv.dtype() != DType::F16
        || width != packed_width
        || tokens_per_batch == 0
        || tokens % tokens_per_batch != 0
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [width])
    {
        return Err(Error::Other(format!(
            "packed vision QKV expects FP16 [tokens,{}], got {:?}",
            packed_width,
            qkv.shape().dims()
        )));
    }

    #[cfg(apxinf_cutlass_fmha)]
    if tokens_per_batch == 256 && heads == 16 && head_dim == 72 {
        let biased = bias
            .map(|value| bias_f16(ctx, qkv, Some(value)))
            .transpose()?;
        let packed = biased.as_ref().unwrap_or(qkv);
        let output = f16_output(ctx, tokens, projection_width)?;
        unsafe {
            let batches = tokens / tokens_per_batch;
            if may_prepare_native_resources() {
                let status = ffi::apxinf_static_prepare_cutlass_mha_packed_qkv_f16(
                    gpu_ptr(packed)?,
                    output.ptr(),
                    batches as i32,
                    tokens_per_batch as i32,
                    heads as i32,
                    head_dim as i32,
                    ctx.stream().handle(),
                );
                if status != 0 {
                    return Err(Error::Cuda(format!(
                        "packed CUTLASS FMHA resource preparation failed with status {status}"
                    )));
                }
            }
            let status = ffi::apxinf_static_cutlass_mha_packed_qkv_f16(
                gpu_ptr(packed)?,
                output.ptr(),
                batches as i32,
                tokens_per_batch as i32,
                heads as i32,
                head_dim as i32,
                ctx.stream().handle(),
            );
            if status == 0 {
                return Ok(make_gpu_tensor(
                    Shape::new(vec![tokens, heads, head_dim]),
                    DType::F16,
                    ctx.device_id(),
                    output,
                ));
            }
        }
    }

    let split = split_qkv_bias_f16(ctx, qkv, bias, heads, head_dim)?;
    mha_f16(ctx, &split.q, &split.k, &split.v, tokens_per_batch)
}

pub fn mha_f16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    tokens_per_batch: usize,
) -> Result<Tensor> {
    let shape = q.shape().dims();
    if q.dtype() != DType::F16
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || shape.len() != 3
        || k.shape() != q.shape()
        || v.shape() != q.shape()
        || shape[2] > 256
        || tokens_per_batch == 0
        || shape[0] % tokens_per_batch != 0
    {
        return Err(Error::Other(
            "static inference MHA expects matching FP16 [tokens,heads,head_dim] tensors".into(),
        ));
    }
    let output = output_buffer(ctx, q.size_in_bytes())?;
    #[cfg(apxinf_cutlass_fmha)]
    if tokens_per_batch == 256 && shape[1] == 16 && shape[2] == 72 {
        unsafe {
            let batches = shape[0] / tokens_per_batch;
            if may_prepare_native_resources() {
                let status = ffi::apxinf_static_prepare_cutlass_mha_f16(
                    gpu_ptr(q)?,
                    gpu_ptr(k)?,
                    gpu_ptr(v)?,
                    output.ptr(),
                    batches as i32,
                    tokens_per_batch as i32,
                    tokens_per_batch as i32,
                    shape[1] as i32,
                    shape[1] as i32,
                    shape[2] as i32,
                    ctx.stream().handle(),
                );
                if status != 0 {
                    return Err(Error::Cuda(format!(
                        "CUTLASS FMHA resource preparation failed with status {status}"
                    )));
                }
            }
            let status = ffi::apxinf_static_cutlass_mha_f16(
                gpu_ptr(q)?,
                gpu_ptr(k)?,
                gpu_ptr(v)?,
                output.ptr(),
                batches as i32,
                tokens_per_batch as i32,
                shape[1] as i32,
                shape[1] as i32,
                shape[2] as i32,
                ctx.stream().handle(),
            );
            if status == 0 {
                return Ok(make_gpu_tensor(
                    Shape::new(vec![tokens, heads, head_dim]),
                    DType::F16,
                    ctx.device_id(),
                    output,
                ));
            }
        }
    }
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_mha_flash_f16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            tokens_per_batch as i32,
            (shape[0] / tokens_per_batch) as i32,
            shape[1] as i32,
            shape[2] as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

pub fn mqa_f16(ctx: &CudaContext, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if q.dtype() != DType::F16
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || q_shape.len() != 3
        || k_shape.len() != 3
        || v.shape().dims() != k_shape
        || k_shape[0] != q_shape[0]
        || k_shape[1] != 1
        || k_shape[2] != q_shape[2]
    {
        return Err(Error::Other(
            "static inference self MQA expects Q [T,H,D], K/V [T,1,D] FP16".into(),
        ));
    }
    #[cfg(apxinf_fa2_f16_sm100)]
    {
        return fa2_mqa_f16(ctx, q, k, v, k_shape[0]);
    }
    #[cfg(not(apxinf_fa2_f16_sm100))]
    cublas_mqa_f16(ctx, q, k, v, k_shape[0])
}

pub fn mqa_f16_e4m3_522(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output_scale: f32,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if q.dtype() != DType::F16
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || q_shape.len() != 3
        || k_shape.len() != 3
        || v.shape().dims() != k_shape
        || q_shape != [522, 8, 256]
        || k_shape != [522, 1, 256]
        || v.shape().dims() != k_shape
        || !output_scale.is_finite()
        || output_scale <= 0.0
    {
        return Err(Error::Other(format!(
            "FA2 direct E4M3 requires FP16 Q [522,8,256], K/V [522,1,256] and finite positive scale; got q={q_shape:?}, k={k_shape:?}, v={:?}, scale={output_scale}",
            v.shape().dims()
        )));
    }
    #[cfg(apxinf_fa2_direct_e4m3_sm100)]
    {
        let output = output_buffer(ctx, q.numel())?;
        let lse_elements = q_shape[0]
            .checked_mul(q_shape[1])
            .ok_or_else(|| Error::Other("FA2 direct E4M3 LSE size overflow".into()))?;
        let softmax_lse = output_buffer(
            ctx,
            lse_elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| Error::Other("FA2 direct E4M3 LSE byte size overflow".into()))?,
        )?;
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_fa2_f16_direct_e4m3_522(
                gpu_ptr(q)?,
                gpu_ptr(k)?,
                gpu_ptr(v)?,
                output.ptr(),
                softmax_lse.ptr(),
                1,
                522,
                522,
                8,
                1,
                256,
                output_scale,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        return Ok(make_gpu_tensor(
            q.shape().clone(),
            DType::F8E4M3,
            ctx.device_id(),
            output,
        ));
    }
    #[cfg(not(apxinf_fa2_direct_e4m3_sm100))]
    Err(Error::Other(
        "FA2 direct E4M3 requires an SM100-family FA2 build".into(),
    ))
}
