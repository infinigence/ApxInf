//! Rotary-position and QKV layout operator contracts.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

pub use super::attention::QkvTensors;
use super::contracts::{
    bf16_output, check_cuda, checked_bytes, f16_output, gpu_ptr, make_gpu_tensor, matrix_shape,
    matrix_tensor, optional_ptr, require_address, require_buffers, require_finite,
    unsupported_dtype,
};
use crate::buffer::{CudaBuffer, CudaDeviceAddress};
use crate::context::CudaContext;
use crate::ffi;

/// Device-resident cosine and sine tensors for half-split RoPE.
#[derive(Debug, Clone)]
pub struct RopeTables {
    pub cosine: Tensor,
    pub sine: Tensor,
}

/// Apply half-split RoPE using caller-provided BF16 cosine and sine tensors.
///
/// `input` is contiguous `[tokens, heads, head_dim]`; `cosine` and `sine` are
/// `[tokens, head_dim]`. Owning the metadata outside the kernel supports
/// arbitrary position schedules and scaling conventions while keeping the
/// device operation model-neutral and graph-capturable.
pub fn apply_precomputed_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    cosine: &Tensor,
    sine: &Tensor,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    if input.dtype() != DType::BF16
        || cosine.dtype() != DType::BF16
        || sine.dtype() != DType::BF16
        || dims.len() != 3
        || dims[0] == 0
        || dims[1] == 0
        || dims[2] == 0
        || dims[2] % 2 != 0
        || cosine.shape().dims() != [dims[0], dims[2]]
        || sine.shape() != cosine.shape()
        || input.device() != apxinf_core::Device::Cuda(ctx.device_id())
        || cosine.device() != input.device()
        || sine.device() != input.device()
    {
        return Err(Error::Other(
            "precomputed RoPE expects CUDA BF16 input [tokens,heads,head_dim] and cosine/sine [tokens,head_dim]"
                .into(),
        ));
    }
    let tokens = i32::try_from(dims[0])
        .map_err(|_| Error::Other("precomputed RoPE token count exceeds i32".into()))?;
    let heads = i32::try_from(dims[1])
        .map_err(|_| Error::Other("precomputed RoPE head count exceeds i32".into()))?;
    let head_dim = i32::try_from(dims[2])
        .map_err(|_| Error::Other("precomputed RoPE head dimension exceeds i32".into()))?;
    let output = bf16_output(ctx, dims[0] * dims[1], dims[2])?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_rope_precomputed_bf16(
            gpu_ptr(input)?,
            gpu_ptr(cosine)?,
            gpu_ptr(sine)?,
            output.ptr(),
            tokens,
            heads,
            head_dim,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// Build `[sin, cos]` sinusoidal embeddings for a BF16 scalar schedule.
///
/// Frequency construction and transcendental evaluation happen on the CUDA
/// device in FP32; only the final `[steps, dimension]` output is rounded to
/// BF16. This avoids substituting host-libm results for a CUDA reference.
pub fn sinusoidal_time_embedding_bf16(
    ctx: &CudaContext,
    times: &Tensor,
    dimension: usize,
    min_period: f32,
    max_period: f32,
) -> Result<Tensor> {
    require_finite("sinusoidal time embedding", &[min_period, max_period])?;
    let dims = times.shape().dims();
    if times.dtype() != DType::BF16
        || dims.len() != 1
        || dims[0] == 0
        || dimension == 0
        || dimension % 2 != 0
        || min_period <= 0.0
        || max_period < min_period
        || times.device() != apxinf_core::Device::Cuda(ctx.device_id())
    {
        return Err(Error::Other(
            "sinusoidal time embedding expects CUDA BF16 times [steps], positive ordered periods, and a non-zero even dimension"
                .into(),
        ));
    }
    let steps = i32::try_from(dims[0])
        .map_err(|_| Error::Other("time embedding step count exceeds i32".into()))?;
    let dimension_i32 = i32::try_from(dimension)
        .map_err(|_| Error::Other("time embedding dimension exceeds i32".into()))?;
    let output = bf16_output(ctx, dims[0], dimension)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_sinusoidal_time_embedding_bf16(
            gpu_ptr(times)?,
            output.ptr(),
            steps,
            dimension_i32,
            min_period,
            max_period,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, dims[0], dimension, output))
}

/// Build BF16 half-split RoPE cosine/sine tables from explicit positions.
///
/// `linear_factor` divides inverse frequency (`1.0` is ordinary RoPE). All
/// frequency, angle, and transcendental work runs in CUDA FP32 before one BF16
/// table store, avoiding CPU-libm substitution.
pub fn rope_tables_bf16(
    ctx: &CudaContext,
    positions: &CudaBuffer,
    tokens: usize,
    head_dim: usize,
    theta: f32,
    linear_factor: f32,
) -> Result<RopeTables> {
    require_finite("RoPE tables", &[theta, linear_factor])?;
    let positions_bytes = tokens
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| Error::Other("RoPE position byte size overflow".into()))?;
    if tokens == 0
        || head_dim == 0
        || head_dim % 2 != 0
        || theta <= 0.0
        || linear_factor <= 0.0
        || positions.device() != ctx.device_id()
        || positions.len() < positions_bytes
    {
        return Err(Error::Other(
            "RoPE tables require device u32 positions, positive theta/factor, and a non-zero even head dimension"
                .into(),
        ));
    }
    let tokens_i32 = i32::try_from(tokens)
        .map_err(|_| Error::Other("RoPE table token count exceeds i32".into()))?;
    let head_dim_i32 = i32::try_from(head_dim)
        .map_err(|_| Error::Other("RoPE table head dimension exceeds i32".into()))?;
    let cosine = bf16_output(ctx, tokens, head_dim)?;
    let sine = bf16_output(ctx, tokens, head_dim)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_rope_tables_bf16(
            positions.ptr(),
            cosine.ptr(),
            sine.ptr(),
            tokens_i32,
            head_dim_i32,
            theta,
            linear_factor,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(RopeTables {
        cosine: matrix_tensor(ctx, tokens, head_dim, cosine),
        sine: matrix_tensor(ctx, tokens, head_dim, sine),
    })
}

/// Apply RoPE into caller-owned storage using a device-resident position.
#[allow(clippy::too_many_arguments)]
pub fn apply_into(
    ctx: &CudaContext,
    dtype: DType,
    input: &CudaBuffer,
    output: &CudaBuffer,
    head_dim: usize,
    heads: usize,
    theta: f32,
    position: CudaDeviceAddress,
) -> Result<()> {
    require_finite("RoPE", &[theta])?;
    if head_dim % 2 != 0 || theta <= 0.0 {
        return Err(Error::Other(
            "RoPE requires an even head dimension and positive theta".into(),
        ));
    }
    let bytes = checked_bytes(dtype, &[heads, head_dim], "RoPE")?;
    require_buffers(
        ctx,
        "RoPE",
        &[("input", input, bytes), ("output", output, bytes)],
    )?;
    require_address(ctx, "RoPE", "position", position, 4)?;
    let status = unsafe {
        match dtype {
            DType::F32 => ffi::apxinf_rope_decode_f32(
                input.ptr(),
                output.ptr(),
                head_dim as u32,
                heads as u32,
                theta,
                position.ptr(),
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_rope_decode_bf16(
                input.ptr(),
                output.ptr(),
                head_dim as u32,
                heads as u32,
                theta,
                position.ptr(),
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(Error::Other(format!(
                    "decode RoPE does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}

/// Apply multimodal RoPE into caller-owned BF16 storage.
#[allow(clippy::too_many_arguments)]
pub fn apply_mrope_bf16_into(
    ctx: &CudaContext,
    input: &CudaBuffer,
    output: &CudaBuffer,
    head_dim: usize,
    heads: usize,
    theta: f32,
    positions: CudaDeviceAddress,
    section_h: usize,
    section_w: usize,
) -> Result<()> {
    require_finite("MRoPE", &[theta])?;
    if head_dim % 2 != 0 || section_h + section_w > head_dim / 2 || theta <= 0.0 {
        return Err(Error::Other(
            "MRoPE received invalid head/section dimensions or theta".into(),
        ));
    }
    let bytes = checked_bytes(DType::BF16, &[heads, head_dim], "MRoPE")?;
    require_buffers(
        ctx,
        "MRoPE",
        &[("input", input, bytes), ("output", output, bytes)],
    )?;
    require_address(ctx, "MRoPE", "positions", positions, 12)?;
    check_cuda(unsafe {
        ffi::apxinf_rope_mrope_decode_bf16(
            input.ptr(),
            output.ptr(),
            head_dim as u32,
            heads as u32,
            theta,
            positions.ptr(),
            section_h as u32,
            section_w as u32,
            ctx.stream().handle(),
        )
    })
}

/// Apply BF16 RoPE to K and write it directly to KV cache.
#[allow(clippy::too_many_arguments)]
pub fn apply_k_write_cache_bf16(
    ctx: &CudaContext,
    input: &CudaBuffer,
    cache: &CudaBuffer,
    head_dim: usize,
    kv_heads: usize,
    max_seq_len: usize,
    theta: f32,
    position: CudaDeviceAddress,
) -> Result<()> {
    require_finite("RoPE K write", &[theta])?;
    require_buffers(
        ctx,
        "RoPE K write",
        &[
            (
                "input",
                input,
                checked_bytes(DType::BF16, &[kv_heads, head_dim], "RoPE K write")?,
            ),
            (
                "cache",
                cache,
                checked_bytes(
                    DType::BF16,
                    &[kv_heads, max_seq_len, head_dim],
                    "RoPE K write",
                )?,
            ),
        ],
    )?;
    require_address(ctx, "RoPE K write", "position", position, 4)?;
    check_cuda(unsafe {
        ffi::apxinf_rope_k_write_bf16(
            input.ptr(),
            cache.ptr(),
            head_dim as u32,
            kv_heads as u32,
            max_seq_len as u32,
            theta,
            position.ptr(),
            ctx.stream().handle(),
        )
    })
}

/// Rotary Position Embedding (RoPE) on CUDA. Dispatches on dtype.
pub fn apply(
    ctx: &CudaContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    pos_offset: u32,
) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let seq_len = if dims.len() == 2 { 1 } else { dims[0] };

    let out_bytes = input.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    unsafe {
        let res = match input.dtype() {
            DType::F32 => ffi::apxinf_rope_f32(
                gpu_ptr(input)?,
                out_buf.ptr(),
                head_dim as u32,
                n_heads as u32,
                seq_len as u32,
                rope_theta,
                pos_offset,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_rope_bf16(
                gpu_ptr(input)?,
                out_buf.ptr(),
                head_dim as u32,
                n_heads as u32,
                seq_len as u32,
                rope_theta,
                pos_offset,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("rope", dtype),
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

/// Qwen3-VL multimodal RoPE (bf16, GPT-J rotate_half + interleaved axis
/// assignment across the three position axes {T,H,W}).
///
/// `input`  : `[seq_len, n_heads, head_dim]` bf16
/// `pos_ids`: `[seq_len, 3]` u32 on device (t, h, w per token)
/// `sections`: `[24, 20, 20]` for Qwen3-VL. Only `sec_h` and `sec_w` matter
///            at the kernel level — the T section is the leftover.
pub fn apply_mrope(
    ctx: &CudaContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    sections: [usize; 3],
    pos_ids: &CudaBuffer,
) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let seq_len = if dims.len() == 2 { 1 } else { dims[0] };
    let out_bytes = input.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    if input.dtype() != DType::BF16 {
        return Err(Error::Other("rope_mrope: only BF16 supported".into()));
    }

    unsafe {
        let res = ffi::apxinf_rope_mrope_bf16(
            gpu_ptr(input)?,
            out_buf.ptr(),
            head_dim as u32,
            n_heads as u32,
            seq_len as u32,
            theta,
            pos_ids.ptr(),
            sections[1] as u32,
            sections[2] as u32,
            ctx.stream().handle(),
        );
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }

    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::BF16,
        device_id,
        out_buf,
    ))
}

/// Vision 2D-RoPE (bf16). `input` `[seq, heads, head_dim]`; `pos_ids` flat
/// u32 slice of length `seq * 2` (h, w per token). head_dim=64 for Qwen3-VL.
pub fn apply_vision_2d(
    ctx: &CudaContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    pos_ids: &CudaBuffer,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 {
        return Err(Error::Other("rope_vision_2d: only BF16 supported".into()));
    }
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let seq_len = if dims.len() == 2 { 1 } else { dims[0] };
    let out_buf = CudaBuffer::alloc_zeros(input.size_in_bytes(), device_id).map_err(Error::Cuda)?;
    unsafe {
        let res = ffi::apxinf_rope_vision_2d_bf16(
            gpu_ptr(input)?,
            out_buf.ptr(),
            head_dim as u32,
            n_heads as u32,
            seq_len as u32,
            theta,
            pos_ids.ptr(),
            ctx.stream().handle(),
        );
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::BF16,
        device_id,
        out_buf,
    ))
}

/// Batched RoPE with half-split pairs. Dispatches on dtype.
pub fn apply_batched(
    ctx: &CudaContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    pos_offset: u32,
) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let seq_len = if dims.len() == 2 { 1 } else { dims[0] };

    let out_bytes = input.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    unsafe {
        let res = match input.dtype() {
            DType::F32 => ffi::apxinf_rope_batched_f32(
                gpu_ptr(input)?,
                out_buf.ptr(),
                head_dim as u32,
                n_heads as u32,
                seq_len as u32,
                rope_theta,
                pos_offset,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_rope_batched_bf16(
                gpu_ptr(input)?,
                out_buf.ptr(),
                head_dim as u32,
                n_heads as u32,
                seq_len as u32,
                rope_theta,
                pos_offset,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("rope_batched", dtype),
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
#[allow(clippy::too_many_arguments)]
fn qkv_rope_impl(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    caches: Option<(&Tensor, &Tensor, usize)>,
) -> Result<QkvTensors> {
    let (tokens, width) = matrix_shape(qkv, "QKV RoPE")?;
    let expected = (q_heads + 2 * kv_heads) * head_dim;
    if qkv.dtype() != DType::BF16
        || width != expected
        || head_dim > 256
        || head_dim % 2 != 0
        || bias
            .is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [expected])
    {
        return Err(Error::Other(
            "static inference BF16 QKV RoPE shape mismatch".into(),
        ));
    }
    let q_buffer = bf16_output(ctx, tokens * q_heads, head_dim)?;
    let owned_k = caches
        .is_none()
        .then(|| bf16_output(ctx, tokens * kv_heads, head_dim))
        .transpose()?;
    let owned_v = caches
        .is_none()
        .then(|| bf16_output(ctx, tokens * kv_heads, head_dim))
        .transpose()?;
    let (k_ptr, v_ptr, output_offset) = if let Some((k, v, offset)) = caches {
        let cache_shape = k.shape().dims();
        if k.dtype() != DType::BF16
            || v.dtype() != DType::BF16
            || cache_shape.len() != 2
            || v.shape().dims() != cache_shape
            || cache_shape[1] != head_dim
            || kv_heads != 1
            || offset + tokens > cache_shape[0]
        {
            return Err(Error::Other(
                "static inference BF16 cached QKV shape mismatch".into(),
            ));
        }
        (gpu_ptr(k)?, gpu_ptr(v)?, offset)
    } else {
        (
            owned_k.as_ref().unwrap().ptr(),
            owned_v.as_ref().unwrap().ptr(),
            0,
        )
    };
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qkv_rope_bf16(
            gpu_ptr(qkv)?,
            optional_ptr(bias)?,
            q_buffer.ptr(),
            k_ptr,
            v_ptr,
            tokens as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            theta,
            position_offset as i32,
            output_offset as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    let q = make_gpu_tensor(
        Shape::new(vec![tokens, q_heads, head_dim]),
        DType::BF16,
        ctx.device_id(),
        q_buffer,
    );
    if let (Some(k), Some(v)) = (owned_k, owned_v) {
        Ok(QkvTensors {
            q,
            k: make_gpu_tensor(
                Shape::new(vec![tokens, kv_heads, head_dim]),
                DType::BF16,
                ctx.device_id(),
                k,
            ),
            v: make_gpu_tensor(
                Shape::new(vec![tokens, kv_heads, head_dim]),
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

#[allow(clippy::too_many_arguments)]
pub fn split_qkv_apply_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<QkvTensors> {
    qkv_rope_impl(
        ctx,
        qkv,
        bias,
        q_heads,
        kv_heads,
        head_dim,
        theta,
        position_offset,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn apply_q_write_kv_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    k_cache: &Tensor,
    v_cache: &Tensor,
    output_offset: usize,
) -> Result<Tensor> {
    Ok(qkv_rope_impl(
        ctx,
        qkv,
        bias,
        q_heads,
        kv_heads,
        head_dim,
        theta,
        position_offset,
        Some((k_cache, v_cache, output_offset)),
    )?
    .q)
}
pub fn split_qkv_apply_f16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<QkvTensors> {
    let (tokens, width) = matrix_shape(qkv, "QKV RoPE")?;
    let expected = (q_heads + 2 * kv_heads) * head_dim;
    if qkv.dtype() != DType::F16
        || width != expected
        || head_dim > 256
        || head_dim % 2 != 0
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [expected])
    {
        return Err(Error::Other(format!(
            "static inference QKV RoPE expected FP16 [tokens,{expected}], got {:?}",
            qkv.shape().dims()
        )));
    }
    let q_buffer = f16_output(ctx, tokens * q_heads, head_dim)?;
    let k_buffer = f16_output(ctx, tokens * kv_heads, head_dim)?;
    let v_buffer = f16_output(ctx, tokens * kv_heads, head_dim)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qkv_rope_f16(
            gpu_ptr(qkv)?,
            bias.map(gpu_ptr)
                .transpose()?
                .unwrap_or(std::ptr::null_mut()),
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            tokens as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            theta,
            position_offset as i32,
            0,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(QkvTensors {
        q: make_gpu_tensor(
            Shape::new(vec![tokens, q_heads, head_dim]),
            DType::F16,
            ctx.device_id(),
            q_buffer,
        ),
        k: make_gpu_tensor(
            Shape::new(vec![tokens, kv_heads, head_dim]),
            DType::F16,
            ctx.device_id(),
            k_buffer,
        ),
        v: make_gpu_tensor(
            Shape::new(vec![tokens, kv_heads, head_dim]),
            DType::F16,
            ctx.device_id(),
            v_buffer,
        ),
    })
}

/// Apply QKV bias/RoPE while writing suffix K/V directly into persistent caches.
pub fn apply_q_write_kv_f16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    k_cache: &Tensor,
    v_cache: &Tensor,
    kv_output_offset: usize,
) -> Result<Tensor> {
    let (tokens, width) = matrix_shape(qkv, "cached QKV RoPE")?;
    let expected = (q_heads + 2 * kv_heads) * head_dim;
    let cache_shape = k_cache.shape().dims();
    if qkv.dtype() != DType::F16
        || width != expected
        || head_dim > 256
        || head_dim % 2 != 0
        || kv_heads != 1
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [expected])
        || k_cache.dtype() != DType::F16
        || v_cache.dtype() != DType::F16
        || cache_shape.len() != 2
        || v_cache.shape().dims() != cache_shape
        || cache_shape[1] != head_dim
        || kv_output_offset + tokens > cache_shape[0]
    {
        return Err(Error::Other(format!(
            "static inference cached QKV RoPE shape mismatch: qkv={:?}, k_cache={cache_shape:?}, v_cache={:?}",
            qkv.shape().dims(),
            v_cache.shape().dims()
        )));
    }
    let q_buffer = f16_output(ctx, tokens * q_heads, head_dim)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qkv_rope_f16(
            gpu_ptr(qkv)?,
            bias.map(gpu_ptr)
                .transpose()?
                .unwrap_or(std::ptr::null_mut()),
            q_buffer.ptr(),
            gpu_ptr(k_cache)?,
            gpu_ptr(v_cache)?,
            tokens as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            theta,
            position_offset as i32,
            kv_output_offset as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![tokens, q_heads, head_dim]),
        DType::F16,
        ctx.device_id(),
        q_buffer,
    ))
}
