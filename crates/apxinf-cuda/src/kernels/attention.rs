//! Model-neutral attention contracts and workspace orchestration.

use std::sync::OnceLock;

use apxinf_core::{DType, Device, Error, KvCache, Result, Shape, Tensor};

use super::contracts::{
    check_cuda, checked_bytes, f16_output, gpu_ptr, make_gpu_tensor, matrix_shape, optional_ptr,
    require_address, require_buffers, require_finite, unsupported_dtype,
};
use super::elementwise::{bias_f16, concat_rows_f16};
use crate::buffer::{CudaBuffer, CudaDeviceAddress};
use crate::context::CudaContext;
use crate::cublas::CublasTranspose;
use crate::ffi;
use crate::workspace::{may_prepare_native_resources, output_buffer, uninitialized_buffer};
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

fn strided_batch_elements(batch: usize, stride: usize, matrix: usize) -> Result<usize> {
    batch
        .saturating_sub(1)
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(matrix))
        .ok_or_else(|| Error::Other("strided GQA batch span overflow".into()))
}

fn element_stride(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::Other("strided GQA stride exceeds i64".into()))
}

fn batch_count(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Other("strided GQA batch exceeds i32".into()))
}

#[allow(clippy::too_many_arguments)]
fn copy_buffer_2d(
    ctx: &CudaContext,
    source: &CudaBuffer,
    source_offset: usize,
    source_pitch: usize,
    destination: &CudaBuffer,
    destination_offset: usize,
    destination_pitch: usize,
    width: usize,
    rows: usize,
) -> Result<()> {
    if rows == 0 || width == 0 {
        return Ok(());
    }
    if source.device() != ctx.device_id() || destination.device() != ctx.device_id() {
        return Err(Error::Other("2D GQA copy crosses CUDA devices".into()));
    }
    if width > source_pitch || width > destination_pitch {
        return Err(Error::Other("2D GQA copy width exceeds pitch".into()));
    }
    let source_end = source_offset
        .checked_add(
            rows.saturating_sub(1)
                .checked_mul(source_pitch)
                .and_then(|offset| offset.checked_add(width))
                .ok_or_else(|| Error::Other("2D GQA source span overflow".into()))?,
        )
        .ok_or_else(|| Error::Other("2D GQA source offset overflow".into()))?;
    let destination_end = destination_offset
        .checked_add(
            rows.saturating_sub(1)
                .checked_mul(destination_pitch)
                .and_then(|offset| offset.checked_add(width))
                .ok_or_else(|| Error::Other("2D GQA destination span overflow".into()))?,
        )
        .ok_or_else(|| Error::Other("2D GQA destination offset overflow".into()))?;
    if source_end > source.len() || destination_end > destination.len() {
        return Err(Error::Other("2D GQA copy exceeds allocation".into()));
    }
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpy2DAsync(
            destination.ptr().cast::<u8>().add(destination_offset).cast(),
            destination_pitch,
            source.ptr().cast::<u8>().add(source_offset).cast(),
            source_pitch,
            width,
            rows,
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_gqa_rows(
    ctx: &CudaContext,
    source: &CudaBuffer,
    destination: &CudaBuffer,
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    gqa_ratio: usize,
    row_width: usize,
    element_bytes: usize,
) -> Result<()> {
    let group_bytes = gqa_ratio
        .checked_mul(row_width)
        .and_then(|value| value.checked_mul(element_bytes))
        .ok_or_else(|| Error::Other("packed GQA row width overflow".into()))?;
    let row_bytes = n_heads
        .checked_mul(row_width)
        .and_then(|value| value.checked_mul(element_bytes))
        .ok_or_else(|| Error::Other("GQA row pitch overflow".into()))?;
    for kv_head in 0..n_kv_heads {
        copy_buffer_2d(
            ctx,
            source,
            kv_head * group_bytes,
            row_bytes,
            destination,
            kv_head * seq_len * group_bytes,
            group_bytes,
            group_bytes,
            seq_len,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn unpack_gqa_rows(
    ctx: &CudaContext,
    source: &CudaBuffer,
    destination: &CudaBuffer,
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    gqa_ratio: usize,
    row_width: usize,
    element_bytes: usize,
) -> Result<()> {
    let group_bytes = gqa_ratio
        .checked_mul(row_width)
        .and_then(|value| value.checked_mul(element_bytes))
        .ok_or_else(|| Error::Other("packed GQA row width overflow".into()))?;
    let row_bytes = n_heads
        .checked_mul(row_width)
        .and_then(|value| value.checked_mul(element_bytes))
        .ok_or_else(|| Error::Other("GQA row pitch overflow".into()))?;
    for kv_head in 0..n_kv_heads {
        copy_buffer_2d(
            ctx,
            source,
            kv_head * seq_len * group_bytes,
            group_bytes,
            destination,
            kv_head * group_bytes,
            row_bytes,
            group_bytes,
            seq_len,
        )?;
    }
    Ok(())
}

fn batched_gqa_prefill_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| match std::env::var("APXINF_BATCHED_GQA_PREFILL") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(value) if value == "0" => Ok(false),
            Ok(value) if value == "1" => Ok(true),
            Ok(value) => Err(format!(
                "APXINF_BATCHED_GQA_PREFILL must be 0 or 1, got `{value}`"
            )),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("APXINF_BATCHED_GQA_PREFILL must be UTF-8".into())
            }
        })
        .clone()
        .map_err(Error::Other)
}

fn causal_fa2_gqa_prefill_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| match std::env::var("APXINF_FA2_GQA_PREFILL") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(value) if value == "0" => Ok(false),
            Ok(value) if value == "1" => Ok(true),
            Ok(value) => Err(format!(
                "APXINF_FA2_GQA_PREFILL must be 0 or 1, got `{value}`"
            )),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("APXINF_FA2_GQA_PREFILL must be UTF-8".into())
            }
        })
        .clone()
        .map_err(Error::Other)
}

const MAX_SOFTMAX_EXP_CACHE_COLS: usize = 11_264;
const MAX_PREFILL_SOFTMAX_EXP_CACHE_COLS: usize = 4_096;
const MAX_INPLACE_SCALE_QUERY_TOKENS: usize = 256;
const MIN_CAUSAL_FA2_GQA_KV_LEN: usize = 4_097;

fn softmax_exp_cache_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| match std::env::var("APXINF_SOFTMAX_EXP_CACHE") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(value) if value == "0" => Ok(false),
            Ok(value) if value == "1" => Ok(true),
            Ok(value) => Err(format!(
                "APXINF_SOFTMAX_EXP_CACHE must be 0 or 1, got `{value}`"
            )),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("APXINF_SOFTMAX_EXP_CACHE must be UTF-8".into())
            }
        })
        .clone()
        .map_err(Error::Other)
}

fn softmax_exp_cache_long_fallback_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(
            || match std::env::var("APXINF_SOFTMAX_EXP_CACHE_LONG_FALLBACK") {
                Err(std::env::VarError::NotPresent) => Ok(false),
                Ok(value) if value == "0" => Ok(false),
                Ok(value) if value == "1" => Ok(true),
                Ok(value) => Err(format!(
                    "APXINF_SOFTMAX_EXP_CACHE_LONG_FALLBACK must be 0 or 1, got `{value}`"
                )),
                Err(std::env::VarError::NotUnicode(_)) => {
                    Err("APXINF_SOFTMAX_EXP_CACHE_LONG_FALLBACK must be UTF-8".into())
                }
            },
        )
        .clone()
        .map_err(Error::Other)
}

fn softmax_global_exp_cache_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| match std::env::var("APXINF_SOFTMAX_GLOBAL_EXP_CACHE") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(value) if value == "0" => Ok(false),
            Ok(value) if value == "1" => Ok(true),
            Ok(value) => Err(format!(
                "APXINF_SOFTMAX_GLOBAL_EXP_CACHE must be 0 or 1, got `{value}`"
            )),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("APXINF_SOFTMAX_GLOBAL_EXP_CACHE must be UTF-8".into())
            }
        })
        .clone()
        .map_err(Error::Other)
}

fn softmax_inplace_scale_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| match std::env::var("APXINF_SOFTMAX_INPLACE_SCALE") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(value) if value == "0" => Ok(false),
            Ok(value) if value == "1" => Ok(true),
            Ok(value) => Err(format!(
                "APXINF_SOFTMAX_INPLACE_SCALE must be 0 or 1, got `{value}`"
            )),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("APXINF_SOFTMAX_INPLACE_SCALE must be UTF-8".into())
            }
        })
        .clone()
        .map_err(Error::Other)
}

fn vision_full_fa2_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| match std::env::var("APXINF_VISION_FULL_FA2") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(value) if value == "0" => Ok(false),
            Ok(value) if value == "1" => Ok(true),
            Ok(value) => Err(format!(
                "APXINF_VISION_FULL_FA2 must be 0 or 1, got `{value}`"
            )),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("APXINF_VISION_FULL_FA2 must be UTF-8".into())
            }
        })
        .clone()
        .map_err(Error::Other)
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

#[allow(clippy::too_many_arguments)]
fn gqa_scores_batched(
    ctx: &CudaContext,
    dtype: DType,
    query: &Tensor,
    query_offset: usize,
    query_stride: usize,
    key_cache: &CudaBuffer,
    key_offset: usize,
    scores: &CudaBuffer,
    scores_offset: usize,
    scores_stride: usize,
    batch: usize,
    gqa_ratio: usize,
    kv_len: usize,
    head_dim: usize,
) -> Result<()> {
    let element_bytes = dtype.size_in_bytes();
    let query_elements = strided_batch_elements(batch, query_stride, gqa_ratio * head_dim)?;
    let score_elements = strided_batch_elements(batch, scores_stride, gqa_ratio * kv_len)?;
    let query = tensor_slice(
        query,
        query_offset * element_bytes,
        query_elements * element_bytes,
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
        score_elements * element_bytes,
    )?;
    ctx.cublas()
        .batched_gemm_ex(
            dtype,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            gqa_ratio,
            kv_len,
            head_dim,
            1.0,
            &query,
            head_dim as i32,
            element_stride(query_stride)?,
            &key,
            head_dim as i32,
            0,
            0.0,
            &output,
            kv_len as i32,
            element_stride(scores_stride)?,
            batch_count(batch)?,
        )
        .map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
fn gqa_values_batched(
    ctx: &CudaContext,
    dtype: DType,
    attention: &Tensor,
    attention_offset: usize,
    attention_stride: usize,
    value_cache: &CudaBuffer,
    value_offset: usize,
    output: &CudaBuffer,
    output_offset: usize,
    output_stride: usize,
    batch: usize,
    gqa_ratio: usize,
    kv_len: usize,
    head_dim: usize,
) -> Result<()> {
    let element_bytes = dtype.size_in_bytes();
    let attention_elements =
        strided_batch_elements(batch, attention_stride, gqa_ratio * kv_len)?;
    let output_elements = strided_batch_elements(batch, output_stride, gqa_ratio * head_dim)?;
    let attention = tensor_slice(
        attention,
        attention_offset * element_bytes,
        attention_elements * element_bytes,
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
        output_elements * element_bytes,
    )?;
    ctx.cublas()
        .batched_gemm_ex(
            dtype,
            CublasTranspose::None,
            CublasTranspose::None,
            gqa_ratio,
            head_dim,
            kv_len,
            1.0,
            &attention,
            kv_len as i32,
            element_stride(attention_stride)?,
            &value,
            head_dim as i32,
            0,
            0.0,
            &output,
            head_dim as i32,
            element_stride(output_stride)?,
            batch_count(batch)?,
        )
        .map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
fn gqa_scores_flattened(
    ctx: &CudaContext,
    dtype: DType,
    query: &CudaBuffer,
    query_offset: usize,
    key_cache: &CudaBuffer,
    key_offset: usize,
    scores: &CudaBuffer,
    scores_offset: usize,
    batch: usize,
    gqa_ratio: usize,
    kv_len: usize,
    head_dim: usize,
) -> Result<()> {
    let element_bytes = dtype.size_in_bytes();
    let query = buffer_slice(
        query,
        query_offset * element_bytes,
        batch * gqa_ratio * head_dim * element_bytes,
    )?;
    let key = buffer_slice(
        key_cache,
        key_offset * element_bytes,
        kv_len * head_dim * element_bytes,
    )?;
    let output = buffer_slice(
        scores,
        scores_offset * element_bytes,
        batch * gqa_ratio * kv_len * element_bytes,
    )?;
    ctx.cublas()
        .gemm_ex(
            dtype,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            batch * gqa_ratio,
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
fn gqa_values_flattened(
    ctx: &CudaContext,
    dtype: DType,
    attention: &Tensor,
    attention_offset: usize,
    value_cache: &CudaBuffer,
    value_offset: usize,
    output: &CudaBuffer,
    output_offset: usize,
    batch: usize,
    gqa_ratio: usize,
    kv_len: usize,
    head_dim: usize,
) -> Result<()> {
    let element_bytes = dtype.size_in_bytes();
    let attention = tensor_slice(
        attention,
        attention_offset * element_bytes,
        batch * gqa_ratio * kv_len * element_bytes,
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
        batch * gqa_ratio * head_dim * element_bytes,
    )?;
    ctx.cublas()
        .gemm_ex(
            dtype,
            CublasTranspose::None,
            CublasTranspose::None,
            batch * gqa_ratio,
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

#[cfg(apxinf_fa2_causal_sm80)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn causal_fa2_gqa_prefill(
    ctx: &CudaContext,
    query: &Tensor,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    query_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    key_tokens: usize,
    max_seq_len: usize,
) -> Result<Tensor> {
    let query_dims = query.shape().dims();
    if query.device() != Device::Cuda(ctx.device_id())
        || query.dtype() != DType::BF16
        || query_dims != [query_tokens, query_heads, head_dim]
        || query_tokens <= 1
        || query_tokens > key_tokens
        || key_tokens == 0
        || key_tokens > max_seq_len
        || query_heads != 16
        || kv_heads != 2
        || head_dim != 128
    {
        return Err(Error::Other(format!(
            "causal GQA FA2 supports only CUDA BF16 suffix prefill [q,16,128], kv_heads=2 and kv_len 1..=max; got {:?}, q={query_tokens}, kv={key_tokens}/{max_seq_len}, heads={query_heads}/{kv_heads}, dim={head_dim}",
            query_dims
        )));
    }
    if key_cache.device() != ctx.device_id() || value_cache.device() != ctx.device_id() {
        return Err(Error::Other(
            "causal GQA FA2 cache crosses CUDA devices".into(),
        ));
    }
    let cache_bytes = kv_heads
        .checked_mul(max_seq_len)
        .and_then(|value| value.checked_mul(head_dim))
        .and_then(|value| value.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("causal GQA FA2 cache size overflow".into()))?;
    if key_cache.len() < cache_bytes || value_cache.len() < cache_bytes {
        return Err(Error::Other(format!(
            "causal GQA FA2 cache is too small: need {cache_bytes}, got {}/{}",
            key_cache.len(),
            value_cache.len()
        )));
    }

    let output = uninitialized_buffer(ctx, query.size_in_bytes())?;
    let lse_bytes = query_tokens
        .checked_mul(query_heads)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| Error::Other("causal GQA FA2 LSE size overflow".into()))?;
    let softmax_lse = uninitialized_buffer(ctx, lse_bytes)?;
    let dimension = |name: &str, value: usize| {
        i32::try_from(value)
            .map_err(|_| Error::Other(format!("causal GQA FA2 {name} exceeds i32")))
    };
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_fa2_causal_strided_kv_bf16(
            gpu_ptr(query)?,
            key_cache.ptr(),
            value_cache.ptr(),
            output.ptr(),
            softmax_lse.ptr(),
            dimension("query tokens", query_tokens)?,
            dimension("key tokens", key_tokens)?,
            dimension("query heads", query_heads)?,
            dimension("KV heads", kv_heads)?,
            dimension("head dimension", head_dim)?,
            dimension("maximum sequence length", max_seq_len)?,
            (head_dim as f32).sqrt().recip(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    static PATH_LOGGED: OnceLock<()> = OnceLock::new();
    if PATH_LOGGED.set(()).is_ok() {
        eprintln!(
            "ApxInf causal GQA FA2 long prefill: query={query_tokens}, kv={key_tokens}, heads={query_heads}/{kv_heads}, dim={head_dim}"
        );
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![query_tokens, query_heads * head_dim]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
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
    sdpa_with_batched_prefill(
        ctx,
        query,
        kv,
        layer_idx,
        n_heads,
        n_kv_heads,
        head_dim,
        kv_len,
        max_seq_len,
        kv_offset,
        batched_gqa_prefill_enabled()?,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_with_batched_prefill(
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
    use_batched_prefill: bool,
) -> Result<Tensor> {
    sdpa_with_batched_prefill_fa2_min_kv(
        ctx,
        query,
        kv,
        layer_idx,
        n_heads,
        n_kv_heads,
        head_dim,
        kv_len,
        max_seq_len,
        kv_offset,
        use_batched_prefill,
        MIN_CAUSAL_FA2_GQA_KV_LEN,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa_with_batched_prefill_fa2_min_kv(
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
    use_batched_prefill: bool,
    causal_fa2_min_kv_len: usize,
    use_scaled_exp_cache: bool,
) -> Result<Tensor> {
    if causal_fa2_min_kv_len == 0 {
        return Err(Error::Other(
            "causal GQA FA2 minimum KV length must be positive".into(),
        ));
    }
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
    if causal_fa2_gqa_prefill_enabled()?
        && seq_len > 1
        && kv_len >= causal_fa2_min_kv_len
    {
        if !use_batched_prefill {
            return Err(Error::Other(
                "causal GQA FA2 long prefill requires APXINF_BATCHED_GQA_PREFILL=1".into(),
            ));
        }
        if usize::try_from(kv_offset)
            .ok()
            .and_then(|offset| offset.checked_add(seq_len))
            != Some(kv_len)
        {
            return Err(Error::Other(format!(
                "causal GQA FA2 requires suffix queries: offset {kv_offset} + query {seq_len} != kv {kv_len}"
            )));
        }
        #[cfg(apxinf_fa2_causal_sm80)]
        {
            return causal_fa2_gqa_prefill(
                ctx,
                query,
                cache.k_buffer(layer_idx),
                cache.v_buffer(layer_idx),
                seq_len,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_len,
                max_seq_len,
            );
        }
        #[cfg(not(apxinf_fa2_causal_sm80))]
        {
            return Err(Error::Other(
                "causal GQA FA2 long prefill is unavailable for this CUDA target".into(),
            ));
        }
    }
    let use_flattened_prefill = use_batched_prefill
        && dtype == DType::BF16
        && seq_len > 1
        && kv_len > MAX_PREFILL_SOFTMAX_EXP_CACHE_COLS;
    if use_scaled_exp_cache
        && (dtype != DType::BF16
            || seq_len <= 1
            || kv_len > MAX_PREFILL_SOFTMAX_EXP_CACHE_COLS
            || use_flattened_prefill)
    {
        return Err(Error::Other(
            "scaled exp-cache prefill requires non-flattened BF16 with 2..=4096 KV tokens".into(),
        ));
    }
    let packed_query = if use_flattened_prefill {
        let packed = uninitialized_buffer(ctx, query.size_in_bytes())?;
        let source = CudaBuffer::from_tensor(query).map_err(Error::Cuda)?;
        pack_gqa_rows(
            ctx,
            &source,
            &packed,
            seq_len,
            n_heads,
            n_kv_heads,
            gqa_ratio,
            head_dim,
            element_bytes,
        )?;
        Some(packed)
    } else {
        None
    };
    let scores = uninitialized_buffer(ctx, seq_len * n_heads * kv_len * element_bytes)?;
    let key_cache = cache.k_buffer(layer_idx);

    for kv_head in 0..n_kv_heads {
        if let Some(packed_query) = &packed_query {
            gqa_scores_flattened(
                ctx,
                dtype,
                packed_query,
                kv_head * seq_len * gqa_ratio * head_dim,
                key_cache,
                kv_head * max_seq_len * head_dim,
                &scores,
                kv_head * seq_len * gqa_ratio * kv_len,
                seq_len,
                gqa_ratio,
                kv_len,
                head_dim,
            )?;
        } else if use_batched_prefill && seq_len > 1 {
            gqa_scores_batched(
                ctx,
                dtype,
                query,
                kv_head * gqa_ratio * head_dim,
                n_heads * head_dim,
                key_cache,
                kv_head * max_seq_len * head_dim,
                &scores,
                kv_head * gqa_ratio * kv_len,
                n_heads * kv_len,
                seq_len,
                gqa_ratio,
                kv_len,
                head_dim,
            )?;
        } else {
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
    }

    let scores = scores.into_tensor(Shape::new(vec![seq_len * n_heads, kv_len]), dtype);
    let score_scale = 1.0 / (head_dim as f32).sqrt();
    let use_inplace_softmax = use_flattened_prefill
        && seq_len <= MAX_INPLACE_SCALE_QUERY_TOKENS
        && softmax_inplace_scale_enabled()?;
    let attention = if use_scaled_exp_cache {
        softmax_causal_bf16_scaled_exp_cache(ctx, &scores, kv_offset, n_heads as u32, score_scale)?
    } else if use_inplace_softmax {
        softmax_causal_bf16_scaled_in_place_gqa_packed(
            ctx,
            scores,
            kv_offset,
            n_heads as u32,
            gqa_ratio as u32,
            score_scale,
        )?
    } else if use_flattened_prefill {
        softmax_causal_bf16_scaled_plain_gqa_packed(
            ctx,
            &scores,
            kv_offset,
            n_heads as u32,
            gqa_ratio as u32,
            score_scale,
        )?
    } else if dtype == DType::BF16
        && seq_len > 1
        && kv_len > MAX_PREFILL_SOFTMAX_EXP_CACHE_COLS
    {
        softmax_causal_bf16_scaled_plain(
            ctx,
            &scores,
            kv_offset,
            n_heads as u32,
            score_scale,
        )?
    } else {
        let scores = super::elementwise::scale(ctx, &scores, score_scale)?;
        softmax_causal(ctx, &scores, kv_offset, n_heads as u32)?
    };

    let output = uninitialized_buffer(ctx, seq_len * n_heads * head_dim * element_bytes)?;
    let packed_output = if use_flattened_prefill {
        Some(uninitialized_buffer(
            ctx,
            seq_len * n_heads * head_dim * element_bytes,
        )?)
    } else {
        None
    };
    let value_cache = cache.v_buffer(layer_idx);
    for kv_head in 0..n_kv_heads {
        if let Some(packed_output) = &packed_output {
            gqa_values_flattened(
                ctx,
                dtype,
                &attention,
                kv_head * seq_len * gqa_ratio * kv_len,
                value_cache,
                kv_head * max_seq_len * head_dim,
                packed_output,
                kv_head * seq_len * gqa_ratio * head_dim,
                seq_len,
                gqa_ratio,
                kv_len,
                head_dim,
            )?;
        } else if use_batched_prefill && seq_len > 1 {
            gqa_values_batched(
                ctx,
                dtype,
                &attention,
                kv_head * gqa_ratio * kv_len,
                n_heads * kv_len,
                value_cache,
                kv_head * max_seq_len * head_dim,
                &output,
                kv_head * gqa_ratio * head_dim,
                n_heads * head_dim,
                seq_len,
                gqa_ratio,
                kv_len,
                head_dim,
            )?;
        } else {
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
    }

    if let Some(packed_output) = &packed_output {
        unpack_gqa_rows(
            ctx,
            packed_output,
            &output,
            seq_len,
            n_heads,
            n_kv_heads,
            gqa_ratio,
            head_dim,
            element_bytes,
        )?;
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
    let out_buf = uninitialized_buffer(ctx, out_bytes)?;

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
/// Supports head dimensions through 128, including Qwen2.5-Omni's 80.
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
    if head_dim == 0 || head_dim > 128 {
        return Err(Error::Other(
            "vision_sdpa: head_dim must be in 1..=128".into(),
        ));
    }
    let requested_fa2 = head_dim == 80 && vision_full_fa2_enabled()?;
    #[cfg(apxinf_fa2_sm80)]
    {
        if head_dim != 80 || requested_fa2 {
            return fa2_attention(
                ctx, q, k, v, 1, seq_len, seq_len, n_heads, n_heads, head_dim,
            )?
            .reshape(vec![seq_len, n_heads * head_dim]);
        }
    }
    #[cfg(all(not(apxinf_fa2_sm80), apxinf_fa2_vision_sm80))]
    {
        if head_dim <= 96 && (head_dim != 80 || requested_fa2) {
            return fa2_attention(
                ctx, q, k, v, 1, seq_len, seq_len, n_heads, n_heads, head_dim,
            )?
            .reshape(vec![seq_len, n_heads * head_dim]);
        }
    }
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_vision_sm80)))]
    if requested_fa2 {
        return Err(Error::Other(
            "APXINF_VISION_FULL_FA2 requires an SM80-family FA2 build".into(),
        ));
    }
    let device_id = ctx.device_id();
    let out_bytes = seq_len * n_heads * head_dim * DType::BF16.size_in_bytes();
    let out_buf = output_buffer(ctx, out_bytes)?;
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

/// Non-causal BF16 attention limited to equal group identifiers.
pub fn grouped(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    group_ids: &CudaBuffer,
) -> Result<Tensor> {
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        return Err(Error::Other("grouped_sdpa: only BF16 supported".into()));
    }
    if head_dim == 0 || head_dim > 128 || group_ids.len() != seq_len * 4 {
        return Err(Error::Other("grouped_sdpa: invalid head dimension or group ids".into()));
    }
    let device_id = ctx.device_id();
    let out_bytes = seq_len * n_heads * head_dim * DType::BF16.size_in_bytes();
    let out_buf = output_buffer(ctx, out_bytes)?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    unsafe {
        ffi::check_cuda(ffi::apxinf_grouped_sdpa_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            out_buf.ptr(),
            seq_len as u32,
            n_heads as u32,
            head_dim as u32,
            scale,
            group_ids.ptr(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![seq_len, n_heads * head_dim]),
        DType::BF16,
        device_id,
        out_buf,
    ))
}

/// Grouped BF16 vision attention over ascending original key-index lists.
/// The plan buffers are immutable request metadata cached by the CUDA backend.
pub fn grouped_indexed(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    group_ids: &CudaBuffer,
    group_offsets: &CudaBuffer,
    group_indices: &CudaBuffer,
    group_count: usize,
) -> Result<Tensor> {
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        return Err(Error::Other(
            "indexed grouped_sdpa: only BF16 supported".into(),
        ));
    }
    if head_dim == 0
        || head_dim > 128
        || group_count == 0
        || group_ids.len() != seq_len * std::mem::size_of::<u32>()
        || group_offsets.len() != (group_count + 1) * std::mem::size_of::<u32>()
        || group_indices.len() != seq_len * std::mem::size_of::<u32>()
    {
        return Err(Error::Other(
            "indexed grouped_sdpa: invalid shape or group plan".into(),
        ));
    }
    let seq_len = u32::try_from(seq_len)
        .map_err(|_| Error::Other("indexed grouped_sdpa sequence exceeds u32".into()))?;
    let n_heads = u32::try_from(n_heads)
        .map_err(|_| Error::Other("indexed grouped_sdpa heads exceed u32".into()))?;
    let head_dim = u32::try_from(head_dim)
        .map_err(|_| Error::Other("indexed grouped_sdpa head dimension exceeds u32".into()))?;
    let group_count = u32::try_from(group_count)
        .map_err(|_| Error::Other("indexed grouped_sdpa group count exceeds u32".into()))?;
    let out_bytes = seq_len as usize
        * n_heads as usize
        * head_dim as usize
        * DType::BF16.size_in_bytes();
    let out_buf = output_buffer(ctx, out_bytes)?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    unsafe {
        ffi::check_cuda(ffi::apxinf_grouped_indexed_sdpa_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            out_buf.ptr(),
            seq_len,
            n_heads,
            head_dim,
            scale,
            group_ids.ptr(),
            group_offsets.ptr(),
            group_indices.ptr(),
            group_count,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![
            seq_len as usize,
            n_heads as usize * head_dim as usize,
        ]),
        DType::BF16,
        ctx.device_id(),
        out_buf,
    ))
}

/// Pack equal-group rows, execute one variable-length FA2 call, and restore
/// the model-owned token order. The caller owns and caches the immutable
/// permutation and cumulative group offsets.
#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_vision_sm80))]
#[allow(clippy::too_many_arguments)]
pub fn grouped_varlen_fa2(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    group_offsets: &CudaBuffer,
    group_indices: &CudaBuffer,
    group_count: usize,
    max_group_size: usize,
    inputs_prepacked: bool,
) -> Result<Tensor> {
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        return Err(Error::Other(
            "grouped varlen FA2: only BF16 supported".into(),
        ));
    }
    let expected_shape = [seq_len, n_heads, head_dim];
    if q.shape().dims() != expected_shape
        || k.shape().dims() != expected_shape
        || v.shape().dims() != expected_shape
    {
        return Err(Error::Other(
            "grouped varlen FA2: Q/K/V shape mismatch".into(),
        ));
    }
    if head_dim == 0
        || head_dim > 96
        || group_count == 0
        || max_group_size == 0
        || max_group_size > seq_len
        || group_offsets.len() != (group_count + 1) * std::mem::size_of::<u32>()
        || group_indices.len() != seq_len * std::mem::size_of::<u32>()
    {
        return Err(Error::Other(
            "grouped varlen FA2: invalid shape or group plan".into(),
        ));
    }
    let row_elements = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("grouped varlen FA2 row width overflow".into()))?;
    if row_elements % 8 != 0 {
        return Err(Error::Other(
            "grouped varlen FA2 requires a BF16 row width divisible by 8".into(),
        ));
    }
    let seq_len_u32 = u32::try_from(seq_len)
        .map_err(|_| Error::Other("grouped varlen FA2 sequence exceeds u32".into()))?;
    let row_elements_u32 = u32::try_from(row_elements)
        .map_err(|_| Error::Other("grouped varlen FA2 row width exceeds u32".into()))?;
    let batch_i32 = i32::try_from(group_count)
        .map_err(|_| Error::Other("grouped varlen FA2 group count exceeds i32".into()))?;
    let total_i32 = i32::try_from(seq_len)
        .map_err(|_| Error::Other("grouped varlen FA2 sequence exceeds i32".into()))?;
    let max_group_i32 = i32::try_from(max_group_size)
        .map_err(|_| Error::Other("grouped varlen FA2 group size exceeds i32".into()))?;
    let heads_i32 = i32::try_from(n_heads)
        .map_err(|_| Error::Other("grouped varlen FA2 heads exceed i32".into()))?;
    let head_dim_i32 = i32::try_from(head_dim)
        .map_err(|_| Error::Other("grouped varlen FA2 head dimension exceeds i32".into()))?;

    let tensor_bytes = q.size_in_bytes();
    if k.size_in_bytes() != tensor_bytes || v.size_in_bytes() != tensor_bytes {
        return Err(Error::Other(
            "grouped varlen FA2: Q/K/V byte size mismatch".into(),
        ));
    }
    let packed_inputs = if inputs_prepacked {
        None
    } else {
        Some((
            uninitialized_buffer(ctx, tensor_bytes)?,
            uninitialized_buffer(ctx, tensor_bytes)?,
            uninitialized_buffer(ctx, tensor_bytes)?,
        ))
    };
    let packed_output = uninitialized_buffer(ctx, tensor_bytes)?;
    let output = uninitialized_buffer(ctx, tensor_bytes)?;
    let lse_bytes = seq_len
        .checked_mul(n_heads)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| Error::Other("grouped varlen FA2 LSE size overflow".into()))?;
    let softmax_lse = uninitialized_buffer(ctx, lse_bytes)?;

    let (query, key, value) = if let Some((packed_q, packed_k, packed_v)) = &packed_inputs {
        unsafe {
            ffi::check_cuda(ffi::apxinf_pack_grouped_qkv_bf16(
                gpu_ptr(q)?, gpu_ptr(k)?, gpu_ptr(v)?,
                packed_q.ptr(), packed_k.ptr(), packed_v.ptr(),
                group_indices.ptr(), seq_len_u32, row_elements_u32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        (packed_q.ptr(), packed_k.ptr(), packed_v.ptr())
    } else {
        (gpu_ptr(q)?, gpu_ptr(k)?, gpu_ptr(v)?)
    };

    unsafe {
        ffi::check_cuda(ffi::apxinf_static_fa2_varlen_bf16(
            query, key, value,
            packed_output.ptr(), softmax_lse.ptr(), group_offsets.ptr(),
            batch_i32, total_i32, max_group_i32, heads_i32, heads_i32,
            head_dim_i32, (head_dim as f32).sqrt().recip(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
        ffi::check_cuda(ffi::apxinf_unpack_grouped_rows_bf16(
            packed_output.ptr(), output.ptr(), group_indices.ptr(),
            seq_len_u32, row_elements_u32, ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }

    Ok(make_gpu_tensor(
        Shape::new(vec![seq_len, row_elements]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// Causal attention mask on CUDA. Dispatches on dtype.
pub fn causal_mask(ctx: &CudaContext, input: &Tensor, kv_offset: u32) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let rows = dims[dims.len() - 2];
    let cols = *dims.last().unwrap();

    let out_bytes = input.size_in_bytes();
    let out_buf = output_buffer(ctx, out_bytes)?;

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
    softmax_causal_with_exp_cache(
        ctx,
        input,
        kv_offset,
        n_heads,
        softmax_exp_cache_enabled()?,
    )
}

pub(crate) fn softmax_causal_with_exp_cache(
    ctx: &CudaContext,
    input: &Tensor,
    kv_offset: u32,
    n_heads: u32,
    use_exp_cache: bool,
) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let rows = dims[dims.len() - 2];
    let cols = *dims.last().unwrap();

    if use_exp_cache
        && rows == n_heads as usize
        && cols > MAX_SOFTMAX_EXP_CACHE_COLS
        && softmax_global_exp_cache_enabled()?
    {
        return softmax_causal_with_global_exp_cache(ctx, input, kv_offset, n_heads);
    }

    if use_exp_cache {
        if input.dtype() != DType::BF16 {
            return Err(Error::Other(
                "attention softmax exp cache supports only BF16".into(),
            ));
        }
        if rows == n_heads as usize
            && cols > MAX_SOFTMAX_EXP_CACHE_COLS
            && !softmax_exp_cache_long_fallback_enabled()?
        {
            return Err(Error::Other(format!(
                "attention softmax exp cache cols {cols} exceed tested limit {MAX_SOFTMAX_EXP_CACHE_COLS}; set APXINF_SOFTMAX_EXP_CACHE_LONG_FALLBACK=1 for the exact scalar path"
            )));
        }
    }
    let use_exp_cache = use_exp_cache
        && ((rows == n_heads as usize && cols <= MAX_SOFTMAX_EXP_CACHE_COLS)
            || (rows != n_heads as usize && cols <= MAX_PREFILL_SOFTMAX_EXP_CACHE_COLS));

    let out_bytes = input.size_in_bytes();
    let out_buf = uninitialized_buffer(ctx, out_bytes)?;

    unsafe {
        let res = if use_exp_cache {
            ffi::apxinf_attention_softmax_bf16_exp_cache(
                gpu_ptr(input)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                kv_offset,
                n_heads,
                ctx.stream().handle(),
            )
        } else {
            match input.dtype() {
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
                    1.0,
                    ctx.stream().handle(),
                ),
                dtype => return unsupported_dtype("attention_softmax", dtype),
            }
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

pub(crate) fn softmax_causal_bf16_scaled_plain(
    ctx: &CudaContext,
    input: &Tensor,
    kv_offset: u32,
    n_heads: u32,
    score_scale: f32,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 {
        return Err(Error::Other(
            "scaled plain attention softmax requires BF16 input".into(),
        ));
    }
    require_finite("attention score scale", &[score_scale])?;
    let dims = input.shape().dims();
    let rows = dims[dims.len() - 2];
    let cols = dims[dims.len() - 1];
    let output = uninitialized_buffer(ctx, input.size_in_bytes())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_attention_softmax_bf16(
            gpu_ptr(input)?,
            output.ptr(),
            cols as u32,
            rows as u32,
            kv_offset,
            n_heads,
            score_scale,
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

pub(crate) fn softmax_causal_bf16_scaled_exp_cache(
    ctx: &CudaContext,
    input: &Tensor,
    kv_offset: u32,
    n_heads: u32,
    score_scale: f32,
) -> Result<Tensor> {
    require_finite("scaled exp-cache attention score", &[score_scale])?;
    let dims = input.shape().dims();
    let rows = dims[dims.len() - 2];
    let cols = dims[dims.len() - 1];
    if input.dtype() != DType::BF16
        || rows <= n_heads as usize
        || rows % n_heads as usize != 0
        || !(2..=MAX_PREFILL_SOFTMAX_EXP_CACHE_COLS).contains(&cols)
        || score_scale <= 0.0
    {
        return Err(Error::Other(
            "scaled exp-cache softmax requires positive scale and BF16 prefill rows with 2..=4096 columns"
                .into(),
        ));
    }
    let output = uninitialized_buffer(ctx, input.size_in_bytes())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_attention_softmax_bf16_scaled_exp_cache(
            gpu_ptr(input)?,
            output.ptr(),
            cols as u32,
            rows as u32,
            kv_offset,
            n_heads,
            score_scale,
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

pub(crate) fn softmax_causal_bf16_scaled_plain_gqa_packed(
    ctx: &CudaContext,
    input: &Tensor,
    kv_offset: u32,
    n_heads: u32,
    gqa_ratio: u32,
    score_scale: f32,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || gqa_ratio == 0 || n_heads % gqa_ratio != 0 {
        return Err(Error::Other(
            "packed GQA scaled softmax requires BF16 and divisible heads".into(),
        ));
    }
    require_finite("attention score scale", &[score_scale])?;
    let dims = input.shape().dims();
    let rows = dims[dims.len() - 2];
    let cols = dims[dims.len() - 1];
    if rows % n_heads as usize != 0 {
        return Err(Error::Other(
            "packed GQA scaled softmax rows must be divisible by heads".into(),
        ));
    }
    let output = uninitialized_buffer(ctx, input.size_in_bytes())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_attention_softmax_bf16_gqa_packed(
            gpu_ptr(input)?,
            output.ptr(),
            cols as u32,
            rows as u32,
            kv_offset,
            n_heads,
            gqa_ratio,
            score_scale,
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

pub(crate) fn softmax_causal_bf16_scaled_in_place_gqa_packed(
    ctx: &CudaContext,
    input: Tensor,
    kv_offset: u32,
    n_heads: u32,
    gqa_ratio: u32,
    score_scale: f32,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || gqa_ratio == 0 || n_heads % gqa_ratio != 0 {
        return Err(Error::Other(
            "in-place packed GQA softmax requires BF16 and divisible heads".into(),
        ));
    }
    require_finite("attention score scale", &[score_scale])?;
    let shape = input.shape().clone();
    let dims = shape.dims();
    let rows = dims[dims.len() - 2];
    let cols = dims[dims.len() - 1];
    if rows % n_heads as usize != 0 {
        return Err(Error::Other(
            "in-place packed GQA softmax rows must be divisible by heads".into(),
        ));
    }
    let output = CudaBuffer::from_tensor(&input).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_attention_softmax_bf16_scale_in_place(
            output.ptr(),
            cols as u32,
            rows as u32,
            kv_offset,
            n_heads,
            gqa_ratio,
            score_scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output.into_tensor(shape, DType::BF16))
}

pub(crate) fn softmax_causal_with_global_exp_cache(
    ctx: &CudaContext,
    input: &Tensor,
    kv_offset: u32,
    n_heads: u32,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    if input.dtype() != DType::BF16
        || dims.len() < 2
        || dims[dims.len() - 2] != n_heads as usize
    {
        return Err(Error::Other(
            "global softmax exp cache requires BF16 decode rows".into(),
        ));
    }
    let rows = dims[dims.len() - 2];
    let cols = dims[dims.len() - 1];
    let out_buf = uninitialized_buffer(ctx, input.size_in_bytes())?;
    let numerator_bytes = rows
        .checked_mul(cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| Error::Other("global softmax numerator size overflow".into()))?;
    let numerators = uninitialized_buffer(ctx, numerator_bytes)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_attention_softmax_bf16_global_exp_cache(
            gpu_ptr(input)?,
            out_buf.ptr(),
            numerators.ptr(),
            cols as u32,
            rows as u32,
            kv_offset,
            n_heads,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::BF16,
        ctx.device_id(),
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
    let projection_bytes = checked_bytes(
        DType::BF16,
        &[tokens, projection_width],
        "vision QKV split output",
    )?;
    let q = uninitialized_buffer(ctx, projection_bytes)?;
    let k = uninitialized_buffer(ctx, projection_bytes)?;
    let v = uninitialized_buffer(ctx, projection_bytes)?;
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

/// Split a biased packed GQA projection into unequal Q and K/V widths.
pub fn split_gqa_qkv_bias_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<QkvTensors> {
    let (tokens, width) = matrix_shape(qkv, "GQA QKV split")?;
    let q_width = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("GQA Q width overflow".into()))?;
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("GQA KV width overflow".into()))?;
    let expected = kv_width
        .checked_mul(2)
        .and_then(|packed_kv| q_width.checked_add(packed_kv))
        .ok_or_else(|| Error::Other("GQA packed width overflow".into()))?;
    if qkv.dtype() != DType::BF16
        || width != expected
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [width])
    {
        return Err(Error::Other("BF16 packed GQA QKV shape mismatch".into()));
    }
    let q = uninitialized_buffer(
        ctx,
        checked_bytes(DType::BF16, &[tokens, q_width], "GQA Q split output")?,
    )?;
    let kv_bytes = checked_bytes(DType::BF16, &[tokens, kv_width], "GQA KV split output")?;
    let k = uninitialized_buffer(ctx, kv_bytes)?;
    let v = uninitialized_buffer(ctx, kv_bytes)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_gqa_qkv_split_bias_bf16(
            gpu_ptr(qkv)?,
            optional_ptr(bias)?,
            q.ptr(),
            k.ptr(),
            v.ptr(),
            tokens as i32,
            q_width as i32,
            kv_width as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(QkvTensors {
        q: make_gpu_tensor(
            Shape::new(vec![tokens, q_heads, head_dim]),
            DType::BF16,
            ctx.device_id(),
            q,
        ),
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
}

#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_vision_sm80))]
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

#[cfg(apxinf_fa2_sm80)]
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
        && head_dim == 256
}

#[cfg(apxinf_fa2_sm80)]
#[allow(clippy::too_many_arguments)]
fn fa2_attention_splitkv(
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
        ffi::check_cuda(ffi::apxinf_static_fa2_bf16_splitkv(
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
                ctx, q, k, v, 1, q_shape[0], key_tokens, q_shape[1], 1, q_shape[2],
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
    #[cfg(apxinf_fa2_sm80)]
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
    #[cfg(apxinf_cutlass_fmha)]
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
    #[cfg(not(apxinf_fa2_sm80))]
    let output = output_buffer(ctx, q.size_in_bytes())?;
    #[cfg(not(apxinf_fa2_sm80))]
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
    #[cfg(not(apxinf_fa2_sm80))]
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
                tokens_per_batch as i32,
                shape[1] as i32,
                shape[1] as i32,
                shape[2] as i32,
                ctx.stream().handle(),
            );
            if status == 0 {
                return Ok(make_gpu_tensor(
                    q.shape().clone(),
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
