use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::{ffi, CudaBuffer, CudaContext, CudaDeviceAddress};

pub const Q_HEADS: usize = 24;
pub const KV_HEADS: usize = 4;
pub const HEAD_DIM: usize = 256;
pub const WIDTH: usize = Q_HEADS * HEAD_DIM;
pub const MAX_SPLIT_CTA: usize = 16;
pub const SPLIT_CTA_CANDIDATE_COUNT: usize = 16;
pub const SPLIT_CTA_CANDIDATE_MIN_KV_BUCKET: usize = 256;

/// Layer-screened SM89 candidate policy.  Callers must opt into this policy;
/// the model default remains the incumbent until full-token E2E promotion.
pub fn split_cta_candidate_for_bucket(bucket_kv_len: usize) -> Option<usize> {
    (bucket_kv_len >= SPLIT_CTA_CANDIDATE_MIN_KV_BUCKET).then_some(SPLIT_CTA_CANDIDATE_COUNT)
}

pub struct SplitCtaWorkspace {
    partial_max: CudaBuffer,
    partial_sum: CudaBuffer,
    partial_accumulator: CudaBuffer,
}

impl SplitCtaWorkspace {
    pub fn new(ctx: &CudaContext) -> Result<Self> {
        let scalars = Q_HEADS * MAX_SPLIT_CTA * std::mem::size_of::<f32>();
        let accumulators = scalars * HEAD_DIM;
        Ok(Self {
            partial_max: CudaBuffer::alloc(scalars, ctx.device_id()).map_err(Error::Cuda)?,
            partial_sum: CudaBuffer::alloc(scalars, ctx.device_id()).map_err(Error::Cuda)?,
            partial_accumulator: CudaBuffer::alloc(accumulators, ctx.device_id())
                .map_err(Error::Cuda)?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_write(
    ctx: &CudaContext,
    q_projection: &Tensor,
    k_projection: &Tensor,
    v_projection: &Tensor,
    q_norm_weight: &Tensor,
    k_norm_weight: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    gate: &Tensor,
    position: CudaDeviceAddress,
) -> Result<()> {
    if ctx.caps().sm != 89 {
        return Err(Error::Other(format!(
            "Qwen3.5 split-CTA attention is frozen for SM89, got SM{}",
            ctx.caps().sm
        )));
    }
    let device = Device::Cuda(ctx.device_id());
    let contracts = [
        (q_projection, DType::BF16, vec![1, 2 * WIDTH]),
        (k_projection, DType::BF16, vec![1, KV_HEADS * HEAD_DIM]),
        (v_projection, DType::BF16, vec![1, KV_HEADS * HEAD_DIM]),
        (q_norm_weight, DType::BF16, vec![HEAD_DIM]),
        (k_norm_weight, DType::BF16, vec![HEAD_DIM]),
        (query, DType::BF16, vec![Q_HEADS, HEAD_DIM]),
        (key, DType::BF16, vec![KV_HEADS, HEAD_DIM]),
        (value, DType::BF16, vec![KV_HEADS, HEAD_DIM]),
        (gate, DType::BF16, vec![Q_HEADS, HEAD_DIM]),
    ];
    for (tensor, dtype, shape) in contracts {
        if tensor.dtype() != dtype || tensor.device() != device || tensor.shape().dims() != shape {
            return Err(Error::Other(
                "Qwen3.5 attention prepare contract mismatch".into(),
            ));
        }
    }
    if position.device() != ctx.device_id() || position.len() < 3 * 4 {
        return Err(Error::Other(
            "Qwen3.5 attention mRoPE position contract mismatch".into(),
        ));
    }
    let buffers = [
        q_projection,
        k_projection,
        v_projection,
        q_norm_weight,
        k_norm_weight,
        query,
        key,
        value,
        gate,
    ]
    .map(|tensor| CudaBuffer::from_tensor(tensor).map_err(Error::Cuda))
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_attention_prepare_bf16(
            buffers[0].ptr(),
            buffers[1].ptr(),
            buffers[2].ptr(),
            buffers[3].ptr(),
            buffers[4].ptr(),
            buffers[5].ptr(),
            buffers[6].ptr(),
            buffers[7].ptr(),
            buffers[8].ptr(),
            position.ptr(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_m8_write(
    ctx: &CudaContext,
    q_projection: &Tensor,
    k_projection: &Tensor,
    v_projection: &Tensor,
    q_norm_weight: &Tensor,
    k_norm_weight: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    gate: &Tensor,
    positions: CudaDeviceAddress,
) -> Result<()> {
    if ctx.caps().sm != 89 {
        return Err(Error::Other(format!(
            "Qwen3.5 M8 attention prepare is frozen for SM89, got SM{}",
            ctx.caps().sm
        )));
    }
    let device = Device::Cuda(ctx.device_id());
    let dims = q_projection.shape().dims();
    let tokens = dims.first().copied().unwrap_or(0);
    if dims != [tokens, 2 * WIDTH] || !(1..=8).contains(&tokens) {
        return Err(Error::Other(
            "Qwen3.5 M8 attention q projection contract mismatch".into(),
        ));
    }
    let contracts = [
        (q_projection, DType::BF16, vec![tokens, 2 * WIDTH]),
        (k_projection, DType::BF16, vec![tokens, KV_HEADS * HEAD_DIM]),
        (v_projection, DType::BF16, vec![tokens, KV_HEADS * HEAD_DIM]),
        (q_norm_weight, DType::BF16, vec![HEAD_DIM]),
        (k_norm_weight, DType::BF16, vec![HEAD_DIM]),
        (query, DType::BF16, vec![tokens, Q_HEADS, HEAD_DIM]),
        (key, DType::BF16, vec![tokens, KV_HEADS, HEAD_DIM]),
        (value, DType::BF16, vec![tokens, KV_HEADS, HEAD_DIM]),
        (gate, DType::BF16, vec![tokens, Q_HEADS, HEAD_DIM]),
    ];
    for (tensor, dtype, shape) in contracts {
        if tensor.dtype() != dtype || tensor.device() != device || tensor.shape().dims() != shape {
            return Err(Error::Other(
                "Qwen3.5 M8 attention prepare contract mismatch".into(),
            ));
        }
    }
    if positions.device() != ctx.device_id() || positions.len() < tokens * 3 * 4 {
        return Err(Error::Other(
            "Qwen3.5 M8 attention mRoPE positions contract mismatch".into(),
        ));
    }
    let buffers = [
        q_projection,
        k_projection,
        v_projection,
        q_norm_weight,
        k_norm_weight,
        query,
        key,
        value,
        gate,
    ]
    .map(|tensor| CudaBuffer::from_tensor(tensor).map_err(Error::Cuda))
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_attention_prepare_m8_bf16(
            buffers[0].ptr(),
            buffers[1].ptr(),
            buffers[2].ptr(),
            buffers[3].ptr(),
            buffers[4].ptr(),
            buffers[5].ptr(),
            buffers[6].ptr(),
            buffers[7].ptr(),
            buffers[8].ptr(),
            positions.ptr(),
            tokens as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

pub fn gate_write(ctx: &CudaContext, input: &Tensor, gate: &Tensor, output: &Tensor) -> Result<()> {
    let device = Device::Cuda(ctx.device_id());
    for tensor in [input, gate, output] {
        if tensor.dtype() != DType::BF16
            || tensor.device() != device
            || tensor.shape().dims() != [Q_HEADS, HEAD_DIM]
        {
            return Err(Error::Other(
                "Qwen3.5 attention gate contract mismatch".into(),
            ));
        }
    }
    let input = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let gate = CudaBuffer::from_tensor(gate).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_attention_gate_bf16(
            input.ptr(),
            gate.ptr(),
            output.ptr(),
            WIDTH as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

pub fn gate_m8_write(
    ctx: &CudaContext,
    input: &Tensor,
    gate: &Tensor,
    output: &Tensor,
) -> Result<()> {
    let device = Device::Cuda(ctx.device_id());
    let dims = input.shape().dims();
    let tokens = dims.first().copied().unwrap_or(0);
    let expected = [tokens, Q_HEADS, HEAD_DIM];
    if !(1..=8).contains(&tokens)
        || [input, gate, output].iter().any(|tensor| {
            tensor.dtype() != DType::BF16
                || tensor.device() != device
                || tensor.shape().dims() != expected
        })
    {
        return Err(Error::Other(
            "Qwen3.5 M8 attention gate contract mismatch".into(),
        ));
    }
    let input = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let gate = CudaBuffer::from_tensor(gate).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_attention_gate_bf16(
            input.ptr(),
            gate.ptr(),
            output.ptr(),
            (tokens * WIDTH) as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

/// SM89 long-context decode candidate with cross-CTA sequence splitting.
///
/// The path is intentionally explicit and fail-closed.  The incumbent
/// one-CTA-per-Q-head implementation remains the short-context/default path.
#[allow(clippy::too_many_arguments)]
pub fn flash_split_cta_write(
    ctx: &CudaContext,
    query: &Tensor,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &Tensor,
    workspace: &SplitCtaWorkspace,
    split_count: usize,
    bucket_kv_len: usize,
    max_seq_len: usize,
    scale: f32,
    position: CudaDeviceAddress,
) -> Result<()> {
    let device = Device::Cuda(ctx.device_id());
    for tensor in [query, output] {
        if tensor.dtype() != DType::BF16
            || tensor.device() != device
            || tensor.shape().dims() != [Q_HEADS, HEAD_DIM]
        {
            return Err(Error::Other(
                "Qwen3.5 split-CTA attention tensor contract mismatch".into(),
            ));
        }
    }
    if !matches!(split_count, 2 | 4 | 8 | 16)
        || bucket_kv_len == 0
        || bucket_kv_len > max_seq_len
        || !scale.is_finite()
        || scale <= 0.0
        || max_seq_len > i32::MAX as usize
        || bucket_kv_len > i32::MAX as usize
    {
        return Err(Error::Other(
            "Qwen3.5 split-CTA attention launch contract mismatch".into(),
        ));
    }
    let cache_bytes = KV_HEADS
        .checked_mul(max_seq_len)
        .and_then(|value| value.checked_mul(HEAD_DIM))
        .and_then(|value| value.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("Qwen3.5 split-CTA cache size overflow".into()))?;
    for (name, buffer) in [("key", key_cache), ("value", value_cache)] {
        if buffer.device() != ctx.device_id() || buffer.len() < cache_bytes {
            return Err(Error::Other(format!(
                "Qwen3.5 split-CTA {name} cache contract mismatch"
            )));
        }
    }
    let scalar_bytes = Q_HEADS * MAX_SPLIT_CTA * std::mem::size_of::<f32>();
    let accumulator_bytes = scalar_bytes * HEAD_DIM;
    for (name, buffer, bytes) in [
        ("partial max", &workspace.partial_max, scalar_bytes),
        ("partial sum", &workspace.partial_sum, scalar_bytes),
        (
            "partial accumulator",
            &workspace.partial_accumulator,
            accumulator_bytes,
        ),
    ] {
        if buffer.device() != ctx.device_id() || buffer.len() < bytes {
            return Err(Error::Other(format!(
                "Qwen3.5 split-CTA {name} workspace contract mismatch"
            )));
        }
    }
    if position.device() != ctx.device_id() || position.len() < 4 {
        return Err(Error::Other(
            "Qwen3.5 split-CTA position contract mismatch".into(),
        ));
    }
    let query = CudaBuffer::from_tensor(query).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_attention_flash_split_cta_bf16(
            query.ptr(),
            key_cache.ptr(),
            value_cache.ptr(),
            workspace.partial_max.ptr(),
            workspace.partial_sum.ptr(),
            workspace.partial_accumulator.ptr(),
            output.ptr(),
            split_count as i32,
            bucket_kv_len as i32,
            max_seq_len as i32,
            scale,
            position.ptr(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn flash_split_cta_buffer_write(
    ctx: &CudaContext,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &CudaBuffer,
    workspace: &SplitCtaWorkspace,
    split_count: usize,
    bucket_kv_len: usize,
    max_seq_len: usize,
    scale: f32,
    position: CudaDeviceAddress,
) -> Result<()> {
    let vector_bytes = WIDTH * DType::BF16.size_in_bytes();
    for (name, buffer) in [("query", query), ("output", output)] {
        if buffer.device() != ctx.device_id() || buffer.len() < vector_bytes {
            return Err(Error::Other(format!(
                "Qwen3.5 split-CTA {name} buffer contract mismatch"
            )));
        }
    }
    if !matches!(split_count, 2 | 4 | 8 | 16)
        || bucket_kv_len == 0
        || bucket_kv_len > max_seq_len
        || !scale.is_finite()
        || scale <= 0.0
        || max_seq_len > i32::MAX as usize
        || bucket_kv_len > i32::MAX as usize
    {
        return Err(Error::Other(
            "Qwen3.5 split-CTA buffer launch contract mismatch".into(),
        ));
    }
    let cache_bytes = KV_HEADS
        .checked_mul(max_seq_len)
        .and_then(|value| value.checked_mul(HEAD_DIM))
        .and_then(|value| value.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("Qwen3.5 split-CTA cache size overflow".into()))?;
    for (name, buffer) in [("key", key_cache), ("value", value_cache)] {
        if buffer.device() != ctx.device_id() || buffer.len() < cache_bytes {
            return Err(Error::Other(format!(
                "Qwen3.5 split-CTA {name} cache contract mismatch"
            )));
        }
    }
    let scalar_bytes = Q_HEADS * MAX_SPLIT_CTA * std::mem::size_of::<f32>();
    let accumulator_bytes = scalar_bytes * HEAD_DIM;
    for (name, buffer, bytes) in [
        ("partial max", &workspace.partial_max, scalar_bytes),
        ("partial sum", &workspace.partial_sum, scalar_bytes),
        (
            "partial accumulator",
            &workspace.partial_accumulator,
            accumulator_bytes,
        ),
    ] {
        if buffer.device() != ctx.device_id() || buffer.len() < bytes {
            return Err(Error::Other(format!(
                "Qwen3.5 split-CTA {name} workspace contract mismatch"
            )));
        }
    }
    if position.device() != ctx.device_id() || position.len() < 4 {
        return Err(Error::Other(
            "Qwen3.5 split-CTA position contract mismatch".into(),
        ));
    }
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_attention_flash_split_cta_bf16(
            query.ptr(),
            key_cache.ptr(),
            value_cache.ptr(),
            workspace.partial_max.ptr(),
            workspace.partial_sum.ptr(),
            workspace.partial_accumulator.ptr(),
            output.ptr(),
            split_count as i32,
            bucket_kv_len as i32,
            max_seq_len as i32,
            scale,
            position.ptr(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_cta_candidate_starts_at_screened_bucket() {
        assert_eq!(split_cta_candidate_for_bucket(0), None);
        assert_eq!(split_cta_candidate_for_bucket(128), None);
        assert_eq!(split_cta_candidate_for_bucket(255), None);
        assert_eq!(split_cta_candidate_for_bucket(256), Some(16));
        assert_eq!(split_cta_candidate_for_bucket(32768), Some(16));
    }
}
