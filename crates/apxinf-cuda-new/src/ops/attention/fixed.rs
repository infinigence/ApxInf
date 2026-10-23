//! Fixed and direct-launch attention operators.
//!
//! The legacy helpers take `[tokens, heads, head_dim]` tensors; the cuda-new
//! Attention contract is rank-4 `[batch, tokens, heads, head_dim]`. These
//! adapters reshape at the boundary — reshape on a contiguous tensor is a
//! metadata change, not a copy.
//!
//! Causality follows the legacy helper it replaces: `mqa_bf16` and `mha_bf16`
//! are non-causal over the valid keys; `causal_gqa_bf16` is causal with the
//! query block at the end of the key range.

use apxinf_core::{DType, Device, Error, KvCache, Result, Shape, Tensor};

use crate::cublas::CublasTranspose;
use crate::{ops, CudaBuffer, CudaContext, CudaKvCache};

/// Q/K/V triple produced by a packed-QKV split. Mirrors the legacy struct.
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

fn scale_legacy(ctx: &CudaContext, input: Tensor, factor: f32) -> Result<Tensor> {
    use crate::ffi::abi::llm_decode as abi;
    let dtype = input.dtype();
    let input_buffer = CudaBuffer::from_tensor(&input).map_err(Error::Cuda)?;
    let output = CudaBuffer::alloc(input.size_in_bytes(), ctx.device_id()).map_err(Error::Cuda)?;
    let count = u32::try_from(input.numel())
        .map_err(|_| Error::Other("attention score count exceeds u32".into()))?;
    unsafe {
        crate::ops::validation::check_cuda(match dtype {
            DType::F32 => abi::apxinf_llmdec_scale_f32(
                input_buffer.ptr(),
                output.ptr(),
                count,
                factor,
                ctx.stream().handle(),
            ),
            DType::BF16 => abi::apxinf_llmdec_scale_bf16(
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
    use crate::ffi::abi::llm_decode as abi;
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
            DType::F32 => abi::apxinf_llmdec_attention_softmax_f32(
                input_buffer.ptr(),
                output.ptr(),
                cols,
                rows,
                kv_offset,
                n_heads,
                ctx.stream().handle(),
            ),
            DType::BF16 => abi::apxinf_llmdec_attention_softmax_bf16(
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

fn rank3(tensor: &Tensor, what: &str) -> Result<[usize; 3]> {
    let dims = tensor.shape().dims();
    if dims.len() != 3 {
        return Err(Error::Other(format!(
            "{what} requires rank-3 [tokens, heads, dim]"
        )));
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
    let k4 = k.reshape(vec![
        1,
        k.shape().numel() / (kv_heads * head_dim),
        kv_heads,
        head_dim,
    ])?;
    let v4 = v.reshape(vec![
        1,
        v.shape().numel() / (kv_heads * head_dim),
        kv_heads,
        head_dim,
    ])?;
    let mut out = ctx.allocate_output(
        Shape::new(vec![1, query_tokens, query_heads, head_dim]),
        DType::BF16,
    )?;
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

// ── Direct FA2 dispatch (legacy fast path) ──────────────────────────────────
//
// The tuned `kv_cache_attention` route keys its plan on the full Spec,
// including `key_capacity` and `key_tokens`. The legacy shim presents whole
// KV caches, so `key_capacity != key_tokens` keeps FA2 out of the tuned
// candidate set, and decode grows `key_tokens` every step, so every step is a
// new Spec that re-prepares. This direct route reproduces the legacy crate's
// raw FA2 dispatch: no tuning state, caller-planned split counts, buffers
// from the session-aware workspace. `Ok(None)` means "take the tuned route".

/// Split-KV eligibility, ported verbatim from the legacy crate.
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

/// Choose split-KV parallelism from the logical work shape.
///
/// The occupancy policy follows the documented heuristic in official
/// FlashAttention 2.7.4.post1 (BSD-3-Clause), expressed here independently in
/// Rust so the raw-pointer CUDA wrapper only marshals an already-made choice.
/// Ported verbatim from the legacy crate.
fn plan_fa2_split_count(
    batches: usize,
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    head_dim: usize,
    multiprocessors: usize,
) -> usize {
    if batches == 0
        || query_tokens == 0
        || key_tokens == 0
        || query_heads == 0
        || head_dim == 0
        || multiprocessors == 0
    {
        return 1;
    }
    let key_tile = if head_dim <= 64 {
        256
    } else if head_dim <= 128 {
        128
    } else {
        64
    };
    let key_tiles = key_tokens.div_ceil(key_tile);
    let row_tiles = query_tokens.div_ceil(64);
    let base_jobs = batches
        .saturating_mul(query_heads)
        .saturating_mul(row_tiles);
    let execution_slots = multiprocessors.saturating_mul(2);
    if execution_slots == 0
        || key_tiles == 0
        || base_jobs.saturating_mul(5) >= execution_slots.saturating_mul(4)
    {
        return 1;
    }

    let limit = 128.min(execution_slots).min(key_tiles);
    let eligible =
        |splits: usize| splits == 1 || key_tiles.div_ceil(splits) != key_tiles.div_ceil(splits - 1);
    let utilization = |splits: usize| {
        let jobs = base_jobs.saturating_mul(splits);
        let waves = jobs as f32 / execution_slots as f32;
        waves / waves.ceil()
    };
    let peak = (1..=limit)
        .filter(|&splits| eligible(splits))
        .map(utilization)
        .fold(0.0f32, f32::max);
    (1..=limit)
        .find(|&splits| eligible(splits) && utilization(splits) >= 0.85 * peak)
        .unwrap_or(1)
}

/// Direct FA2 BF16 attention over the legacy-shim surface. Returns `Ok(None)`
/// when FA2 is not compiled in or the shape falls outside the kernel
/// contract, so callers keep the tuned route as fallback. K/V may be whole
/// caches: the kernel reads only the leading `key_tokens` rows of the
/// contiguous `[*, kv_heads, head_dim]` storage, exactly like the legacy
/// `apxinf_static_fa2_bf16*` path.
#[allow(clippy::too_many_arguments)]
fn fa2_direct_buffer(
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
    output: Option<CudaBuffer>,
) -> Result<Option<CudaBuffer>> {
    use crate::ops::validation::check_cuda;
    use crate::ffi::abi::attention as abi;
    use crate::workspace::output_buffer;

    if !crate::FA2_DECODE {
        return Ok(None);
    }
    // Escape hatch mirroring legacy APXINF_DISABLE_FA2_SPLITKV: route back
    // through the tuned operator for A/B diagnosis.
    if std::env::var_os("APXINF_DISABLE_FA2_DIRECT").is_some() {
        return Ok(None);
    }
    let kv_rows = batches.checked_mul(key_tokens);
    let kv_elements = kv_rows
        .and_then(|rows| rows.checked_mul(kv_heads))
        .and_then(|elements| elements.checked_mul(head_dim));
    let q_elements = batches
        .checked_mul(query_tokens)
        .and_then(|elements| elements.checked_mul(query_heads))
        .and_then(|elements| elements.checked_mul(head_dim));
    match (q_elements, kv_elements) {
        (Some(q_elements), Some(kv_elements))
            if [q, k, v].into_iter().all(|t| t.dtype() == DType::BF16)
                && batches > 0
                && query_tokens > 0
                && key_tokens > 0
                && query_heads > 0
                && kv_heads > 0
                && query_heads % kv_heads == 0
                && (1..=256).contains(&head_dim)
                && (!causal || key_tokens >= query_tokens)
                && q.shape().numel() == q_elements
                && k.shape().numel() >= kv_elements
                && v.shape().numel() >= kv_elements => {}
        _ => return Ok(None),
    }
    let to_i32 = |value: usize| i32::try_from(value).ok();
    let (
        Some(batch_i),
        Some(query_tokens_i),
        Some(key_tokens_i),
        Some(query_heads_i),
        Some(kv_heads_i),
        Some(head_dim_i),
    ) = (
        to_i32(batches),
        to_i32(query_tokens),
        to_i32(key_tokens),
        to_i32(query_heads),
        to_i32(kv_heads),
        to_i32(head_dim),
    )
    else {
        return Ok(None);
    };
    let scale = (head_dim as f32).sqrt().recip();
    let output = match output {
        Some(output) if output.device() == ctx.device_id() && output.len() == q.size_in_bytes() => {
            output
        }
        Some(output) => {
            return Err(Error::Other(format!(
                "FA2 output buffer mismatch: expected CUDA {} and {} bytes, got CUDA {} and {} bytes",
                ctx.device_id(),
                q.size_in_bytes(),
                output.device(),
                output.len()
            )));
        }
        None => output_buffer(ctx, q.size_in_bytes())?,
    };
    let lse_elements = batches
        .checked_mul(query_heads)
        .and_then(|value| value.checked_mul(query_tokens))
        .ok_or_else(|| Error::Other("FA2 LSE size overflow".into()))?;
    let softmax_lse = output_buffer(
        ctx,
        lse_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::Other("FA2 LSE byte size overflow".into()))?,
    )?;
    // The legacy head-256 GQA fold (`apxinf_fa2_head_special`) reshaped a
    // decode step's GQA groups into the row dimension before planning. The
    // cuda-new vendored `fa2.cu` has no fold adapter, so plan on the caller's
    // dimensions as the non-special legacy arm did.
    let num_splits =
        if fa2_splitkv_enabled(query_tokens, key_tokens, query_heads, kv_heads, head_dim) {
            plan_fa2_split_count(
                batches,
                query_tokens,
                key_tokens,
                query_heads,
                head_dim,
                ctx.caps().multiprocessor_count as usize,
            )
        } else {
            1
        };
    // The vendored split-KV entry rejects `num_splits <= 1`; a one-way split
    // is the plain kernel.
    if num_splits > 1 {
        let block_n = if head_dim <= 64 {
            256
        } else if head_dim <= 128 {
            128
        } else {
            64
        };
        let max_splits = key_tokens.div_ceil(block_n).min(128);
        let lse_accum_bytes = max_splits
            .checked_mul(lse_elements)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Other("FA2 split-KV LSE accum overflow".into()))?;
        let softmax_lse_accum = output_buffer(ctx, lse_accum_bytes)?;
        let o_accum_bytes = lse_accum_bytes
            .checked_mul(head_dim)
            .ok_or_else(|| Error::Other("FA2 split-KV O accum overflow".into()))?;
        let o_accum = output_buffer(ctx, o_accum_bytes)?;
        check_cuda(unsafe {
            abi::apxinf_cn_fa2_bf16_splitkv(
                gpu_ptr(q)?,
                gpu_ptr(k)?,
                gpu_ptr(v)?,
                output.ptr(),
                softmax_lse.ptr(),
                softmax_lse_accum.ptr(),
                o_accum.ptr(),
                batch_i,
                query_tokens_i,
                key_tokens_i,
                query_heads_i,
                kv_heads_i,
                head_dim_i,
                scale,
                num_splits as i32,
                causal as i32,
                ctx.stream().handle(),
            )
        })?;
    } else {
        check_cuda(unsafe {
            abi::apxinf_cn_fa2_bf16(
                gpu_ptr(q)?,
                gpu_ptr(k)?,
                gpu_ptr(v)?,
                output.ptr(),
                softmax_lse.ptr(),
                batch_i,
                query_tokens_i,
                key_tokens_i,
                query_heads_i,
                kv_heads_i,
                head_dim_i,
                scale,
                causal as i32,
                ctx.stream().handle(),
            )
        })?;
    }
    Ok(Some(output))
}

#[allow(clippy::too_many_arguments)]
fn fa2_direct(
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
) -> Result<Option<Tensor>> {
    Ok(fa2_direct_buffer(
        ctx,
        q,
        k,
        v,
        batches,
        query_tokens,
        key_tokens,
        query_heads,
        kv_heads,
        head_dim,
        causal,
        None,
    )?
    .map(|output| make_gpu_tensor(q.shape().clone(), DType::BF16, ctx.device_id(), output)))
}

#[cfg(test)]
mod splitkv_planner_tests {
    use super::plan_fa2_split_count;

    fn official_fa2_choice(base_jobs: usize, execution_slots: usize, key_tiles: usize) -> usize {
        if base_jobs as f32 >= 0.8 * execution_slots as f32 {
            return 1;
        }
        let limit = 128.min(execution_slots).min(key_tiles);
        let eligible = |splits: usize| {
            splits == 1 || key_tiles.div_ceil(splits) != key_tiles.div_ceil(splits - 1)
        };
        let scores = (1..=limit)
            .map(|splits| {
                if !eligible(splits) {
                    return 0.0;
                }
                let waves = (base_jobs * splits) as f32 / execution_slots as f32;
                waves / waves.ceil()
            })
            .collect::<Vec<_>>();
        let peak = scores.iter().copied().fold(0.0f32, f32::max);
        (1..=limit)
            .find(|&splits| eligible(splits) && scores[splits - 1] >= 0.85 * peak)
            .unwrap_or(1)
    }

    #[test]
    fn pi05_orin_shape_uses_three_splits() {
        assert_eq!(plan_fa2_split_count(1, 10, 532, 8, 256, 16), 3);
    }

    #[test]
    fn saturated_or_single_tile_work_stays_unsplit() {
        assert_eq!(plan_fa2_split_count(1, 64, 512, 32, 256, 16), 1);
        assert_eq!(plan_fa2_split_count(1, 10, 32, 8, 256, 16), 1);
    }

    #[test]
    fn invalid_or_overflowing_work_stays_unsplit() {
        assert_eq!(plan_fa2_split_count(0, 1, 64, 1, 256, 16), 1);
        assert_eq!(plan_fa2_split_count(1, 0, 64, 1, 256, 16), 1);
        assert_eq!(plan_fa2_split_count(1, 1, 0, 1, 256, 16), 1);
        assert_eq!(plan_fa2_split_count(1, 1, 64, 0, 256, 16), 1);
        assert_eq!(plan_fa2_split_count(1, 1, 64, 1, 0, 16), 1);
        assert_eq!(plan_fa2_split_count(1, 1, 64, 1, 256, 0), 1);
        assert_eq!(
            plan_fa2_split_count(usize::MAX, 64, usize::MAX, usize::MAX, 256, usize::MAX),
            1
        );
    }

    #[test]
    fn agrees_with_official_fa2_policy_over_supported_domain() {
        for multiprocessors in 1..=128 {
            let execution_slots = multiprocessors * 2;
            for query_heads in 1..=256 {
                for key_tiles in 1..=128 {
                    let expected = official_fa2_choice(query_heads, execution_slots, key_tiles);
                    let actual = plan_fa2_split_count(
                        1,
                        1,
                        key_tiles * 64,
                        query_heads,
                        256,
                        multiprocessors,
                    );
                    assert_eq!(
                        actual, expected,
                        "SMs={multiprocessors}, heads={query_heads}, key_tiles={key_tiles}"
                    );
                }
            }
        }
    }
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
    let [query_tokens, query_heads, head_dim] = rank3(q, "attention query")?;
    if let Some(output) = fa2_direct(
        ctx,
        q,
        k,
        v,
        1,
        query_tokens,
        key_tokens,
        query_heads,
        1,
        head_dim,
        false,
    )? {
        return Ok(output);
    }
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
    // Legacy FA2 ran dense per-batch MHA as batch = views, query/key tokens =
    // tokens_per_batch, equal query/kv heads.
    if let Some(output) = fa2_direct(
        ctx,
        q,
        k,
        v,
        batches,
        tokens_per_batch,
        tokens_per_batch,
        heads,
        heads,
        head_dim,
        false,
    )? {
        return Ok(output);
    }
    let q4 = q.reshape(vec![batches, tokens_per_batch, heads, head_dim])?;
    let k4 = k.reshape(vec![batches, tokens_per_batch, heads, head_dim])?;
    let v4 = v.reshape(vec![batches, tokens_per_batch, heads, head_dim])?;
    let mut out = ctx.allocate_output(
        Shape::new(vec![batches, tokens_per_batch, heads, head_dim]),
        DType::BF16,
    )?;
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
    let [query_tokens, query_heads, head_dim] = rank3(q, "attention query")?;
    if let Some(output) = fa2_direct(
        ctx,
        q,
        k,
        v,
        1,
        query_tokens,
        key_tokens,
        query_heads,
        kv_heads,
        head_dim,
        true,
    )? {
        return Ok(output);
    }
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
    use crate::ffi::abi::{status, multimodal_attention as abi};
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
    let bias_buffer = bias
        .map(CudaBuffer::from_tensor)
        .transpose()
        .map_err(Error::Cuda)?;
    let q_buffer = CudaBuffer::from_tensor(&q).map_err(Error::Cuda)?;
    let k_buffer = CudaBuffer::from_tensor(&k).map_err(Error::Cuda)?;
    let v_buffer = CudaBuffer::from_tensor(&v).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_multimodal_attention_gqa_qkv_mrope_cache_bf16(
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
    use crate::ffi::abi::{status, multimodal_attention as abi};
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
        return Err(Error::Other(
            "vision QKV RoPE position ids too short".into(),
        ));
    }
    let shape = Shape::new(vec![tokens, heads, head_dim]);
    let q = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let k = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let v = ctx.allocate_output(shape, DType::BF16)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let qkv_buffer = CudaBuffer::from_tensor(qkv).map_err(Error::Cuda)?;
    let bias_buffer = bias
        .map(CudaBuffer::from_tensor)
        .transpose()
        .map_err(Error::Cuda)?;
    let q_buffer = CudaBuffer::from_tensor(&q).map_err(Error::Cuda)?;
    let k_buffer = CudaBuffer::from_tensor(&k).map_err(Error::Cuda)?;
    let v_buffer = CudaBuffer::from_tensor(&v).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_multimodal_attention_vision_qkv_rope_bf16(
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
    use crate::ffi::abi::{status, multimodal_attention as abi};
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
        return Err(Error::Other(
            "segmented MHA offsets buffer too short".into(),
        ));
    }
    if host_offsets.first() != Some(&0)
        || host_offsets.last().map(|&value| value as usize) != Some(shape[0])
        || host_offsets
            .windows(2)
            .any(|pair| pair[1] < pair[0] || (pair[1] - pair[0]) as usize > max_tokens)
    {
        return Err(Error::Other(
            "segmented MHA offsets are inconsistent".into(),
        ));
    }

    // Match the legacy segmented fast path: launch the tuned/direct FA2
    // implementation for each contiguous segment. The generic candidate is
    // a correctness fallback, but its one-CTA-per-query kernel serializes the
    // softmax and is particularly slow for WallOSS's 80-wide vision heads.
    // All scratch and output allocations use the session workspace so graph
    // capture/replay keeps stable addresses.
    if crate::FA2_DECODE
        && std::env::var_os("APXINF_DISABLE_FA2_DIRECT").is_none()
        && (1..=256).contains(&shape[2])
    {
        use crate::workspace::output_buffer;

        let heads = shape[1];
        let head_dim = shape[2];
        let row_bytes = heads
            .checked_mul(head_dim)
            .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("segmented MHA row size overflow".into()))?;
        let output = output_buffer(ctx, q.size_in_bytes())?;
        let q_buffer = CudaBuffer::from_tensor(q).map_err(Error::Cuda)?;
        let k_buffer = CudaBuffer::from_tensor(k).map_err(Error::Cuda)?;
        let v_buffer = CudaBuffer::from_tensor(v).map_err(Error::Cuda)?;

        for bounds in host_offsets.windows(2) {
            let start = bounds[0] as usize;
            let tokens = (bounds[1] - bounds[0]) as usize;
            if tokens == 0 {
                continue;
            }
            let bytes = tokens
                .checked_mul(row_bytes)
                .ok_or_else(|| Error::Other("segmented MHA slice size overflow".into()))?;
            let segment = |buffer: &CudaBuffer| -> Result<Tensor> {
                buffer
                    .view(start * row_bytes, bytes)
                    .map_err(Error::Cuda)?
                    .as_tensor(Shape::new(vec![tokens, heads, head_dim]), DType::BF16)
                    .map_err(Error::Cuda)
            };
            let (sq, sk, sv) = (
                segment(&q_buffer)?,
                segment(&k_buffer)?,
                segment(&v_buffer)?,
            );
            let output_segment = output.view(start * row_bytes, bytes).map_err(Error::Cuda)?;
            let Some(_) = fa2_direct_buffer(
                ctx,
                &sq,
                &sk,
                &sv,
                1,
                tokens,
                tokens,
                heads,
                heads,
                head_dim,
                false,
                Some(output_segment),
            )?
            else {
                return Err(Error::Other(format!(
                    "direct FA2 unexpectedly rejected segmented attention with {tokens} tokens"
                )));
            };
        }
        return Ok(make_gpu_tensor(
            q.shape().clone(),
            DType::BF16,
            ctx.device_id(),
            output,
        ));
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
        status::check(abi::apxinf_multimodal_attention_segmented_mha_bf16(
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

// ── gr00t attention surface ───────────────────────────────────────────────
//
// `noncausal` reshapes cuda-new's rank-3 attention output back to the flat
// `[query_len, heads * head_dim]` the legacy helper returned. The
// `_hdim96_bm64`, `segmented_*` and `try_*` entries are fixed-shape FA2
// candidates that were not vendored into cuda-new; they return `Ok(None)` so
// the call sites fall back to `noncausal` / `noncausal_strided_qkv`.

/// `noncausal`: non-causal dense attention over `[tokens, heads, head_dim]`,
/// flattened to `[tokens, heads * head_dim]`.
pub fn noncausal(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let q_dims = q.shape().dims();
    if q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || q_dims.len() != 3
        || q_dims[1] != n_heads
        || q_dims[2] != head_dim
        || n_heads == 0
        || head_dim == 0
    {
        return Err(Error::Other(
            "noncausal_sdpa: BF16 [query, heads, head_dim] expected".into(),
        ));
    }
    let key_tokens = k.shape().dims().first().copied().unwrap_or(0);
    let hidden = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("noncausal_sdpa: hidden size overflow".into()))?;
    let tokens = q_dims[0];
    noncausal_gqa_bf16(ctx, q, k, v, key_tokens)?.reshape(vec![tokens, hidden])
}

/// `noncausal_strided_qkv`: non-causal attention over a rank-2 packed
/// `[tokens, 3 * heads * head_dim]` projection, flattened to
/// `[tokens, heads * head_dim]`.
pub fn noncausal_strided_qkv(
    ctx: &CudaContext,
    qkv: &Tensor,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    use crate::ffi::abi::{transformer_bf16 as abi, status};
    use crate::ffi::raw::cuda_runtime as raw;
    let dims = qkv.shape().dims();
    let hidden = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("strided QKV attention hidden size overflow".into()))?;
    if qkv.dtype() != DType::BF16
        || dims.len() != 2
        || n_heads == 0
        || head_dim == 0
        || dims[1] != 3 * hidden
    {
        return Err(Error::Other(format!(
            "strided QKV attention expected CUDA BF16 [tokens,{}]",
            3 * hidden
        )));
    }
    let tokens = dims[0];
    let shape = Shape::new(vec![tokens, n_heads, head_dim]);
    let q = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let k = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let v = ctx.allocate_output(shape, DType::BF16)?;
    let qkv_buffer = CudaBuffer::from_tensor(qkv).map_err(Error::Cuda)?;
    let q_buffer = CudaBuffer::from_tensor(&q).map_err(Error::Cuda)?;
    let k_buffer = CudaBuffer::from_tensor(&k).map_err(Error::Cuda)?;
    let v_buffer = CudaBuffer::from_tensor(&v).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_transformer_bf16_split_strided_qkv_bf16(
            qkv_buffer.ptr(),
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            i32::try_from(tokens).map_err(|_| Error::Other("token count exceeds i32".into()))?,
            i32::try_from(hidden).map_err(|_| Error::Other("hidden size exceeds i32".into()))?,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    noncausal_gqa_bf16(ctx, &q, &k, &v, tokens)?.reshape(vec![tokens, hidden])
}

/// `noncausal_hdim96_bm64`: fixed-shape sm87/sm110 FA2 candidate. Not
/// vendored; callers fall back to `noncausal`.
pub fn noncausal_hdim96_bm64(
    _ctx: &CudaContext,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _n_heads: usize,
    _head_dim: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}

/// `noncausal_strided_qkv_hdim96_bm64`: fixed-shape companion. Not vendored.
pub fn noncausal_strided_qkv_hdim96_bm64(
    _ctx: &CudaContext,
    _qkv: &Tensor,
    _n_heads: usize,
    _head_dim: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}

/// `causal_gqa_prefill_bf16`: causal GQA prefill. Routes through cuda-new's
/// tuned causal attention operator; the FA2 fixed-shape fast path is not
/// vendored, so this always takes the generic route.
pub fn causal_gqa_prefill_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Option<Tensor>> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if [q, k, v]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || q_shape.len() != 3
        || k_shape.len() != 3
        || v.shape() != k.shape()
        || q_shape[0] == 0
        || q_shape[0] != k_shape[0]
        || q_shape[1] == 0
        || k_shape[1] == 0
        || q_shape[1] % k_shape[1] != 0
        || q_shape[2] == 0
        || q_shape[2] > 256
        || q_shape[2] != k_shape[2]
    {
        return Err(Error::Other(
            "causal BF16 GQA prefill shape mismatch".into(),
        ));
    }
    let key_tokens = k_shape[0];
    Ok(Some(causal_gqa_bf16(ctx, q, k, v, key_tokens)?))
}

/// `segmented_noncausal_contiguous_output_bf16`: fixed-shape FA2 segmented
/// candidate. Not vendored; callers fall back to their per-segment path.
#[allow(clippy::too_many_arguments)]
pub fn segmented_noncausal_contiguous_output_bf16(
    _ctx: &CudaContext,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _segment_lengths: &[usize],
    _n_heads: usize,
    _head_dim: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}

/// `try_noncausal_batched_equal`: equal-length batched FA2 candidate. Not
/// vendored; callers fall back to `noncausal`.
#[allow(clippy::too_many_arguments)]
pub fn try_noncausal_batched_equal(
    _ctx: &CudaContext,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _batches: usize,
    _sequence_len: usize,
    _n_heads: usize,
    _head_dim: usize,
) -> Result<Option<Tensor>> {
    Ok(None)
}

// ── smolvla attention surface ───────────────────────────────────────────────
//
// BF16 entries reuse the tuned cuda-new dense-attention operator — the same
// route `mqa_bf16`/`mha_bf16`/`causal_gqa_bf16` take — with the legacy
// composition (row views over the shared prefix, per-view MHA) kept intact.
// FP16 entries are the cuBLAS-composed Xavier kernels, ported verbatim into
// native/adapters/attention/fixed_f16.cu.

use crate::ops::validation::{gpu_ptr, make_gpu_tensor, matrix_shape};

fn row_view(
    ctx: &CudaContext,
    input: &Tensor,
    start_row: usize,
    end_row: usize,
    dtype: DType,
) -> Result<Tensor> {
    let shape = input.shape().dims();
    if input.dtype() != dtype
        || input.device() != Device::Cuda(ctx.device_id())
        || shape.len() != 3
        || start_row >= end_row
        || end_row > shape[0]
    {
        return Err(Error::Other(format!(
            "{dtype} attention row view expects a 3-D CUDA tensor with a non-empty row range"
        )));
    }
    let row_elements = shape[1]
        .checked_mul(shape[2])
        .ok_or_else(|| Error::Other("attention row view element count overflow".into()))?;
    let bytes = end_row
        .checked_sub(start_row)
        .and_then(|rows| rows.checked_mul(row_elements))
        .and_then(|elements| elements.checked_mul(dtype.size_in_bytes()))
        .ok_or_else(|| Error::Other("attention row view byte size overflow".into()))?;
    let offset = start_row
        .checked_mul(row_elements)
        .and_then(|elements| elements.checked_mul(dtype.size_in_bytes()))
        .ok_or_else(|| Error::Other("attention row view offset overflow".into()))?;
    let buffer = CudaBuffer::from_tensor(input)
        .and_then(|buffer| buffer.view(offset, bytes))
        .map_err(Error::Cuda)?;
    buffer
        .as_tensor(
            Shape::new(vec![end_row - start_row, shape[1], shape[2]]),
            dtype,
        )
        .map_err(Error::Cuda)
}

/// `full_gqa_bf16`: non-causal BF16 GQA for prefix-only cross attention.
pub fn full_gqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.shape() != k.shape()
        || q_shape.len() != 3
        || k_shape.len() != 3
        || q_shape[1] == 0
        || q_shape[1] % k_shape[1] != 0
        || q_shape[2] != k_shape[2]
        || key_tokens == 0
        || key_tokens != k_shape[0]
    {
        return Err(Error::Other("non-causal BF16 GQA shape mismatch".into()));
    }
    noncausal_gqa_bf16(ctx, q, k, v, key_tokens)
}

/// `prefix_gqa_bf16`: two-block non-causal BF16 GQA used by SmolVLA's prefix.
///
/// The shared prefix (vision and language) attends only to itself, while the
/// trailing state tokens attend to both blocks. This mirrors LeRobot's
/// prefix-LM attention without materializing a full 2-D mask.
pub fn prefix_gqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    shared_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if q_shape.len() != 3 || k_shape.len() != 3 {
        return Err(Error::Other(
            "prefix BF16 GQA requires rank-3 Q/K tensors".into(),
        ));
    }
    let key_tokens = k_shape[0];
    if shared_tokens == 0 || shared_tokens >= key_tokens || q_shape[0] != key_tokens {
        return Err(Error::Other(
            "prefix BF16 GQA requires 0 < shared tokens < key tokens".into(),
        ));
    }
    let shared_q = row_view(ctx, q, 0, shared_tokens, DType::BF16)?;
    let shared_k = row_view(ctx, k, 0, shared_tokens, DType::BF16)?;
    let shared_v = row_view(ctx, v, 0, shared_tokens, DType::BF16)?;
    let state_q = row_view(ctx, q, shared_tokens, key_tokens, DType::BF16)?;
    let shared_output = full_gqa_bf16(ctx, &shared_q, &shared_k, &shared_v, shared_tokens)?;
    let state_output = full_gqa_bf16(ctx, &state_q, k, v, key_tokens)?;
    let head_width = q_shape[1] * q_shape[2];
    let shared_rows = shared_output.reshape(vec![shared_tokens, head_width])?;
    let state_rows = state_output.reshape(vec![key_tokens - shared_tokens, head_width])?;
    crate::ops::elementwise::concat_rows_bf16(ctx, &shared_rows, &state_rows)?
        .reshape(vec![key_tokens, q_shape[1], q_shape[2]])
}

/// `vision_mha_bf16`: non-causal BF16 vision MHA batched per camera view.
/// Routes through the dense per-batch MHA, the same composed route the legacy
/// non-FA2 builds took.
pub fn vision_mha_bf16(
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
            "static inference BF16 vision MHA shape mismatch".into(),
        ));
    }
    mha_bf16(ctx, q, k, v, tokens_per_batch)
}

/// `full_gqa_f16`: non-causal FP16 GQA using Xavier FP16 tensor-core GEMMs.
pub fn full_gqa_f16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    use crate::ffi::abi::transformer_f16 as abi;
    use crate::ffi::raw::cublas::check_cublas;
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if q.dtype() != DType::F16
        || k.dtype() != DType::F16
        || v.shape() != k.shape()
        || q_shape.len() != 3
        || k_shape.len() != 3
        || q_shape[1] == 0
        || q_shape[1] % k_shape[1] != 0
        || q_shape[2] != k_shape[2]
        || key_tokens == 0
        || key_tokens != k_shape[0]
    {
        return Err(Error::Other("non-causal FP16 GQA shape mismatch".into()));
    }
    let output = crate::workspace::output_buffer(ctx, q.size_in_bytes())?;
    unsafe {
        check_cublas(abi::apxinf_transformer_f16_cublas_gqa_f16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            q_shape[0] as i32,
            key_tokens as i32,
            q_shape[1] as i32,
            k_shape[1] as i32,
            q_shape[2] as i32,
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

/// `suffix_causal_gqa_f16`: FP16 GQA for a causal suffix following a fully
/// visible prefix. Query row `i` attends to all `key_offset + i + 1` keys.
pub fn suffix_causal_gqa_f16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
    key_offset: usize,
) -> Result<Tensor> {
    use crate::ffi::abi::transformer_f16 as abi;
    use crate::ffi::raw::cublas::check_cublas;
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if q.dtype() != DType::F16
        || k.dtype() != DType::F16
        || v.shape() != k.shape()
        || q_shape.len() != 3
        || k_shape.len() != 3
        || q_shape[1] == 0
        || q_shape[1] % k_shape[1] != 0
        || q_shape[2] != k_shape[2]
        || key_tokens == 0
        || key_tokens != k_shape[0]
        || key_offset >= key_tokens
        || q_shape[0] + key_offset != key_tokens
    {
        return Err(Error::Other("causal suffix FP16 GQA shape mismatch".into()));
    }
    let output = crate::workspace::output_buffer(ctx, q.size_in_bytes())?;
    unsafe {
        check_cublas(abi::apxinf_transformer_f16_cublas_gqa_causal_f16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            q_shape[0] as i32,
            key_tokens as i32,
            q_shape[1] as i32,
            k_shape[1] as i32,
            q_shape[2] as i32,
            key_offset as i32,
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

/// `prefix_gqa_f16`: the FP16 two-block prefix attention.
pub fn prefix_gqa_f16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    shared_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if q_shape.len() != 3 || k_shape.len() != 3 {
        return Err(Error::Other(
            "prefix FP16 GQA requires rank-3 Q/K tensors".into(),
        ));
    }
    let key_tokens = k_shape[0];
    if shared_tokens == 0
        || shared_tokens >= key_tokens
        || q_shape[0] != key_tokens
        || q.dtype() != DType::F16
    {
        return Err(Error::Other(
            "prefix FP16 GQA requires 0 < shared tokens < key tokens".into(),
        ));
    }
    let shared_q = row_view(ctx, q, 0, shared_tokens, DType::F16)?;
    let shared_k = row_view(ctx, k, 0, shared_tokens, DType::F16)?;
    let shared_v = row_view(ctx, v, 0, shared_tokens, DType::F16)?;
    let state_q = row_view(ctx, q, shared_tokens, key_tokens, DType::F16)?;
    let shared_output = full_gqa_f16(ctx, &shared_q, &shared_k, &shared_v, shared_tokens)?;
    let state_output = full_gqa_f16(ctx, &state_q, k, v, key_tokens)?;
    let head_width = q_shape[1] * q_shape[2];
    let shared_rows = shared_output.reshape(vec![shared_tokens, head_width])?;
    let state_rows = state_output.reshape(vec![key_tokens - shared_tokens, head_width])?;
    crate::ops::elementwise::concat_rows_f16(ctx, &shared_rows, &state_rows)?
        .reshape(vec![key_tokens, q_shape[1], q_shape[2]])
}

/// `vision_mha_f16`: non-causal FP16 vision MHA that keeps each camera view
/// independent, composed from per-view cuBLAS GQA like the legacy helper.
pub fn vision_mha_f16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let shape = q.shape().dims();
    if q.dtype() != DType::F16
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || shape.len() != 3
        || k.shape() != q.shape()
        || v.shape() != q.shape()
        || shape[1] == 0
        || tokens_per_view == 0
        || shape[0] % tokens_per_view != 0
    {
        return Err(Error::Other(
            "FP16 vision MHA expects matching [tokens,heads,head_dim] tensors".into(),
        ));
    }
    let views = shape[0] / tokens_per_view;
    let head_width = shape[1]
        .checked_mul(shape[2])
        .ok_or_else(|| Error::Other("FP16 vision MHA width overflow".into()))?;
    let mut output = None;
    for view in 0..views {
        let start = view * tokens_per_view;
        let end = start + tokens_per_view;
        let view_q = row_view(ctx, q, start, end, DType::F16)?;
        let view_k = row_view(ctx, k, start, end, DType::F16)?;
        let view_v = row_view(ctx, v, start, end, DType::F16)?;
        let view_output = full_gqa_f16(ctx, &view_q, &view_k, &view_v, tokens_per_view)?
            .reshape(vec![tokens_per_view, head_width])?;
        output = Some(match output {
            Some(output) => crate::ops::elementwise::concat_rows_f16(ctx, &output, &view_output)?,
            None => view_output,
        });
    }
    output
        .ok_or_else(|| Error::Other("FP16 vision MHA requires at least one view".into()))?
        .reshape(vec![shape[0], shape[1], shape[2]])
}

/// `split_qkv_bias_f16`: split a `[tokens, 3*heads*dim]` packed FP16 vision
/// projection into Q/K/V with an optional packed bias, no rotation. The
/// cuda-new RoPE operator's dtype-templated `split_qkv_bias` kernel is the
/// verbatim legacy kernel.
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
    let shape = Shape::new(vec![tokens, heads, head_dim]);
    let mut q = ctx.allocate_output(shape.clone(), DType::F16)?;
    let mut k = ctx.allocate_output(shape.clone(), DType::F16)?;
    let mut v = ctx.allocate_output(shape, DType::F16)?;
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

/// GQA scaled-dot-product attention over the portable backend's head-major
/// CUDA KV cache. This is the legacy algorithm and layout contract, moved
/// into cuda-new without changing its model-facing behavior.
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
        use crate::ffi::abi::llm_decode as abi;
        let gather = |src: &CudaBuffer| -> Result<CudaBuffer> {
            let dst = CudaBuffer::alloc(
                seq_len * n_kv_heads * head_dim * element_bytes,
                ctx.device_id(),
            )
            .map_err(Error::Cuda)?;
            unsafe {
                crate::ops::validation::check_cuda(abi::apxinf_llmdec_kv_cache_gather_bf16(
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
        return causal_gqa_bf16(ctx, query, &key, &value, seq_len)?
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
pub fn softmax_f32_into(
    ctx: &CudaContext,
    scores: &CudaBuffer,
    output: &CudaBuffer,
    cols: usize,
    heads: usize,
    position: crate::CudaDeviceAddress,
) -> Result<()> {
    use crate::ops::validation::{check_cuda, checked_bytes, require_address, require_buffers};
    use crate::ffi::abi::llm_decode as abi;
    let bytes = checked_bytes(DType::F32, &[heads, cols], "attention softmax")?;
    require_buffers(
        ctx,
        "attention softmax",
        &[("scores", scores, bytes), ("output", output, bytes)],
    )?;
    require_address(ctx, "attention softmax", "position", position, 4)?;
    check_cuda(unsafe {
        abi::apxinf_llmdec_attention_softmax_decode_f32(
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
    position: crate::CudaDeviceAddress,
) -> Result<()> {
    use crate::ops::validation::{
        check_cuda, checked_bytes, require_address, require_buffers, require_finite,
    };
    use crate::ffi::abi::llm_decode as abi;
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
        abi::apxinf_llmdec_flash_attn_decode_bf16(
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
