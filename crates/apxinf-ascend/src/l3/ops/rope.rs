//! RoPE (Rotary Position Embedding) operations.
//!
//! Uses `aclnnApplyRotaryPosEmb` for the rotation. QKV splitting and
//! KV cache writes are handled via D2D copies.
//!
//! cos/sin tables are cached in a thread-local keyed by
//! `(seq_len, head_dim, theta, position_offset, n_heads)` so that
//! graph capture can reuse the warmup-pass device buffers without
//! re-issuing H2D transfers (forbidden during RI capture).

use std::cell::RefCell;

use apxinf_core::{DType, Device, Error, Result, Shape, Storage, Tensor};

use crate::l3::context::AscendNewContext;

// ── Args structs ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeSemantic {
    /// Grouped-query split with rotary embedding on Q and K.
    SplitQkvRope,
    /// Vision MHA split with optional bias, no rotation.
    SplitQkvBias,
}

pub struct RopeArgs<'a> {
    pub semantic: RopeSemantic,
    pub qkv: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub q: &'a mut Tensor,
    pub k: &'a mut Tensor,
    pub v: &'a mut Tensor,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub theta: f32,
    pub position_offset: usize,
    pub kv_output_offset: usize,
}

// ── cos/sin device cache (RI-capture compatible) ─────────────────────

use std::collections::HashMap;

struct CosSinEntry {
    cos_ptr: usize,
    sin_ptr: usize,
    // Keep device buffers alive
    _cos_buf: crate::l3::buffer::AscendBuffer,
    _sin_buf: crate::l3::buffer::AscendBuffer,
}

thread_local! {
    static COS_SIN_CACHE: RefCell<HashMap<(usize, usize, u32, usize, usize), CosSinEntry>> =
        RefCell::new(HashMap::new());
}

/// Compute cos/sin on CPU, H2D to device, and cache the device buffers.
/// On subsequent calls with the same key, returns cached pointers
/// without any H2D transfer — safe for RI capture.
fn get_or_compute_cos_sin(
    ctx: &AscendNewContext,
    seq_len: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    n_heads: usize,
) -> Result<(usize, usize)> {
    let key = (seq_len, head_dim, theta.to_bits(), position_offset, n_heads);

    // Check cache
    let cached = COS_SIN_CACHE.with(|c| c.borrow().contains_key(&key));
    if cached {
        return COS_SIN_CACHE.with(|c| {
            let map = c.borrow();
            let entry = map.get(&key).unwrap();
            let result = (entry.cos_ptr, entry.sin_ptr);
            Ok(result)
        });
    }

    // Compute cos/sin on CPU
    let half_dim = head_dim / 2;
    let mut cos_half = Vec::with_capacity(seq_len * half_dim);
    let mut sin_half = Vec::with_capacity(seq_len * half_dim);
    for s in 0..seq_len {
        let pos = (position_offset + s) as f64;
        for i in 0..half_dim {
            let freq = (theta as f64).powf(-2.0 * (i as f64) / (head_dim as f64));
            let angle = pos * freq;
            cos_half.push(angle.cos() as f32);
            sin_half.push(angle.sin() as f32);
        }
    }

    // cos_n1_fix_applied: aclnnRotaryPositionEmbedding expects cos/sin N axis = 1
    // (broadcast to all heads). Filling [seq, 1, head_dim] instead of [seq, n_heads, head_dim].
    // Build full head_dim cos/sin in HALF layout: [c0,c1,...,c127, c0,c1,...,c127]
    // This matches Gemma's `cat((freqs, freqs), dim=-1)` and ACL mode=0 (half).
    let total_elems = seq_len * head_dim;
    let mut cos_bf16: Vec<u16> = Vec::with_capacity(total_elems);
    let mut sin_bf16: Vec<u16> = Vec::with_capacity(total_elems);
    for s in 0..seq_len {
        // First half: c0, c1, ..., c(half_dim-1)
        for i in 0..half_dim {
            let c_bits = cos_half[s * half_dim + i].to_bits();
            let c_bf16 = (c_bits >> 16) as u16;
            cos_bf16.push(c_bf16);
        }
        // Second half: repeat c0, c1, ..., c(half_dim-1)
        for i in 0..half_dim {
            let c_bits = cos_half[s * half_dim + i].to_bits();
            let c_bf16 = (c_bits >> 16) as u16;
            cos_bf16.push(c_bf16);
        }
        // Same for sin
        for i in 0..half_dim {
            let s_bits = sin_half[s * half_dim + i].to_bits();
            let s_bf16 = (s_bits >> 16) as u16;
            sin_bf16.push(s_bf16);
        }
        for i in 0..half_dim {
            let s_bits = sin_half[s * half_dim + i].to_bits();
            let s_bf16 = (s_bits >> 16) as u16;
            sin_bf16.push(s_bf16);
        }
    }

    let cos_bytes_vec: Vec<u8> = bytemuck::cast_slice(&cos_bf16).to_vec();
    let sin_bytes_vec: Vec<u8> = bytemuck::cast_slice(&sin_bf16).to_vec();
    let cos_bytes_len = cos_bytes_vec.len();
    let sin_bytes_len = sin_bytes_vec.len();

    // H2D transfer (outside RI capture — warmup path).
    // Use session-aware alloc to avoid aclrtMemset during RI capture.
    let cos_buf = crate::l3::buffer::AscendBuffer::alloc_zeros(cos_bytes_len, ctx.device_id())?;
    let sin_buf = crate::l3::buffer::AscendBuffer::alloc_zeros(sin_bytes_len, ctx.device_id())?;
    cos_buf.copy_from_host(&cos_bytes_vec)?;
    sin_buf.copy_from_host(&sin_bytes_vec)?;

    let cos_ptr = cos_buf.raw_ptr();
    let sin_ptr = sin_buf.raw_ptr();

    // Cache the device buffers
    COS_SIN_CACHE.with(|c| {
        c.borrow_mut().insert(key, CosSinEntry {
            cos_ptr,
            sin_ptr,
            _cos_buf: cos_buf,
            _sin_buf: sin_buf,
        });
    });

    Ok((cos_ptr, sin_ptr))
}

// ── Implementation ────────────────────────────────────────────────────

/// RoPE: split QKV, apply rotary embedding, optionally write K/V to cache.
///
/// QKV layout: [seq_len, (q_heads + 2*kv_heads) * head_dim]  (2D contiguous)
/// Q output: [seq_len, q_heads, head_dim]  (contiguous)
/// K output: [seq_len, kv_heads, head_dim] (contiguous)
/// V output: [seq_len, kv_heads, head_dim] (contiguous)
///
/// QKV split uses `aclnnSplitWithSize` (single op, replaces per-token D2D loop).
/// Bias add uses `aclnnAdd` with broadcasting (replaces per-token add_inplace loop).
pub fn rope(ctx: &AscendNewContext, args: RopeArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.qkv.dtype();

    let qkv_dims = args.qkv.shape().dims();
    let seq_len = qkv_dims[0];
    let head_dim = args.head_dim;
    let kv_heads = args.kv_heads;
    let q_heads = args.q_heads;

    let elem_size = dtype.size_in_bytes();
    let q_cols = q_heads * head_dim;
    let kv_cols = kv_heads * head_dim;
    let total_cols = q_cols + 2 * kv_cols;

    let qkv_ptr = device_ptr(args.qkv)?;
    let q_out_ptr = device_ptr(args.q)?;
    let k_out_base = device_ptr(args.k)?;
    let v_out_base = device_ptr(args.v)?;

    // Apply kv_output_offset: when writing K/V into a persistent cache, the
    // caller passes the row offset where this call's K/V should be written
    // (e.g. after the prefix rows). cuda-new handles this in its ABI kernel;
    // ascend must shift the destination pointer manually. Without this, every
    // denoise step would overwrite the prefix K/V at the cache start instead
    // of appending after it.
    let kv_row_bytes = kv_cols * elem_size;
    let kv_offset_bytes = args.kv_output_offset * kv_row_bytes;
    let k_out_ptr = k_out_base + kv_offset_bytes;
    let v_out_ptr = v_out_base + kv_offset_bytes;

    // QKV is 2D [seq_len, total_cols], contiguous.
    // Split Q, K, V via aclnnSlice (3 ops, replaces 3*seq_len per-token D2D copies).
    let qkv_shape = Shape::new(vec![seq_len, total_cols]);
    super::aclnn_slice_2d(qkv_ptr, &qkv_shape, q_out_ptr, seq_len, q_cols, 0, dtype, stream)?;
    super::aclnn_slice_2d(qkv_ptr, &qkv_shape, k_out_ptr, seq_len, kv_cols, q_cols as i64, dtype, stream)?;
    super::aclnn_slice_2d(qkv_ptr, &qkv_shape, v_out_ptr, seq_len, kv_cols, (q_cols + kv_cols) as i64, dtype, stream)?;

    // Add bias if present (for SplitQkvBias semantic).
    // Uses aclnnExpand to tile bias [1, cols] → [seq_len, cols], then aclnnAdd.
    // This replaces the per-token add_inplace loop (was 2*seq_len ops/section → 3 ops/section).
    if let Some(bias) = args.bias {
        let bias_ptr = device_ptr(bias)?;
        add_inplace_2d(ctx, q_out_ptr, bias_ptr, seq_len, q_cols, dtype)?;
        add_inplace_2d(ctx, k_out_ptr, bias_ptr + q_cols * elem_size, seq_len, kv_cols, dtype)?;
        add_inplace_2d(ctx, v_out_ptr, bias_ptr + (q_cols + kv_cols) * elem_size, seq_len, kv_cols, dtype)?;
    }

    // Apply RoPE to Q (and K if SplitQkvRope)
    if args.semantic == RopeSemantic::SplitQkvRope {
        // cos/sin are cached on device — warmup pass computes + H2D,
        // capture pass reuses cached device pointers (no H2D during RI capture).
        let (cos_q_ptr, sin_q_ptr) = get_or_compute_cos_sin(
            ctx, seq_len, head_dim, args.theta, args.position_offset, q_heads,
        )?;
        let (cos_k_ptr, sin_k_ptr) = get_or_compute_cos_sin(
            ctx, seq_len, head_dim, args.theta, args.position_offset, kv_heads,
        )?;

        let cos_q_shape = Shape::new(vec![seq_len, 1, head_dim]);
        let sin_q_shape = Shape::new(vec![seq_len, 1, head_dim]);
        let q_shape = Shape::new(vec![seq_len, q_heads, head_dim]);
        let acl_q = crate::acl_tensor::AclTensor::from_device(q_out_ptr, &q_shape, dtype);
        let acl_cos_q = crate::acl_tensor::AclTensor::from_device(cos_q_ptr, &cos_q_shape, dtype);
        let acl_sin_q = crate::acl_tensor::AclTensor::from_device(sin_q_ptr, &sin_q_shape, dtype);

        crate::acl_tensor::run_op(
            |ws, exec| unsafe {
                crate::ffi::aclnn::aclnnRotaryPositionEmbeddingGetWorkspaceSize(
                    acl_q.raw(),
                    acl_cos_q.raw(),
                    acl_sin_q.raw(),
                    0,
                    acl_q.raw(),
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe {
                crate::ffi::aclnn::aclnnRotaryPositionEmbedding(ws, ws_size, exec, s)
            },
            stream,
            "aclnnRotaryPositionEmbedding(Q)",
        )
        .map_err(|e| Error::Other(format!("rope Q: {e}")))?;

        let cos_k_shape = Shape::new(vec![seq_len, 1, head_dim]);
        let sin_k_shape = Shape::new(vec![seq_len, 1, head_dim]);
        let k_shape = Shape::new(vec![seq_len, kv_heads, head_dim]);
        let acl_k = crate::acl_tensor::AclTensor::from_device(k_out_ptr, &k_shape, dtype);
        let acl_cos_k = crate::acl_tensor::AclTensor::from_device(cos_k_ptr, &cos_k_shape, dtype);
        let acl_sin_k = crate::acl_tensor::AclTensor::from_device(sin_k_ptr, &sin_k_shape, dtype);

        crate::acl_tensor::run_op(
            |ws, exec| unsafe {
                crate::ffi::aclnn::aclnnRotaryPositionEmbeddingGetWorkspaceSize(
                    acl_k.raw(),
                    acl_cos_k.raw(),
                    acl_sin_k.raw(),
                    0,
                    acl_k.raw(),
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe {
                crate::ffi::aclnn::aclnnRotaryPositionEmbedding(ws, ws_size, exec, s)
            },
            stream,
            "aclnnRotaryPositionEmbedding(K)",
        )
        .map_err(|e| Error::Other(format!("rope K: {e}")))?;
    }

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────

fn device_ptr(tensor: &Tensor) -> Result<usize> {
    match tensor.storage() {
        Storage::Gpu { handle, .. } => Ok(handle.ptr()),
        _ => Err(Error::Other(format!(
            "expected NPU tensor, got {:?}",
            tensor.device()
        ))),
    }
}

/// Element-wise add: dst[seq_len, cols] += bias_tiled[seq_len, cols].
/// Uses aclnnExpand to tile bias [1, cols] → [seq_len, cols], then aclnnInplaceAdd.
/// 2 ops per section (expand + inplace_add), no copy-back needed.
fn add_inplace_2d(
    ctx: &AscendNewContext,
    dst: usize,
    bias: usize,
    seq_len: usize,
    cols: usize,
    dtype: DType,
) -> Result<()> {
    let elem_size = dtype.size_in_bytes();
    let numel = seq_len * cols;
    let bytes = numel * elem_size;
    if bytes == 0 {
        return Ok(());
    }

    let out_shape = Shape::new(vec![seq_len, cols]);
    let bias_shape = Shape::new(vec![1, cols]);

    // Step 1: Expand bias [1, cols] → bias_tiled [seq_len, cols]
    let bias_tiled_buf = crate::l3::buffer::AscendBuffer::alloc_zeros_session_aware(bytes, ctx.device_id())?;
    let bias_tiled_ptr = bias_tiled_buf.raw_ptr();

    let acl_bias = crate::acl_tensor::AclTensor::from_device(bias, &bias_shape, dtype);
    let acl_bias_tiled = crate::acl_tensor::AclTensor::from_device(bias_tiled_ptr, &out_shape, dtype);
    let acl_expand_shape = crate::acl_tensor::AclIntArray::new(&[seq_len as i64, cols as i64]);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnExpandGetWorkspaceSize(
                acl_bias.raw(),
                acl_expand_shape.raw(),
                acl_bias_tiled.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnExpand(ws, ws_size, exec, s)
        },
        ctx.raw_stream(),
        "aclnnExpand (rope bias tile)",
    )
    .map_err(|e| Error::Other(format!("rope add_inplace_2d expand: {e}")))?;

    // Step 2: dst += bias_tiled via aclnnInplaceAdd (no copy-back needed)
    super::aclnn_inplace_add(
        dst,
        bias_tiled_ptr,
        &out_shape,
        &out_shape,
        dtype,
        ctx.raw_stream(),
    )
    .map_err(|e| Error::Other(format!("rope add_inplace_2d: {e}")))?;
    Ok(())
}
