//! Attention operations — FAS (FlashAttentionScore).
//!
//! Uses `aclnnFlashAttentionScore` for all attention computation.
//! PFA (PromptFlashAttention) has no compiled kernels on 950PR (CANN 9.1.0),
//! but FAS has 1428+ tiling key variants available.
//!
//! Key requirement: softmaxMaxOut/softmaxSumOut must be [batch, heads, seq, 8] FP32.

use apxinf_core::{DType, Error, Result, Shape, Storage, Tensor};

use crate::l3::context::AscendNewContext;

// ── Args structs ──────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct AttentionPolicy {
    pub workspace_limit: usize,
    pub online_tune: bool,
    pub allow_fallback: bool,
    pub graph_safe: bool,
    pub deterministic: bool,
    pub cache_dir: Option<String>,
}

impl Default for AttentionPolicy {
    fn default() -> Self {
        Self {
            workspace_limit: 256 * 1024 * 1024,
            online_tune: false,
            allow_fallback: true,
            graph_safe: true,
            deterministic: false,
            cache_dir: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionMask {
    None,
    Causal,
}

pub struct AttentionArgs<'a> {
    pub query: &'a Tensor,
    pub key: &'a Tensor,
    pub value: &'a Tensor,
    pub out: &'a mut Tensor,
    pub mask: AttentionMask,
    pub scale: f32,
    pub output_scale: f32,
    pub policy: AttentionPolicy,
}

impl<'a> AttentionArgs<'a> {
    pub fn new(query: &'a Tensor, key: &'a Tensor, value: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            query,
            key,
            value,
            out,
            mask: AttentionMask::None,
            scale: 0.0, // 0 = auto (1/sqrt(head_dim))
            output_scale: 1.0,
            policy: AttentionPolicy::default(),
        }
    }
}

// ── Implementation ────────────────────────────────────────────────────

/// Attention via `aclnnFlashAttentionScore`.
///
/// Input layout: BSND (B=batch, S=seq, N=heads, D=head_dim).
/// FAS on 950PR supports head_dim=72 natively (no padding needed).
///
/// Required softmax output tensors:
/// - softmaxMaxOut: [batch, heads, seq, 8] FP32
/// - softmaxSumOut: [batch, heads, seq, 8] FP32
/// - softmaxOutOut: same shape as Q (optional, can be NULL)
pub fn attention(ctx: &AscendNewContext, args: AttentionArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.query.dtype();

    let q_dims = args.query.shape().dims();
    let kv_dims = args.key.shape().dims();

    // Parse Q shape: [batch, seq, heads, head_dim] (BSND layout)
    let (batch, seq_len, n_heads, head_dim) = match q_dims.len() {
        4 => (q_dims[0], q_dims[1], q_dims[2], q_dims[3]),
        3 => (1, q_dims[0], q_dims[1], q_dims[2]), // [seq, heads, dim] → B=1
        _ => return Err(Error::Other(format!(
            "attention: unexpected Q rank {}, dims={:?}", q_dims.len(), q_dims
        ))),
    };
    let n_kv_heads = match kv_dims.len() {
        4 => kv_dims[2],
        3 => kv_dims[1],
        _ => return Err(Error::Other(format!(
            "attention: unexpected K rank {}, dims={:?}", kv_dims.len(), kv_dims
        ))),
    };

    let scale_value = if args.scale > 0.0 {
        args.scale as f64
    } else {
        1.0 / (head_dim as f64).sqrt()
    };

    // sparseMode: 0=none, 2=causal
    let sparse_mode = match args.mask {
        AttentionMask::None => 0i64,
        AttentionMask::Causal => 2i64,
    };

    // Create ACL tensors for Q, K, V, Out
    let acl_q = acl_tensor(args.query, dtype);
    let acl_k = acl_tensor(args.key, dtype);
    let acl_v = acl_tensor(args.value, dtype);
    let acl_out = acl_tensor(args.out, dtype);

    // Allocate softmax output buffers
    // softmaxMaxOut and softmaxSumOut: [batch, heads, seq, 8] FP32
    let sm_elem_count = batch * n_heads * seq_len * 8;
    let sm_bytes = sm_elem_count * 4; // FP32 = 4 bytes
    let sm_max_buf = crate::l3::buffer::AscendBuffer::alloc_zeros_session_aware(sm_bytes, ctx.device_id())?;
    let sm_sum_buf = crate::l3::buffer::AscendBuffer::alloc_zeros_session_aware(sm_bytes, ctx.device_id())?;

    let sm_shape = Shape::new(vec![batch, n_heads, seq_len, 8]);
    let acl_sm_max = crate::acl_tensor::AclTensor::from_device(
        sm_max_buf.raw_ptr(), &sm_shape, DType::F32,
    );
    let acl_sm_sum = crate::acl_tensor::AclTensor::from_device(
        sm_sum_buf.raw_ptr(), &sm_shape, DType::F32,
    );

    let layout = std::ffi::CString::new("BSND").unwrap();

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnFlashAttentionScoreGetWorkspaceSize(
                acl_q.raw(),
                acl_k.raw(),
                acl_v.raw(),
                std::ptr::null(), // realShift
                std::ptr::null(), // dropMask
                std::ptr::null(), // paddingMask
                std::ptr::null(), // attenMask
                std::ptr::null(), // prefix
                scale_value,
                1.0,              // keepProb
                seq_len as i64,   // preTokens
                seq_len as i64,   // nextTokens
                n_heads as i64,   // headNum
                layout.as_ptr(),  // inputLayout
                0,                // innerPrecise
                sparse_mode,      // sparseMode
                acl_sm_max.raw(), // softmaxMaxOut
                acl_sm_sum.raw(), // softmaxSumOut
                std::ptr::null(), // softmaxOutOut (optional)
                acl_out.raw(),    // attentionOutOut
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnFlashAttentionScore(ws, ws_size, exec, s)
        },
        stream,
        "aclnnFlashAttentionScore",
    )
    .map_err(|e| Error::Other(format!("attention: {e}")))?;

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

fn acl_tensor(tensor: &Tensor, dtype: apxinf_core::DType) -> crate::acl_tensor::AclTensor {
    crate::acl_tensor::AclTensor::from_device(
        device_ptr(tensor).unwrap_or(0),
        tensor.shape(),
        dtype,
    )
}
