//! Qwen3.5 hybrid linear-attention kernel contracts (bf16).

use apxinf_core::{Error, Result};

use super::contracts::check_cuda;
use crate::buffer::{CudaBuffer, HostMappedBuffer};
use crate::context::CudaContext;
use crate::ffi;

/// Argmax over `count` bf16 logits; writes the winning index to the mapped
/// buffer. The caller synchronizes before reading it back.
pub fn argmax_bf16(
    ctx: &CudaContext,
    logits: &CudaBuffer,
    count: usize,
    out: &HostMappedBuffer,
) -> Result<()> {
    let bytes = count
        .checked_mul(2)
        .ok_or_else(|| Error::Other("argmax_bf16: overflow".into()))?;
    if logits.len() < bytes || out.len() < 4 {
        return Err(Error::Other("argmax_bf16: buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_argmax_bf16(
            logits.ptr(),
            count as u32,
            out.address().ptr(),
            ctx.stream().handle(),
        )
    })
}
/// MLP activation: `out = silu(gate) * up` elementwise over `count` bf16.
pub fn silu_mul(
    ctx: &CudaContext,
    gate: &CudaBuffer,
    up: &CudaBuffer,
    out: &CudaBuffer,
    count: usize,
) -> Result<()> {
    let bytes = count
        .checked_mul(2)
        .ok_or_else(|| Error::Other("silu_mul: overflow".into()))?;
    if gate.len() < bytes || up.len() < bytes || out.len() < bytes {
        return Err(Error::Other("silu_mul: buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_silu_mul(
            gate.ptr(),
            up.ptr(),
            out.ptr(),
            count as i64,
            ctx.stream().handle(),
        )
    })
}
/// Causal depthwise conv + SiLU with carry state.
/// `input` is `[seq, channels]` bf16, `weight` `[channels, kernel]` bf16,
/// `state` `[(kernel-1), channels]` f32 (oldest..newest, updated in place).
/// Writes `[seq, channels]` bf16 to `output`.
pub fn conv_silu(
    ctx: &CudaContext,
    input: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    state: &mut CudaBuffer,
    seq: usize,
    channels: usize,
    kernel: usize,
) -> Result<()> {
    if seq == 0 || channels == 0 || kernel <= 1 {
        return Err(Error::Other("conv_silu: invalid dimensions".into()));
    }
    let bytes_in = seq
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("conv_silu: input overflow".into()))?;
    let bytes_w = channels
        .checked_mul(kernel)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("conv_silu: weight overflow".into()))?;
    let bytes_state = (kernel - 1)
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| Error::Other("conv_silu: state overflow".into()))?;
    if input.len() < bytes_in || weight.len() < bytes_w || state.len() < bytes_state {
        return Err(Error::Other("conv_silu: buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_conv_silu(
            input.ptr(),
            weight.ptr(),
            output.ptr(),
            state.ptr(),
            seq as i32,
            channels as i32,
            kernel as i32,
            ctx.stream().handle(),
        )
    })
}

/// Gated delta-rule recurrence (one launch sweeps the whole sequence).
///
/// `qkv` is `[seq, k_heads*kdim*2 + v_heads*vdim]` bf16 (post-conv);
/// `a`/`b` are `[seq, v_heads]` bf16; `a_log`/`dt_bias` are `[v_heads]` bf16;
/// `recurrent` is `[v_heads, kdim, vdim]` f32, updated in place; `out` is
/// `[seq, v_heads*vdim]` bf16.
#[allow(clippy::too_many_arguments)]
pub fn delta_step(
    ctx: &CudaContext,
    qkv: &CudaBuffer,
    a: &CudaBuffer,
    b: &CudaBuffer,
    a_log: &CudaBuffer,
    dt_bias: &CudaBuffer,
    recurrent: &mut CudaBuffer,
    out: &CudaBuffer,
    seq: usize,
    k_heads: usize,
    v_heads: usize,
    kdim: usize,
    vdim: usize,
) -> Result<()> {
    if seq == 0 || k_heads == 0 || v_heads == 0 || kdim == 0 || vdim == 0 || vdim % 32 != 0 {
        return Err(Error::Other("delta_step: invalid dimensions".into()));
    }
    let row = k_heads * kdim * 2 + v_heads * vdim;
    let bytes_qkv = seq
        .checked_mul(row)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("delta_step: qkv overflow".into()))?;
    let bytes_ab = seq
        .checked_mul(v_heads)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("delta_step: a/b overflow".into()))?;
    let bytes_heads = v_heads
        .checked_mul(2)
        .ok_or_else(|| Error::Other("delta_step: heads overflow".into()))?;
    let bytes_state = v_heads
        .checked_mul(kdim)
        .and_then(|v| v.checked_mul(vdim))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| Error::Other("delta_step: state overflow".into()))?;
    let bytes_out = seq
        .checked_mul(v_heads)
        .and_then(|v| v.checked_mul(vdim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("delta_step: output overflow".into()))?;
    if qkv.len() < bytes_qkv
        || a.len() < bytes_ab
        || b.len() < bytes_ab
        || a_log.len() < bytes_heads
        || dt_bias.len() < bytes_heads
        || recurrent.len() < bytes_state
        || out.len() < bytes_out
    {
        return Err(Error::Other("delta_step: buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_delta_step(
            qkv.ptr(),
            a.ptr(),
            b.ptr(),
            a_log.ptr(),
            dt_bias.ptr(),
            recurrent.ptr(),
            out.ptr(),
            seq as i32,
            k_heads as i32,
            v_heads as i32,
            kdim as i32,
            vdim as i32,
            ctx.stream().handle(),
        )
    })
}

/// Gated RMSNorm: `out = rms_norm(input) * weight * silu(z)` per head row.
pub fn gated_norm(
    ctx: &CudaContext,
    input: &CudaBuffer,
    z: &CudaBuffer,
    weight: &CudaBuffer,
    out: &CudaBuffer,
    seq: usize,
    v_heads: usize,
    vdim: usize,
    eps: f32,
) -> Result<()> {
    if seq == 0 || v_heads == 0 || vdim == 0 || !eps.is_finite() {
        return Err(Error::Other("gated_norm: invalid dimensions".into()));
    }
    let bytes_row = seq
        .checked_mul(v_heads)
        .and_then(|v| v.checked_mul(vdim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("gated_norm: row overflow".into()))?;
    let bytes_w = vdim
        .checked_mul(2)
        .ok_or_else(|| Error::Other("gated_norm: weight overflow".into()))?;
    if input.len() < bytes_row || z.len() < bytes_row || out.len() < bytes_row || weight.len() < bytes_w {
        return Err(Error::Other("gated_norm: buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_gated_norm(
            input.ptr(),
            z.ptr(),
            weight.ptr(),
            out.ptr(),
            seq as i32,
            v_heads as i32,
            vdim as i32,
            eps,
            ctx.stream().handle(),
        )
    })
}

/// Split q_proj output into q/gate, RMSNorm q per head, apply partial RoPE.
#[allow(clippy::too_many_arguments)]
pub fn q_split_norm_rope(
    ctx: &CudaContext,
    q_gate: &CudaBuffer,
    q_norm_w: &CudaBuffer,
    q_out: &CudaBuffer,
    gate_out: &CudaBuffer,
    seq: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    start_pos: u32,
) -> Result<()> {
    if seq == 0 || heads == 0 || head_dim == 0 || rotary_dim == 0 || !theta.is_finite() {
        return Err(Error::Other("q_split_norm_rope: invalid dimensions".into()));
    }
    let bytes_qg = seq
        .checked_mul(heads)
        .and_then(|v| v.checked_mul(head_dim * 2))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("q_split_norm_rope: q_gate overflow".into()))?;
    let bytes_q = seq
        .checked_mul(heads)
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("q_split_norm_rope: q overflow".into()))?;
    let bytes_w = head_dim
        .checked_mul(2)
        .ok_or_else(|| Error::Other("q_split_norm_rope: weight overflow".into()))?;
    if q_gate.len() < bytes_qg
        || q_out.len() < bytes_q
        || gate_out.len() < bytes_q
        || q_norm_w.len() < bytes_w
    {
        return Err(Error::Other("q_split_norm_rope: buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_q_split_norm_rope(
            q_gate.ptr(),
            q_norm_w.ptr(),
            q_out.ptr(),
            gate_out.ptr(),
            seq as i32,
            heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            theta,
            start_pos,
            ctx.stream().handle(),
        )
    })
}

/// RMSNorm k per head, apply partial RoPE, append into the K cache at
/// positions `start_pos .. start_pos + seq`.
#[allow(clippy::too_many_arguments)]
pub fn k_norm_rope_append(
    ctx: &CudaContext,
    k_in: &CudaBuffer,
    k_norm_w: &CudaBuffer,
    k_cache: &mut CudaBuffer,
    seq: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    start_pos: u32,
    max_seq_len: usize,
) -> Result<()> {
    if seq == 0 || n_kv_heads == 0 || head_dim == 0 || rotary_dim == 0 || max_seq_len == 0 {
        return Err(Error::Other("k_norm_rope_append: invalid dimensions".into()));
    }
    let bytes_k = seq
        .checked_mul(n_kv_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("k_norm_rope_append: k overflow".into()))?;
    let bytes_w = head_dim
        .checked_mul(2)
        .ok_or_else(|| Error::Other("k_norm_rope_append: weight overflow".into()))?;
    let bytes_cache = n_kv_heads
        .checked_mul(max_seq_len)
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("k_norm_rope_append: cache overflow".into()))?;
    if k_in.len() < bytes_k || k_cache.len() < bytes_cache || k_norm_w.len() < bytes_w {
        return Err(Error::Other("k_norm_rope_append: buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_k_norm_rope_append(
            k_in.ptr(),
            k_norm_w.ptr(),
            k_cache.ptr(),
            seq as i32,
            n_kv_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            theta,
            start_pos,
            max_seq_len as i32,
            ctx.stream().handle(),
        )
    })
}

/// `out = sigmoid(gate) * x` elementwise bf16.
pub fn sigmoid_mul(
    ctx: &CudaContext,
    gate: &CudaBuffer,
    x: &CudaBuffer,
    out: &CudaBuffer,
    count: usize,
) -> Result<()> {
    let bytes = count
        .checked_mul(2)
        .ok_or_else(|| Error::Other("sigmoid_mul: overflow".into()))?;
    if gate.len() < bytes || x.len() < bytes || out.len() < bytes {
        return Err(Error::Other("sigmoid_mul: buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_sigmoid_mul(
            gate.ptr(),
            x.ptr(),
            out.ptr(),
            count as i64,
            ctx.stream().handle(),
        )
    })
}

/// Causal GQA flash attention for the full-attention prefill.
#[allow(clippy::too_many_arguments)]
pub fn flash_prefill(
    ctx: &CudaContext,
    q: &CudaBuffer,
    k_cache: &CudaBuffer,
    v_cache: &CudaBuffer,
    out: &CudaBuffer,
    seq: usize,
    heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    start_pos: u32,
    max_seq_len: usize,
) -> Result<()> {
    if seq == 0 || heads == 0 || n_kv_heads == 0 || heads % n_kv_heads != 0 || head_dim == 0 {
        return Err(Error::Other("flash_prefill: invalid dimensions".into()));
    }
    let bytes_q = seq
        .checked_mul(heads)
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("flash_prefill: q overflow".into()))?;
    let bytes_cache = n_kv_heads
        .checked_mul(max_seq_len)
        .and_then(|v| v.checked_mul(head_dim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("flash_prefill: cache overflow".into()))?;
    if q.len() < bytes_q || k_cache.len() < bytes_cache || v_cache.len() < bytes_cache || out.len() < bytes_q {
        return Err(Error::Other("flash_prefill: buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_flash_prefill(
            q.ptr(),
            k_cache.ptr(),
            v_cache.ptr(),
            out.ptr(),
            seq as i32,
            heads as i32,
            n_kv_heads as i32,
            head_dim as i32,
            scale,
            start_pos,
            max_seq_len as i32,
            ctx.stream().handle(),
        )
    })
}
