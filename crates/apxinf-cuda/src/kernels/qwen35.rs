//! Qwen3.5 hybrid linear-attention kernel contracts (bf16).

use apxinf_core::{DType, Error, Result};

use super::contracts::check_cuda;
use crate::buffer::{CudaBuffer, CudaDeviceAddress, HostMappedBuffer};
use crate::context::CudaContext;
use crate::ffi;

fn kernel_prof(ctx: &CudaContext, name: &str) -> Option<ffi::cudaEvent_t> {
    if std::env::var_os("APXINF_KERNEL_PROF").is_none() {
        return None;
    }
    let mut e0: ffi::cudaEvent_t = std::ptr::null_mut();
    unsafe {
        ffi::check_cuda(ffi::cudaEventCreate(&mut e0)).ok()?;
        ffi::check_cuda(ffi::cudaEventRecord(e0, ctx.stream().handle())).ok()?;
    }
    Some(e0)
}

fn kernel_prof_end(ctx: &CudaContext, name: &str, e0: ffi::cudaEvent_t) {
    let mut e1: ffi::cudaEvent_t = std::ptr::null_mut();
    unsafe {
        if ffi::check_cuda(ffi::cudaEventCreate(&mut e1)).is_err() {
            return;
        }
        ffi::check_cuda(ffi::cudaEventRecord(e1, ctx.stream().handle())).ok();
        ffi::check_cuda(ffi::cudaEventSynchronize(e1)).ok();
        let mut ms: f32 = 0.0;
        ffi::check_cuda(ffi::cudaEventElapsedTime(&mut ms, e0, e1)).ok();
        if ms > 0.05 {
            eprintln!("[kernel] {name} : {ms:.3} ms");
        }
        ffi::check_cuda(ffi::cudaEventDestroy(e0)).ok();
        ffi::check_cuda(ffi::cudaEventDestroy(e1)).ok();
    }
}

/// Exact one-launch argmax over `count` bf16 logits. The bounded geometry is
/// part of the ABI contract; callers retain [`argmax_bf16_parallel`] fallback.
pub fn argmax_bf16_single_launch(
    ctx: &CudaContext,
    logits: &CudaBuffer,
    count: usize,
    partials: &CudaBuffer,
    arrivals: &CudaBuffer,
    out: &HostMappedBuffer,
) -> Result<()> {
    argmax_bf16_single_launch_at(ctx, logits, count, partials, arrivals, out, 0)
}

/// Exact one-launch argmax into an indexed mapped-host result slot.
pub fn argmax_bf16_single_launch_at(
    ctx: &CudaContext,
    logits: &CudaBuffer,
    count: usize,
    partials: &CudaBuffer,
    arrivals: &CudaBuffer,
    out: &HostMappedBuffer,
    index: usize,
) -> Result<()> {
    const PARTIAL_CAPACITY: usize = 128;
    const PARTIAL_BYTES: usize = 8;
    let bytes = count
        .checked_mul(2)
        .ok_or_else(|| Error::Other("argmax_bf16_single_launch: overflow".into()))?;
    let count = u32::try_from(count)
        .map_err(|_| Error::Other("argmax_bf16_single_launch: count exceeds u32".into()))?;
    let byte_offset = index
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| Error::Other("argmax result offset overflow".into()))?;
    let out_address = out
        .address_at(byte_offset, std::mem::size_of::<u32>())
        .map_err(Error::Cuda)?;
    if count == 0 || logits.len() < bytes
        || partials.len() < PARTIAL_CAPACITY * PARTIAL_BYTES
        || arrivals.len() < 4 {
        return Err(Error::Other("argmax_bf16_single_launch: invalid buffers".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_argmax_bf16_single_launch(
            logits.ptr(), count, partials.ptr(), PARTIAL_CAPACITY as u32,
            arrivals.ptr(), out_address.ptr(), ctx.stream().handle(),
        )
    })
}

/// Exact multi-block argmax over `count` bf16 logits. The caller synchronizes
/// before reading the mapped output.
pub fn argmax_bf16_parallel(
    ctx: &CudaContext,
    logits: &CudaBuffer,
    count: usize,
    partials: &CudaBuffer,
    out: &HostMappedBuffer,
) -> Result<()> {
    const PARTIAL_CAPACITY: usize = 128;
    const PARTIAL_BYTES: usize = 8;
    let bytes = count
        .checked_mul(2)
        .ok_or_else(|| Error::Other("argmax_bf16_parallel: overflow".into()))?;
    let count = u32::try_from(count)
        .map_err(|_| Error::Other("argmax_bf16_parallel: count exceeds u32".into()))?;
    if count == 0 || logits.len() < bytes
        || partials.len() < PARTIAL_CAPACITY * PARTIAL_BYTES || out.len() < 4 {
        return Err(Error::Other("argmax_bf16_parallel: invalid buffers".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_argmax_bf16_parallel(
            logits.ptr(), count, partials.ptr(), PARTIAL_CAPACITY as u32,
            out.address().ptr(), ctx.stream().handle(),
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
    let prof_ev = kernel_prof(ctx, "silu_mul");
    let prof_result = check_cuda(unsafe {
        ffi::apxinf_qwen35_silu_mul(
            gate.ptr(),
            up.ptr(),
            out.ptr(),
            count as i64,
            ctx.stream().handle(),
        )
    });
    if let Some(ev) = prof_ev {
        kernel_prof_end(ctx, "silu_mul", ev);
    }
    prof_result
}
/// Prepare the unchanged delta recurrence launch before CUDA graph capture.
/// This performs the one-time dynamic shared-memory opt-in without enqueuing
/// work or mutating model state.
pub fn prepare_exact_gdn() -> Result<()> {
    check_cuda(unsafe { ffi::apxinf_qwen35_prepare_delta_step() })
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
    let prof_ev = kernel_prof(ctx, "conv_silu");
    let prof_result = check_cuda(unsafe {
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
    });
    if let Some(ev) = prof_ev {
        kernel_prof_end(ctx, "conv_silu", ev);
    }
    prof_result
}

/// Normalizes q/k per (token, k_head) in a parallel prepass; `qk_out` is
/// `[seq, k_heads, 2, kdim]` bf16.
#[allow(clippy::too_many_arguments)]
pub fn delta_norm_prepass(
    ctx: &CudaContext,
    qkv: &CudaBuffer,
    qk_out: &CudaBuffer,
    seq: usize,
    k_heads: usize,
    v_heads: usize,
    kdim: usize,
    vdim: usize,
) -> Result<()> {
    let required = seq
        .checked_mul(k_heads)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_mul(kdim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("delta_norm_prepass size overflow".into()))?;
    if qk_out.len() < required {
        return Err(Error::Other("delta_norm_prepass: qk_out buffer too small".into()));
    }
    let prof_ev = kernel_prof(ctx, "delta_norm_prepass");
    let prof_result = check_cuda(unsafe {
        ffi::apxinf_qwen35_delta_norm_prepass(
            qkv.ptr(),
            qk_out.ptr(),
            seq as i32,
            k_heads as i32,
            v_heads as i32,
            kdim as i32,
            vdim as i32,
            ctx.stream().handle(),
        )
    });
    if let Some(ev) = prof_ev {
        kernel_prof_end(ctx, "delta_norm_prepass", ev);
    }
    prof_result
}

/// Gated delta-rule recurrence (one launch sweeps the whole sequence).
///
/// `qkv` is `[seq, k_heads*kdim*2 + v_heads*vdim]` bf16 (post-conv);
/// `qk_norm` is the normalized `[seq, k_heads, 2, kdim]` q/k from the
/// prepass; `a`/`b` are `[seq, v_heads]` bf16; `a_log`/`dt_bias` are
/// `[v_heads]` bf16; `recurrent` is `[v_heads, kdim, vdim]` f32, updated in
/// place; `out` is `[seq, v_heads*vdim]` bf16.
#[allow(clippy::too_many_arguments)]
pub fn delta_step(
    ctx: &CudaContext,
    qkv: &CudaBuffer,
    qk_norm: &CudaBuffer,
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
    let bytes_qk = seq
        .checked_mul(k_heads)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_mul(kdim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("delta_step: qk_norm overflow".into()))?;
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
        || qk_norm.len() < bytes_qk
        || a.len() < bytes_ab
        || b.len() < bytes_ab
        || a_log.len() < bytes_heads
        || dt_bias.len() < bytes_heads
        || recurrent.len() < bytes_state
        || out.len() < bytes_out
    {
        return Err(Error::Other("delta_step: buffer too small".into()));
    }
    let prof_ev = kernel_prof(ctx, "delta_step");
    let prof_result = check_cuda(unsafe {
        ffi::apxinf_qwen35_delta_step(
            qkv.ptr(),
            qk_norm.ptr(),
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
    });
    if let Some(ev) = prof_ev {
        kernel_prof_end(ctx, "delta_step", ev);
    }
    prof_result
}

/// Prefill-only delta recurrence for Qwen3.5's exact 128x128 head geometry.
/// Four independent 32-value tiles own disjoint recurrent-state columns and
/// each tile visits tokens serially, preserving the causal update order.
#[allow(clippy::too_many_arguments)]
pub fn prefill_delta_step(
    ctx: &CudaContext,
    qkv: &CudaBuffer,
    qk_norm: &CudaBuffer,
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
    if !(2..=512).contains(&seq)
        || k_heads == 0
        || v_heads == 0
        || v_heads % k_heads != 0
        || kdim != 128
        || vdim != 128
    {
        return Err(Error::Other("prefill_delta_step: unsupported dimensions".into()));
    }
    let row = k_heads
        .checked_mul(kdim)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_add(v_heads.checked_mul(vdim)?))
        .ok_or_else(|| Error::Other("prefill_delta_step: row overflow".into()))?;
    let qkv_bytes = seq
        .checked_mul(row)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("prefill_delta_step: qkv overflow".into()))?;
    let qk_bytes = seq
        .checked_mul(k_heads)
        .and_then(|v| v.checked_mul(2 * kdim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("prefill_delta_step: qk overflow".into()))?;
    let scalar_bytes = seq
        .checked_mul(v_heads)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("prefill_delta_step: scalar overflow".into()))?;
    let value_bytes = seq
        .checked_mul(v_heads)
        .and_then(|v| v.checked_mul(vdim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("prefill_delta_step: value overflow".into()))?;
    let state_bytes = v_heads
        .checked_mul(kdim)
        .and_then(|v| v.checked_mul(vdim))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| Error::Other("prefill_delta_step: state overflow".into()))?;
    if qkv.len() < qkv_bytes
        || qk_norm.len() < qk_bytes
        || a.len() < scalar_bytes
        || b.len() < scalar_bytes
        || a_log.len() < v_heads * 2
        || dt_bias.len() < v_heads * 2
        || recurrent.len() < state_bytes
        || out.len() < value_bytes
    {
        return Err(Error::Other("prefill_delta_step: buffer too small".into()));
    }
    let prof_ev = kernel_prof(ctx, "prefill_delta_step");
    let prof_result = check_cuda(unsafe {
        ffi::apxinf_qwen35_prefill_delta_step(
            qkv.ptr(),
            qk_norm.ptr(),
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
    });
    if let Some(ev) = prof_ev {
        kernel_prof_end(ctx, "prefill_delta_step", ev);
    }
    prof_result
}

/// Prepare the opt-in 4-warp shared-work prefill recurrence.
pub fn prepare_prefill_delta_step_4w() -> Result<bool> {
    let mut supported = 0;
    check_cuda(unsafe {
        ffi::apxinf_qwen35_prepare_prefill_delta_step_4w(&mut supported)
    })?;
    Ok(supported != 0)
}

/// One CTA/value-head, four warps owning the historical 32-column V tiles.
#[allow(clippy::too_many_arguments)]
pub fn prefill_delta_step_4w(
    ctx: &CudaContext,
    qkv: &CudaBuffer,
    qk_norm: &CudaBuffer,
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
    if !(2..=512).contains(&seq) || kdim != 128 || vdim != 128
        || k_heads == 0 || v_heads == 0 || v_heads % k_heads != 0
    {
        return Err(Error::Other("prefill 4w unsupported dimensions".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_prefill_delta_step_4w(
            qkv.ptr(), qk_norm.ptr(), a.ptr(), b.ptr(), a_log.ptr(),
            dt_bias.ptr(), recurrent.ptr(), out.ptr(), seq as i32,
            k_heads as i32, v_heads as i32, kdim as i32, vdim as i32,
            ctx.stream().handle(),
        )
    })
}

/// Prepare the opt-in two-warp shared-work prefill recurrence.
pub fn prepare_prefill_delta_step_2w() -> Result<bool> {
    let mut supported = 0;
    check_cuda(unsafe {
        ffi::apxinf_qwen35_prepare_prefill_delta_step_2w(&mut supported)
    })?;
    Ok(supported != 0)
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_delta_step_2w(
    ctx: &CudaContext,
    qkv: &CudaBuffer,
    qk_norm: &CudaBuffer,
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
    if !(2..=512).contains(&seq) || kdim != 128 || vdim != 128
        || k_heads == 0 || v_heads == 0 || v_heads % k_heads != 0
    {
        return Err(Error::Other("prefill 2w unsupported dimensions".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_prefill_delta_step_2w(
            qkv.ptr(), qk_norm.ptr(), a.ptr(), b.ptr(), a_log.ptr(),
            dt_bias.ptr(), recurrent.ptr(), out.ptr(), seq as i32,
            k_heads as i32, v_heads as i32, kdim as i32, vdim as i32,
            ctx.stream().handle(),
        )
    })
}


/// Preflight the exact fused GDN tail before causal convolution mutates qkv
/// and conv state. Unsupported dynamic-shared-memory geometry returns false
/// so the caller can use the unchanged eager sequence.
pub fn prepare_norm_delta_gated() -> Result<bool> {
    let mut supported = 0;
    check_cuda(unsafe { ffi::apxinf_qwen35_prepare_norm_delta_gated(&mut supported) })?;
    Ok(supported != 0)
}

/// Exact fusion of q/k normalization, delta recurrence, and gated RMSNorm for
/// Qwen3.5's 128-wide heads. `qk_out` and `delta_out` are still materialized
/// through bf16 exactly as in the eager kernels.
#[allow(clippy::too_many_arguments)]
pub fn norm_delta_gated(
    ctx: &CudaContext,
    qkv: &CudaBuffer,
    qk_out: &CudaBuffer,
    a: &CudaBuffer,
    b: &CudaBuffer,
    a_log: &CudaBuffer,
    dt_bias: &CudaBuffer,
    z: &CudaBuffer,
    weight: &CudaBuffer,
    recurrent: &mut CudaBuffer,
    delta_out: &CudaBuffer,
    out: &CudaBuffer,
    seq: usize,
    k_heads: usize,
    v_heads: usize,
    kdim: usize,
    vdim: usize,
    eps: f32,
) -> Result<()> {
    if seq == 0
        || k_heads == 0
        || v_heads == 0
        || v_heads % k_heads != 0
        || kdim != 128
        || vdim != 128
        || !eps.is_finite()
        || eps < 0.0
    {
        return Err(Error::Other("norm_delta_gated: unsupported dimensions".into()));
    }
    let row = k_heads
        .checked_mul(kdim)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_add(v_heads.checked_mul(vdim)?))
        .ok_or_else(|| Error::Other("norm_delta_gated: row overflow".into()))?;
    let qkv_bytes = seq
        .checked_mul(row)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("norm_delta_gated: qkv overflow".into()))?;
    let qk_bytes = seq
        .checked_mul(k_heads)
        .and_then(|v| v.checked_mul(2 * kdim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("norm_delta_gated: qk overflow".into()))?;
    let scalar_bytes = seq
        .checked_mul(v_heads)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("norm_delta_gated: scalar overflow".into()))?;
    let value_bytes = seq
        .checked_mul(v_heads)
        .and_then(|v| v.checked_mul(vdim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("norm_delta_gated: value overflow".into()))?;
    let state_bytes = v_heads
        .checked_mul(kdim)
        .and_then(|v| v.checked_mul(vdim))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| Error::Other("norm_delta_gated: state overflow".into()))?;
    if qkv.len() < qkv_bytes
        || qk_out.len() < qk_bytes
        || a.len() < scalar_bytes
        || b.len() < scalar_bytes
        || a_log.len() < v_heads * 2
        || dt_bias.len() < v_heads * 2
        || z.len() < value_bytes
        || weight.len() < vdim * 2
        || recurrent.len() < state_bytes
        || delta_out.len() < value_bytes
        || out.len() < value_bytes
    {
        return Err(Error::Other("norm_delta_gated: buffer too small".into()));
    }
    let prof_ev = kernel_prof(ctx, "norm_delta_gated");
    let prof_result = check_cuda(unsafe {
        ffi::apxinf_qwen35_norm_delta_gated(
            qkv.ptr(),
            qk_out.ptr(),
            a.ptr(),
            b.ptr(),
            a_log.ptr(),
            dt_bias.ptr(),
            z.ptr(),
            weight.ptr(),
            recurrent.ptr(),
            delta_out.ptr(),
            out.ptr(),
            seq as i32,
            k_heads as i32,
            v_heads as i32,
            kdim as i32,
            vdim as i32,
            eps,
            ctx.stream().handle(),
        )
    });
    if let Some(ev) = prof_ev {
        kernel_prof_end(ctx, "norm_delta_gated", ev);
    }
    prof_result
}

/// Preflight the decode-only packed GDN kernel before it can mutate either
/// recurrent state. Unsupported dynamic-shared-memory geometry returns false.
pub fn prepare_packed_delta_gated() -> Result<bool> {
    let mut supported = 0;
    check_cuda(unsafe { ffi::apxinf_qwen35_prepare_packed_delta_gated(&mut supported) })?;
    Ok(supported != 0)
}

/// Packed Qwen3.5 GDN for the K=V=128, conv-width-4 layout. This fuses causal
/// conv, q/k norm, delta gating/update, and gated RMSNorm while retaining bf16
/// materialization boundaries.
#[allow(clippy::too_many_arguments)]
pub fn packed_delta_gated(
    ctx: &CudaContext,
    qkv: &CudaBuffer,
    conv_weight: &CudaBuffer,
    conv_state: &mut CudaBuffer,
    a: &CudaBuffer,
    b: &CudaBuffer,
    a_log: &CudaBuffer,
    dt_bias: &CudaBuffer,
    z: &CudaBuffer,
    weight: &CudaBuffer,
    recurrent: &mut CudaBuffer,
    out: &CudaBuffer,
    seq: usize,
    k_heads: usize,
    v_heads: usize,
    kdim: usize,
    vdim: usize,
    conv_kernel: usize,
    eps: f32,
) -> Result<()> {
    if seq != 1
        || k_heads == 0
        || v_heads == 0
        || v_heads % k_heads != 0
        || kdim != 128
        || vdim != 128
        || conv_kernel != 4
        || !eps.is_finite()
        || eps < 0.0
    {
        return Err(Error::Other("packed_delta_gated: unsupported dimensions".into()));
    }
    let row = k_heads
        .checked_mul(kdim)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_add(v_heads.checked_mul(vdim)?))
        .ok_or_else(|| Error::Other("packed_delta_gated: row overflow".into()))?;
    let bf16_rows = seq
        .checked_mul(row)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("packed_delta_gated: qkv overflow".into()))?;
    let bf16_heads = seq
        .checked_mul(v_heads)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("packed_delta_gated: head overflow".into()))?;
    let bf16_values = seq
        .checked_mul(v_heads)
        .and_then(|v| v.checked_mul(vdim))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("packed_delta_gated: value overflow".into()))?;
    let state_bytes = v_heads
        .checked_mul(kdim)
        .and_then(|v| v.checked_mul(vdim))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| Error::Other("packed_delta_gated: state overflow".into()))?;
    let conv_weight_bytes = row
        .checked_mul(conv_kernel)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("packed_delta_gated: conv weight overflow".into()))?;
    let conv_state_bytes = row
        .checked_mul(conv_kernel - 1)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| Error::Other("packed_delta_gated: conv state overflow".into()))?;
    if qkv.len() < bf16_rows
        || conv_weight.len() < conv_weight_bytes
        || conv_state.len() < conv_state_bytes
        || a.len() < bf16_heads
        || b.len() < bf16_heads
        || a_log.len() < v_heads * 2
        || dt_bias.len() < v_heads * 2
        || z.len() < bf16_values
        || weight.len() < vdim * 2
        || recurrent.len() < state_bytes
        || out.len() < bf16_values
    {
        return Err(Error::Other("packed_delta_gated: buffer too small".into()));
    }
    let prof_ev = kernel_prof(ctx, "packed_delta_gated");
    let prof_result = check_cuda(unsafe {
        ffi::apxinf_qwen35_packed_delta_gated(
            qkv.ptr(),
            conv_weight.ptr(),
            conv_state.ptr(),
            a.ptr(),
            b.ptr(),
            a_log.ptr(),
            dt_bias.ptr(),
            z.ptr(),
            weight.ptr(),
            recurrent.ptr(),
            out.ptr(),
            seq as i32,
            k_heads as i32,
            v_heads as i32,
            kdim as i32,
            vdim as i32,
            conv_kernel as i32,
            eps,
            ctx.stream().handle(),
        )
    });
    if let Some(ev) = prof_ev {
        kernel_prof_end(ctx, "packed_delta_gated", ev);
    }
    prof_result
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
    let prof_ev = kernel_prof(ctx, "gated_norm");
    let prof_result = check_cuda(unsafe {
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
    });
    if let Some(ev) = prof_ev {
        kernel_prof_end(ctx, "gated_norm", ev);
    }
    prof_result
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
    let prof_ev = kernel_prof(ctx, "q_split_norm_rope");
    let prof_result = check_cuda(unsafe {
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
    });
    if let Some(ev) = prof_ev {
        kernel_prof_end(ctx, "q_split_norm_rope", ev);
    }
    prof_result
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
    let prof_ev = kernel_prof(ctx, "k_norm_rope_append");
    let prof_result = check_cuda(unsafe {
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
    });
    if let Some(ev) = prof_ev {
        kernel_prof_end(ctx, "k_norm_rope_append", ev);
    }
    prof_result
}
/// Decode-only fusion of q split/norm/RoPE and k norm/RoPE/cache append.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_append(
    ctx: &CudaContext,
    q_gate: &CudaBuffer,
    q_norm_w: &CudaBuffer,
    k_in: &CudaBuffer,
    k_norm_w: &CudaBuffer,
    q_out: &CudaBuffer,
    gate_out: &CudaBuffer,
    k_cache: &mut CudaBuffer,
    seq: usize,
    heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    position: CudaDeviceAddress,
    max_seq_len: usize,
) -> Result<()> {
    if seq != 1 || heads == 0 || n_kv_heads == 0 || head_dim == 0 ||
        rotary_dim == 0 || max_seq_len == 0 || !theta.is_finite() ||
        position.len() < std::mem::size_of::<u32>()
    {
        return Err(Error::Other("qk_norm_rope_append: unsupported dimensions".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_qk_norm_rope_append(
            q_gate.ptr(), q_norm_w.ptr(), k_in.ptr(), k_norm_w.ptr(),
            q_out.ptr(), gate_out.ptr(), k_cache.ptr(), seq as i32,
            heads as i32, n_kv_heads as i32, head_dim as i32,
            rotary_dim as i32, theta, position.ptr() as *const u32,
            max_seq_len as i32, ctx.stream().handle(),
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
    let prof_ev = kernel_prof(ctx, "sigmoid_mul");
    let prof_result = check_cuda(unsafe {
        ffi::apxinf_qwen35_sigmoid_mul(
            gate.ptr(),
            x.ptr(),
            out.ptr(),
            count as i64,
            ctx.stream().handle(),
        )
    });
    if let Some(ev) = prof_ev {
        kernel_prof_end(ctx, "sigmoid_mul", ev);
    }
    prof_result
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
    let prof_ev = kernel_prof(ctx, "flash_prefill");
    let prof_result = check_cuda(unsafe {
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
    });
    if let Some(ev) = prof_ev {
        kernel_prof_end(ctx, "flash_prefill", ev);
    }
    prof_result
}

/// Exact fused seq=1 full attention for the production Qwen3.5 shape.
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_gated_256(
    ctx: &CudaContext,
    q: &CudaBuffer,
    k_cache: &CudaBuffer,
    v_cache: &CudaBuffer,
    gate: &CudaBuffer,
    out: &CudaBuffer,
    heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    position: CudaDeviceAddress,
    max_seq_len: usize,
) -> Result<()> {
    if heads != 24 || n_kv_heads != 4 || head_dim != 256 || max_seq_len == 0 {
        return Err(Error::Other("flash_decode_gated_256: unsupported shape".into()));
    }
    let row_bytes = heads * head_dim * 2;
    let cache_bytes = n_kv_heads
        .checked_mul(max_seq_len)
        .and_then(|v| v.checked_mul(head_dim * 2))
        .ok_or_else(|| Error::Other("flash_decode_gated_256: cache overflow".into()))?;
    if q.len() < row_bytes || gate.len() < row_bytes || out.len() < row_bytes ||
        k_cache.len() < cache_bytes || v_cache.len() < cache_bytes {
        return Err(Error::Other("flash_decode_gated_256: buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_flash_decode_gated_256(
            q.ptr(), k_cache.ptr(), v_cache.ptr(), gate.ptr(), out.ptr(), scale,
            position.ptr() as *const u32, max_seq_len as i32, ctx.stream().handle(),
        )
    })
}

/// Exact two-pass seq=1 attention with the same eight warp streams and merge
/// order as [`flash_decode_gated_256`], but 48 partial CTAs for higher SM use.
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_gated_256_split(
    ctx: &CudaContext,
    q: &CudaBuffer,
    k_cache: &CudaBuffer,
    v_cache: &CudaBuffer,
    gate: &CudaBuffer,
    out: &CudaBuffer,
    partials: &CudaBuffer,
    scale: f32,
    position: CudaDeviceAddress,
    max_seq_len: usize,
) -> Result<()> {
    const PARTIAL_BYTES: usize = 24 * 8 * (2 + 8 * 32) * 4;
    if partials.len() < PARTIAL_BYTES {
        return Err(Error::Other("split attention partial buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_flash_decode_gated_256_split(
            q.ptr(), k_cache.ptr(), v_cache.ptr(), gate.ptr(), out.ptr(),
            partials.ptr(), scale, position.ptr() as *const u32,
            max_seq_len as i32, ctx.stream().handle(),
        )
    })
}

/// Exact two-warp-CTA partial attention candidate.
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_gated_256_split_2w(
    ctx: &CudaContext,
    q: &CudaBuffer,
    k_cache: &CudaBuffer,
    v_cache: &CudaBuffer,
    gate: &CudaBuffer,
    out: &CudaBuffer,
    partials: &CudaBuffer,
    scale: f32,
    position: CudaDeviceAddress,
    max_seq_len: usize,
) -> Result<()> {
    const PARTIAL_BYTES: usize = 24 * 8 * (2 + 8 * 32) * 4;
    if partials.len() < PARTIAL_BYTES {
        return Err(Error::Other("split attention partial buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_flash_decode_gated_256_split_2w(
            q.ptr(), k_cache.ptr(), v_cache.ptr(), gate.ptr(), out.ptr(),
            partials.ptr(), scale, position.ptr() as *const u32,
            max_seq_len as i32, ctx.stream().handle(),
        )
    })
}

/// Exact one-warp-CTA partial attention candidate.
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_gated_256_split_1w(
    ctx: &CudaContext,
    q: &CudaBuffer,
    k_cache: &CudaBuffer,
    v_cache: &CudaBuffer,
    gate: &CudaBuffer,
    out: &CudaBuffer,
    partials: &CudaBuffer,
    scale: f32,
    position: CudaDeviceAddress,
    max_seq_len: usize,
) -> Result<()> {
    const PARTIAL_BYTES: usize = 24 * 8 * (2 + 8 * 32) * 4;
    if partials.len() < PARTIAL_BYTES {
        return Err(Error::Other("split attention partial buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_flash_decode_gated_256_split_1w(
            q.ptr(), k_cache.ptr(), v_cache.ptr(), gate.ptr(), out.ptr(),
            partials.ptr(), scale, position.ptr() as *const u32,
            max_seq_len as i32, ctx.stream().handle(),
        )
    })
}

/// Exact GQA-aware partial attention sharing one staged KV row across 2/3/6
/// Q-head warps. Partial layout and merge order remain unchanged.
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_gated_256_gqa(
    ctx: &CudaContext,
    q: &CudaBuffer,
    k_cache: &CudaBuffer,
    v_cache: &CudaBuffer,
    gate: &CudaBuffer,
    out: &CudaBuffer,
    partials: &CudaBuffer,
    scale: f32,
    position: CudaDeviceAddress,
    max_seq_len: usize,
    group: usize,
) -> Result<()> {
    const PARTIAL_BYTES: usize = 24 * 8 * (2 + 8 * 32) * 4;
    if !matches!(group, 2 | 3 | 6) || partials.len() < PARTIAL_BYTES {
        return Err(Error::Other("GQA attention candidate invalid geometry".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_flash_decode_gated_256_gqa(
            q.ptr(), k_cache.ptr(), v_cache.ptr(), gate.ptr(), out.ptr(),
            partials.ptr(), scale, position.ptr() as *const u32,
            max_seq_len as i32, group as i32, ctx.stream().handle(),
        )
    })
}

/// Scores = q @ k^T for one query head via the established plain cuBLAS
/// GEMM. `scores` is `[seq, row_stride]` f32, `kt` is the already-transposed
/// cache for this query head's KV group `[head_dim, visible]` bf16, and `q`
/// is `[seq, heads, head_dim]` bf16.
#[allow(clippy::too_many_arguments)]
pub fn attention_gqa_dot(
    ctx: &CudaContext,
    q: &CudaBuffer,
    kt: &CudaBuffer,
    scores: &CudaBuffer,
    head: usize,
    seq: usize,
    visible: usize,
    heads: usize,
    head_dim: usize,
    row_stride: usize,
) -> Result<()> {
    if head >= heads || head_dim == 0 || visible == 0 || row_stride < visible {
        return Err(Error::Other("attention_gqa_dot: invalid dimensions".into()));
    }
    let q_head_bytes = head_dim
        .checked_mul(2)
        .ok_or_else(|| Error::Other("attention_gqa_dot: q head overflow".into()))?;
    let q_row_elems = heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("attention_gqa_dot: q row overflow".into()))?;
    let score_bytes = seq
        .checked_mul(row_stride)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| Error::Other("attention_gqa_dot: score overflow".into()))?;
    let required_q_bytes = seq
        .checked_mul(q_row_elems)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| Error::Other("attention_gqa_dot: q size overflow".into()))?;
    if q.len() < required_q_bytes || scores.len() < score_bytes {
        return Err(Error::Other("attention_gqa_dot: buffer too small".into()));
    }
    let q_offset = head
        .checked_mul(q_head_bytes)
        .ok_or_else(|| Error::Other("attention_gqa_dot: q offset overflow".into()))?;
    let q_view = q
        .view(q_offset, q.len().saturating_sub(q_offset))
        .map_err(Error::Cuda)?;
    let score_view = scores.view(0, score_bytes).map_err(Error::Cuda)?;
    ctx.cublas()
        .gemm_bf16_f32(
            seq,
            visible,
            head_dim,
            1.0,
            &q_view,
            q_row_elems as i32,
            kt,
            0.0,
            &score_view,
            row_stride as i32,
        )
        .map_err(Error::Cuda)
}


/// out = p @ v for one query head using the established f32 cuBLAS GEMM.
/// `p` is `[seq, row_stride]`, `vf32` is `[visible, head_dim]`, and `out`
/// retains the full `[seq, heads, head_dim]` row stride.
#[allow(clippy::too_many_arguments)]
pub fn attention_gqa_pv(
    ctx: &CudaContext,
    p: &CudaBuffer,
    vf32: &CudaBuffer,
    out: &CudaBuffer,
    head: usize,
    seq: usize,
    visible: usize,
    heads: usize,
    head_dim: usize,
    row_stride: usize,
    beta: f32,
) -> Result<()> {
    if head >= heads || head_dim == 0 || visible == 0 || row_stride < visible {
        return Err(Error::Other("attention_gqa_pv: invalid dimensions".into()));
    }
    let p_bytes = seq
        .saturating_sub(1)
        .checked_mul(row_stride)
        .and_then(|value| value.checked_add(visible))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| Error::Other("attention_gqa_pv: score overflow".into()))?;
    let out_head_bytes = head_dim
        .checked_mul(4)
        .ok_or_else(|| Error::Other("attention_gqa_pv: head overflow".into()))?;
    let out_row_elems = heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("attention_gqa_pv: output row overflow".into()))?;
    let required_v_bytes = visible
        .checked_mul(head_dim)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| Error::Other("attention_gqa_pv: V size overflow".into()))?;
    let required_out_bytes = seq
        .checked_mul(out_row_elems)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| Error::Other("attention_gqa_pv: output size overflow".into()))?;
    if p.len() < p_bytes || vf32.len() < required_v_bytes || out.len() < required_out_bytes {
        return Err(Error::Other("attention_gqa_pv: buffer too small".into()));
    }
    let out_offset = head
        .checked_mul(out_head_bytes)
        .ok_or_else(|| Error::Other("attention_gqa_pv: output offset overflow".into()))?;
    let p_view = p.view(0, p_bytes).map_err(Error::Cuda)?;
    let out_view = out
        .view(out_offset, out.len().saturating_sub(out_offset))
        .map_err(Error::Cuda)?;
    ctx.cublas()
        .gemm_ld_f32(
            seq,
            head_dim,
            visible,
            1.0,
            &p_view,
            row_stride as i32,
            vf32,
            beta,
            &out_view,
            out_row_elems as i32,
        )
        .map_err(Error::Cuda)
}

/// vf32 = f32(v) for the attention output GEMM.
pub fn v_to_f32(
    ctx: &CudaContext,
    v: &CudaBuffer,
    vf32: &CudaBuffer,
    visible: usize,
    head_dim: usize,
) -> Result<()> {
    check_cuda(unsafe {
        ffi::apxinf_qwen35_v_to_f32(
            v.ptr(),
            vf32.ptr(),
            visible as i32,
            head_dim as i32,
            ctx.stream().handle(),
        )
    })
}

/// Row-wise softmax over the causal prefix of the scores matrix.
#[allow(clippy::too_many_arguments)]
pub fn attention_softmax_rows(
    ctx: &CudaContext,
    scores: &CudaBuffer,
    l_out: &CudaBuffer,
    head_base: usize,
    seq: usize,
    heads: usize,
    visible: usize,
    row_stride: usize,
    start_pos: u32,
    scale: f32,
) -> Result<()> {
    check_cuda(unsafe {
        ffi::apxinf_qwen35_attention_softmax_rows(
            scores.ptr(),
            l_out.ptr(),
            head_base as i32,
            seq as i32,
            heads as i32,
            visible as i32,
            row_stride as i32,
            start_pos as i32,
            scale,
            ctx.stream().handle(),
        )
    })
}

/// out = bf16(pv / l) row-wise; pv is f32, out is bf16.
pub fn scale_out(
    ctx: &CudaContext,
    pv: &CudaBuffer,
    l: &CudaBuffer,
    out: &CudaBuffer,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<()> {
    check_cuda(unsafe {
        ffi::apxinf_qwen35_scale_out(
            pv.ptr(),
            l.ptr(),
            out.ptr(),
            seq as i32,
            heads as i32,
            head_dim as i32,
            ctx.stream().handle(),
        )
    })
}

/// `out = bf16(bf16(pv / l) * sigmoid(gate))`, retaining the former BF16
/// boundary while fusing the scale and gate launches for chunked prefill.
#[allow(clippy::too_many_arguments)]
pub fn scale_out_gated(
    ctx: &CudaContext,
    pv: &CudaBuffer,
    l: &CudaBuffer,
    gate: &CudaBuffer,
    out: &CudaBuffer,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<()> {
    if seq <= 1 || heads == 0 || head_dim == 0 || head_dim > 1024 {
        return Err(Error::Other("scale_out_gated: invalid dimensions".into()));
    }
    let rows = seq
        .checked_mul(heads)
        .ok_or_else(|| Error::Other("scale_out_gated: row overflow".into()))?;
    let elems = rows
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("scale_out_gated: element overflow".into()))?;
    let pv_bytes = elems
        .checked_mul(4)
        .ok_or_else(|| Error::Other("scale_out_gated: pv overflow".into()))?;
    let l_bytes = rows
        .checked_mul(4)
        .ok_or_else(|| Error::Other("scale_out_gated: l overflow".into()))?;
    let bf16_bytes = elems
        .checked_mul(2)
        .ok_or_else(|| Error::Other("scale_out_gated: bf16 overflow".into()))?;
    if pv.len() < pv_bytes || l.len() < l_bytes || gate.len() < bf16_bytes || out.len() < bf16_bytes {
        return Err(Error::Other("scale_out_gated: buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_scale_out_gated(
            pv.ptr(),
            l.ptr(),
            gate.ptr(),
            out.ptr(),
            seq as i32,
            heads as i32,
            head_dim as i32,
            ctx.stream().handle(),
        )
    })
}

/// Transposes a k cache slice for the scores GEMM: `kt` is
/// `[head_dim, visible]` bf16 from `k` `[visible, head_dim]`.
pub fn transpose_kt(
    ctx: &CudaContext,
    k: &CudaBuffer,
    kt: &CudaBuffer,
    visible: usize,
    head_dim: usize,
) -> Result<()> {
    let required = visible
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::Other("transpose_kt size overflow".into()))?;
    if kt.len() < required {
        return Err(Error::Other("transpose_kt: kt buffer too small".into()));
    }
    check_cuda(unsafe {
        ffi::apxinf_qwen35_transpose_kt(
            k.ptr(),
            kt.ptr(),
            visible as i32,
            head_dim as i32,
            ctx.stream().handle(),
        )
    })
}
