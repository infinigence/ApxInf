//! Gated DeltaNet primitives: the recurrent step, its gates, the causal
//! convolution window, and the gated output norm.
//!
//! The recurrence is implemented from the architecture's documented form and
//! has **not** been checked against a reference engine running this
//! checkpoint. The tests here establish that the kernels compute the stated
//! recurrence, not that the recurrence is the model's.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{gdn as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// Shape of the recurrent state a GDN layer carries between tokens.
///
/// `[v_heads, v_dim, k_dim]` f32. For Qwen3.8-27B that is
/// 48 x 128 x 128 x 4 B = 3 MiB per layer, so 48 layers hold ~151 MiB of state
/// that decode reads and writes every token -- a bandwidth cost comparable to
/// a sizeable fraction of the weights, and one worth measuring rather than
/// assuming away.
pub fn gdn_state_elements(v_heads: usize, v_dim: usize, k_dim: usize) -> usize {
    v_heads * v_dim * k_dim
}

/// One recurrent step: decay, delta-rule correction, rank-1 update, readout.
///
/// `state` is `[v_heads, v_dim, k_dim]` f32, updated in place. `q`/`k` are
/// `[k_heads, k_dim]`, `v` is `[v_heads, v_dim]`; value head `h` reads k-head
/// `h / (v_heads / k_heads)`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_step(
    ctx: &CudaContext,
    state: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    output: &Tensor,
    k_heads: usize,
) -> Result<()> {
    let state_dims = state.shape().dims().to_vec();
    if state_dims.len() != 3 {
        return Err(invalid("GDN state must be [v_heads, v_dim, k_dim]"));
    }
    let (v_heads, v_dim, k_dim) = (state_dims[0], state_dims[1], state_dims[2]);
    if k_heads == 0 || v_heads % k_heads != 0 {
        return Err(invalid("GDN value heads must be a multiple of key heads"));
    }
    let state_buffer = tensor_storage(ctx, state, DType::F32, &state_dims)?;
    let q_buffer = tensor_storage(ctx, q, DType::BF16, &[k_heads, k_dim])?;
    let k_buffer = tensor_storage(ctx, k, DType::BF16, &[k_heads, k_dim])?;
    let v_buffer = tensor_storage(ctx, v, DType::BF16, &[v_heads, v_dim])?;
    let decay_buffer = tensor_storage(ctx, decay, DType::F32, &[v_heads])?;
    let beta_buffer = tensor_storage(ctx, beta, DType::F32, &[v_heads])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &[v_heads, v_dim])?;
    unsafe {
        status::check(abi::apxinf_gdn_recurrent_step(
            state_buffer.ptr(),
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            decay_buffer.ptr(),
            beta_buffer.ptr(),
            output_buffer.ptr(),
            v_heads as i64,
            k_heads as i64,
            v_dim as i64,
            k_dim as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Per-head RMSNorm followed by the swish output gate.
pub fn gdn_gated_norm(
    ctx: &CudaContext,
    input: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    epsilon: f32,
) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("GDN gated norm expects [heads, head_dim]"));
    }
    let (heads, head_dim) = (dims[0], dims[1]);
    let input_buffer = tensor_storage(ctx, input, DType::BF16, &dims)?;
    let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &[head_dim])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_gdn_gated_norm(
            input_buffer.ptr(),
            gate_buffer.ptr(),
            weight_buffer.ptr(),
            output_buffer.ptr(),
            heads as i64,
            head_dim as i64,
            epsilon,
            ctx.stream().handle(),
        ))
    }
}

/// Advance the causal convolution by one token, then apply SiLU.
///
/// `window` is `[channels, kernel_width]` f32 holding the last
/// `kernel_width` inputs per channel.
pub fn gdn_causal_conv_step(
    ctx: &CudaContext,
    window: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
) -> Result<()> {
    let window_dims = window.shape().dims().to_vec();
    if window_dims.len() != 2 {
        return Err(invalid("GDN conv window must be [channels, kernel_width]"));
    }
    let (channels, kernel_width) = (window_dims[0], window_dims[1]);
    let window_buffer = tensor_storage(ctx, window, DType::F32, &window_dims)?;
    let input_buffer = tensor_storage(ctx, input, DType::BF16, &[channels])?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &window_dims)?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &[channels])?;
    unsafe {
        status::check(abi::apxinf_gdn_causal_conv_step(
            window_buffer.ptr(),
            input_buffer.ptr(),
            weight_buffer.ptr(),
            output_buffer.ptr(),
            channels as i64,
            kernel_width as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Causal depthwise conv1d over a whole prompt, then SiLU.
///
/// `input`/`output` are `[tokens, channels]` BF16 and `weight` is
/// `[channels, kernel_width]` BF16. When `window` is `Some`, the last
/// `kernel_width` inputs are written back in the `[channels, kernel_width]`
/// f32 layout `gdn_causal_conv_step` expects, so decode continues from a
/// prompt without re-running it. Passing `None` skips that and is only
/// correct when nothing will decode afterwards.
pub fn gdn_causal_conv_forward(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    window: Option<&Tensor>,
    tokens: usize,
    channels: usize,
    kernel_width: usize,
) -> Result<()> {
    let flat = [tokens, channels];
    let input_buffer = tensor_storage(ctx, input, DType::BF16, &flat)?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &[channels, kernel_width])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &flat)?;
    let window_buffer = match window {
        Some(tensor) => Some(tensor_storage(
            ctx,
            tensor,
            DType::F32,
            &[channels, kernel_width],
        )?),
        None => None,
    };
    unsafe {
        status::check(abi::apxinf_gdn_causal_conv_forward(
            input_buffer.ptr(),
            weight_buffer.ptr(),
            output_buffer.ptr(),
            window_buffer
                .as_ref()
                .map(|buffer| buffer.ptr())
                .unwrap_or(std::ptr::null_mut()),
            tokens as i64,
            channels as i64,
            kernel_width as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Sequence-axis `gdn_decay_and_beta`.
///
/// `a`/`b` are `[tokens, heads]` BF16 and `decay`/`beta` `[tokens, heads]`
/// f32. `a_log` and `dt_bias` stay per-head `[heads]`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decay_and_beta_seq(
    ctx: &CudaContext,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    tokens: usize,
    heads: usize,
) -> Result<()> {
    let rows = [tokens, heads];
    let a_buffer = tensor_storage(ctx, a, DType::BF16, &rows)?;
    let b_buffer = tensor_storage(ctx, b, DType::BF16, &rows)?;
    let a_log_buffer = tensor_storage(ctx, a_log, DType::BF16, &[heads])?;
    let dt_bias_buffer = tensor_storage(ctx, dt_bias, DType::BF16, &[heads])?;
    let decay_buffer = tensor_storage(ctx, decay, DType::F32, &rows)?;
    let beta_buffer = tensor_storage(ctx, beta, DType::F32, &rows)?;
    unsafe {
        status::check(abi::apxinf_gdn_decay_and_beta_seq(
            a_buffer.ptr(),
            b_buffer.ptr(),
            a_log_buffer.ptr(),
            dt_bias_buffer.ptr(),
            decay_buffer.ptr(),
            beta_buffer.ptr(),
            tokens as i64,
            heads as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Sequence-axis `gdn_gated_norm`. Tensors are `[tokens, heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_gated_norm_seq(
    ctx: &CudaContext,
    input: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    epsilon: f32,
) -> Result<()> {
    let dims = [tokens, heads, head_dim];
    let input_buffer = tensor_storage(ctx, input, DType::BF16, &dims)?;
    let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &[head_dim])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_gdn_gated_norm_seq(
            input_buffer.ptr(),
            gate_buffer.ptr(),
            weight_buffer.ptr(),
            output_buffer.ptr(),
            tokens as i64,
            heads as i64,
            head_dim as i64,
            epsilon,
            ctx.stream().handle(),
        ))
    }
}

/// L2-normalize each head in place.
pub fn gdn_l2_normalize_heads(ctx: &CudaContext, data: &Tensor, epsilon: f32) -> Result<()> {
    let dims = data.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("GDN head normalization expects [heads, head_dim]"));
    }
    let buffer = tensor_storage(ctx, data, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_gdn_l2_normalize_heads(
            buffer.ptr(),
            dims[0] as i64,
            dims[1] as i64,
            epsilon,
            ctx.stream().handle(),
        ))
    }
}

/// `decay = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decay_and_beta(
    ctx: &CudaContext,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
) -> Result<()> {
    let dims = a.shape().dims().to_vec();
    if dims.len() != 1 {
        return Err(invalid("GDN gate inputs must be [heads]"));
    }
    let heads = dims[0];
    let a_buffer = tensor_storage(ctx, a, DType::BF16, &dims)?;
    let b_buffer = tensor_storage(ctx, b, DType::BF16, &dims)?;
    let a_log_buffer = tensor_storage(ctx, a_log, DType::BF16, &dims)?;
    let dt_bias_buffer = tensor_storage(ctx, dt_bias, DType::BF16, &dims)?;
    let decay_buffer = tensor_storage(ctx, decay, DType::F32, &dims)?;
    let beta_buffer = tensor_storage(ctx, beta, DType::F32, &dims)?;
    unsafe {
        status::check(abi::apxinf_gdn_decay_and_beta(
            a_buffer.ptr(),
            b_buffer.ptr(),
            a_log_buffer.ptr(),
            dt_bias_buffer.ptr(),
            decay_buffer.ptr(),
            beta_buffer.ptr(),
            heads as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Chunked parallel scan over `seq_padded` tokens (prefill).
///
/// A faithful port of `torch_chunk_gated_delta_rule` (the forward-substitution
/// export path). Produces the same `core_attn_out` and leaves the recurrent
/// `state` in exactly the form the single-token `gdn_recurrent_step` expects,
/// so a prompt run through this scan and then decoded continues from an
/// identical state.
///
/// `q`/`k` are `[seq_padded, k_heads, k_dim]` BF16 (post-conv, pre-l2norm);
/// `v` is `[seq_padded, v_heads, k_dim]` BF16; `g`/`beta` are
/// `[seq_padded, v_heads]` f32. `out` is `[seq_padded, v_heads, k_dim]` BF16.
/// `state` is `[v_heads, v_dim, k_dim]` f32, updated in place. q/k are
/// L2-normalized in fp32 and q scaled by `k_dim**-0.5` inside the kernel, to
/// match the reference. `seq_padded` must be a multiple of `chunk_size`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_scan(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    out: &Tensor,
    state: &Tensor,
    v_heads: usize,
    k_heads: usize,
    chunk_size: usize,
) -> Result<()> {
    let state_dims = state.shape().dims().to_vec();
    if state_dims.len() != 3 {
        return Err(invalid("GDN state must be [v_heads, v_dim, k_dim]"));
    }
    let (sv_heads, v_dim, k_dim) = (state_dims[0], state_dims[1], state_dims[2]);
    if sv_heads != v_heads || v_dim != k_dim {
        return Err(invalid("GDN chunk scan expects square v_dim == k_dim state"));
    }
    if k_heads == 0 || v_heads % k_heads != 0 {
        return Err(invalid("GDN value heads must be a multiple of key heads"));
    }
    if chunk_size == 0 {
        return Err(invalid("GDN chunk size must be positive"));
    }
    let q_dims = q.shape().dims().to_vec();
    if q_dims.len() != 3 {
        return Err(invalid("GDN chunk scan q must be [seq, k_heads, k_dim]"));
    }
    let seq_padded = q_dims[0];
    if seq_padded % chunk_size != 0 {
        return Err(invalid("GDN chunk scan seq must be a multiple of chunk_size"));
    }
    let num_chunks = seq_padded / chunk_size;

    let q_buffer = tensor_storage(ctx, q, DType::BF16, &[seq_padded, k_heads, k_dim])?;
    let k_buffer = tensor_storage(ctx, k, DType::BF16, &[seq_padded, k_heads, k_dim])?;
    let v_buffer = tensor_storage(ctx, v, DType::BF16, &[seq_padded, v_heads, v_dim])?;
    let g_buffer = tensor_storage(ctx, g, DType::F32, &[seq_padded, v_heads])?;
    let beta_buffer = tensor_storage(ctx, beta, DType::F32, &[seq_padded, v_heads])?;
    let out_buffer = tensor_storage(ctx, out, DType::BF16, &[seq_padded, v_heads, v_dim])?;
    let state_buffer = tensor_storage(ctx, state, DType::F32, &state_dims)?;
    unsafe {
        status::check(abi::apxinf_gdn_chunk_scan(
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            g_buffer.ptr(),
            beta_buffer.ptr(),
            out_buffer.ptr(),
            state_buffer.ptr(),
            seq_padded as i64,
            v_heads as i64,
            k_heads as i64,
            chunk_size as i64,
            k_dim as i64,
            num_chunks as i64,
            // Contiguous [seq, heads, dim] inputs: one token is heads*dim apart.
            (k_heads * k_dim) as i64,
            (k_heads * k_dim) as i64,
            (v_heads * k_dim) as i64,
            ctx.stream().handle(),
        ))
    }
}

/// `gdn_chunk_scan` over q, k and v interleaved in one projection row.
///
/// The GDN input projection produces `[seq, q | k | v]` in a single row, and
/// the causal conv keeps that packing. Splitting it into three contiguous
/// tensors would cost a full read and write of the conv output per layer, so
/// instead each operand points at its own offset inside the fused buffer and
/// carries the full row width as its stride.
///
/// `fused` is `[seq_padded, row_width]` BF16 with q first, then k, then v.
/// `q_offset`, `k_offset` and `v_offset` are element offsets into a row.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_scan_interleaved(
    ctx: &CudaContext,
    fused: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    out: &Tensor,
    state: &Tensor,
    seq_padded: usize,
    row_width: usize,
    offsets: [usize; 3],
    v_heads: usize,
    k_heads: usize,
    chunk_size: usize,
    k_dim: usize,
) -> Result<()> {
    if seq_padded % chunk_size != 0 {
        return Err(invalid("GDN chunk scan needs a multiple of chunk_size"));
    }
    let [q_offset, k_offset, v_offset] = offsets;
    if q_offset + k_heads * k_dim > row_width
        || k_offset + k_heads * k_dim > row_width
        || v_offset + v_heads * k_dim > row_width
    {
        return Err(invalid("GDN chunk scan offsets run past the fused row"));
    }
    let num_chunks = seq_padded / chunk_size;
    let element = DType::BF16.size_in_bytes();

    let fused_buffer = tensor_storage(ctx, fused, DType::BF16, &[seq_padded, row_width])?;
    let g_buffer = tensor_storage(ctx, g, DType::F32, &[seq_padded, v_heads])?;
    let beta_buffer = tensor_storage(ctx, beta, DType::F32, &[seq_padded, v_heads])?;
    let out_buffer =
        tensor_storage(ctx, out, DType::BF16, &[seq_padded, v_heads, k_dim])?;
    let state_buffer = tensor_storage(ctx, state, DType::F32, &[v_heads, k_dim, k_dim])?;

    // Byte offsets into the one fused allocation; the kernel walks rows itself.
    let base = fused_buffer.ptr() as *const u8;
    unsafe {
        status::check(abi::apxinf_gdn_chunk_scan(
            base.add(q_offset * element) as *const std::ffi::c_void,
            base.add(k_offset * element) as *const std::ffi::c_void,
            base.add(v_offset * element) as *const std::ffi::c_void,
            g_buffer.ptr(),
            beta_buffer.ptr(),
            out_buffer.ptr(),
            state_buffer.ptr(),
            seq_padded as i64,
            v_heads as i64,
            k_heads as i64,
            chunk_size as i64,
            k_dim as i64,
            num_chunks as i64,
            row_width as i64,
            row_width as i64,
            row_width as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Chunked gated delta rule on the vendored FlashInfer Cake kernel.
///
/// The fast path for prefill. Our own `gdn_chunk_scan` stays as the reference
/// it is checked against; this one measured 0.427 ms at 2048 tokens on the
/// 16/48-head shape against 48 ms for a layer of the hand-written scan.
///
/// Differences from `gdn_chunk_scan`, all of which the caller must honour:
///
/// * `q`, `k`, `v` and `out` are **FP16**, not BF16. The head-generic variant
///   our 48 value heads force exists only with FP16 I/O; the probe in
///   `devlocal/qwen38-nvfp4/reports/flashinfer-gdn-feasibility.md` measured
///   four orders of magnitude of headroom on real activations.
/// * `q` and `k` must **already be L2-normalized**, and `q` must **not** be
///   pre-scaled -- pass `scale` instead.
/// * `tokens` needs no padding. The kernel clamps its TMA descriptors per
///   sequence, so the `chunk_size` multiple `gdn_chunk_scan` requires is gone.
///
/// `state` is `[v_heads, 128, 128]` f32 in `[H, V, K]` order, updated in
/// place -- the same layout `gdn_recurrent_step` uses, so decode continues
/// without conversion.
#[allow(clippy::too_many_arguments)]
pub fn flashinfer_gdn_prefill(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &Tensor,
    gate_log: &Tensor,
    beta: &Tensor,
    cu_seqlens: &Tensor,
    state: &Tensor,
    workspace: &Tensor,
    tokens: usize,
    q_heads: usize,
    v_heads: usize,
    num_seqs: usize,
    scale: f32,
) -> Result<()> {
    if v_heads % q_heads != 0 {
        return Err(invalid("GDN value heads must be a multiple of query heads"));
    }
    let head_dim = 128usize;
    let q_dims = [tokens, q_heads, head_dim];
    let v_dims = [tokens, v_heads, head_dim];

    let q_buffer = tensor_storage(ctx, q, DType::F16, &q_dims)?;
    let k_buffer = tensor_storage(ctx, k, DType::F16, &q_dims)?;
    let v_buffer = tensor_storage(ctx, v, DType::F16, &v_dims)?;
    let out_buffer = tensor_storage(ctx, out, DType::F16, &v_dims)?;
    let gate_buffer = tensor_storage(ctx, gate_log, DType::F32, &[tokens, v_heads])?;
    let beta_buffer = tensor_storage(ctx, beta, DType::F32, &[tokens, v_heads])?;
    let seqlens_buffer = tensor_storage(ctx, cu_seqlens, DType::I32, &[num_seqs + 1])?;
    let state_buffer =
        tensor_storage(ctx, state, DType::F32, &[v_heads, head_dim, head_dim])?;
    // The kernel treats this as raw bytes; validating it against its own
    // declared shape keeps the check meaningful without inventing a dtype.
    let workspace_dims = workspace.shape().dims().to_vec();
    let workspace_buffer =
        tensor_storage(ctx, workspace, workspace.dtype(), &workspace_dims)?;

    unsafe {
        status::check(abi::apxinf_flashinfer_gdn_prefill(
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            out_buffer.ptr(),
            gate_buffer.ptr(),
            beta_buffer.ptr(),
            seqlens_buffer.ptr(),
            state_buffer.ptr(),
            workspace_buffer.ptr(),
            tokens as i64,
            q_heads as i64,
            v_heads as i64,
            num_seqs as i64,
            scale,
            ctx.stream().handle(),
        ))
    }
}

/// Scratch bytes `flashinfer_gdn_prefill` needs for its TMA rewrites.
pub fn flashinfer_gdn_workspace_bytes(v_heads: usize, num_seqs: usize) -> usize {
    unsafe {
        abi::apxinf_flashinfer_gdn_workspace_bytes(v_heads as i64, num_seqs as i64)
            .max(0) as usize
    }
}

/// One-pass conversion of a GDN projection into FlashInfer's input form.
///
/// `flashinfer_gdn_prefill` wants q/k/v split into contiguous FP16 tensors
/// with q and k L2-normalized and q unscaled, plus a linear-space decay. Doing
/// those as separate ops would reread the projection four times; at 2048
/// tokens it is 40 MB, so the conversion would rival the kernel it feeds.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prepare_flashinfer(
    ctx: &CudaContext,
    fused: &Tensor,
    q_out: &Tensor,
    k_out: &Tensor,
    v_out: &Tensor,
    g: &Tensor,
    alpha: &Tensor,
    tokens: usize,
    row_width: usize,
    k_heads: usize,
    v_heads: usize,
    dim: usize,
    epsilon: f32,
) -> Result<()> {
    if 2 * k_heads * dim + v_heads * dim > row_width {
        return Err(invalid("GDN projection row is too narrow for q, k and v"));
    }
    let fused_buffer = tensor_storage(ctx, fused, DType::BF16, &[tokens, row_width])?;
    let q_buffer = tensor_storage(ctx, q_out, DType::F16, &[tokens, k_heads, dim])?;
    let k_buffer = tensor_storage(ctx, k_out, DType::F16, &[tokens, k_heads, dim])?;
    let v_buffer = tensor_storage(ctx, v_out, DType::F16, &[tokens, v_heads, dim])?;
    let g_buffer = tensor_storage(ctx, g, DType::F32, &[tokens, v_heads])?;
    let alpha_buffer = tensor_storage(ctx, alpha, DType::F32, &[tokens, v_heads])?;
    unsafe {
        status::check(abi::apxinf_gdn_prepare_flashinfer(
            fused_buffer.ptr(),
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            g_buffer.ptr(),
            alpha_buffer.ptr(),
            tokens as i64,
            row_width as i64,
            k_heads as i64,
            v_heads as i64,
            dim as i64,
            epsilon,
            ctx.stream().handle(),
        ))
    }
}

/// Widen an FP16 buffer to BF16.
///
/// The FlashInfer scan emits FP16 because the head-generic variant our 48
/// value heads force has no BF16 build; everything downstream of it here is
/// BF16.
pub fn convert_f16_to_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    output: &Tensor,
    count: usize,
) -> Result<()> {
    let input_buffer = tensor_storage(ctx, input, DType::F16, input.shape().dims())?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, output.shape().dims())?;
    if input.shape().numel() < count || output.shape().numel() < count {
        return Err(invalid("FP16 widening runs past one of its buffers"));
    }
    unsafe {
        status::check(abi::apxinf_gdn_widen_f16_to_bf16(
            input_buffer.ptr(),
            output_buffer.ptr(),
            count as i64,
            ctx.stream().handle(),
        ))
    }
}
