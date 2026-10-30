//! Gated DeltaNet primitives: the recurrent step, its gates, the causal
//! convolution window, and the gated output norm.
//!
//! The recurrence is implemented from the architecture's documented form and
//! has **not** been checked against a reference engine running this
//! checkpoint. The tests here establish that the kernels compute the stated
//! recurrence, not that the recurrence is the model's.

use apxinf_core::{DType, Error, Result, Tensor};

use crate::ffi::abi::{gdn as abi, status};
use crate::ops::gemm::contracts::{dtype, invalid, tensor_storage};
use crate::{CudaBuffer, CudaContext};

mod launches;

const FIRST_PREFILL_CHUNK: usize = 64;
const CONTINUATION_CHUNK: usize = 64;

fn launch(ctx: &CudaContext, spec: &abi::Spec, bindings: &abi::Bindings) -> Result<()> {
    unsafe { status::check(abi::apxinf_gdn_launch(ctx.runtime(), spec, bindings)) }
}

/// Complete first-prefill causal-convolution + gated-delta scan semantic.
///
/// Candidate support and fallback are internal: callers always receive the
/// same logical `[tokens, value_heads * value_dim]` output and state updates.
pub struct GdnFirstPrefillArgs<'a> {
    pub zba: &'a Tensor,
    pub conv_weight: &'a Tensor,
    pub new_conv_state: &'a Tensor,
    pub a_log: &'a Tensor,
    /// Canonical FP32 decay bias used by the portable implementation.
    pub dt_bias: &'a Tensor,
    /// Optional BF16-rounded representation of the same weights. Exact-shape
    /// candidates may consume it without making the model select a provider.
    pub rounded_dt_bias: Option<&'a Tensor>,
    pub final_state: &'a Tensor,
    pub output: &'a Tensor,
    /// `[key_heads, value_heads, key_dim, value_dim]`.
    pub geometry: [usize; 4],
}
fn scan_scratch(
    ctx: &CudaContext,
    heads: usize,
    seq_pad: usize,
    seq: usize,
    width_bytes: usize,
) -> Result<CudaBuffer> {
    if seq_pad == seq {
        return crate::ops::scratch_buffer(ctx, (heads * seq_pad * width_bytes).max(1));
    }
    crate::ops::scratch_buffer_tail_zeroed(ctx, heads, seq_pad * width_bytes, seq * width_bytes)
}

fn zeroed(ctx: &CudaContext, bytes: usize) -> Result<CudaBuffer> {
    crate::ops::scratch_buffer_zeroed(ctx, bytes.max(1))
}

fn prepare_fixed(
    ctx: &CudaContext,
    key_dim: usize,
    value_dim: usize,
    chunk_size: usize,
) -> Result<()> {
    if crate::workspace::may_prepare_native_resources() {
        unsafe {
            status::check(abi::apxinf_gdn_prepare_fixed(
                ctx.device_id() as i32,
                key_dim as i32,
                value_dim as i32,
                chunk_size as i32,
            ))?;
        }
    }
    Ok(())
}

fn prepare_chunk_scan(ctx: &CudaContext) -> Result<()> {
    if crate::workspace::may_prepare_native_resources() {
        unsafe {
            status::check(abi::apxinf_gdn_chunk_scan_prepare(ctx.device_id() as i32))?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
pub fn gdn_first_prefill(ctx: &CudaContext, args: GdnFirstPrefillArgs<'_>) -> Result<()> {
    let [key_heads, value_heads, key_dim, value_dim] = args.geometry;
    let seq = args.zba.shape().dims().first().copied().unwrap_or(0);
    let key_width = key_heads
        .checked_mul(key_dim)
        .ok_or_else(|| invalid("GDN key width overflow"))?;
    let value_width = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| invalid("GDN value width overflow"))?;
    let conv_width = 2 * key_width + value_width;
    let row_width = conv_width + value_width + 2 * value_heads;
    if seq == 0
        || key_heads == 0
        || value_heads == 0
        || value_heads % key_heads != 0
        || key_dim != value_dim
        || args.zba.shape().dims() != [seq, row_width]
        || args.conv_weight.shape().dims() != [conv_width, 4]
        || args.new_conv_state.shape().dims() != [conv_width, 4]
        || args.a_log.shape().dims() != [value_heads]
        || args.dt_bias.shape().dims() != [value_heads]
        || args.dt_bias.dtype() != DType::F32
        || args.rounded_dt_bias.is_some_and(|tensor| {
            tensor.shape().dims() != [value_heads] || tensor.dtype() != DType::BF16
        })
        || args.final_state.shape().dims() != [value_heads, value_dim, key_dim]
        || args.output.shape().dims() != [seq, value_width]
    {
        return Err(invalid("invalid first-prefill GDN contract"));
    }
    let final_state = CudaBuffer::from_tensor(args.final_state).map_err(Error::Cuda)?;
    if let Some(fast) = launches::try_causal_conv_gdn_prefill_bf16(
        ctx,
        args.zba,
        args.conv_weight,
        args.new_conv_state,
        args.a_log,
        args.rounded_dt_bias,
        &final_state,
        args.geometry,
    )? {
        let source = CudaBuffer::from_tensor(&fast).map_err(Error::Cuda)?;
        let destination = CudaBuffer::from_tensor(args.output).map_err(Error::Cuda)?;
        destination
            .copy_from_device_async(&source, args.output.size_in_bytes(), ctx.stream())
            .map_err(Error::Cuda)?;
        return Ok(());
    }

    prepare_fixed(ctx, key_dim, value_dim, FIRST_PREFILL_CHUNK)?;

    use launches as la;
    let conv_out =
        ctx.allocate_output(apxinf_core::Shape::new(vec![seq, conv_width]), DType::BF16)?;
    la::causal_conv1d_silu_bf16(
        ctx,
        args.zba,
        args.conv_weight,
        None,
        &conv_out,
        args.new_conv_state,
        4,
    )?;

    let seq_pad = seq.div_ceil(FIRST_PREFILL_CHUNK) * FIRST_PREFILL_CHUNK;
    let f32_bytes = DType::F32.size_in_bytes();
    let qk_bf16 =
        la::gdn_qk_bf16_prefill_supported(ctx, false, key_dim, value_dim, FIRST_PREFILL_CHUNK);
    let qk_bytes = if qk_bf16 {
        DType::BF16.size_in_bytes()
    } else {
        f32_bytes
    };
    let q_buf = scan_scratch(ctx, value_heads, seq_pad, seq, key_dim * qk_bytes)?;
    let k_buf = scan_scratch(ctx, value_heads, seq_pad, seq, key_dim * qk_bytes)?;
    let v_buf = if qk_bf16 {
        None
    } else {
        Some(scan_scratch(
            ctx,
            value_heads,
            seq_pad,
            seq,
            value_dim * DType::BF16.size_in_bytes(),
        )?)
    };
    let beta_buf = scan_scratch(ctx, value_heads, seq_pad, seq, f32_bytes)?;
    let g_buf = scan_scratch(ctx, value_heads, seq_pad, seq, f32_bytes)?;

    if qk_bf16 {
        la::gdn_qk_prep_qk_bf16(
            ctx,
            &conv_out,
            &q_buf,
            &k_buf,
            seq_pad,
            key_heads,
            value_heads,
            key_dim,
            key_width,
            false,
            1e-6,
        )?;
        la::gdn_gate_prep(
            ctx,
            args.zba,
            conv_width + value_width,
            conv_width + value_width + value_heads,
            args.dt_bias,
            args.a_log,
            &beta_buf,
            &g_buf,
            seq_pad,
            value_heads,
        )?;
    } else {
        la::gdn_qk_prep(
            ctx,
            &conv_out,
            &q_buf,
            &k_buf,
            seq_pad,
            key_heads,
            value_heads,
            key_dim,
            key_width,
            false,
            1e-6,
        )?;
        la::gdn_vb_prep(
            ctx,
            &conv_out,
            args.zba,
            conv_width + value_width,
            conv_width + value_width + value_heads,
            args.dt_bias,
            args.a_log,
            v_buf
                .as_ref()
                .ok_or_else(|| invalid("GDN V scratch missing"))?,
            &beta_buf,
            &g_buf,
            seq_pad,
            value_heads,
            value_dim,
            2 * key_width,
        )?;
    }

    let chunks = seq_pad / FIRST_PREFILL_CHUNK;
    let g_cum = zeroed(ctx, value_heads * seq_pad * f32_bytes)?;
    let a_buf = zeroed(
        ctx,
        value_heads * chunks * FIRST_PREFILL_CHUNK * FIRST_PREFILL_CHUNK * f32_bytes,
    )?;
    let t_buf = zeroed(
        ctx,
        value_heads
            * chunks
            * FIRST_PREFILL_CHUNK
            * FIRST_PREFILL_CHUNK
            * DType::BF16.size_in_bytes(),
    )?;
    let vt_buf = scan_scratch(
        ctx,
        value_heads,
        seq_pad,
        seq,
        value_dim * DType::BF16.size_in_bytes(),
    )?;
    let kcd_buf = scan_scratch(
        ctx,
        value_heads,
        seq_pad,
        seq,
        key_dim * DType::BF16.size_in_bytes(),
    )?;
    la::gdn_cumsum(
        ctx,
        &g_buf,
        &g_cum,
        seq_pad,
        value_heads,
        FIRST_PREFILL_CHUNK,
    )?;
    if qk_bf16 {
        la::gdn_attn_raw_solve_f1_qk_bf16(
            ctx,
            &q_buf,
            &k_buf,
            &beta_buf,
            &g_cum,
            &a_buf,
            &t_buf,
            seq_pad,
            value_heads,
            key_dim,
            FIRST_PREFILL_CHUNK,
        )?;
        la::gdn_chunk_gemm_tri_k_bf16_direct_v(
            ctx,
            &a_buf,
            &conv_out,
            &k_buf,
            &beta_buf,
            &g_cum,
            &vt_buf,
            &kcd_buf,
            seq_pad,
            value_heads,
            key_dim,
            value_dim,
            FIRST_PREFILL_CHUNK,
            2 * key_width,
        )?;
        la::gdn_chunk_state_qk_bf16(
            ctx,
            &q_buf,
            &k_buf,
            &g_cum,
            &t_buf,
            &vt_buf,
            &kcd_buf,
            &final_state,
            args.output,
            seq_pad,
            value_heads,
            key_dim,
            value_dim,
            FIRST_PREFILL_CHUNK,
        )?;
    } else {
        la::gdn_attn_raw(
            ctx,
            &q_buf,
            &k_buf,
            &beta_buf,
            &g_cum,
            &a_buf,
            &t_buf,
            seq_pad,
            value_heads,
            key_dim,
            FIRST_PREFILL_CHUNK,
        )?;
        la::gdn_tri_solve(ctx, &a_buf, value_heads * chunks, FIRST_PREFILL_CHUNK)?;
        la::gdn_chunk_gemm(
            ctx,
            &a_buf,
            v_buf
                .as_ref()
                .ok_or_else(|| invalid("GDN V scratch missing"))?,
            &k_buf,
            &beta_buf,
            &g_cum,
            &vt_buf,
            &kcd_buf,
            seq_pad,
            value_heads,
            key_dim,
            value_dim,
            FIRST_PREFILL_CHUNK,
        )?;
        la::gdn_chunk_state(
            ctx,
            &q_buf,
            &k_buf,
            &g_cum,
            &t_buf,
            &vt_buf,
            &kcd_buf,
            &final_state,
            args.output,
            seq_pad,
            value_heads,
            key_dim,
            value_dim,
            FIRST_PREFILL_CHUNK,
        )?;
    }
    Ok(())
}

/// Complete stateful GDN execution after the first prefill.
///
/// The model supplies only semantic tensors. Physical Q/K representation,
/// scratch layout and the recurrent-versus-chunked candidate are private to
/// this family.
pub struct GdnContinuationArgs<'a> {
    pub zba: &'a Tensor,
    pub conv_weight: &'a Tensor,
    pub current_conv_state: &'a Tensor,
    pub new_conv_state: &'a Tensor,
    pub a_log: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub recurrent_state: &'a Tensor,
    pub output: &'a Tensor,
    pub kernel_width: usize,
    /// `[key_heads, value_heads, key_dim, value_dim]`.
    pub geometry: [usize; 4],
}

#[allow(clippy::too_many_lines)]
pub fn gdn_continue(ctx: &CudaContext, args: GdnContinuationArgs<'_>) -> Result<()> {
    use launches as la;

    let [key_heads, value_heads, key_dim, value_dim] = args.geometry;
    let seq = args.zba.shape().dims().first().copied().unwrap_or(0);
    let key_width = key_heads
        .checked_mul(key_dim)
        .ok_or_else(|| invalid("GDN key width overflow"))?;
    let value_width = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| invalid("GDN value width overflow"))?;
    let conv_width = 2 * key_width + value_width;
    let row_width = conv_width + value_width + 2 * value_heads;
    if seq == 0
        || key_heads == 0
        || value_heads == 0
        || value_heads % key_heads != 0
        || key_dim != value_dim
        || args.kernel_width == 0
        || args.zba.shape().dims() != [seq, row_width]
        || args.conv_weight.shape().dims() != [conv_width, args.kernel_width]
        || args.current_conv_state.shape().dims() != [conv_width, args.kernel_width]
        || args.new_conv_state.shape().dims() != [conv_width, args.kernel_width]
        || args.a_log.shape().dims() != [value_heads]
        || args.dt_bias.shape().dims() != [value_heads]
        || args.recurrent_state.shape().dims() != [value_heads, value_dim, key_dim]
        || args.output.shape().dims() != [seq, value_width]
    {
        return Err(invalid("invalid stateful GDN contract"));
    }

    prepare_fixed(ctx, key_dim, value_dim, CONTINUATION_CHUNK)?;

    let conv_out =
        ctx.allocate_output(apxinf_core::Shape::new(vec![seq, conv_width]), DType::BF16)?;
    la::causal_conv1d_silu_bf16(
        ctx,
        args.zba,
        args.conv_weight,
        Some(args.current_conv_state),
        &conv_out,
        args.new_conv_state,
        args.kernel_width,
    )?;

    let recurrent_decode = seq == 1;
    let seq_pad = if recurrent_decode {
        1
    } else {
        seq.div_ceil(CONTINUATION_CHUNK) * CONTINUATION_CHUNK
    };
    let f32_bytes = DType::F32.size_in_bytes();
    let qk_bf16 = la::gdn_qk_bf16_prefill_supported(
        ctx,
        recurrent_decode,
        key_dim,
        value_dim,
        CONTINUATION_CHUNK,
    );
    let qk_bytes = if qk_bf16 {
        DType::BF16.size_in_bytes()
    } else {
        f32_bytes
    };
    let q_buf = scan_scratch(ctx, value_heads, seq_pad, seq, key_dim * qk_bytes)?;
    let k_buf = scan_scratch(ctx, value_heads, seq_pad, seq, key_dim * qk_bytes)?;
    let v_buf = if qk_bf16 {
        None
    } else {
        Some(scan_scratch(
            ctx,
            value_heads,
            seq_pad,
            seq,
            value_dim * DType::BF16.size_in_bytes(),
        )?)
    };
    let beta_buf = scan_scratch(ctx, value_heads, seq_pad, seq, f32_bytes)?;
    let g_buf = scan_scratch(ctx, value_heads, seq_pad, seq, f32_bytes)?;

    if qk_bf16 {
        la::gdn_qk_prep_qk_bf16(
            ctx,
            &conv_out,
            &q_buf,
            &k_buf,
            seq_pad,
            key_heads,
            value_heads,
            key_dim,
            key_width,
            false,
            1e-6,
        )?;
        la::gdn_gate_prep(
            ctx,
            args.zba,
            conv_width + value_width,
            conv_width + value_width + value_heads,
            args.dt_bias,
            args.a_log,
            &beta_buf,
            &g_buf,
            seq_pad,
            value_heads,
        )?;
    } else {
        la::gdn_qk_prep(
            ctx,
            &conv_out,
            &q_buf,
            &k_buf,
            seq_pad,
            key_heads,
            value_heads,
            key_dim,
            key_width,
            recurrent_decode,
            1e-6,
        )?;
        la::gdn_vb_prep(
            ctx,
            &conv_out,
            args.zba,
            conv_width + value_width,
            conv_width + value_width + value_heads,
            args.dt_bias,
            args.a_log,
            v_buf
                .as_ref()
                .ok_or_else(|| invalid("GDN V scratch missing"))?,
            &beta_buf,
            &g_buf,
            seq_pad,
            value_heads,
            value_dim,
            2 * key_width,
        )?;
    }

    let recurrent_state = CudaBuffer::from_tensor(args.recurrent_state).map_err(Error::Cuda)?;
    if recurrent_decode {
        la::gdn_recurrent(
            ctx,
            &q_buf,
            &k_buf,
            v_buf
                .as_ref()
                .ok_or_else(|| invalid("GDN V scratch missing"))?,
            &beta_buf,
            &g_buf,
            &recurrent_state,
            args.output,
            value_heads,
            key_dim,
            value_dim,
        )?;
        return Ok(());
    }

    let chunks = seq_pad / CONTINUATION_CHUNK;
    let g_cum = zeroed(ctx, value_heads * seq_pad * f32_bytes)?;
    let a_buf = zeroed(
        ctx,
        value_heads * chunks * CONTINUATION_CHUNK * CONTINUATION_CHUNK * f32_bytes,
    )?;
    let t_buf = zeroed(
        ctx,
        value_heads
            * chunks
            * CONTINUATION_CHUNK
            * CONTINUATION_CHUNK
            * DType::BF16.size_in_bytes(),
    )?;
    let vt_buf = scan_scratch(
        ctx,
        value_heads,
        seq_pad,
        seq,
        value_dim * DType::BF16.size_in_bytes(),
    )?;
    let kcd_buf = scan_scratch(
        ctx,
        value_heads,
        seq_pad,
        seq,
        key_dim * DType::BF16.size_in_bytes(),
    )?;
    la::gdn_cumsum(
        ctx,
        &g_buf,
        &g_cum,
        seq_pad,
        value_heads,
        CONTINUATION_CHUNK,
    )?;
    if qk_bf16 {
        la::gdn_attn_raw_solve_f1_qk_bf16(
            ctx,
            &q_buf,
            &k_buf,
            &beta_buf,
            &g_cum,
            &a_buf,
            &t_buf,
            seq_pad,
            value_heads,
            key_dim,
            CONTINUATION_CHUNK,
        )?;
        la::gdn_chunk_gemm_tri_k_bf16_direct_v(
            ctx,
            &a_buf,
            &conv_out,
            &k_buf,
            &beta_buf,
            &g_cum,
            &vt_buf,
            &kcd_buf,
            seq_pad,
            value_heads,
            key_dim,
            value_dim,
            CONTINUATION_CHUNK,
            2 * key_width,
        )?;
        la::gdn_chunk_state_qk_bf16(
            ctx,
            &q_buf,
            &k_buf,
            &g_cum,
            &t_buf,
            &vt_buf,
            &kcd_buf,
            &recurrent_state,
            args.output,
            seq_pad,
            value_heads,
            key_dim,
            value_dim,
            CONTINUATION_CHUNK,
        )?;
    } else {
        la::gdn_attn_raw(
            ctx,
            &q_buf,
            &k_buf,
            &beta_buf,
            &g_cum,
            &a_buf,
            &t_buf,
            seq_pad,
            value_heads,
            key_dim,
            CONTINUATION_CHUNK,
        )?;
        la::gdn_tri_solve(ctx, &a_buf, value_heads * chunks, CONTINUATION_CHUNK)?;
        la::gdn_chunk_gemm(
            ctx,
            &a_buf,
            v_buf
                .as_ref()
                .ok_or_else(|| invalid("GDN V scratch missing"))?,
            &k_buf,
            &beta_buf,
            &g_cum,
            &vt_buf,
            &kcd_buf,
            seq_pad,
            value_heads,
            key_dim,
            value_dim,
            CONTINUATION_CHUNK,
        )?;
        la::gdn_chunk_state(
            ctx,
            &q_buf,
            &k_buf,
            &g_cum,
            &t_buf,
            &vt_buf,
            &kcd_buf,
            &recurrent_state,
            args.output,
            seq_pad,
            value_heads,
            key_dim,
            value_dim,
            CONTINUATION_CHUNK,
        )?;
    }
    Ok(())
}

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
    let mut spec = abi::Spec::new(abi::operation::RECURRENT_STEP, 2);
    spec.input_dtype = dtype(DType::BF16)?;
    spec.output_dtype = dtype(DType::BF16)?;
    spec.key_heads = k_heads as i64;
    spec.value_heads = v_heads as i64;
    spec.key_dim = k_dim as i64;
    spec.value_dim = v_dim as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.state = state_buffer.ptr();
    bindings.q = q_buffer.ptr();
    bindings.k = k_buffer.ptr();
    bindings.v = v_buffer.ptr();
    bindings.decay = decay_buffer.ptr();
    bindings.beta = beta_buffer.ptr();
    bindings.output = output_buffer.ptr();
    launch(ctx, &spec, &bindings)
}

/// Optional quantized output produced by [`gdn_gated_norm`].
///
/// `input_scale` preserves the existing static-scale quantization contract;
/// the output format itself is inferred from `tensor`.
#[derive(Clone, Copy)]
pub struct GdnQuantizedOutput<'a> {
    pub tensor: &'a Tensor,
    pub input_scale: f32,
}

/// Complete gated RMSNorm semantic.
///
/// Rank-2 tensors describe one recurrent step as `[heads, head_dim]`; rank-3
/// tensors describe a sequence as `[tokens, heads, head_dim]`. The input dtype
/// and optional quantized output select the preserved native implementation
/// internally rather than multiplying public L3 entry points.
#[derive(Clone, Copy)]
pub enum GdnGate<'a> {
    /// Gate tensor with the same logical shape as the normalized input.
    Dense(&'a Tensor),
    /// Gate values embedded in a wider row. `heads` consecutive gate heads
    /// begin at `column_offset`; flattened input rows select the corresponding
    /// source row and head.
    Strided {
        tensor: &'a Tensor,
        column_offset: usize,
        heads: usize,
    },
}

pub struct GdnGatedNormArgs<'a> {
    pub input: &'a Tensor,
    pub gate: GdnGate<'a>,
    pub weight: &'a Tensor,
    pub output: &'a Tensor,
    pub quantized: Option<GdnQuantizedOutput<'a>>,
    pub epsilon: f32,
}

/// Per-head RMSNorm followed by the swish output gate.
pub fn gdn_gated_norm(ctx: &CudaContext, args: GdnGatedNormArgs<'_>) -> Result<()> {
    match (args.input.shape().dims().len(), args.gate, args.quantized) {
        (2, GdnGate::Strided { .. }, None) => gdn_gated_norm_strided(ctx, &args),
        (2, GdnGate::Dense(_), None) => gdn_gated_norm_step(ctx, &args),
        (3, GdnGate::Dense(_), None) => gdn_gated_norm_sequence(ctx, &args),
        (3, GdnGate::Dense(_), Some(quantized)) => gdn_gated_norm_quantized(ctx, &args, quantized),
        (2, _, Some(_)) => Err(invalid(
            "GDN gated norm quantization expects [tokens, heads, head_dim]",
        )),
        (_, GdnGate::Strided { .. }, _) => Err(invalid(
            "strided GDN gate requires unquantized rank-2 input",
        )),
        _ => Err(invalid(
            "GDN gated norm expects [heads, head_dim] or [tokens, heads, head_dim]",
        )),
    }
}

fn dense_gate<'a>(args: &'a GdnGatedNormArgs<'a>) -> Result<&'a Tensor> {
    match args.gate {
        GdnGate::Dense(gate) => Ok(gate),
        GdnGate::Strided { .. } => Err(invalid("GDN gated norm expected a dense gate")),
    }
}

fn gdn_gated_norm_strided(ctx: &CudaContext, args: &GdnGatedNormArgs<'_>) -> Result<()> {
    let GdnGate::Strided {
        tensor: gate,
        column_offset,
        heads,
    } = args.gate
    else {
        return Err(invalid("GDN gated norm expected a strided gate"));
    };
    let dims = args.input.shape().dims();
    let gate_dims = gate.shape().dims();
    if dims.len() != 2 || gate_dims.len() != 2 {
        return Err(invalid("strided GDN gated norm expects rank-2 tensors"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let (gate_rows, gate_width) = (gate_dims[0], gate_dims[1]);
    if rows == 0
        || cols == 0
        || heads == 0
        || rows % heads != 0
        || gate_rows != rows / heads
        || column_offset
            .checked_add(
                heads
                    .checked_mul(cols)
                    .ok_or_else(|| invalid("strided GDN gate extent overflow"))?,
            )
            .is_none_or(|end| end > gate_width)
        || args.weight.shape().dims() != [cols]
        || args.output.shape().dims() != dims
        || !args.epsilon.is_finite()
        || args.epsilon <= 0.0
    {
        return Err(invalid("invalid strided GDN gated norm contract"));
    }
    let input = tensor_storage(ctx, args.input, DType::BF16, dims)?;
    let gate = tensor_storage(ctx, gate, DType::BF16, gate_dims)?;
    let weight = tensor_storage(ctx, args.weight, DType::BF16, &[cols])?;
    let output = tensor_storage(ctx, args.output, DType::BF16, dims)?;
    let mut spec = abi::Spec::new(abi::operation::GATED_NORM, 2);
    spec.input_dtype = dtype(DType::BF16)?;
    spec.output_dtype = dtype(DType::BF16)?;
    spec.layout = abi::layout::STRIDED_GATE;
    spec.tokens = rows as i64;
    spec.heads = heads as i64;
    spec.head_dim = cols as i64;
    spec.gate_row_stride = gate_width as i64;
    spec.gate_column_offset = column_offset as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.input = input.ptr();
    bindings.gate = gate.ptr();
    bindings.weight = weight.ptr();
    bindings.output = output.ptr();
    bindings.epsilon = args.epsilon;
    launch(ctx, &spec, &bindings)
}

fn gdn_gated_norm_step(ctx: &CudaContext, args: &GdnGatedNormArgs<'_>) -> Result<()> {
    let GdnGatedNormArgs {
        input,
        weight,
        output,
        epsilon,
        ..
    } = args;
    let gate = dense_gate(args)?;
    let dims = input.shape().dims().to_vec();
    let (heads, head_dim) = (dims[0], dims[1]);
    let input_buffer = tensor_storage(ctx, input, DType::BF16, &dims)?;
    let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &[head_dim])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &dims)?;
    let mut spec = abi::Spec::new(abi::operation::GATED_NORM, 2);
    spec.input_dtype = dtype(DType::BF16)?;
    spec.output_dtype = dtype(DType::BF16)?;
    spec.heads = heads as i64;
    spec.head_dim = head_dim as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.input = input_buffer.ptr();
    bindings.gate = gate_buffer.ptr();
    bindings.weight = weight_buffer.ptr();
    bindings.output = output_buffer.ptr();
    bindings.epsilon = *epsilon;
    launch(ctx, &spec, &bindings)
}

fn gdn_gated_norm_quantized(
    ctx: &CudaContext,
    args: &GdnGatedNormArgs<'_>,
    quantized: GdnQuantizedOutput<'_>,
) -> Result<()> {
    let GdnGatedNormArgs {
        input,
        weight,
        output,
        epsilon,
        ..
    } = args;
    let gate = dense_gate(args)?;
    let dims = input.shape().dims().to_vec();
    if dims.len() != 3 || dims[2] != 128 || dims[0] == 0 || dims[1] == 0 {
        return Err(invalid(
            "GDN norm quantization expects nonempty [tokens, heads, 128]",
        ));
    }
    dims[0]
        .checked_mul(dims[1])
        .filter(|value| *value <= i32::MAX as usize)
        .ok_or_else(|| invalid("GDN norm quantization row extent overflows i32"))?;
    if !epsilon.is_finite()
        || *epsilon <= 0.0
        || !quantized.input_scale.is_finite()
        || quantized.input_scale <= 0.0
        || !(1.0 / quantized.input_scale).is_finite()
        || !matches!(input.dtype(), DType::F16 | DType::BF16)
    {
        return Err(invalid(
            "GDN norm quantization requires positive finite epsilon and scale",
        ));
    }
    let input_buffer = tensor_storage(ctx, input, input.dtype(), &dims)?;
    let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &[128])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &dims)?;
    let quantized_buffer = tensor_storage(ctx, quantized.tensor, DType::F8E4M3, &dims)?;
    let mut spec = abi::Spec::new(abi::operation::GATED_NORM, 3);
    spec.input_dtype = dtype(input.dtype())?;
    spec.output_dtype = dtype(DType::BF16)?;
    spec.quantization = abi::quantization::STATIC_E4M3;
    spec.tokens = dims[0] as i64;
    spec.heads = dims[1] as i64;
    spec.head_dim = dims[2] as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.input = input_buffer.ptr();
    bindings.gate = gate_buffer.ptr();
    bindings.weight = weight_buffer.ptr();
    bindings.output = output_buffer.ptr();
    bindings.quantized = quantized_buffer.ptr();
    bindings.epsilon = *epsilon;
    bindings.input_scale = quantized.input_scale;
    launch(ctx, &spec, &bindings)
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
    let mut spec = abi::Spec::new(abi::operation::CAUSAL_CONV, 1);
    spec.input_dtype = dtype(DType::BF16)?;
    spec.output_dtype = dtype(DType::BF16)?;
    spec.has_window = 1;
    spec.channels = channels as i64;
    spec.kernel_width = kernel_width as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.input = input_buffer.ptr();
    bindings.weight = weight_buffer.ptr();
    bindings.output = output_buffer.ptr();
    bindings.window = window_buffer.ptr();
    launch(ctx, &spec, &bindings)
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
    let mut spec = abi::Spec::new(abi::operation::CAUSAL_CONV, 2);
    spec.input_dtype = dtype(DType::BF16)?;
    spec.output_dtype = dtype(DType::BF16)?;
    spec.has_window = u32::from(window_buffer.is_some());
    spec.tokens = tokens as i64;
    spec.channels = channels as i64;
    spec.kernel_width = kernel_width as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.input = input_buffer.ptr();
    bindings.weight = weight_buffer.ptr();
    bindings.output = output_buffer.ptr();
    bindings.window = window_buffer
        .as_ref()
        .map_or(std::ptr::null_mut(), |buffer| buffer.ptr());
    launch(ctx, &spec, &bindings)
}

/// Complete GDN decay/beta preparation contract.
pub struct GdnDecayAndBetaArgs<'a> {
    pub a: &'a Tensor,
    pub b: &'a Tensor,
    pub a_log: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub decay: &'a Tensor,
    pub beta: &'a Tensor,
}

/// `decay = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)`.
///
/// Rank-1 `a`/`b` describe one recurrent step; rank-2 tensors describe
/// `[tokens, heads]`. Geometry is inferred from the tensors.
pub fn gdn_decay_and_beta(ctx: &CudaContext, args: GdnDecayAndBetaArgs<'_>) -> Result<()> {
    match args.a.shape().dims().len() {
        1 => gdn_decay_and_beta_step(ctx, &args),
        2 => gdn_decay_and_beta_sequence(ctx, &args),
        _ => Err(invalid(
            "GDN gate inputs must be [heads] or [tokens, heads]",
        )),
    }
}

fn gdn_decay_and_beta_sequence(ctx: &CudaContext, args: &GdnDecayAndBetaArgs<'_>) -> Result<()> {
    let GdnDecayAndBetaArgs {
        a,
        b,
        a_log,
        dt_bias,
        decay,
        beta,
    } = args;
    let rows = a.shape().dims().to_vec();
    let (tokens, heads) = (rows[0], rows[1]);
    let a_buffer = tensor_storage(ctx, a, DType::BF16, &rows)?;
    let b_buffer = tensor_storage(ctx, b, DType::BF16, &rows)?;
    let a_log_buffer = tensor_storage(ctx, a_log, DType::BF16, &[heads])?;
    let dt_bias_buffer = tensor_storage(ctx, dt_bias, DType::BF16, &[heads])?;
    let decay_buffer = tensor_storage(ctx, decay, DType::F32, &rows)?;
    let beta_buffer = tensor_storage(ctx, beta, DType::F32, &rows)?;
    let mut spec = abi::Spec::new(abi::operation::DECAY_AND_BETA, 2);
    spec.input_dtype = dtype(DType::BF16)?;
    spec.output_dtype = dtype(DType::F32)?;
    spec.tokens = tokens as i64;
    spec.heads = heads as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.a = a_buffer.ptr();
    bindings.b = b_buffer.ptr();
    bindings.a_log = a_log_buffer.ptr();
    bindings.dt_bias = dt_bias_buffer.ptr();
    bindings.decay = decay_buffer.ptr();
    bindings.beta = beta_buffer.ptr();
    launch(ctx, &spec, &bindings)
}

/// Sequence implementation. FP16 inputs are rounded to BF16 before
/// normalization, matching the existing explicit conversion contract.
fn gdn_gated_norm_sequence(ctx: &CudaContext, args: &GdnGatedNormArgs<'_>) -> Result<()> {
    let GdnGatedNormArgs {
        input,
        weight,
        output,
        epsilon,
        ..
    } = args;
    let gate = dense_gate(args)?;
    let dims = input.shape().dims().to_vec();
    let (tokens, heads, head_dim) = (dims[0], dims[1], dims[2]);
    let input_dtype = input.dtype();
    if input_dtype != DType::BF16 && input_dtype != DType::F16 {
        return Err(invalid("GDN sequence gated norm expects BF16 or F16 input"));
    }
    let input_buffer = tensor_storage(ctx, input, input_dtype, &dims)?;
    let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &[head_dim])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &dims)?;
    let mut spec = abi::Spec::new(abi::operation::GATED_NORM, 3);
    spec.input_dtype = dtype(input_dtype)?;
    spec.output_dtype = dtype(DType::BF16)?;
    spec.tokens = tokens as i64;
    spec.heads = heads as i64;
    spec.head_dim = head_dim as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.input = input_buffer.ptr();
    bindings.gate = gate_buffer.ptr();
    bindings.weight = weight_buffer.ptr();
    bindings.output = output_buffer.ptr();
    bindings.epsilon = *epsilon;
    launch(ctx, &spec, &bindings)
}

/// L2-normalize each head in place.
pub fn gdn_l2_normalize_heads(ctx: &CudaContext, data: &Tensor, epsilon: f32) -> Result<()> {
    let dims = data.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("GDN head normalization expects [heads, head_dim]"));
    }
    let buffer = tensor_storage(ctx, data, DType::BF16, &dims)?;
    let mut spec = abi::Spec::new(abi::operation::L2_NORMALIZE, 2);
    spec.input_dtype = dtype(DType::BF16)?;
    spec.output_dtype = dtype(DType::BF16)?;
    spec.heads = dims[0] as i64;
    spec.head_dim = dims[1] as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.output = buffer.ptr();
    bindings.epsilon = epsilon;
    launch(ctx, &spec, &bindings)
}

fn gdn_decay_and_beta_step(ctx: &CudaContext, args: &GdnDecayAndBetaArgs<'_>) -> Result<()> {
    let GdnDecayAndBetaArgs {
        a,
        b,
        a_log,
        dt_bias,
        decay,
        beta,
    } = args;
    let dims = a.shape().dims().to_vec();
    let heads = dims[0];
    let a_buffer = tensor_storage(ctx, a, DType::BF16, &dims)?;
    let b_buffer = tensor_storage(ctx, b, DType::BF16, &dims)?;
    let a_log_buffer = tensor_storage(ctx, a_log, DType::BF16, &dims)?;
    let dt_bias_buffer = tensor_storage(ctx, dt_bias, DType::BF16, &dims)?;
    let decay_buffer = tensor_storage(ctx, decay, DType::F32, &dims)?;
    let beta_buffer = tensor_storage(ctx, beta, DType::F32, &dims)?;
    let mut spec = abi::Spec::new(abi::operation::DECAY_AND_BETA, 1);
    spec.input_dtype = dtype(DType::BF16)?;
    spec.output_dtype = dtype(DType::F32)?;
    spec.heads = heads as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.a = a_buffer.ptr();
    bindings.b = b_buffer.ptr();
    bindings.a_log = a_log_buffer.ptr();
    bindings.dt_bias = dt_bias_buffer.ptr();
    bindings.decay = decay_buffer.ptr();
    bindings.beta = beta_buffer.ptr();
    launch(ctx, &spec, &bindings)
}

/// Q/K/V bindings consumed by [`gdn_chunk_scan`].
#[derive(Clone, Copy)]
pub enum GdnChunkScanInput<'a> {
    /// Separate contiguous `[seq, heads, dim]` tensors.
    Split {
        q: &'a Tensor,
        k: &'a Tensor,
        v: &'a Tensor,
    },
    /// One `[seq, row_width]` projection with per-row element offsets for Q/K/V.
    Interleaved {
        fused: &'a Tensor,
        offsets: [usize; 3],
    },
}

/// Complete chunked GDN scan contract.
pub struct GdnChunkScanArgs<'a> {
    pub input: GdnChunkScanInput<'a>,
    pub gate_log: &'a Tensor,
    pub beta: &'a Tensor,
    pub output: &'a Tensor,
    pub state: &'a Tensor,
    pub key_heads: usize,
    pub chunk_size: usize,
}

/// Chunked parallel scan over padded prefill tokens.
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
pub fn gdn_chunk_scan(ctx: &CudaContext, args: GdnChunkScanArgs<'_>) -> Result<()> {
    match args.input {
        GdnChunkScanInput::Split { q, k, v } => gdn_chunk_scan_split(ctx, &args, q, k, v),
        GdnChunkScanInput::Interleaved { fused, offsets } => {
            gdn_chunk_scan_interleaved(ctx, &args, fused, offsets)
        }
    }
}

fn gdn_chunk_scan_split(
    ctx: &CudaContext,
    args: &GdnChunkScanArgs<'_>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<()> {
    let state_dims = args.state.shape().dims().to_vec();
    if state_dims.len() != 3 {
        return Err(invalid("GDN state must be [v_heads, v_dim, k_dim]"));
    }
    let (v_heads, v_dim, k_dim) = (state_dims[0], state_dims[1], state_dims[2]);
    if v_dim != k_dim {
        return Err(invalid(
            "GDN chunk scan expects square v_dim == k_dim state",
        ));
    }
    let k_heads = args.key_heads;
    if k_heads == 0 || v_heads % k_heads != 0 {
        return Err(invalid("GDN value heads must be a multiple of key heads"));
    }
    if args.chunk_size == 0 {
        return Err(invalid("GDN chunk size must be positive"));
    }
    let q_dims = q.shape().dims().to_vec();
    if q_dims.len() != 3 {
        return Err(invalid("GDN chunk scan q must be [seq, k_heads, k_dim]"));
    }
    let seq_padded = q_dims[0];
    if seq_padded % args.chunk_size != 0 {
        return Err(invalid(
            "GDN chunk scan seq must be a multiple of chunk_size",
        ));
    }
    let num_chunks = seq_padded / args.chunk_size;

    let q_buffer = tensor_storage(ctx, q, DType::BF16, &[seq_padded, k_heads, k_dim])?;
    let k_buffer = tensor_storage(ctx, k, DType::BF16, &[seq_padded, k_heads, k_dim])?;
    let v_buffer = tensor_storage(ctx, v, DType::BF16, &[seq_padded, v_heads, v_dim])?;
    let g_buffer = tensor_storage(ctx, args.gate_log, DType::F32, &[seq_padded, v_heads])?;
    let beta_buffer = tensor_storage(ctx, args.beta, DType::F32, &[seq_padded, v_heads])?;
    let out_buffer = tensor_storage(ctx, args.output, DType::BF16, &[seq_padded, v_heads, v_dim])?;
    let state_buffer = tensor_storage(ctx, args.state, DType::F32, &state_dims)?;
    prepare_chunk_scan(ctx)?;
    let mut spec = abi::Spec::new(abi::operation::CHUNK_SCAN, 3);
    spec.input_dtype = dtype(DType::BF16)?;
    spec.output_dtype = dtype(DType::BF16)?;
    spec.layout = abi::layout::SPLIT_QKV;
    spec.tokens = seq_padded as i64;
    spec.key_heads = k_heads as i64;
    spec.value_heads = v_heads as i64;
    spec.key_dim = k_dim as i64;
    spec.value_dim = v_dim as i64;
    spec.chunk_size = args.chunk_size as i64;
    spec.num_chunks = num_chunks as i64;
    // Contiguous [seq, heads, dim] inputs: one token is heads*dim apart.
    spec.q_row_stride = (k_heads * k_dim) as i64;
    spec.k_row_stride = (k_heads * k_dim) as i64;
    spec.v_row_stride = (v_heads * k_dim) as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.q = q_buffer.ptr();
    bindings.k = k_buffer.ptr();
    bindings.v = v_buffer.ptr();
    bindings.gate_log = g_buffer.ptr();
    bindings.beta = beta_buffer.ptr();
    bindings.output = out_buffer.ptr();
    bindings.state = state_buffer.ptr();
    launch(ctx, &spec, &bindings)
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
fn gdn_chunk_scan_interleaved(
    ctx: &CudaContext,
    args: &GdnChunkScanArgs<'_>,
    fused: &Tensor,
    offsets: [usize; 3],
) -> Result<()> {
    let state_dims = args.state.shape().dims();
    if state_dims.len() != 3 || state_dims[1] != state_dims[2] {
        return Err(invalid(
            "GDN chunk scan expects square [v_heads, dim, dim] state",
        ));
    }
    let (v_heads, k_dim) = (state_dims[0], state_dims[1]);
    let k_heads = args.key_heads;
    if k_heads == 0 || v_heads % k_heads != 0 || args.chunk_size == 0 {
        return Err(invalid("invalid GDN chunk scan head or chunk geometry"));
    }
    let fused_dims = fused.shape().dims();
    if fused_dims.len() != 2 {
        return Err(invalid("interleaved GDN input must be [seq, row_width]"));
    }
    let (seq_padded, row_width) = (fused_dims[0], fused_dims[1]);
    if seq_padded % args.chunk_size != 0 {
        return Err(invalid("GDN chunk scan needs a multiple of chunk_size"));
    }
    let [q_offset, k_offset, v_offset] = offsets;
    if q_offset + k_heads * k_dim > row_width
        || k_offset + k_heads * k_dim > row_width
        || v_offset + v_heads * k_dim > row_width
    {
        return Err(invalid("GDN chunk scan offsets run past the fused row"));
    }
    let num_chunks = seq_padded / args.chunk_size;
    let element = DType::BF16.size_in_bytes();

    let fused_buffer = tensor_storage(ctx, fused, DType::BF16, &[seq_padded, row_width])?;
    let g_buffer = tensor_storage(ctx, args.gate_log, DType::F32, &[seq_padded, v_heads])?;
    let beta_buffer = tensor_storage(ctx, args.beta, DType::F32, &[seq_padded, v_heads])?;
    let out_buffer = tensor_storage(ctx, args.output, DType::BF16, &[seq_padded, v_heads, k_dim])?;
    let state_buffer = tensor_storage(ctx, args.state, DType::F32, state_dims)?;
    prepare_chunk_scan(ctx)?;

    // Byte offsets into the one fused allocation; the kernel walks rows itself.
    let base = fused_buffer.ptr() as *const u8;
    let mut spec = abi::Spec::new(abi::operation::CHUNK_SCAN, 3);
    spec.input_dtype = dtype(DType::BF16)?;
    spec.output_dtype = dtype(DType::BF16)?;
    spec.layout = abi::layout::INTERLEAVED_QKV;
    spec.tokens = seq_padded as i64;
    spec.key_heads = k_heads as i64;
    spec.value_heads = v_heads as i64;
    spec.key_dim = k_dim as i64;
    spec.value_dim = k_dim as i64;
    spec.chunk_size = args.chunk_size as i64;
    spec.num_chunks = num_chunks as i64;
    spec.q_row_stride = row_width as i64;
    spec.k_row_stride = row_width as i64;
    spec.v_row_stride = row_width as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    unsafe {
        bindings.q = base.add(q_offset * element) as *const std::ffi::c_void;
        bindings.k = base.add(k_offset * element) as *const std::ffi::c_void;
        bindings.v = base.add(v_offset * element) as *const std::ffi::c_void;
    }
    bindings.gate_log = g_buffer.ptr();
    bindings.beta = beta_buffer.ptr();
    bindings.output = out_buffer.ptr();
    bindings.state = state_buffer.ptr();
    launch(ctx, &spec, &bindings)
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
pub fn gdn_prefill(
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
    let state_buffer = tensor_storage(ctx, state, DType::F32, &[v_heads, head_dim, head_dim])?;
    // The kernel treats this as raw bytes; validating it against its own
    // declared shape keeps the check meaningful without inventing a dtype.
    let workspace_dims = workspace.shape().dims().to_vec();
    let workspace_buffer = tensor_storage(ctx, workspace, workspace.dtype(), &workspace_dims)?;

    let mut spec = abi::Spec::new(abi::operation::PREFILL, 3);
    spec.input_dtype = dtype(DType::F16)?;
    spec.output_dtype = dtype(DType::F16)?;
    spec.layout = abi::layout::SPLIT_QKV;
    spec.tokens = tokens as i64;
    spec.key_heads = q_heads as i64;
    spec.value_heads = v_heads as i64;
    spec.key_dim = head_dim as i64;
    spec.value_dim = head_dim as i64;
    spec.num_sequences = num_seqs as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.q = q_buffer.ptr();
    bindings.k = k_buffer.ptr();
    bindings.v = v_buffer.ptr();
    bindings.output = out_buffer.ptr();
    bindings.gate_log = gate_buffer.ptr();
    bindings.beta = beta_buffer.ptr();
    bindings.cu_seqlens = seqlens_buffer.ptr();
    bindings.state = state_buffer.ptr();
    bindings.workspace = workspace_buffer.ptr();
    bindings.scale = scale;
    launch(ctx, &spec, &bindings)
}

/// Scratch bytes `gdn_prefill` needs for its TMA rewrites.
pub fn gdn_prefill_workspace_bytes(v_heads: usize, num_seqs: usize) -> usize {
    unsafe {
        abi::apxinf_gdn_prefill_workspace_bytes(v_heads as i64, num_seqs as i64).max(0) as usize
    }
}

/// One-pass conversion of a GDN projection into FlashInfer's input form.
///
/// `gdn_prefill` wants q/k/v split into contiguous FP16 tensors
/// with q and k L2-normalized and q unscaled, plus a linear-space decay. Doing
/// those as separate ops would reread the projection four times; at 2048
/// tokens it is 40 MB, so the conversion would rival the kernel it feeds.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prepare_prefill(
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
    let mut spec = abi::Spec::new(abi::operation::PREPARE_PREFILL, 2);
    spec.input_dtype = dtype(DType::BF16)?;
    spec.output_dtype = dtype(DType::F16)?;
    spec.layout = abi::layout::INTERLEAVED_QKV;
    spec.tokens = tokens as i64;
    spec.row_width = row_width as i64;
    spec.key_heads = k_heads as i64;
    spec.value_heads = v_heads as i64;
    spec.head_dim = dim as i64;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.input = fused_buffer.ptr();
    bindings.q_output = q_buffer.ptr();
    bindings.k_output = k_buffer.ptr();
    bindings.v_output = v_buffer.ptr();
    bindings.gate_log = g_buffer.ptr();
    bindings.alpha = alpha_buffer.ptr();
    bindings.epsilon = epsilon;
    launch(ctx, &spec, &bindings)
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_conv_prepare(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    window: &Tensor,
    q_out: &Tensor,
    k_out: &Tensor,
    v_out: &Tensor,
    decay: &Tensor,
    alpha: &Tensor,
    tokens: usize,
    k_heads: usize,
    v_heads: usize,
    epsilon: f32,
) -> Result<()> {
    let channels = k_heads
        .checked_mul(2)
        .and_then(|heads| heads.checked_add(v_heads))
        .and_then(|heads| heads.checked_mul(128))
        .ok_or_else(|| invalid("fused GDN channel count overflow"))?;
    if tokens == 0
        || tokens > i32::MAX as usize
        || k_heads == 0
        || v_heads == 0
        || channels > 536_870_911
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err(invalid("invalid fused GDN dimensions or epsilon"));
    }
    let input_buffer = tensor_storage(ctx, input, DType::BF16, &[tokens, channels])?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &[channels, 4])?;
    let window_buffer = tensor_storage(ctx, window, DType::F32, &[channels, 4])?;
    let q_buffer = tensor_storage(ctx, q_out, DType::F16, &[tokens, k_heads, 128])?;
    let k_buffer = tensor_storage(ctx, k_out, DType::F16, &[tokens, k_heads, 128])?;
    let v_buffer = tensor_storage(ctx, v_out, DType::F16, &[tokens, v_heads, 128])?;
    let decay_buffer = tensor_storage(ctx, decay, DType::F32, &[tokens, v_heads])?;
    let alpha_buffer = tensor_storage(ctx, alpha, DType::F32, &[tokens, v_heads])?;
    let mut spec = abi::Spec::new(abi::operation::CONV_PREPARE, 2);
    spec.input_dtype = dtype(DType::BF16)?;
    spec.output_dtype = dtype(DType::F16)?;
    spec.layout = abi::layout::INTERLEAVED_QKV;
    spec.tokens = tokens as i64;
    spec.key_heads = k_heads as i64;
    spec.value_heads = v_heads as i64;
    spec.head_dim = 128;
    spec.channels = channels as i64;
    spec.kernel_width = 4;
    let mut bindings = abi::Bindings::new(ctx.stream().handle());
    bindings.input = input_buffer.ptr();
    bindings.weight = weight_buffer.ptr();
    bindings.window = window_buffer.ptr();
    bindings.q_output = q_buffer.ptr();
    bindings.k_output = k_buffer.ptr();
    bindings.v_output = v_buffer.ptr();
    bindings.decay = decay_buffer.ptr();
    bindings.alpha = alpha_buffer.ptr();
    bindings.epsilon = epsilon;
    launch(ctx, &spec, &bindings)
}
