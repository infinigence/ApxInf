//! Linear-attention / hybrid-recurrent operator contracts.
//!
//! Model-neutral safe wrappers for the gated delta rule (chunked prefill and
//! rank-1 recurrent decode), depthwise causal conv1d with SiLU, gated RMSNorm,
//! partial rotary application from precomputed tables, adaLN helpers, and
//! small dtype/layout utilities. Physical implementations live in
//! `kernels/custom/linear_attention.cuh`; every wrapper owns validation and
//! the raw FFI call.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, check_cuda, gpu_ptr, make_gpu_tensor, matrix_shape, optional_ptr,
    require_buffers,
};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;

fn expect_bf16(tensor: &Tensor, name: &str) -> Result<()> {
    if tensor.dtype() != DType::BF16 {
        return Err(Error::Other(format!("linear-attention {name} expects BF16")));
    }
    Ok(())
}

fn f32_bytes(elements: usize) -> Result<usize> {
    elements
        .checked_mul(DType::F32.size_in_bytes())
        .ok_or_else(|| Error::Other("linear-attention buffer size overflow".into()))
}

/// Cast an F32 tensor to BF16 (round-to-nearest-even).
pub fn cast_f32_to_bf16(ctx: &CudaContext, input: &Tensor, output: &Tensor) -> Result<()> {
    if input.dtype() != DType::F32 || output.dtype() != DType::BF16 || input.numel() != output.numel() {
        return Err(Error::Other("cast f32->bf16 shape/dtype mismatch".into()));
    }
    unsafe {
        check_cuda(ffi::apxinf_static_cast_f32_bf16(
            gpu_ptr(input)?,
            gpu_ptr(output)?,
            input.numel() as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Cast a BF16 tensor to F32 (exact widening).
pub fn cast_bf16_to_f32(ctx: &CudaContext, input: &Tensor, output: &Tensor) -> Result<()> {
    if input.dtype() != DType::BF16 || output.dtype() != DType::F32 || input.numel() != output.numel() {
        return Err(Error::Other("cast bf16->f32 shape/dtype mismatch".into()));
    }
    unsafe {
        check_cuda(ffi::apxinf_static_cast_bf16_f32(
            gpu_ptr(input)?,
            gpu_ptr(output)?,
            input.numel() as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Depthwise causal conv1d + SiLU over a strided token-major stream.
///
/// `x` is `[seq, x_row_stride]` BF16; the mixed projection occupies its first
/// `channels` columns. `out` is dense `[seq, channels]`. `state` (optional)
/// and `new_state` are `[channels, kernel_size]`; the new state is written to
/// `new_state`, which must not alias `state`.
pub fn causal_conv1d_silu_bf16(
    ctx: &CudaContext,
    x: &Tensor,
    weight: &Tensor,
    state: Option<&Tensor>,
    out: &Tensor,
    new_state: &Tensor,
    kernel_size: usize,
) -> Result<()> {
    let (seq, x_stride) = matrix_shape(x, "causal conv1d")?;
    let (out_seq, channels) = matrix_shape(out, "causal conv1d")?;
    if seq == 0
        || out_seq != seq
        || channels == 0
        || x_stride < channels
        || kernel_size == 0
        || kernel_size > 8
        || weight.shape().dims() != [channels, kernel_size]
        || new_state.shape().dims() != [channels, kernel_size]
        || state.is_some_and(|value| value.shape().dims() != [channels, kernel_size])
    {
        return Err(Error::Other("causal conv1d shape mismatch".into()));
    }
    for tensor in [x, weight, out, new_state] {
        expect_bf16(tensor, "causal conv1d")?;
    }
    if let Some(state) = state {
        expect_bf16(state, "causal conv1d")?;
    }
    unsafe {
        check_cuda(ffi::apxinf_static_causal_conv1d_silu_bf16(
            gpu_ptr(x)?,
            gpu_ptr(weight)?,
            optional_ptr(state)?,
            gpu_ptr(out)?,
            gpu_ptr(new_state)?,
            channels as i32,
            seq as i32,
            kernel_size as i32,
            x_stride as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Gated-delta-rule q/k preparation: L2-normalize q/k rows of the post-conv
/// stream (fp32, eps), scale q by 1/sqrt(head_k_dim), scatter key heads to the
/// value-head layout. Outputs are caller-owned fp32 buffers
/// `[num_v_heads, seq_pad, head_k_dim]` (zero-filled tail by the caller).
pub fn gdn_qk_prep(
    ctx: &CudaContext,
    conv_out: &Tensor,
    q: &CudaBuffer,
    k: &CudaBuffer,
    seq_pad: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    key_dim: usize,
    eps: f32,
) -> Result<()> {
    let (seq, conv_dim) = matrix_shape(conv_out, "GDN qk prep")?;
    if seq == 0
        || seq > seq_pad
        || conv_dim < 2 * key_dim
        || key_dim != num_k_heads * head_k_dim
        || num_v_heads == 0
        || num_k_heads == 0
        || num_v_heads % num_k_heads != 0
        || head_k_dim == 0
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other("GDN qk prep shape mismatch".into()));
    }
    expect_bf16(conv_out, "GDN qk prep")?;
    require_buffers(
        ctx,
        "GDN qk prep",
        &[
            ("q", q, f32_bytes(num_v_heads * seq_pad * head_k_dim)?),
            ("k", k, f32_bytes(num_v_heads * seq_pad * head_k_dim)?),
        ],
    )?;
    let scale = 1.0f32 / (head_k_dim as f32).sqrt();
    unsafe {
        check_cuda(ffi::apxinf_static_gdn_qk_prep_bf16(
            gpu_ptr(conv_out)?,
            q.ptr(),
            k.ptr(),
            seq as i32,
            seq_pad as i32,
            conv_dim as i32,
            key_dim as i32,
            num_v_heads as i32,
            head_k_dim as i32,
            scale,
            eps,
            ctx.stream().handle(),
        ))
    }
}

/// GDN value/beta/decay preparation. `b_col`/`a_col` are the column offsets of
/// the beta and decay projections inside the fused `zba` output; `v_offset` is
/// the value channel offset inside `conv_out`. Outputs: fp32 head-major
/// `v [num_v_heads, seq_pad, head_v_dim]`, `beta/g [num_v_heads, seq_pad]`.
pub fn gdn_vb_prep(
    ctx: &CudaContext,
    conv_out: &Tensor,
    zba: &Tensor,
    b_col: usize,
    a_col: usize,
    dt_bias: &Tensor,
    a_log: &Tensor,
    v: &CudaBuffer,
    beta: &CudaBuffer,
    g: &CudaBuffer,
    seq_pad: usize,
    num_v_heads: usize,
    head_v_dim: usize,
    v_offset: usize,
) -> Result<()> {
    let (seq, conv_dim) = matrix_shape(conv_out, "GDN vb prep")?;
    let (zba_seq, zba_width) = matrix_shape(zba, "GDN vb prep")?;
    if seq == 0
        || seq > seq_pad
        || zba_seq != seq
        || conv_dim < v_offset + num_v_heads * head_v_dim
        || a_col + num_v_heads > zba_width
        || b_col >= a_col
        || num_v_heads == 0
        || head_v_dim == 0
        || dt_bias.shape().dims() != [num_v_heads]
        || a_log.shape().dims() != [num_v_heads]
        || dt_bias.dtype() != DType::F32
        || a_log.dtype() != DType::F32
    {
        return Err(Error::Other("GDN vb prep shape mismatch".into()));
    }
    expect_bf16(conv_out, "GDN vb prep")?;
    expect_bf16(zba, "GDN vb prep")?;
    require_buffers(
        ctx,
        "GDN vb prep",
        &[
            ("v", v, f32_bytes(num_v_heads * seq_pad * head_v_dim)?),
            ("beta", beta, f32_bytes(num_v_heads * seq_pad)?),
            ("g", g, f32_bytes(num_v_heads * seq_pad)?),
        ],
    )?;
    let element = DType::BF16.size_in_bytes();
    let b_ptr = gpu_ptr(zba)?.cast::<u8>().wrapping_add(b_col * element).cast();
    let a_ptr = gpu_ptr(zba)?.cast::<u8>().wrapping_add(a_col * element).cast();
    unsafe {
        check_cuda(ffi::apxinf_static_gdn_vb_prep_bf16(
            gpu_ptr(conv_out)?,
            b_ptr,
            a_ptr,
            gpu_ptr(dt_bias)?,
            gpu_ptr(a_log)?,
            v.ptr(),
            beta.ptr(),
            g.ptr(),
            seq as i32,
            seq_pad as i32,
            conv_dim as i32,
            v_offset as i32,
            num_v_heads as i32,
            zba_width as i32,
            head_v_dim as i32,
            ctx.stream().handle(),
        ))
    }
}

/// Chunk-local inclusive cumsum of the decay `g`.
pub fn gdn_cumsum(
    ctx: &CudaContext,
    g: &CudaBuffer,
    g_cum: &CudaBuffer,
    seq_pad: usize,
    num_v_heads: usize,
    chunk_size: usize,
) -> Result<()> {
    if seq_pad == 0 || seq_pad % chunk_size != 0 || chunk_size == 0 || num_v_heads == 0 {
        return Err(Error::Other("GDN cumsum shape mismatch".into()));
    }
    require_buffers(
        ctx,
        "GDN cumsum",
        &[
            ("g", g, f32_bytes(num_v_heads * seq_pad)?),
            ("g_cum", g_cum, f32_bytes(num_v_heads * seq_pad)?),
        ],
    )?;
    unsafe {
        check_cuda(ffi::apxinf_static_gdn_cumsum_f32(
            g.ptr(),
            g_cum.ptr(),
            seq_pad as i32,
            num_v_heads as i32,
            chunk_size as i32,
            ctx.stream().handle(),
        ))
    }
}

/// Raw masked intra-chunk matrices: A = -(k_beta @ k^T) * decay (strictly
/// lower), T = (q @ k^T) * decay (lower). Caller-owned fp32 buffers
/// `[num_v_heads, chunks, chunk_size, chunk_size]`.
pub fn gdn_attn_raw(
    ctx: &CudaContext,
    q: &CudaBuffer,
    k: &CudaBuffer,
    beta: &CudaBuffer,
    g_cum: &CudaBuffer,
    a: &CudaBuffer,
    t: &CudaBuffer,
    seq_pad: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    chunk_size: usize,
) -> Result<()> {
    let chunks = seq_pad.checked_div(chunk_size).unwrap_or(0);
    if seq_pad == 0 || chunks == 0 || seq_pad % chunk_size != 0 || num_v_heads == 0 {
        return Err(Error::Other("GDN attn raw shape mismatch".into()));
    }
    let matrix = chunk_size * chunk_size;
    require_buffers(
        ctx,
        "GDN attn raw",
        &[
            ("q", q, f32_bytes(num_v_heads * seq_pad * head_k_dim)?),
            ("k", k, f32_bytes(num_v_heads * seq_pad * head_k_dim)?),
            ("beta", beta, f32_bytes(num_v_heads * seq_pad)?),
            ("g_cum", g_cum, f32_bytes(num_v_heads * seq_pad)?),
            ("a", a, f32_bytes(num_v_heads * chunks * matrix)?),
            ("t", t, f32_bytes(num_v_heads * chunks * matrix)?),
        ],
    )?;
    unsafe {
        check_cuda(ffi::apxinf_static_gdn_attn_raw_f32(
            q.ptr(),
            k.ptr(),
            beta.ptr(),
            g_cum.ptr(),
            a.ptr(),
            t.ptr(),
            seq_pad as i32,
            num_v_heads as i32,
            head_k_dim as i32,
            chunk_size as i32,
            ctx.stream().handle(),
        ))
    }
}

/// In-place forward substitution + identity on each chunk matrix, producing
/// (I - A)^-1. `matrices` is the total chunk-matrix count (heads * chunks).
pub fn gdn_tri_solve(ctx: &CudaContext, a: &CudaBuffer, matrices: usize, chunk_size: usize) -> Result<()> {
    if matrices == 0 || chunk_size == 0 || chunk_size > 128 {
        return Err(Error::Other("GDN triangular solve shape mismatch".into()));
    }
    require_buffers(
        ctx,
        "GDN triangular solve",
        &[("a", a, f32_bytes(matrices * chunk_size * chunk_size)?)],
    )?;
    unsafe {
        check_cuda(ffi::apxinf_static_gdn_tri_solve_f32(
            a.ptr(),
            matrices as i32,
            chunk_size as i32,
            ctx.stream().handle(),
        ))
    }
}

/// VT = A @ (v * beta), KCD = A @ (k * beta * exp(g_cum)) per chunk.
pub fn gdn_chunk_gemm(
    ctx: &CudaContext,
    a: &CudaBuffer,
    v: &CudaBuffer,
    k: &CudaBuffer,
    beta: &CudaBuffer,
    g_cum: &CudaBuffer,
    vt: &CudaBuffer,
    kcd: &CudaBuffer,
    seq_pad: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_size: usize,
) -> Result<()> {
    let chunks = seq_pad.checked_div(chunk_size).unwrap_or(0);
    if seq_pad == 0 || chunks == 0 || seq_pad % chunk_size != 0 || num_v_heads == 0 {
        return Err(Error::Other("GDN chunk gemm shape mismatch".into()));
    }
    require_buffers(
        ctx,
        "GDN chunk gemm",
        &[
            ("a", a, f32_bytes(num_v_heads * chunks * chunk_size * chunk_size)?),
            ("v", v, f32_bytes(num_v_heads * seq_pad * head_v_dim)?),
            ("k", k, f32_bytes(num_v_heads * seq_pad * head_k_dim)?),
            ("beta", beta, f32_bytes(num_v_heads * seq_pad)?),
            ("g_cum", g_cum, f32_bytes(num_v_heads * seq_pad)?),
            ("vt", vt, f32_bytes(num_v_heads * chunks * chunk_size * head_v_dim)?),
            ("kcd", kcd, f32_bytes(num_v_heads * chunks * chunk_size * head_k_dim)?),
        ],
    )?;
    unsafe {
        check_cuda(ffi::apxinf_static_gdn_chunk_gemm_f32(
            a.ptr(),
            v.ptr(),
            k.ptr(),
            beta.ptr(),
            g_cum.ptr(),
            vt.ptr(),
            kcd.ptr(),
            seq_pad as i32,
            num_v_heads as i32,
            head_k_dim as i32,
            head_v_dim as i32,
            chunk_size as i32,
            ctx.stream().handle(),
        ))
    }
}

/// Sequential chunk recurrence with state read-in/write-back; emits the
/// token-major BF16 layer output `[seq, num_v_heads * head_v_dim]`.
pub fn gdn_chunk_state(
    ctx: &CudaContext,
    q: &CudaBuffer,
    k: &CudaBuffer,
    g_cum: &CudaBuffer,
    t: &CudaBuffer,
    vt: &CudaBuffer,
    kcd: &CudaBuffer,
    state: &CudaBuffer,
    out: &Tensor,
    seq_pad: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_size: usize,
) -> Result<()> {
    let chunks = seq_pad.checked_div(chunk_size).unwrap_or(0);
    let (seq, out_width) = matrix_shape(out, "GDN chunk state")?;
    if seq == 0
        || chunks == 0
        || seq_pad % chunk_size != 0
        || seq > seq_pad
        || out_width != num_v_heads * head_v_dim
        || chunk_size * head_v_dim / 256 > 32
        || num_v_heads == 0
    {
        return Err(Error::Other("GDN chunk state shape mismatch".into()));
    }
    expect_bf16(out, "GDN chunk state")?;
    require_buffers(
        ctx,
        "GDN chunk state",
        &[
            ("q", q, f32_bytes(num_v_heads * seq_pad * head_k_dim)?),
            ("k", k, f32_bytes(num_v_heads * seq_pad * head_k_dim)?),
            ("g_cum", g_cum, f32_bytes(num_v_heads * seq_pad)?),
            ("t", t, f32_bytes(num_v_heads * chunks * chunk_size * chunk_size)?),
            ("vt", vt, f32_bytes(num_v_heads * chunks * chunk_size * head_v_dim)?),
            ("kcd", kcd, f32_bytes(num_v_heads * chunks * chunk_size * head_k_dim)?),
            ("state", state, f32_bytes(num_v_heads * head_k_dim * head_v_dim)?),
        ],
    )?;
    unsafe {
        check_cuda(ffi::apxinf_static_gdn_chunk_state_f32(
            q.ptr(),
            k.ptr(),
            g_cum.ptr(),
            t.ptr(),
            vt.ptr(),
            kcd.ptr(),
            state.ptr(),
            gpu_ptr(out)?,
            seq as i32,
            seq_pad as i32,
            num_v_heads as i32,
            head_k_dim as i32,
            head_v_dim as i32,
            chunk_size as i32,
            chunks as i32,
            out_width as i32,
            ctx.stream().handle(),
        ))
    }
}

/// Single-token recurrent update. q/k/v are head-major fp32
/// `[num_v_heads, head_dim]` (seq_pad == 1 layout); state read-in/write-back;
/// out is `[1, num_v_heads * head_v_dim]` BF16.
pub fn gdn_recurrent(
    ctx: &CudaContext,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    beta: &CudaBuffer,
    g: &CudaBuffer,
    state: &CudaBuffer,
    out: &Tensor,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Result<()> {
    let (rows, out_width) = matrix_shape(out, "GDN recurrent")?;
    if rows != 1 || out_width != num_v_heads * head_v_dim || num_v_heads == 0 {
        return Err(Error::Other("GDN recurrent shape mismatch".into()));
    }
    expect_bf16(out, "GDN recurrent")?;
    require_buffers(
        ctx,
        "GDN recurrent",
        &[
            ("q", q, f32_bytes(num_v_heads * head_k_dim)?),
            ("k", k, f32_bytes(num_v_heads * head_k_dim)?),
            ("v", v, f32_bytes(num_v_heads * head_v_dim)?),
            ("beta", beta, f32_bytes(num_v_heads)?),
            ("g", g, f32_bytes(num_v_heads)?),
            ("state", state, f32_bytes(num_v_heads * head_k_dim * head_v_dim)?),
        ],
    )?;
    unsafe {
        check_cuda(ffi::apxinf_static_gdn_recurrent_f32(
            q.ptr(),
            k.ptr(),
            v.ptr(),
            beta.ptr(),
            g.ptr(),
            state.ptr(),
            gpu_ptr(out)?,
            num_v_heads as i32,
            head_k_dim as i32,
            head_v_dim as i32,
            ctx.stream().handle(),
        ))
    }
}

/// Gated RMSNorm: rows of `x` (BF16) normalized (fp32), scaled by `weight`,
/// multiplied by silu(z). `z` lives in column slices of a fused projection:
/// row `r` reads `z[r / z_heads, z_col_offset + (r % z_heads) * cols ..]`.
pub fn gated_rms_silu(
    ctx: &CudaContext,
    x: &Tensor,
    z: &Tensor,
    z_col_offset: usize,
    z_heads: usize,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(x, "gated rms silu")?;
    let (z_rows, z_width) = matrix_shape(z, "gated rms silu")?;
    if rows == 0
        || z_heads == 0
        || rows % z_heads != 0
        || z_rows != rows / z_heads
        || z_col_offset + z_heads * cols > z_width
        || weight.shape().dims() != [cols]
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other("gated rms silu shape mismatch".into()));
    }
    for tensor in [x, z, weight] {
        expect_bf16(tensor, "gated rms silu")?;
    }
    let output = bf16_output(ctx, rows, cols)?;
    let z_ptr = gpu_ptr(z)?;
    unsafe {
        check_cuda(ffi::apxinf_static_gated_rms_silu_bf16(
            gpu_ptr(x)?,
            z_ptr,
            gpu_ptr(weight)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            z_heads as i32,
            z_width as i64,
            z_col_offset as i64,
            eps,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// RMSNorm with zero-init (1 + weight) semantics, fp32 compute, BF16 storage.
pub fn rms_norm_plus1(ctx: &CudaContext, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "rms norm plus1")?;
    if rows == 0 || cols == 0 || weight.shape().dims() != [cols] || !eps.is_finite() || eps <= 0.0 {
        return Err(Error::Other("rms norm plus1 shape mismatch".into()));
    }
    expect_bf16(input, "rms norm plus1")?;
    expect_bf16(weight, "rms norm plus1")?;
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        check_cuda(ffi::apxinf_static_rms_norm_plus1_bf16(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// Partial rotary from precomputed BF16 cos/sin tables `[rows, rotary_dim]`.
/// `x` is `[rows, heads, head_dim]`; leading `rotary_dim` channels rotate,
/// the rest pass through.
pub fn partial_rope_table(
    ctx: &CudaContext,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 3
        || dims[1] != heads
        || dims[2] != head_dim
        || rotary_dim == 0
        || rotary_dim % 2 != 0
        || rotary_dim > head_dim
        || cos.shape().dims() != [dims[0], rotary_dim]
        || sin.shape().dims() != [dims[0], rotary_dim]
    {
        return Err(Error::Other("partial rope table shape mismatch".into()));
    }
    for tensor in [x, cos, sin] {
        expect_bf16(tensor, "partial rope table")?;
    }
    let rows = dims[0];
    let output = output_buffer(ctx, x.size_in_bytes())?;
    unsafe {
        check_cuda(ffi::apxinf_static_partial_rope_table_bf16(
            gpu_ptr(x)?,
            gpu_ptr(cos)?,
            gpu_ptr(sin)?,
            output.ptr(),
            rows as i32,
            heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        x.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// Full-attention (q|gate) input preparation: per-head (1 + w) RMSNorm on q/k,
/// partial rotary from tables, K/V append into caller-owned caches at
/// `cache_offset`. `fused` is the fused qkv projection `[seq, width]`; q_out is
/// `[seq, q_heads, head_dim]`; caches are `[capacity, kv_heads, head_dim]`.
pub fn full_attn_prepare(
    ctx: &CudaContext,
    fused: &Tensor,
    q_norm_w: &Tensor,
    k_norm_w: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    q_out: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    cache_offset: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f32,
) -> Result<()> {
    let (seq, width) = matrix_shape(fused, "full attention prepare")?;
    let expected = q_heads * 2 * head_dim + 2 * kv_heads * head_dim;
    let q_dims = q_out.shape().dims();
    let cache_dims = k_cache.shape().dims();
    if seq == 0
        || width < expected
        || q_dims != [seq, q_heads, head_dim]
        || cache_dims.len() != 3
        || cache_dims[1] != kv_heads
        || cache_dims[2] != head_dim
        || v_cache.shape().dims() != cache_dims
        || cache_offset + seq > cache_dims[0]
        || q_norm_w.shape().dims() != [head_dim]
        || k_norm_w.shape().dims() != [head_dim]
        || cos.shape().dims() != [seq, rotary_dim]
        || sin.shape().dims() != [seq, rotary_dim]
        || rotary_dim == 0
        || rotary_dim % 2 != 0
        || rotary_dim > head_dim
    {
        return Err(Error::Other("full attention prepare shape mismatch".into()));
    }
    for tensor in [fused, q_norm_w, k_norm_w, cos, sin, q_out, k_cache, v_cache] {
        expect_bf16(tensor, "full attention prepare")?;
    }
    unsafe {
        check_cuda(ffi::apxinf_static_full_attn_prepare_bf16(
            gpu_ptr(fused)?,
            gpu_ptr(q_norm_w)?,
            gpu_ptr(k_norm_w)?,
            gpu_ptr(cos)?,
            gpu_ptr(sin)?,
            gpu_ptr(q_out)?,
            gpu_ptr(k_cache)?,
            gpu_ptr(v_cache)?,
            seq as i32,
            cache_offset as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            width as i64,
            (kv_heads * head_dim) as i64,
            eps,
            ctx.stream().handle(),
        ))
    }
}

/// In-place post-attention sigmoid gate: `attn *= sigmoid(gate)` where the
/// gate is read from the (q|gate) fused projection rows.
pub fn sigmoid_gate_mul(ctx: &CudaContext, attn: &Tensor, fused: &Tensor, heads: usize, head_dim: usize) -> Result<()> {
    let (rows, width) = matrix_shape(attn, "sigmoid gate")?;
    let (fused_rows, fused_width) = matrix_shape(fused, "sigmoid gate")?;
    if rows == 0
        || width != heads * head_dim
        || fused_rows != rows
        || fused_width < 2 * heads * head_dim
    {
        return Err(Error::Other("sigmoid gate shape mismatch".into()));
    }
    expect_bf16(attn, "sigmoid gate")?;
    expect_bf16(fused, "sigmoid gate")?;
    unsafe {
        check_cuda(ffi::apxinf_static_sigmoid_gate_mul_bf16(
            gpu_ptr(attn)?,
            gpu_ptr(fused)?,
            rows as i32,
            heads as i32,
            head_dim as i32,
            fused_width as i64,
            ctx.stream().handle(),
        ))
    }
}

/// adaLN normalization: out = bf16(bf16(rms(x)*w) * bf16(1 + scale) + shift).
pub fn adaln_rms_norm(
    ctx: &CudaContext,
    x: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(x, "adaln rms norm")?;
    if rows == 0
        || weight.shape().dims() != [cols]
        || scale.shape().dims() != [cols]
        || shift.shape().dims() != [cols]
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other("adaln rms norm shape mismatch".into()));
    }
    for tensor in [x, weight, scale, shift] {
        expect_bf16(tensor, "adaln rms norm")?;
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        check_cuda(ffi::apxinf_static_adaln_rms_norm_bf16(
            gpu_ptr(x)?,
            gpu_ptr(weight)?,
            gpu_ptr(scale)?,
            gpu_ptr(shift)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// adaLN residual: out = bf16(residual + bf16(proj * bf16(1 + gate))).
pub fn adaln_gate_residual(
    ctx: &CudaContext,
    proj: &Tensor,
    residual: &Tensor,
    gate: &Tensor,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(proj, "adaln gate residual")?;
    if rows == 0
        || residual.shape().dims() != [rows, cols]
        || gate.shape().dims() != [cols]
    {
        return Err(Error::Other("adaln gate residual shape mismatch".into()));
    }
    for tensor in [proj, residual, gate] {
        expect_bf16(tensor, "adaln gate residual")?;
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        check_cuda(ffi::apxinf_static_adaln_gate_residual_bf16(
            gpu_ptr(proj)?,
            gpu_ptr(residual)?,
            gpu_ptr(gate)?,
            output.ptr(),
            (rows * cols) as i64,
            cols as i32,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// Expert group-major fused QKV preparation: split (query, gate, key, value),
/// per-head plain RMSNorm on q/k, partial rotary from tables. Outputs are
/// caller tensors: q/gate `[seq, q_heads, head_dim]`, k/v `[seq, kv_heads, head_dim]`.
pub fn expert_qkv_prepare(
    ctx: &CudaContext,
    fused: &Tensor,
    q_norm_w: &Tensor,
    k_norm_w: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    q_out: &Tensor,
    gate_out: &Tensor,
    k_out: &Tensor,
    v_out: &Tensor,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f32,
) -> Result<()> {
    let (seq, width) = matrix_shape(fused, "expert qkv prepare")?;
    let heads_per_group = q_heads.checked_div(kv_heads).unwrap_or(0);
    let expected = kv_heads * (2 * heads_per_group + 2) * head_dim;
    if seq == 0
        || kv_heads == 0
        || q_heads % kv_heads != 0
        || width < expected
        || q_out.shape().dims() != [seq, q_heads, head_dim]
        || gate_out.shape().dims() != [seq, q_heads, head_dim]
        || k_out.shape().dims() != [seq, kv_heads, head_dim]
        || v_out.shape().dims() != [seq, kv_heads, head_dim]
        || q_norm_w.shape().dims() != [head_dim]
        || k_norm_w.shape().dims() != [head_dim]
        || cos.shape().dims() != [seq, rotary_dim]
        || sin.shape().dims() != [seq, rotary_dim]
        || rotary_dim == 0
        || rotary_dim % 2 != 0
        || rotary_dim > head_dim
    {
        return Err(Error::Other("expert qkv prepare shape mismatch".into()));
    }
    for tensor in [fused, q_norm_w, k_norm_w, cos, sin, q_out, gate_out, k_out, v_out] {
        expect_bf16(tensor, "expert qkv prepare")?;
    }
    unsafe {
        check_cuda(ffi::apxinf_static_expert_qkv_prepare_bf16(
            gpu_ptr(fused)?,
            gpu_ptr(q_norm_w)?,
            gpu_ptr(k_norm_w)?,
            gpu_ptr(cos)?,
            gpu_ptr(sin)?,
            gpu_ptr(q_out)?,
            gpu_ptr(gate_out)?,
            gpu_ptr(k_out)?,
            gpu_ptr(v_out)?,
            seq as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            width as i64,
            eps,
            ctx.stream().handle(),
        ))
    }
}

/// In-place post-attention sigmoid gate from a standalone gate tensor.
pub fn expert_sigmoid_gate_mul(ctx: &CudaContext, attn: &Tensor, gate: &Tensor) -> Result<()> {
    if attn.shape().dims() != gate.shape().dims() || attn.numel() == 0 {
        return Err(Error::Other("expert sigmoid gate shape mismatch".into()));
    }
    expect_bf16(attn, "expert sigmoid gate")?;
    expect_bf16(gate, "expert sigmoid gate")?;
    unsafe {
        check_cuda(ffi::apxinf_static_expert_sigmoid_gate_mul_bf16(
            gpu_ptr(attn)?,
            gpu_ptr(gate)?,
            attn.numel() as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Fourier features: `[rows, point_dim * num_features * 2]` with per-channel
/// cat(sin, cos) layout.
pub fn fourier_features(
    ctx: &CudaContext,
    waypoints: &Tensor,
    freqs: &Tensor,
    point_dim: usize,
    num_features: usize,
) -> Result<Tensor> {
    let (rows, width) = matrix_shape(waypoints, "fourier features")?;
    if rows == 0
        || width != point_dim
        || point_dim == 0
        || num_features == 0
        || freqs.shape().dims() != [num_features]
    {
        return Err(Error::Other("fourier features shape mismatch".into()));
    }
    expect_bf16(waypoints, "fourier features")?;
    expect_bf16(freqs, "fourier features")?;
    let out_width = point_dim * num_features * 2;
    let output = bf16_output(ctx, rows, out_width)?;
    unsafe {
        check_cuda(ffi::apxinf_static_fourier_features_bf16(
            gpu_ptr(waypoints)?,
            gpu_ptr(freqs)?,
            output.ptr(),
            rows as i32,
            point_dim as i32,
            num_features as i32,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, out_width]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// Column-concatenate seven `[rows|1, cols]` sources into `[rows, 7 * cols]`;
/// `broadcast_mask` bit `i` broadcasts source `i` row 0 across rows.
pub fn concat7_cols(ctx: &CudaContext, srcs: [&Tensor; 7], broadcast_mask: u32) -> Result<Tensor> {
    let (first_rows, cols) = matrix_shape(srcs[0], "concat7")?;
    if first_rows == 0 || cols == 0 {
        return Err(Error::Other("concat7 shape mismatch".into()));
    }
    for (index, src) in srcs.iter().enumerate() {
        let (rows, width) = matrix_shape(src, "concat7")?;
        let expect_rows = if (broadcast_mask >> index) & 1 == 1 { 1 } else { first_rows };
        if rows != expect_rows || width != cols {
            return Err(Error::Other("concat7 shape mismatch".into()));
        }
        expect_bf16(src, "concat7")?;
    }
    let output = bf16_output(ctx, first_rows, 7 * cols)?;
    unsafe {
        check_cuda(ffi::apxinf_static_concat7_cols_bf16(
            gpu_ptr(srcs[0])?,
            gpu_ptr(srcs[1])?,
            gpu_ptr(srcs[2])?,
            gpu_ptr(srcs[3])?,
            gpu_ptr(srcs[4])?,
            gpu_ptr(srcs[5])?,
            gpu_ptr(srcs[6])?,
            output.ptr(),
            first_rows as i32,
            cols as i32,
            broadcast_mask as i32,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        Shape::new(vec![first_rows, 7 * cols]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// In-place flow-matching Euler update on an F32 state:
/// `w += (endpoint - w) / remaining * step`.
pub fn flow_update(
    ctx: &CudaContext,
    w: &Tensor,
    endpoint: &Tensor,
    remaining: f32,
    step: f32,
) -> Result<()> {
    if w.dtype() != DType::F32
        || endpoint.dtype() != DType::F32
        || w.shape().dims() != endpoint.shape().dims()
        || w.numel() == 0
        || !remaining.is_finite()
        || remaining <= 0.0
        || !step.is_finite()
    {
        return Err(Error::Other("flow update shape mismatch".into()));
    }
    unsafe {
        check_cuda(ffi::apxinf_static_flow_update_f32(
            gpu_ptr(w)?,
            gpu_ptr(endpoint)?,
            remaining,
            step,
            w.numel() as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Set the listed columns of one logits row to -inf (EOS suppression).
pub fn suppress_logits(ctx: &CudaContext, logits: &Tensor, row: usize, ids: &CudaBuffer) -> Result<()> {
    let (rows, cols) = matrix_shape(logits, "logits suppression")?;
    if row >= rows {
        return Err(Error::Other("logits suppression row out of range".into()));
    }
    expect_bf16(logits, "logits suppression")?;
    let count = ids.len() / std::mem::size_of::<u32>();
    if count == 0 {
        return Ok(());
    }
    let row_ptr = gpu_ptr(logits)?.cast::<u8>().wrapping_add(row * cols * DType::BF16.size_in_bytes()).cast();
    unsafe {
        check_cuda(ffi::apxinf_static_suppress_logits_bf16(
            row_ptr,
            ids.ptr().cast(),
            count as i32,
            ctx.stream().handle(),
        ))
    }
}

/// Exact (erf) GELU, BF16 storage.
pub fn gelu_exact(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    expect_bf16(input, "gelu exact")?;
    let output = output_buffer(ctx, input.size_in_bytes())?;
    unsafe {
        check_cuda(ffi::apxinf_static_gelu_exact_bf16(
            gpu_ptr(input)?,
            output.ptr(),
            input.numel() as i64,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// Merge `factor` consecutive rows into one row:
/// `[rows, cols] -> [rows / factor, cols * factor]`.
pub fn merge_rows(ctx: &CudaContext, input: &Tensor, factor: usize) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "merge rows")?;
    if rows == 0 || factor == 0 || rows % factor != 0 {
        return Err(Error::Other("merge rows shape mismatch".into()));
    }
    expect_bf16(input, "merge rows")?;
    let out_rows = rows / factor;
    let output = bf16_output(ctx, out_rows, cols * factor)?;
    unsafe {
        check_cuda(ffi::apxinf_static_merge_rows_bf16(
            gpu_ptr(input)?,
            output.ptr(),
            (out_rows * cols * factor) as i64,
            cols as i32,
            factor as i32,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        Shape::new(vec![out_rows, cols * factor]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// Non-causal joint GQA attention over caller-owned K/V (BF16, FA2 backend).
///
/// `q` is `[q_tokens, q_heads, head_dim]`; `k`/`v` are `[key_tokens, kv_heads,
/// head_dim]` with `q_heads % kv_heads == 0`. Returns `[q_tokens, q_heads,
/// head_dim]`. Hybrid diffusion experts use this primitive to read an external
/// cache concatenated with their own tokens; it shares the FA2 provider path
/// with `attention::causal_gqa_bf16` but applies no causal mask.
pub fn gqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if [q, k, v].into_iter().any(|tensor| tensor.dtype() != DType::BF16)
        || q_shape.len() != 3
        || k_shape.len() != 3
        || v.shape() != k.shape()
        || k_shape[0] < key_tokens
        || k_shape[2] != q_shape[2]
        || k_shape[1] == 0
        || q_shape[1] % k_shape[1] != 0
        || key_tokens == 0
    {
        return Err(Error::Other(
            "non-causal joint BF16 GQA shape mismatch".into(),
        ));
    }
    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    {
        let output = output_buffer(ctx, q.size_in_bytes())?;
        let lse_elements = q_shape[0]
            .checked_mul(q_shape[1])
            .ok_or_else(|| Error::Other("joint GQA LSE size overflow".into()))?;
        let softmax_lse = output_buffer(
            ctx,
            lse_elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| Error::Other("joint GQA LSE byte overflow".into()))?,
        )?;
        unsafe {
            check_cuda(ffi::apxinf_static_fa2_bf16(
                gpu_ptr(q)?,
                gpu_ptr(k)?,
                gpu_ptr(v)?,
                output.ptr(),
                softmax_lse.ptr(),
                1,
                q_shape[0] as i32,
                key_tokens as i32,
                q_shape[1] as i32,
                k_shape[1] as i32,
                q_shape[2] as i32,
                (q_shape[2] as f32).sqrt().recip(),
                ctx.stream().handle(),
            ))?;
        }
        return Ok(make_gpu_tensor(
            q.shape().clone(),
            DType::BF16,
            ctx.device_id(),
            output,
        ));
    }
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    Err(Error::Other(
        "non-causal joint BF16 GQA requires the FA2 backend".into(),
    ))
}
