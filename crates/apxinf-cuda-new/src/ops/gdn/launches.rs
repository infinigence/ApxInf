#[cfg(apxinf_segmented_attention_sm110)]
use apxinf_core::Shape;
use apxinf_core::{DType, Error, Result, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi::abi::gdn as gdn_abi;
use crate::ops::validation::{
    check_cuda, expect_bf16, gpu_ptr, matrix_shape, optional_ptr, require_buffers,
};
#[cfg(apxinf_segmented_attention_sm110)]
use crate::workspace::output_buffer;

/// Fused first-prefill causal convolution and gated delta-rule scan.
/// Returns no implementation for unsupported geometry or device, before any
/// allocation. Stateful continuation uses the recurrent operators instead.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_causal_conv_gdn_prefill_bf16(
    ctx: &CudaContext,
    zba: &Tensor,
    conv_weight: &Tensor,
    new_conv_state: &Tensor,
    a_log: &Tensor,
    rounded_dt_bias: Option<&Tensor>,
    final_state: &CudaBuffer,
    geometry: [usize; 4],
) -> Result<Option<Tensor>> {
    if !cfg!(apxinf_segmented_attention_sm110)
        || ctx.caps().sm != 110
        || ctx.caps().multiprocessor_count != 20
        || geometry != [16, 32, 128, 128]
        || zba.shape().dims() != [3387, 12352]
        || conv_weight.shape().dims() != [8192, 4]
        || rounded_dt_bias.is_none()
    {
        return Ok(None);
    }
    #[cfg(apxinf_segmented_attention_sm110)]
    {
        let dt_bias = rounded_dt_bias.expect("candidate support checked above");
        if new_conv_state.shape().dims() != [8192, 4]
            || a_log.shape().dims() != [32]
            || a_log.dtype() != DType::F32
            || dt_bias.shape().dims() != [32]
        {
            return Err(Error::Other(
                "fused GDN prefill state/weight contract mismatch".into(),
            ));
        }
        for tensor in [zba, conv_weight, new_conv_state, dt_bias] {
            expect_bf16(tensor, "fused GDN prefill")?;
        }
        let z = CudaBuffer::from_tensor(zba).map_err(Error::Cuda)?;
        let weight = CudaBuffer::from_tensor(conv_weight).map_err(Error::Cuda)?;
        let conv_state = CudaBuffer::from_tensor(new_conv_state).map_err(Error::Cuda)?;
        let log = CudaBuffer::from_tensor(a_log).map_err(Error::Cuda)?;
        let bias = CudaBuffer::from_tensor(dt_bias).map_err(Error::Cuda)?;
        require_buffers(
            ctx,
            "fused GDN prefill",
            &[
                ("zba", &z, 3387 * 12352 * 2),
                ("conv_weight", &weight, 8192 * 4 * 2),
                ("conv_state", &conv_state, 8192 * 4 * 2),
                ("a_log", &log, 32 * 4),
                ("dt_bias", &bias, 32 * 2),
                ("final_state", final_state, 32 * 128 * 128 * 4),
            ],
        )?;
        if crate::workspace::may_prepare_native_resources() {
            let status = unsafe { gdn_abi::apxinf_gdn_bf16_prefill_sm110_init() };
            if status != 0 {
                return Err(Error::Other(format!(
                    "native GDN prefill prepare failed: {status}"
                )));
            }
        }
        const PAD: usize = 3392;
        let scratch = output_buffer(ctx, 2 * PAD * 32 * 4 + 10240 + 256)?;
        if scratch.ptr() as usize & 255 != 0 {
            return Err(Error::Other(
                "fused GDN scratch requires 256-byte alignment".into(),
            ));
        }
        let q = output_buffer(ctx, PAD * 16 * 128 * 2)?;
        let k = output_buffer(ctx, PAD * 16 * 128 * 2)?;
        let v = output_buffer(ctx, PAD * 32 * 128 * 2)?;
        let a = output_buffer(ctx, PAD * 32 * 2)?;
        let b = output_buffer(ctx, PAD * 32 * 2)?;
        let state = output_buffer(ctx, 32 * 128 * 128 * 4)?;
        let output = output_buffer(ctx, PAD * 32 * 128 * 2)?;
        let offsets = output_buffer(ctx, 2 * 4)?;
        let status = unsafe {
            gdn_abi::apxinf_gdn_bf16_prefill_sm110(
                z.ptr(),
                weight.ptr(),
                conv_state.ptr(),
                log.ptr().cast::<f32>(),
                bias.ptr(),
                scratch.ptr(),
                q.ptr(),
                k.ptr(),
                v.ptr(),
                a.ptr(),
                b.ptr(),
                state.ptr().cast::<f32>(),
                final_state.ptr().cast::<f32>(),
                output.ptr(),
                offsets.ptr().cast::<i32>(),
                3387,
                PAD as i32,
                ctx.stream().handle(),
            )
        };
        if status != 0 {
            return Err(Error::Other(format!(
                "native GDN prefill enqueue failed: {status}"
            )));
        }
        return output
            .view(0, 3387 * 4096 * 2)
            .map_err(Error::Cuda)?
            .as_tensor(Shape::new(vec![3387, 4096]), DType::BF16)
            .map(Some)
            .map_err(Error::Cuda);
    }
    #[cfg(not(apxinf_segmented_attention_sm110))]
    let _ = (new_conv_state, a_log, rounded_dt_bias, final_state);
    #[cfg(not(apxinf_segmented_attention_sm110))]
    Ok(None)
}

fn f32_bytes(elements: usize) -> Result<usize> {
    elements
        .checked_mul(DType::F32.size_in_bytes())
        .ok_or_else(|| Error::Other("linear-attention buffer size overflow".into()))
}

/// The GDN value buffer carries the convolution output unchanged, so it is held
/// at BF16 width rather than widened on the way out of the preparation kernel.
fn bf16_bytes(elements: usize) -> Result<usize> {
    elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("linear-attention buffer size overflow".into()))
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value)
            .ok_or_else(|| Error::Other("linear-attention buffer size overflow".into()))
    })
}

fn ffi_i32(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Other("linear-attention dimension exceeds i32".into()))
}

/// Cast an F32 tensor to BF16 (round-to-nearest-even).
pub(super) fn causal_conv1d_silu_bf16(
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
        check_cuda(gdn_abi::apxinf_gdn_causal_conv1d_silu_bf16(
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

/// Whether a GDN prefill pipeline can use physically BF16 Q/K throughout.
/// The caller chooses once before allocation; typed launches reject violations.
pub(super) fn gdn_qk_bf16_prefill_supported(
    ctx: &CudaContext,
    recurrent: bool,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_size: usize,
) -> bool {
    unsafe {
        gdn_abi::apxinf_gdn_qk_bf16_supported(
            ctx.device_id() as i32,
            i32::from(recurrent),
            head_k_dim as i32,
            head_v_dim as i32,
            chunk_size as i32,
        ) != 0
    }
}

/// Gated-delta-rule q/k preparation: L2-normalize q/k rows of the post-conv
/// stream and scatter key heads to the value-head layout. Prefill materializes
/// BF16-normalized values; decode retains FP32 and applies the query scale.
/// Outputs are caller-owned fp32 buffers
/// `[num_v_heads, seq_pad, head_k_dim]` (zero-filled tail by the caller).
pub(super) fn gdn_qk_prep(
    ctx: &CudaContext,
    conv_out: &Tensor,
    q: &CudaBuffer,
    k: &CudaBuffer,
    seq_pad: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    key_dim: usize,
    recurrent: bool,
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
    let scale = (1.0f64 / (head_k_dim as f64).sqrt()) as f32;
    unsafe {
        check_cuda(gdn_abi::apxinf_gdn_qk_prep_bf16(
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
            i32::from(recurrent),
            ctx.stream().handle(),
        ))
    }
}

/// Physical BF16 Q/K producer for a selected prefill route.
pub(super) fn gdn_qk_prep_qk_bf16(
    ctx: &CudaContext,
    conv_out: &Tensor,
    q: &CudaBuffer,
    k: &CudaBuffer,
    seq_pad: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    key_dim: usize,
    recurrent: bool,
    eps: f32,
) -> Result<()> {
    if recurrent {
        return Err(Error::Other("BF16 Q/K prep is prefill only".into()));
    }
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
            ("q", q, bf16_bytes(num_v_heads * seq_pad * head_k_dim)?),
            ("k", k, bf16_bytes(num_v_heads * seq_pad * head_k_dim)?),
        ],
    )?;
    let scale = (1.0f64 / (head_k_dim as f64).sqrt()) as f32;
    unsafe {
        check_cuda(gdn_abi::apxinf_gdn_qk_prep_qk_bf16(
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
            i32::from(recurrent),
            ctx.stream().handle(),
        ))
    }
}

/// GDN value/beta/decay preparation. `b_col`/`a_col` are the column offsets of
/// the beta and decay projections inside the fused `zba` output; `v_offset` is
/// the value channel offset inside `conv_out`. Outputs: head-major
/// `v [num_v_heads, seq_pad, head_v_dim]` at BF16 width, and fp32
/// `beta/g [num_v_heads, seq_pad]`.
pub(super) fn gdn_vb_prep(
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
            ("v", v, bf16_bytes(num_v_heads * seq_pad * head_v_dim)?),
            ("beta", beta, f32_bytes(num_v_heads * seq_pad)?),
            ("g", g, f32_bytes(num_v_heads * seq_pad)?),
        ],
    )?;
    let element = DType::BF16.size_in_bytes();
    let b_ptr = gpu_ptr(zba)?
        .cast::<u8>()
        .wrapping_add(b_col * element)
        .cast();
    let a_ptr = gpu_ptr(zba)?
        .cast::<u8>()
        .wrapping_add(a_col * element)
        .cast();
    unsafe {
        check_cuda(gdn_abi::apxinf_gdn_vb_prep_bf16(
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

#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_gate_prep(
    ctx: &CudaContext,
    zba: &Tensor,
    b_col: usize,
    a_col: usize,
    dt_bias: &Tensor,
    a_log: &Tensor,
    beta: &CudaBuffer,
    g: &CudaBuffer,
    seq_pad: usize,
    num_v_heads: usize,
) -> Result<()> {
    let (seq, zba_width) = matrix_shape(zba, "GDN gate prep")?;
    let a_end = a_col
        .checked_add(num_v_heads)
        .ok_or_else(|| Error::Other("GDN gate prep column overflow".into()))?;
    if seq == 0
        || seq > seq_pad
        || b_col >= a_col
        || a_end > zba_width
        || num_v_heads == 0
        || dt_bias.shape().dims() != [num_v_heads]
        || a_log.shape().dims() != [num_v_heads]
        || dt_bias.dtype() != DType::F32
        || a_log.dtype() != DType::F32
    {
        return Err(Error::Other("GDN gate prep shape mismatch".into()));
    }
    expect_bf16(zba, "GDN gate prep")?;
    let zba_buffer = CudaBuffer::from_tensor(zba).map_err(Error::Cuda)?;
    let dt_bias_buffer = CudaBuffer::from_tensor(dt_bias).map_err(Error::Cuda)?;
    let a_log_buffer = CudaBuffer::from_tensor(a_log).map_err(Error::Cuda)?;
    let gate_elements = checked_product(&[num_v_heads, seq_pad])?;
    require_buffers(
        ctx,
        "GDN gate prep",
        &[
            (
                "zba",
                &zba_buffer,
                bf16_bytes(checked_product(&[seq, zba_width])?)?,
            ),
            ("dt_bias", &dt_bias_buffer, f32_bytes(num_v_heads)?),
            ("a_log", &a_log_buffer, f32_bytes(num_v_heads)?),
            ("beta", beta, f32_bytes(gate_elements)?),
            ("g", g, f32_bytes(gate_elements)?),
        ],
    )?;
    let element = DType::BF16.size_in_bytes();
    let b_offset = b_col
        .checked_mul(element)
        .ok_or_else(|| Error::Other("GDN gate prep byte offset overflow".into()))?;
    let a_offset = a_col
        .checked_mul(element)
        .ok_or_else(|| Error::Other("GDN gate prep byte offset overflow".into()))?;
    let b_ptr = gpu_ptr(zba)?.cast::<u8>().wrapping_add(b_offset).cast();
    let a_ptr = gpu_ptr(zba)?.cast::<u8>().wrapping_add(a_offset).cast();
    unsafe {
        check_cuda(gdn_abi::apxinf_gdn_gate_prep_bf16(
            b_ptr,
            a_ptr,
            gpu_ptr(dt_bias)?,
            gpu_ptr(a_log)?,
            beta.ptr(),
            g.ptr(),
            ffi_i32(seq)?,
            ffi_i32(seq_pad)?,
            ffi_i32(num_v_heads)?,
            ffi_i32(zba_width)?,
            ctx.stream().handle(),
        ))
    }
}

/// Chunk-local inclusive cumsum of the decay `g`.
pub(super) fn gdn_cumsum(
    ctx: &CudaContext,
    g: &CudaBuffer,
    g_cum: &CudaBuffer,
    seq_pad: usize,
    num_v_heads: usize,
    chunk_size: usize,
) -> Result<()> {
    if seq_pad == 0 || chunk_size == 0 || seq_pad % chunk_size != 0 || num_v_heads == 0 {
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
        check_cuda(gdn_abi::apxinf_gdn_cumsum_f32(
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
pub(super) fn gdn_attn_raw(
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
            ("t", t, bf16_bytes(num_v_heads * chunks * matrix)?),
        ],
    )?;
    unsafe {
        check_cuda(gdn_abi::apxinf_gdn_attn_raw_f32(
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

/// Fixed-shape F1 for physically BF16 Q/K; never falls back to FP32.
pub(super) fn gdn_attn_raw_solve_f1_qk_bf16(
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
    if !gdn_qk_bf16_prefill_supported(ctx, false, head_k_dim, head_k_dim, chunk_size) {
        return Err(Error::Other("BF16 Q/K F1 route mismatch".into()));
    }
    if seq_pad == 0 || seq_pad % chunk_size != 0 || num_v_heads == 0 {
        return Err(Error::Other("GDN raw+solve F1 shape mismatch".into()));
    }
    let chunks = seq_pad / chunk_size;
    require_buffers(
        ctx,
        "GDN raw+solve F1",
        &[
            ("q", q, bf16_bytes(num_v_heads * seq_pad * head_k_dim)?),
            ("k", k, bf16_bytes(num_v_heads * seq_pad * head_k_dim)?),
            ("beta", beta, f32_bytes(num_v_heads * seq_pad)?),
            ("g_cum", g_cum, f32_bytes(num_v_heads * seq_pad)?),
            (
                "a",
                a,
                f32_bytes(num_v_heads * chunks * chunk_size * chunk_size)?,
            ),
            (
                "t",
                t,
                bf16_bytes(num_v_heads * chunks * chunk_size * chunk_size)?,
            ),
        ],
    )?;
    unsafe {
        check_cuda(gdn_abi::apxinf_gdn_attn_raw_solve_f1_qk_bf16(
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
        ))?;
    }
    Ok(())
}

/// In-place forward substitution + identity on each chunk matrix, producing
/// (I - A)^-1. `matrices` is the total chunk-matrix count (heads * chunks).
pub(super) fn gdn_tri_solve(
    ctx: &CudaContext,
    a: &CudaBuffer,
    matrices: usize,
    chunk_size: usize,
) -> Result<()> {
    if matrices == 0 || chunk_size == 0 || chunk_size > 128 {
        return Err(Error::Other("GDN triangular solve shape mismatch".into()));
    }
    require_buffers(
        ctx,
        "GDN triangular solve",
        &[("a", a, f32_bytes(matrices * chunk_size * chunk_size)?)],
    )?;
    unsafe {
        check_cuda(gdn_abi::apxinf_gdn_tri_solve_f32(
            a.ptr(),
            matrices as i32,
            chunk_size as i32,
            ctx.stream().handle(),
        ))
    }
}

/// VT = A @ (v * beta), KCD = A @ (k * beta * exp(g_cum)) per chunk.
pub(super) fn gdn_chunk_gemm(
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
            (
                "a",
                a,
                f32_bytes(num_v_heads * chunks * chunk_size * chunk_size)?,
            ),
            ("v", v, bf16_bytes(num_v_heads * seq_pad * head_v_dim)?),
            ("k", k, f32_bytes(num_v_heads * seq_pad * head_k_dim)?),
            ("beta", beta, f32_bytes(num_v_heads * seq_pad)?),
            ("g_cum", g_cum, f32_bytes(num_v_heads * seq_pad)?),
            (
                "vt",
                vt,
                bf16_bytes(num_v_heads * chunks * chunk_size * head_v_dim)?,
            ),
            (
                "kcd",
                kcd,
                bf16_bytes(num_v_heads * chunks * chunk_size * head_k_dim)?,
            ),
        ],
    )?;
    unsafe {
        check_cuda(gdn_abi::apxinf_gdn_chunk_gemm_f32(
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
            ctx.device_id() as i32,
            ctx.stream().handle(),
        ))
    }
}

/// Typed direct-V W/U: reads V straight from the convolution output instead of
/// a head-major copy.
#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_chunk_gemm_tri_k_bf16_direct_v(
    ctx: &CudaContext,
    a: &CudaBuffer,
    conv_out: &Tensor,
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
    v_offset: usize,
) -> Result<()> {
    if head_k_dim != 128 || head_v_dim != 128 || chunk_size != 64 {
        return Err(Error::Other("BF16 K direct-V W/U route mismatch".into()));
    }
    let (seq, conv_dim) = matrix_shape(conv_out, "GDN direct-V W/U")?;
    expect_bf16(conv_out, "GDN direct-V W/U")?;
    let conv_buffer = CudaBuffer::from_tensor(conv_out).map_err(Error::Cuda)?;
    let v_end = v_offset
        .checked_add(checked_product(&[num_v_heads, head_v_dim])?)
        .ok_or_else(|| Error::Other("GDN direct-V offset overflow".into()))?;
    let chunks = seq_pad.checked_div(chunk_size).unwrap_or(0);
    if seq == 0
        || seq > seq_pad
        || seq_pad % chunk_size != 0
        || chunks == 0
        || num_v_heads == 0
        || v_end > conv_dim
    {
        return Err(Error::Other("GDN direct-V W/U shape mismatch".into()));
    }
    // The value source has exactly `seq` token rows. Padded rows are masked in
    // the CUDA kernel before indexing this tensor; they are never read.
    let gate_elements = checked_product(&[num_v_heads, seq_pad])?;
    let matrix_elements = checked_product(&[num_v_heads, chunks, chunk_size, chunk_size])?;
    let vt_elements = checked_product(&[num_v_heads, chunks, chunk_size, head_v_dim])?;
    let kcd_elements = checked_product(&[num_v_heads, chunks, chunk_size, head_k_dim])?;
    require_buffers(
        ctx,
        "GDN direct-V W/U",
        &[
            (
                "conv_out",
                &conv_buffer,
                bf16_bytes(checked_product(&[seq, conv_dim])?)?,
            ),
            ("a", a, f32_bytes(matrix_elements)?),
            (
                "k",
                k,
                bf16_bytes(checked_product(&[gate_elements, head_k_dim])?)?,
            ),
            ("beta", beta, f32_bytes(gate_elements)?),
            ("g_cum", g_cum, f32_bytes(gate_elements)?),
            ("vt", vt, bf16_bytes(vt_elements)?),
            ("kcd", kcd, bf16_bytes(kcd_elements)?),
        ],
    )?;
    let conv_ptr = gpu_ptr(conv_out)?;
    let seq_i32 = ffi_i32(seq)?;
    let seq_pad_i32 = ffi_i32(seq_pad)?;
    let conv_dim_i32 = ffi_i32(conv_dim)?;
    let v_offset_i32 = ffi_i32(v_offset)?;
    let num_v_heads_i32 = ffi_i32(num_v_heads)?;
    let head_k_dim_i32 = ffi_i32(head_k_dim)?;
    let head_v_dim_i32 = ffi_i32(head_v_dim)?;
    let chunk_size_i32 = ffi_i32(chunk_size)?;
    unsafe {
        check_cuda(gdn_abi::apxinf_gdn_chunk_gemm_tri_k_bf16_direct_v(
            a.ptr(),
            conv_ptr,
            k.ptr(),
            beta.ptr(),
            g_cum.ptr(),
            vt.ptr(),
            kcd.ptr(),
            seq_i32,
            seq_pad_i32,
            conv_dim_i32,
            v_offset_i32,
            num_v_heads_i32,
            head_k_dim_i32,
            head_v_dim_i32,
            chunk_size_i32,
            ctx.stream().handle(),
        ))
    }
}

/// Sequential chunk recurrence with state read-in/write-back; emits the
/// token-major BF16 layer output `[seq, num_v_heads * head_v_dim]`.
pub(super) fn gdn_chunk_state(
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
            (
                "t",
                t,
                bf16_bytes(num_v_heads * chunks * chunk_size * chunk_size)?,
            ),
            (
                "vt",
                vt,
                bf16_bytes(num_v_heads * chunks * chunk_size * head_v_dim)?,
            ),
            (
                "kcd",
                kcd,
                bf16_bytes(num_v_heads * chunks * chunk_size * head_k_dim)?,
            ),
            (
                "state",
                state,
                f32_bytes(num_v_heads * head_k_dim * head_v_dim)?,
            ),
        ],
    )?;
    unsafe {
        check_cuda(gdn_abi::apxinf_gdn_chunk_state_f32(
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
            ctx.device_id() as i32,
            ctx.stream().handle(),
        ))
    }
}

/// LOSSY WMMA scan consuming physical BF16 Q/K.
pub(super) fn gdn_chunk_state_qk_bf16(
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
    if head_k_dim != 128 || head_v_dim != 128 || chunk_size != 64 {
        return Err(Error::Other("BF16 Q/K state route mismatch".into()));
    }
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
            ("q", q, bf16_bytes(num_v_heads * seq_pad * head_k_dim)?),
            ("k", k, bf16_bytes(num_v_heads * seq_pad * head_k_dim)?),
            ("g_cum", g_cum, f32_bytes(num_v_heads * seq_pad)?),
            (
                "t",
                t,
                bf16_bytes(num_v_heads * chunks * chunk_size * chunk_size)?,
            ),
            (
                "vt",
                vt,
                bf16_bytes(num_v_heads * chunks * chunk_size * head_v_dim)?,
            ),
            (
                "kcd",
                kcd,
                bf16_bytes(num_v_heads * chunks * chunk_size * head_k_dim)?,
            ),
            (
                "state",
                state,
                f32_bytes(num_v_heads * head_k_dim * head_v_dim)?,
            ),
        ],
    )?;
    unsafe {
        check_cuda(gdn_abi::apxinf_gdn_chunk_state_qk_bf16(
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
pub(super) fn gdn_recurrent(
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
            ("v", v, bf16_bytes(num_v_heads * head_v_dim)?),
            ("beta", beta, f32_bytes(num_v_heads)?),
            ("g", g, f32_bytes(num_v_heads)?),
            (
                "state",
                state,
                f32_bytes(num_v_heads * head_k_dim * head_v_dim)?,
            ),
        ],
    )?;
    unsafe {
        check_cuda(gdn_abi::apxinf_gdn_recurrent_f32(
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
            ctx.device_id() as i32,
            ctx.stream().handle(),
        ))
    }
}
