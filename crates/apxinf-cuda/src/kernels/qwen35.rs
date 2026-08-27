//! Qwen3.5-specific bf16 kernels (C1 eager path).
//!
//! These wrappers operate on `CudaBuffer`s and write into caller-owned
//! output buffers, matching the `gemm::write` convention. All tensors are
//! bf16 except the AWQ packed/scale/zero-point sources and the f32 state.

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use apxinf_core::{Error, Result};

fn need(ctx: &CudaContext, name: &str, dev: usize, buffer: &CudaBuffer, bytes: usize) -> Result<()> {
    if dev != ctx.device_id() {
        return Err(Error::Other(format!(
            "qwen35 `{name}` is on CUDA{dev}, expected CUDA{}",
            ctx.device_id()
        )));
    }
    if buffer.len() < bytes {
        return Err(Error::Other(format!(
            "qwen35 `{name}` needs {bytes} bytes, has {}",
            buffer.len()
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dequant_w4a16_bf16_into(
    ctx: &CudaContext,
    packed: &CudaBuffer,
    scale: &CudaBuffer,
    zp: &CudaBuffer,
    out: &CudaBuffer,
    out_dim: usize,
    in_dim: usize,
) -> Result<()> {
    let groups = in_dim / 32;
    let dev = ctx.device_id();
    need(ctx, "packed", dev, packed, out_dim * (in_dim / 8) * 4)?;
    need(ctx, "scale", dev, scale, out_dim * groups * 2)?;
    need(ctx, "zp", dev, zp, out_dim.div_ceil(8) * groups * 4)?;
    need(ctx, "out", dev, out, in_dim * out_dim * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_dequant_w4a16_bf16(
            packed.ptr(),
            scale.ptr(),
            zp.ptr(),
            out.ptr(),
            out_dim as i32,
            in_dim as i32,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

pub fn mul_bf16_into(ctx: &CudaContext, a: &CudaBuffer, b: &CudaBuffer, out: &CudaBuffer, n: usize) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "a", dev, a, n * 2)?;
    need(ctx, "b", dev, b, n * 2)?;
    need(ctx, "out", dev, out, n * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_mul_bf16(a.ptr(), b.ptr(), out.ptr(), n as i64, ctx.stream().handle())
    })
    .map_err(Error::Cuda)
}

pub fn silu_bf16_into(ctx: &CudaContext, a: &CudaBuffer, out: &CudaBuffer, n: usize) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "a", dev, a, n * 2)?;
    need(ctx, "out", dev, out, n * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_silu_bf16(a.ptr(), out.ptr(), n as i64, ctx.stream().handle())
    })
    .map_err(Error::Cuda)
}

pub fn sigmoid_mul_bf16_into(ctx: &CudaContext, a: &CudaBuffer, b: &CudaBuffer, out: &CudaBuffer, n: usize) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "a", dev, a, n * 2)?;
    need(ctx, "b", dev, b, n * 2)?;
    need(ctx, "out", dev, out, n * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_sigmoid_mul_bf16(a.ptr(), b.ptr(), out.ptr(), n as i64, ctx.stream().handle())
    })
    .map_err(Error::Cuda)
}

pub fn accum_bf16_into(ctx: &CudaContext, dst: &CudaBuffer, src: &CudaBuffer, n: usize) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "dst", dev, dst, n * 2)?;
    need(ctx, "src", dev, src, n * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_accum_bf16(dst.ptr(), src.ptr(), n as i64, ctx.stream().handle())
    })
    .map_err(Error::Cuda)
}

pub fn rms_norm_bf16_into(
    ctx: &CudaContext,
    x: &CudaBuffer,
    w: &CudaBuffer,
    out: &CudaBuffer,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "x", dev, x, rows * cols * 2)?;
    need(ctx, "w", dev, w, cols * 2)?;
    need(ctx, "out", dev, out, rows * cols * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_rms_norm_bf16(
            x.ptr(),
            w.ptr(),
            out.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

pub fn qg_split_bf16_into(
    ctx: &CudaContext,
    qg: &CudaBuffer,
    q: &CudaBuffer,
    gate: &CudaBuffer,
    total: usize,
    heads: usize,
    hd: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "qg", dev, qg, total * 2)?;
    need(ctx, "q", dev, q, total * 2)?;
    need(ctx, "gate", dev, gate, total * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_qg_split_bf16(
            qg.ptr(),
            q.ptr(),
            gate.ptr(),
            total as i64,
            heads as i32,
            hd as i32,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

pub fn partial_rope_bf16_inplace(
    ctx: &CudaContext,
    x: &CudaBuffer,
    cos: &CudaBuffer,
    sin: &CudaBuffer,
    rows: usize,
    heads: usize,
    hd: usize,
    half: usize,
    max_len: usize,
    pos0: &CudaBuffer,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "x", dev, x, rows * hd * 2)?;
    need(ctx, "cos", dev, cos, max_len * half * 2)?;
    need(ctx, "sin", dev, sin, max_len * half * 2)?;
    need(ctx, "pos0", dev, pos0, 4)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_partial_rope_bf16(
            x.ptr(),
            cos.ptr(),
            sin.ptr(),
            (rows * half) as i64,
            heads as i32,
            hd as i32,
            half as i32,
            pos0.ptr(),
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

pub fn conv_silu_bf16_into(
    ctx: &CudaContext,
    x: &CudaBuffer,
    w: &CudaBuffer,
    out: &CudaBuffer,
    l: usize,
    conv_dim: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "x", dev, x, l * conv_dim * 2)?;
    need(ctx, "w", dev, w, conv_dim * 4 * 2)?;
    need(ctx, "out", dev, out, l * conv_dim * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_conv_silu_bf16(x.ptr(), w.ptr(), out.ptr(), l as i32, conv_dim as i32, ctx.stream().handle())
    })
    .map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
pub fn delta_recurrence_bf16_into(
    ctx: &CudaContext,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    beta: &CudaBuffer,
    g: &CudaBuffer,
    state: &CudaBuffer,
    out: &CudaBuffer,
    l: usize,
    nv: usize,
    kd: usize,
    vd: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "q", dev, q, l * nv * kd * 2)?;
    need(ctx, "k", dev, k, l * nv * kd * 2)?;
    need(ctx, "v", dev, v, l * nv * vd * 2)?;
    need(ctx, "beta", dev, beta, l * nv * 2)?;
    need(ctx, "g", dev, g, l * nv * 2)?;
    need(ctx, "state", dev, state, nv * kd * vd * 4)?;
    need(ctx, "out", dev, out, l * nv * vd * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_delta_recurrence_bf16(
            q.ptr(),
            k.ptr(),
            v.ptr(),
            beta.ptr(),
            g.ptr(),
            state.ptr(),
            out.ptr(),
            l as i32,
            nv as i32,
            kd as i32,
            vd as i32,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
pub fn attention_bf16_into(
    ctx: &CudaContext,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    gate: &CudaBuffer,
    out: &CudaBuffer,
    l: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "q", dev, q, l * heads * hd * 2)?;
    need(ctx, "k", dev, k, l * kv_heads * hd * 2)?;
    need(ctx, "v", dev, v, l * kv_heads * hd * 2)?;
    need(ctx, "gate", dev, gate, l * heads * hd * 2)?;
    need(ctx, "out", dev, out, l * heads * hd * 2)?;
    // TEST_FENG: FA2 causal prefill for long sequences (naive kernel
    // stays for L<=128 where its O(L^2) cost is negligible).
    #[cfg(apxinf_fa2_sm80)]
    if l > 128 {
        return attention_prefill_fa2(ctx, q, k, v, gate, out, l, heads, kv_heads, hd);
    }
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_attention_bf16(
            q.ptr(),
            k.ptr(),
            v.ptr(),
            gate.ptr(),
            out.ptr(),
            l as i32,
            heads as i32,
            kv_heads as i32,
            hd as i32,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

#[cfg(apxinf_fa2_sm80)]
#[allow(clippy::too_many_arguments)]
fn attention_prefill_fa2(
    ctx: &CudaContext,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    gate: &CudaBuffer,
    out: &CudaBuffer,
    l: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    // Scratch for the FA2 log-sum-exp output; the vendored adapter
    // requires a valid pointer even though we discard the values.
    let lse = CudaBuffer::alloc(l * heads * std::mem::size_of::<f32>(), dev)
        .map_err(Error::Other)?;
    let softmax_scale = (hd as f32).sqrt().recip();
    ffi::check_cuda(unsafe {
        ffi::apxinf_static_fa2_f16(
            q.ptr(),
            k.ptr(),
            v.ptr(),
            out.ptr(),
            lse.ptr(),
            1i32,
            l as i32,
            l as i32,
            heads as i32,
            kv_heads as i32,
            hd as i32,
            softmax_scale,
            1i32, // causal prefill
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)?;
    // Qwen3.5 gated-attention epilogue: out = sigmoid(gate) * attn.
    sigmoid_mul_bf16_into(ctx, gate, out, out, l * heads * hd)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_split_bf16_into(
    ctx: &CudaContext,
    conv: &CudaBuffer,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    l: usize,
    nk: usize,
    nv: usize,
    kd: usize,
    vd: usize,
    conv_dim: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "conv", dev, conv, l * conv_dim * 2)?;
    need(ctx, "q", dev, q, l * nv * kd * 2)?;
    need(ctx, "k", dev, k, l * nv * kd * 2)?;
    need(ctx, "v", dev, v, l * nv * vd * 2)?;
    let dims = [l as i32, nk as i32, nv as i32, kd as i32, vd as i32, conv_dim as i32];
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_conv_split_bf16(
            conv.ptr(),
            q.ptr(),
            k.ptr(),
            v.ptr(),
            dims[0], dims[1], dims[2], dims[3], dims[4], dims[5],
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

pub fn l2norm_bf16_into(
    ctx: &CudaContext,
    x: &CudaBuffer,
    out: &CudaBuffer,
    rows: usize,
    cols: usize,
    eps: f32,
    scale: f32,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "x", dev, x, rows * cols * 2)?;
    need(ctx, "out", dev, out, rows * cols * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_l2norm_bf16(
            x.ptr(),
            out.ptr(),
            rows as i32,
            cols as i32,
            eps,
            scale,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
pub fn gemm_w4a16_bf16(
    ctx: &CudaContext,
    a: &CudaBuffer,
    packed: &CudaBuffer,
    scale: &CudaBuffer,
    zp: &CudaBuffer,
    c: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "a", dev, a, m * k * 2)?;
    need(ctx, "packed", dev, packed, n * k.div_ceil(8) * 4)?;
    need(ctx, "scale", dev, scale, n * (k / 32) * 2)?;
    need(ctx, "zp", dev, zp, n.div_ceil(8) * (k / 32) * 4)?;
    need(ctx, "c", dev, c, m * n * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_gemm_w4a16_bf16(
            a.ptr(),
            packed.ptr(),
            scale.ptr(),
            zp.ptr(),
            c.ptr(),
            m as i32,
            n as i32,
            k as i32,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
pub fn beta_g_bf16(
    ctx: &CudaContext,
    a: &CudaBuffer,
    b: &CudaBuffer,
    a_log: &CudaBuffer,
    dt_bias: &CudaBuffer,
    beta: &CudaBuffer,
    g: &CudaBuffer,
    total: usize,
    nv: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "a", dev, a, total * 2)?;
    need(ctx, "b", dev, b, total * 2)?;
    need(ctx, "a_log", dev, a_log, nv * 4)?;
    need(ctx, "dt_bias", dev, dt_bias, nv * 4)?;
    need(ctx, "beta", dev, beta, total * 2)?;
    need(ctx, "g", dev, g, total * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_beta_g_bf16(
            a.ptr(),
            b.ptr(),
            a_log.ptr(),
            dt_bias.ptr(),
            beta.ptr(),
            g.ptr(),
            total as i32,
            nv as i32,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}


pub fn copy_bf16(ctx: &CudaContext, src: &CudaBuffer, dst: &CudaBuffer, n: usize) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "src", dev, src, n * 2)?;
    need(ctx, "dst", dev, dst, n * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_copy_bf16(src.ptr(), dst.ptr(), n as i64, ctx.stream().handle())
    })
    .map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode_bf16_into(
    ctx: &CudaContext,
    q: &CudaBuffer,
    kcache: &CudaBuffer,
    vcache: &CudaBuffer,
    gate: &CudaBuffer,
    out: &CudaBuffer,
    seq: &CudaBuffer,
    heads: usize,
    kv_heads: usize,
    hd: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "q", dev, q, heads * hd * 2)?;
    need(ctx, "kcache", dev, kcache, 1)?;
    need(ctx, "vcache", dev, vcache, 1)?;
    need(ctx, "gate", dev, gate, heads * hd * 2)?;
    need(ctx, "out", dev, out, heads * hd * 2)?;
    need(ctx, "seq", dev, seq, 4)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_attention_decode_bf16(
            q.ptr(),
            kcache.ptr(),
            vcache.ptr(),
            gate.ptr(),
            out.ptr(),
            seq.ptr(),
            heads as i32,
            kv_heads as i32,
            hd as i32,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

pub fn conv_step_silu_bf16_into(
    ctx: &CudaContext,
    cur: &CudaBuffer,
    hist: &CudaBuffer,
    w: &CudaBuffer,
    out: &CudaBuffer,
    conv_dim: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "cur", dev, cur, conv_dim * 2)?;
    need(ctx, "hist", dev, hist, 3 * conv_dim * 2)?;
    need(ctx, "w", dev, w, conv_dim * 4 * 2)?;
    need(ctx, "out", dev, out, conv_dim * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_conv_step_silu_bf16(
            cur.ptr(),
            hist.ptr(),
            w.ptr(),
            out.ptr(),
            conv_dim as i32,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
pub fn delta_step_bf16_into(
    ctx: &CudaContext,
    q: &CudaBuffer,
    k: &CudaBuffer,
    v: &CudaBuffer,
    beta: &CudaBuffer,
    g: &CudaBuffer,
    state: &CudaBuffer,
    out: &CudaBuffer,
    nv: usize,
    kd: usize,
    vd: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "q", dev, q, nv * kd * 2)?;
    need(ctx, "k", dev, k, nv * kd * 2)?;
    need(ctx, "v", dev, v, nv * vd * 2)?;
    need(ctx, "beta", dev, beta, nv * 2)?;
    need(ctx, "g", dev, g, nv * 2)?;
    need(ctx, "state", dev, state, nv * kd * vd * 4)?;
    need(ctx, "out", dev, out, nv * vd * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_delta_step_bf16(
            q.ptr(),
            k.ptr(),
            v.ptr(),
            beta.ptr(),
            g.ptr(),
            state.ptr(),
            out.ptr(),
            nv as i32,
            kd as i32,
            vd as i32,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

pub fn copy_at_bf16(
    ctx: &CudaContext,
    src: &CudaBuffer,
    dst_base: &CudaBuffer,
    pos: &CudaBuffer,
    stride_elems: usize,
    n: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "src", dev, src, n * 2)?;
    need(ctx, "dst_base", dev, dst_base, 1)?;
    need(ctx, "pos", dev, pos, 4)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_copy_at_bf16(
            src.ptr(),
            dst_base.ptr(),
            pos.ptr(),
            stride_elems as i32,
            n as i32,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

pub fn gemm_f16(
    ctx: &CudaContext,
    a: &CudaBuffer,
    b: &CudaBuffer,
    c: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "a", dev, a, m * k * 2)?;
    need(ctx, "b", dev, b, k * n * 2)?;
    need(ctx, "c", dev, c, m * n * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_gemm_f16(
            a.ptr(),
            b.ptr(),
            c.ptr(),
            m as i32,
            n as i32,
            k as i32,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

pub fn gemm_w4a16_m1_bf16(
    ctx: &CudaContext,
    a: &CudaBuffer,
    packed: &CudaBuffer,
    scale: &CudaBuffer,
    zp: &CudaBuffer,
    c: &CudaBuffer,
    n: usize,
    k: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "a", dev, a, k * 2)?;
    need(ctx, "packed", dev, packed, n * k.div_ceil(8) * 4)?;
    need(ctx, "scale", dev, scale, n * (k / 32) * 2)?;
    need(ctx, "zp", dev, zp, n.div_ceil(8) * (k / 32) * 4)?;
    need(ctx, "c", dev, c, n * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_gemm_w4a16_m1_bf16(
            a.ptr(),
            packed.ptr(),
            scale.ptr(),
            zp.ptr(),
            c.ptr(),
            n as i32,
            k as i32,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}

pub fn embed_gather_f16(
    ctx: &CudaContext,
    table: &CudaBuffer,
    ids: &CudaBuffer,
    out: &CudaBuffer,
    tokens: usize,
    hidden: usize,
) -> Result<()> {
    let dev = ctx.device_id();
    need(ctx, "ids", dev, ids, tokens * 4)?;
    need(ctx, "out", dev, out, tokens * hidden * 2)?;
    ffi::check_cuda(unsafe {
        ffi::apxinf_qwen_embed_gather_f16(
            table.ptr(),
            ids.ptr(),
            out.ptr(),
            tokens as i64,
            hidden as i32,
            ctx.stream().handle(),
        )
    })
    .map_err(Error::Cuda)
}
