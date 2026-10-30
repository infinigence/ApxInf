//! Causal joint transformer block. Only observed video K/V is eligible for
//! publication; generated video and action tokens remain call-local outputs.
use super::super::{
    backend::{allocate, binary, ops, transfers, Context, Policies},
    weights::dit::{Block, Linear, Norm, DIM, FFN, HEADS},
};
use crate::dreamzero::backend;
use apxinf_core::{Error, Result, Shape, Tensor};

pub(super) fn rows(x: &Tensor, start: usize, count: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 2 || start.checked_add(count).is_none_or(|end| end > dims[0]) {
        return Err(Error::Other(
            "DreamZero matrix row view out of bounds".into(),
        ));
    }
    let width = dims[1];
    backend::CudaBuffer::from_tensor(x)
        .map_err(Error::Cuda)?
        .view(
            start * width * x.dtype().size_in_bytes(),
            count * width * x.dtype().size_in_bytes(),
        )
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![count, width]), x.dtype())
        .map_err(Error::Cuda)
}
pub(super) fn projection(
    ctx: &Context,
    p: &Policies,
    x: &Tensor,
    w: &Linear,
    out: &mut Tensor,
) -> Result<()> {
    let mut gemm = ops::GemmArgs::new(x, &w.weight, out);
    gemm.policy = p.gemm.clone();
    // Match the fixed BF16 reference's Lt workspace budget. Its narrow output
    // head uses F32 split-K; a different budget changes reduction grouping.
    gemm.policy.workspace_limit = gemm.policy.workspace_limit.min(1 << 20);
    gemm.weight_layout = ops::GemmWeightLayout::OutputMajor;
    ops::gemm_bias_single_round(
        ctx,
        ops::GemmBiasArgs {
            gemm,
            bias: &w.bias,
        },
    )?;
    Ok(())
}
fn norm(ctx: &Context, x: &Tensor, w: &Norm, out: &mut Tensor) -> Result<()> {
    ops::norm(
        ctx,
        ops::NormArgs::layer(x, &w.weight, &w.bias, out, 1e-6),
    )?;
    Ok(())
}
fn rms(ctx: &Context, x: &Tensor, weight: &Tensor, out: &mut Tensor) -> Result<()> {
    crate::dreamzero::backend::staged_rms(ctx, ops::NormArgs::rms(x, weight, out, 1e-6))?;
    Ok(())
}
fn attend(
    ctx: &Context,
    p: &Policies,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let shape = |x: &Tensor| vec![1, x.shape().dims()[0], HEADS, DIM / HEADS];
    let q = q.reshape(shape(q))?;
    let k = k.reshape(shape(k))?;
    let v = v.reshape(shape(v))?;
    let mut out = out.reshape(shape(out))?;
    let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut out);
    args.policy = p.attention.clone();
    // Keep self, text and image attention on the same linear-memory cold
    // path, including short cross-attention shapes that fit dense scratch.
    args.policy.workspace_limit = 2 << 20;
    ops::attention(ctx, args)?;
    Ok(())
}

pub(in crate::dreamzero) struct BlockWorkspace {
    rows: usize,
    video_rows: usize,
    history_rows: usize,
    pub hidden: Tensor,
    normed: Tensor,
    shifted: Tensor,
    q_raw: Tensor,
    k_raw: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attended: Tensor,
    projected: Tensor,
    residual: Tensor,
    gated: Tensor,
    up: Tensor,
    activated: Tensor,
    kv_k: Tensor,
    kv_v: Tensor,
    text_k_raw: Tensor,
    text_k: Tensor,
    text_v: Tensor,
    image_k_raw: Tensor,
    image_k: Tensor,
    image_v: Tensor,
    image_attended: Tensor,
    modulation: Tensor,
    scale: Tensor,
    unit_norm: Norm,
}
impl BlockWorkspace {
    pub fn new(
        ctx: &Context,
        video_rows: usize,
        action_rows: usize,
        history_rows: usize,
    ) -> Result<Self> {
        if ![880, 1760].contains(&video_rows)
            || ![0, 25].contains(&action_rows)
            || history_rows % 880 != 0
            || history_rows + video_rows > 7920
        {
            return Err(Error::Other("unsupported DreamZero block geometry".into()));
        }
        let n = video_rows + action_rows;
        let h = || allocate(ctx, &[n, DIM]);
        let text = || allocate(ctx, &[512, DIM]);
        let image = || allocate(ctx, &[257, DIM]);
        Ok(Self {
            rows: n,
            video_rows,
            history_rows,
            hidden: h()?,
            normed: h()?,
            shifted: h()?,
            q_raw: h()?,
            k_raw: h()?,
            q: h()?,
            k: h()?,
            v: h()?,
            attended: h()?,
            projected: h()?,
            residual: h()?,
            gated: h()?,
            up: allocate(ctx, &[n, FFN])?,
            activated: allocate(ctx, &[n, FFN])?,
            kv_k: allocate(ctx, &[history_rows + n, DIM])?,
            kv_v: allocate(ctx, &[history_rows + n, DIM])?,
            text_k_raw: text()?,
            text_k: text()?,
            text_v: text()?,
            image_k_raw: image()?,
            image_k: image()?,
            image_v: image()?,
            image_attended: h()?,
            modulation: allocate(ctx, &[12, DIM])?,
            scale: allocate(ctx, &[1, DIM])?,
            unit_norm: Norm {
                weight: transfers::to_cuda(
                    &Tensor::from_bf16(vec![DIM], &vec![half::bf16::ONE; DIM])?,
                    ctx.device_id(),
                )?,
                bias: transfers::to_cuda(
                    &Tensor::from_bf16(vec![DIM], &vec![half::bf16::ZERO; DIM])?,
                    ctx.device_id(),
                )?,
            },
        })
    }
    /// Time projections contain one six-vector group for video and one for
    /// action/state. This represents independent modality schedules directly.
    fn bind_modulation(&mut self, ctx: &Context, w: &Tensor, time: &Tensor) -> Result<()> {
        if time.shape().dims() != [12, DIM] {
            return Err(Error::Other(
                "DreamZero time modulation must be [12,5120]".into(),
            ));
        }
        for group in 0..2 {
            binary(
                ctx,
                ops::PointwiseSemantic::Add,
                &rows(time, group * 6, 6)?,
                w,
                &mut rows(&self.modulation, group * 6, 6)?,
            )?;
        }
        Ok(())
    }
    fn modulate(&mut self, ctx: &Context, shift: usize, scale: usize) -> Result<()> {
        for (group, start, count) in [
            (0, 0, self.video_rows),
            (1, self.video_rows, self.rows - self.video_rows),
        ] {
            if count == 0 {
                continue;
            }
            let scale_input = rows(&self.modulation, group * 6 + scale, 1)?;
            let mut args = ops::PointwiseArgs::new(
                ops::PointwiseSemantic::ScalarAdd,
                &scale_input,
                &mut self.scale,
            );
            args.dt = 1.;
            ops::pointwise(ctx, args)?;
            binary(
                ctx,
                ops::PointwiseSemantic::BroadcastMultiply,
                &rows(&self.normed, start, count)?,
                &self.scale,
                &mut rows(&self.gated, start, count)?,
            )?;
            binary(
                ctx,
                ops::PointwiseSemantic::BroadcastAdd,
                &rows(&self.gated, start, count)?,
                &rows(&self.modulation, group * 6 + shift, 1)?,
                &mut rows(&self.shifted, start, count)?,
            )?;
        }
        Ok(())
    }
    fn gate(&mut self, ctx: &Context, gate: usize) -> Result<()> {
        for (group, start, count) in [
            (0, 0, self.video_rows),
            (1, self.video_rows, self.rows - self.video_rows),
        ] {
            if count == 0 {
                continue;
            }
            binary(
                ctx,
                ops::PointwiseSemantic::BroadcastMultiply,
                &rows(&self.projected, start, count)?,
                &rows(&self.modulation, group * 6 + gate, 1)?,
                &mut rows(&self.gated, start, count)?,
            )?;
        }
        Ok(())
    }
}

/// `next_k/v` receive the candidate observed-video cache. The caller decides
/// whether to commit it; running a denoising step never mutates `history_k/v`.
#[allow(clippy::too_many_arguments)]
pub(in crate::dreamzero) fn block(
    ctx: &Context,
    p: &Policies,
    w: &Block,
    s: &mut BlockWorkspace,
    time: &Tensor,
    text: &Tensor,
    image: &Tensor,
    rotary: &ops::ComplexRotaryTable,
    history_k: Option<&Tensor>,
    history_v: Option<&Tensor>,
    observed_cache: Option<(&Tensor, &Tensor)>,
) -> Result<()> {
    if text.shape().dims() != [512, DIM]
        || image.shape().dims() != [257, DIM]
        || observed_cache.is_some_and(|(k, v)| {
            k.shape().dims() != [s.history_rows + s.video_rows, DIM] || v.shape() != k.shape()
        })
    {
        return Err(Error::Other(
            "DreamZero block context/cache shape mismatch".into(),
        ));
    }
    match (history_k, history_v) {
        (None, None) if s.history_rows == 0 => (),
        (Some(k), Some(v))
            if k.shape().dims() == [s.history_rows, DIM] && v.shape() == k.shape() =>
        {
            transfers::copy_cuda_to_cuda(ctx, k, &rows(&s.kv_k, 0, s.history_rows)?)?;
            transfers::copy_cuda_to_cuda(ctx, v, &rows(&s.kv_v, 0, s.history_rows)?)?;
        }
        _ => {
            return Err(Error::Other(
                "DreamZero history cache shape mismatch".into(),
            ))
        }
    }
    s.bind_modulation(ctx, &w.modulation, time)?;
    norm(ctx, &s.hidden, &s.unit_norm, &mut s.normed)?;
    s.modulate(ctx, 0, 1)?;
    let a = &w.self_attention;
    projection(ctx, p, &s.shifted, &a.q, &mut s.q_raw)?;
    projection(ctx, p, &s.shifted, &a.k, &mut s.k_raw)?;
    projection(ctx, p, &s.shifted, &a.v, &mut s.v)?;
    rms(ctx, &s.q_raw, &a.norm_q, &mut s.q)?;
    rms(ctx, &s.k_raw, &a.norm_k, &mut s.k)?;
    let shape = vec![s.rows, HEADS, DIM / HEADS];
    ops::complex_rotary(
        ctx,
        rotary,
        &s.q.reshape(shape.clone())?,
        &mut s.q_raw.reshape(shape.clone())?,
    )?;
    ops::complex_rotary(
        ctx,
        rotary,
        &s.k.reshape(shape.clone())?,
        &mut s.k_raw.reshape(shape)?,
    )?;
    transfers::copy_cuda_to_cuda(ctx, &s.k_raw, &rows(&s.kv_k, s.history_rows, s.rows)?)?;
    transfers::copy_cuda_to_cuda(ctx, &s.v, &rows(&s.kv_v, s.history_rows, s.rows)?)?;
    if let Some((next_k, next_v)) = observed_cache {
        let observed_rows = s.history_rows + s.video_rows;
        transfers::copy_cuda_to_cuda(ctx, &rows(&s.kv_k, 0, observed_rows)?, next_k)?;
        transfers::copy_cuda_to_cuda(ctx, &rows(&s.kv_v, 0, observed_rows)?, next_v)?;
    }
    attend(ctx, p, &s.q_raw, &s.kv_k, &s.kv_v, &mut s.attended)?;
    projection(ctx, p, &s.attended, &a.o, &mut s.projected)?;
    s.gate(ctx, 2)?;
    binary(
        ctx,
        ops::PointwiseSemantic::Add,
        &s.hidden,
        &s.gated,
        &mut s.residual,
    )?;

    norm(ctx, &s.residual, &w.norm3, &mut s.normed)?;
    let a = &w.cross_attention;
    projection(ctx, p, &s.normed, &a.q, &mut s.q_raw)?;
    rms(ctx, &s.q_raw, &a.norm_q, &mut s.q)?;
    projection(ctx, p, text, &a.k, &mut s.text_k_raw)?;
    rms(ctx, &s.text_k_raw, &a.norm_k, &mut s.text_k)?;
    projection(ctx, p, text, &a.v, &mut s.text_v)?;
    attend(ctx, p, &s.q, &s.text_k, &s.text_v, &mut s.attended)?;
    projection(ctx, p, image, &w.image_k, &mut s.image_k_raw)?;
    rms(ctx, &s.image_k_raw, &w.image_norm_k, &mut s.image_k)?;
    projection(ctx, p, image, &w.image_v, &mut s.image_v)?;
    attend(ctx, p, &s.q, &s.image_k, &s.image_v, &mut s.image_attended)?;
    binary(
        ctx,
        ops::PointwiseSemantic::Add,
        &s.attended,
        &s.image_attended,
        &mut s.shifted,
    )?;
    projection(ctx, p, &s.shifted, &a.o, &mut s.projected)?;
    binary(
        ctx,
        ops::PointwiseSemantic::Add,
        &s.residual,
        &s.projected,
        &mut s.hidden,
    )?;

    norm(ctx, &s.hidden, &s.unit_norm, &mut s.normed)?;
    s.modulate(ctx, 3, 4)?;
    projection(ctx, p, &s.shifted, &w.up, &mut s.up)?;
    let mut activation = ops::PointwiseArgs::new(
        ops::PointwiseSemantic::BiasActivation,
        &s.up,
        &mut s.activated,
    );
    activation.activation = ops::PointwiseActivation::Gelu;
    ops::pointwise(ctx, activation)?;
    projection(ctx, p, &s.activated, &w.down, &mut s.projected)?;
    s.gate(ctx, 5)?;
    binary(
        ctx,
        ops::PointwiseSemantic::Add,
        &s.hidden,
        &s.gated,
        &mut s.residual,
    )?;
    transfers::copy_cuda_to_cuda(ctx, &s.residual, &s.hidden)
}
