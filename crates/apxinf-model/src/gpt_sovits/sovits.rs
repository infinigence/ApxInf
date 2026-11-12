//! SoVITS acoustic encoder for GPT-SoVITS V5.

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda_new::CudaContext as Context;

use super::config::{
    SOVITS_DECODER, SOVITS_FF, SOVITS_HEADS, SOVITS_HIDDEN, SOVITS_INTER, SOVITS_WINDOW,
    SOVITS_WN_LAYERS,
};
use super::ops;
use super::weights::{ids_tensor, Linear, Weights};

// The MRTE cross-attention lives in the decoder-width style space, not in the
// 192-wide `SOVITS_INTER` space of the surrounding convolutions.
const MRTE_HIDDEN: usize = SOVITS_DECODER;
const MRTE_HEADS: usize = 4;

struct AttnLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    emb_k: Tensor,
    emb_v: Tensor,
    ln1_w: Tensor,
    ln1_b: Tensor,
    ff1_w: Tensor,
    ff1_b: Option<Tensor>,
    ff2_w: Tensor,
    ff2_b: Option<Tensor>,
    ln2_w: Tensor,
    ln2_b: Tensor,
}

struct EncoderStack {
    layers: Vec<AttnLayer>,
}

impl EncoderStack {
    fn load(weights: &mut Weights, ctx: &Context, prefix: &str, count: usize) -> Result<Self> {
        let mut layers = Vec::with_capacity(count);
        for index in 0..count {
            let base = format!("{prefix}.{index}");
            layers.push(AttnLayer {
                q: Linear::load(weights, ctx, &format!("{base}.q"))?,
                k: Linear::load(weights, ctx, &format!("{base}.k"))?,
                v: Linear::load(weights, ctx, &format!("{base}.v"))?,
                o: Linear::load(weights, ctx, &format!("{base}.o"))?,
                emb_k: weights.get(ctx, &format!("{base}.emb_k"))?.reshape(
                    apxinf_core::Shape::new(vec![
                        2 * SOVITS_WINDOW + 1,
                        SOVITS_HIDDEN / SOVITS_HEADS,
                    ]),
                )?,
                emb_v: weights.get(ctx, &format!("{base}.emb_v"))?.reshape(
                    apxinf_core::Shape::new(vec![
                        2 * SOVITS_WINDOW + 1,
                        SOVITS_HIDDEN / SOVITS_HEADS,
                    ]),
                )?,
                ln1_w: weights.get(ctx, &format!("{base}.ln1.w"))?,
                ln1_b: weights.get(ctx, &format!("{base}.ln1.b"))?,
                ff1_w: weights.get(ctx, &format!("{base}.ff1.w"))?,
                ff1_b: weights.get_optional(ctx, &format!("{base}.ff1.b"))?,
                ff2_w: weights.get(ctx, &format!("{base}.ff2.w"))?,
                ff2_b: weights.get_optional(ctx, &format!("{base}.ff2.b"))?,
                ln2_w: weights.get(ctx, &format!("{base}.ln2.w"))?,
                ln2_b: weights.get(ctx, &format!("{base}.ln2.b"))?,
            });
        }
        Ok(Self { layers })
    }

    fn forward(&self, ctx: &Context, input: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims().to_vec();
        let len = dims[0];
        let mut x = input.clone();
        for layer in &self.layers {
            let q = layer.q.apply(ctx, &x)?;
            let k = layer.k.apply(ctx, &x)?;
            let v = layer.v.apply(ctx, &x)?;
            let head_dim = SOVITS_HIDDEN / SOVITS_HEADS;
            let q = q.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                SOVITS_HEADS,
                head_dim,
            ]))?;
            let k = k.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                SOVITS_HEADS,
                head_dim,
            ]))?;
            let v = v.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                SOVITS_HEADS,
                head_dim,
            ]))?;
            let attended = ops::relative_attention(
                ctx,
                &q,
                &k,
                &v,
                &layer.emb_k,
                &layer.emb_v,
                SOVITS_WINDOW,
            )?;
            let attended = attended.reshape(apxinf_core::Shape::new(vec![len, SOVITS_HIDDEN]))?;
            let projected = layer.o.apply(ctx, &attended)?;
            let summed = ops::residual_add(ctx, &projected, &x)?;
            let normed = ops::layer_norm(ctx, &summed, &layer.ln1_w, &layer.ln1_b, 1e-5)?;

            let ff = ops::conv1d(
                ctx,
                &normed.reshape(apxinf_core::Shape::new(vec![1, len, SOVITS_HIDDEN]))?,
                &layer.ff1_w,
                layer.ff1_b.as_ref(),
                1,
                1,
                1,
                1,
            )?;
            let ff = ops::relu(ctx, &ff)?;
            let ff = ops::conv1d(ctx, &ff, &layer.ff2_w, layer.ff2_b.as_ref(), 1, 1, 1, 1)?;
            let ff = ff.reshape(apxinf_core::Shape::new(vec![len, SOVITS_HIDDEN]))?;
            let summed = ops::residual_add(ctx, &ff, &normed)?;
            x = ops::layer_norm(ctx, &summed, &layer.ln2_w, &layer.ln2_b, 1e-5)?;
        }
        Ok(x)
    }
}

struct Mrte {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    c_pre_w: Tensor,
    c_pre_b: Option<Tensor>,
    text_pre_w: Tensor,
    text_pre_b: Option<Tensor>,
    c_post_w: Tensor,
    c_post_b: Option<Tensor>,
}

impl Mrte {
    fn load(weights: &mut Weights, ctx: &Context) -> Result<Self> {
        Ok(Self {
            q: Linear::load(weights, ctx, "sovits.mrte.q")?,
            k: Linear::load(weights, ctx, "sovits.mrte.k")?,
            v: Linear::load(weights, ctx, "sovits.mrte.v")?,
            o: Linear::load(weights, ctx, "sovits.mrte.o")?,
            c_pre_w: weights.get(ctx, "sovits.mrte.c_pre.w")?,
            c_pre_b: weights.get_optional(ctx, "sovits.mrte.c_pre.b")?,
            text_pre_w: weights.get(ctx, "sovits.mrte.text_pre.w")?,
            text_pre_b: weights.get_optional(ctx, "sovits.mrte.text_pre.b")?,
            c_post_w: weights.get(ctx, "sovits.mrte.c_post.w")?,
            c_post_b: weights.get_optional(ctx, "sovits.mrte.c_post.b")?,
        })
    }

    fn forward(
        &self,
        ctx: &Context,
        content: &Tensor,
        text: &Tensor,
        ge: &Tensor,
    ) -> Result<Tensor> {
        let len = content.shape().dims()[0];
        let text_len = text.shape().dims()[0];
        let content_3d = content.reshape(apxinf_core::Shape::new(vec![1, len, SOVITS_INTER]))?;
        let c = ops::conv1d(
            ctx,
            &content_3d,
            &self.c_pre_w,
            self.c_pre_b.as_ref(),
            1,
            0,
            1,
            1,
        )?;
        let text_3d = text.reshape(apxinf_core::Shape::new(vec![1, text_len, SOVITS_INTER]))?;
        let t = ops::conv1d(
            ctx,
            &text_3d,
            &self.text_pre_w,
            self.text_pre_b.as_ref(),
            1,
            0,
            1,
            1,
        )?;
        let head_dim = MRTE_HIDDEN / MRTE_HEADS;
        let q = self.q.apply(
            ctx,
            &c.reshape(apxinf_core::Shape::new(vec![len, MRTE_HIDDEN]))?,
        )?;
        let k = self.k.apply(
            ctx,
            &t.reshape(apxinf_core::Shape::new(vec![text_len, MRTE_HIDDEN]))?,
        )?;
        let v = self.v.apply(
            ctx,
            &t.reshape(apxinf_core::Shape::new(vec![text_len, MRTE_HIDDEN]))?,
        )?;
        let q = q.reshape(apxinf_core::Shape::new(vec![1, len, MRTE_HEADS, head_dim]))?;
        let k = k.reshape(apxinf_core::Shape::new(vec![
            1, text_len, MRTE_HEADS, head_dim,
        ]))?;
        let v = v.reshape(apxinf_core::Shape::new(vec![
            1, text_len, MRTE_HEADS, head_dim,
        ]))?;
        let attended = ops::attention(ctx, &q, &k, &v, MRTE_HEADS)?;
        let attended = attended.reshape(apxinf_core::Shape::new(vec![len, MRTE_HIDDEN]))?;
        let projected = self.o.apply(ctx, &attended)?;
        let summed = ops::residual_add(
            ctx,
            &projected,
            &c.reshape(apxinf_core::Shape::new(vec![len, MRTE_HIDDEN]))?,
        )?;
        let with_style = ops::add_bias(ctx, &summed, ge)?;
        let post = ops::conv1d(
            ctx,
            &with_style.reshape(apxinf_core::Shape::new(vec![1, len, MRTE_HIDDEN]))?,
            &self.c_post_w,
            self.c_post_b.as_ref(),
            1,
            0,
            1,
            1,
        )?;
        Ok(post.reshape(apxinf_core::Shape::new(vec![len, SOVITS_INTER]))?)
    }
}

pub struct TextEncoder {
    ssl_proj_w: Tensor,
    ssl_proj_b: Option<Tensor>,
    encoder_ssl: EncoderStack,
    encoder_text: EncoderStack,
    encoder_tail: EncoderStack,
    text_emb: Tensor,
    mrte: Mrte,
}

impl TextEncoder {
    fn load(weights: &mut Weights, ctx: &Context) -> Result<Self> {
        Ok(Self {
            ssl_proj_w: weights.get(ctx, "sovits.enc.ssl_proj.w")?,
            ssl_proj_b: weights.get_optional(ctx, "sovits.enc.ssl_proj.b")?,
            encoder_ssl: EncoderStack::load(weights, ctx, "sovits.enc.ssl", 3)?,
            encoder_text: EncoderStack::load(weights, ctx, "sovits.enc.text", 6)?,
            encoder_tail: EncoderStack::load(weights, ctx, "sovits.enc.tail", 3)?,
            text_emb: weights.get(ctx, "sovits.enc.text_emb")?,
            mrte: Mrte::load(weights, ctx)?,
        })
    }

    fn forward(
        &self,
        ctx: &Context,
        quantized: &Tensor,
        text_ids: &Tensor,
        ge: &Tensor,
    ) -> Result<Tensor> {
        let len = quantized.shape().dims()[0];
        let y = ops::conv1d(
            ctx,
            &quantized.reshape(apxinf_core::Shape::new(vec![1, len, 768]))?,
            &self.ssl_proj_w,
            self.ssl_proj_b.as_ref(),
            1,
            0,
            1,
            1,
        )?;
        let mut y = y.reshape(apxinf_core::Shape::new(vec![len, SOVITS_HIDDEN]))?;
        y = self.encoder_ssl.forward(ctx, &y)?;

        let text = ops::embedding(ctx, &self.text_emb, text_ids)?;
        let text = self.encoder_text.forward(ctx, &text)?;
        let mut y = self.mrte.forward(ctx, &y, &text, ge)?;
        y = self.encoder_tail.forward(ctx, &y)?;
        Ok(y)
    }
}

struct RefEncoder {
    spec0: Linear,
    spec3: Linear,
    temp0_x1: Linear,
    temp0_x2: Linear,
    temp1_x1: Linear,
    temp1_x2: Linear,
    attn_q: Linear,
    attn_k: Linear,
    attn_v: Linear,
    attn_o: Linear,
    attn_q_scale: Tensor,
    attn_q_shift: Tensor,
    out: Linear,
}

impl RefEncoder {
    fn load(weights: &mut Weights, ctx: &Context) -> Result<Self> {
        Ok(Self {
            spec0: Linear::load(weights, ctx, "sovits.refenc.spec0")?,
            spec3: Linear::load(weights, ctx, "sovits.refenc.spec3")?,
            temp0_x1: Linear::load(weights, ctx, "sovits.refenc.temp0.x1")?,
            temp0_x2: Linear::load(weights, ctx, "sovits.refenc.temp0.x2")?,
            temp1_x1: Linear::load(weights, ctx, "sovits.refenc.temp1.x1")?,
            temp1_x2: Linear::load(weights, ctx, "sovits.refenc.temp1.x2")?,
            attn_q: Linear::load(weights, ctx, "sovits.refenc.attn.q")?,
            attn_k: Linear::load(weights, ctx, "sovits.refenc.attn.k")?,
            attn_v: Linear::load(weights, ctx, "sovits.refenc.attn.v")?,
            attn_o: Linear::load(weights, ctx, "sovits.refenc.attn.o")?,
            // The reference MelStyleEncoder scales attention logits by
            // sqrt(d_model=128), while the shared attention operator scales by
            // sqrt(head_dim=64). Pre-scale q by sqrt(64 / 128).
            attn_q_scale: super::weights::fp16_tensor(
                ctx,
                &[128],
                &vec![std::f32::consts::FRAC_1_SQRT_2; 128],
            )?,
            attn_q_shift: super::weights::fp16_tensor(ctx, &[128], &[0.0; 128])?,
            out: Linear::load(weights, ctx, "sovits.refenc.out")?,
        })
    }

    fn forward(&self, ctx: &Context, spec: &Tensor) -> Result<Tensor> {
        let len = spec.shape().dims()[0];
        let mut x = self.spec0.apply(ctx, spec)?;
        x = ops::mish(ctx, &x)?;
        x = self.spec3.apply(ctx, &x)?;
        x = ops::mish(ctx, &x)?;
        for (first, second) in [
            (&self.temp0_x1, &self.temp0_x2),
            (&self.temp1_x1, &self.temp1_x2),
        ] {
            let x3 = x.reshape(apxinf_core::Shape::new(vec![1, len, 128]))?;
            let a = ops::conv1d(ctx, &x3, &first.w, first.b.as_ref(), 1, 2, 1, 1)?;
            let b = ops::conv1d(ctx, &x3, &second.w, second.b.as_ref(), 1, 2, 1, 1)?;
            let a = a.reshape(apxinf_core::Shape::new(vec![len, 128]))?;
            let b = b.reshape(apxinf_core::Shape::new(vec![len, 128]))?;
            let gated = ops::glu_separate(ctx, &a, &b)?;
            x = ops::residual_add(ctx, &gated, &x)?;
        }
        let q = self.attn_q.apply(ctx, &x)?;
        let q = ops::affine(ctx, &q, &self.attn_q_scale, &self.attn_q_shift)?;
        let k = self.attn_k.apply(ctx, &x)?;
        let v = self.attn_v.apply(ctx, &x)?;
        let q = q.reshape(apxinf_core::Shape::new(vec![1, len, 2, 64]))?;
        let k = k.reshape(apxinf_core::Shape::new(vec![1, len, 2, 64]))?;
        let v = v.reshape(apxinf_core::Shape::new(vec![1, len, 2, 64]))?;
        let attended = ops::attention(ctx, &q, &k, &v, 2)?;
        let attended = attended.reshape(apxinf_core::Shape::new(vec![len, 128]))?;
        let projected = self.attn_o.apply(ctx, &attended)?;
        x = ops::residual_add(ctx, &projected, &x)?;
        let x = self.out.apply(ctx, &x)?;
        ops::mean_rows(ctx, &x)
    }
}

struct Wn {
    pre_w: Tensor,
    pre_b: Option<Tensor>,
    in_w: Vec<Tensor>,
    in_b: Vec<Option<Tensor>>,
    res_w: Vec<Tensor>,
    res_b: Vec<Option<Tensor>>,
    skip_w: Vec<Tensor>,
    skip_b: Vec<Option<Tensor>>,
    cond_w: Vec<Tensor>,
    cond_b: Vec<Tensor>,
    proj_w: Tensor,
    proj_b: Option<Tensor>,
}

impl Wn {
    fn load(weights: &mut Weights, ctx: &Context) -> Result<Self> {
        let mut in_w = Vec::new();
        let mut in_b = Vec::new();
        let mut res_w = Vec::new();
        let mut res_b = Vec::new();
        let mut skip_w = Vec::new();
        let mut skip_b = Vec::new();
        let mut cond_w = Vec::new();
        let mut cond_b = Vec::new();
        for index in 0..SOVITS_WN_LAYERS {
            in_w.push(weights.get(ctx, &format!("sovits.wns.in.{index}.w"))?);
            in_b.push(weights.get_optional(ctx, &format!("sovits.wns.in.{index}.b"))?);
            if index + 1 < SOVITS_WN_LAYERS {
                res_w.push(weights.get(ctx, &format!("sovits.wns.rs.{index}.res.w"))?);
                res_b.push(weights.get_optional(ctx, &format!("sovits.wns.rs.{index}.res.b"))?);
            }
            skip_w.push(weights.get(ctx, &format!("sovits.wns.rs.{index}.skip.w"))?);
            skip_b.push(weights.get_optional(ctx, &format!("sovits.wns.rs.{index}.skip.b"))?);
            cond_w.push(weights.get(ctx, &format!("sovits.wns.cond.{index}.w"))?);
            cond_b.push(weights.get(ctx, &format!("sovits.wns.cond.{index}.b"))?);
        }
        Ok(Self {
            pre_w: weights.get(ctx, "sovits.wns.pre.w")?,
            pre_b: weights.get_optional(ctx, "sovits.wns.pre.b")?,
            in_w,
            in_b,
            res_w,
            res_b,
            skip_w,
            skip_b,
            cond_w,
            cond_b,
            proj_w: weights.get(ctx, "sovits.wns.proj.w")?,
            proj_b: weights.get_optional(ctx, "sovits.wns.proj.b")?,
        })
    }

    fn forward(&self, ctx: &Context, input: &Tensor, ge: &Tensor) -> Result<Tensor> {
        let len = input.shape().dims()[0];
        let mut x = ops::conv1d(
            ctx,
            &input.reshape(apxinf_core::Shape::new(vec![1, len, SOVITS_DECODER]))?,
            &self.pre_w,
            self.pre_b.as_ref(),
            1,
            0,
            1,
            1,
        )?;
        x = x.reshape(apxinf_core::Shape::new(vec![len, SOVITS_DECODER]))?;
        let mut output = ops::zeros_like(ctx, &x)?;
        let ge_3d = ge.reshape(apxinf_core::Shape::new(vec![1, 1, SOVITS_DECODER]))?;
        for index in 0..SOVITS_WN_LAYERS {
            let x3 = x.reshape(apxinf_core::Shape::new(vec![1, len, SOVITS_DECODER]))?;
            let x_in = ops::conv1d(
                ctx,
                &x3,
                &self.in_w[index],
                self.in_b[index].as_ref(),
                1,
                2,
                1,
                1,
            )?;
            let x_in = x_in.reshape(apxinf_core::Shape::new(vec![len, 1024]))?;
            let g = ops::conv1d(
                ctx,
                &ge_3d,
                &self.cond_w[index],
                Some(&self.cond_b[index]),
                1,
                0,
                1,
                1,
            )?;
            let g = g.reshape(apxinf_core::Shape::new(vec![1024]))?;
            let gated_input = ops::add_bias(ctx, &x_in, &g)?;
            let acts = ops::tanh_gate(ctx, &gated_input, None)?;
            let acts3 = acts.reshape(apxinf_core::Shape::new(vec![1, len, SOVITS_DECODER]))?;
            if index + 1 < SOVITS_WN_LAYERS {
                let res = ops::conv1d(
                    ctx,
                    &acts3,
                    &self.res_w[index],
                    self.res_b[index].as_ref(),
                    1,
                    0,
                    1,
                    1,
                )?;
                let res = res.reshape(apxinf_core::Shape::new(vec![len, SOVITS_DECODER]))?;
                x = ops::residual_add(ctx, &res, &x)?;
            }
            let skip = ops::conv1d(
                ctx,
                &acts3,
                &self.skip_w[index],
                self.skip_b[index].as_ref(),
                1,
                0,
                1,
                1,
            )?;
            let skip = skip.reshape(apxinf_core::Shape::new(vec![len, SOVITS_DECODER]))?;
            output = ops::residual_add(ctx, &skip, &output)?;
        }
        let out3 = output.reshape(apxinf_core::Shape::new(vec![1, len, SOVITS_DECODER]))?;
        let projected = ops::conv1d(ctx, &out3, &self.proj_w, self.proj_b.as_ref(), 1, 0, 1, 1)?;
        let final_output = projected.reshape(apxinf_core::Shape::new(vec![len, SOVITS_DECODER]))?;
        Ok(final_output)
    }
}

pub struct Sovits {
    text_encoder: TextEncoder,
    ref_encoder: RefEncoder,
    bridge_w: Tensor,
    bridge_b: Option<Tensor>,
    wn: Wn,
    codebook: Tensor,
}

impl Sovits {
    pub fn load(weights: &mut Weights, ctx: &Context) -> Result<Self> {
        Ok(Self {
            text_encoder: TextEncoder::load(weights, ctx)?,
            ref_encoder: RefEncoder::load(weights, ctx)?,
            bridge_w: weights.get(ctx, "sovits.bridge.w")?,
            bridge_b: weights.get_optional(ctx, "sovits.bridge.b")?,
            wn: Wn::load(weights, ctx)?,
            codebook: weights.get(ctx, "sovits.quant.codebook")?,
        })
    }

    /// Compute the style embedding from the 704-bin reference spectrogram.
    pub fn style(&self, ctx: &Context, spec_704: &Tensor) -> Result<Tensor> {
        self.ref_encoder.forward(ctx, spec_704)
    }

    /// Decode semantic codes into 100 Hz mel conditioning features.
    pub fn decode_features(
        &self,
        ctx: &Context,
        codes: &Tensor,
        text_ids: &[u32],
        ge: &Tensor,
        speed_factor: f32,
    ) -> Result<Tensor> {
        let text_ids = ids_tensor(ctx, text_ids)?;
        let upsample_codes = repeat_row_ids(ctx, codes.numel())?;
        let upsample_bridge = repeat_row_ids(ctx, codes.numel() * 2)?;
        self.decode_features_device(
            ctx,
            codes,
            &text_ids,
            ge,
            &upsample_codes,
            &upsample_bridge,
            speed_factor,
        )
    }

    pub(crate) fn decode_features_device(
        &self,
        ctx: &Context,
        codes: &Tensor,
        text_ids: &Tensor,
        ge: &Tensor,
        upsample_codes: &Tensor,
        upsample_bridge: &Tensor,
        speed_factor: f32,
    ) -> Result<Tensor> {
        if !speed_factor.is_finite() || speed_factor <= 0.0 {
            return Err(Error::Other(format!(
                "SoVITS speed_factor must be finite and positive, got {speed_factor}"
            )));
        }
        let quantized = ops::embedding(ctx, &self.codebook, codes)?;
        let quantized = ops::embedding(ctx, &quantized, upsample_codes)?;
        let encoded = self.text_encoder.forward(ctx, &quantized, text_ids, ge)?;
        // Official `TextEncoder.forward(..., speed)` interpolates the encoder
        // output before `proj`; it does not resample the waveform. Reproduce
        // that latent-length change here.
        let encoded = if (speed_factor - 1.0).abs() > 1e-6 {
            let input_frames = encoded.shape().dims()[0];
            let output_frames = ops::speed_output_frames(input_frames, speed_factor);
            ops::linear_interpolate_rows(ctx, &encoded, output_frames)?
        } else {
            encoded
        };
        let len = encoded.shape().dims()[0];
        let encoded = encoded.reshape(apxinf_core::Shape::new(vec![1, len, SOVITS_INTER]))?;
        let bridged = ops::conv1d(
            ctx,
            &encoded,
            &self.bridge_w,
            self.bridge_b.as_ref(),
            1,
            0,
            1,
            1,
        )?;
        let bridged = ops::leaky_relu_one_percent(ctx, &bridged)?;
        let bridged = bridged.reshape(apxinf_core::Shape::new(vec![len, SOVITS_DECODER]))?;
        let bridged = if (speed_factor - 1.0).abs() > 1e-6 {
            // The prepared row map is sized for native speed. Rebuild the
            // nearest-neighbour 2x expansion for the speed-adjusted length.
            ops::upsample2(ctx, &bridged)?
        } else {
            ops::embedding(ctx, &bridged, upsample_bridge)?
        };
        self.wn.forward(ctx, &bridged, ge)
    }
}

pub fn hidden() -> usize {
    SOVITS_HIDDEN
}

pub fn ff() -> usize {
    SOVITS_FF
}

// Immutable row maps are created before capture and retained by the graph.
pub(crate) fn repeat_row_ids(ctx: &Context, rows: usize) -> Result<Tensor> {
    let ids = (0..rows as u32)
        .flat_map(|row| [row, row])
        .collect::<Vec<_>>();
    ids_tensor(ctx, &ids)
}
