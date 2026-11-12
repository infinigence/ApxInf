//! Chinese HuBERT feature encoder plus the semantic-token quantizer.

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda_new::CudaContext as Context;

use super::config::{HUBERT_CONV_DIM, HUBERT_HEADS, HUBERT_HIDDEN, HUBERT_LAYERS};
use super::ops;
use super::weights::{fp16_tensor, Linear, Weights};

const CONV_KERNEL: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
const CONV_STRIDE: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];

struct EncoderLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln_w: Tensor,
    ln_b: Tensor,
    ff1: Linear,
    ff2: Linear,
    out_ln_w: Tensor,
    out_ln_b: Tensor,
}

pub struct Hubert {
    convs: Vec<Tensor>,
    conv_bias: Vec<Option<Tensor>>,
    group_w: Tensor,
    group_b: Tensor,
    proj_w: Tensor,
    proj_b: Option<Tensor>,
    proj_ln_w: Tensor,
    proj_ln_b: Tensor,
    pos_w: Tensor,
    pos_b: Option<Tensor>,
    enc_ln_w: Tensor,
    enc_ln_b: Tensor,
    layers: Vec<EncoderLayer>,
    ssl_proj_w: Tensor,
    ssl_proj_b: Option<Tensor>,
    codebook_t: Tensor,
    code_bias: Tensor,
}

impl Hubert {
    pub fn load(weights: &mut Weights, ctx: &Context) -> Result<Self> {
        let mut convs = Vec::new();
        let mut conv_bias = Vec::new();
        for index in 0..7 {
            convs.push(weights.get(ctx, &format!("hubert.feat.conv{index}.w"))?);
            conv_bias.push(weights.get_optional(ctx, &format!("hubert.feat.conv{index}.b"))?);
        }
        let group_w = weights.get(ctx, "hubert.feat.norm0.w")?;
        let group_b = weights.get(ctx, "hubert.feat.norm0.b")?;
        let proj_w = weights.get(ctx, "hubert.feat_proj.proj.w")?;
        let proj_b = weights.get_optional(ctx, "hubert.feat_proj.proj.b")?;
        let proj_ln_w = weights.get(ctx, "hubert.feat_proj.ln.w")?;
        let proj_ln_b = weights.get(ctx, "hubert.feat_proj.ln.b")?;
        let pos_w = weights.get(ctx, "hubert.posconv.w")?;
        let pos_b = weights.get_optional(ctx, "hubert.posconv.b")?;
        let enc_ln_w = weights.get(ctx, "hubert.enc_ln.w")?;
        let enc_ln_b = weights.get(ctx, "hubert.enc_ln.b")?;
        let mut layers = Vec::with_capacity(HUBERT_LAYERS);
        for index in 0..HUBERT_LAYERS {
            let prefix = format!("hubert.layers.{index}");
            layers.push(EncoderLayer {
                q: Linear::load(weights, ctx, &format!("{prefix}.q"))?,
                k: Linear::load(weights, ctx, &format!("{prefix}.k"))?,
                v: Linear::load(weights, ctx, &format!("{prefix}.v"))?,
                o: Linear::load(weights, ctx, &format!("{prefix}.o"))?,
                ln_w: weights.get(ctx, &format!("{prefix}.ln.w"))?,
                ln_b: weights.get(ctx, &format!("{prefix}.ln.b"))?,
                ff1: Linear::load(weights, ctx, &format!("{prefix}.ff1"))?,
                ff2: Linear::load(weights, ctx, &format!("{prefix}.ff2"))?,
                out_ln_w: weights.get(ctx, &format!("{prefix}.out_ln.w"))?,
                out_ln_b: weights.get(ctx, &format!("{prefix}.out_ln.b"))?,
            });
        }
        let ssl_proj_w = weights.get(ctx, "hubert.ssl_proj.w")?;
        let ssl_proj_b = weights.get_optional(ctx, "hubert.ssl_proj.b")?;

        // The quantizer codebook is stored [1024, 768]. Host-side transposition
        // and the column bias are load-time constants.
        let host_codebook = weights.host_clone("sovits.quant.codebook")?;
        let values = host_codebook.to_f32_vec()?;
        let bins = host_codebook.shape().dims()[0];
        let dim = host_codebook.shape().dims()[1];
        let mut transposed = vec![0.0f32; bins * dim];
        for bin in 0..bins {
            for channel in 0..dim {
                transposed[channel * bins + bin] = values[bin * dim + channel];
            }
        }
        let mut bias = vec![0.0f32; bins];
        for bin in 0..bins {
            let mut sum = 0.0f32;
            for channel in 0..dim {
                let value = values[bin * dim + channel];
                sum += value * value;
            }
            bias[bin] = -0.5 * sum;
        }
        let codebook = weights.to_device(ctx, &host_codebook)?;
        weights.put_device("sovits.quant.codebook", &codebook);
        let codebook_t = fp16_tensor(ctx, &[dim, bins], &transposed)?;
        let code_bias = fp16_tensor(ctx, &[bins], &bias)?;

        Ok(Self {
            convs,
            conv_bias,
            group_w,
            group_b,
            proj_w,
            proj_b,
            proj_ln_w,
            proj_ln_b,
            pos_w,
            pos_b,
            enc_ln_w,
            enc_ln_b,
            layers,
            ssl_proj_w,
            ssl_proj_b,
            codebook_t,
            code_bias,
        })
    }

    /// Encode 16 kHz mono PCM into the `[1, frames, 768]` pre-quantizer
    /// HuBERT representation used by the comparison harness.
    pub fn encode_features(&self, ctx: &Context, audio_16k: &[f32]) -> Result<Tensor> {
        if audio_16k.len() < 400 {
            return Err(Error::Other(
                "reference audio must contain at least 400 samples".into(),
            ));
        }
        let input = fp16_tensor(ctx, &[1, audio_16k.len(), 1], audio_16k)?;
        self.encode_features_device(ctx, &input)
    }

    pub(crate) fn semantic_frames(&self, samples: usize) -> Result<usize> {
        let mut frames = samples;
        for index in 0..7 {
            frames = frames
                .checked_sub(CONV_KERNEL[index])
                .ok_or_else(|| Error::Other("reference audio too short".into()))?
                / CONV_STRIDE[index]
                + 1;
        }
        Ok(frames
            .checked_sub(self.ssl_proj_w.shape().dims()[2])
            .ok_or_else(|| Error::Other("reference has no semantic frames".into()))?
            / 2
            + 1)
    }

    pub(crate) fn encode_features_device(&self, ctx: &Context, input: &Tensor) -> Result<Tensor> {
        let mut hidden = input.clone();
        // Quantizer decisions can change at sub-ULP feature differences. Keep
        // the validated convolution accumulation order for this frontend.
        for index in 0..7 {
            hidden = ops::conv1d(
                ctx,
                &hidden,
                &self.convs[index],
                self.conv_bias[index].as_ref(),
                CONV_STRIDE[index],
                0,
                1,
                1,
            )?;

            if index == 0 {
                hidden = ops::column_norm(ctx, &hidden, &self.group_w, &self.group_b)?;
            }
            hidden = ops::gelu_exact(ctx, &hidden)?;
        }
        let dims = hidden.shape().dims().to_vec();
        let hidden = hidden.reshape(apxinf_core::Shape::new(vec![dims[1], HUBERT_CONV_DIM]))?;
        let normalized = ops::layer_norm(ctx, &hidden, &self.proj_ln_w, &self.proj_ln_b, 1e-5)?;

        let projected =
            ops::linear_bias_accumulator(ctx, &normalized, &self.proj_w, self.proj_b.as_ref())?;
        let projected_3d =
            projected.reshape(apxinf_core::Shape::new(vec![1, dims[1], HUBERT_HIDDEN]))?;

        // Positional convolution (kernel 128, groups 16, same padding). The
        // even kernel produces one extra frame which HubertSamePadLayer trims.
        let positional = ops::conv1d_ordered(
            ctx,
            &projected_3d,
            &self.pos_w,
            self.pos_b.as_ref(),
            1,
            64,
            1,
            16,
        )?;

        let positional = ops::gelu_exact(ctx, &positional)?;
        let positional_dims = positional.shape().dims().to_vec();
        let positional = if positional_dims[1] == dims[1] + 1 {
            ops::slice_rows(
                ctx,
                &positional.reshape(apxinf_core::Shape::new(vec![
                    positional_dims[1],
                    HUBERT_HIDDEN,
                ]))?,
                0,
                dims[1],
            )?
            .reshape(apxinf_core::Shape::new(vec![1, dims[1], HUBERT_HIDDEN]))?
        } else {
            positional
        };
        let mut hidden = ops::residual_add(ctx, &positional, &projected_3d)?;
        let len = hidden.shape().dims()[1];
        let flat = hidden.reshape(apxinf_core::Shape::new(vec![len, HUBERT_HIDDEN]))?;
        let flat = ops::layer_norm(ctx, &flat, &self.enc_ln_w, &self.enc_ln_b, 1e-5)?;
        hidden = flat.reshape(apxinf_core::Shape::new(vec![1, len, HUBERT_HIDDEN]))?;

        for layer in &self.layers {
            let len = hidden.shape().dims()[1];
            let flat = hidden.reshape(apxinf_core::Shape::new(vec![len, HUBERT_HIDDEN]))?;
            let q = layer.q.apply(ctx, &flat)?;
            let k = layer.k.apply(ctx, &flat)?;
            let v = layer.v.apply(ctx, &flat)?;
            let head_dim = HUBERT_HIDDEN / HUBERT_HEADS;
            let q = q.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                HUBERT_HEADS,
                head_dim,
            ]))?;
            let k = k.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                HUBERT_HEADS,
                head_dim,
            ]))?;
            let v = v.reshape(apxinf_core::Shape::new(vec![
                1,
                len,
                HUBERT_HEADS,
                head_dim,
            ]))?;
            let attended = ops::attention(ctx, &q, &k, &v, HUBERT_HEADS)?;
            let attended = attended.reshape(apxinf_core::Shape::new(vec![len, HUBERT_HIDDEN]))?;

            let projected = layer.o.apply(ctx, &attended)?;
            let summed = ops::residual_add(ctx, &projected, &flat)?;
            let normed = ops::layer_norm(ctx, &summed, &layer.ln_w, &layer.ln_b, 1e-5)?;
            let ff = layer.ff1.apply(ctx, &normed)?;
            let ff = ops::gelu_exact(ctx, &ff)?;
            let ff = layer.ff2.apply(ctx, &ff)?;
            let summed = ops::residual_add(ctx, &ff, &normed)?;
            let normed = ops::layer_norm(ctx, &summed, &layer.out_ln_w, &layer.out_ln_b, 1e-5)?;
            hidden = normed.reshape(apxinf_core::Shape::new(vec![1, len, HUBERT_HIDDEN]))?;
        }
        Ok(hidden)
    }

    pub(crate) fn quantize(&self, ctx: &Context, hidden: &Tensor) -> Result<Tensor> {
        let len = hidden.shape().dims()[1];
        let flat = hidden.reshape(apxinf_core::Shape::new(vec![len, HUBERT_HIDDEN]))?;
        let flat_3d = flat.reshape(apxinf_core::Shape::new(vec![1, len, HUBERT_HIDDEN]))?;
        let ssl = ops::conv1d_ordered(
            ctx,
            &flat_3d,
            &self.ssl_proj_w,
            self.ssl_proj_b.as_ref(),
            2,
            0,
            1,
            1,
        )?;
        let frames = ssl.shape().dims()[1];
        let ssl = ssl.reshape(apxinf_core::Shape::new(vec![frames, HUBERT_HIDDEN]))?;
        let distances = ops::gemm(ctx, &ssl, &self.codebook_t, Some(&self.code_bias))?;
        {
            let indices = ctx.allocate_output(
                apxinf_core::Shape::new(vec![frames]),
                apxinf_core::DType::I32,
            )?;
            apxinf_cuda_new::ops::argmax_rows(ctx, &distances, &indices)?;
            return Ok(indices);
        }
    }

    /// Encode 16 kHz mono PCM into 25 Hz semantic tokens.
    pub fn encode(&self, ctx: &Context, audio_16k: &[f32]) -> Result<Tensor> {
        let features = self.encode_features(ctx, audio_16k)?;
        self.quantize(ctx, &features)
    }
}
