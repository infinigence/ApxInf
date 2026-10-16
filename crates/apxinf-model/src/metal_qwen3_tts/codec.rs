//! The 12 Hz waveform decoder (mlx-audio `Qwen3TTSSpeechTokenizerDecoder`)
//! with the parity seams, in NLC layout throughout.
//!
//! As in mlx-audio, the decoder transformer uses a full causal mask; the
//! checkpoint's `sliding_window` is not applied.
use super::config::codec::*;
use super::config::CODE_GROUPS;
use super::ops::Ops;
use super::weights::{Codec, CodecLayer, Conv, ConvNext, Snake};
use apxinf_core::Result;
use apxinf_mlx::{Array, MlxDType};

const CHUNK: usize = 300;
const LEFT_CONTEXT: usize = 25;

pub struct Decoder {
    w: Codec,
    ops: Ops,
    inv_freq: Array,
}

impl Decoder {
    pub fn new(w: Codec, ops: &Ops) -> Result<Self> {
        let inv_freq = ops.inv_freq(HEAD_DIM, ROPE_THETA)?;
        inv_freq.eval()?;
        Ok(Self {
            w,
            ops: ops.clone(),
            inv_freq,
        })
    }

    fn scalar(&self, value: f64, like: &Array) -> Result<Array> {
        self.ops.scalar(value, like.dtype())
    }

    /// `CausalConv1d`: left padding `(K - 1) * dilation`, then `nn.Conv1d`.
    fn causal(&self, x: &Array, c: &Conv, dilation: usize, groups: usize) -> Result<Array> {
        let pad = (c.weight.shape()[1] - 1) * dilation;
        let x = if pad > 0 {
            x.pad(&[0, pad, 0], &[0, 0, 0])?
        } else {
            x.clone()
        };
        x.conv1d(&c.weight, 1, 0, dilation, groups)?.add(&c.bias)
    }

    /// `nn.ConvTranspose1d`, then the causal right trim of `K - stride`.
    fn upsample(&self, x: &Array, c: &Conv, stride: usize) -> Result<Array> {
        let y = x
            .conv_transpose1d(&c.weight, stride, 0, 1, 0, 1)?
            .add(&c.bias)?;
        let trim = c.weight.shape()[1] - stride;
        if trim == 0 {
            return Ok(y);
        }
        let length = y.shape()[1];
        y.slice_axis(1, 0, length - trim)
    }

    /// `SnakeBeta`: `x + 1 / (exp(beta) + 1e-9) * sin(x * exp(alpha)) ** 2`.
    fn snake(&self, x: &Array, s: &Snake) -> Result<Array> {
        let alpha = s.alpha.exp()?;
        let beta = s.beta.exp()?;
        let inverse = self
            .scalar(1.0, &beta)?
            .div(&beta.add(&self.scalar(1e-9, &beta)?)?)?;
        let wave = x.mul(&alpha)?.sin()?.pow(&self.scalar(2.0, x)?)?;
        x.add(&inverse.mul(&wave)?)
    }

    /// The parity codec RMSNorm: FP32 statistics, cast, then the weight.
    fn norm(&self, x: &Array, weight: &Array) -> Result<Array> {
        let work = x.cast(MlxDType::F32)?;
        let variance = work.mul(&work)?.mean(&[-1], true)?;
        let eps = self.ops.scalar(EPS as f64, MlxDType::F32)?;
        work.mul(&variance.add(&eps)?.rsqrt()?)?
            .cast(x.dtype())?
            .mul(weight)
    }

    fn layer(
        &self,
        l: &CodecLayer,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let [b, t, _] = *x.shape() else {
            unreachable!("decoder hidden states are [B, T, C]")
        };
        let h = self.norm(x, &l.input_norm)?;
        let a = &l.attention;
        let split = |y: Array| -> Result<Array> {
            y.reshape(&[b, t, HEADS, HEAD_DIM])?
                .transpose(&[0, 2, 1, 3])
        };
        let (q, k) =
            self.ops
                .rope_eager(&split(a.q.apply(&h)?)?, &split(a.k.apply(&h)?)?, cos, sin)?;
        let v = split(a.v.apply(&h)?)?;
        let scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let out = q.sdpa(&k, &v, scale, false, mask)?;
        let out =
            a.o.apply(
                &out.transpose(&[0, 2, 1, 3])?
                    .reshape(&[b, t, HEADS * HEAD_DIM])?,
            )?;
        let x = x.add(&l.attention_scale.mul(&out)?)?;
        let h = self.norm(&x, &l.post_norm)?;
        let m = &l.mlp;
        let y = m
            .down
            .apply(&self.ops.swiglu(&m.gate.apply(&h)?, &m.up.apply(&h)?)?)?;
        x.add(&l.mlp_scale.mul(&y)?)
    }

    fn transformer(&self, x: &Array) -> Result<Array> {
        let x = self.w.input_proj.apply(x)?;
        let t = x.shape()[1];
        let half = HEAD_DIM / 2;
        let positions = Array::arange(self.ops.stream(), 0.0, t as f32, 1.0, MlxDType::I32)?
            .reshape(&[1, 1, t])?
            .cast(MlxDType::F32)?;
        let freqs = self
            .inv_freq
            .reshape(&[1, half, 1])?
            .mul(&positions)?
            .transpose(&[0, 2, 1])?;
        let emb = Array::concat(&[&freqs, &freqs], 2)?;
        let cos = emb.cos()?.cast(x.dtype())?;
        let sin = emb.sin()?.cast(x.dtype())?;
        let mask = (t > 1)
            .then(|| self.ops.causal_mask(t, x.dtype()))
            .transpose()?;
        let mut x = x;
        for l in &self.w.layers {
            x = self.layer(l, &x, &cos, &sin, mask.as_ref())?;
        }
        self.w.output_proj.apply(&self.norm(&x, &self.w.norm)?)
    }

    fn convnext(&self, x: &Array, c: &ConvNext) -> Result<Array> {
        let groups = c.dwconv.weight.shape()[0];
        let y = self.causal(x, &c.dwconv, 1, groups)?;
        let y = y.fast_layer_norm(Some(&c.norm_weight), Some(&c.norm_bias), 1e-6)?;
        let y = c.pwconv2.apply(&self.ops.gelu(&c.pwconv1.apply(&y)?)?)?;
        x.add(&c.gamma.mul(&y)?)
    }

    /// Codebook lookup `[1, T]` -> `[1, 256, T]` (`VectorQuantization.decode`).
    fn lookup(&self, codebook: &Array, codes: &Array) -> Result<Array> {
        codebook.take(codes, 0)?.transpose(&[0, 2, 1])
    }

    /// Residual VQ over `rows` of `codes` `[1, 16, T]`: BF16 ordered sums, then
    /// the 1x1 output projection; `[1, 512, T]`.
    fn rvq(&self, codes: &Array, rows: std::ops::Range<usize>, output: &Array) -> Result<Array> {
        let t = codes.shape()[2];
        let code = |i: usize| codes.slice(&[0, i, 0], &[1, i + 1, t])?.reshape(&[1, t]);
        let first = self.lookup(&self.w.codebooks[rows.start], &code(rows.start)?)?;
        let mut q = Array::zeros(self.ops.stream(), first.shape(), first.dtype())?.add(&first)?;
        for i in rows.start + 1..rows.end {
            q = q.add(&self.lookup(&self.w.codebooks[i], &code(i)?)?)?;
        }
        q.transpose(&[0, 2, 1])?
            .conv1d(output, 1, 0, 1, 1)?
            .transpose(&[0, 2, 1])
    }

    /// One decoder call on codes `[1, 16, T]`: waveform `[1, 1, T * 1920]` BF16.
    fn call(&self, codes: &Array) -> Result<Array> {
        let w = &self.w;
        let hidden = self
            .rvq(codes, 0..1, &w.first_output)?
            .add(&self.rvq(codes, 1..CODE_GROUPS, &w.rest_output)?)?
            .transpose(&[0, 2, 1])?;
        // pre_conv with FP32 convolution and bias, then one cast.
        let padded = hidden.pad(&[0, 2, 0], &[0, 0, 0])?;
        let hidden = padded
            .cast(MlxDType::F32)?
            .conv1d(&w.pre_conv_weight, 1, 0, 1, 1)?
            .add(&w.pre_conv_bias)?
            .cast(padded.dtype())?;
        let mut x = self.transformer(&hidden)?;
        for ((up, next), &factor) in w.upsample.iter().zip(&UPSAMPLING_RATIOS) {
            x = self.convnext(&self.upsample(&x, up, factor)?, next)?;
        }
        x = self.causal(&x, &w.initial, 1, 1)?;
        for block in &w.blocks {
            x = self.upsample(&self.snake(&x, &block.snake)?, &block.upsample, block.rate)?;
            for r in &block.residuals {
                let y = self.causal(&self.snake(&x, &r.act1)?, &r.conv1, r.dilation, 1)?;
                let y = self.causal(&self.snake(&y, &r.act2)?, &r.conv2, 1, 1)?;
                x = y.add(&x)?;
            }
        }
        let x = self.causal(&self.snake(&x, &w.final_snake)?, &w.final_conv, 1, 1)?;
        let x = x.transpose(&[0, 2, 1])?;
        x.clip(&self.scalar(-1.0, &x)?, &self.scalar(1.0, &x)?)
    }

    /// `chunked_decode` of codes `[1, 16, F]`: `F * 1920` samples, as host `f32`.
    pub fn decode(&self, codes: &Array) -> Result<Vec<f32>> {
        let frames = codes.shape()[2];
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < frames {
            let end = (start + CHUNK).min(frames);
            let context = if start > LEFT_CONTEXT {
                LEFT_CONTEXT
            } else {
                start
            };
            let wav = self.call(&codes.slice(&[0, 0, start - context], &[1, CODE_GROUPS, end])?)?;
            let samples = wav.shape()[2];
            chunks.push(wav.slice(&[0, 0, context * SAMPLES_PER_FRAME], &[1, 1, samples])?);
            start = end;
        }
        let audio = Array::concat(&chunks.iter().collect::<Vec<_>>(), 2)?
            .reshape(&[frames * SAMPLES_PER_FRAME])?;
        audio.cast(MlxDType::F32)?.to_f32_vec()
    }
}
