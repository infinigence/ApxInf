//! The single-stream DiT (mflux `Qwen21Transformer`): the text prefix runs
//! once per prompt and caches each block's post-RoPE K/V; every denoise step
//! then runs one compiled image-token forward against that cache.
use super::ops::Ops;
use super::weights::{Block, Transformer, DIM, HEADS, HEAD_DIM};
use apxinf_core::Result;
use apxinf_mlx::fusions::InterleavedQkNormRope;
use apxinf_mlx::{Array, Compiled, MlxDType};
use std::rc::Rc;

const EPS: f32 = 1e-6;

pub struct Dit {
    w: Transformer,
    qk: InterleavedQkNormRope,
    /// `Qwen21Timesteps._freqs`, `[1, 128]` FP32.
    freqs: Array,
    ops: Ops,
}

/// Per-block post-RoPE text K/V, `[1, 32, T, 128]` BF16 each.
pub struct TextCache {
    kv: Vec<(Array, Array)>,
}

impl Dit {
    pub fn new(w: Transformer, ops: &Ops) -> Result<Self> {
        let stream = ops.stream();
        let freqs = super::tables::timestep_frequencies(128);
        Ok(Self {
            w,
            qk: InterleavedQkNormRope::new(stream)?,
            freqs: Array::from_f32(stream, &[1, 128], &freqs)?,
            ops: ops.clone(),
        })
    }

    /// `Qwen21TimeTextEmbed` for FP32 timestep rows `[2]`: BF16 `[2, 4096]`.
    fn time_embedding(&self, rows: &Array) -> Result<Array> {
        let args = rows
            .reshape(&[2, 1])?
            .mul(&self.ops.f32(1000.0)?)?
            .mul(&self.freqs)?;
        let projected = Array::concat(&[&args.cos()?, &args.sin()?], 1)?.cast(MlxDType::BF16)?;
        let hidden = self.ops.silu(&self.w.time_1.apply(&projected)?)?;
        self.w.time_2.apply(&hidden)
    }

    /// Shared modulation split into `(mod1, mod2)` rows `[2, 8192]`.
    fn modulation(&self, temb: &Array) -> Result<(Array, Array)> {
        let m = self.w.modulation.apply(&self.ops.silu(temb)?)?;
        Ok((
            m.slice_axis(1, 0, 2 * DIM)?,
            m.slice_axis(1, 2 * DIM, 4 * DIM)?,
        ))
    }

    fn modulate(&self, h: &Array, scale: &Array) -> Result<Array> {
        let one = self.ops.scalar(1.0, scale.dtype())?;
        h.fast_layer_norm(None, None, EPS)?.mul(&one.add(scale)?)
    }

    fn txt_in(&self, prompt: &Array) -> Result<Array> {
        // Qwen21ZeroCenterRMSNorm: x * rrms * (weight + 1) in FP32.
        let x = prompt.cast(MlxDType::F32)?;
        let rrms = x
            .mul(&x)?
            .mean(&[-1], true)?
            .add(&self.ops.f32(EPS)?)?
            .rsqrt()?;
        let scale = self
            .w
            .txt_norm
            .cast(MlxDType::F32)?
            .add(&self.ops.f32(1.0)?)?;
        let normed = x.mul(&rrms)?.mul(&scale)?.cast(prompt.dtype())?;
        let hidden = self.ops.gelu_approx(&self.w.txt_in.apply(&normed)?)?;
        self.w.txt_out.apply(&hidden)
    }

    fn attention_inputs(
        &self,
        b: &Block,
        x: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<(Array, Array, Array)> {
        let tokens = x.shape()[1];
        let (q, k) = self.qk.call(
            &b.to_q.apply(x)?,
            &b.to_k.apply(x)?,
            &b.norm_q,
            &b.norm_k,
            cos,
            sin,
            HEADS,
        )?;
        let v = b
            .to_v
            .apply(x)?
            .reshape(&[1, tokens, HEADS, HEAD_DIM])?
            .transpose(&[0, 2, 1, 3])?;
        Ok((q, k, v))
    }

    fn attention_output(&self, b: &Block, o: &Array) -> Result<Array> {
        let tokens = o.shape()[2];
        let o = o.transpose(&[0, 2, 1, 3])?.reshape(&[1, tokens, DIM])?;
        b.to_out.apply(&o)
    }

    fn mlp(&self, b: &Block, x: &Array) -> Result<Array> {
        let gated = self.ops.silu(&b.gate.apply(x)?)?.mul(&b.proj.apply(x)?)?;
        b.out.apply(&gated)
    }

    /// One block over the image tokens (`text_kv = Some`) or the causal text
    /// prefix (`None`, also returning its K/V).
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        b: &Block,
        h: &Array,
        text_kv: Option<(&Array, &Array)>,
        mod1: &Array,
        mod2: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<(Array, Option<(Array, Array)>)> {
        let half = |m: &Array| -> Result<(Array, Array)> {
            Ok((m.slice_axis(2, 0, DIM)?, m.slice_axis(2, DIM, 2 * DIM)?))
        };
        let (scale1, gate1) = half(mod1)?;
        let (scale2, gate2) = half(mod2)?;
        let x = self.modulate(h, &scale1)?;
        let (q, k, v) = self.attention_inputs(b, &x, cos, sin)?;
        let scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let (o, cached) = match text_kv {
            Some((key_text, value_text)) => {
                let k = Array::concat(&[key_text, &k], 2)?;
                let v = Array::concat(&[value_text, &v], 2)?;
                (q.sdpa(&k, &v, scale, false, None)?, None)
            }
            None => {
                let o = q.sdpa(&k, &v, scale, true, None)?;
                (o, Some((k.contiguous()?, v.contiguous()?)))
            }
        };
        let o = self.attention_output(b, &o)?;
        let h = h.add(&gate1.tanh()?.mul(&o)?)?;
        let y = self.mlp(b, &self.modulate(&h, &scale2)?)?;
        Ok((h.add(&gate2.tanh()?.mul(&y)?)?, cached))
    }

    /// Row `row` of `[2, C]` modulation, broadcast over `tokens`.
    fn row(m: &Array, row: usize, tokens: usize) -> Result<Array> {
        let width = m.shape()[1];
        m.slice_axis(0, row, row + 1)?
            .reshape(&[1, 1, width])?
            .broadcast_to(&[1, tokens, width])
    }

    /// The text prefix with the t = 0 modulation row; FP32 RoPE tables `[T, 64]`.
    pub fn text_cache(&self, prompt: &Array, cos: &Array, sin: &Array) -> Result<TextCache> {
        let stream = self.ops.stream();
        let tokens = prompt.shape()[1];
        let temb = self.time_embedding(&Array::zeros(stream, &[2], MlxDType::F32)?)?;
        let (mod1, mod2) = self.modulation(&temb)?;
        let (mod1, mod2) = (Self::row(&mod1, 1, tokens)?, Self::row(&mod2, 1, tokens)?);
        let mut h = self.txt_in(prompt)?;
        let mut kv = Vec::with_capacity(self.w.blocks.len());
        for b in &self.w.blocks {
            let (next, cached) = self.block(b, &h, None, &mod1, &mod2, cos, sin)?;
            h = next;
            kv.push(cached.expect("the text prefix returns its K/V"));
        }
        let mut arrays = vec![h];
        arrays.extend(kv.iter().flat_map(|(k, v)| [k.clone(), v.clone()]));
        stream.eval(&arrays)?;
        Ok(TextCache { kv })
    }

    /// Image-token forward: latents `[1, N, 64]` BF16, timestep `[1]` FP32,
    /// FP32 RoPE tables `[N, 64]`, then each block's text K/V.
    fn image_forward(&self, inputs: &[Array]) -> Result<Array> {
        let stream = self.ops.stream();
        let (latents, timestep, cos, sin) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
        let tokens = latents.shape()[1];
        let rows = Array::concat(&[timestep, &Array::zeros(stream, &[1], MlxDType::F32)?], 0)?;
        let temb = self.time_embedding(&rows)?;
        let (mod1, mod2) = self.modulation(&temb)?;
        let (mod1, mod2) = (Self::row(&mod1, 0, tokens)?, Self::row(&mod2, 0, tokens)?);
        let mut h = self.w.img_in.apply(latents)?;
        for (b, kv) in self.w.blocks.iter().zip(inputs[4..].chunks(2)) {
            let text = Some((&kv[0], &kv[1]));
            h = self.block(b, &h, text, &mod1, &mod2, cos, sin)?.0;
        }
        let scale = self.w.norm_out.apply(&self.ops.silu(&temb)?)?;
        let scale = Self::row(&scale, 0, tokens)?;
        self.w.proj_out.apply(&self.modulate(&h, &scale)?)
    }
}

/// The compiled denoise step of one request; it owns the transformer.
pub struct Step {
    compiled: Compiled,
    dit: Rc<Dit>,
}

impl Step {
    pub fn new(dit: Dit) -> Result<Self> {
        let dit = Rc::new(dit);
        let traced = dit.clone();
        let compiled = Compiled::new(dit.ops.stream(), 1, move |inputs| {
            Ok(vec![traced.image_forward(inputs)?])
        })?;
        Ok(Self { compiled, dit })
    }

    pub fn dit(&self) -> &Dit {
        &self.dit
    }

    /// Velocity prediction `[1, N, 64]` BF16 for one timestep.
    pub fn call(
        &self,
        latents: &Array,
        timestep: &Array,
        cos: &Array,
        sin: &Array,
        cache: &TextCache,
    ) -> Result<Array> {
        let mut inputs = vec![latents.clone(), timestep.clone(), cos.clone(), sin.clone()];
        inputs.extend(cache.kv.iter().flat_map(|(k, v)| [k.clone(), v.clone()]));
        Ok(self.compiled.call(&inputs)?.remove(0))
    }
}
