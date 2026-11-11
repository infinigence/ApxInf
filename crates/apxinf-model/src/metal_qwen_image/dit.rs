//! The single-stream DiT (mflux `Qwen21Transformer`): the text prefix runs
//! once per prompt and caches each block's post-RoPE K/V; every denoise step
//! then runs one compiled image-token forward against that cache.
use super::ops::Ops;
use super::weights::{Block, GateUp, Transformer, DIM, HEADS, HEAD_DIM, Q4};
use apxinf_core::{Error, Result};
use apxinf_mlx::fusions::{
    GatedResidualLayerNormModulate, InterleavedQkNormRope, LayerNormModulate, MppBf16GemmNt,
    MppGemmDecodeQ4Rows, MppGemmGatedResidual, MppGemmHeadRmsRope, MppGemmHeadsWithPrefix,
    MppGemmSwigluPairs,
};
use apxinf_mlx::{Array, Compiled, MlxDType, Stream};
use std::rc::Rc;

const EPS: f32 = 1e-6;
/// Smallest image-token count that uses the MPP GEMM. MLX quantized matmul
/// switches to another reduction for small output grids (observed on M4 at
/// 64 tokens), where MPP no longer reproduces it.
pub const MPP_MIN_TOKENS: usize = 1024;
/// Image-token counts that run the fused blocks: multiples of 64 within the
/// range their kernels admit, from the smallest qualified image (512²).
pub const FUSED_TOKENS: std::ops::RangeInclusive<usize> = 1024..=4096;

/// The image-token block as seven fused kernels (see the family README):
/// norm + modulation, Q/K with head RMSNorm and RoPE, V in the head-major
/// text + image layout, SDPA, the output projection decoding the packed MLP
/// gate/up weight, gated residual + norm + modulation, SwiGLU, and the MLP down
/// projection with its gated residual.
pub struct Fused {
    norm: LayerNormModulate,
    residual_norm: GatedResidualLayerNormModulate,
    qk: MppGemmHeadRmsRope,
    value: MppGemmHeadsWithPrefix,
    output: MppGemmDecodeQ4Rows,
    swiglu: MppGemmSwigluPairs,
    down: MppGemmGatedResidual,
}

impl Fused {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            norm: LayerNormModulate::new(stream)?,
            residual_norm: GatedResidualLayerNormModulate::new(stream)?,
            qk: MppGemmHeadRmsRope::new(stream)?,
            value: MppGemmHeadsWithPrefix::new(stream)?,
            output: MppGemmDecodeQ4Rows::new(stream)?,
            swiglu: MppGemmSwigluPairs::new(stream)?,
            down: MppGemmGatedResidual::new(stream)?,
        })
    }
}

/// The MLP gate and up Q4 weights `[MLP, DIM / 8]` interleaved in 32-row
/// pairs, `[2 MLP, DIM / 8]` (scales and biases alike), as the fused SwiGLU
/// reads them once decoded. Rows `64 j .. 64 j + 31` are gate rows `32 j ..`,
/// the next 32 the matching up rows.
pub fn interleave_gate_up(gate: &Q4, up: &Q4) -> Result<Q4> {
    let pair = |g: &Array, u: &Array| -> Result<Array> {
        let [rows, columns] = *g.shape() else {
            return Err(Error::Contract("gate/up weights must be 2-D"));
        };
        if u.shape() != g.shape() || u.dtype() != g.dtype() || rows % 32 != 0 {
            return Err(Error::Contract(
                "gate and up weights must match in rows that are a multiple of 32",
            ));
        }
        let split = |a: &Array| a.reshape(&[rows / 32, 1, 32, columns]);
        Array::concat(&[&split(g)?, &split(u)?], 1)?.reshape(&[2 * rows, columns])
    };
    Ok(Q4 {
        weight: pair(&gate.weight, &up.weight)?,
        scales: pair(&gate.scales, &up.scales)?,
        biases: pair(&gate.biases, &up.biases)?,
    })
}

pub struct Dit {
    w: Transformer,
    qk: InterleavedQkNormRope,
    /// `Qwen21Timesteps._freqs`, `[1, 128]` FP32.
    freqs: Array,
    ops: Ops,
    /// The image-token GEMM, when it reproduces MLX quantized matmul here.
    gemm: Option<Rc<MppBf16GemmNt>>,
    /// The fused image blocks, when selected for this architecture.
    fused: Option<Rc<Fused>>,
}

/// Per-block post-RoPE text K/V, `[1, 32, T, 128]` BF16 each.
pub struct TextCache {
    kv: Vec<(Array, Array)>,
}

impl Dit {
    pub fn new(
        w: Transformer,
        ops: &Ops,
        gemm: Option<Rc<MppBf16GemmNt>>,
        fused: Option<Rc<Fused>>,
    ) -> Result<Self> {
        let stream = ops.stream();
        let freqs = super::tables::timestep_frequencies(128);
        Ok(Self {
            w,
            qk: InterleavedQkNormRope::new(stream)?,
            freqs: Array::from_f32(stream, &[1, 128], &freqs)?,
            ops: ops.clone(),
            gemm,
            fused,
        })
    }

    /// Whether `tokens` image tokens run the fused blocks.
    pub fn fuses(&self, tokens: usize) -> bool {
        self.fused.is_some() && tokens.is_multiple_of(64) && FUSED_TOKENS.contains(&tokens)
    }

    /// Interleave every block's gate/up weights for the fused blocks and drop
    /// the originals, one block at a time, after the text prefix (their only
    /// other reader) has run. Requests that do not fuse keep them separate.
    pub fn interleave_for_image(&mut self, tokens: usize) -> Result<()> {
        if !self.fuses(tokens) {
            return Ok(());
        }
        let stream = self.ops.stream().clone();
        for b in &mut self.w.blocks {
            if let GateUp::Separate { gate, up } = &b.gate_up {
                let packed = interleave_gate_up(gate, up)?;
                stream.eval(&[
                    packed.weight.clone(),
                    packed.scales.clone(),
                    packed.biases.clone(),
                ])?;
                b.gate_up = GateUp::Interleaved(packed);
            }
        }
        Ok(())
    }

    /// `nn.QuantizedLinear`. At least `MPP_MIN_TOKENS` image tokens in a
    /// multiple of 64 decode the weight per call and use the probe-selected
    /// MPP GEMM. Full-image parity requires separate qualification for the
    /// selected device and profile.
    fn project(&self, w: &Q4, x: &Array, image: bool) -> Result<Array> {
        let tokens = x.shape()[1];
        match &self.gemm {
            Some(gemm) if image && tokens.is_multiple_of(64) && tokens >= MPP_MIN_TOKENS => {
                gemm.call(x, &w.weight.dequantize(&w.scales, &w.biases, 64, 4)?)
            }
            _ => w.apply(x),
        }
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
        image: bool,
    ) -> Result<(Array, Array, Array)> {
        let tokens = x.shape()[1];
        let (q, k) = self.qk.call(
            &self.project(&b.to_q, x, image)?,
            &self.project(&b.to_k, x, image)?,
            &b.norm_q,
            &b.norm_k,
            cos,
            sin,
            HEADS,
        )?;
        let v = self
            .project(&b.to_v, x, image)?
            .reshape(&[1, tokens, HEADS, HEAD_DIM])?
            .transpose(&[0, 2, 1, 3])?;
        Ok((q, k, v))
    }

    fn attention_output(&self, b: &Block, o: &Array, image: bool) -> Result<Array> {
        let tokens = o.shape()[2];
        let o = o.transpose(&[0, 2, 1, 3])?.reshape(&[1, tokens, DIM])?;
        self.project(&b.to_out, &o, image)
    }

    fn mlp(&self, b: &Block, x: &Array, image: bool) -> Result<Array> {
        let GateUp::Separate { gate, up } = &b.gate_up else {
            return Err(Error::Contract(
                "interleaved gate/up weights run only in the fused image blocks",
            ));
        };
        let gated = self.ops.silu(&self.project(gate, x, image)?)?;
        let gated = gated.mul(&self.project(up, x, image)?)?;
        self.project(&b.out, &gated, image)
    }

    /// The image-token block on the fused kernels. `mod1`/`mod2` are the
    /// broadcast `[1, T, 2 DIM]` `[scale | gate]` rows, read without copies.
    #[allow(clippy::too_many_arguments)]
    fn fused_block(
        &self,
        f: &Fused,
        gate_up: &Q4,
        b: &Block,
        h: &Array,
        (key_text, value_text): (&Array, &Array),
        mod1: &Array,
        mod2: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        let tokens = h.shape()[1];
        let dense = |w: &Q4| w.weight.dequantize(&w.scales, &w.biases, 64, 4);
        let x = h.contiguous()?;
        let attn_input = f.norm.call(&x, mod1)?;
        let q = f.qk.call(
            &attn_input,
            &dense(&b.to_q)?,
            &b.norm_q,
            cos,
            sin,
            key_text,
            false,
        )?;
        let k = f.qk.call(
            &attn_input,
            &dense(&b.to_k)?,
            &b.norm_k,
            cos,
            sin,
            key_text,
            true,
        )?;
        let v = f.value.call(&attn_input, &dense(&b.to_v)?, value_text)?;
        let scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let o = q
            .sdpa(&k, &v, scale, false, None)?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, tokens, DIM])?;
        let (attention, gate_up) = f.output.call(
            &o,
            &dense(&b.to_out)?,
            &gate_up.weight,
            &gate_up.scales,
            &gate_up.biases,
        )?;
        let (x, mlp_input) = f.residual_norm.call(&x, &attention, mod1, mod2)?;
        let gated = f.swiglu.call(&mlp_input, &gate_up)?;
        f.down.call(&gated, &dense(&b.out)?, &x, mod2)
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
        let tokens = h.shape()[1];
        if let (Some(f), Some(text), GateUp::Interleaved(gate_up)) =
            (&self.fused, text_kv, &b.gate_up)
        {
            if self.fuses(tokens) {
                let h = self.fused_block(f, gate_up, b, h, text, mod1, mod2, cos, sin)?;
                return Ok((h, None));
            }
        }
        let half = |m: &Array| -> Result<(Array, Array)> {
            Ok((m.slice_axis(2, 0, DIM)?, m.slice_axis(2, DIM, 2 * DIM)?))
        };
        let (scale1, gate1) = half(mod1)?;
        let (scale2, gate2) = half(mod2)?;
        let x = self.modulate(h, &scale1)?;
        let image = text_kv.is_some();
        let (q, k, v) = self.attention_inputs(b, &x, cos, sin, image)?;
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
        let o = self.attention_output(b, &o, image)?;
        let h = h.add(&gate1.tanh()?.mul(&o)?)?;
        let y = self.mlp(b, &self.modulate(&h, &scale2)?, image)?;
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
}

impl Step {
    pub fn prepare(dit: Dit, inputs: &[Array]) -> Result<(Self, Array)> {
        let stream = dit.ops.stream().clone();
        let compiled = Compiled::new(&stream, 1, move |inputs| {
            Ok(vec![dit.image_forward(inputs)?])
        })?;
        let first = compiled.prepare(inputs)?.remove(0);
        Ok((Self { compiled }, first))
    }

    /// Velocity prediction `[1, N, 64]` BF16 for one timestep.
    pub fn inputs(
        latents: &Array,
        timestep: &Array,
        cos: &Array,
        sin: &Array,
        cache: &TextCache,
    ) -> Vec<Array> {
        let mut inputs = vec![latents.clone(), timestep.clone(), cos.clone(), sin.clone()];
        inputs.extend(cache.kv.iter().flat_map(|(k, v)| [k.clone(), v.clone()]));
        inputs
    }

    pub fn call(&self, inputs: &[Array]) -> Result<Array> {
        Ok(self.compiled.call(inputs)?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_up_rows_interleave_in_32_row_pairs() {
        let s = Stream::cpu().unwrap();
        let rows = |offset: i32, n: usize, columns: usize| {
            let values: Vec<i32> = (0..(n * columns) as i32).map(|v| offset + v).collect();
            Array::from_i32(&s, &[n, columns], &values).unwrap()
        };
        let q4 = |offset: i32| Q4 {
            weight: rows(offset, 64, 2),
            scales: rows(offset + 1_000, 64, 1),
            biases: rows(offset + 2_000, 64, 1),
        };
        let (gate, up) = (q4(0), q4(10_000));
        let packed = interleave_gate_up(&gate, &up).unwrap();
        assert_eq!(packed.weight.shape(), [128, 2]);
        let bytes = |a: &Array| -> Vec<i32> {
            a.to_bytes()
                .unwrap()
                .chunks(4)
                .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
                .collect()
        };
        let (g, u, p) = (
            bytes(&gate.weight),
            bytes(&up.weight),
            bytes(&packed.weight),
        );
        for row in 0..128 {
            let (pair, half, within) = (row / 64, (row / 32) % 2, row % 32);
            let source = if half == 0 { &g } else { &u };
            let original = (32 * pair + within) * 2;
            assert_eq!(
                p[row * 2..row * 2 + 2],
                source[original..original + 2],
                "row {row}"
            );
        }
        assert_eq!(bytes(&packed.scales)[32], bytes(&up.scales)[0]);
        assert_eq!(bytes(&packed.biases)[64], bytes(&gate.biases)[32]);
        let odd = Q4 {
            weight: rows(0, 48, 2),
            scales: rows(0, 48, 1),
            biases: rows(0, 48, 1),
        };
        assert!(interleave_gate_up(&odd, &odd).is_err());
    }
}
