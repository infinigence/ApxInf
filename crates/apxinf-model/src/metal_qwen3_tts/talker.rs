//! The Talker (28 layers, first codebook) and the code predictor (5 layers,
//! the other 15 codebooks), following mlx-audio with the parity seams.
use super::config::{EPS, HEADS, HEAD_DIM, KV_HEADS, ROPE_THETA};
use super::ops::Ops;
use super::weights::{Attention, Layer, Mlp, Predictor, Talker};
use apxinf_core::Result;
use apxinf_mlx::{Array, MlxDType};

/// mlx_lm `KVCache`: storage grows in steps of 256 positions.
#[derive(Default)]
pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: usize,
}

impl KvCache {
    const STEP: usize = 256;

    pub fn offset(&self) -> usize {
        self.offset
    }

    fn update(&mut self, k: &Array, v: &Array) -> Result<(Array, Array)> {
        let prev = self.offset;
        let n = k.shape()[2];
        let capacity = self.keys.as_ref().map_or(0, |a| a.shape()[2]);
        if self.keys.is_none() || prev + n > capacity {
            let [b, heads, _, dim] = *k.shape() else {
                unreachable!("cached keys are [B, H, L, D]")
            };
            let steps = (Self::STEP + n - 1) / Self::STEP;
            let shape = [b, heads, steps * Self::STEP, dim];
            let grow = |old: &Option<Array>, dtype| -> Result<Array> {
                let fresh = Array::zeros(k.stream(), &shape, dtype)?;
                match old {
                    None => Ok(fresh),
                    Some(old) => {
                        let old = if !prev.is_multiple_of(Self::STEP) {
                            old.slice_axis(2, 0, prev)?
                        } else {
                            old.clone()
                        };
                        Array::concat(&[&old, &fresh], 2)
                    }
                }
            };
            self.keys = Some(grow(&self.keys, k.dtype())?);
            self.values = Some(grow(&self.values, v.dtype())?);
        }
        self.offset += n;
        let start = Array::from_i32(k.stream(), &[1], &[prev as i32])?;
        let keys = self.keys.as_ref().unwrap().slice_update(k, &start, &[2])?;
        let values = self
            .values
            .as_ref()
            .unwrap()
            .slice_update(v, &start, &[2])?;
        let fetched = (
            keys.slice_axis(2, 0, self.offset)?,
            values.slice_axis(2, 0, self.offset)?,
        );
        self.keys = Some(keys);
        self.values = Some(values);
        Ok(fetched)
    }
}

pub fn new_caches(n: usize) -> Vec<KvCache> {
    (0..n).map(|_| KvCache::default()).collect()
}

/// `nn.Embedding`: `weight[ids]`.
pub fn embed(table: &Array, ids: &Array) -> Result<Array> {
    table.take(ids, 0)
}

pub struct Model {
    pub talker: Talker,
    pub predictor: Predictor,
    ops: Ops,
    /// Rotary inverse frequencies `[64]` FP32, computed once like the reference.
    inv_freq: Array,
    scale: f32,
}

fn heads(x: &Array, n: usize) -> Result<Array> {
    let [b, l, _] = *x.shape() else {
        unreachable!("projections are [B, L, H * D]")
    };
    x.reshape(&[b, l, n, HEAD_DIM])
}

impl Model {
    pub fn new(talker: Talker, predictor: Predictor, ops: &Ops) -> Result<Self> {
        let inv_freq = ops.inv_freq(HEAD_DIM, ROPE_THETA)?;
        inv_freq.eval()?;
        Ok(Self {
            talker,
            predictor,
            ops: ops.clone(),
            inv_freq,
            scale: (HEAD_DIM as f64).powf(-0.5) as f32,
        })
    }

    /// `ResizeMLP` with the parity SiLU.
    pub fn text_projection(&self, x: &Array) -> Result<Array> {
        let t = &self.talker;
        t.text_fc2.apply(&self.ops.silu(&t.text_fc1.apply(x)?)?)
    }

    fn mlp(&self, mlp: &Mlp, x: &Array) -> Result<Array> {
        mlp.down
            .apply(&self.ops.swiglu(&mlp.gate.apply(x)?, &mlp.up.apply(x)?)?)
    }

    fn attention(
        &self,
        a: &Attention,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        cache: &mut KvCache,
    ) -> Result<Array> {
        let [b, l, _] = *x.shape() else {
            unreachable!("hidden states are [B, L, C]")
        };
        let (q_norm, k_norm) = a.qk_norm.as_ref().expect("Talker attention has Q/K norms");
        let q = heads(&a.q.apply(x)?, HEADS)?.fast_rms_norm(q_norm, EPS)?;
        let k = heads(&a.k.apply(x)?, KV_HEADS)?.fast_rms_norm(k_norm, EPS)?;
        let v = heads(&a.v.apply(x)?, KV_HEADS)?;
        let order = [0, 2, 1, 3];
        let (q, k) = self
            .ops
            .rope(&q.transpose(&order)?, &k.transpose(&order)?, cos, sin)?;
        let (k, v) = cache.update(&k, &v.transpose(&order)?)?;
        let out = q.sdpa(&k, &v, self.scale, false, mask)?;
        a.o.apply(&out.transpose(&order)?.reshape(&[b, l, HEADS * HEAD_DIM])?)
    }

    fn layer(
        &self,
        layer: &Layer,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        cache: &mut KvCache,
    ) -> Result<Array> {
        let h = x.add(&self.attention(
            &layer.attention,
            &x.fast_rms_norm(&layer.input_norm, EPS)?,
            cos,
            sin,
            mask,
            cache,
        )?)?;
        h.add(&self.mlp(&layer.mlp, &h.fast_rms_norm(&layer.post_norm, EPS)?)?)
    }

    /// Positions `[offset, offset + L)` as FP32 `[1, L]`.
    fn positions(&self, offset: usize, l: usize) -> Result<Array> {
        Array::arange(
            self.ops.stream(),
            offset as f32,
            (offset + l) as f32,
            1.0,
            MlxDType::I32,
        )?
        .reshape(&[1, l])?
        .cast(MlxDType::F32)
    }

    /// Talker forward over `[1, L, 2048]`: logits `[1, L, 3072]` and hidden states.
    pub fn talker(&self, embeds: &Array, cache: &mut [KvCache]) -> Result<(Array, Array)> {
        let l = embeds.shape()[1];
        let pos = self.positions(cache[0].offset(), l)?;
        // Interleaved MRoPE over three position axes. Text-only requests give
        // every axis the same positions, so the interleave selects equal values.
        let half = HEAD_DIM / 2;
        let inv = self
            .inv_freq
            .reshape(&[1, 1, half, 1])?
            .broadcast_to(&[3, 1, half, 1])?;
        let freqs = inv
            .matmul(&pos.reshape(&[1, 1, 1, l])?.broadcast_to(&[3, 1, 1, l])?)?
            .transpose(&[0, 1, 3, 2])?
            .slice_axis(0, 0, 1)?
            .reshape(&[1, l, half])?;
        let emb = Array::concat(&[&freqs, &freqs], 2)?;
        let cos = emb.cos()?.cast(embeds.dtype())?;
        let sin = emb.sin()?.cast(embeds.dtype())?;
        let mask = (l > 1)
            .then(|| self.ops.causal_mask(l, embeds.dtype()))
            .transpose()?;
        let mut x = embeds.clone();
        for (layer, cache) in self.talker.layers.iter().zip(cache.iter_mut()) {
            x = self.layer(layer, &x, &cos, &sin, mask.as_ref(), cache)?;
        }
        let hidden = x.fast_rms_norm(&self.talker.norm, EPS)?;
        Ok((self.talker.codec_head.apply(&hidden)?, hidden))
    }

    /// One code predictor step over `[1, L, 2048]` with head `group`: logits `[1, L, 2048]`.
    pub fn predictor(&self, inputs: &Array, group: usize, cache: &mut [KvCache]) -> Result<Array> {
        let p = &self.predictor;
        // FP32 product and bias, then one cast (ReferenceBiasedProjection).
        let x = p
            .to_mtp_bias
            .addmm(
                &inputs.cast(MlxDType::F32)?,
                &p.to_mtp_weight.transpose(&[1, 0])?,
                1.0,
                1.0,
            )?
            .cast(inputs.dtype())?;
        let l = x.shape()[1];
        let half = HEAD_DIM / 2;
        let freqs = self
            .inv_freq
            .reshape(&[1, half, 1])?
            .mul(&self.positions(cache[0].offset(), l)?.reshape(&[1, 1, l])?)?
            .transpose(&[0, 2, 1])?;
        let emb = Array::concat(&[&freqs, &freqs], 2)?;
        let cos = emb.cos()?.cast(x.dtype())?;
        let sin = emb.sin()?.cast(x.dtype())?;
        let mask = (l > 1)
            .then(|| self.ops.causal_mask(l, x.dtype()))
            .transpose()?;
        let mut x = x;
        for (layer, cache) in p.layers.iter().zip(cache.iter_mut()) {
            x = self.layer(layer, &x, &cos, &sin, mask.as_ref(), cache)?;
        }
        p.heads[group].apply(&x.fast_rms_norm(&p.norm, EPS)?)
    }
}
