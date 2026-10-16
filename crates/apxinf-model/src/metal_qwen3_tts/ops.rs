//! Arithmetic shared by the Talker, code predictor and waveform decoder, in
//! the reference's operation order. Functions the reference wraps in
//! `mx.compile` run through `Compiled` so MLX builds the same fused graphs.
use apxinf_core::Result;
use apxinf_mlx::{Array, Compiled, MlxDType, Stream};
use std::rc::Rc;

#[derive(Clone)]
pub struct Ops {
    stream: Stream,
    gelu: Rc<Compiled>,
    rope: Rc<Compiled>,
}

/// `rotate_half` for the last axis.
fn rotate_half(x: &Array) -> Result<Array> {
    let rank = x.shape().len();
    let half = x.shape()[rank - 1] / 2;
    let x1 = x.slice_axis(rank - 1, 0, half)?;
    let x2 = x.slice_axis(rank - 1, half, 2 * half)?;
    Array::concat(&[&x2.neg()?, &x1], rank - 1)
}

/// `q * cos + rotate_half(q) * sin` with `cos`, `sin` `[B, L, D]` broadcast over heads.
fn rotate(q: &Array, k: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
    let [b, l, d] = *cos.shape() else {
        unreachable!("rotary tables are [B, L, D]")
    };
    let cos = cos.reshape(&[b, 1, l, d])?;
    let sin = sin.reshape(&[b, 1, l, d])?;
    let apply = |x: &Array| x.mul(&cos)?.add(&rotate_half(x)?.mul(&sin)?);
    Ok((apply(q)?, apply(k)?))
}

impl Ops {
    pub fn new(stream: &Stream) -> Result<Self> {
        // nn.gelu: x * (1 + erf(x / sqrt(2))) / 2, compiled shapeless.
        let s = stream.clone();
        let gelu = Compiled::with_shapeless(stream, 1, true, move |a| {
            let x = &a[0];
            let c = |v: f64| Array::scalar(&s, v as f32, x.dtype());
            let inner = c(1.0)?.add(&x.div(&c(std::f64::consts::SQRT_2)?)?.erf()?)?;
            Ok(vec![x.mul(&inner)?.div(&c(2.0)?)?])
        })?;
        // mlx-audio apply_rotary_pos_emb / apply_multimodal_rotary_pos_emb.
        let rope = Compiled::new(stream, 2, |a| {
            let (q, k) = rotate(&a[0], &a[1], &a[2], &a[3])?;
            Ok(vec![q, k])
        })?;
        Ok(Self {
            stream: stream.clone(),
            gelu: Rc::new(gelu),
            rope: Rc::new(rope),
        })
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// A Python float operand, converted to `dtype` as MLX converts it.
    pub fn scalar(&self, value: f64, dtype: MlxDType) -> Result<Array> {
        Array::scalar(&self.stream, value as f32, dtype)
    }

    pub fn gelu(&self, x: &Array) -> Result<Array> {
        Ok(self.gelu.call(std::slice::from_ref(x))?.remove(0))
    }

    /// The compiled RoPE of the Talker and code predictor.
    pub fn rope(&self, q: &Array, k: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
        let mut out = self
            .rope
            .call(&[q.clone(), k.clone(), cos.clone(), sin.clone()])?;
        let k = out.pop().unwrap();
        Ok((out.pop().unwrap(), k))
    }

    /// The waveform decoder's uncompiled RoPE.
    pub fn rope_eager(
        &self,
        q: &Array,
        k: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<(Array, Array)> {
        rotate(q, k, cos, sin)
    }

    /// The parity SiLU: FP32 `x / (1 + exp(-x))`, then one cast back.
    pub fn silu(&self, x: &Array) -> Result<Array> {
        let work = x.cast(MlxDType::F32)?;
        let one = self.scalar(1.0, MlxDType::F32)?;
        work.div(&one.add(&work.neg()?.exp()?)?)?.cast(x.dtype())
    }

    /// The parity SwiGLU: FP32 SiLU of the gate, rounded, times `up`.
    pub fn swiglu(&self, gate: &Array, up: &Array) -> Result<Array> {
        self.silu(gate)?.mul(up)
    }

    /// `nn.MultiHeadAttention.create_additive_causal_mask(n)` cast to `dtype`:
    /// `(i < j) * finfo(float32).min`.
    pub fn causal_mask(&self, n: usize, dtype: MlxDType) -> Result<Array> {
        let values: Vec<f32> = (0..n * n)
            .map(|i| if i / n < i % n { f32::MIN } else { -0.0 })
            .collect();
        Array::from_f32(&self.stream, &[n, n], &values)?.cast(dtype)
    }

    /// `1 / (base ** (arange(0, dim, 2, float32) / dim))`, on the device.
    pub fn inv_freq(&self, dim: usize, base: f32) -> Result<Array> {
        let exponent = Array::arange(&self.stream, 0.0, dim as f32, 2.0, MlxDType::F32)?
            .div(&self.scalar(dim as f64, MlxDType::F32)?)?;
        let one = self.scalar(1.0, MlxDType::F32)?;
        one.div(&Array::scalar(&self.stream, base, MlxDType::F32)?.pow(&exponent)?)
    }
}
