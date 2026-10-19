//! Arithmetic shared by the Talker, code predictor and waveform decoder, in
//! the reference's operation order. Functions the reference wraps in
//! `mx.compile` run through `Compiled` so MLX builds the same fused graphs.
use super::config::{HEADS, HEAD_DIM, KV_HEADS};
use apxinf_core::Result;
use apxinf_mlx::{Array, Compiled, MlxDType, Stream};
use std::rc::Rc;

#[derive(Clone)]
pub struct Ops {
    stream: Stream,
    gelu: Rc<Compiled>,
    rope: Rc<Compiled>,
}

#[cfg(test)]
mod tests {
    use super::super::config::{HEADS, HEAD_DIM, KV_HEADS};
    use super::*;

    fn check_prepared_profiles(stream: &Stream) -> Result<()> {
        let ops = Ops::new(&stream)?;
        for length in [1, 2, 17] {
            if length == 17 {
                ops.prepare_rope(length)?;
            }
            let query = Array::from_f32(
                stream,
                &[1, HEADS, length, HEAD_DIM],
                &vec![0.25; HEADS * length * HEAD_DIM],
            )?
            .cast(MlxDType::BF16)?;
            let key = Array::from_f32(
                stream,
                &[1, KV_HEADS, length, HEAD_DIM],
                &vec![-0.5; KV_HEADS * length * HEAD_DIM],
            )?
            .cast(MlxDType::BF16)?;
            let angle = Array::from_f32(
                &stream,
                &[1, length, HEAD_DIM],
                &vec![0.75; length * HEAD_DIM],
            )?
            .cast(MlxDType::BF16)?;
            stream.eval(&[query.clone(), key.clone(), angle.clone()])?;
            let before = stream.counters().trace_callbacks;
            let (actual_query, actual_key) = ops.rope(&query, &key, &angle, &angle)?;
            stream.eval(&[actual_query.clone(), actual_key.clone()])?;
            assert_eq!(
                stream.counters().trace_callbacks,
                before,
                "rotary length {length}"
            );
            let cold = Compiled::new(stream, 2, |inputs| {
                let (query, key) = rotate(&inputs[0], &inputs[1], &inputs[2], &inputs[3])?;
                Ok(vec![query, key])
            })?;
            let expected = cold.call_and_eval(&[query, key, angle.clone(), angle])?;
            assert_eq!(actual_query.to_bytes()?, expected[0].to_bytes()?);
            assert_eq!(actual_key.to_bytes()?, expected[1].to_bytes()?);
        }
        let values = Array::from_f32(&stream, &[1, 17, 4096], &vec![0.75; 17 * 4096])?
            .cast(MlxDType::BF16)?;
        values.eval()?;
        let before = stream.counters().trace_callbacks;
        ops.gelu(&values)?.eval()?;
        assert_eq!(stream.counters().trace_callbacks, before);
        Ok(())
    }

    #[test]
    fn fixed_rotary_and_gelu_profiles_are_prepared_at_construction() -> Result<()> {
        check_prepared_profiles(&Stream::cpu()?)
    }

    #[test]
    #[ignore = "requires native Apple Metal"]
    fn rotary_and_gelu_preparation_matches_cold_metal() -> Result<()> {
        check_prepared_profiles(&Stream::metal(0)?)
    }
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
        drop(gelu.prepare(&[Array::zeros(stream, &[1, 2, 2], MlxDType::BF16)?])?);
        let ops = Self {
            stream: stream.clone(),
            gelu: Rc::new(gelu),
            rope: Rc::new(rope),
        };
        for length in [1, 2] {
            ops.prepare_rope(length)?;
        }
        Ok(ops)
    }

    pub fn prepare_rope(&self, length: usize) -> Result<()> {
        let query = Array::zeros(&self.stream, &[1, HEADS, length, HEAD_DIM], MlxDType::BF16)?;
        let key = Array::zeros(
            &self.stream,
            &[1, KV_HEADS, length, HEAD_DIM],
            MlxDType::BF16,
        )?;
        let table = Array::zeros(&self.stream, &[1, length, HEAD_DIM], MlxDType::BF16)?;
        drop(self.rope.prepare(&[query, key, table.clone(), table])?);
        Ok(())
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
