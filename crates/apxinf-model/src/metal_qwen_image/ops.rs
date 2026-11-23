//! Element-wise helpers that MLX `nn` defines as shapeless compiled functions.
//! Calling them through `Compiled` keeps the reference's fused graphs.
use apxinf_core::Result;
use apxinf_mlx::{Array, Compiled, MlxDType, Stream};
use std::rc::Rc;

#[derive(Clone)]
pub struct Ops {
    stream: Stream,
    silu: Rc<Compiled>,
    gelu_approx: Rc<Compiled>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared_activations(stream: &Stream) -> Result<()> {
        let ops = Ops::new(stream)?;
        for (dtype, shape) in [
            (MlxDType::BF16, vec![2, 4096]),
            (MlxDType::BF16, vec![1, 1, 12288]),
            (MlxDType::BF16, vec![1, 17, 12288]),
            (MlxDType::F32, vec![1, 96, 1, 1]),
            (MlxDType::F32, vec![1, 96, 1, 17]),
            (MlxDType::F32, vec![1, 96, 17, 1]),
            (MlxDType::F32, vec![1, 96, 17, 19]),
        ] {
            let count = shape.iter().product();
            let values: Vec<f32> = (0..count)
                .map(|index| ((index % 31) as f32 - 15.0) / 4.0)
                .collect();
            let input = Array::from_f32(stream, &shape, &values)?.cast(dtype)?;
            input.eval()?;
            let before = stream.counters();
            let actual = ops.silu(&input)?;
            actual.eval()?;
            let after = stream.counters();
            assert_eq!(after.trace_callbacks, before.trace_callbacks, "{shape:?}");
            assert_eq!(after.uploads_bytes, before.uploads_bytes);
            assert_eq!(after.downloads_bytes, before.downloads_bytes);
            let cold = Compiled::with_shapeless(stream, 1, true, |args| {
                Ok(vec![args[0].mul(&args[0].sigmoid()?)?])
            })?;
            assert_eq!(
                actual.to_bytes()?,
                cold.call_and_eval(&[input])?[0].to_bytes()?
            );
        }
        for length in [1, 17] {
            let input = Array::from_f32(stream, &[1, length, 4], &vec![0.75; length * 4])?
                .cast(MlxDType::BF16)?;
            input.eval()?;
            let before = stream.counters();
            let output = ops.gelu_approx(&input)?;
            output.eval()?;
            let after = stream.counters();
            assert_eq!(after.trace_callbacks, before.trace_callbacks);
            assert_eq!(after.uploads_bytes, before.uploads_bytes);
            assert_eq!(after.downloads_bytes, before.downloads_bytes);
            let retained = output.to_bytes()?;
            let changed = input.neg()?;
            assert_ne!(ops.gelu_approx(&changed)?.to_bytes()?, retained);
            assert_eq!(output.to_bytes()?, retained);
        }
        Ok(())
    }

    #[test]
    fn activations_are_prepared_before_first_cpu_use() -> Result<()> {
        prepared_activations(&Stream::cpu()?)
    }

    #[test]
    #[ignore = "requires native Apple Metal"]
    fn activations_are_prepared_before_first_metal_use() -> Result<()> {
        prepared_activations(&Stream::metal(0)?)
    }
}

impl Ops {
    pub fn new(stream: &Stream) -> Result<Self> {
        // nn.silu: x * sigmoid(x)
        let silu =
            Compiled::with_shapeless(stream, 1, true, |a| Ok(vec![a[0].mul(&a[0].sigmoid()?)?]))?;
        // nn.gelu_approx: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3)))
        let gelu_stream = stream.clone();
        let gelu_approx = Compiled::with_shapeless(stream, 1, true, move |a| {
            let x = &a[0];
            let c = |v: f64| Array::scalar(&gelu_stream, v as f32, x.dtype());
            let cubic = c(0.044715)?.mul(&x.pow(&c(3.0)?)?)?;
            let inner = c((2.0 / std::f64::consts::PI).sqrt())?.mul(&x.add(&cubic)?)?;
            let half = c(0.5)?.mul(x)?;
            Ok(vec![half.mul(&c(1.0)?.add(&inner.tanh()?)?)?])
        })?;
        for shape in [vec![2, 2], vec![1, 1, 2], vec![1, 2, 2]] {
            drop(silu.prepare(&[Array::zeros(stream, &shape, MlxDType::BF16)?])?);
        }
        for height in [1, 2] {
            for width in [1, 2] {
                drop(silu.prepare(&[Array::zeros(
                    stream,
                    &[1, 2, height, width],
                    MlxDType::F32,
                )?])?);
            }
        }
        for tokens in [1, 2] {
            drop(gelu_approx.prepare(&[Array::zeros(stream, &[1, tokens, 2], MlxDType::BF16)?])?);
        }
        Ok(Self {
            stream: stream.clone(),
            silu: Rc::new(silu),
            gelu_approx: Rc::new(gelu_approx),
        })
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// A Python float operand of an FP32 or BF16 array, as MLX converts it.
    pub fn scalar(&self, value: f32, dtype: MlxDType) -> Result<Array> {
        Array::scalar(&self.stream, value, dtype)
    }

    pub fn f32(&self, value: f32) -> Result<Array> {
        self.scalar(value, MlxDType::F32)
    }

    pub fn silu(&self, x: &Array) -> Result<Array> {
        Ok(self.silu.call(std::slice::from_ref(x))?.remove(0))
    }

    pub fn gelu_approx(&self, x: &Array) -> Result<Array> {
        Ok(self.gelu_approx.call(std::slice::from_ref(x))?.remove(0))
    }
}
