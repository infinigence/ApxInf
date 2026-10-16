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
