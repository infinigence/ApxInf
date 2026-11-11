use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda_new::{transfers, CudaContext as Context};

use super::ops;

pub struct Weights {
    host: HashMap<String, Tensor>,
    device: HashMap<String, Tensor>,
}

impl Weights {
    pub fn load(directory: &Path) -> Result<Self> {
        let mut host = HashMap::new();
        for name in ["gpt", "sovits", "vocoder", "hubert", "roberta"] {
            let path = directory.join(format!("{name}.safetensors"));
            let (tensors, _) = apxinf_loader::safetensors::load_native_path(&path)
                .map_err(|error| Error::Other(format!("load {}: {error}", path.display())))?;
            host.extend(tensors);
        }
        Ok(Self {
            host,
            device: HashMap::new(),
        })
    }

    /// Upload a tensor once and return a shallow clone of the device tensor.
    pub fn get(&mut self, ctx: &Context, key: &str) -> Result<Tensor> {
        if let Some(tensor) = self.device.get(key) {
            return Ok(tensor.clone());
        }
        let host = self
            .host
            .remove(key)
            .ok_or_else(|| Error::Other(format!("missing gpt_sovits weight {key}")))?;
        if host.dtype() != ops::DTYPE {
            return Err(Error::Other(format!(
                "weight {key} has dtype {}, expected {}",
                host.dtype(),
                ops::DTYPE
            )));
        }
        let device = transfers::to_cuda(&host, ctx.device_id())?;
        self.device.insert(key.to_owned(), device.clone());
        Ok(device)
    }

    pub fn get_optional(&mut self, ctx: &Context, key: &str) -> Result<Option<Tensor>> {
        if self.host.contains_key(key) || self.device.contains_key(key) {
            self.get(ctx, key).map(Some)
        } else {
            Ok(None)
        }
    }

    pub fn has(&self, key: &str) -> bool {
        self.host.contains_key(key) || self.device.contains_key(key)
    }

    /// Read a host tensor without uploading it (used for small constants).
    pub fn host(&mut self, key: &str) -> Result<Tensor> {
        self.host
            .remove(key)
            .ok_or_else(|| Error::Other(format!("missing gpt_sovits weight {key}")))
    }

    pub fn host_clone(&self, key: &str) -> Result<Tensor> {
        self.host
            .get(key)
            .cloned()
            .ok_or_else(|| Error::Other(format!("missing gpt_sovits weight {key}")))
    }

    pub fn put_device(&mut self, key: &str, tensor: &Tensor) {
        self.device.insert(key.to_owned(), tensor.clone());
    }

    pub fn to_device(&self, ctx: &Context, tensor: &Tensor) -> Result<Tensor> {
        transfers::to_cuda(tensor, ctx.device_id())
    }
}

pub struct Linear {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl Linear {
    pub fn load(weights: &mut Weights, ctx: &Context, prefix: &str) -> Result<Self> {
        let w = weights.get(ctx, &format!("{prefix}.w"))?;
        let b = weights.get_optional(ctx, &format!("{prefix}.b"))?;
        Ok(Self { w, b })
    }

    pub fn apply(&self, ctx: &Context, input: &Tensor) -> Result<Tensor> {
        if self.uses_accumulator_semantics(input) {
            self.apply_accumulator(ctx, input)
        } else {
            ops::linear(ctx, input, &self.w, self.b.as_ref())
        }
    }

    pub fn apply_seq(&self, ctx: &Context, input: &Tensor) -> Result<Tensor> {
        if self.uses_accumulator_semantics(input) {
            let dims = input.shape().dims().to_vec();
            if dims.len() != 3 {
                return Err(Error::Other("linear_seq expects rank-3 input".into()));
            }
            let rows = dims[0] * dims[1];
            let out_cols = *self
                .w
                .shape()
                .dims()
                .get(1)
                .ok_or_else(|| Error::Other("linear_seq expects rank-2 weight".into()))?;
            let flat = input.reshape(apxinf_core::Shape::new(vec![rows, dims[2]]))?;
            let out = self.apply_accumulator(ctx, &flat)?;
            out.reshape(apxinf_core::Shape::new(vec![dims[0], dims[1], out_cols]))
        } else {
            ops::linear_seq(ctx, input, &self.w, self.b.as_ref())
        }
    }

    pub fn apply_accumulator(&self, ctx: &Context, input: &Tensor) -> Result<Tensor> {
        ops::linear_bias_accumulator(ctx, input, &self.w, self.b.as_ref())
    }

    fn uses_accumulator_semantics(&self, input: &Tensor) -> bool {
        input.dtype() == ops::DTYPE && self.b.is_some()
    }
}

#[derive(Clone)]
pub struct Conv {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl Conv {
    pub fn load(weights: &mut Weights, ctx: &Context, prefix: &str) -> Result<Self> {
        let w = weights.get(ctx, &format!("{prefix}.w"))?;
        let b = weights.get_optional(ctx, &format!("{prefix}.b"))?;
        Ok(Self { w, b })
    }

    pub fn apply(
        &self,
        ctx: &Context,
        input: &Tensor,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor> {
        ops::conv1d(
            ctx,
            input,
            &self.w,
            self.b.as_ref(),
            stride,
            padding,
            dilation,
            groups,
        )
    }

    pub fn transpose(
        &self,
        ctx: &Context,
        input: &Tensor,
        stride: usize,
        padding: usize,
    ) -> Result<Tensor> {
        ops::conv_transpose1d(ctx, input, &self.w, self.b.as_ref(), stride, padding, 0)
    }
}

pub fn normalize(weights: &mut Weights, ctx: &Context, prefix: &str) -> Result<(Tensor, Tensor)> {
    Ok((
        weights.get(ctx, &format!("{prefix}.w"))?,
        weights.get(ctx, &format!("{prefix}.b"))?,
    ))
}

pub fn ids_tensor(ctx: &Context, ids: &[u32]) -> Result<Tensor> {
    let ids: Vec<i32> = ids.iter().map(|value| *value as i32).collect();
    let host = Tensor::from_i32(vec![ids.len()], &ids)?;
    transfers::to_cuda(&host, ctx.device_id())
}

pub fn f32_tensor(ctx: &Context, shape: &[usize], values: &[f32]) -> Result<Tensor> {
    let host = Tensor::from_f32(shape.to_vec(), values)?;
    transfers::to_cuda(&host, ctx.device_id())
}

pub fn fp16_tensor(ctx: &Context, shape: &[usize], values: &[f32]) -> Result<Tensor> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        let value = half::f16::from_f32(*value);
        bytes.extend_from_slice(&value.to_bits().to_ne_bytes());
    }
    let host = Tensor::from_raw(
        apxinf_core::Shape::new(shape.to_vec()),
        apxinf_core::DType::F16,
        apxinf_core::Device::Cpu,
        bytes,
    )?;
    transfers::to_cuda(&host, ctx.device_id())
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::DType;

    fn values(tensor: &Tensor) -> Vec<f32> {
        assert_eq!(tensor.dtype(), DType::F16);
        let buffer = apxinf_cuda_new::CudaBuffer::from_tensor(tensor).unwrap();
        let mut bytes = vec![0; buffer.len()];
        buffer.copy_to_host(&mut bytes).unwrap();
        bytes
            .chunks_exact(2)
            .map(|value| half::f16::from_bits(u16::from_ne_bytes([value[0], value[1]])).to_f32())
            .collect()
    }

    #[test]
    fn fp16_biased_linear_apply_uses_accumulator_semantics() {
        let ctx = Context::new(0).unwrap();
        let input = fp16_tensor(
            &ctx,
            &[2, 4],
            &[0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0],
        )
        .unwrap();
        let weight = fp16_tensor(
            &ctx,
            &[4, 3],
            &[
                0.5, -0.25, 0.125, 0.25, 0.5, -0.125, -0.5, 0.25, 0.0, 0.125, -0.5, 0.25,
            ],
        )
        .unwrap();
        let bias = fp16_tensor(&ctx, &[3], &[0.125, -0.25, 0.5]).unwrap();
        let linear = Linear {
            w: weight,
            b: Some(bias),
        };

        assert_eq!(
            values(&linear.apply(&ctx, &input).unwrap()),
            values(&linear.apply_accumulator(&ctx, &input).unwrap())
        );

        let sequence = input
            .reshape(apxinf_core::Shape::new(vec![1, 2, 4]))
            .unwrap();
        assert_eq!(
            values(&linear.apply_seq(&ctx, &sequence).unwrap()),
            values(&linear.apply_accumulator(&ctx, &input).unwrap())
        );
    }
}
