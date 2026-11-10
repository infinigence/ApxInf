//! Device-ready static-FP8 linear weights.

use apxinf_core::{Backend, DType, Error, Result, Tensor};

#[cfg(feature = "cuda")]
use super::backend::{kernels, RuntimeBackend};
use super::{quantize_e4m3_absmax, LinearWeights};

#[derive(Debug)]
pub struct Fp8LinearWeights {
    /// `[input, output]` CUDA E4M3 matrix.
    pub weight: Tensor,
    pub weight_scale: f32,
    /// Bias stays FP16 and is fused into the consumer kernel.
    pub bias: Option<Tensor>,
}

impl Fp8LinearWeights {
    #[cfg(feature = "cuda")]
    pub fn as_kernel_view(&self) -> kernels::gemm::Fp8WeightView<'_> {
        kernels::gemm::Fp8WeightView {
            values_e4m3: &self.weight,
            scale: self.weight_scale,
        }
    }

    pub fn from_host(linear: &LinearWeights, backend: &dyn Backend) -> Result<Self> {
        Self::from_host_parts(&[linear], backend)
    }

    /// Concatenate projections along their output dimension before applying
    /// one absmax quantization scale. This produces graph-ready QKV and
    /// gate/up matrices without runtime concatenation or mixed descales.
    pub fn from_host_parts(linears: &[&LinearWeights], backend: &dyn Backend) -> Result<Self> {
        if linears.is_empty() {
            return Err(Error::Other("cannot pack an empty FP8 linear group".into()));
        }
        let weight = concat_host_2d(&linears.iter().map(|x| &x.weight).collect::<Vec<_>>())?;
        #[cfg(feature = "cuda")]
        let (weight, weight_scale) =
            if let Some(cuda_backend) = backend.as_any().downcast_ref::<RuntimeBackend>() {
                // Quantizing billions of parameters with the scalar CPU E4M3
                // encoder is prohibitively slow on Jetson. Upload FP16 once and
                // let the CUDA conversion kernel produce the resident FP8 matrix.
                let (weight_f16, amax) = fp16_host_and_amax(&weight)?;
                let weight_scale = if amax == 0.0 {
                    1.0
                } else {
                    amax / super::E4M3_MAX
                };
                let weight_f16 = backend.to_device(&weight_f16)?;
                let weight = kernels::quantization::quantize_f16_e4m3(
                    cuda_backend.context(),
                    &weight_f16,
                    weight_scale,
                )?;
                (weight, weight_scale)
            } else {
                let quantized = quantize_e4m3_absmax(&weight)?;
                (backend.to_device(&quantized.values)?, quantized.scale)
            };
        #[cfg(not(feature = "cuda"))]
        let (weight, weight_scale) = {
            let quantized = quantize_e4m3_absmax(&weight)?;
            (backend.to_device(&quantized.values)?, quantized.scale)
        };
        let bias = if linears.iter().all(|x| x.bias.is_none()) {
            None
        } else if linears.iter().all(|x| x.bias.is_some()) {
            let biases = linears
                .iter()
                .map(|x| x.bias.as_ref().unwrap())
                .collect::<Vec<_>>();
            Some(backend.to_device(&concat_host_1d_f16(&biases)?)?)
        } else {
            return Err(Error::Other(
                "cannot pack projections with a mixture of present and absent biases".into(),
            ));
        };
        Ok(Self {
            weight,
            weight_scale,
            bias,
        })
    }
}

#[cfg(feature = "cuda")]
fn fp16_host_and_amax(tensor: &Tensor) -> Result<(Tensor, f32)> {
    let values = tensor.to_f32_vec()?;
    let amax = values
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    let values = values
        .into_iter()
        .map(half::f16::from_f32)
        .collect::<Vec<_>>();
    Ok((
        Tensor::from_f16(tensor.shape().dims().to_vec(), &values)?,
        amax,
    ))
}

pub fn fp16_to_device(tensor: &Tensor, backend: &dyn Backend) -> Result<Tensor> {
    let values = tensor.to_f32_vec()?;
    let values = values
        .iter()
        .map(|value| half::f16::from_f32(*value))
        .collect::<Vec<_>>();
    backend.to_device(&Tensor::from_f16(tensor.shape().dims().to_vec(), &values)?)
}

pub(super) fn concat_host_2d(tensors: &[&Tensor]) -> Result<Tensor> {
    let first = tensors
        .first()
        .ok_or_else(|| Error::Other("empty tensor concatenation".into()))?;
    let dims = first.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!("expected 2D weight, got {dims:?}")));
    }
    let rows = dims[0];
    let widths = tensors
        .iter()
        .map(|tensor| {
            let dims = tensor.shape().dims();
            if dims.len() != 2 || dims[0] != rows {
                return Err(Error::Other("packed linear input dimensions differ".into()));
            }
            Ok(dims[1])
        })
        .collect::<Result<Vec<_>>>()?;
    let total_cols = widths.iter().sum::<usize>();
    let sources = tensors
        .iter()
        .map(|tensor| tensor.to_f32_vec())
        .collect::<Result<Vec<_>>>()?;
    let mut output = vec![0.0f32; rows * total_cols];
    for row in 0..rows {
        let mut output_col = 0;
        for (source, width) in sources.iter().zip(&widths) {
            output[row * total_cols + output_col..row * total_cols + output_col + width]
                .copy_from_slice(&source[row * width..(row + 1) * width]);
            output_col += width;
        }
    }
    Tensor::from_f32(vec![rows, total_cols], &output)
}

fn concat_host_1d_f16(tensors: &[&Tensor]) -> Result<Tensor> {
    let mut output = Vec::new();
    for tensor in tensors {
        if tensor.shape().dims().len() != 1 || tensor.dtype() == DType::F8E4M3 {
            return Err(Error::Other("packed biases must be non-FP8 vectors".into()));
        }
        output.extend(tensor.to_f32_vec()?.into_iter().map(half::f16::from_f32));
    }
    Tensor::from_f16(vec![output.len()], &output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::CpuBackend;

    fn linear(weight: &[f32], shape: [usize; 2], bias: Option<&[f32]>) -> LinearWeights {
        LinearWeights {
            weight: Tensor::from_f32(shape.to_vec(), weight).unwrap(),
            bias: bias.map(|x| Tensor::from_f32(vec![x.len()], x).unwrap()),
        }
    }

    #[test]
    fn packs_qkv_before_quantization() {
        let q = linear(&[1., 2., 3., 4.], [2, 2], Some(&[1., 2.]));
        let k = linear(&[5., 6.], [2, 1], Some(&[3.]));
        let v = linear(&[7., 8.], [2, 1], Some(&[4.]));
        let packed = Fp8LinearWeights::from_host_parts(&[&q, &k, &v], &CpuBackend).unwrap();
        assert_eq!(packed.weight.shape().dims(), &[2, 4]);
        assert_eq!(packed.weight.dtype(), DType::F8E4M3);
        let bias = packed.bias.unwrap();
        assert_eq!(bias.dtype(), DType::F16);
        assert_eq!(bias.to_f32_vec().unwrap(), vec![1., 2., 3., 4.]);
    }
}
