//! Device-ready BF16 linear weights for π0-FAST.

use apxinf_core::{Backend, DType, Error, Result, Tensor};

use super::{device_weights::concat_host_2d, LinearWeights};

#[derive(Debug)]
pub struct Bf16LinearWeights {
    /// Physical row-major `[input, output]` matrix.
    pub weight: Tensor,
    /// Reserved for the packed dual-GeGLU layout used by the π0.5 runtime.
    /// π0-FAST packs gate/up plainly today (`false`, no alternates), so every
    /// fused-GeGLU call takes the plain path.
    pub bf16_dual_geglu_interleaved: bool,
    pub bf16_dual_geglu_auto_interleaved: Option<Tensor>,
    pub bf16_sm89_geglu_interleaved: Option<Tensor>,
    pub bias: Option<Tensor>,
}

impl Bf16LinearWeights {
    pub fn from_host(linear: &LinearWeights, backend: &dyn Backend) -> Result<Self> {
        Self::from_host_parts(&[linear], backend)
    }

    /// Pack projections along the output dimension so QKV and gate/up each
    /// remain one tensor-core GEMM.
    pub fn from_host_parts(linears: &[&LinearWeights], backend: &dyn Backend) -> Result<Self> {
        if linears.is_empty() {
            return Err(Error::Other("cannot pack an empty BF16 linear group".into()));
        }
        let plain_weight = concat_host_2d(
            &linears
                .iter()
                .map(|linear| &linear.weight)
                .collect::<Vec<_>>(),
        )?;
        let weight = bf16_to_device(&plain_weight, backend)?;
        let bias = if linears.iter().all(|linear| linear.bias.is_none()) {
            None
        } else if linears.iter().all(|linear| linear.bias.is_some()) {
            Some(concat_biases_bf16(
                &linears
                    .iter()
                    .map(|linear| linear.bias.as_ref().unwrap())
                    .collect::<Vec<_>>(),
                backend,
            )?)
        } else {
            return Err(Error::Other(
                "cannot pack BF16 projections with mixed bias presence".into(),
            ));
        };
        Ok(Self {
            weight,
            bf16_dual_geglu_interleaved: false,
            bf16_dual_geglu_auto_interleaved: None,
            bf16_sm89_geglu_interleaved: None,
            bias,
        })
    }
}

pub fn bf16_to_device(tensor: &Tensor, backend: &dyn Backend) -> Result<Tensor> {
    if tensor.dtype() == DType::F8E4M3 {
        return Err(Error::Other(
            "cannot convert scale-less E4M3 data to BF16".into(),
        ));
    }
    let values = tensor
        .to_f32_vec()?
        .into_iter()
        .map(half::bf16::from_f32)
        .collect::<Vec<_>>();
    backend.to_device(&Tensor::from_bf16(tensor.shape().dims().to_vec(), &values)?)
}

/// Upload a host tensor unchanged in FP32.
///
/// PaliGemma deliberately keeps the SigLIP patch embedding (weight, bias, and
/// position embedding) in FP32 while the rest of the tower is BF16, so these
/// three tensors bypass the BF16 conversion.
pub fn f32_to_device(tensor: &Tensor, backend: &dyn Backend) -> Result<Tensor> {
    if tensor.dtype() != DType::F32 {
        return Err(Error::Other(format!(
            "π0-FAST FP32 passthrough expects F32, got {}",
            tensor.dtype()
        )));
    }
    let values = tensor.to_f32_vec()?;
    backend.to_device(&Tensor::from_f32(tensor.shape().dims().to_vec(), &values)?)
}

fn concat_biases_bf16(tensors: &[&Tensor], backend: &dyn Backend) -> Result<Tensor> {
    let mut values = Vec::new();
    for tensor in tensors {
        if tensor.shape().dims().len() != 1 || tensor.dtype() == DType::F8E4M3 {
            return Err(Error::Other(
                "packed BF16 biases must be non-FP8 vectors".into(),
            ));
        }
        values.extend(tensor.to_f32_vec()?.into_iter().map(half::bf16::from_f32));
    }
    backend.to_device(&Tensor::from_bf16(vec![values.len()], &values)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::CpuBackend;

    #[test]
    fn packs_qkv_as_native_bf16() {
        let linear = |weight: &[f32], shape: [usize; 2]| LinearWeights {
            weight: Tensor::from_f32(shape.to_vec(), weight).unwrap(),
            bias: None,
        };
        let q = linear(&[1., 2., 3., 4.], [2, 2]);
        let k = linear(&[5., 6.], [2, 1]);
        let v = linear(&[7., 8.], [2, 1]);
        let packed = Bf16LinearWeights::from_host_parts(&[&q, &k, &v], &CpuBackend).unwrap();
        assert_eq!(packed.weight.shape().dims(), &[2, 4]);
        assert_eq!(packed.weight.dtype(), DType::BF16);
        assert!(packed.bias.is_none());
        assert!(!packed.bf16_dual_geglu_interleaved);
    }
}
