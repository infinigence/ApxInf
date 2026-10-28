//! Host checkpoint storage, source validation and host tensor construction.
use apxinf_core::{DType, Error, Result, Tensor};
use std::{collections::HashMap, path::Path};

pub(in crate::cuda_fastwam) struct Store {
    tensors: HashMap<String, Tensor>,
}

#[derive(Clone, Copy)]
pub(in crate::cuda_fastwam) enum Bias {
    Required,
    Forbidden,
}

impl Store {
    pub(in crate::cuda_fastwam) fn load(path: &Path) -> Result<Self> {
        let (tensors, _) =
            apxinf_loader::safetensors::load_native_path(path).map_err(Error::Other)?;
        Ok(Self { tensors })
    }
    pub(in crate::cuda_fastwam) fn take(&mut self, name: &str) -> Result<Tensor> {
        self.tensors
            .remove(name)
            .ok_or_else(|| Error::Other(format!("Fast-WAM checkpoint missing {name}")))
    }
    pub(in crate::cuda_fastwam) fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub(super) fn take_linear(
        &mut self,
        prefix: &str,
        bias: Bias,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let raw = self.take(&format!("{prefix}.weight"))?;
        let dims = raw.shape().dims();
        if dims.len() != 2 || dims.contains(&0) || raw.dtype() != DType::BF16 {
            return Err(Error::Other(format!(
                "{prefix} requires a nonempty rank-2 BF16 weight"
            )));
        }
        let n = dims[0];
        let name = format!("{prefix}.bias");
        let bias = match bias {
            Bias::Required => {
                let bias = self.take(&name)?;
                if bias.shape().dims() != [n] || bias.dtype() != raw.dtype() {
                    return Err(Error::Other(format!("invalid Fast-WAM linear bias {name}")));
                }
                Some(bias)
            }
            Bias::Forbidden => {
                if self.contains(&name) {
                    return Err(Error::Other(format!(
                        "Fast-WAM bias-free projection has unexpected {name}"
                    )));
                }
                None
            }
        };
        Ok((raw, bias))
    }
}

pub(in crate::cuda_fastwam) fn validate_tensor(
    name: &str,
    tensor: &Tensor,
    dims: &[usize],
    dtype: DType,
) -> Result<()> {
    if tensor.shape().dims() != dims || tensor.dtype() != dtype {
        return Err(Error::Other(format!(
            "Fast-WAM {name}: expected {dtype:?} {dims:?}, got {:?} {:?}",
            tensor.dtype(),
            tensor.shape()
        )));
    }
    Ok(())
}

pub(in crate::cuda_fastwam) fn host_tensor(
    shape: Vec<usize>,
    values: &[f32],
    dtype: DType,
) -> Result<Tensor> {
    match dtype {
        DType::F32 => Tensor::from_f32(shape, values),
        DType::BF16 => Tensor::from_bf16(
            shape,
            &values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        ),
        _ => Err(Error::Other(
            "Fast-WAM checkpoint expects BF16 or F32 weights".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::Shape;

    fn store(weight_shape: Vec<usize>, bias: Option<(Vec<usize>, DType)>) -> Store {
        let mut tensors = HashMap::from([(
            "projection.weight".into(),
            Tensor::zeros(Shape::new(weight_shape), DType::BF16),
        )]);
        if let Some((shape, dtype)) = bias {
            tensors.insert(
                "projection.bias".into(),
                Tensor::zeros(Shape::new(shape), dtype),
            );
        }
        Store { tensors }
    }

    #[test]
    fn fastwam_linear_bias_presence_matches_architecture() {
        let mut missing = store(vec![3, 2], None);
        assert!(missing
            .take_linear("projection", Bias::Required)
            .err()
            .unwrap()
            .to_string()
            .contains("missing projection.bias"));
        let mut unexpected = store(vec![3, 2], Some((vec![3], DType::BF16)));
        assert!(unexpected
            .take_linear("projection", Bias::Forbidden)
            .err()
            .unwrap()
            .to_string()
            .contains("unexpected projection.bias"));
        for (contract, bias) in [
            (Bias::Required, Some((vec![3], DType::BF16))),
            (Bias::Forbidden, None),
        ] {
            assert!(store(vec![3, 2], bias)
                .take_linear("projection", contract)
                .is_ok());
        }
    }

    #[test]
    fn fastwam_linear_rejects_invalid_weight_and_bias_shapes() {
        for shape in [vec![6], vec![1, 3, 2]] {
            assert!(store(shape, None)
                .take_linear("projection", Bias::Forbidden)
                .is_err());
        }
        for (shape, dtype) in [
            (vec![2], DType::BF16),
            (vec![1, 3], DType::BF16),
            (vec![3], DType::F32),
        ] {
            assert!(store(vec![3, 2], Some((shape, dtype)))
                .take_linear("projection", Bias::Required)
                .is_err());
        }
    }
}
