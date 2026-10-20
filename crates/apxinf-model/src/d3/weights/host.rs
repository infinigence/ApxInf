//! Host-side ownership of the official d3 package weights.

use std::collections::HashMap;

use apxinf_core::{DType, Error, Result, Tensor};

use crate::d3::D3PackageConfig;

/// The two BF16 backbone namespaces from `model.safetensors.index.json`.
#[derive(Debug)]
pub struct D3CheckpointWeights {
    pub(crate) language: HashMap<String, Tensor>,
    pub(crate) visual: HashMap<String, Tensor>,
}

/// The separate FP32 decision readout from `readout.safetensors`.
#[derive(Debug)]
pub struct D3ReadoutWeights {
    pub(crate) weight: Tensor,
}

impl D3CheckpointWeights {
    pub fn from_map(tensors: HashMap<String, Tensor>) -> Result<Self> {
        let mut language = HashMap::new();
        let mut visual = HashMap::new();
        let mut unexpected = Vec::new();
        for (name, tensor) in tensors {
            if name.starts_with("language_model.") {
                language.insert(name, tensor);
            } else if name.starts_with("visual.") {
                visual.insert(name, tensor);
            } else {
                unexpected.push(name);
            }
        }
        unexpected.sort_unstable();
        if let Some(first) = unexpected.first() {
            return Err(Error::Other(format!(
                "d3 weights: {} tensors outside language_model/visual, first: {first}",
                unexpected.len()
            )));
        }
        for required in [
            "language_model.embed_tokens.weight",
            "language_model.norm.weight",
        ] {
            if !language.contains_key(required) {
                return Err(Error::Other(format!("d3 weights: missing {required}")));
            }
        }
        if visual.is_empty() {
            return Err(Error::Other(
                "d3 weights: the official visual.* tower is required".into(),
            ));
        }
        Ok(Self { language, visual })
    }

    pub fn language_tensor_count(&self) -> usize {
        self.language.len()
    }

    pub fn visual_tensor_count(&self) -> usize {
        self.visual.len()
    }
}

impl D3ReadoutWeights {
    pub fn from_map(
        config: &D3PackageConfig,
        mut tensors: HashMap<String, Tensor>,
    ) -> Result<Self> {
        let weight = tensors
            .remove("weight")
            .ok_or_else(|| Error::Other("d3 readout: missing weight".into()))?;
        if let Some(first) = tensors.keys().min() {
            return Err(Error::Other(format!(
                "d3 readout: unexpected tensor {first}"
            )));
        }
        let expected = [
            config.decision.codes.len(),
            config.model.text_config.hidden_size,
        ];
        if weight.shape().dims() != expected || weight.dtype() != DType::F32 {
            return Err(Error::Other(format!(
                "d3 readout: weight must be FP32 {expected:?}, got {} {:?}",
                weight.dtype(),
                weight.shape().dims()
            )));
        }
        Ok(Self { weight })
    }
}

/// Transpose a checkpoint `[out, in]` matrix into cuda-new's `[in, out]` layout.
pub(super) fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "d3 transpose expected 2D tensor, got {dims:?}"
        )));
    }
    let (rows, cols) = (dims[0], dims[1]);
    const TILE: usize = 32;
    match tensor.dtype() {
        DType::F32 => {
            let data = tensor.as_f32()?;
            let mut output = vec![0.0f32; rows * cols];
            for ii in (0..rows).step_by(TILE) {
                for jj in (0..cols).step_by(TILE) {
                    for i in ii..(ii + TILE).min(rows) {
                        for j in jj..(jj + TILE).min(cols) {
                            output[j * rows + i] = data[i * cols + j];
                        }
                    }
                }
            }
            Tensor::from_f32(vec![cols, rows], &output)
        }
        DType::BF16 => {
            let data = tensor.as_bf16()?;
            let mut output = vec![half::bf16::from_f32(0.0); rows * cols];
            for ii in (0..rows).step_by(TILE) {
                for jj in (0..cols).step_by(TILE) {
                    for i in ii..(ii + TILE).min(rows) {
                        for j in jj..(jj + TILE).min(cols) {
                            output[j * rows + i] = data[i * cols + j];
                        }
                    }
                }
            }
            Tensor::from_bf16(vec![cols, rows], &output)
        }
        dtype => Err(Error::Other(format!(
            "d3 transpose does not support {dtype}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_namespaces_are_strict() {
        let mut map = HashMap::new();
        map.insert(
            "language_model.embed_tokens.weight".into(),
            Tensor::zeros(vec![2, 2], DType::BF16),
        );
        map.insert(
            "language_model.norm.weight".into(),
            Tensor::zeros(vec![2], DType::BF16),
        );
        map.insert(
            "visual.patch_embed.proj.weight".into(),
            Tensor::zeros(vec![2, 2], DType::BF16),
        );
        let weights = D3CheckpointWeights::from_map(map).unwrap();
        assert_eq!(weights.language_tensor_count(), 2);
        assert_eq!(weights.visual_tensor_count(), 1);
    }

    #[test]
    fn checkpoint_rejects_foreign_prefix() {
        let mut map = HashMap::new();
        map.insert(
            "model.language_model.norm.weight".into(),
            Tensor::zeros(vec![2], DType::BF16),
        );
        let error = D3CheckpointWeights::from_map(map).unwrap_err();
        assert!(error.to_string().contains("outside language_model/visual"));
    }

    #[test]
    fn projection_transposes_bf16() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>();
        let tensor = Tensor::from_bf16(vec![2, 3], &values).unwrap();
        let output = transpose_2d(&tensor).unwrap();
        assert_eq!(output.shape().dims(), &[3, 2]);
        assert_eq!(output.to_f32_vec().unwrap(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
