//! Host checkpoint storage and strict source validation. No device computation.
use apxinf_core::{DType, Error, Result, Tensor};
use std::{collections::HashMap, path::Path};
pub(super) struct Store {
    tensors: HashMap<String, Tensor>,
}
impl Store {
    pub(super) fn new(path: &Path) -> Result<Self> {
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|e| Error::Other(format!("load {}: {e}", path.display())))?;
        Ok(Self { tensors })
    }
    pub(super) fn take(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let tensor = self
            .tensors
            .remove(name)
            .ok_or_else(|| Error::Other(format!("missing Cosmos Policy weight {name}")))?;
        if tensor.dtype() != DType::BF16 || tensor.shape().dims() != shape {
            return Err(Error::Other(format!(
                "Cosmos Policy weight {name}: expected BF16 {shape:?}, got {} {:?}",
                tensor.dtype(),
                tensor.shape().dims()
            )));
        }
        if tensor.as_bf16()?.iter().any(|x| !x.to_f32().is_finite()) {
            return Err(Error::Other(format!(
                "nonfinite Cosmos Policy weight {name}"
            )));
        }
        Ok(tensor)
    }

    pub(super) fn finish(self) -> Result<()> {
        if !self.tensors.is_empty() {
            let mut names: Vec<_> = self.tensors.into_keys().collect();
            names.sort();
            return Err(Error::Other(format!(
                "unexpected Cosmos Policy weights: {names:?}"
            )));
        }
        Ok(())
    }
}
