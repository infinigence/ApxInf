//! Lazy-upload BF16 weight store for a converted qwen3-asr checkpoint.
//!
//! The checkpoint is a single `weights.safetensors` whose tensors are all BF16
//! except `frontend.mel_filter` (F32); names and layouts follow the converter's
//! manifest.

use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_cuda_new::{transfers, CudaContext as Context};

pub const DTYPE: DType = DType::BF16;

pub struct Weights {
    host: HashMap<String, Tensor>,
    device: HashMap<String, Tensor>,
}

impl Weights {
    pub fn load(directory: &Path) -> Result<Self> {
        let path = directory.join("weights.safetensors");
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(&path)
            .map_err(|error| Error::Other(format!("load {}: {error}", path.display())))?;
        Ok(Self {
            host: tensors,
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
            .ok_or_else(|| Error::Other(format!("missing qwen3-asr weight {key}")))?;
        if host.dtype() != DTYPE {
            return Err(Error::Other(format!(
                "weight {key} has dtype {}, expected {DTYPE}",
                host.dtype()
            )));
        }
        let device = transfers::to_cuda(&host, ctx.device_id())?;
        self.device.insert(key.to_owned(), device.clone());
        Ok(device)
    }

    /// Read a host tensor without uploading it (frontend constants).
    pub fn host_clone(&self, key: &str) -> Result<Tensor> {
        self.host
            .get(key)
            .cloned()
            .ok_or_else(|| Error::Other(format!("missing qwen3-asr weight {key}")))
    }
}
