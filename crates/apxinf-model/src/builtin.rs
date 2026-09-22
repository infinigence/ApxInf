//! Built-in model registrations used by [`crate::AutoModel`].

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};

use crate::auto::{LoadOptions, LoadedModel};
use crate::llama::{GeneralLlama, LlamaWeights};
#[cfg(feature = "cuda")]
use crate::qwen35::Qwen35Model;
use crate::qwen3vl::{GeneralQwen3VL, Qwen3VLConfig};
use crate::registry;

/// Register every implementation shipped in this crate. Re-registering is
/// harmless and keeps `AutoModel::load_model` self-contained for users.
pub fn register_builtin_models() {
    registry::register("llama", load_llama);
    registry::register("qwen3_vl", load_qwen3vl);
    registry::register("qwen3vl", load_qwen3vl);
    #[cfg(feature = "cuda")]
    {
        registry::register("qwen3_5", load_qwen35);
        registry::register("qwen35", load_qwen35);
    }
    registry::register("qwen_drive", load_qwen_drive);

    #[cfg(feature = "cuda")]
    crate::pi05::register_builtin();
    #[cfg(feature = "cuda")]
    crate::walloss::register_builtin();
    #[cfg(feature = "cuda")]
    crate::pi0fast::register_builtin();
    #[cfg(feature = "cuda")]
    crate::gr00t::register_builtin();
}

#[cfg(feature = "cuda")]
fn load_qwen35(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    _options: &LoadOptions,
) -> Result<LoadedModel> {
    let model = Qwen35Model::from_path(path, backend, None)?;
    Ok(LoadedModel::text(Box::new(model)))
}
fn load_llama(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    let (mut tensors, metadata) = apxinf_loader::safetensors::load_native_path(path)
        .map_err(|error| Error::Other(format!("load {}: {error}", path.display())))?;
    if let Some(dtype) = options.text_weight_dtype {
        if !matches!(dtype, DType::F32 | DType::BF16) {
            return Err(Error::Other(format!(
                "Llama text weights support f32 or bf16, not {dtype}"
            )));
        }
    }
    if matches!(device, Device::Cpu) || options.text_weight_dtype == Some(DType::F32) {
        upcast_bf16_weights(&mut tensors)?;
    }
    let config = apxinf_loader::safetensors::config_from_metadata(&metadata);
    let weights = LlamaWeights::from_map(&config, tensors)?;
    Ok(LoadedModel::text(Box::new(GeneralLlama::new(
        config, weights, backend,
    )?)))
}

fn load_qwen3vl(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    _options: &LoadOptions,
) -> Result<LoadedModel> {
    let model_dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let config = Qwen3VLConfig::from_json_file(&model_dir.join("config.json"))?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
        .map_err(|error| Error::Other(format!("load {}: {error}", path.display())))?;
    let model = GeneralQwen3VL::from_weights_with_backend(config, tensors, backend)?;
    Ok(LoadedModel::text(Box::new(model)))
}

/// Load only the Qwen-Drive VLM through the text registry. Planning weights
/// require an explicit planner path through the planning-capable model/policy
/// entry point; an adjacent directory does not add capabilities to LlmTrait.
fn load_qwen_drive(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    _options: &LoadOptions,
) -> Result<LoadedModel> {
    #[cfg(feature = "cuda")]
    {
        let model_dir = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or_else(|| Path::new("."))
        };
        let model = crate::qwen_drive::QwenDriveModel::load_with_backend(model_dir, None, backend)?;
        Ok(LoadedModel::text(Box::new(model)))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (path, device, backend);
        Err(Error::Other(
            "qwen_drive requires the cuda feature (native CUDA deployment); \
             this build has no CUDA support"
                .into(),
        ))
    }
}

fn upcast_bf16_weights(tensors: &mut HashMap<String, Tensor>) -> Result<()> {
    for tensor in tensors.values_mut() {
        if tensor.dtype() != DType::BF16 {
            continue;
        }
        let shape = tensor.shape().dims().to_vec();
        *tensor = Tensor::from_f32(shape, &tensor.to_f32_vec()?)?;
    }
    Ok(())
}
