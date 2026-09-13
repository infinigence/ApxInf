//! Built-in model registrations used by [`crate::AutoModel`].

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};

use crate::auto::{LoadOptions, LoadedModel};
use crate::llama::{GeneralLlama, LlamaWeights};
use crate::qwen3vl::{GeneralQwen3VL, Qwen3VLConfig};
use crate::registry;

/// Register every implementation shipped in this crate. Re-registering is
/// harmless and keeps `AutoModel::load_model` self-contained for users.
pub fn register_builtin_models() {
    registry::register("llama", load_llama);
    registry::register("qwen3_vl", load_qwen3vl);
    registry::register("qwen3vl", load_qwen3vl);
    registry::register("qwen_drive", load_qwen_drive);
    registry::register("qwen_drive-cuda", load_qwen_drive);

    #[cfg(feature = "cuda")]
    crate::pi05::register_builtin();
    #[cfg(feature = "cuda")]
    crate::walloss::register_builtin();
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

/// Load the Qwen-Drive VLM through the maintained registry surface. When the
/// checkpoint ships the SFT planner head beside the VLM it is attached so the
/// direct-planning flow works out of the box; the Python policy passes an
/// explicit planner path for the RL head. CUDA-only: other devices fail
/// clearly here.
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
        let planner_dir = model_dir.join("planner-sft");
        let planner = if planner_dir.is_dir() {
            Some(planner_dir.as_path())
        } else {
            None
        };
        let model = crate::qwen_drive::QwenDriveModel::load_with_backend(
            model_dir, planner, backend,
        )?;
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
