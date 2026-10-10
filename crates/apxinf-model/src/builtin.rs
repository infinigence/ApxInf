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
    #[cfg(feature = "cuda-new")]
    registry::register("qwen38", load_qwen38);
    #[cfg(feature = "cuda-new")]
    registry::register("qwen3_8", load_qwen38);
    // The NVFP4 checkpoint's config.json says model_type "qwen3_5" /
    // "qwen3_5_text"; register both so AutoModel's detection works on the
    // unmodified checkpoint directory.
    #[cfg(feature = "cuda-new")]
    registry::register("qwen3_5", load_qwen38);
    #[cfg(feature = "cuda-new")]
    registry::register("qwen3_5_text", load_qwen38);

    #[cfg(feature = "cuda")]
    crate::walloss::register_builtin();
    #[cfg(feature = "cuda")]
    crate::pi0fast::register_builtin();
    #[cfg(feature = "cuda")]
    crate::gr00t::register_builtin();
    #[cfg(feature = "cuda")]
    crate::smolvla::register_builtin();
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

fn load_qwen_drive(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    #[cfg(feature = "cuda")]
    {
        crate::qwen_drive::load::load_registered(path, device, backend, options)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (path, device, backend, options);
        Err(Error::Other(
            "qwen_drive planning requires the CUDA feature and a CUDA device".into(),
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


/// Qwen3.8-27B-NVFP4 on CUDA: safetensors shards in the checkpoint directory,
/// validated default execution (FlashInfer-GDN prefill, CUDA-graph decode,
/// split-KV attention).
fn load_qwen38(
    path: &Path,
    device: Device,
    _backend: Arc<dyn Backend>,
    _options: &LoadOptions,
) -> Result<LoadedModel> {
    #[cfg(feature = "cuda-new")]
    {
        use crate::llm_trait::LlmTrait;
        let (tensors, metadata) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|error| Error::Other(format!("load {}: {error}", path.display())))?;
        let config = apxinf_loader::safetensors::config_from_metadata(&metadata);
        let model = crate::qwen38::Qwen38::load(config, tensors, device)?;
        Ok(LoadedModel::text(Box::new(model)))
    }
    #[cfg(not(feature = "cuda-new"))]
    {
        let _ = (path, device);
        Err(Error::Other("qwen38 requires the cuda-new feature".into()))
    }
}
