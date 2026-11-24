//! LFM3-1.7B native MLX implementation: no external engine, no Python runtime.
//!
//! LFM3 is a hybrid checkpoint: 13 gated width-7 short-conv layers, 6 unbounded
//! attention layers and 5 sliding-window (256) attention layers, with sandwich
//! RMSNorm, a per-head sigmoid attention gate and two independent partial-RoPE
//! parameter sets. See `README.md` for the supported contract and scope.

pub mod config;
mod model;
mod weights;
mod q4;
pub use model::{native_4bit_enabled, prefill_fast_enabled, Lfm3Model};

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, DType, Device, Error, Result};
use config::Lfm3Config;
use std::{path::Path, sync::Arc};

/// Registry loader; construction retains the exact backend stream chosen by AutoModel.
pub fn load(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Metal(_)) || backend.device() != device {
        return Err(Error::UnsupportedDevice(device));
    }
    admit(options)?;
    let json = std::fs::read_to_string(path.join("config.json"))
        .map_err(|e| Error::Other(format!("cannot read LFM3 config: {e}")))?;
    let config = Lfm3Config::from_json(&json)?;
    config.validate_checkpoint_scope()?;
    let (mut tensors, _) = apxinf_loader::safetensors::load_native_path(path).map_err(Error::Other)?;
    // The BF16 checkpoint keys are `model.*`; the Hugging Face MLX-quantized
    // export wraps them in `language_model.` and adds `{scales,biases}` next
    // to each packed `weight`. Normalizing the prefix once here means the
    // existing `model.*` lookup paths work unchanged for both arms, and the
    // default (env-unset) arm still never inspects a quantized key.
    if q4::enabled() {
        let prefix = "language_model.";
        for key in tensors
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect::<Vec<_>>()
        {
            let new = key[prefix.len()..].to_string();
            let value = tensors.remove(&key).expect("key present");
            tensors.insert(new, value);
        }
    }
    Ok(LoadedModel::text(Box::new(Lfm3Model::new(
        config, tensors, backend,
    )?)))
}

/// Render the lane ChatML prompt the frozen corpus and the registered gate use.
///
/// The checkpoint's `chat_template.jinja` is deliberately **not** used here. Even
/// with `enable_thinking=false` that template appends `<think>` and the
/// end-of-thinking token to the assistant prefix, i.e. the Hugging Face
/// thinking-prefill protocol. That is a different protocol: it was never frozen
/// against our corpus and it needs its own reference capture and gate. The
/// verified protocol ends the assistant prefix after `<|im_start|>assistant\n`
/// and lets the checkpoint emit `<think>` itself as its first generated token.
///
/// The prompt starts with the snapshot's BOS text token, which the tokenizer
/// resolves to id 124894, exactly like the reference lane's `load_tokens`.
pub fn lane_chat_prompt(messages: &[serde_json::Value]) -> Result<String> {
    const IM_START: &str = "<|im_start|>";
    const IM_END: &str = "<|im_end|>";
    let mut text = String::from("<|startoftext|>");
    let mut saw_user = false;
    for message in messages {
        let role = message
            .get("role")
            .and_then(|value| value.as_str())
            .ok_or_else(|| Error::Other("LFM3 lane ChatML: every message needs a role".into()))?;
        let content = match message.get("content") {
            Some(serde_json::Value::String(text)) => text.as_str(),
            None => "",
            Some(_) => {
                return Err(Error::Other(
                    "LFM3 lane ChatML: message content must be a string".into(),
                ))
            }
        };
        if !matches!(role, "system" | "user" | "assistant") {
            return Err(Error::Other(format!(
                "LFM3 lane ChatML: role `{role}` is unsupported (system, user, assistant)"
            )));
        }
        saw_user |= role == "user";
        text.push_str(IM_START);
        text.push_str(role);
        text.push('\n');
        text.push_str(content);
        text.push_str(IM_END);
        text.push('\n');
    }
    if !saw_user {
        return Err(Error::Other(
            "LFM3 lane ChatML: the conversation needs a user turn".into(),
        ));
    }
    // Generation prompt: nothing after the newline, so the model emits `<think>`
    // itself. Do not add the Hugging Face thinking prefix.
    text.push_str(IM_START);
    text.push_str("assistant\n");
    Ok(text)
}

/// Device-independent option admission, checked before the checkpoint is read.
///
/// This port implements exactly one prepared representation: the original BF16
/// checkpoint executed eagerly. Everything that would silently change numerics
/// or claim an unprepared capability is rejected here instead.
fn admit(options: &LoadOptions) -> Result<()> {
    if options.synthetic.is_some()
        || options.config.is_some()
        || !options.assets.is_empty()
        || options.calibration_path.is_some()
        || options.tuning_path.is_some()
        || options.uniform_fp8_scale.is_some()
        || options.autotune
        || !matches!(
            options.precision,
            ModelPrecision::Auto | ModelPrecision::Bf16
        )
        || options
            .text_weight_dtype
            .is_some_and(|dtype| dtype != DType::BF16)
    {
        return Err(Error::Other(
            "metal_lfm3 requires the original BF16 checkpoint and no prepared variant; CUDA calibration/tuning, FP8 and synthetic options are unsupported".into(),
        ));
    }
    match options.model_variant.as_deref().unwrap_or("auto") {
        "auto" | "bf16-eager" => Ok(()),
        other => Err(Error::Other(format!(
            "unsupported LFM3 MLX variant `{other}`; implemented: auto, bf16-eager"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(variant: Option<&str>) -> LoadOptions {
        LoadOptions {
            model_variant: variant.map(str::to_owned),
            ..LoadOptions::default()
        }
    }

    #[test]
    fn admission_needs_no_device_or_checkpoint() {
        assert!(admit(&options(None)).is_ok());
        assert!(admit(&options(Some("auto"))).is_ok());
        assert!(admit(&options(Some("bf16-eager"))).is_ok());
        assert!(admit(&options(Some("mixed-w8"))).is_err());
        let mut with_asset = options(None);
        with_asset.assets.insert("draft".into(), "draft".into());
        assert!(admit(&with_asset).is_err());
        let mut tuned = options(None);
        tuned.autotune = true;
        assert!(admit(&tuned).is_err());
        let mut fp32 = options(None);
        fp32.text_weight_dtype = Some(DType::F32);
        assert!(admit(&fp32).is_err());
    }
}
