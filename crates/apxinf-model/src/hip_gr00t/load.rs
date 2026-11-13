//! Load a GR00T N1.7 checkpoint onto a HIP device.
//!
//! Registered as `Gr00tN1d7-hip` (plus the aliases `gr00t-hip` and
//! `gr00t_n1_7-hip`), so `AutoModel` reaches it through the registry the way
//! it reaches `Gr00tN1d7-cuda`: `config.json:model_type` names the family and
//! the device suffix picks the backend.

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, Result};
use apxinf_hip::{HipBackend, HipContext};

use super::config::Gr00tConfig;
use super::model;
use super::model_runner::HipGr00tRunner;
use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::vla::VlaRuntime;

/// The only asset name this family accepts, for checkpoints that predate the
/// `assets/` manifest layout.
const BACKBONE_ASSET: &str = "backbone";

pub(crate) fn register_builtin() {
    for name in ["Gr00tN1d7-hip", "gr00t-hip", "gr00t_n1_7-hip"] {
        crate::registry::register(name, load_registered);
    }
}

fn load_registered(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    // Reuse the context the registry already opened for this device rather
    // than opening a second one: two contexts on one device would allocate
    // and synchronize independently.
    let hip = backend
        .as_any()
        .downcast_ref::<HipBackend>()
        .ok_or_else(|| {
            Error::Other(format!("GR00T on HIP was given a non-HIP backend for {device}"))
        })?;
    Ok(LoadedModel::Vla(Box::new(load_runner(
        path,
        Arc::clone(hip.context()),
        options,
    )?)))
}

/// Build the runner directly, for benches and tests that already own a
/// context and do not want to go through `AutoModel`.
pub fn load_runner(
    path: &Path,
    ctx: Arc<HipContext>,
    options: &LoadOptions,
) -> Result<HipGr00tRunner> {
    reject_unsupported_options(options)?;

    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let backbone_path = super::assets::resolve_assets(
        root,
        options.assets.get(BACKBONE_ASSET).map(PathBuf::as_path),
    )?;

    let config = Gr00tConfig::from_json_file(&root.join("config.json"))?;
    model::validate_servable(&config)?;
    let (backbone_config, mut weights) =
        super::weights::host::Gr00tWeights::from_safetensors(&config, &backbone_path, root)?;

    eprintln!(
        "[apxinf] GR00T N1.7 model_variant=bf16 (hip): {} Qwen layers, {} DiT blocks, {} flow steps",
        config.select_layer, config.diffusion.num_layers, config.num_inference_timesteps
    );
    // Build the BF16 blocks, then let the model take the embodiment category
    // tensors out of the host weights before the rest is dropped: on gfx1151
    // host RAM is device DRAM, so retained host copies of uploaded tensors
    // are a real memory cost.
    let blocks = model::blocks::Bf16Blocks::new(ctx, config, backbone_config, &weights)?;
    let model = model::build_bf16_model(blocks, &mut weights)?;
    drop(weights);
    let runner = HipGr00tRunner::new(Rc::new(model));
    debug_assert!(runner.contract().action_shape[1] > 0);
    Ok(runner)
}

/// Refuse options this port cannot honour, rather than loading something the
/// caller did not ask for.
fn reject_unsupported_options(options: &LoadOptions) -> Result<()> {
    if options.synthetic.is_some() || options.config.is_some() {
        return Err(Error::Other(
            "GR00T does not support the PI0.5 synthetic/config load options".into(),
        ));
    }
    if options.uniform_fp8_scale.is_some() || options.calibration_path.is_some() {
        return Err(Error::Other(
            "GR00T on HIP is BF16 only: gfx1151 has no FP8 tensor cores, so FP8 calibration \
             options cannot be honoured"
                .into(),
        ));
    }
    match options.precision {
        ModelPrecision::Auto | ModelPrecision::Bf16 => {}
        other => {
            return Err(Error::Other(format!(
                "GR00T on HIP supports BF16 only, got precision {other:?}"
            )))
        }
    }
    match options.model_variant.as_deref().unwrap_or("auto") {
        "auto" | "bf16" => {}
        other => {
            return Err(Error::Other(format!(
                "GR00T on HIP supports model_variant auto or bf16, got {other}"
            )))
        }
    }
    if let Some(name) = options
        .assets
        .keys()
        .find(|name| name.as_str() != BACKBONE_ASSET)
    {
        return Err(Error::Other(format!(
            "GR00T does not recognize the named asset {name:?}; expected only {BACKBONE_ASSET:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> LoadOptions {
        LoadOptions::default()
    }

    #[test]
    fn bf16_and_auto_are_the_only_accepted_precisions() {
        reject_unsupported_options(&options()).unwrap();
        let mut bf16 = options();
        bf16.precision = ModelPrecision::Bf16;
        reject_unsupported_options(&bf16).unwrap();
        let mut variant = options();
        variant.model_variant = Some("bf16".into());
        reject_unsupported_options(&variant).unwrap();
    }

    #[test]
    fn quantized_requests_are_refused_with_the_hardware_reason() {
        // gfx1151 has no FP8 tensor cores. Silently serving BF16 instead would
        // make a latency comparison meaningless and hide the missing support.
        let mut fp8 = options();
        fp8.precision = ModelPrecision::Fp8;
        let error = reject_unsupported_options(&fp8).unwrap_err().to_string();
        assert!(error.contains("BF16 only"), "{error}");

        let mut calibrated = options();
        calibrated.calibration_path = Some(PathBuf::from("/tmp/calibration.json"));
        let error = reject_unsupported_options(&calibrated)
            .unwrap_err()
            .to_string();
        assert!(error.contains("FP8 tensor cores"), "{error}");

        let mut variant = options();
        variant.model_variant = Some("fp8_static".into());
        assert!(reject_unsupported_options(&variant).is_err());
    }

    #[test]
    fn only_the_backbone_asset_is_recognized() {
        let mut backbone = options();
        backbone
            .assets
            .insert(BACKBONE_ASSET.into(), PathBuf::from("/models/cosmos"));
        reject_unsupported_options(&backbone).unwrap();

        let mut unknown = options();
        unknown
            .assets
            .insert("tokenizer".into(), PathBuf::from("/models/tok"));
        let error = reject_unsupported_options(&unknown).unwrap_err().to_string();
        assert!(error.contains("tokenizer"), "{error}");
    }
}
