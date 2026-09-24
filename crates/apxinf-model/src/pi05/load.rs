//! Load PI0.5 assets and assemble the computation and model_runner modules.
use super::model::{
    build_bf16_model_with_policies, build_fp8_static_model_with_policies,
    build_int8_dynamic_model_with_policies, L3Policies, ModelVariant,
};
use super::*;
use crate::auto::{cuda_recipe_options, LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Device, Error};
use apxinf_core::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) fn load_with_cuda_new(
    path: &Path,
    device: Device,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    let Device::Cuda(device_id) = device else {
        return Err(Error::Other("PI0.5 requires CUDA".into()));
    };
    let context = Arc::new(backend::Context::new(device_id).map_err(Error::Cuda)?);
    Ok(LoadedModel::Vla(Box::new(load_model_runner(
        path, context, options,
    )?)))
}

pub(super) fn load_model_runner(
    path: &Path,
    backend: Arc<backend::Context>,
    options: &LoadOptions,
) -> Result<Pi05ModelRunner> {
    let policies = cuda_l3_policies(options)?;
    load_model_runner_with_policies(path, backend, options, policies)
}

fn load_model_runner_with_policies(
    path: &Path,
    backend: Arc<backend::Context>,
    options: &LoadOptions,
    policies: L3Policies,
) -> Result<Pi05ModelRunner> {
    let cuda = &*backend;
    let root = artifact_root(path);
    let config_path = root.join("config.json");
    let config = Arc::new(if let Some(cfg) = options.config.clone() {
        cfg
    } else if config_path.is_file() {
        Pi05Config::from_json_file(&config_path)?
    } else {
        Pi05Config::default()
    });
    let synthetic = options.synthetic;
    let host_weights = match synthetic {
        Some(synthetic) => Pi05Weights::synthetic(&config, synthetic.seed)?,
        None => Pi05Weights::from_safetensors(&config, path)?,
    };
    // Synthetic (checkpoint-free) loads must not pick up stray calibration/tuning
    // files from the working directory; only honor explicitly passed paths.
    let calibration_path = options.calibration_path.clone().or_else(|| {
        (synthetic.is_none())
            .then(|| existing(root.join("calibration.json")))
            .flatten()
    });
    if options.precision != ModelPrecision::Auto {
        return Err(Error::Other(
            "PI0.5 uses model_variant instead of precision".into(),
        ));
    }
    let sm = cuda.caps().sm;
    let model_variant = options
        .model_variant
        .as_deref()
        .unwrap_or("auto")
        .parse::<ModelVariantChoice>()?
        .resolve(
            sm,
            calibration_path.is_some() || options.uniform_fp8_scale.is_some(),
        )
        .ensure_supported(sm)?;
    eprintln!("[apxinf] PI0.5 model_variant={}", model_variant.as_str());

    let model = match model_variant {
        ModelVariantChoice::Fp8Static => {
            let scales = if let Some(scale) = options.uniform_fp8_scale {
                Arc::new(Fp8StaticActivationScales::uniform(&config, scale)?)
            } else {
                let calibration_path = calibration_path.ok_or_else(|| {
                    Error::Other(
                        "FP8 PI0.5 requires LoadOptions.calibration_path or calibration.json"
                            .into(),
                    )
                })?;
                let checkpoint = checkpoint_identity(path)?;
                let calibration =
                    Fp8StaticCalibration::from_json_file(&calibration_path, &config, &checkpoint)?;
                Arc::new(Fp8StaticActivationScales::from_calibration(
                    &config,
                    &calibration,
                )?)
            };
            let weights = Arc::new(Fp8StaticWeights::from_host(
                &host_weights,
                &*backend,
            )?);
            let time_embeddings = Arc::new(upload_time_embeddings_fp8_static(&config, &*backend)?);
            ModelVariant::Fp8Static {
                model: build_fp8_static_model_with_policies(
                    Arc::clone(&backend),
                    Arc::clone(&config),
                    weights,
                    scales,
                    policies,
                )?,
                time_embeddings,
            }
        }
        ModelVariantChoice::Bf16 => {
            let weights = Arc::new(Bf16Weights::from_host(
                &host_weights,
                &*backend,
            )?);
            let time_embeddings = Arc::new(upload_time_embeddings_bf16(&config, &*backend)?);
            ModelVariant::Bf16 {
                model: build_bf16_model_with_policies(
                    Arc::clone(&backend),
                    Arc::clone(&config),
                    weights,
                    policies,
                )?,
                time_embeddings,
            }
        }
        ModelVariantChoice::Int8Dynamic => {
            let weights = Arc::new(Int8DynamicWeights::from_host(&host_weights, cuda)?);
            let time_embeddings =
                Arc::new(upload_time_embeddings_int8_dynamic(&config, &*backend)?);
            ModelVariant::Int8Dynamic {
                model: build_int8_dynamic_model_with_policies(
                    Arc::clone(&backend),
                    Arc::clone(&config),
                    weights,
                    policies,
                )?,
                time_embeddings,
            }
        }
        ModelVariantChoice::Auto => unreachable!("automatic precision was resolved"),
    };

    Ok(Pi05ModelRunner::new(backend, config, model))
}

fn cuda_l3_policies(options: &LoadOptions) -> Result<L3Policies> {
    let recipe = cuda_recipe_options(options)?;
    Ok(L3Policies::for_recipe(recipe.cache_dir, recipe.online_tune))
}

fn artifact_root(path: &Path) -> &Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    }
}

fn existing(path: PathBuf) -> Option<PathBuf> {
    path.is_file().then_some(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::SyntheticWeights;

    fn temporary_directory(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "apxinf-pi05-load-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn assert_recipe_policy(
        policies: &L3Policies,
        expected_cache: Option<&str>,
        expected_online_tune: bool,
    ) {
        assert_eq!(policies.gemm.cache_dir.as_deref(), expected_cache);
        assert_eq!(policies.attention.cache_dir.as_deref(), expected_cache);
        assert_eq!(policies.gemm.online_tune, expected_online_tune);
        assert_eq!(policies.attention.online_tune, expected_online_tune);
    }

    #[test]
    fn production_recipe_policy_preserves_disabled_autotune_and_cache_path() {
        let directory = temporary_directory("disabled-autotune");
        let cache = directory.join("explicit.recipes");
        let options = LoadOptions {
            tuning_path: Some(cache.clone()),
            autotune: false,
            ..LoadOptions::default()
        };

        let policies = cuda_l3_policies(&options).unwrap();
        assert_recipe_policy(&policies, cache.to_str(), false);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn production_recipe_policy_preserves_enabled_autotune_and_json_sibling() {
        let directory = temporary_directory("enabled-autotune");
        let legacy = directory.join("legacy-tactics.json");
        let options = LoadOptions {
            tuning_path: Some(legacy.clone()),
            autotune: true,
            ..LoadOptions::default()
        };

        let policies = cuda_l3_policies(&options).unwrap();
        assert_recipe_policy(&policies, legacy.with_extension("recipes").to_str(), true);
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn small_synthetic_config() -> Pi05Config {
        let transformer = GemmaVariantConfig {
            width: 8,
            depth: 1,
            mlp_dim: 16,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 8,
        };
        Pi05Config {
            action_dim: 2,
            action_horizon: 1,
            max_token_len: 2,
            num_flow_steps: 1,
            num_views: 1,
            image_size: 2,
            patch_size: 2,
            vision_width: 8,
            vision_depth: 1,
            vision_mlp_dim: 16,
            vision_heads: 1,
            vision_head_dim: 8,
            vocab_size: 8,
            language: transformer,
            action_expert: transformer,
            ..Pi05Config::default()
        }
    }

    #[test]
    #[ignore = "requires CUDA SM100+"]
    fn production_load_wires_recipe_policy_into_all_model_variants() {
        let backend = Arc::new(backend::Context::new(0).unwrap());
        let directory = temporary_directory("production-wiring");
        let cases = [
            ("bf16", false, "bf16.recipes", "bf16.recipes"),
            ("fp8_static", true, "fp8.json", "fp8.recipes"),
            ("int8_dynamic", false, "int8.recipes", "int8.recipes"),
        ];

        for (variant, online_tune, input_name, expected_name) in cases {
            let tuning_path = directory.join(input_name);
            let expected_cache = directory.join(expected_name);
            let runner = load_model_runner(
                Path::new("."),
                Arc::clone(&backend),
                &LoadOptions {
                    model_variant: Some(variant.into()),
                    tuning_path: Some(tuning_path),
                    autotune: online_tune,
                    config: Some(small_synthetic_config()),
                    synthetic: Some(SyntheticWeights { seed: 7 }),
                    uniform_fp8_scale: (variant == "fp8_static").then_some(1.0),
                    ..LoadOptions::default()
                },
            )
            .unwrap();
            let snapshot = runner.l3_policy_snapshot();
            assert_eq!(snapshot.gemm_cache_dir.as_deref(), expected_cache.to_str());
            assert_eq!(
                snapshot.attention_cache_dir.as_deref(),
                expected_cache.to_str()
            );
            assert_eq!(snapshot.gemm_online_tune, online_tune);
            assert_eq!(snapshot.attention_online_tune, online_tune);
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}
