//! PI0.5 asset loading and compute implementation construction.
use super::backend::RuntimeBackend;
use super::blocks::{Bf16Blocks, Fp8StaticBlocks, Int8DynamicBlocks};
use super::network::Pi05Network;
use super::*;
use apxinf_core::{Backend, Result, Tensor};
use std::sync::Arc;
pub fn build_bf16_network(
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    weights: Arc<Bf16Weights>,
) -> Result<Arc<Pi05Network<Bf16Blocks>>> {
    Ok(Arc::new(Pi05Network::from_blocks(Bf16Blocks::new(
        backend, config, weights,
    )?)))
}
pub fn build_fp8_static_network(
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    weights: Arc<Fp8StaticWeights>,
    scales: Arc<Fp8StaticActivationScales>,
) -> Result<Arc<Pi05Network<Fp8StaticBlocks>>> {
    Ok(Arc::new(Pi05Network::from_blocks(Fp8StaticBlocks::new(
        backend, config, weights, scales,
    )?)))
}
pub fn build_int8_dynamic_network(
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    weights: Arc<Int8DynamicWeights>,
) -> Result<Arc<Pi05Network<Int8DynamicBlocks>>> {
    Ok(Arc::new(Pi05Network::from_blocks(Int8DynamicBlocks::new(
        backend, config, weights,
    )?)))
}
pub fn upload_time_embeddings_bf16(
    config: &Pi05Config,
    backend: &dyn Backend,
) -> Result<Vec<Tensor>> {
    (0..config.num_flow_steps)
        .map(|step| {
            let time = config.flow_start_time * (1.0 - step as f32 / config.num_flow_steps as f32);
            let values = sinusoidal_time_embedding(
                time,
                config.action_expert.width,
                config.time_min_period,
                config.time_max_period,
            )
            .into_iter()
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>();
            backend.to_device(&Tensor::from_bf16(
                vec![1, config.action_expert.width],
                &values,
            )?)
        })
        .collect()
}
pub fn upload_time_embeddings_fp8_static(
    config: &Pi05Config,
    backend: &dyn Backend,
) -> Result<Vec<Tensor>> {
    (0..config.num_flow_steps)
        .map(|step| {
            let time = config.flow_start_time * (1.0 - step as f32 / config.num_flow_steps as f32);
            let values = sinusoidal_time_embedding(
                time,
                config.action_expert.width,
                config.time_min_period,
                config.time_max_period,
            )
            .into_iter()
            .map(half::f16::from_f32)
            .collect::<Vec<_>>();
            let tensor = Tensor::from_f16(vec![1, config.action_expert.width], &values)?;
            backend.to_device(&tensor)
        })
        .collect()
}

pub use upload_time_embeddings_bf16 as upload_time_embeddings_int8_dynamic;

use super::backend::DeviceBuffer;
use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::vla::InferenceSpec;
use apxinf_core::DType;
use apxinf_core::{Device, Error};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
pub(super) fn load_registered(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    Ok(LoadedModel::Vla(Box::new(load_session(
        path, backend, options,
    )?)))
}

pub(super) fn load_session(
    path: &Path,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<Pi05Session> {
    let backend = crate::accelerator::cuda::downcast_arc(backend)
        .ok_or_else(|| Error::Other("PI0.5 is only registered for CUDA".into()))?;
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
            "PI0.5 uses compute_variant instead of precision".into(),
        ));
    }
    let compute_variant = options
        .compute_variant
        .as_deref()
        .unwrap_or("auto")
        .parse::<ComputeVariant>()?
        .resolve(
            cuda.context().caps().sm,
            calibration_path.is_some() || options.uniform_fp8_scale.is_some(),
        );
    eprintln!(
        "[apxinf] PI0.5 compute_variant={}",
        compute_variant.as_str()
    );

    let compute = match compute_variant {
        ComputeVariant::Fp8Static => {
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
                config.language_dual_geglu_shape_possible(),
            )?);
            let time_embeddings = Arc::new(upload_time_embeddings_fp8_static(&config, &*backend)?);
            LoadedCompute::Fp8Static {
                network: build_fp8_static_network(
                    Arc::clone(&backend),
                    Arc::clone(&config),
                    weights,
                    scales,
                )?,
                time_embeddings,
            }
        }
        ComputeVariant::Bf16 => {
            let weights = Arc::new(Bf16Weights::from_host(
                &host_weights,
                &*backend,
                config.language_dual_geglu_shape_possible(),
            )?);
            let time_embeddings = Arc::new(upload_time_embeddings_bf16(&config, &*backend)?);
            LoadedCompute::Bf16 {
                network: build_bf16_network(Arc::clone(&backend), Arc::clone(&config), weights)?,
                time_embeddings,
            }
        }
        ComputeVariant::Int8Dynamic => {
            let weights = Arc::new(Int8DynamicWeights::from_host(&host_weights, cuda)?);
            let time_embeddings =
                Arc::new(upload_time_embeddings_int8_dynamic(&config, &*backend)?);
            LoadedCompute::Int8Dynamic {
                network: build_int8_dynamic_network(
                    Arc::clone(&backend),
                    Arc::clone(&config),
                    weights,
                )?,
                time_embeddings,
            }
        }
        ComputeVariant::Auto => unreachable!("automatic precision was resolved"),
    };

    Ok(Pi05Session {
        backend,
        config,
        compute,
        prepared: RefCell::new(None),
    })
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

#[derive(Clone)]
pub(super) enum LoadedCompute {
    Fp8Static {
        network: Fp8StaticNetwork,
        time_embeddings: Arc<Vec<Tensor>>,
    },
    Bf16 {
        network: Bf16Network,
        time_embeddings: Arc<Vec<Tensor>>,
    },
    Int8Dynamic {
        network: Int8DynamicNetwork,
        time_embeddings: Arc<Vec<Tensor>>,
    },
}

impl LoadedCompute {
    pub(super) fn input_dtype(&self) -> DType {
        match self {
            Self::Fp8Static { .. } => DType::F16,
            Self::Bf16 { .. } | Self::Int8Dynamic { .. } => DType::BF16,
        }
    }

    pub(super) fn captured_patch_dtype(&self, raw_rgb: bool) -> DType {
        match (self, raw_rgb) {
            (Self::Fp8Static { .. }, true) => DType::F8E4M3,
            _ => self.input_dtype(),
        }
    }

    pub(super) fn infer(
        &self,
        patches: &Tensor,
        token_ids: &DeviceBuffer,
        token_count: usize,
        noise: &Tensor,
        prequantized_fp8_patches: bool,
    ) -> Result<Tensor> {
        match self {
            Self::Fp8Static {
                network,
                time_embeddings,
                ..
            } if prequantized_fp8_patches => {
                network.infer_native(patches, token_ids, token_count, noise, time_embeddings)
            }
            Self::Fp8Static {
                network,
                time_embeddings,
                ..
            } => network.infer(patches, token_ids, token_count, noise, time_embeddings),
            Self::Bf16 {
                network,
                time_embeddings,
            } => network.infer(patches, token_ids, token_count, noise, time_embeddings),
            Self::Int8Dynamic {
                network,
                time_embeddings,
            } => network.infer(patches, token_ids, token_count, noise, time_embeddings),
        }
    }

    pub(super) fn capture(
        &self,
        spec: &InferenceSpec,
        patches: &Tensor,
        token_ids: &DeviceBuffer,
        noise: &Tensor,
    ) -> Result<CapturedGraph> {
        let input = match spec.image_layout {
            Some(layout) => {
                super::prepare::CaptureInput::Rgb(super::session::kernel_image_layout(layout))
            }
            None => super::prepare::CaptureInput::Patches(patches),
        };
        match self {
            Self::Fp8Static {
                network,
                time_embeddings,
                ..
            } => super::prepare::capture(
                network,
                input,
                token_ids,
                spec.token_count,
                noise,
                time_embeddings,
            ),
            Self::Bf16 {
                network,
                time_embeddings,
            } => super::prepare::capture(
                network,
                input,
                token_ids,
                spec.token_count,
                noise,
                time_embeddings,
            ),
            Self::Int8Dynamic {
                network,
                time_embeddings,
            } => super::prepare::capture(
                network,
                input,
                token_ids,
                spec.token_count,
                noise,
                time_embeddings,
            ),
        }
    }
}

impl LoadedCompute {
    pub(super) fn preprocess_rgb(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: super::Pi05ImageLayout,
    ) -> Result<()> {
        use super::prepare::PrepareBlocks;
        match self {
            Self::Bf16 { network, .. } => network.blocks.preprocess(images, patches, layout),
            Self::Fp8Static { network, .. } => network.blocks.preprocess(images, patches, layout),
            Self::Int8Dynamic { network, .. } => network.blocks.preprocess(images, patches, layout),
        }
    }
    pub(super) fn calibrate(
        &self,
        patches: &Tensor,
        tokens: &DeviceBuffer,
        count: usize,
        noise: &Tensor,
    ) -> Result<std::collections::BTreeMap<String, f32>> {
        match self {
            Self::Bf16 {
                network,
                time_embeddings,
            } => network.calibrate(patches, tokens, count, noise, time_embeddings),
            _ => Err(Error::Other(
                "PI0.5 calibration requires compute_variant=bf16".into(),
            )),
        }
    }
}
