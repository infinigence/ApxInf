//! Native Inkling-2-Mini (`inkling2_loop`) on Apple Metal.
//!
//! Native Metal/MLX execution engine with 3-pass looped attention residuals,
//! banded relative-bias NAX/flash attention, NVFP4/MXFP8/affine routed MoE,
//! compiled single-token decode & exact multi-row verify envelopes, certified
//! q8/g64 LM-head screen, and 3-layer DSpark MTP drafting.

pub mod cache;
pub mod config;
pub mod draft;
pub mod model;
pub mod weights;

pub use config::{Config, MAX_CONTEXT};
pub use model::{BenchmarkRun, MetalInkling2, MtpCycleStats, MtpStats};

use crate::{LoadOptions, LoadedModel};
use apxinf_core::{Backend, Device, Error, Result};
use apxinf_mlx::MlxBackend;
use std::{path::Path, sync::Arc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Nvfp4Mixed,
    Oq4eMtp,
}

impl Variant {
    pub fn parse(raw: Option<&str>) -> Result<Self> {
        match raw {
            None | Some("auto") | Some("nvfp4-mixed") => Ok(Self::Nvfp4Mixed),
            Some("oq4e-mtp") => Ok(Self::Oq4eMtp),
            Some(_) => Err(Error::Contract(
                "metal_inkling2 supports only 'nvfp4-mixed' (default) or 'oq4e-mtp'",
            )),
        }
    }
}

pub fn load(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Metal(_)) {
        return Err(Error::UnsupportedDevice(device));
    }
    let native = backend
        .as_any()
        .downcast_ref::<MlxBackend>()
        .ok_or(Error::Contract("Inkling2 requires the native MLX backend"))?
        .clone();
    if native.device() != device {
        return Err(Error::DeviceMismatch {
            expected: device,
            got: native.device(),
        });
    }
    let directory = if path.is_dir() {
        path
    } else {
        path.parent()
            .ok_or(Error::Contract("checkpoint file has no parent"))?
    };
    let model = MetalInkling2::load(directory, device, native, options.model_variant.as_deref())?;
    Ok(LoadedModel::text(Box::new(model)))
}
