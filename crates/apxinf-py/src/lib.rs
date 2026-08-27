//! PyO3 bindings for the ApxInf VLA runtime.
//!
//! This crate is the **binding layer** of the ApxInf Python frontend. It exposes
//! bare-model inference (no processor) directly to Python, `import`-ready, and
//! removes the subprocess + stdio hop the old serving path relied on.
//!
//! The public inference tier is numpy-in / numpy-out (host):
//!
//! * **L1** [`Model::infer_rgb`] — caller supplies resized RGB `uint8` images;
//!   vision→patches runs inside the Rust CUDA graph.
//!
//! **L0** [`Model::infer_patches`] (caller supplies pre-computed `patches`,
//! equivalent to a Rust `Observation(Patches)`) is implemented but **not part of
//! the public API**: it is exposed to Python only under the private
//! `_infer_patches` name for L0/L1 consistency tests, and may change or be
//! removed without notice.
//!
//! Both return the **normalized-domain** action as a `float32` numpy array of
//! shape `[action_horizon, action_dim]`. Inference accepts optional exact
//! caller-provided noise. When it is omitted, a model-owned counter-based stream
//! generates the initial latent directly in the runtime's device buffer. The
//! `*_seeded` variants remain available for explicitly keyed replay. No processor
//! lives here.
//!
//! The pi05 runtime is only registered on CUDA devices, so real inference
//! requires the `cuda` feature and a CUDA machine; without it the module still
//! imports and reports shape contracts, but `load` errors for pi05.

use std::cell::Cell;
use std::path::{Path, PathBuf};

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use apxinf_core::{Device, RngKey, Shape, Tensor};
use apxinf_model::{
    AutoModel, ImageLayout, LoadOptions, LoadedModel, ModelPrecision, Observation, Pi05Config,
    SyntheticWeights, VisionObservation, VlaRequest,
};

/// Map any Rust error into a Python `RuntimeError`.
fn runtime_err<E: std::fmt::Display>(error: E) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn parse_device(spec: &str) -> PyResult<Device> {
    let (kind, index) = match spec.split_once(':') {
        Some((kind, index)) => {
            let index = index.parse::<usize>().map_err(|_| {
                PyValueError::new_err(format!(
                    "apxinf_py.load: invalid device index in `{spec}` (expected e.g. `cuda:0`)"
                ))
            })?;
            (kind, index)
        }
        None => (spec, 0),
    };
    match kind {
        "cuda" => Ok(Device::Cuda(index)),
        "cpu" => Ok(Device::Cpu),
        other => Err(PyValueError::new_err(format!(
            "apxinf_py.load: unknown device `{other}` (expected cuda|cpu)"
        ))),
    }
}

fn parse_precision(spec: &str) -> PyResult<ModelPrecision> {
    match spec {
        "auto" => Ok(ModelPrecision::Auto),
        "fp8" => Ok(ModelPrecision::Fp8),
        "bf16" => Ok(ModelPrecision::Bf16),
        "int8" | "w8a8" => Ok(ModelPrecision::W8A8),
        other => Err(PyValueError::new_err(format!(
            "apxinf_py.load: unknown precision `{other}` (expected auto|fp8|bf16|int8)"
        ))),
    }
}

fn parse_layout(spec: &str) -> PyResult<ImageLayout> {
    match spec {
        "nhwc" => Ok(ImageLayout::Nhwc),
        "nchw" => Ok(ImageLayout::Nchw),
        other => Err(PyValueError::new_err(format!(
            "apxinf_py.infer_rgb: layout must be nhwc|nchw, got `{other}`"
        ))),
    }
}

/// Resolve the pi05 config the same way the loader does, so shape-contract
/// queries and input validation match the runtime `AutoModel` builds.
/// `LoadedModel` does not carry the config, so we read `config.json` (falling
/// back to `Pi05Config::default()` when absent), matching the runtime loader.
fn load_config(checkpoint: &Path) -> PyResult<Pi05Config> {
    let root = if checkpoint.is_dir() {
        checkpoint
    } else {
        checkpoint.parent().unwrap_or_else(|| Path::new("."))
    };
    let config_path = root.join("config.json");
    if config_path.is_file() {
        Pi05Config::from_json_file(&config_path).map_err(runtime_err)
    } else {
        Ok(Pi05Config::default())
    }
}

/// A loaded pi05 model handle. Holds the runtime plus the resolved config used
/// for shape-contract queries and input validation.
///
/// The pi05 runtime uses `Rc`/`RefCell` internally and is therefore not `Send`;
/// the handle is `unsendable` and must be used from the thread that created it.
#[pyclass(unsendable)]
pub struct Model {
    model: LoadedModel,
    config: Pi05Config,
    device: Device,
    sampling_seed: Cell<u64>,
    sampling_draw: Cell<u64>,
}

impl Model {
    fn patch_rows(&self) -> usize {
        self.config.num_views * self.config.patches_per_view()
    }

    fn patch_width(&self) -> usize {
        3 * self.config.patch_size * self.config.patch_size
    }

    fn validate_tokens(&self, token_ids: &[u32]) -> PyResult<()> {
        if token_ids.is_empty() {
            return Err(PyValueError::new_err(
                "apxinf_py.infer: token_ids must be non-empty",
            ));
        }
        if token_ids.len() > self.config.max_token_len {
            return Err(PyValueError::new_err(format!(
                "apxinf_py.infer: token_ids length {} exceeds max_token_len {}",
                token_ids.len(),
                self.config.max_token_len
            )));
        }
        Ok(())
    }

    /// Validate `noise` shape and build a CPU f32 tensor. The runtime normalizes
    /// f32 CPU tensors to its input dtype, so numpy f32 is accepted directly.
    fn noise_tensor(&self, noise: PyReadonlyArray2<'_, f32>) -> PyResult<Tensor> {
        let expected = [self.config.action_horizon, self.config.action_dim];
        let shape = noise.shape();
        if shape.len() != 2 || shape[0] != expected[0] || shape[1] != expected[1] {
            return Err(PyValueError::new_err(format!(
                "apxinf_py.infer: noise expected shape [{}, {}], got {:?}",
                expected[0], expected[1], shape
            )));
        }
        let data = noise.as_slice().map_err(|_| {
            PyValueError::new_err("apxinf_py.infer: noise must be C-contiguous float32")
        })?;
        Tensor::from_f32(
            Shape::new(vec![expected[0], expected[1]]),
            data,
        )
        .map_err(runtime_err)
    }

    fn action_array<'py>(
        &self,
        py: Python<'py>,
        flat: Vec<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let horizon = self.config.action_horizon;
        let dim = self.config.action_dim;
        if flat.len() != horizon * dim {
            return Err(PyRuntimeError::new_err(format!(
                "apxinf_py.infer: model returned {} values, expected {} ({}x{})",
                flat.len(),
                horizon * dim,
                horizon,
                dim
            )));
        }
        if let Some(bad) = flat.iter().find(|value| !value.is_finite()) {
            return Err(PyRuntimeError::new_err(format!(
                "apxinf_py.infer: model produced non-finite action value {bad}"
            )));
        }
        let array = Array2::from_shape_vec((horizon, dim), flat).map_err(runtime_err)?;
        Ok(array.into_pyarray_bound(py))
    }

    fn next_sampling_rng(&self) -> PyResult<RngKey> {
        let draw = self.sampling_draw.get();
        let next = draw
            .checked_add(1)
            .ok_or_else(|| PyRuntimeError::new_err("implicit sampling draw counter overflow"))?;
        self.sampling_draw.set(next);
        Ok(RngKey::new(self.sampling_seed.get(), 0, draw))
    }

    /// Run with an exact caller-provided latent. This is the correctness and
    /// OpenPI-parity path.
    fn run_provided<'py>(
        &self,
        py: Python<'py>,
        observation: Observation,
        latent: Tensor,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let request = VlaRequest::provided(&observation, &latent);
        let flat = self.model.infer_host_f32(&request).map_err(runtime_err)?;
        self.action_array(py, flat)
    }

    /// Run with a standard-normal latent generated into the prepared device
    /// buffer, avoiding a host allocation and H2D latent copy.
    fn run_generated<'py>(
        &self,
        py: Python<'py>,
        observation: Observation,
        rng: RngKey,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let request = VlaRequest::generated(&observation, rng);
        let flat = self.model.infer_host_f32(&request).map_err(runtime_err)?;
        self.action_array(py, flat)
    }
}

#[pymethods]
impl Model {
    /// Load a pi05 checkpoint through the unified `AutoModel` frontend.
    ///
    /// * `model` — model name, e.g. `"pi05"`.
    /// * `path` — checkpoint directory or index file.
    /// * `device` — `cuda:N` (default) or `cpu`.
    /// * `precision` — `auto` (default), `fp8`, `bf16`, or `int8`.
    /// * `calibration` / `tactics` — optional FP8 calibration / tactics json.
    /// * `sampling_seed` — seed for the implicit device-side noise stream used
    ///   when inference is called without `noise`.
    /// * `action_horizon` — override the checkpoint's chunk length. `None`
    ///   (default) runs the native `config.json` value; an explicit value wins
    ///   over it. The horizon is a sequence length, not a weight dimension, so
    ///   the same weights load and run at the requested chunk length.
    /// * `num_views` — serve fewer cameras than the checkpoint declares.
    ///
    /// `num_views` exists because a deployment often has fewer cameras than the
    /// checkpoint was trained with. Dropping the trailing views is numerically
    /// equivalent to openpi zero-padding and masking them — a masked view is
    /// excluded from attention and consumes no RoPE position, and the vision
    /// tower has no per-slot parameters — while saving one view's worth of patch
    /// tokens per step. Nothing weight-shaped depends on the count; it only sizes
    /// the prefix, so this is a load-time constant, not a per-request one.
    #[staticmethod]
    #[pyo3(signature = (model, path, device="cuda:0", precision="auto", calibration=None, tactics=None, action_horizon=None, num_views=None, sampling_seed=0))]
    fn load(
        model: &str,
        path: PathBuf,
        device: &str,
        precision: &str,
        calibration: Option<PathBuf>,
        tactics: Option<PathBuf>,
        action_horizon: Option<usize>,
        num_views: Option<usize>,
        sampling_seed: u64,
    ) -> PyResult<Self> {
        let device = parse_device(device)?;
        let mut config = load_config(&path)?;
        // Only hand the loader an explicit config when the caller actually
        // overrode something; otherwise it reads `config.json` itself, exactly
        // as before.
        let mut overridden = false;
        if let Some(horizon) = action_horizon {
            config.action_horizon = horizon;
            overridden = true;
        }
        if let Some(views) = num_views {
            if views == 0 || views > config.num_views {
                return Err(PyValueError::new_err(format!(
                    "apxinf_py.load: num_views={views} must be in 1..={} (the \
                     checkpoint's view count); a checkpoint cannot serve more \
                     cameras than it was trained on",
                    config.num_views
                )));
            }
            config.num_views = views;
            overridden = true;
        }
        // Validate once, after every override, so a combination that is
        // individually valid but jointly is not still fails here.
        if overridden {
            config.validate().map_err(runtime_err)?;
        }
        let options = LoadOptions {
            model_name: Some(model.to_owned()),
            precision: parse_precision(precision)?,
            calibration_path: calibration,
            tuning_path: tactics,
            config: overridden.then(|| config.clone()),
            ..LoadOptions::default()
        };
        let loaded = AutoModel::load_model(device, &path, &options).map_err(runtime_err)?;
        Ok(Self {
            model: loaded,
            config,
            device,
            sampling_seed: Cell::new(sampling_seed),
            sampling_draw: Cell::new(0),
        })
    }

    /// Build a **checkpoint-free** pi05 model with deterministic random weights.
    ///
    /// Latency depends only on tensor shape and dtype, so the L0/L1 engine can be
    /// benchmarked with no trained weights and no checkpoint on disk. The
    /// architecture is `Pi05Config::default()` overlaid with the given shape
    /// parameters, letting one call sweep 2/3-view, image size, horizon, etc.
    ///
    /// * `model` — model name, e.g. `"pi05"`.
    /// * `device` — `cuda:N` (default) or `cpu`.
    /// * `precision` — `bf16` (default), `fp8`, or `int8`.
    /// * `calibration` — for FP8: `"uniform:<scale>"` for a uniform activation
    ///   scale (no calibration file), or a path to a calibration json.
    /// * `tactics` — optional FP8 tactics json (a synthetic FP8 run falls back to
    ///   the kernel's default tactic when omitted).
    /// * `seed` — RNG seed for reproducible weights.
    /// * `sampling_seed` — independent seed for implicit device-side noise.
    #[staticmethod]
    #[pyo3(signature = (
        model,
        device="cuda:0",
        precision="bf16",
        num_views=2,
        image_size=224,
        action_horizon=10,
        action_dim=32,
        num_flow_steps=10,
        max_token_len=200,
        calibration=None,
        tactics=None,
        seed=0,
        sampling_seed=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn random(
        model: &str,
        device: &str,
        precision: &str,
        num_views: usize,
        image_size: usize,
        action_horizon: usize,
        action_dim: usize,
        num_flow_steps: usize,
        max_token_len: usize,
        calibration: Option<String>,
        tactics: Option<PathBuf>,
        seed: u64,
        sampling_seed: u64,
    ) -> PyResult<Self> {
        let device = parse_device(device)?;
        let config = Pi05Config {
            num_views,
            image_size,
            action_horizon,
            action_dim,
            num_flow_steps,
            max_token_len,
            ..Pi05Config::default()
        };
        config.validate().map_err(runtime_err)?;

        let (calibration_path, uniform_fp8_scale) = match calibration {
            Some(spec) if spec.starts_with("uniform:") => {
                let scale = spec["uniform:".len()..].parse::<f32>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "apxinf_py.random: invalid uniform FP8 scale in `{spec}`"
                    ))
                })?;
                (None, Some(scale))
            }
            Some(spec) => (Some(PathBuf::from(spec)), None),
            None => (None, None),
        };

        let options = LoadOptions {
            model_name: Some(model.to_owned()),
            precision: parse_precision(precision)?,
            text_weight_dtype: None,
            calibration_path,
            tuning_path: tactics,
            config: Some(config.clone()),
            synthetic: Some(SyntheticWeights { seed }),
            uniform_fp8_scale,
            ..LoadOptions::default()
        };
        let loaded = AutoModel::load_model(device, Path::new(""), &options)
            .map_err(runtime_err)?;
        Ok(Self {
            model: loaded,
            config,
            device,
            sampling_seed: Cell::new(sampling_seed),
            sampling_draw: Cell::new(0),
        })
    }

    /// **L0** (internal, not public API) — infer from pre-computed patches. No
    /// processor is applied. Exposed to Python as the private `_infer_patches`
    /// for L0/L1 consistency tests only; may change or be removed without notice.
    ///
    /// * `patches` — `float32` `[num_views * patches_per_view, 3 * patch_size^2]`.
    /// * `token_ids` — `uint32` `[token_count]` (1..=max_token_len).
    /// * `noise` — optional `float32` `[action_horizon, action_dim]`; omission
    ///   uses the model's internal device-side sampling stream.
    ///
    /// Returns the normalized-domain action, `float32` `[action_horizon, action_dim]`.
    #[pyo3(name = "_infer_patches", signature = (patches, token_ids, noise=None))]
    fn infer_patches<'py>(
        &self,
        py: Python<'py>,
        patches: PyReadonlyArray2<'py, f32>,
        token_ids: PyReadonlyArray1<'py, u32>,
        noise: Option<PyReadonlyArray2<'py, f32>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let expected = [self.patch_rows(), self.patch_width()];
        let shape = patches.shape();
        if shape.len() != 2 || shape[0] != expected[0] || shape[1] != expected[1] {
            return Err(PyValueError::new_err(format!(
                "apxinf_py._infer_patches: patches expected shape [{}, {}], got {:?}",
                expected[0], expected[1], shape
            )));
        }
        let patch_data = patches.as_slice().map_err(|_| {
            PyValueError::new_err("apxinf_py._infer_patches: patches must be C-contiguous float32")
        })?;
        let tokens = token_ids
            .as_slice()
            .map_err(|_| {
                PyValueError::new_err("apxinf_py._infer_patches: token_ids must be C-contiguous uint32")
            })?
            .to_vec();
        self.validate_tokens(&tokens)?;

        let patch_tensor = Tensor::from_f32(
            Shape::new(vec![expected[0], expected[1]]),
            patch_data,
        )
        .map_err(runtime_err)?;

        let observation = Observation {
            vision: VisionObservation::Patches(patch_tensor),
            token_ids: tokens,
        };
        match noise {
            Some(noise) => {
                let noise_tensor = self.noise_tensor(noise)?;
                self.run_provided(py, observation, noise_tensor)
            }
            None => self.run_generated(py, observation, self.next_sampling_rng()?),
        }
    }

    /// **L0** seeded variant for runtime tests. The initial standard-normal
    /// latent is generated directly in the prepared CUDA input buffer.
    #[pyo3(name = "_infer_patches_seeded", signature = (patches, token_ids, seed, sequence=0, draw=0))]
    fn infer_patches_seeded<'py>(
        &self,
        py: Python<'py>,
        patches: PyReadonlyArray2<'py, f32>,
        token_ids: PyReadonlyArray1<'py, u32>,
        seed: u64,
        sequence: u64,
        draw: u64,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let expected = [self.patch_rows(), self.patch_width()];
        let shape = patches.shape();
        if shape.len() != 2 || shape[0] != expected[0] || shape[1] != expected[1] {
            return Err(PyValueError::new_err(format!(
                "apxinf_py._infer_patches_seeded: patches expected shape [{}, {}], got {:?}",
                expected[0], expected[1], shape
            )));
        }
        let patch_data = patches.as_slice().map_err(|_| {
            PyValueError::new_err(
                "apxinf_py._infer_patches_seeded: patches must be C-contiguous float32",
            )
        })?;
        let tokens = token_ids
            .as_slice()
            .map_err(|_| {
                PyValueError::new_err(
                    "apxinf_py._infer_patches_seeded: token_ids must be C-contiguous uint32",
                )
            })?
            .to_vec();
        self.validate_tokens(&tokens)?;
        let patch_tensor = Tensor::from_f32(
            Shape::new(vec![expected[0], expected[1]]),
            patch_data,
        )
        .map_err(runtime_err)?;
        let observation = Observation {
            vision: VisionObservation::Patches(patch_tensor),
            token_ids: tokens,
        };
        self.run_generated(py, observation, RngKey::new(seed, sequence, draw))
    }

    /// **L1** — infer from resized RGB `uint8` images; vision→patches runs in the
    /// Rust CUDA graph.
    ///
    /// * `rgb_u8` — `uint8` images, `num_views * image_size * image_size * 3`
    ///   bytes total, in `layout` order.
    /// * `layout` — `"nhwc"` or `"nchw"`.
    /// * `token_ids` — `uint32` `[token_count]`.
    /// * `noise` — optional `float32` `[action_horizon, action_dim]`; omission
    ///   uses the model's internal device-side sampling stream.
    ///
    /// Returns the normalized-domain action, `float32` `[action_horizon, action_dim]`.
    #[pyo3(signature = (rgb_u8, layout, token_ids, noise=None))]
    fn infer_rgb<'py>(
        &self,
        py: Python<'py>,
        rgb_u8: PyReadonlyArrayDyn<'py, u8>,
        layout: &str,
        token_ids: PyReadonlyArray1<'py, u32>,
        noise: Option<PyReadonlyArray2<'py, f32>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let layout = parse_layout(layout)?;
        let expected_bytes =
            self.config.num_views * self.config.image_size * self.config.image_size * 3;
        let bytes = rgb_u8
            .as_slice()
            .map_err(|_| {
                PyValueError::new_err("apxinf_py.infer_rgb: rgb_u8 must be C-contiguous uint8")
            })?
            .to_vec();
        if bytes.len() != expected_bytes {
            return Err(PyValueError::new_err(format!(
                "apxinf_py.infer_rgb: rgb_u8 expected {} bytes ({} views x {}x{}x3), got {}",
                expected_bytes,
                self.config.num_views,
                self.config.image_size,
                self.config.image_size,
                bytes.len()
            )));
        }
        let tokens = token_ids
            .as_slice()
            .map_err(|_| {
                PyValueError::new_err("apxinf_py.infer_rgb: token_ids must be C-contiguous uint32")
            })?
            .to_vec();
        self.validate_tokens(&tokens)?;
        let observation = Observation {
            vision: VisionObservation::RgbU8 { bytes, layout },
            token_ids: tokens,
        };
        match noise {
            Some(noise) => {
                let noise_tensor = self.noise_tensor(noise)?;
                self.run_provided(py, observation, noise_tensor)
            }
            None => self.run_generated(py, observation, self.next_sampling_rng()?),
        }
    }

    /// Seeded L1 inference. This avoids creating or transferring a host noise
    /// array and fills the latent on the runtime's CUDA stream.
    #[pyo3(signature = (rgb_u8, layout, token_ids, seed, sequence=0, draw=0))]
    fn infer_rgb_seeded<'py>(
        &self,
        py: Python<'py>,
        rgb_u8: PyReadonlyArrayDyn<'py, u8>,
        layout: &str,
        token_ids: PyReadonlyArray1<'py, u32>,
        seed: u64,
        sequence: u64,
        draw: u64,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let layout = parse_layout(layout)?;
        let expected_bytes =
            self.config.num_views * self.config.image_size * self.config.image_size * 3;
        let bytes = rgb_u8
            .as_slice()
            .map_err(|_| {
                PyValueError::new_err(
                    "apxinf_py.infer_rgb_seeded: rgb_u8 must be C-contiguous uint8",
                )
            })?
            .to_vec();
        if bytes.len() != expected_bytes {
            return Err(PyValueError::new_err(format!(
                "apxinf_py.infer_rgb_seeded: rgb_u8 expected {} bytes ({} views x {}x{}x3), got {}",
                expected_bytes,
                self.config.num_views,
                self.config.image_size,
                self.config.image_size,
                bytes.len()
            )));
        }
        let tokens = token_ids
            .as_slice()
            .map_err(|_| {
                PyValueError::new_err(
                    "apxinf_py.infer_rgb_seeded: token_ids must be C-contiguous uint32",
                )
            })?
            .to_vec();
        self.validate_tokens(&tokens)?;
        let observation = Observation {
            vision: VisionObservation::RgbU8 { bytes, layout },
            token_ids: tokens,
        };
        self.run_generated(py, observation, RngKey::new(seed, sequence, draw))
    }

    /// Reset the implicit device-side sampling stream used when `noise=None`.
    /// Supplying a seed also replaces the configured seed.
    #[pyo3(signature = (seed=None))]
    fn reset_sampling(&self, seed: Option<u64>) {
        if let Some(seed) = seed {
            self.sampling_seed.set(seed);
        }
        self.sampling_draw.set(0);
    }

    /// Device string, e.g. `"cuda:0"`.
    #[getter]
    fn device(&self) -> String {
        match self.device {
            Device::Cuda(index) => format!("cuda:{index}"),
            Device::Cpu => "cpu".to_string(),
        }
    }

    #[getter]
    fn action_dim(&self) -> usize {
        self.config.action_dim
    }

    #[getter]
    fn action_horizon(&self) -> usize {
        self.config.action_horizon
    }

    #[getter]
    fn num_views(&self) -> usize {
        self.config.num_views
    }

    #[getter]
    fn image_size(&self) -> usize {
        self.config.image_size
    }

    #[getter]
    fn patch_size(&self) -> usize {
        self.config.patch_size
    }

    #[getter]
    fn patches_per_view(&self) -> usize {
        self.config.patches_per_view()
    }

    #[getter]
    fn max_token_len(&self) -> usize {
        self.config.max_token_len
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(device={}, action=[{}, {}], views={}, image={}, patch={})",
            self.device(),
            self.config.action_horizon,
            self.config.action_dim,
            self.config.num_views,
            self.config.image_size,
            self.config.patch_size,
        )
    }
}

#[pymodule]
fn apxinf_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_forms() {
        assert!(matches!(parse_device("cuda").unwrap(), Device::Cuda(0)));
        assert!(matches!(parse_device("cuda:1").unwrap(), Device::Cuda(1)));
        assert!(matches!(parse_device("cpu").unwrap(), Device::Cpu));
        assert!(parse_device("tpu").is_err());
        assert!(parse_device("cuda:x").is_err());
    }

    #[test]
    fn parses_precision_forms() {
        assert_eq!(parse_precision("auto").unwrap(), ModelPrecision::Auto);
        assert_eq!(parse_precision("fp8").unwrap(), ModelPrecision::Fp8);
        assert_eq!(parse_precision("bf16").unwrap(), ModelPrecision::Bf16);
        assert_eq!(parse_precision("int8").unwrap(), ModelPrecision::W8A8);
        assert_eq!(parse_precision("w8a8").unwrap(), ModelPrecision::W8A8);
        assert!(parse_precision("fp4").is_err());
    }

    #[test]
    fn parses_layout_forms() {
        assert_eq!(parse_layout("nhwc").unwrap(), ImageLayout::Nhwc);
        assert_eq!(parse_layout("nchw").unwrap(), ImageLayout::Nchw);
        assert!(parse_layout("hwcn").is_err());
    }

    #[test]
    fn missing_config_falls_back_to_default() {
        let config = load_config(Path::new("/nonexistent/checkpoint")).unwrap();
        assert_eq!(config, Pi05Config::default());
    }
}
