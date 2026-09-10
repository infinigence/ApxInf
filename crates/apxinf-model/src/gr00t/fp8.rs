//! Internal static-FP8 contracts for GR00T N1.7.
//!
//! This module is intentionally private: selecting FP8 reuses the existing
//! [`crate::ModelPrecision`] option while calibration remains an offline
//! artifact. It does not add another model-facing runtime interface.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use apxinf_core::{Backend, DType, Error, Result, Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::accelerator::cuda::{kernels, DeviceBuffer, RuntimeBackend};

use super::Gr00tLinearWeights;

const CALIBRATION_SCHEMA: &str = "apxinf.gr00t-n1.7.fp8-calibration.v1";
const E4M3_MAX: f32 = 448.0;

/// Derives the per-tensor E4M3 weight scale that maps the maximum-magnitude
/// weight onto the largest representable E4M3 value. A zero-magnitude tensor
/// yields a unit scale so quantization stays finite. Shared by every FP8
/// weight path so the calibration math has a single definition and a single
/// place to test.
fn e4m3_weight_scale(max_abs: f32) -> f32 {
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / E4M3_MAX
    }
}

/// GR00T-local output-channel-quantized W8A8 weights.
///
/// The layout matches the model-neutral CUDA W8A8 kernel contract. Keeping the
/// small packing wrapper here avoids coupling one model family to Pi0.5's
/// private weight implementation.
pub(super) struct Gr00tInt8LinearWeights {
    weight_output_major: DeviceBuffer,
    weight_scales: Tensor,
    bias: Option<Tensor>,
    input_dim: usize,
    output_dim: usize,
}

impl std::fmt::Debug for Gr00tInt8LinearWeights {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Gr00tInt8LinearWeights")
            .field("weight_bytes", &self.weight_output_major.len())
            .field("weight_scales", &self.weight_scales.shape().dims())
            .field("has_bias", &self.bias.is_some())
            .field("input_dim", &self.input_dim)
            .field("output_dim", &self.output_dim)
            .finish()
    }
}

impl Gr00tInt8LinearWeights {
    fn from_host(weight: &Tensor, bias: Option<&Tensor>, backend: &RuntimeBackend) -> Result<Self> {
        let (quantized, scales, input_dim, output_dim) = quantize_output_channels(weight)?;
        let bytes = quantized
            .into_iter()
            .map(|value| value as u8)
            .collect::<Vec<_>>();
        let weight_output_major =
            DeviceBuffer::alloc(bytes.len(), backend.device_id()).map_err(Error::Cuda)?;
        weight_output_major
            .copy_from_host(&bytes)
            .map_err(Error::Cuda)?;
        let weight_scales = backend.to_device(&Tensor::from_f32(vec![output_dim], &scales)?)?;
        let bias = bias
            .map(|tensor| {
                if tensor.shape().dims() != [output_dim] || tensor.dtype() == DType::F8E4M3 {
                    return Err(Error::Other(format!(
                        "GR00T INT8 bias must be a non-FP8 vector of width {output_dim}, got {} {:?}",
                        tensor.dtype(),
                        tensor.shape().dims()
                    )));
                }
                let values = tensor
                    .to_f32_vec()?
                    .into_iter()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>();
                backend.to_device(&Tensor::from_bf16(vec![output_dim], &values)?)
            })
            .transpose()?;
        Ok(Self {
            weight_output_major,
            weight_scales,
            bias,
            input_dim,
            output_dim,
        })
    }

    fn gemm(&self, backend: &RuntimeBackend, activation: &Tensor) -> Result<Tensor> {
        kernels::gemm::w8a8(backend.context(), activation, self.as_kernel_view())
    }

    fn gemm_quantized(
        &self,
        backend: &RuntimeBackend,
        activation: &kernels::gemm::W8A8Activation,
    ) -> Result<Tensor> {
        kernels::gemm::gemm_quantized_w8a8(backend.context(), activation, self.as_kernel_view())
    }

    fn as_kernel_view(&self) -> kernels::gemm::W8A8WeightView<'_> {
        kernels::gemm::W8A8WeightView {
            values_i8: &self.weight_output_major,
            scales_f32: &self.weight_scales,
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            scale_mode: kernels::gemm::W8A8ScaleMode::DynamicRowPerOutputChannel,
            layout: kernels::gemm::W8A8Layout::OutputMajor,
        }
    }
}

fn quantize_output_channels(tensor: &Tensor) -> Result<(Vec<i8>, Vec<f32>, usize, usize)> {
    if tensor.dtype() == DType::F8E4M3 {
        return Err(Error::Other(
            "cannot quantize a scale-less E4M3 matrix to INT8".into(),
        ));
    }
    let dims = tensor.shape().dims();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(Error::Other(format!(
            "GR00T INT8 weight must be a non-empty matrix, got {dims:?}"
        )));
    }
    let (input_dim, output_dim) = (dims[0], dims[1]);
    let values = tensor.to_f32_vec()?;
    let mut quantized = vec![0i8; input_dim * output_dim];
    let mut scales = vec![0.0f32; output_dim];
    for output in 0..output_dim {
        let mut maximum = 0.0f32;
        for input in 0..input_dim {
            maximum = maximum.max(values[input * output_dim + output].abs());
        }
        let scale = (maximum / 127.0).max(1.0e-12);
        scales[output] = scale;
        for input in 0..input_dim {
            let value = (values[input * output_dim + output] / scale)
                .round()
                .clamp(-128.0, 127.0);
            quantized[output * input_dim + input] = value as i8;
        }
    }
    Ok((quantized, scales, input_dim, output_dim))
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CalibrationDocument {
    schema: String,
    checkpoint_config_sha256: String,
    fixture_manifest_sha256: String,
    activation_scales: BTreeMap<String, f32>,
}

/// Validated, immutable activation scales for one checkpoint and calibration
/// campaign. Scale names use complete checkpoint-style projection paths.
#[derive(Debug)]
pub(super) struct Gr00tFp8Calibration {
    checkpoint_config_sha256: String,
    fixture_manifest_sha256: String,
    activation_scales: BTreeMap<String, f32>,
}

#[derive(Clone, Debug)]
pub(super) struct Gr00tFp8Collector {
    output_path: std::path::PathBuf,
    checkpoint_config_sha256: String,
    fixture_manifest_sha256: String,
    activation_amax: Arc<Mutex<BTreeMap<String, f32>>>,
}

impl Gr00tFp8Collector {
    pub(super) fn from_env(checkpoint_path: &Path) -> Result<Option<Self>> {
        let Some(output_path) = std::env::var_os("APXINF_GR00T_FP8_CALIBRATION_OUTPUT") else {
            return Ok(None);
        };
        let fixture_manifest_sha256 =
            std::env::var("APXINF_GR00T_FP8_FIXTURE_SHA256").map_err(|_| {
                Error::Other(
                    "GR00T FP8 calibration capture requires APXINF_GR00T_FP8_FIXTURE_SHA256".into(),
                )
            })?;
        validate_sha256("fixture_manifest_sha256", &fixture_manifest_sha256)?;
        let config_path = checkpoint_path.join("config.json");
        let bytes = std::fs::read(&config_path).map_err(|error| {
            Error::Other(format!(
                "read GR00T checkpoint config {}: {error}",
                config_path.display()
            ))
        })?;
        Ok(Some(Self {
            output_path: output_path.into(),
            checkpoint_config_sha256: format!("{:x}", Sha256::digest(bytes)),
            fixture_manifest_sha256,
            activation_amax: Arc::new(Mutex::new(BTreeMap::new())),
        }))
    }

    pub(super) fn observe(
        &self,
        name: &str,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<()> {
        let input = if input.device() == apxinf_core::Device::Cpu {
            input.clone()
        } else {
            backend.synchronize()?;
            backend.to_cpu(input)?
        };
        let maximum = input
            .to_f32_vec()?
            .into_iter()
            .fold(0.0f32, |value, next| value.max(next.abs()));
        let mut values = self.activation_amax.lock().map_err(|_| {
            Error::Other("GR00T FP8 calibration collector mutex was poisoned".into())
        })?;
        values
            .entry(name.to_owned())
            .and_modify(|value| *value = value.max(maximum))
            .or_insert(maximum);
        Ok(())
    }

    pub(super) fn save(&self) -> Result<()> {
        let values = self.activation_amax.lock().map_err(|_| {
            Error::Other("GR00T FP8 calibration collector mutex was poisoned".into())
        })?;
        let activation_scales = values
            .iter()
            .map(|(name, maximum)| (name.clone(), (maximum / E4M3_MAX).max(f32::MIN_POSITIVE)))
            .collect();
        let document = CalibrationDocument {
            schema: CALIBRATION_SCHEMA.into(),
            checkpoint_config_sha256: self.checkpoint_config_sha256.clone(),
            fixture_manifest_sha256: self.fixture_manifest_sha256.clone(),
            activation_scales,
        };
        let bytes = serde_json::to_vec_pretty(&document)
            .map_err(|error| Error::Other(format!("serialize GR00T FP8 calibration: {error}")))?;
        std::fs::write(&self.output_path, bytes).map_err(|error| {
            Error::Other(format!(
                "write GR00T FP8 calibration {}: {error}",
                self.output_path.display()
            ))
        })
    }
}

impl Gr00tFp8Calibration {
    pub(super) fn from_json_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|error| {
            Error::Other(format!(
                "read GR00T FP8 calibration {}: {error}",
                path.display()
            ))
        })?;
        let document: CalibrationDocument = serde_json::from_slice(&bytes).map_err(|error| {
            Error::Other(format!(
                "parse GR00T FP8 calibration {}: {error}",
                path.display()
            ))
        })?;
        Self::from_document(document)
    }

    fn from_document(document: CalibrationDocument) -> Result<Self> {
        if document.schema != CALIBRATION_SCHEMA {
            return Err(Error::Other(format!(
                "unsupported GR00T FP8 calibration schema {:?}; expected {CALIBRATION_SCHEMA:?}",
                document.schema
            )));
        }
        validate_sha256(
            "checkpoint_config_sha256",
            &document.checkpoint_config_sha256,
        )?;
        validate_sha256("fixture_manifest_sha256", &document.fixture_manifest_sha256)?;
        if document.activation_scales.is_empty() {
            return Err(Error::Other(
                "GR00T FP8 calibration contains no activation scales".into(),
            ));
        }
        for (name, scale) in &document.activation_scales {
            if name.is_empty() || name.trim() != name {
                return Err(Error::Other(format!(
                    "invalid GR00T FP8 activation-scale name {name:?}"
                )));
            }
            if !scale.is_finite() || *scale <= 0.0 {
                return Err(Error::Other(format!(
                    "GR00T FP8 activation scale {name:?} must be finite and positive, got {scale}"
                )));
            }
        }
        Ok(Self {
            checkpoint_config_sha256: document.checkpoint_config_sha256,
            fixture_manifest_sha256: document.fixture_manifest_sha256,
            activation_scales: document.activation_scales,
        })
    }

    pub(super) fn scale(&self, name: &str) -> Result<f32> {
        self.activation_scales.get(name).copied().ok_or_else(|| {
            Error::Other(format!(
                "GR00T FP8 calibration is missing activation scale {name:?}"
            ))
        })
    }

    pub(super) fn fixture_manifest_sha256(&self) -> &str {
        &self.fixture_manifest_sha256
    }

    pub(super) fn validate_checkpoint(&self, config_path: &Path) -> Result<()> {
        let bytes = std::fs::read(config_path).map_err(|error| {
            Error::Other(format!(
                "read GR00T checkpoint config {}: {error}",
                config_path.display()
            ))
        })?;
        let actual = format!("{:x}", Sha256::digest(bytes));
        if actual != self.checkpoint_config_sha256 {
            return Err(Error::Other(format!(
                "GR00T FP8 calibration checkpoint mismatch: config {} hashes to {}, calibration requires {}",
                config_path.display(), actual, self.checkpoint_config_sha256
            )));
        }
        Ok(())
    }
}

/// Device-only linear weight. FP8 keeps BF16 bias/output so the existing
/// GR00T attention, residual and normalization contracts remain unchanged.
#[derive(Debug)]
pub(super) enum Gr00tDeviceLinearWeights {
    Bf16 {
        weights: Gr00tLinearWeights,
        calibration: Option<(String, Gr00tFp8Collector)>,
    },
    Fp8 {
        weight: Tensor,
        weight_scale: f32,
        activation_scale: f32,
        bias: Option<Tensor>,
    },
    /// SM80-family W8A8 path using the shared CUDA kernel contract. The output
    /// remains BF16, so surrounding GR00T operators are unchanged.
    W8A8 { weight: Gr00tInt8LinearWeights },
}

pub(super) enum Gr00tQuantizedLinearInput {
    Fp8(Tensor),
    W8A8(kernels::gemm::W8A8Activation),
}

impl Gr00tDeviceLinearWeights {
    pub(super) fn bf16(
        weights: Gr00tLinearWeights,
        name: &str,
        calibration: Option<&Gr00tFp8Collector>,
        backend: &RuntimeBackend,
    ) -> Result<Self> {
        Ok(Self::Bf16 {
            weights: Gr00tLinearWeights {
                weight: backend.to_device(&weights.weight)?,
                bias: backend.to_device(&weights.bias)?,
            },
            calibration: calibration.map(|collector| (name.to_owned(), collector.clone())),
        })
    }

    pub(super) fn fp8(
        weights: Gr00tLinearWeights,
        activation_scale: f32,
        backend: &RuntimeBackend,
    ) -> Result<Self> {
        if !activation_scale.is_finite() || activation_scale <= 0.0 {
            return Err(Error::Other(format!(
                "GR00T FP8 activation scale must be finite and positive, got {activation_scale}"
            )));
        }
        let maximum = weights
            .weight
            .to_f32_vec()?
            .into_iter()
            .fold(0.0f32, |value, next| value.max(next.abs()));
        let weight_scale = e4m3_weight_scale(maximum);
        let weight = backend.to_device(&weights.weight)?;
        let weight =
            kernels::quantization::quantize_bf16_e4m3(backend.context(), &weight, weight_scale)?;
        Ok(Self::Fp8 {
            weight,
            weight_scale,
            activation_scale,
            bias: Some(backend.to_device(&weights.bias)?),
        })
    }

    pub(super) fn fp8_matrix(
        weight: &Tensor,
        activation_scale: f32,
        backend: &RuntimeBackend,
    ) -> Result<Self> {
        if !activation_scale.is_finite() || activation_scale <= 0.0 {
            return Err(Error::Other(format!(
                "GR00T FP8 activation scale must be finite and positive, got {activation_scale}"
            )));
        }
        let maximum = weight
            .to_f32_vec()?
            .into_iter()
            .fold(0.0f32, |value, next| value.max(next.abs()));
        let weight_scale = e4m3_weight_scale(maximum);
        let weight = backend.to_device(weight)?;
        let weight =
            kernels::quantization::quantize_bf16_e4m3(backend.context(), &weight, weight_scale)?;
        Ok(Self::Fp8 {
            weight,
            weight_scale,
            activation_scale,
            bias: None,
        })
    }

    pub(super) fn w8a8(weights: Gr00tLinearWeights, backend: &RuntimeBackend) -> Result<Self> {
        Ok(Self::W8A8 {
            weight: Gr00tInt8LinearWeights::from_host(
                &weights.weight,
                Some(&weights.bias),
                backend,
            )?,
        })
    }

    pub(super) fn w8a8_matrix(weight: &Tensor, backend: &RuntimeBackend) -> Result<Self> {
        Ok(Self::W8A8 {
            weight: Gr00tInt8LinearWeights::from_host(weight, None, backend)?,
        })
    }

    pub(super) fn forward(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor> {
        match self {
            Self::Bf16 {
                weights,
                calibration,
            } => {
                if let Some((name, collector)) = calibration {
                    collector.observe(name, input, backend)?;
                }
                backend.matmul(input, &weights.weight)
            }
            Self::Fp8 {
                weight,
                weight_scale,
                activation_scale,
                ..
            } => {
                let input = kernels::quantization::quantize_bf16_e4m3(
                    backend.context(),
                    input,
                    *activation_scale,
                )?;
                kernels::gemm::fp8_bf16(
                    backend.context(),
                    &input,
                    *activation_scale,
                    kernels::gemm::Fp8WeightView {
                        values_e4m3: weight,
                        scale: *weight_scale,
                        dual_geglu_interleaved: false,
                        dual_geglu_auto_interleaved: None,
                    },
                )
            }
            Self::W8A8 { weight } => weight.gemm(backend, input),
        }
    }

    pub(super) fn activation_scale(&self) -> Option<f32> {
        match self {
            Self::Bf16 { .. } => None,
            Self::Fp8 {
                activation_scale, ..
            } => Some(*activation_scale),
            Self::W8A8 { .. } => None,
        }
    }

    pub(super) fn can_share_quantized_input_with(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Fp8 {
                    activation_scale: left,
                    ..
                },
                Self::Fp8 {
                    activation_scale: right,
                    ..
                },
            ) => left == right,
            (Self::W8A8 { weight: left }, Self::W8A8 { weight: right }) => {
                left.input_dim == right.input_dim
            }
            _ => false,
        }
    }

    pub(super) fn quantize_reusable_input(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Gr00tQuantizedLinearInput> {
        match self {
            Self::Fp8 {
                activation_scale, ..
            } => Ok(Gr00tQuantizedLinearInput::Fp8(
                kernels::quantization::quantize_bf16_e4m3(
                    backend.context(),
                    input,
                    *activation_scale,
                )?,
            )),
            Self::Bf16 { .. } => Err(Error::Other(
                "BF16 linear does not have a reusable quantized input".into(),
            )),
            Self::W8A8 { .. } => Ok(Gr00tQuantizedLinearInput::W8A8(
                kernels::gemm::quantize_w8a8_activation(backend.context(), input)?,
            )),
        }
    }

    pub(super) fn quantize_bias_gelu_reusable_input(
        &self,
        input: &Tensor,
        bias: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Gr00tQuantizedLinearInput> {
        match self {
            Self::Fp8 {
                activation_scale, ..
            } => Ok(Gr00tQuantizedLinearInput::Fp8(
                kernels::activation::bias_gelu_quant_bf16_e4m3(
                    backend.context(),
                    input,
                    bias,
                    *activation_scale,
                )?,
            )),
            Self::Bf16 { .. } => Err(Error::Other(
                "GR00T fused bias GELU quantization requires quantized output weights".into(),
            )),
            Self::W8A8 { .. } => Err(Error::Other(
                "GR00T fused bias GELU quantization requires W8A8 dispatch".into(),
            )),
        }
    }

    pub(super) fn is_fp8(&self) -> bool {
        matches!(self, Self::Fp8 { .. })
    }

    pub(super) fn is_w8a8(&self) -> bool {
        matches!(self, Self::W8A8 { .. })
    }

    pub(super) fn is_quantized(&self) -> bool {
        !matches!(self, Self::Bf16 { .. })
    }

    pub(super) fn forward_reusable_quantized(
        &self,
        input: &Gr00tQuantizedLinearInput,
        backend: &RuntimeBackend,
    ) -> Result<Tensor> {
        match (self, input) {
            (Self::Fp8 { .. }, Gr00tQuantizedLinearInput::Fp8(input)) => {
                self.forward_quantized(input, backend)
            }
            (Self::W8A8 { weight }, Gr00tQuantizedLinearInput::W8A8(input)) => {
                weight.gemm_quantized(backend, input)
            }
            _ => Err(Error::Other(
                "GR00T quantized input and linear precision do not match".into(),
            )),
        }
    }

    pub(super) fn forward_w8a8_silu_mul(
        &self,
        gate: &Tensor,
        up: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Tensor> {
        let Self::W8A8 { weight } = self else {
            return Err(Error::Other(
                "GR00T fused SiLU-mul W8A8 dispatch requires W8A8 weights".into(),
            ));
        };
        let activation =
            kernels::gemm::quantize_w8a8_silu_mul_activation(backend.context(), gate, up)?;
        weight.gemm_quantized(backend, &activation)
    }

    pub(super) fn forward_fp8_quantized_bias(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Tensor> {
        let Self::Fp8 {
            weight,
            weight_scale,
            activation_scale,
            bias,
        } = self
        else {
            return Err(Error::Other(
                "GR00T fused-bias FP8 dispatch requires FP8 weights".into(),
            ));
        };
        kernels::gemm::fp8_bias_bf16(
            backend.context(),
            input,
            *activation_scale,
            kernels::gemm::Fp8WeightView {
                values_e4m3: weight,
                scale: *weight_scale,
                dual_geglu_interleaved: false,
                dual_geglu_auto_interleaved: None,
            },
            bias.as_ref().ok_or_else(|| {
                Error::Other("GR00T fused-bias FP8 projection requires bias".into())
            })?,
        )
    }

    pub(super) fn forward_fp8_bias(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Tensor> {
        let Self::Fp8 {
            activation_scale, ..
        } = self
        else {
            return Err(Error::Other(
                "GR00T fused-bias FP8 dispatch requires FP8 weights".into(),
            ));
        };
        let input =
            kernels::quantization::quantize_bf16_e4m3(backend.context(), input, *activation_scale)?;
        self.forward_fp8_quantized_bias(&input, backend)
    }

    pub(super) fn quantize_input(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        self.activation_scale()
            .map(|scale| kernels::quantization::quantize_bf16_e4m3(backend.context(), input, scale))
            .transpose()
    }

    pub(super) fn forward_quantized(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Tensor> {
        let Self::Fp8 {
            weight,
            weight_scale,
            activation_scale,
            ..
        } = self
        else {
            return Err(Error::Other(
                "GR00T pre-quantized linear dispatch requires FP8 weights".into(),
            ));
        };
        kernels::gemm::fp8_bf16(
            backend.context(),
            input,
            *activation_scale,
            kernels::gemm::Fp8WeightView {
                values_e4m3: weight,
                scale: *weight_scale,
                dual_geglu_interleaved: false,
                dual_geglu_auto_interleaved: None,
            },
        )
    }

    pub(super) fn bias(&self) -> Option<&Tensor> {
        match self {
            Self::Bf16 { weights, .. } => Some(&weights.bias),
            Self::Fp8 { bias, .. } => bias.as_ref(),
            Self::W8A8 { weight } => weight.bias.as_ref(),
        }
    }
}

fn validate_sha256(field: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(Error::Other(format!(
            "GR00T FP8 {field} must be a 64-character hexadecimal SHA-256"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        e4m3_weight_scale, quantize_output_channels, CalibrationDocument, Gr00tFp8Calibration,
        CALIBRATION_SCHEMA, E4M3_MAX,
    };
    use apxinf_core::Tensor;
    use std::collections::BTreeMap;

    #[test]
    fn e4m3_weight_scale_maps_max_abs_onto_representable_maximum() {
        // A non-zero maximum magnitude scales so the largest weight lands on
        // E4M3_MAX; the derivation matches the inline math every FP8 path used.
        let scale = e4m3_weight_scale(8.96);
        assert_eq!(scale, 8.96f32 / E4M3_MAX);
        assert!((8.96f32 / scale - E4M3_MAX).abs() <= f32::EPSILON * E4M3_MAX);
        // A zero-magnitude tensor falls back to a unit scale so quantization
        // stays finite instead of dividing by zero.
        assert_eq!(e4m3_weight_scale(0.0), 1.0);
    }

    #[test]
    fn e4m3_weight_scale_keeps_scaled_weights_within_e4m3_range() {
        // Every finite weight divided by the derived scale stays inside the
        // representable E4M3 range, for both signs and the extreme value.
        let weights = [-8.96f32, -3.5, -0.0, 0.0, 1.25, 8.96];
        let maximum = weights
            .iter()
            .fold(0.0f32, |value, next| value.max(next.abs()));
        let scale = e4m3_weight_scale(maximum);
        for &weight in &weights {
            let scaled = weight / scale;
            assert!(scaled.abs() <= E4M3_MAX + f32::EPSILON * E4M3_MAX);
        }
    }

    fn document() -> CalibrationDocument {
        CalibrationDocument {
            schema: CALIBRATION_SCHEMA.into(),
            checkpoint_config_sha256: "a".repeat(64),
            fixture_manifest_sha256: "b".repeat(64),
            activation_scales: BTreeMap::from([("action_head.test".into(), 0.25)]),
        }
    }

    #[test]
    fn calibration_requires_complete_identity_and_positive_scales() {
        let calibration = Gr00tFp8Calibration::from_document(document()).unwrap();
        assert_eq!(calibration.scale("action_head.test").unwrap(), 0.25);
        assert!(calibration.scale("action_head.missing").is_err());

        let mut invalid = document();
        invalid.activation_scales.insert("bad".into(), 0.0);
        assert!(Gr00tFp8Calibration::from_document(invalid).is_err());

        let mut invalid = document();
        invalid.checkpoint_config_sha256 = "not-a-digest".into();
        assert!(Gr00tFp8Calibration::from_document(invalid).is_err());
    }

    #[test]
    fn int8_weights_are_quantized_per_output_channel_and_transposed() {
        let weight = Tensor::from_f32(vec![3, 2], &[1.0, -10.0, 2.0, 0.0, 3.0, 10.0]).unwrap();
        let (quantized, scales, input, output) = quantize_output_channels(&weight).unwrap();
        assert_eq!((input, output), (3, 2));
        assert_eq!(quantized, vec![42, 85, 127, -127, 0, 127]);
        assert!((scales[0] - 3.0 / 127.0).abs() < 1.0e-7);
        assert!((scales[1] - 10.0 / 127.0).abs() < 1.0e-7);
    }
}
