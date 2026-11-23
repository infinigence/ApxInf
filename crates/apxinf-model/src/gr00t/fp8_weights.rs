//! Static-FP8 calibration and device weights for GR00T N1.7.
//!
//! This module is intentionally private: selecting FP8 reuses the existing
//! [`crate::ModelPrecision`] option while calibration remains an offline
//! artifact. It does not add another model-facing runtime interface.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use apxinf_core::{DType, Error, Result, Shape, Tensor};
use serde::Deserialize;

use super::action_weights::Gr00tLinearWeights;
use super::assets::checkpoint_identity;
#[cfg(test)]
use super::assets::LocalSha256;
use super::backend::{ops, RuntimeBackend};
use super::device_weights::DeviceLinearWeights;

const CALIBRATION_SCHEMA: &str = "apxinf.fp8-calibration.v1";
const MODEL_FAMILY: &str = "gr00t";
const FP8_FORMAT: &str = "e4m3fn";
const CALIBRATION_STATISTIC: &str = "absmax";
const CALIBRATION_SCALE_RULE: &str = "max(amax*margin/448,1e-8)";
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationDocument {
    schema: String,
    model: CalibrationModel,
    quantization: CalibrationQuantization,
    calibration_data: CalibrationData,
    seed_policy: CalibrationSeedPolicy,
    source_revision: String,
    device: BTreeMap<String, String>,
    plan: CalibrationPlan,
    observed_sites: Vec<String>,
    scales: BTreeMap<String, CalibrationScale>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationModel {
    family: String,
    checkpoint: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationQuantization {
    format: String,
    statistic: String,
    scale_rule: String,
    margin: f32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationData {
    identity: String,
    kind: String,
    production: bool,
    sample_count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationSeedPolicy {
    algorithm: String,
    base_seed: u64,
    sample_sequence: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationPlan {
    sites: Vec<String>,
    consumers: BTreeMap<String, String>,
    statistics: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationScale {
    amax: f32,
    scale: f32,
}

/// Validated static-FP8 scales indexed by stable logical consumer ID.
#[derive(Debug)]
pub(super) struct Gr00tFp8Calibration {
    activation_scales: BTreeMap<String, f32>,
}

impl Gr00tFp8Calibration {
    pub(super) fn from_json_file(
        path: &Path,
        checkpoint_path: &Path,
        backbone_path: &Path,
        expected_consumers: &[String],
    ) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|error| {
            Error::Other(format!(
                "read GR00T FP8 calibration {}: {error}",
                path.display()
            ))
        })?;
        let checkpoint = checkpoint_identity(checkpoint_path, backbone_path)?;
        Self::from_json_str(&raw, &checkpoint, expected_consumers)
    }

    fn from_json_str(raw: &str, checkpoint: &str, expected_consumers: &[String]) -> Result<Self> {
        let document: CalibrationDocument = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("GR00T FP8 calibration JSON: {error}")))?;
        if document.schema != CALIBRATION_SCHEMA {
            return Err(Error::Other(format!(
                "GR00T FP8 calibration schema mismatch: expected {CALIBRATION_SCHEMA}, got {}",
                document.schema
            )));
        }
        if document.model.family != MODEL_FAMILY {
            return Err(Error::Other(format!(
                "GR00T FP8 calibration model family mismatch: {}",
                document.model.family
            )));
        }
        if document.model.checkpoint != checkpoint {
            return Err(Error::Other(format!(
                "GR00T FP8 calibration checkpoint identity mismatch: profile={}, runtime={checkpoint}; regenerate calibration for the current weights and processor resources",
                document.model.checkpoint
            )));
        }
        if document.quantization.format != FP8_FORMAT
            || document.quantization.statistic != CALIBRATION_STATISTIC
            || document.quantization.scale_rule != CALIBRATION_SCALE_RULE
            || !document.quantization.margin.is_finite()
            || document.quantization.margin < 1.0
        {
            return Err(Error::Other(
                "GR00T FP8 calibration quantization contract mismatch".into(),
            ));
        }
        if document.calibration_data.identity.is_empty()
            || document.calibration_data.sample_count == 0
            || document.source_revision.is_empty()
            || document.source_revision == "unknown"
            || document.seed_policy.algorithm.is_empty()
            || document.seed_policy.sample_sequence.is_empty()
            || document
                .device
                .get("requested")
                .map_or(true, String::is_empty)
            || document.device.get("host").map_or(true, String::is_empty)
        {
            return Err(Error::Other(
                "GR00T FP8 calibration manifest is incomplete".into(),
            ));
        }
        if document.calibration_data.kind != "representative"
            || !document.calibration_data.production
        {
            return Err(Error::Other(
                "GR00T FP8 calibration requires representative production data".into(),
            ));
        }
        let _ = document.seed_policy.base_seed;

        let expected_consumers = expected_consumers.iter().cloned().collect::<BTreeSet<_>>();
        let expected_sites = expected_consumers
            .iter()
            .map(|consumer| format!("{consumer}.input"))
            .collect::<BTreeSet<_>>();
        let plan_sites = unique_strings("plan.sites", &document.plan.sites)?;
        let observed_sites = unique_strings("observed_sites", &document.observed_sites)?;
        let scale_sites = document.scales.keys().cloned().collect::<BTreeSet<_>>();
        let plan_consumers = document
            .plan
            .consumers
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        if plan_sites != expected_sites
            || observed_sites != expected_sites
            || scale_sites != expected_sites
            || plan_consumers != expected_consumers
        {
            return Err(Error::Other(
                "GR00T FP8 calibration coverage is missing, unknown, or incompatible".into(),
            ));
        }
        for (consumer, site) in &document.plan.consumers {
            if site != &format!("{consumer}.input") {
                return Err(Error::Other(format!(
                    "GR00T FP8 consumer {consumer:?} maps to incompatible site {site:?}"
                )));
            }
        }
        if document.plan.statistics.len() != expected_sites.len()
            || document.plan.statistics.iter().any(|(site, statistic)| {
                !expected_sites.contains(site) || statistic != CALIBRATION_STATISTIC
            })
        {
            return Err(Error::Other(
                "GR00T FP8 calibration statistics do not match the execution plan".into(),
            ));
        }

        let mut activation_scales = BTreeMap::new();
        for consumer in expected_consumers {
            let site = &document.plan.consumers[&consumer];
            let entry = &document.scales[site];
            if !entry.amax.is_finite()
                || entry.amax < 0.0
                || !entry.scale.is_finite()
                || entry.scale <= 0.0
            {
                return Err(Error::Other(format!(
                    "GR00T FP8 calibration entry {site:?} is non-finite or non-positive"
                )));
            }
            let expected_scale = ((entry.amax as f64 * document.quantization.margin as f64)
                / E4M3_MAX as f64)
                .max(1.0e-8);
            if !expected_scale.is_finite()
                || expected_scale > f32::MAX as f64
                || (entry.scale as f64 - expected_scale).abs() > expected_scale * 1.0e-5
            {
                return Err(Error::Other(format!(
                    "GR00T FP8 calibration entry {site:?} has inconsistent amax/scale values"
                )));
            }
            activation_scales.insert(consumer, entry.scale);
        }
        Ok(Self { activation_scales })
    }

    pub(super) fn scale(&self, consumer: &str) -> Result<f32> {
        self.activation_scales
            .get(consumer)
            .copied()
            .ok_or_else(|| {
                Error::Other(format!(
                    "GR00T FP8 calibration is missing consumer {consumer:?}"
                ))
            })
    }
}

fn unique_strings(label: &str, values: &[String]) -> Result<BTreeSet<String>> {
    let set = values.iter().cloned().collect::<BTreeSet<_>>();
    if set.len() != values.len() {
        return Err(Error::Other(format!(
            "GR00T FP8 calibration {label} contains duplicate values"
        )));
    }
    Ok(set)
}

/// Static-FP8 device linear. Bias and output stay BF16 so residual and
/// normalization boundaries match the reference graph.
#[derive(Debug)]
pub(super) struct Gr00tFp8LinearWeights {
    weight: Tensor,
    weight_scale: f32,
    activation_scale: f32,
    bias: Option<Tensor>,
}

impl Gr00tFp8LinearWeights {
    fn quantize_activation(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor> {
        ops::quantization::quantize_fixed_scale(backend.context(), input, self.activation_scale)
    }

    pub(super) fn from_host(
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
            ops::quantization::quantize_fixed_scale(backend.context(), &weight, weight_scale)?;
        Ok(Self {
            weight,
            weight_scale,
            activation_scale,
            bias: Some(backend.to_device(&weights.bias)?),
        })
    }

    pub(super) fn matrix(
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
            ops::quantization::quantize_fixed_scale(backend.context(), &weight, weight_scale)?;
        Ok(Self {
            weight,
            weight_scale,
            activation_scale,
            bias: None,
        })
    }
}

impl DeviceLinearWeights for Gr00tFp8LinearWeights {
    type ReusableInput = Tensor;

    fn forward(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor> {
        let input = self.quantize_activation(input, backend)?;
        self.forward_quantized_tensor(&input, backend)
    }

    fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    fn activation_scale(&self) -> Option<f32> {
        Some(self.activation_scale)
    }

    fn can_share_quantized_input_with(&self, other: &Self) -> bool {
        self.activation_scale == other.activation_scale
    }

    fn quantize_reusable_input(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::ReusableInput>> {
        Ok(Some(self.quantize_activation(input, backend)?))
    }

    fn quantize_bias_gelu_reusable_input(
        &self,
        input: &Tensor,
        bias: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::ReusableInput>> {
        Ok(Some(ops::pointwise::bias_gelu_quantized(
            backend.context(),
            input,
            bias,
            self.activation_scale,
        )?))
    }

    fn residual_layer_norm_quantized(
        &self,
        projection: &Tensor,
        residual: &Tensor,
        norm_weight: &Tensor,
        norm_bias: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        let hidden = backend
            .context()
            .allocate_output(projection.shape().clone(), DType::BF16)?;
        let normalized = backend
            .context()
            .allocate_output(projection.shape().clone(), DType::F8E4M3)?;
        ops::fused_add_norm(
            backend.context(),
            ops::FusedAddNormArgs {
                input: projection,
                bias: None,
                residual,
                gate: None,
                hidden: &hidden,
                normalized: None,
                kind: ops::NormKind::Layer,
                affine: ops::Affine::Weight {
                    weight: norm_weight,
                    bias: Some(norm_bias),
                    offset: false,
                },
                quantized: Some(ops::QuantOut::StaticE4m3 {
                    output: &normalized,
                    scale: self.activation_scale,
                }),
                numerics: ops::NormNumerics::Flexible,
                bias_order: ops::BiasOrder::SingleRound,
                eps,
            },
        )?;
        Ok(Some((hidden, normalized)))
    }

    fn supports_fused_bias_gelu_quantization(&self) -> bool {
        true
    }

    fn forward_reusable_quantized(
        &self,
        input: &Self::ReusableInput,
        backend: &RuntimeBackend,
    ) -> Result<Tensor> {
        self.forward_quantized_tensor(input, backend)
    }

    fn quantize_tensor_input(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        Ok(Some(self.quantize_activation(input, backend)?))
    }

    fn forward_quantized_tensor(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor> {
        let a = input.shape().dims();
        let b = self.weight.shape().dims();
        if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
            return Err(Error::Other(format!(
                "GR00T FP8 GEMM shape mismatch: {a:?} @ {b:?}"
            )));
        }
        let mut output = backend
            .context()
            .allocate_output(Shape::new(vec![a[0], b[1]]), DType::BF16)?;
        let mut args = ops::gemm::GemmArgs::new(input, &self.weight, &mut output);
        args.quantization = ops::gemm::GemmQuantization::Fp8UnitScale;
        args.alpha = self.activation_scale * self.weight_scale;
        ops::gemm::gemm(backend.context(), args)?;
        Ok(output)
    }

    fn adaptive_layer_norm_quantized(
        &self,
        input: &Tensor,
        modulation: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        let normalized = backend
            .context()
            .allocate_output(input.shape().clone(), DType::BF16)?;
        let quantized = backend
            .context()
            .allocate_output(input.shape().clone(), DType::F8E4M3)?;
        ops::norm::norm(
            backend.context(),
            ops::NormArgs {
                input,
                output: Some(&normalized),
                kind: ops::NormKind::Layer,
                affine: ops::Affine::Adaptive { style: modulation },
                quantized: Some(ops::QuantOut::StaticE4m3 {
                    output: &quantized,
                    scale: self.activation_scale,
                }),
                numerics: ops::NormNumerics::Flexible,
                eps,
            },
        )?;
        Ok(Some((normalized, quantized)))
    }

    fn rms_norm_quantized(
        &self,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::ReusableInput>> {
        let output = backend
            .context()
            .allocate_output(input.shape().clone(), DType::F8E4M3)?;
        ops::norm::norm(
            backend.context(),
            ops::NormArgs {
                input,
                output: None,
                kind: ops::NormKind::Rms,
                affine: ops::Affine::Weight {
                    weight,
                    bias: None,
                    offset: false,
                },
                quantized: Some(ops::QuantOut::StaticE4m3 {
                    output: &output,
                    scale: self.activation_scale,
                }),
                numerics: ops::NormNumerics::Flexible,
                eps,
            },
        )?;
        Ok(Some(output))
    }

    fn uses_quantized_output(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{
        checkpoint_identity, e4m3_weight_scale, Gr00tFp8Calibration, LocalSha256,
        CALIBRATION_SCHEMA, E4M3_MAX,
    };
    use std::path::Path;

    #[test]
    fn sha256_matches_the_standard_vector() {
        let mut digest = LocalSha256::new();
        digest.update(b"abc");
        assert_eq!(
            digest.finish_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn checkpoint_identity_binds_primary_and_backbone() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/checkpoint_identity");
        assert_eq!(
            checkpoint_identity(&fixture, &fixture).unwrap(),
            "sha256:db8f2f5b6bf722562e68038fe7f0f7cd8f0acc4deec17d24ca69a126150fb256"
        );
    }

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

    fn document(amax: f32, scale: f32) -> String {
        serde_json::json!({
            "schema": CALIBRATION_SCHEMA,
            "model": {"family": "gr00t", "checkpoint": "sha256:test"},
            "quantization": {
                "format": "e4m3fn",
                "statistic": "absmax",
                "scale_rule": "max(amax*margin/448,1e-8)",
                "margin": 1.0
            },
            "calibration_data": {
                "identity": "sha256:fixture",
                "kind": "representative",
                "production": true,
                "sample_count": 10
            },
            "seed_policy": {
                "algorithm": "seed-plus-sample-context-v1",
                "base_seed": 7,
                "sample_sequence": "[base_seed,sample_index]"
            },
            "source_revision": "0123456789abcdef",
            "device": {"requested": "cuda:0", "host": "test-host"},
            "plan": {
                "sites": ["action_head.test.input"],
                "consumers": {"action_head.test": "action_head.test.input"},
                "statistics": {"action_head.test.input": "absmax"}
            },
            "observed_sites": ["action_head.test.input"],
            "scales": {
                "action_head.test.input": {"amax": amax, "scale": scale}
            }
        })
        .to_string()
    }

    #[test]
    fn calibration_requires_complete_identity_and_positive_scales() {
        let calibration = Gr00tFp8Calibration::from_json_str(
            &document(112.0, 0.25),
            "sha256:test",
            &["action_head.test".into()],
        )
        .unwrap();
        assert_eq!(calibration.scale("action_head.test").unwrap(), 0.25);
        assert!(calibration.scale("action_head.missing").is_err());
        assert!(Gr00tFp8Calibration::from_json_str(
            &document(112.0, 0.0),
            "sha256:test",
            &["action_head.test".into()],
        )
        .is_err());
        assert!(Gr00tFp8Calibration::from_json_str(
            &document(112.0, 0.25),
            "sha256:other",
            &["action_head.test".into()],
        )
        .is_err());
        assert!(Gr00tFp8Calibration::from_json_str(
            &document(112.0, 0.25),
            "sha256:test",
            &["action_head.other".into()],
        )
        .is_err());
    }

    #[test]
    fn calibration_file_rejects_mutated_processor_resources() {
        struct Fixture(std::path::PathBuf);
        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let fixture = Fixture(std::env::temp_dir().join(format!(
            "apxinf-gr00t-calibration-processor-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )));
        let root = &fixture.0;
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(root.join("model.safetensors"), b"test weights").unwrap();
        for name in [
            "processor_config.json",
            "statistics.json",
            "embodiment_id.json",
        ] {
            std::fs::write(root.join(name), b"{}\n").unwrap();
        }
        let identity = checkpoint_identity(root, root).unwrap();
        let profile = root.join("calibration.json");
        std::fs::write(
            &profile,
            document(112.0, 0.25).replace("sha256:test", &identity),
        )
        .unwrap();
        let consumers = ["action_head.test".into()];
        assert!(Gr00tFp8Calibration::from_json_file(&profile, root, root, &consumers).is_ok());
        for name in [
            "processor_config.json",
            "statistics.json",
            "embodiment_id.json",
        ] {
            let path = root.join(name);
            std::fs::write(&path, b"{\"changed\":true}\n").unwrap();
            let error =
                Gr00tFp8Calibration::from_json_file(&profile, root, root, &consumers).unwrap_err();
            assert!(error.to_string().contains("checkpoint identity mismatch"));
            std::fs::write(&path, b"{}\n").unwrap();
        }
    }

    #[test]
    fn calibration_rejects_scale_that_does_not_match_amax() {
        assert!(Gr00tFp8Calibration::from_json_str(
            &document(112.0, 0.5),
            "sha256:test",
            &["action_head.test".into()],
        )
        .is_err());
    }

    #[test]
    fn calibration_rejects_non_production_synthetic_data() {
        let mut synthetic: serde_json::Value =
            serde_json::from_str(&document(112.0, 0.25)).unwrap();
        synthetic["calibration_data"]["kind"] = "synthetic-zero-fixture".into();
        synthetic["calibration_data"]["production"] = false.into();

        let error = Gr00tFp8Calibration::from_json_str(
            &synthetic.to_string(),
            "sha256:test",
            &["action_head.test".into()],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("requires representative production data"));
    }
}
