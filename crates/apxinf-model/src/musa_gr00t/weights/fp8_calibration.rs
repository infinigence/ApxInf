use super::super::assets::checkpoint_identity;
use apxinf_core::{Error, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const CALIBRATION_SCHEMA: &str = "apxinf.fp8-calibration.v1";
const MODEL_FAMILY: &str = "gr00t";
const FP8_FORMAT: &str = "e4m3fn";
const CALIBRATION_STATISTIC: &str = "absmax";
const CALIBRATION_SCALE_RULE: &str = "max(amax*margin/448,1e-8)";
const E4M3_MAX: f32 = 448.0;
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
pub(crate) struct Gr00tFp8Calibration {
    activation_scales: BTreeMap<String, f32>,
}

impl Gr00tFp8Calibration {
    pub(crate) fn from_json_file(
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
        let benchmark = document.calibration_data.kind == "fixed-input-benchmark"
            && !document.calibration_data.production
            && std::env::var("APXINF_GR00T_ALLOW_BENCHMARK_CALIBRATION").as_deref() == Ok("1");
        if !benchmark
            && (document.calibration_data.kind != "representative"
                || !document.calibration_data.production)
        {
            return Err(Error::Other(
                "GR00T FP8 calibration requires representative production data".into(),
            ));
        }
        if benchmark {
            eprintln!("GR00T FP8: benchmark-only calibration; not validated for deployment");
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

    pub(crate) fn scale(&self, consumer: &str) -> Result<f32> {
        if let Some(prefix) = consumer.strip_suffix(".fused_qkv") {
            let mut scale = 0.0f32;
            for projection in ["query", "key", "value"] {
                scale = scale.max(self.scale(&format!("{prefix}.{projection}"))?);
            }
            return Ok(scale);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> serde_json::Value {
        serde_json::json!({
            "schema": CALIBRATION_SCHEMA,
            "model": {"family": "gr00t", "checkpoint": "test-checkpoint"},
            "quantization": {"format": FP8_FORMAT, "statistic": CALIBRATION_STATISTIC,
                "scale_rule": CALIBRATION_SCALE_RULE, "margin": 1.0},
            "calibration_data": {"identity": "test-data", "sample_count": 1,
                "kind": "representative", "production": true},
            "seed_policy": {"algorithm": "fixed", "sample_sequence": "fixed", "base_seed": 0},
            "source_revision": "test-revision", "device": {"requested": "musa:1", "host": "test"},
            "plan": {"sites": ["linear.input"], "consumers": {"linear": "linear.input"},
                "statistics": {"linear.input": "absmax"}},
            "observed_sites": ["linear.input"],
            "scales": {"linear.input": {"amax": 448.0, "scale": 1.0}}
        })
    }

    #[test]
    fn calibration_validates_identity_coverage_and_scale() {
        let expected = ["linear".into()];
        let valid = document();
        assert_eq!(
            Gr00tFp8Calibration::from_json_str(&valid.to_string(), "test-checkpoint", &expected)
                .unwrap()
                .scale("linear")
                .unwrap(),
            1.0
        );
        assert!(
            Gr00tFp8Calibration::from_json_str(&valid.to_string(), "other", &expected).is_err()
        );
        let mut invalid = document();
        invalid["scales"]["linear.input"]["scale"] = serde_json::json!(0.5);
        assert!(Gr00tFp8Calibration::from_json_str(
            &invalid.to_string(),
            "test-checkpoint",
            &expected
        )
        .is_err());
        let mut invalid = document();
        invalid["observed_sites"] = serde_json::json!([]);
        assert!(Gr00tFp8Calibration::from_json_str(
            &invalid.to_string(),
            "test-checkpoint",
            &expected
        )
        .is_err());
    }

    #[test]
    fn fused_qkv_uses_largest_shared_input_scale() {
        let calibration = Gr00tFp8Calibration {
            activation_scales: [
                ("attention.query".into(), 0.1),
                ("attention.key".into(), 0.3),
                ("attention.value".into(), 0.2),
            ]
            .into_iter()
            .collect(),
        };
        assert_eq!(calibration.scale("attention.fused_qkv").unwrap(), 0.3);
        assert!(calibration.scale("other.fused_qkv").is_err());
    }
}
