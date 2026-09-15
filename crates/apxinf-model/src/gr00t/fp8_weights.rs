//! Static-FP8 calibration and device weights for GR00T N1.7.
//!
//! This module is intentionally private: selecting FP8 reuses the existing
//! [`crate::ModelPrecision`] option while calibration remains an offline
//! artifact. It does not add another model-facing runtime interface.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use apxinf_core::{Backend, Error, Result, Tensor};
use serde::Deserialize;

use super::action_weights::Gr00tLinearWeights;
use super::backend::{kernels, RuntimeBackend};
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
                "GR00T FP8 calibration checkpoint identity mismatch: profile={}, runtime={checkpoint}",
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
            kernels::quantization::quantize_bf16_e4m3(backend.context(), &weight, weight_scale)?;
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
            kernels::quantization::quantize_bf16_e4m3(backend.context(), &weight, weight_scale)?;
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
        let input = kernels::quantization::quantize_bf16_e4m3(
            backend.context(),
            input,
            self.activation_scale,
        )?;
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
        Ok(Some(kernels::quantization::quantize_bf16_e4m3(
            backend.context(),
            input,
            self.activation_scale,
        )?))
    }

    fn quantize_bias_gelu_reusable_input(
        &self,
        input: &Tensor,
        bias: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::ReusableInput>> {
        Ok(Some(kernels::activation::bias_gelu_quant_bf16_e4m3(
            backend.context(),
            input,
            bias,
            self.activation_scale,
        )?))
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
        Ok(Some(kernels::quantization::quantize_bf16_e4m3(
            backend.context(),
            input,
            self.activation_scale,
        )?))
    }

    fn forward_quantized_tensor(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor> {
        kernels::gemm::fp8_bf16(
            backend.context(),
            input,
            self.activation_scale,
            kernels::gemm::Fp8WeightView {
                values_e4m3: &self.weight,
                scale: self.weight_scale,
                dual_geglu_interleaved: false,
                dual_geglu_auto_interleaved: None,
            },
        )
    }

    fn adaptive_layer_norm_quantized(
        &self,
        input: &Tensor,
        modulation: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        let (normalized, quantized) = kernels::norm::adaptive_layer_quant_bf16_e4m3(
            backend.context(),
            input,
            modulation,
            eps,
            self.activation_scale,
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
        Ok(Some(kernels::norm::rms_quant_bf16_e4m3(
            backend.context(),
            input,
            weight,
            eps,
            self.activation_scale,
        )?))
    }

    fn uses_quantized_output(&self) -> bool {
        true
    }
}

/// Content identity shared with the model-neutral Python calibration runner.
///
/// Only weight shards participate in the identity. Paths are canonicalized
/// relative to the checkpoint root, sorted bytewise, and delimited before the
/// file contents so Rust validates exactly the artifact Python generated.
fn checkpoint_identity(checkpoint: &Path, backbone: &Path) -> Result<String> {
    let primary = single_checkpoint_identity(checkpoint)?;
    let backbone = single_checkpoint_identity(backbone)?;
    let mut digest = LocalSha256::new();
    for (name, identity) in [("primary", primary), ("backbone", backbone)] {
        digest.update(name.as_bytes());
        digest.update(&[0]);
        digest.update(identity.as_bytes());
        digest.update(&[0]);
    }
    Ok(format!("sha256:{}", digest.finish_hex()))
}

fn single_checkpoint_identity(path: &Path) -> Result<String> {
    let (root, files) = if path.is_dir() {
        let index = path.join("model.safetensors.index.json");
        let model = path.join("model.safetensors");
        let files = if index.is_file() {
            checkpoint_index_files(&index)?
        } else if model.is_file() {
            vec![model]
        } else {
            let mut files = Vec::new();
            collect_safetensors(path, &mut files)?;
            files
        };
        (path.to_path_buf(), files)
    } else if path
        .file_name()
        .and_then(|name| name.to_str())
        .map_or(false, |name| name.ends_with(".index.json"))
    {
        (
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            checkpoint_index_files(path)?,
        )
    } else {
        (
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            vec![path.to_path_buf()],
        )
    };
    if files.is_empty() || files.iter().any(|file| !file.is_file()) {
        return Err(Error::Other(format!(
            "cannot resolve GR00T checkpoint files from {}",
            path.display()
        )));
    }
    let mut canonical = files
        .into_iter()
        .map(|file| {
            let relative = file.strip_prefix(&root).map_err(|_| {
                Error::Other(format!(
                    "GR00T checkpoint file {} is outside {}",
                    file.display(),
                    root.display()
                ))
            })?;
            let relative = relative
                .components()
                .map(|part| {
                    part.as_os_str().to_str().ok_or_else(|| {
                        Error::Other(format!(
                            "GR00T checkpoint path is not canonical UTF-8: {}",
                            file.display()
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?
                .join("/");
            Ok((relative, file))
        })
        .collect::<Result<Vec<_>>>()?;
    canonical.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    let mut digest = LocalSha256::new();
    for (relative, file) in canonical {
        digest.update(relative.as_bytes());
        digest.update(&[0]);
        let mut handle = File::open(&file)
            .map_err(|error| Error::Other(format!("read {}: {error}", file.display())))?;
        let mut buffer = [0u8; 1024 * 1024];
        loop {
            let count = handle
                .read(&mut buffer)
                .map_err(|error| Error::Other(format!("read {}: {error}", file.display())))?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
    }
    Ok(format!("sha256:{}", digest.finish_hex()))
}

fn checkpoint_index_files(index_path: &Path) -> Result<Vec<PathBuf>> {
    let raw = std::fs::read_to_string(index_path)
        .map_err(|error| Error::Other(format!("read {}: {error}", index_path.display())))?;
    let index: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|error| Error::Other(format!("GR00T checkpoint index JSON: {error}")))?;
    let weight_map = index
        .get("weight_map")
        .and_then(|value| value.as_object())
        .ok_or_else(|| Error::Other("GR00T checkpoint index has no weight_map".into()))?;
    let mut names = BTreeSet::new();
    for value in weight_map.values() {
        let name = value
            .as_str()
            .ok_or_else(|| Error::Other("GR00T checkpoint index has a non-string shard".into()))?;
        let relative = Path::new(name);
        if relative.is_absolute()
            || relative
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err(Error::Other(format!(
                "GR00T checkpoint index has an unsafe shard path: {name}"
            )));
        }
        names.insert(name.to_owned());
    }
    let root = index_path.parent().unwrap_or_else(|| Path::new("."));
    Ok(names.into_iter().map(|name| root.join(name)).collect())
}

fn collect_safetensors(directory: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(directory)
        .map_err(|error| Error::Other(format!("read {}: {error}", directory.display())))?
    {
        let path = entry
            .map_err(|error| Error::Other(error.to_string()))?
            .path();
        if path.is_dir() {
            collect_safetensors(&path, files)?;
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors") {
            files.push(path);
        }
    }
    Ok(())
}

struct LocalSha256 {
    state: [u32; 8],
    buffer: Vec<u8>,
    bytes: u64,
}

impl LocalSha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: Vec::with_capacity(64),
            bytes: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.bytes += input.len() as u64;
        if !self.buffer.is_empty() {
            let needed = 64 - self.buffer.len();
            let take = needed.min(input.len());
            self.buffer.extend_from_slice(&input[..take]);
            input = &input[take..];
            if self.buffer.len() == 64 {
                let block: [u8; 64] = self.buffer.as_slice().try_into().unwrap();
                self.compress(&block);
                self.buffer.clear();
            }
        }
        while input.len() >= 64 {
            let block: &[u8; 64] = input[..64].try_into().unwrap();
            self.compress(block);
            input = &input[64..];
        }
        self.buffer.extend_from_slice(input);
    }

    fn finish_hex(mut self) -> String {
        let bit_len = self.bytes * 8;
        self.buffer.push(0x80);
        while self.buffer.len() % 64 != 56 {
            self.buffer.push(0);
        }
        self.buffer.extend_from_slice(&bit_len.to_be_bytes());
        let blocks = std::mem::take(&mut self.buffer);
        for chunk in blocks.chunks_exact(64) {
            self.compress(chunk.try_into().unwrap());
        }
        self.state
            .iter()
            .map(|word| format!("{word:08x}"))
            .collect::<Vec<_>>()
            .join("")
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut words = [0u32; 64];
        for (index, bytes) in block.chunks_exact(4).enumerate() {
            words[index] = u32::from_be_bytes(bytes.try_into().unwrap());
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (state, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *state = state.wrapping_add(value);
        }
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
            "sha256:d23faece91e5ba14630dd918ed491b712b32c905f153a2bbb5065f355b5df094"
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
