//! Static FP8 E4M3 quantization used by the Thor execution profile.
//!
//! Matrix weights use one scale per tensor. Activations obtain named scales
//! from a calibration file so CUDA graph replay never performs a reduction or
//! allocates scale tensors. Attention probabilities, residuals, and norms stay
//! FP16.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use apxinf_core::{Error, Result, Tensor};
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::pi05::Pi05Config;

/// Largest finite NVIDIA/CUDA E4M3 value (`0x7e`).
pub const E4M3_MAX: f32 = 448.0;

#[derive(Debug, Clone)]
pub struct Fp8Tensor {
    pub values: Tensor,
    /// Dequantization multiplier: `real = fp8(values) * scale`.
    pub scale: f32,
    pub amax: f32,
}

pub const CALIBRATION_SCHEMA: &str = "apxinf.pi05.fp8-calibration.v1";
pub const FP8_FORMAT: &str = "e4m3fn";
pub const CALIBRATION_STATISTIC: &str = "absmax";
pub const CALIBRATION_SCALE_RULE: &str = "max(amax*margin/448,1e-8)";

/// Stable logical activation sites consumed by the static-FP8 execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pi05CalibrationPlan {
    sites: Vec<String>,
    vision_layers: Vec<LayerCalibrationSites>,
    language_layers: Vec<LayerCalibrationSites>,
    action_layers: Vec<LayerCalibrationSites>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerCalibrationSites {
    pub attention_norm: String,
    pub attention_output: Option<String>,
    pub mlp_norm: Option<String>,
    pub mlp_activation: Option<String>,
}

impl Pi05CalibrationPlan {
    pub fn for_config(config: &Pi05Config) -> Self {
        let mut sites = vec!["vision.patch_input".to_owned()];
        let vision_layers = (0..config.vision_depth)
            .map(|index| layer_sites("vision", index, true))
            .collect::<Vec<_>>();
        extend_sites(&mut sites, &vision_layers);
        sites.push("vision.post_norm".to_owned());
        let language_layers = (0..config.language.depth)
            .map(|index| layer_sites("language", index, index + 1 < config.language.depth))
            .collect::<Vec<_>>();
        extend_sites(&mut sites, &language_layers);
        sites.extend([
            "action.input".to_owned(),
            "time.input".to_owned(),
            "time.hidden".to_owned(),
            "action.conditioning".to_owned(),
        ]);
        let action_layers = (0..config.action_expert.depth)
            .map(|index| layer_sites("action", index, true))
            .collect::<Vec<_>>();
        extend_sites(&mut sites, &action_layers);
        sites.push("action.final_norm".to_owned());
        Self {
            sites,
            vision_layers,
            language_layers,
            action_layers,
        }
    }

    pub fn sites(&self) -> &[String] {
        &self.sites
    }

    pub fn vision_layers(&self) -> &[LayerCalibrationSites] {
        &self.vision_layers
    }

    pub fn language_layers(&self) -> &[LayerCalibrationSites] {
        &self.language_layers
    }

    pub fn action_layers(&self) -> &[LayerCalibrationSites] {
        &self.action_layers
    }
}

fn layer_sites(prefix: &str, index: usize, compute_tail: bool) -> LayerCalibrationSites {
    let named = |suffix| format!("{prefix}.layers.{index}.{suffix}");
    LayerCalibrationSites {
        attention_norm: named("attention_norm"),
        attention_output: compute_tail.then(|| named("attention_output")),
        mlp_norm: compute_tail.then(|| named("mlp_norm")),
        mlp_activation: compute_tail.then(|| named("mlp_activation")),
    }
}

fn extend_sites(sites: &mut Vec<String>, layers: &[LayerCalibrationSites]) {
    for layer in layers {
        sites.push(layer.attention_norm.clone());
        sites.extend(
            [
                layer.attention_output.clone(),
                layer.mlp_norm.clone(),
                layer.mlp_activation.clone(),
            ]
            .into_iter()
            .flatten(),
        );
    }
}

#[derive(Debug, Clone)]
pub struct StaticFp8Calibration {
    scales: HashMap<String, f32>,
    warnings: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationProfile {
    schema: String,
    model: ProfileModel,
    quantization: ProfileQuantization,
    calibration_data: CalibrationData,
    seed_policy: SeedPolicy,
    source_revision: String,
    device: DeviceSummary,
    plan: ProfilePlan,
    observed_sites: Vec<String>,
    scales: ScaleMap,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileModel {
    family: String,
    checkpoint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileQuantization {
    format: String,
    statistic: String,
    scale_rule: String,
    margin: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationData {
    identity: String,
    kind: String,
    production: bool,
    sample_count: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SeedPolicy {
    algorithm: String,
    base_seed: u64,
    sample_sequence: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeviceSummary {
    requested: String,
    host: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfilePlan {
    sites: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScaleEntry {
    amax: f32,
    scale: f32,
}

struct ScaleMap(BTreeMap<String, ScaleEntry>);

impl<'de> Deserialize<'de> for ScaleMap {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(ScaleMapVisitor)
    }
}

struct ScaleMapVisitor;

impl<'de> Visitor<'de> for ScaleMapVisitor {
    type Value = ScaleMap;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a map of unique PI0.5 calibration sites")
    }

    fn visit_map<M>(self, mut access: M) -> std::result::Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut scales = BTreeMap::new();
        while let Some((site, entry)) = access.next_entry::<String, ScaleEntry>()? {
            if scales.insert(site.clone(), entry).is_some() {
                return Err(serde::de::Error::custom(format!(
                    "duplicate calibration scale site `{site}`"
                )));
            }
        }
        Ok(ScaleMap(scales))
    }
}

impl StaticFp8Calibration {
    pub fn from_json_file(path: &Path, config: &Pi05Config, checkpoint: &str) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        let calibration = Self::from_json_str(&raw, config, checkpoint)?;
        for warning in calibration.warnings() {
            eprintln!("warning: {warning}");
        }
        Ok(calibration)
    }

    /// Parse and fully validate a profile before an FP8 runtime is constructed.
    pub fn from_json_str(raw: &str, config: &Pi05Config, checkpoint: &str) -> Result<Self> {
        let profile: CalibrationProfile = serde_json::from_str(raw)
            .map_err(|e| Error::Other(format!("π0.5 FP8 calibration JSON: {e}")))?;
        if profile.schema != CALIBRATION_SCHEMA {
            return Err(Error::Other(format!(
                "π0.5 FP8 calibration schema mismatch: expected {CALIBRATION_SCHEMA}, got {}",
                profile.schema
            )));
        }
        if profile.model.family != "pi05" {
            return Err(Error::Other(format!(
                "π0.5 FP8 calibration model family mismatch: {}",
                profile.model.family
            )));
        }
        let mut warnings = Vec::new();
        if profile.model.checkpoint != checkpoint {
            warnings.push(format!(
                concat!(
                    "π0.5 FP8 calibration checkpoint identity mismatch: ",
                    "profile={}, runtime={}; continuing with the supplied profile"
                ),
                profile.model.checkpoint, checkpoint
            ));
        }
        if profile.quantization.format != FP8_FORMAT
            || profile.quantization.statistic != CALIBRATION_STATISTIC
            || profile.quantization.scale_rule != CALIBRATION_SCALE_RULE
        {
            return Err(Error::Other(
                "π0.5 FP8 calibration quantization contract mismatch".into(),
            ));
        }
        if !profile.quantization.margin.is_finite() || profile.quantization.margin < 1.0 {
            return Err(Error::Other(
                "π0.5 FP8 calibration has invalid margin".into(),
            ));
        }
        if profile.calibration_data.identity.is_empty()
            || profile.calibration_data.kind.is_empty()
            || profile.calibration_data.sample_count == 0
            || profile.source_revision.is_empty()
            || profile.source_revision == "unknown"
            || profile.device.requested.is_empty()
            || profile.device.host.is_empty()
            || profile.seed_policy.algorithm.is_empty()
            || profile.seed_policy.sample_sequence.is_empty()
        {
            return Err(Error::Other(
                "π0.5 FP8 calibration manifest is incomplete".into(),
            ));
        }
        match (
            profile.calibration_data.kind.as_str(),
            profile.calibration_data.production,
        ) {
            ("representative", true) | ("synthetic-zero-fixture", false) => {}
            _ => {
                return Err(Error::Other(
                    "π0.5 FP8 calibration data kind/production label is ambiguous".into(),
                ))
            }
        }
        if profile.calibration_data.identity.starts_with("synthetic:")
            && profile.calibration_data.production
        {
            return Err(Error::Other(
                "π0.5 FP8 calibration synthetic data cannot be labeled production".into(),
            ));
        }
        if profile.seed_policy.algorithm != "numpy-pcg64-seed-sequence-v1"
            || profile.seed_policy.sample_sequence != "[base_seed,sample_index]"
        {
            return Err(Error::Other(
                "π0.5 FP8 calibration seed policy is incompatible".into(),
            ));
        }
        let _ = profile.seed_policy.base_seed;

        let expected = Pi05CalibrationPlan::for_config(config);
        let expected_set = expected.sites().iter().cloned().collect::<BTreeSet<_>>();
        let plan_set = unique_sites("plan", &profile.plan.sites)?;
        let observed_set = unique_sites("observed_sites", &profile.observed_sites)?;
        let scale_set = profile.scales.0.keys().cloned().collect::<BTreeSet<_>>();
        if plan_set != expected_set || observed_set != expected_set || scale_set != expected_set {
            return Err(Error::Other(
                "π0.5 FP8 calibration site coverage is missing, unknown, or incompatible".into(),
            ));
        }

        let mut scales = HashMap::with_capacity(profile.scales.0.len());
        for (name, entry) in profile.scales.0 {
            if !entry.amax.is_finite()
                || entry.amax < 0.0
                || !entry.scale.is_finite()
                || entry.scale <= 0.0
            {
                return Err(Error::Other(format!(
                    "π0.5 FP8 calibration entry `{name}` is non-finite or non-positive"
                )));
            }
            let expected_scale = ((entry.amax as f64 * profile.quantization.margin as f64)
                / E4M3_MAX as f64)
                .max(1.0e-8);
            if !expected_scale.is_finite()
                || expected_scale > f32::MAX as f64
                || (entry.scale as f64 - expected_scale).abs() > expected_scale * 1.0e-5
            {
                return Err(Error::Other(format!(
                    "π0.5 FP8 calibration entry `{name}` has ambiguous amax/scale values"
                )));
            }
            scales.insert(name, entry.scale);
        }
        Ok(Self { scales, warnings })
    }

    pub fn scale(&self, name: &str) -> Result<f32> {
        self.scales
            .get(name)
            .copied()
            .ok_or_else(|| Error::Other(format!("missing π0.5 FP8 activation scale `{name}`")))
    }

    pub fn len(&self) -> usize {
        self.scales.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scales.is_empty()
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }
}

fn unique_sites(label: &str, sites: &[String]) -> Result<BTreeSet<String>> {
    let set = sites.iter().cloned().collect::<BTreeSet<_>>();
    if set.len() != sites.len() {
        return Err(Error::Other(format!(
            "π0.5 FP8 calibration {label} contains ambiguous duplicate sites"
        )));
    }
    Ok(set)
}

/// Content identity shared by calibration generation and runtime validation.
pub fn checkpoint_identity(path: &Path) -> Result<String> {
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
        .is_some_and(|name| name.ends_with(".index.json"))
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
    if files.is_empty() {
        return Err(Error::Other(format!(
            "checkpoint {} has no safetensors",
            path.display()
        )));
    }
    let mut canonical = files
        .into_iter()
        .map(|file| {
            let relative = file.strip_prefix(&root).map_err(|_| {
                Error::Other(format!(
                    "checkpoint file {} is outside {}",
                    file.display(),
                    root.display()
                ))
            })?;
            let relative = relative
                .components()
                .map(|part| {
                    part.as_os_str().to_str().ok_or_else(|| {
                        Error::Other(format!(
                            "checkpoint path is not canonical UTF-8: {}",
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
    let mut digest = Sha256::new();
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
        .map_err(|error| Error::Other(format!("checkpoint index JSON: {error}")))?;
    let weight_map = index
        .get("weight_map")
        .and_then(|value| value.as_object())
        .ok_or_else(|| Error::Other("checkpoint index has no weight_map".into()))?;
    let mut names = BTreeSet::new();
    for value in weight_map.values() {
        let name = value
            .as_str()
            .ok_or_else(|| Error::Other("checkpoint index has a non-string shard".into()))?;
        let relative = Path::new(name);
        if relative.is_absolute()
            || relative
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err(Error::Other(format!(
                "checkpoint index has an unsafe shard path: {name}"
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
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
            files.push(path);
        }
    }
    Ok(())
}

struct Sha256 {
    state: [u32; 8],
    buffer: Vec<u8>,
    bytes: u64,
}

impl Sha256 {
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

/// Quantize a CPU F32/F16/BF16 tensor using a static per-tensor scale.
pub fn quantize_e4m3(tensor: &Tensor, scale: f32) -> Result<Fp8Tensor> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Other(format!(
            "FP8 quantization scale must be finite and positive, got {scale}"
        )));
    }
    let source = tensor.to_f32_vec()?;
    let amax = source.iter().fold(0.0f32, |m, value| m.max(value.abs()));
    let inverse_scale = scale.recip();
    let bytes = source
        .iter()
        .map(|value| encode_e4m3(*value * inverse_scale))
        .collect::<Vec<_>>();
    Ok(Fp8Tensor {
        values: Tensor::from_f8_e4m3(tensor.shape().dims().to_vec(), &bytes)?,
        scale,
        amax,
    })
}

/// Select the standard absmax scale and quantize a weight matrix.
pub fn quantize_e4m3_absmax(tensor: &Tensor) -> Result<Fp8Tensor> {
    let source = tensor.to_f32_vec()?;
    let amax = source.iter().fold(0.0f32, |m, value| m.max(value.abs()));
    // All-zero matrices use scale 1 so their representation remains valid.
    let scale = if amax == 0.0 { 1.0 } else { amax / E4M3_MAX };
    quantize_e4m3(tensor, scale)
}

/// Decode on CPU for tests, cache inspection, and correctness diagnostics.
pub fn dequantize_e4m3(tensor: &Fp8Tensor) -> Result<Tensor> {
    let values = tensor
        .values
        .as_f8_e4m3()?
        .iter()
        .map(|byte| decode_e4m3(*byte) * tensor.scale)
        .collect::<Vec<_>>();
    Tensor::from_f32(tensor.values.shape().dims().to_vec(), &values)
}

/// CUDA-compatible saturating finite E4M3 encoding, round-to-nearest-even.
pub fn encode_e4m3(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let value = value.abs();
    if value == 0.0 {
        return sign;
    }
    if !value.is_finite() || value >= E4M3_MAX {
        return sign | 0x7e;
    }

    // Subnormals have a fixed 2^-9 quantum.
    if value < 2f32.powi(-6) {
        let mantissa = round_ties_even(value * 512.0).min(7) as u8;
        return sign | mantissa;
    }

    let exponent = value.log2().floor() as i32;
    let mut exponent_bits = exponent + 7;
    let normalized = value / 2f32.powi(exponent) - 1.0;
    let mut mantissa = round_ties_even(normalized * 8.0) as i32;
    if mantissa == 8 {
        exponent_bits += 1;
        mantissa = 0;
    }
    if exponent_bits > 15 || (exponent_bits == 15 && mantissa >= 7) {
        return sign | 0x7e;
    }
    sign | ((exponent_bits as u8) << 3) | mantissa as u8
}

pub fn decode_e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 {
        return sign * mantissa as f32 * 2f32.powi(-9);
    }
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    sign * (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
}

fn round_ties_even(value: f32) -> u32 {
    let floor = value.floor();
    let fraction = value - floor;
    let floor_u = floor as u32;
    if fraction > 0.5 || (fraction == 0.5 && floor_u & 1 == 1) {
        floor_u + 1
    } else {
        floor_u
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::DType;

    #[test]
    fn canonical_e4m3_values_match_cuda_layout() {
        assert_eq!(encode_e4m3(0.0), 0x00);
        assert_eq!(encode_e4m3(-0.0), 0x80);
        assert_eq!(encode_e4m3(1.0), 0x38);
        assert_eq!(encode_e4m3(-1.0), 0xb8);
        assert_eq!(encode_e4m3(448.0), 0x7e);
        assert_eq!(decode_e4m3(0x01), 2f32.powi(-9));
        assert_eq!(decode_e4m3(0x7e), 448.0);
        assert!(decode_e4m3(0x7f).is_nan());
    }

    #[test]
    fn encoding_saturates_and_rounds_ties_to_even() {
        assert_eq!(encode_e4m3(f32::INFINITY), 0x7e);
        assert_eq!(encode_e4m3(1000.0), 0x7e);
        // Halfway between 1.0 (mantissa 0, even) and 1.125 (mantissa 1).
        assert_eq!(encode_e4m3(1.0625), 0x38);
        // Halfway between mantissas 1 and 2 rounds to mantissa 2.
        assert_eq!(encode_e4m3(1.1875), 0x3a);
    }

    #[test]
    fn absmax_quantization_preserves_shape_and_range() {
        let source = Tensor::from_f32(vec![2, 2], &[-2.0, -0.5, 0.5, 2.0]).unwrap();
        let quantized = quantize_e4m3_absmax(&source).unwrap();
        assert_eq!(quantized.values.dtype(), DType::F8E4M3);
        assert_eq!(quantized.values.shape(), source.shape());
        assert_eq!(quantized.amax, 2.0);
        let output = dequantize_e4m3(&quantized).unwrap();
        let output = output.as_f32().unwrap();
        assert_eq!(output[0], -2.0);
        assert_eq!(output[3], 2.0);
    }

    fn test_config() -> Pi05Config {
        let mut config = Pi05Config::thor_two_view();
        config.vision_depth = 1;
        config.language.depth = 1;
        config.action_expert.depth = 1;
        config
    }

    fn profile(config: &Pi05Config) -> serde_json::Value {
        let sites = Pi05CalibrationPlan::for_config(config).sites().to_vec();
        let scales = sites
            .iter()
            .map(|site| {
                (
                    site.clone(),
                    serde_json::json!({"amax": 448.0, "scale": 1.0}),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        serde_json::json!({
            "schema": CALIBRATION_SCHEMA,
            "model": {"family": "pi05", "checkpoint": "sha256:test"},
            "quantization": {
                "format": FP8_FORMAT,
                "statistic": CALIBRATION_STATISTIC,
                "scale_rule": CALIBRATION_SCALE_RULE,
                "margin": 1.0
            },
            "calibration_data": {
                "identity": "dataset:test",
                "kind": "representative",
                "production": true,
                "sample_count": 2
            },
            "seed_policy": {
                "algorithm": "numpy-pcg64-seed-sequence-v1",
                "base_seed": 0,
                "sample_sequence": "[base_seed,sample_index]"
            },
            "source_revision": "abc123",
            "device": {"requested": "cuda:0", "host": "test-host"},
            "plan": {"sites": sites},
            "observed_sites": Pi05CalibrationPlan::for_config(config).sites(),
            "scales": scales
        })
    }

    fn parse(value: &serde_json::Value, config: &Pi05Config) -> Result<StaticFp8Calibration> {
        StaticFp8Calibration::from_json_str(
            &serde_json::to_string(value).unwrap(),
            config,
            "sha256:test",
        )
    }

    #[test]
    fn strict_profile_accepts_exact_execution_plan() {
        let config = test_config();
        let plan = Pi05CalibrationPlan::for_config(&config);
        assert!(plan
            .sites()
            .contains(&"language.layers.0.attention_norm".to_owned()));
        assert!(!plan
            .sites()
            .contains(&"language.layers.0.attention_output".to_owned()));
        let calibration = parse(&profile(&config), &config).unwrap();
        assert_eq!(calibration.len(), plan.sites().len());
        assert_eq!(calibration.scale("vision.patch_input").unwrap(), 1.0);
    }

    #[test]
    fn strict_profile_rejects_incompatible_contract() {
        let config = test_config();
        for (pointer, value) in [
            ("/schema", serde_json::json!("wrong-schema")),
            ("/model/family", serde_json::json!("other-model")),
            ("/quantization/format", serde_json::json!("e5m2")),
            ("/source_revision", serde_json::json!("unknown")),
        ] {
            let mut document = profile(&config);
            *document.pointer_mut(pointer).unwrap() = value;
            assert!(
                parse(&document, &config).is_err(),
                "accepted mutation at {pointer}"
            );
        }
    }

    #[test]
    fn checkpoint_identity_mismatch_warns_without_blocking_profile_load() {
        let config = test_config();
        let mut document = profile(&config);
        document["model"]["checkpoint"] = serde_json::json!("sha256:other-checkpoint");

        let calibration = parse(&document, &config).unwrap();

        assert_eq!(calibration.warnings().len(), 1);
        assert!(calibration.warnings()[0].contains("checkpoint identity mismatch"));
    }

    #[test]
    fn strict_profile_rejects_missing_unknown_ambiguous_and_non_finite_data() {
        let config = test_config();
        let mut missing = profile(&config);
        missing["scales"]
            .as_object_mut()
            .unwrap()
            .remove("vision.patch_input");
        assert!(parse(&missing, &config).is_err());

        let mut unknown = profile(&config);
        unknown["unexpected"] = serde_json::json!(true);
        assert!(parse(&unknown, &config).is_err());

        let mut ambiguous = profile(&config);
        ambiguous["scales"]["vision.patch_input"]["scale"] = serde_json::json!(2.0);
        assert!(parse(&ambiguous, &config).is_err());

        let mut overflow = profile(&config);
        overflow["quantization"]["margin"] = serde_json::json!(2.0);
        for entry in overflow["scales"].as_object_mut().unwrap().values_mut() {
            entry["scale"] = serde_json::json!(2.0);
        }
        overflow["scales"]["vision.patch_input"]["amax"] = serde_json::json!(f32::MAX);
        assert!(parse(&overflow, &config).is_err());

        let mut mislabeled = profile(&config);
        mislabeled["calibration_data"]["identity"] =
            serde_json::json!("synthetic:zero-observation-v1");
        assert!(parse(&mislabeled, &config).is_err());

        let duplicate = serde_json::to_string(&profile(&config)).unwrap().replace(
            "\"scales\":{",
            "\"scales\":{\"vision.patch_input\":{\"amax\":448.0,\"scale\":1.0},",
        );
        assert!(StaticFp8Calibration::from_json_str(&duplicate, &config, "sha256:test").is_err());

        let raw = serde_json::to_string(&profile(&config))
            .unwrap()
            .replace("\"amax\":448.0", "\"amax\":1e400");
        assert!(StaticFp8Calibration::from_json_str(&raw, &config, "sha256:test").is_err());
    }

    #[test]
    fn sha256_matches_the_standard_vector() {
        let mut digest = Sha256::new();
        digest.update(b"abc");
        assert_eq!(
            digest.finish_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn checkpoint_identity_matches_shared_cross_language_fixture() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/checkpoint_identity");
        let expected = std::fs::read_to_string(fixture.join("expected.sha256"))
            .unwrap()
            .trim()
            .to_owned();

        assert_eq!(checkpoint_identity(&fixture).unwrap(), expected);
        assert_eq!(
            checkpoint_identity(&fixture.join("model.safetensors.index.json")).unwrap(),
            expected
        );
    }
}

#[cfg(feature = "cuda")]
pub use activation_scales::Fp8StaticActivationScales;
#[cfg(feature = "cuda")]
mod activation_scales {
    use crate::pi05::{
        Fp8StaticTransformerLayerScales, Fp8StaticVisionLayerScales, LayerCalibrationSites,
        Pi05CalibrationPlan, Pi05Config, StaticFp8Calibration,
    };
    use apxinf_core::{Error, Result};
    #[derive(Clone, Debug)]
    pub struct Fp8StaticActivationScales {
        pub vision_patch_input: f32,
        pub vision_layers: Vec<Fp8StaticVisionLayerScales>,
        pub vision_post_norm: f32,
        pub language_layers: Vec<Fp8StaticTransformerLayerScales>,
        pub action_input: f32,
        pub time_input: f32,
        pub time_hidden: f32,
        pub conditioning: f32,
        pub action_layers: Vec<Fp8StaticTransformerLayerScales>,
        pub action_final_norm: f32,
    }

    impl Fp8StaticActivationScales {
        /// Resolve every graph activation scale from a named calibration file.
        pub fn from_calibration(
            config: &Pi05Config,
            calibration: &StaticFp8Calibration,
        ) -> Result<Self> {
            let plan = Pi05CalibrationPlan::for_config(config);
            let optional_scale = |site: &Option<String>| -> Result<f32> {
                site.as_deref()
                    .map(|name| calibration.scale(name))
                    .transpose()
                    .map(|scale| scale.unwrap_or(1.0))
            };
            let transformer_layer =
                |sites: &LayerCalibrationSites| -> Result<Fp8StaticTransformerLayerScales> {
                    Ok(Fp8StaticTransformerLayerScales {
                        attention_norm: calibration.scale(&sites.attention_norm)?,
                        attention_output: optional_scale(&sites.attention_output)?,
                        mlp_norm: optional_scale(&sites.mlp_norm)?,
                        mlp_activation: optional_scale(&sites.mlp_activation)?,
                    })
                };
            let vision_layers = plan
                .vision_layers()
                .iter()
                .map(|sites| {
                    Ok(Fp8StaticVisionLayerScales {
                        attention_norm: calibration.scale(&sites.attention_norm)?,
                        attention_output: calibration
                            .scale(sites.attention_output.as_deref().expect("vision tail site"))?,
                        mlp_norm: calibration
                            .scale(sites.mlp_norm.as_deref().expect("vision tail site"))?,
                        mlp_activation: calibration
                            .scale(sites.mlp_activation.as_deref().expect("vision tail site"))?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let language_layers = plan
                .language_layers()
                .iter()
                .map(transformer_layer)
                .collect::<Result<Vec<_>>>()?;
            let action_layers = plan
                .action_layers()
                .iter()
                .map(transformer_layer)
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                vision_patch_input: calibration.scale("vision.patch_input")?,
                vision_layers,
                vision_post_norm: calibration.scale("vision.post_norm")?,
                language_layers,
                action_input: calibration.scale("action.input")?,
                time_input: calibration.scale("time.input")?,
                time_hidden: calibration.scale("time.hidden")?,
                conditioning: calibration.scale("action.conditioning")?,
                action_layers,
                action_final_norm: calibration.scale("action.final_norm")?,
            })
        }

        /// Useful for kernel smoke tests. Production inference should load named,
        /// measured scales from `StaticFp8Calibration`.
        pub fn uniform(config: &Pi05Config, scale: f32) -> Result<Self> {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(Error::Other(format!("invalid uniform FP8 scale {scale}")));
            }
            let transformer = Fp8StaticTransformerLayerScales {
                attention_norm: scale,
                attention_output: scale,
                mlp_norm: scale,
                mlp_activation: scale,
            };
            let vision = Fp8StaticVisionLayerScales {
                attention_norm: scale,
                attention_output: scale,
                mlp_norm: scale,
                mlp_activation: scale,
            };
            Ok(Self {
                vision_patch_input: scale,
                vision_layers: vec![vision; config.vision_depth],
                vision_post_norm: scale,
                language_layers: vec![transformer; config.language.depth],
                action_input: scale,
                time_input: scale,
                time_hidden: scale,
                conditioning: scale,
                action_layers: vec![transformer; config.action_expert.depth],
                action_final_norm: scale,
            })
        }

        pub(in crate::pi05) fn validate(&self, config: &Pi05Config) -> Result<()> {
            if self.vision_layers.len() != config.vision_depth
                || self.language_layers.len() != config.language.depth
                || self.action_layers.len() != config.action_expert.depth
            {
                return Err(Error::Other(
                    "π0.5 activation calibration depth mismatch".into(),
                ));
            }
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Fp8StaticTransformerLayerScales {
    /// Output scale for the first (Ada)RMSNorm.
    pub attention_norm: f32,
    /// Input scale for the attention output projection.
    pub attention_output: f32,
    /// Output scale for the second (Ada)RMSNorm.
    pub mlp_norm: f32,
    /// Output scale for GELU-tanh(gate) * up.
    pub mlp_activation: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Fp8StaticVisionLayerScales {
    pub attention_norm: f32,
    pub attention_output: f32,
    pub mlp_norm: f32,
    pub mlp_activation: f32,
}
