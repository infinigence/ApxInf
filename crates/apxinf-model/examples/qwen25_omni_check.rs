//! Exact-token verifier for the pinned native Qwen2.5-Omni Thinker path.
//!
//! Python is used only to unpack the canonical `.npz`; model inference is
//! exclusively the native ApxInf CUDA implementation.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use apxinf_core::{DType, Device, Tensor};
use apxinf_model::{AudioInput, AutoModel, ImageInput, LlmInput, LoadOptions, ModelPrecision};
use serde_json::Value;

const MODEL_ID: &str = "Qwen/Qwen2.5-Omni-3B";
const MODEL_REVISION: &str = "f75b40e3da2003cdd6e1829b1f420ca70797c34e";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: qwen25_omni_check MODEL_DIR REFERENCE.npz".into());
    }
    let model_dir = PathBuf::from(&args[0]);
    let reference_path = PathBuf::from(&args[1]);
    let reference = Reference::load(&reference_path)?;
    if reference.model_id != MODEL_ID || reference.model_revision != MODEL_REVISION {
        return Err(format!(
            "reference identity mismatch: {}@{}",
            reference.model_id, reference.model_revision
        )
        .into());
    }
    if reference.expected.len() != 10 {
        return Err(format!(
            "reference must contain exactly 10 greedy tokens, got {}",
            reference.expected.len()
        )
        .into());
    }

    let options = LoadOptions {
        precision: ModelPrecision::Bf16,
        text_weight_dtype: Some(DType::BF16),
        ..LoadOptions::default()
    };
    let mut model = AutoModel::load_model(Device::Cuda(0), &model_dir, &options)?;
    let image_input = reference
        .image
        .as_ref()
        .map(|image| ImageInput::new(&image.pixels, &image.grids));
    let audio_input = reference.audio.as_ref().map(|audio| {
        AudioInput::new(
            &audio.features,
            &audio.mask,
            &audio.feature_lengths,
            &audio.token_counts,
        )
    });
    let input = LlmInput::with_media(&reference.tokens, image_input, audio_input);
    let (actual, _) = model.generate_streaming(input, 10, |_| {}, None)?;
    let passed = actual == reference.expected;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema":"apxinf.qwen25_omni.reference_check.v1",
            "model_id":MODEL_ID,
            "model_revision":MODEL_REVISION,
            "case":reference.case,
            "prompt_tokens":reference.tokens.len(),
            "expected_tokens":reference.expected,
            "actual_tokens":actual,
            "exact_trajectory":passed,
            "fallback_active":false,
        }))?
    );
    if !passed {
        return Err("native Qwen2.5-Omni token trajectory differs from HF reference".into());
    }
    Ok(())
}

struct Reference {
    _temp: TempReference,
    model_id: String,
    model_revision: String,
    case: String,
    tokens: Vec<u32>,
    expected: Vec<u32>,
    image: Option<ImageReference>,
    audio: Option<AudioReference>,
}

struct ImageReference {
    pixels: Tensor,
    grids: Vec<[u32; 3]>,
}

struct AudioReference {
    features: Tensor,
    mask: Tensor,
    feature_lengths: Vec<u32>,
    token_counts: Vec<u32>,
}

impl Reference {
    fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        if !path.is_file() {
            return Err(format!("reference does not exist: {}", path.display()).into());
        }
        let temp = TempReference::new()?;
        let script = r#"
import json
import pathlib
import sys
import numpy as np

source, output_dir = sys.argv[1:]
data = np.load(source, allow_pickle=False)
required = {"tokens", "greedy_tokens", "model_id", "model_revision", "case"}
missing = sorted(required.difference(data.files))
if missing:
    raise ValueError("reference missing arrays: " + ", ".join(missing))
meta = {
    "tokens": data["tokens"].astype(np.int64).reshape(-1).tolist(),
    "greedy_tokens": data["greedy_tokens"].astype(np.int64).reshape(-1).tolist(),
    "model_id": str(data["model_id"].item()),
    "model_revision": str(data["model_revision"].item()),
    "case": str(data["case"].item()),
}
out = pathlib.Path(output_dir)
if "image_pixel_values" in data.files:
    if "image_grid_thw" not in data.files:
        raise ValueError("image reference missing image_grid_thw")
    np.save(out / "image.npy", data["image_pixel_values"].astype(np.float32))
    meta["image_grid_thw"] = data["image_grid_thw"].astype(np.int64).reshape(-1, 3).tolist()
if "audio_input_features" in data.files:
    audio_required = {
        "audio_attention_mask", "audio_feature_lengths", "audio_token_counts"
    }
    audio_missing = sorted(audio_required.difference(data.files))
    if audio_missing:
        raise ValueError("audio reference missing arrays: " + ", ".join(audio_missing))
    np.save(out / "audio.npy", data["audio_input_features"].astype(np.float32))
    np.save(out / "mask.npy", data["audio_attention_mask"].astype(np.float32))
    meta["audio_feature_lengths"] = data["audio_feature_lengths"].astype(np.int64).reshape(-1).tolist()
    meta["audio_token_counts"] = data["audio_token_counts"].astype(np.int64).reshape(-1).tolist()
print(json.dumps(meta, separators=(",", ":")))
"#;
        let output = Command::new("python3")
            .arg("-c")
            .arg(script)
            .arg(path)
            .arg(&temp.path)
            .output()?;
        if !output.status.success() {
            return Err(format!(
                "extract reference: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
        let metadata: Value = serde_json::from_slice(&output.stdout)?;
        let tokens = u32_array(&metadata, "tokens")?;
        let expected = u32_array(&metadata, "greedy_tokens")?;
        let model_id = string_value(&metadata, "model_id")?;
        let model_revision = string_value(&metadata, "model_revision")?;
        let case = string_value(&metadata, "case")?;

        let image = if let Some(grids) = metadata.get("image_grid_thw") {
            let grids = grids
                .as_array()
                .ok_or("image_grid_thw must be an array")?
                .iter()
                .map(|grid| {
                    let values = grid.as_array().ok_or("image grid must be an array")?;
                    if values.len() != 3 {
                        return Err("image grid must have three elements".into());
                    }
                    Ok([
                        u32_value(&values[0], "grid T")?,
                        u32_value(&values[1], "grid H")?,
                        u32_value(&values[2], "grid W")?,
                    ])
                })
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            Some(ImageReference {
                pixels: read_npy_f32_as_bf16(&temp.path.join("image.npy"))?,
                grids,
            })
        } else {
            None
        };
        let audio = if metadata.get("audio_feature_lengths").is_some() {
            Some(AudioReference {
                features: read_npy_f32_as_bf16(&temp.path.join("audio.npy"))?,
                mask: read_npy_f32_as_bf16(&temp.path.join("mask.npy"))?,
                feature_lengths: u32_array(&metadata, "audio_feature_lengths")?,
                token_counts: u32_array(&metadata, "audio_token_counts")?,
            })
        } else {
            None
        };
        if image.is_some() && audio.is_some() {
            return Err("reference contains unsupported simultaneous image and audio".into());
        }
        match (case.as_str(), image.is_some(), audio.is_some()) {
            ("text", false, false) | ("image", true, false) | ("audio", false, true) => {}
            _ => return Err("reference case does not match its media arrays".into()),
        }
        Ok(Self {
            _temp: temp,
            model_id,
            model_revision,
            case,
            tokens,
            expected,
            image,
            audio,
        })
    }
}

struct TempReference {
    path: PathBuf,
}

impl TempReference {
    fn new() -> Result<Self, std::io::Error> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "apxinf-qwen25-omni-check-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path)?;
        Ok(Self { path })
    }
}

impl Drop for TempReference {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn string_value(value: &Value, key: &str) -> Result<String, Box<dyn std::error::Error>> {
    value[key]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{key} must be a string").into())
}

fn u32_array(value: &Value, key: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    value[key]
        .as_array()
        .ok_or_else(|| format!("{key} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, item)| u32_value(item, &format!("{key}[{index}]")))
        .collect()
}

fn u32_value(value: &Value, name: &str) -> Result<u32, Box<dyn std::error::Error>> {
    value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .ok_or_else(|| format!("{name} must be a u32").into())
}

fn read_npy_f32_as_bf16(path: &Path) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{} is not an NPY array", path.display()).into());
    }
    let major = bytes[6];
    let (header_start, header_len) = if major == 1 {
        (10, u16::from_le_bytes([bytes[8], bytes[9]]) as usize)
    } else if bytes.len() >= 12 {
        (
            12,
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        )
    } else {
        return Err("truncated NPY header".into());
    };
    let data_start = header_start + header_len;
    if data_start > bytes.len() {
        return Err("truncated NPY data".into());
    }
    let header = std::str::from_utf8(&bytes[header_start..data_start])?;
    if !header.contains("'<f4'") && !header.contains("\"<f4\"") {
        return Err(format!("{} must contain little-endian f32", path.display()).into());
    }
    let shape = parse_shape(header)?;
    let raw = &bytes[data_start..];
    if raw.len() % 4 != 0 {
        return Err("NPY f32 payload length is not divisible by four".into());
    }
    let expected = shape.iter().try_fold(1usize, |count, dimension| {
        count.checked_mul(*dimension).ok_or("NPY shape overflow")
    })?;
    if raw.len() / 4 != expected {
        return Err("NPY shape does not match payload length".into());
    }
    let values = raw
        .chunks_exact(4)
        .map(|chunk| half::bf16::from_f32(f32::from_le_bytes(chunk.try_into().unwrap())))
        .collect::<Vec<_>>();
    Ok(Tensor::from_bf16(shape, &values)?)
}

fn parse_shape(header: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let shape_index = header.find("shape").ok_or("NPY header has no shape")?;
    let open = header[shape_index..]
        .find('(')
        .map(|offset| shape_index + offset)
        .ok_or("NPY shape has no opening parenthesis")?;
    let close = header[open..]
        .find(')')
        .map(|offset| open + offset)
        .ok_or("NPY shape has no closing parenthesis")?;
    header[open + 1..close]
        .split(',')
        .filter(|value| !value.trim().is_empty())
        .map(|value| Ok(value.trim().parse::<usize>()?))
        .collect()
}
