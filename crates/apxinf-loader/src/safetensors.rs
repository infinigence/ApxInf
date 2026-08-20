//! SafeTensors format loader.
//!
//! Format spec:
//!   [8 bytes LE u64] header_size
//!   [header_size bytes] JSON metadata
//!   [rest of file] raw tensor data
//!
//! JSON header maps tensor name → { "dtype", "shape", "data_offsets": [start, end] }
//! Offsets are relative to the start of the data section (after the header).

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::{Component, Path};

use memmap2::Mmap;
use serde::Deserialize;

use bytemuck;
use apxinf_core::{DType, Device, Shape, Tensor};

use crate::config::ModelConfig;

/// A raw tensor entry from the SafeTensors JSON header.
#[derive(Debug, Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Load a SafeTensors file. Returns all tensors on CPU.
///
/// **BF16 tensors are upcast to F32 at load time.** This is the legacy path
/// used by TinyLlama (fp32 workspace). For the bf16 CUDA path (Qwen3-VL,
/// TinyLlama-bf16), use `load_native` which preserves the on-disk dtype.
///
/// # Returns
/// `(tensors, config_metadata)` — the tensor map and any metadata
/// key-value strings found in the `__metadata__` header entry.
pub fn load(path: &Path) -> Result<(HashMap<String, Tensor>, HashMap<String, String>), String> {
    load_impl(path, /* upcast_bf16 */ true)
}

/// Load a SafeTensors file preserving on-disk dtype (no BF16 → F32 upcast).
///
/// Use this when the target device is CUDA and you want native bf16
/// tensors (roughly halves memory + weight-streaming bandwidth vs the
/// upcasting `load()`). The returned tensors sit on CPU as bf16; call
/// your backend's `to_device` to move them.
pub fn load_native(
    path: &Path,
) -> Result<(HashMap<String, Tensor>, HashMap<String, String>), String> {
    load_impl(path, /* upcast_bf16 */ false)
}

/// Load either one SafeTensors file, a Hugging Face SafeTensors index, or a
/// checkpoint directory containing `model.safetensors.index.json`.
pub fn load_native_path(
    path: &Path,
) -> Result<(HashMap<String, Tensor>, HashMap<String, String>), String> {
    if path.is_dir() {
        let index = path.join("model.safetensors.index.json");
        if index.is_file() {
            return load_native_sharded(&index);
        }
        let model = path.join("model.safetensors");
        if model.is_file() {
            return load_native(&model);
        }
        let mut candidates = std::fs::read_dir(path)
            .map_err(|e| {
                format!(
                    "failed to read checkpoint directory {}: {e}",
                    path.display()
                )
            })?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|candidate| {
                candidate.extension().and_then(|value| value.to_str()) == Some("safetensors")
            })
            .collect::<Vec<_>>();
        candidates.sort();
        return match candidates.as_slice() {
            [only] => load_native(only),
            [] => Err(format!(
                "no SafeTensors model or index in {}",
                path.display()
            )),
            _ => Err(format!(
                "multiple SafeTensors files but no model.safetensors.index.json in {}",
                path.display()
            )),
        };
    }
    if path.extension().and_then(|value| value.to_str()) == Some("json") {
        load_native_sharded(path)
    } else {
        load_native(path)
    }
}

/// Load all shards named by a Hugging Face `*.safetensors.index.json` file.
/// Each indexed tensor is checked against the shard in which it was found.
pub fn load_native_sharded(
    index_path: &Path,
) -> Result<(HashMap<String, Tensor>, HashMap<String, String>), String> {
    let raw = std::fs::read_to_string(index_path)
        .map_err(|e| format!("failed to read {}: {e}", index_path.display()))?;
    let index: SafetensorsIndex = serde_json::from_str(&raw)
        .map_err(|e| format!("invalid SafeTensors index {}: {e}", index_path.display()))?;
    if index.weight_map.is_empty() {
        return Err(format!(
            "SafeTensors index {} has an empty weight_map",
            index_path.display()
        ));
    }
    let parent = index_path.parent().unwrap_or_else(|| Path::new("."));
    let shards = index.weight_map.values().cloned().collect::<BTreeSet<_>>();
    let mut tensors = HashMap::with_capacity(index.weight_map.len());
    let mut metadata = HashMap::new();

    for shard in shards {
        let shard_path = Path::new(&shard);
        if shard_path.is_absolute()
            || shard_path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
        {
            return Err(format!(
                "unsafe shard path `{shard}` in {}",
                index_path.display()
            ));
        }
        let (shard_tensors, shard_metadata) = load_native(&parent.join(&shard))?;
        metadata.extend(shard_metadata);
        for (name, tensor) in shard_tensors {
            let Some(expected_shard) = index.weight_map.get(&name) else {
                continue;
            };
            if expected_shard != &shard {
                return Err(format!(
                    "tensor `{name}` was found in `{shard}`, index assigns it to `{expected_shard}`"
                ));
            }
            if tensors.insert(name.clone(), tensor).is_some() {
                return Err(format!("duplicate indexed tensor `{name}`"));
            }
        }
    }
    let missing = index
        .weight_map
        .keys()
        .filter(|name| !tensors.contains_key(*name))
        .take(8)
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "SafeTensors shards are missing indexed tensors: {}",
            missing.join(", ")
        ));
    }
    Ok((tensors, metadata))
}

fn load_impl(
    path: &Path,
    upcast_bf16: bool,
) -> Result<(HashMap<String, Tensor>, HashMap<String, String>), String> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap failed: {e}"))? };

    if mmap.len() < 8 {
        return Err("file too small to be a SafeTensors file".into());
    }

    // Read 8-byte LE header length
    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;

    if mmap.len() < 8 + header_len {
        return Err(format!(
            "file too small: expected at least {} bytes, got {}",
            8 + header_len,
            mmap.len()
        ));
    }

    // Parse JSON header
    let header_json = std::str::from_utf8(&mmap[8..8 + header_len])
        .map_err(|e| format!("invalid UTF-8 in header: {e}"))?;

    let raw: HashMap<String, serde_json::Value> =
        serde_json::from_str(header_json).map_err(|e| format!("JSON parse error: {e}"))?;

    // Extract metadata (optional __metadata__ key)
    let mut metadata = HashMap::new();
    if let Some(serde_json::Value::Object(meta)) = raw.get("__metadata__") {
        for (k, v) in meta {
            if let Some(s) = v.as_str() {
                metadata.insert(k.clone(), s.to_string());
            }
        }
    }

    // Data section starts after the 8-byte length + header
    let data_start = 8 + header_len;

    let mut tensors = HashMap::new();

    for (name, value) in &raw {
        if name == "__metadata__" {
            continue;
        }

        let info: TensorInfo = serde_json::from_value(value.clone())
            .map_err(|e| format!("failed to parse tensor '{name}': {e}"))?;

        let dtype = parse_dtype(&info.dtype)
            .ok_or_else(|| format!("unsupported dtype '{}' for tensor '{name}'", info.dtype))?;

        let [start, end] = info.data_offsets;
        let abs_start = data_start + start;
        let abs_end = data_start + end;

        if abs_end > mmap.len() {
            return Err(format!(
                "tensor '{name}': data_offsets [{start}, {end}] exceed file size {}",
                mmap.len()
            ));
        }

        let bytes = mmap[abs_start..abs_end].to_vec();
        let shape = Shape::from(info.shape);

        // BF16: upcast to F32 for the legacy path, or keep native for bf16 CUDA.
        let tensor = if dtype == DType::BF16 && upcast_bf16 {
            let bf16_bytes: &[u8] = &bytes;
            let bf16_data: &[half::bf16] = bytemuck::cast_slice(bf16_bytes);
            let f32_data: Vec<f32> = bf16_data.iter().map(|b| b.to_f32()).collect();
            Tensor::from_f32(shape.dims().to_vec(), &f32_data)
                .map_err(|e| format!("tensor '{name}' bf16->f32 conversion error: {e}"))?
        } else {
            Tensor::from_raw(shape, dtype, Device::Cpu, bytes)
                .map_err(|e| format!("tensor '{name}' construction error: {e}"))?
        };

        tensors.insert(name.clone(), tensor);
    }

    Ok((tensors, metadata))
}

/// Try to extract a ModelConfig from SafeTensors metadata.
/// Falls back to TinyLlama defaults for any missing fields.
pub fn config_from_metadata(metadata: &HashMap<String, String>) -> ModelConfig {
    // HuggingFace stores config as JSON in the __metadata__ section
    // Key: "config" (or individual keys like "hidden_size", etc.)
    // Try to parse a JSON config blob first
    if let Some(config_json) = metadata.get("config") {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(config_json) {
            return ModelConfig {
                hidden_size: v["hidden_size"].as_u64().unwrap_or(2048) as usize,
                intermediate_size: v["intermediate_size"].as_u64().unwrap_or(5632) as usize,
                n_layers: v["num_hidden_layers"].as_u64().unwrap_or(22) as usize,
                n_heads: v["num_attention_heads"].as_u64().unwrap_or(32) as usize,
                n_kv_heads: v["num_key_value_heads"].as_u64().unwrap_or(4) as usize,
                vocab_size: v["vocab_size"].as_u64().unwrap_or(32000) as usize,
                max_seq_len: v["max_position_embeddings"].as_u64().unwrap_or(2048) as usize,
                rope_theta: v["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
                rms_norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            };
        }
    }
    // Fall back to per-key metadata (some exporters write individual keys)
    ModelConfig {
        hidden_size: parse_meta_usize(metadata, "hidden_size", 2048),
        intermediate_size: parse_meta_usize(metadata, "intermediate_size", 5632),
        n_layers: parse_meta_usize(metadata, "num_hidden_layers", 22),
        n_heads: parse_meta_usize(metadata, "num_attention_heads", 32),
        n_kv_heads: parse_meta_usize(metadata, "num_key_value_heads", 4),
        vocab_size: parse_meta_usize(metadata, "vocab_size", 32000),
        max_seq_len: parse_meta_usize(metadata, "max_position_embeddings", 2048),
        rope_theta: parse_meta_f32(metadata, "rope_theta", 10000.0),
        rms_norm_eps: parse_meta_f32(metadata, "rms_norm_eps", 1e-5),
    }
}

fn parse_meta_usize(m: &HashMap<String, String>, key: &str, default: usize) -> usize {
    m.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn parse_meta_f32(m: &HashMap<String, String>, key: &str, default: f32) -> f32 {
    m.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn parse_dtype(s: &str) -> Option<DType> {
    match s {
        "F32" => Some(DType::F32),
        "F16" => Some(DType::F16),
        "BF16" => Some(DType::BF16),
        "F8_E4M3" => Some(DType::F8E4M3),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Build a minimal SafeTensors file in memory.
    fn make_safetensors(tensors: &[(&str, DType, &[usize], &[u8])]) -> Vec<u8> {
        // Build JSON header
        let mut data_offset = 0usize;
        let mut tensor_entries = Vec::new();

        for (name, dtype, shape, data) in tensors {
            let dtype_str = match dtype {
                DType::F32 => "F32",
                DType::F16 => "F16",
                DType::BF16 => "BF16",
                DType::F8E4M3 => "F8_E4M3",
            };
            let shape_json: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            let end = data_offset + data.len();
            tensor_entries.push(format!(
                r#""{name}": {{"dtype": "{dtype_str}", "shape": [{shape}], "data_offsets": [{start}, {end}]}}"#,
                shape = shape_json.join(", "),
                start = data_offset,
            ));
            data_offset = end;
        }

        let header_json = format!("{{{}}}", tensor_entries.join(", "));
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut out = Vec::new();
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(header_bytes);
        for (_, _, _, data) in tensors {
            out.extend_from_slice(data);
        }
        out
    }

    #[test]
    fn test_load_f32_tensor() {
        let data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|x: &f32| x.to_le_bytes())
            .collect();
        let file_bytes = make_safetensors(&[("weight", DType::F32, &[2, 2], &data)]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();

        let (tensors, _) = load(tmp.path()).unwrap();
        assert!(tensors.contains_key("weight"));
        let t = &tensors["weight"];
        assert_eq!(t.shape().dims(), &[2, 2]);
        assert_eq!(t.dtype(), DType::F32);
        let got = t.as_f32().unwrap();
        assert_eq!(got, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_load_multiple_tensors() {
        let a_data: Vec<u8> = vec![1.0f32, 2.0]
            .iter()
            .flat_map(|x: &f32| x.to_le_bytes())
            .collect();
        let b_data: Vec<u8> = vec![10.0f32, 20.0, 30.0]
            .iter()
            .flat_map(|x: &f32| x.to_le_bytes())
            .collect();
        let file_bytes = make_safetensors(&[
            ("a", DType::F32, &[2], &a_data),
            ("b", DType::F32, &[3], &b_data),
        ]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();

        let (tensors, _) = load(tmp.path()).unwrap();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors["a"].numel(), 2);
        assert_eq!(tensors["b"].numel(), 3);
    }

    #[test]
    fn test_load_bf16_tensor() {
        use half::bf16;
        let bf16_data: Vec<u8> = vec![bf16::from_f32(1.0), bf16::from_f32(2.0)]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let file_bytes = make_safetensors(&[("w", DType::BF16, &[2], &bf16_data)]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();

        let (tensors, _) = load(tmp.path()).unwrap();
        // BF16 is converted to F32 during loading for compatibility
        assert_eq!(tensors["w"].dtype(), DType::F32);
        let data = tensors["w"].as_f32().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-3);
        assert!((data[1] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_load_native_bf16_preserves_dtype() {
        use half::bf16;
        let bf16_data: Vec<u8> = vec![bf16::from_f32(1.5), bf16::from_f32(-2.25)]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let file_bytes = make_safetensors(&[("w", DType::BF16, &[2], &bf16_data)]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();

        // load_native keeps bf16 as bf16 — the on-disk dtype is preserved.
        let (tensors, _) = load_native(tmp.path()).unwrap();
        assert_eq!(tensors["w"].dtype(), DType::BF16);
        let data = tensors["w"].as_bf16().unwrap();
        assert!((data[0].to_f32() - 1.5).abs() < 1e-3);
        assert!((data[1].to_f32() - -2.25).abs() < 1e-3);
    }

    #[test]
    fn test_unsupported_dtype_skipped() {
        // Build a header with an INT8 tensor — should fail gracefully
        let header_json = r#"{"x": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]}}"#;
        let header_len = header_json.len() as u64;
        let mut file_bytes = header_len.to_le_bytes().to_vec();
        file_bytes.extend_from_slice(header_json.as_bytes());
        file_bytes.extend_from_slice(&[0u8; 4]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&file_bytes).unwrap();

        let result = load(tmp.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unsupported dtype"));
    }

    #[test]
    fn test_load_native_sharded_index() {
        let directory = tempfile::tempdir().unwrap();
        let a_data = [half::f16::from_f32(1.5)]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let b_data = [half::f16::from_f32(-2.0)]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(
            directory.path().join("model-00001-of-00002.safetensors"),
            make_safetensors(&[("a", DType::F16, &[1], &a_data)]),
        )
        .unwrap();
        std::fs::write(
            directory.path().join("model-00002-of-00002.safetensors"),
            make_safetensors(&[("b", DType::F16, &[1], &b_data)]),
        )
        .unwrap();
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let (tensors, _) = load_native_path(directory.path()).unwrap();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors["a"].dtype(), DType::F16);
        assert_eq!(tensors["b"].as_f16().unwrap()[0].to_f32(), -2.0);
    }
}
