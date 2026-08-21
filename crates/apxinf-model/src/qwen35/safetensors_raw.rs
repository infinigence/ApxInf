//! Minimal SafeTensors reader for the raw tensor map.
//!
//! `apxinf-loader` rejects `I32`/`I64` tensors (packed AWQ weights), so this
//! module reads shards directly and returns every tensor as raw bytes with
//! its dtype string and shape intact.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct RawTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl RawTensor {
    pub fn as_i32(&self) -> Result<Vec<i32>, String> {
        if self.dtype != "I32" {
            return Err(format!("tensor dtype is {}, not I32", self.dtype));
        }
        Ok(self
            .bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn as_bf16(&self) -> Result<Vec<half::bf16>, String> {
        if self.dtype != "BF16" {
            return Err(format!("tensor dtype is {}, not BF16", self.dtype));
        }
        Ok(self
            .bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    }

    pub fn as_f32(&self) -> Result<Vec<f32>, String> {
        if self.dtype != "F32" {
            return Err(format!("tensor dtype is {}, not F32", self.dtype));
        }
        Ok(self
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

/// Load every tensor from a checkpoint directory (index + shards) or a single
/// `model.safetensors` file.
pub fn load_checkpoint(dir: &Path) -> Result<HashMap<String, RawTensor>, String> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        return load_sharded(&index_path);
    }

    let single = dir.join("model.safetensors");
    if !single.is_file() {
        return Err(format!(
            "neither {} nor {} exist",
            index_path.display(),
            single.display()
        ));
    }
    read_file_tensors(&single)
}

fn load_sharded(index_path: &Path) -> Result<HashMap<String, RawTensor>, String> {
    let raw = std::fs::read_to_string(index_path)
        .map_err(|e| format!("read {}: {e}", index_path.display()))?;
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("parse index: {e}"))?;
    let weight_map = parsed
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "index.json has no weight_map object".to_string())?;

    let mut shards: BTreeSet<PathBuf> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(PathBuf::from))
        .collect();
    if shards.is_empty() {
        shards.insert(PathBuf::from("model.safetensors"));
    }

    let parent = index_path.parent().unwrap_or_else(|| Path::new("."));
    let mut out = HashMap::with_capacity(weight_map.len());
    for shard in shards {
        let file = parent.join(&shard);
        let tensors = read_file_tensors(&file)?;
        out.extend(tensors);
    }
    Ok(out)
}

fn read_file_tensors(file: &Path) -> Result<HashMap<String, RawTensor>, String> {
    let data = std::fs::read(file).map_err(|e| format!("read {}: {e}", file.display()))?;
    if data.len() < 8 {
        return Err(format!("{} is not a SafeTensors file", file.display()));
    }

    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    if data.len() < 8 + header_len {
        return Err(format!("truncated header in {}", file.display()));
    }
    let header: serde_json::Value = serde_json::from_slice(&data[8..8 + header_len])
        .map_err(|e| format!("parse header of {}: {e}", file.display()))?;
    let base = 8 + header_len;
    let entries = header
        .as_object()
        .ok_or_else(|| format!("header of {} is not an object", file.display()))?;

    let mut out = HashMap::with_capacity(entries.len());
    for (name, info) in entries {
        let dtype = info
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let shape = info
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|x| x as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let offsets = info
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("missing data_offsets for {name}"))?;
        let start = offsets
            .first()
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("bad start offset for {name}"))?
            as usize;
        let end = offsets
            .get(1)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("bad end offset for {name}"))?
            as usize;
        if base + end > data.len() {
            return Err(format!("data out of range for {name}"));
        }
        out.insert(
            name.clone(),
            RawTensor {
                dtype,
                shape,
                bytes: data[base + start..base + end].to_vec(),
            },
        );
    }
    Ok(out)
}
