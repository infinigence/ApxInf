//! Zero-copy, dtype-agnostic SafeTensors archive.
//!
//! [`crate::safetensors::load_native_path`] materialises every tensor as an
//! `apxinf_core::Tensor`, which requires a supported [`apxinf_core::DType`]
//! and copies each tensor into host memory. Quantized checkpoints (AutoAWQ,
//! GPTQ) carry packed `I32` tensors and tens of thousands of small expert
//! tensors, so they need a different access pattern: keep every shard
//! memory-mapped, resolve tensor names through the Hugging Face index, and
//! hand out borrowed byte slices with their on-disk dtype string.
//!
//! Model code decides how to interpret the bytes (for example uploading packed
//! INT4 weights straight to a device buffer). No new `DType` variants are
//! introduced here.

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::{Component, Path, PathBuf};

use memmap2::Mmap;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct RawTensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Location of one tensor inside a mapped shard.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    shard: usize,
    start: usize,
    end: usize,
}

impl TensorEntry {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_len(&self) -> usize {
        self.end - self.start
    }
}

/// Borrowed view of a tensor's raw bytes.
#[derive(Debug, Clone, Copy)]
pub struct TensorBytes<'a> {
    pub dtype: &'a str,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

/// One or more memory-mapped SafeTensors shards addressed by tensor name.
pub struct SafetensorsArchive {
    shards: Vec<Mmap>,
    shard_paths: Vec<PathBuf>,
    entries: HashMap<String, TensorEntry>,
    metadata: HashMap<String, String>,
}

impl SafetensorsArchive {
    /// Open a checkpoint file, a `*.safetensors.index.json`, or a directory
    /// containing either `model.safetensors.index.json` or a single
    /// `*.safetensors` file.
    pub fn open(path: &Path) -> Result<Self, String> {
        if path.is_dir() {
            let index = path.join("model.safetensors.index.json");
            if index.is_file() {
                return Self::open_index(&index);
            }
            let model = path.join("model.safetensors");
            if model.is_file() {
                return Self::open_files(&[model]);
            }
            let mut candidates = std::fs::read_dir(path)
                .map_err(|e| format!("failed to read checkpoint directory {}: {e}", path.display()))?
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|candidate| {
                    candidate.extension().and_then(|value| value.to_str()) == Some("safetensors")
                })
                .collect::<Vec<_>>();
            candidates.sort();
            return match candidates.len() {
                1 => Self::open_files(&candidates),
                0 => Err(format!("no SafeTensors model or index in {}", path.display())),
                _ => Err(format!(
                    "multiple SafeTensors files but no model.safetensors.index.json in {}",
                    path.display()
                )),
            };
        }
        if path.extension().and_then(|value| value.to_str()) == Some("json") {
            Self::open_index(path)
        } else {
            Self::open_files(std::slice::from_ref(&path.to_path_buf()))
        }
    }

    /// Open every shard referenced by a Hugging Face index and verify that each
    /// indexed tensor lives in its assigned shard.
    pub fn open_index(index_path: &Path) -> Result<Self, String> {
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
        let shard_names = index.weight_map.values().cloned().collect::<BTreeSet<_>>();
        let mut paths = Vec::with_capacity(shard_names.len());
        for shard in &shard_names {
            let shard_path = Path::new(shard);
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
            paths.push(parent.join(shard));
        }
        let archive = Self::open_files(&paths)?;
        for (name, shard) in &index.weight_map {
            let Some(entry) = archive.entries.get(name) else {
                return Err(format!("SafeTensors shards are missing indexed tensor `{name}`"));
            };
            let actual = archive.shard_paths[entry.shard]
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("");
            if actual != shard {
                return Err(format!(
                    "tensor `{name}` was found in `{actual}`, index assigns it to `{shard}`"
                ));
            }
        }
        Ok(archive)
    }

    /// Map an explicit list of shard files.
    pub fn open_files(paths: &[PathBuf]) -> Result<Self, String> {
        let mut shards = Vec::with_capacity(paths.len());
        let mut shard_paths = Vec::with_capacity(paths.len());
        let mut entries = HashMap::new();
        let mut metadata = HashMap::new();
        for (shard_index, path) in paths.iter().enumerate() {
            let file =
                File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
            let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap failed: {e}"))? };
            if mmap.len() < 8 {
                return Err(format!("{} is too small to be a SafeTensors file", path.display()));
            }
            let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            if mmap.len() < 8 + header_len {
                return Err(format!(
                    "{}: header length {header_len} exceeds file size {}",
                    path.display(),
                    mmap.len()
                ));
            }
            let header = std::str::from_utf8(&mmap[8..8 + header_len])
                .map_err(|e| format!("{}: invalid UTF-8 in header: {e}", path.display()))?;
            let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header)
                .map_err(|e| format!("{}: JSON parse error: {e}", path.display()))?;
            let data_start = 8 + header_len;
            for (name, value) in raw {
                if name == "__metadata__" {
                    if let serde_json::Value::Object(meta) = value {
                        for (k, v) in meta {
                            if let Some(s) = v.as_str() {
                                metadata.insert(k, s.to_string());
                            }
                        }
                    }
                    continue;
                }
                let info: RawTensorInfo = serde_json::from_value(value)
                    .map_err(|e| format!("failed to parse tensor '{name}': {e}"))?;
                let [start, end] = info.data_offsets;
                if start > end || data_start + end > mmap.len() {
                    return Err(format!(
                        "tensor '{name}': data_offsets [{start}, {end}] exceed file size {}",
                        mmap.len()
                    ));
                }
                let expected = info.shape.iter().product::<usize>()
                    * dtype_size(&info.dtype).unwrap_or(0);
                if expected != 0 && expected != end - start {
                    return Err(format!(
                        "tensor '{name}': {} bytes on disk, shape {:?} of {} needs {expected}",
                        end - start,
                        info.shape,
                        info.dtype
                    ));
                }
                let entry = TensorEntry {
                    dtype: info.dtype,
                    shape: info.shape,
                    shard: shard_index,
                    start: data_start + start,
                    end: data_start + end,
                };
                if entries.insert(name.clone(), entry).is_some() {
                    return Err(format!("duplicate tensor `{name}` across shards"));
                }
            }
            shards.push(mmap);
            shard_paths.push(path.clone());
        }
        Ok(Self {
            shards,
            shard_paths,
            entries,
            metadata,
        })
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn entry(&self, name: &str) -> Option<&TensorEntry> {
        self.entries.get(name)
    }

    /// Borrow a tensor's raw bytes together with its dtype string and shape.
    pub fn get(&self, name: &str) -> Result<TensorBytes<'_>, String> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| format!("tensor `{name}` not found in archive"))?;
        Ok(TensorBytes {
            dtype: &entry.dtype,
            shape: &entry.shape,
            bytes: &self.shards[entry.shard][entry.start..entry.end],
        })
    }

    /// Borrow a tensor and check dtype and shape in one step.
    pub fn get_checked(
        &self,
        name: &str,
        dtype: &str,
        shape: &[usize],
    ) -> Result<TensorBytes<'_>, String> {
        let view = self.get(name)?;
        if view.dtype != dtype || view.shape != shape {
            return Err(format!(
                "tensor `{name}` is {} {:?}, expected {dtype} {shape:?}",
                view.dtype, view.shape
            ));
        }
        Ok(view)
    }
}

/// Element size for the SafeTensors dtype strings this crate recognises.
/// Returns `None` for unknown strings so callers can still address the bytes.
pub fn dtype_size(dtype: &str) -> Option<usize> {
    Some(match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "F8_E4M3" | "F8_E5M2" | "I8" | "U8" | "BOOL" => 1,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_safetensors(tensors: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        let mut offset = 0usize;
        let mut entries = Vec::new();
        for (name, dtype, shape, data) in tensors {
            let shape_json: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            let end = offset + data.len();
            entries.push(format!(
                r#""{name}": {{"dtype": "{dtype}", "shape": [{}], "data_offsets": [{offset}, {end}]}}"#,
                shape_json.join(", ")
            ));
            offset = end;
        }
        let header = format!("{{{}}}", entries.join(", "));
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        for (_, _, _, data) in tensors {
            out.extend_from_slice(data);
        }
        out
    }

    #[test]
    fn reads_i32_tensors_zero_copy() {
        let values: Vec<u8> = [7i32, -3, 0x0123_4567]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = make_safetensors(&[("w.qweight", "I32", &[1, 3], &values)]);
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&bytes).unwrap();
        let archive = SafetensorsArchive::open(tmp.path()).unwrap();
        let view = archive.get_checked("w.qweight", "I32", &[1, 3]).unwrap();
        assert_eq!(view.bytes, values.as_slice());
        assert!(archive.get_checked("w.qweight", "F16", &[1, 3]).is_err());
        assert!(archive.get("missing").is_err());
    }

    #[test]
    fn opens_sharded_index_and_validates_assignment() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("model-00001-of-00002.safetensors"),
            make_safetensors(&[("a", "F16", &[2], &[1, 0, 2, 0])]),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model-00002-of-00002.safetensors"),
            make_safetensors(&[("b", "I32", &[1], &[1, 2, 3, 4])]),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let archive = SafetensorsArchive::open(dir.path()).unwrap();
        assert_eq!(archive.len(), 2);
        assert_eq!(archive.get("b").unwrap().bytes, &[1, 2, 3, 4]);

        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model-00002-of-00002.safetensors","b":"model-00001-of-00002.safetensors"}}"#,
        )
        .unwrap();
        assert!(SafetensorsArchive::open(dir.path()).is_err());
    }

    #[test]
    fn rejects_size_mismatch() {
        let bytes = make_safetensors(&[("x", "F32", &[2], &[0u8; 4])]);
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&bytes).unwrap();
        assert!(SafetensorsArchive::open(tmp.path()).is_err());
    }
}
