//! Sharded F32 safetensors reader for the LingBot-VLA 2.0 checkpoint.
//!
//! The six shards (25.5 GB of F32) are memory-mapped, never read into the heap as a whole. Each
//! tensor is converted on demand, so peak host memory is one tensor (at most the 1.5 GB embedding
//! table, which is converted row-block by row-block) on top of the page cache.

use apxinf_core::{Error, Result};
use apxinf_loader::MappedFile;
use half::bf16;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    shard: usize,
    start: usize,
    end: usize,
}

pub struct Checkpoint {
    files: Vec<MappedFile>,
    tensors: HashMap<String, TensorInfo>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(format!("LingBot-VLA 2.0 checkpoint: {}", message.into()))
}

impl Checkpoint {
    /// Open every `*.safetensors` shard under `dir` and index its tensors.
    pub fn open(dir: &Path) -> Result<Self> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| invalid(format!("read {}: {e}", dir.display())))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(invalid(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        let mut files = Vec::with_capacity(shards.len());
        let mut tensors = HashMap::new();
        for (shard, path) in shards.iter().enumerate() {
            let file =
                MappedFile::open(path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
            let bytes: &[u8] = file.as_ref();
            if bytes.len() < 8 {
                return Err(invalid(format!("{} is truncated", path.display())));
            }
            let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value = serde_json::from_slice(
                bytes
                    .get(8..8 + header_len)
                    .ok_or_else(|| invalid(format!("{} header is truncated", path.display())))?,
            )
            .map_err(|e| invalid(format!("{} header: {e}", path.display())))?;
            let data_start = 8 + header_len;
            for (name, meta) in header.as_object().into_iter().flatten() {
                if name == "__metadata__" {
                    continue;
                }
                if meta["dtype"] != "F32" {
                    return Err(invalid(format!(
                        "{name}: dtype {} is not F32",
                        meta["dtype"]
                    )));
                }
                let shape: Vec<usize> = meta["shape"]
                    .as_array()
                    .ok_or_else(|| invalid(format!("{name}: missing shape")))?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as usize)
                    .collect();
                let offsets = meta["data_offsets"]
                    .as_array()
                    .ok_or_else(|| invalid(format!("{name}: missing data_offsets")))?;
                let (s, e) = (
                    offsets[0].as_u64().unwrap_or(0) as usize,
                    offsets[1].as_u64().unwrap_or(0) as usize,
                );
                if data_start + e > bytes.len() || e - s != shape.iter().product::<usize>() * 4 {
                    return Err(invalid(format!(
                        "{name}: byte range does not match its shape"
                    )));
                }
                tensors.insert(
                    name.clone(),
                    TensorInfo {
                        shape,
                        shard,
                        start: data_start + s,
                        end: data_start + e,
                    },
                );
            }
            files.push(file);
        }
        Ok(Self { files, tensors })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        self.info(name).map(|i| i.shape.as_slice())
    }

    fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))
    }

    fn raw(&self, name: &str) -> Result<&[u8]> {
        let info = self.info(name)?;
        Ok(&AsRef::<[u8]>::as_ref(&self.files[info.shard])[info.start..info.end])
    }

    /// Require an exact shape before any data is read.
    pub fn expect(&self, name: &str, shape: &[usize]) -> Result<()> {
        let got = self.shape(name)?;
        if got != shape {
            return Err(invalid(format!(
                "{name}: shape {got:?}, expected {shape:?}"
            )));
        }
        Ok(())
    }

    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        self.expect(name, shape)?;
        Ok(self
            .raw(name)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// F32 -> BF16 (round to nearest even), the dtype every non-quantised tensor runs in.
    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<bf16>> {
        self.expect(name, shape)?;
        Ok(self
            .raw(name)?
            .chunks_exact(4)
            .map(|c| bf16::from_f32(f32::from_le_bytes(c.try_into().unwrap())))
            .collect())
    }

    /// Rows `[row0, row0 + rows)` of a rank-2 tensor as F32, for tensors too large to hold whole.
    pub fn f32_rows(&self, name: &str, row0: usize, rows: usize) -> Result<Vec<f32>> {
        let shape = self.shape(name)?;
        let [total, cols] = shape else {
            return Err(invalid(format!("{name}: expected a matrix, got {shape:?}")));
        };
        if row0 + rows > *total {
            return Err(invalid(format!(
                "{name}: rows {row0}+{rows} exceed {total}"
            )));
        }
        let raw = self.raw(name)?;
        Ok(raw[row0 * cols * 4..(row0 + rows) * cols * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_shard(dir: &Path, name: &str, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (tname, shape, values) in tensors {
            let start = data.len();
            for v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(
                (*tname).into(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        out.extend_from_slice(&data);
        std::fs::write(dir.join(name), out).unwrap();
    }

    #[test]
    fn reads_tensors_across_shards_and_checks_shapes() {
        let dir = std::env::temp_dir().join(format!("lingbot_ckpt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_shard(
            &dir,
            "a.safetensors",
            &[("x", vec![2, 3], vec![1., 2., 3., 4., 5., 6.])],
        );
        write_shard(
            &dir,
            "b.safetensors",
            &[("y", vec![4], vec![0.5, -1.0, 2.0, 3.0])],
        );
        let ck = Checkpoint::open(&dir).unwrap();
        assert_eq!(ck.tensors.len(), 2);
        assert_eq!(ck.f32("x", &[2, 3]).unwrap(), vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(ck.f32_rows("x", 1, 1).unwrap(), vec![4., 5., 6.]);
        assert_eq!(ck.bf16("y", &[4]).unwrap()[1], bf16::from_f32(-1.0));
        assert!(ck.f32("x", &[3, 2]).is_err());
        assert!(ck.f32("missing", &[1]).is_err());
        std::fs::remove_dir_all(dir).ok();
    }
}
