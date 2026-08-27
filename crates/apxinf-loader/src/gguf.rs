//! GGUF format loader (v2 and v3).
//!
//! Binary format:
//!   [4 bytes] magic "GGUF"
//!   [4 bytes LE u32] version (2 or 3)
//!   [8 bytes LE u64] tensor_count
//!   [8 bytes LE u64] metadata_kv_count
//!   [metadata_kv_count × kv entries]
//!   [tensor_count × tensor_info entries]
//!   [alignment padding]
//!   [tensor data]
//!
//! We only load F32 and BF16 tensors; all others are skipped with a warning.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use bytemuck;
use byteorder::{ReadBytesExt, LE};
use memmap2::Mmap;

use apxinf_core::{DType, Device, Shape, Tensor};

use crate::config::ModelConfig;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" LE

// ── GGUF value types ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        if let GgufValue::Uint32(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        if let GgufValue::Uint64(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        if let GgufValue::Float32(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        if let GgufValue::Float64(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let GgufValue::String(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            GgufValue::Uint32(v) => Some(*v as usize),
            GgufValue::Uint64(v) => Some(*v as usize),
            GgufValue::Int32(v) => Some(*v as usize),
            _ => None,
        }
    }
}

// ── GGUF tensor types we care about ────────────────────────────────

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_BF16: u32 = 30;

/// Loaded GGUF file contents.
#[derive(Debug)]
pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, Tensor>,
    pub skipped: Vec<String>, // tensor names with unsupported types
}

/// Load a GGUF file. Only F32 and BF16 tensors are loaded.
pub fn load(path: &Path) -> Result<GgufFile, String> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap failed: {e}"))? };
    let mut cursor = std::io::Cursor::new(&mmap[..]);

    // ── Header ──────────────────────────────────────────────────────
    let magic = cursor
        .read_u32::<LE>()
        .map_err(|e| format!("read magic: {e}"))?;
    if magic != GGUF_MAGIC {
        return Err(format!("not a GGUF file (magic = 0x{magic:08x})"));
    }

    let version = cursor
        .read_u32::<LE>()
        .map_err(|e| format!("read version: {e}"))?;
    if version != 2 && version != 3 {
        return Err(format!(
            "unsupported GGUF version {version} (expected 2 or 3)"
        ));
    }

    let tensor_count = cursor
        .read_u64::<LE>()
        .map_err(|e| format!("read tensor_count: {e}"))? as usize;
    let kv_count = cursor
        .read_u64::<LE>()
        .map_err(|e| format!("read kv_count: {e}"))? as usize;

    // ── Metadata key-value pairs ────────────────────────────────────
    let mut metadata = HashMap::new();
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut cursor)?;
        let value_type = cursor
            .read_u32::<LE>()
            .map_err(|e| format!("read value type: {e}"))?;
        let value = read_gguf_value(&mut cursor, value_type)?;
        metadata.insert(key, value);
    }

    // ── Tensor info entries ─────────────────────────────────────────
    struct TensorInfo {
        name: String,
        shape: Vec<usize>,
        ggml_type: u32,
        offset: u64,
    }

    let mut tensor_infos = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut cursor)?;
        let n_dims = cursor
            .read_u32::<LE>()
            .map_err(|e| format!("read n_dims: {e}"))? as usize;
        // GGUF stores dimensions in reverse order (innermost first)
        let mut shape_rev = vec![0u64; n_dims];
        for d in &mut shape_rev {
            *d = cursor
                .read_u64::<LE>()
                .map_err(|e| format!("read dim: {e}"))?;
        }
        // Reverse to get standard row-major [outer, ..., inner]
        let shape: Vec<usize> = shape_rev.iter().rev().map(|&d| d as usize).collect();
        let ggml_type = cursor
            .read_u32::<LE>()
            .map_err(|e| format!("read ggml_type: {e}"))?;
        let offset = cursor
            .read_u64::<LE>()
            .map_err(|e| format!("read offset: {e}"))?;
        tensor_infos.push(TensorInfo {
            name,
            shape,
            ggml_type,
            offset,
        });
    }

    // ── Data section ─────────────────────────────────────────────────
    // Data starts at the next alignment boundary after the header.
    let alignment: usize = metadata
        .get("general.alignment")
        .and_then(|v| v.as_usize())
        .unwrap_or(32);

    let header_end = cursor.position() as usize;
    let data_start = align_to(header_end, alignment);

    // ── Build tensors ────────────────────────────────────────────────
    let mut tensors = HashMap::new();
    let mut skipped = Vec::new();

    for info in tensor_infos {
        let dtype = match info.ggml_type {
            GGML_TYPE_F32 => DType::F32,
            GGML_TYPE_BF16 => DType::BF16,
            GGML_TYPE_F16 => {
                // F16 → convert to F32 on load (simple path for now)
                let numel: usize = info.shape.iter().product();
                let byte_start = data_start + info.offset as usize;
                let byte_end = byte_start + numel * 2;
                if byte_end > mmap.len() {
                    return Err(format!("tensor '{}' f16 data out of bounds", info.name));
                }
                // Convert f16 bytes to f32
                let src: &[u16] = bytemuck::cast_slice(&mmap[byte_start..byte_end]);
                let f32_data: Vec<f32> = src
                    .iter()
                    .map(|&bits| half::f16::from_bits(bits).to_f32())
                    .collect();
                let bytes: Vec<u8> = bytemuck::cast_slice(&f32_data).to_vec();
                let shape = Shape::from(info.shape);
                let tensor = Tensor::from_raw(shape, DType::F32, Device::Cpu, bytes)
                    .map_err(|e| format!("tensor '{}': {e}", info.name))?;
                tensors.insert(info.name, tensor);
                continue;
            }
            other => {
                skipped.push(format!("{} (ggml_type={})", info.name, other));
                continue;
            }
        };

        let numel: usize = info.shape.iter().product();
        let elem_size = dtype.size_in_bytes();
        let byte_start = data_start + info.offset as usize;
        let byte_end = byte_start + numel * elem_size;

        if byte_end > mmap.len() {
            return Err(format!(
                "tensor '{}': data [{byte_start}, {byte_end}] out of bounds (file={})",
                info.name,
                mmap.len()
            ));
        }

        let bytes = mmap[byte_start..byte_end].to_vec();
        let shape = Shape::from(info.shape);
        let tensor = Tensor::from_raw(shape, dtype, Device::Cpu, bytes)
            .map_err(|e| format!("tensor '{}': {e}", info.name))?;
        tensors.insert(info.name, tensor);
    }

    Ok(GgufFile {
        metadata,
        tensors,
        skipped,
    })
}

/// Extract a ModelConfig from GGUF metadata.
pub fn config_from_metadata(metadata: &HashMap<String, GgufValue>) -> ModelConfig {
    let get_usize = |key: &str, default: usize| -> usize {
        metadata
            .get(key)
            .and_then(|v| v.as_usize())
            .unwrap_or(default)
    };
    let get_f32 = |key: &str, default: f32| -> f32 {
        metadata
            .get(key)
            .and_then(|v| match v {
                GgufValue::Float32(x) => Some(*x),
                GgufValue::Float64(x) => Some(*x as f32),
                _ => None,
            })
            .unwrap_or(default)
    };

    // Standard GGUF metadata keys for Llama models
    ModelConfig {
        hidden_size: get_usize("llama.embedding_length", 2048),
        intermediate_size: get_usize("llama.feed_forward_length", 5632),
        n_layers: get_usize("llama.block_count", 22),
        n_heads: get_usize("llama.attention.head_count", 32),
        n_kv_heads: get_usize("llama.attention.head_count_kv", 4),
        vocab_size: get_usize("llama.vocab_size", 32000),
        max_seq_len: get_usize("llama.context_length", 2048),
        rope_theta: get_f32("llama.rope.freq_base", 10000.0),
        rms_norm_eps: get_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
    }
}

// ── Binary reading helpers ───────────────────────────────────────────

fn read_gguf_string(cursor: &mut std::io::Cursor<&[u8]>) -> Result<String, String> {
    let len = cursor
        .read_u64::<LE>()
        .map_err(|e| format!("read string len: {e}"))? as usize;
    let mut buf = vec![0u8; len];
    std::io::Read::read_exact(cursor, &mut buf).map_err(|e| format!("read string bytes: {e}"))?;
    String::from_utf8(buf).map_err(|e| format!("invalid UTF-8 in string: {e}"))
}

fn read_gguf_value(
    cursor: &mut std::io::Cursor<&[u8]>,
    value_type: u32,
) -> Result<GgufValue, String> {
    match value_type {
        0 => Ok(GgufValue::Uint8(
            cursor.read_u8().map_err(|e| format!("u8: {e}"))?,
        )),
        1 => Ok(GgufValue::Int8(
            cursor.read_i8().map_err(|e| format!("i8: {e}"))?,
        )),
        2 => Ok(GgufValue::Uint16(
            cursor.read_u16::<LE>().map_err(|e| format!("u16: {e}"))?,
        )),
        3 => Ok(GgufValue::Int16(
            cursor.read_i16::<LE>().map_err(|e| format!("i16: {e}"))?,
        )),
        4 => Ok(GgufValue::Uint32(
            cursor.read_u32::<LE>().map_err(|e| format!("u32: {e}"))?,
        )),
        5 => Ok(GgufValue::Int32(
            cursor.read_i32::<LE>().map_err(|e| format!("i32: {e}"))?,
        )),
        6 => Ok(GgufValue::Float32(
            cursor.read_f32::<LE>().map_err(|e| format!("f32: {e}"))?,
        )),
        7 => {
            let b = cursor.read_u8().map_err(|e| format!("bool: {e}"))?;
            Ok(GgufValue::Bool(b != 0))
        }
        8 => Ok(GgufValue::String(read_gguf_string(cursor)?)),
        9 => {
            let elem_type = cursor
                .read_u32::<LE>()
                .map_err(|e| format!("array elem type: {e}"))?;
            let count = cursor
                .read_u64::<LE>()
                .map_err(|e| format!("array count: {e}"))? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_gguf_value(cursor, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::Uint64(
            cursor.read_u64::<LE>().map_err(|e| format!("u64: {e}"))?,
        )),
        11 => Ok(GgufValue::Int64(
            cursor.read_i64::<LE>().map_err(|e| format!("i64: {e}"))?,
        )),
        12 => Ok(GgufValue::Float64(
            cursor.read_f64::<LE>().map_err(|e| format!("f64: {e}"))?,
        )),
        other => Err(format!("unknown GGUF value type {other}")),
    }
}

fn align_to(offset: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        return offset;
    }
    (offset + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Build a minimal GGUF v3 file in memory.
    struct GgufBuilder {
        data: Vec<u8>,
        tensor_info: Vec<u8>,
        tensor_data: Vec<u8>,
        tensor_count: u64,
        kv_count: u64,
    }

    impl GgufBuilder {
        fn new() -> Self {
            let mut data = Vec::new();
            // magic
            data.write_u32::<LE>(GGUF_MAGIC).unwrap();
            // version 3
            data.write_u32::<LE>(3).unwrap();
            // tensor_count placeholder
            data.extend_from_slice(&0u64.to_le_bytes());
            // kv_count placeholder
            data.extend_from_slice(&0u64.to_le_bytes());
            Self {
                data,
                tensor_info: Vec::new(),
                tensor_data: Vec::new(),
                tensor_count: 0,
                kv_count: 0,
            }
        }

        fn add_metadata_u32(&mut self, key: &str, val: u32) {
            let mut tmp = std::mem::take(&mut self.data);
            self.write_string_to(&mut tmp, key);
            tmp.write_u32::<LE>(4).unwrap(); // value_type = u32
            tmp.write_u32::<LE>(val).unwrap();
            self.data = tmp;
            self.kv_count += 1;
        }

        fn write_string_to(&self, buf: &mut Vec<u8>, s: &str) {
            buf.write_u64::<LE>(s.len() as u64).unwrap();
            buf.extend_from_slice(s.as_bytes());
        }

        fn add_metadata_str(&mut self, key: &str, val: &str) {
            let mut tmp = std::mem::take(&mut self.data);
            self.write_string_to(&mut tmp, key);
            tmp.write_u32::<LE>(8).unwrap(); // value_type = string
            self.write_string_to(&mut tmp, val);
            self.data = tmp;
            self.kv_count += 1;
        }

        fn add_tensor_f32(&mut self, name: &str, shape: &[u64], values: &[f32]) {
            // Tensor info
            let mut info = Vec::new();
            self.write_string_to(&mut info, name);
            info.write_u32::<LE>(shape.len() as u32).unwrap();
            // GGUF stores dims reversed
            for &d in shape.iter().rev() {
                info.write_u64::<LE>(d).unwrap();
            }
            info.write_u32::<LE>(GGML_TYPE_F32).unwrap();
            info.write_u64::<LE>(self.tensor_data.len() as u64).unwrap(); // offset
            self.tensor_info.extend_from_slice(&info);

            // Tensor data
            for v in values {
                self.tensor_data.write_f32::<LE>(*v).unwrap();
            }
            self.tensor_count += 1;
        }

        fn build(mut self) -> Vec<u8> {
            // Patch counts
            let tc = self.tensor_count.to_le_bytes();
            let kc = self.kv_count.to_le_bytes();
            self.data[8..16].copy_from_slice(&tc);
            self.data[16..24].copy_from_slice(&kc);

            // Assemble: header + tensor_info + alignment padding + tensor_data
            let mut out = self.data;
            out.extend_from_slice(&self.tensor_info);
            // Align to 32 bytes
            let pad = align_to(out.len(), 32) - out.len();
            out.extend(vec![0u8; pad]);
            out.extend_from_slice(&self.tensor_data);
            out
        }
    }

    #[test]
    fn test_load_f32_tensor() {
        let mut b = GgufBuilder::new();
        b.add_tensor_f32("weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let bytes = b.build();

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&bytes).unwrap();

        let gguf = load(tmp.path()).unwrap();
        assert!(gguf.tensors.contains_key("weight"));
        let t = &gguf.tensors["weight"];
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        let got = t.as_f32().unwrap();
        assert_eq!(got, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_load_metadata() {
        let mut b = GgufBuilder::new();
        b.add_metadata_u32("llama.block_count", 22);
        b.add_metadata_u32("llama.embedding_length", 2048);
        b.add_tensor_f32("x", &[2], &[1.0, 2.0]);
        let bytes = b.build();

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&bytes).unwrap();

        let gguf = load(tmp.path()).unwrap();
        assert_eq!(gguf.metadata["llama.block_count"].as_usize(), Some(22));
        assert_eq!(
            gguf.metadata["llama.embedding_length"].as_usize(),
            Some(2048)
        );
    }

    #[test]
    fn test_config_from_metadata() {
        let mut b = GgufBuilder::new();
        b.add_metadata_u32("llama.block_count", 32);
        b.add_metadata_u32("llama.attention.head_count", 32);
        b.add_metadata_u32("llama.attention.head_count_kv", 8);
        b.add_metadata_u32("llama.embedding_length", 4096);
        b.add_metadata_u32("llama.feed_forward_length", 11008);
        b.add_metadata_u32("llama.vocab_size", 32000);
        b.add_metadata_u32("llama.context_length", 4096);
        let bytes = b.build();

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&bytes).unwrap();

        let gguf = load(tmp.path()).unwrap();
        let config = config_from_metadata(&gguf.metadata);
        assert_eq!(config.n_layers, 32);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.n_kv_heads, 8);
        assert_eq!(config.head_dim(), 128); // 4096 / 32
    }

    #[test]
    fn test_multiple_tensors() {
        let mut b = GgufBuilder::new();
        b.add_tensor_f32("a", &[3], &[1.0, 2.0, 3.0]);
        b.add_tensor_f32("b", &[2, 2], &[4.0, 5.0, 6.0, 7.0]);
        let bytes = b.build();

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&bytes).unwrap();

        let gguf = load(tmp.path()).unwrap();
        assert_eq!(gguf.tensors.len(), 2);
        assert_eq!(gguf.tensors["a"].numel(), 3);
        assert_eq!(gguf.tensors["b"].numel(), 4);
    }

    #[test]
    fn test_bad_magic() {
        let bytes = vec![0xAA, 0xBB, 0xCC, 0xDD, 0, 0, 0, 0];
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&bytes).unwrap();
        let result = load(tmp.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a GGUF file"));
    }
}
