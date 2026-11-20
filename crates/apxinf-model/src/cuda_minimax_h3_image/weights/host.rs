//! Host checkpoint access: one safetensors file, header parsed once, tensors read on demand by offset.
//!
//! The shared loader copies every tensor of a file into memory. The INT8 DiT is 19.5 GiB and the text
//! encoder 25.3 GiB, so this family reads one tensor at a time and lets the caller upload and drop it.
//! No device state and no model knowledge lives here.
use apxinf_core::{DType, Error, Result};
use serde::Deserialize;
use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Header entry of one tensor. `dtype` is `None` for the byte-typed `U8` metadata tensors.
#[derive(Clone, Debug)]
pub(crate) struct Entry {
    pub dtype: Option<DType>,
    pub raw_dtype: String,
    pub shape: Vec<usize>,
    pub start: u64,
    pub len: u64,
}

pub(crate) struct Checkpoint {
    path: PathBuf,
    file: File,
    entries: HashMap<String, Entry>,
    metadata: HashMap<String, String>,
    read: BTreeSet<String>,
}

fn dtype_of(raw: &str) -> Option<DType> {
    match raw {
        "F32" => Some(DType::F32),
        "F16" => Some(DType::F16),
        "BF16" => Some(DType::BF16),
        "I8" => Some(DType::I8),
        "I32" => Some(DType::I32),
        _ => None,
    }
}

/// `MemAvailable` of a `/proc/meminfo` text, in bytes.
fn parse_mem_available(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kib.checked_mul(1024)
}

/// Bytes left under the container's memory limit, from a cgroup `limit` and `usage` text (`max` means unlimited).
fn parse_cgroup_headroom(limit: &str, usage: &str) -> Option<u64> {
    let limit: u64 = limit.trim().parse().ok()?;
    let usage: u64 = usage.trim().parse().ok()?;
    // cgroup v1 reports "no limit" as a number near the top of the range
    (limit < 1 << 60).then(|| limit.saturating_sub(usage))
}

/// Memory the process can still pin without swapping or being killed: the host's `MemAvailable`, and the container's
/// remaining limit when it has one. `None` when the platform reports neither.
pub(crate) fn available_host_memory() -> Option<u64> {
    let host = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| parse_mem_available(&text));
    let read = |path: &str| std::fs::read_to_string(path).ok();
    let container = [
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ]
    .into_iter()
    .find_map(|(limit, usage)| parse_cgroup_headroom(&read(limit)?, &read(usage)?));
    match (host, container) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

fn io_error(path: &Path, what: &str, e: std::io::Error) -> Error {
    Error::Other(format!("{what} {}: {e}", path.display()))
}

impl Checkpoint {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| io_error(path, "open", e))?;
        let total = file
            .metadata()
            .map_err(|e| io_error(path, "stat", e))?
            .len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|e| io_error(path, "read header length of", e))?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 || header_len > 64 << 20 || 8 + header_len > total {
            return Err(Error::Other(format!(
                "{}: invalid safetensors header length {header_len}",
                path.display()
            )));
        }
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)
            .map_err(|e| io_error(path, "read header of", e))?;
        let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(&header)
            .map_err(|e| Error::Other(format!("{}: header JSON: {e}", path.display())))?;
        let data_start = 8 + header_len;
        let mut metadata = HashMap::new();
        let mut entries = HashMap::with_capacity(raw.len());
        for (name, value) in raw {
            if name == "__metadata__" {
                if let serde_json::Value::Object(map) = value {
                    for (k, v) in map {
                        if let Some(s) = v.as_str() {
                            metadata.insert(k, s.to_owned());
                        }
                    }
                }
                continue;
            }
            let entry: RawEntry = serde_json::from_value(value)
                .map_err(|e| Error::Other(format!("{}: tensor {name}: {e}", path.display())))?;
            let [begin, end] = entry.data_offsets;
            let inside = data_start
                .checked_add(end)
                .is_some_and(|stop| stop <= total);
            if end < begin || !inside {
                return Err(Error::Other(format!(
                    "{}: tensor {name} offsets [{begin}, {end}] exceed the file",
                    path.display()
                )));
            }
            let dtype = dtype_of(&entry.dtype);
            let elements = entry
                .shape
                .iter()
                .try_fold(1u64, |a, d| a.checked_mul(*d as u64));
            if let Some(dtype) = dtype {
                let bytes = elements.and_then(|n| n.checked_mul(dtype.size_in_bytes() as u64));
                if bytes != Some(end - begin) {
                    return Err(Error::Other(format!(
                        "{}: tensor {name} byte length disagrees with {} {:?}",
                        path.display(),
                        entry.dtype,
                        entry.shape
                    )));
                }
            }
            entries.insert(
                name,
                Entry {
                    dtype,
                    raw_dtype: entry.dtype,
                    shape: entry.shape,
                    start: data_start + begin,
                    len: end - begin,
                },
            );
        }
        Ok(Self {
            path: path.to_owned(),
            file,
            entries,
            metadata,
            read: BTreeSet::new(),
        })
    }

    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Shape and dtype check without reading data. `dtype == None` accepts only byte tensors.
    pub fn expect(&self, name: &str, dtype: Option<DType>, shape: &[usize]) -> Result<&Entry> {
        let entry = self.entries.get(name).ok_or_else(|| {
            Error::Other(format!("{}: missing tensor {name}", self.path.display()))
        })?;
        if entry.dtype != dtype || entry.shape != shape {
            return Err(Error::Other(format!(
                "{}: tensor {name} is {} {:?}, expected {dtype:?} {shape:?}",
                self.path.display(),
                entry.raw_dtype,
                entry.shape
            )));
        }
        Ok(entry)
    }

    /// `(start, len)` in bytes of a tensor after checking dtype and shape, without reading it. For callers that map the
    /// file and move the bytes themselves.
    pub fn locate(
        &self,
        name: &str,
        dtype: Option<DType>,
        shape: &[usize],
    ) -> Result<(usize, usize)> {
        let entry = self.expect(name, dtype, shape)?;
        Ok((entry.start as usize, entry.len as usize))
    }

    /// Read the raw bytes of a tensor after checking dtype and shape.
    pub fn read(&mut self, name: &str, dtype: Option<DType>, shape: &[usize]) -> Result<Vec<u8>> {
        let (start, len) = {
            let entry = self.expect(name, dtype, shape)?;
            (entry.start, entry.len)
        };
        let mut bytes = vec![0u8; len as usize];
        self.file
            .seek(SeekFrom::Start(start))
            .and_then(|_| self.file.read_exact(&mut bytes))
            .map_err(|e| io_error(&self.path, &format!("read tensor {name} of"), e))?;
        self.read.insert(name.to_owned());
        Ok(bytes)
    }

    /// Read a byte-typed (`U8`) metadata tensor of any length.
    pub fn read_u8(&mut self, name: &str) -> Result<Vec<u8>> {
        let (start, len) = {
            let entry = self.entries.get(name).ok_or_else(|| {
                Error::Other(format!("{}: missing tensor {name}", self.path.display()))
            })?;
            if entry.raw_dtype != "U8" {
                return Err(Error::Other(format!(
                    "{}: tensor {name} is {}, expected U8",
                    self.path.display(),
                    entry.raw_dtype
                )));
            }
            (entry.start, entry.len)
        };
        let mut bytes = vec![0u8; len as usize];
        self.file
            .seek(SeekFrom::Start(start))
            .and_then(|_| self.file.read_exact(&mut bytes))
            .map_err(|e| io_error(&self.path, &format!("read tensor {name} of"), e))?;
        self.read.insert(name.to_owned());
        Ok(bytes)
    }

    /// Names of all tensors in the file, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.entries.keys().cloned().collect();
        names.sort();
        names
    }

    /// Mark every tensor under `prefix` as deliberately unused (for example the VAE encoder when only decoding).
    /// Returns how many tensors were skipped so a caller can assert the expected group size.
    pub fn ignore_prefix(&mut self, prefix: &str) -> usize {
        let names: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        let count = names.len();
        self.read.extend(names);
        count
    }

    /// Fail if the file carries a tensor the loader never consumed, so a changed layout is noticed.
    pub fn finish(&self) -> Result<()> {
        let mut unused: Vec<_> = self
            .entries
            .keys()
            .filter(|k| !self.read.contains(*k))
            .cloned()
            .collect();
        if unused.is_empty() {
            return Ok(());
        }
        unused.sort();
        unused.truncate(8);
        Err(Error::Other(format!(
            "{}: unused tensors (first 8): {unused:?}",
            self.path.display()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(dir: &Path, header: &str, data: &[u8]) -> PathBuf {
        let path = dir.join("t.safetensors");
        let mut f = File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(data).unwrap();
        path
    }

    fn temp_dir(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("apxinf-h3-host-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reads_by_offset_checks_shape_and_reports_unused_tensors() {
        let dir = temp_dir("ok");
        let header = r#"{"__metadata__":{"config":"{}"},"a":{"dtype":"I8","shape":[2,3],"data_offsets":[0,6]},"b":{"dtype":"F32","shape":[2],"data_offsets":[6,14]},"q":{"dtype":"U8","shape":[2],"data_offsets":[14,16]}}"#;
        let mut data = vec![1u8, 2, 3, 4, 5, 6];
        data.extend(1.5f32.to_le_bytes());
        data.extend(2.5f32.to_le_bytes());
        data.extend([7u8, 8]);
        let path = write_file(&dir, header, &data);
        let mut file = Checkpoint::open(&path).unwrap();
        assert_eq!(file.metadata("config"), Some("{}"));
        assert_eq!(
            file.read("a", Some(DType::I8), &[2, 3]).unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert!(
            file.read("a", Some(DType::I8), &[3, 2]).is_err(),
            "shape is part of the contract"
        );
        assert!(
            file.read("a", Some(DType::BF16), &[2, 3]).is_err(),
            "dtype is part of the contract"
        );
        assert!(file.read("missing", Some(DType::I8), &[1]).is_err());
        let err = file.finish().unwrap_err().to_string();
        assert!(err.contains("\"b\"") && err.contains("\"q\""), "{err}");
        assert_eq!(file.read("b", Some(DType::F32), &[2]).unwrap().len(), 8);
        assert_eq!(file.read_u8("q").unwrap(), vec![7, 8]);
        assert!(
            file.read_u8("a").is_err(),
            "only U8 tensors are byte metadata"
        );
        file.finish().unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn locates_tensors_without_reading_and_parses_meminfo() {
        let dir = temp_dir("locate");
        let header = r#"{"a":{"dtype":"I8","shape":[2,3],"data_offsets":[0,6]},"q":{"dtype":"U8","shape":[2],"data_offsets":[6,8]}}"#;
        let path = write_file(&dir, header, &[0u8; 8]);
        let file = Checkpoint::open(&path).unwrap();
        let data_start = 8 + header.len();
        assert_eq!(
            file.locate("a", Some(DType::I8), &[2, 3]).unwrap(),
            (data_start, 6)
        );
        assert!(
            file.locate("a", Some(DType::I8), &[3, 2]).is_err(),
            "shape is part of the contract"
        );
        assert_eq!(
            parse_mem_available("MemTotal: 10 kB\nMemAvailable:   2048 kB\n"),
            Some(2 << 20)
        );
        assert_eq!(parse_mem_available("MemTotal: 10 kB\n"), None);
        assert_eq!(parse_cgroup_headroom("1000\n", "400\n"), Some(600));
        assert_eq!(parse_cgroup_headroom("max\n", "400\n"), None, "no limit");
        assert_eq!(
            parse_cgroup_headroom("9223372036854771712\n", "400\n"),
            None,
            "v1 spelling of no limit"
        );
        assert_eq!(parse_cgroup_headroom("100\n", "400\n"), Some(0));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_corrupt_headers_and_inconsistent_lengths() {
        let dir = temp_dir("bad");
        let overrun = r#"{"a":{"dtype":"I8","shape":[4],"data_offsets":[0,40]}}"#;
        assert!(Checkpoint::open(&write_file(&dir, overrun, &[0; 4])).is_err());
        let mismatch = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#;
        assert!(Checkpoint::open(&write_file(&dir, mismatch, &[0; 8])).is_err());
        assert!(Checkpoint::open(&write_file(&dir, "not json", &[])).is_err());
        // offsets and shapes taken from the header must not wrap around the checks that bound them by the file
        let wrapping =
            r#"{"a":{"dtype":"I8","shape":[4],"data_offsets":[0,18446744073709551615]}}"#;
        assert!(Checkpoint::open(&write_file(&dir, wrapping, &[0; 4])).is_err());
        let huge_shape =
            r#"{"a":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,0]}}"#;
        assert!(Checkpoint::open(&write_file(&dir, huge_shape, &[])).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
