//! Opt-in persistent cache of the DiT text K/V for repeated prompts.
//!
//! The denoise loop reads the prompt only through the per-block text K/V that
//! `Dit::text_cache` computes once per request. A hit loads the bytes of an
//! earlier computation and skips both the text encoder and `text_cache`. The
//! key identifies the inputs of that computation, so a hit produces the same
//! image as a miss: the prompt tokens, the code and MLX library that run it,
//! the OS and GPU, and fingerprints of both shard sets (see [`Identity`]).
//!
//! Each entry is one SafeTensors file named by its key. Its modification time
//! is its last use: a hit sets it, and eviction deletes the oldest entries
//! first. The directory listing is the only index, and each write publishes a
//! uniquely named temporary file by rename, so processes can share a
//! directory without a lock. Cache problems never fail a request; they print
//! one line and count as a miss.

use super::dit::TextCache;
use super::text::PromptTokens;
use super::weights::{BLOCKS, HEADS, HEAD_DIM};
use crate::PromptCacheOptions;
use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, MlxDType, Stream};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Bump when the entry layout changes.
const FORMAT: u32 = 1;
const ENTRY_SUFFIX: &str = ".safetensors";
const TEMP_SUFFIX: &str = ".tmp";
/// A temporary file this old belongs to a writer that died; live writes take seconds.
const ABANDONED_WRITE: Duration = Duration::from_secs(3600);
/// Checkpoint shards are identified by their headers plus this many evenly
/// spaced payload windows, so same-shape weights with different values differ.
const SAMPLES: u64 = 8;
const SAMPLE_BYTES: u64 = 64 << 10;
/// Entries hold prompt-derived activations: readable by the owner only.
const ENTRY_MODE: u32 = 0o600;

/// Everything besides the prompt tokens that determines the cached bytes.
#[derive(Clone, serde::Serialize)]
struct Identity {
    format: u32,
    apxinf_version: &'static str,
    /// Size and modification time of the executable or library that contains
    /// this code, which change with any rebuild of the code that computes the K/V.
    code: String,
    mlx_version: &'static str,
    /// Size and modification time of the loaded MLX library and its Metal
    /// kernel library, which a rebuild of the same MLX version changes.
    mlx_library: String,
    /// Digest of the OS release record: Metal compiles MLX kernels at run time.
    os: String,
    gpu_architecture: String,
    mpp_image_projections: bool,
    text_encoder: String,
    transformer: String,
}

pub struct PromptCache {
    dir: PathBuf,
    max_bytes: u64,
    /// Serialized [`Identity`], hashed into every key.
    identity: Vec<u8>,
}

impl Identity {
    fn new(checkpoint: &Path, stream: &Stream, mpp_image_projections: bool) -> Result<Self> {
        let mlx = apxinf_mlx::library_path()
            .ok_or_else(|| Error::Other("the MLX library path is unknown".into()))?;
        let metallib = mlx.with_file_name("mlx.metallib");
        Ok(Self {
            format: FORMAT,
            apxinf_version: env!("CARGO_PKG_VERSION"),
            code: file_stamp(&code_image()?)?,
            mlx_version: apxinf_mlx::version(),
            mlx_library: format!(
                "{}; {}",
                file_stamp(&mlx)?,
                file_stamp(&metallib).unwrap_or_default()
            ),
            os: hex(&Sha256::digest(std::fs::read(
                "/System/Library/CoreServices/SystemVersion.plist",
            )?)),
            gpu_architecture: stream
                .metal_caps()
                .map_or_else(String::new, |caps| caps.architecture.clone()),
            mpp_image_projections,
            text_encoder: shard_fingerprint(&checkpoint.join("text_encoder"))?,
            transformer: shard_fingerprint(&checkpoint.join("transformer"))?,
        })
    }
}

/// The executable or dynamic library this code was loaded from.
fn code_image() -> Result<PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    let found = unsafe { libc::dladdr(code_image as *const libc::c_void, &mut info) };
    if found == 0 || info.dli_fname.is_null() {
        return Err(Error::Other("the loaded code image is unknown".into()));
    }
    let name = unsafe { std::ffi::CStr::from_ptr(info.dli_fname) };
    Ok(PathBuf::from(std::ffi::OsStr::from_bytes(name.to_bytes())))
}

impl PromptCache {
    pub fn open(
        options: &PromptCacheOptions,
        checkpoint: &Path,
        stream: &Stream,
        mpp_image_projections: bool,
    ) -> Result<Self> {
        let identity = Identity::new(checkpoint, stream, mpp_image_projections)?;
        std::fs::create_dir_all(&options.dir)?;
        Ok(Self {
            dir: options.dir.clone(),
            max_bytes: options.max_bytes.get(),
            identity: serialize(&identity)?,
        })
    }

    /// The cached text K/V for `tokens`, or `None` on a miss.
    pub fn load(&self, tokens: &PromptTokens, stream: &Stream) -> Option<TextCache> {
        let path = self.entry_path(tokens);
        let before = std::fs::metadata(&path).ok()?;
        match read_entry(&path, text_len(tokens), stream) {
            Ok(kv) => {
                // Record the use; a failure only makes this entry older for eviction.
                let _ = File::options()
                    .write(true)
                    .open(&path)
                    .and_then(|file| file.set_modified(SystemTime::now()));
                Some(TextCache::from_kv(kv))
            }
            Err(error) => {
                // Another process may have evicted or replaced the entry while
                // it was read; only delete the file that failed to read.
                let unchanged = std::fs::metadata(&path).is_ok_and(|after| {
                    after.ino() == before.ino() && after.modified().ok() == before.modified().ok()
                });
                if unchanged {
                    eprintln!(
                        "Qwen-Image prompt cache: discarding {}: {error}",
                        path.display()
                    );
                    let _ = std::fs::remove_file(&path);
                }
                None
            }
        }
    }

    /// Stores `cache` for `tokens`, evicting least recently used entries
    /// first so that the stored entries, this one included, fit the bound.
    pub fn store(&self, tokens: &PromptTokens, cache: &TextCache) {
        if let Err(error) = self.try_store(tokens, cache) {
            eprintln!("Qwen-Image prompt cache: entry not stored: {error}");
        }
    }

    fn try_store(&self, tokens: &PromptTokens, cache: &TextCache) -> Result<()> {
        let header = entry_header(text_len(tokens))?;
        let bytes = 8 + header.len() as u64 + 2 * (BLOCKS * tensor_bytes(text_len(tokens))) as u64;
        if bytes > self.max_bytes {
            return Ok(());
        }
        self.evict(self.max_bytes - bytes)?;
        let path = self.entry_path(tokens);
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        static WRITES: AtomicU64 = AtomicU64::new(0);
        let temp = self.dir.join(format!(
            ".{name}.{}-{}{TEMP_SUFFIX}",
            std::process::id(),
            WRITES.fetch_add(1, Ordering::Relaxed)
        ));
        let written = (|| {
            let mut file = File::options()
                .write(true)
                .create_new(true)
                .mode(ENTRY_MODE)
                .open(&temp)?;
            file.write_all(&(header.len() as u64).to_le_bytes())?;
            file.write_all(&header)?;
            for (k, v) in cache.kv() {
                file.write_all(&k.to_bytes()?)?;
                file.write_all(&v.to_bytes()?)?;
            }
            file.sync_all()?;
            Ok::<_, Error>(file.metadata()?.len())
        })();
        match written {
            Ok(len) if len == bytes => std::fs::rename(&temp, &path)?,
            outcome => {
                let _ = std::fs::remove_file(&temp);
                outcome?;
                return Err(Error::Other(format!("entry is not {bytes} bytes")));
            }
        }
        // Another process may have stored entries meanwhile.
        self.evict(self.max_bytes)
    }

    /// Deletes the least recently used entries until the stored entries take
    /// at most `budget` bytes, and temporary files of writers that died.
    /// Other files in the directory are never touched.
    fn evict(&self, budget: u64) -> Result<()> {
        let mut entries = Vec::new();
        for item in std::fs::read_dir(&self.dir)? {
            let item = item?;
            let (Some(name), Ok(metadata)) = (
                item.file_name().to_str().map(str::to_owned),
                item.metadata(),
            ) else {
                continue;
            };
            let modified = metadata.modified()?;
            if is_entry_name(&name) {
                entries.push((modified, name, metadata.len()));
            } else if is_temp_name(&name)
                && modified.elapsed().is_ok_and(|age| age > ABANDONED_WRITE)
            {
                let _ = std::fs::remove_file(item.path());
            }
        }
        entries.sort();
        let mut total: u64 = entries.iter().map(|(_, _, len)| len).sum();
        for (_, name, len) in entries {
            if total <= budget {
                break;
            }
            match std::fs::remove_file(self.dir.join(&name)) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            total -= len;
        }
        Ok(())
    }

    fn entry_path(&self, tokens: &PromptTokens) -> PathBuf {
        self.dir
            .join(format!("{}{ENTRY_SUFFIX}", key(&self.identity, tokens)))
    }
}

fn key(identity: &[u8], tokens: &PromptTokens) -> String {
    let mut hasher = Sha256::new();
    hasher.update((identity.len() as u64).to_le_bytes());
    hasher.update(identity);
    hasher.update((tokens.drop as u64).to_le_bytes());
    for id in &tokens.ids {
        hasher.update(id.to_le_bytes());
    }
    hex(&hasher.finalize())
}

fn serialize(identity: &Identity) -> Result<Vec<u8>> {
    serde_json::to_vec(identity).map_err(|error| Error::Other(error.to_string()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn is_entry_name(name: &str) -> bool {
    name.strip_suffix(ENTRY_SUFFIX)
        .is_some_and(|key| key.len() == 64 && key.bytes().all(|b| b.is_ascii_hexdigit()))
}

fn is_temp_name(name: &str) -> bool {
    name.strip_prefix('.')
        .and_then(|rest| rest.strip_suffix(TEMP_SUFFIX))
        .and_then(|rest| rest.rsplit_once('.'))
        .is_some_and(|(entry, writer)| {
            is_entry_name(entry) && writer.bytes().all(|b| b.is_ascii_digit() || b == b'-')
        })
}

fn text_len(tokens: &PromptTokens) -> usize {
    tokens.ids.len() - tokens.drop
}

fn tensor_bytes(text_len: usize) -> usize {
    text_len * HEADS * HEAD_DIM * 2
}

fn tensor_names() -> impl Iterator<Item = (String, String)> {
    (0..BLOCKS).map(|block| (format!("k.{block}"), format!("v.{block}")))
}

/// SafeTensors header for one entry: BF16 `[1, HEADS, text_len, HEAD_DIM]`
/// K and V per block, stored in block order, padded to 8 bytes.
fn entry_header(text_len: usize) -> Result<Vec<u8>> {
    let size = tensor_bytes(text_len);
    let mut tensors = serde_json::Map::new();
    for (index, name) in tensor_names().flat_map(|(k, v)| [k, v]).enumerate() {
        tensors.insert(
            name,
            serde_json::json!({
                "dtype": "BF16",
                "shape": [1, HEADS, text_len, HEAD_DIM],
                "data_offsets": [index * size, (index + 1) * size],
            }),
        );
    }
    let mut header =
        serde_json::to_vec(&tensors).map_err(|error| Error::Other(error.to_string()))?;
    header.resize(header.len().next_multiple_of(8), b' ');
    Ok(header)
}

fn read_entry(path: &Path, text_len: usize, stream: &Stream) -> Result<Vec<(Array, Array)>> {
    let mut tensors = apxinf_mlx::load_safetensors(stream, &[path], |_| true)?;
    let shape = [1, HEADS, text_len, HEAD_DIM];
    let mut take = |name: &str| {
        tensors
            .remove(name)
            .filter(|array| array.dtype() == MlxDType::BF16 && array.shape() == shape)
            .ok_or_else(|| Error::Other(format!("`{name}` is missing or is not BF16 {shape:?}")))
    };
    let kv = tensor_names()
        .map(|(k, v)| Ok((take(&k)?, take(&v)?)))
        .collect::<Result<Vec<_>>>()?;
    if !tensors.is_empty() {
        return Err(Error::Other("unexpected tensors".into()));
    }
    Ok(kv)
}

fn file_stamp(path: &Path) -> Result<String> {
    let metadata = std::fs::metadata(path)?;
    let modified = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    Ok(format!(
        "{} bytes, modified {}ns",
        metadata.len(),
        modified.as_nanos()
    ))
}

/// Fingerprint of a SafeTensors shard set: per shard, its file name, size,
/// modification time, header and sampled payload windows. It does not read
/// every byte: a same-size edit between the windows that keeps the
/// modification time is not detected. Paths are left out, so the same files
/// reached through another directory or link share entries.
fn shard_fingerprint(dir: &Path) -> Result<String> {
    let mut files = apxinf_loader::safetensors::safetensors_files(dir).map_err(Error::Other)?;
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    let mut hasher = Sha256::new();
    for path in files {
        let name = path
            .file_name()
            .map(|name| name.as_encoded_bytes())
            .unwrap_or_default();
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name);
        hasher.update(file_stamp(&path)?.as_bytes());
        let mut file = File::open(&path)?;
        let size = file.metadata()?.len();
        let mut header_len = [0; 8];
        file.read_exact(&mut header_len)?;
        let header_len = u64::from_le_bytes(header_len);
        let data = 8u64
            .checked_add(header_len)
            .filter(|&start| start <= size)
            .ok_or_else(|| Error::Other(format!("{}: header exceeds the file", path.display())))?;
        let mut header = vec![0; header_len as usize];
        file.read_exact(&mut header)?;
        hasher.update(&header);
        let span = size - data;
        let mut window = vec![0; SAMPLE_BYTES.min(span) as usize];
        for sample in 0..SAMPLES {
            let offset = data + (span - window.len() as u64) * sample / (SAMPLES - 1);
            file.read_exact_at(&mut window, offset)?;
            hasher.update(&window);
        }
    }
    Ok(hex(&hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Identity {
        Identity {
            format: FORMAT,
            apxinf_version: "0.1.0",
            code: "1 bytes, modified 1ns".into(),
            mlx_version: "0.32.3",
            mlx_library: "2 bytes, modified 1ns; 3 bytes, modified 1ns".into(),
            os: "os".into(),
            gpu_architecture: "applegpu_g16g".into(),
            mpp_image_projections: true,
            text_encoder: "text".into(),
            transformer: "dit".into(),
        }
    }

    fn tokens(ids: &[u32], drop: usize) -> PromptTokens {
        PromptTokens {
            ids: ids.to_vec(),
            drop,
        }
    }

    fn cache(label: &str, max_bytes: u64) -> PromptCache {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "apxinf-prompt-cache-{label}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        PromptCache {
            dir,
            max_bytes,
            identity: serialize(&identity()).unwrap(),
        }
    }

    fn text_cache(stream: &Stream, tokens: &PromptTokens, seed: u8) -> TextCache {
        let size = tensor_bytes(text_len(tokens));
        let tensor = |salt: usize| {
            let bytes: Vec<u8> = (0..size)
                .map(|i| (i * 31 + salt * 7 + seed as usize) as u8)
                .collect();
            Array::from_bytes(
                stream,
                &[1, HEADS, text_len(tokens), HEAD_DIM],
                MlxDType::BF16,
                &bytes,
            )
            .unwrap()
        };
        TextCache::from_kv(
            (0..BLOCKS)
                .map(|b| (tensor(2 * b), tensor(2 * b + 1)))
                .collect(),
        )
    }

    fn bytes_of(cache: &TextCache) -> Vec<Vec<u8>> {
        cache
            .kv()
            .iter()
            .flat_map(|(k, v)| [k.to_bytes().unwrap(), v.to_bytes().unwrap()])
            .collect()
    }

    fn entry_bytes(tokens: &PromptTokens) -> u64 {
        8 + entry_header(text_len(tokens)).unwrap().len() as u64
            + 2 * (BLOCKS * tensor_bytes(text_len(tokens))) as u64
    }

    #[test]
    fn every_identity_field_and_the_tokens_change_the_key() {
        let base = identity();
        let mutants: Vec<(&str, Identity)> = vec![
            (
                "format",
                Identity {
                    format: FORMAT + 1,
                    ..base.clone()
                },
            ),
            (
                "apxinf_version",
                Identity {
                    apxinf_version: "0.1.1",
                    ..base.clone()
                },
            ),
            (
                "code",
                Identity {
                    code: "1 bytes, modified 2ns".into(),
                    ..base.clone()
                },
            ),
            (
                "mlx_library",
                Identity {
                    mlx_library: "2 bytes, modified 2ns".into(),
                    ..base.clone()
                },
            ),
            (
                "mlx_version",
                Identity {
                    mlx_version: "0.32.4",
                    ..base.clone()
                },
            ),
            (
                "os",
                Identity {
                    os: "other".into(),
                    ..base.clone()
                },
            ),
            (
                "gpu_architecture",
                Identity {
                    gpu_architecture: "applegpu_g13s".into(),
                    ..base.clone()
                },
            ),
            (
                "mpp_image_projections",
                Identity {
                    mpp_image_projections: false,
                    ..base.clone()
                },
            ),
            (
                "text_encoder",
                Identity {
                    text_encoder: "other".into(),
                    ..base.clone()
                },
            ),
            (
                "transformer",
                Identity {
                    transformer: "other".into(),
                    ..base.clone()
                },
            ),
        ];
        let fields = serde_json::to_value(&base).unwrap();
        assert_eq!(
            fields.as_object().unwrap().len(),
            mutants.len(),
            "every identity field needs a case here"
        );
        let prompt = tokens(&[5, 6, 7, 8], 2);
        let want = key(&serialize(&base).unwrap(), &prompt);
        for (field, mutant) in &mutants {
            assert_ne!(
                key(&serialize(mutant).unwrap(), &prompt),
                want,
                "{field} is not in the key"
            );
        }
        let identity = serialize(&base).unwrap();
        assert_eq!(key(&identity, &tokens(&[5, 6, 7, 8], 2)), want);
        for other in [
            tokens(&[5, 6, 7, 9], 2),
            tokens(&[5, 6, 7], 2),
            tokens(&[5, 6, 7, 8], 1),
        ] {
            assert_ne!(key(&identity, &other), want);
        }
    }

    fn write_shard(dir: &Path, value: u8) {
        std::fs::create_dir_all(dir).unwrap();
        let header = br#"{"w":{"dtype":"U8","shape":[1048576],"data_offsets":[0,1048576]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend(std::iter::repeat_n(value, 1 << 20));
        let path = dir.join("model.safetensors");
        std::fs::write(&path, bytes).unwrap();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(UNIX_EPOCH + Duration::from_secs(1_000_000))
            .unwrap();
    }

    #[test]
    fn identity_reads_each_component_from_its_own_source() {
        let root = std::env::temp_dir().join(format!("apxinf-identity-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_shard(&root.join("text_encoder"), 1);
        write_shard(&root.join("transformer"), 2);
        let stream = Stream::cpu().unwrap();
        let identity = Identity::new(&root, &stream, true).unwrap();
        assert_eq!(
            identity.text_encoder,
            shard_fingerprint(&root.join("text_encoder")).unwrap()
        );
        assert_eq!(
            identity.transformer,
            shard_fingerprint(&root.join("transformer")).unwrap()
        );
        assert_ne!(identity.text_encoder, identity.transformer);
        assert_eq!(identity.mlx_version, apxinf_mlx::version());
        assert!(identity.mpp_image_projections);
        assert!(
            !Identity::new(&root, &stream, false)
                .unwrap()
                .mpp_image_projections
        );
        let code = code_image().unwrap();
        assert_eq!(identity.code, file_stamp(&code).unwrap());
        assert_eq!(
            code,
            std::env::current_exe().unwrap(),
            "unit tests link this code statically"
        );
        let mlx = apxinf_mlx::library_path().unwrap();
        assert_eq!(
            identity.mlx_library,
            format!(
                "{}; {}",
                file_stamp(&mlx).unwrap(),
                file_stamp(&mlx.with_file_name("mlx.metallib")).unwrap()
            )
        );
        let os = hex(&Sha256::digest(
            std::fs::read("/System/Library/CoreServices/SystemVersion.plist").unwrap(),
        ));
        assert_eq!(identity.os, os);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn entries_are_readable_by_the_owner_only() {
        let stream = Stream::cpu().unwrap();
        let cache = cache("mode", 1 << 30);
        let prompt = tokens(&[1, 2, 3], 1);
        cache.store(&prompt, &text_cache(&stream, &prompt, 0));
        let mode = std::fs::metadata(cache.entry_path(&prompt)).unwrap().mode();
        assert_eq!(mode & 0o777, ENTRY_MODE);
    }

    #[test]
    fn a_hit_returns_the_stored_bytes() {
        let stream = Stream::cpu().unwrap();
        let cache = cache("roundtrip", 1 << 30);
        let prompt = tokens(&[1, 2, 3, 4, 5], 2);
        assert!(cache.load(&prompt, &stream).is_none());
        let stored = text_cache(&stream, &prompt, 9);
        cache.store(&prompt, &stored);
        let hit = cache.load(&prompt, &stream).expect("hit after store");
        assert_eq!(bytes_of(&hit), bytes_of(&stored));
        assert!(cache.load(&tokens(&[1, 2, 3, 4, 6], 2), &stream).is_none());
    }

    #[test]
    fn eviction_removes_the_least_recently_used_entry() {
        let stream = Stream::cpu().unwrap();
        let prompts: Vec<_> = (0..3).map(|i| tokens(&[9, i, i + 1], 1)).collect();
        let cache = cache("lru", 2 * entry_bytes(&prompts[0]));
        cache.store(&prompts[0], &text_cache(&stream, &prompts[0], 0));
        std::thread::sleep(Duration::from_millis(20));
        cache.store(&prompts[1], &text_cache(&stream, &prompts[1], 1));
        std::thread::sleep(Duration::from_millis(20));
        // Using the oldest entry makes the second one the least recently used.
        assert!(cache.load(&prompts[0], &stream).is_some());
        std::thread::sleep(Duration::from_millis(20));
        cache.store(&prompts[2], &text_cache(&stream, &prompts[2], 2));
        assert!(
            cache.load(&prompts[1], &stream).is_none(),
            "least recently used survived"
        );
        assert!(
            cache.load(&prompts[0], &stream).is_some(),
            "recently used entry was evicted"
        );
        assert!(
            cache.load(&prompts[2], &stream).is_some(),
            "new entry was evicted"
        );
    }

    #[test]
    fn the_bound_holds_and_oversized_entries_are_not_stored() {
        let stream = Stream::cpu().unwrap();
        let short = tokens(&[1, 2, 3], 1);
        let long = tokens(&[1, 2, 3, 4, 5, 6, 7, 8], 1);
        let cache = cache("bound", entry_bytes(&long) - 1);
        cache.store(&short, &text_cache(&stream, &short, 0));
        cache.store(&long, &text_cache(&stream, &long, 0));
        assert!(cache.load(&long, &stream).is_none());
        assert!(
            cache.load(&short, &stream).is_some(),
            "an entry that is not stored evicted others"
        );
        let foreign = cache.dir.join("notes.txt");
        std::fs::write(&foreign, b"not an entry").unwrap();
        for i in 10..14 {
            let prompt = tokens(&[1, 2, i], 1);
            cache.store(&prompt, &text_cache(&stream, &prompt, 0));
        }
        let stored: u64 = std::fs::read_dir(&cache.dir)
            .unwrap()
            .map(|item| item.unwrap())
            .filter(|item| is_entry_name(item.file_name().to_str().unwrap()))
            .map(|item| item.metadata().unwrap().len())
            .sum();
        assert!(
            stored <= cache.max_bytes,
            "{stored} bytes stored over {}",
            cache.max_bytes
        );
        assert!(
            foreign.exists(),
            "eviction deleted a file the cache does not own"
        );
    }

    #[test]
    fn a_damaged_entry_is_a_miss_and_is_removed() {
        let stream = Stream::cpu().unwrap();
        let cache = cache("damaged", 1 << 30);
        let prompt = tokens(&[1, 2, 3, 4], 1);
        cache.store(&prompt, &text_cache(&stream, &prompt, 3));
        let path = cache.entry_path(&prompt);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 1);
        std::fs::write(&path, bytes).unwrap();
        assert!(cache.load(&prompt, &stream).is_none());
        assert!(!path.exists());
    }

    #[test]
    fn abandoned_temporary_files_are_removed() {
        let stream = Stream::cpu().unwrap();
        let cache = cache("abandoned", 1 << 30);
        let name = format!(".{}{ENTRY_SUFFIX}.4242-0{TEMP_SUFFIX}", "a".repeat(64));
        assert!(is_temp_name(&name));
        let (old, fresh) = (
            cache.dir.join(&name),
            cache.dir.join(name.replace("4242-0", "4242-1")),
        );
        for path in [&old, &fresh] {
            std::fs::write(path, b"partial").unwrap();
        }
        File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .set_modified(SystemTime::now() - ABANDONED_WRITE * 2)
            .unwrap();
        let prompt = tokens(&[1, 2, 3], 1);
        cache.store(&prompt, &text_cache(&stream, &prompt, 0));
        assert!(!old.exists(), "abandoned write kept");
        assert!(fresh.exists(), "a write in progress was deleted");
    }

    #[test]
    fn shard_fingerprint_follows_payload_values_not_the_path() {
        let root =
            std::env::temp_dir().join(format!("apxinf-shard-fingerprint-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let write = |name: &str, value: u8| {
            write_shard(&root.join(name), value);
            root.join(name)
        };
        let (a, moved, changed) = (write("a", 1), write("moved", 1), write("changed", 2));
        assert_eq!(
            shard_fingerprint(&a).unwrap(),
            shard_fingerprint(&moved).unwrap()
        );
        assert_ne!(
            shard_fingerprint(&a).unwrap(),
            shard_fingerprint(&changed).unwrap()
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
