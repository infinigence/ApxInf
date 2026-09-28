//! GR00T-local resource discovery and calibration content identities.
//!
//! The bundled layout needs Cosmos configuration and processor files, not a
//! second tensor checkpoint. Explicit legacy assets retain their old identity.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use apxinf_core::{Error, Result};
use serde::Deserialize;

const DEFAULT_SUBDIR: &str = "assets/cosmos";
const MANIFEST: &str = "apxinf_assets.json";
const SCHEMA: &str = "apxinf.gr00t-assets.v1";
const RESOURCE_NAMES: &[&str] = &[
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "generation_config.json",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AssetManifest {
    schema: String,
    #[serde(deserialize_with = "unique_files")]
    files: BTreeMap<String, String>,
}

fn unique_files<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, String>, D::Error> {
    struct Files;
    impl<'de> serde::de::Visitor<'de> for Files {
        type Value = BTreeMap<String, String>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("unique asset paths mapped to SHA256 strings")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> std::result::Result<Self::Value, M::Error> {
            let mut files = BTreeMap::new();
            while let Some((name, hash)) = map.next_entry::<String, String>()? {
                if files.insert(name.clone(), hash).is_some() {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate GR00T asset {name:?}"
                    )));
                }
            }
            Ok(files)
        }
    }
    deserializer.deserialize_map(Files)
}

/// Match the family policy's local default while preserving explicit snapshots.
pub(super) fn resolve_assets(model_dir: &Path, explicit: Option<&Path>) -> Result<PathBuf> {
    let root = explicit
        .map(Path::to_path_buf)
        .unwrap_or_else(|| model_dir.join(DEFAULT_SUBDIR));
    if root.join(MANIFEST).exists() || root.join(MANIFEST).is_symlink() {
        asset_identity(&root)?;
    } else if explicit.is_none() {
        return Err(Error::Other(format!(
            "GR00T local assets require {}. Prepare them with Gr00tPolicy.prepare_assets(MODEL, COSMOS), or supply the legacy backbone asset",
            root.join(MANIFEST).display()
        )));
    }
    Ok(root)
}

fn validate_resource_name(name: &str) -> Result<()> {
    let parts: Vec<_> = name.split('/').collect();
    if name.contains(['\\', '\0'])
        || parts.iter().any(|p| matches!(*p, "" | "." | ".."))
        || !(RESOURCE_NAMES.contains(&name)
            || (parts.len() == 2 && parts[0] == "chat_templates" && parts[1].ends_with(".jinja")))
    {
        return Err(Error::Other(format!(
            "unsupported GR00T asset path {name:?}"
        )));
    }
    Ok(())
}

fn resource_names(root: &Path) -> Result<BTreeSet<String>> {
    let mut names: BTreeSet<String> = RESOURCE_NAMES
        .iter()
        .filter(|name| root.join(name).exists() || root.join(name).is_symlink())
        .map(|name| (*name).to_owned())
        .collect();
    let templates = root.join("chat_templates");
    if templates.exists() || templates.is_symlink() {
        if templates.is_symlink() || !templates.is_dir() {
            return Err(Error::Other(
                "GR00T chat_templates must be a regular directory".into(),
            ));
        }
        for entry in std::fs::read_dir(&templates).map_err(|e| Error::Other(e.to_string()))? {
            let entry = entry.map_err(|e| Error::Other(e.to_string()))?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| Error::Other("GR00T asset path is not UTF-8".into()))?;
            if name.ends_with(".jinja") {
                names.insert(format!("chat_templates/{name}"));
            }
        }
    }
    Ok(names)
}

fn file_digest(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
    let mut digest = LocalSha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|e| Error::Other(e.to_string()))?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(digest.finish_hex())
}

/// Validate the files used by the official processor, then hash their content.
fn asset_identity(root: &Path) -> Result<String> {
    let path = root.join(MANIFEST);
    if path.is_symlink() {
        return Err(Error::Other(
            "GR00T asset manifest must not be a symlink".into(),
        ));
    }
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
    let manifest: AssetManifest = serde_json::from_str(&raw)
        .map_err(|e| Error::Other(format!("GR00T asset manifest: {e}")))?;
    if manifest.schema != SCHEMA {
        return Err(Error::Other(format!(
            "unsupported GR00T asset schema {}",
            manifest.schema
        )));
    }
    for required in [
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ] {
        if !manifest.files.contains_key(required) {
            return Err(Error::Other(format!(
                "missing required GR00T asset {required}"
            )));
        }
    }
    if !manifest.files.contains_key("tokenizer.json")
        && !(manifest.files.contains_key("vocab.json") && manifest.files.contains_key("merges.txt"))
    {
        return Err(Error::Other(
            "GR00T assets require tokenizer.json or vocab.json and merges.txt".into(),
        ));
    }
    for name in resource_names(root)? {
        if !manifest.files.contains_key(&name) {
            return Err(Error::Other(format!("unlisted GR00T asset {name}")));
        }
    }
    let mut digest = LocalSha256::new();
    digest.update(SCHEMA.as_bytes());
    digest.update(&[0]);
    for (name, expected) in &manifest.files {
        validate_resource_name(name)?;
        if expected.len() != 64
            || !expected
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(Error::Other(format!(
                "invalid lowercase SHA256 for GR00T asset {name}"
            )));
        }
        let file = root.join(name);
        if file.is_symlink() || !file.is_file() {
            return Err(Error::Other(format!(
                "GR00T asset must be a regular file: {name}"
            )));
        }
        if &file_digest(&file)? != expected {
            return Err(Error::Other(format!("GR00T asset SHA256 mismatch: {name}")));
        }
        digest.update(name.as_bytes());
        digest.update(&[0]);
        digest.update(expected.as_bytes());
        digest.update(&[0]);
    }
    Ok(format!("sha256:{}", digest.finish_hex()))
}

/// Content identity shared with the model-neutral Python calibration runner.
///
/// Bundled assets use a versioned resource identity. Legacy explicit snapshots
/// retain the original two-weight-root identity, without relabeling profiles.
pub(super) fn checkpoint_identity(checkpoint: &Path, backbone: &Path) -> Result<String> {
    let primary = single_checkpoint_identity(checkpoint)?;
    let (name, backbone) =
        if backbone.join(MANIFEST).exists() || backbone.join(MANIFEST).is_symlink() {
            ("cosmos-assets-v1", asset_identity(backbone)?)
        } else {
            ("backbone", single_checkpoint_identity(backbone)?)
        };
    let mut digest = LocalSha256::new();
    for (name, identity) in [("primary", primary), (name, backbone)] {
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

pub(super) struct LocalSha256 {
    state: [u32; 8],
    buffer: Vec<u8>,
    bytes: u64,
}

impl LocalSha256 {
    pub(super) fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: Vec::with_capacity(64),
            bytes: 0,
        }
    }

    pub(super) fn update(&mut self, mut input: &[u8]) {
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

    pub(super) fn finish_hex(mut self) -> String {
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
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "apxinf-gr00t-assets-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            let bundle = root.join(DEFAULT_SUBDIR);
            std::fs::create_dir_all(&bundle).unwrap();
            std::fs::write(root.join("model.safetensors"), b"primary-v1").unwrap();
            let mut files = BTreeMap::new();
            for name in [
                "config.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
                "tokenizer.json",
            ] {
                std::fs::write(bundle.join(name), b"{}\n").unwrap();
                files.insert(name, file_digest(&bundle.join(name)).unwrap());
            }
            std::fs::write(
                bundle.join(MANIFEST),
                serde_json::to_vec(&serde_json::json!({
                    "schema": SCHEMA, "files": files
                }))
                .unwrap(),
            )
            .unwrap();
            Self(root)
        }
        fn bundle(&self) -> PathBuf {
            self.0.join(DEFAULT_SUBDIR)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn resource_and_checkpoint_identities_match_independent_python_vectors() {
        let fixture = Fixture::new();
        assert_eq!(
            asset_identity(&fixture.bundle()).unwrap(),
            "sha256:d3d2b2cf917e47140c6242930536c92f9e8ca8b4e2871d773814d3b63ea26e75"
        );
        assert_eq!(
            checkpoint_identity(&fixture.0, &fixture.bundle()).unwrap(),
            "sha256:652d1192021049f00f97e3450a39f340271cbd5d97f3d37c10d91feffb0004d6"
        );
        assert_eq!(resolve_assets(&fixture.0, None).unwrap(), fixture.bundle());
        let renamed = fixture.0.join("relocated-cosmos");
        std::fs::rename(fixture.bundle(), &renamed).unwrap();
        assert_eq!(
            asset_identity(&renamed).unwrap(),
            "sha256:d3d2b2cf917e47140c6242930536c92f9e8ca8b4e2871d773814d3b63ea26e75"
        );
        assert_eq!(
            checkpoint_identity(&fixture.0, &renamed).unwrap(),
            "sha256:652d1192021049f00f97e3450a39f340271cbd5d97f3d37c10d91feffb0004d6"
        );
    }

    #[test]
    fn legacy_calibration_identity_stays_unchanged() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/checkpoint_identity");
        assert_eq!(
            checkpoint_identity(&fixture, &fixture).unwrap(),
            "sha256:d23faece91e5ba14630dd918ed491b712b32c905f153a2bbb5065f355b5df094"
        );
    }

    #[test]
    fn absent_default_manifest_is_not_treated_as_a_legacy_snapshot() {
        let fixture = Fixture::new();
        std::fs::remove_file(fixture.bundle().join(MANIFEST)).unwrap();
        assert!(resolve_assets(&fixture.0, None).is_err());
        assert_eq!(
            resolve_assets(&fixture.0, Some(&fixture.bundle())).unwrap(),
            fixture.bundle()
        );
    }

    #[test]
    fn changed_and_missing_resources_are_rejected_before_loading() {
        let fixture = Fixture::new();
        let file = fixture.bundle().join("tokenizer.json");
        std::fs::write(&file, b"different vocabulary").unwrap();
        assert!(asset_identity(&fixture.bundle())
            .unwrap_err()
            .to_string()
            .contains("SHA256 mismatch"));
        std::fs::remove_file(file).unwrap();
        assert!(asset_identity(&fixture.bundle()).is_err());
    }

    #[test]
    fn unlisted_processor_resources_are_rejected() {
        let fixture = Fixture::new();
        std::fs::write(fixture.bundle().join("chat_template.json"), b"{}").unwrap();
        assert!(asset_identity(&fixture.bundle())
            .unwrap_err()
            .to_string()
            .contains("unlisted"));
    }

    #[test]
    fn manifest_version_unknown_fields_duplicates_and_unsafe_paths_are_rejected() {
        let fixture = Fixture::new();
        let path = fixture.bundle().join(MANIFEST);
        let original = std::fs::read_to_string(&path).unwrap();
        for changed in [
            original.replace(SCHEMA, "apxinf.gr00t-assets.v999"),
            original.replacen('{', "{\"unknown\":true,", 1),
            original.replacen(
                "\"files\":{",
                "\"files\":{\"config.json\":\"duplicate\",",
                1,
            ),
            original.replace("tokenizer.json", "../tokenizer.json"),
        ] {
            std::fs::write(&path, changed).unwrap();
            assert!(asset_identity(&fixture.bundle()).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn resource_symlinks_cannot_escape_the_validated_bundle() {
        let fixture = Fixture::new();
        let file = fixture.bundle().join("tokenizer.json");
        std::fs::remove_file(&file).unwrap();
        std::os::unix::fs::symlink(fixture.bundle().join("config.json"), file).unwrap();
        assert!(asset_identity(&fixture.bundle()).is_err());
    }
}
