//! The on-disk package of one MiniMax-H3 (video and audio) deployment: the Comfy-Org INT8 ConvRot DiT, the video and audio VAEs and
//! the text encoder, plus a tokenizer, in one directory named by a small manifest so `AutoModel` can recognise it without guessing from
//! file names.
//!
//! ```text
//! <dir>/apxinf_h3.json                                 {"schema": "apxinf.minimax-h3.v1", ...file names...}
//! <dir>/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
//! <dir>/vae/minimax_h3_video_vae_fp16.safetensors
//! <dir>/vae/minimax_h3_audio_vae_fp32.safetensors
//! <dir>/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
//! <dir>/tokenizer/{vocab.json, merges.txt, tokenizer_config.json}      (Qwen2 tokenizer of the Comfy-Org repo)
//! ```
#[cfg(feature = "cuda")]
use apxinf_core::{Error, Result};
#[cfg(feature = "cuda")]
use serde::Deserialize;
use std::path::Path;
#[cfg(feature = "cuda")]
use std::path::PathBuf;

pub const MANIFEST: &str = "apxinf_h3.json";
pub const SCHEMA: &str = "apxinf.minimax-h3.v1";

#[cfg(feature = "cuda")]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest {
    schema: String,
    transformer: String,
    video_vae: String,
    audio_vae: String,
    text_encoder: String,
    tokenizer: String,
}

/// Resolved, existing paths of the four parts.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Package {
    pub transformer: PathBuf,
    pub video_vae: PathBuf,
    pub audio_vae: PathBuf,
    pub text_encoder: PathBuf,
    pub tokenizer: PathBuf,
}

/// Whether `dir` carries this family's manifest (used by `AutoModel` detection; does not validate the contents).
pub(crate) fn is_package(dir: &Path) -> bool {
    dir.join(MANIFEST).is_file()
}

#[cfg(feature = "cuda")]
fn inside(root: &Path, relative: &str, what: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(Error::Other(format!(
            "{MANIFEST}: {what} `{relative}` must be a relative path inside the package"
        )));
    }
    Ok(root.join(relative_path))
}

#[cfg(feature = "cuda")]
impl Package {
    pub fn open(dir: &Path) -> Result<Self> {
        let manifest = dir.join(MANIFEST);
        let raw = std::fs::read_to_string(&manifest)
            .map_err(|e| Error::Other(format!("read {}: {e}", manifest.display())))?;
        let parsed: RawManifest = serde_json::from_str(&raw)
            .map_err(|e| Error::Other(format!("parse {}: {e}", manifest.display())))?;
        if parsed.schema != SCHEMA {
            return Err(Error::Other(format!(
                "{}: schema `{}` is not `{SCHEMA}`",
                manifest.display(),
                parsed.schema
            )));
        }
        let package = Self {
            transformer: inside(dir, &parsed.transformer, "transformer")?,
            video_vae: inside(dir, &parsed.video_vae, "video_vae")?,
            audio_vae: inside(dir, &parsed.audio_vae, "audio_vae")?,
            text_encoder: inside(dir, &parsed.text_encoder, "text_encoder")?,
            tokenizer: inside(dir, &parsed.tokenizer, "tokenizer")?,
        };
        for (what, path, directory) in [
            ("transformer", &package.transformer, false),
            ("video vae", &package.video_vae, false),
            ("audio vae", &package.audio_vae, false),
            ("text encoder", &package.text_encoder, false),
            ("tokenizer", &package.tokenizer, true),
        ] {
            let exists = if directory {
                path.is_dir()
            } else {
                path.is_file()
            };
            if !exists {
                return Err(Error::Other(format!(
                    "MiniMax-H3 package is missing its {what}: {}",
                    path.display()
                )));
            }
        }
        Ok(package)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("apxinf-h3-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn manifest(transformer: &str) -> String {
        format!(
            r#"{{"schema":"{SCHEMA}","transformer":"{transformer}","video_vae":"v.safetensors","audio_vae":"a.safetensors","text_encoder":"t.safetensors","tokenizer":"tok"}}"#
        )
    }

    #[test]
    fn a_complete_package_resolves_and_a_missing_part_is_named() {
        let dir = scratch("ok");
        std::fs::write(dir.join(MANIFEST), manifest("d.safetensors")).unwrap();
        assert!(is_package(&dir));
        let missing = Package::open(&dir).unwrap_err().to_string();
        assert!(
            missing.contains("transformer") && missing.contains("d.safetensors"),
            "{missing}"
        );
        for f in [
            "d.safetensors",
            "v.safetensors",
            "a.safetensors",
            "t.safetensors",
        ] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        std::fs::create_dir(dir.join("tok")).unwrap();
        let package = Package::open(&dir).unwrap();
        assert_eq!(package.video_vae, dir.join("v.safetensors"));
        assert_eq!(package.audio_vae, dir.join("a.safetensors"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn wrong_schema_unknown_fields_and_escaping_paths_are_rejected() {
        let dir = scratch("bad");
        assert!(!is_package(&dir));
        let write = |text: &str| std::fs::write(dir.join(MANIFEST), text).unwrap();
        write(&manifest("d.safetensors").replace(SCHEMA, "other.v9"));
        assert!(Package::open(&dir)
            .unwrap_err()
            .to_string()
            .contains("schema"));
        write(&manifest("d.safetensors").replace("\"tokenizer\"", "\"extra\":1,\"tokenizer\""));
        assert!(
            Package::open(&dir).is_err(),
            "unknown manifest fields are rejected"
        );
        write(&manifest("../outside.safetensors"));
        assert!(Package::open(&dir)
            .unwrap_err()
            .to_string()
            .contains("relative path"));
        write(&manifest("/etc/passwd"));
        assert!(Package::open(&dir)
            .unwrap_err()
            .to_string()
            .contains("relative path"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
