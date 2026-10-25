use crate::contracts::{strict_json, AudioLimits, AudioTask};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

pub const AUDIO_COMMIT: &str = "f976eab8b47a3cada5aeb3b52a78451ee06117bc";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeIdentity {
    pub python: String,
    pub mlx: String,
    pub mlx_audio: String,
    pub mlx_audio_commit: String,
    pub packages: BTreeMap<String, String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: PathBuf,
    pub sha256: String,
    pub role: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub task: AudioTask,
    pub model: String,
    pub provider: String,
    pub family: String,
    pub precision: String,
    pub model_dir: PathBuf,
    pub runtime: RuntimeIdentity,
    pub artifacts: Vec<Artifact>,
    pub languages: Vec<String>,
    pub voices: BTreeMap<String, PathBuf>,
}
#[derive(Clone)]
pub struct Config {
    pub manifest: Manifest,
    pub manifest_path: PathBuf,
    pub revision: String,
    pub capability_revision: String,
    pub python: PathBuf,
    pub worker: PathBuf,
    pub pressure: bool,
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn file_digest(path: &Path, cancelled: &AtomicBool) -> Result<String, String> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|e| e.to_string())?;
    let before = file.metadata().map_err(|e| e.to_string())?;
    if !before.is_file() {
        return Err("artifact is not a regular file".into());
    }
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        if cancelled.load(Ordering::SeqCst) {
            return Err("startup cancelled".into());
        }
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let after = file.metadata().map_err(|e| e.to_string())?;
    if (
        before.ino(),
        before.len(),
        before.mtime(),
        before.mtime_nsec(),
    ) != (after.ino(), after.len(), after.mtime(), after.mtime_nsec())
    {
        return Err("artifact changed during identity check".into());
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn capabilities(manifest: &Manifest) -> serde_json::Value {
    serde_json::json!({"task": manifest.task, "languages": manifest.languages,
        "voices": manifest.voices.keys().collect::<Vec<_>>(), "limits": AudioLimits::default(),
        "cancellation": "provider_boundary_or_process_reap"})
}

impl Config {
    pub fn load(
        manifest_path: &Path,
        python: PathBuf,
        worker: PathBuf,
        fixture: bool,
        pressure: bool,
        cancelled: Arc<AtomicBool>,
    ) -> Result<Self, String> {
        if pressure && !crate::media::normal_pressure() {
            return Err("capacity_unavailable: host pressure is not normal".into());
        }
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(manifest_path)
            .map_err(|e| e.to_string())?;
        let mut raw = Vec::new();
        file.take(crate::contracts::MAX_FRAME as u64 + 1)
            .read_to_end(&mut raw)
            .map_err(|e| e.to_string())?;
        let manifest: Manifest =
            serde_json::from_value(strict_json(&raw)?).map_err(|e| e.to_string())?;
        if manifest.schema != "apxinf-audio-manifest/1"
            || manifest.model.trim().is_empty()
            || manifest.languages.is_empty()
            || manifest.languages.len() > 32
            || manifest.voices.len() > 32
        {
            return Err("invalid audio manifest".into());
        }
        if manifest.provider == "cpu_fixture" {
            if !fixture
                || pressure
                || manifest.family != "fixture"
                || !manifest.artifacts.is_empty()
            {
                return Err(
                    "CPU fixture requires explicit allowance and disabled pressure policy".into(),
                );
            }
        } else if manifest.provider != "mlx_audio"
            || fixture
            || !pressure
            || manifest.runtime.mlx != "0.32.4"
            || manifest.runtime.mlx_audio != "0.6.1"
            || manifest.runtime.mlx_audio_commit != AUDIO_COMMIT
            || manifest.precision != "frozen_artifact"
            || !manifest.model_dir.is_absolute()
            || manifest.artifacts.is_empty()
            || !matches!(
                (manifest.task, manifest.family.as_str()),
                (AudioTask::Asr, "whisper") | (AudioTask::Tts, "kokoro")
            )
        {
            return Err("unsupported or unpinned audio runtime profile".into());
        }
        if manifest.task == AudioTask::Tts && manifest.voices.is_empty() {
            return Err("TTS requires frozen voices".into());
        }
        if manifest.task == AudioTask::Asr && !manifest.voices.is_empty() {
            return Err("ASR cannot declare voices".into());
        }
        let identity_shape = crate::contracts::Frame::new(
            "11111111-1111-4111-8111-111111111111",
            crate::contracts::Message::Ready {
                task: manifest.task,
                model: manifest.model.clone(),
                model_revision: "0".repeat(64),
                capability_revision: "0".repeat(64),
                manifest_revision: "0".repeat(64),
                provider: manifest.provider.clone(),
                languages: manifest.languages.clone(),
                voices: manifest.voices.keys().cloned().collect(),
                cancellation: "provider_boundary_or_process_reap".into(),
                limits: AudioLimits::default(),
            },
        );
        if !identity_shape.validate() {
            return Err("invalid manifest capability fields".into());
        }
        if manifest.provider == "mlx_audio" {
            if !manifest.runtime.packages.contains_key("numpy")
                || manifest.runtime.packages.keys().any(|name| {
                    ["mlx", "mlx-audio"].contains(
                        &name
                            .to_ascii_lowercase()
                            .split(['-', '_', '.'])
                            .filter(|part| !part.is_empty())
                            .collect::<Vec<_>>()
                            .join("-")
                            .as_str(),
                    )
                })
            {
                return Err("invalid extra runtime package identity".into());
            }
            let mut seen = std::collections::HashSet::new();
            for artifact in &manifest.artifacts {
                if !artifact.path.is_absolute()
                    || !seen.insert(artifact.path.clone())
                    || !["model", "config", "preprocess", "voice", "frontend"]
                        .contains(&artifact.role.as_str())
                {
                    return Err("invalid artifact declaration".into());
                }
                let metadata =
                    fs::symlink_metadata(&artifact.path).map_err(|_| "missing_resource")?;
                if !metadata.is_file()
                    || artifact.path.canonicalize().map_err(|e| e.to_string())? != artifact.path
                {
                    return Err("artifact must be a canonical regular file".into());
                }
                if file_digest(&artifact.path, &cancelled)? != artifact.sha256 {
                    return Err("artifact digest mismatch".into());
                }
            }
            for voice in manifest.voices.values() {
                if !voice.is_absolute()
                    || voice.extension().and_then(|s| s.to_str()) != Some("safetensors")
                    || !manifest
                        .artifacts
                        .iter()
                        .any(|a| a.path == *voice && a.role == "voice")
                {
                    return Err("voice requires a frozen local safetensors artifact".into());
                }
            }
            for role in ["model", "config", "preprocess"] {
                if !manifest.artifacts.iter().any(|a| {
                    a.role == role
                        || (role == "preprocess"
                            && manifest.family == "kokoro"
                            && a.role == "frontend")
                }) {
                    return Err(format!("missing {role} identity"));
                }
            }
            if manifest.task == AudioTask::Tts
                && (manifest.languages != ["en"]
                    || !manifest.artifacts.iter().any(|a| a.role == "frontend")
                    || !manifest.runtime.packages.contains_key("misaki")
                    || !manifest.runtime.packages.contains_key("spacy")
                    || !manifest.runtime.packages.contains_key("en_core_web_sm"))
            {
                return Err("Kokoro requires a frozen English frontend profile".into());
            }
        }
        let capability_revision = sha256(&serde_json::to_vec(&capabilities(&manifest)).unwrap());
        Ok(Self {
            manifest,
            manifest_path: manifest_path.canonicalize().map_err(|e| e.to_string())?,
            revision: sha256(&raw),
            capability_revision,
            python,
            worker,
            pressure,
        })
    }
}
