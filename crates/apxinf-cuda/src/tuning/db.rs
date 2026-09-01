use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use apxinf_core::{Error, Result};

use crate::context::CudaLibraryVersions;
use crate::device_caps::CudaDeviceCaps;

use super::key::{
    DeviceFingerprint, Epilogue, GemmLayout, GemmOp, GemmTuningKey, ScaleMode, TuningDType,
};
use super::store::{GemmTuningRecord, TacticStore};
use super::tactic::{decode_cublaslt_custom_tactic, TacticBackend, TacticId};

pub const TUNING_SCHEMA_V1: &str = "apxinf.cuda.tuning.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuningDbHeader {
    pub schema: String,
    pub kernel_build_id: Option<String>,
    pub device_name: Option<String>,
    pub sm: Option<u32>,
    pub cuda_version: Option<String>,
    pub cublas_version: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TuningDb {
    pub header: TuningDbHeader,
    records: Vec<ParsedGemmRecord>,
}

#[derive(Clone, Debug, PartialEq)]
struct ParsedGemmRecord {
    op: GemmOp,
    device: Option<DeviceFingerprint>,
    m: usize,
    n: usize,
    k: usize,
    activation_dtype: TuningDType,
    weight_dtype: TuningDType,
    output_dtype: TuningDType,
    layout: GemmLayout,
    scale_mode: ScaleMode,
    epilogue: Epilogue,
    workspace_limit: usize,
    tactic: TacticId,
    implementation_version: Option<u32>,
    milliseconds: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CompatibilityRejections {
    total: usize,
    implementation_version: usize,
    library_version: usize,
}

impl TuningDb {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        Self::from_json_str(&raw)
    }

    /// Parse both the versioned v1 database and the preserved pre-v1 PI0.5
    /// JSON format. Legacy model/profile fields are metadata only and never
    /// enter a physical tuning key.
    pub fn from_json_str(raw: &str) -> Result<Self> {
        let root: serde_json::Value = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("CUDA tuning JSON: {error}")))?;
        let object = root
            .as_object()
            .ok_or_else(|| Error::Other("CUDA tuning database must be a JSON object".into()))?;
        let declared_schema = object.get("schema").is_some();
        let schema = object
            .get("schema")
            .and_then(|value| value.as_str())
            .unwrap_or("apxinf.cuda.tuning.legacy-pi05")
            .to_string();
        if schema != TUNING_SCHEMA_V1 && schema != "apxinf.cuda.tuning.legacy-pi05" {
            return Err(Error::Other(format!(
                "unsupported CUDA tuning schema `{schema}`"
            )));
        }
        let device_name = object
            .get("device_name")
            .or_else(|| object.get("device"))
            .and_then(|value| value.as_str())
            .map(str::to_string);
        let sm = object
            .get("sm")
            .and_then(|value| value.as_u64())
            .and_then(|value| u32::try_from(value).ok())
            .or_else(|| device_name.as_deref().and_then(parse_sm));
        let header = TuningDbHeader {
            schema,
            kernel_build_id: object
                .get("kernel_build_id")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            device_name,
            sm,
            cuda_version: object
                .get("cuda_version")
                .or_else(|| object.get("cuda"))
                .and_then(|value| value.as_str())
                .map(str::to_string),
            cublas_version: object
                .get("cublas_version")
                .or_else(|| object.get("cublas"))
                .and_then(|value| value.as_str())
                .map(str::to_string),
        };
        if declared_schema && header.schema == TUNING_SCHEMA_V1 {
            validate_v1_header(&header)?;
        }
        let mut records = match object.get("records") {
            Some(value) => parse_v1_records(value)?,
            None => parse_legacy_tactics(object.get("tactics").ok_or_else(|| {
                Error::Other("CUDA tuning database has neither records nor tactics".into())
            })?)?,
        };
        records.sort_by_key(|record| (record.op as u8, record.m, record.n, record.k));
        Ok(Self { header, records })
    }

    pub fn build_store(
        &self,
        caps: &CudaDeviceCaps,
        versions: &CudaLibraryVersions,
    ) -> Result<TacticStore> {
        TacticStore::from_gemm_records(self.build_records(caps, versions)?)
    }

    pub fn header_for_cuda(
        caps: &CudaDeviceCaps,
        versions: &CudaLibraryVersions,
    ) -> TuningDbHeader {
        TuningDbHeader {
            schema: TUNING_SCHEMA_V1.into(),
            // Kept as diagnostic provenance only. Compatibility is checked
            // per provider through `implementation_version` below.
            kernel_build_id: Some(super::KERNEL_BUILD_ID.into()),
            device_name: Some(caps.device_name.clone()),
            sm: Some(caps.sm),
            cuda_version: Some(major_minor(&versions.cuda)),
            cublas_version: Some(major_minor(&versions.cublas)),
        }
    }

    /// Merge one exact winner with the latest on-disk database while holding
    /// an inter-process lock, then atomically replace the JSON file.
    pub fn merge_record_atomic(
        path: &Path,
        header: &TuningDbHeader,
        caps: &CudaDeviceCaps,
        versions: &CudaLibraryVersions,
        record: GemmTuningRecord,
    ) -> Result<TacticStore> {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|error| {
            Error::Other(format!(
                "create tuning directory {}: {error}",
                parent.display()
            ))
        })?;
        let _lock = DatabaseLock::acquire(path)?;
        let mut store = if path.is_file() {
            Self::from_json_file(path)?.build_store(caps, versions)?
        } else {
            TacticStore::default()
        };
        store.upsert_gemm(record);
        write_store_atomic(path, header, &store)?;
        Ok(store)
    }

    /// Materialize device-specific physical records. Callers loading several
    /// databases can merge their resulting stores before the one-time install.
    pub fn build_records(
        &self,
        caps: &CudaDeviceCaps,
        versions: &CudaLibraryVersions,
    ) -> Result<Vec<GemmTuningRecord>> {
        if let Some(sm) = self.header.sm {
            if sm != caps.sm {
                return Err(Error::Other(format!(
                    "tuning database targets SM{sm}, current device is SM{}",
                    caps.sm
                )));
            }
        }
        let (records, rejected) = self.build_compatible_records(caps, versions);
        if rejected.total != 0 {
            eprintln!(
                "[apxinf] warning: ignored {} incompatible CUDA tuning record(s) for {} (implementation version mismatch: {}, CUDA/cuBLAS version mismatch: {})",
                rejected.total,
                self.header.device_name.as_deref().unwrap_or("unknown device"),
                rejected.implementation_version,
                rejected.library_version,
            );
        }
        Ok(records)
    }

    fn build_compatible_records(
        &self,
        caps: &CudaDeviceCaps,
        versions: &CudaLibraryVersions,
    ) -> (Vec<GemmTuningRecord>, CompatibilityRejections) {
        // Device name and the global build id are provenance. They must not
        // invalidate unrelated providers. SM and the schema are the
        // database-wide compatibility boundary; library and kernel contract
        // changes are rejected record-by-record below.
        let cuda_compatible =
            versions_compatible(self.header.cuda_version.as_deref(), versions.cuda.as_str());
        let cublas_compatible = versions_compatible(
            self.header.cublas_version.as_deref(),
            versions.cublas.as_str(),
        );
        let device = DeviceFingerprint::from(caps);
        let mut rejected = CompatibilityRejections::default();
        let records = self
            .records
            .iter()
            .filter_map(|record| {
                let implementation_compatible =
                    record.implementation_version.map_or(true, |version| {
                        version == record.tactic.backend.implementation_version()
                    });
                let library_compatible = match record.tactic.backend {
                    TacticBackend::Cutlass
                    | TacticBackend::CutlassFp8DualGeGlu
                    | TacticBackend::CutlassBf16DualGeGluM522
                    | TacticBackend::CutlassBf16DualGeGluM533 => cuda_compatible,
                    TacticBackend::CublasLt
                    | TacticBackend::CublasLtCustom
                    | TacticBackend::CublasLtCustomBias
                    | TacticBackend::CublasLtCustomSplitSerial
                    | TacticBackend::CublasLtCustomSplitGeGluCutlass
                    | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto
                    | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3
                    | TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm
                    | TacticBackend::CublasLtCustomSplitGeGluCutlassBf16
                    | TacticBackend::Vendor => cuda_compatible && cublas_compatible,
                };
                if !implementation_compatible || !library_compatible {
                    rejected.total += 1;
                    rejected.implementation_version += usize::from(!implementation_compatible);
                    rejected.library_version += usize::from(!library_compatible);
                    return None;
                }
                Some(GemmTuningRecord {
                    key: GemmTuningKey {
                        op: record.op,
                        device: record.device.unwrap_or(device),
                        m: record.m,
                        n: record.n,
                        k: record.k,
                        activation_dtype: record.activation_dtype,
                        weight_dtype: record.weight_dtype,
                        output_dtype: record.output_dtype,
                        layout: record.layout,
                        scale_mode: record.scale_mode,
                        epilogue: record.epilogue,
                        workspace_limit: record.workspace_limit,
                    },
                    tactic: record.tactic,
                    implementation_version: record.implementation_version,
                    milliseconds: record.milliseconds,
                })
            })
            .collect();
        (records, rejected)
    }
}

fn major_minor(version: &str) -> String {
    version.split('.').take(2).collect::<Vec<_>>().join(".")
}

fn versions_compatible(expected: Option<&str>, actual: &str) -> bool {
    expected.map_or(true, |expected| {
        major_minor(expected) == major_minor(actual)
    })
}

fn write_store_atomic(path: &Path, header: &TuningDbHeader, store: &TacticStore) -> Result<()> {
    let mut records = store.gemm_records().collect::<Vec<_>>();
    records.sort_by(|left, right| {
        (
            left.key.op as u8,
            left.key.device.sm,
            left.key.device.multiprocessor_count,
            left.key.m,
            left.key.n,
            left.key.k,
            left.key.activation_dtype as u8,
            left.key.weight_dtype as u8,
            left.key.output_dtype as u8,
            left.key.layout as u8,
            left.key.scale_mode as u8,
            left.key.epilogue as u8,
        )
            .cmp(&(
                right.key.op as u8,
                right.key.device.sm,
                right.key.device.multiprocessor_count,
                right.key.m,
                right.key.n,
                right.key.k,
                right.key.activation_dtype as u8,
                right.key.weight_dtype as u8,
                right.key.output_dtype as u8,
                right.key.layout as u8,
                right.key.scale_mode as u8,
                right.key.epilogue as u8,
            ))
            .then_with(|| left.key.workspace_limit.cmp(&right.key.workspace_limit))
            .then_with(|| (left.tactic.backend as u8).cmp(&(right.tactic.backend as u8)))
            .then_with(|| left.tactic.value.cmp(&right.tactic.value))
    });
    let records = records
        .into_iter()
        .map(|record| {
            serde_json::json!({
                "key": super::report::key_json(&record.key),
                "tactic": {
                    "backend": super::report::backend_name(record.tactic.backend),
                    "id": record.tactic.value,
                    "implementation_version": record
                        .implementation_version
                        .unwrap_or_else(|| record.tactic.backend.implementation_version()),
                },
                "milliseconds": record.milliseconds,
            })
        })
        .collect::<Vec<_>>();
    let mut root = serde_json::json!({
        "schema": header.schema,
        "device_name": header.device_name,
        "sm": header.sm,
        "cuda_version": header.cuda_version,
        "cublas_version": header.cublas_version,
        "records": records,
    });
    if let Some(build_id) = header.kernel_build_id.as_deref() {
        root["kernel_build_id"] = serde_json::json!(build_id);
    }
    write_json_atomic(path, &root)
}

pub(crate) fn write_json_atomic(path: &Path, value: &serde_json::Value) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| {
        Error::Other(format!(
            "create tuning directory {}: {error}",
            parent.display()
        ))
    })?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("tactics.json");
    let temporary = parent.join(format!(".{file_name}.{}.tmp", std::process::id()));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| Error::Other(format!("create {}: {error}", temporary.display())))?;
        serde_json::to_writer_pretty(&mut file, value)
            .map_err(|error| Error::Other(format!("encode {}: {error}", path.display())))?;
        file.write_all(b"\n")
            .and_then(|_| file.sync_all())
            .map_err(|error| Error::Other(format!("flush {}: {error}", temporary.display())))?;
        fs::rename(&temporary, path).map_err(|error| {
            Error::Other(format!(
                "replace {} with {}: {error}",
                path.display(),
                temporary.display()
            ))
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub(crate) struct DatabaseLock {
    path: PathBuf,
}

impl DatabaseLock {
    pub(crate) fn acquire(database: &Path) -> Result<Self> {
        let path = database.with_extension("json.lock");
        for _ in 0..200 {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    writeln!(file, "{}", std::process::id()).map_err(|error| {
                        Error::Other(format!("write tuning lock {}: {error}", path.display()))
                    })?;
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    thread::sleep(Duration::from_millis(25));
                }
                Err(error) => {
                    return Err(Error::Other(format!(
                        "create tuning lock {}: {error}",
                        path.display()
                    )))
                }
            }
        }
        Err(Error::Other(format!(
            "timed out waiting for tuning lock {}",
            path.display()
        )))
    }
}

impl Drop for DatabaseLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn validate_v1_header(header: &TuningDbHeader) -> Result<()> {
    let missing = [
        ("device_name", header.device_name.as_deref()),
        ("cuda_version", header.cuda_version.as_deref()),
        ("cublas_version", header.cublas_version.as_deref()),
    ]
    .into_iter()
    .filter_map(|(name, value)| {
        value
            .filter(|value| !value.is_empty())
            .is_none()
            .then_some(name)
    })
    .collect::<Vec<_>>();
    if !missing.is_empty() || header.sm.is_none() {
        let mut missing = missing;
        if header.sm.is_none() {
            missing.push("sm");
        }
        return Err(Error::Other(format!(
            "CUDA tuning v1 header is missing {}",
            missing.join(", ")
        )));
    }
    Ok(())
}

fn parse_v1_records(value: &serde_json::Value) -> Result<Vec<ParsedGemmRecord>> {
    let records = value
        .as_array()
        .ok_or_else(|| Error::Other("CUDA tuning records must be an array".into()))?;
    records
        .iter()
        .enumerate()
        .map(|(index, value)| parse_v1_record(index, value))
        .collect()
}

fn parse_v1_record(index: usize, value: &serde_json::Value) -> Result<ParsedGemmRecord> {
    let record = value
        .as_object()
        .ok_or_else(|| Error::Other(format!("CUDA tuning record {index} must be an object")))?;
    let key = record
        .get("key")
        .map(|value| {
            value.as_object().ok_or_else(|| {
                Error::Other(format!("CUDA tuning record {index} key must be an object"))
            })
        })
        .transpose()?
        .unwrap_or(record);
    let label = format!("record {index}");
    let op = match required_string(key, "op", &label)? {
        "bf16" => GemmOp::Bf16,
        "w8a8" => GemmOp::W8A8,
        "fp8_f16" => GemmOp::Fp8F16,
        value => return invalid_field(&label, "op", value),
    };
    let activation_dtype = parse_dtype(required_string(key, "activation_dtype", &label)?, &label)?;
    let weight_dtype = parse_dtype(required_string(key, "weight_dtype", &label)?, &label)?;
    let output_dtype = parse_dtype(required_string(key, "output_dtype", &label)?, &label)?;
    let layout = match required_string(key, "layout", &label)? {
        "row_major" => GemmLayout::RowMajor,
        "weight_output_major" => GemmLayout::WeightOutputMajor,
        value => return invalid_field(&label, "layout", value),
    };
    let scale_mode = match required_string(key, "scale_mode", &label)? {
        "none" => ScaleMode::None,
        "per_tensor" => ScaleMode::PerTensor,
        "dynamic_row_per_output_channel" => ScaleMode::DynamicRowPerOutputChannel,
        value => return invalid_field(&label, "scale_mode", value),
    };
    let epilogue = match required_string(key, "epilogue", &label)? {
        "none" => Epilogue::None,
        "bias" => Epilogue::Bias,
        "bias_gelu" => Epilogue::BiasGelu,
        "bias_residual" => Epilogue::BiasResidual,
        value => return invalid_field(&label, "epilogue", value),
    };
    let device = key
        .get("device")
        .map(|value| parse_device(value, &label))
        .transpose()?;
    let (backend, tactic) = parse_v1_tactic(record, &label)?;
    validate_tactic(&label, backend, tactic)?;
    Ok(ParsedGemmRecord {
        op,
        device,
        m: required_usize(key, "m", &label)?,
        n: required_usize(key, "n", &label)?,
        k: required_usize(key, "k", &label)?,
        activation_dtype,
        weight_dtype,
        output_dtype,
        layout,
        scale_mode,
        epilogue,
        workspace_limit: optional_usize(key, "workspace_limit", &label)?.unwrap_or(usize::MAX),
        tactic: TacticId {
            backend,
            value: tactic,
        },
        implementation_version: record
            .get("tactic")
            .and_then(serde_json::Value::as_object)
            .and_then(|tactic| tactic.get("implementation_version"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|version| u32::try_from(version).ok()),
        milliseconds: record
            .get("milliseconds")
            .and_then(serde_json::Value::as_f64),
    })
}

fn parse_legacy_tactics(value: &serde_json::Value) -> Result<Vec<ParsedGemmRecord>> {
    let tactics = value
        .as_object()
        .ok_or_else(|| Error::Other("CUDA tuning tactics must be an object".into()))?;
    let mut records = Vec::with_capacity(tactics.len());
    for (key, value) in tactics {
        let (m, n, k) = parse_legacy_fp8_key(key)?;
        let tactic = value
            .get("tactic")
            .and_then(serde_json::Value::as_i64)
            .or_else(|| value.as_i64())
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| Error::Other(format!("CUDA tactic {key} has no valid tactic id")))?;
        let backend = parse_backend(
            value
                .get("backend")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("cutlass"),
            key,
        )?;
        validate_tactic(key, backend, tactic)?;
        records.push(ParsedGemmRecord {
            op: GemmOp::Fp8F16,
            device: None,
            m,
            n,
            k,
            activation_dtype: TuningDType::F8E4M3,
            weight_dtype: TuningDType::F8E4M3,
            output_dtype: TuningDType::F16,
            layout: GemmLayout::RowMajor,
            scale_mode: ScaleMode::PerTensor,
            // The legacy custom-bias records were measured on PI0.5's
            // language down projection, whose physical contract includes
            // both bias and residual. Preserve that meaning during migration
            // instead of letting the record shadow a plain GEMM.
            epilogue: if backend == TacticBackend::CublasLtCustomBias {
                Epilogue::BiasResidual
            } else {
                Epilogue::None
            },
            workspace_limit: usize::MAX,
            tactic: TacticId {
                backend,
                value: tactic,
            },
            implementation_version: None,
            milliseconds: value
                .get("milliseconds")
                .and_then(serde_json::Value::as_f64),
        });
    }
    Ok(records)
}

fn parse_v1_tactic(
    record: &serde_json::Map<String, serde_json::Value>,
    label: &str,
) -> Result<(TacticBackend, i32)> {
    let tactic = record
        .get("tactic")
        .ok_or_else(|| Error::Other(format!("CUDA tuning {label} has no tactic")))?;
    let (backend, value) = if let Some(object) = tactic.as_object() {
        let backend = required_string(object, "backend", label)?;
        let value = object
            .get("id")
            .or_else(|| object.get("value"))
            .and_then(serde_json::Value::as_i64);
        (backend, value)
    } else {
        (required_string(record, "backend", label)?, tactic.as_i64())
    };
    let value = value
        .and_then(|value| i32::try_from(value).ok())
        .ok_or_else(|| Error::Other(format!("CUDA tuning {label} has invalid tactic id")))?;
    Ok((parse_backend(backend, label)?, value))
}

fn parse_backend(value: &str, label: &str) -> Result<TacticBackend> {
    match value {
        "cutlass" => Ok(TacticBackend::Cutlass),
        "cublaslt" => Ok(TacticBackend::CublasLt),
        "cublaslt_custom" => Ok(TacticBackend::CublasLtCustom),
        "cublaslt_custom_bias" => Ok(TacticBackend::CublasLtCustomBias),
        "cublaslt_custom_split_serial" => Ok(TacticBackend::CublasLtCustomSplitSerial),
        "cublaslt_custom_split_geglu_cutlass" => Ok(TacticBackend::CublasLtCustomSplitGeGluCutlass),
        "cublaslt_custom_split_geglu_cutlass_2sm_auto" => {
            Ok(TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto)
        }
        "cublaslt_custom_split_geglu_cutlass_2sm_stage3" => {
            Ok(TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3)
        }
        "cublaslt_custom_split_geglu_cutlass_m522_explicit_2sm" => {
            Ok(TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm)
        }
        // Historical names remain read-only compatibility aliases. New
        // configs use the semantic backend name without encoding M in it.
        "cutlass_fp8_dual_geglu"
        | "cutlass_fp8_dual_geglu_m522"
        | "cutlass_fp8_dual_geglu_m533" => Ok(TacticBackend::CutlassFp8DualGeGlu),
        "cutlass_bf16_dual_geglu_m522" => Ok(TacticBackend::CutlassBf16DualGeGluM522),
        "cutlass_bf16_dual_geglu_m533" => Ok(TacticBackend::CutlassBf16DualGeGluM533),
        "cublaslt_custom_split_geglu_cutlass_bf16" => {
            Ok(TacticBackend::CublasLtCustomSplitGeGluCutlassBf16)
        }
        "vendor" => Ok(TacticBackend::Vendor),
        value => invalid_field(label, "backend", value),
    }
}

fn parse_dtype(value: &str, label: &str) -> Result<TuningDType> {
    match value {
        "f32" => Ok(TuningDType::F32),
        "f16" => Ok(TuningDType::F16),
        "bf16" => Ok(TuningDType::Bf16),
        "f8e4m3" => Ok(TuningDType::F8E4M3),
        "i8" => Ok(TuningDType::I8),
        "i32" => Ok(TuningDType::I32),
        value => invalid_field(label, "dtype", value),
    }
}

fn parse_device(value: &serde_json::Value, label: &str) -> Result<DeviceFingerprint> {
    let object = value
        .as_object()
        .ok_or_else(|| Error::Other(format!("CUDA tuning {label} device must be an object")))?;
    let sm = required_usize(object, "sm", label)?;
    let multiprocessor_count = required_usize(object, "multiprocessor_count", label)?;
    Ok(DeviceFingerprint {
        sm: u32::try_from(sm)
            .map_err(|_| Error::Other(format!("CUDA tuning {label} sm exceeds u32")))?,
        multiprocessor_count: u32::try_from(multiprocessor_count).map_err(|_| {
            Error::Other(format!(
                "CUDA tuning {label} multiprocessor_count exceeds u32"
            ))
        })?,
    })
}

fn required_string<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
    label: &str,
) -> Result<&'a str> {
    object
        .get(field)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::Other(format!("CUDA tuning {label} requires string `{field}`")))
}

fn required_usize(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    label: &str,
) -> Result<usize> {
    optional_usize(object, field, label)?
        .ok_or_else(|| Error::Other(format!("CUDA tuning {label} requires integer `{field}`")))
}

fn optional_usize(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    label: &str,
) -> Result<Option<usize>> {
    object
        .get(field)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    Error::Other(format!(
                        "CUDA tuning {label} field `{field}` must fit usize"
                    ))
                })
        })
        .transpose()
}

fn invalid_field<T>(label: &str, field: &str, value: &str) -> Result<T> {
    Err(Error::Other(format!(
        "CUDA tuning {label} has invalid {field} `{value}`"
    )))
}

fn validate_tactic(key: &str, backend: TacticBackend, tactic: i32) -> Result<()> {
    let valid = match backend {
        TacticBackend::Cutlass => (0..=7).contains(&tactic),
        TacticBackend::CublasLt => (0..64).contains(&tactic),
        TacticBackend::CublasLtCustom
        | TacticBackend::CublasLtCustomBias
        | TacticBackend::CublasLtCustomSplitSerial
        | TacticBackend::CublasLtCustomSplitGeGluCutlass
        | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto
        | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3
        | TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm
        | TacticBackend::CublasLtCustomSplitGeGluCutlassBf16 => {
            decode_cublaslt_custom_tactic(tactic).is_some()
        }
        TacticBackend::CutlassFp8DualGeGlu
        | TacticBackend::CutlassBf16DualGeGluM522
        | TacticBackend::CutlassBf16DualGeGluM533 => tactic == 0,
        TacticBackend::Vendor => tactic == 0,
    };
    if valid {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "CUDA tactic {key} has invalid {backend:?} id {tactic}"
        )))
    }
}

fn parse_sm(device: &str) -> Option<u32> {
    let (_, suffix) = device.rsplit_once("sm_")?;
    let digits = suffix
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    digits.parse().ok()
}

fn parse_legacy_fp8_key(key: &str) -> Result<(usize, usize, usize)> {
    let rest = key
        .strip_prefix("fp8_f16_m")
        .ok_or_else(|| Error::Other(format!("unsupported CUDA tuning key `{key}`")))?;
    let (m, rest) = rest
        .split_once("_n")
        .ok_or_else(|| Error::Other(format!("invalid CUDA tuning key `{key}`")))?;
    let (n, k) = rest
        .split_once("_k")
        .ok_or_else(|| Error::Other(format!("invalid CUDA tuning key `{key}`")))?;
    let parse = |value: &str| {
        value
            .parse::<usize>()
            .map_err(|error| Error::Other(format!("invalid CUDA tuning key `{key}`: {error}")))
    };
    Ok((parse(m)?, parse(n)?, parse(k)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_caps::CudaArchFamily;

    fn record(m: usize, tactic: i32, milliseconds: f64) -> GemmTuningRecord {
        GemmTuningRecord {
            key: GemmTuningKey {
                op: GemmOp::Fp8F16,
                device: DeviceFingerprint::from(&caps(87)),
                m,
                n: 1024,
                k: 1024,
                activation_dtype: TuningDType::F8E4M3,
                weight_dtype: TuningDType::F8E4M3,
                output_dtype: TuningDType::F16,
                layout: GemmLayout::RowMajor,
                scale_mode: ScaleMode::PerTensor,
                epilogue: Epilogue::None,
                workspace_limit: usize::MAX,
            },
            tactic: TacticId {
                backend: TacticBackend::Cutlass,
                value: tactic,
            },
            implementation_version: Some(TacticBackend::Cutlass.implementation_version()),
            milliseconds: Some(milliseconds),
        }
    }

    fn caps(sm: u32) -> CudaDeviceCaps {
        CudaDeviceCaps {
            device_name: "test".into(),
            compute_major: sm / 10,
            compute_minor: sm % 10,
            sm,
            multiprocessor_count: 16,
            arch_family: CudaDeviceCaps::classify(sm),
        }
    }

    fn versions() -> CudaLibraryVersions {
        CudaLibraryVersions {
            cuda: "12.6.1".into(),
            cublas: "12.6.4".into(),
        }
    }

    #[test]
    fn parses_legacy_database_without_model_in_key() {
        let db = TuningDb::from_json_str(
            r#"{"device":"Jetson Thor sm_110a","tactics":{"fp8_f16_m10_n2560_k1024":{"backend":"cutlass","tactic":4,"milliseconds":0.01}}}"#,
        )
        .unwrap();
        assert_eq!(db.header.sm, Some(110));
        let store = db.build_store(&caps(110), &versions()).unwrap();
        assert_eq!(store.len(), 1);
        assert!(store
            .gemm_records()
            .all(|record| format!("{:?}", record.key).find("pi05").is_none()));
    }

    #[test]
    fn migrates_legacy_custom_bias_as_bias_residual() {
        let db = TuningDb::from_json_str(
            r#"{"device":"Thor sm_110","tactics":{"fp8_f16_m522_n2048_k16384":{"backend":"cublaslt_custom_bias","tactic":18926998}}}"#,
        )
        .unwrap();
        let store = db.build_store(&caps(110), &versions()).unwrap();
        let record = store.gemm_records().next().unwrap();
        assert_eq!(record.key.epilogue, Epilogue::BiasResidual);
    }

    #[test]
    fn fp8_dual_geglu_backend_accepts_semantic_and_legacy_names() {
        for name in [
            "cutlass_fp8_dual_geglu",
            "cutlass_fp8_dual_geglu_m522",
            "cutlass_fp8_dual_geglu_m533",
        ] {
            assert_eq!(
                parse_backend(name, "test backend").unwrap(),
                TacticBackend::CutlassFp8DualGeGlu
            );
        }
    }

    #[test]
    fn parses_v1_full_physical_record() {
        let db = TuningDb::from_json_str(&format!(
            r#"{{
                "schema":"apxinf.cuda.tuning.v1",
                "kernel_build_id":"{}",
                "device_name":"test",
                "sm":87,
                "cuda_version":"12.6",
                "cublas_version":"12.6.4",
                "records":[{{
                    "key":{{
                        "op":"w8a8",
                        "device":{{"sm":87,"multiprocessor_count":16}},
                        "m":11,"n":1024,"k":2048,
                        "activation_dtype":"i8","weight_dtype":"i8","output_dtype":"bf16",
                        "layout":"weight_output_major",
                        "scale_mode":"dynamic_row_per_output_channel",
                        "epilogue":"bias",
                        "workspace_limit":4096
                    }},
                    "tactic":{{"backend":"vendor","id":0}},
                    "milliseconds":0.04
                }}]
            }}"#,
            super::super::KERNEL_BUILD_ID
        ))
        .unwrap();
        let store = db.build_store(&caps(87), &versions()).unwrap();
        let record = store.gemm_records().next().unwrap();
        assert_eq!(record.key.op, GemmOp::W8A8);
        assert_eq!(record.key.device.sm, 87);
        assert_eq!(record.key.layout, GemmLayout::WeightOutputMajor);
        assert_eq!(record.key.workspace_limit, 4096);
        assert_eq!(record.tactic.backend, TacticBackend::Vendor);
    }

    #[test]
    fn rejects_wrong_device() {
        let db = TuningDb::from_json_str(
            r#"{"device":"Orin sm_87","tactics":{"fp8_f16_m1_n8_k8":{"tactic":0}}}"#,
        )
        .unwrap();
        assert!(db.build_store(&caps(110), &versions()).is_err());
    }

    #[test]
    fn treats_cuda_and_cublas_versions_as_provenance() {
        let database = |cuda: &str, cublas: &str| {
            format!(
                r#"{{"schema":"apxinf.cuda.tuning.v1","kernel_build_id":"{}","device_name":"test","sm":87,"cuda_version":"{cuda}","cublas_version":"{cublas}","tactics":{{}}}}"#,
                super::super::KERNEL_BUILD_ID
            )
        };
        let compatible = TuningDb::from_json_str(&database("12.6", "12.6.4")).unwrap();
        assert!(compatible.build_store(&caps(87), &versions()).is_ok());

        let wrong_cuda = TuningDb::from_json_str(&database("13.0", "12.6.4")).unwrap();
        assert!(wrong_cuda
            .build_store(&caps(87), &versions())
            .unwrap()
            .is_empty());

        let wrong_cublas = TuningDb::from_json_str(&database("12.6", "12.7")).unwrap();
        assert!(wrong_cublas
            .build_store(&caps(87), &versions())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn accepts_v1_header_without_kernel_build_id() {
        let db = TuningDb::from_json_str(
            r#"{"schema":"apxinf.cuda.tuning.v1","device_name":"test","sm":87,"cuda_version":"12.6","cublas_version":"12.6.4","tactics":{}}"#,
        )
        .unwrap();
        assert!(db.build_store(&caps(87), &versions()).is_ok());
    }

    #[test]
    fn bundled_orin_and_rtx4090_databases_declare_v1_sm_headers() {
        for (raw, sm) in [
            (
                include_str!("../../../../configs/tuning/nvidia/orin-sm87/tactics.json"),
                87,
            ),
            (
                include_str!("../../../../configs/tuning/nvidia/rtx4090-sm89/tactics.json"),
                89,
            ),
        ] {
            let db = TuningDb::from_json_str(raw).unwrap();
            assert_eq!(db.header.schema, TUNING_SCHEMA_V1);
            assert_eq!(db.header.sm, Some(sm));
        }
    }

    #[test]
    fn kernel_build_id_is_provenance_only() {
        let db = TuningDb::from_json_str(
            r#"{
                "schema":"apxinf.cuda.tuning.v1",
                "kernel_build_id":"a-different-build",
                "device_name":"test",
                "sm":87,
                "cuda_version":"12.6",
                "cublas_version":"12.6.4",
                "records":[
                    {"key":{"op":"fp8_f16","m":8,"n":1024,"k":1024,"activation_dtype":"f8e4m3","weight_dtype":"f8e4m3","output_dtype":"f16","layout":"row_major","scale_mode":"per_tensor","epilogue":"none"},"tactic":{"backend":"cutlass","id":0,"implementation_version":1},"milliseconds":0.1}
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(db.build_store(&caps(87), &versions()).unwrap().len(), 1);
    }

    #[test]
    fn implementation_version_invalidates_only_stale_records() {
        let db = TuningDb::from_json_str(
            r#"{
                "schema":"apxinf.cuda.tuning.v1",
                "device_name":"test",
                "sm":87,
                "cuda_version":"12.6",
                "cublas_version":"12.6.4",
                "records":[
                    {"key":{"op":"fp8_f16","m":8,"n":1024,"k":1024,"activation_dtype":"f8e4m3","weight_dtype":"f8e4m3","output_dtype":"f16","layout":"row_major","scale_mode":"per_tensor","epilogue":"none"},"tactic":{"backend":"cublaslt","id":0,"implementation_version":2},"milliseconds":0.1},
                    {"key":{"op":"fp8_f16","m":9,"n":1024,"k":1024,"activation_dtype":"f8e4m3","weight_dtype":"f8e4m3","output_dtype":"f16","layout":"row_major","scale_mode":"per_tensor","epilogue":"none"},"tactic":{"backend":"cutlass","id":0,"implementation_version":1},"milliseconds":0.1}
                ]
            }"#,
        )
        .unwrap();
        let store = db.build_store(&caps(87), &versions()).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(
            store.gemm_records().next().unwrap().tactic.backend,
            TacticBackend::Cutlass
        );
    }

    #[test]
    fn summarizes_implementation_and_library_rejections() {
        let db = TuningDb::from_json_str(
            r#"{
                "schema":"apxinf.cuda.tuning.v1",
                "device_name":"test",
                "sm":87,
                "cuda_version":"12.6",
                "cublas_version":"13.0",
                "records":[
                    {"key":{"op":"fp8_f16","m":8,"n":1024,"k":1024,"activation_dtype":"f8e4m3","weight_dtype":"f8e4m3","output_dtype":"f16","layout":"row_major","scale_mode":"per_tensor","epilogue":"none"},"tactic":{"backend":"cublaslt","id":0,"implementation_version":1}},
                    {"key":{"op":"fp8_f16","m":9,"n":1024,"k":1024,"activation_dtype":"f8e4m3","weight_dtype":"f8e4m3","output_dtype":"f16","layout":"row_major","scale_mode":"per_tensor","epilogue":"none"},"tactic":{"backend":"cutlass","id":0,"implementation_version":2}},
                    {"key":{"op":"fp8_f16","m":10,"n":1024,"k":1024,"activation_dtype":"f8e4m3","weight_dtype":"f8e4m3","output_dtype":"f16","layout":"row_major","scale_mode":"per_tensor","epilogue":"none"},"tactic":{"backend":"cutlass","id":0,"implementation_version":1}}
                ]
            }"#,
        )
        .unwrap();

        let (records, rejected) = db.build_compatible_records(&caps(87), &versions());
        assert_eq!(records.len(), 1);
        assert_eq!(
            rejected,
            CompatibilityRejections {
                total: 2,
                implementation_version: 1,
                library_version: 1,
            }
        );
    }

    #[test]
    fn atomic_merge_appends_new_keys_and_deduplicates_existing_keys() {
        let directory = std::env::temp_dir().join(format!(
            "apxinf-tuning-db-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = directory.join("tactics.json");
        let caps = caps(87);
        let versions = versions();
        let header = TuningDb::header_for_cuda(&caps, &versions);

        assert_eq!(
            TuningDb::merge_record_atomic(&path, &header, &caps, &versions, record(8, 1, 0.2))
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            TuningDb::merge_record_atomic(&path, &header, &caps, &versions, record(9, 2, 0.3))
                .unwrap()
                .len(),
            2
        );
        let merged =
            TuningDb::merge_record_atomic(&path, &header, &caps, &versions, record(8, 3, 0.1))
                .unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(
            merged.lookup_gemm_exact(&record(8, 0, 0.0).key),
            Some(TacticId {
                backend: TacticBackend::Cutlass,
                value: 3,
            })
        );

        let persisted = TuningDb::from_json_file(&path).unwrap();
        assert_eq!(
            persisted.header.kernel_build_id.as_deref(),
            Some(super::super::KERNEL_BUILD_ID)
        );
        assert_eq!(persisted.build_store(&caps, &versions).unwrap().len(), 2);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn library_change_invalidates_only_the_affected_provider_records() {
        let db = TuningDb::from_json_str(
            r#"{
                "schema":"apxinf.cuda.tuning.v1",
                "device_name":"old name",
                "sm":87,
                "cuda_version":"12.6",
                "cublas_version":"13.0",
                "records":[
                    {"key":{"op":"fp8_f16","m":8,"n":1024,"k":1024,"activation_dtype":"f8e4m3","weight_dtype":"f8e4m3","output_dtype":"f16","layout":"row_major","scale_mode":"per_tensor","epilogue":"none"},"tactic":{"backend":"cublaslt","id":0,"implementation_version":1},"milliseconds":0.1},
                    {"key":{"op":"fp8_f16","m":9,"n":1024,"k":1024,"activation_dtype":"f8e4m3","weight_dtype":"f8e4m3","output_dtype":"f16","layout":"row_major","scale_mode":"per_tensor","epilogue":"none"},"tactic":{"backend":"cutlass","id":0,"implementation_version":1},"milliseconds":0.1}
                ]
            }"#,
        )
        .unwrap();
        let store = db.build_store(&caps(87), &versions()).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(
            store.gemm_records().next().unwrap().tactic.backend,
            TacticBackend::Cutlass
        );
    }

    #[test]
    fn rejects_incomplete_v1_header_but_preserves_legacy_compatibility() {
        assert!(TuningDb::from_json_str(
            r#"{"schema":"apxinf.cuda.tuning.v1","device_name":"test","sm":87,"tactics":{}}"#
        )
        .is_err());
        assert!(TuningDb::from_json_str(r#"{"device":"test sm_87","tactics":{}}"#).is_ok());
    }

    #[test]
    fn test_caps_are_consistent() {
        assert_eq!(caps(87).arch_family, CudaArchFamily::Sm80);
    }
}
