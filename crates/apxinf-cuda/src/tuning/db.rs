use std::path::Path;

use apxinf_core::{Error, Result};

use crate::context::CudaLibraryVersions;
use crate::device_caps::CudaDeviceCaps;

use super::key::{
    DeviceFingerprint, Epilogue, GemmLayout, GemmOp, GemmTuningKey, ScaleMode, TuningDType,
};
use super::store::{
    decode_cublaslt_custom_tactic, GemmTuningRecord, TacticBackend, TacticId, TacticStore,
};

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
    milliseconds: Option<f64>,
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

    /// Materialize device-specific physical records. Callers loading several
    /// databases can merge their resulting stores before the one-time install.
    pub fn build_records(
        &self,
        caps: &CudaDeviceCaps,
        versions: &CudaLibraryVersions,
    ) -> Result<Vec<GemmTuningRecord>> {
        if self.header.schema == TUNING_SCHEMA_V1
            && self.header.device_name.as_deref() != Some(caps.device_name.as_str())
        {
            return Err(Error::Other(format!(
                "tuning database targets GPU `{}`, current device is `{}`",
                self.header.device_name.as_deref().unwrap_or("<missing>"),
                caps.device_name
            )));
        }
        if let Some(sm) = self.header.sm {
            if sm != caps.sm {
                return Err(Error::Other(format!(
                    "tuning database targets SM{sm}, current device is SM{}",
                    caps.sm
                )));
            }
        }
        if let Some(expected) = self.header.kernel_build_id.as_deref() {
            let actual = super::KERNEL_BUILD_ID;
            if expected != actual {
                return Err(Error::Other(format!(
                    "tuning database kernel build `{expected}` != current `{actual}`"
                )));
            }
        }
        validate_version("CUDA", self.header.cuda_version.as_deref(), &versions.cuda)?;
        validate_version(
            "cuBLAS",
            self.header.cublas_version.as_deref(),
            &versions.cublas,
        )?;
        let device = DeviceFingerprint::from(caps);
        Ok(self
            .records
            .iter()
            .map(|record| GemmTuningRecord {
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
                milliseconds: record.milliseconds,
            })
            .collect())
    }
}

fn validate_v1_header(header: &TuningDbHeader) -> Result<()> {
    let missing = [
        ("kernel_build_id", header.kernel_build_id.as_deref()),
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

fn validate_version(kind: &str, expected: Option<&str>, actual: &str) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let expected = version_components(expected).ok_or_else(|| {
        Error::Other(format!(
            "tuning database has invalid {kind} version `{expected}`"
        ))
    })?;
    let actual_components = version_components(actual).ok_or_else(|| {
        Error::Other(format!(
            "runtime returned invalid {kind} version `{actual}`"
        ))
    })?;
    if actual_components.starts_with(&expected) {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "tuning database targets {kind} {}, current runtime is {actual}",
            expected
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(".")
        )))
    }
}

fn version_components(value: &str) -> Option<Vec<u32>> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    value
        .split('.')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()
        .ok()
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
            epilogue: Epilogue::None,
            workspace_limit: usize::MAX,
            tactic: TacticId {
                backend,
                value: tactic,
            },
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
        "i64" => Ok(TuningDType::I64),
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
        TacticBackend::Vendor => tactic >= 0,
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
                    "tactic":{{"backend":"vendor","id":3}},
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
    fn validates_declared_cuda_and_cublas_versions() {
        let database = |cuda: &str, cublas: &str| {
            format!(
                r#"{{"schema":"apxinf.cuda.tuning.v1","kernel_build_id":"{}","device_name":"test","sm":87,"cuda_version":"{cuda}","cublas_version":"{cublas}","tactics":{{}}}}"#,
                super::super::KERNEL_BUILD_ID
            )
        };
        let compatible = TuningDb::from_json_str(&database("12.6", "12.6.4")).unwrap();
        assert!(compatible.build_store(&caps(87), &versions()).is_ok());

        let wrong_cuda = TuningDb::from_json_str(&database("13.0", "12.6.4")).unwrap();
        assert!(wrong_cuda.build_store(&caps(87), &versions()).is_err());

        let wrong_cublas = TuningDb::from_json_str(&database("12.6", "12.7")).unwrap();
        assert!(wrong_cublas.build_store(&caps(87), &versions()).is_err());
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
