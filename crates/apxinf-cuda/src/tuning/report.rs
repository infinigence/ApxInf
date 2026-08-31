//! Serializable diagnostics for autotune runs.

use std::path::Path;

use apxinf_core::{Error, Result};
use serde_json::{json, Value};

use crate::context::CudaLibraryVersions;
use crate::device_caps::CudaDeviceCaps;

use super::{CandidateMeasurement, GemmTuningKey, TacticBackend, TuningOutcome};

const REPORT_SCHEMA_V1: &str = "apxinf.cuda.tuning-report.v1";

/// Replace the latest report for one exact key and atomically publish the
/// resulting hardware-wide report file.
pub(crate) fn append_outcome(
    path: &Path,
    caps: &CudaDeviceCaps,
    versions: &CudaLibraryVersions,
    outcome: &TuningOutcome,
) -> Result<()> {
    let _lock = super::db::DatabaseLock::acquire(path)?;
    let mut runs = if path.is_file() {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|error| Error::Other(format!("parse {}: {error}", path.display())))?;
        if value.get("schema").and_then(Value::as_str) != Some(REPORT_SCHEMA_V1) {
            return Err(Error::Other(format!(
                "unsupported CUDA tuning report schema in {}",
                path.display()
            )));
        }
        value
            .get("runs")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| Error::Other(format!("{} has no runs array", path.display())))?
    } else {
        Vec::new()
    };
    let key = key_json(&outcome.winner.key);
    runs.retain(|run| run.get("key") != Some(&key));
    runs.push(outcome_json(outcome));
    // Sort by the complete serialized physical key. Several epilogues may
    // legitimately share M/N/K, so a partial shape sort would make output
    // order depend on the input report's previous ordering.
    runs.sort_by_cached_key(|run| {
        serde_json::to_string(run.get("key").unwrap_or(&Value::Null)).unwrap_or_default()
    });
    let root = json!({
        "schema": REPORT_SCHEMA_V1,
        "device_name": caps.device_name.as_str(),
        "sm": caps.sm,
        "cuda_version": versions.cuda.as_str(),
        "cublas_version": versions.cublas.as_str(),
        "kernel_build_id": super::KERNEL_BUILD_ID,
        "runs": runs,
    });
    super::db::write_json_atomic(path, &root)
}

pub fn outcome_json(outcome: &TuningOutcome) -> Value {
    json!({
        "key": key_json(&outcome.winner.key),
        "winner": tactic_json(
            outcome.winner.tactic.backend,
            outcome.winner.tactic.value,
            outcome.winner.milliseconds,
        ),
        "candidates": outcome
            .candidates
            .iter()
            .map(measurement_json)
            .collect::<Vec<_>>(),
    })
}

fn measurement_json(measurement: &CandidateMeasurement) -> Value {
    tactic_json(
        measurement.tactic.backend,
        measurement.tactic.value,
        measurement.milliseconds,
    )
    .as_object()
    .cloned()
    .map(|mut value| {
        value.insert("correct".into(), json!(measurement.correct));
        Value::Object(value)
    })
    .expect("tactic JSON is an object")
}

pub(crate) fn key_json(key: &GemmTuningKey) -> Value {
    json!({
        "op": match key.op {
            super::GemmOp::Bf16 => "bf16",
            super::GemmOp::W8A8 => "w8a8",
            super::GemmOp::Fp8F16 => "fp8_f16",
        },
        "device": {
            "sm": key.device.sm,
            "multiprocessor_count": key.device.multiprocessor_count,
        },
        "m": key.m,
        "n": key.n,
        "k": key.k,
        "activation_dtype": dtype_name(key.activation_dtype),
        "weight_dtype": dtype_name(key.weight_dtype),
        "output_dtype": dtype_name(key.output_dtype),
        "layout": match key.layout {
            super::GemmLayout::RowMajor => "row_major",
            super::GemmLayout::WeightOutputMajor => "weight_output_major",
        },
        "scale_mode": match key.scale_mode {
            super::ScaleMode::None => "none",
            super::ScaleMode::PerTensor => "per_tensor",
            super::ScaleMode::DynamicRowPerOutputChannel => "dynamic_row_per_output_channel",
        },
        "epilogue": match key.epilogue {
            super::Epilogue::None => "none",
            super::Epilogue::Bias => "bias",
            super::Epilogue::BiasGelu => "bias_gelu",
            super::Epilogue::BiasResidual => "bias_residual",
        },
        "workspace_limit": key.workspace_limit,
    })
}

pub(crate) fn tactic_json(backend: TacticBackend, id: i32, milliseconds: Option<f64>) -> Value {
    json!({
        "backend": backend_name(backend),
        "id": id,
        "implementation_version": backend.implementation_version(),
        "milliseconds": milliseconds,
    })
}

pub(crate) fn backend_name(backend: TacticBackend) -> &'static str {
    match backend {
        TacticBackend::Cutlass => "cutlass",
        TacticBackend::CublasLt => "cublaslt",
        TacticBackend::CublasLtCustom => "cublaslt_custom",
        TacticBackend::CublasLtCustomBias => "cublaslt_custom_bias",
        TacticBackend::CublasLtCustomSplitSerial => "cublaslt_custom_split_serial",
        TacticBackend::CublasLtCustomSplitGeGluCutlass => "cublaslt_custom_split_geglu_cutlass",
        TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto => {
            "cublaslt_custom_split_geglu_cutlass_2sm_auto"
        }
        TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3 => {
            "cublaslt_custom_split_geglu_cutlass_2sm_stage3"
        }
        TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm => {
            "cublaslt_custom_split_geglu_cutlass_m522_explicit_2sm"
        }
        TacticBackend::CutlassFp8DualGeGlu => "cutlass_fp8_dual_geglu",
        TacticBackend::CutlassBf16DualGeGluM522 => "cutlass_bf16_dual_geglu_m522",
        TacticBackend::CutlassBf16DualGeGluM533 => "cutlass_bf16_dual_geglu_m533",
        TacticBackend::CublasLtCustomSplitGeGluCutlassBf16 => {
            "cublaslt_custom_split_geglu_cutlass_bf16"
        }
        TacticBackend::Vendor => "vendor",
    }
}

fn dtype_name(dtype: super::TuningDType) -> &'static str {
    match dtype {
        super::TuningDType::F32 => "f32",
        super::TuningDType::F16 => "f16",
        super::TuningDType::Bf16 => "bf16",
        super::TuningDType::F8E4M3 => "f8e4m3",
        super::TuningDType::I8 => "i8",
        super::TuningDType::I32 => "i32",
    }
}
