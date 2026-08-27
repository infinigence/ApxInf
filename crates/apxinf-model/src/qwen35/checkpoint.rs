use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use apxinf_core::{DType, Error, Result};
use apxinf_loader::safetensors::{self, CheckpointManifest, TensorManifestEntry};

use super::{Qwen35Config, Qwen35LayerType};

#[derive(Clone, Debug)]
pub struct Qwen35CheckpointReport {
    pub tensor_count: usize,
    pub shard_count: usize,
    pub tensor_bytes: u64,
    pub dtype_counts: BTreeMap<DType, usize>,
    pub quantized_linears: usize,
    pub linear_attention_layers: usize,
    pub full_attention_layers: usize,
    pub ignored_modules: usize,
}

impl Qwen35CheckpointReport {
    pub fn inspect(model_dir: &Path, config: &Qwen35Config) -> Result<Self> {
        let manifest = safetensors::inspect_path(model_dir)
            .map_err(|error| Error::Other(format!("inspect qwen3.5 checkpoint: {error}")))?;
        let quantized_linears = validate_quantized_linears(&manifest, config)?;
        Ok(Self {
            tensor_count: manifest.tensors.len(),
            shard_count: manifest.shards.len(),
            tensor_bytes: manifest.tensor_bytes,
            dtype_counts: manifest.dtype_counts(),
            quantized_linears,
            linear_attention_layers: config
                .text
                .layer_types
                .iter()
                .filter(|layer| **layer == Qwen35LayerType::LinearAttention)
                .count(),
            full_attention_layers: config
                .text
                .layer_types
                .iter()
                .filter(|layer| **layer == Qwen35LayerType::FullAttention)
                .count(),
            ignored_modules: config.quantization.ignored_modules.len(),
        })
    }
}

fn validate_quantized_linears(
    manifest: &CheckpointManifest,
    config: &Qwen35Config,
) -> Result<usize> {
    let packed_bases = manifest
        .tensors
        .iter()
        .filter_map(|tensor| tensor.name.strip_suffix(".weight_packed"))
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    if packed_bases.is_empty() {
        return Err(Error::Other(
            "qwen3.5 checkpoint has no compressed-tensors weight_packed tensors".into(),
        ));
    }

    for base in &packed_bases {
        if config
            .quantization
            .ignored_modules
            .iter()
            .any(|ignored| ignored == base)
        {
            return Err(Error::Other(format!(
                "qwen3.5 checkpoint quantizes `{base}`, but config marks it ignored"
            )));
        }
        let packed = required_tensor(manifest, &format!("{base}.weight_packed"))?;
        let scale = required_tensor(manifest, &format!("{base}.weight_scale"))?;
        let zero = required_tensor(manifest, &format!("{base}.weight_zero_point"))?;
        let shape = required_tensor(manifest, &format!("{base}.weight_shape"))?;
        require_dtype(packed, DType::I32)?;
        require_dtype(scale, DType::BF16)?;
        require_dtype(zero, DType::I32)?;
        require_dtype(shape, DType::I64)?;
        if shape.shape != [2] {
            return Err(Error::Other(format!(
                "qwen3.5 `{base}.weight_shape` header is {:?}, expected [2]",
                shape.shape
            )));
        }
        let logical_shape = safetensors::read_small_i64(shape)
            .map_err(|error| Error::Other(format!("read `{base}.weight_shape`: {error}")))?;
        if logical_shape.len() != 2 || logical_shape.iter().any(|dimension| *dimension <= 0) {
            return Err(Error::Other(format!(
                "qwen3.5 `{base}.weight_shape` values must be two positive dimensions, got {logical_shape:?}"
            )));
        }
        let output = usize::try_from(logical_shape[0])
            .map_err(|_| Error::Other(format!("qwen3.5 `{base}` output dimension overflow")))?;
        let input = usize::try_from(logical_shape[1])
            .map_err(|_| Error::Other(format!("qwen3.5 `{base}` input dimension overflow")))?;
        validate_quantized_shapes(
            base,
            output,
            input,
            &packed.shape,
            &scale.shape,
            &zero.shape,
            config.quantization.num_bits,
            config.quantization.group_size,
        )?;
    }

    for tensor in &manifest.tensors {
        for suffix in [".weight_scale", ".weight_zero_point", ".weight_shape"] {
            if let Some(base) = tensor.name.strip_suffix(suffix) {
                if !packed_bases.contains(base) {
                    return Err(Error::Other(format!(
                        "qwen3.5 tensor `{}` has no `{base}.weight_packed` companion",
                        tensor.name
                    )));
                }
            }
        }
    }
    Ok(packed_bases.len())
}

#[allow(clippy::too_many_arguments)]
fn validate_quantized_shapes(
    base: &str,
    output: usize,
    input: usize,
    packed: &[usize],
    scale: &[usize],
    zero: &[usize],
    num_bits: usize,
    group_size: usize,
) -> Result<()> {
    if num_bits == 0 || 32 % num_bits != 0 || group_size == 0 {
        return Err(Error::Other(format!(
            "qwen3.5 `{base}` has invalid packing contract bits={num_bits}, group={group_size}"
        )));
    }
    let pack_factor = 32 / num_bits;
    if input % pack_factor != 0 || input % group_size != 0 || output % pack_factor != 0 {
        return Err(Error::Other(format!(
            "qwen3.5 `{base}` logical shape [{output}, {input}] is not divisible by pack={pack_factor}, group={group_size}"
        )));
    }
    let expected_packed = [output, input / pack_factor];
    let expected_scale = [output, input / group_size];
    let expected_zero = [output / pack_factor, input / group_size];
    if packed != expected_packed || scale != expected_scale || zero != expected_zero {
        return Err(Error::Other(format!(
            "qwen3.5 `{base}` packed layout mismatch: logical=[{output}, {input}], packed={packed:?} expected={expected_packed:?}, scale={scale:?} expected={expected_scale:?}, zero={zero:?} expected={expected_zero:?}"
        )));
    }
    Ok(())
}

fn required_tensor<'a>(
    manifest: &'a CheckpointManifest,
    name: &str,
) -> Result<&'a TensorManifestEntry> {
    manifest
        .tensor(name)
        .ok_or_else(|| Error::Other(format!("qwen3.5 checkpoint missing `{name}`")))
}

fn require_dtype(tensor: &TensorManifestEntry, expected: DType) -> Result<()> {
    if tensor.dtype != expected {
        return Err(Error::Other(format!(
            "qwen3.5 tensor `{}` is {}, expected {expected}",
            tensor.name, tensor.dtype
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_group32_asymmetric_packed_shapes() {
        validate_quantized_shapes(
            "mlp.gate_proj",
            17408,
            5120,
            &[17408, 640],
            &[17408, 160],
            &[2176, 160],
            4,
            32,
        )
        .unwrap();
    }

    #[test]
    fn rejects_zero_point_axis_drift() {
        let error = validate_quantized_shapes(
            "mlp.gate_proj",
            17408,
            5120,
            &[17408, 640],
            &[17408, 160],
            &[17408, 20],
            4,
            32,
        )
        .unwrap_err();
        assert!(error.to_string().contains("zero="), "{error}");
    }
}
