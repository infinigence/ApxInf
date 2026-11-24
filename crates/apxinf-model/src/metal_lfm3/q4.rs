//! QUEUE #8: one MLX affine group-64 4-bit projection (packed U32 weights +
//! BF16 scales/biases), loaded from the official 4-bit SafeTensors layout
//! (`<prefix>model.<...>.{weight,scales,biases}`; `Weights::load` strips the
//! Hugging Face `language_model.` wrapper before calling in).
//!
//! The multiplication is `Array::quantized_matmul` with `mode="affine"`
//! (bridge op 31), the same MLX op the Python lane's packed4 engine runs, so
//! G1 (native-4bit vs lane-4bit token parity) has a chance by construction.
//! A missing/incomplete triple is not an error: the caller falls back to the
//! BF16 path.

use apxinf_core::{DType, Device, Error, Result, Tensor};
use apxinf_mlx::{Array, MlxDType, Stream};
use std::collections::HashMap;

/// `true` when the worker was launched with the QUEUE #8 switch. Same source
/// as the banner field `native_4bit` (empty-arm gate).
pub fn enabled() -> bool {
    std::env::var_os("APXINF_LFM3_4BIT").is_some_and(|v| v == "1")
}

/// A group-64 affine 4-bit projection: packed `[out, in/8]` U32 weight and
/// BF16 `[out, in/64]` scales/biases. Dequantized value `(q - bias) * scale`.
#[derive(Clone)]
pub struct Q4 {
    weight: Array,
    scales: Array,
    biases: Array,
}

impl Q4 {
    /// `nn.QuantizedLinear` without bias: `x [.., in] @ W^T -> [.., out]`.
    pub fn apply(&self, x: &Array) -> Result<Array> {
        x.quantized_matmul(&self.weight, &self.scales, &self.biases, true, 64, 4)
    }

    /// The packed/scale/bias arrays to warm at load (uploaded bytes).
    pub fn arrays(&self) -> [Array; 3] {
        [self.weight.clone(), self.scales.clone(), self.biases.clone()]
    }

    /// Gather `ids` rows of the tied table and dequantize only those rows to
    /// BF16 `[ids, in]` — the input side of the quantized embedding. Decode
    /// projections never call this.
    pub fn take_rows(&self, ids: &Array) -> Result<Array> {
        let packed = self.weight.take(ids, 0)?;
        let scales = self.scales.take(ids, 0)?;
        let biases = self.biases.take(ids, 0)?;
        packed.dequantize(&scales, &biases, 64, 4)
    }
}

/// Load one tensor as an MLX array of `dtype`, consuming it from `map`.
/// Returns `Ok(None)` when the key is absent.
fn take_array(
    map: &mut HashMap<String, Tensor>,
    key: &str,
    dtype: MlxDType,
    shape: &[usize],
    stream: &Stream,
) -> Result<Option<Array>> {
    let Some(tensor) = map.remove(key) else {
        return Ok(None);
    };
    if tensor.device() != Device::Cpu {
        return Err(Error::UnsupportedDevice(tensor.device()));
    }
    if tensor.shape().dims() != shape {
        return Err(Error::ShapeMismatch {
            expected: format!("{key} {shape:?}"),
            got: format!("{:?}", tensor.shape()),
        });
    }
    // apxinf_loader surfaces safetensors U32 as I32 (apxinf_core has no
    // unsigned dtype); BF16 stays BF16. `Tensor::from_raw` keeps the payload
    // bytes verbatim, so `Array::from_bytes` with MlxDType::U32 reinterprets
    // the words as unsigned without any bit conversion.
    let want_core = if dtype == MlxDType::U32 {
        DType::I32
    } else {
        DType::BF16
    };
    if tensor.dtype() != want_core {
        return Err(Error::UnsupportedDType {
            got: tensor.dtype(),
            allowed: "LFM3 Q4 checkpoint (I32-backed U32 weights, BF16 scales/biases)",
        });
    }
    let bytes = tensor
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::Other("Q4 weight has no CPU storage".into()))?;
    Ok(Some(Array::from_bytes(stream, shape, dtype, bytes)?))
}

/// Load the quantized triple for one logical BF16 weight name (e.g.
/// `model.layers.0.feed_forward.w1.weight`) if and only if all three
/// tensors are present with the expected shapes. `None` means "not
/// quantized, use the BF16 path" — the map is left untouched so the BF16
/// loader produces the normal error for a partial triple.
pub fn take(
    map: &mut HashMap<String, Tensor>,
    logical: &str,
    input: usize,
    output: usize,
    stream: &Stream,
) -> Result<Option<Q4>> {
    let stem = logical.trim_end_matches(".weight");
    let parts = ["weight", "scales", "biases"];
    let complete = parts.iter().all(|part| {
        map.get(&format!("{stem}.{part}")).is_some_and(|tensor| {
            let (want, cols) = if *part == "weight" {
                (DType::I32, input / 8)
            } else {
                (DType::BF16, input / 64)
            };
            tensor.dtype() == want && tensor.shape().dims() == [output, cols]
        })
    });
    if !complete {
        return Ok(None);
    }
    Ok(Some(Q4 {
        weight: take_array(map, &format!("{stem}.weight"), MlxDType::U32, &[output, input / 8], stream)?
            .expect("triple completeness checked above"),
        scales: take_array(map, &format!("{stem}.scales"), MlxDType::BF16, &[output, input / 64], stream)?
            .expect("triple completeness checked above"),
        biases: take_array(map, &format!("{stem}.biases"), MlxDType::BF16, &[output, input / 64], stream)?
            .expect("triple completeness checked above"),
    }))
}
