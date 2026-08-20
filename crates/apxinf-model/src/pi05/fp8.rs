//! Static FP8 E4M3 quantization used by the Thor execution profile.
//!
//! Matrix weights use one scale per tensor. Activations obtain named scales
//! from a calibration file so CUDA graph replay never performs a reduction or
//! allocates scale tensors. Attention probabilities, residuals, and norms stay
//! FP16.

use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{Error, Result, Tensor};

/// Largest finite NVIDIA/CUDA E4M3 value (`0x7e`).
pub const E4M3_MAX: f32 = 448.0;

#[derive(Debug, Clone)]
pub struct Fp8Tensor {
    pub values: Tensor,
    /// Dequantization multiplier: `real = fp8(values) * scale`.
    pub scale: f32,
    pub amax: f32,
}

#[derive(Debug, Clone, Default)]
pub struct StaticFp8Calibration {
    scales: HashMap<String, f32>,
}

impl StaticFp8Calibration {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        Self::from_json_str(&raw)
    }

    /// Accept either `{ "name": scale }`, a nested `scales` object,
    /// or entries shaped as `{ "scale": number }` / `{ "amax": number }`.
    pub fn from_json_str(raw: &str) -> Result<Self> {
        let root: serde_json::Value = serde_json::from_str(raw)
            .map_err(|e| Error::Other(format!("π0.5 FP8 calibration JSON: {e}")))?;
        let values = root.get("scales").unwrap_or(&root);
        let object = values
            .as_object()
            .ok_or_else(|| Error::Other("π0.5 FP8 calibration must be a JSON object".into()))?;
        let mut scales = HashMap::with_capacity(object.len());
        for (name, value) in object {
            let scale = value
                .as_f64()
                .or_else(|| value.get("scale").and_then(|v| v.as_f64()))
                .or_else(|| {
                    value
                        .get("amax")
                        .and_then(|v| v.as_f64())
                        .map(|amax| amax / E4M3_MAX as f64)
                })
                .ok_or_else(|| {
                    Error::Other(format!(
                        "π0.5 FP8 calibration entry `{name}` has no numeric scale or amax"
                    ))
                })? as f32;
            if !scale.is_finite() || scale <= 0.0 {
                return Err(Error::Other(format!(
                    "π0.5 FP8 calibration entry `{name}` has invalid scale {scale}"
                )));
            }
            scales.insert(name.clone(), scale);
        }
        Ok(Self { scales })
    }

    pub fn scale(&self, name: &str) -> Result<f32> {
        self.scales
            .get(name)
            .copied()
            .ok_or_else(|| Error::Other(format!("missing π0.5 FP8 activation scale `{name}`")))
    }

    pub fn len(&self) -> usize {
        self.scales.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scales.is_empty()
    }
}

/// Quantize a CPU F32/F16/BF16 tensor using a static per-tensor scale.
pub fn quantize_e4m3(tensor: &Tensor, scale: f32) -> Result<Fp8Tensor> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Other(format!(
            "FP8 quantization scale must be finite and positive, got {scale}"
        )));
    }
    let source = tensor.to_f32_vec()?;
    let amax = source.iter().fold(0.0f32, |m, value| m.max(value.abs()));
    let inverse_scale = scale.recip();
    let bytes = source
        .iter()
        .map(|value| encode_e4m3(*value * inverse_scale))
        .collect::<Vec<_>>();
    Ok(Fp8Tensor {
        values: Tensor::from_f8_e4m3(tensor.shape().dims().to_vec(), &bytes)?,
        scale,
        amax,
    })
}

/// Select the standard absmax scale and quantize a weight matrix.
pub fn quantize_e4m3_absmax(tensor: &Tensor) -> Result<Fp8Tensor> {
    let source = tensor.to_f32_vec()?;
    let amax = source.iter().fold(0.0f32, |m, value| m.max(value.abs()));
    // All-zero matrices use scale 1 so their representation remains valid.
    let scale = if amax == 0.0 { 1.0 } else { amax / E4M3_MAX };
    quantize_e4m3(tensor, scale)
}

/// Decode on CPU for tests, cache inspection, and correctness diagnostics.
pub fn dequantize_e4m3(tensor: &Fp8Tensor) -> Result<Tensor> {
    let values = tensor
        .values
        .as_f8_e4m3()?
        .iter()
        .map(|byte| decode_e4m3(*byte) * tensor.scale)
        .collect::<Vec<_>>();
    Tensor::from_f32(tensor.values.shape().dims().to_vec(), &values)
}

/// CUDA-compatible saturating finite E4M3 encoding, round-to-nearest-even.
pub fn encode_e4m3(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let value = value.abs();
    if value == 0.0 {
        return sign;
    }
    if !value.is_finite() || value >= E4M3_MAX {
        return sign | 0x7e;
    }

    // Subnormals have a fixed 2^-9 quantum.
    if value < 2f32.powi(-6) {
        let mantissa = round_ties_even(value * 512.0).min(7) as u8;
        return sign | mantissa;
    }

    let exponent = value.log2().floor() as i32;
    let mut exponent_bits = exponent + 7;
    let normalized = value / 2f32.powi(exponent) - 1.0;
    let mut mantissa = round_ties_even(normalized * 8.0) as i32;
    if mantissa == 8 {
        exponent_bits += 1;
        mantissa = 0;
    }
    if exponent_bits > 15 || (exponent_bits == 15 && mantissa >= 7) {
        return sign | 0x7e;
    }
    sign | ((exponent_bits as u8) << 3) | mantissa as u8
}

pub fn decode_e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 {
        return sign * mantissa as f32 * 2f32.powi(-9);
    }
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    sign * (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
}

fn round_ties_even(value: f32) -> u32 {
    let floor = value.floor();
    let fraction = value - floor;
    let floor_u = floor as u32;
    if fraction > 0.5 || (fraction == 0.5 && floor_u & 1 == 1) {
        floor_u + 1
    } else {
        floor_u
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::DType;

    #[test]
    fn canonical_e4m3_values_match_cuda_layout() {
        assert_eq!(encode_e4m3(0.0), 0x00);
        assert_eq!(encode_e4m3(-0.0), 0x80);
        assert_eq!(encode_e4m3(1.0), 0x38);
        assert_eq!(encode_e4m3(-1.0), 0xb8);
        assert_eq!(encode_e4m3(448.0), 0x7e);
        assert_eq!(decode_e4m3(0x01), 2f32.powi(-9));
        assert_eq!(decode_e4m3(0x7e), 448.0);
        assert!(decode_e4m3(0x7f).is_nan());
    }

    #[test]
    fn encoding_saturates_and_rounds_ties_to_even() {
        assert_eq!(encode_e4m3(f32::INFINITY), 0x7e);
        assert_eq!(encode_e4m3(1000.0), 0x7e);
        // Halfway between 1.0 (mantissa 0, even) and 1.125 (mantissa 1).
        assert_eq!(encode_e4m3(1.0625), 0x38);
        // Halfway between mantissas 1 and 2 rounds to mantissa 2.
        assert_eq!(encode_e4m3(1.1875), 0x3a);
    }

    #[test]
    fn absmax_quantization_preserves_shape_and_range() {
        let source = Tensor::from_f32(vec![2, 2], &[-2.0, -0.5, 0.5, 2.0]).unwrap();
        let quantized = quantize_e4m3_absmax(&source).unwrap();
        assert_eq!(quantized.values.dtype(), DType::F8E4M3);
        assert_eq!(quantized.values.shape(), source.shape());
        assert_eq!(quantized.amax, 2.0);
        let output = dequantize_e4m3(&quantized).unwrap();
        let output = output.as_f32().unwrap();
        assert_eq!(output[0], -2.0);
        assert_eq!(output[3], 2.0);
    }

    #[test]
    fn calibration_accepts_scales_and_amax() {
        let calibration = StaticFp8Calibration::from_json_str(
            r#"{"scales":{"vision.input":0.25,"action.q":{"amax":448.0}}}"#,
        )
        .unwrap();
        assert_eq!(calibration.scale("vision.input").unwrap(), 0.25);
        assert_eq!(calibration.scale("action.q").unwrap(), 1.0);
        assert!(calibration.scale("missing").is_err());
    }
}
