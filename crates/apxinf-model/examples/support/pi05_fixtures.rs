//! Fixture inputs, error metrics and stage dumps shared by the π0.5 examples
//! that compare backends (`pi05_hip_bench`, `pi05_stage_dump`).
//!
//! Inputs use the cross-backend bundle's formats — NHWC `u8` images, `u32le`
//! token ids, `bf16` noise as `u16le` — and metrics are `pi05_bench`'s, so a
//! HIP run is judged exactly as a CUDA run against the same fixtures.

#![allow(dead_code)]

use std::path::Path;

pub type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

/// The input/output geometry fixtures depend on — plain numbers, so this
/// support module depends on neither backend's config type.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub views: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub action_horizon: usize,
    pub action_dim: usize,
}

/// One request's inputs, as host data.
pub struct Fixture {
    /// `views * image_size² * 3` bytes, NHWC.
    pub images: Vec<u8>,
    pub token_ids: Vec<u32>,
    /// BF16-exact values, `[action_horizon, action_dim]`.
    pub noise: Vec<f32>,
}

impl Fixture {
    /// Read fixture files, or fall back to `pi05_bench`'s deterministic
    /// synthetic images, zero token ids and zero noise for any not given.
    pub fn load(
        config: Geometry,
        token_count: usize,
        images: Option<&str>,
        token_ids: Option<&str>,
        noise: Option<&str>,
    ) -> AnyResult<Self> {
        let image_bytes = config.views * config.image_size * config.image_size * 3;
        let images = match images {
            Some(path) => read_exact(path, image_bytes)?,
            None => (0..image_bytes)
                .map(|i| ((i * 73 + i / 11 + 19) & 0xff) as u8)
                .collect(),
        };
        let token_ids = match token_ids {
            Some(path) => read_exact(path, token_count * 4)?
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            None => vec![0; token_count],
        };
        let elements = config.action_horizon * config.action_dim;
        let noise = match noise {
            Some(path) => read_exact(path, elements * 2)?
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
                .collect(),
            None => vec![0.0; elements],
        };
        Ok(Self {
            images,
            token_ids,
            noise,
        })
    }

    /// The images as `[views * patches_per_view, 3 * patch²]` BF16 patches,
    /// normalized to `[-1, 1]` on the host — the same values the device
    /// RGB→patches kernels produce.
    pub fn patches(&self, config: Geometry) -> Vec<half::bf16> {
        let (size, patch) = (config.image_size, config.patch_size);
        let per_side = size / patch;
        let width = 3 * patch * patch;
        let mut out = vec![half::bf16::ZERO; config.views * per_side * per_side * width];
        for view in 0..config.views {
            for py in 0..per_side {
                for px in 0..per_side {
                    let row = view * per_side * per_side + py * per_side + px;
                    for channel in 0..3 {
                        for dy in 0..patch {
                            for dx in 0..patch {
                                let (y, x) = (py * patch + dy, px * patch + dx);
                                let byte =
                                    self.images[((view * size + y) * size + x) * 3 + channel];
                                out[row * width + channel * patch * patch + dy * patch + dx] =
                                    half::bf16::from_f32((f32::from(byte) / 255.0) * 2.0 - 1.0);
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

fn read_exact(path: &str, len: usize) -> AnyResult<Vec<u8>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != len {
        return Err(format!("{path}: expected {len} bytes, got {}", bytes.len()).into());
    }
    Ok(bytes)
}

/// A reference file's `raw_actions`.
pub fn reference_actions(path: &str, config: Geometry) -> AnyResult<Vec<f32>> {
    let document: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let actions = document
        .get("raw_actions")
        .and_then(serde_json::Value::as_array)
        .ok_or("reference `raw_actions` is missing or not an array")?
        .iter()
        .map(|v| {
            v.as_f64()
                .map(|v| v as f32)
                .ok_or("reference action is not numeric")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected = config.action_horizon * config.action_dim;
    if actions.len() != expected {
        return Err(format!(
            "reference has {} actions, expected {expected}",
            actions.len()
        )
        .into());
    }
    Ok(actions)
}

/// `pi05_bench`'s comparison metrics, in F64.
pub fn metrics(actual: &[f32], expected: &[f32]) -> AnyResult<serde_json::Value> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err(format!("comparing {} values with {}", actual.len(), expected.len()).into());
    }
    if actual.iter().chain(expected).any(|v| !v.is_finite()) {
        return Err("comparison contains a non-finite value".into());
    }
    let (mut dot, mut aa, mut ee, mut err2, mut err_sum, mut max_abs) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0f64);
    for (&a, &e) in actual.iter().zip(expected) {
        let (a, e) = (f64::from(a), f64::from(e));
        dot += a * e;
        aa += a * a;
        ee += e * e;
        err2 += (a - e) * (a - e);
        err_sum += (a - e).abs();
        max_abs = max_abs.max((a - e).abs());
    }
    let cosine = if aa == 0.0 || ee == 0.0 {
        f64::from(aa == ee)
    } else {
        dot / (aa.sqrt() * ee.sqrt())
    };
    let relative_l2 = if ee == 0.0 {
        if err2 == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (err2 / ee).sqrt()
    };
    Ok(serde_json::json!({
        "cosine": cosine,
        "relative_l2": relative_l2,
        "max_abs": max_abs,
        "mean_abs": err_sum / actual.len() as f64,
        "rmse": (err2 / actual.len() as f64).sqrt(),
    }))
}

/// The BF16 acceptance gate of `doc/pi05-cuda-regression.md`.
pub fn passes_bf16_gate(metrics: &serde_json::Value) -> bool {
    metrics["cosine"].as_f64().is_some_and(|c| c >= 0.999)
        && metrics["relative_l2"].as_f64().is_some_and(|r| r <= 0.05)
}

/// P50/P95/min/max/mean of millisecond samples.
pub fn latency_json(samples: &[f64]) -> serde_json::Value {
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    let at = |q: f64| ordered[((ordered.len() - 1) as f64 * q).round() as usize];
    serde_json::json!({
        "p50": at(0.5),
        "p95": at(0.95),
        "min": ordered[0],
        "max": ordered[ordered.len() - 1],
        "mean": samples.iter().sum::<f64>() / samples.len() as f64,
        "samples": samples.len(),
    })
}

/// Write `values` as a little-endian F32 `.npy` of `shape`.
pub fn write_npy(dir: &Path, name: &str, shape: &[usize], values: &[f32]) -> AnyResult<()> {
    assert_eq!(
        shape.iter().product::<usize>(),
        values.len(),
        "{name}: shape and data disagree"
    );
    let dims = shape.iter().map(|d| format!("{d},")).collect::<String>();
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}), }}");
    // Magic (6) + version (2) + length (2) + header + newline, padded to 64.
    let padding = (64 - (10 + header.len() + 1) % 64) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend((header.len() as u16).to_le_bytes());
    bytes.extend(header.as_bytes());
    bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(format!("{name}.npy")), bytes)?;
    Ok(())
}
