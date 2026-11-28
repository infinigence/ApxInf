//! The ANE half of the LFM3 MLP column split.
//!
//! `APXINF_LFM3_ANE=<columns>` splits the intermediate dimension at `<columns>`
//! (or, given a fraction, at the pre-registered `S = round(share * 7680 / 64) *
//! 64` ANE width): the lower columns stay on the GPU and the upper ones run as
//! one FP16 Core ML program per layer on the Neural Engine. This module owns the
//! second half, so the split has one implementation instead of a call site
//! buried in `model.rs`.
//!
//! The program is the chain measured by issue #151 (`fp16_arm` in
//! `crates/apxinf-coreml/examples/lfm3_mlp_parity_probe.rs`):
//!
//! * gate and up as FP16 1x1 convolutions over the ANE columns;
//! * SiLU as `0.5 v (1 + tanh(v / 2))`;
//! * gate/up normalized by their per-token maxima (`1/cap`, floor `1/16`) so the
//!   normalized product cannot overflow FP16;
//! * down as K-split FP16 convolutions (`ceil(width / 4096)` near-equal parts),
//!   summed;
//! * output `y` restored on the host as `y * gate_scale * up_scale` in FP32.
//!
//! The switch is only read at model load. `APXINF_LFM3_ANE` unset (or `0`) never
//! builds or loads a Core ML program, and that arm is byte-identical to the
//! GPU-only path. A non-zero split is deliberately not byte-identical: the
//! projection's column count changes, so GEMM tiling and reduction order can
//! differ (#128). The token gate in
//! `doc/lfm3-ane-fp16-port-preregister.md` decides that, not a byte comparison.
//!
//! The program is compiled through [`ProgramCache`] and must be placed entirely
//! on the Neural Engine; any conv whose preferred device is not the ANE rejects
//! the program instead of silently reporting a CPU/GPU fallback as an ANE run.

#[cfg(feature = "coreml")]
use super::weights::Linear;
use apxinf_core::{Error, Result};
#[cfg(feature = "coreml")]
use apxinf_coreml::{
    ComputeDeviceKind, ComputeUnits, ExpectedPlacement, Model, Operand, Package, ProgramBuilder,
    ProgramCache, Var,
};
#[cfg(not(feature = "coreml"))]
use apxinf_mlx::Array;
#[cfg(feature = "coreml")]
use apxinf_mlx::{Array, MlxDType};
#[cfg(feature = "coreml")]
use half::f16;
#[cfg(feature = "coreml")]
use std::path::PathBuf;
#[cfg(feature = "coreml")]
use std::time::Duration;

/// LFM3-1.7B geometry; the checkpoint pins both.
#[cfg(feature = "coreml")]
const HIDDEN: usize = 2048;
/// The shipped ANE tile (`#151`/Clef), and the only tile this port compiles.
#[cfg(feature = "coreml")]
const TILE: usize = 512;
/// Down K-split width; the program uses `ceil(width / DOWN_PART)` parts.
#[cfg(feature = "coreml")]
const DOWN_PART: usize = 4096;
#[cfg(feature = "coreml")]
const FACTOR_SCALE: f64 = 16.0;
#[cfg(feature = "coreml")]
const DOWN_ABS_SUM_LIMIT: f64 = 8192.0;

/// One layer's ANE column share.
#[cfg_attr(feature = "coreml", allow(dead_code))]
pub(crate) struct AneSlice {
    /// First intermediate column owned by this slice (`share`).
    start: usize,
    /// One past the last column (`intermediate`).
    end: usize,
    /// The loaded FP16 program. `None` only before [`AneSlice::load`].
    #[cfg(feature = "coreml")]
    model: Option<Model>,
}

impl AneSlice {
    pub(crate) fn new(start: usize, end: usize) -> Result<Self> {
        if start >= end {
            return Err(Error::Other(format!(
                "ANE slice needs start < end, got {start}..{end}"
            )));
        }
        Ok(Self {
            start,
            end,
            #[cfg(feature = "coreml")]
            model: None,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn width(&self) -> usize {
        self.end - self.start
    }

    /// `silu(gate) * up` restricted to `[start, end)`, from projections the
    /// caller already computed for the full intermediate width.
    ///
    /// This is the GPU-only fallback for builds without the `coreml` feature.
    /// The `coreml` arm below never computes the ANE columns on the GPU.
    #[cfg(not(feature = "coreml"))]
    pub(crate) fn from_parts(&self, gate: &Array, up: &Array) -> Result<Array> {
        let gate = gate.slice_axis(1, self.start, self.end)?;
        let up = up.slice_axis(1, self.start, self.end)?;
        let g = gate.mul(&gate.sigmoid()?)?;
        g.mul(&up)
    }
}

#[cfg(feature = "coreml")]
impl AneSlice {
    /// Builds and loads this layer's FP16 program from the full BF16 weights.
    ///
    /// The caller keeps only the GPU share afterwards; the program owns the FP16
    /// constants for `[start, end)`.
    pub(crate) fn load(
        cache: &ProgramCache,
        start: usize,
        end: usize,
        w1: &Linear,
        w3: &Linear,
        w2: &Linear,
    ) -> Result<Self> {
        let package = program(w1, w3, w2, start, end)?;
        let mut slice = Self::new(start, end)?;
        slice.model = Some(cache.load_with_placement(
            &package,
            ComputeUnits::CpuAndNeuralEngine,
            Duration::from_secs(600),
            admit,
        )?);
        Ok(slice)
    }

    /// `down(silu(x·W1ᵀ) * (x·W3ᵀ))` over columns `[start, end)`, restored to
    /// BF16 `[1, tokens, HIDDEN]` on `x`'s stream.
    ///
    /// The input is the post-`ffn_norm` activation `[1, tokens, 2048]`. Tokens
    /// are laid out as 512-token channel-first tiles; the last tile is zero
    /// padded and only its real rows are returned, so a 128-token prefill pays
    /// the 4x tile it deploys with.
    pub(crate) fn predict(&self, x: &Array) -> Result<Array> {
        let model = self.model.as_ref().ok_or_else(|| {
            Error::Other("LFM3 ANE slice was used before its Core ML program loaded".into())
        })?;
        let [1, tokens, channels] = *x.shape() else {
            return Err(Error::Other(format!(
                "LFM3 ANE input must be [1, tokens, {HIDDEN}], got {:?}",
                x.shape()
            )));
        };
        if channels != HIDDEN {
            return Err(Error::Other(format!(
                "LFM3 ANE input needs {HIDDEN} channels, got {channels}"
            )));
        }
        if tokens == 0 {
            return Err(Error::Other("LFM3 ANE input has no tokens".into()));
        }
        let tiles = tokens.div_ceil(TILE);
        let padded = if tiles * TILE > tokens {
            let zeros = Array::zeros(x.stream(), &[1, tiles * TILE - tokens, channels], x.dtype())?;
            Array::concat(&[x, &zeros], 1)?
        } else {
            x.clone()
        };
        let packed = padded
            .reshape(&[tiles, TILE, channels])?
            .transpose(&[0, 2, 1])?
            .cast(MlxDType::F16)?
            .contiguous()?;
        let bytes = packed.to_bytes_ready()?;
        let values: Vec<f16> = bytes
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]))
            .collect();

        let stream = x.stream();
        let mut parts = Vec::with_capacity(tiles);
        for (index, tile) in values.chunks_exact(HIDDEN * TILE).enumerate() {
            let rows = (tokens - index * TILE).min(TILE);
            let out = model.predict(
                &[("x", tile, &[1, HIDDEN, 1, TILE])],
                &[
                    ("y", &[1, HIDDEN, 1, TILE]),
                    ("gate_scale", &[1, 1, 1, TILE]),
                    ("up_scale", &[1, 1, 1, TILE]),
                ],
            )?;
            let scale = |values: &[f16]| -> Result<Array> {
                Ok(
                    Array::from_bytes(stream, &[TILE, 1], MlxDType::F16, &f16_bytes(values))?
                        .slice_axis(0, 0, rows)?
                        .cast(MlxDType::F32)?,
                )
            };
            let y = Array::from_bytes(stream, &[HIDDEN, TILE], MlxDType::F16, &f16_bytes(&out[0]))?
                .transpose(&[1, 0])?
                .slice_axis(0, 0, rows)?
                .cast(MlxDType::F32)?;
            parts.push(
                y.mul(&scale(&out[1])?)?
                    .mul(&scale(&out[2])?)?
                    .cast(MlxDType::BF16)?
                    .reshape(&[1, rows, HIDDEN])?,
            );
        }
        Array::concat(&parts.iter().collect::<Vec<_>>(), 1)
    }
}

/// The FP16 Core ML program for one layer's `[start, end)` share.
#[cfg(feature = "coreml")]
fn program(w1: &Linear, w3: &Linear, w2: &Linear, start: usize, end: usize) -> Result<Package> {
    let width = end - start;
    if width == 0 {
        return Err(Error::Contract("LFM3 ANE program needs a non-empty share"));
    }
    let gate = rows(w1, start, end)?;
    let up = rows(w3, start, end)?;
    let down = columns(w2, start, end)?;
    let parts = width.div_ceil(DOWN_PART);
    let base = width / parts;
    let extra = width % parts;
    let sizes: Vec<usize> = (0..parts)
        .map(|part| base + usize::from(part < extra))
        .collect();
    let down: Vec<Vec<f16>> = {
        let mut offset = 0;
        sizes
            .iter()
            .map(|&size| {
                let part = column_slice(&down, width, offset, size);
                offset += size;
                part
            })
            .collect()
    };
    let cap = factor_cap(&down, &sizes)?;

    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, HIDDEN, 1, TILE])?;
    let g = b.conv_fp16(&x, &gate, &[width, HIDDEN, 1, 1])?;
    let u = b.conv_fp16(&x, &up, &[width, HIDDEN, 1, 1])?;
    let half = b.mul(&g, Operand::Scalar(f16::from_f32(0.5)), None)?;
    let t = b.tanh(&half)?;
    let t1 = b.add(&t, Operand::Scalar(f16::ONE), None)?;
    let silu = b.mul(&half, Operand::Var(&t1), None)?;
    let sa = b.abs(&silu)?;
    let s_max = b.reduce_max(&sa, &[1], true)?;
    let ua = b.abs(&u)?;
    let u_max = b.reduce_max(&ua, &[1], true)?;
    let inverse_cap = f16::from_f64(1.0 / cap);
    let floor = f16::from_f64(1.0 / FACTOR_SCALE);
    let sm = b.mul(&s_max, Operand::Scalar(inverse_cap), None)?;
    let s_scale = b.maximum(&sm, Operand::Scalar(floor), Some("gate_scale"))?;
    let um = b.mul(&u_max, Operand::Scalar(inverse_cap), None)?;
    let u_scale = b.maximum(&um, Operand::Scalar(floor), Some("up_scale"))?;
    let sn = b.real_div(&silu, Operand::Var(&s_scale), None)?;
    let un = b.real_div(&u, Operand::Var(&u_scale), None)?;
    let a = b.mul(&sn, Operand::Var(&un), None)?;
    let xs: Vec<Var> = if parts > 1 {
        b.split(&a, &sizes, 1)?
    } else {
        vec![a]
    };
    let mut y: Option<Var> = None;
    for (xi, weights) in xs.iter().zip(&down) {
        let part_width = weights.len() / HIDDEN;
        let yi = b.conv_fp16(xi, weights, &[HIDDEN, part_width, 1, 1])?;
        y = Some(match y {
            None => yi,
            Some(acc) => b.add(&acc, Operand::Var(&yi), None)?,
        });
    }
    let y = b.identity(&y.expect("at least one down part"), Some("y"))?;
    b.finish(&[&y, &s_scale, &u_scale])
}

/// Rows `[start, end)` of a canonical `[HIDDEN, intermediate]` projection as
/// row-major `[width, HIDDEN]` FP16.
#[cfg(feature = "coreml")]
fn rows(linear: &Linear, start: usize, end: usize) -> Result<Vec<f16>> {
    let weight = linear
        .bf16()?
        .slice_axis(1, start, end)?
        .transpose(&[1, 0])?
        .contiguous()?;
    to_f16(&weight)
}

/// Input columns `[start, end)` of a canonical `[intermediate, HIDDEN]`
/// projection as row-major `[HIDDEN, width]` FP16.
#[cfg(feature = "coreml")]
fn columns(linear: &Linear, start: usize, end: usize) -> Result<Vec<f16>> {
    let weight = linear
        .bf16()?
        .slice_axis(0, start, end)?
        .transpose(&[1, 0])?
        .contiguous()?;
    to_f16(&weight)
}

#[cfg(feature = "coreml")]
fn to_f16(weight: &Array) -> Result<Vec<f16>> {
    let values = weight.cast(MlxDType::F32)?.to_bytes_ready()?;
    values
        .chunks_exact(4)
        .map(|bytes| {
            let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if !value.is_finite() || value.abs() > 65504.0 {
                return Err(Error::Contract(
                    "LFM3 ANE weight is outside the FP16 finite range",
                ));
            }
            Ok(f16::from_f32(value))
        })
        .collect()
}

/// Columns `[start, start + width)` of a row-major `[rows, cols]` matrix.
#[cfg(feature = "coreml")]
fn column_slice(w: &[f16], cols: usize, start: usize, width: usize) -> Vec<f16> {
    w.chunks_exact(cols)
        .flat_map(|row| row[start..start + width].iter().copied())
        .collect()
}

/// The probe's `factor_cap` over the down parts: bound the largest restored row
/// sum so the normalization stays inside FP16.
#[cfg(feature = "coreml")]
fn factor_cap(parts: &[Vec<f16>], sizes: &[usize]) -> Result<f64> {
    let mut row_sums = vec![0f64; HIDDEN];
    for (weights, &width) in parts.iter().zip(sizes) {
        for (row, sum) in row_sums.iter_mut().enumerate() {
            let abs: f64 = weights[row * width..(row + 1) * width]
                .iter()
                .map(|v| f64::from(v.to_f32().abs()))
                .sum();
            *sum += abs;
        }
    }
    let max_row_sum = row_sums.iter().fold(1f64, |m, &v| m.max(v));
    let cap = 2f64.powf(
        FACTOR_SCALE
            .min((DOWN_ABS_SUM_LIMIT / max_row_sum).sqrt())
            .log2()
            .floor(),
    );
    if cap < 1.0 {
        return Err(Error::Contract(
            "LFM3 ANE down weights need factor scales outside the FP16 range",
        ));
    }
    Ok(cap)
}

#[cfg(feature = "coreml")]
fn admit(facts: &ExpectedPlacement) -> Result<()> {
    let mut convs = 0usize;
    for function in &facts.functions {
        for operation in &function.operations {
            if !(operation.operator == "conv" || operation.operator.ends_with(".conv")) {
                continue;
            }
            convs += 1;
            let on_ane = operation
                .usage
                .as_ref()
                .and_then(|usage| usage.preferred.as_ref())
                .is_some_and(|device| device.kind == ComputeDeviceKind::NeuralEngine);
            if !on_ane {
                return Err(Error::Other(format!(
                    "LFM3 ANE program has a conv that is not NeuralEngine-preferred at {:?}",
                    operation.path
                )));
            }
        }
    }
    if convs == 0 {
        let operators: Vec<&str> = facts
            .functions
            .iter()
            .flat_map(|function| function.operations.iter())
            .map(|operation| operation.operator.as_str())
            .collect();
        return Err(Error::Other(format!(
            "LFM3 ANE program reported no conv operations (operators: {operators:?})"
        )));
    }
    Ok(())
}

#[cfg(feature = "coreml")]
fn f16_bytes(values: &[f16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Where compiled LFM3 programs live. Same content-addressed cache as the other
/// Metal families, so a program is compiled once per macOS build.
#[cfg(feature = "coreml")]
pub(crate) fn cache_dir() -> PathBuf {
    std::env::var_os("APXINF_COREML_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join("Library/Caches/apxinf/coreml")
        })
}

#[cfg(all(test, feature = "coreml"))]
mod tests {
    use super::*;

    #[test]
    fn split_geometry_needs_a_non_empty_share() {
        assert!(AneSlice::new(7680, 7680).is_err());
        assert!(AneSlice::new(7681, 7680).is_err());
        assert_eq!(AneSlice::new(5632, 7680).unwrap().width(), 2048);
    }

    #[test]
    fn down_parts_cover_the_share_without_a_remainder() {
        for width in [1usize, 64, 2048, 4096, 4097, 5632, 7680] {
            let parts = width.div_ceil(DOWN_PART);
            let base = width / parts;
            let extra = width % parts;
            let sizes: Vec<usize> = (0..parts)
                .map(|part| base + usize::from(part < extra))
                .collect();
            assert_eq!(sizes.iter().sum::<usize>(), width);
            assert!(sizes.iter().all(|&size| size > 0));
        }
    }
}
