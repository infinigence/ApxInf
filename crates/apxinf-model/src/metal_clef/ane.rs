//! The Neural Engine share of each Clef MLP in the GPU+ANE column
//! composition. Intermediate columns `[GPU_COLUMNS, 12288)` run as one Core ML
//! program per layer; the GPU computes the rest with the GPU composition's MLP,
//! and the two shares are summed on the GPU. Portions adapted from EngineTailor
//! (MIT); see NOTICE.
//!
//! The program: gate and up as per-row symmetric INT8 1x1 convolutions, SiLU
//! as `0.5 v (1 + tanh(v / 2))`, gate and up normalized by their row maxima
//! before the product, down as K-split INT8 convolutions, 512-token tiles.
//! Weights are quantized on the host from the Q4 checkpoint.
use super::weights::{Mlp, Q4};
use apxinf_core::{Error, Result};
use apxinf_coreml::{ComputeUnits, Model, Operand, Package, ProgramBuilder, ProgramCache};
use apxinf_mlx::{Array, MlxDType, Stream};
use half::f16;

/// Hidden width and MLP width of the only geometry this split was built for.
const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 12288;
/// `12288 - round(12288 * 5632 / 12288 / 256) * 256`: the GPU keeps the first
/// 6656 columns, the Neural Engine runs the other 5632.
pub const GPU_COLUMNS: usize = 6656;
const TILE: usize = 512;
const FACTOR_SCALE: f64 = 16.0;
const DOWN_ABS_SUM_LIMIT: f64 = 8192.0;
const DOWN_PART: usize = 4096;

/// FP16 values of rows `rows` and columns `cols` of a Q4 group-64 weight,
/// dequantized on the device as `mx.dequantize` does (BF16 scales), then
/// rounded to FP16.
fn dense(q: &Q4, rows: (usize, usize), cols: (usize, usize)) -> Result<Vec<f16>> {
    let slice = |a: &Array, per: usize| -> Result<Array> {
        a.slice_axis(0, rows.0, rows.1)?
            .slice_axis(1, cols.0 / per, cols.1 / per)
    };
    let w = slice(&q.weight, 8)?
        .dequantize(&slice(&q.scales, 64)?, &slice(&q.biases, 64)?, 64, 4)?
        .cast(MlxDType::F16)?;
    Ok(w.to_bytes()?
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]))
        .collect())
}

/// Symmetric per-row INT8 with FP16 scales: `scale = max|w| / 127` in FP32,
/// rounded to FP16; `q = clip(rint(w / scale), -127, 127)`.
fn int8_per_row(w: &[f16], cols: usize) -> (Vec<i8>, Vec<f16>) {
    let mut q = Vec::with_capacity(w.len());
    let mut scales = Vec::with_capacity(w.len() / cols);
    for row in w.chunks_exact(cols) {
        let max = row.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
        let scale = f16::from_f32(max / 127.0);
        let s = scale.to_f32();
        scales.push(scale);
        q.extend(row.iter().map(|v| {
            if s == 0.0 {
                0
            } else {
                (v.to_f32() / s).round_ties_even().clamp(-127.0, 127.0) as i8
            }
        }));
    }
    (q, scales)
}

/// Columns `[start, start + width)` of a row-major `[rows, cols]` matrix.
fn columns(w: &[f16], cols: usize, start: usize, width: usize) -> Vec<f16> {
    w.chunks_exact(cols)
        .flat_map(|row| row[start..start + width].iter().copied())
        .collect()
}

/// One layer's Core ML program for the Neural Engine share of `mlp`.
pub fn program(mlp: &Mlp) -> Result<Package> {
    let n = INTERMEDIATE - GPU_COLUMNS;
    if mlp.gate.rows() != INTERMEDIATE || mlp.down.rows() != HIDDEN {
        return Err(Error::Contract(
            "Clef GPU+ANE needs the 4096 x 12288 MLP geometry",
        ));
    }
    let (qg, sg) = int8_per_row(
        &dense(&mlp.gate, (GPU_COLUMNS, INTERMEDIATE), (0, HIDDEN))?,
        HIDDEN,
    );
    let (qu, su) = int8_per_row(
        &dense(&mlp.up, (GPU_COLUMNS, INTERMEDIATE), (0, HIDDEN))?,
        HIDDEN,
    );
    let wd = dense(&mlp.down, (0, HIDDEN), (GPU_COLUMNS, INTERMEDIATE))?;
    let parts = n.div_ceil(DOWN_PART);
    let width = n / parts;
    let down: Vec<(Vec<i8>, Vec<f16>)> = (0..parts)
        .map(|p| int8_per_row(&columns(&wd, n, p * width, width), width))
        .collect();
    // Bound the dot products of the INT8 weights the Neural Engine uses.
    let mut row_sums = vec![0f64; HIDDEN];
    for (qd, sd) in &down {
        for (r, sum) in row_sums.iter_mut().enumerate() {
            let abs: f64 = qd[r * width..(r + 1) * width]
                .iter()
                .map(|&v| f64::from(v).abs())
                .sum();
            *sum += f64::from(sd[r].to_f32()) * abs;
        }
    }
    let max_row_sum = row_sums.iter().fold(1f64, |m, &v| m.max(v));
    let factor_cap = 2f64.powf(
        FACTOR_SCALE
            .min((DOWN_ABS_SUM_LIMIT / max_row_sum).sqrt())
            .log2()
            .floor(),
    );
    if factor_cap < 1.0 {
        return Err(Error::Contract(
            "Clef GPU+ANE down weights need factor scales outside the FP16 range",
        ));
    }

    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, HIDDEN, 1, TILE]);
    let wg = b.constexpr_blockwise_shift_scale(&qg, &[n, HIDDEN, 1, 1], &sg, &[n, 1, 1, 1])?;
    let g = b.conv(&x, &wg)?;
    let wu = b.constexpr_blockwise_shift_scale(&qu, &[n, HIDDEN, 1, 1], &su, &[n, 1, 1, 1])?;
    let u = b.conv(&x, &wu)?;
    let half = b.mul(&g, Operand::Scalar(f16::from_f32(0.5)), None)?;
    let t = b.tanh(&half);
    let t1 = b.add(&t, Operand::Scalar(f16::ONE), None)?;
    let s = b.mul(&half, Operand::Var(&t1), None)?;
    let sa = b.abs(&s);
    let s_max = b.reduce_max(&sa, &[1], true)?;
    let ua = b.abs(&u);
    let u_max = b.reduce_max(&ua, &[1], true)?;
    let inverse_cap = Operand::Scalar(f16::from_f64(1.0 / factor_cap));
    let floor = f16::from_f64(1.0 / FACTOR_SCALE);
    let sm = b.mul(&s_max, inverse_cap, None)?;
    let s_scale = b.maximum(&sm, Operand::Scalar(floor), Some("gate_scale"))?;
    let um = b.mul(
        &u_max,
        Operand::Scalar(f16::from_f64(1.0 / factor_cap)),
        None,
    )?;
    let u_scale = b.maximum(&um, Operand::Scalar(floor), Some("up_scale"))?;
    // Normalize before the product so a large SwiGLU value cannot overflow first.
    let sn = b.real_div(&s, Operand::Var(&s_scale), None)?;
    let un = b.real_div(&u, Operand::Var(&u_scale), None)?;
    let a = b.mul(&sn, Operand::Var(&un), None)?;
    let xs = if parts > 1 {
        b.split(&a, &vec![width; parts], 1)?
    } else {
        vec![a]
    };
    let mut y = None;
    for (xi, (qd, sd)) in xs.iter().zip(&down) {
        let wd =
            b.constexpr_blockwise_shift_scale(qd, &[HIDDEN, width, 1, 1], sd, &[HIDDEN, 1, 1, 1])?;
        let yi = b.conv(xi, &wd)?;
        y = Some(match y {
            None => yi,
            Some(acc) => b.add(&acc, Operand::Var(&yi), None)?,
        });
    }
    let y = b.identity(&y.expect("at least one down part"), Some("y"));
    b.finish(&[&y, &s_scale, &u_scale])
}

/// The GPU share of `mlp`: gate and up rows and down columns `[0, GPU_COLUMNS)`,
/// as contiguous Q4 copies.
pub fn gpu_share(mlp: &Mlp) -> Result<Mlp> {
    let rows = |q: &Q4| -> Result<Q4> {
        Ok(Q4 {
            weight: q.weight.slice_axis(0, 0, GPU_COLUMNS)?.contiguous()?,
            scales: q.scales.slice_axis(0, 0, GPU_COLUMNS)?.contiguous()?,
            biases: q.biases.slice_axis(0, 0, GPU_COLUMNS)?.contiguous()?,
        })
    };
    let d = &mlp.down;
    let share = Mlp {
        gate: rows(&mlp.gate)?,
        up: rows(&mlp.up)?,
        down: Q4 {
            weight: d.weight.slice_axis(1, 0, GPU_COLUMNS / 8)?.contiguous()?,
            scales: d.scales.slice_axis(1, 0, GPU_COLUMNS / 64)?.contiguous()?,
            biases: d.biases.slice_axis(1, 0, GPU_COLUMNS / 64)?.contiguous()?,
        },
    };
    let s = share.gate.weight.stream();
    s.eval(&[
        share.gate.weight.clone(),
        share.gate.scales.clone(),
        share.gate.biases.clone(),
        share.up.weight.clone(),
        share.up.scales.clone(),
        share.up.biases.clone(),
        share.down.weight.clone(),
        share.down.scales.clone(),
        share.down.biases.clone(),
    ])?;
    Ok(share)
}

/// Compiled-program cache: `APXINF_COREML_CACHE`, else
/// `~/Library/Caches/apxinf/coreml`. Programs compile once per macOS build.
pub fn cache_dir() -> std::path::PathBuf {
    std::env::var_os("APXINF_COREML_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join("Library/Caches/apxinf/coreml")
        })
}

/// The little-endian bytes of FP16 values, without a copy (Apple Silicon only).
fn f16_bytes(values: &[f16]) -> &[u8] {
    // SAFETY: f16 is a plain 2-byte value without padding, so its storage is valid as bytes; the
    // `coreml` feature targets little-endian aarch64 macOS only.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

/// One layer's loaded Neural Engine program.
pub struct AneMlp {
    model: Model,
}

impl AneMlp {
    pub fn load(cache: &ProgramCache, mlp: &Mlp) -> Result<Self> {
        let package = program(mlp)?;
        let admission = super::ane_admission::Admission::new(&package, [1, HIDDEN, 1, TILE])?;
        Ok(Self {
            model: cache.load_with_placement(
                &package,
                ComputeUnits::CpuAndNeuralEngine,
                std::time::Duration::from_secs(60),
                |facts| admission.check(facts),
            )?,
        })
    }

    pub fn warm(&self) -> Result<()> {
        let shape = [1, HIDDEN, 1, TILE];
        let scale_shape = [1, 1, 1, TILE];
        let input = vec![f16::ZERO; HIDDEN * TILE];
        drop(self.model.predict(
            &[("x", &input, &shape)],
            &[
                ("y", &shape),
                ("gate_scale", &scale_shape),
                ("up_scale", &scale_shape),
            ],
        )?);
        Ok(())
    }

    /// BF16 `[1, T, 4096]` as the little-endian bytes of FP16 channel-first
    /// tiles `[tiles, 4096, 512]`, zero-padded to whole tiles. Evaluates `h`.
    pub fn tiles(h: &Array) -> Result<Vec<u8>> {
        let [1, tokens, channels] = *h.shape() else {
            return Err(Error::Contract("Clef GPU+ANE tiles take [1, T, 4096]"));
        };
        let tiles = tokens.div_ceil(TILE);
        let padded = if tiles * TILE > tokens {
            let zeros = Array::zeros(h.stream(), &[1, tiles * TILE - tokens, channels], h.dtype())?;
            Array::concat(&[h, &zeros], 1)?
        } else {
            h.clone()
        };
        let packed = padded
            .reshape(&[tiles, TILE, channels])?
            .transpose(&[0, 2, 1])?
            .cast(MlxDType::F16)?;
        packed.to_bytes()
    }

    /// Predicts every tile of [`tiles`](Self::tiles) and restores BF16
    /// `[1, T, 4096]` on `stream`: rows times both scales in FP32.
    pub fn predict(&self, stream: &Stream, tiles: &[u8], tokens: usize) -> Result<Array> {
        // SAFETY: every bit pattern is a valid f16; `align_to` returns only the aligned middle.
        let (head, aligned, tail) = unsafe { tiles.align_to::<f16>() };
        let copied: Vec<f16>;
        let values: &[f16] = if head.is_empty() && tail.is_empty() {
            aligned
        } else {
            copied = tiles
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]))
                .collect();
            &copied
        };
        if values.is_empty() || values.len() % (HIDDEN * TILE) != 0 {
            return Err(Error::Contract(
                "Clef GPU+ANE tiles must be whole [4096, 512] tiles",
            ));
        }
        let mut parts = Vec::with_capacity(values.len() / (HIDDEN * TILE));
        for (i, tile) in values.chunks_exact(HIDDEN * TILE).enumerate() {
            let rows = TILE.min(tokens - i * TILE);
            let out = self.model.predict(
                &[("x", tile, &[1, HIDDEN, 1, TILE])],
                &[
                    ("y", &[1, HIDDEN, 1, TILE]),
                    ("gate_scale", &[1, 1, 1, TILE]),
                    ("up_scale", &[1, 1, 1, TILE]),
                ],
            )?;
            let scale = |v: &[f16]| -> Result<Array> {
                Array::from_bytes(stream, &[TILE, 1], MlxDType::F16, f16_bytes(v))?
                    .slice_axis(0, 0, rows)?
                    .cast(MlxDType::F32)
            };
            let y = Array::from_bytes(stream, &[HIDDEN, TILE], MlxDType::F16, f16_bytes(&out[0]))?
                .transpose(&[1, 0])?
                .slice_axis(0, 0, rows)?
                .cast(MlxDType::F32)?;
            parts.push(
                y.mul(&scale(&out[1])?)?
                    .mul(&scale(&out[2])?)?
                    .reshape(&[1, rows, HIDDEN])?
                    .cast(MlxDType::BF16)?,
            );
        }
        Array::concat(&parts.iter().collect::<Vec<_>>(), 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_rows_match_the_python_rounding() {
        let w: Vec<f16> = [0.5f32, -1.27, 0.635, 0.0, 0.0, 0.0, 0.0, 0.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let (q, s) = int8_per_row(&w, 4);
        let scale = f16::from_f32(f16::from_f32(1.27).to_f32() / 127.0);
        assert_eq!(s, [scale, f16::ZERO]);
        let expect = |v: f32| (f16::from_f32(v).to_f32() / scale.to_f32()).round_ties_even() as i8;
        assert_eq!(&q[..4], &[expect(0.5), -127, expect(0.635), 0]);
        assert_eq!(&q[4..], &[0; 4]);
    }

    /// Every layer's program from the Q4 checkpoint matches a package that a
    /// coremltools 9.0 reference build of the column split compiled: the
    /// same weight file bytes and the same decoded program (writer metadata
    /// and `buildInfo` excluded).
    #[test]
    #[ignore = "requires the Clef checkpoint, recorded packages and a Metal device"]
    fn column_programs_match_the_recorded_packages() {
        use sha2::{Digest, Sha256};
        use std::collections::HashMap;
        let var = |n: &str| std::env::var(n).unwrap_or_else(|_| panic!("set {n}"));
        let checkpoint = std::path::PathBuf::from(var("APXINF_CLEF_CHECKPOINT"));
        let read = |n: &str| std::fs::read_to_string(checkpoint.join(n)).unwrap();
        let config =
            super::super::Config::from_json(&read("config.json"), &read("joint_head_config.json"))
                .unwrap();
        let stream = Stream::metal(0).unwrap();
        let backbone =
            super::super::weights::Backbone::load(&config, &stream, &checkpoint).unwrap();
        let mut recorded = HashMap::new();
        for entry in std::fs::read_dir(var("APXINF_CLEF_COLUMNS_RECORDED")).unwrap() {
            let dir = entry.unwrap().path().join("Data/com.apple.CoreML");
            if let Ok(weights) = std::fs::read(dir.join("weights/weight.bin")) {
                recorded.insert(Sha256::digest(&weights).to_vec(), dir);
            }
        }
        let comparable = |mut m: apxinf_coreml::proto::Model| {
            m.description.as_mut().unwrap().metadata = None;
            m.ml_program.as_mut().unwrap().attributes.clear();
            m
        };
        let mut matched = 0;
        for (i, layer) in backbone.layers.iter().enumerate() {
            let package = program(&layer.mlp).unwrap();
            let Some(dir) = recorded.get(&Sha256::digest(&package.weights).to_vec()) else {
                println!("layer {i}: no recorded package has this weight file");
                continue;
            };
            let theirs: apxinf_coreml::proto::Model =
                prost::Message::decode(&std::fs::read(dir.join("model.mlmodel")).unwrap()[..])
                    .unwrap();
            let same = comparable(theirs) == comparable(package.model.clone());
            println!(
                "layer {i}: weight file matches {}; program {}",
                dir.display(),
                if same { "matches" } else { "DIFFERS" }
            );
            matched += usize::from(same);
        }
        assert_eq!(
            matched,
            backbone.layers.len(),
            "layers matching a recorded package"
        );
    }

    #[test]
    fn columns_take_a_block_of_every_row() {
        let w: Vec<f16> = (0..12).map(|v| f16::from_f32(v as f32)).collect();
        let c: Vec<f32> = columns(&w, 4, 1, 2).iter().map(|v| v.to_f32()).collect();
        assert_eq!(c, [1., 2., 5., 6., 9., 10.]);
    }
}
