//! The Neural Engine share of each Clef MLP in the EngineTailor GPU+ANE
//! frontier A04 (campaign/clef-flash9b-m1pro fe66a747 `a04.py`, MIT; see
//! NOTICE). Intermediate columns `[GPU_COLUMNS, 12288)` run as one Core ML
//! program per layer; the GPU computes the rest with the C14 MLP, and the two
//! shares are summed on the GPU.
//!
//! The program follows `_ane_mlp_program`: gate and up as per-row symmetric
//! INT8 1x1 convolutions, SiLU as `0.5 v (1 + tanh(v / 2))`, gate and up
//! normalized by their row maxima before the product, down as K-split INT8
//! convolutions, 512-token tiles. Weights are quantized on the host from the
//! Q4 checkpoint exactly as the Python helper does.
use super::weights::{Mlp, Q4};
use apxinf_core::{Error, Result};
use apxinf_coreml::{ComputeUnits, Model, Operand, Package, ProgramBuilder, ProgramCache, Var};
use apxinf_mlx::{Array, MlxDType, Stream};
use half::f16;

/// Hidden width and MLP width of the only geometry A04 was built for.
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
            "Clef A04 needs the 4096 x 12288 MLP geometry",
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
            "Clef A04 down weights need factor scales outside the FP16 range",
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

/// m4a012 (M4 campaign `stk011-m4a012-fp16-1248`): per MLP, the `M4_INT8`
/// neurons whose int8 encoding adds the least error run on the Neural Engine
/// as per-channel int8, the next `M4_FP16` as FP16, and the rest on the GPU.
pub const M4_INT8: usize = 5632;
pub const M4_FP16: usize = 1248;
pub const M4_TILE: usize = 128;

/// numpy 2.x `einsum("ij,ij->i", r, r, dtype=float64)` on Apple Silicon: one
/// two-lane accumulator, each 8-element block fused-multiply-added from its
/// last pair to its first, the tail in pairs; the lanes are summed last.
fn einsum_squares(r: &[f64]) -> f64 {
    let mut acc = [0f64; 2];
    let blocks = r.len() / 8;
    for b in 0..blocks {
        for k in (0..4).rev() {
            for (l, a) in acc.iter_mut().enumerate() {
                let v = r[8 * b + 2 * k + l];
                *a = v.mul_add(v, *a);
            }
        }
    }
    for pair in r[8 * blocks..].chunks(2) {
        for (a, &v) in acc.iter_mut().zip(pair) {
            *a = v.mul_add(v, *a);
        }
    }
    acc[0] + acc[1]
}

/// numpy's pairwise summation of a contiguous float64 array (`sum`).
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(0.0, |s, &v| s + v);
    }
    if n <= 128 {
        let mut r: [f64; 8] = a[..8].try_into().unwrap();
        let whole = n - n % 8;
        for block in a[8..whole].chunks_exact(8) {
            for (x, &v) in r.iter_mut().zip(block) {
                *x += v;
            }
        }
        let head = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        return a[whole..].iter().fold(head, |s, &v| s + v);
    }
    let half = n / 2 - (n / 2) % 8;
    pairwise_sum(&a[..half]) + pairwise_sum(&a[half..])
}

/// `_row_stats`: per row, the squared norm and the squared error of its
/// per-channel int8 encoding (FP32 arithmetic, float64 sums).
fn row_stats(w: &[f16], cols: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut norm, mut err) = (Vec::new(), Vec::new());
    for row in w.chunks_exact(cols) {
        let values: Vec<f32> = row.iter().map(|v| v.to_f32()).collect();
        let max = values.iter().fold(0f32, |m, v| m.max(v.abs()));
        let s = f16::from_f32(max / 127.0).to_f32();
        let wide: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
        let e: Vec<f64> = values
            .iter()
            .map(|&v| {
                let q = if s == 0.0 {
                    0.0
                } else {
                    (v / s).round_ties_even().clamp(-127.0, 127.0)
                };
                f64::from(v - q * s)
            })
            .collect();
        norm.push(einsum_squares(&wide));
        err.push(einsum_squares(&e));
    }
    (norm, err)
}

/// `_column_stats` of a `[rows, cols]` down weight: per column the squared
/// norm (column sums in 16 row blocks, each block summed row by row, the
/// blocks added in order) and the mean squared rounding error per-row int8
/// steps add to any column.
fn column_stats(w: &[f16], cols: usize) -> (Vec<f64>, f64) {
    let rows = w.len() / cols;
    let mut norm = vec![0f64; cols];
    let mut steps = Vec::with_capacity(rows);
    // np.array_split(w, 16): the first rows % 16 blocks take one more row.
    let (base, extra) = (rows / 16, rows % 16);
    let mut start = 0;
    for b in 0..16 {
        let end = start + base + usize::from(b < extra);
        let mut block = vec![0f64; cols];
        for row in w[start * cols..end * cols].chunks_exact(cols) {
            for (s, v) in block.iter_mut().zip(row) {
                let v = f64::from(v.to_f32());
                *s = v.mul_add(v, *s);
            }
            let max = row.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
            steps.push(f64::from(max) / 127.0);
        }
        for (n, s) in norm.iter_mut().zip(&block) {
            *n += s;
        }
        start = end;
    }
    let squares: Vec<f64> = steps.iter().map(|s| s * s).collect();
    (norm, pairwise_sum(&squares) / 12.0)
}

/// FP16 values of a whole Q4 group-64 weight (`_dense`).
fn dense_all(q: &Q4) -> Result<Vec<f16>> {
    let (rows, cols) = (q.rows(), q.weight.shape()[1] * 8);
    dense(q, (0, rows), (0, cols))
}

/// m4a012's neuron split of one MLP (`_split_neurons`): (GPU, Neural Engine
/// int8, Neural Engine FP16) intermediate indices, each ascending. Neurons are
/// ranked by the error their int8 encoding would add,
/// `|d|^2 (|u|^2 e(g) + |g|^2 e(u)) + |g|^2 |u|^2 e(d)`, with a stable sort.
pub struct NeuronSplit {
    pub gpu: Vec<usize>,
    pub int8: Vec<usize>,
    pub fp16: Vec<usize>,
}

/// The three dense weights of one MLP, decoded once for the split and the
/// program: gate and up `[INTERMEDIATE, HIDDEN]`, down `[HIDDEN, INTERMEDIATE]`.
pub struct DenseMlp {
    gate: Vec<f16>,
    up: Vec<f16>,
    down: Vec<f16>,
}

impl DenseMlp {
    pub fn new(mlp: &Mlp) -> Result<Self> {
        if mlp.gate.rows() != INTERMEDIATE || mlp.down.rows() != HIDDEN {
            return Err(Error::Contract(
                "Clef m4a012 needs the 4096 x 12288 MLP geometry",
            ));
        }
        Ok(Self {
            gate: dense_all(&mlp.gate)?,
            up: dense_all(&mlp.up)?,
            down: dense_all(&mlp.down)?,
        })
    }

    pub fn split(&self) -> NeuronSplit {
        let (ng, eg) = row_stats(&self.gate, HIDDEN);
        let (nu, eu) = row_stats(&self.up, HIDDEN);
        let (nd, ed) = column_stats(&self.down, INTERMEDIATE);
        let score: Vec<f64> = (0..INTERMEDIATE)
            .map(|j| nd[j] * (nu[j] * eg[j] + ng[j] * eu[j]) + ng[j] * nu[j] * ed)
            .collect();
        let mut order: Vec<usize> = (0..INTERMEDIATE).collect();
        order.sort_by(|&a, &b| score[a].total_cmp(&score[b]));
        let sorted = |s: &[usize]| {
            let mut v = s.to_vec();
            v.sort_unstable();
            v
        };
        NeuronSplit {
            int8: sorted(&order[..M4_INT8]),
            fp16: sorted(&order[M4_INT8..M4_INT8 + M4_FP16]),
            gpu: sorted(&order[M4_INT8 + M4_FP16..]),
        }
    }

    /// Rows `neurons` of a `[INTERMEDIATE, HIDDEN]` weight.
    fn rows(w: &[f16], neurons: &[usize]) -> Vec<f16> {
        neurons
            .iter()
            .flat_map(|&n| w[n * HIDDEN..(n + 1) * HIDDEN].iter().copied())
            .collect()
    }

    /// Columns `neurons` of the down weight, `[HIDDEN, neurons.len()]`.
    fn down_columns(&self, neurons: &[usize]) -> Vec<f16> {
        self.down
            .chunks_exact(INTERMEDIATE)
            .flat_map(|row| neurons.iter().map(move |&n| row[n]))
            .collect()
    }

    /// m4a012's Neural Engine program for this MLP (`_ane_mlp_program`), as
    /// coremltools 9.0 writes it: FP16 gate conv and the following `* 0.5`
    /// fused into one conv with a zero bias.
    pub fn program(&self, split: &NeuronSplit) -> Result<Package> {
        let (n, nf) = (split.int8.len(), split.fp16.len());
        let (qg, sg) = int8_per_row(&Self::rows(&self.gate, &split.int8), HIDDEN);
        let (qu, su) = int8_per_row(&Self::rows(&self.up, &split.int8), HIDDEN);
        let fg = Self::rows(&self.gate, &split.fp16);
        let fu = Self::rows(&self.up, &split.fp16);
        let fd = self.down_columns(&split.fp16);
        let wd = self.down_columns(&split.int8);
        let parts = n.div_ceil(DOWN_PART);
        let width = n / parts;
        let down: Vec<(Vec<i8>, Vec<f16>)> = (0..parts)
            .map(|p| int8_per_row(&columns(&wd, n, p * width, width), width))
            .collect();
        // Bound the dot products of the int8 and FP16 weights the Neural Engine uses.
        let mut row_sums = vec![0f64; HIDDEN];
        for (qd, sd) in &down {
            for (r, sum) in row_sums.iter_mut().enumerate() {
                let abs: Vec<f64> = qd[r * width..(r + 1) * width]
                    .iter()
                    .map(|&v| f64::from(v).abs())
                    .collect();
                *sum += f64::from(sd[r].to_f32()) * pairwise_sum(&abs);
            }
        }
        for (r, sum) in row_sums.iter_mut().enumerate() {
            let abs: Vec<f64> = fd[r * nf..(r + 1) * nf]
                .iter()
                .map(|v| f64::from(v.to_f32()).abs())
                .collect();
            *sum += pairwise_sum(&abs);
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
                "Clef m4a012 down weights need factor scales outside the FP16 range",
            ));
        }

        let half = Operand::Scalar(f16::from_f32(0.5));
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[1, HIDDEN, 1, M4_TILE]);
        let wg = b.constexpr_blockwise_shift_scale(&qg, &[n, HIDDEN, 1, 1], &sg, &[n, 1, 1, 1])?;
        let g = b.conv(&x, &wg)?;
        let wu = b.constexpr_blockwise_shift_scale(&qu, &[n, HIDDEN, 1, 1], &su, &[n, 1, 1, 1])?;
        let u = b.conv(&x, &wu)?;
        // swish(gf)'s `* 0.5` is the program's third mul (after swish(g)'s two).
        let gf_half =
            b.conv_fp16_scaled(&x, &fg, &[nf, HIDDEN, 1, 1], f16::from_f32(0.5), "mul_2")?;
        let uf = b.conv_fp16(&x, &fu, &[nf, HIDDEN, 1, 1])?;
        let swish = |b: &mut ProgramBuilder, half: &Var| -> Result<Var> {
            let t = b.tanh(half);
            let t1 = b.add(&t, Operand::Scalar(f16::ONE), None)?;
            b.mul(half, Operand::Var(&t1), None)
        };
        let g_half = b.mul(&g, half, None)?;
        let s = swish(&mut b, &g_half)?;
        let sf = swish(&mut b, &gf_half)?;
        let peak = |b: &mut ProgramBuilder, v: &Var, w: &Var| -> Result<Var> {
            let va = b.abs(v);
            let vm = b.reduce_max(&va, &[1], true)?;
            let wa = b.abs(w);
            let wm = b.reduce_max(&wa, &[1], true)?;
            b.maximum(&vm, Operand::Var(&wm), None)
        };
        let s_max = peak(&mut b, &s, &sf)?;
        let u_max = peak(&mut b, &u, &uf)?;
        let floor = Operand::Scalar(f16::from_f64(1.0 / FACTOR_SCALE));
        let sm = b.mul(
            &s_max,
            Operand::Scalar(f16::from_f64(1.0 / factor_cap)),
            None,
        )?;
        let s_scale = b.maximum(&sm, floor, Some("gate_scale"))?;
        let um = b.mul(
            &u_max,
            Operand::Scalar(f16::from_f64(1.0 / factor_cap)),
            None,
        )?;
        let u_scale = b.maximum(
            &um,
            Operand::Scalar(f16::from_f64(1.0 / FACTOR_SCALE)),
            Some("up_scale"),
        )?;
        // Normalize before the product so a large SwiGLU value cannot overflow first.
        let normalized = |b: &mut ProgramBuilder, s: &Var, u: &Var| -> Result<Var> {
            let sn = b.real_div(s, Operand::Var(&s_scale), None)?;
            let un = b.real_div(u, Operand::Var(&u_scale), None)?;
            b.mul(&sn, Operand::Var(&un), None)
        };
        let a = normalized(&mut b, &s, &u)?;
        let af = normalized(&mut b, &sf, &uf)?;
        let xs = if parts > 1 {
            b.split(&a, &vec![width; parts], 1)?
        } else {
            vec![a]
        };
        let mut ys = Vec::with_capacity(parts + 1);
        for (xi, (qd, sd)) in xs.iter().zip(&down) {
            let wd = b.constexpr_blockwise_shift_scale(
                qd,
                &[HIDDEN, width, 1, 1],
                sd,
                &[HIDDEN, 1, 1, 1],
            )?;
            ys.push(b.conv(xi, &wd)?);
        }
        ys.push(b.conv_fp16(&af, &fd, &[HIDDEN, nf, 1, 1])?);
        let mut y = ys[0].clone();
        for z in &ys[1..] {
            y = b.add(&y, Operand::Var(z), None)?;
        }
        let y = b.identity(&y, Some("y"));
        b.finish(&[&y, &s_scale, &u_scale])
    }
}

/// The GPU share of `mlp`: gate and up rows and down columns `[0, GPU_COLUMNS)`,
/// as contiguous Q4 copies (EngineTailor `_MLPView`).
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
        Self::load_package(cache, &program(mlp)?)
    }

    pub fn load_package(cache: &ProgramCache, package: &Package) -> Result<Self> {
        Ok(Self {
            model: cache.load(package, ComputeUnits::CpuAndNeuralEngine)?,
        })
    }

    /// m4a012: one Core ML batch prediction over the FP16 channel-first tiles
    /// `[tiles, 4096, M4_TILE]` (evaluated here). Returns the program's `y` as
    /// `[tiles, 4096, M4_TILE]` and both scales as `[tiles * M4_TILE]`, FP16 on
    /// `stream`, as `np.stack` of the per-tile results lays them out.
    pub fn predict_tiles(&self, stream: &Stream, packed: &Array) -> Result<[Array; 3]> {
        let [tiles, HIDDEN, M4_TILE] = *packed.shape() else {
            return Err(Error::Contract(
                "Clef m4a012 tiles must be [tiles, 4096, 128]",
            ));
        };
        // Wait for the tiles only: the GPU share queued after them keeps running.
        let bytes = packed.to_bytes_ready()?;
        // SAFETY: every bit pattern is a valid f16; `align_to` returns only the aligned middle.
        let (head, aligned, tail) = unsafe { bytes.align_to::<f16>() };
        let copied: Vec<f16>;
        let values: &[f16] = if head.is_empty() && tail.is_empty() {
            aligned
        } else {
            copied = bytes
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]))
                .collect();
            &copied
        };
        let batch: Vec<Vec<&[f16]>> = values
            .chunks_exact(HIDDEN * M4_TILE)
            .map(|tile| vec![tile])
            .collect();
        let results = self.model.predict_batch(
            &[("x", &[1, HIDDEN, 1, M4_TILE])],
            &batch,
            &[
                ("y", &[1, HIDDEN, 1, M4_TILE]),
                ("gate_scale", &[1, 1, 1, M4_TILE]),
                ("up_scale", &[1, 1, 1, M4_TILE]),
            ],
        )?;
        let stacked = |k: usize, shape: &[usize]| -> Result<Array> {
            let mut bytes = Vec::with_capacity(shape.iter().product::<usize>() * 2);
            for r in &results {
                bytes.extend_from_slice(f16_bytes(&r[k]));
            }
            Array::from_bytes(stream, shape, MlxDType::F16, &bytes)
        };
        Ok([
            stacked(0, &[tiles, HIDDEN, M4_TILE])?,
            stacked(1, &[tiles * M4_TILE])?,
            stacked(2, &[tiles * M4_TILE])?,
        ])
    }

    /// `_to_channels_first_tiles`: BF16 `[1, T, 4096]` as FP16 channel-first
    /// tiles `[1, 4096, 1, 512]`, zero-padded to whole tiles. Evaluates `h`.
    pub fn tiles(h: &Array) -> Result<Vec<Vec<f16>>> {
        let [1, tokens, channels] = *h.shape() else {
            return Err(Error::Contract("Clef A04 tiles take [1, T, 4096]"));
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
        let values: Vec<f16> = packed
            .to_bytes()?
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]))
            .collect();
        Ok(values
            .chunks_exact(channels * TILE)
            .map(<[f16]>::to_vec)
            .collect())
    }

    /// Predicts every tile and restores BF16 `[1, T, 4096]` on `stream`
    /// (`_from_channels_first_gpu`): rows times both scales in FP32.
    pub fn predict(&self, stream: &Stream, tiles: &[Vec<f16>], tokens: usize) -> Result<Array> {
        let bytes = |v: &[f16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let mut parts = Vec::with_capacity(tiles.len());
        for (i, tile) in tiles.iter().enumerate() {
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
                Array::from_bytes(stream, &[TILE, 1], MlxDType::F16, &bytes(v))?
                    .slice_axis(0, 0, rows)?
                    .cast(MlxDType::F32)
            };
            let y = Array::from_bytes(stream, &[HIDDEN, TILE], MlxDType::F16, &bytes(&out[0]))?
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

    /// Every layer's program from the Q4 checkpoint matches a package that
    /// the EngineTailor engine compiled with coremltools 9.0 (in the
    /// directory `recorded_env` names): the same weight file bytes and the
    /// same decoded program (writer metadata and `buildInfo` excluded).
    fn layer_programs_match_recorded(
        recorded_env: &str,
        frontier: &str,
        build: impl Fn(&Mlp) -> Package,
    ) {
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
        for entry in std::fs::read_dir(var(recorded_env)).unwrap() {
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
            let package = build(&layer.mlp);
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
            "layers matching a recorded {frontier} package"
        );
    }

    /// A-K: the A04 programs (`APXINF_A04_RECORDED`).
    #[test]
    #[ignore = "requires the Clef checkpoint, recorded A04 packages and a Metal device"]
    fn layer_programs_match_the_recorded_a04_packages() {
        layer_programs_match_recorded("APXINF_A04_RECORDED", "A04", |mlp| program(mlp).unwrap());
    }

    /// M-K: the m4a012 programs (`APXINF_M4A012_RECORDED`). A weight file that
    /// matches no package also means the neuron split differs.
    #[test]
    #[ignore = "requires the Clef checkpoint, recorded m4a012 packages and a Metal device"]
    fn layer_programs_match_the_recorded_m4a012_packages() {
        layer_programs_match_recorded("APXINF_M4A012_RECORDED", "m4a012", |mlp| {
            let dense = DenseMlp::new(mlp).unwrap();
            dense.program(&dense.split()).unwrap()
        });
    }

    #[test]
    fn columns_take_a_block_of_every_row() {
        let w: Vec<f16> = (0..12).map(|v| f16::from_f32(v as f32)).collect();
        let c: Vec<f32> = columns(&w, 4, 1, 2).iter().map(|v| v.to_f32()).collect();
        assert_eq!(c, [1., 2., 5., 6., 9., 10.]);
    }
}
