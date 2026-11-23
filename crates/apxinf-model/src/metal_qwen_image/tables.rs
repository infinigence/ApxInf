//! Host-side constant tables that mflux computes with NumPy float32.
//!
//! NumPy (2.x, SIMD builds with FMA) evaluates float32 `sin`/`cos` with its own
//! Cody-Waite reduction and minimax polynomials, which differ from the system
//! libm in the last bit for many inputs. The tables here reproduce those
//! results so the rotary tables and VAE blending ramps match the reference.

const TWO_OVER_PI: u32 = 0x3f22_f983;
const PIO2_HIGH: u32 = 0xbfc9_0fd8;
const PIO2_MED: u32 = 0xb4a8_885a;
const PIO2_LOW: u32 = 0xa7c2_34c4;
const RINT_MAGIC: u32 = 0x4b40_0000;
const COS_POLY: [u32; 5] = [
    0x37cc_730b,
    0xbab6_036e,
    0x3d2a_aa9e,
    0xbf00_0000,
    0x3f80_0000,
];
const SIN_POLY: [u32; 4] = [0x363e_9dde, 0xb950_35dd, 0x3c08_88cd, 0xbe2a_aaab];
const MAX_SIN: u32 = 0x47e5_5dff;
const MAX_COS: u32 = 0x478b_9a08;

fn c(bits: u32) -> f32 {
    f32::from_bits(bits)
}

fn numpy_sincos(x: f32, cosine: bool) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.abs() > c(if cosine { MAX_COS } else { MAX_SIN }) {
        return if cosine { x.cos() } else { x.sin() };
    }
    let quadrant = (x * c(TWO_OVER_PI) + c(RINT_MAGIC)) - c(RINT_MAGIC);
    let mut r = quadrant.mul_add(c(PIO2_HIGH), x);
    r = quadrant.mul_add(c(PIO2_MED), r);
    r = quadrant.mul_add(c(PIO2_LOW), r);
    let r2 = r * r;
    let mut cos = c(COS_POLY[0]).mul_add(r2, c(COS_POLY[1]));
    for &k in &COS_POLY[2..] {
        cos = cos.mul_add(r2, c(k));
    }
    let mut sin = c(SIN_POLY[0]).mul_add(r2, c(SIN_POLY[1]));
    for &k in &SIN_POLY[2..] {
        sin = sin.mul_add(r2, c(k));
    }
    sin = sin.mul_add(r2, 0.0).mul_add(r, r);
    let quadrant = quadrant as i32 + i32::from(cosine);
    let value = if quadrant & 1 == 0 { sin } else { cos };
    if quadrant & 2 == 2 {
        0.0 - value
    } else {
        value
    }
}

/// `np.cos` on a float32 scalar.
pub fn numpy_cos(x: f32) -> f32 {
    numpy_sincos(x, true)
}

/// `np.sin` on a float32 scalar.
pub fn numpy_sin(x: f32) -> f32 {
    numpy_sincos(x, false)
}

/// Row index of `Qwen21Rope` tables: positions 0..8191, then -1024..-1.
pub fn rope_row(position: i64) -> Option<usize> {
    match position {
        0..=8191 => Some(position as usize),
        -1024..=-1 => Some((8192 + 1024 + position) as usize),
        _ => None,
    }
}

/// `Qwen21Rope._rope_params` for one axis: `[9216, dim / 2]` cos and sin.
pub fn rope_axis(dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let omega: Vec<f32> = (0..dim / 2)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
        .collect();
    let rows = 8192 + 1024;
    let mut cos = Vec::with_capacity(rows * omega.len());
    let mut sin = Vec::with_capacity(rows * omega.len());
    for row in 0..rows {
        let position = if row < 8192 {
            row as i64
        } else {
            row as i64 - 9216
        } as f32;
        for &w in &omega {
            let angle = position * w;
            cos.push(numpy_cos(angle));
            sin.push(numpy_sin(angle));
        }
    }
    (cos, sin)
}

/// `Qwen21Timesteps._freqs`: `exp(-ln(10000) * i / half)` in float64, as float32.
pub fn timestep_frequencies(half: usize) -> Vec<f32> {
    (0..half)
        .map(|i| (-(10000f64.ln()) * (i as f32) as f64 / half as f64).exp() as f32)
        .collect()
}

/// `VAETiler._cos_ramp(n)`: `0.5 - 0.5 * cos(linspace(0, 1, n, float32) * pi)`.
pub fn cos_ramp(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let step = if n > 1 { 1.0 / (n - 1) as f64 } else { 0.0 };
    (0..n)
        .map(|i| {
            let t = (i as f64 * step) as f32;
            0.5 - 0.5 * numpy_cos(t * std::f32::consts::PI)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn digest(values: &[f32]) -> String {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        format!("{:x}", Sha256::digest(bytes))
    }

    // Recorded from NumPy 2.5.4 (ASIMD build) on Apple M4 with mflux's formulas.
    #[test]
    fn tables_match_recorded_numpy_float32_digests() {
        let (cos16, sin16) = rope_axis(16, 10000.0);
        assert_eq!(
            digest(&cos16),
            "18e9f4489e5ed08cba06d55bd7e8d63447291e14883e860c0497e2529894d53c"
        );
        assert_eq!(
            digest(&sin16),
            "aabed13cadfb7e4c4d9ab89f69bf2a413b1bd63bfc85312677b1193540089dd0"
        );
        let (cos56, sin56) = rope_axis(56, 10000.0);
        assert_eq!(
            digest(&cos56),
            "49bd29327ada0bd4033b6eab30df7aeb20c0f9e75330d25d812e2a0405dc64bb"
        );
        assert_eq!(
            digest(&sin56),
            "bb1cb9efb4725e209cdbe43290aecfc40a5d0eaa0101b3045968b98722c0bf2e"
        );
        assert_eq!(
            digest(&cos_ramp(128)),
            "7a3c85956e2ca60e6bbb3b1e105be3f5d2e7d13ea13a32eadf4a2f4c0d325b8c"
        );
        assert_eq!(
            digest(&timestep_frequencies(128)),
            "77c4766be1f3df24192a4756598ca5d9be61c1963cfa71a01237c50ff6f7950f"
        );
    }

    #[test]
    fn spot_values_cover_signs_quadrants_and_the_libm_fallback() {
        let x = [
            0.0f32, -0.0, 1.0, -2.5, 3.14159265, 1e5, 2e5, -1e6, 71476.0625, 71476.07,
        ];
        let cos: Vec<u32> = x.iter().map(|&v| numpy_cos(v).to_bits()).collect();
        let sin: Vec<u32> = x.iter().map(|&v| numpy_sin(v).to_bits()).collect();
        assert_eq!(
            cos,
            [
                0x3f800000, 0x3f800000, 0x3f0a5140, 0xbf4d17bf, 0xbf800000, 0xbf7fd61c, 0x3f7f587e,
                0x3f6fcefd, 0x3def8fc6, 0x3dff71be
            ]
        );
        assert_eq!(
            sin,
            [
                0x0, 0x80000000, 0x3f576aa5, 0xbf193579, 0xb3bbbd2e, 0x3d126d55, 0xbd92555f,
                0x3eb33259, 0xbf7e3e19, 0xbf7e0039
            ]
        );
        assert!(numpy_cos(f32::NAN).is_nan());
        assert_eq!(rope_row(0), Some(0));
        assert_eq!(rope_row(-1), Some(9215));
        assert_eq!(rope_row(-1024), Some(8192));
        assert_eq!(rope_row(8192), None);
    }
}
