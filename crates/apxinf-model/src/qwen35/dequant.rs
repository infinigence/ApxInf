//! Dequantization for compressed-tensors `pack-quantized` W4A16 weights.
//!
//! Layout (verified against compressed_tensors 0.11.0):
//! * `weight_packed`: int32 `[out, ceil(in/8)]`; eight 4-bit nibbles per
//!   int32, the 0th nibble in the least-significant 4 bits.
//! * `weight_scale`: bf16 `[out, in/group]` (`group = 32`).
//! * `weight_zero_point`: int32 `[ceil(out/8), in/group]`, nibbles packed
//!   along the **row** (out) dimension, same nibble order.
//!
//! Dequant: `W[o,i] = scale[o, g] * (w4[o,i] - zp[o,g])`, `g = i / 32`,
//! where `w4` and `zp` are signed 4-bit values in `[-8, 7]`.

use half::bf16;

pub const GROUP_SIZE: usize = 32;

#[inline]
fn nibble(packed: i32, idx: usize) -> i32 {
    ((packed >> (4 * (idx & 7))) & 0xF) - 8
}

/// Unpack one int32 nibble lane. Kept public for tests.
pub fn unpack_int4_lane(value: i32) -> [i8; 8] {
    let mut out = [0i8; 8];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = nibble(value, i) as i8;
    }
    out
}

/// Pack eight signed int4 nibbles into one int32 (LSB first). Tests use this
/// to build expected inputs.
pub fn pack_int4_lane(vals: &[i8]) -> i32 {
    debug_assert!(vals.len() == 8);
    vals.iter().enumerate().fold(0i32, |acc, (i, &v)| {
        acc | (((v as i32 + 8) & 0xF) << (4 * i))
    })
}

/// Dequantize one packed linear weight into bf16 `dst[out * inp]`.
pub fn dequantize_w4a16(
    out: usize,
    inp: usize,
    packed: &[i32],
    scale: &[bf16],
    zp_packed: &[i32],
    dst: &mut Vec<bf16>,
) -> Result<(), String> {
    if inp % GROUP_SIZE != 0 {
        return Err(format!("input dim {inp} not divisible by group size {GROUP_SIZE}"));
    }
    let groups = inp / GROUP_SIZE;
    let packed_cols = inp.div_ceil(8);
    let zp_rows = out.div_ceil(8);
    if packed.len() < out * packed_cols {
        return Err("weight_packed too small".to_string());
    }
    if scale.len() < out * groups {
        return Err("weight_scale too small".to_string());
    }
    if zp_packed.len() < zp_rows * groups {
        return Err("weight_zero_point too small".to_string());
    }

    dst.clear();
    dst.reserve(out * inp);
    for o in 0..out {
        let zp_lane = zp_packed[(o / 8) * groups..(o / 8) * groups + groups].to_vec();
        for g in 0..groups {
            let zp = nibble(zp_lane[g], o % 8) as i32;
            let s = scale[o * groups + g].to_f32();
            for j in 0..GROUP_SIZE {
                let i = g * GROUP_SIZE + j;
                let w4 = nibble(packed[o * packed_cols + i / 8], i % 8) as i32;
                dst.push(bf16::from_f32(s * (w4 - zp) as f32));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_matches_compressed_tensors_bit_order() {
        // 0x12345678: nibbles (LSB first) are 8,7,6,5,4,3,2,1.
        // After the -8 offset: 0,-1,-2,-3,-4,-5,-6,-7.
        assert_eq!(
            unpack_int4_lane(0x1234_5678),
            [0, -1, -2, -3, -4, -5, -6, -7]
        );
        assert_eq!(pack_int4_lane(&[0, -1, -2, -3, -4, -5, -6, -7]), 0x1234_5678);
    }

    #[test]
    fn dequant_single_group() {
        // out=8, in=32, single group. zp nibbles = [0,-1,-2,-3,-4,-5,-6,-7].
        let zp_packed = [0x1234_5678i32];
        // Make every weight nibble 7 (+1 relative to zp of row o).
        // W[o,i] = scale*(7 - zp_o). scale = 2.0.
        let mut packed = vec![0i32; 8 * 4]; // out=8, in/8=4
        for lane in packed.iter_mut() {
            *lane = 0x7777_7777;
        }
        let scale = vec![bf16::from_f32(2.0); 8];
        let mut dst = Vec::new();
        dequantize_w4a16(8, 32, &packed, &scale, &zp_packed, &mut dst).unwrap();
        for o in 0..8usize {
            // zp nibbles are [0,-1,-2,-3,-4,-5,-6,-7]; w4 nibbles are all 7 => -1.
            let zp = (o as i32) * -1;
            let expect = 2.0f32 * (-1.0f32 - zp as f32);
            for i in 0..32usize {
                let got = dst[o * 32 + i].to_f32();
                assert!((got - expect).abs() < 0.11, "o={o} got {got} expect {expect}");
            }
        }
    }
}
