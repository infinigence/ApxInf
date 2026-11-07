#![cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]

use apxinf_core::Result;
use apxinf_mlx::fusions::Bf16QkTableRope;
use apxinf_mlx::{Array, Compiled, MlxDType as D, Stream};
use half::bf16;

const POSITIONS: [usize; 4] = [0, 1024, 16384, 40959];

#[derive(Clone, Copy, Debug)]
enum Values {
    Normal,
    Large,
    SignedZero,
}

fn upload(s: &Stream, shape: &[usize], values: &[bf16]) -> Result<Array> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|x| x.to_bits().to_ne_bytes())
        .collect();
    Array::from_bytes(s, shape, D::BF16, &bytes)
}

fn bits(x: &Array) -> Result<Vec<u16>> {
    assert_eq!(x.dtype(), D::BF16);
    Ok(x.to_bytes()?
        .chunks_exact(2)
        .map(|b| u16::from_ne_bytes([b[0], b[1]]))
        .collect())
}

fn input(
    s: &Stream,
    heads: usize,
    dim: usize,
    values: Values,
    strided: bool,
    seed: usize,
) -> Result<Array> {
    let values: Vec<bf16> = (0..heads * dim)
        .map(|i| match values {
            Values::SignedZero => bf16::from_bits(if (i + seed) % 3 == 0 { 0x8000 } else { 0 }),
            Values::Normal | Values::Large => {
                let scale = if matches!(values, Values::Large) {
                    1048576.
                } else {
                    1.
                };
                bf16::from_f32((((i * 37 + seed) % 251) as f32 - 125.) / 19. * scale)
            }
        })
        .collect();
    if strided {
        upload(s, &[1, dim, 1, heads], &values)?.transpose(&[0, 3, 2, 1])
    } else {
        upload(s, &[1, 1, heads, dim], &values)?.transpose(&[0, 2, 1, 3])
    }
}

fn tables(s: &Stream, dim: usize, unequal_halves: bool) -> Result<[Array; 2]> {
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for position in POSITIONS {
        for column in 0..dim {
            let exponent = (2 * (column % (dim / 2))) as f32 / dim as f32;
            let angle = position as f32 / 1_000_000_f32.powf(exponent);
            let (mut sn, mut cs) = angle.sin_cos();
            if unequal_halves && column >= dim / 2 {
                cs = cs * 0.625 + 0.125;
                sn = sn * -0.75 - 0.25;
            }
            cos.push(bf16::from_f32(cs));
            sin.push(bf16::from_f32(sn));
        }
    }
    Ok([
        upload(s, &[1, 1, POSITIONS.len(), dim], &cos)?,
        upload(s, &[1, 1, POSITIONS.len(), dim], &sin)?,
    ])
}

fn reference(x: &Array, cos: &Array, sin: &Array) -> Result<Vec<u16>> {
    let dim = x.shape()[3];
    let x = bits(x)?;
    let cos = bits(cos)?;
    let sin = bits(sin)?;
    Ok(x.iter()
        .enumerate()
        .map(|(i, &value)| {
            let column = i % dim;
            let partner = if column < dim / 2 {
                i + dim / 2
            } else {
                i - dim / 2
            };
            let rotated = x[partner] ^ if column < dim / 2 { 0x8000 } else { 0 };
            let left = bf16::from_f32(
                bf16::from_bits(value).to_f32() * bf16::from_bits(cos[column]).to_f32(),
            );
            let right = bf16::from_f32(
                bf16::from_bits(rotated).to_f32() * bf16::from_bits(sin[column]).to_f32(),
            );
            bf16::from_f32(left.to_f32() + right.to_f32()).to_bits()
        })
        .collect())
}

fn ordered(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let dim = x.shape()[3];
    let half = Array::concat(
        &[
            &x.slice_axis(3, dim / 2, dim)?.neg()?,
            &x.slice_axis(3, 0, dim / 2)?,
        ],
        3,
    )?;
    x.mul(cos)?.add(&half.mul(sin)?)
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn table_rope_matches_bf16_bits_for_tiny_and_checkpoint_views() -> Result<()> {
    let s = Stream::metal(0)?;
    let kernel = Bf16QkTableRope::new(&s)?;
    let composed = Compiled::new(&s, 2, |a| {
        Ok(vec![
            ordered(&a[0], &a[2], &a[3])?,
            ordered(&a[1], &a[2], &a[3])?,
        ])
    })?;
    for (q_heads, k_heads, dim) in [(2, 1, 8), (16, 8, 128)] {
        for values in [Values::Normal, Values::Large, Values::SignedZero] {
            for strided in [false, true] {
                let q = input(&s, q_heads, dim, values, strided, 11)?;
                let k = input(&s, k_heads, dim, values, strided, 23)?;
                if matches!(values, Values::SignedZero) {
                    for x in [&q, &k] {
                        let observed = bits(x)?;
                        assert!(observed.contains(&0) && observed.contains(&0x8000));
                    }
                }
                for unequal_halves in [false, true] {
                    let [cos_table, sin_table] = tables(&s, dim, unequal_halves)?;
                    for (row, position) in POSITIONS.into_iter().enumerate() {
                        let cos = cos_table.slice_axis(2, row, row + 1)?;
                        let sin = sin_table.slice_axis(2, row, row + 1)?;
                        let args = [q.clone(), k.clone(), cos, sin];
                        let before = args.iter().map(bits).collect::<Result<Vec<_>>>()?;
                        let expected = [
                            reference(&q, &args[2], &args[3])?,
                            reference(&k, &args[2], &args[3])?,
                        ];
                        let composed = composed.call_and_eval(&args)?;
                        let (qo, ko) = kernel.call(&q, &k, &args[2], &args[3])?;
                        for (part, output) in [qo, ko].iter().enumerate() {
                            assert_eq!(
                                (output.shape(), output.dtype()),
                                (args[part].shape(), D::BF16)
                            );
                            assert_eq!(bits(output)?, expected[part], "part {part}, D{dim}, {values:?}, strided {strided}, unequal {unequal_halves}, position {position}");
                            assert_eq!(bits(&composed[part])?, expected[part], "ordered reference part {part}, D{dim}, {values:?}, position {position}");
                        }
                        assert_eq!(args.iter().map(bits).collect::<Result<Vec<_>>>()?, before);
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn table_rope_compiled_replay_uses_changed_operands_and_retains_outputs() -> Result<()> {
    let s = Stream::metal(0)?;
    for (q_heads, k_heads, dim) in [(2, 1, 8), (16, 8, 128)] {
        let kernel = Bf16QkTableRope::new(&s)?;
        let compiled = Compiled::new(&s, 2, move |a| {
            let (q, k) = kernel.call(&a[0], &a[1], &a[2], &a[3])?;
            Ok(vec![q, k])
        })?;
        let [cos, sin] = tables(&s, dim, true)?;
        let mut args = [
            input(&s, q_heads, dim, Values::Normal, true, 11)?,
            input(&s, k_heads, dim, Values::Normal, true, 23)?,
            cos.slice_axis(2, 0, 1)?,
            sin.slice_axis(2, 0, 1)?,
        ];
        let first = compiled.prepare(&args)?;
        let first_bits = first.iter().map(bits).collect::<Result<Vec<_>>>()?;
        let traces = s.counters().trace_callbacks;
        let replacements = [
            input(&s, q_heads, dim, Values::Normal, true, 47)?,
            input(&s, k_heads, dim, Values::Normal, true, 61)?,
            cos.slice_axis(2, 1, 2)?,
            sin.slice_axis(2, 2, 3)?,
        ];
        let mut previous = first_bits.clone();
        for (operand, replacement) in replacements.into_iter().enumerate() {
            args[operand] = replacement;
            let outputs = compiled.call_and_eval(&args)?;
            assert_eq!(s.counters().trace_callbacks, traces);
            let actual = outputs.iter().map(bits).collect::<Result<Vec<_>>>()?;
            let expected = vec![
                reference(&args[0], &args[2], &args[3])?,
                reference(&args[1], &args[2], &args[3])?,
            ];
            assert_eq!(actual, expected, "changed operand {operand}, D{dim}");
            assert_ne!(actual, previous, "operand {operand} must affect a result");
            assert_eq!(
                first.iter().map(bits).collect::<Result<Vec<_>>>()?,
                first_bits
            );
            previous = actual;
        }
        let pending = compiled.call(&args)?;
        assert_eq!(s.counters().trace_callbacks, traces);
        drop(compiled);
        assert_eq!(
            pending.iter().map(bits).collect::<Result<Vec<_>>>()?,
            previous
        );
        assert_eq!(
            first.iter().map(bits).collect::<Result<Vec<_>>>()?,
            first_bits
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn table_rope_rejects_unsupported_shapes_dtypes_and_extent() -> Result<()> {
    let s = Stream::metal(0)?;
    let kernel = Bf16QkTableRope::new(&s)?;
    let args = [
        Array::zeros(&s, &[1, 2, 1, 8], D::BF16)?,
        Array::zeros(&s, &[1, 1, 1, 8], D::BF16)?,
        Array::zeros(&s, &[1, 1, 1, 8], D::BF16)?,
        Array::zeros(&s, &[1, 1, 1, 8], D::BF16)?,
    ];
    for operand in 0..4 {
        let mut invalid = args.clone();
        invalid[operand] = invalid[operand].cast(D::F32)?;
        assert!(kernel
            .call(&invalid[0], &invalid[1], &invalid[2], &invalid[3])
            .is_err());
    }
    for (operand, shape) in [
        (0, vec![2, 8]),
        (0, vec![2, 2, 1, 8]),
        (0, vec![1, 2, 2, 8]),
        (0, vec![1, 0, 1, 8]),
        (1, vec![1, 0, 1, 8]),
        (1, vec![1, 1, 1, 6]),
        (1, vec![1, 1, 2, 8]),
        (1, vec![8]),
        (2, vec![8]),
        (3, vec![1, 1, 2, 8]),
    ] {
        let mut invalid = args.clone();
        invalid[operand] = Array::zeros(&s, &shape, D::BF16)?;
        assert!(
            kernel
                .call(&invalid[0], &invalid[1], &invalid[2], &invalid[3])
                .is_err(),
            "operand {operand}, shape {shape:?}"
        );
    }
    for dim in [0, 7] {
        let q = Array::zeros(&s, &[1, 2, 1, dim], D::BF16)?;
        let k = Array::zeros(&s, &[1, 1, 1, dim], D::BF16)?;
        assert!(kernel.call(&q, &k, &k, &k).is_err());
    }
    let scalar = Array::scalar(&s, 0., D::BF16)?;
    let huge = scalar.broadcast_to(&[1, i32::MAX as usize, 1, 2])?;
    let small = scalar.broadcast_to(&[1, 1, 1, 2])?;
    assert!(kernel.call(&huge, &small, &small, &small).is_err());
    Ok(())
}
