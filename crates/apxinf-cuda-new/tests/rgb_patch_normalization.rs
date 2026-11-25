//! RGB u8 -> patch normalization recipes of the Gather family, checked against independent CPU
//! references over every byte value, all three channels, two views, both source layouts and every
//! supported output dtype. The references re-state each recipe's arithmetic operation by operation
//! (IEEE FP32 for each step, one rounding to the output dtype); they share no code with the kernel.
use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::*;
use half::{bf16, f16};

const VIEWS: usize = 2;
const SIZE: usize = 16;
const PATCH: usize = 2;
const PER_SIDE: usize = SIZE / PATCH;
const ROWS: usize = VIEWS * PER_SIDE * PER_SIDE;
const COLS: usize = 3 * PATCH * PATCH;

/// Byte of pixel (view, y, x) in channel c, rotated by `shift`. With shift 0 every (view, channel)
/// plane holds each of the 256 byte values exactly once, at positions that differ between planes.
fn byte_at(view: usize, y: usize, x: usize, c: usize, shift: u8) -> u8 {
    (((y * SIZE + x + 37 * c + 101 * view) % 256) as u8).wrapping_add(shift)
}

fn image(nhwc: bool, shift: u8) -> Vec<u8> {
    let mut out = vec![0u8; VIEWS * SIZE * SIZE * 3];
    for v in 0..VIEWS {
        for y in 0..SIZE {
            for x in 0..SIZE {
                for c in 0..3 {
                    let i = if nhwc {
                        ((v * SIZE + y) * SIZE + x) * 3 + c
                    } else {
                        ((v * 3 + c) * SIZE + y) * SIZE + x
                    };
                    out[i] = byte_at(v, y, x, c, shift);
                }
            }
        }
    }
    out
}

#[test]
fn the_test_image_covers_every_byte_in_every_plane() {
    for shift in [0u8, 1, 200] {
        for v in 0..VIEWS {
            for c in 0..3 {
                let mut seen = [false; 256];
                for y in 0..SIZE {
                    for x in 0..SIZE {
                        seen[byte_at(v, y, x, c, shift) as usize] = true;
                    }
                }
                assert!(seen.iter().all(|&s| s), "view {v} channel {c} shift {shift} misses a byte value");
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Recipe {
    /// `RgbToPatches`: `(float(b) / 255) * 2 - 1`, true FP32 division.
    UnitRange,
    /// `RgbToPatchesStaged`: cast to the output dtype, divide by 127.5, subtract 1, rounding after each step.
    Staged,
    /// `RgbToPatchesReciprocal`: `((float(b) * float(1/255)) * 2) - 1`, FP32 steps, one output rounding.
    Reciprocal,
}

fn round_to(dtype: DType, v: f32) -> f32 {
    match dtype {
        DType::F32 => v,
        DType::BF16 => bf16::from_f32(v).to_f32(),
        DType::F16 => f16::from_f32(v).to_f32(),
        _ => unreachable!(),
    }
}

fn normalize(recipe: Recipe, dtype: DType, b: u8) -> f32 {
    match recipe {
        Recipe::UnitRange => round_to(dtype, (b as f32 / 255.0f32) * 2.0f32 - 1.0f32),
        Recipe::Staged => {
            let t = round_to(dtype, b as f32);
            let q = round_to(dtype, t / 127.5f32);
            round_to(dtype, q - 1.0f32)
        }
        Recipe::Reciprocal => {
            let reciprocal = (1.0f64 / 255.0f64) as f32;
            round_to(dtype, ((b as f32 * reciprocal) * 2.0f32) - 1.0f32)
        }
    }
}

/// Expected `[ROWS, COLS]` patches: rows are (view, patch_y, patch_x), columns are (channel, dy, dx).
fn expected(recipe: Recipe, dtype: DType, shift: u8) -> Vec<f32> {
    let mut out = vec![0f32; ROWS * COLS];
    for v in 0..VIEWS {
        for py in 0..PER_SIDE {
            for px in 0..PER_SIDE {
                let row = (v * PER_SIDE + py) * PER_SIDE + px;
                for c in 0..3 {
                    for dy in 0..PATCH {
                        for dx in 0..PATCH {
                            let b = byte_at(v, py * PATCH + dy, px * PATCH + dx, c, shift);
                            out[row * COLS + (c * PATCH + dy) * PATCH + dx] = normalize(recipe, dtype, b);
                        }
                    }
                }
            }
        }
    }
    out
}

fn output(dtype: DType, rows: usize) -> Result<Tensor, String> {
    let buffer = CudaBuffer::alloc_zeros(rows * COLS * dtype.size_in_bytes(), 0)?;
    buffer.as_tensor(Shape::new(vec![rows, COLS]), dtype)
}

fn run_gather(ctx: &CudaContext, semantic: GatherSemantic, dtype: DType, nhwc: bool) -> Result<Vec<f32>, String> {
    let bytes = image(nhwc, 0);
    let input = CudaBuffer::alloc(bytes.len(), 0)?;
    input.copy_from_host(&bytes)?;
    let mut out = output(dtype, ROWS)?;
    let mut args = GatherArgs::rgb_to_patches(
        &input,
        &mut out,
        GatherPatchGeometry { views: VIEWS, image_size: SIZE, patch_size: PATCH, nhwc },
    );
    args.semantic = semantic;
    gather(ctx, args).map_err(|e| e.to_string())?;
    ctx.synchronize()?;
    Ok(read(&out))
}

fn read(t: &Tensor) -> Vec<f32> {
    let mut b = vec![0u8; t.size_in_bytes()];
    CudaBuffer::from_tensor(t).unwrap().copy_to_host(&mut b).unwrap();
    match t.dtype() {
        DType::F32 => b.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect(),
        DType::BF16 => b.chunks_exact(2).map(|c| bf16::from_bits(u16::from_ne_bytes(c.try_into().unwrap())).to_f32()).collect(),
        DType::F16 => b.chunks_exact(2).map(|c| f16::from_bits(u16::from_ne_bytes(c.try_into().unwrap())).to_f32()).collect(),
        _ => unreachable!(),
    }
}

fn assert_bits_equal(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i].to_bits() != want[i].to_bits()).collect();
    assert!(bad.is_empty(), "{what}: {} of {} elements differ, first at {} (got {}, want {})", bad.len(), got.len(), bad[0], got[bad[0]], want[bad[0]]);
}

// ---------------------------------------------------------------- existing recipes (must not change)

#[test]
fn unit_range_recipe_matches_the_cpu_reference() {
    let ctx = CudaContext::new(0).unwrap();
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        for nhwc in [true, false] {
            let got = run_gather(&ctx, GatherSemantic::RgbToPatches, dtype, nhwc).unwrap();
            assert_bits_equal(&got, &expected(Recipe::UnitRange, dtype, 0), &format!("RgbToPatches {dtype:?} nhwc={nhwc}"));
        }
    }
}

#[test]
fn staged_recipe_matches_the_cpu_reference() {
    let ctx = CudaContext::new(0).unwrap();
    for dtype in [DType::BF16, DType::F16] {
        for nhwc in [true, false] {
            let got = run_gather(&ctx, GatherSemantic::RgbToPatchesStaged, dtype, nhwc).unwrap();
            assert_bits_equal(&got, &expected(Recipe::Staged, dtype, 0), &format!("RgbToPatchesStaged {dtype:?} nhwc={nhwc}"));
        }
    }
    // The staged recipe stays unavailable for F32 output.
    assert!(run_gather(&ctx, GatherSemantic::RgbToPatchesStaged, DType::F32, true).is_err());
}

// ---------------------------------------------------------------- reciprocal recipe

#[test]
fn reciprocal_recipe_matches_the_cpu_reference() {
    let ctx = CudaContext::new(0).unwrap();
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        for nhwc in [true, false] {
            let got = run_gather(&ctx, GatherSemantic::RgbToPatchesReciprocal, dtype, nhwc).unwrap();
            assert_bits_equal(&got, &expected(Recipe::Reciprocal, dtype, 0), &format!("RgbToPatchesReciprocal {dtype:?} nhwc={nhwc}"));
        }
    }
}

#[test]
fn reciprocal_is_a_distinct_recipe_within_one_output_step_of_the_plain_one() {
    let ctx = CudaContext::new(0).unwrap();
    let plain = run_gather(&ctx, GatherSemantic::RgbToPatches, DType::BF16, true).unwrap();
    let recip = run_gather(&ctx, GatherSemantic::RgbToPatchesReciprocal, DType::BF16, true).unwrap();
    let differing: Vec<usize> = (0..plain.len()).filter(|&i| plain[i].to_bits() != recip[i].to_bits()).collect();
    assert!(!differing.is_empty(), "the reciprocal recipe must not alias the plain recipe");
    for &i in &differing {
        let (a, b) = (bf16::from_f32(plain[i]).to_bits() as i32, bf16::from_f32(recip[i]).to_bits() as i32);
        assert_eq!((a & 0x8000), (b & 0x8000), "sign differs at {i}");
        assert_eq!((a - b).abs(), 1, "element {i}: {} vs {} is not one BF16 step", plain[i], recip[i]);
    }
}

#[test]
fn reciprocal_is_replayed_from_a_captured_graph_with_updated_bytes() {
    let ctx = CudaContext::new(0).unwrap();
    let bytes = image(true, 0);
    let input = CudaBuffer::alloc(bytes.len(), 0).unwrap();
    input.copy_from_host(&bytes).unwrap();
    let out = output(DType::BF16, ROWS).unwrap();
    let graph = capture(&ctx, || {
        let mut target = out.clone();
        let mut args = GatherArgs::rgb_to_patches(
            &input,
            &mut target,
            GatherPatchGeometry { views: VIEWS, image_size: SIZE, patch_size: PATCH, nhwc: true },
        );
        args.semantic = GatherSemantic::RgbToPatchesReciprocal;
        gather(&ctx, args)
    })
    .unwrap();
    for shift in [0u8, 1, 127, 200] {
        input.copy_from_host(&image(true, shift)).unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_bits_equal(&read(&out), &expected(Recipe::Reciprocal, DType::BF16, shift), &format!("replay with byte shift {shift}"));
    }
}

#[test]
fn reciprocal_rejects_bad_dtype_shape_and_geometry() {
    let ctx = CudaContext::new(0).unwrap();
    let bytes = image(true, 0);
    let input = CudaBuffer::alloc(bytes.len(), 0).unwrap();
    input.copy_from_host(&bytes).unwrap();
    let geometry = GatherPatchGeometry { views: VIEWS, image_size: SIZE, patch_size: PATCH, nhwc: true };
    // integer output is not a normalized-pixel dtype
    let mut int_out = output(DType::I32, ROWS).unwrap();
    let mut args = GatherArgs::rgb_to_patches(&input, &mut int_out, geometry);
    args.semantic = GatherSemantic::RgbToPatchesReciprocal;
    assert!(gather(&ctx, args).is_err());
    // output shape disagreeing with the geometry
    let mut short = output(DType::BF16, ROWS - 1).unwrap();
    let mut args = GatherArgs::rgb_to_patches(&input, &mut short, geometry);
    args.semantic = GatherSemantic::RgbToPatchesReciprocal;
    assert!(gather(&ctx, args).is_err());
    // image size not divisible by the patch size
    let mut out = output(DType::BF16, ROWS).unwrap();
    let mut args = GatherArgs::rgb_to_patches(&input, &mut out, GatherPatchGeometry { patch_size: 3, ..geometry });
    args.semantic = GatherSemantic::RgbToPatchesReciprocal;
    assert!(gather(&ctx, args).is_err());
}
