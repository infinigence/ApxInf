//! NVFP4 block-scaled GEMM: independent-reference test for every candidate.
//!
//! The reference is computed on the host in f64 from the same FP4 code points
//! and E4M3 block scales the device reads, so it validates the whole chain --
//! packing, the load-time scale relayout, candidate selection and the kernel --
//! rather than any one link.

use super::framework::{bytes_tensor, tensor, values, zeros_tensor};
use super::*;
use crate::CudaContext;
use apxinf_core::{DType, Tensor};

fn validate_all_candidates<'a>(
    ctx: &CudaContext,
    args: GemmArgs<'a>,
    bias: Option<&'a Tensor>,
    expected: &'a [f32],
) -> apxinf_core::Result<()> {
    let normalized =
        super::contracts::normalize(ctx, args, super::contracts::Semantic::Gemm, bias)?;
    super::execution::validate_candidates(ctx, &normalized, expected)
}

/// e2m1 magnitudes indexed by the low three bits; bit 3 carries the sign.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e2m1_code(index: usize, salt: usize) -> u8 {
    let magnitude = ((index * salt + 1) % 8) as u8;
    let negative = index % 3 == 0;
    if negative { magnitude | 0x8 } else { magnitude }
}

fn e2m1_value(code: u8) -> f64 {
    let magnitude = E2M1[(code & 0x7) as usize] as f64;
    if code & 0x8 != 0 { -magnitude } else { magnitude }
}

/// Pack one row-major FP4 operand, two values per byte, low nibble first.
fn pack_operand(rows: usize, k: usize, salt: usize) -> (Vec<u8>, Vec<f64>) {
    let mut packed = vec![0u8; rows * k / 2];
    let mut values = vec![0.0; rows * k];
    for row in 0..rows {
        for column in 0..k {
            let flat = row * k + column;
            let code = e2m1_code(flat, salt);
            values[flat] = e2m1_value(code);
            let byte = row * (k / 2) + column / 2;
            if column % 2 == 0 {
                packed[byte] = code;
            } else {
                packed[byte] |= code << 4;
            }
        }
    }
    (packed, values)
}

/// Encode a positive float as unsigned E4M3 (bias 7). For the non-negative
/// values a scale holds this is bit-identical to signed E4M3, which is why the
/// checkpoint's `F8_E4M3` scale bytes can be fed to the kernel unchanged.
fn ue4m3_code(value: f32) -> u8 {
    if !(value > 0.0) {
        return 0;
    }
    let exponent = value.log2().floor();
    let mantissa = value / exponent.exp2();
    let mut biased = exponent as i32 + 7;
    let mut fraction = ((mantissa - 1.0) * 8.0).round() as i32;
    if fraction > 7 {
        fraction = 0;
        biased += 1;
    }
    let biased = biased.clamp(0, 15);
    ((biased as u8) << 3) | (fraction.clamp(0, 7) as u8)
}

fn ue4m3_value(code: u8) -> f64 {
    let exponent = ((code >> 3) & 0xF) as i32;
    let fraction = (code & 0x7) as f64;
    if exponent == 0 {
        return 0.0;
    }
    (1.0 + fraction / 8.0) * (exponent as f64 - 7.0).exp2()
}

fn block_scales(rows: usize, blocks: usize, salt: usize) -> (Vec<u8>, Vec<f64>) {
    let mut codes = vec![0u8; rows * blocks];
    let mut values = vec![0.0; rows * blocks];
    for index in 0..rows * blocks {
        let magnitude = 0.5 + 0.25 * ((index * salt) % 5) as f32;
        let code = ue4m3_code(magnitude);
        codes[index] = code;
        values[index] = ue4m3_value(code);
    }
    (codes, values)
}

/// Move checkpoint-order scales into the layout the kernel reads.
fn packed_scale_tensor(
    ctx: &CudaContext,
    codes: &[u8],
    rows: usize,
    k: usize,
    block_size: u32,
) -> apxinf_core::Result<Tensor> {
    let device = ctx.device_id();
    let blocks = k / block_size as usize;
    let source = bytes_tensor(device, vec![rows, blocks], DType::F8E4M3, codes);
    let bytes = crate::ops::nvfp4_scale_buffer_bytes(rows, k, block_size)?;
    let destination = zeros_tensor(device, vec![bytes], DType::F8E4M3);
    crate::ops::nvfp4_pack_block_scales(ctx, &source, &destination, rows, k, block_size)?;
    ctx.synchronize().map_err(apxinf_core::Error::Cuda)?;
    Ok(destination)
}

#[test]
fn nvfp4_all_candidates_match_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let device = ctx.device_id();
    let (m, n, k, block_size) = (256usize, 256usize, 512usize, 16u32);
    let blocks = k / block_size as usize;

    let (a_packed, a_values) = pack_operand(m, k, 7);
    let (b_packed, b_values) = pack_operand(n, k, 5);
    let (a_scale_codes, a_scale_values) = block_scales(m, blocks, 3);
    let (b_scale_codes, b_scale_values) = block_scales(n, blocks, 2);

    // A is [M, K/2]; B keeps the checkpoint's own [N, K/2] orientation.
    let a = bytes_tensor(device, vec![m, k / 2], DType::E2M1Pair, &a_packed);
    let b = bytes_tensor(device, vec![n, k / 2], DType::E2M1Pair, &b_packed);
    let a_scales = packed_scale_tensor(&ctx, &a_scale_codes, m, k, block_size).unwrap();
    let b_scales = packed_scale_tensor(&ctx, &b_scale_codes, n, k, block_size).unwrap();

    // A per-tensor scale folds into alpha, which is how a ModelOpt checkpoint's
    // weight_scale_2 and input_scale reach the kernel.
    let alpha = 0.375_f32;

    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut accumulator = 0.0f64;
            for index in 0..k {
                let block = index / block_size as usize;
                accumulator += a_values[row * k + index]
                    * a_scale_values[row * blocks + block]
                    * b_values[column * k + index]
                    * b_scale_values[column * blocks + block];
            }
            expected[row * n + column] = (accumulator * alpha as f64) as f32;
        }
    }

    let mut out = zeros_tensor(device, vec![m, n], DType::BF16);
    let mut args = GemmArgs::nvfp4(&a, &a_scales, &b, &b_scales, block_size, alpha, &mut out);
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
    validate_all_candidates(&ctx, args, None, &expected).unwrap();
}

#[test]
fn nvfp4_rejects_operands_that_are_not_packed_fp4() {
    let ctx = CudaContext::new(0).unwrap();
    let device = ctx.device_id();
    let (m, n, k, block_size) = (64usize, 64usize, 128usize, 16u32);
    let blocks = k / block_size as usize;

    let (b_packed, _) = pack_operand(n, k, 5);
    let (a_scale_codes, _) = block_scales(m, blocks, 3);
    let (b_scale_codes, _) = block_scales(n, blocks, 2);

    // BF16 activation against an FP4 weight: the contract must reject this
    // rather than reinterpret the bytes.
    let a = zeros_tensor(device, vec![m, k / 2], DType::BF16);
    let b = bytes_tensor(device, vec![n, k / 2], DType::E2M1Pair, &b_packed);
    let a_scales = packed_scale_tensor(&ctx, &a_scale_codes, m, k, block_size).unwrap();
    let b_scales = packed_scale_tensor(&ctx, &b_scale_codes, n, k, block_size).unwrap();
    let mut out = zeros_tensor(device, vec![m, n], DType::BF16);

    let args = GemmArgs::nvfp4(&a, &a_scales, &b, &b_scales, block_size, 1.0, &mut out);
    assert!(crate::ops::gemm(&ctx, args).is_err());
}

#[test]
fn nvfp4_scale_buffer_is_padded_beyond_the_checkpoint_layout() {
    // The kernel's atom layout pads the logical [rows, K/block] grid, so a
    // loader that sizes its buffer as rows*K/block would overrun. Pin that the
    // queried size is the larger one.
    let (rows, k, block_size) = (17408usize, 5120usize, 16u32);
    let logical = rows * (k / block_size as usize);
    let actual = crate::ops::nvfp4_scale_buffer_bytes(rows, k, block_size).unwrap();
    assert!(
        actual >= logical,
        "atom layout {actual} must not be smaller than the logical grid {logical}"
    );
}

#[test]
fn quantized_activation_reproduces_the_bf16_projection() {
    // The W4A4 chain is only useful if quantizing the activation preserves the
    // projection. Compare a BF16 activation quantized on device, run through
    // the NVFP4 GEMM, against the same projection computed in f64 from the
    // unquantized BF16 values.
    let ctx = CudaContext::new(0).unwrap();
    let device = ctx.device_id();
    let (m, n, k, block_size) = (64usize, 128usize, 512usize, 16u32);
    let blocks = k / block_size as usize;

    // A smoothly varying activation with a per-row magnitude spread, which is
    // what block scaling exists to handle.
    let activation_values: Vec<f32> = (0..m * k)
        .map(|index| {
            let row = index / k;
            let column = index % k;
            let magnitude = 1.0 + (row % 4) as f32 * 3.0;
            magnitude * ((column as f32 * 0.037).sin() + 0.25 * (row as f32 * 0.11).cos())
        })
        .collect();
    let activation = tensor(device, vec![m, k], &activation_values);

    let (b_packed, b_values) = pack_operand(n, k, 5);
    let (b_scale_codes, b_scale_values) = block_scales(n, blocks, 2);
    let b = bytes_tensor(device, vec![n, k / 2], DType::E2M1Pair, &b_packed);
    let b_scales = packed_scale_tensor(&ctx, &b_scale_codes, n, k, block_size).unwrap();

    // A plausible calibration: input_scale sets the representable range to
    // roughly the activation's actual amax, as ModelOpt's calibration would.
    let amax = activation_values.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let input_scale = amax / (6.0 * 448.0);

    let packed = zeros_tensor(device, vec![m, k / 2], DType::E2M1Pair);
    let scale_bytes = crate::ops::nvfp4_scale_buffer_bytes(m, k, block_size).unwrap();
    let a_scales = zeros_tensor(device, vec![scale_bytes], DType::F8E4M3);
    crate::ops::nvfp4_quantize_activation(
        &ctx,
        &activation,
        &packed,
        &a_scales,
        input_scale,
        block_size,
        crate::ops::ScaleLayout::GemmAtom,
    )
    .unwrap();

    let mut out = zeros_tensor(device, vec![m, n], DType::BF16);
    let args = GemmArgs::nvfp4(
        &packed,
        &a_scales,
        &b,
        &b_scales,
        block_size,
        input_scale,
        &mut out,
    );
    crate::ops::gemm(&ctx, args).unwrap();
    ctx.synchronize().map_err(apxinf_core::Error::Cuda).unwrap();

    let produced = values(&out);

    // Measure relative L2 over the whole output, not per-element relative
    // error. FP4 carries ~2 significant bits, so an output that happens to sum
    // near zero has a large *relative* error while contributing almost nothing
    // to the projection -- per-element ratios would report that as failure
    // while the layer is fine.
    let mut error_energy = 0.0f64;
    let mut signal_energy = 0.0f64;
    for row in 0..m {
        for column in 0..n {
            let mut accumulator = 0.0f64;
            for index in 0..k {
                // Reference uses the *unquantized* activation: this measures
                // the cost of quantization, not just kernel arithmetic.
                accumulator += activation_values[row * k + index] as f64
                    * b_values[column * k + index]
                    * b_scale_values[column * blocks + index / block_size as usize];
            }
            let got = produced[row * n + column] as f64;
            error_energy += (got - accumulator).powi(2);
            signal_energy += accumulator.powi(2);
        }
    }
    let relative_l2 = (error_energy / signal_energy).sqrt();

    // Diagnostic: a constant ratio points at a scale that is applied in the
    // wrong place; scattered ratios point at a layout or packing fault.
    for (row, column) in [(0usize, 0usize), (0, 1), (1, 0), (5, 7), (32, 64)] {
        let mut accumulator = 0.0f64;
        for index in 0..k {
            accumulator += activation_values[row * k + index] as f64
                * b_values[column * k + index]
                * b_scale_values[column * blocks + index / block_size as usize];
        }
        println!(
            "  [{row},{column}] got={:.5} expected={:.5} ratio={:.5}",
            produced[row * n + column],
            accumulator,
            produced[row * n + column] as f64 / accumulator
        );
    }
    println!("activation quantization relative L2: {relative_l2:.4}");
    // Quantizing both operands to ~2 significant bits over a 512-deep
    // reduction lands in the low percent. A wrong scale direction or nibble
    // order would be an order of magnitude worse.
    assert!(
        relative_l2 < 0.12,
        "quantized projection drifted by relative L2 {relative_l2}"
    );
}

#[test]
fn nvfp4_gemv_matches_the_block_scaled_gemm() {
    // The decode path uses a GEMV; prefill uses the block-scaled GEMM. They
    // must agree, or decode and prefill would disagree on the same weights.
    //
    // The GEMV also reads scales in the checkpoint's own row-major layout
    // rather than the kernel's atom layout, so this covers that difference
    // too: both sides are fed from the same checkpoint-order bytes.
    let ctx = CudaContext::new(0).unwrap();
    let device = ctx.device_id();
    let (n, k, block_size) = (512usize, 512usize, 16u32);
    let blocks = k / block_size as usize;

    let (a_packed, _) = pack_operand(1, k, 7);
    let (b_packed, _) = pack_operand(n, k, 5);
    let (a_scale_codes, _) = block_scales(1, blocks, 3);
    let (b_scale_codes, _) = block_scales(n, blocks, 2);
    let alpha = 0.375_f32;

    let a = bytes_tensor(device, vec![1, k / 2], DType::E2M1Pair, &a_packed);
    let b = bytes_tensor(device, vec![n, k / 2], DType::E2M1Pair, &b_packed);

    // GEMM: scales go through the atom relayout.
    let a_atom = packed_scale_tensor(&ctx, &a_scale_codes, 1, k, block_size).unwrap();
    let b_atom = packed_scale_tensor(&ctx, &b_scale_codes, n, k, block_size).unwrap();
    let mut gemm_out = zeros_tensor(device, vec![1, n], DType::BF16);
    crate::ops::gemm(
        &ctx,
        GemmArgs::nvfp4(&a, &a_atom, &b, &b_atom, block_size, alpha, &mut gemm_out),
    )
    .unwrap();

    // GEMV: scales stay in checkpoint order.
    let a_rows = bytes_tensor(device, vec![1, blocks], DType::F8E4M3, &a_scale_codes);
    let b_rows = bytes_tensor(device, vec![n, blocks], DType::F8E4M3, &b_scale_codes);
    let gemv_out = zeros_tensor(device, vec![n], DType::BF16);
    crate::ops::nvfp4_gemv(&ctx, &b, &b_rows, &a, &a_rows, &gemv_out, alpha).unwrap();
    ctx.synchronize().map_err(apxinf_core::Error::Cuda).unwrap();

    let from_gemm = values(&gemm_out);
    let from_gemv = values(&gemv_out);
    let mut worst = 0.0f64;
    for (gemm, gemv) in from_gemm.iter().zip(from_gemv.iter()) {
        let denominator = (gemm.abs() as f64).max(1e-2);
        worst = worst.max((*gemm as f64 - *gemv as f64).abs() / denominator);
    }
    println!("gemv vs block-scaled gemm: worst relative {worst:.5}");
    // Both accumulate in f32 and write BF16 but sum in different orders, so
    // this is a rounding bound, not an exactness claim.
    assert!(worst < 2e-2, "GEMV and GEMM disagree by {worst}");
}
