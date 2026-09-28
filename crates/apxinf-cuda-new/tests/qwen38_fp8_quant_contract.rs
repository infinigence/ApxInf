// FP8 E4M3 quantization contract (report 64, defect D3).
//
// The reference semantics are vLLM v0.28.0's real CUDA output: round to
// nearest even, saturate at +-448 without NaN, and keep subnormal codes
// instead of flushing them to signed zero. The old frexpf/bit-twiddling
// ladders zeroed every |value| below the smallest normal, losing 1.13M
// non-zero elements on one real 2048x6144 prefill quantization (round 61).
//
// The oracle enumerates all 126 positive E4M3 codes in f64 and picks the
// nearest with ties to even; it never calls the code under test. Coverage is
// every finite BF16 bit pattern times five scales times the three dispatch
// modes (scalar, vector, native-pair), plus a 13-element tail-path shape.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext};

fn upload(ctx: &CudaContext, bytes: &[u8], count: usize, dtype: DType) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(vec![count]), dtype).unwrap()
}

/// Independent E4M3 oracle: nearest of the 126 positive finite codes in f64,
/// ties to even, SATFINITE saturation, sign preserved (including -0.0).
fn nearest_e4m3(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 128 } else { 0 };
    let magnitude = (value as f64).abs();
    if magnitude >= 448.0 {
        return sign | 126;
    }
    let mut best_code = 0_u8;
    let mut best_distance = magnitude;
    for code in 1_u8..=126 {
        let exponent = (code >> 3) as i32;
        let fraction = (code & 7) as f64;
        let decoded = if exponent == 0 {
            fraction / 512.0
        } else {
            (1.0 + fraction / 8.0) * 2.0_f64.powi(exponent - 7)
        };
        let distance = (magnitude - decoded).abs();
        if distance < best_distance || (distance == best_distance && code % 2 == 0) {
            best_code = code;
            best_distance = distance;
        }
    }
    sign | best_code
}

#[test]
fn fp8_finite_bf16_domain_matches_independent_rne_oracle() {
    let ctx = CudaContext::new(0).unwrap();
    let old_vector = std::env::var_os("APXINF_QWEN38_VECTOR_ELEMENTWISE");
    let old_pair = std::env::var_os("APXINF_FP8_NATIVE_PAIR");
    // Every BF16 bit pattern that is not Inf/NaN, both signs.
    let full_domain: Vec<u16> = (0_u16..=u16::MAX)
        .filter(|bits| bits & 0x7f80 != 0x7f80)
        .collect();
    assert_eq!(full_domain.len(), 65280);
    // A 13-element shape exercises the scalar kernel's tail elements, which
    // the 8-aligned full domain never reaches. Values sit around the
    // subnormal boundary, the midpoints and the saturation edge.
    let fallback = vec![
        0, 0x8000, 0x3a80, 0xba80, 0x3b00, 0xbb00, 0x3c00,
        0xbc00, 0x3c70, 0xbc70, 0x3f80, 0x7f7f, 0xff7f,
    ];
    let mut checked_values = 0;
    for bits in [&full_domain, &fallback] {
        let count = bits.len();
        let bytes: Vec<u8> = bits.iter().flat_map(|value| value.to_le_bytes()).collect();
        let input = upload(&ctx, &bytes, count, DType::BF16);
        for scale in [0.5_f32, 1.0, 2.0, 0.1121651828289032, 0.0322265625] {
            // FP32 reciprocal multiply, matching the kernel and vLLM.
            let inverse = 1.0_f32 / scale;
            let expected: Vec<u8> = bits
                .iter()
                .map(|value| nearest_e4m3(half::bf16::from_bits(*value).to_f32() * inverse))
                .collect();
            for (vector, pair) in [("0", "0"), ("1", "0"), ("1", "1")] {
                std::env::set_var("APXINF_QWEN38_VECTOR_ELEMENTWISE", vector);
                std::env::set_var("APXINF_FP8_NATIVE_PAIR", pair);
                let output = upload(&ctx, &vec![0x7f; count], count, DType::F8E4M3);
                ops::quantize_fp8_per_tensor(&ctx, &input, &output, scale).unwrap();
                ctx.synchronize().unwrap();
                let mut actual = vec![0; count];
                CudaBuffer::from_tensor(&output)
                    .unwrap()
                    .copy_to_host(&mut actual)
                    .unwrap();
                for index in 0..count {
                    assert_eq!(
                        actual[index], expected[index],
                        "bits={:04x} scale={scale} vector={vector} pair={pair}",
                        bits[index]
                    );
                }
                checked_values += count;
            }
        }
    }
    for (name, old_value) in [
        ("APXINF_QWEN38_VECTOR_ELEMENTWISE", old_vector),
        ("APXINF_FP8_NATIVE_PAIR", old_pair),
    ] {
        if let Some(value) = old_value {
            std::env::set_var(name, value);
        } else {
            std::env::remove_var(name);
        }
    }
    assert_eq!(checked_values, 979_395);
    println!(
        "PASS {checked_values} FP8 codes, finite BF16 domain, five scales, \
         three dispatch modes and scalar fallback"
    );
}
