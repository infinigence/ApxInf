// NVFP4 quantization contract (report 64, defects D1/D2/D4).
//
// The reference semantics are vLLM v0.28.0's real CUDA quantization output:
//   * FP4 E2M1 codes round to nearest with ties to even and preserve the
//     sign of -0.0 (D1: the old comparison ladder rounded half-up).
//   * E4M3 block scales keep subnormals, codes 1..=8 (D2: the old frexpf
//     ladder flushed them, zeroing whole 16-element blocks).
//   * Fused SwiGLU materializes bf16(bf16(silu(gate)) * up) before
//     quantizing (D4: pure-FP32 SwiGLU shifted 207k FP4 codes on real
//     prefill inputs, Codex round 63).
//
// The oracle is implemented independently in f64 (candidate-by-candidate
// nearest-even search); it never calls the code under test. The scale
// division is deliberately written as a multiply by the FP32 reciprocal --
// both our kernel and vLLM divide that way, and an exact-division oracle
// falsely reports 84 midpoint mismatches (Codex round 60).

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext};

fn upload(ctx: &CudaContext, bytes: &[u8], dims: Vec<usize>, dtype: DType) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn download(tensor: &Tensor, count: usize) -> Vec<u8> {
    let mut bytes = vec![0; count];
    CudaBuffer::from_tensor(tensor).unwrap().copy_to_host(&mut bytes).unwrap();
    bytes
}

/// Exact value of a non-negative E4M3 code, subnormals included.
fn scale_value(code: u8) -> f32 {
    let exponent = (code >> 3) as i32;
    let fraction = (code & 7) as f32;
    if exponent == 0 {
        fraction / 512.0
    } else {
        (1.0 + fraction / 8.0) * 2.0_f32.powi(exponent - 7)
    }
}

/// Independent E2M1 oracle: nearest representable magnitude in f64 with ties
/// to even (even = even code index), sign bit from the FP32 sign.
fn nearest_e2m1_code(value: f32) -> u8 {
    let magnitudes = [0.0_f64, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = (value as f64).abs();
    let mut best = 0_usize;
    for candidate in 1..magnitudes.len() {
        let distance = (magnitude - magnitudes[candidate]).abs();
        let best_distance = (magnitude - magnitudes[best]).abs();
        if distance < best_distance || (distance == best_distance && candidate % 2 == 0) {
            best = candidate;
        }
    }
    best as u8 | if value.is_sign_negative() { 8 } else { 0 }
}

/// Every positive finite E4M3 scale code (1..=126) times both signs, with a
/// 16-element pattern containing all four FP4 midpoints (0.25/1.25/2.5/5.0),
/// a signed zero and amax = 6 * scale so the expected block scale is exact.
#[test]
fn nvfp4_rne_midpoints_and_subnormal_block_scales() {
    let ctx = CudaContext::new(0).unwrap();
    let pattern = [
        0.0_f32, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
        2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, -0.25,
    ];
    let mut values = Vec::new();
    let mut expected_scales = Vec::new();
    let mut expected_codes = Vec::new();
    for code in 1_u8..=126 {
        for sign in [1.0_f32, -1.0] {
            let scale = scale_value(code);
            let row: Vec<f32> = pattern
                .iter()
                .map(|value| half::bf16::from_f32(value * scale * sign).to_f32())
                .collect();
            // The pattern is engineered so BF16 storage is lossless at the
            // block maximum: the stored amax must be exactly 6 * scale, which
            // encodes back to `code` with no rounding.
            assert_eq!(
                row.iter().map(|value| value.abs()).fold(0.0_f32, f32::max),
                6.0 * scale
            );
            expected_scales.push(code);
            // FP32 reciprocal multiply, matching the kernel and vLLM.
            let inverse = 1.0_f32 / scale;
            expected_codes.extend(row.iter().map(|value| nearest_e2m1_code(*value * inverse)));
            values.extend(row);
        }
    }
    let rows = expected_scales.len();
    assert_eq!(rows, 252);
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect();
    let input = upload(&ctx, &bytes, vec![rows, 16], DType::BF16);
    let packed = upload(&ctx, &vec![0xaa; rows * 8], vec![rows, 8], DType::E2M1Pair);
    let scales = upload(&ctx, &vec![0xff; rows], vec![rows], DType::F8E4M3);
    ops::nvfp4_quantize_activation(&ctx, &input, &packed, &scales, 1.0, 16, ops::ScaleLayout::RowMajor)
        .unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(download(&scales, rows), expected_scales);
    let codes: Vec<u8> = download(&packed, rows * 8)
        .into_iter()
        .flat_map(|byte| [byte & 15, byte >> 4])
        .collect();
    assert_eq!(codes.len(), 4032);
    for index in 0..codes.len() {
        assert_eq!(
            codes[index], expected_codes[index],
            "element={index} block_scale={}",
            expected_scales[index / 16]
        );
    }
    println!(
        "PASS 126 positive finite E4M3 scales, both signs, 4032 FP4 codes \
         with independent nearest-even oracle"
    );
}

/// D4 cross-check: the fused SwiGLU quantization (scalar and vector dispatch)
/// must match a real vLLM v0.28.0 fixture bit for bit, and the standalone
/// swiglu kernel must produce the fixture's BF16 intermediate. This pins the
/// bf16(bf16(silu(gate)) * up) rounding chain at both boundaries.
#[test]
fn swiglu_bf16_boundaries_match_independent_vllm_fixture() {
    let ctx = CudaContext::new(0).unwrap();
    let previous = std::env::var_os("APXINF_QWEN38_VECTOR_ELEMENTWISE");
    // 32 gate + 32 up BF16 values captured from a real vLLM prefill, plus the
    // BF16 SwiGLU output, packed FP4 and block scales vLLM computed for them.
    let input_bits: [u16; 64] = [
        15797, 48543, 48487, 48286, 48590, 48747, 48535, 48307,
        48104, 15596, 48588, 15767, 15697, 15846, 48437, 48619,
        48551, 15625, 48603, 48515, 48380, 48276, 48626, 15688,
        48458, 48341, 48564, 48416, 15665, 48379, 15620, 48387,
        15666, 48510, 48060, 48026, 48612, 15827, 48340, 47775,
        15361, 48276, 48408, 15914, 15691, 15658, 48462, 15666,
        48278, 15556, 48537, 48428, 15677, 48603, 15644, 48547,
        48583, 48442, 15835, 48581, 15877, 48331, 15664, 15337,
    ];
    let output_bits: [u16; 32] = [
        15107, 15128, 14628, 14396, 15279, 48171, 14960, 14172,
        47081, 47498, 15078, 15313, 15018, 15137, 14990, 47898,
        14908, 14805, 15223, 15019, 47671, 14972, 47883, 47875,
        15129, 14873, 48019, 15090, 15164, 14788, 14904, 47339,
    ];
    let packed_bytes = [50_u8, 0, 245, 1, 136, 98, 49, 177, 18, 54, 58, 221, 37, 79, 22, 130];
    let scale_bytes = [12_u8, 5];
    for rows in [1, 3, 65] {
        let input_bytes: Vec<u8> = input_bits
            .repeat(rows)
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let expected_bf16: Vec<u8> = output_bits
            .repeat(rows)
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let input = upload(&ctx, &input_bytes, vec![rows, 64], DType::BF16);
        let output = upload(&ctx, &vec![0xff; rows * 64], vec![rows, 32], DType::BF16);
        ops::swiglu(&ctx, &input, &output).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(download(&output, rows * 64), expected_bf16);
        for vector in ["0", "1"] {
            std::env::set_var("APXINF_QWEN38_VECTOR_ELEMENTWISE", vector);
            let packed = upload(&ctx, &vec![0xaa; rows * 16], vec![rows, 16], DType::E2M1Pair);
            let scales = upload(&ctx, &vec![0xff; rows * 2], vec![rows, 2], DType::F8E4M3);
            ops::nvfp4_quantize_swiglu(
                &ctx,
                &input,
                &packed,
                &scales,
                1.0_f32 / 12.923076629638672_f32,
                16,
                ops::ScaleLayout::RowMajor,
            )
            .unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(download(&packed, rows * 16), packed_bytes.repeat(rows));
            assert_eq!(download(&scales, rows * 2), scale_bytes.repeat(rows));
        }
        println!(
            "PASS rows={rows} vLLM v0.28.0 real-input fixture: standalone BF16, \
             fused scalar/vector FP4 and block scales"
        );
    }
    if let Some(value) = previous {
        std::env::set_var("APXINF_QWEN38_VECTOR_ELEMENTWISE", value);
    } else {
        std::env::remove_var("APXINF_QWEN38_VECTOR_ELEMENTWISE");
    }
}
