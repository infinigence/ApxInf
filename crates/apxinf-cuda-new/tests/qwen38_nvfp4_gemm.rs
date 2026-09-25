//! End-to-end path for a real NVFP4 weight: checkpoint bytes -> device ->
//! block-scale relayout -> block-scaled GEMM.
//!
//! The in-crate precision test already covers the kernel against synthetic
//! operands. What this adds is the part synthetic data cannot check: that the
//! bytes a ModelOpt checkpoint actually stores are consumed correctly, with
//! the real packing, the real scale layout and the real per-tensor scales.
//!
//! ```text
//! APXINF_QWEN38_CHECKPOINT=/path/to/Qwen3.8-27B-NVFP4 \
//!   bash crates/apxinf-cuda-new/test-new.sh \
//!     test -p apxinf-cuda --test qwen38_nvfp4_gemm -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::{ops, CudaBuffer, CudaContext};

/// e2m1 magnitudes indexed by the low three bits; bit 3 is the sign.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e2m1_value(code: u8) -> f64 {
    let magnitude = E2M1[(code & 0x7) as usize] as f64;
    if code & 0x8 != 0 { -magnitude } else { magnitude }
}

/// Decode E4M3 (bias 7). Scale tensors are non-negative, where the unsigned
/// and signed encodings coincide, which is why checkpoint bytes reach the
/// kernel unchanged.
fn e4m3_value(code: u8) -> f64 {
    let exponent = ((code >> 3) & 0x0F) as i32;
    let mantissa = (code & 0x07) as f64;
    if exponent == 0 {
        return mantissa / 8.0 * (-6.0f64).exp2();
    }
    let magnitude = (1.0 + mantissa / 8.0) * ((exponent - 7) as f64).exp2();
    if code & 0x80 != 0 { -magnitude } else { magnitude }
}

fn bf16_to_f64(raw: u16) -> f64 {
    f32::from_bits((raw as u32) << 16) as f64
}

fn upload(ctx: &CudaContext, host: &Tensor) -> Tensor {
    let bytes = host.shape().numel() * host.dtype().size_in_bytes();
    let cpu = match host.storage() {
        apxinf_core::Storage::Cpu(data) => data,
        _ => panic!("expected a CPU tensor"),
    };
    let buffer = CudaBuffer::alloc(bytes, ctx.device_id()).unwrap();
    buffer.copy_from_host(&cpu[..bytes]).unwrap();
    buffer
        .as_tensor(Shape::new(host.shape().dims().to_vec()), host.dtype())
        .unwrap()
}

fn checkpoint() -> HashMap<String, Tensor> {
    let path: PathBuf = std::env::var_os("APXINF_QWEN38_CHECKPOINT")
        .expect("set APXINF_QWEN38_CHECKPOINT")
        .into();
    apxinf_loader::safetensors::load_native_path(&path)
        .expect("checkpoint failed to load")
        .0
}

#[test]
#[ignore = "requires the 20 GiB Qwen3.8-27B-NVFP4 checkpoint and a GPU"]
fn real_gate_proj_weight_runs_through_the_nvfp4_gemm() {
    let ctx = CudaContext::new(0).unwrap();
    let tensors = checkpoint();
    let prefix = "model.language_model.layers.0.mlp.gate_proj";

    let weight_host = &tensors[&format!("{prefix}.weight")];
    let scale_host = &tensors[&format!("{prefix}.weight_scale")];
    let weight_scale_2 = tensors[&format!("{prefix}.weight_scale_2")]
        .to_f32_vec()
        .unwrap()[0];
    let input_scale = tensors[&format!("{prefix}.input_scale")]
        .to_f32_vec()
        .unwrap()[0];

    assert_eq!(weight_host.dtype(), DType::E2M1Pair);
    let n = weight_host.shape().dims()[0];
    let k = weight_host.shape().dims()[1] * 2;
    let block = 16usize;
    let blocks = k / block;
    assert_eq!(scale_host.shape().dims(), &[n, blocks]);
    println!("gate_proj N={n} K={k} weight_scale_2={weight_scale_2:e} input_scale={input_scale:e}");

    // Keep the test to a slice of the projection: the full 17408 rows would
    // dominate runtime without testing anything the slice does not.
    let rows = 512usize;
    let weight_bytes = weight_host.as_e2m1_pairs().unwrap();
    let scale_bytes = scale_host.as_f8_e4m3().unwrap();
    let weight_slice = &weight_bytes[..rows * k / 2];
    let scale_slice = &scale_bytes[..rows * blocks];

    let weight_cpu =
        Tensor::from_e2m1_pairs(Shape::new(vec![rows, k / 2]), weight_slice).unwrap();
    let scale_cpu = Tensor::from_f8_e4m3(Shape::new(vec![rows, blocks]), scale_slice).unwrap();
    let weight = upload(&ctx, &weight_cpu);
    let checkpoint_scales = upload(&ctx, &scale_cpu);

    // Relayout the weight scales once, exactly as a loader would.
    let scale_bytes_needed = ops::nvfp4_scale_buffer_bytes(rows, k, block as u32).unwrap();
    let weight_scales = CudaBuffer::alloc(scale_bytes_needed, ctx.device_id())
        .unwrap()
        .as_tensor(Shape::new(vec![scale_bytes_needed]), DType::F8E4M3)
        .unwrap();
    ops::nvfp4_pack_block_scales(
        &ctx,
        &checkpoint_scales,
        &weight_scales,
        rows,
        k,
        block as u32,
    )
    .unwrap();

    // A deterministic FP4 activation with unit block scales: the point is the
    // weight path, so the activation stays simple and exactly representable.
    let m = 8usize;
    let mut activation_bytes = vec![0u8; m * k / 2];
    for (index, byte) in activation_bytes.iter_mut().enumerate() {
        *byte = ((index % 7 + 1) | ((index % 5 + 1) << 4)) as u8;
    }
    let activation = upload(
        &ctx,
        &Tensor::from_e2m1_pairs(Shape::new(vec![m, k / 2]), &activation_bytes).unwrap(),
    );
    let unit = 0x38u8; // E4M3 for 1.0
    let activation_scales_cpu =
        Tensor::from_f8_e4m3(Shape::new(vec![m, blocks]), &vec![unit; m * blocks]).unwrap();
    let activation_checkpoint_scales = upload(&ctx, &activation_scales_cpu);
    let activation_scale_bytes = ops::nvfp4_scale_buffer_bytes(m, k, block as u32).unwrap();
    let activation_scales = CudaBuffer::alloc(activation_scale_bytes, ctx.device_id())
        .unwrap()
        .as_tensor(Shape::new(vec![activation_scale_bytes]), DType::F8E4M3)
        .unwrap();
    ops::nvfp4_pack_block_scales(
        &ctx,
        &activation_checkpoint_scales,
        &activation_scales,
        m,
        k,
        block as u32,
    )
    .unwrap();

    // Both per-tensor scales multiply the projection, so they are alpha.
    let alpha = weight_scale_2 * input_scale;

    let mut out = CudaBuffer::alloc(m * rows * 2, ctx.device_id())
        .unwrap()
        .as_tensor(Shape::new(vec![m, rows]), DType::BF16)
        .unwrap();
    let args = ops::GemmArgs::nvfp4(
        &activation,
        &activation_scales,
        &weight,
        &weight_scales,
        block as u32,
        alpha,
        &mut out,
    );
    ops::gemm(&ctx, args).unwrap();
    ctx.synchronize().unwrap();

    // Host reference in f64 straight from the checkpoint bytes.
    let mut produced = vec![0u8; m * rows * 2];
    CudaBuffer::from_tensor(&out)
        .unwrap()
        .copy_to_host(&mut produced)
        .unwrap();

    let mut worst = 0.0f64;
    for row in 0..m {
        for column in 0..rows {
            let mut accumulator = 0.0f64;
            for index in 0..k {
                let a_byte = activation_bytes[row * (k / 2) + index / 2];
                let a_code = if index % 2 == 0 { a_byte & 0x0F } else { a_byte >> 4 };
                let b_byte = weight_slice[column * (k / 2) + index / 2];
                let b_code = if index % 2 == 0 { b_byte & 0x0F } else { b_byte >> 4 };
                let b_scale = e4m3_value(scale_slice[column * blocks + index / block]);
                accumulator += e2m1_value(a_code) * e2m1_value(b_code) * b_scale;
            }
            let expected = accumulator * alpha as f64;
            let offset = (row * rows + column) * 2;
            let got = bf16_to_f64(u16::from_le_bytes([produced[offset], produced[offset + 1]]));
            let relative = (got - expected).abs() / expected.abs().max(1e-3);
            worst = worst.max(relative);
        }
    }
    println!("worst relative error against the checkpoint bytes: {worst:.5}");
    // BF16 output over a 5120-deep reduction; 3% bounds rounding while still
    // catching a wrong scale direction or a mis-ordered nibble.
    assert!(worst < 3e-2, "worst relative error {worst}");
}
