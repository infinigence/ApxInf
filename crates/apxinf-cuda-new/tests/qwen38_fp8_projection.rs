//! The FP8 projection path on real attention weights.
//!
//! Every projection outside the MLP -- all of `self_attn` and all of
//! `linear_attn` -- is FP8 in this checkpoint, and each carries *scalar*
//! `weight_scale` and `input_scale` rather than the `[M]`/`[N]` vectors the
//! scaled-FP8 GEMM arm expects. The claim this test exists to check is that
//! those projections need no new quantization contract: quantize the
//! activation against `input_scale`, run unit-scale FP8, and let
//! `alpha = weight_scale * input_scale` carry both.
//!
//! If that is wrong, attention and GDN need a per-tensor FP8 contract, and it
//! is much cheaper to find out here than after the model is assembled.
//!
//! ```text
//! APXINF_QWEN38_CHECKPOINT=/path/to/Qwen3.8-27B-NVFP4 \
//!   bash crates/apxinf-cuda-new/test-new.sh \
//!     test -p apxinf-cuda --test qwen38_fp8_projection -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::{ops, CudaBuffer, CudaContext};

fn checkpoint() -> HashMap<String, Tensor> {
    let path: PathBuf = std::env::var_os("APXINF_QWEN38_CHECKPOINT")
        .expect("set APXINF_QWEN38_CHECKPOINT")
        .into();
    apxinf_loader::safetensors::load_native_path(&path)
        .expect("checkpoint failed to load")
        .0
}

fn cpu_bytes(tensor: &Tensor) -> &[u8] {
    match tensor.storage() {
        apxinf_core::Storage::Cpu(data) => data,
        _ => panic!("expected a CPU tensor"),
    }
}

fn upload(ctx: &CudaContext, bytes: &[u8], dims: Vec<usize>, dtype: DType) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn zeros(ctx: &CudaContext, dims: Vec<usize>, dtype: DType) -> Tensor {
    let bytes = dims.iter().product::<usize>() * dtype.size_in_bytes();
    let buffer = CudaBuffer::alloc(bytes, ctx.device_id()).unwrap();
    buffer.copy_from_host(&vec![0u8; bytes]).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

/// Decode OCP E4M3 (bias 7).
fn e4m3_value(code: u8) -> f64 {
    let exponent = ((code >> 3) & 0x0F) as i32;
    let mantissa = (code & 0x07) as f64;
    let magnitude = if exponent == 0 {
        mantissa / 8.0 * (-6.0f64).exp2()
    } else {
        (1.0 + mantissa / 8.0) * ((exponent - 7) as f64).exp2()
    };
    if code & 0x80 != 0 { -magnitude } else { magnitude }
}

fn bf16_to_f64(raw: u16) -> f64 {
    f32::from_bits((raw as u32) << 16) as f64
}

fn read_bf16(tensor: &Tensor) -> Vec<f64> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|value| bf16_to_f64(u16::from_le_bytes([value[0], value[1]])))
        .collect()
}

#[test]
#[ignore = "requires the 20 GiB Qwen3.8-27B-NVFP4 checkpoint and a GPU"]
fn per_tensor_fp8_projection_needs_no_new_contract() {
    let ctx = CudaContext::new(0).unwrap();
    let tensors = checkpoint();

    // Layer 3 is the first full-attention layer. q_proj is the widest of the
    // attention projections and the one whose output feeds the output gate.
    let prefix = "model.language_model.layers.3.self_attn.q_proj";
    let weight_host = &tensors[&format!("{prefix}.weight")];
    assert_eq!(weight_host.dtype(), DType::F8E4M3);
    let weight_scale = tensors[&format!("{prefix}.weight_scale")]
        .to_f32_vec()
        .unwrap()[0];
    let input_scale = tensors[&format!("{prefix}.input_scale")]
        .to_f32_vec()
        .unwrap()[0];

    let n_full = weight_host.shape().dims()[0];
    let k = weight_host.shape().dims()[1];
    println!(
        "q_proj [{n_full}, {k}]  weight_scale={weight_scale:e}  input_scale={input_scale:e}"
    );
    assert_eq!(k, 5120);
    // 12288 = 2 * (24 heads * 256): q and its output gate share one projection.
    assert_eq!(n_full, 12288);

    // Slice the projection: the full width would dominate runtime without
    // testing anything the slice does not. The GEMM contract stores B as
    // [K, N], and the checkpoint stores [N, K], so the slice is transposed
    // on the host once.
    let n = 512usize;
    let weight_bytes = cpu_bytes(weight_host);
    let mut transposed = vec![0u8; k * n];
    for row in 0..n {
        for column in 0..k {
            transposed[column * n + row] = weight_bytes[row * k + column];
        }
    }
    let weight = upload(&ctx, &transposed, vec![k, n], DType::F8E4M3);

    // A plausible post-norm activation: RMSNorm output has unit-ish scale.
    let m = 32usize;
    let activation_values: Vec<f32> = (0..m * k)
        .map(|index| {
            let row = index / k;
            let column = index % k;
            ((column as f32 * 0.021).sin() + 0.3 * (row as f32 * 0.17).cos()) * 1.5
        })
        .collect();
    let activation_bytes: Vec<u8> = activation_values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect();
    let activation = upload(&ctx, &activation_bytes, vec![m, k], DType::BF16);

    let quantized = zeros(&ctx, vec![m, k], DType::F8E4M3);
    ops::quantize_fp8_per_tensor(&ctx, &activation, &quantized, input_scale).unwrap();

    // The whole point: unit-scale FP8 with both per-tensor scales in alpha.
    let mut out = zeros(&ctx, vec![m, n], DType::BF16);
    let mut args = ops::GemmArgs::new(&quantized, &weight, &mut out);
    args.quantization = ops::GemmQuantization::Fp8UnitScale;
    args.alpha = weight_scale * input_scale;
    ops::gemm(&ctx, args).unwrap();
    ctx.synchronize().unwrap();

    let produced = read_bf16(&out);

    // Reference from the checkpoint bytes and the *unquantized* activation, so
    // this measures the cost of FP8 quantization rather than kernel arithmetic.
    let mut error_energy = 0.0f64;
    let mut signal_energy = 0.0f64;
    for row in 0..m {
        for column in 0..n {
            let mut accumulator = 0.0f64;
            for index in 0..k {
                accumulator += activation_values[row * k + index] as f64
                    * e4m3_value(weight_bytes[column * k + index])
                    * weight_scale as f64;
            }
            let got = produced[row * n + column];
            error_energy += (got - accumulator).powi(2);
            signal_energy += accumulator.powi(2);
        }
    }
    let relative_l2 = (error_energy / signal_energy).sqrt();
    println!("per-tensor FP8 projection relative L2: {relative_l2:.5}");

    // E4M3 has a 3-bit mantissa, so 4 significant bits: worst-case relative
    // rounding is 2^-4 = 6.25% and the RMS over a uniform distribution is
    // ~3.6%. The weights are already E4M3 in the checkpoint and the reference
    // decodes those same bytes, so essentially all of this is the *activation*
    // quantization. Measured 2.7%, which sits just under the analytic RMS.
    //
    // The bound is 4% rather than something tighter because tightening it
    // further would not catch a real fault: a scale applied in the wrong place
    // lands orders of magnitude out, not a few percent.
    assert!(
        relative_l2 < 0.04,
        "per-tensor FP8 projection drifted by relative L2 {relative_l2}"
    );

    // What this settles: the FP8 projections need no new quantization
    // contract. Unit-scale FP8 plus `alpha = weight_scale * input_scale`
    // reproduces the projection, so attention and GDN can reuse the existing
    // arm and only NVFP4 required a new one.
}
