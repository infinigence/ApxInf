//! A complete Qwen3.8 MLP block on real NVFP4 weights, end to end.
//!
//! RMSNorm -> quantize -> fused gate/up NVFP4 GEMM -> SwiGLU -> quantize ->
//! down NVFP4 GEMM -> residual add.
//!
//! This is the first point where the pieces are exercised as a layer rather
//! than individually, so it is also where a per-layer latency number becomes
//! meaningful. The MLP is ~50% of this model's weights, so its decode cost
//! sets most of the token budget.
//!
//! ```text
//! APXINF_QWEN38_CHECKPOINT=/path/to/Qwen3.8-27B-NVFP4 \
//!   bash crates/apxinf-cuda-new/test-new.sh \
//!     test -p apxinf-cuda --test qwen38_mlp_layer -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::{ops, CudaBuffer, CudaContext};

const HIDDEN: usize = 5120;
const INTERMEDIATE: usize = 17408;
const BLOCK: u32 = 16;

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

fn upload_bytes(ctx: &CudaContext, bytes: &[u8], dims: Vec<usize>, dtype: DType) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn device_zeros(ctx: &CudaContext, dims: Vec<usize>, dtype: DType) -> Tensor {
    let bytes = dims.iter().product::<usize>() * dtype.size_in_bytes();
    let buffer = CudaBuffer::alloc(bytes, ctx.device_id()).unwrap();
    buffer.copy_from_host(&vec![0u8; bytes]).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn scalar(tensors: &HashMap<String, Tensor>, name: &str) -> f32 {
    tensors[name].to_f32_vec().unwrap()[0]
}

/// One NVFP4 projection: packed weight plus its relaid-out block scales and
/// the alpha that folds both per-tensor scales.
struct Projection {
    weight: Tensor,
    scales: Tensor,
    input_scale: f32,
    alpha: f32,
    out_features: usize,
    in_features: usize,
}

impl Projection {
    /// Concatenate gate and up along N into one [2N, K/2] operand.
    ///
    /// Both carry identical `input_scale` and `weight_scale_2` in every layer
    /// of this checkpoint, so the fused GEMM is exact -- no requantization and
    /// no epilogue correction.
    fn fused_gate_up(
        ctx: &CudaContext,
        tensors: &HashMap<String, Tensor>,
        layer: usize,
    ) -> Projection {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        let gate = &tensors[&format!("{prefix}.gate_proj.weight")];
        let up = &tensors[&format!("{prefix}.up_proj.weight")];
        let gate_scale = &tensors[&format!("{prefix}.gate_proj.weight_scale")];
        let up_scale = &tensors[&format!("{prefix}.up_proj.weight_scale")];

        let input_scale = scalar(tensors, &format!("{prefix}.gate_proj.input_scale"));
        let weight_scale_2 = scalar(tensors, &format!("{prefix}.gate_proj.weight_scale_2"));
        assert_eq!(
            input_scale,
            scalar(tensors, &format!("{prefix}.up_proj.input_scale")),
            "fusing gate and up requires a shared input_scale"
        );
        assert_eq!(
            weight_scale_2,
            scalar(tensors, &format!("{prefix}.up_proj.weight_scale_2")),
            "fusing gate and up requires a shared weight_scale_2"
        );

        let mut weight_bytes = Vec::with_capacity(cpu_bytes(gate).len() * 2);
        weight_bytes.extend_from_slice(cpu_bytes(gate));
        weight_bytes.extend_from_slice(cpu_bytes(up));
        let weight = upload_bytes(
            ctx,
            &weight_bytes,
            vec![2 * INTERMEDIATE, HIDDEN / 2],
            DType::E2M1Pair,
        );

        let mut scale_bytes = Vec::with_capacity(cpu_bytes(gate_scale).len() * 2);
        scale_bytes.extend_from_slice(cpu_bytes(gate_scale));
        scale_bytes.extend_from_slice(cpu_bytes(up_scale));
        let checkpoint_scales = upload_bytes(
            ctx,
            &scale_bytes,
            vec![2 * INTERMEDIATE, HIDDEN / BLOCK as usize],
            DType::F8E4M3,
        );
        let scales = relayout(ctx, &checkpoint_scales, 2 * INTERMEDIATE, HIDDEN);

        Projection {
            weight,
            scales,
            input_scale,
            alpha: input_scale * weight_scale_2,
            out_features: 2 * INTERMEDIATE,
            in_features: HIDDEN,
        }
    }

    fn down(ctx: &CudaContext, tensors: &HashMap<String, Tensor>, layer: usize) -> Projection {
        let prefix = format!("model.language_model.layers.{layer}.mlp.down_proj");
        let weight = upload_bytes(
            ctx,
            cpu_bytes(&tensors[&format!("{prefix}.weight")]),
            vec![HIDDEN, INTERMEDIATE / 2],
            DType::E2M1Pair,
        );
        let checkpoint_scales = upload_bytes(
            ctx,
            cpu_bytes(&tensors[&format!("{prefix}.weight_scale")]),
            vec![HIDDEN, INTERMEDIATE / BLOCK as usize],
            DType::F8E4M3,
        );
        let scales = relayout(ctx, &checkpoint_scales, HIDDEN, INTERMEDIATE);
        let input_scale = scalar(tensors, &format!("{prefix}.input_scale"));
        let weight_scale_2 = scalar(tensors, &format!("{prefix}.weight_scale_2"));
        Projection {
            weight,
            scales,
            input_scale,
            alpha: input_scale * weight_scale_2,
            out_features: HIDDEN,
            in_features: INTERMEDIATE,
        }
    }
}

fn relayout(ctx: &CudaContext, checkpoint_scales: &Tensor, rows: usize, k: usize) -> Tensor {
    let bytes = ops::nvfp4_scale_buffer_bytes(rows, k, BLOCK).unwrap();
    let destination = device_zeros(ctx, vec![bytes], DType::F8E4M3);
    ops::nvfp4_pack_block_scales(ctx, checkpoint_scales, &destination, rows, k, BLOCK).unwrap();
    destination
}

/// Scratch buffers for one MLP block at a fixed token count.
struct Scratch {
    normalized: Tensor,
    quantized: Tensor,
    quantized_scales: Tensor,
    fused: Tensor,
    activated: Tensor,
    activated_quantized: Tensor,
    activated_scales: Tensor,
    projected: Tensor,
}

impl Scratch {
    fn new(ctx: &CudaContext, tokens: usize) -> Scratch {
        Scratch {
            normalized: device_zeros(ctx, vec![tokens, HIDDEN], DType::BF16),
            quantized: device_zeros(ctx, vec![tokens, HIDDEN / 2], DType::E2M1Pair),
            quantized_scales: device_zeros(
                ctx,
                vec![ops::nvfp4_scale_buffer_bytes(tokens, HIDDEN, BLOCK).unwrap()],
                DType::F8E4M3,
            ),
            fused: device_zeros(ctx, vec![tokens, 2 * INTERMEDIATE], DType::BF16),
            activated: device_zeros(ctx, vec![tokens, INTERMEDIATE], DType::BF16),
            activated_quantized: device_zeros(
                ctx,
                vec![tokens, INTERMEDIATE / 2],
                DType::E2M1Pair,
            ),
            activated_scales: device_zeros(
                ctx,
                vec![ops::nvfp4_scale_buffer_bytes(tokens, INTERMEDIATE, BLOCK).unwrap()],
                DType::F8E4M3,
            ),
            projected: device_zeros(ctx, vec![tokens, HIDDEN], DType::BF16),
        }
    }
}

/// residual = residual + MLP(RMSNorm(residual))
///
/// `fused` selects whether the norm and activation feed their consumers
/// through a BF16 intermediate or hand packed FP4 straight over. The two are
/// numerically equivalent; only the bandwidth differs.
#[allow(clippy::too_many_arguments)]
fn mlp_block(
    ctx: &CudaContext,
    residual: &Tensor,
    norm_weight: &Tensor,
    gate_up: &Projection,
    down: &Projection,
    scratch: &mut Scratch,
    tokens: usize,
    fused: bool,
) -> apxinf_core::Result<()> {
    if fused {
        ops::nvfp4_quantize_rms_norm(
            ctx,
            residual,
            norm_weight,
            &scratch.quantized,
            &scratch.quantized_scales,
            1e-6,
            gate_up.input_scale,
            BLOCK,
            ops::ScaleLayout::GemmAtom,
        )?;
    } else {
        ops::rms_norm(ctx, residual, norm_weight, &scratch.normalized, 1e-6)?;
        ops::nvfp4_quantize_activation(
            ctx,
            &scratch.normalized,
            &scratch.quantized,
            &scratch.quantized_scales,
            gate_up.input_scale,
            BLOCK,
            ops::ScaleLayout::GemmAtom,
        )?;
    }
    ops::gemm(
        ctx,
        ops::GemmArgs::nvfp4(
            &scratch.quantized,
            &scratch.quantized_scales,
            &gate_up.weight,
            &gate_up.scales,
            BLOCK,
            gate_up.alpha,
            &mut scratch.fused,
        ),
    )?;

    if fused {
        ops::nvfp4_quantize_swiglu(
            ctx,
            &scratch.fused,
            &scratch.activated_quantized,
            &scratch.activated_scales,
            down.input_scale,
            BLOCK,
            ops::ScaleLayout::GemmAtom,
        )?;
    } else {
        ops::swiglu(ctx, &scratch.fused, &scratch.activated)?;
        ops::nvfp4_quantize_activation(
            ctx,
            &scratch.activated,
            &scratch.activated_quantized,
            &scratch.activated_scales,
            down.input_scale,
            BLOCK,
            ops::ScaleLayout::GemmAtom,
        )?;
    }
    ops::gemm(
        ctx,
        ops::GemmArgs::nvfp4(
            &scratch.activated_quantized,
            &scratch.activated_scales,
            &down.weight,
            &down.scales,
            BLOCK,
            down.alpha,
            &mut scratch.projected,
        ),
    )?;

    ops::add_into(ctx, &scratch.projected, residual)?;
    let _ = (tokens, gate_up.out_features, gate_up.in_features, down.out_features, down.in_features);
    Ok(())
}

#[test]
#[ignore = "requires the 20 GiB Qwen3.8-27B-NVFP4 checkpoint and a GPU"]
fn nvfp4_mlp_block_runs_and_is_timed() {
    let ctx = CudaContext::new(0).unwrap();
    let load_start = Instant::now();
    let tensors = checkpoint();
    println!("checkpoint load: {:.2}s", load_start.elapsed().as_secs_f64());

    let build_start = Instant::now();
    let gate_up = Projection::fused_gate_up(&ctx, &tensors, 0);
    let down = Projection::down(&ctx, &tensors, 0);
    let norm_weight = upload_bytes(
        &ctx,
        cpu_bytes(&tensors["model.language_model.layers.0.post_attention_layernorm.weight"]),
        vec![HIDDEN],
        DType::BF16,
    );
    ctx.synchronize().unwrap();
    println!(
        "device weights ready: {:.2}s (fused gate/up {}x{}, down {}x{})",
        build_start.elapsed().as_secs_f64(),
        gate_up.out_features,
        gate_up.in_features,
        down.out_features,
        down.in_features
    );

    // Correctness before speed: the fused path must produce the same block
    // output as the separate one, or the timing below is measuring the wrong
    // computation.
    {
        let tokens = 256usize;
        let mut scratch = Scratch::new(&ctx, tokens);
        let seed: Vec<u8> = (0..tokens * HIDDEN * 2)
            .map(|index| ((index * 37 + 11) % 251) as u8)
            .collect();

        let mut outputs = Vec::new();
        for fused in [false, true] {
            let residual = upload_bytes(&ctx, &seed, vec![tokens, HIDDEN], DType::BF16);
            mlp_block(
                &ctx, &residual, &norm_weight, &gate_up, &down, &mut scratch, tokens, fused,
            )
            .unwrap();
            ctx.synchronize().unwrap();
            let mut bytes = vec![0u8; tokens * HIDDEN * 2];
            CudaBuffer::from_tensor(&residual)
                .unwrap()
                .copy_to_host(&mut bytes)
                .unwrap();
            outputs.push(bytes);
        }
        let mismatches = outputs[0]
            .chunks_exact(2)
            .zip(outputs[1].chunks_exact(2))
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "fused vs separate: {mismatches} of {} outputs differ",
            tokens * HIDDEN
        );
        // Both paths do identical arithmetic in identical order, so this is an
        // exact-equality check, not a tolerance.
        assert_eq!(mismatches, 0, "fused path changed the block output");
    }

    for &tokens in &[1usize, 256, 512, 1024] {
        let mut scratch = Scratch::new(&ctx, tokens);
        let residual_values: Vec<u8> = (0..tokens * HIDDEN * 2)
            .map(|index| ((index * 37 + 11) % 251) as u8)
            .collect();
        let residual = upload_bytes(&ctx, &residual_values, vec![tokens, HIDDEN], DType::BF16);

        // Weight bytes this block reads: packed FP4 plus one scale per 16.
        let weight_bytes = (2.0 * INTERMEDIATE as f64 * HIDDEN as f64
            + HIDDEN as f64 * INTERMEDIATE as f64)
            * (0.5 + 1.0 / 16.0);
        let flops = 2.0 * tokens as f64
            * (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE) as f64;

        let mut timings = [0.0f64; 2];
        for (slot, fused) in [(0usize, false), (1usize, true)] {
            // Warm up: the first call tunes and prepares native executions.
            mlp_block(
                &ctx, &residual, &norm_weight, &gate_up, &down, &mut scratch, tokens, fused,
            )
            .unwrap();
            ctx.synchronize().unwrap();

            let iterations = if tokens == 1 { 50 } else { 20 };
            let start = Instant::now();
            for _ in 0..iterations {
                mlp_block(
                    &ctx, &residual, &norm_weight, &gate_up, &down, &mut scratch, tokens, fused,
                )
                .unwrap();
            }
            ctx.synchronize().unwrap();
            timings[slot] = start.elapsed().as_secs_f64() / iterations as f64;
        }

        for (label, per_call) in [("separate", timings[0]), ("fused   ", timings[1])] {
            println!(
                "tokens={tokens:5} {label}  {:8.3} ms/block  {:7.2} TF/s  {:6.1} GB/s  \
                 -> 64 layers = {:7.2} ms",
                per_call * 1e3,
                flops / per_call / 1e12,
                weight_bytes / per_call / 1e9,
                per_call * 64.0 * 1e3
            );
        }
        println!(
            "              fusing the norm and activation into the quantizer: {:+.1}%",
            (timings[1] / timings[0] - 1.0) * 100.0
        );
    }
}
