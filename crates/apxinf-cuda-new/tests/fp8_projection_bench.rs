//! Standalone rate measurement for the FP8 projection shapes that dominate
//! Qwen3.8 GDN prefill.
//!
//! The end-to-end stage timer cannot separate "the candidate is slow" from
//! "the framework around the candidate is slow", because it also contains the
//! activation quantization, the per-call device synchronize that `ops::gemm`
//! performs outside an execution session, and any vendor epilogue pass. This
//! runs the same contract -- `GemmQuantization::Fp8UnitScale`, a non-unit
//! alpha, a `[K, N]` row-major weight, a BF16 output -- on synthetic operands
//! and reports the achieved rate for the GEMM call alone.
//!
//! ```text
//! APXINF_GEMM_REPORT=1 bash crates/apxinf-cuda-new/test-new.sh \
//!   test -p apxinf-cuda --test fp8_projection_bench --release -- \
//!   --ignored --nocapture --test-threads=1
//! ```

use std::time::Instant;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::{ops, CudaBuffer, CudaContext};

fn device_tensor(ctx: &CudaContext, bytes: &[u8], dims: Vec<usize>, dtype: DType) -> Tensor {
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

/// A spread of finite E4M3 codes. Constant operands would let a kernel look
/// good for reasons unrelated to arithmetic, and denormal or NaN codes would
/// make the comparison against a reference meaningless.
fn e4m3_pattern(count: usize, seed: usize) -> Vec<u8> {
    const CODES: [u8; 8] = [0x38, 0x3c, 0x34, 0x40, 0xb8, 0x30, 0x44, 0xbc];
    (0..count)
        .map(|index| CODES[(index * 7 + seed * 3) % CODES.len()])
        .collect()
}

fn measure(ctx: &CudaContext, label: &str, m: usize, k: usize, n: usize, output_dtype: DType) {
    let a = device_tensor(ctx, &e4m3_pattern(m * k, 1), vec![m, k], DType::F8E4M3);
    let b = device_tensor(ctx, &e4m3_pattern(k * n, 5), vec![k, n], DType::F8E4M3);
    let mut out = zeros(ctx, vec![m, n], output_dtype);

    // alpha is the checkpoint-shaped `weight_scale * input_scale`; a unit
    // alpha would select a different equivalence class in the tuning key.
    let alpha = 0.0123_f32;
    let mut run = || {
        let mut args = ops::GemmArgs::new(&a, &b, &mut out);
        args.quantization = ops::GemmQuantization::Fp8UnitScale;
        args.alpha = alpha;
        ops::gemm(ctx, args).unwrap();
    };

    let tuning = Instant::now();
    run();
    let tuning = tuning.elapsed().as_secs_f64() * 1e3;

    for _ in 0..3 {
        run();
    }
    ctx.synchronize().unwrap();

    const ITERATIONS: usize = 20;
    let mut samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        run();
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let median = samples[ITERATIONS / 2];
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    println!(
        "{label:<34} [{m}, {k}] x [{k}, {n}] -> {output_dtype:?}  \
         first(incl tune) {tuning:8.2} ms  median {median:7.3} ms  \
         best {:7.3} ms  {:6.1} TFLOP/s",
        samples[0],
        flops / (median * 1e-3) / 1e12,
    );
}

#[test]
#[ignore = "benchmark; needs a GPU and is not a correctness check"]
fn fp8_projection_rate() {
    let ctx = CudaContext::new(0).unwrap();
    // The four FP8 projection shapes a 2048-token GDN prefill runs, largest
    // first. QKV_WIDTH=10240, Z_WIDTH=6144, HIDDEN=5120.
    measure(&ctx, "gdn in_proj_qkv", 2048, 5120, 10240, DType::BF16);
    measure(&ctx, "gdn in_proj_z", 2048, 5120, 6144, DType::BF16);
    measure(&ctx, "gdn out_proj", 2048, 6144, 5120, DType::BF16);
    measure(&ctx, "attention q_proj", 2048, 5120, 12288, DType::BF16);
    // The F16 arm is what the CUTLASS candidate accepted before this branch;
    // it bounds how much of any BF16 gap is the output dtype itself.
    measure(&ctx, "gdn in_proj_qkv (f16 out)", 2048, 5120, 10240, DType::F16);
}
