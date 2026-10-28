//! Shared backend-framework contract, integration, and regression tests.
//!
//! Adding an L3 operator normally does not require editing this file. Extend it
//! only when the operator introduces or changes shared behavior such as keys,
//! candidate selection, alignment, caching, resources, or lifetimes.

use super::*;
use crate::{CudaBuffer, CudaContext};
use apxinf_core::{DType, Shape, Tensor};
use half::{bf16, f16};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

unsafe extern "C" {
    fn apxinf_gemm_test_hardware_fingerprint(
        uuid: *const u8,
        uuid_size: usize,
        multiprocessor_count: i32,
        total_global_memory: u64,
        memory_clock_rate: i32,
        performance: c_int,
        output: *mut c_char,
        capacity: usize,
    ) -> usize;
    fn apxinf_gemm_test_resource_prefilter(device: c_int) -> c_int;
    fn apxinf_framework_test_registry_contract(error: *mut c_char, capacity: usize) -> c_int;
    fn apxinf_framework_test_recipe_db_contract(
        directory: *const c_char,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn apxinf_framework_test_autotune_contract(
        graph_safe: c_int,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
}

fn run_framework_contract(name: &str, invoke: impl FnOnce(*mut c_char, usize) -> c_int) {
    let mut error = vec![0 as c_char; 4096];
    let passed = invoke(error.as_mut_ptr(), error.len());
    let detail = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    assert_eq!(passed, 1, "{name} failed: {detail}");
}

struct FrameworkTestDirectory(PathBuf);

impl FrameworkTestDirectory {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock predates UNIX epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "apxinf-framework-recipe-contract-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path)
            .unwrap_or_else(|error| panic!("create {}: {error}", path.display()));
        Self(path)
    }
}

impl Drop for FrameworkTestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn registry_requires_the_exact_identity_triple() {
    run_framework_contract("registry exact identity", |error, capacity| unsafe {
        apxinf_framework_test_registry_contract(error, capacity)
    });
}

#[test]
fn recipe_db_round_trips_updates_and_rejects_invalid_reads() {
    let directory = FrameworkTestDirectory::new();
    let path = CString::new(directory.0.to_string_lossy().as_bytes()).unwrap();
    run_framework_contract("recipe DB persistence", |error, capacity| unsafe {
        apxinf_framework_test_recipe_db_contract(path.as_ptr(), error, capacity)
    });
}

#[test]
fn autotune_filters_failures_and_times_every_valid_configuration() {
    run_framework_contract("CUDA autotune selection", |error, capacity| unsafe {
        apxinf_framework_test_autotune_contract(0, error, capacity)
    });
}

#[test]
fn graph_safe_autotune_captures_the_winner() {
    run_framework_contract("CUDA Graph-safe autotune", |error, capacity| unsafe {
        apxinf_framework_test_autotune_contract(1, error, capacity)
    });
}

#[test]
fn merged_gemm_contract_rejects_pre_merge_spec_version() {
    let ctx = CudaContext::new(0).unwrap();
    let activation = zeros_tensor(0, vec![1, 16], DType::BF16);
    let weight = zeros_tensor(0, vec![16, 16], DType::BF16);
    let mut output = zeros_tensor(0, vec![1, 16], DType::BF16);
    let args = GemmArgs::new(&activation, &weight, &mut output);
    let mut normalized =
        super::contracts::normalize(&ctx, args, super::contracts::Semantic::Gemm, None).unwrap();
    assert_eq!(normalized.spec.version, 8);
    normalized.spec.version = 7;
    let error = match super::execution::prepare(&ctx, normalized) {
        Ok(_) => panic!("pre-merge GEMM spec must be rejected"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("invalid GEMM Spec"), "{error}");
}

fn hardware_fingerprint(
    uuid: [u8; 16],
    sms: i32,
    memory: u64,
    memory_clock_rate: i32,
    performance: bool,
) -> String {
    let mut output = vec![0 as c_char; 512];
    let length = unsafe {
        apxinf_gemm_test_hardware_fingerprint(
            uuid.as_ptr(),
            uuid.len(),
            sms,
            memory,
            memory_clock_rate,
            if performance { 1 } else { 0 },
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert!(length < output.len());
    unsafe { CStr::from_ptr(output.as_ptr()) }
        .to_str()
        .unwrap()
        .to_owned()
}

#[test]
fn uuid_does_not_partition_persistent_tuning_cache() {
    let first = [0x11; 16];
    let second = [0xee; 16];
    assert_eq!(
        hardware_fingerprint(first, 14, 32 << 30, 1_500_000, true),
        hardware_fingerprint(second, 14, 32 << 30, 1_500_000, true),
        "equivalent GPUs with different UUIDs must share tuned recipes"
    );
    assert_eq!(
        hardware_fingerprint(first, 14, 32 << 30, 1_500_000, false),
        hardware_fingerprint(second, 14, 32 << 30, 1_500_000, false)
    );
}

#[test]
fn workspace_budget_rejects_before_provider_create() {
    assert_eq!(unsafe { apxinf_gemm_test_resource_prefilter(0) }, 1);
}

pub(super) fn tensor(device: usize, shape: Vec<usize>, values: &[f32]) -> Tensor {
    let host: Vec<_> = values.iter().map(|value| bf16::from_f32(*value)).collect();
    let bytes = unsafe {
        std::slice::from_raw_parts(
            host.as_ptr().cast::<u8>(),
            host.len() * std::mem::size_of::<bf16>(),
        )
    };
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}

pub(super) fn f16_tensor(device: usize, shape: Vec<usize>, values: &[f32]) -> Tensor {
    let host: Vec<_> = values.iter().map(|value| f16::from_f32(*value)).collect();
    let bytes = unsafe {
        std::slice::from_raw_parts(
            host.as_ptr().cast::<u8>(),
            host.len() * std::mem::size_of::<f16>(),
        )
    };
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::F16).unwrap()
}

pub(super) fn bf16_bits_tensor(device: usize, shape: Vec<usize>, values: &[u16]) -> Tensor {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    bytes_tensor(device, shape, DType::BF16, &bytes)
}

pub(super) fn f32_tensor(device: usize, shape: Vec<usize>, values: &[f32]) -> Tensor {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    bytes_tensor(device, shape, DType::F32, &bytes)
}

pub(super) fn zeros_tensor(device: usize, shape: Vec<usize>, dtype: DType) -> Tensor {
    let bytes = vec![0; shape.iter().product::<usize>() * dtype.size_in_bytes()];
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), dtype).unwrap()
}

pub(super) fn bytes_tensor(device: usize, shape: Vec<usize>, dtype: DType, bytes: &[u8]) -> Tensor {
    assert_eq!(
        bytes.len(),
        shape.iter().product::<usize>() * dtype.size_in_bytes()
    );
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), dtype).unwrap()
}

pub(super) fn scales(device: usize, values: &[f32]) -> Tensor {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer
        .as_tensor(Shape::new(vec![values.len()]), DType::F32)
        .unwrap()
}

fn prepare_test_gemm(
    ctx: &CudaContext,
    args: GemmArgs<'_>,
) -> apxinf_core::Result<std::rc::Rc<super::execution::Execution>> {
    super::execution::prepare(
        ctx,
        super::contracts::normalize(ctx, args, super::contracts::Semantic::Gemm, None)?,
    )
}

#[test]
fn gemm_rejects_partially_overlapping_output_storage() {
    let ctx = CudaContext::new(0).unwrap();
    let backing = CudaBuffer::alloc(12, 0).unwrap();
    let a = backing
        .view(0, 8)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 2]), DType::BF16)
        .unwrap();
    let b = tensor(0, vec![2, 2], &[1.0; 4]);
    let mut out = backing
        .view(4, 8)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 2]), DType::BF16)
        .unwrap();

    let error = gemm(&ctx, GemmArgs::new(&a, &b, &mut out)).unwrap_err();
    assert!(error.to_string().contains("overlaps read-only A storage"));
}

#[test]
fn gemm_rejects_storage_misaligned_for_its_dtype() {
    let ctx = CudaContext::new(0).unwrap();
    let backing = CudaBuffer::alloc(9, 0).unwrap();
    let a = backing
        .view(1, 8)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 2]), DType::BF16)
        .unwrap();
    let b = tensor(0, vec![2, 2], &[1.0; 4]);
    let mut out = tensor(0, vec![2, 2], &[0.0; 4]);

    let error = gemm(&ctx, GemmArgs::new(&a, &b, &mut out)).unwrap_err();
    assert!(error.to_string().contains("2-byte dtype"), "{error}");
}

#[test]
fn gpu_e2e_candidate_alignment_is_selected_and_keyed_from_actual_bindings() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (64, 64, 64);
    let a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &vec![0x38; m * k]);
    let b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &vec![0x38; k * n]);

    // Seed the in-memory Recipe cache with the naturally aligned binding class.
    let mut aligned_out = zeros_tensor(0, vec![m, n], DType::F16);
    let mut aligned_args = GemmArgs::new(&a, &b, &mut aligned_out);
    aligned_args.quantization = GemmQuantization::Fp8UnitScale;
    aligned_args.policy.allow_fallback = false;
    // Empty cache_dir disables recipe persistence; this test must observe a
    // fresh tune, not a recipe left in the default shared directory.
    aligned_args.policy.cache_dir = Some(String::new());
    let aligned =
        super::contracts::normalize(&ctx, aligned_args, super::contracts::Semantic::Gemm, None)
            .unwrap();
    assert!(aligned.spec.output_alignment >= 16);
    let aligned_instance = super::execution::prepare(&ctx, aligned).unwrap();
    eprintln!("GPU_ALIGNMENT_ALIGNED {}", aligned_instance.summary());

    // Offset by two bytes: valid for F16, but below CUTLASS's 16-byte output
    // requirement.  This must form a different Recipe key and skip CUTLASS.
    let output_bytes = m * n * DType::F16.size_in_bytes();
    let backing = CudaBuffer::alloc(output_bytes + 2, 0).unwrap();
    let mut misaligned_out = backing
        .view(2, output_bytes)
        .unwrap()
        .as_tensor(Shape::new(vec![m, n]), DType::F16)
        .unwrap();
    let mut misaligned_args = GemmArgs::new(&a, &b, &mut misaligned_out);
    misaligned_args.quantization = GemmQuantization::Fp8UnitScale;
    misaligned_args.policy.allow_fallback = false;
    misaligned_args.policy.cache_dir = Some(String::new());
    let misaligned = super::contracts::normalize(
        &ctx,
        misaligned_args,
        super::contracts::Semantic::Gemm,
        None,
    )
    .unwrap();
    assert_eq!(misaligned.spec.output_alignment, 2);
    let execution = super::execution::prepare(&ctx, misaligned).unwrap();
    let summary = execution.summary().to_owned();
    eprintln!("GPU_ALIGNMENT_MISALIGNED {summary}");
    assert!(summary.contains("cutlass-fp8=skip(alignment)"), "{summary}");
    assert!(
        summary.contains("cublasLt+custom-epilogue=skip(alignment)"),
        "{summary}"
    );
    assert!(
        summary.contains("cublasLt-native-fp8+custom-epilogue=skip(alignment)"),
        "{summary}"
    );
    assert!(
        !summary.contains("source=memory"),
        "alignment key aliased: {summary}"
    );

    execution.enqueue().unwrap();
    ctx.synchronize().unwrap();
    assert!(f16_values(&misaligned_out)
        .iter()
        .all(|&value| value == k as f32));
}

pub(super) fn values(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|value| bf16::from_bits(u16::from_ne_bytes([value[0], value[1]])).to_f32())
        .collect()
}

fn f16_values(tensor: &Tensor) -> Vec<f32> {
    assert_eq!(tensor.dtype(), DType::F16);
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|value| half::f16::from_bits(u16::from_ne_bytes([value[0], value[1]])).to_f32())
        .collect()
}

fn f32_values(tensor: &Tensor) -> Vec<f32> {
    assert_eq!(tensor.dtype(), DType::F32);
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
        .collect()
}

#[test]
fn fp8_unit_scale_f32_output_does_not_round_through_f16() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (1, 16, 16);
    // E4M3 448 is 0x7e. The dot product is finite in F32 but overflows F16.
    let a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &vec![0x7e; m * k]);
    let b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &vec![0x7e; k * n]);
    let mut out = zeros_tensor(0, vec![m, n], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    args.policy.online_tune = false;

    gemm(&ctx, args).unwrap();

    let expected = 448.0 * 448.0 * k as f32;
    assert_eq!(expected, 3_211_264.0);
    assert!(f32_values(&out).iter().all(|&value| value == expected));
}

#[test]
fn fp8_unit_scale_bf16_output_does_not_round_through_f16() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (1, 16, 16);
    // BF16 can represent this dot product exactly, while F16 overflows it.
    let a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &vec![0x7e; m * k]);
    let b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &vec![0x7e; k * n]);
    let mut out = zeros_tensor(0, vec![m, n], DType::BF16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    args.policy.online_tune = true;
    args.policy.allow_fallback = false;

    let normalized =
        super::contracts::normalize(&ctx, args, super::contracts::Semantic::Gemm, None).unwrap();
    let expected = 448.0 * 448.0 * k as f32;
    assert_eq!(expected, 3_211_264.0);
    super::execution::validate_candidates(&ctx, &normalized, &vec![expected; m * n]).unwrap();
    let execution = super::execution::prepare(&ctx, normalized).unwrap();

    execution.enqueue().unwrap();
    ctx.synchronize().unwrap();

    assert!(values(&out).iter().all(|&value| value == expected));
}

/// Decode one OCP E4M3 code (bias 7) exactly, so the reference below is the
/// arithmetic the kernels are supposed to perform rather than an approximation
/// of it.
fn e4m3_value(code: u8) -> f32 {
    let exponent = ((code >> 3) & 0x0F) as i32;
    let mantissa = (code & 0x07) as f32;
    let magnitude = if exponent == 0 {
        mantissa / 8.0 * (-6.0f32).exp2()
    } else {
        (1.0 + mantissa / 8.0) * ((exponent - 7) as f32).exp2()
    };
    if code & 0x80 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Every FP8 candidate eligible for a BF16 projection must reproduce the
/// projection at a shape with more than one tile.
///
/// `fp8_unit_scale_bf16_output_does_not_round_through_f16` also runs
/// `validate_candidates`, but at M=1, K=16, N=16 -- one tile, one K iteration,
/// and a single output row. A mainloop or epilogue that indexes wrongly
/// across tiles passes that and fails here. This matters because the CUTLASS
/// FP8 candidate accepts BF16 outputs, which is every projection in the
/// Qwen3.8 checkpoint, and `validate_candidates` rejects an implementation
/// that has no executable configuration -- so this also proves the candidate
/// is genuinely reached rather than silently skipped.
#[test]
fn gpu_e2e_bf16_fp8_candidates_reproduce_a_multi_tile_projection() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (128usize, 512usize, 256usize);
    // A spread of finite E4M3 codes. Constant operands would hide an indexing
    // fault, because every element of the result would be identical.
    const CODES: [u8; 8] = [0x38, 0x3c, 0x34, 0x40, 0xb8, 0x30, 0x44, 0xbc];
    let a_codes: Vec<u8> = (0..m * k)
        .map(|i| CODES[(i * 7 + 1) % CODES.len()])
        .collect();
    let b_codes: Vec<u8> = (0..k * n)
        .map(|i| CODES[(i * 5 + 3) % CODES.len()])
        .collect();
    let a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &a_codes);
    let b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &b_codes);
    let mut out = zeros_tensor(0, vec![m, n], DType::BF16);

    // The checkpoint-shaped alpha: `weight_scale * input_scale`, not unit.
    let alpha = 0.0123_f32;
    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut accumulator = 0.0f32;
            for index in 0..k {
                accumulator +=
                    e4m3_value(a_codes[row * k + index]) * e4m3_value(b_codes[index * n + column]);
            }
            expected[row * n + column] = accumulator * alpha;
        }
    }
    // Guard the magnitude as well as the direction: a candidate that is the
    // right shape and the wrong scale scores near-perfect cosine.
    let peak = expected.iter().fold(0.0f32, |best, v| best.max(v.abs()));
    assert!(
        peak > 0.1 && peak < 1.0e4,
        "degenerate reference peak {peak}"
    );

    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    args.alpha = alpha;
    args.policy.online_tune = true;
    args.policy.allow_fallback = false;
    args.policy.cache_dir = Some(String::new());
    let normalized =
        super::contracts::normalize(&ctx, args, super::contracts::Semantic::Gemm, None).unwrap();
    super::execution::validate_candidates(&ctx, &normalized, &expected).unwrap();

    // And the candidate the tuner actually selects has to land there too.
    let execution = super::execution::prepare(&ctx, normalized).unwrap();
    let summary = execution.summary().to_owned();
    eprintln!("GPU_BF16_FP8_CANDIDATES {summary}");
    assert!(
        summary.contains("cutlass-fp8#"),
        "the CUTLASS FP8 candidate was not reached for a BF16 output: {summary}"
    );
    execution.enqueue().unwrap();
    ctx.synchronize().unwrap();
    let produced = values(&out);
    let mut error_energy = 0.0f64;
    let mut signal_energy = 0.0f64;
    for (got, want) in produced.iter().zip(&expected) {
        error_energy += (*got as f64 - *want as f64).powi(2);
        signal_energy += (*want as f64).powi(2);
    }
    let relative_l2 = (error_energy / signal_energy).sqrt();
    eprintln!("GPU_BF16_FP8_RELATIVE_L2 {relative_l2:.6}");
    // BF16 carries 8 mantissa bits, so a correct kernel lands near 1e-3. The
    // bound is deliberately far above that and far below any real fault.
    assert!(
        relative_l2 < 0.01,
        "winner drifted by relative L2 {relative_l2}"
    );
}

#[test]
fn gpu_e2e_autotune_times_candidates_and_checks_winner_capture() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (64, 64, 64);
    // E4M3 1.0 is 0x38.  This contract is supported by cuBLAS, cuBLASLt,
    // cuBLASLt native FP8, and the SM100-family CUTLASS implementation.
    let a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &vec![0x38; m * k]);
    let b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &vec![0x38; k * n]);
    let mut out = zeros_tensor(0, vec![m, n], DType::F16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    args.policy.online_tune = true;
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
    // Empty cache_dir disables recipe persistence; this test asserts on a
    // fresh tune summary and must not read the default shared directory.
    args.policy.cache_dir = Some(String::new());

    let normalized =
        super::contracts::normalize(&ctx, args, super::contracts::Semantic::Gemm, None).unwrap();
    let execution = super::execution::prepare(&ctx, normalized).unwrap();
    let summary = execution.summary().to_owned();
    eprintln!("GPU_E2E_CANDIDATES {summary}");
    assert!(
        summary.contains("winner-graph=capture-pass"),
        "the eager winner was not validated for CUDA Graph capture: {summary}"
    );
    assert!(
        summary.contains("=timed("),
        "no candidate was timed: {summary}"
    );

    execution.enqueue().unwrap();
    ctx.synchronize().unwrap();
    assert!(f16_values(&out).iter().all(|&value| value == k as f32));
}

#[test]
fn gpu_e2e_graph_replay_overwrites_sentinel_and_matches_numeric_result() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (7, 19, 22);
    let a = tensor(0, vec![m, k], &vec![1.0; m * k]);
    let b = tensor(0, vec![k, n], &vec![1.0; k * n]);
    let mut out = tensor(0, vec![m, n], &vec![0.0; m * n]);
    let observed = CudaBuffer::from_tensor(&out).unwrap();
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;

    let normalized =
        super::contracts::normalize(&ctx, args, super::contracts::Semantic::Gemm, None).unwrap();
    let execution = super::execution::prepare(&ctx, normalized).unwrap();
    let graph = crate::capture(&ctx, || execution.enqueue()).unwrap();

    let sentinel: Vec<u8> = (0..m * n)
        .flat_map(|_| bf16::from_f32(-123.0).to_bits().to_ne_bytes())
        .collect();
    observed.copy_from_host(&sentinel).unwrap();
    assert!(values(&out).iter().all(|&value| value == -123.0));

    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    let actual = values(&out);
    assert!(
        actual.iter().all(|&value| value == k as f32),
        "captured graph replay did not overwrite the sentinel: {actual:?}"
    );
    eprintln!("GPU_E2E_GRAPH replay overwrote sentinel and matched {k}");
}

#[test]
fn bf16_gemm_numeric_recipe_and_graph() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (7, 19, 22);
    let a_values: Vec<_> = (0..m * k)
        .map(|i| ((i * 7 % 17) as f32 - 8.0) / 16.0)
        .collect();
    let b_values: Vec<_> = (0..k * n)
        .map(|i| ((i * 3 % 13) as f32 - 6.0) / 16.0)
        .collect();
    let a = tensor(0, vec![m, k], &a_values);
    let b = tensor(0, vec![k, n], &b_values);
    let mut out = tensor(0, vec![m, n], &vec![0.0; m * n]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.cache_dir = Some("/tmp/apxinf-gemm-design-tests".into());
    let execution = super::execution::prepare(
        &ctx,
        super::contracts::normalize(&ctx, args, super::contracts::Semantic::Gemm, None).unwrap(),
    )
    .unwrap();
    eprintln!("GEMM {}", execution.summary());
    execution.enqueue().unwrap();
    ctx.synchronize().unwrap();
    let actual = values(&out);
    for row in 0..m {
        for column in 0..n {
            let expected: f32 = (0..k)
                .map(|inner| a_values[row * k + inner] * b_values[inner * n + column])
                .sum();
            assert!((actual[row * n + column] - expected).abs() < 0.01);
        }
    }

    let mut cached_args = GemmArgs::new(&a, &b, &mut out);
    cached_args.policy.cache_dir = Some("/tmp/apxinf-gemm-design-tests".into());
    cached_args.policy.online_tune = false;
    cached_args.policy.allow_fallback = false;
    gemm(&ctx, cached_args).unwrap();

    let graph = crate::capture(&ctx, || execution.enqueue()).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), actual);
    drop(a);
    drop(b);
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), actual);
}

fn scratch_cache_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "apxinf-{}-{}-{}",
        label,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A Recipe is shared by every caller whose Spec matches, and alpha is no longer
/// part of that Spec. Executing the same shape twice with different alphas
/// must therefore still scale each result independently: if alpha were cached
/// inside the Execution the second execution would silently reuse the first one.
#[test]
fn gpu_e2e_alpha_is_bound_per_execution_not_baked_into_the_recipe() {
    let cache_dir = scratch_cache_dir("gemm-alpha-binding");
    let cache = cache_dir.to_string_lossy().into_owned();
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (5, 11, 8);
    let a_values: Vec<f32> = (0..m * k)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) / 8.0)
        .collect();
    let b_values: Vec<f32> = (0..k * n)
        .map(|i| ((i * 3 % 11) as f32 - 5.0) / 8.0)
        .collect();
    let a = tensor(0, vec![m, k], &a_values);
    let b = tensor(0, vec![k, n], &b_values);

    let run = |alpha: f32| {
        let mut out = tensor(0, vec![m, n], &vec![0.0; m * n]);
        let mut args = GemmArgs::new(&a, &b, &mut out);
        args.alpha = alpha;
        args.policy.cache_dir = Some(cache.clone());
        let execution = super::execution::prepare(
            &ctx,
            super::contracts::normalize(&ctx, args, super::contracts::Semantic::Gemm, None)
                .unwrap(),
        )
        .unwrap();
        let summary = execution.summary().to_owned();
        execution.enqueue().unwrap();
        ctx.synchronize().unwrap();
        (values(&out), summary)
    };
    let expected = |alpha: f32| -> Vec<f32> {
        let mut values = vec![0.0f32; m * n];
        for row in 0..m {
            for column in 0..n {
                values[row * n + column] = alpha
                    * (0..k)
                        .map(|inner| a_values[row * k + inner] * b_values[inner * n + column])
                        .sum::<f32>();
            }
        }
        values
    };

    let (doubled, first) = run(2.0);
    let (tripled, second) = run(3.0);
    eprintln!("GPU_E2E_ALPHA first=[{first}] second=[{second}]");
    for (index, want) in expected(2.0).into_iter().enumerate() {
        assert!(
            (doubled[index] - want).abs() < 0.02,
            "alpha=2 element {index} is {} but should be {want}",
            doubled[index]
        );
    }
    for (index, want) in expected(3.0).into_iter().enumerate() {
        assert!(
            (tripled[index] - want).abs() < 0.02,
            "alpha=3 element {index} is {} but should be {want} \
             (a shared Recipe reused the previous alpha)",
            tripled[index]
        );
    }
    assert_ne!(
        doubled, tripled,
        "alpha must change the numeric result even when the Recipe is shared"
    );
    std::fs::remove_dir_all(cache_dir).unwrap();
}

/// Only the unit/non-unit scale predicate belongs to the tuning key, so calls
/// that differ solely by alpha must land on one recipe instead of tuning once
/// per calibration scale.
#[test]
fn gpu_e2e_scale_values_do_not_fragment_the_tuning_cache() {
    let cache_dir = scratch_cache_dir("gemm-scale-fragmentation");
    let cache = cache_dir.to_string_lossy().into_owned();
    let (m, k, n) = (6, 9, 12);
    let run = |ctx: &CudaContext, alpha: f32, online_tune: bool| -> String {
        let a = tensor(0, vec![m, k], &vec![0.5; m * k]);
        let b = tensor(0, vec![k, n], &vec![0.25; k * n]);
        let mut out = tensor(0, vec![m, n], &vec![0.0; m * n]);
        let mut args = GemmArgs::new(&a, &b, &mut out);
        args.alpha = alpha;
        args.policy.cache_dir = Some(cache.clone());
        args.policy.online_tune = online_tune;
        args.policy.allow_fallback = false;
        let execution = super::execution::prepare(
            ctx,
            super::contracts::normalize(ctx, args, super::contracts::Semantic::Gemm, None).unwrap(),
        )
        .unwrap();
        execution.summary().to_owned()
    };

    let ctx = CudaContext::new(0).unwrap();
    let tuned = run(&ctx, 2.0, true);
    assert!(
        tuned.contains("source=tuned"),
        "the first call must actually tune: {tuned}"
    );
    let reused = run(&ctx, 3.0, true);
    assert!(
        reused.contains("source=memory"),
        "a different alpha must hit the same tuned Recipe: {reused}"
    );
    // A fresh runtime has no in-memory Recipe, so this proves the persisted
    // recipe is shared too and that tuning is not repeated.
    let restored = run(&CudaContext::new(0).unwrap(), 4.0, false);
    assert!(
        restored.contains("source=recipe"),
        "a different alpha must restore the persisted recipe: {restored}"
    );
    std::fs::remove_dir_all(cache_dir).unwrap();
}

#[test]
fn gpu_e2e_persisted_recipe_restores_in_new_process() {
    const CHILD_CACHE: &str = "APXINF_RECIPE_RESTORE_CHILD";
    let run = |cache: String, online_tune: bool| {
        let ctx = CudaContext::new(0).unwrap();
        let a = tensor(0, vec![4, 8], &[0.5; 32]);
        let b = tensor(0, vec![8, 6], &[0.25; 48]);
        let mut out = tensor(0, vec![4, 6], &[0.0; 24]);
        let mut args = GemmArgs::new(&a, &b, &mut out);
        args.policy.cache_dir = Some(cache);
        args.policy.online_tune = online_tune;
        args.policy.allow_fallback = false;
        prepare_test_gemm(&ctx, args).unwrap().summary().to_owned()
    };

    if let Ok(cache) = std::env::var(CHILD_CACHE) {
        let restored = run(cache, false);
        assert!(
            restored.contains("source=recipe"),
            "child process did not restore the persisted Recipe: {restored}"
        );
        return;
    }

    let cache_dir = scratch_cache_dir("gemm-process-recipe");
    let cache = cache_dir.to_string_lossy().into_owned();
    let tuned = run(cache.clone(), true);
    assert!(tuned.contains("source=tuned"), "{tuned}");

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "ops::tests::framework::gpu_e2e_persisted_recipe_restores_in_new_process",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_CACHE, &cache)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Recipe restore child failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(cache_dir).unwrap();
}

#[test]
fn gpu_e2e_default_cache_and_autotune_switches() {
    const CHILD_PHASE: &str = "APXINF_DEFAULT_RECIPE_CHILD";
    if let Ok(phase) = std::env::var(CHILD_PHASE) {
        let allow_tune = !matches!(phase.as_str(), "cold" | "restore" | "override-restore");
        let ctx = CudaContext::new_with_autotune(0, allow_tune).unwrap();
        let default_dir = std::path::Path::new(ctx.default_cache_dir());
        assert!(default_dir.starts_with("configs/tuning/nvidia"));
        if phase == "cold" {
            std::fs::create_dir_all(default_dir).unwrap();
            std::fs::write(default_dir.join("tactics.json"), "not a recipe database").unwrap();
        } else if phase == "restore" {
            std::fs::remove_file(default_dir.join("tactics.json")).unwrap();
        }
        let cache_dir = match phase.as_str() {
            "override" | "override-restore" => Some("explicit-recipes".to_owned()),
            "no-persistence" => Some(String::new()),
            _ => None,
        };
        let online_tune = phase != "policy-off";
        let expected = match phase.as_str() {
            "cold" | "policy-off" => "source=fallback",
            "restore" | "warm-enabled" | "override-restore" => "source=recipe",
            _ => "source=tuned",
        };
        let input = tensor(0, vec![4, 8], &[0.5; 32]);
        let weight = tensor(0, vec![8, 6], &[0.25; 48]);
        let mut output = tensor(0, vec![4, 6], &[0.0; 24]);
        let mut args = GemmArgs::new(&input, &weight, &mut output);
        args.policy.cache_dir = cache_dir.clone();
        args.policy.online_tune = online_tune;
        let execution = prepare_test_gemm(&ctx, args).unwrap();
        eprintln!("{phase} GEMM: {}", execution.summary());
        assert!(
            execution.summary().contains(expected),
            "{}",
            execution.summary()
        );
        execution.enqueue().unwrap();
        ctx.synchronize().unwrap();
        assert!(values(&output)
            .iter()
            .all(|value| (*value - 1.0).abs() < 0.01));

        let query = tensor(0, vec![1, 4, 1, 32], &[0.5; 128]);
        let key = tensor(0, vec![1, 4, 1, 32], &[0.25; 128]);
        let value = tensor(0, vec![1, 4, 1, 32], &[0.75; 128]);
        let mut output = tensor(0, vec![1, 4, 1, 32], &[0.0; 128]);
        let mut args = AttentionArgs::new(&query, &key, &value, &mut output);
        args.policy.cache_dir = cache_dir;
        args.policy.online_tune = online_tune;
        let normalized = crate::ops::attention::contracts::normalize(&ctx, args).unwrap();
        let execution = crate::ops::attention::execution::prepare(&ctx, normalized).unwrap();
        eprintln!("{phase} Attention: {}", execution.summary());
        assert!(
            execution.summary().contains(expected),
            "{}",
            execution.summary()
        );
        execution.enqueue_for_test().unwrap();
        ctx.synchronize().unwrap();
        assert!(values(&output)
            .iter()
            .all(|value| (*value - 0.75).abs() < 0.01));
        return;
    }

    fn recipes(
        directory: &std::path::Path,
    ) -> std::collections::BTreeMap<PathBuf, (Vec<u8>, SystemTime)> {
        let mut snapshot = std::collections::BTreeMap::new();
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                snapshot.extend(recipes(&path));
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("recipe") {
                snapshot.insert(
                    path.clone(),
                    (
                        std::fs::read(&path).unwrap(),
                        std::fs::metadata(path).unwrap().modified().unwrap(),
                    ),
                );
            }
        }
        snapshot
    }

    let directory = FrameworkTestDirectory::new();
    let mut previous: std::collections::BTreeMap<PathBuf, (Vec<u8>, SystemTime)> =
        std::collections::BTreeMap::new();
    for (phase, expected_count) in [
        ("cold", 0),
        ("policy-off", 0),
        ("tune", 2),
        ("restore", 2),
        ("warm-enabled", 2),
        ("override", 4),
        ("override-restore", 4),
        ("no-persistence", 4),
    ] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "ops::tests::framework::gpu_e2e_default_cache_and_autotune_switches",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD_PHASE, phase)
            .current_dir(&directory.0)
            .output()
            .unwrap();
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        assert!(
            output.status.success(),
            "{phase}: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        let current = recipes(&directory.0);
        assert_eq!(current.len(), expected_count, "{phase}");
        for (path, content) in &previous {
            assert_eq!(
                current.get(path),
                Some(content),
                "{phase} rewrote {}",
                path.display()
            );
        }
        previous = current;
    }
}

#[test]
fn gpu_e2e_invalid_persisted_recipe_is_retuned() {
    let cache_dir = scratch_cache_dir("gemm-invalid-recipe");
    let cache = cache_dir.to_string_lossy().into_owned();
    let run = |ctx: &CudaContext| {
        let a = tensor(0, vec![4, 8], &[0.5; 32]);
        let b = tensor(0, vec![8, 6], &[0.25; 48]);
        let mut out = tensor(0, vec![4, 6], &[0.0; 24]);
        let mut args = GemmArgs::new(&a, &b, &mut out);
        args.policy.cache_dir = Some(cache.clone());
        args.policy.online_tune = true;
        args.policy.allow_fallback = false;
        prepare_test_gemm(ctx, args).unwrap().summary().to_owned()
    };

    let first = run(&CudaContext::new(0).unwrap());
    assert!(first.contains("source=tuned"), "{first}");
    for entry in std::fs::read_dir(&cache_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|value| value.to_str()) != Some("recipe") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let key = text.lines().next().unwrap();
        std::fs::write(path, format!("{key}\n999 999 999 999\n")).unwrap();
    }

    let restored = run(&CudaContext::new(0).unwrap());
    assert!(
        restored.contains("source=tuned"),
        "an invalid persisted Recipe was accepted instead of retuned: {restored}"
    );
    std::fs::remove_dir_all(cache_dir).unwrap();
}

#[test]
fn captured_graph_retains_session_execution_and_tensor_storage() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let observed = out.clone();
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    prepare_with_session(&session, || gemm(&ctx, GemmArgs::new(&a, &b, &mut out))).unwrap();

    let graph = crate::capture(&ctx, || {
        with_session(&session, || gemm(&ctx, GemmArgs::new(&a, &b, &mut out)))
    })
    .unwrap();

    drop(session);
    drop(a);
    drop(b);
    drop(out);
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert!(values(&observed).iter().all(|&value| value == 3.0));
}

#[test]
fn quantization_contract_becomes_part_of_the_gemm_key() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (2, 16, 8);
    let row_scales = scales(0, &[1.0; 2]);
    let channel_scales = scales(0, &[1.0; 8]);

    let fp8_a = zeros_tensor(0, vec![m, k], DType::F8E4M3);
    let fp8_b = zeros_tensor(0, vec![k, n], DType::F8E4M3);
    let mut fp8_out = zeros_tensor(0, vec![m, n], DType::BF16);
    let fp8 = super::contracts::normalize(
        &ctx,
        GemmArgs::fp8(&fp8_a, &row_scales, &fp8_b, &channel_scales, &mut fp8_out),
        super::contracts::Semantic::Gemm,
        None,
    )
    .unwrap();
    assert_eq!(fp8.spec.quantization, 2);
    assert_eq!(fp8.spec.accumulation_dtype, 0);

    let int8_a = zeros_tensor(0, vec![m, k], DType::I8);
    let int8_b = zeros_tensor(0, vec![k, n], DType::I8);
    let mut int8_out = zeros_tensor(0, vec![m, n], DType::BF16);
    let w8a8 = super::contracts::normalize(
        &ctx,
        GemmArgs::w8a8(
            &int8_a,
            &row_scales,
            &int8_b,
            &channel_scales,
            &mut int8_out,
        ),
        super::contracts::Semantic::Gemm,
        None,
    )
    .unwrap();
    assert_eq!(w8a8.spec.quantization, 3);
    assert_eq!(w8a8.spec.accumulation_dtype, 5);
}

#[test]
fn scaled_fp8_and_w8a8_use_the_canonical_kn_weight_contract() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (2, 16, 8);
    let row_scales = scales(0, &[0.5, 0.25]);
    let channel_scales = scales(0, &[2.0; 8]);

    // E4M3 1.0 is 0x38. Both rows therefore produce K * row * channel.
    let fp8_a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &vec![0x38; m * k]);
    let fp8_b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &vec![0x38; k * n]);
    let mut fp8_out = zeros_tensor(0, vec![m, n], DType::BF16);
    let mut fp8_args = GemmArgs::fp8(&fp8_a, &row_scales, &fp8_b, &channel_scales, &mut fp8_out);
    fp8_args.policy.online_tune = false;
    gemm(&ctx, fp8_args).unwrap();
    let fp8_values = values(&fp8_out);
    assert!(fp8_values[..n].iter().all(|&value| value == 16.0));
    assert!(fp8_values[n..].iter().all(|&value| value == 8.0));

    let int8_a = bytes_tensor(0, vec![m, k], DType::I8, &vec![1; m * k]);
    let int8_b = bytes_tensor(0, vec![k, n], DType::I8, &vec![1; k * n]);
    let mut int8_out = zeros_tensor(0, vec![m, n], DType::BF16);
    let mut int8_args = GemmArgs::w8a8(
        &int8_a,
        &row_scales,
        &int8_b,
        &channel_scales,
        &mut int8_out,
    );
    int8_args.policy.online_tune = false;
    gemm(&ctx, int8_args).unwrap();
    let int8_values = values(&int8_out);
    assert!(int8_values[..n].iter().all(|&value| value == 16.0));
    assert!(int8_values[n..].iter().all(|&value| value == 8.0));
}

#[test]
fn w8a8_orin_shape_uses_direct_cutlass_candidate() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (41, 1536, 6144);
    let mut a_data = vec![0; m * k];
    for row in 0..m {
        a_data[row * k + (row * 37) % k] = 1;
    }
    let b_data = (0..k * n)
        .map(|index| ((3 * (index / n) + 5 * (index % n)) % 7 + 1) as u8)
        .collect::<Vec<_>>();
    let row_scale_values = (0..m)
        .map(|row| if row % 2 == 0 { 0.5 } else { 1.0 })
        .collect::<Vec<_>>();
    let channel_scale_values = (0..n)
        .map(|column| if column % 2 == 0 { 2.0 } else { 1.0 })
        .collect::<Vec<_>>();
    let a = bytes_tensor(0, vec![m, k], DType::I8, &a_data);
    let b = bytes_tensor(0, vec![k, n], DType::I8, &b_data);
    let row_scales = scales(0, &row_scale_values);
    let channel_scales = scales(0, &channel_scale_values);
    let mut out = zeros_tensor(0, vec![m, n], DType::BF16);
    let weight = crate::ops::RepackedWeight::new(&b).unwrap();
    let mut args = GemmArgs::w8a8(&a, &row_scales, &b, &channel_scales, &mut out)
        .with_repacked_weight(&weight);
    args.policy.online_tune = false;
    args.policy.allow_fallback = ctx.caps().sm != 87;
    let cache_dir = scratch_cache_dir("w8a8-sm87-cutlass");
    args.policy.cache_dir = Some(cache_dir.to_string_lossy().into_owned());

    let normalized =
        super::contracts::normalize(&ctx, args, super::contracts::Semantic::Gemm, None).unwrap();
    if ctx.caps().sm == 87 {
        super::execution::seed_recipe(&ctx, &normalized, 3, 6, 1, 0).unwrap();
    }
    let execution = super::execution::prepare(&ctx, normalized).unwrap();
    if ctx.caps().sm == 87 {
        assert!(execution
            .summary()
            .starts_with("cutlass-w8a8-sm87 config=0 "));
        assert!(execution.summary().contains("source=recipe"));
    } else {
        assert!(execution
            .summary()
            .starts_with("cublas+custom-epilogue config=0 "));
        assert!(execution.summary().contains("source=fallback"));
    }

    execution.enqueue().unwrap();
    let graph = crate::capture(&ctx, || execution.enqueue()).unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    let actual = values(&out);
    for row in 0..m {
        let input = (row * 37) % k;
        for column in 0..n {
            let expected = b_data[input * n + column] as f32
                * row_scale_values[row]
                * channel_scale_values[column];
            assert_eq!(actual[row * n + column], expected);
        }
    }
    std::fs::remove_dir_all(cache_dir).unwrap();
}

#[test]
fn execution_session_reuses_one_native_execution_across_forward_traversals() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    super::execution::reset_execution_create_count();
    prepare_with_session(&session, || {
        let mut args = GemmArgs::new(&a, &b, &mut out);
        args.policy.online_tune = false;
        gemm(&ctx, args)
    })
    .unwrap();
    assert_eq!(super::execution::execution_create_count(), 1);
    super::execution::reset_execution_sync_count();

    for _ in 0..2 {
        with_session(&session, || {
            let mut args = GemmArgs::new(&a, &b, &mut out);
            args.policy.online_tune = false;
            gemm(&ctx, args)
        })
        .unwrap();
    }

    // The first public L3 traversal creates provider resources and performs the
    // one explicit enqueue. Repeating it must hit the session Execution.
    assert_eq!(super::execution::execution_create_count(), 1);
    assert_eq!(super::execution::execution_sync_count(), 0);
    assert!(values(&out).iter().all(|&value| value == 3.0));
}

#[test]
fn gpu_e2e_cached_recipe_rebuilds_and_drops_provider_private_executions() {
    let ctx = CudaContext::new(0).unwrap();

    for iteration in 0..4 {
        // Every iteration uses new A/B/output allocations. The computation
        // key and Recipe stay the same, while the bound Execution must be
        // rebuilt for the new addresses.
        let a = tensor(0, vec![8, 16], &[1.0; 8 * 16]);
        let b = tensor(0, vec![16, 8], &[1.0; 16 * 8]);
        let mut out = tensor(0, vec![8, 8], &[0.0; 8 * 8]);
        let mut args = GemmArgs::new(&a, &b, &mut out);
        args.policy.online_tune = true;
        args.policy.allow_fallback = false;

        let prepared = prepare_test_gemm(&ctx, args).unwrap();
        if iteration != 0 {
            assert!(
                prepared.summary().contains("source=memory-recipe"),
                "provider Execution was not rebuilt from the cached Recipe: {}",
                prepared.summary()
            );
        }
        prepared.enqueue().unwrap();
        ctx.synchronize().unwrap();
        drop(prepared);

        assert!(values(&out).iter().all(|&value| value == 16.0));
    }
}

#[test]
fn tuning_isolated_from_user_buffers_and_replay_results() {
    let cache_dir = scratch_cache_dir("acceptance-f11-tuning-isolation");
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0, 2.0, -1.0, 0.0, 3.0, 1.0]);
    let b = tensor(0, vec![3, 2], &[1.0, 2.0, 0.0, -1.0, 2.0, 1.0]);

    let a_buffer = CudaBuffer::from_tensor(&a).unwrap();
    let b_buffer = CudaBuffer::from_tensor(&b).unwrap();
    let mut a_before = vec![0; a_buffer.len()];
    let mut b_before = vec![0; b_buffer.len()];
    a_buffer.copy_to_host(&mut a_before).unwrap();
    b_buffer.copy_to_host(&mut b_before).unwrap();

    let output_bytes = 2 * 2 * DType::BF16.size_in_bytes();
    let backing = CudaBuffer::alloc(output_bytes + 4, 0).unwrap();
    let guard_word = bf16::from_f32(-77.0).to_bits().to_ne_bytes();
    let sentinel_word = bf16::from_f32(-123.0).to_bits().to_ne_bytes();
    let mut initial = Vec::with_capacity(output_bytes + 4);
    initial.extend_from_slice(&guard_word);
    for _ in 0..4 {
        initial.extend_from_slice(&sentinel_word);
    }
    initial.extend_from_slice(&guard_word);
    backing.copy_from_host(&initial).unwrap();

    let mut out = backing
        .view(2, output_bytes)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 2]), DType::BF16)
        .unwrap();
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.cache_dir = Some(cache_dir.to_string_lossy().into_owned());
    args.policy.online_tune = true;
    args.policy.allow_fallback = false;

    // Native prepare may tune and allocate provider resources, but it must not
    // execute the serving binding or otherwise touch caller-owned buffers.
    let execution = prepare_test_gemm(&ctx, args).unwrap();

    let mut after_prepare = vec![0; initial.len()];
    backing.copy_to_host(&mut after_prepare).unwrap();
    assert_eq!(
        after_prepare, initial,
        "prepare/tuning modified user output"
    );

    let mut a_after = vec![0; a_buffer.len()];
    let mut b_after = vec![0; b_buffer.len()];
    a_buffer.copy_to_host(&mut a_after).unwrap();
    b_buffer.copy_to_host(&mut b_after).unwrap();
    assert_eq!(a_after, a_before, "tuning modified input A");
    assert_eq!(b_after, b_before, "tuning modified input B");

    execution.enqueue().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), vec![-1.0, -1.0, 2.0, -2.0]);

    let mut final_bytes = vec![0; initial.len()];
    backing.copy_to_host(&mut final_bytes).unwrap();
    assert_eq!(&final_bytes[..2], &guard_word);
    assert_eq!(&final_bytes[final_bytes.len() - 2..], &guard_word);
    std::fs::remove_dir_all(cache_dir).unwrap();
}

#[test]
fn gpu_e2e_three_view_split_geglu_recipe_resources_and_replay() {
    let ctx = CudaContext::new(0).unwrap();
    let directory = FrameworkTestDirectory::new();
    let activation = zeros_tensor(0, vec![778, 2048], DType::F8E4M3);
    let weight = zeros_tensor(0, vec![2048, 32768], DType::F8E4M3);
    let mut output = zeros_tensor(0, vec![778, 16384], DType::F8E4M3);
    let mut normalize = |workspace_limit, allow_fallback| {
        let mut args = GemmArgs::new(&activation, &weight, &mut output)
            .with_immutable_weight(WeightVersion::new(1));
        args.quantization = GemmQuantization::Fp8UnitScale;
        args.policy.online_tune = false;
        args.policy.allow_fallback = allow_fallback;
        args.policy.workspace_limit = workspace_limit;
        args.policy.cache_dir = Some(directory.0.to_string_lossy().into_owned());
        super::contracts::normalize(&ctx, args, super::contracts::Semantic::GemmGeglu, None)
            .unwrap()
    };
    let budget = 256 * 1024 * 1024;
    let fallback = super::execution::prepare(&ctx, normalize(budget, true)).unwrap();
    assert!(fallback.summary().contains("cublas+custom-epilogue"));
    fallback.enqueue().unwrap();
    ctx.synchronize().unwrap();
    drop(fallback);
    for configuration in 0..3 {
        let normalized = normalize(budget, false);
        super::execution::seed_recipe(&ctx, &normalized, 3, 5, 1, configuration).unwrap();
        let execution = super::execution::prepare(&ctx, normalized).unwrap();
        assert!(execution.summary().contains("cublasLt+cutlass-split-geglu"));
        execution.enqueue().unwrap();
        let graph = crate::capture(&ctx, || execution.enqueue()).unwrap();
        graph.replay().unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
    }
    let normalized = normalize(budget, false);
    assert!(super::execution::seed_recipe(&ctx, &normalized, 3, 5, 1, 24).is_err());
    let insufficient = normalize(778 * 32768 * 2 - 1, false);
    super::execution::seed_recipe(&ctx, &insufficient, 3, 5, 1, 0).unwrap();
    assert!(super::execution::prepare(&ctx, insufficient).is_err());
    let mut negative_alpha = normalize(budget, false);
    negative_alpha.bindings.alpha = -1.0;
    negative_alpha.spec.alpha_is_unit = 0;
    super::execution::seed_recipe(&ctx, &negative_alpha, 3, 5, 1, 0).unwrap();
    assert!(super::execution::prepare(&ctx, negative_alpha).is_err());
    drop(normalize);
    ctx.synchronize().unwrap();
    let host = crate::transfers::to_cpu(&output).unwrap();
    assert!(host.as_f8_e4m3().unwrap().iter().all(|value| *value == 0));
    // T21 three-view rows (M=789) are also a pinned production shape; the
    // split-GeGLU candidate accepts them. An unrelated M stays rejected.
    let activation = zeros_tensor(0, vec![789, 2048], DType::F8E4M3);
    let mut output = zeros_tensor(0, vec![789, 16384], DType::F8E4M3);
    let mut args = GemmArgs::new(&activation, &weight, &mut output);
    args.quantization = GemmQuantization::Fp8UnitScale;
    args.policy.cache_dir = Some(directory.0.to_string_lossy().into_owned());
    let normalized =
        super::contracts::normalize(&ctx, args, super::contracts::Semantic::GemmGeglu, None)
            .unwrap();
    super::execution::seed_recipe(&ctx, &normalized, 3, 5, 1, 0).unwrap();
    let activation = zeros_tensor(0, vec![800, 2048], DType::F8E4M3);
    let mut output = zeros_tensor(0, vec![800, 16384], DType::F8E4M3);
    let mut args = GemmArgs::new(&activation, &weight, &mut output);
    args.quantization = GemmQuantization::Fp8UnitScale;
    args.policy.cache_dir = Some(directory.0.to_string_lossy().into_owned());
    let normalized =
        super::contracts::normalize(&ctx, args, super::contracts::Semantic::GemmGeglu, None)
            .unwrap();
    assert!(super::execution::seed_recipe(&ctx, &normalized, 3, 5, 1, 0).is_err());
}

#[test]
fn gpu_repacked_weight_rejects_independent_state_and_canonical_alias() {
    let ctx = CudaContext::new(0).unwrap();
    let b = zeros_tensor(0, vec![16, 16], DType::BF16);
    let alias = b.clone();
    let state = crate::ops::RepackedWeight::new(&b).unwrap();
    assert!(crate::ops::RepackedWeight::new(&alias).is_err());
    let shared = state.clone();
    assert!(std::sync::Arc::ptr_eq(&state.state, &shared.state));
    let view = CudaBuffer::from_tensor(&b)
        .unwrap()
        .view(32, 256)
        .unwrap()
        .as_tensor(Shape::new(vec![8, 16]), DType::BF16)
        .unwrap();
    assert!(crate::ops::RepackedWeight::new(&view).is_err());
    let independent = zeros_tensor(0, vec![16, 16], DType::BF16);
    assert!(crate::ops::RepackedWeight::new(&independent).is_ok());
    // Even after the public state is dropped, aliases must not become canonical.
    drop(state);
    drop(shared);
    for (k, binding) in [(16, &alias), (8, &view)] {
        let a = zeros_tensor(0, vec![1, k], DType::BF16);
        let mut out = zeros_tensor(0, vec![1, 16], DType::BF16);
        let normalized = super::contracts::normalize(
            &ctx,
            GemmArgs::new(&a, binding, &mut out),
            super::contracts::Semantic::Gemm,
            None,
        )
        .unwrap();
        let error = super::execution::prepare(&ctx, normalized)
            .err()
            .expect("canonical alias must fail");
        assert!(error.to_string().contains("layout state"), "{error}");
    }
}

#[test]
fn gpu_repacked_registration_does_not_retain_allocation() {
    let _ctx = CudaContext::new(0).unwrap();
    let b = zeros_tensor(0, vec![16, 16], DType::BF16);
    let buffer = CudaBuffer::from_tensor(&b).unwrap();
    let owner = std::sync::Arc::downgrade(buffer.allocation_owner());
    let state = crate::ops::RepackedWeight::new(&b).unwrap();
    drop(buffer);
    drop(b);
    assert!(owner.upgrade().is_some());
    drop(state);
    assert!(owner.upgrade().is_none(), "registry must not retain GPU memory");
    // A new allocation (including an address reused by CUDA) can be adopted.
    let next = zeros_tensor(0, vec![16, 16], DType::BF16);
    assert!(crate::ops::RepackedWeight::new(&next).is_ok());
}

#[test]
fn gpu_canonical_weight_does_not_opt_into_inplace_packing() {
    let ctx = CudaContext::new(0).unwrap();
    if ctx.caps().sm != 87 {
        return;
    }
    let (m, k, n) = (8, 256, 128);
    let a = bytes_tensor(0, vec![m, k], DType::I8, &vec![1; m * k]);
    let source = (0..k * n).map(|i| (i % 7) as u8).collect::<Vec<_>>();
    let b = bytes_tensor(0, vec![k, n], DType::I8, &source);
    let rows = scales(0, &vec![1.0; m]);
    let channels = scales(0, &vec![1.0; n]);
    let mut out = zeros_tensor(0, vec![m, n], DType::BF16);
    let cache = scratch_cache_dir("canonical-weight");
    let mut args = GemmArgs::w8a8(&a, &rows, &b, &channels, &mut out)
        .with_immutable_weight(WeightVersion::new(7));
    args.policy.online_tune = false;
    args.policy.cache_dir = Some(cache.to_string_lossy().into_owned());
    let normalized =
        super::contracts::normalize(&ctx, args, super::contracts::Semantic::Gemm, None).unwrap();
    // An immutable Tensor is still canonical; only RepackedWeight permits mutation.
    assert!(super::execution::seed_recipe(&ctx, &normalized, 3, 6, 1, 0).is_err());
    let execution = super::execution::prepare(&ctx, normalized).unwrap();
    execution.enqueue().unwrap();
    ctx.synchronize().unwrap();
    let mut actual = vec![0; source.len()];
    CudaBuffer::from_tensor(&b)
        .unwrap()
        .copy_to_host(&mut actual)
        .unwrap();
    assert_eq!(actual, source);
    for (i, value) in values(&out).into_iter().enumerate() {
        let expected = (0..k)
            .map(|row| source[row * n + i % n] as f32)
            .sum::<f32>();
        assert_eq!(value, bf16::from_f32(expected).to_f32());
    }
    std::fs::remove_dir_all(cache).unwrap();
}

#[test]
fn gpu_repacked_weight_survives_new_io_shape_and_graph() {
    let ctx = CudaContext::new(0).unwrap();
    if ctx.caps().sm != 87 {
        eprintln!("W8A8 inplace lifecycle regression requires SM87");
        return;
    }
    let (k, n) = (64, 128);
    let source = (0..k * n)
        .map(|i| ((i / n + 3 * (i % n)) % 7) as u8)
        .collect::<Vec<_>>();
    let b = bytes_tensor(0, vec![k, n], DType::I8, &source);
    let layout = crate::ops::RepackedWeight::new(&b).unwrap();
    let channels = scales(0, &vec![1.0; n]);
    let cache_dir = scratch_cache_dir("inplace-weight-lifecycle");
    let mut retained_graph = None;
    let mut retained_output = None;
    for (run, m) in [8, 17, 8].into_iter().enumerate() {
        let mut input = vec![0; m * k];
        for row in 0..m {
            input[row * k + row % k] = 1;
        }
        let a = bytes_tensor(0, vec![m, k], DType::I8, &input);
        let rows = scales(0, &vec![1.0; m]);
        let mut output = zeros_tensor(0, vec![m, n], DType::BF16);
        let mut args =
            GemmArgs::w8a8(&a, &rows, &b, &channels, &mut output).with_repacked_weight(&layout);
        args.policy.online_tune = false;
        args.policy.allow_fallback = false;
        args.policy.cache_dir = Some(cache_dir.to_string_lossy().into_owned());
        let mut normalized =
            super::contracts::normalize(&ctx, args, super::contracts::Semantic::Gemm, None)
                .unwrap();
        normalized.spec.b_layout = layout.state.lock().unwrap().layout;
        if run == 0 {
            let expected = (0..m * n)
                .map(|i| source[(i / n) % k * n + i % n] as f32)
                .collect::<Vec<_>>();
            super::execution::validate_candidates(&ctx, &normalized, &expected).unwrap();
            let mut unchanged = vec![0; source.len()];
            CudaBuffer::from_tensor(&b)
                .unwrap()
                .copy_to_host(&mut unchanged)
                .unwrap();
            assert_eq!(
                unchanged, source,
                "candidate trials must not rewrite model weights"
            );
        }
        super::execution::seed_recipe(&ctx, &normalized, 3, 6, 1, 0).unwrap();
        let execution = super::execution::prepare(&ctx, normalized).unwrap();
        assert!(execution.summary().contains("workspace=0 "));
        execution.enqueue().unwrap();
        ctx.synchronize().unwrap();
        for (index, actual) in values(&output).into_iter().enumerate() {
            assert_eq!(actual, source[(index / n) % k * n + index % n] as f32);
        }
        if run == 0 {
            let mut packed = vec![0; k * n];
            CudaBuffer::from_tensor(&b)
                .unwrap()
                .copy_to_host(&mut packed)
                .unwrap();
            for row in 0..k {
                for col in 0..n {
                    assert_eq!(packed[col * k + row], source[row * n + col]);
                }
            }
            let mut canonical_output = zeros_tensor(0, vec![m, n], DType::BF16);
            let args = GemmArgs::w8a8(&a, &rows, &b, &channels, &mut canonical_output);
            let normalized =
                super::contracts::normalize(&ctx, args, super::contracts::Semantic::Gemm, None)
                    .unwrap();
            let error = super::execution::prepare(&ctx, normalized)
                .err()
                .expect("packed bytes cannot be consumed as canonical");
            assert!(error.to_string().contains("layout state"), "{error}");
            retained_graph = Some(crate::capture(&ctx, || execution.enqueue()).unwrap());
            retained_output = Some(output);
        }
        // No session or execution cache remains for the next set of IO.
    }
    retained_graph.as_ref().unwrap().replay().unwrap();
    ctx.synchronize().unwrap();
    for (index, actual) in values(retained_output.as_ref().unwrap())
        .into_iter()
        .enumerate()
    {
        assert_eq!(actual, source[(index / n) % k * n + index % n] as f32);
    }
    // A shape/semantic change may use the layout-aware portable baseline.
    let a = bytes_tensor(0, vec![1, k], DType::I8, &vec![1; k]);
    let rows = scales(0, &[1.0]);
    let mut output = zeros_tensor(0, vec![1, n], DType::BF16);
    let mut args =
        GemmArgs::w8a8(&a, &rows, &b, &channels, &mut output).with_repacked_weight(&layout);
    args.policy.online_tune = false;
    args.policy.cache_dir = Some(String::new());
    crate::ops::gemm(&ctx, args).unwrap();
    for (col, actual) in values(&output).into_iter().enumerate() {
        let sum = (0..k).map(|row| source[row * n + col] as f32).sum::<f32>();
        assert_eq!(actual, bf16::from_f32(sum).to_f32());
    }
    std::fs::remove_dir_all(cache_dir).unwrap();
}

#[test]
fn gpu_repacked_geglu_preserves_old_graph_and_generic_shape() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n, implementation, version) = match ctx.caps().sm {
        89 => (10, 1024, 8192, 4, 1),
        100 | 101 | 110 => (522, 2048, 32768, 3, 2),
        _ => {
            eprintln!("GeGLU packing regression requires SM89 or SM100-family");
            return;
        }
    };
    let b_data = (0..k * n)
        .map(|i| ((i / n + 3 * (i % n)) % 17) as f32 / 16.0 - 0.5)
        .collect::<Vec<_>>();
    let bytes = b_data
        .iter()
        .flat_map(|&v| bf16::from_f32(v).to_bits().to_ne_bytes())
        .collect::<Vec<_>>();
    let b = bytes_tensor(0, vec![k, n], DType::BF16, &bytes);
    let layout = crate::ops::RepackedWeight::new(&b).unwrap();
    let cache_dir = scratch_cache_dir("inplace-geglu-lifecycle");
    let mut graph = None;
    let mut graph_output = None;
    for (run, rows) in [m, 1, m].into_iter().enumerate() {
        let mut input = vec![bf16::ZERO; rows * k];
        for row in 0..rows {
            input[row * k + row % k] = bf16::ONE;
        }
        let input_bytes = input
            .iter()
            .flat_map(|v| v.to_bits().to_ne_bytes())
            .collect::<Vec<_>>();
        let a = bytes_tensor(0, vec![rows, k], DType::BF16, &input_bytes);
        let mut out = zeros_tensor(0, vec![rows, n / 2], DType::BF16);
        let mut args = GemmArgs::new(&a, &b, &mut out).with_repacked_weight(&layout);
        args.policy.online_tune = false;
        args.policy.allow_fallback = rows == 1;
        args.policy.cache_dir = Some(cache_dir.to_string_lossy().into_owned());
        let mut normalized =
            super::contracts::normalize(&ctx, args, super::contracts::Semantic::GemmGeglu, None)
                .unwrap();
        normalized.spec.b_layout = layout.state.lock().unwrap().layout;
        if rows != 1 {
            super::execution::seed_recipe(&ctx, &normalized, 3, implementation, version, 0)
                .unwrap();
        }
        let execution = super::execution::prepare(&ctx, normalized).unwrap();
        if rows != 1 {
            assert!(execution.summary().contains("workspace=0 "));
        }
        execution.enqueue().unwrap();
        ctx.synchronize().unwrap();
        let check = |out: &Tensor| {
            for (i, actual) in values(out).into_iter().enumerate() {
                let row = i / (n / 2);
                let col = i % (n / 2);
                let gate = b_data[(row % k) * n + col];
                let up = b_data[(row % k) * n + col + n / 2];
                let expected = 0.5
                    * gate
                    * (1.0 + (0.7978845608 * (gate + 0.044715 * gate.powi(3))).tanh())
                    * up;
                assert!(
                    (actual - expected).abs() < 0.003,
                    "{i}: {actual} vs {expected}"
                );
            }
        };
        check(&out);
        if run == 0 {
            graph = Some(crate::capture(&ctx, || execution.enqueue()).unwrap());
            graph_output = Some(out);
        } else {
            graph.as_ref().unwrap().replay().unwrap();
            ctx.synchronize().unwrap();
            check(graph_output.as_ref().unwrap());
        }
    }
    std::fs::remove_dir_all(cache_dir).unwrap();
}

#[test]
fn gpu_repacked_fp8_crosses_geglu_and_plain_gemm() {
    let ctx = CudaContext::new(0).unwrap();
    if !matches!(ctx.caps().sm, 100 | 101 | 110) {
        eprintln!("FP8 packing regression requires SM100-family");
        return;
    }
    let (k, n) = (2048, 32768);
    let codes = [0x20, 0x28, 0x30, 0x38];
    let bytes = (0..k * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            codes[(r + c / 128 + c + usize::from(c >= n / 2)) % codes.len()]
        })
        .collect::<Vec<_>>();
    let canonical = bytes_tensor(0, vec![k, n], DType::F8E4M3, &bytes);
    let committed = bytes_tensor(0, vec![k, n], DType::F8E4M3, &bytes);
    let layout = crate::ops::RepackedWeight::new(&committed).unwrap();
    let cache_dir = scratch_cache_dir("inplace-fp8-cross-semantic");
    for m in [522, 1, 522] {
        let fused = m != 1;
        let mut a_bytes = vec![0; m * k];
        for row in 0..m {
            a_bytes[row * k + row % k] = 0x38;
        }
        let a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &a_bytes);
        let dtype = if fused { DType::F8E4M3 } else { DType::F16 };
        let width = if fused { n / 2 } else { n };
        let mut expected = zeros_tensor(0, vec![m, width], dtype);
        let mut actual = zeros_tensor(0, vec![m, width], dtype);
        for (b, out, state) in [
            (&canonical, &mut expected, None),
            (&committed, &mut actual, Some(&layout)),
        ] {
            let mut args = GemmArgs::new(&a, b, out).with_immutable_weight(WeightVersion::new(0));
            if let Some(state) = state {
                args = args.with_repacked_weight(state);
            }
            args.quantization = GemmQuantization::Fp8UnitScale;
            args.policy.online_tune = false;
            args.policy.allow_fallback = !fused || state.is_none();
            args.policy.cache_dir = Some(cache_dir.to_string_lossy().into_owned());
            let semantic = if fused {
                super::contracts::Semantic::GemmGeglu
            } else {
                super::contracts::Semantic::Gemm
            };
            let mut normalized = super::contracts::normalize(&ctx, args, semantic, None).unwrap();
            if let Some(state) = state {
                normalized.spec.b_layout = state.state.lock().unwrap().layout;
            }
            if fused && state.is_some() {
                super::execution::seed_recipe(&ctx, &normalized, 3, 2, 2, 0).unwrap();
            }
            let execution = super::execution::prepare(&ctx, normalized).unwrap();
            execution.enqueue().unwrap();
        }
        ctx.synchronize().unwrap();
        let read = |tensor: &Tensor| {
            let buffer = CudaBuffer::from_tensor(tensor).unwrap();
            let mut bytes = vec![0; buffer.len()];
            buffer.copy_to_host(&mut bytes).unwrap();
            bytes
        };
        assert_eq!(read(&actual), read(&expected), "M={m}");
        assert_eq!(
            read(&canonical),
            bytes,
            "reference weights remain canonical"
        );
    }
    std::fs::remove_dir_all(cache_dir).unwrap();
}
