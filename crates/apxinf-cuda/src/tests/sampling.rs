use apxinf_core::{
    standard_normal_f32, Backend, DType, NextTokenLogits, RngKey, SamplingBackend, Tensor,
    TokenPenalties, TokenSamplingInit, TokenSamplingParams, TokenSamplingSpec, TokenSelection,
};
use half::{bf16, f16};

use crate::CudaBackend;

fn gpu_logits(backend: &CudaBackend, values: &[f32], shape: Vec<usize>) -> Tensor {
    backend
        .to_device(&Tensor::from_f32(shape, values).unwrap())
        .unwrap()
}

#[test]
fn gpu_greedy_uses_multiblock_reduction_and_lowest_tie() {
    let backend = CudaBackend::new(0).expect("CUDA device required");
    let vocab = 32_768;
    let mut values = vec![-10.0f32; vocab * 2];
    values[17] = 100.0; // Must be ignored: only the last row is sampled.
    values[vocab + 21_000] = 5.0;
    values[vocab + 123] = 5.0;
    let logits = gpu_logits(&backend, &values, vec![2, vocab]);
    let params = TokenSamplingParams::greedy();
    let mut sampler = backend
        .create_token_sampler(TokenSamplingSpec {
            vocab_size: vocab,
            max_sequence_len: 8,
        })
        .unwrap();
    sampler
        .begin(TokenSamplingInit {
            prompt_token_ids: &[1, 2],
            params: &params,
            rng: RngKey::default(),
        })
        .unwrap();
    let sample = sampler
        .sample(NextTokenLogits::last(&logits, vocab).unwrap())
        .unwrap();
    assert_eq!(sample.token_id, 123);
}

#[test]
fn gpu_greedy_logprob_matches_cpu_for_large_tied_logits() {
    let backend = CudaBackend::new(0).expect("CUDA device required");
    let cpu = apxinf_core::CpuBackend;
    let cpu_logits = Tensor::from_f32(vec![1, 3], &[f32::MAX, f32::MAX, 0.0]).unwrap();
    let gpu_logits = backend.to_device(&cpu_logits).unwrap();
    let params = TokenSamplingParams {
        return_logprob: true,
        ..TokenSamplingParams::greedy()
    };
    let spec = TokenSamplingSpec {
        vocab_size: 3,
        max_sequence_len: 4,
    };
    let init = TokenSamplingInit {
        prompt_token_ids: &[2],
        params: &params,
        rng: RngKey::default(),
    };
    let mut cpu_sampler = cpu.create_token_sampler(spec).unwrap();
    let mut gpu_sampler = backend.create_token_sampler(spec).unwrap();
    cpu_sampler.begin(init).unwrap();
    gpu_sampler.begin(init).unwrap();

    let expected = cpu_sampler
        .sample(NextTokenLogits::last(&cpu_logits, 3).unwrap())
        .unwrap();
    let actual = gpu_sampler
        .sample(NextTokenLogits::last(&gpu_logits, 3).unwrap())
        .unwrap();
    assert_eq!(actual.token_id, expected.token_id);
    assert!((actual.logprob.unwrap() - expected.logprob.unwrap()).abs() < 1e-6);
}

#[test]
fn gpu_penalties_and_random_sampling_match_cpu_reference() {
    let backend = CudaBackend::new(0).expect("CUDA device required");
    let cpu = apxinf_core::CpuBackend;
    let vocab = 257;
    let values = (0..vocab)
        .map(|index| ((index * 37 % 101) as f32 - 50.0) * 0.07)
        .collect::<Vec<_>>();
    let params = TokenSamplingParams {
        selection: TokenSelection::Random {
            temperature: 0.73,
            top_k: Some(64),
            top_p: 0.91,
        },
        penalties: TokenPenalties {
            repetition: 1.15,
            frequency: 0.2,
            presence: 0.1,
        },
        return_logprob: true,
    };
    let spec = TokenSamplingSpec {
        vocab_size: vocab,
        max_sequence_len: 16,
    };
    let init = TokenSamplingInit {
        prompt_token_ids: &[3, 3, 7, 11],
        params: &params,
        rng: RngKey::new(0x1234_5678, 9, 2),
    };
    for cpu_logits in [
        Tensor::from_f32(vec![1, vocab], &values).unwrap(),
        Tensor::from_f16(
            vec![1, vocab],
            &values
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        Tensor::from_bf16(
            vec![1, vocab],
            &values
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
    ] {
        let gpu_logits = backend.to_device(&cpu_logits).unwrap();
        let mut cpu_sampler = cpu.create_token_sampler(spec).unwrap();
        let mut gpu_sampler = backend.create_token_sampler(spec).unwrap();
        cpu_sampler.begin(init).unwrap();
        gpu_sampler.begin(init).unwrap();
        for _ in 0..5 {
            let expected = cpu_sampler
                .sample(NextTokenLogits::last(&cpu_logits, vocab).unwrap())
                .unwrap();
            let actual = gpu_sampler
                .sample(NextTokenLogits::last(&gpu_logits, vocab).unwrap())
                .unwrap();
            assert_eq!(
                actual.token_id,
                expected.token_id,
                "dtype={}",
                cpu_logits.dtype()
            );
            assert!(
                (actual.logprob.unwrap() - expected.logprob.unwrap()).abs() < 3e-5,
                "dtype={}, actual={actual:?}, expected={expected:?}",
                cpu_logits.dtype()
            );
        }
    }
}

#[test]
fn gpu_standard_normal_is_reproducible_and_distributed() {
    let backend = CudaBackend::new(0).expect("CUDA device required");
    let output = backend
        .to_device(&Tensor::zeros(vec![100_000], DType::F32))
        .unwrap();
    let mut generator = backend.create_normal_generator(output).unwrap();
    let key = RngKey::new(17, 23, 42);
    generator.generate(key).unwrap();
    backend.synchronize().unwrap();
    let first = backend
        .to_cpu(generator.output())
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec();
    let expected = standard_normal_f32(first.len(), key);
    let max_error = first
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error < 2e-5,
        "CPU/CUDA normal RNG max error={max_error}"
    );
    generator.generate(key).unwrap();
    backend.synchronize().unwrap();
    let second = backend
        .to_cpu(generator.output())
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec();
    assert_eq!(first, second);
    let mean = first.iter().sum::<f32>() / first.len() as f32;
    let variance = first
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / first.len() as f32;
    assert!(mean.abs() < 0.02, "mean={mean}");
    assert!((variance - 1.0).abs() < 0.03, "variance={variance}");

    let expected = standard_normal_f32(257, key);
    for dtype in [DType::F16, DType::BF16] {
        let output = backend.to_device(&Tensor::zeros(vec![257], dtype)).unwrap();
        let mut generator = backend.create_normal_generator(output).unwrap();
        generator.generate(key).unwrap();
        backend.synchronize().unwrap();
        let actual = backend
            .to_cpu(generator.output())
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let tolerance = if dtype == DType::F16 { 0.002 } else { 0.02 };
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "dtype={dtype}, actual={actual}, expected={expected}"
            );
        }
    }
}
