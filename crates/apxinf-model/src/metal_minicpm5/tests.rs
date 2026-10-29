//! Real-checkpoint lifecycle checks, separate from numerical qualification.
use super::{config::Config, weights::Weights, MiniCpm5, Variant};
use crate::{
    GenerationBoundary, GenerationOptions, GenerationRequest, LlmInput, LlmTrait,
    TextPreparationState,
};
use apxinf_core::{Error, Result, Tensor};
use apxinf_mlx::{clear_cache, memory_stats, MlxBackend};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{env, path::PathBuf};

fn last_logits(model: &MiniCpm5, output: &Tensor) -> Result<Vec<f32>> {
    let values = model.backend.array(output)?.to_f32_vec()?;
    Ok(values[values.len() - model.vocab_size()..].to_vec())
}

fn check_chunk_logits(expected: &[f32], actual: &[f32], length: usize) {
    assert_eq!(expected.len(), actual.len());
    let mut maximum_error = 0.0f32;
    for (&whole, &chunked) in expected.iter().zip(actual) {
        assert!(
            whole.is_finite() && chunked.is_finite(),
            "nonfinite logits at {length}"
        );
        let error = (whole - chunked).abs();
        maximum_error = maximum_error.max(error);
        assert!(
            error <= 0.5 + 0.02 * whole.abs(),
            "BF16 tolerance at {length}: {whole} versus {chunked}"
        );
    }
    let greedy = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(
        greedy(expected),
        greedy(actual),
        "greedy token changed at {length}"
    );
    eprintln!(
        "prompt_tokens={length} maximum_logit_error={maximum_error} greedy_token={}",
        greedy(actual)
    );
}

#[test]
#[ignore = "requires official MiniCPM checkpoint, isolated process and shared Metal lock"]
fn real_checkpoint_chunked_prefill_equivalence_and_cancellation() -> Result<()> {
    let checkpoint = env::var_os("APXINF_MINICPM_CHECKPOINT")
        .map(PathBuf::from)
        .ok_or(Error::Contract(
            "set APXINF_MINICPM_CHECKPOINT for this ignored test",
        ))?;
    let tokenizer = apxinf_tokenizer::Tokenizer::from_file(checkpoint.join("tokenizer.json"))?;
    let prompt = tokenizer.encode(include_str!(
        "../../tests/fixtures/minicpm-context-prompt.txt"
    ))?;
    assert!(
        prompt.len() >= 513,
        "original prompt fixture must cover every boundary"
    );
    let config = Config::from_json(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    let backend = MlxBackend::new(0)?;
    let (map, _) =
        apxinf_loader::safetensors::load_native_path(&checkpoint).map_err(Error::Other)?;
    let weights = Weights::load(&config, backend.stream(), map)?;
    let mut model = MiniCpm5::new(config, weights, backend, Variant::Compiled)?;
    let mut recovery_reference = Vec::new();
    for length in [255, 256, 257, 513] {
        let ids = &prompt[..length];
        model.reset();
        model.prepare(length, 1)?;
        let full = model.forward(ids, 0)?;
        assert_eq!(full.shape().dims(), [length, model.vocab_size()]);
        let expected = last_logits(&model, &full)?;
        drop(full);
        model.reset();
        model.prepare(length, 1)?;
        let mut boundaries = Vec::new();
        let chunked = model
            .prefill_controlled(LlmInput::text(ids), &mut |boundary| {
                boundaries.push(boundary);
                Ok(true)
            })?
            .unwrap();
        assert_eq!(chunked.shape().dims(), [1, model.vocab_size()]);
        check_chunk_logits(&expected, &last_logits(&model, &chunked)?, length);
        assert_eq!(model.offset, length);
        let mut expected_boundaries = vec![GenerationBoundary::PrefillStart];
        expected_boundaries.extend(
            (256..length)
                .step_by(256)
                .chain(std::iter::once(length))
                .map(|consumed_tokens| GenerationBoundary::PrefillProgress { consumed_tokens }),
        );
        assert_eq!(boundaries, expected_boundaries);
        if length == 257 {
            recovery_reference = expected;
        }
    }

    model.reset();
    model.prepare(513, 1)?;
    let mut invalid = prompt[..513].to_vec();
    invalid[400] = model.vocab_size() as u32;
    let mut callbacks = 0;
    assert!(model
        .prefill_controlled(LlmInput::text(&invalid), &mut |_| {
            callbacks += 1;
            Ok(true)
        })
        .is_err());
    assert_eq!(callbacks, 0);
    assert_eq!(model.offset, 0);
    assert_eq!(
        model.preparation_status().state,
        TextPreparationState::Ready
    );

    let before = model.prefill_controlled(LlmInput::text(&prompt[..513]), &mut |boundary| {
        assert_eq!(boundary, GenerationBoundary::PrefillStart);
        Ok(false)
    })?;
    assert!(before.is_none());
    assert_eq!(model.offset, 0);
    let options = GenerationOptions::greedy(1, None);
    let mut boundaries = Vec::new();
    let cancelled = model.generate_streaming_controlled_dyn(
        GenerationRequest {
            input: LlmInput::text(&prompt[..513]),
            options: &options,
        },
        &mut |boundary| {
            boundaries.push(boundary);
            Ok(matches!(boundary, GenerationBoundary::PrefillStart))
        },
    )?;
    assert!(cancelled.tokens.is_empty());
    assert_eq!(
        boundaries,
        vec![
            GenerationBoundary::PrefillStart,
            GenerationBoundary::PrefillProgress {
                consumed_tokens: 256
            }
        ]
    );
    assert_eq!(model.offset, 256);
    model.release_request_state()?;
    assert_eq!(model.offset, 0);
    assert_eq!(
        model.preparation_status().state,
        TextPreparationState::Unprepared
    );
    model.prepare(257, 1)?;
    let recovered = model.prefill(LlmInput::text(&prompt[..257]))?;
    check_chunk_logits(&recovery_reference, &last_logits(&model, &recovered)?, 257);
    drop(recovered);
    model.release_request_state()?;
    let mut resumed_boundaries = Vec::new();
    let resumed = model.generate_streaming_controlled_dyn(
        GenerationRequest {
            input: LlmInput::text(&prompt[..257]),
            options: &options,
        },
        &mut |boundary| {
            resumed_boundaries.push(boundary);
            Ok(true)
        },
    )?;
    assert_eq!(resumed.tokens.len(), 1);
    let expected_token = recovery_reference
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32;
    assert_eq!(resumed.tokens[0].token_id, expected_token);
    assert_eq!(
        resumed_boundaries,
        vec![
            GenerationBoundary::PrefillStart,
            GenerationBoundary::PrefillProgress {
                consumed_tokens: 256
            },
            GenerationBoundary::PrefillProgress {
                consumed_tokens: 257
            },
            GenerationBoundary::Token(resumed.tokens[0]),
        ]
    );
    assert_eq!(
        model.preparation_status().prepared_sequence_lengths,
        vec![1, 256, 257]
    );
    model.release_request_state()?;
    eprintln!(
        "whole_input_rejection=true cancel_before=true cancel_after=256 release_and_recovery=true"
    );
    Ok(())
}

#[derive(Deserialize)]
struct ReplayInputs {
    cases: Vec<ReplayCase>,
}

#[derive(Deserialize)]
struct ReplayCase {
    id: String,
    prompt_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LogitsDigest {
    bytes: usize,
    all_rows: [u8; 32],
    last_row: [u8; 32],
}

/// The download is intentional test observation, never a runtime model helper.
fn digest(backend: &MlxBackend, output: &Tensor) -> Result<LogitsDigest> {
    let bytes = backend.array(output)?.to_bytes()?;
    let row_bytes = output.shape().dims()[1] * output.dtype().size_in_bytes();
    assert!(bytes.len() >= row_bytes);
    Ok(LogitsDigest {
        bytes: bytes.len(),
        all_rows: Sha256::digest(&bytes).into(),
        last_row: Sha256::digest(&bytes[bytes.len() - row_bytes..]).into(),
    })
}

fn full_forward(model: &mut MiniCpm5, case: &ReplayCase) -> Result<Tensor> {
    model.reset();
    model.prepare(case.prompt_token_ids.len(), 1)?;
    assert_eq!(
        model.preparation_status().state,
        TextPreparationState::Ready
    );
    let output = model.forward(&case.prompt_token_ids, 0)?;
    assert_eq!(
        output.shape().dims(),
        [case.prompt_token_ids.len(), model.vocab_size()],
        "public forward must retain every logit row for {}",
        case.id
    );
    Ok(output)
}

#[test]
#[ignore = "requires official MiniCPM checkpoint and exclusive shared Metal locks"]
fn real_checkpoint_long_chunk_and_bucket_boundary() -> Result<()> {
    let checkpoint = PathBuf::from(
        env::var_os("APXINF_MINICPM_CHECKPOINT")
            .ok_or(Error::Contract("set APXINF_MINICPM_CHECKPOINT"))?,
    );
    let config = Config::from_json(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    let backend = MlxBackend::new(0)?;
    let (map, _) =
        apxinf_loader::safetensors::load_native_path(&checkpoint).map_err(Error::Other)?;
    let weights = Weights::load(&config, backend.stream(), map)?;
    let mut model = MiniCpm5::new(config, weights, backend.clone(), Variant::Compiled)?;
    model.prepare(3, 1)?;
    let short = model.prefill(LlmInput::text(&[1, 7, 3]))?;
    let expected = digest(&backend, &short)?;
    model.reset();
    model.prepare(4351, 2)?;
    assert_eq!(model.preparation_status().kv_capacity, Some(4608));
    model.prefill(LlmInput::text(&vec![7; 4351]))?;
    let traces = backend.stream().counters().trace_callbacks;
    model.forward(&[2], 4351)?;
    model.forward(&[3], 4352)?;
    assert_eq!(backend.stream().counters().trace_callbacks, traces);
    assert!(model.forward(&[3], 4608).is_err());
    model.reset();
    model.prepare(3, 1)?;
    assert_eq!(model.preparation_status().kv_capacity, Some(256));
    let repeated = model.prefill(LlmInput::text(&[1, 7, 3]))?;
    assert_eq!(digest(&backend, &repeated)?, expected);
    assert_eq!(digest(&backend, &short)?, expected);
    Ok(())
}

/// Run only this test in an isolated process under the shared Metal lock:
/// `minicpm5::tests::real_checkpoint_full_forward_retention_and_release
/// --exact --ignored --test-threads=1 --nocapture`.
/// APXINF_MINICPM_CHECKPOINT names the checkpoint directory and
/// APXINF_DSPARK_REPLAY_INPUT names a source-bound JSON with two real prompts.
/// This checks lifecycle/determinism, not agreement with a numerical oracle.
#[test]
#[ignore = "requires official MiniCPM checkpoint, real replay inputs, isolated process and shared Metal lock"]
fn real_checkpoint_full_forward_retention_and_release() -> Result<()> {
    let path = |name| {
        env::var_os(name)
            .map(PathBuf::from)
            .ok_or_else(|| Error::Other(format!("set {name} for this ignored test")))
    };
    let checkpoint = path("APXINF_MINICPM_CHECKPOINT")?;
    let replay = path("APXINF_DSPARK_REPLAY_INPUT")?;
    let inputs: ReplayInputs = serde_json::from_slice(&std::fs::read(replay)?)
        .map_err(|e| Error::Other(format!("MiniCPM lifecycle replay inputs: {e}")))?;
    if inputs.cases.len() < 2
        || inputs.cases[..2]
            .iter()
            .any(|case| case.id.is_empty() || case.prompt_token_ids.is_empty())
        || inputs.cases[0].prompt_token_ids == inputs.cases[1].prompt_token_ids
    {
        return Err(Error::Contract(
            "MiniCPM lifecycle test needs two distinct nonempty real prompts",
        ));
    }
    let config = Config::from_json(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    let backend = MlxBackend::new(0)?;
    backend.stream().synchronize()?;
    clear_cache()?;
    let baseline = memory_stats()?;
    let mut expected: [Option<LogitsDigest>; 2] = [None, None];
    for cycle in 0..2 {
        let (map, _) =
            apxinf_loader::safetensors::load_native_path(&checkpoint).map_err(Error::Other)?;
        let weights = Weights::load(&config, backend.stream(), map)?;
        // The observer backend owns only the stream, never weights or a model.
        let mut model = MiniCpm5::new(config.clone(), weights, backend.clone(), Variant::Compiled)?;
        assert_eq!(
            model.preparation_status().state,
            TextPreparationState::Unprepared
        );
        // Reverse order on the second load: both prompts get a fresh-model run.
        let first = cycle;
        let changed = 1 - cycle;
        let retained = full_forward(&mut model, &inputs.cases[first])?;
        let original = digest(&backend, &retained)?;
        if let Some(previous) = &expected[first] {
            assert_eq!(&original, previous, "fresh-model logits changed");
        }
        expected[first] = Some(original.clone());

        let next = full_forward(&mut model, &inputs.cases[changed])?;
        let changed_digest = digest(&backend, &next)?;
        assert_ne!(
            original.last_row, changed_digest.last_row,
            "changed real input did not change final logits"
        );
        if let Some(previous) = &expected[changed] {
            assert_eq!(
                &changed_digest, previous,
                "reset leaked prior request state"
            );
        }
        expected[changed] = Some(changed_digest);
        drop(next);
        assert_eq!(
            digest(&backend, &retained)?,
            original,
            "subsequent forward mutated retained output"
        );

        let repeated = full_forward(&mut model, &inputs.cases[first])?;
        assert_eq!(
            digest(&backend, &repeated)?,
            original,
            "reset/reprepare changed the original logits"
        );
        drop(repeated);
        model.reset();
        assert_eq!(
            digest(&backend, &retained)?,
            original,
            "reset mutated retained output"
        );
        drop(model);
        backend.stream().synchronize()?;
        assert_eq!(
            digest(&backend, &retained)?,
            original,
            "model drop invalidated retained output"
        );
        drop(retained);
        backend.stream().synchronize()?;
        clear_cache()?;
        let released = memory_stats()?;
        assert_eq!(
            released.active_bytes, baseline.active_bytes,
            "model/output retained live MLX allocations in cycle {cycle}: {released:?}"
        );
        assert_eq!(released.cache_bytes, 0);
        eprintln!("cycle={cycle} first_case={} full_logits={original:?} baseline={baseline:?} released={released:?}", inputs.cases[first].id);
    }
    Ok(())
}

#[test]
#[ignore = "requires official MiniCPM checkpoint, real replay inputs and exclusive shared Metal locks"]
fn real_checkpoint_prepared_code_reuses_with_fresh_request_state() -> Result<()> {
    let checkpoint = PathBuf::from(env::var_os("APXINF_MINICPM_CHECKPOINT")
        .ok_or(Error::Contract("set APXINF_MINICPM_CHECKPOINT"))?);
    let replay = PathBuf::from(env::var_os("APXINF_DSPARK_REPLAY_INPUT")
        .ok_or(Error::Contract("set APXINF_DSPARK_REPLAY_INPUT"))?);
    let inputs: ReplayInputs = serde_json::from_slice(&std::fs::read(replay)?)
        .map_err(|e| Error::Other(e.to_string()))?;
    if inputs.cases.len() < 2 || inputs.cases[..2].iter().any(|c| c.prompt_token_ids.is_empty()) {
        return Err(Error::Contract("two nonempty real prompts are required"));
    }
    let config = Config::from_json(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    let backend = MlxBackend::new(0)?;
    let (map, _) = apxinf_loader::safetensors::load_native_path(&checkpoint).map_err(Error::Other)?;
    let weights = Weights::load(&config, backend.stream(), map)?;
    let mut model = MiniCpm5::new(config, weights, backend.clone(), Variant::Compiled)?;
    let options = crate::GenerationOptions::greedy(8, None);
    let run = |model: &mut MiniCpm5, index: usize| {
        model.generate_streaming_with_options(crate::GenerationRequest {
            input: LlmInput::text(&inputs.cases[index].prompt_token_ids), options: &options,
        }, |_| {}).map(|output| output.token_ids())
    };
    let first = run(&mut model, 0)?;
    model.release_request_state_retaining_preparation()?;
    assert_eq!(model.preparation_status().state, TextPreparationState::Unprepared);
    let traces = backend.stream().counters().trace_callbacks;
    assert_eq!(run(&mut model, 0)?, first);
    assert_eq!(backend.stream().counters().trace_callbacks, traces,
        "same-profile request traced after prepared-code retention");
    model.release_request_state_retaining_preparation()?;
    let changed = run(&mut model, 1)?;
    model.release_request_state()?;
    assert_eq!(run(&mut model, 1)?, changed, "retained preparation leaked prior request state");
    model.release_request_state()?;
    assert_eq!(run(&mut model, 0)?, first, "changed profile altered the original output");
    let old_capacity = model.preparation_status().kv_capacity.unwrap();
    model.reset();
    model.prepare(old_capacity + 1, 8)?;
    assert!(model.preparation_status().kv_capacity.unwrap() > old_capacity);
    assert_eq!(run(&mut model, 1)?, changed, "ordinary capacity change leaked prior state");
    Ok(())
}
