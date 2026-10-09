//! Real-checkpoint lifecycle checks, separate from numerical qualification.
use super::{config::Config, weights::Weights, MiniCpm5, Variant};
use crate::{LlmTrait, TextPreparationState};
use apxinf_core::{Error, Result, Tensor};
use apxinf_mlx::{clear_cache, memory_stats, MlxBackend};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{env, path::PathBuf};

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
