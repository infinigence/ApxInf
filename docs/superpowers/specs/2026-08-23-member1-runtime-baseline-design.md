# Member1 Runtime Baseline Design

## Scope

This design covers the first member1 integration slice: reproducible R0 capture,
M2-O0 handoff validation, the runtime admission contract, and the production
boundary needed to connect a future member2 protocol adapter to a bounded GPU
worker. It does not change the evaluator contract, protocol semantics, loader
ownership, or member3 experiment scripts.

## Frozen Inputs

- Repository commit at inspection: `81dad4753f2aa72b77f8deddbe7fb290b3d1789e`.
- Model: local `Qwen3.8-27B-AWQ-INT4` artifact, HF cache revision
  `63768c10df38c0395e12ef49edac1bd539eaeeea`.
- Model vocabulary: `text_config.vocab_size=248320`; image token `248056` is
  valid even though tokenizer vocabulary is `248044`.
- Contract SHA256:
  `520349b1279c3bf999a6848b296c23d20cdaeab7420934e9196c90018bac7433`.
- Server baseline: CUDA 12.8, driver `580.82.07`, RTX 4090 UUIDs fixed by the
  collaboration workflow; GPU1 is oracle/replay and GPU0 is formal scoring.
- The kernel journal is unavailable on this host, so R0 remains blocked until
  an equivalent Xid evidence command is available.

## Architecture

The model-facing runtime exposes only model-neutral request, capability,
admission, cancellation, token-event, usage, and request-error types. HTTP code
will depend on this interface and will not receive CUDA types. A single runtime
owner consumes a bounded channel, owns mutable model state and CUDA resources,
and serializes GPU work. Each request carries an independent state handle and a
cancel flag; completion, cancellation, admission rejection, and CUDA failure
must release request state through RAII.

Admission has two layers. Protocol validation remains member2-owned. Runtime
capacity validation checks the total `prompt_tokens + max_new_tokens` budget,
model vocabulary boundary, configured parallel request limit, and calibrated
device bytes. Capacity rejection happens before any kernel launch. Health is
allowed to report `ok` only while the runtime health probe succeeds; a damaged
CUDA context becomes a service error and requires worker recovery.

## Data and Artifact Flow

1. Capture environment, model manifest hashes, contract hash, GPU UUID, and
   command into an R0 record. Never copy model weights into Git.
2. Require a complete M2-O0 handoff containing generator SHA, schema version,
   input manifest hash, layer/stage selection, model revision, and replay
   command. Without it, do not run a real checkpoint oracle.
3. With `/tmp/apxinf-gpu-job.lock` held and GPU1 explicitly selected, run one
   oracle job. Store only approved manifest/golden outputs and checksums under
   `/mnt/chuangxin/team2/artifacts/apxinf/oracle/<revision>/<commit-sha>/`.
4. Release GPU1 before any GPU0 base/reliability job. Formal evidence is always
   tagged with GPU0's UUID.

## Correctness and Reliability Gates

The first code slice is test-first. Tests cover model-vocabulary admission,
the total-length budget, cancellation before launch, bounded worker behavior,
state cleanup after errors, and W4 metadata directionality. The runtime must
not introduce CPU, external vLLM/Transformers, other GPU, or other model
fallbacks. Feature-off behavior is the rollback point for every future kernel
candidate.

## Rollback

The rollback point is the last integrated commit with all feature flags at their
reference/eager values. Runtime adapter additions must remain additive and
revertible. Any oracle or server failure is recorded with its full commit SHA,
GPU UUID, command, artifact path, SHA256, and failure reason before retrying.
