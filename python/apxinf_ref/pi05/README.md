# pi05 torch-eager gold

Deterministic Pi0.5 reference produced by the vendored openpi `PI0Pytorch`
snapshot. The output is a self-verifying artifact (schema
`apxinf.pi05.torch-eager-gold.v1`) that an ApxInf engine replay is compared
against.

Two halves, split by what they need:

- `torch_gold/artifact.py` — the fixture, the stored encoding, the manifest, the
  writer's gate and the preflight a replay runs before scoring (`patchify`,
  `reference_actions`, `replay_problems`). Imports no torch at module scope, so
  it loads (and is tested) on a host with no device stack.
- `torch_gold/capture.py` — drives the model through
  `PI0Pytorch.sample_actions`, replaces `_preprocess_observation` so no
  tokenizer or camera preprocessing runs, and captures every tensor on the
  forward boundary plus sampled layer signatures. Needs torch; import it only
  after torch (or torchada) is loaded.

The fixture is synthetic and seeded: images are canonical NCHW float32, token
ids and noise come from `numpy.default_rng(seed)`, and the Euler schedule
travels with it (`num_flow_steps`, `dt`, `flow_start_time`). Defaults are seed
0, 2 views, 10 tokens, 10 flow steps.

## Checkpoint

`pi05_libero_pytorch` from the openpi assets (`openpi-assets/checkpoints/`),
as used by the ApxInf pi05 runner:

```
<checkpoint>/                     self-describing, one directory per checkpoint
  config.json                     action_dim 32, action_horizon 10, bfloat16,
                                  paligemma gemma_2b, action expert gemma_300m
  model.safetensors               sha256 d648dc6c5a4da5ccec752297f564e4259c66a6f08bad8b5f6919f86e0f82edb5
  assets/physical-intelligence/libero/norm_stats.json
                                  (informational; the synthetic fixture bypasses normalization)
```

`--checkpoint` defaults to `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`
and can be set with `APXINF_PI05_CHECKPOINT`. Each run hashes the whole
`model.safetensors` and records the digest in the manifest, so a baseline names
the exact weights it was produced from; a different digest means a different
baseline.

The tokenizer (`paligemma_tokenizer.model`, sha256 `8986bb4f…`) lives in the
openpi checkout and is not used by the synthetic fixture.

## Generate

```bash
python python/apxinf_ref/pi05/generate_pi05_torch_gold.py \
  --device cuda:0 \
  --checkpoint /path/to/pi05_libero_pytorch \
  --output devlocal/pi05-gold/orin
```

Run it from the repository root with the Python that has torch; no
`PYTHONPATH` is needed because the entry point installs its own directory.
Keep the defaults (`--sample-count 256`, `--warmup-runs 3`, `--timed-runs 3`)
when producing a baseline: changing them changes `debug.npz` or the manifest's
`sampling` block.

Artifacts are not committed. The ignored `devlocal/pi05-gold/<device>/` layout
holds one baseline per host:

```
devlocal/pi05-gold/orin/
  inputs.npz      canonical fixture, the only file `--input` reads
  debug.npz       full boundary tensors, stage signatures, raw_actions
  manifest.json   schema, runtime, checkpoint identity, per-array metadata,
                  statistics, checks
```

The writer stages the files, reads them back and re-checks every array against
the manifest, then publishes atomically. A run that cannot verify exits 3 and
leaves the output directory untouched (`--force` replaces an existing
artifact).

## Replay an existing fixture

```bash
python python/apxinf_ref/pi05/generate_pi05_torch_gold.py \
  --device musa:0 \
  --checkpoint /path/to/checkpoint \
  --input <path-to-inputs.npz> \
  --output devlocal/pi05-gold/musa
```

The replay prints the cosine similarity of its own capture against the artifact
the fixture came from. That readout is printed, never stored, so two runs over
one fixture still produce byte-identical manifest bytes.

## Consuming an artifact

`inputs.npz` is the engine input: `images` (NCHW float32), `token_ids`,
`state`, `noise`, `num_flow_steps`, `dt`, `flow_start_time`.
`debug.npz` holds the reference tensors; decode each with its manifest entry
(`debug.<name>`: `dtype`, `encoding`, `shape`). bfloat16 arrays are stored as
raw uint16 bit patterns (`encoding: "bfloat16-bits"`), float32/int64/bool are
native. `torch_gold.artifact.auditor_values(array, dtype=…, encoding=…)` is the
decoder used by the replay tooling.

`manifest.json` is also the identity record: `checkpoint.sha256`, `runtime`
(torch, python, platform, precision, transformers, numpy), `fixture` (generated
vs replayed), `sampling`, `model_config`, and `checks` (all true on a published
artifact).
