# Portable Oracle Job Generator

`generate_golden.py` prepares a canonical selective-oracle job for the frozen
Qwen3.5 checkpoint. Local manifest-only use reads only `config.json` and
`generation_config.json`; it does not load weights, expand BF16 tensors, or
write guessed golden values.

## Manifest-only bundle

```bash
python3 tools/oracle/generate_golden.py \
  --model-dir /mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4 \
  --output-dir /mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/<commit-sha> \
  --revision 63768c10df38c0395e12ef49edac1bd539eaeeea \
  --input-manifest tools/oracle/manifests/m2-o0-short.json \
  --layers 0,3,31,32,60,63 \
  --stages embedding,layer_hidden,gdn_state,kv_state,logits,tokens \
  --max-new-tokens 128
```

Layer lists accept comma-separated IDs, repeated flags, and inclusive ranges
such as `--layers 0-63`. If layers are supplied without stages, the generator
selects `layer_hidden`. The supported stages are `embedding`, `layer_hidden`,
`gdn_state`, `kv_state`, `logits`, and `tokens`.

The bundle contains canonical JSON control files and an empty `artifacts/`
directory. Every declared artifact starts as `pending`; no output token ID,
decoded text, hidden state, recurrent state, KV state, or logit value exists
until a real runner completes successfully.

## Server runner contract

Member1 supplies a checkpoint-specific runner on GPU1:

```bash
python3 tools/oracle/generate_golden.py <manifest-only flags above> \
  --runner /absolute/path/to/qwen35_oracle_runner \
  --runner-arg value
```

The runner is executed directly, without a shell, and receives:

```text
APXINF_ORACLE_JOB_MANIFEST=/absolute/path/job-manifest.json
APXINF_ORACLE_OUTPUT_DIR=/absolute/path/artifacts
```

It must write exactly the files declared by `artifact-manifest.json` plus
`artifact-report.json`. The report schema is
`apxinf.oracle-artifact-report.v1`; it must include
`generation: {completion_tokens, stop_reason}` where `stop_reason` is `eos` or
`budget`. Every artifact record contains `file`, `schema_ref`, `dtype`, fully
resolved positive integer `shape`, and `sha256`. Missing, extra, duplicate,
wrong-shard, dtype/schema/shape/hash, invalid token, byte-size, non-finite F32,
early-stop/EOS-policy, metadata-drift, symlink, or nonzero-runner cases fail
before the bundle is marked complete. The runner cannot replace the
`artifacts/` directory or mutate control manifests.

The real checkpoint runner, GPU UUID, peak VRAM, raw artifact hashes, and any
approved export are recorded by member1 in
`docs/collaboration/records/M2-O0-oracle-handoff.md`.

## Local tests

```bash
python3 -m unittest tools.oracle.test_generate_golden -v
```

The tests use only temporary metadata and synthetic artifact files. They do
not need `transformers`, vLLM, `huggingface_hub`, CUDA, or checkpoint weights.
