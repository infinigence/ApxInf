# Member3 campaign scaffolding

This directory contains offline preparation assets for the performance and
bonus replay lane. It does not run a server, benchmark, CUDA kernel, or model
download. The read-only evaluation contract remains under
`benchmarks/qwen38_4090/evaluation/` and must not be edited.

## Shape inventory

```text
python scripts/campaign/shape_inventory.py \
  --config fixtures/qwen35-metadata/config.json \
  --index fixtures/qwen35-metadata/model.safetensors.index.json \
  --output benchmarks/campaign/manifests/qwen38-shape-inventory.json
```

The output records fixture hashes, model/layer facts, tensor-name categories,
W4 packing direction examples, and the exact committed `metadata.total_size`.
The index has no per-tensor SafeTensors headers, so category byte totals are
intentionally left unknown rather than guessed.

## Paired A/B evidence

Use `manifests/w4-gemv-baseline.json` as a structure-only template. Validate it
with:

```text
python scripts/campaign/validate_experiment.py \
  benchmarks/campaign/manifests/w4-gemv-baseline.json --mode template
```

Before a server replay, fill every placeholder and run `--mode ready`. A
candidate must have feature-off/on commands, one primary variable, a full
commit SHA, pinned model/contract/input hashes, RTX 4090 environment details,
warmup 1 and measured 5 or more, finite latency samples with sample-CV <= 10%,
correctness/reliability/recovery evidence, raw artifact path and SHA256, and a
rollback SHA/command. Any NaN, OOM, fallback, Xid, failed recovery, or CV over
10% is rejected. A valid rejected record may stop before five repeats, but it
must preserve the failed gate and rollback evidence. Only GPU0 replay on an RTX
4090 with `evidence_scope=official` may be accepted; all other GPUs are
development evidence.

The server replay command is deliberately a placeholder in the template. It
must be replaced by the member1 `BASE_GOOD` service command and replayed under
the GPU job lock on GPU0 before entering the final report.
