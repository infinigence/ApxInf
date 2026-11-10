# Qwen3.5 GPU1 Oracle Runner Design

## Goal

Provide the member1 checkpoint-specific executable required by the M2-O0
handoff. It runs only on the fixed GPU1 UUID, consumes the generator's frozen
job manifest, and writes the declared selective golden bundle. It is an offline
oracle tool, not a production runtime and not a fallback path.

## Boundary

The runner uses Transformers 5.15.1's native `Qwen3_5ForConditionalGeneration`
implementation with the local checkpoint revision. It may use CUDA and the
checkpoint's compressed-tensors loader. It must not invoke vLLM, an external
HTTP service, CPU inference, another model, or another GPU. Production Rust
services remain independent of this tool.

The generator owns control manifests and validates the final artifact set. The
runner receives `APXINF_ORACLE_JOB_MANIFEST` and
`APXINF_ORACLE_OUTPUT_DIR`, reads input/generation/selection metadata, and
writes only declared artifact files plus `artifact-report.json`. It never
modifies control files or stores weights/BF16 expansions in Git.

## Execution

The runner loads the model with `local_files_only=True`, `low_cpu_mem_usage=True`,
BF16 compute, and one CUDA device selected by the caller. It performs greedy
prefill/decode for the requested budget, retaining the model's native cache.
Forward hooks capture selected decoder-layer hidden states. The native cache is
inspected for selected full-attention KV and linear-attention recurrent/conv
state. Embedding, logits, and exact output tokens are captured directly.

If a requested internal state cannot be represented with a stable resolved
shape, the runner fails closed before writing a complete report. Every F32
artifact is finite and little-endian; logits include one top-1/top-2 margin per
generated token. EOS handling follows the frozen `[248046, 248044]` policy.

## Safety and evidence

The server command holds `/tmp/apxinf-gpu-job.lock`, sets the target GPU UUID,
records command output and peak VRAM, and writes the raw bundle under the
revision/commit artifact directory. A failed or partial run is an incident and
is not reused as correctness evidence. Only the manifest, schema, and approved
minimal golden files may leave the server artifact channel.

## Verification

Before GPU execution, unit tests cover manifest parsing, artifact path safety,
greedy stop behavior, state extraction shape checks, and report generation on a
tiny synthetic model. The real job is accepted only when the generator's own
hash/shape/dtype/finite-value checks pass and the recorded GPU UUID, revision,
contract, peak VRAM, command, and SHA256 values are complete.
