# Qwen3.5 Reference Runtime Design

Date: 2026-08-23  
Owner: member1 (server/runtime integrator)  
Model revision: `63768c10df38c0395e12ef49edac1bd539eaeeea`  
Contract: `apxinf.qwen38_27b.inference_interface.v1`

## Goal and non-goals

Build an auditable, text-only Qwen3.5 reference vertical slice that consumes
member2's stable `LoaderManifest` and `ProtocolRuntime` APIs, executes model
semantics on the selected CUDA device, streams tokens incrementally, and fails
closed on unknown checkpoint layouts, unsupported modalities, wrong GPU UUIDs,
or insufficient device budget. CPU code is limited to synthetic unit fixtures;
it is never a production fallback. Performance kernels, MTP, vision, long
context, and multi-request scheduling remain disabled until the reference path
has fresh correctness and reliability evidence.

## Architecture

### 1. Checkpoint inventory and loader

`apxinf-loader` remains the owner of SafeTensors header/index parsing and the
immutable manifest schema. `qwen35::loader` will consume that API, parse the
checkpoint `config.json` and `generation_config.json`, verify revision, model
vocabulary (`248320`), 64 layer types, EOS IDs (`248046`, `248044`), tensor
names, shapes, dtypes, and mixed quantization roles. It records a sorted
inventory digest and opens tensor payloads lazily by shard. Unknown layouts,
missing tensors, duplicate names, symlinks, tokenizer-vocabulary boundaries,
or implicit full-BF16 expansion are hard errors.

Synthetic W4 fixtures are separate from real checkpoint loading. They cover
K-packed weights, K-grouped-32 scales, N-packed zero-points, nibble order,
group boundaries and tails without copying real weight bytes.

### 2. W4 reference projection

`qwen35::w4` owns a checked logical view over packed `I32` words and BF16/F32
scales. The reference operation consumes nibbles directly and computes
`(q - zero_point) * scale` per output/group. A CUDA implementation is exposed
through a narrow `apxinf-cuda` production interface; the first GPU path may use
small, shape-checked dequant/GEMM kernels but must not materialize the complete
model in BF16. CPU reference math is used only by synthetic tests.

### 3. Layer state and executor

`qwen35::attention` implements RMSNorm, q/k norm, partial RoPE (first 64 of
256 dimensions), GQA, output gate (`q_proj` split into q and sigmoid gate),
full-attention KV append/read, MLP and residuals. `qwen35::gdn` implements
causal-conv ring state and FP32 eager recurrent state with explicit reset and
cancel guards. `qwen35::model` owns the 64-layer schedule (48 GDN layers,
16 full-attention layers), embedding, prefill, single-token decode, final norm,
BF16 lm-head/logits, greedy argmax, EOS/budget handling, and per-request state.
Each stage exposes a small reference method so selective oracle captures can be
compared before enabling the next stage. State is request-local and released by
RAII on success, cancellation, capacity rejection, and execution error.

### 4. Runtime/protocol adapter

`qwen35::runtime` adapts the executor to member2's `ProtocolRuntime` trait.
`capabilities()` reports measured model/device budget, `parallel_requests=1`,
`multimodal=false`, and `stub=false`. `start()` returns a genuinely incremental
`TokenStream`; every `next_token` performs at most one decode step and checks
the request cancellation token. Both request cancellation and stream
cancellation release the worker permit and all model state. Queue/capacity
errors map to the existing 503 semantics; execution errors are request-scoped
and cannot poison the next request.

### 5. Production entry

`src/main.rs` retains the old CLI only as an explicit compatibility command and
adds a strict `serve` command. Serve requires the pinned model directory,
revision, GPU UUID (`GPU-d074a13d-dbb6-fceb-4caf-a45be9be9281` for formal
jobs), bind address, and bounded queue. Startup validates CUDA availability,
UUID, config/inventory, calibrated memory budget, and a warmup forward; any
failure exits non-zero. No model/device fallback is attempted. `/health` and
`/v1/evaluations/generate` are mounted through the existing protocol service;
shutdown drops the worker and waits for state reclamation.

### 6. Evidence and release

Every real GPU job acquires `/tmp/apxinf-gpu-job.lock`, records queue ID,
commit, environment, command, GPU UUID, health-before/after, correctness,
reliability, logs, VRAM peak, incident (if any), and SHA256 manifest under the
revision/commit/stage artifact directory. `sha256sum -c manifest.sha256` is a
release gate. GPU0 correctness precedes reliability and performance. Xid
evidence remains `blocked/unknown` when kernel journal access is denied; it is
never synthesized.

## Error and recovery policy

Admission rejects empty prompts, token IDs outside `[0,248320)`, over-budget
requests, unsupported temperature/modality, and device-budget exhaustion before
launch. CUDA/NaN/Inf errors terminate only the active request, drop its state,
run a health probe, and keep the service unhealthy until a successful warmup
proves recovery. Any missing reliability boolean makes the campaign ineligible.

## Verification order

1. Manifest/config inventory and synthetic W4 pack/dequant tests.
2. CUDA W4 projection fixture against the CPU reference.
3. RMSNorm/embedding/RoPE and one GDN/full-attention layer captures.
4. Selective early/middle/late oracle comparisons.
5. 64-layer text-only prefill/decode trajectory, EOS, budget and cancellation.
6. Real `ProtocolRuntime` streaming, invalid gates and recovery.
7. GPU0 P1 correctness job, then reliability campaign and release artifact.

Until a stage has fresh evidence, its production capability is disabled rather
than replaced by a proxy, hard-coded token, another model, or another device.
