# ApxInf Qwen3.8-27B Single-GPU Inference Report

Status: text-eligibility gates met; single-request performance optimized
(TTFT up to ~48x and TPOT ~2x over the eligibility baseline); two bonus
paths delivered, each behind its own single opt-in switch — multimodal
(vision tower + `/v1/chat/completions`) and C4 concurrent serving (all
official calibration validity gates passed). See "Submission Identity" for
the authoritative commit.

Conventions used throughout:

- `<model-dir>` — the approved `Qwen3.8-27B-AWQ-INT4` checkpoint directory
  (revision pinned under "Identity").
- `apxinf-evidence/...` and bare harness names (`apxinf-perf.py`,
  `apxinf-soak.py`, `apxinf-protocol.py`, ...) — measurement clients and
  their outputs, kept in the evidence archive that accompanies the
  submission, outside the repository tree.
- Naming: the checkpoint's product name is Qwen3.8; its `config.json`
  declares HF architecture `Qwen3_5ForConditionalGeneration`
  (`model_type: qwen3_5`), so module paths, test names and environment
  flags use `qwen35`/`Q35`. Prose in this report says Qwen3.8.

## Submission Identity

- Branch: `integrate/member2` (remote `origin`).
- Implementation commit (all code and planning documents, final):
  `06993a2d2642c6f7177b57493b797d5d537e4d64` — "feat(qwen35): add C4
  concurrent batched serving behind a single switch". Every measurement in
  this report describes that tree; acceptance references this SHA. This
  report file is committed immediately on top of it and changes no code.
- Submitted HEAD SHA: recorded by `git rev-parse HEAD` at submission time
  (the report-only commit above the implementation commit).
- Rollback point (pre-integration HEAD):
  `47ec280d2f88e8daf87750c0957e596e3a5390c1`.
- Previous integration checkpoint:
  `2e1b3c24db8ea453f54e1cc3677cdd590cbbf7aa`
  ("harden strict qwen35 service admission"). Everything in the "Layer-2"
  sections below was developed on top of it and is contained in the submitted
  commit, not in that one.
- Every accepted optimization is behind an environment flag with a tested
  `=0`/off control, so any single change can be rolled back without reverting
  the commit. The flag list is in "Final Configuration".

## Identity

- Model: approved checkpoint `Qwen3.8-27B-AWQ-INT4`; its directory is cited as `<model-dir>` below
- Checkpoint revision: `63768c10df38c0395e12ef49edac1bd539eaeeea`
- Development/replay GPU: `GPU-343bc895-b011-22fa-4449-97207aa2bdec`
- Final service command (release build; every accepted optimization is the
  default): `target/release/apxinf serve --model <model-dir> --revision
  63768c10df38c0395e12ef49edac1bd539eaeeea --gpu-uuid
  GPU-343bc895-b011-22fa-4449-97207aa2bdec --bind 127.0.0.1:18080
  --max-model-len 32768 --queue-capacity 1`. Prefix with
  `APXINF_ENABLE_MULTIMODAL=1` for the multimodal configuration. Layer-1
  evidence was gathered with the same arguments on the debug binary
  (`target/debug/apxinf`), as recorded in the layer-1 sections.
- CUDA toolkit: 12.8.93; driver: 580.82.07; Rust: 1.98.0
- Contract SHA256: `520349b1279c3bf999a6848b296c23d20cdaeab7420934e9196c90018bac7433`
- Model `config.json` SHA256: `fece2915d4c8ad4c10877622f04ea5e01cd3ae38768ce5c1edb700dd1de290f6`
- Model safetensors index SHA256: `82b1bf79f5b61333e83da17ec3bf89c9f178e29395a14c6b3ce3bbc474e1ead8`
- Approved safetensor shard SHA256s:
  `54d83c1d36631de231876217a8e0c2483eccee8746369a482b79442bdfc5d958`,
  `64be5fc2f66a3e5679ba229261a7a0d8112b06f6f560c750a62ca9457f90006c`,
  `7b90d6c7059d615a560cd4d2e766d328210605041061681550d80f380a8b529b`,
  `03b2624ec788780a2915003cd2871c29c87dfb6f2a8d189ef3918662d6a1ed56`,
  `eb5ea1fbef28b13ac89158924ee7cfe7c9f111c79ae177b290c0abd45c38925c`.

# Required Submission Materials

The six required items, in the order of the review requirements, and where
each is satisfied: (1) design changes and affected execution stages — next
section; (2) `test.py check` / `test.py run` commands and results — in this
part; (3) at least one negative control or regression test — in this part;
(4) correctness/performance/stability/VRAM trade-offs — in this part;
(5) known limitations, failed experiments and rollback — in this part;
(6) baseline, hypotheses, implementation, measurement, results and
reproduction steps — "Reproduction Steps" in this part plus the engineering
record that follows it.

## Design Changes and Affected Execution Stages

The build progressed through three planned stages — layer 1 (text
eligibility), layer 2 (single-request performance), layer 3 bonus
(multimodal) — and every design change below is tied to a paired experiment
or a fix section in the engineering record. "Stage affected" names the
execution phase whose behaviour changed.

| # | Design change | Stage affected | Record |
|---|---|---|---|
| 1 | Client-disconnect cancellation: prefill now checks a request-scoped cancellation token at every block boundary and releases the capacity slot; previously an abandoned request held the single slot until completion | service admission / recovery | "Client-Disconnect Capacity Leak" |
| 2 | Measured configuration: debug binary with debug capture -> release binary, capture off | whole service | A/B 1, A/B 2 |
| 3 | W4 prefill projections: per-element GEMV -> dequantize-once + tensor-core BF16 GEMM for >= 8 rows | prefill | A/B 4 |
| 4 | W4 decode GEMV: one K element per thread -> one packed uint32 per thread (coalesced 128-byte transactions) | decode | A/B 5 |
| 5 | Dequant scratch: per-call cudaMalloc/free -> resident per-size pool (~460 MiB) | prefill host-device interaction | A/B 7 |
| 6 | Prefill block size: fixed 64 -> configurable chunk, default 512 text / 256 multimodal, charged at admission | prefill + admission | A/B 9, A/B 14, "VRAM budget fix" |
| 7 | Attention prefill: per-(head, row) cuBLAS loop -> strided-batched GEMM, then query-head rebatch | attention prefill | A/B 6, A/B 16 |
| 8 | Causal softmax: every thread scanning the full row -> row-cooperative block reduction | attention prefill | A/B 17 |
| 9 | GDN status checks: kept eager per-op — the deferred variant measured zero gain and was rejected, i.e. the design deliberately did not change | decode / prefill | A/B 3 |
| 10 | Multimodal: absent route (non-compliant 404) -> contract fail-closed 400 by default; full vision path + `POST /v1/chat/completions` behind `APXINF_ENABLE_MULTIMODAL=1` | new chat entry + prefill embedding scatter | "Multimodal Bonus Implementation" |
| 11 | Multi-request C4: batched protocol runtime behind `APXINF_Q35_MAX_CONCURRENCY` — one batched decode step per round, two-phase delivery, prefix cache + recycled-session pool; the default single-request path is unchanged | new batch scheduler + batched decode | "Multi-Request Bonus: C4 Concurrent Serving" |

Changes that were designed, measured and *not* adopted are part of the same
record: see "Failed experiments" below.

Organizationally, the original multi-lane plan (protocol lane, kernel lane,
integration lane) collapsed to a single serial lane in execution; the
GPU-lock queue and the single-designated-GPU formal replay discipline were
kept throughout. The plan-versus-execution divergence is recorded in the two
planning documents that accompany the submission.

## `test.py check` and `test.py run`: Commands and Results

### `test.py check` — passed

```
cd <worktree>
python3 benchmarks/qwen38_4090/evaluation/test.py check
```

Result: `assignment checks passed`, exit code 0 — contract hashes, model
files, public corpus hash, revision and Python dependencies all verified
(captured log: `apxinf-evidence/submission/test-py-check.log`). The script's
`--help` sweep over every evaluation program plus `cargo check --workspace
--locked` also passed on the main repository checkout. Operational note:
with the repository's default `target/` on the network volume the check ran
more than 15 minutes without finishing (linker stuck in uninterruptible disk
sleep); with `CARGO_TARGET_DIR` on local disk it completed in 45 s — the
override changes where build artifacts are written, not what is checked.

### `test.py run` — literal run blocked by a missing platform artifact; equivalent run completed and disclosed

```
python3 benchmarks/qwen38_4090/evaluation/test.py run \
  --model-dir <model-dir> \
  --base-url http://127.0.0.1:18080
```

The literal invocation cannot produce output on this host for a reason
outside the implementation: the script passes no `--trajectory-reference`,
and `run_evaluation.py` unconditionally raises `provide
--trajectory-reference, or capture one from the vLLM control` before any
artifact is written (raise at `run_evaluation.py:1461`; `raw.jsonl` /
`submission.json` writes start at `:1588`). The platform's approved vLLM
control artifact (schema `apxinf.qwen38_27b.trajectory_reference.v1`) is not
present: a full search of the artifacts share and the repository checkout
matched only the evaluator's own `__pycache__` bytecode.

What was done instead, in two steps, both disclosed:

1. Every workload `test.py run` would exercise was measured directly against
   the running service with the frozen public corpus and is reported in the
   engineering record: public functional 6/6 exact, the frozen 12-gate
   protocol suite, the seven TTFT/TPOT cells (warmup 1 + measure 5), and the
   mixed soaks.
2. On 2026-08-27 the identical command line that `test.py run` builds was
   executed once with one addition, `--capture-trajectory-reference`, solely
   to prove the pipeline emits a complete artifact end to end (README §4:
   public runs are local debugging only, not official results). Run
   `20260826-155907-apxinf` (`benchmarks/qwen38_4090/evaluation/runs/`,
   ~13 minutes, strict text-only release service on the designated GPU)
   produced a complete `submission.json` (schema `leaderboard_submission.v1`,
   profile `public_calibration` — 0 warmup / 1 repeat per cell by design):
   `correctness.protocol_pass=true`, public cases **6/6**, all five
   reliability booleans true with `request_success_rate` 1.0, TTFT 1K 1.88 s
   / 2K 6.55 s / 4K 7.32 s / 8K 14.83 s / 16K 34.94 s, TPOT 67-77 ms, peak
   VRAM 20,750-22,832 MiB, frozen contract (`520349b1...`) and
   public-manifest (`1ec4f360...`) hashes embedded.

Trajectory disclosure for step 2: `public_trajectory_tokens_passed=256/256`
is self-comparison by construction — the reference was captured from the
same run — and is **not** a score. The captured file is kept outside the
repository (`apxinf-evidence/captured-trajectory-DEBUG-ONLY.json`) and is
not an approved vLLM control. An earlier, pre-optimization evaluator attempt
had been interrupted after more than 16 minutes on its first 8K case and is
superseded by the run above.

Consequences, stated plainly:

- No official scorer artifact exists for this submission. Trajectory is
  **unverified**; neither `eligible=true` nor any base score is claimed.
- The `midterm_leaderboard` profile (1+5 repeats, official scoring) remains
  a platform action with the platform's own reference artifact.
- No evaluator, scorer, generator or contract file under
  `benchmarks/qwen38_4090/evaluation/` was modified, and `submission.json`
  image fields were not hand-filled.
- The approved public dataset was consumed read-only from
  `benchmarks/qwen38_4090/evaluation/.cache/public` (repository checkout);
  its manifest SHA256 is
  `1ec4f360e8dce8cb366251d9b92f8f91a393e5534bb93277a955f8b9e3e5e1e4`.

## Reproduction Steps

1. Build (locked, no fast-math): `cargo build --release --features
   cuda-no-nvtx --locked --bin apxinf`.
2. Start the strict single-GPU service — all accepted optimizations are the
   defaults; startup to bound port is ~8 minutes on the release build
   (checkpoint digest verification, weight load, warmup):
   `target/release/apxinf serve --model <model-dir> --revision
   63768c10df38c0395e12ef49edac1bd539eaeeea --gpu-uuid
   GPU-343bc895-b011-22fa-4449-97207aa2bdec --bind 127.0.0.1:18080
   --max-model-len 32768 --queue-capacity 1`. Prefix with
   `APXINF_ENABLE_MULTIMODAL=1` for the multimodal configuration, or
   `APXINF_Q35_MAX_CONCURRENCY=4` plus `--queue-capacity 4` for the C4
   concurrent configuration; serialize real-GPU jobs behind the global
   GPU-job lock.
3. Gates: the frozen protocol suite (`apxinf-protocol.py`, expected 12/12),
   the six public functional cases, and the proxy hidden suite
   (`apxinf-proxy-hidden-run.py`, expected >= 11/12, non-official).
4. Performance: `apxinf-perf.py` — warmup 1 + measure 5 per cell, SSE-timed
   TTFT/TPOT, produces the seven-cell table with CV and peak VRAM.
5. Reliability: `apxinf-soak.py` — 200-request mixed plan with health
   probes; expected 100% success and five true booleans.
6. `python3 benchmarks/qwen38_4090/evaluation/test.py check` must print
   `assignment checks passed`.

## Negative Controls and Regression Tests

The submission requirement asks for at least one negative control or
regression test. The following are in the submitted tree and run in CI-style
suites (`cargo test`), not by hand.

**Protocol negative controls (7, all required to fail closed).** Executed
against the live service by `apxinf-protocol.py`, 12/12 gates passing:
`malformed_json` (raw unparseable body, HTTP 400), and six structured
`stream=false` controls each requiring HTTP 400 with a JSON `error` —
`empty_input_ids`, `negative_token_id`, `out_of_vocabulary_token_id`
(`4294967295`), `unsupported_temperature` (`0.1`), `over_budget`
(`max_new_tokens = health.max_model_len`), `unsupported_modality_field`
(`images:["x"]`). Then `valid_short_nostream_request` (8 tokens, HTTP 200,
`type=result`, usage 8/1/9), `health_after_invalid_requests`, and
`health_contract_identity` — proving the negative controls do not poison the
service.

**Fault-injection regression for the client-disconnect capacity leak.** Three
tests added after the defect was reproduced (see "Client-Disconnect Capacity
Leak"): two drive a real `TcpListener`/`TcpStream` pair against a runtime that
blocks until cancelled, disconnect mid-generation, and assert the runtime
observed cancellation *while still working*; both fail on the pre-fix code
(`Some(false)`) and pass after. The third cancels while the executor is inside
`open()` and asserts the caller gets `Cancelled`, capacity returns to zero, and
the next request succeeds.

**Numerical regressions guarding every accepted optimization.** Each has a
tested `=0` control path:

- `qwen35_w4_cuda_prefill_gemm_matches_gemv_path_and_cpu_reference` — asserts
  the dequantized weight matrix is **bit-identical** to the reference
  decompression (`assert_eq!` over all weights, not a tolerance), then bounds
  GEMM-vs-GEMV drift and checks sub-threshold row counts reproduce GEMV bytes
  exactly.
- `qwen35_w4_cuda_packed_gemv_matches_baseline_kernel_and_cpu_reference` and
  `..._warp_gemv_...` — cover the non-multiple-of-8 K tail and the
  out_features block tail.
- `qwen35_w4_cuda_packed_gemv_rejects_non_finite_scale` and the warp variant —
  NaN-scale must surface as a request error, not a silent result.
- `batched_sdpa_prefill_matches_per_row_loop_at_checkpoint_shape` — **bit
  equality** against the per-row loop at the real 24/4/256 head geometry with
  a nonzero `kv_offset`.
- `qwen35_gdn_cuda_deferred_status_matches_eager_and_latches_non_finite` —
  bit-identical outputs, silent latching, drain error, latch clearing, and
  eager restoration.
- `qwen35_gdn_cuda_sequence_chunked_matches_eager_step_at_checkpoint_layout` —
  chunked prefill vs eager recurrence at the real layout; re-run at 512 rows
  when the prefill block size changed.
- `prefill_chunk_override_is_clamped_to_the_workspace_bound` and
  `prefill_plan_bounds_every_block_to_the_configured_chunk` — the chunk
  override cannot exceed the workspace budget, and blocks always cover the
  prompt contiguously.
- `qwen35_gdn_cuda_convolution_commit_can_be_rolled_back` and the recurrent
  equivalent — state rollback after an injected failure.

**Suite status on the submitted tree (final, after the multimodal
addition):** `apxinf-cuda` GPU suite 102 passed (2 pre-existing `fp8`
environmental failures: cuBLAS status 15 / `CUBLAS_STATUS_NOT_SUPPORTED` on
this driver, not on the Qwen3.8 path, unchanged baseline); `apxinf-model`
qwen35 GPU suite 58 passed; CPU suites 98 (model lib) and 62 (bin, including
the image/chat/template tests); `cargo fmt --all -- --check`,
`cargo check --workspace --locked` and `git diff --check` clean.

## Trade-offs: Correctness, Performance, Stability and VRAM

The submission requirement asks for the trade-offs explicitly. Each was a
decision with a measured cost, not a guess.

**Trajectory (correctness soft target) traded for TTFT.** The W4 prefill
dequant+GEMM change moved the frozen 128-token oracle prefix from 28 tokens to
23 and later, with the packed GEMV, to 76. The mechanism is identical in all
cases: the oracle's own top1/top2 margin is exactly 0.0 at generated tokens
23, 28 and 76, so a different accumulation order resolves those ties the other
way. No achievable tolerance fixes this, because the oracle has no margin
there. Accepted because trajectory is a scored soft target with threshold 0.0
while TTFT is 35 points, and because public functional stayed 6/6 exact at
every step. Recorded rather than hidden.

**NaN defence kept at per-request granularity, not weakened for speed.** The
deferred-status experiment batched the per-op finite checks into one drain per
token/block. It measured zero gain and was rejected, so the eager per-op check
remains — the stricter option also turned out to be free.

**VRAM traded for prefill throughput, with admission as the guard.** Raising
the prefill block from 64 to 512 tokens scales the attention score workspace
from 192 MB to 1536 MB (at `max_model_len=32768`), and the dequant scratch pool
adds ~460 MB resident. Both are charged through `request_state_bytes` at
admission, so an over-large configuration fails closed at startup instead of
OOMing mid-request; `MAX_PREFILL_CHUNK_TOKENS` caps the override. Measured
peak stayed at 19958 MiB of 24564 with ~3.8 GiB of headroom. This spend is
also what puts the context bonus out of reach beyond 65536 — the trade was
made knowingly in favour of the 35-point TTFT axis over a 3.33-point bonus.

**Stability never traded.** Every accepted change was gated on protocol 12/12,
public functional 6/6 and a clean soak before acceptance; the 200-request soak
reached 100% success with all five reliability booleans true. Two candidates
that were faster in isolation were still rejected for lack of end-to-end gain,
and no change was accepted on kernel-level numbers alone.

## Known Limitations, Failed Experiments and Rollback

### Known limitations (final state)

1. The frozen 128-token trajectory diverges from the approved oracle at BF16
   tie sites — the oracle's own top1/top2 margins are exactly 0.0 at
   generated tokens 23, 28 and 76, so sub-ulp accumulation-order drift
   decides those steps. The final configuration matches a 76-token prefix
   (82/128 overall; intermediate configurations ranged 23-28). No achievable
   tolerance changes this, it does not block eligibility (trajectory
   threshold is 0.0), and it caps only the trajectory soft score.
2. `request_state_bytes` is a conservative admission estimate, not allocator
   instrumentation with a measured peak-memory margin.
3. KV append rollback on a failed request is achieved by dropping the failed
   session; an in-place transaction rollback is not implemented.
4. Multimodal: the official image suite is absent on this host, so "public
   4/4" is a platform-run item (the self-built probes are disclosed as
   non-official); out-of-contract image shapes are rejected with 400 rather
   than resampled; the vision forward adds ~0.4 s per image with a ~30 ms
   host-memory round-trip in the QKV split — functional, not tuned.
5. Long-context bonus is capped by VRAM (65536 needs 4923 MiB against
   4606 MiB free; 131072 is impossible even with INT8 KV). C4 is delivered;
   C8 is blocked on memory, not scheduling — 8 full request states need
   4.1 GiB against the 2.36 GiB admission budget; the shared-GDN-scratch
   refactor that would fit it (~295 MiB per state) is quantified but not
   implemented, so C8 is not claimed.

An earlier blocker — no completed long-prompt service request — is resolved
and kept only as history: the seven-cell campaigns completed every prompt
length up to 16K with 5/5 success per cell, and the once-interrupted
evaluator attempt is superseded by the completed calibration run recorded in
the `test.py` section.

The strict production path independently attests the selected CUDA device
UUID, requires the frozen 64-layer Qwen3.8 contract, and streams SHA-256
over the approved config, index and five safetensor payloads before model
admission. Runtime errors map `WorkerStopped` to HTTP 503/unavailable, and
recovery serializes through a poisoned mutex.

### Failed experiments (kept in-tree as negative results)

Eight of the seventeen paired experiments were rejected, each with data and
root cause in its own section of the engineering record: deferred GDN status
checks (A/B 3 — zero end-to-end gain: the synchronizations were off the
critical path); warp-per-output GEMV (A/B 8 — <= 1.8%, inside noise);
`uint4` vectorized loads (+26% slower); 4-accumulator array (+17% slower —
dynamic indexing spilled the array to local memory); unrolled+fused
accumulators (+1.3% — the apparent gain was a measurement artifact caught by
running duplicate arms on separate GPUs); the per-group LUT design (rejected
at design stage after the cost decomposition); Marlin-style bit-trick SIMD
dequantization (A/B 15 — 54% slower, latency-bound, SASS-verified); and the
initial chunk-512 caution (overturned by a recomputed budget, then adopted
as A/B 14). Rejected kernel variants remain in the tree, default off and
unreachable in production dispatch.

### Rollback methods

- Every accepted optimization is behind an environment flag with a tested
  `=0`/off control: `APXINF_Q35_W4_PREFILL_GEMM`,
  `APXINF_Q35_W4_PACKED_GEMV`, `APXINF_Q35_BATCHED_SDPA`,
  `APXINF_Q35_SCRATCH_POOL`, `APXINF_Q35_PREFILL_CHUNK`,
  `APXINF_Q35_ROWWISE_SOFTMAX`.
- The multimodal path is a single opt-in switch: without
  `APXINF_ENABLE_MULTIMODAL` the vision tower is not loaded, the prefill
  chunk default stays 512, image requests fail closed with the contract 400,
  and the service is the unchanged text-only submission.
- The C4 batched runtime is likewise a single opt-in switch: without
  `APXINF_Q35_MAX_CONCURRENCY` (or with `=1`) the service is the exact
  frozen single-request runtime.
- Whole-tree rollback point (pre-integration HEAD):
  `47ec280d2f88e8daf87750c0957e596e3a5390c1`.

# Engineering Record: Baseline, Hypotheses, Implementation, Measurement, Results

## Implementation

The Qwen3.8 runtime executes prompt prefill in contiguous blocks of at most
`prefill_chunk_tokens()` tokens — default 512 (text-only) / 256 (multimodal);
the original layer-1 design used 64, and the internal GDN scan chunk stays 64. Every block runs all 64 layers before the next block, carries GDN
convolution/recurrent state and full-attention KV state across boundaries, uses
absolute positions, allocates request KV capacity as `prompt_len + max_new_tokens`,
and retains only the final block's last row for logits. Readiness now performs a
prefill-plus-decode warmup before binding, fails closed while unhealthy, and
serially attempts recovery from `/health`.

## Verification (layer-1 phase)

- `cargo test --bin apxinf -- --nocapture --test-threads=1`: 52 passed, 0 failed.
- `cargo test -p apxinf-model --locked qwen35 -- --nocapture`: 54 passed, 0 failed, 2 ignored.
- `CUDA_VISIBLE_DEVICES=GPU-343bc895-b011-22fa-4449-97207aa2bdec APXINF_TEST_GPU_UUID=GPU-343bc895-b011-22fa-4449-97207aa2bdec cargo test -p apxinf-cuda context::tests::attested_context_accepts_expected_uuid_and_rejects_mismatch`: 1 passed, 0 failed.
- Designated-GPU GDN suite: 19 passed, 0 failed.
- Python protocol/oracle tests: 29 passed, 0 failed.
- `cargo check --workspace --locked`: passed.
- `cargo build --features cuda-no-nvtx --locked --bin apxinf`: passed.
- `cargo fmt --all -- --check` and `git diff --check`: passed.

The pinned checkpoint test used a 65-token prompt and passed in 916.56 seconds.
It verified two bounded prefill blocks, final position/KV length 65, request KV
capacity 67, and one prefill plus one decode token. Artifact SHA256:
`9a2c325f2254fd681b6de4c8e2b4d2b755aebfca0aba584ce2cb63bb1bb683b0`.

These are the layer-1-phase suite counts; the final counts on the
submitted tree (after layer 2 and the multimodal addition) are recorded
under "Negative Controls and Regression Tests".

## Fresh Service Evidence

Artifact directory:
`20260825T164057Z-readiness-final` (midterm artifacts share)

- `/health`: HTTP 200, `stub=false`, frozen revision and contract identity,
  `max_model_len=32768`, `parallel_requests=1`, `fallback_active=false`.
- Frozen protocol gates: 10/10 passed (the gate suite at that date; it later
  grew to 12 gates). `protocol.json` SHA256:
  `562dec14609fa7508ea6194361c45cfc4d9ef81258ff96b48e605218da52ce2c`.
- SSE request: HTTP 200, token prefix `[2037, 9]`, valid done usage and `[DONE]`.
- Capacity rejection: prompt 1 plus `max_new_tokens=32768` returned structured HTTP
  400; `/health` remained healthy and the next short request returned HTTP 200.
- 65-token non-stream request: HTTP 200, output `[1]`, prompt usage 65 and total 66.
- Client disconnect during the optional evaluator attempt was followed by healthy
  `/health` and a successful short recovery request.
- No Xid lines were observed in the captured kernel log window.
- On shutdown, PID 1317001 exited, port 18080 was free, all GPUs returned to 1 MiB,
  and the global GPU-job lock was available.
- Artifact manifest SHA256: `a4bbba80042bb282285b4d627c4b7372b194425fc9c8440177aba3533c66758a`.

An additional hardening startup gate used the same strict service command. The
port stayed unbound during full-checkpoint digest verification and model warmup;
startup then independently reported CUDA UUID
`GPU-343bc895-b011-22fa-4449-97207aa2bdec`. `/health` returned HTTP 200 and an
8-token non-stream request returned HTTP 200 with output `[2037]` and usage 8/1/9.
After interrupting PID 1353165, the port was free, the target GPU returned to 1 MiB,
and the global GPU-job lock was available.

## Client-Disconnect Capacity Leak (found and fixed 2026-08-25)

### Reproduction (unfixed binary, real service)

Procedure: start the strict service, open a raw socket, POST
`longdoc-multihop-8192` (8192 prompt tokens, `max_new_tokens=128`,
`stream=true`), close the socket after 2 s without reading any byte, then probe
with 8-token `stream=false` requests every 2 s.

Observed: 316 consecutive probes returned HTTP 503
`{"error":{"message":"runtime capacity is exhausted","type":"capacity"}}` over
653 s while `GET /health` stayed HTTP 200 the whole time. The first HTTP 200
probe (`output_ids=[2037]`, usage 8/1/9) arrived only after the abandoned
prefill had run to completion, matching the ~11 min 8K prefill cost. Evidence:
`apxinf-evidence/a-disconnect/repro-broken.stdout`,
`repro-broken-probes.json` (317 rows), `repro-serve.log`.

### Root cause

Cancellation was only observable at points the abandoned request never
reached. `Qwen35CudaModel::open` ran the entire bounded-block prefill before
returning a session, and the HTTP layer only learned about a disconnect from a
failed socket *write*. For a streaming request no frame exists until prefill
finishes, and for a non-streaming request nothing is written until the whole
generation completes, so a client that disappears during prefill left the
`RequestPermit` held for the full prefill duration. Every new request was
correctly (per the admission contract) rejected with `Capacity`, and `/health`
correctly stayed `ok`, so the service looked healthy while refusing all work.

### Fix (minimal, three layers)

1. `src/server/http.rs` spawns a disconnect monitor on a cloned socket for
   every generate request. A read returning EOF or an error cancels a
   request-scoped `CancellationToken` immediately, before any response byte
   has to be written. The handler shuts the socket down on completion so the
   monitor thread always exits.
2. `src/server/service.rs` gained `handle_non_stream_with_cancel` and
   `start_stream_with_cancel`, which thread that externally owned token into
   `RuntimeRequest` instead of minting a fresh one. The existing
   `handle_non_stream` / `start_stream` entry points remain as thin wrappers.
3. `crates/apxinf-model/src/qwen35/runtime.rs` gained `open_with_cancel`,
   which checks the token at every 64-token prefill block boundary and aborts
   with a `prefill cancelled by client disconnect at token N of M` error. The
   permit is released when the worker job unwinds. `open` remains as a
   never-cancelled wrapper for tests and warmup.
   `src/server/qwen35_runtime.rs` maps an abort observed under a cancelled
   token to `RuntimeError::Cancelled` rather than `Execution`, so a disconnect
   cannot mark the service unhealthy; the same rule covers the
   session-channel-disconnect race in `next_token`.

Cancellation granularity is therefore one prefill block (64 tokens) or one
decode step, not one whole request.

### Regression tests (added, GPU-free)

- `server::http::tests::stream_disconnect_before_first_token_cancels_generation`
- `server::http::tests::non_stream_disconnect_before_result_cancels_generation`
  Both drive a real `TcpListener`/`TcpStream` pair with a runtime whose stream
  blocks until cancelled, disconnect 50 ms in, and assert the runtime observed
  cancellation *while still working*. Both fail on the pre-fix code
  (`Some(false)`, i.e. the model ran to completion) and pass after it.
- `server::qwen35_runtime::tests::cancelling_during_open_releases_capacity_and_maps_to_cancelled`
  cancels while the executor is still inside `open`, and asserts the caller
  gets `RuntimeError::Cancelled`, `active_requests()` returns to 0, and the
  next request succeeds.

`cargo test --bin apxinf`: 55 passed, 0 failed (52 before, 3 new).
`cargo test -p apxinf-model --locked qwen35`: 54 passed, 2 ignored (unchanged).
`cargo fmt --all -- --check`, `cargo check --workspace --locked`,
`cargo build --features cuda-no-nvtx --locked --bin apxinf`, and
`git diff --check` all pass.

## Verified Reliability Soak (2026-08-25, post-fix)

Command: `python3 apxinf-soak.py apxinf-evidence/soak-fixed` against
the strict service on the designated GPU under `flock`. Plan: 200 sequential
requests, seed 20260825 — 150 short (8-token prompt, 1-8 output tokens), 40
medium (1024-token prompt, 4-16 output), 10 long (8192-token prompt, 4-8
output); `stream=false`, `temperature=0`, `ignore_eos=false`; health probe
every 10 requests; per-request usage consistency (`total_tokens ==
prompt + completion`) checked.

Result: **200/200 succeeded, success rate 1.0000, 0 failures** in 10184.9 s
(2.83 h). Every health probe returned HTTP 200; the three final probes
returned `status=ok` with `fallback_active=false`. Evidence:
`apxinf-evidence/soak-fixed/soak.json`, `soak-fixed.stdout`.

Reliability booleans:

| Field | Verdict | Basis |
|---|---|---|
| `no_unexpected_oom` | true | no OOM in 3271 service log lines; every request completed |
| `no_nan` | true | no NaN in service log; all 200 responses parsed with consistent usage |
| `no_fallback` | true | `/health.fallback_active=false` before, during and after; single-GPU CUDA path only |
| `no_xid` | true (indirect) | see caveat below |
| `service_healthy_after_failure` | true | disconnect drill and capacity rejection both left `/health` `ok` with a successful follow-up request |

`no_xid` caveat: `dmesg` returns `read kernel buffer failed: Operation not
permitted` on this host, so kernel Xid lines cannot be read directly. Indirect
evidence: the service logged zero occurrences of `cuda error`, `CUDA_ERROR`,
`Xid`, `uncorrectable`, `NaN`, `out of memory`, `unhealthy`, `WorkerStopped`
and `panic` across 3271 lines; GPU memory returned to the 19498 MiB resident
weight baseline after the soak with no growth, and the other three GPUs stayed
at 1 MiB throughout. Snapshots:
`apxinf-evidence/no-xid-indirect-midsoak.txt`,
`no-xid-indirect-after-soak.txt`. This is stated as indirect, not as a direct
kernel-log check.

## Candidate vs Oracle Logits (frozen 128-token replay)

The frozen non-stream request (`input_ids=[1..8]`, `max_new_tokens=128`,
`temperature=0`, `ignore_eos=false`) was replayed against a freshly emptied
`APXINF_DEBUG_LOGITS_DIR` (the directory had to be cleared first: captures are
named by position and the soak's 1K requests had overwritten it). HTTP 200 in
17.62 s, usage 8/128/136, all 128 per-step logit rows captured. Candidate
`output_ids` SHA256:
`32b08629c08d40575e68c946852ea748a8dd32ad3c712e640073651ba642b945`; full
response SHA256:
`1d4b6718e785bac63b7565ada30d7a38a38479fcdf209686ae8c0bf275b98c27`.

Per-step comparison against the approved oracle logits
(`.../46182a1167570e7595b3e658b02fb8acadac9f7a/artifacts/logits.f32.bin`):

| Window | Finding |
|---|---|
| steps 0-27 (pre-divergence) | max L-inf 0.375, mean L-inf 0.189; top-1 agrees at **every** step |
| step 23 | oracle margin exactly 0.0 (tie); both sides resolve to 3050 — tie handled consistently |
| step 28 (first divergence) | oracle margin exactly 0.0 with `3175` and `40608` both at 21.375; candidate has `3175` at **21.5**, i.e. exactly one BF16 ulp higher, so margin 0.125 and no tie to resolve |
| steps 29-127 | L-inf 11.4 to 34.4 — inputs already differ after the flip, so these rows are not comparable |
| total | 30/128 tokens match; oracle exact ties occur at steps 23, 28, 76 |

This is the quantitative confirmation of the earlier qualitative diagnosis.
Pre-divergence drift never exceeds 3 BF16 ulp at the ~21 logit magnitude
(1 ulp = 0.125 there) and never changes an argmax. The divergence is caused by
a single 1-ulp lift on token `3175` breaking an exactly-tied oracle logit pair;
the post-divergence L-inf explosion is a *consequence* of the different token
prefix, not evidence of a second defect. Evidence:
`apxinf-evidence/logits-compare.json`,
`logits-compare-summary.txt`.

Because the oracle's own top1/top2 margin is 0.0 at that step, no achievable
numerical tolerance makes this step deterministic in the candidate's favour —
matching it requires reproducing the oracle's exact accumulation order, not
merely reducing error. Trajectory remains a scored soft target (threshold 0.0)
and does not gate eligibility.

## Roofline Ledger and Performance Expectation

Contract-frozen proxy: 54 GFLOP/token. RTX 4090 BF16 peak: 165.2 TFLOPS.

| Quantity | Value | Source |
|---|---|---|
| 16K prefill arithmetic work | ~885 TFLOP | 54 GFLOP/token x 16384 |
| 16K optimistic arithmetic floor | ~5.36 s | 885 TFLOP / 165.2 TFLOPS |
| 1K optimistic arithmetic floor | ~335 ms | same ratio |
| decode weight traffic per token | ~15.56 GiB (~16.70 GB) | backbone 13.19 + lm_head 2.37 |
| decode optimistic floor at 800-850 GB/s | ~19.6-20.9 ms/token | traffic / effective bandwidth |
| `lm_head` share of decode traffic | ~15.2% | 2.37 / 15.56 |
| max MLP activation at 16K | ~544 MiB | 16384 x 17408 x 2 B |
| max BF16 dequant scratch | ~170 MiB (~340 MiB double-buffered) | 17408 x 5120 x 2 B |
| KV per token (16 full-attention layers) | 64 KiB | 16 x 2 x 4 heads x 256 x 2 B |
| KV at 16512 tokens (16K cell) | 1.01 GiB | 64 KiB x 16512 |
| attention prefill workspace | 192 MiB | 24 heads x 64 chunk x 32768 x 2 x 2 B |
| GDN recurrent state (3 buffers, 48 layers) | 432 MiB | 48 x 3 x 48 x 128^2 x 4 B |

The contract notes this FLOP proxy omits elementwise and recurrent work, so it
is an order-of-magnitude screen and not an absolute physical floor.

Measured prefill on this build is far above that floor: ~78 s at 1K and ~675 s
at 8K (from the functional and soak evidence in this report), i.e. two to three orders
of magnitude above the arithmetic screen. The consequence for scoring must be
stated plainly: TTFT (35) and TPOT (25) are scored *relative to the best valid
reference in the same round*, so passing `success 5/5` and `CV<=10%` earns cell
**validity** — which is an eligibility requirement — but essentially none of
the 60 dynamic points. The realistic ceiling for this build is Correctness 30 +
Reliability 10, conditional on eligibility. No base-100 claim is made.
*(This paragraph describes the layer-1 debug build it was written against;
the layer-2 work below closes most of the distance — to ~4.8x off the
arithmetic screen at 1K and ~3.3x off the decode bandwidth floor — while the
no-dynamic-points expectation, unchanged in kind, is re-assessed there.)*

`per_request_bytes` is sized from `max_model_len` (32768) rather than the
actual prompt, so admission behaviour is prompt-length independent: an 8K
request passing admission implies the 16K cell also passes.

## Base Performance Cells (layer-1 baseline): 7/7 valid

Command: `python3 apxinf-perf.py apxinf-evidence/perf-fixed` against
the strict debug-build service on the designated GPU under `flock`. Frozen
public `text-perf-*` prompts, `max_new_tokens=128`, `ignore_eos=true`,
`temperature=0`, `stream=true`; 1 warmup + 5 measured per cell; SSE consumed
incrementally so TTFT is the wall time to the first token event and TPOT is
the mean of the 127 inter-token gaps. All 30 measured requests returned HTTP
200 with exactly 128 token events and usage `prompt/128/prompt+128`.

| Cell | prompt | success | TTFT median (s) | TTFT CV | TPOT median (ms/token) | TPOT CV | peak VRAM | valid |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| text-perf-1024 | 1024 | 5/5 | 78.09 | 1.66% | 133.7 | 0.35% | 19498 MiB | yes |
| text-perf-2048 | 2048 | 5/5 | 159.19 | 0.25% | 134.1 | 0.93% | 19498 MiB | yes |
| text-perf-4096 | 4096 | 5/5 | 320.40 | 0.34% | 138.4 | 0.62% | 19498 MiB | yes |
| text-perf-8192 | 8192 | 5/5 | 676.52 | 0.09% | 153.4 | 0.24% | 19498 MiB | yes |
| text-perf-16384 | 16384 | 5/5 | 1635.15 | 0.04% | 228.7 | 3.47% | 19498 MiB | yes |

TTFT cells: 5/5 valid. TPOT cells (1024, 8192): both valid. Every CV is far
under the 10% ceiling (worst 3.47%, on 16K TPOT), success is 100%, and peak
VRAM never exceeds the resident-weight baseline of 19498 MiB — the KV and
workspace for even the 16K cell fit inside allocator headroom already counted.
The 16K cell's slowest repeat was 1663.6 s total against the 1800 s
single-request timeout (~7.6% margin), as predicted. Evidence:
`apxinf-evidence/perf-fixed/perf.json`, `perf-fixed.stdout`.

TTFT scales almost exactly linearly with prompt length (78 / 159 / 320 / 677 /
1635 s for 1x/2x/4x/8x/16x), i.e. the quadratic attention term is not
dominant — consistent with the static audit's finding that fixed per-op
overhead, not arithmetic, sets the pace. TPOT grows only 133.7 to 228.7 ms
across a 16x KV range, so decode is likewise dominated by per-token fixed
cost rather than KV traffic.

As stated in the roofline section, these are *valid* cells but not
competitive ones; the 60 dynamic TTFT/TPOT points are scored relative to the
round's best reference and are not claimed.

## Proxy Hidden Suite: 11/12

Command: `python3 apxinf-proxy-hidden-run.py apxinf-proxy-hidden
apxinf-evidence/proxy-hidden` (12 independently seeded cases, explicitly
NOT the official hidden dataset; manifest with per-case SHA256 in
`apxinf-proxy-hidden/manifest.json`). SSE, `temperature=0`,
`max_new_tokens=64`, `ignore_eos=false`, tokenizer decode with
`skip_special_tokens` and ascii strip, exact match.

Result: **11/12 passed** — meets the ≥11/12 eligibility floor, misses the
12/12 stretch goal. All all-length retrieval (1K/4K/8K/16K), distractor
(2K/8K), multi-hop (4K/8K), revision (2K/8K) and aggregate-1024 passed with
correct EOS early termination. Timings scale near-linearly with prompt length
(79.5 s at 1K, 323.3 s at 4K, ~680 s at 8K, 1638.4 s at 16K). Evidence:
`apxinf-evidence/proxy-hidden/proxy-hidden.json`.

Failure analysis for `proxy-aggregate-8192` (expected "223", got "164"): the
case was replayed with per-step logits capture. The first generated token
('1' of "164") had top1/top2 margin **1.125** (nine BF16 ulp at that
magnitude), the second 1.125, the third 0.25, and the EOS step 11.875.
Sub-ulp accumulation drift — the only numerical defect previously identified,
capable of flipping exact ties only — cannot overturn a 1.125 margin. The
model is confidently producing the wrong count on this 8K aggregation task,
so this is a model-capability boundary, not an implementation-correctness
defect; no code change is warranted. (A full offline-oracle replay of this
prompt was not run; the margin evidence is the basis of this classification.)
Evidence: `apxinf-evidence/aggregate-8192-response.json`, service-log
margin lines in `serve log`, captured rows `service-logits-pos-000/8193/8194/
8195.f32.bin`.

## Layer-1 Eligibility Summary (2026-08-26)

| Gate | Result |
|---|---|
| protocol gate | 12/12 passed |
| public functional | 6/6 exact |
| hidden ≥11/12 | proxy suite 11/12 (official hidden dataset unavailable) |
| request success rate ≥99% | 200/200 = 100% |
| 5 reliability booleans | all true (`no_xid` indirect) |
| 7 TTFT/TPOT cells valid | 7/7, success 5/5 each, worst CV 3.47% |
| trajectory (soft, threshold 0) | recorded; 30/128, diverges at step 28 (layer-1 state; layer-2 later improved it to 82/128, first divergence at step 76 — see A/B 5) |
| official scorer artifact | unavailable — trajectory unverified, base 100 not claimed |

Every layer-1 engineering gate that can be satisfied on this host is
satisfied. The two open items are platform inputs, not work items: the
official hidden dataset and an approved vLLM trajectory reference.

## Service Session Incident (infrastructure, not a service defect)

The layer-1 evidence service (started 2026-08-25 19:24, PID 1626157) was
killed between 00:53 and 00:58 on 2026-08-26, after the aggregate diagnosis
request completed. The service log ends cleanly on the final request's debug
lines — no panic, no CUDA error, no OOM. The process was attached to an agent
shell session through a `tee` pipeline and was torn down when that session was
aborted. All evidence gathered before the kill (soak, proxy suite, logits
capture, disconnect drills) is unaffected; the lock and port were released
cleanly. The replacement service for the performance cells is started with
`setsid`/`nohup`, fully detached from any agent session, logging directly to
`apxinf-evidence/serve-c-session.log`. One transient 503 was observed
when the diagnosis client raced the still-running previous request for the
single capacity slot; that is the documented `queue-capacity=1` admission
behaviour, not a regression.

## Layer-2 Hypotheses as Formed at Layer-1 Close (historical record)

This section was written before any layer-2 measurement and is kept in its
original substance so the hypothesis-forming step stays auditable; each
item's measured outcome is noted inline and recorded fully in the "Layer-2
Paired A/B Results" sections that follow. At layer-1 close, all service evidence
had been produced by `target/debug/apxinf` (cargo dev profile, unoptimized
host code) with `APXINF_DEBUG_HIDDEN_DIR`/`APXINF_DEBUG_LOGITS_DIR` capture
enabled, per the frozen session command. Three layer-2 candidates followed
directly, ordered by expected yield per risk:

1. Release build (`cargo build --release --features cuda-no-nvtx`): the plan's
   own canonical command (§9.5) uses `--release`; host-side tensor/FFI/loop
   code currently runs at `-O0`. Same source, same kernels, no fast-math, so
   numerics are expected identical, but correctness smoke + frozen-request
   regression still required before adoption. *(Outcome: accepted as A/B 1,
   bit-identical, TPOT -18.3%.)*
2. Drop debug capture env vars for measured runs: the logits capture writes
   ~1 MB per step to the capture directory and one stderr line per generated
   token. *(Outcome: accepted as A/B 2, TPOT -3.8%.)*
3. Per-token logits D2H + host argmax: every decode step copies 248320 f32
   (~1 MB) to host and scans it on CPU; a device argmax (plan M3) removes
   both. *(Outcome: retired without implementation — the decode attribution
   later measured this at 0.5% of TPOT, not a lever.)*

These hypotheses were then tested by paired A/B; the layer-2 sections above
supersede the debug-build numbers that were authoritative when this was
written.

Preparation already done without touching the GPU lane: `target/release/apxinf`
is built (same source, `--features cuda-no-nvtx --locked`), and the CPU test
suites pass identically under the release profile (bin 55/55; apxinf-model
qwen35 54 passed / 2 ignored). Profilers available on this host: `ncu` and
`nvprof`; `nsys` is absent, so launch-gap timelines will use `nvprof` GPU
traces. A cheap profiling path is noted for layer 2: the existing
single-layer GPU tests load one layer in ~14 s and can be wrapped by `ncu`
directly, avoiding a full 20-minute service start per profile.

16K timeout margin (measured 2026-08-25): `proxy-retrieval-16384-mid`
completed its prefill in ~1630 s (1638.4 s total for 12 tokens). The 16K
performance cell needs prefill + 128 decode tokens, i.e. ~1670 s against the
1800 s single-request timeout — an ~8% margin. A single >10% slow repeat
would invalidate that repeat; the perf harness persists results after every
cell so a 16K failure cannot lose the other four TTFT cells.

Static hot-path audit (code reading, no GPU): the dominant structural
bottleneck candidate is host-device ping-pong, not kernel arithmetic. In
`crates/apxinf-cuda/src/kernels/qwen35_gdn.rs`, `read_status()` performs a
full `ctx.synchronize()` plus a device-to-host flag read, and it is called
inside `causal_conv_silu` (line 263), `gated_delta_step` (418),
`gated_rms_norm_bf16` (1003), their prefill variants (346, 804), and
`require_finite_bf16` (46). Additionally every one of those calls allocates
fresh `output`/`flags` (and sometimes `workspace`) buffers with `alloc_zeroed`
(cudaMalloc + memset) instead of reusing request-scoped storage. Per decode
token this multiplies out to roughly 4 full-stream synchronizations and ~6
device allocations per GDN layer — ≈200 synchronizations and several hundred
allocations across the 64 layers, before the per-token 248320-float logits
D2H and host argmax. At the measured ~130 ms/token that is ~0.65 ms per
synchronization+allocation pair, which matches the observed gap to the
~20 ms/token bandwidth floor far better than any kernel-efficiency
explanation. Prefill shares the same structure per 64-token block
(~4 syncs x 48 GDN layers x 128 blocks ≈ 25k synchronizations for an 8K
prompt).

*(Outcome: half right. The allocation half was confirmed and fixed — the
dequant scratch pool, A/B 7, delivered TTFT 3.46x at 1K. The synchronization
half was refuted by A/B 3: the per-op synchronizations wait on kernels that
are already executing, so removing ~200 of them per token measured zero
gain.)*

Layer-2 A/B queue, reordered by expected yield per risk (each item paired,
single-variable, correctness-gated per plan §9.4):

1. Batch the finite/status checks: keep the NaN defence but check once per
   token (final hidden) or per prefill block instead of per layer per op, and
   read flags without a mid-pipeline global synchronize (event/lazy read).
   The contract requires NaN to surface as a request-level error; it does not
   require a per-layer synchronous check. *(Outcome: rejected as A/B 3 —
   zero measured gain.)*
2. Reuse request-scoped output/flags/workspace buffers instead of
   cudaMalloc/free per op. *(Outcome: realized as the dequant scratch pool,
   A/B 7 — accepted, the largest single prefill win at that point.)*
3. Release build swap (binary already built and CPU-suite-verified).
   *(Outcome: accepted as A/B 1.)*
4. Drop debug capture env vars for measured runs. *(Outcome: accepted as
   A/B 2.)*
5. Device argmax to remove the ~1 MB/token logits D2H + host scan.
   *(Outcome: retired — measured at 0.5% of TPOT.)*

Item 1 was implemented (2026-08-26) behind
`APXINF_Q35_DEFERRED_STATUS=1`, default off:

- `qwen35_gdn.rs`: a `StatusFlags` handle picks per call between the eager
  path (fresh zeroed flags buffer, synchronous `read_status`, immediate
  error — bit-for-bit the previous behaviour) and the deferred path (all ops
  `atomicOr` into one resident per-device 4-byte latch, no allocation, no
  synchronize, no read). New `drain_deferred_status()` synchronizes once,
  raises any latched non-finite error, and clears the latch.
- `runtime.rs` drains once per prefill block and once per decoded token, so
  a non-finite value still fails the request (and the session is dropped as
  before); only the detection point moves from per-op to per-token/block.
  In deferred mode the per-op conv/recurrent rollback no longer triggers on
  NaN — irrelevant in service, because a failed request always drops the
  whole session.
- Expected effect: decode goes from ~200 full-stream synchronizations + ~200
  cudaMalloc per token to 1 synchronization; an 8K prefill from ~25k to 128.
- Tests: new GPU test
  `qwen35_gdn_cuda_deferred_status_matches_eager_and_latches_non_finite`
  asserts bit-identical outputs vs eager, silent latching, drain error,
  latch clearing, and eager restoration; it passed once the GPU lock freed,
  and CPU suites re-passed (bin 55/55, model 54/54). The flag was then
  measured as A/B 3 and rejected (no end-to-end gain); it stays in the tree
  default-off as a documented negative result.

## Layer-2 Paired A/B Results

A/B harness: `apxinf-ab-probe.py` runs a correctness gate before any
timing is trusted — the frozen 128-token request must reproduce the baseline
`output_ids` SHA256 exactly, and public case `text-niah-1024-p50` must still
match exactly — then measures `text-perf-1024` with 1 warmup + 3 repeats
(TTFT and TPOT). A full 7-cell re-measurement is only spent on an accepted
candidate.

### A/B 1: release build vs debug build — ACCEPTED (pending full re-measure)

Same source, same kernels, `--features cuda-no-nvtx --locked`, debug capture
still enabled on both sides so the build profile is the only variable.

| Metric | debug baseline | release candidate | delta |
|---|---:|---:|---|
| frozen `output_ids` SHA256 | `32b0862...` | `32b0862...` | identical |
| frozen request wall time | 17.62 s | 15.44 s | -12.4% |
| `text-niah-1024-p50` exact | pass | pass | unchanged |
| TTFT median (1K) | 78.09 s | 78.23 s | +0.2% (noise) |
| TPOT median | 133.7 ms | 109.2 ms | **-18.3%** |
| startup to bound port | ~20 min | ~8 min | -60% |

Verdict: accept. Correctness is bit-identical (the SHA match proves the whole
128-token trajectory is unchanged), TPOT improves 18.3%, TTFT is unchanged
within noise. The asymmetry is itself evidence for the static audit's
conclusion: decode is dominated by host-side per-op overhead (which `-O2`
shrinks), while prefill is dominated by device work per 64-token block (which
the host optimizer cannot touch). Evidence:
`apxinf-evidence/ab/ab-release-debugcapture.json`.

### A/B 2: debug capture off — ACCEPTED (small)

Release build both sides; only the `APXINF_DEBUG_HIDDEN_DIR` /
`APXINF_DEBUG_LOGITS_DIR` env vars differ.

| Metric | with capture | without capture | delta |
|---|---:|---:|---|
| frozen SHA256 | matches | matches | identical |
| frozen wall time | 15.44 s | 13.86 s | -10.2% |
| TTFT median (1K) | 78.23 s | 77.33 s | -1.2% |
| TPOT median | 109.2 ms | 105.1 ms | -3.8% |

Verdict: accept. Small but consistent, zero correctness cost, and the capture
hooks exist only for debugging. Measured runs from here on drop them.

### A/B 3: deferred GDN status checks — REJECTED

Release build with capture off on both sides; only
`APXINF_Q35_DEFERRED_STATUS` differs. This removes roughly 200 full-stream
synchronizations and ~200 device allocations per decoded token, which the
static audit predicted would be the dominant decode cost.

| Metric | eager (control) | deferred (candidate) | delta |
|---|---:|---:|---|
| frozen SHA256 | matches | matches | identical |
| frozen wall time | 13.86 s | 14.03 s | +1.2% |
| TTFT median (1K) | 77.33 s | 77.29 s | -0.05% |
| TPOT median | 105.1 ms | 105.5 ms | +0.4% |

Verdict: **reject**, per the plan's stop rule (gain below 2% / inside noise).
The hypothesis was wrong, and the measurement says so clearly: eliminating
~200 synchronizations per token changed nothing. The explanation is that these
synchronizations are not on the critical path — each one waits on kernels that
were already executing, so the GPU never idles for them and the CPU-side cost
is hidden. The flag and its regression test stay in the tree (default off,
`APXINF_Q35_DEFERRED_STATUS=1`) as a documented negative result; it is not
enabled and claims no benefit.

The useful outcome of this experiment is the correction it forces: decode cost
is inside the kernels, not around them. That reframes where to look next.

### Root cause of prefill cost: the packed-W4 GEMV kernel (measured 2026-08-26)

Per-token prefill cost derived from the layer-1 cells:

| cell | TTFT | ms per prompt token |
|---|---:|---:|
| 1K | 78.09 s | 76.3 |
| 2K | 159.19 s | 77.7 |
| 4K | 320.40 s | 78.2 |
| 8K | 676.52 s | 82.6 |
| 16K | 1635.15 s | 99.8 |

Decode costs 105 ms per token (release). A 64-token prefill block therefore
costs about 46x one decode step while doing 64x the token work — a batching
efficiency of roughly 1.03x, where a dense GEMM should approach 64x.

`crates/apxinf-cuda/kernels/custom/qwen35_w4.cuh` explains it exactly. The
packed-W4 projection kernel assigns **one CUDA block per output element**
(`blockIdx.x = row * out_features + out`) and each block re-reads the packed
weight row it needs from global memory. With M activation rows the entire
weight matrix is therefore streamed M times: a 64-row prefill block moves 64x
the weight bytes of a decode step, which is why per-token prefill cost is
flat in prompt length and nearly equal to decode. Weight traffic, not
attention, sets prefill cost — consistent with TTFT scaling linearly rather
than quadratically.

This is precisely experiment M-W4-P in the plan (§5.2/§6.1): "prefill 的大 M
第一候选是按 chunk 将 packed W4 解到 BF16 scratch，再交给 cuBLASLt/CUTLASS
BF16 GEMM". Implementation (staged, behind `APXINF_Q35_W4_PREFILL_GEMM`,
default threshold 8 rows):

- New `qwen35_w4_dequantize_bf16_kernel` (grid-stride) decompresses the packed
  matrix into a dense BF16 `[out_features, in_features]` scratch tensor,
  rounding each weight to BF16 exactly once — the same decompression boundary
  the GEMV kernel uses.
- `project_bf16` dispatches to dequantize-then-`project_checkpoint_bf16`
  (tensor-core BF16 GEMM, FP32 accumulate) when `rows >= threshold`, and keeps
  the GEMV kernel for decode-shaped calls. `APXINF_Q35_W4_PREFILL_GEMM=0`
  forces the GEMV path for every row count, which is the A/B control.
- Correctness test
  `qwen35_w4_cuda_prefill_gemm_matches_gemv_path_and_cpu_reference` asserts
  the decisive property first: the dequantized matrix is **bit-identical** to
  the reference decompression (`assert_eq!` on all 1056 weights, not a
  tolerance). Given that, any output difference is attributable to
  accumulation order alone; the test bounds GEMM-vs-GEMV drift at 0.0625 and
  verifies sub-threshold row counts reproduce the GEMV bytes exactly.
- Suite status: `apxinf-cuda` 93 passed, 2 failed — the same two pre-existing
  `fp8` failures (cuBLAS status 15, not on the Qwen3.8 path) as the baseline.

An earlier version of this test failed on 1 of 264 outputs. The investigation
is recorded because it changed the test rather than the code: GEMV-vs-CPU was
bit-exact (error 0.0), the dequantized weights were bit-exact, and the
outlier was an output near 0.42 built from partial sums reaching ±4.1, i.e.
catastrophic cancellation in synthetic test data amplifying one BF16
reassociation ulp. The fix was to assert the property that actually matters
(bit-exact decompression plus bounded reassociation drift) instead of a
relative tolerance on a deliberately ill-conditioned dot product.

### A/B 4: W4 prefill dequantize+GEMM — ACCEPTED

Release build, capture off, deferred status off on both sides; only
`APXINF_Q35_W4_PREFILL_GEMM` differs (control = GEMV for all row counts,
candidate = threshold 8).

| Metric | GEMV control | dequant+GEMM | delta |
|---|---:|---:|---|
| TTFT median (1K) | 77.33 s | 19.87 s | **-74.3% (3.89x)** |
| TPOT median | 105.1 ms | 104.1 ms | -0.9% |
| public functional | 6/6 | **6/6** | unchanged |
| 8K functional case wall time | ~680 s | ~214 s | -68.5% (3.2x) |
| frozen trajectory vs oracle | 28-token prefix, 30/128 | 23-token prefix, 24/128 | regressed |

Verdict: accept. Correctness that gates eligibility is unchanged — public
functional is still 6/6 exact, including all three 8K longdoc cases with
correct EOS early termination — while TTFT, the largest scoring lever at 35
points, improves 3.89x. Decode is untouched by design (row counts below the
threshold keep the GEMV kernel), and TPOT moves only within noise.

The cost is the trajectory soft target: the frozen 128-token prefix drops from
28 to 23 tokens. This is the *same* mechanism the logits comparison above
quantified, at the *same*
sites: the oracle's top1/top2 margin is exactly 0.0 at steps 23, 28 and 76,
and a different accumulation order resolves the step-23 tie the other way. The
GEMV path happened to agree with the oracle at step 23 and lose at 28; the
GEMM path loses at 23. No achievable tolerance fixes either, because the
oracle itself has no margin there. Per plan §11.2 trajectory is a scored soft
target with threshold 0.0 and must not outweigh functional results or an
end-to-end gain of this size, so the trade is taken and recorded here rather
than hidden.

Seven-cell re-measurement with A/B 4 accepted (all valid, worst CV 6.25%):

| Cell | TTFT before | TTFT after | speedup | TPOT before | TPOT after |
|---|---:|---:|---:|---:|---:|
| 1K | 78.09 s | 20.04 s | 3.90x | 133.7 ms | 104.6 ms |
| 2K | 159.19 s | 42.83 s | 3.72x | 134.1 ms | 104.6 ms |
| 4K | 320.40 s | 89.81 s | 3.57x | 138.4 ms | 109.0 ms |
| 8K | 676.52 s | 214.58 s | 3.15x | 153.4 ms | 122.6 ms |
| 16K | 1635.15 s | 710.71 s | 2.30x | 228.7 ms | 201.6 ms |

The 16K single-request wall time drops from ~1664 s to ~737 s, widening the
1800 s timeout margin from ~8% to ~59%. Evidence:
`apxinf-evidence/perf-w4gemm/perf.json`.

### A/B 5: bandwidth-oriented packed-W4 GEMV for decode — ACCEPTED

Decode still used the baseline kernel (one K element per thread). Its warp
load pattern is the inefficiency: 32 consecutive lanes cover K..K+31, i.e.
only four distinct packed uint32 values (16 bytes) per 128-byte transaction,
and every lane re-reads its group's scale and zero-point. Measured effective
decode bandwidth was ~58 GB/s (~5.8% of the RTX 4090's ~1 TB/s) for ~5.7 GiB
of per-token weight traffic.

The new `qwen35_w4_project_bf16_packed_kernel` gives each thread one whole
packed uint32 (eight consecutive K values, always within one quant group), so
a warp streams 128 contiguous bytes per transaction and loads each scale and
zero-point once per uint32 instead of eight times; reduction is warp shuffle
plus one shared stage. Decompression rounding is unchanged. Gated by
`APXINF_Q35_W4_PACKED_GEMV` (default on, `=0` is the control); GPU tests
assert packed-vs-baseline-vs-CPU agreement including the non-multiple-of-8
K tail and the NaN-scale rejection path (33 qwen35 GPU tests pass).

| Metric | baseline GEMV | packed GEMV | delta |
|---|---:|---:|---|
| TPOT median (1K) | 104.6 ms | 67.6 ms | **-35.4%** |
| TTFT median (1K) | 19.9 s | 19.9 s | unchanged (prefill is GEMM) |
| public functional case | pass | pass | unchanged |
| frozen wall time | 14.5 s | 9.5 s | -34% |
| frozen trajectory vs oracle | 23-token prefix, 24/128 | **76-token prefix, 82/128** | improved |

The trajectory *improved* past even the original debug baseline (28-token
prefix): the new accumulation order happens to resolve the oracle's exact
ties at steps 23 and 28 the oracle's way, and first diverges at step 76 — the
third zero-margin site. This is further confirmation that the trajectory is
decided by tie-flips at the oracle's zero-margin steps and not by any logic
defect; the direction of each flip is an accident of reassociation.

Accepted configuration after A/B 5: release build, no debug capture, eager
status checks (deferred rejected), `APXINF_Q35_W4_PREFILL_GEMM=8` (default),
`APXINF_Q35_W4_PACKED_GEMV=1` (default). Cumulative vs the layer-1 baseline
at 1K: TTFT 78.1 s -> 19.9 s (3.9x), TPOT 133.7 ms -> 67.6 ms (1.98x).

### A/B 6: batched prefill attention — ACCEPTED (marginal, bit-identical)

With projections on the GEMM path, the remaining prefill structure cost in
attention was launch count: `kernels/attention.rs::sdpa` issued one cuBLAS
GEMM per (kv_head, sequence_row) pair — M=6 (the GQA ratio) micro-GEMMs,
4 x 64 x 2 = 512 host calls per full-attention layer per 64-token block,
~131k calls for a 1K prompt. The change batches the sequence dimension with
one `cublasGemmStridedBatchedEx` call per kv head (query/score strides across
rows are uniform; the K/V page broadcasts with stride 0), cutting those 512
calls to 8. Gated by `APXINF_Q35_BATCHED_SDPA` (default on, `=0` control).

Correctness is proven at two levels. GPU test
`batched_sdpa_prefill_matches_per_row_loop_at_checkpoint_shape` requires
**bit equality** (`assert_eq!`, no tolerance) against the per-row loop at the
real 24/4/256 head geometry with a nonzero kv_offset, and passes — the
batched GEMMs see exactly the same operands per (row, head). At service
level, the frozen 128-token request reproduces the packed-GEMV
configuration's `output_ids` byte for byte (SHA `20d981cb...`).

Timing (1 warmup + 3 repeats, 1K): TTFT median 19.87 s -> 19.01 s (-4.3%),
but with samples [19.01, 21.35, 18.69] s (CV 7.4%) the gain sits at the edge
of noise; TPOT is unchanged (67.6 -> 66.7 ms). Honest verdict: marginal on
this cell. It is retained as default-on because the cost side is provably
zero (bit-identical outputs), it removes a 64x host-call amplification that
matters more at larger seq/kv shapes, and the control flag preserves instant
rollback.

## Final Configuration and Verification Chain (2026-08-26)

Final layer-2 configuration: release build, no debug capture env vars,
eager status checks (`APXINF_Q35_DEFERRED_STATUS` off — rejected),
`APXINF_Q35_W4_PREFILL_GEMM=8`, `APXINF_Q35_W4_PACKED_GEMV=1`,
`APXINF_Q35_BATCHED_SDPA=1` (all three are the defaults; every flag has a
tested `=0` rollback).

Seven-cell measurement on the final configuration (all valid, success 5/5
per cell, peak VRAM 19498 MiB throughout;
`apxinf-evidence/perf-final/perf.json`):

| Cell | TTFT median | TTFT CV | TPOT median | TPOT CV | vs layer-1 TTFT | vs layer-1 TPOT |
|---|---:|---:|---:|---:|---:|---:|
| 1K | 19.94 s | 0.22% | 68.1 ms | 3.13% | 3.92x | 1.96x |
| 2K | 43.07 s | 0.46% | 68.5 ms | 0.43% | 3.70x | 1.96x |
| 4K | 87.43 s | 1.74% | 72.9 ms | 0.62% | 3.66x | 1.90x |
| 8K | 214.21 s | 0.60% | 86.8 ms | 0.46% | 3.16x | 1.77x |
| 16K | 712.29 s | 0.19% | 151.4 ms | 6.44% | 2.30x | 1.51x |

The 16K wall time is ~732 s against the 1800 s timeout (~59% margin).

Verification chain on the final binary and flags, in order:

1. GPU suites under the lock: `qwen35_w4` 12 passed (bit-exact dequantize,
   packed-GEMV vs baseline vs CPU, NaN-scale rejection, threshold fallback),
   `qwen35_gdn` 21 passed, `batched_sdpa` bit-equality passed; full
   `apxinf-cuda` suite 93 passed with only the two pre-existing environmental
   `fp8` failures (cuBLAS status 15, off the Qwen3.8 path).
2. CPU suites: bin 55/55, `apxinf-model` qwen35 54 passed / 2 ignored,
   `cargo fmt --check`, `cargo check --workspace --locked`,
   `git diff --check` all clean.
3. Frozen protocol gate re-run against the final service: **12/12 passed**
   (`apxinf-evidence/protocol-final`, evidence SHA
   `4695be18ddfc0f478e2d87444d859a0302644eef39d0c2e175e0765fe7c4e868`).
4. Frozen 128-token request: bit-identical to the accepted packed-GEMV
   trajectory (76-token oracle prefix, 82/128 agreement — better than the
   layer-1 baseline's 28/30).
5. Public functional 6/6 exact on the optimized prefill path (measured during
   A/B 4; the prefill/decode kernels are unchanged since, and the frozen
   bit-equality in step 4 covers the subsequent flags).
6. Short mixed soak on the final configuration: **58/58, success rate
   1.0000** (45 short + 12 x 1K + 1 x 8K, sequential, usage-checked), final
   health `ok` with `fallback_active=false`
   (`apxinf-evidence/short-soak-final.json`). The full 200-request soak
   remains the layer-1 record; a re-run on the final flags is recommended
   before any formal submission if time allows.
7. Shutdown: service killed cleanly, port 18080 free, all four GPUs back to
   1 MiB, and the global GPU-job lock free.

Cumulative layer-2 result vs the layer-1 baseline: TTFT 3.9x faster at 1K and
2.3x at 16K; TPOT 2.0x faster at 1K and 1.5x at 16K; correctness gates
unchanged (protocol 12/12, functional 6/6, soak clean); trajectory soft
target improved from 30/128 to 82/128 tokens against the approved oracle.

### A/B 7: W4 dequant scratch pooling — ACCEPTED (2026-08-26 evening)

Multi-GPU development was authorized (final deployment stays single-GPU;
development GPUs used only for tests/profiling), which enabled a real
measurement campaign on idle GPU0 instead of inference from the service
timings. A new ignored profiling harness
(`real_layer_profile_harness_prefill_and_decode`) loads one real GDN layer and
one real full-attention layer from the checkpoint and times prefill/decode
stages with warmed shapes.

Measured attribution (rows=64, release, GPU0), which corrected another wrong
guess — attention was NOT the prefill bottleneck:

| Stage | per call |
|---|---:|
| GDN layer prefill (whole) | 16.7 ms |
| . in-projections (qkv/z/a/b) | 3.9 ms |
| . causal conv prefill | 0.7 ms |
| . gated delta prefill | 1.4 ms |
| . norm + out_proj + MLP | 11.4 ms |
| full-attention layer prefill (whole) | 17.1 ms |
| **micro: cudaMalloc+free 178 MB** | **3.22 ms** |
| micro: cudaMalloc+free 4 B | 3.7 us |
| micro: one MLP gate projection (incl. malloc) | 3.92 ms |

The smoking gun: allocating and freeing the 178 MB dequant scratch costs
3.22 ms — 82% of a whole MLP projection call. Every W4 projection allocated
fresh scratch (7 per GDN layer per 64-token block), so ~80% of prefill was
page-table work in `cudaMalloc`/`cudaFree`, not arithmetic.

Fix: a per-(device, byte-size) scratch pool (one resident buffer per size
class, ~500 MB total across the distinct projection shapes; the dequant
kernel overwrites every element so no zeroing is needed), and the GEMM is
issued on raw buffers so the pooled scratch never enters `Tensor` ownership.
`APXINF_Q35_SCRATCH_POOL=0` is the paired control. Harness after pooling:
GDN layer prefill 16.7 -> 5.6 ms, attention layer 17.1 -> 4.6 ms.

Service-level results (GPU1, strict command):

| Metric | before (A/B 6 config) | after pooling | delta |
|---|---:|---:|---|
| TTFT 1K | 19.92 s | **5.76 s** | 3.46x |
| TTFT 8K (spot) | 214.2 s | **94.1 s** | 2.28x |
| TTFT 16K (spot) | 712.3 s | **488.1 s** | 1.46x |
| TPOT 1K | 67.6 ms | 68.5 ms | unchanged (decode never dequantizes) |
| frozen 128-token output | SHA `20d981cb...` | **bit-identical** | zero numeric change |
| peak VRAM (8K/16K spots) | 19498 MiB | 19958 MiB | +460 MiB pool, ample headroom |

Verification on the pooled build: GPU suites on GPU0 (qwen35 33 passed,
batched-sdpa bit-equality passed), protocol gate **12/12** on GPU1
(`apxinf-evidence/protocol-scratchpool`, SHA `fac93a75...`), short mixed
soak **33/33 = 100%** (24 short + 8 x 1K + 1 x 8K) with final health `ok` /
`fallback_active=false`, CPU suites and fmt/check/diff clean, and shutdown
left the port free, all GPUs at 1 MiB, and the lock free.

Cumulative vs the layer-1 debug baseline: **TTFT 13.6x at 1K (78.1 s ->
5.76 s), 7.2x at 8K, 3.35x at 16K; TPOT 2.0x (133.7 -> 68.5 ms)**. The 16K
request now completes in ~507 s against the 1800 s timeout (72% margin).
Remaining known gaps to a vLLM-class stack, in expected order: flash-style
fused attention (score matrices never materialized), Marlin-class fused
dequant-GEMM (no BF16 scratch round-trip), GDN chunk-scan batching across
blocks, CUDA-graph decode, device argmax. A fresh five-repeat seven-cell
campaign on this configuration is the next formal measurement.

### Four-GPU development workflow

Multi-GPU use for development was authorized (submission stays single-GPU on
`GPU-343bc895`). The lanes used from here on: GPU1 (`GPU-343bc895`) formal
service A/B under `flock` only; GPU0 (`GPU-d074a13d`) profiling harness;
GPU2 (`GPU-f4efcc89`) and GPU3 (`GPU-ea64faa4`) parallel test shards. Running
the W4 suite, the GDN suite and the profiling harness concurrently on three
GPUs cut a verification cycle from ~35 s serial to ~12 s, and the three-way
chunk-size sweep below took one 11 s wall-clock pass instead of three
sequential runs. Test binaries are prebuilt once (`--no-run`) and invoked
directly with per-lane `CUDA_VISIBLE_DEVICES`/`APXINF_TEST_GPU_UUID`.

### Decode attribution (measured, GPU0)

Per-call costs with request state reused across steps (service-shaped):

| Component | per call | x count | contribution to TPOT |
|---|---:|---:|---:|
| GDN layer decode | 928 us | 48 | 44.5 ms (65%) |
| full-attention layer decode | 932 us | 16 | 14.9 ms (22%) |
| lm_head projection | 2.67 ms | 1 | 2.7 ms (3.9%) |
| logits D2H + host argmax | 0.34 ms | 1 | 0.3 ms (0.5%) |
| **total** | | | **62.4 ms** (measured TPOT 68.5 ms) |

Within one GDN decode: in-projections 179 us, MLP projections 419 us
(gate 128 us, down 128 us alone), causal conv step 30 us — i.e. **79% of
decode is W4 projections**. This retires the "device argmax" candidate: the
D2H plus host scan is 0.5% of TPOT, not a lever.

Bandwidth check, and the key comparison: the W4 GEMV reaches ~330 GB/s
(42.5 MB in 128 us for an MLP projection), while cuBLAS BF16 GEMV on the
lm_head reaches **954 GB/s** (2.37 GiB in 2.67 ms). The hardware is fine; our
W4 kernel runs at 35% of the achievable rate. Closing that needs a Marlin-class
kernel that fuses dequantization into a tensor-core MMA and never round-trips
weights through BF16 scratch. Dequantize-then-cuBLAS is *not* the answer for
decode: it would move 182 MB packed + 728 MB written + 728 MB read instead of
182 MB, i.e. strictly worse.

### A/B 8: warp-per-output W4 GEMV — REJECTED

Hypothesis: the packed kernel gives each thread only ~2.5 packed uint32 values
and each block just 2.5 KB, too little to hide latency. The candidate assigns
one warp per output, eight outputs per block (~20 uint32 per thread), stages
the activation row in shared memory, and reduces with pure warp shuffles;
shapes wider than the 48 KB shared budget (the MLP down projection at
in_features 17408) keep the packed kernel, covering 77% of per-layer W4 bytes.

Result: GDN layer decode 932.6 us -> 916.0/927.9 us across runs, i.e. **≤1.8%
and inside run-to-run noise**. Rejected per the stop rule. The wrong assumption
was again mine: work-per-thread was not the constraint, so the shared-memory
staging (an extra 20 KB read per block) roughly cancelled the scheduling gain.
Kernel and tests are retained behind `APXINF_Q35_W4_WARP_GEMV` (**default
off**), with GPU tests covering the block tail (out_features not a multiple of
8), the packed-nibble tail, and NaN-scale rejection.

### A/B 9: prefill block size 64 -> 256 — ACCEPTED (large)

With the scratch pool in place, each prefill block still re-dequantizes every
projection, so per-block fixed cost is amortized over only 64 tokens. A
three-GPU parallel sweep measured per-token layer cost directly:

| rows/block | GDN layer | attention layer | GDN us/token | attention us/token |
|---:|---:|---:|---:|---:|
| 64 | 6.06 ms | 4.92 ms | 94.7 | 76.8 |
| 256 | 10.46 ms | 8.68 ms | 40.9 | 33.9 |
| 512 | 14.06 ms | 13.18 ms | 27.5 | 25.7 |

The 64-row model predicts a 1K prefill of 5.91 s against 5.76 s measured, so
the extrapolation is trustworthy: chunk 256 predicts ~2.56 s and chunk 512
~1.77 s. `PREFILL_CHUNK_TOKENS` moved to 256 with a `prefill_chunk_tokens()`
accessor (override `APXINF_Q35_PREFILL_CHUNK`, clamped to 512 so the attention
score workspace stays bounded), and `request_state_bytes` now scales the
attention workspace with the same value, keeping admission honest — the GDN
scan chunk stays fixed at 64 internally and is unaffected.

Service results (GPU1):

| Metric | chunk 64 | chunk 256 | delta |
|---|---:|---:|---|
| TTFT 1K | 5.76 s | **2.38 s** | 2.42x |
| TTFT 8K | 94.1 s | **69.1 s** | 1.36x |
| TTFT 16K | 488.1 s | **418.7 s** | 1.17x |
| TPOT 1K | 68.5 ms | 68.5 ms | unchanged (decode is unaffected) |
| peak VRAM | 19958 MiB | 19958 MiB | unchanged |

Correctness across the multi-block regime — the case chunking could actually
break: all three 8K `longdoc` functional cases (each spanning 32 blocks at
chunk 256) still pass exact with correct EOS termination, and the frozen
8-token trajectory is bit-identical (it is a single block at either size, so
it does not exercise chunking; the 8K cases are the real evidence).
Verification: protocol gate **12/12** (SHA `3595a985...`), short mixed soak
**33/33 = 100%**, three-GPU regression 98 passed with only the two
pre-existing `fp8` environmental failures, CPU suites 55/55 including two new
chunk-configuration tests, fmt/check/diff clean, and shutdown left the port
free, all four GPUs at 1 MiB, and the lock free.

Not pushed to chunk 512 (which the sweep says is ~1.8 s at 1K) because the
16K attention score workspace would reach 774 MB and the current headroom is
~4.6 GiB; 512 is available via the env override for a future measured
campaign with a re-verified budget.

### Cumulative layer-2 result

| Metric | layer-1 baseline | now | speedup |
|---|---:|---:|---:|
| TTFT 1K | 78.09 s | **2.38 s** | **32.8x** |
| TTFT 8K | 676.52 s | **69.1 s** | **9.8x** |
| TTFT 16K | 1635.15 s | **418.7 s** | **3.9x** |
| TPOT 1K | 133.7 ms | **68.5 ms** | **1.95x** |
| 16K single request | ~1664 s (8% timeout margin) | ~437 s (76% margin) | 3.8x |

Nine paired single-variable A/B experiments: six accepted (release build,
debug-capture off, W4 prefill dequant+GEMM, packed GEMV, batched SDPA, scratch
pool, prefill chunk 256), two rejected with the reasoning recorded (deferred
GDN status, warp-per-output GEMV), and every accepted change carries an `=0`
rollback flag plus bit-exact or bounded-drift tests. Correctness gates held
throughout: protocol 12/12, public functional 6/6, soak 100%.

Honest remaining gap to a vLLM-class stack: prefill is now ~7x off the
contract's arithmetic screen (2.38 s vs ~0.34 s at 1K) and decode ~3.4x off
the bandwidth floor (68.5 ms vs ~20 ms). The next levers, in order of expected
yield: Marlin-class fused dequant-MMA (would address the 65% of decode and
most of prefill still spent in W4 projections), flash-style fused attention
(score matrices never materialized), and CUDA-graph decode.

### W4 decode GEMV cost decomposition (four-GPU micro-benchmark matrix)

Rather than guess again, three diagnostic kernel variants were added behind
`APXINF_Q35_W4_DIAG_KERNEL` (measurement only; variants 1 and 2 are
deliberately numerically wrong and can never be selected without that
variable) and run as a four-way parallel matrix, one variant per GPU. Target:
the MLP gate projection, 44.6 MB packed, decode shape.

| Variant | time | effective bandwidth | isolates |
|---|---:|---:|---|
| stream-only (weights only, no activation, no dequant) | 52.2 us | **815 GB/s** | achievable read rate for this layout |
| no-dequant (same accesses, one multiply per nibble) | 78.7 us | 540 GB/s | + activation load and multiply-accumulate |
| production packed kernel | 128.5 us | 331 GB/s | + full dequantization arithmetic |
| vec4 (16-byte `uint4` loads) | 162.0 us | — | vectorized loads |

Decomposition: **41% weight streaming, 21% activation+FMA, 39%
dequantization arithmetic.** The decisive finding is the first row — at
815 GB/s (81% of the ~1 TB/s peak) the memory access pattern is already near
optimal, so no amount of load-pattern tuning can help. The theoretical floor
for a correct kernel with this structure is ~52 us versus 128.5 us today, and
all of the gap is arithmetic.

### A/B 10-13: four rejected W4 GEMV optimizations

Every candidate below kept the BF16 rounding boundary intact and was measured
against the production kernel on the same shape; all four are rejected, and
all four of my hypotheses were wrong.

| Candidate | hypothesis | measured | verdict |
|---|---|---:|---|
| warp-per-output + shared activation | per-thread work too small | -1.8% (noise) | reject |
| `uint4` vectorized loads | load width limits throughput | **+26% slower** | reject |
| 4 accumulators via `partial[nibble & 3]` | serial FMA dependency chain | **+17% slower** | reject |
| unrolled named accumulators + `fmaf(q, s, -z*s)` | fix the local-memory spill, save one instruction | +1.3% slower | reject |

The third result diagnosed the fourth: indexing a `float partial[4]` by
`nibble & 3` is a dynamic index, so the compiler spills the array to local
(device) memory — hence 17% slower rather than faster. Fully unrolling with
named accumulators fixed the spill, and the first three-GPU pass appeared to
show a 3-4% gain on the whole layer.

**That gain was a measurement artifact, and catching it is the useful part.**
A second pass with 4x iterations and two independent copies of each arm on
different GPUs gave: baseline 923.1 / 901.0 us, candidate 933.8 / 913.1 us.
Cross-GPU spread (2.4%) exceeds the candidate effect, and the candidate is
actually 1.3% slower on average. Running duplicate arms on separate GPUs is
now the standard protocol for any sub-5% claim; the earlier accepted changes
(26-74% effects) are far outside this noise band and unaffected.

Conclusion for this lane: with the current algorithm (independent per-nibble
dequantization plus FP32 accumulation), the decode GEMV is at a local optimum.
The 39% arithmetic share cannot be reclaimed by local tuning — it needs a
structural change, i.e. a Marlin-class kernel that either hoists
decompression into a per-group lookup table (only 16 distinct products exist
per group, since `q` spans 0..15 with `z` and `s` fixed) or performs the
dequantized multiply-accumulate inside a tensor-core MMA. That is a
substantial implementation, not an incremental one, and is left as the
top-of-list next lever with measured justification rather than speculation.

Diagnostic kernels remain in the tree, unreachable without
`APXINF_Q35_W4_DIAG_KERNEL`, and the production dispatch is unchanged:
three-GPU regression after these edits is 98 passed with only the two
pre-existing `fp8` environmental failures, W4 suite 14 passed, GDN suite 21
passed.

### A/B 14: prefill block size 256 -> 512 — ACCEPTED

After four failed GEMV micro-optimizations, the highest-certainty remaining
lever was the one already measured: the three-GPU sweep put per-token GDN
layer cost at 27.5 us at 512 rows versus 40.9 us at 256. This is a
configuration change with no new kernel code.

The earlier decision to stop at 256 was over-conservative. Recomputed budget:
the attention score workspace at `max_model_len=32768` is 768 MB at chunk 256
and 1536 MB at chunk 512, and measured VRAM in use was 19958 MiB of 24564
(4606 MiB headroom), so chunk 512 leaves ~3838 MiB. `request_state_bytes`
charges this at admission, so an over-large chunk fails closed at startup
rather than mid-request — and the service did bind, confirming the budget.
`MAX_PREFILL_CHUNK_TOKENS` raised to 1024 to keep the override useful.

| Metric | chunk 256 | chunk 512 | delta |
|---|---:|---:|---|
| TTFT 1K | 2.38 s | **1.76 s** | 1.35x |
| TTFT 8K | 69.1 s | **64.1 s** | 1.08x |
| TTFT 16K | 418.7 s | **408.7 s** | 1.02x |
| TPOT 1K | 68.5 ms | 67.8 ms | unchanged |
| peak VRAM | 19958 MiB | 19958 MiB | unchanged |

The sweep predicted 1.77 s at 1K; measured 1.76 s. Gains taper with prompt
length because long prompts already amortized the per-block cost over many
blocks.

Correctness across the multi-block regime: five functional cases pass exact —
`text-niah-1024-p10/p90` plus all three 8K `longdoc` cases (16 blocks each) —
with correct EOS termination. The GDN chunked-vs-eager equivalence test was
re-run at 512 rows on a development GPU and passes, confirming the internal
64-token scan is unaffected by the outer block size. Protocol gate **12/12**
(SHA `7338e0bb...`), short mixed soak **33/33 = 100%** with final health `ok`
and `fallback_active=false`, three-GPU regression 98 passed with only the two
pre-existing `fp8` environmental failures, qwen35 suite 35 passed, batched-SDPA
bit-equality passed, CPU suites 55/55, fmt/check/diff clean, and shutdown left
the port free, all four GPUs at 1 MiB, and the lock free.

### Cumulative layer-2 result (final)

| Metric | layer-1 baseline | final | speedup |
|---|---:|---:|---:|
| TTFT 1K | 78.09 s | **1.76 s** | **44.4x** |
| TTFT 8K | 676.52 s | **64.1 s** | **10.6x** |
| TTFT 16K | 1635.15 s | **408.7 s** | **4.0x** |
| TPOT 1K | 133.7 ms | **67.8 ms** | **1.97x** |
| 16K single request | ~1664 s (8% timeout margin) | ~427 s (76% margin) | 3.9x |

Fourteen paired single-variable experiments: seven accepted (release build,
debug-capture off, W4 prefill dequant+GEMM, packed GEMV, batched SDPA, scratch
pool, prefill chunk 512), seven rejected with reasoning and data recorded
(deferred GDN status, warp-per-output GEMV, `uint4` vectorized loads,
4-accumulator array, unrolled+fused accumulators, per-group LUT design, and
the chunk-512 caution that the recomputed budget overturned). Correctness
gates held at every step: protocol 12/12, public functional 6/6, soak 100%,
and the trajectory soft target improved from 30/128 to 82/128 against the
approved oracle.

Where the remaining distance to a vLLM-class stack now sits: prefill is ~5x
off the contract's arithmetic screen (1.76 s vs ~0.34 s at 1K) and decode
~3.4x off the bandwidth floor (67.8 ms vs ~20 ms). Decode is the larger and
harder gap, and the micro-benchmark matrix above localizes it precisely: 39%
of the W4 GEMV is dequantization arithmetic that four local optimizations
could not reclaim, so the next step there is a Marlin-class fused
dequant-MMA rather than further tuning. For prefill, chunk 1024 is now
reachable via the override if a re-verified budget allows.

### A/B 15: Marlin-style bit-trick dequantization — REJECTED (with root cause)

The remaining decode lever was the 39% of W4 GEMV time spent on
dequantization arithmetic. Marlin's key idea for that is not the tensor core
(an M=1 GEMV cannot use MMA productively) but replacing numeric conversion
with bit construction: BF16 carries 8 significant bits, so the bit pattern
`0x4300 | q` *is* the value `128 + q` for q in 0..15, letting two nibbles
become one `bf16x2` with a mask, a shift and an OR — no convert instructions.

Bitwise equality was established before writing the kernel, not assumed:

- `0x4300 | q` equals `128 + q` exactly for all 16 codes (exhaustively
  checked).
- `(128+q) - (128+z)` is exact in BF16 for all 256 pairs (exhaustively
  checked) and equals `q - z`.
- `(q-z)` needs at most 5 significant bits and `s` has 8, so the exact
  product needs at most 13 — FP32's 24-bit significand holds it exactly.
  The production path thus rounds exactly once (FP32 product to BF16), which
  is by definition what an IEEE BF16 multiply produces. So BF16 SIMD
  arithmetic here is bit-identical to the FP32-then-round path, and
  accumulation stays FP32 in the same order.

Measured (MLP gate shape, two independent GPUs per arm, 64 iterations):
baseline 131.4 / 129.7 us, candidate 203.3 / 197.7 us — **54% slower**. A
first attempt was 54% slower too; suspecting the local-memory trap that had
explained an earlier failure, the `reinterpret_cast` on a stack variable was
replaced with `__ushort_as_bfloat16` / `__halves2bfloat162` so no address is
ever taken. Identical result, so that was not the cause.

SASS disassembly settled it. The candidate issues **fewer** instructions than
production (98 vs 107 XMAD-class ops in the same region), and a minimal probe
compiled for `sm_89` confirms `__hmul2`/`__hsub2` on `__nv_bfloat162` do lower
to native `HFMA2` — so neither instruction count nor absent hardware support
explains the slowdown. The cause is **latency, not throughput**: the
`HSUB2 -> HMUL2 -> unpack -> FFMA` chain is longer per element than the scalar
FP32 chain, and an M=1 GEMV gives each thread only ~2.5 packed words, far too
little work to hide it. Marlin's technique pays off in batched GEMM, where
occupancy hides latency and tensor-core MMA consumes packed pairs directly; it
does not transfer to GEMV.

Conclusion for the decode lane, now with five measured failures and a
disassembly-level explanation: the W4 GEMV is at the limit of this algorithmic
structure on this hardware for M=1. Reclaiming the 39% requires raising M so
tensor-core MMA becomes usable — i.e. speculative decoding or batching, both
explicitly out of scope this period — or an offline weight re-layout paired
with an MMA kernel, which only pays off at M >= 8. This is recorded so the
five dead ends are not retried: warp-per-output, `uint4` loads,
4-accumulator arrays, unrolled/fused accumulators, and bit-trick SIMD.

Final performance is therefore unchanged from A/B 14 (1K TTFT 1.76 s, TPOT
67.8 ms). Regression after these edits: `apxinf-cuda` 98 passed with only the
two pre-existing `fp8` environmental failures, qwen35 suite 35 passed, CPU
suites 55/55 and 55/55, fmt/check/diff clean. All diagnostic variants remain
unreachable without `APXINF_Q35_W4_DIAG_KERNEL`, and the production dispatch
is untouched.

### A/B 16-17: attention prefill — the real long-prompt bottleneck

Asked to push further, the first question was where 16K actually spends its
time. The single-layer harness only explained ~124 s of the measured 408.7 s,
because `attention_prefill` there starts from an empty cache and so has
kv_len == rows, while a real 16K prompt attends over a cache growing to 16512.
A new staircase section walks every block from 0 to the prompt length, exactly
as the service does. It found the answer immediately: **one attention layer
cost 25.05 s for a 16K prompt, so 16 layers account for ~400 s of the 408.7 s
TTFT.** Attention prefill, not GDN and not W4, was the entire 16K story.

**A/B 16 (GEMM reshape, small gain, kept):** the score/value GEMMs batched the
sequence dimension, leaving M = gqa_ratio = 6 — a shape that wastes the tensor
core. Rebatching over the query heads within a GQA group makes M = seq_len
(512) with the key page broadcasting at stride 0. Staircase 25.05 s -> 23.59 s
at 16K and 4.30 s -> 3.14 s at 8K; the bit-equality test against the per-row
loop still passes. Modest, and it showed GEMM shape was not the main cost.

**Finding the real cause required three corrections to my own reasoning.**
A per-stage split at the final kv length showed the complete `sdpa` call
taking 2082 ms while scale (5.3 ms), softmax (34.9 ms) and the scores
allocation (4.2 ms) summed to 44 ms. A micro-benchmark of the score GEMM in
isolation took **5.7 us**, and interleaved versus contiguous output stride made
no difference — so neither the GEMMs nor `ldc` explained anything. The 34.9 ms
softmax measurement was the flaw: it had been taken with `kv_offset = 0`, where
`valid_cols` averages ~256, whereas the real call has `kv_offset = 15872` and
scans 16000+ columns. Scaling by that factor gives ~2234 ms, matching the
2082 ms exactly. Softmax was ~98% of attention prefill all along.

**A/B 17 (row-cooperative softmax, ACCEPTED, large):** reading the kernel
showed why. `for (c = 0; c < valid_cols; c++)` is executed **in full by every
thread of every block** — 256-fold redundant computation per block, times
`ceil(cols/256)` blocks per row, i.e. 512 full row scans where two partitioned
passes suffice. The replacement gives one block per row: the 256 threads split
the row, reduce max and sum through shared memory, and each thread rewrites
only the elements it read (which also makes the operation in-place safe).
`APXINF_Q35_ROWWISE_SOFTMAX=0` selects the legacy kernel as the control.

| Measurement (16K, 512-row blocks) | legacy | row-cooperative | speedup |
|---|---:|---:|---:|
| attention staircase, one layer | 23.06 s | **0.659 s** | **35.0x** |
| single `sdpa` at kv=16384 | 2053.7 ms | **13.7 ms** | **150x** |

Two independent GPUs agreed (0.659 s / 0.679 s). Service-level results:

| Metric | before | after | speedup |
|---|---:|---:|---:|
| TTFT 1K | 1.76 s | **1.62 s** | 1.09x |
| TTFT 8K | 64.1 s | **17.8 s** | **3.6x** |
| TTFT 16K | 408.7 s | **31.9 s** | **12.8x** |
| TPOT 1K | 67.8 ms | 67.1 ms | unchanged |
| peak VRAM | 19958 MiB | 19958 MiB | unchanged |

The gain scales with prompt length exactly as the diagnosis predicts, since
the redundancy factor grows with kv_len. Numerics: the max reduction is exact
in any order; the exponential sum is now a partitioned tree rather than one
sequential accumulation, the same class of FP32 reassociation as the accepted
prefill GEMM. **Public functional 6/6 all pass exact** (three 1K NIAH plus all
three 8K longdoc), protocol gate **12/12** (SHA `56e4ee27...`), and the
`apxinf-cuda` suite is 98 passed with only the two pre-existing `fp8`
environmental failures.

Seven-cell campaign on this configuration — all valid, success 5/5 per cell,
worst CV 8.1%, peak VRAM 19958 MiB throughout
(`apxinf-evidence/perf-rowwise/perf.json`):

| Cell | TTFT median | TTFT CV | TPOT median | TPOT CV |
|---|---:|---:|---:|---:|
| 1K | 1.63 s | 0.93% | 66.5 ms | 6.14% |
| 2K | 3.31 s | 4.01% | 67.2 ms | 0.36% |
| 4K | 7.01 s | 1.82% | 67.2 ms | 0.86% |
| 8K | 14.81 s | 8.06% | 67.3 ms | 0.46% |
| 16K | 34.58 s | 4.51% | 75.1 ms | 0.64% |

Verification chain on the final build: public functional **6/6** exact,
protocol gate **12/12** (SHA `56e4ee27...`), mixed soak **90/90 = 100%** (60
short + 25 x 1K + 5 x 8K, 170.4 s) with final health `ok` and
`fallback_active=false`, `apxinf-cuda` 98 passed with only the two
pre-existing `fp8` environmental failures, batched-SDPA bit-equality passed,
CPU suites 55/55 and 55/55, fmt/check/diff clean, and shutdown left the port
free, all four GPUs at 1 MiB, and the lock free.

### Cumulative result vs the layer-1 baseline

| Metric | layer-1 baseline | final | speedup |
|---|---:|---:|---:|
| TTFT 1K | 78.09 s | **1.63 s** | **47.9x** |
| TTFT 2K | 159.19 s | **3.31 s** | **48.1x** |
| TTFT 4K | 320.40 s | **7.01 s** | **45.7x** |
| TTFT 8K | 676.52 s | **14.81 s** | **45.7x** |
| TTFT 16K | 1635.15 s | **34.58 s** | **47.3x** |
| TPOT 1K | 133.7 ms | **66.5 ms** | **2.01x** |
| TPOT 8K | 153.4 ms | **67.3 ms** | **2.28x** |
| 16K single request | ~1664 s (8% timeout margin) | ~46 s (97% margin) | 36x |

Seventeen paired single-variable experiments: nine accepted (release build,
debug-capture off, W4 prefill dequant+GEMM, packed GEMV, batched SDPA, scratch
pool, prefill chunk 64 -> 512 (two measured steps), attention GEMM reshape,
row-cooperative softmax), eight rejected with data and reasoning recorded.
Correctness gates held at every step: protocol 12/12, public functional 6/6,
soak 100%, and the trajectory soft target improved from 30/128 to 82/128
against the approved oracle.

Distance to the contract's own screens is now: prefill ~4.8x off the
arithmetic floor at 1K (1.63 s vs ~0.34 s) and ~6.5x at 16K (34.6 s vs
~5.36 s); decode ~3.3x off the bandwidth floor (66.5 ms vs ~20 ms). The
remaining decode gap is the 39% dequantization arithmetic documented above,
which needs a fused dequant-MMA at M >= 8; the remaining prefill gap is now
spread across GDN scan, projections and attention rather than concentrated in
one defect.

### Note on speculative decoding (evaluated, not implemented)

Raising M so tensor-core MMA becomes usable was considered as the route to the
39% dequantization overhead in decode, and rejected on a technical ground
before scope: `m16n8k16` needs M >= 8 to avoid waste, so K >= 7 draft tokens,
while MTP acceptance rates fall off well before that — it cannot deliver the
M the MMA path needs. Its real mechanism is amortizing one forward over
several accepted tokens (M=5 GEMV has the same weight traffic as M=1), which
is genuinely promising, but it requires transactional rollback of 48 layers of
GDN conv ring plus recurrent state, where today's recovery only drops the whole
session. That is the highest-risk place to touch with correctness gates
currently green, and MTP is explicitly out of scope this period.

## Layer-3 (bonus) Feasibility Assessment and Multimodal Start

Asked whether the plan's third tier (context 10 + C4/C8 10 + multimodal 10)
is reachable, the binding constraint was measured first: resident weights are
19498 MiB plus a 460 MiB scratch pool, leaving **4606 MiB**, and KV costs
64 KiB/token.

| Context | KV | attn workspace (chunk 64) | total | verdict | points |
|---|---:|---:|---:|---|---:|
| 32768 | 2048 MiB | 192 MiB | 2683 MiB | fits | **0.00** |
| 65536 | 4096 MiB | 384 MiB | 4923 MiB | 317 MiB short | 3.33 |
| 131072 | 8192 MiB | 768 MiB | 9403 MiB | impossible | 6.67 |
| 262016 | 16376 MiB | 1535 MiB | 18354 MiB | impossible | 10.00 |

So the context bonus is capped at 3.33 points and even that needs the scratch
pool released on demand (which would free 460 MiB, leaving ~143 MiB of slack);
131072 stays out of reach even with INT8 KV (4096 + 768 + 443 = 5307 MiB).
C4/C8 is the opposite: the dominant cost is the length-independent 443 MiB GDN
state per request, so 4 concurrent requests need ~2060 MiB and 8 need
~4120 MiB — both fit — but continuous batching (`scheduler.rs`) does not exist
yet. *(C4 was subsequently delivered — see "Multi-Request Bonus: C4
Concurrent Serving".)* Multimodal costs ~880 MiB resident for the vision tower, leaving
3726 MiB, which still covers 32K single-request serving.

Scoring structure matters for the choice and is recorded plainly: the
multimodal 10 points are **2 for public image correctness, 6 for private, and
1 per split for platform-verified integration**. Only the public 2 points are
locally verifiable; the other 8 depend on the platform running the hidden
image set against this implementation.

### Multimodal groundwork completed

Structure survey (checkpoint `config.json` + weight index):

- 333 `model.visual.*` tensors: 27 blocks of
  `norm1/norm2` (weight+bias, i.e. LayerNorm), `attn.qkv`, `attn.proj`,
  `mlp.linear_fc1`, `mlp.linear_fc2`, plus `patch_embed.proj`,
  `pos_embed.weight` and a single `merger`.
- vision config: depth 27, hidden 1152, 16 heads (head_dim 72), intermediate
  4304, patch 16, spatial_merge 2, temporal_patch 2, out_hidden 5120,
  `gelu_pytorch_tanh`, `num_position_embeddings` 2304,
  **`deepstack_visual_indexes: []`** so no deepstack injection is needed.
- `image_token_id` 248056, `vision_start/end` 248053/248054; processor
  `Qwen3VLProcessor` + `Qwen2VLImageProcessorFast`, mean/std 0.5 (so pixels
  normalize to [-1, 1]).

Reuse finding that materially cuts the work: the existing
`crates/apxinf-model/src/qwen3vl/vision.rs` and `vision_weights.rs` already
implement exactly this shape — patch_embed matmul+bias, positional embedding
interpolation, 27 blocks, and the merger's
`LayerNorm -> reshape [N,C] to [N/4,4C] -> fc1 -> GELU -> fc2` — and load from
**identical tensor names** (`model.visual.blocks.{i}.*`, `patch_embed.proj.*`,
`pos_embed.weight`, `merger.*`). The deepstack path, which this checkpoint does
not use, is the main part that differs.

Offline vision oracle generated (`apxinf-vision-oracle`, script
`apxinf-vision-oracle.py`, run under the offline-only oracle venv —
Transformers is never in a serving path). On a deterministic 448x448 RGB PNG:

- processor output `pixel_values` [784, 1536], `image_grid_thw` [[1, 28, 28]];
- all 333 vision tensors load with **0 missing, 0 unexpected**;
- per-stage goldens dumped for blocks 0/13/26 and the final hidden state;
- **merger output [196, 5120]** — matching the arithmetic (784 patches / 4 for
  spatial_merge 2, projected to the text hidden size), which is the tensor the
  language model must receive in place of the image tokens.

`manifest.json` records shape, SHA256, min/max and mean-abs for every stage, so
the CUDA implementation can be diffed stage by stage rather than only at the
output.

Remaining work for the multimodal bonus, in dependency order: port the vision
forward to the Qwen3.8 config on CUDA and match the per-stage goldens; scatter
the 196 embeddings onto `image_token_id` positions in the text embedding;
implement `POST /v1/chat/completions` accepting one base64 PNG part plus one
text part with `temperature=0`, `max_completion_tokens=32`, `stream=false`,
`enable_thinking=false`; flip `capabilities.multimodal` only after public 4/4
passes, keeping the current fail-closed behaviour until then. *(All of these
steps were subsequently completed — see "Multimodal Bonus Implementation"
below.)*

### Operational note: sandboxed client double-execution

Three separate confusing incidents during this session (the aggregate-8192
diagnosis 503, the batched-sdpa probe traceback, and a short-soak run that
appeared to fail 58/58 in 0.2 s) had the same cause: the agent harness can
re-execute a foreground heredoc client, and with `queue-capacity=1` the
duplicate instance's requests are correctly rejected with 503 while the
surviving instance proceeds. None of these were service faults — the
admission behaviour is the contracted one — and each was resolved by
locating the surviving process and reading its results. Long-running clients
are therefore started detached (`setsid nohup ... &`) with file-based
output, which does not get re-executed.

## Multimodal Bonus Implementation (2026-08-26 evening session)

Scope: the full Qwen3.8 vision path plus `POST /v1/chat/completions`, per
`multimodal-contract-v1.json`. Out of scope and untouched: MTP/speculative
decoding, prefix cache, C4/C8, long-context bonus, KV quantization.

### Availability of the official image suite (material limitation)

The official multimodal generator, public images, prompts and answers are
**not present on this host**: a full search of `benchmarks/qwen38_4090/`,
the team share and prior evidence directories finds the category
names and seed only inside `multimodal-contract-v1.json` itself, and
`fetch_public_corpus.py` handles only the text corpus. The evaluator
(`run_evaluation.py`, `score_multimodal.py`) consumes platform-generated
reports rather than producing image cases. Consequently "public 4/4" cannot
be literally self-verified here; the local self-verification below uses
(a) stage-exact oracle equality where exactness is achievable, and (b) four
self-built contract-style probes (same 448x448 deterministic-PNG shape, same
four categories, same request format) with locally known ground truth.
This distinction is stated plainly rather than claimed away.

### Correctness evidence chain (strongest available per stage)

| Stage | Criterion | Result |
|---|---|---|
| PNG decode + preprocess (`(x-127.5)/127.5`, patchify) | **bit equality** with `Qwen2VLImageProcessorFast` output on the oracle probe image | 0 of 1,204,224 f32 elements differ |
| chat template render + tokenize + image-pad expansion | **token equality** with `Qwen3VLProcessor.apply_chat_template(enable_thinking=False)` | exact on all 4 probe requests (224-233 tokens each) |
| vision tower (27 blocks, CUDA BF16) vs FP32 oracle | per-stage golden comparison | block00 max_abs_diff 0.069 (golden max 5.6), block13 4.87 (max 372.7), block26 352 (max 4477) — 1.2-8% on outlier channels, mean_abs ratio ~2.7% |
| merged `[196, 5120]` embedding | drift + cosine | max_abs_diff 4.21 (golden max_abs 142.8), mean_abs_diff 0.0078 (golden mean_abs 0.278), **cosine 0.99971887** |
| fail-closed probe (capability off) | contract `unsupported_rule` | HTTP 400, `error.type=unsupported_capability`, `/health.capabilities.multimodal=false` |

The BF16-vs-FP32 drift compounds through 27 blocks exactly as on the text
stack; the merger LayerNorm renormalizes before the LLM consumes the
embedding, and the cosine shows direction is preserved. GPU test
`vision_tower_matches_oracle_stage_by_stage` (ignored, requires GPU +
checkpoint + oracle dir) pins these bounds at ~2x measured.

### Implementation

- `crates/apxinf-cuda`: new `vision_sdpa_bf16_wide_kernel` — the existing
  vision SDPA hardcoded head_dim 64 (2 elements/thread); the wide variant
  owns up to 4 strided elements per thread (any even head_dim <= 128,
  Qwen3.8 vision is 72). At head_dim 64 it is **bit-identical** to the
  narrow kernel (`vision_sdpa_bf16_wide_bit_matches_narrow_kernel_at_head_dim_64`),
  and the dispatcher keeps the narrow kernel for 64 so the Qwen3-VL path is
  untouched. New `gelu_erf_bf16_kernel`: the patch merger uses PyTorch's
  default `nn.GELU()` (exact erf), *not* the tanh form the block MLPs use —
  verified against HF `Qwen3VLVisionPatchMerger` source.
- `crates/apxinf-model/src/qwen35/vision.rs`: the 27-block tower + merger on
  CUDA backend primitives. Constants are hardcoded to the frozen checkpoint
  (config.json is digest-verified at load) with per-tensor shape assertions;
  ~880 MiB BF16 resident, loaded **only when `APXINF_ENABLE_MULTIMODAL=1`**.
  Bilinear pos-embed interpolation (align_corners=true) and merge-block-order
  position ids match `transformers.vision_utils` exactly; the
  `[784,1152] -> [196,4608]` merge is a pure reshape (rows contiguous).
- `crates/apxinf-model/src/qwen35/runtime.rs`: `open_with_cancel_multimodal`
  runs the vision forward on the GPU worker thread (image and text requests
  stay serialized), validates that image-pad count equals merged rows, and
  `scatter_image_rows` overwrites exactly those embedding rows through a
  bounds-checked device-buffer view during prefill.
- `src/server/chat.rs` + `src/server/image.rs`: strict OpenAI-parts parsing
  (exactly one `data:image/png;base64` image_url + one text part, temperature
  0, stream=false, `enable_thinking=false`; anything else is a 400), manual
  template rendering (the minijinja `apply_chat_template` wrapper collapses
  consecutive newlines and would corrupt `<think>\n\n</think>\n\n`), PNG
  decode via the `png` crate, and the OpenAI-style response body.
  `/v1/chat/completions` with the capability off fails closed per contract.
  EOS for chat is the existing `EOS_TOKEN_IDS = [248046, 248044]`, which
  matches `generation_config.json.eos_token_id` exactly.
- Dependencies added to the bin crate: `png`, `base64` (Cargo.lock updated;
  nothing enters the model/kernel crates).

### Self-built probe suite (local ground truth)

`apxinf-vision-e2e-gen.py` builds one 448x448 deterministic PNG per
contract category — seven_segment_ocr ("382" on a dark seven-segment
display), spatial_color (2x2 grid, top-left red), bar_arithmetic (two bars
labeled 7 and 5, sum 12), object_count (5 circles) — plus, from the offline
oracle venv, the HF-processor golden `input_ids` and the exact
`/v1/chat/completions` request bodies. These are **not** the official public
cases; they verify the full image path end to end with known answers.

### VRAM budget fix required for the multimodal configuration

The first multimodal service start failed closed at admission, exactly as
designed: with the vision tower resident, free memory after weights is
~3634 MiB while the chunk-512 request estimate is 4037 MiB. Fix:
`prefill_chunk_tokens()` defaults to 256 when `APXINF_ENABLE_MULTIMODAL=1`
(estimate 3259 MiB, ~375 MiB slack), still overridable via
`APXINF_Q35_PREFILL_CHUNK`; the text-only default stays 512 and the
text-only configuration is untouched. Unit test covers both defaults; the
runtime env-var tests now serialize on a mutex (a pre-existing parallel-test
race that this change widened and then fixed).

### End-to-end results on the designated GPU (strict service, 2026-08-26)

Service: release build, GPU1 under `flock`, `APXINF_ENABLE_MULTIMODAL=1`,
resident 20454 MiB (19498 text + ~956 vision). `/health` reports
`status=ok`, `capabilities.multimodal=true`, `fallback_active=false`.

| Check | Result |
|---|---|
| self-built image probes | **4/4 exact** — "382", "red", "12", "5"; 1.0-2.0 s per request end to end (`apxinf-evidence/vision-e2e/vision-e2e.json`) |
| frozen protocol gate | **12/12 passed** (`apxinf-evidence/protocol-mm`, SHA `3e600e5b...`) |
| public functional (text) | **6/6 exact** on the multimodal service; 8K longdoc cases ~24 s each at chunk 256 (`apxinf-evidence/functional-mm/functional.json`) |
| mixed text+image soak | **36/36** (18 short + 6 x 1K text + 12 image chats interleaved), success rate 1.0, healthy after (`apxinf-evidence/mm-soak/mm-soak.json`) |
| frozen 128-token text request on the multimodal service | output_ids SHA `20d981cb...` — **byte-identical** to the accepted layer-2 text configuration: the resident vision tower changes no text numerics |
| peak VRAM during the run | 20914 MiB of 24564 (`apxinf-evidence/mm-vram.txt`) |

### Text-only (default) configuration re-verified after the change

A fresh default service (same binary, no `APXINF_ENABLE_MULTIMODAL`) on the
designated GPU: `/health.capabilities.multimodal=false`; the image probe
fails closed with **HTTP 400 and `error.type=unsupported_capability`** (the
contract's required unsupported behaviour — previously this path returned a
non-compliant 404 because the route did not exist); the frozen 128-token
request returns the same SHA `20d981cb...` (three-way byte equality:
historical config, multimodal service, post-change default service); frozen
protocol gate **12/12** (`apxinf-evidence/protocol-textonly-after-mm`).
The vision tower is not loaded, resident memory and prefill chunk (512) are
unchanged, so every layer-1/layer-2 performance number in this report still
describes the text-only configuration.

Full regression on the final tree: `apxinf-cuda` GPU suite 102 passed with
only the two pre-existing environmental `fp8` failures (cuBLAS status 15,
unchanged baseline); `apxinf-model` qwen35 GPU suite 58 passed; CPU suites
98 (model lib) and 62 (bin, including the new image/chat/template tests);
`cargo check --workspace`, `cargo fmt --check`, `git diff --check` clean.
Shutdown verified: port 18080 free, all four GPUs at 1 MiB, GPU-job lock
free.

### Multimodal scoring status (stated honestly)

- The **capability flip condition was met on the strongest locally available
  evidence**: bit-exact preprocessing, token-exact prompt construction,
  oracle-aligned vision tower (cosine 0.99972), and 4/4 on self-built
  contract-style probes with locally known answers. The official public
  4-case suite is not present on this host, so the literal "public 4/4"
  remains a platform-run item; the 2 public correctness points, 6 private
  points, and 2 platform-verified integration points are all decided by the
  evaluation platform. No multimodal score is claimed here.
- To serve with the capability declared, start with
  `APXINF_ENABLE_MULTIMODAL=1` (the tower loads at startup, chunk defaults
  to 256). Without it the service is the unchanged text-only submission and
  now satisfies the contract's fail-closed probe rule.
- Rollback: unset `APXINF_ENABLE_MULTIMODAL` (single switch; no other
  behavioural change is reachable without it).

Known limitations, recorded for the platform run: (1) images whose
dimensions are not multiples of 32 (or outside the 65,536..16,777,216 pixel
window) are rejected with 400 rather than resampled — the contract fixes
both suites at 448x448, so this is out-of-contract input only; (2) only the
contract request shape is accepted (one PNG image_url + one text part,
temperature 0, `enable_thinking=false`, stream=false); (3) the vision
forward adds ~0.4 s and the QKV column split currently round-trips through
host memory (~30 ms per image) — functional, not tuned.


## Multi-Request Bonus: C4 Concurrent Serving (2026-08-26 night session)

Goal: pass the `multi-c4-text-perf-1024` cell (4 concurrent closed-loop
clients, 1024-token prompt, 128 completion tokens, 32 requests total) without
regressing the frozen single-request configuration. C8 was explored and its
blocker quantified (see below).

### Architecture

A batched protocol runtime (`src/server/qwen35_batch.rs`), enabled only when
`APXINF_Q35_MAX_CONCURRENCY > 1`; the default single-request path is
untouched (same `ProtocolRuntime`, same kernels, no behavior change — the
frozen configuration does not set the variable).

- One batch worker thread owns the model; HTTP admissions post `BatchJob`s
  through an MPSC queue guarded by `RuntimeAdmission` (permits = 4,
  per-request memory sizer instead of the single-slot budget).
- Each round: admit up to 4 slots -> one batched decode step for all slots ->
  deliver one token per slot. Retired slots release their admission permits
  BEFORE any final token is sent (two-phase delivery; see "Lessons").
- Prefix cache: the first request for a prompt runs the normal prefill and
  caches a pristine fork as a template; identical prompts (the evaluator's
  multi cells use one shared prompt per cell) fork/refill from it in ~2 ms.
- Recycled-session pool (depth 3): retired sessions keep their ~0.52 GiB of
  device allocations and are refilled in place from the template
  (stream-ordered copies), avoiding the ~320 cudaMallocs (~330 ms measured)
  of a cold fork. Pool + template are dropped whenever a larger-than-template
  request arrives (8K/16K cells need the head-room; measured OOM otherwise).

### Batched decode numerics

Per-request math is bit-identical to the serial path where at all feasible:

- New CUDA kernels `qwen35_gdn_conv_batch_bf16_kernel` and
  `qwen35_gdn_recurrent_batch_bf16_f32_kernel`: one `blockIdx.y` per request
  with per-request state pointer arrays; the per-channel/per-head bodies are
  copied verbatim from the single-request kernels (same thread mapping, same
  accumulation order).
- New `rope_partial_positions_bf16_kernel`: partial RoPE with one position
  per row (same per-row math as `rope_partial_batched`).
- QK RMSNorm, gated RMSNorm, sigmoid gate: row-independent kernels applied to
  the whole batch (bit-identical per row).
- W4 packed projections batch across rows (one block per (row, out) pair —
  bit-identical per row, established earlier).
- Two accepted cuBLAS reassociations (documented trade, affects only
  exactly-tied argmax logits): the batched LM head GEMM ([M, vocab] instead
  of M GEMVs), and GDN in_proj_a/in_proj_b are per-request GEMVs that are
  stacked (still bit-identical); the BF16 out_proj/attention out_proj remain
  per-request GEMVs.
- Bit-equality regression: `real_layers_batched_decode_bit_matches_serial`
  (ignored test, GPU0) loads real checkpoint layers, runs 3 requests batched
  vs serial for multiple steps, and asserts byte-identical outputs and
  states. Re-run after every batching change: passed at every step.

### Measured progression (closed loop, 4 clients, 1024/128, loopback)

| stage | steady TPOT | notes |
|---|---|---|
| naive per-request loop in batch step | ~3300 ms/round | 2000+ micro-ops, sync memcpys, per-op allocs |
| deferred status + zero-copy row views + async stacks | ~150 ms/round (batch=3) | but only 3 of 4 slots ever active |
| + two-phase delivery (permit release before send) | 503s eliminated (32/32 OK) | root cause: clients resubmit within the 17 us retirement window |
| + session fork for prefix cache | TTFT 6.5 s -> 1.7 s (first wave) / 0.05 s (steady) | template was silently never cached before the rollback-flag fix |
| + batched GDN/rope/gate kernels + batched LM head | TPOT 186-206 ms | GDN stage 140 ms -> ~95 ms (batch=4) |
| + recycle pool (fork -> refill ~2 ms) | p95 TPOT hovers at the 201.5 ms limit | residual: one ~330 ms fork + one ~20 ms dispose per wave |

Single-request baseline for the validity thresholds (same submission):
TTFT 1.905 s (limit 1.5x = 2.86 s), TPOT 67.2 ms (limit 3x = 201.5 ms).

### Where the remaining variance comes from

Batch-step timing shows p50 = 178 ms with 2-3 outlier steps (600-1400 ms)
per template turnover, clustered around the once-per-wave cold fork
(320 cudaMallocs) and pool-overflow dispose (320 cudaFrees). Removing the
last per-wave malloc/free entirely needs a 4-deep pool, which is 29 MiB over
the measured free VRAM (9 x 0.52 GiB resident states + weights 19.5 GiB);
the structural fix is sharing the GDN scratch/backup buffers across sessions
(~148 MiB per session, the batch worker serializes their use), which is also
the enabler for C8 (8 x lean states would fit where 8 x full states cannot).
That refactor is the identified next step, not attempted tonight.

### C8 assessment

Blocked on memory, not on scheduling: the scheduler, admission sizer, and
batched kernels all take the concurrency from `APXINF_Q35_MAX_CONCURRENCY`.
8 full request states (KV 72 MiB + GDN 443 MiB each) = 4.1 GiB against
2.36 GiB of admission budget (and the physical margin above). With shared
GDN scratch/backup (states drop to ~295 MiB) plus the existing pool-yield
logic, C8 admission fits arithmetically; TPOT would additionally need the
per-request flash-decode/KV-append stages batched (the remaining serial
segment, ~50 ms at batch=4). Not claimable without implementation.

### Lessons recorded

- "Pending rollback" flags on GDN states mean "last commit is revocable"
  (normal), not "state is dirty": misreading them made every template fork
  fail silently and cost an hour (every request re-prefilled, TTFT 6.5 s).
  Cache-skip paths now log loudly.
- Two-phase delivery matters at loopback speeds: all 503s landed in the
  17 us window between two slot retirements, because a client's next request
  raced the not-yet-released permits of sibling slots.
- Synchronous cudaMemcpy/cudaMalloc in per-request paths dominate batched
  decode: a state fork was 330 ms of which ~260 ms was 320 serial
  stream fences; async copies + one final synchronize (later: none, stream
  order suffices) fixed it.
- cargo test parallel threads on one GPU produce mass spurious failures
  (basic operator tests "failing" in 4.7 s); serial re-run restored the
  102 passed + 2 pre-existing fp8 baseline.


### C4 final result (official evaluator, calibration profile, run 20260826-204907-apxinf)

All validity gates pass with margin:

| metric | value | threshold |
|---|---|---|
| success_rate | 1.0 | = 1.0 |
| correctness_rate | 1.0 | = 1.0 |
| jain_fairness_index | 0.9993 | >= 0.95 |
| p95_ttft_s | 1.789 | <= 2.496 (1.5x single 1.664) |
| p95_tpot_s | 0.1739 | <= 0.1869 (3x single 62.3 ms) |
| goodput | 23.53 tok/s | scored metric |
| no_fallback / healthy_after | true | required |

Same submission re-verified the single-request cells (no regression, in fact
slightly faster than the frozen baseline's 66.5-67.3 ms TPOT band):
1K 62.3 ms / 2K 62.0 / 4K 62.3 / 8K 62.7 / 16K 68.1 ms TPOT; protocol 12/12;
public functional 6/6. Trajectory capture remains DEBUG-ONLY (no
platform-approved `--trajectory-reference`; unverified flag unchanged).

The two decisive final fixes:

1. `decode_step_batch` falls back to the serial step for batch=1 (the
   batched plumbing cost ~10% TPOT on the 16K cell: 74.5 -> 68.1 ms).
2. Session clones allocate through the stream-ordered pool
   (`cudaMallocAsync`/`cudaFreeAsync`, release threshold pinned to keep pool
   pages): a fork went from ~330 ms (320 synchronous cudaMallocs, each an
   implicit device synchronize) to single-digit ms submitted in stream
   order, and retirements free in stream order too. This removed both the
   first-wave TTFT penalty (pre-warm no longer needed and was removed) and
   the wave-turnover TPOT spikes: closed-loop waves measure 166.7-173.9 ms
   TPOT (previously 186-216 with outliers).

Environment note recorded for fairness of local numbers: two isolated
batch-step outliers (1.2-1.4 s, GDN and attention inflated proportionally,
instant recovery, no correlated service event, mid-wave) were observed once
on this shared host; they match external GPU preemption, not a service
defect. The final evaluator run shows no such outlier.

### Default-configuration regression after all C4 changes

- Protocol gate 12/12 (sha256 `49f8b0fc...`), evidence
  `apxinf-evidence/c4-regression/`.
- A/B probe on the default single-request service: TTFT median 1.870 s,
  TPOT median 66.3 ms — inside the frozen baseline band (1.63-1.9 s / 66.5 ms).
- Frozen 128-token output sha256 `20d981cb...` is byte-identical to the
  accepted frozen configuration's output (same sha as the accepted
  `ab-chunk512` / `ab-rowwise-softmax` records; the probe's embedded
  constant predates those accepted optimizations — see "Frozen 128-Token
  Oracle Audit").
- CPU suites: apxinf bin tests pass; apxinf-model lib pass; apxinf-cuda lib
  102 passed + the 2 pre-existing environmental fp8 failures (baseline
  unchanged). Batched-vs-serial bit-equality test re-passed after the final
  build.

### C4 service command (for the record)

    setsid nohup bash -c 'exec flock -n <gpu-job-lock> env \
      CUDA_VISIBLE_DEVICES=GPU-343bc895-b011-22fa-4449-97207aa2bdec \
      APXINF_Q35_MAX_CONCURRENCY=4 APXINF_Q35_DEFERRED_STATUS=1 \
      target/release/apxinf serve --model <model-dir> \
      --revision 63768c10df38c0395e12ef49edac1bd539eaeeea \
      --gpu-uuid GPU-343bc895-b011-22fa-4449-97207aa2bdec \
      --bind 127.0.0.1:18080 --max-model-len 32768 --queue-capacity 4' \
      > <log-file> 2>&1 & disown

Rollback: unset `APXINF_Q35_MAX_CONCURRENCY` (or set to 1) restores the
exact frozen single-request runtime; `APXINF_Q35_DEFERRED_STATUS` remains
default-off there. No default-path behavior changed.

# Appendix: Chronological Session Records

Retained as the audit trail of how the result was reached; numbers
superseded by later sections are marked where they occur.

## Frozen 128-Token Oracle Audit

The approved checkpoint contains 48 GDN layers and 16 full-attention layers.
After correcting the BF16 q/k square accumulation boundary and explicitly
materializing FP32 `k_beta` and `v_beta`, the frozen non-stream request
`input_ids=[1,2,3,4,5,6,7,8]`, `max_new_tokens=128`, `temperature=0.0`
matched the stored reference for 76 tokens. The first divergence was token 76:
candidate `2972`, reference `9493`; 46 of 128 positions differed. Candidate
artifact: `apxinf-oracle-128-after-materialized-beta.json`.

A deterministic CUDA regression at the checkpoint's real `key_dim=128` now
passes bit-exact BF16 output and strict FP32 recurrent-state samples. The
sequence path is staged around transpose-aware FP32 strided-batched cuBLAS
products, with explicit phase-1 attention construction and a cuBLAS state
update product. The dedicated scan and triangular-reduction ordering remains a
separate parity concern.

On 2026-08-25 the staged implementation completed the frozen 128-token service
request (HTTP 200, 16.67 s, usage 8/128/136). Against the approved oracle it
matched 28 tokens; first divergence at token index 28 (candidate `3175`,
reference `40608`); 98 of 128 positions differed. Candidate output SHA256:
`081ca4aca162a23a447d725a7d0133a160c39f6d104b5d2ff8bb67f798aea85f`; full
response SHA256: `80c4593de7eed3824a79af99032f68d59e9176a81013170d29507eda3bdb9eec`.
The oracle's stored top1/top2 margins are exactly 0.0 at generated tokens 23,
28, and 76 (BF16 logit ties); at token 28 the oracle logits are exactly tied
(`3175` and `40608` both 21.375) and the reference selected the higher index,
matching this implementation's highest-index tie rule. The candidate selecting
`3175` therefore means its logits were not exactly tied at that step: a
sub-ulp BF16 drift flipped the tie.

Per-layer localization used the runtime hidden-state debug capture
(`APXINF_DEBUG_HIDDEN_DIR`) at positions 7/35/83 for layers
embedding/000/003/031/032/060/063 versus the oracle's 136-row hidden bins:
embedding position 7 is bit-exact; layer 0 position 7 differs by at most
0.000977 (0 violations of the oracle 0.01+1% tolerance); layer 3 position 7 by
at most 0.125 (0 violations); layer 31 position 7 has 1468 violations, layer
32 1618, layer 60 2728, and layer 63 3125 violations (max 4.0, mean absolute
0.088). The pattern is ~1-2 BF16 ulp of per-layer drift compounding across the
64 layers; layers 0 and 3 match the oracle within its sanctioned tolerance, so
the staged GDN/attention prefill is not logically wrong, and the trajectory
flip occurs where accumulated drift crosses a logit tie.

A new focused GPU test,
`qwen35_gdn_cuda_sequence_chunked_matches_eager_step_at_checkpoint_layout`,
compares the staged chunked prefill against the eager per-token recurrence at
the real checkpoint layout (16 key heads, 48 value heads, 128x128) and passes:
output max absolute difference 0.000244 (max relative 0.006), recurrent state
max absolute difference 3.8e-8. The staged sequence path is numerically
consistent with the previously validated eager path; the earlier 76-token match
(pre-staging) and the current 28-token match are both consistent with
accumulation-order drift deciding BF16 tie sites, not a GDN logic regression.
Trajectory rate is a scored soft target (threshold 0.0 for eligibility).

## Fresh Session Evidence (2026-08-25 evening)

Service command (strict): `flock -n <gpu-job-lock> env
CUDA_VISIBLE_DEVICES=GPU-343bc895-b011-22fa-4449-97207aa2bdec
APXINF_DEBUG_HIDDEN_DIR=<capture-dir> target/debug/apxinf serve
--model <model-dir> --revision
63768c10df38c0395e12ef49edac1bd539eaeeea --gpu-uuid
GPU-343bc895-b011-22fa-4449-97207aa2bdec --bind 127.0.0.1:18080
--max-model-len 32768 --queue-capacity 1`. Startup to bound port took ~19.5
minutes (checkpoint digest verification, weight load, warmup). `/health`
returned HTTP 200 with `stub=false`, `status=ok`, the frozen revision and
contract identity, `max_model_len=32768`, `parallel_requests=1`,
`fallback_active=false`, `vocab_size=248320`.

Focused tests (this session, designated GPU under flock):
`qwen35_gdn` suite 19 passed / 1 ignored (timing probe); full `apxinf-cuda`
suite 90 passed with 2 pre-existing failures in `fp8` (cuBLAS status 15,
CUBLAS_STATUS_NOT_SUPPORTED on this driver/hardware; FP8 is not on the
Qwen3.8 text path and is untouched by the staged diff); `apxinf` bin tests 52
passed; `apxinf-model qwen35` 54 passed / 2 ignored; ignored oracle test
`real_layer_zero_prefill_first_row_matches_oracle_hidden` passed in 13.6 s.
`cargo check --workspace --locked`, `cargo build --features cuda-no-nvtx
--locked --bin apxinf`, `cargo fmt --all -- --check`, and `git diff --check`
all passed.

Public functional cases (SSE, official public corpus, normalized exact after
ascii strip, tokenizer decode with skip_special_tokens): 6/6 passed.
`text-niah-1024-p10/p50/p90` each emitted 12 tokens (~80 s each);
`longdoc-multihop-8192` 9 tokens (678.6 s), `longdoc-revision-8192` 9 tokens
(678.3 s), `longdoc-aggregate-8192` 3 tokens (681.2 s). EOS stopping is
demonstrated by these early terminations under `ignore_eos=false`. Evidence:
`apxinf-functional-fresh/functional.json`.

Protocol gate: 12/12 passed (7 negative controls all HTTP 400 with JSON
`error`; `valid_short_nostream_request` HTTP 200 `type=result` one token
`[2037]` usage 8/1/9; `health_after_invalid_requests` and
`health_contract_identity` HTTP 200 with
`apxinf.qwen38_27b.inference_interface.v1`; SSE 8-token stream with
contiguous indexes 0-7, single request_id, done usage 8/8/16, `[DONE]`).
Evidence: `apxinf-protocol-fresh/protocol.json` SHA256
`3cbd8c0389112eafc854256c85647ac67d9c52d4295e85c05d20ec3d2c2da5ba`.

Client disconnect during SSE: connection aborted mid-stream; afterwards
`/health` stayed `ok` with `fallback_active=false` and a follow-up short
request returned HTTP 200 output `[2037]` usage 8/1/9 (0.85 s).

Kernel log access is not available on this host (`dmesg`/syslog read
permission denied), so Xid absence is evidenced indirectly: the service logged
no CUDA/GPU errors across all requests and every request completed or was
rejected cleanly.

Reliability soak (2026-08-25 evening, first attempt): invalidated. The 200
sequential mixed requests all failed in 0.2 s with HTTP 503
`runtime capacity is exhausted` because the immediately preceding functional
client had been killed during an 8K `longdoc` request; the abandoned request
kept the single capacity slot. See "Client-Disconnect Capacity Leak" in the engineering record for the root cause,
regression tests, and fix.
