# Qwen-Drive RTX 4090 performance pilot

Correctness differences versus the official reference are accepted for this performance experiment; this is not a correctness qualification.

DSH with infini-ai/kimi-k3 removed unconditional diagnostics from general.rs. The source candidate is `1931881583b975925c7e7b2b086ead35c2d0a66dee27cd6dc74fdbbe28d2027f`. Real model computation and all 49 source files were retained. Baseline source candidate: `1f2c1e52500d2aa50afec1bfba033a3e163bc1571614477d8b145920c7bbb028`.

Workload: four fixed public VQA scenes, batch 1, BF16, exactly 64 generated tokens; one warmup and three measured requests per scene. Host measures complete policy.infer latency, excluding weight loading and input-file loading. No profiler. This is one baseline/candidate measurement batch, not an alternating replicated qualification.

| Scene | Baseline median (s) | Candidate median (s) | Speedup |
|---|---:|---:|---:|
| 0 | 3.661982 | 3.588105 | 1.0206x |
| 1 | 3.667362 | 3.579821 | 1.0245x |
| 2 | 3.678464 | 3.571942 | 1.0298x |
| 3 | 3.660940 | 3.593195 | 1.0189x |

Geometric mean speedup: 1.023421x (approximately 2.29 percent latency reduction). All requests met the fixed output count. The 1.05x performance target is not reached; optimization continues. No claim applies to full-length generation, direct/reasoning/perception, throughput under concurrency, or general serving.

The installed Nsight Systems 2023.3.1 failed processing a separate profiler attempt and exported no CUDA kernel records; its partial API statistics are invalid for attribution. The timing result above comes from the independent no-profiler benchmark.


## Follow-up repeat: invalid source/library binding

The subsequent baseline-then-candidate batch recorded baseline medians
[3.676311, 3.705151, 3.687875, 3.674581] seconds and candidate medians
[3.673305, 3.677313, 3.680310, 3.686379] seconds. Its apparent paired ratio of
1.001804x is **invalid as a candidate comparison**: both jobs loaded the same
baseline native library, SHA256
`80846182c679b72338fe450b74e4b0a7249faa00b1e9eed7c2027a04e657b670`.
The earlier interpretation that this batch failed to reproduce the gain is
withdrawn. The initial 1.023421x observation remains unreplicated.

A later prefill-candidate replay exposed the same defect: source contents changed,
but the source modification time preceded Cargo's build timestamp, so Cargo
reused the preceding library. Performance verifiers checked workload execution
but did not detect this source/library mismatch.

The experimental Host now explicitly timestamps changed inputs and rebuilds the
model and Python bridge packages before measurement, retaining CUDA dependency
caches. Fresh paired measurements are recorded below; broader qualification remains pending.

## Rebuilt prefill candidate: paired replay

Candidate `3d200540927725dd9fa9f2e8f655264c8bd297db3d08c9d40611ea375065b8d7`
adds last-row prefill final norm/LM-head selection to diagnostic removal.
The repaired build produced candidate library
`430a5f470d41ad2d135ef11274af8c331fd4cd4e49a12bb9fdfd2c927fc4ae59`,
matching its initial build and differing from the baseline library above.

| Scene | Fresh baseline median (s) | Candidate median (s) | Paired speedup |
|---|---:|---:|---:|
| 0 | 3.680739 | 3.565502 | 1.032320x |
| 1 | 3.725618 | 3.602023 | 1.034313x |
| 2 | 3.675358 | 3.605967 | 1.019243x |
| 3 | 3.697119 | 3.568985 | 1.035902x |

Paired geometric mean speedup is **1.030423x** (2.95 percent latency reduction).
Against the frozen original baseline, the same candidate samples yield
1.022760x; the canonical 1.05x gate remains unmet. All four scenes generated
64 tokens. This replay supports a modest improvement for the combined candidate,
not isolated attribution to last-row selection or a broad serving qualification.

The prototype crops logits in the shared text path and therefore does not retain
the public multi-token forward output shape. It must be scoped to generation
before promotion as a general API-compatible implementation. This source change
is not included in the diagnostic-removal commit documented above.

## Rebuilt diagnostic-only paired replay

The diagnostic-only candidate `1931881583b975925c7e7b2b086ead35c2d0a66dee27cd6dc74fdbbe28d2027f` was rebuilt as library `241496aa17864ec4cdf4bfec9e0e6db70e2439b004b6d40e9875dce28e109ba1`, matching its initial build and distinct from baseline. Fresh baseline medians were [3.642118, 3.666379, 3.665816, 3.678474] seconds; candidate medians were [3.598071, 3.609911, 3.620278, 3.636127] seconds. Paired geometric speedup was **1.013026x** (1.29 percent latency reduction), with all four scenes generating 64 tokens. The fixed-original-baseline ratio was 1.014134x, below the 1.05x gate. This supports a small diagnostic-removal benefit; subtracting it from the separate prefill paired batch would not isolate prefill's effect because the batches ran at different times.

## Position-cache candidate: first valid measurement

DSH/Kimi-K3 repaired proposal serialization in revision7 and submitted candidate `287766b4a414133295c2efb2aaf8de50e9685378e9b29bdf23e841649e6d5ca4` (proposal `2f4b2fc3e64782484f7c6361c2a838faf980a75f1f2ef906e1e53f2cde549882`). The repaired build produced library `80733db3eadc95ac2a5bd51ac7ddd32b6884c67ce5be7a425b5951a12a400535`. Four scene medians were [3.488187, 3.529561, 3.489460, 3.492119] seconds, all with 64 generated tokens. Geometric speedup versus the frozen baseline was **1.047829x** (4.56 percent latency reduction); valid=true, passed=false. The 1.05x target remains unmet and replay is pending.

The prototype adds a process-global host position table and grid-keyed device tensor cache. It does not own these caches per model, bound the number of grids, or key on model weights/device context. Those lifecycle limitations and the shared forward-output-shape limitation remain; this measurement applies only to the fixed single-model process workload. It is not general deployment qualification.

## Position-cache paired replay: threshold-sensitive

The same source and library were rebuilt for a fresh baseline-then-candidate replay. Baseline medians [3.648873, 3.665581, 3.671292, 3.660577] seconds; candidate medians [3.508858, 3.491607, 3.482623, 3.471670] seconds; all four scenes generated 64 tokens. Speedup against the frozen original baseline was 1.051171x (this replay's verifier passed), but the fresh paired ratio was **1.049563x**. The initial batch was 1.047829x; the geometric mean of these two frozen-baseline batch ratios is 1.049499x (descriptive only, not a new acceptance rule). This is a threshold-sensitive result, not robust evidence of clearing 1.05x. Preserve the original failed receipt and the passed replay; continue optimization rather than selecting the passing batch. Cache lifetime and public forward-shape limitations remain.

## Packed input projections: initial performance pass

DSH/Kimi-K3 revision8 submitted candidate `5d04061b2ed7826aa6dee367e1fdf1ef8e5584e98a8bdcdaa4c97329b1b083d8` (proposal `f166545ec96a09fb36f65e00cf6d210080d89b5037b2d1a8600ade7843ca735e`). Rebuilt library SHA256: `e1ea3d204b4b970b7c3aac3db807a408f361d29ea13d43d1dcbb940a771e340f`. With stacked input projections replacing separate GEMMs plus concat, four scene medians were [3.300674, 3.294143, 3.289608, 3.292775] seconds. All scenes generated64tokens; valid=true, passed=true; geometric speedup **1.113190x** (10.17 percent latency reduction) against the frozen baseline. Fresh paired replay is pending. The existing cache lifetime/forward-shape limitations remain; reference numerical parity stays waived.

## Packed-projection replay and Mission completion

The rebuilt candidate repeated at scene medians [3.316539, 3.274849, 3.287977, 3.284270] seconds, all64tokens. Frozen-baseline speedup was **1.114349x**, and speedup relative to the fresh paired baseline was **1.120495x**. Both initial and replay candidate batches passed; the replay library matched `e1ea3d204b4b970b7c3aac3db807a408f361d29ea13d43d1dcbb940a771e340f`. KerSor's fixed Host verifier completed the current VQA64 Mission successfully. This snapshot preserves the measured experimental implementation, including its documented process-global cache and public-forward shape limitations; it is not a general API/deployment qualification. Further performance work and broader workloads remain separate from this completed pilot.

## Successor64/256: decode workspace candidate r2

Candidate `ee36b1c93639a2723f9146ddcb20c2a2f604f11c9b9ac251c09f4c652b9ac965` (proposal `94a896aa1a8a3fef664096d5c58f7b0bf45425978ac9656d4150d20a7957fe36`) reused model-owned single-token intermediate buffers. Rebuilt native library `88a08483027f13549312d19c38ac46bb5223a26290cb42cde33437eb21be3c8e` differs from baseline. Eight-cell output lengths matched (64/64/64/64;251/88/256/256), valid=true, passed=false. Geometric speedup vs the packed-projection successor baseline was **1.005600x** (~0.56percent latency reduction). 64-cell medians [3.231200,3.231659,3.268582,3.270116]s;256-budget medians [5.973086,3.569045,6.042304,6.009752]s. Two longer cells slightly regressed (~0.97percent and1.12percent); no cell exceeded the5percent regression cap. This initial result does not support the predicted large allocation-removal benefit and is not a promoted speedup. Keep the frozen baseline and choose further changes from evidence; do not claim call-count estimates as measured latency savings.

## Successor64/256: mRoPE cache candidate r3

Candidate `75513b8cdf5b7e6cef767f80cc70a9606edfc70a61e30c9c7a365301e7b90b97` (proposal `ab191bb07b60c9179fac93ba839a73cb19007da231804c32930aaa7dd87314b2`) stacked a bounded model-owned mRoPE window on r2 workspace reuse. Rebuilt library `abf3efcdb9e517aafa87fe57d07fac854aeb4ca88cd9823e9d56cc66572bd554`. Eight-cell output lengths matched; valid=true, passed=false. Geometric speedup versus frozen packed-projection baseline **1.008897x** (~0.88percent latency reduction). 64-cell medians [3.281621,3.348221,3.265024,3.260978]s;256-budget medians [5.799268,3.592477,5.847541,5.905729]s. Long scenes0/2/3 improved around2percent, but scene1@64 regressed2.16percent; no cell exceeded5percent regression cap. Source estimates remain insufficient; several samples contain latency spikes, and the late-decode pacing rise persisted. No robust promotion or mechanism attribution from this single batch. Further work should target costs with meaningful margin; lifecycle/interface limitations remain open.

## Successor r4: cross-module decode buffer reuse

DSH/Kimi-K3 candidate `88b8de2ce96444d67213dc40cec1707f73fd46a321d2da981204f88e64b04f7b` added output-buffer variants and model-owned decode scratch across CUDA wrappers. The first eight-cell batch measured **1.060107x** versus the frozen packed-projection baseline, valid=true and passed=true; output counts matched in all eight cells. The current Mission completed its timing verifier. Independent baseline/candidate replay was started on 2026-09-14 and remains pending. Existing public-forward last-row cropping remains a delivery limitation; timing acceptance does not establish full deployment readiness.
