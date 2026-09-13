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
