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
