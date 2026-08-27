# SGLang vs vLLM vs ApxInf Comparison - Iteration 29

Date: 2026-08-26
Model: `../model/qwen`, Qwen3.8-27B-AWQ-INT4, pinned revision from the benchmark contract
Hardware: one RTX 4090 per process, compute capability 8.9
Workload: pretokenized 1,024-token prompt, temperature 0, greedy decode, 128 output tokens, one request at a time

## Executive result

| Backend | Runtime | Startup | 1K TTFT | 1K TPOT | 1K decode | Exact frozen trajectory | Nsight CUDA kernels |
|---|---|---|---:|---:|---:|---|---|
| ApxInf | current default | pass | 1.194 s profiled | 25.834 ms profiled | 38.71 tok/s | 128/128 | present |
| vLLM | 0.27.1, FP8 KV | pass | 0.770 s profiled | 20.284 ms profiled | 49.30 tok/s | 93/128 local comparison | present |
| SGLang | 0.5.9, torch 2.9.1+cu128 | **fail before readiness** | unavailable | unavailable | unavailable | unavailable | none |

The vLLM result is a runtime comparison, not an exact-correctness claim: its output trajectory differs from the ApxInf frozen reference, as already observed in the local vLLM control. ApxInf passes the frozen trajectory gate.

SGLang has no valid throughput result for this checkpoint in the installed environment. It reaches Qwen3.5 model construction but rejects the asymmetric compressed-tensors scheme before loading weights. No SGLang latency or kernel value is fabricated.

## Identical workload and commands

The same `text-perf-1024` case was used for both viable backends:

- input IDs: public dataset `benchmarks/qwen38_4090/evaluation/.cache/public`;
- prompt length: 1,024 tokens;
- output budget: 128 tokens;
- temperature: 0;
- EOS ignored;
- one request, batch 1;
- streaming token IDs collected by the client.

ApxInf default profile:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --cuda-graph-trace=node --output=/tmp/cmp-apxinf \
  ./target/release/apxinf serve --model ../model/qwen \
  --host 127.0.0.1 --port 8053
```

vLLM profile:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --cuda-graph-trace=node --output=/tmp/cmp-vllm \
  ../model/vllm/.venv/bin/vllm serve ../model/qwen \
  --served-model-name vllm-control \
  --max-model-len 16512 --gpu-memory-utilization 0.98 \
  --kv-cache-dtype fp8 --trust-remote-code --max-num-seqs 1 \
  --tensor-parallel-size 1 --host 127.0.0.1 --port 8054
```

SGLang profile attempt:

```bash
SGLANG_DISABLE_CUDNN_CHECK=1 \
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --cuda-graph-trace=node --output=/tmp/compare-sglang \
  ../model/sglang/.venv/bin/python \
  ../model/sglang/.venv/lib/python3.12/site-packages/sglang/launch_server.py \
  --model-path ../model/qwen --served-model-name sglang-control \
  --context-length 16512 --dtype bfloat16 \
  --quantization compressed-tensors --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 1 --host 127.0.0.1 --port 8051 \
  --trust-remote-code --language-only --enable-layerwise-nvtx-marker \
  --skip-server-warmup
```

A second SGLang attempt with `--quantization awq_marlin` was also made. SGLang rejected the override because the checkpoint config declares `compressed-tensors`.

## Runtime versions

- vLLM: `0.27.1`, torch `2.13.0+cu130`.
- SGLang: `0.5.9`, torch `2.9.1+cu128`, CuDNN 9.10.
- ApxInf: native Rust/CUDA implementation, current release build.

SGLang first stopped on its explicit CuDNN compatibility guard: torch 2.9.1 with CuDNN 9.10 is known to require CuDNN >=9.15. The documented `SGLANG_DISABLE_CUDNN_CHECK=1` bypass was used once. After bypassing, SGLang failed at:

```text
NotImplementedError: No compressed-tensors compatible scheme was found.
```

The failing layer was `linear_attn.in_proj_qkv`. Source inspection shows the installed SGLang compressed-tensors WNA16 scheme requires symmetric weights, while this checkpoint declares:

```json
{
  "format": "pack-quantized",
  "group_size": 32,
  "num_bits": 4,
  "strategy": "group",
  "symmetric": false
}
```

Therefore the SGLang blocker is format/scheme incompatibility, not a measured inference bottleneck.

## Client measurements

ApxInf 1K profiled request:

- tokens: 128;
- TTFT: **1.1938 s**;
- TPOT: **25.834 ms**;
- E2E: **4.4747 s**;
- output SHA: `7eedbc78e930361a167ea9dec3f827d5ea9aeb25148e18fb854c5efe36e85bea`;
- exact frozen trajectory: **128/128**.

vLLM 1K profiled request:

- tokens: 128;
- TTFT: **0.7695 s**;
- TPOT: **20.284 ms**;
- E2E: **3.3457 s**;
- output SHA: `7577a39279f2e7677eb700353d806fd88ff685826d5806a4aab650659823aede`;
- local frozen-reference comparison: **93/128**.

The earlier two-cell profiled vLLM control also measured 8K TPOT around 20.56 ms/token. The same backend’s canonical control artifact records 1K 20.220 ms and 8K 20.512 ms in the public comparison setup.

## Nsight module breakdown

`nsys stats` reports were exported with:

```bash
nsys stats --force-export=true \
  --report nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --format csv /tmp/<backend>.nsys-rep
```

The following buckets aggregate CUDA kernel names by module. They include model startup, prefill, and the profiled decode request; they are not substituted for the client TPOT measurement.

### ApxInf

| Module | Kernel instances | GPU time |
|---|---:|---:|
| CUTLASS BF16 GEMM | 3,836 | 391.347 ms |
| Marlin W4A16 | 7,627 | 365.647 ms |
| GDN prefill | 96 | 253.884 ms |
| Other kernels | 11,082 | 235.756 ms |
| Qwen W4 projections | 1,205 | 144.694 ms |
| FlashAttention decode | 401 | 57.400 ms |
| RMSNorm | 3,494 | 31.638 ms |
| GDN decode | 1,205 | 20.759 ms |
| LM-head GEMV | 2,436 | 10.819 ms |

The final no-environment ApxInf profile has the required stage markers. In its decode-only interval:

| Stage | Instances | Range elapsed |
|---|---:|---:|
| `Qwen/GDN` | 400 | 83.084 ms |
| `Qwen/attention` | 400 | 28.651 ms |
| `Qwen/MLP` | 448 | 11.871 ms |
| `Qwen/LM head` | 25 | 0.237 ms |

Decode-only memory/allocation facts from the final default profile:

- H2D: **0 bytes**, 0 operations;
- `cudaMalloc`: **0**;
- device-to-device memset: **0**;
- graph-node kernels: 15,600 over 25 intervals, 624/token;
- kernel launches: 21,700 over 25 intervals, 868/token;
- summed GPU kernel time: 25.330 ms/token;
- official canonical TPOT: 25.592 ms/token.

### vLLM

| Module | Kernel instances | GPU time |
|---|---:|---:|
| FlashAttention prefill/decode | 27 | 4,324.054 ms |
| Marlin W4A16 | 18,141 | 2,017.584 ms |
| Other kernels | 5,114 | 433.790 ms |
| Fused Triton/elementwise | 28,353 | 217.136 ms |
| LM-head GEMV | 3,307 | 187.319 ms |
| RMSNorm | 9,049 | 30.599 ms |
| GDN | 3,129 | 23.182 ms |
| Conv1D | 3,081 | 6.675 ms |

### Per-module decode-normalized comparison

The following table re-buckets the kernel symbols inside each backend's argmax-delimited profiled interval. Values are **summed GPU kernel time divided by the number of intervals in that trace**: 25 ApxInf intervals and 67 vLLM intervals. They are module-level profiler values, not replacements for the client-observed TPOT, because the vLLM trace includes graph warmup/capture activity and the two traces have different interval counts.

| Normalized module | ApxInf ms/token | ApxInf share | vLLM ms/token | vLLM share | Difference / interpretation |
|---|---:|---:|---:|---:|---|
| W4 Marlin | **14.543** | **57.50%** | **20.950** | **82.03%** | Both are dominated by Marlin; vLLM routes more W4 work through Marlin, while ApxInf splits some projections into its custom raw path. |
| Custom/raw W4 projections | **5.764** | **22.79%** | 0 | 0 | ApxInf-only bucket: raw QKV-alt and related custom W4 kernels remain a separate projection schedule. |
| Combined quantized W4 projection work | **20.307** | **80.29%** | **20.950** | **82.03%** | Primary ApxInf bottleneck. Eliminating host overhead cannot remove this GPU weight sweep. |
| Attention | **2.290** | **9.05%** | **0.195** | **0.76%** | ApxInf decode attention is costlier in this bucket; fused FlashAttention is already enabled, but the trace bucket boundaries differ. |
| RMSNorm | **1.148** | **4.54%** | **0.333** | **1.30%** | ApxInf has more standalone norm work; vLLM folds more norm/epilogue work into Triton graphs. |
| GDN / recurrent path | **0.827** | **3.27%** | **0.452** | **1.77%** | Secondary. ApxInf packed GDN is not the limiting stage. |
| LM-head GEMV | **0.429** | **1.69%** | **2.754** | **10.79%** | ApxInf's exact W4 head is much smaller than vLLM's dense-head GEMV in this bucket. |
| Fused Triton/elementwise | **0.201** | **0.79%** | **0.591** | **2.31%** | vLLM absorbs more post-projection work into fused Triton kernels. |
| KV cache | **0.051** | **0.20%** | **0.046** | **0.18%** | Equivalent secondary cost. |
| Argmax/sampling | **0.006** | **0.02%** | **0.007** | **0.03%** | Negligible in both engines. |
| Other | **0.036** | **0.14%** | **0.144** | **0.57%** | Small residual bucket. |

### Module conclusion

The side-by-side view changes the optimization priority:

1. **Quantized W4 projection execution is the shared dominant cost**: 80.29% of ApxInf's normalized profiled GPU time versus 82.03% for vLLM.
2. **ApxInf's distinctive penalty is split projection ownership**: 22.79% is in custom/raw W4 projection kernels outside Marlin. The next high-value work is to make those projections Marlin-compatible without changing BF16 rounding or trajectory order.
3. **ApxInf attention is secondary but visibly larger** in the normalized bucket (2.290 vs 0.195 ms/token). This is the next module to inspect after W4 projection layout; profile interval composition makes this comparison directional rather than a strict kernel-equivalent ratio.
4. **LM head is no longer an ApxInf bottleneck** after the exact W4 cutover: 0.429 vs vLLM's 2.754 normalized ms/token.
5. **GDN, KV cache, argmax, and host transfers are not bottlenecks**. ApxInf's isolated decode interval has zero H2D and zero allocation events.

The vLLM trace contains extensive startup and graph-capture work. Its dominant named kernels are:

- FlashAttention forward: 4.324 s aggregate across 27 instances;
- Marlin M=1: 1.086 s aggregate across 17,376 instances;
- Marlin M=4: 0.932 s aggregate across 765 instances;
- LM-head GEMV: 169.9 ms across 64 instances;
- packed recurrent GDN: 20.4 ms across 3,032 instances.

A direct trace interval around the FlashAttention kernel family measured 4.570 s GPU time across the captured mixed workload; it contains the two requested cells and startup/capture-adjacent activity, so the client TPOT remains the fair latency metric.

## Flamegraph / folded-stack artifact

Saved folded module stacks:

- `comparison_flamegraph_29.folded`

Contents are generated from Nsight kernel aggregates and can be rendered with Brendan Gregg’s `flamegraph.pl` if installed:

```bash
flamegraph.pl comparison_flamegraph_29.folded > comparison_flamegraph_29.svg
```

The folded stack is module-level because Nsight provides CUDA kernel symbols, not CPU call-stack samples for the GPU critical path. It preserves the evidence needed to compare module contribution without inventing a Python flamegraph for a CUDA-bound workload.

## Bottleneck diagnosis: ApxInf

### Primary bottleneck: quantized projection weight sweep

ApxInf’s dominant GPU families are Marlin W4A16, raw QKV-alt W4, and CUTLASS BF16 projection GEMMs. In the final default trace, Marlin alone is **356.6 ms** in the partial capture and the raw QKV-alt family is **144.4 ms**. The decode-only interval has no H2D, no allocation, and no memset work, so host/runtime optimization cannot close the remaining gap.

This matches the client comparison:

- ApxInf: **25.834 ms/token** profiled, **38.71 tok/s**;
- vLLM: **20.284 ms/token** profiled, **49.30 tok/s**;
- ApxInf is approximately **27.4% slower in TPOT** for this single 1K request.

The important difference is not “CUDA graph versus no graph”: both use graph paths. vLLM’s Marlin path and fused Triton/compiled epilogues execute the same quantized projections with fewer separately materialized intermediate operations and a more aggressively fused runtime. ApxInf has already eliminated transfer/allocation overhead; its remaining deficit is in the GPU projection schedule and exact BF16 intermediate boundaries.

### Secondary bottlenecks

1. ApxInf’s CUTLASS BF16 GEMM family remains a large prefill/projection cost.
2. The 16 full-attention layers produce context-dependent TPOT growth; fused FlashAttention-256 reduces this component but does not remove it.
3. GDN decode is relatively small: about 20.8 ms aggregate in the full trace, versus hundreds of milliseconds from projection families.
4. ApxInf’s exact dense-to-W4 LM-head cutover reduced VRAM by approximately 1.77 GiB and improved TPOT, but LM-head GEMV is not the dominant remaining cost.

### Not the bottleneck

- Decode H2D: zero in the isolated ApxInf interval.
- Decode `cudaMalloc/cudaFree`: zero.
- Token selection: approximately 5.5 microseconds per argmax kernel.
- NVTX marker overhead: static names, no hot-path allocation.

## SGLang result

SGLang could not serve this checkpoint from the supplied venv:

1. Initial attempt stopped on the explicit torch/CuDNN guard; CuDNN was 9.10 while the guard requires >=9.15 for torch 2.9.1.
2. With `SGLANG_DISABLE_CUDNN_CHECK=1`, startup reached model construction and failed with `NotImplementedError: No compressed-tensors compatible scheme was found` at `linear_attn.in_proj_qkv`.
3. An `awq_marlin` override was rejected because the checkpoint config declares `compressed-tensors`.
4. The SGLang Nsight artifacts contain no CUDA kernel data for a valid inference request.

SGLang is therefore **startup-incompatible with this asymmetric compressed-tensors checkpoint in the installed 0.5.9 environment**. A no-quantization or converted-checkpoint run would not be the requested same-checkpoint comparison and was not substituted.

## Artifacts and hashes

| Artifact | SHA-256 |
|---|---|
| `/tmp/cmp-apxinf.nsys-rep` | `c7e6b0d8dfef159c2bb12aa7260324ed5a7802314393dd96d6b7c024ba53d9` |
| `/tmp/cmp-vllm.nsys-rep` | `f93852f7d3caad7495980a54ca92ddb9019b1d26c77691c1006a07c2ed9defc3` |
| `/tmp/compare-sglang.nsys-rep` | `1af9e1721bf49901b60e857037c13b43bb4756700d3a762c06a81f8985d10854` |
| `/tmp/compare-sglang-awq.nsys-rep` | `39df34eceb8ca13c550c3aa4dc5c30d17af3efd80f064df9a9f8591d23422908` |
| `comparison_flamegraph_29.folded` | generated module-level folded stacks |

Comparison limitations:

- The vLLM profile contains startup, CUDA-graph capture, and two sequential workload requests; the ApxInf profile contains one 1K request. Client TPOT values are the fair direct comparison.
- SGLang has no valid inference sample because the installed runtime cannot load this asymmetric compressed-tensors model.
- Nsight Compute hardware counters were not available; no DRAM bandwidth or occupancy numbers are claimed.
- The official ApxInf run and comparison report use the exact pinned public workload; hidden cases and private leaderboard repeats are unavailable locally.
