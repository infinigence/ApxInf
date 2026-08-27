# vLLM versus ApxInf profiler comparison

Date: 2026-08-24  
Model: `../model/qwen`  
Hardware: RTX 4090; vLLM profiler run on visible GPU 2

## Conclusion

The bottleneck is the quantized linear layer, specifically Marlin/W4 projection work in vLLM and the corresponding custom W4 projection path in ApxInf. vLLM's PyTorch profiler recorded **311.415 ms** in Marlin kernels during the profiled 1K request context and **63.961 ms** in the decode Marlin kernel family. The GDN attention core was **78.184 ms CPU-total / 11.363 ms CUDA-total** across 48 calls. Full attention was **2.744 ms** across 16 FlashInfer prefill kernels. The vLLM profiler therefore agrees with the earlier ApxInf Nsight result: optimize W4 linear projections first; GDN and attention are secondary for short decode.

## vLLM profiler setup

The vLLM process used the installed virtual environment:

```bash
/mnt/chuangxin/team3/work/model/vllm/.venv/bin/vllm serve ../model/qwen \
  --served-model-name qwen-profile \
  --max-model-len 16512 \
  --gpu-memory-utilization 0.98 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --max-num-seqs 1 \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 --port 8017 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/mnt/chuangxin/team3/work/ApxInf/target/vllm-profiler","torch_profiler_with_stack":false,"torch_profiler_use_gzip":false,"delay_iterations":1,"max_iterations":8,"warmup_iterations":1,"active_iterations":5,"wait_iterations":0}'
```

Profiling was controlled through:

```bash
curl -X POST http://127.0.0.1:8017/start_profile
curl -X POST http://127.0.0.1:8017/stop_profile
```

Artifacts:

- `target/vllm-profiler/rank0.1787579846432427651.pt.trace.json`
- `target/vllm-profiler/profiler_out_0.txt`
- `target/vllm-profiler/is-ddg3cxgfesglewqb-devmachine-0_400962.async_llm.1787579848171261370.pt.trace.json`

The trace contained 27,788 events and 10.7 MB of JSON. The async frontend trace contained only seven events; the rank-0 worker trace contains the useful CUDA/module data.

## vLLM module summary

From `profiler_out_0.txt`:

| Module/kernel | CUDA time | Calls | Interpretation |
|---|---:|---:|---|
| Marlin W4 kernel, prefill/context family | 311.415 ms | 255 | Dominant quantized projection workload in the profiled context |
| Marlin W4 kernel, decode family | 63.961 ms | 1,020 | Dominant per-token decode projection workload |
| `vllm::qwen_gdn_attention_core` | 11.363 ms CUDA; 78.184 ms CPU-total | 48 | GDN core; not the short-decode bottleneck |
| BF16 `aten::mm` | 14.728 ms | 54 | Dense projections / output operations |
| BF16 cuBLAS GEMV family | 13.270 ms | 5 | Dense vector projections / LM-head-like work |
| FlashInfer paged full attention | 2.744 ms | 16 | Full-attention layers for the profiled context |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 2.196 ms | 48 | GDN chunk recurrence |
| `chunk_fwd_kernel_o` | 1.989 ms | 48 | GDN output stage |
| Packed recurrent GDN decode kernel | 1.366 ms | 192 | Decode recurrent update |
| Causal convolution update | 0.352 ms | 192 | GDN convolution update |
| Fused post-convolution kernel | 0.750 ms | 48 | GDN post-convolution preparation |
| `aten::argmax` | 32.960 us | 5 | Sampling; negligible |

The trace also reports CUDA graph launch activity, fused Triton RMSNorm/add/Silu operations, and 398 `aten::to` operations. These are secondary to Marlin in GPU time.

## ApxInf comparison

Latest valid ApxInf iteration-14 Nsight node trace:

- W4 projection kernels: **46.31 ms/token**, **85.61%** of graph GPU time.
- Dense LM head: **2.653 ms/token**.
- Full attention: **2.413 ms/token at 1K**.
- GDN recurrence plus convolution: approximately **0.72 ms/token**.
- Control, graph launch, argmax, synchronization, and service envelope: at most **0.40 ms/token**.

ApxInf 1K decode was **18.35 tok/s / 54.49 ms TPOT**. The one-GPU vLLM control was **49.45 tok/s / 20.22 ms TPOT**. The measured decode ratio is **0.371×**.

### Bottleneck ranking

1. **W4 projection implementation/layout** — dominant in both systems. vLLM uses Marlin's load-time-repacked layout; ApxInf uses a custom one-row W4 BF16 MMA path.
2. **ApxInf dense LM head** — 2.653 ms/token, material but far smaller than the W4 gap.
3. **Full attention at long context** — FlashInfer is efficient in vLLM; ApxInf's context-dependent full-attention scan grows sharply with prefix length.
4. **Prefill W4 handling** — ApxInf materializes dequantized BF16 weights per projection/chunk; vLLM performs packed Marlin operations directly.
5. **GDN recurrence** — ApxInf is slower structurally than vLLM's packed recurrent path, but the measured short-decode contribution is too small to explain the global gap.
6. **Sampling/service** — negligible for the one-request throughput comparison.

## Nsight Systems and nvprof

A vLLM Nsight Systems launch was attempted. The capture artifact exists:

- `target/vllm-nsys/vllm.nsys-rep` — 3,050,186 bytes
- `target/vllm-nsys/vllm.sqlite` — 53,628,928 bytes

This capture contains startup/API activity but no usable CUDA kernel or GPU memory tables because the profiled server was interrupted during model import/startup. `nsys stats` explicitly reported that the SQLite file did not contain CUDA kernel or GPU memory data. It must not be used as a kernel-time measurement.

`nvprof` was also probed directly on an RTX 4090:

```text
Warning: This version of nvprof doesn't support the underlying device, GPU profiling skipped
Warning: nvprof is not supported on devices with compute capability 8.0 and higher.
Use NVIDIA Nsight Systems for GPU tracing and NVIDIA Nsight Compute for GPU profiling.
```

The exact warning is retained in `target/vllm-nvprof.txt`. Nsight Systems and PyTorch profiler are the valid evidence sources on this GPU; nvprof cannot provide kernel metrics on sm_89.

## Recommended optimization order

1. Port or reproduce Marlin's repacked W4 layout and asymmetric group-32 metadata handling in ApxInf.
2. Fuse QKV/QKVZ and gate/up projection schedules around the Marlin-compatible kernel rather than relying only on CUDA Graph replay.
3. Remove ApxInf's repeated full-matrix prefill dequantization; use in-kernel fragment dequantization for M=128 and larger.
4. Match vLLM's fused GDN preparation and packed recurrent decode only after W4 work is addressed.
5. Match FlashInfer's paged full-attention strategy for long-context requests.
6. Leave sampling and HTTP optimization for last; their measured contribution is below one percent of ApxInf TPOT.

No profiler claim is made from the failed Nsight capture. The module conclusion is grounded in the vLLM PyTorch trace and the valid ApxInf iteration-14 Nsight trace.
