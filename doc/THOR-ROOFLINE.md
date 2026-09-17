# Qwen-Drive-1.0-4B on Jetson AGX Thor (sm_110) — measured roofline

Written before any optimisation, so every later claim has a bar to clear.
Device: thor-3, NVIDIA Thor sm_110, 20 SMs @ 1.05 GHz, 256-bit LPDDR5X,
32 MB L2, 228 KB shared/SM, JetPack R39.2.1, CUDA 13.2.86, driver 595.78.

## Sustained device-memory bandwidth (measured, not spec)

4 GB buffers so the 32 MB L2 cannot serve the stream; grid-stride float4,
blocks = 16 x SM count, 5 replays timed with CUDA events.

| device | read | copy (r+w) | spec |
|---|---:|---:|---:|
| Thor sm_110 | **259.7 GB/s** | 236.1 GB/s | 273 GB/s |
| Orin sm_87 (orin2) | 154.4 GB/s | 176.9 GB/s | 204.8 GB/s |

Orin's *measured* read is 154.4 GB/s, not the 204.8 GB/s bus figure. PR72's
decode at 152 GB/s is therefore **98.4% of the practical roofline**, not 74%.
There is no BF16 decode headroom left on Orin, and the same will be true here
once the port is tuned.

## Bytes the decode must move, per token

From the checkpoint's own tensor table (723 tensors, 9.079 GB on disk), text
model only — the 667 MB vision tower runs once per scene, not per token.

| block | MB/token | share |
|---|---:|---:|
| MLP gate/up/down, 32 layers | 4529.7 | 53.9% |
| GDN linear_attn in_proj_qkv + in_proj_z + out_proj, 24 layers | 2013.2 | 23.9% |
| lm_head / embed_tokens (tied, vocab 248320 x 2560) | 1271.4 | 15.1% |
| full self_attn q/k/v/o, 8 layers | 587.1 | 7.0% |
| norms, conv1d, A_log, dt_bias | ~10 | 0.1% |
| **total** | **8411** | |

Matches PR72's 8.41 GB independently.

Architecture note: 24 of 32 text layers are GDN linear attention (O(1) decode
state) and only 8 are full attention with a KV cache, so at 64 generated
tokens the KV traffic is negligible and decode is pure weight streaming.
`mtp_num_hidden_layers: 1` appears in config.json but **no `mtp.*` tensor
exists in the checkpoint**, so there is no free self-speculative draft head.

## Decode floor

| precision | bytes/token | floor at 259.7 GB/s | 64 tokens |
|---|---:|---:|---:|
| BF16 (today) | 8.41 GB | 32.4 ms | 2.07 s |
| lossless-packed BF16 (see below) | 5.55 GB | 21.4 ms | 1.37 s |
| FP8 weights | ~4.3 GB | 16.6 ms | 1.06 s |
| NVFP4 weights | ~2.1 GB | 8.1 ms | 0.52 s |

Orin PR72 measured 55.2 ms/token; its floor was 8.41/154.4 = 54.5 ms.

## What a 10x target implies

PR72's scene is 6.0143 s = 2.49 s fixed (vision+prefill) + 64 x 55.2 ms decode.
10x is 0.60 s per scene. Decode alone in that budget needs 8.41 GB in 0.35 s =
**1529 GB/s, 5.9x the measured bandwidth**. Even at NVFP4 decode is 0.52 s and
has already spent the whole budget. 10x end-to-end is not reachable by any
kernel work on this model and this device; it needs fewer bytes *and* fewer
weight sweeps (speculation), and even then lands near 6x.

## Lossless byte reduction — measured, and it is real

Constraint from the owner: a format change must not cost precision. BF16
weights of this checkpoint are compressible **exactly**, because the exponent
field is low-entropy while sign+mantissa is not:

| tensor | exponent entropy | sign+mantissa entropy | total | lossless ratio |
|---|---:|---:|---:|---:|
| layers.5.mlp.down_proj | 2.584 b | 7.972 b | 10.556 b | 1.516x |
| layers.5.mlp.gate_proj | 2.567 b | 7.972 b | 10.539 b | 1.518x |
| layers.5.linear_attn.in_proj_qkv | 2.574 b | 7.972 b | 10.546 b | 1.517x |
| layers.3.self_attn.q_proj | 2.634 b | 7.972 b | 10.606 b | 1.509x |
| embed_tokens | 2.582 b | 7.972 b | 10.554 b | 1.516x |

So ~1.52x is available with **bit-identical weights** — the gate cannot
distinguish it, because the reconstruction is the same 16 bits.

The naive fixed-width version of this does not work: per-block exponent
*range* is 9-11 at the median and 15-17 at p99 for blocks of 64-256, because
the distribution is concentrated but long-tailed. A fixed 3-bit block offset
covers only 45-49% of 64-weight blocks. The scheme has to be a short code on
the frequent exponents with an escape, sized so blocks stay independently
addressable. Reconstruction must sustain ~400 GB/s of output to be worth it.

## Consequence for megakernel / persistent runtime

Orin's decode is at 98.4% MBU, which says launch gaps and kernel boundaries
are **not** the bottleneck in BF16. A persistent megakernel earns its
complexity only after the bytes come down, when latency and launch overhead
become a visible share again. Order: bytes first, megakernel second.

---

# Measured baseline, and where Thor's advantage goes missing

## PR72 as ported, unmodified, on thor-3

Fixed four-scene 64-token VQA workload, warmup 1 + 3 measured, warm median.
Split measured by running the same scene at max_new_tokens 1 and 64 on one
policy instance, so no instrumentation and no phase attribution is assumed.

| | Orin PR72 | Thor PR72 | ratio |
|---|---:|---:|---:|
| per scene | 6.0143 s | **5.1395 s** | 1.17x |
| fixed cost (vision + prefill) | 2.49 s | **2.421 s** | 1.03x |
| decode | 55.2 ms/token | **42.48 ms/token** | 1.30x |

Decode is 42.48 ms against a 32.4 ms byte floor: **76.3% MBU**, so ~24% is
there for the taking. The fixed cost did not move at all, and that is the
whole story of the missing speedup.

## Unit throughput, measured on both boards

| unit | Thor | Orin | ratio |
|---|---:|---:|---:|
| BF16 tensor core, prefill shapes | 164-195 TFLOP/s | 15.4-16.0 TFLOP/s | **~11x** |
| DRAM read, sustained | 259.7 GB/s | 154.4 GB/s | 1.68x |
| **fp32 IEEE on CUDA cores (SIMT)** | **5.43 TFLOP/s** | **3.42 TFLOP/s** | **1.59x** |

Thor is 11x Orin on tensor cores and 1.59x on fp32 CUDA cores. The prefill
runs on the second row.

## Kernel table, one scene, nsys, 4.94 s of GPU in a 5.14 s wall (96% busy)

Everything below is fp32 on CUDA cores, or feeds something that is:

| kernel | ms | instances |
|---|---:|---:|
| `gdn_chunk_state_kernel<8>` | 525.0 | 24 (one per GDN layer, 21.9 ms each) |
| `cutlass_80_simt_sgemm_128x64_8x5_tn` | 370.3 | 4608 |
| `row_softmax_f32_bf16_kernel` | 260.1 | 288 |
| `gdn_recurrent_kernel` | 118.9 | 1512 |
| `gdn_chunk_gemm_kernel<16>` | 117.2 | 24 |
| `cast_bf16_to_f32_kernel` | 116.1 | 1584 |
| `cutlass_80_simt_sgemm_256x128_8x4_nn` | 97.6 | 384 |
| `cutlass_80_simt_sgemm_128x128_8x4_tn` | 89.7 | 256 |
| `gdn_attn_raw_kernel` | 72.2 | 24 |
| `gdn_block_inverse64_kernel` | 9.3 | 24 |

≈1.42 s of the 2.42 s fixed cost. `gdn_chunk_state` alone is 10.7% of the
scene and is **unchanged from Orin** — PR72 took it from 799 ms to 555 ms
there, and Thor runs it at 525 ms, because its tile constants were chosen for
sm_87 and it never touches a tensor core.

The decode side is healthy by comparison: `nvjet_sm110_tst_64x8_..._splitK_NNT`
moves 90 MB in 334 us = 269 GB/s, at the roofline.

## Is there a precision-safe way to put fp32 work on the tensor cores?

cuBLAS on this box, 4096^3, accuracy against an fp64 device reference:

| mode | TFLOP/s | max rel err | mean rel err |
|---|---:|---:|---:|
| fp32 IEEE (SIMT) | 5.43 | 9.7e-3 | 5.8e-6 |
| fp32 emulated BF16x9 | **1.86** | 3.2e-3 | **2.7e-6** |
| fp32 "fast" TF32 | 18.61 | **2.656** | 2.3e-3 |
| fp32 "fast" BF16 | 18.60 | **2.656** | 2.3e-3 |

BF16x9 is the accurate one — mean error *below* IEEE fp32's, as designed — but
it runs at 1.86 TFLOP/s, slower than SIMT, and `CUBLAS_EMULATION_STRATEGY`
(default/performant/eager) does not change it. The two fast modes are the
truncation PR72 already rejected on accuracy; max rel err 2.66 confirms why.

So cuBLAS offers no drop-in precision-safe fp32 acceleration here. The GDN
kernels have to be restructured — split the math so the parts that are already
on the BF16 grid use tensor cores and the parts that need fp32 keep it — not
merely re-dispatched to a different compute type.

## Also found

All **116** Thor tuning records are rejected at load: recorded under CUDA/cuBLAS
13.0, this box is 13.2. Every GEMM in the baseline is running the provider's
safe default, so the tuning system contributed nothing to the number above.
