# Qwen-Drive on Jetson AGX Thor — what landed, what did not, what is blocked

Companion to THOR-ROOFLINE.md, which holds the measured device limits this is
all scored against. Branch `codex/qwen-drive-thor-sm110`, from PR #72.

## Where it ended

Fixed four-scene 64-token VQA workload, warm median, `control/verify_perf_orin.py`:

| | per scene | fixed cost | decode |
|---|---:|---:|---:|
| Orin, PR72 | 6.0143 s | 2.49 s | 55.2 ms/token |
| Thor, PR72 as ported | 5.1395 s | 2.421 s | 42.48 ms/token |
| Thor, this branch | **3.833 s** | **1.431 s** | **37.5 ms/token** |

Official geomean against the Thor baseline: **1.3306x** (1.326 / 1.337 / 1.328 /
1.332 across the four scenes). Against Orin's PR72: 1.57x.

The fixed cost carried the work: 2.421 -> 1.431 s, 41% off. Decode moved 12%
and cannot move much further — see below.

## What landed

| change | effect | why it was there to take |
|---|---|---|
| build fix for `mha_f16`'s CUTLASS FMHA branch | builds at all | the branch had never been compiled: every prior device is outside the SM100 family |
| head-256 FA2 on the SM100 family | 5.1395 -> 4.6871 s | `is_fa2_sm80_family` gated the dispatch adapters, not the kernels, which were already compiled |
| GDN chunk tiles chosen from the device | -> 4.6157 s | Orin's 8/16 are not Thor's 4/4; the chunk-state curve steepens, 16 goes from 8% worse to 26% worse |
| prefill GEMM tactics re-recorded for CUDA 13.2 | -> 4.6133 s | all 116 shipped records were rejected at load; the decode half of the new ones was rejected on accuracy |
| vision head-64 attention onto FA2 | -> 3.9741 s | same gate as the head-256 case; the fallback was 628.6 ms of fp32 on CUDA cores |
| GDN decode recurrence split across threads | -> 3.8877 s | 13% occupancy and the state touched four times where twice is the work |
| chunk-state block width 512 -> 1024 | -> 3.8600 s | Thor has 20 SMs and 228 KB of shared memory against Orin's 16 and 164 KB |
| vec8 elementwise half of the Orin WIP | -> 3.833 s | prefill-sized instances; bit-identical |

## What did not, and the number that stopped it

**Autotuned decode GEMM tactics.** Worth 2.5%, all of it in the decode half,
which drops VQA token agreement 27% and raises reasoning trajectory RMS 14%.
The prefill half is accuracy-neutral and worth 0.05%. Shipped prefill only.

**Bit-exact packed weights (13.0625 bits against 16).** The format is sound and
verified: zero blocks over a five-bit exponent range, zero reconstruction
mismatches, and a refusal guard that caught the one vision projection that does
not fit. The GEMV over it wins above n=36864 (1.28-1.61x) and loses 2.7x at and
below n=18432, which is where decode lives; in the engine it costs
38.16 -> 54.36 ms/token. Ablation rules out the extra planes (the lo plane
alone, half of BF16's bytes, is already slower), register pressure (40 against
42, no spills) and trip count (sixteen weights per lane measures the same).
Cause unidentified. Committed default-off behind `APXINF_PACKED_DECODE`.

**Tiling the GDN chunk-state inter-chunk term.** The intra term already hoists
its shared read out of the cell loop; the inter term does not, and hoisting it
the same way removes 768 of about 4100 memory operations per chunk. It is 10%
slower: fixed cost 1.6059 s against 1.4606. No spills, no occupancy change, so
the shared reads were not the limiter. Reverted.

**The BF16 shared tiles from the Orin WIP.** Worth taking on Orin, where an SM
holds 164 KB and the float tiles allowed two blocks. Thor holds 228 KB and
already fits them, so the change only adds conversions: 1.4904 s against
1.4596 for neither.

**CUDA-graph decode.** 39.474 against 39.231 ms/token. The same non-result as
on Orin, and for the same reason: the allocation cache already removed what
graphs remove.

## Why decode is nearly finished, and what that says about a megakernel

Kernel table split at the prefill/decode boundary, one scene:

    PREFILL  1184.8 ms over 1061 launches
      gdn_chunk_state<4>   452.3 ms   38.2% of prefill, 12.5% of the scene
      gdn_chunk_gemm<4>    107.7 ms
      gdn_attn_raw          72.2 ms
      (GDN scan is 55% of prefill)

    DECODE   2428.1 ms over 35181 launches
      four cuBLAS GEMV families  2189.4 ms   90.2%
      splitKreduce                 79.8 ms    3.3%
      everything else             159.0 ms    6.5%

Decode is 90% weight sweep, and that sweep moves 8.41 GB per token at about
250 GB/s against a measured 259.7 GB/s roofline. A perfect megakernel or
persistent runtime — fusing every norm, activation, conv and gate into the
GEMV epilogues and never launching them separately — has 159 ms to win, 4.1%
of the scene. The lever decode actually wants is fewer bytes, which is why the
packed form was tried and why its failure matters.

The remaining prefill target is the GDN scan, 650 ms of fp32 on CUDA cores
where Thor is only 1.59x Orin while its tensor cores are 11x. Its inner loops
are GEMM-shaped and one operand (the carried state, and `v_new`) is already on
the BF16 grid, so a tensor-core form with the fp32 operand split into two or
three BF16 terms would be near-exact. That is the next thing worth building.

## What is blocking the next round

`ncu` cannot run on this box: `ERR_NVGPUCTRPERM`. Two of the results above are
"cause unidentified" for exactly that reason. Enabling it is one root-level
driver setting (`NVreg_RestrictProfilingToAdminUsers=0`) and would turn both
into answerable questions.

## How things were measured

- One measurement at a time, through `gpulock`. A perf run that shared the GPU
  with a sweep reported 9.24 s for a 4.70 s scene and dragged the geomean from
  1.09 to 0.92.
- `control/precision_probe.py` for accuracy, not the four-mode gate. The gate
  reports scene 0's maximum trajectory error, which on this checkpoint only
  takes the two adjacent BF16 output ULPs 0.0403 and 0.0806, so it flips on
  changes that move nothing.
- fp64 operator oracles where a route had to be chosen:
  `vision_segmented_mha_error_against_fp64_oracle` and
  `gdn_recurrent_decode_error_against_fp64_oracle`. The first overturned the
  gate's verdict — FA2 is the most accurate of the three vision routes, not the
  least — and the second needed an input with a 2^16 magnitude ramp along the
  reduction axis before the regrouping it was testing could show at all.
- Microbenchmarks one variant per process, on buffers past the 32 MB L2. In one
  process the same five variants came out non-monotonic and reordered between
  builds.
