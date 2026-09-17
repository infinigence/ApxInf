# Qwen-Drive on Jetson AGX Thor — what landed, what did not, what is blocked

Companion to THOR-ROOFLINE.md, which holds the measured device limits this is
scored against. Branch `codex/qwen-drive-thor-sm110`, from PR #72.

## Where it ended

Fixed four-scene 64-token VQA workload, warm median, `control/verify_perf_orin.py`:

| | per scene | fixed cost | decode |
|---|---:|---:|---:|
| Orin, PR72 | 6.0143 s | 2.49 s | 55.2 ms/token |
| Thor, PR72 as ported | 5.1395 s | 2.421 s | 42.48 ms/token |
| Thor, this branch | **3.594 s** | **1.152 s** | **37.9 ms/token** |

Official geomean against the Thor baseline: **1.4296x** (1.424 / 1.435 / 1.427 /
1.433 across the four scenes). Against Orin's PR72: 1.67x.

The fixed cost carried it: 2.421 -> 1.152 s, 52% off. Decode moved 11% and
cannot move much further; the reason is below.

## What landed

| change | per scene | why it was there to take |
|---|---:|---|
| build fix for `mha_f16`'s CUTLASS FMHA branch | builds at all | the branch had never been compiled on any device |
| head-256 FA2 on the SM100 family | 5.1395 -> 4.6871 | the gate was on the dispatch adapters, not the kernels, which were already compiled |
| GDN chunk tiles chosen from the device | -> 4.6157 | Orin's 8/16 are not Thor's 4/4 |
| prefill GEMM tactics re-recorded for CUDA 13.2 | -> 4.6133 | all 116 shipped records were rejected at load |
| vision head-64 attention onto FA2 | -> 3.9741 | the fallback was 628.6 ms of fp32 on CUDA cores |
| GDN decode recurrence split across threads | -> 3.8877 | 13% occupancy, state touched four times where twice is the work |
| chunk-state block width 512 -> 1024 | -> 3.8600 | 20 SMs and 228 KB of shared against Orin's 16 and 164 |
| vec8 elementwise half of the Orin WIP | -> 3.833 | prefill-sized instances; bit-identical |
| **GDN chunk-state scan on tensor cores** | **-> 3.594** | 452 ms of GEMM-shaped work in scalar fp32 |

The last one is the largest single change in the campaign and the one that
needed the most care, so it is worth stating on its own. The scan's four inner
products all have a right-hand operand that is already on the BF16 grid -- the
kernel rounds the carried state itself, and v_new before both the intra term
and the state update. Rounding the *left* operand to BF16 too looked free for
the same reason and is not: against an fp64 reference of the scan it costs
3.5x, 1.418808e-3 to 4.954247e-3. Carrying the left operand as two BF16 terms
instead, x = hi + lo, makes every partial product exact and lands at
1.422947e-3 -- 0.3% -- while still taking the fixed cost from 1.4623 to
1.1909 s. End to end it is also more accurate on every metric that moves:
VQA token agreement 0.32498 -> 0.608759, direct trajectory mean/rms/p99
0.013483/0.036036/0.161133 -> 0.011252/0.027358/0.097656.

## What did not, and the number that stopped it

**Autotuned decode GEMM tactics.** 2.5%, all of it in the decode half, which
drops VQA token agreement 27% and raises reasoning trajectory RMS 14%. Prefill
half shipped; it is worth 0.05%.

**Bit-exact packed weights, 13.0625 bits against 16.** The format is sound:
zero blocks over a five-bit exponent range on real tensors, zero reconstruction
mismatches, and a refusal guard that caught the one vision projection that does
not fit. The GEMV over it wins above n=36864 and loses 2.7x at and below
n=18432, which is where decode lives. A ladder that adds one piece at a time
(`probes/steps.cu`, one variant per process) puts the whole difference on the
*number of load instructions*, not the bytes:

    0 stream (lo only, 22.5 MB)  148.3 us      5 BF16 reference (45 MB)  217 us
    1 + the x vector             237.0 us
    2 + the reconstruction       234.5 us
    3 + the multiply             282.0 us
    4 + the three exponent planes 608  us

The byte stream is faster than BF16; the reconstruction is free; the three
extra streams cost +326 us. A single-stream layout would not have the
crossover at all, and 13.0625 bits has no naturally aligned single-stream
packing -- that is the open problem. Left in, default off, with the width gate
set to where it wins; enabling it for the embedding matrix alone measures
within noise end to end (37.90/37.90 against 37.93/38.05 ms/token).

**Tensor-core chunk GEMM.** Built and works, 1.9% of the scene, off by
default. This kernel rounds both outputs to BF16 so the scalar form is nearly
exact -- 1.369621e-7 against fp64 -- and the split form is 2.959516e-6, 21.6x
more. Both are far below the 3.9e-3 grid the results are written on and 470x
below the error the scan that consumes them already carries, but the
end-to-end probe moves with it, and 1.9% is not enough to spend that on
without being asked. `APXINF_GDN_CHUNK_GEMM_WMMA=1`.

**Tiling the chunk-state inter-chunk term.** Removes 768 of about 4100 memory
operations per chunk and is 10% slower: 1.6059 s against 1.4606. No spills, no
occupancy change. Reverted; superseded by the tensor-core form.

**The BF16 shared tiles from the Orin WIP.** Worth taking on Orin, where an SM
holds 164 KB; Thor holds 228 and already fits the float tiles, so the change
only adds conversions -- 1.4904 s against 1.4596 for neither.

**CUDA-graph decode.** 39.474 against 39.231 ms/token, the same non-result as
on Orin and for the same reason.

## Why decode is nearly finished, and what that says about a megakernel

Kernel table split at the prefill/decode boundary, one scene:

    PREFILL  918.6 ms over 1061 launches
      gdn_chunk_state_wmma<split>  187.8 ms   20.5% of prefill  (was 452.3)
      gdn_chunk_gemm<4>            107.9 ms   11.7%
      gdn_attn_raw                  72.2 ms    7.9%

    DECODE   ~2440 ms over 35181 launches
      four cuBLAS GEMV families    ~2189 ms   90.2%
      splitKreduce                   79.8 ms   3.3%
      everything else               159.0 ms   6.5%

Decode is 90% weight sweep, 8.41 GB per token at about 250 GB/s against a
measured 259.7 GB/s roofline. A perfect megakernel or persistent runtime --
every norm, activation, conv and gate fused into the GEMV epilogues, nothing
launched separately -- has 159 ms to win, 4.1% of the scene. What decode wants
is fewer bytes, which is why the packed form was built and why its load-count
problem is the one worth solving next.

`gdn_attn_raw` is the remaining prefill target, 2% of the scene. Unlike the two
kernels already moved, *both* its operands are fp32, so a tensor-core form
needs three or four passes rather than two, and one of its outputs feeds a
triangular solve.

## What is blocking the next round

`ncu` cannot run on this box: `ERR_NVGPUCTRPERM`. The packed GEMV's load-count
problem is localised but not explained, and a profiler would say in one run
whether it is LSU issue, MIO queueing or something else. Enabling it is one
root-level driver setting, `NVreg_RestrictProfilingToAdminUsers=0`.

## How things were measured

- One measurement at a time, through `gpulock`. A perf run that shared the GPU
  with a sweep reported 9.24 s for a 4.70 s scene and dragged the geomean from
  1.09 to 0.92.
- `control/precision_probe.py` for accuracy, not the four-mode gate. The gate
  reports scene 0's maximum trajectory error, which on this checkpoint only
  takes the two adjacent BF16 output ULPs 0.0403 and 0.0806, so it flips on
  changes that move nothing.
- fp64 operator oracles wherever a route had to be chosen: vision segmented
  MHA, the GDN decode recurrence, the GDN chunk-state scan, the GDN chunk
  GEMM. The first overturned the gate's verdict; the third decided between two
  tensor-core forms that the end-to-end numbers could not separate; the fourth
  kept a working optimisation switched off.
- Microbenchmarks one variant per process, on buffers past the 32 MB L2. In one
  process the same five variants came out non-monotonic and reordered between
  builds, and one BF16 reference measured 596 us in a cold process and 217 in a
  warm one.
