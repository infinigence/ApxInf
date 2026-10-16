# SmolVLA

This module contains the native ApxInf implementation of SmolVLA for LIBERO.
It intentionally does not depend on PyTorch, Transformers, Hugging Face model
code, or another Python inference framework. The Python policy uses the
repository's native tokenizer implementation; all model execution is handled by
ApxInf Rust code and CUDA kernels.

## Module layout

| File | Responsibility |
| --- | --- |
| `config.rs` | Model dimensions, image/patch settings, action shape, flow schedule, and checkpoint config parsing. |
| `weights.rs` | Safetensors loading, tensor layout conversion, weight packing, upload, and BF16/FP16 conversion. |
| `model.rs` | Vision encoder, VLM prefix encoder, action expert, cross-attention, flow matching, and phase timing. |
| `runtime.rs` | `VlaRuntime` integration, request preparation, initial-noise handling, and inference contract. |
| `load.rs` | Registry loading and checkpoint path resolution. |

The Python-facing integration is in `python/apxinf/apxinf/policies/impls/smolvla.py`.
It owns observation normalization, action denormalization, tokenizer setup, and
the `smolvla` / `smolvla_libero` policy registration. The maintained benchmark
and LIBERO entry points are `scripts/bench_smolvla.py` and
`scripts/eval_smolvla_libero.py`.

## Inference path

1. Accept two RGB views plus robot state and the tokenized task prompt.
2. Convert HWC/NHWC `uint8` images directly into patch-major normalized tensors
   on the GPU.
3. Encode the vision tower, project robot state, and build the language/vision
   prefix.
4. Run the VLM transformer and retain per-layer prefix keys and values.
5. Run the 10-step flow-matching action expert. The schedule advances from
   `1.0` to `0.1`; each step contains action/time projections, expert
   self-attention, cross-attention to the VLM prefix, and SwiGLU MLPs.
6. Slice the final latent to the configured action dimensions. For the LIBERO
   checkpoint, the output is `[50, 7]`.

Checkpoint configs use LeRobot's `num_vlm_layers` semantics: `0` means the
VLM is not cropped, so the LIBERO checkpoint uses all 32 text-model layers.
Configs without that field retain the historical 16-layer default.

The default model variant is BF16. The optional FP16 variant converts the
uploaded model tensors to FP16 and uses the FP16 GEMM path, which is
tensor-core-capable on Xavier's `sm_72`. BF16 and FP16 dispatch to matching
CUDA kernels for normalization, activation, attention, preprocessing, and
elementwise operations.

## Optimizations

- **GPU image preprocessing:** the `uint8`-to-patch kernel performs
  normalization, channel reordering, and patch-major layout conversion in one
  GPU operation instead of constructing patches on the CPU.
- **Packed projections:** QKV projections and gate/up MLP projections are
  packed into single GEMMs. Related biases are batched where possible.
- **FP16 execution:** all model-side GEMM operands use FP16 in the FP16
  variant, avoiding mixed FP32/BF16 GEMM fallbacks.
- **Fused CUDA kernels:** FP16 paths include RMSNorm, SwiGLU, QKV split with
  RoPE, bias+SiLU, concat, Euler flow update, and output slicing kernels.
- **Specialized attention kernels:** prefix GQA, suffix causal GQA, and full
  cross-attention paths avoid unnecessary decode-time attention work and keep
  attention entirely on the GPU.
- **GEMM tactics:** exact-shape cuBLAS tactic selection can be supplied through
  `--tactics`; the reported measurements use the tuned Xavier tactic table.
- **Stream-ordered allocation reuse:** operator outputs use an exact-size,
  stream-keyed CUDA allocation cache by default. This removes the thousands of
  blocking `cudaMalloc`/`cudaFree` pairs formerly issued by one inference. Set
  `APXINF_CUDA_ALLOC_CACHE=0` to disable the cache.
- **Cross-attention prefix K/V reuse:** the VLM prefix is fixed across the ten
  action-denoising steps, so its cross-attention keys and values are projected
  once after prefix encoding and reused by every step. This removes 144
  repeated GEMMs per inference while preserving the same inputs, weights, and
  projections.
- **Uninitialized FP16 GEMM output:** the FP16 GEMM path writes every output
  element with `beta = 0`, so its output buffer is allocated without an
  avoidable `cudaMemset`.
- **Phase profiling:** CUDA events separately measure preprocessing, prefix
  embedding, VLM transformer, action expert, and output slicing.

SmolVLA cross-attention uses half-split RoPE for the query:
`x[..., :d/2]` rotates with `x[..., d/2:]`. This is intentionally different
from the interleaved-pair RoPE used elsewhere in the CUDA stack. The dedicated
FP16 half-split kernel fixed the original numerical mismatch with the reference
implementation.

## Numerical validation

The FP16 implementation was compared against a fixed-input LeRobot reference
rollout after the RoPE correction:

| Metric | Result |
| --- | --- |
| End-to-end relative Frobenius error | `0.006532` |
| First-step velocity relative error | `0.007712` |
| Cross-attention layer relative errors | `0.0065`–`0.0179` |
| Output shape | `[50, 7]` |
| Output values | all finite |

The CUDA test
`rope_half_split_f16_matches_fp32_reference` in
`crates/apxinf-cuda/src/tests/operators.rs` covers the corrected RoPE layout.

## Performance

Measurements were taken on the local Xavier `sm_72` GPU with the FP16 variant,
two `512x512` cameras, and the tuned GEMM tactic table. The baseline used 20
iterations after 3 warmups; the eager result uses 50 iterations after 3
warmups:

| Stage or metric | Baseline p50 | Current p50 |
| --- | ---: | ---: |
| End-to-end | `395.7 ms` | `387.0 ms` |
| Model | `390.4 ms` | `381.9 ms` |
| Preprocess | `4.4 ms` | `3.5 ms` |
| Prefix embedding | `193.0 ms` | `183.5 ms` |
| VLM transformer | `29.1 ms` | `28.3 ms` |
| VLM prefix total | `222.4 ms` | `211.8 ms` |
| Action expert | `160.8 ms` | `164.2 ms` |
| Output slicing | `0.09 ms` | `0.05 ms` |

The current end-to-end p50 is about `8.7 ms` lower than the known baseline.
The action-expert p50 varies between short runs, so the small stage-level
increase should not be interpreted as a regression from the cross K/V change.
The remaining dominant costs are still prefix embedding and the action expert;
the VLM transformer is comparatively small after the prefix is built.

The whole-model CUDA Graph path captures RGB preprocessing, prefix
construction, cross-attention K/V preparation, all ten action-denoise steps,
and output slicing in one graph. Its workspace uses lifetime-aware sub-block
reuse rather than retaining every intermediate at a unique address. On the
same 50-iteration benchmark, the graph path reaches `312.4 ms` model p50,
`317.6 ms` end-to-end p50, and `311.9 ms` graph p50. The graph workspace peak
is about `45 MiB`; a `96 MiB` reservation is used to leave headroom for
address fragmentation. The fixed-input graph output is bitwise equal to the
eager output, so the graph changes execution scheduling but not the policy
computation. Set `APXINF_SMOLVLA_NO_GRAPH=1` to run the fixed eager path.
## LIBERO spatial result

The corrected FP16 implementation was evaluated with
`scripts/eval_smolvla_libero.py` on `libero_spatial`, one rollout per task:

| Setting | Value |
| --- | --- |
| Tasks | 10 |
| Trials per task | 1 |
| Completed runs | 10 |
| Successes | 8 |
| Success rate | **80%** |
| Failed tasks | task `4`, task `7` |
| Max steps | 520 |
| Replan interval | 5 actions |
| Seed | 7 |
| Mean model time | `435.3 ms/call` |
| Mean inference time | `450.1 ms/call` |

The latest pure-eager run (`APXINF_SMOLVLA_NO_GRAPH=1`) completed `6/10`;
tasks `1`, `4`, `7`, and `8` reached the 520-step timeout. Re-running those
four failed tasks once succeeded on tasks `1` and `7`, while tasks `4` and `8`
still timed out. Together with the two earlier `8/10` runs, this indicates
LIBERO rollout variability despite the fixed seed. A fixed-input comparison
against the pre-optimization output is bitwise equal, so the policy
computation itself is unchanged.

## Validation commands

```bash
CUDA_PATH=/path/to/cuda cargo check -p apxinf-model --features cuda
CUDA_PATH=/path/to/cuda cargo test -p apxinf-cuda rope_half_split_f16_matches_fp32_reference
python -m pytest python/apxinf/tests/test_smolvla_policy.py
```

Benchmark and LIBERO evaluation use the maintained scripts:

```bash
python scripts/bench_smolvla.py --model-dir /path/to/checkpoint \
  --model-variant fp16 --num-views 2 --image-size 512 --profile
python scripts/eval_smolvla_libero.py --model-dir /path/to/checkpoint \
  --model-variant fp16 --suite libero_spatial --tasks all \
  --trials-per-task 1
```
