# Qwen3-MoE optimization acceptance

Run on Thor-U, from the checkout after sourcing `/opt/data/dev/env.sh`:

```sh
/opt/data/dev/venv/bin/python -m unittest discover -s scripts/qwen3moe -p 'test_*.py'
/opt/data/dev/venv/bin/python scripts/qwen3moe/accept.py
APXINF_QWEN3MOE_GROUPED=1 /opt/data/dev/venv/bin/python scripts/qwen3moe/accept.py --milestone M1
```

Each invocation builds the CLI and verifier, replays `raw0` and `chat0` under
teacher forcing, checks the complete finite logit dumps, generates text through
the CLI, and benchmarks ISL 128/1024/4096 with OSL 128. Results and logs live in
unique directories under `/opt/data/dev/acceptance`; `results.jsonl` includes
failed runs. Inspect `generate.log` for coherent output before promotion.
The binary-capable source patch, source hashes, `source-changes.tar.gz`
(including untracked kernel contents), environment knobs, and exact commands
identify the tested candidate. Each snapshot also records deleted files and
symlink targets without following links; an output directory inside the
checkout is excluded from source capture. Keep the source tree fixed
throughout a run; use a separate directory for independent operator probes.

The numerical gate requires maximum error below 1.1 (the reproduced baseline
maximum is 1.0005), below the reference top-1/top-2 margin at every non-tie step,
and final-step error no more than three times initial error. Only `raw0` step 2
has a documented tie exemption. This envelope is specific to these fixtures;
new long-context fixtures need their own reference evidence.

For a claimed arithmetic-preserving change, pass `--previous /path/to/run` to
also compare full binary logit dumps byte for byte. `--isl` and `--osl` accept
comma-separated grids. M1/M2/M3 enforce cumulative throughput gates at
ISL 1024, OSL 128. M4 requires additional workspace and long-context reference
evidence and deliberately cannot be closed by this short-case harness alone.

Standalone probes:

```sh
nvcc -O3 -arch=sm_101 scripts/qwen3moe/grouped_probe.cu \
  -I/usr/local/cuda-12.8/thor/targets/aarch64-linux/include \
  -L/usr/local/cuda-12.8/thor/targets/aarch64-linux/lib -lcublas \
  -o /opt/data/dev/grouped_probe
/opt/data/dev/grouped_probe
nvcc -O3 -arch=sm_101 scripts/qwen3moe/gqa_probe.cu -o /opt/data/dev/gqa_probe
/opt/data/dev/gqa_probe
nvcc -O3 -arch=sm_101 scripts/qwen3moe/pdl_probe.cu -o /opt/data/dev/pdl_probe
/opt/data/dev/pdl_probe
```

Operator timings are diagnostic; only unprofiled end-to-end CLI measurements
can accept a performance milestone. Avoid other GPU work during acceptance.

Current opt-in candidate: M1/M2 acceptance gates passed on 2026-09-08;
remaining implementation tasks, M3/M4, and long-context acceptance are open:

```sh
export APXINF_QWEN3MOE_GROUPED=marlin APXINF_QWEN3MOE_MARLIN_M=64
export APXINF_QWEN3MOE_MARLIN_TUNE=constants
export APXINF_QWEN3MOE_MARLIN_F16=1 APXINF_QWEN3MOE_CHUNKED=1
export APXINF_QWEN3MOE_FUSED_QKV=1 APXINF_QWEN3MOE_DECODE_QKV=1
export APXINF_QWEN3MOE_DECODE_RESIDUAL=1 APXINF_QWEN3MOE_PREFILL_COMBINE=1
export APXINF_QWEN3MOE_GQA_VECTOR=1
export APXINF_QWEN3MOE_MAX_SEQ_LEN=9216
export APXINF_QWEN3MOE_BLOCKED=2 APXINF_QWEN3MOE_GEMV_MAGIC=1
export APXINF_QWEN3MOE_GEMV_PAIR=1
export APXINF_QWEN3MOE_SILU_LUT=1 APXINF_QWEN3MOE_F32_LOGITS=1
export APXINF_QWEN3MOE_DENSE_CACHE=2
```

The output head always retains checkpoint BF16 precision. `MAX_SEQ_LEN` must
cover ISL plus OSL; 9216 supports the 8192/128 test. Chunked prefill uses a
1024-token activation capacity, reports exact workspace bytes, and retains at
most four graphs. Persistent KV and weight storage are reported separately
from activation scratch. All optimization paths remain opt-in during the
campaign. `MARLIN_SILU=1` failed full-model numerical acceptance and `BLOCKED=1`
regressed full-model decode speed; neither is part of the candidate above.
`BLOCKED=2` preserves the original device/mapped placement of decode weights
and keeps checkpoint-format copies in additional mapped storage. Together
with exact conversion and wider residual loads, this candidate measured
3183.7 prefill and 81.56 decode tok/s at ISL1024, OSL128, with full
logit byte equality (run `20260908T063955.660095Z-92ec635b`). Its same-build
comparison improves prefill by 5.7% without a decode regression.
`GQA_VECTOR=1` independently enables vector GQA after the flag fix, validated
in `20260908T072800.917105Z-92ec635b`: 3186.8 prefill / 81.36 decode tok/s,
both strict references, complete byte equality, and coherent CLI output.
`MARLIN_TUNE=n256` failed numerical acceptance. `n128` also failed
the strict gate (raw0 maximum 1.381); matching greedy tokens is insufficient.

`--verify-only` records a numerical-only run and cannot claim a performance
milestone. `--cases isl128,isl1024,isl4096,isl8192 --reference /path/to/long`
checks named long-context fixtures. The cached CPU reference is validated
against both original full-forward cases before preparing new fixtures:

```sh
python scripts/qwen3moe/cached_reference.py --model "$MODEL" --out "$REF/cached"   --case raw0 --case-json "$REF/raw0.json" --check-npz "$REF/raw0.npz"
python scripts/qwen3moe/cached_reference.py --model "$MODEL" --out "$REF/long"   --case isl8192 --isl 8192
```

Do not run the CPU reference during performance acceptance: its shared-memory
traffic can also affect Thor's measured GPU throughput.

The scaled FP16 representation diagnostic now passes raw0, chat0, and all
four long-context fixtures with natural routing. It preserves FP32 residuals
and router inputs and retains the checkpoint's BF16 head values. It still
uses CPU FP32 matrix multiplication on reconstructed values, so these results
do not establish CUDA numerical acceptance.

`cuda_product_precision_reference.py` isolates the next uncertainty by moving
AWQ matrix products to CUDA while leaving the remaining operations on CPU.
It does not change the production runtime or the acceptance references.
Build and run its operator checks before replaying a model case:

```sh
nvcc -O3 -arch=sm_101 -shared -Xcompiler=-fPIC \
  scripts/qwen3moe/cuda_product_precision.cu \
  -I/usr/local/cuda-12.8/thor/targets/aarch64-linux/include \
  -L/usr/local/cuda-12.8/thor/targets/aarch64-linux/lib -lcublas \
  -o /opt/data/dev/libqwen_cuda_product_precision.so
python scripts/qwen3moe/cuda_product_precision_reference.py \
  --library /opt/data/dev/libqwen_cuda_product_precision.so \
  --products scaled3 --operator-only --out /opt/data/dev/cuda-product-operator-scaled3
python scripts/qwen3moe/cuda_product_precision_reference.py \
  --library /opt/data/dev/libqwen_cuda_product_precision.so \
  --products scaled3 --model "$MODEL" --case-json "$REF/raw0.json" \
  --reference "$REF/raw0.npz" --out /opt/data/dev/cuda-product-raw0-scaled3
```

Use separate output directories for `--products fp32`, `scaled3`, and
`scaled4`. Each run records complete finite logits, natural router selections,
per-step reference margins, and source/library hashes. Host transfers are
included in the replay, so its elapsed time is not an inference benchmark.
The library and model replay still require validation on Thor-U.

Additional experimental paths and probes:

- `APXINF_QWEN3MOE_HEAD_COMPENSATED=1` retains every BF16 head weight and
  represents its normalized input as BF16 high/low rows. One two-row GEMM
  and FP32 addition produce FP32 logits. The extra 1,223,680 bytes are shared
  by prefill/decode and independent of ISL. This accuracy experiment remains
  outside the accepted configuration: both original strict cases pass, but
  all four long-context cases still fail. Default-off complete logit bytes
  match the accepted vector-flag build.
- `APXINF_QWEN3MOE_GQA_BALANCED=1` selects vector attention with complete
  32-token tiles distributed across splits above 512 tokens. It independently
  enables GQA; do not combine it with `GQA_MMA`. Short contexts retain the
  original boundaries. Both original strict references and M2 pass, and
  same-build ISL1024 decode improves 81.58 → 82.84 tok/s. Longer contexts do
  not improve in the paired benchmark, and all four long-context numerical
  gates fail. Their bytes change. The accepted configuration keeps it disabled.
- `head_norm_probe.cu` checks fused normalization/high-low decomposition
  against the separate composition and an independent FP64 RMS calculation.
  The model shape takes 6.46 versus 8.20 microseconds; all nine shapes pass.
- `APXINF_QWEN3MOE_SHORT_GEMV=32` is an example threshold, not a measured
  recommendation. It evaluates experts with the existing batched decode
  GEMV/SwiGLU operations for up to that many tokens (valid range 0–256,
  default 0). Threshold selection needs a prompt-length sweep and numerical
  checks; it does not exempt any fixture from the gate.
- `partial_rows_probe.cu` compares multi-token residual-partial reduction
  against an independent CPU FP32 sum followed by the original norm kernel.
- `pdl_gemv_probe.cu` checks the actual residual/RMSNorm → blocked GEMV pair
  with and without PDL, eager and captured. No model route enables PDL yet.

Compile these two probes with `nvcc -std=c++17 -O3 -arch=sm_101`, adding
`-I/usr/local/cuda-12.8/thor/targets/aarch64-linux/include` on Thor-U. Run them
separately from performance acceptance. Local `cargo check --features cuda`
on a machine without CUDA verifies Rust types only.

`router_probe.cu` checks the router store change against an independent
CPU softmax and stable-sort oracle, including tied logits. The change retains
each selected weight in its owning lane instead of reading another lane's
global store without a memory barrier. It passes the GPU oracle and whole-model byte comparison recorded below. The PDL templates are guarded for SM90 or newer; their default
specializations remain available on older targets.

Thor validation on 2026-09-08: router passes 110 cases, multi-token residual
passes all 20 byte comparisons, and PDL preserves partial outputs in both
eager and captured execution. Captured PDL pairs are slightly slower, so the
model still leaves it disabled. The router/default route passes full-model
byte comparison, CLI integrity, and measures 2774.1 prefill / 77.74 decode
tok/s at ISL1024. That earlier run missed M1; the current candidate above passes it.

The short-GEMV route failed raw0 numerical acceptance. Its diagnostic sweep
shows a speed benefit at five tokens (59.7 → 43.1 ms) but a regression from
16 through 256 tokens. Keep the default threshold zero pending numerical
work; the example value above is not an accepted setting.

`APXINF_QWEN3MOE_F32_LOGITS=1` retains FP32 output from the BF16 head GEMM.
With head compensation disabled, both head weights and inputs stay BF16. This removes output-storage rounding,
which by itself exceeds the ISL4096 step7 reference margin, and does not relax
the gate. `bf16_f32_output_probe.cu` checks this mixed cuBLAS operation against
independent host dot products, including the full vocabulary width. Both original model references pass with this option enabled.

For numerical diagnosis, set `APXINF_QWEN3MOE_LAYER_TRACE` to a new JSONL path
before running the verifier. This records each layer's last-token residual,
FFN input, router logits, expert IDs, and router weights. It forces eager
execution and synchronizes host reads, so its timings are invalid. Existing
trace files are never overwritten. Compare prefill records with the cached
CPU reference using `compare_layer_trace.py --reference ... --trace ... --out ...`;
layer signatures summarize differences and cannot prove elementwise equality.

`DENSE_CACHE=1` (with the `APXINF_QWEN3MOE_` prefix) is rejected: its additional
1.812 GB mapped BF16 attention cache preserves logits but increases ISL1024
TTFT from 369.6 to 463.4 ms. `DENSE_CACHE=2` reserves space for a device cache
before packed-weight placement; it passes full logit byte equality and is
part of the accepted M1/M2 configuration.

`norm_rounding_reference.py` isolates the effect of rounding checkpoint norm
weights to BF16 while leaving all other CPU reference arithmetic FP32. Its
outputs are diagnostic and must not replace acceptance references. Run it
only while GPU performance tests are stopped.

`silu_vector_probe.cu` compares the inter=768 vector specialization against the
scalar LUT implementation across prompt sizes, all finite FP16 gates, and
eight up values. The production specialization passes byte equality and measures
157 µs versus 320 µs at 8192 routed rows. Together with the vector routed
combine, it passes full-model byte equality and the M1/M2 performance gates.

`F16_RMS_WEIGHTS=1` preserves the checkpoint FP16 RMS weights, while keeping
Q/K norm and the BF16 head unchanged. Its kernels pass 36 bit checks and six
independent numerical checks, but full-model raw0 error reaches 1.5985 and
fails the 1.1 gate. Keep it disabled; default-off model byte equality passes.

Current long-context checks with FP32 logits still fail all four strict gates.
ISL128/1024 retain greedy agreement, while ISL4096/8192 disagree at step7.
FP32 output storage removes one rounding limitation; upstream accuracy remains
unresolved. These results do not close M4.

`GQA_MMA=1` evaluates eight real heads with m16n8k16 tensor instructions,
avoiding the padded16-head WMMA work. The operator passes90 FP64-oracle
cases, and both original full-model logit dumps remain byte-identical. Its
first unprofiled run passes M1/M2 but shows no consistent end-to-end decode
speedup, despite faster isolated attention. The completed same-build
comparison shows a regression at ISL 1024, and vector-load WMMA is faster
at every measured model length. Keep `GQA_MMA` disabled.

`gemv_persistent_probe.cu` and `gemv_occupancy_probe.cu` preserve all partial
output bytes but do not establish a net projection speedup. Their scheduling
variants live in `gemv_schedule_probe.cuh`; production GEMV is unchanged.
`marlin_stages_probe.cu` rejects the slower two-stage pipeline.

`head_input_rounding_reference.py` captures final FP32 activations from the
independent CPU reference, then isolates BF16/FP16 input rounding through the
unchanged checkpoint head. It verifies replay against an existing reference,
stores separate diagnostic evidence, and must run outside GPU benchmarks.

`GQA_VECTOR=1` is now part of the accepted candidate. Its16-byte Q/K/V loads
preserve the existing WMMA arithmetic and pass30 operator byte checks plus
full original model-logit equality. It improves decode at1024/4096/8192 to
78.97/69.17/59.27 tok/s. It is mutually exclusive with `GQA_MMA=1`.

`GEMV_PAIR=1` is part of the accepted blocked-weight decode candidate. It reuses
exact Marlin U4 conversion, specializes group size 128, and vectorizes shared
reduction stores without changing FP32 accumulation order. The generic group
fallback and all four model shapes pass byte comparisons in
`gemv_tuned_probe.cu`. Full-model default-off and enabled strict checks,
complete logit byte comparisons, CLI integrity, and cumulative M2 pass.

`activation_precision_reference.py` evaluates CPU precision hypotheses while
keeping checkpoint head values unchanged. FP16 activations improve raw0, but
the 4096-token diagnostic still fails the strict margin checks. Its output is
diagnostic evidence, never a replacement acceptance reference.

`MARLIN_TUNE=constants` is part of the accepted prefill candidate. It requires M64 and
FP16 compute, retains the existing tile and reduction schedules, and removes
adapter options that are always fixed. Its mapped-weight operator checks pass
all 12 short/long and uniform/skewed/empty-expert combinations. Whole-model
strict checks, byte equality, CLI integrity, and cumulative M2 pass.

The activation diagnostic can save every expert selection and use an exact
FP32 control as a `--route-oracle`. This forces only expert membership while
recomputing probabilities and weights from the diagnostic run. It isolates
routing effects and must never be used as model acceptance or as a runtime
shortcut. The oracle must match the prompt, teacher-forcing tokens, and
independent FP32 logits.
