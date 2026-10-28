# PI0-FAST benchmark

Prepare the real LeRobot checkpoint, its normalization files, and local
PaliGemma/FAST tokenizers. The policy discovers tokenizer directories under
`assets/paligemma-tokenizer` and `assets/fast-tokenizer` in the model directory.
No per-machine tokenizer path or environment override is needed with that layout.

```sh
hf download lerobot/pi0fast-libero-v044 --local-dir /models/pi0fast-libero-v044
hf download google/paligemma-3b-pt-224 --include '*token*' 'special_tokens_map.json' \
  --local-dir /models/pi0fast-libero-v044/assets/paligemma-tokenizer
hf download jadechoghari/fast-libero-tokenizer-mean-std \
  --local-dir /models/pi0fast-libero-v044/assets/fast-tokenizer
```

Record the exact downloaded revisions and checkpoint/calibration hashes for a
reproduction campaign. Access to the checkpoint's declared tokenizer may require
accepting its upstream access terms. Benchmarking never downloads resources.

## Controlled token-length benchmark (no frame file)

Use a real checkpoint with constructed images/state to measure exact decode
lengths. This requires a native binding built with `max_new_tokens` support.

First prepare a device/precision-specific tactic database with the same input
shapes. Run from the repository root. Keep `--seed 0` and the decode lengths
unchanged between preparation and measurement:

```sh
# Prepare tactics once; do not use this run as the reported performance result.
python scripts/bench_pi0_fast.py \
  --model-dir /models/pi0fast-libero-v044 --model-variant bf16 \
  --state-key observation/state --layer l1 --mode fixed --seed 0 \
  --warmup 10 --samples 30 \
  --tactics devlocal/model-bench-inputs/pi0fast/thor-bf16-tactics.json --autotune \
  --out devlocal/model-bench-inputs/pi0fast/prepare-bf16.json

# Measure in a new process with online tuning disabled, reusing that database.
python scripts/bench_pi0_fast.py \
  --model-dir /models/pi0fast-libero-v044 --model-variant bf16 \
  --state-key observation/state --layer l1 --mode fixed --seed 0 \
  --warmup 30 --samples 30 \
  --tactics devlocal/model-bench-inputs/pi0fast/thor-bf16-tactics.json \
  --out devlocal/model-bench-inputs/pi0fast/fixed-bf16.json
```

No LIBERO installation, dataset, simulator, or `.npz` input is required for these
commands. The checkpoint and its tokenizer/normalization assets are still needed.
For FP8, use `--model-variant fp8_static --calibration /path/to/matching-calibration.json`
in both commands, with separate FP8 tactic and output paths. Use separate tactic
paths for each board, precision and compatible CUDA/cuBLAS version. Reuse a
compatible prepared database directly when one is available.

Omitting `--tactics` selects the runtime's default database or fallback algorithms;
it does not reproduce this preparation step and can be substantially slower.
Keep the database hash from the JSON report together with the timings. The seed
is already 0 by default; spelling it out preserves the input contract in copied
commands. A fixed seed stabilizes the input, but does not by itself stabilize
machine load, clocks or kernel selection.
The measurement command uses 30 warmups per length (the CLI default is 10).
Inspect the saved raw samples for changes over time as well as the fit: a high
R² alone does not establish stable performance.

The default lengths are **12, 16, 20, 28, 32, 40** generated tokens. They were
selected from the retained 2026-10-22 natural-stop measurements at source
`08f72843`, using the same ten recorded LIBERO frames across three configurations:

| Configuration | Natural generated lengths, frame order |
|---|---|
| Orin BF16 | 28, 27, 16, 28, 17, 20, 17, 21, 21, 37 |
| Thor BF16 | 28, 27, 15, 28, 17, 21, 17, 21, 21, 37 |
| Thor FP8 | 21, 27, 15, 22, 21, 17, 18, 21, 23, 32 |

Counts include any generated `Action:` preamble and the terminator. This is ten unique observations, **not thirty
independent scenes**; repeated runs and baseline copies were excluded. The
pooled min/P25/P50/P75/P90/max are 15/17.25/21/27/28.4/37 (linear quantiles).
Round the minimum down to a multiple of four, P25/P50/P75 to the nearest four,
and P90/maximum up: 12/16/20/28/32/40. Four is just a convenient measurement grid,
not an alignment requirement. The range brackets these observations; it is not
a claim about all tasks/checkpoints. Use `--decode-lengths N ...` to override it
when a broader survey supports different lengths. At least two distinct positive
counts are required, all within the checkpoint's `max_action_tokens` capacity.

Fixed mode uses only the **first** prepared input for every length. Each call
runs exactly N generated tokens, ignoring `|`; N includes the first token from
prefill, so there are N-1 subsequent autoregressive forwards. The native loop
actually ends at N; it does not decode 256 tokens and truncate. KV/workspace
capacity stays checkpoint-sized. Timings include the native call and final token
return, exclude input construction and prefix comparisons, and omit per-token
EOS readbacks. Lengths are interleaved in ascending/descending rounds after
warming up each length. Every returned sequence must equal the prefix of the
longest decode on that input.

JSON `fixed_decode` stores prepared-input hashes and prompt length, the stop policy, requested counts, all samples,
per-length P50/P95, and a fit `T(N) = fixed_ms + N * per_step_ms` over medians.
The intercept is a regression coefficient; `fixed_ms + per_step_ms` estimates
the cost through the first generated token. This is a controlled compute
measurement, not the natural action latency or an action-quality evaluation.
In particular, the natural-stop path performs an EOS readback per token and
uses different observations. **Do not substitute these fitted values for the
historical README Prefix/Per Token numbers without remeasurement and explicit
labeling.** Natural-stop comparisons remain available below.

## Latency

By default, the script constructs inputs deterministically from the checkpoint's
image/state shapes; users do not need to create an input file. All timing
excludes input construction. L1 measures prepared RGB and tokens through native
token return; L2 includes policy preprocessing and action decoding. These
boundaries match the PI0.5 benchmark's L1/L2 convention. PI0-FAST currently
times its autoregressive native call; its runtime does not implement captured
CUDA Graph replay.

```sh
python scripts/bench_pi0_fast.py \
  --model-dir /models/pi0fast-libero-v044 --model-variant bf16 \
  --state-key observation/state --layer l1 --mode latency \
  --frames-count 10 --warmup 10 --samples 30 \
  --tactics devlocal/model-bench-inputs/pi0fast/thor-bf16-tactics.json --autotune \
  --out devlocal/model-bench-inputs/pi0fast/thor-bf16.json
```

This input-free command measures complete-request latency. It does not produce
the README's Prefix and Per Token values. Run it on Thor, Orin and RTX 4090 for
the corresponding latency check. For Thor FP8, use `--model-variant fp8_static
--calibration /path/to/matching-calibration.json` and a separate output path.
`--calibration` supplies FP8 activation scales. `--tactics` reaches the native
GEMV/GEMM selector. This command creates a device-specific database with
`--autotune`; omit `--autotune` and reuse that database for subsequent
measurements. Use separate stores for each hardware/precision combination and
preserve their CUDA/cuBLAS and kernel identity; never patch a stale store's
identity. To check the original script's default tactic path, omit both tactic
flags. That baseline can be substantially slower and can generate different
tokens, so record which path was used and compare like with like.

To measure the original prefix/per-token fit, provide recorded LIBERO frames
with `base_raw` or `base_flipped`, `wrist_raw` or `wrist_flipped`, `state`, and
`task` arrays in an `.npz` file:

```sh
python scripts/bench_pi0_fast.py \
  --model-dir /models/pi0fast-libero-v044 --model-variant bf16 \
  --state-key observation/state --layer l1 --mode ar \
  --frames /path/to/libero_frames.npz --frame-variant raw \
  --survey 20 --repeats 5 \
  --tactics devlocal/model-bench-inputs/pi0fast/thor-bf16-tactics.json \
  --out devlocal/model-bench-inputs/pi0fast/thor-bf16-ar.json
```

The script surveys frames for distinct observed decode lengths and fits latency
against token count. The result is available in `ar.fit.fixed_ms` and
`ar.fit.per_step_ms` only when at least two lengths occur. These fields have the
same meaning as the README Prefix and Per Token columns; reproducing the
published values also requires the original workload and measurement setup.
The synthetic default is sufficient for the complete-request latency check but
does not guarantee a usable AR fit.

Lock CPU/GPU/EMC clocks and fan and exclude other GPU work. Keep raw samples,
engine/native-binary and model/calibration identities. The script adds artifact
identities to its JSON report. Keep the historical table values as reference
until their original frame workload and measurement setup are reproduced.

## Real task evaluation

LIBERO success rates require simulator observations and the checkpoint's action
normalization. The shared evaluator uses the policy's declared state width, so
PI0-FAST receives both mirrored finger joints in its eight-value state:

```sh
python scripts/eval_libero.py --backend in-process \
  --model-dir /models/pi0fast-libero-v044 --precision bf16 \
  --suite libero_10 --trials-per-task 10 --seed 7 \
  --results-jsonl devlocal/pi0fast-eval/results.jsonl \
  --summary-json devlocal/pi0fast-eval/summary.json
```

In APXinf-robo, use `apxinf-robo eval-libero` with the same model, suite and
precision. Constructed benchmark inputs are not used for task scoring.
