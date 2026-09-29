# PI0-FAST benchmark and evaluation

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

## Latency

Inputs are constructed deterministically from the checkpoint's image/state
shapes. All timing excludes input construction. L1 measures prepared RGB and
tokens through native token return; L2 includes policy preprocessing and action
decoding. These boundaries match the PI0.5 benchmark's L1/L2 convention.

```sh
python scripts/bench_pi0_fast.py \
  --model-dir /models/pi0fast-libero-v044 --precision bf16 \
  --state-key observation/state --layer l1 --mode all \
  --frames-count 1 --repeats 5 --warmup 10 --samples 30 \
  --tactics devlocal/pi0fast-bench/thor-bf16-tactics.json --autotune \
  --out devlocal/pi0fast-bench/thor-bf16.json
```

Run this command on Thor, Orin and RTX 4090 for their BF16 rows. For Thor FP8,
use `--precision fp8 --calibration /path/to/matching-calibration.json` and a
separate output and tactic path. `--tactics` reaches the native GEMV/GEMM
selector; `--calibration` supplies FP8 activation scales. Generate tactics once
with `--autotune`, then omit that flag and reuse the database for measurement.
Use separate stores for each hardware/precision combination and preserve their
CUDA/cuBLAS and kernel identity; never patch a stale store's identity.


`layers_ms` reports complete request timings. `ar.fit.fixed_ms` and
`ar.fit.per_step_ms` correspond to the README Prefix and Per Token columns.
The fit uses the same observation at multiple verified stopping points from its
full token stream. Each stop token must first occur at the requested position,
and every measured prefix must match the full stream. The command fails when
the output cannot provide two distinct stopping points. Check the fit's R² and
raw points before reporting those columns; do not substitute full-request
latency for prefix latency.

Lock CPU/GPU/EMC clocks and fan and exclude other GPU work. Keep raw samples,
engine/native-binary and model/calibration identities. The script adds artifact
identities to its JSON report. Historical real-input timings are reference
values until each constructed-input hardware/precision row is measured.

## Accuracy

Accuracy requires real simulator observations and the checkpoint's action
normalization; constructed benchmark inputs are never used to compute success.

```sh
python scripts/eval_libero.py --backend in-process \
  --model-dir /models/pi0fast-libero-v044 --precision bf16 \
  --suite libero_10 --trials-per-task 10 --seed 7 \
  --results-jsonl devlocal/pi0fast-eval/results.jsonl \
  --summary-json devlocal/pi0fast-eval/summary.json
```

In APXinf-robo, use `apxinf-robo eval-libero` with the same arguments. Its
`scripts/bench_pi0_fast.py` calls the pinned engine's benchmark and loads the
policy through Robo. Both evaluators preserve the checkpoint's eight-value
state, including both mirrored finger joints.
