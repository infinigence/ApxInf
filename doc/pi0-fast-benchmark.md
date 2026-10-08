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
  --model-dir /models/pi0fast-libero-v044 --precision bf16 \
  --state-key observation/state --layer l1 --mode latency \
  --frames-count 10 --warmup 10 --samples 30 \
  --tactics devlocal/model-bench-inputs/pi0fast/thor-bf16-tactics.json --autotune \
  --out devlocal/model-bench-inputs/pi0fast/thor-bf16.json
```

This input-free command measures complete-request latency. It does not produce
the README's Prefix and Per Token values. Run it on Thor, Orin and RTX 4090 for
the corresponding latency check. For Thor FP8, use `--precision fp8
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
  --model-dir /models/pi0fast-libero-v044 --precision bf16 \
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
