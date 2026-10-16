# Reproduce d3-mini on Decision Index 0.3

This guide reproduces the **public text** Decision Index 0.3 score for
`vllm-sr/d3-mini` with ApxInf. It does not run the private tests used by the
leaderboard's Full score or any vision benchmark.

## Published baseline and validated ApxInf result

The [official d3-mini model card](https://huggingface.co/vllm-sr/d3-mini#text-jev-decision-index-032)
reports a **54.9** public text score and a **67.7** public vision score for Jev
Decision Index 0.3.2. ApxInf has reproduced the public text suite only; the
vision suite was not run.

| Result | Scope | Public score | Completed | Coverage | Wall time | Median latency | P95 latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official d3-mini model card | text | **54.9** | - | - | - | - | - |
| Official d3-mini model card | vision | **67.7** | - | - | - | - | - |
| ApxInf on Jetson Thor | text | **54.91** | **140,178 / 140,178** | **100%** | **about 14 h 53 min** | **128 ms** | **1,165.7 ms** |

All **140,178** ApxInf result rows had status `ok`; there were no malformed or
duplicate rows. The ApxInf row was produced with the `cuda` feature and
`--autotune`. The **67.7** vision score above is the published model-card
result, not an ApxInf reproduction.

The suite takes about 7 GB of downloads and 17 GB of build space. The complete
d3-mini package is about 9 GB. Runtime depends strongly on the GPU.

## 1. Clone ApxInf

```bash
git clone https://github.com/infinigence/ApxInf.git
cd ApxInf
export APXINF_DIR="$PWD"
```

All commands below assume a Linux NVIDIA system with a working CUDA toolkit and
Rust toolchain. On Jetson, make sure `nvcc` and Cargo are on `PATH`:

```bash
export CUDA_PATH=/usr/local/cuda
export PATH="$CUDA_PATH/bin:$HOME/.cargo/bin:$PATH"
```

## 2. Create an isolated Python environment

Do not install the evaluation dependencies into the system Python. Debian and
Ubuntu protect it with PEP 668 and report `externally-managed-environment`.

```bash
python3 -m venv "$APXINF_DIR/.venv-d3-index"
source "$APXINF_DIR/.venv-d3-index/bin/activate"
python -m pip install --upgrade pip huggingface_hub
```

If `venv` is unavailable, install the matching distribution package first
(for example `python3-venv` or `python3-full`). Do not use
`--break-system-packages`.

The suite includes gated HLE data. Accept its terms on Hugging Face and log in:

```bash
hf auth login
```

## 3. Download the complete d3-mini package

```bash
export REPRO_ROOT="$APXINF_DIR/.d3-decision-index"
export MODEL_DIR="$REPRO_ROOT/d3-mini"
export KIT_DIR="$REPRO_ROOT/decision-index"
export SUITE_DIR="$REPRO_ROOT/suite-0.3"
export RUN_DIR="$REPRO_ROOT/run"
mkdir -p "$REPRO_ROOT" "$RUN_DIR"

hf download vllm-sr/d3-mini --local-dir "$MODEL_DIR"
```

Download the whole repository, not only the two safetensor shards. ApxInf also
validates and uses `MODEL_MANIFEST.json`, `config.json`,
`decision_config.json`, tokenizer files and `readout.safetensors`.

## 4. Build the pinned public Decision Index 0.3 suite

The result above used Decision Index commit
`f8552a13b643d39f61185f5951bb5fed16bedf51`.

```bash
git clone https://github.com/apolinario/decision-index.git "$KIT_DIR"
git -C "$KIT_DIR" checkout f8552a13b643d39f61185f5951bb5fed16bedf51
python -m pip install "transformers==5.17.0" -e "${KIT_DIR}[transformers,rebuild]"

export HF_HUB_DISABLE_XET=1
python -m decision_index suite rebuild --work "$REPRO_ROOT/work"
python -m decision_index suite import \
  --edition 0.3 \
  --dir "$SUITE_DIR" \
  --rows "$REPRO_ROOT/work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz" \
  --added-rows "$REPRO_ROOT/work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz" \
  --gsm8k-rows "$REPRO_ROOT/work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz"
python -m decision_index suite verify --edition 0.3 --dir "$SUITE_DIR"
```

The verified uncompressed suite hashes are:

```text
selected rows  b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5
added rows     7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89
GSM8K rows    759334894858a37d6bd7bd07298eac8e55da236e31dad327ccd06a9979f4738a
```

## 5. Prepare ApxInf's pre-tokenized records

```bash
python "$APXINF_DIR/scripts/d3_prepare_decision_index.py" \
  --model-dir "$MODEL_DIR" \
  --suite-dir "$SUITE_DIR" \
  --edition 0.3 \
  --out "$RUN_DIR/records.jsonl"

test "$(wc -l < "$RUN_DIR/records.jsonl")" -eq 140178
```

The script uses the official package's tokenizer and `d3_format.py`. Do not
replace either when comparing with the published score.

## 6. Compile with the CUDA feature

`cuda-new` is a deprecated feature alias. The supported build command is:

```bash
cd "$APXINF_DIR"
cargo build --locked --release \
  -p apxinf-model \
  --features cuda \
  --example d3_decision_index
```

## 7. Run the full benchmark

The runner keeps fixed-shape CUDA Graph profiles. The safe defaults retain at
most 12 text profiles and a total capacity of 30,000 tokens. They were selected
after an unbounded cache exhausted a 24 GiB RTX 4090. Pass them explicitly so
the run configuration is visible in logs:

```bash
"$APXINF_DIR/target/release/examples/d3_decision_index" \
  "$MODEL_DIR" \
  "$RUN_DIR/records.jsonl" \
  "$RUN_DIR/results.jsonl" \
  --autotune \
  --profile-limit 12 \
  --profile-token-budget 30000 \
  2>&1 | tee -a "$RUN_DIR/run.log"
```

`--autotune` tunes a recipe on the first miss, reuses it from the process cache,
and persists it under the runtime's default hardware/toolkit-specific recipe
directory. A later process started from the same working directory restores
those recipes without `--tuning-dir`. Add
`--tuning-dir "$RUN_DIR/recipes"` only when an explicit cache location is
preferred.

### Resume after interruption

Run the exact same command again with the same records and results paths. The
runner removes an incomplete trailing JSON line, reads completed `run_id`
values and appends only missing requests. A new process restores compatible
recipes from the default persistent cache; it retunes only missing or
invalidated recipes.

## 8. Score with the official scorer

```bash
python -m decision_index score \
  --edition 0.3 \
  --suite-dir "$SUITE_DIR" \
  --results "$RUN_DIR/results.jsonl" \
  --engine apxinf-d3-cuda-new \
  --out "$RUN_DIR"
```

Verify the headline fields:

```bash
python - <<'PY'
import json
import os

path = os.path.join(os.environ["RUN_DIR"], "scores.json")
score = json.load(open(path, encoding="utf-8"))
print({
    "decision_index": score["decision_index"],
    "completed": score["completed"],
    "complete": score["complete"],
    "counts": score["counts"],
})
assert score["decision_index"] == 54.91
assert score["completed"] == 140178
assert score["complete"] is True
assert score["counts"] == {"ok": 140178}
PY
```

Expected output:

```text
{'decision_index': 54.91, 'completed': 140178, 'complete': True, 'counts': {'ok': 140178}}
```

`54.91` is the reproducible public index. The leaderboard's Full score also
contains private tests and cannot be reproduced from the public kit.

## Optional quick validation

Before starting the full run, prepare and execute a small deterministic prefix:

```bash
python "$APXINF_DIR/scripts/d3_prepare_decision_index.py" \
  --model-dir "$MODEL_DIR" \
  --suite-dir "$SUITE_DIR" \
  --edition 0.3 \
  --limit 25 \
  --out "$RUN_DIR/quick-25-records.jsonl"

"$APXINF_DIR/target/release/examples/d3_decision_index" \
  "$MODEL_DIR" \
  "$RUN_DIR/quick-25-records.jsonl" \
  "$RUN_DIR/quick-25-results.jsonl" \
  --autotune \
  --profile-limit 12 \
  --profile-token-budget 30000

test "$(wc -l < "$RUN_DIR/quick-25-results.jsonl")" -eq 25
```

This validates model loading, CUDA execution, autotuning and result formatting.
It is not statistically meaningful and must not be reported as a Decision
Index score.
