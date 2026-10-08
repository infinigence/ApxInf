#!/usr/bin/env bash
# This harness must come from the reviewed default branch, not the PR checkout.
set -euo pipefail

if [[ $# != 4 ]]; then
  echo 'usage: run_pi05_ci.sh thor|orin SHA CANDIDATE_DIR OUTPUT_DIR' >&2
  exit 2
fi
board=$1
revision=$2
candidate=$(realpath "$3")
output=$(realpath -m "$4")
trusted=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ "$board" == thor || "$board" == orin ]]
[[ "$revision" =~ ^[0-9a-f]{40}$ ]]
[[ $(git -C "$candidate" rev-parse HEAD) == "$revision" ]]
: "${APXINF_CI_PYTHON:?set the ApxInf Python environment}"
: "${APXINF_CI_BANK:?set the frozen bank JSON path}"
: "${APXINF_CI_BANK_SHA256:?set the approved bank digest}"
: "${APXINF_CI_PREFLIGHT:?set the operator-owned hardware preflight executable}"
: "${APXINF_CI_PREFLIGHT_SHA256:?set the approved preflight digest}"

# Developers use this same lock for their GPU commands. Never kill their jobs.
exec 9>"${APXINF_CI_GPU_LOCK:-/tmp/apxinf-${board}-gpu.lock}"
reserve_gpu() {
  if ! flock -n 9; then
    echo 'GPU reserved; resubmit after the current development job finishes.' >&2
    exit 75
  fi
}
reserve_gpu
mkdir -p "$output"
[[ $(sha256sum "$APXINF_CI_PREFLIGHT" | cut -d ' ' -f 1) == "$APXINF_CI_PREFLIGHT_SHA256" ]]
"$APXINF_CI_PREFLIGHT" "$board" > "$output/reservation-preflight.log" 2>&1
[[ -z $(git -C "$candidate" status --porcelain) ]]
flock -u 9
cd "$candidate"
PYO3_PYTHON="$APXINF_CI_PYTHON" cargo build --locked --release -p apxinf-py \
  --features cuda,extension-module --target-dir "$candidate/target" \
  --message-format=json-render-diagnostics > "$output/build.jsonl" 2> "$output/build.log"
artifact=$("$APXINF_CI_PYTHON" - "$output/build.jsonl" <<'PY'
import json
import sys

artifacts = []
with open(sys.argv[1]) as stream:
    for line in stream:
        item = json.loads(line)
        if item.get("reason") == "compiler-artifact" and item["target"]["name"] == "apxinf_py":
            artifacts.extend(path for path in item["filenames"] if path.endswith(".so"))
if len(artifacts) != 1:
    raise ValueError("Cargo did not produce exactly one apxinf_py extension")
print(artifacts[0])
PY
)
mkdir -p "$output/python"
ln -sf "$artifact" "$output/python/apxinf_py.so"
reserve_gpu
"$APXINF_CI_PREFLIGHT" "$board" > "$output/preflight.log" 2>&1
export PYTHONPATH="$output/python${PYTHONPATH:+:$PYTHONPATH}"
status=0
"$APXINF_CI_PYTHON" "$trusted/pi05_ci.py" run \
  --bank "$APXINF_CI_BANK" --bank-sha256 "$APXINF_CI_BANK_SHA256" \
  --revision "$revision" --hardware "$board" --output-dir "$output" || status=$?
"$APXINF_CI_PREFLIGHT" "$board" > "$output/postflight.log" 2>&1
exit "$status"
