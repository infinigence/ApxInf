#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$script_dir/test-new.sh" \
  test -p apxinf-cuda ops::tests::framework::gpu_e2e_ -- \
  --nocapture --test-threads=1
