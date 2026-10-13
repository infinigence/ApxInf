#!/usr/bin/env bash
# Deprecated shim — use `just test-l3` from the repository root instead.
# Kept so existing documentation and muscle memory keep working; it simply
# forwards legacy `test ...` calls to the just recipe.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
if [ "${1:-}" = test ]; then
  shift
fi
exec just test-l3 "$@"
