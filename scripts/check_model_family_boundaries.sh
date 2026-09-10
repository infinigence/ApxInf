#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="$repo_root/crates/apxinf-model/src"

# These directories are documented top-level infrastructure rather than model
# families. Add a directory only after its shared ownership is reviewed.
shared_dirs=(profiling vla)

is_shared_dir() {
    local candidate="$1"
    local shared
    for shared in "${shared_dirs[@]}"; do
        if [[ "$candidate" == "$shared" ]]; then
            return 0
        fi
    done
    return 1
}

# Composite model families may depend on a separately registered backbone only
# through an explicitly reviewed edge. GR00T N1.7 embeds Qwen3-VL as its
# released vision-language backbone; duplicating that architecture would make
# checkpoint and numerical fixes diverge between the standalone and VLA paths.
is_allowed_family_dependency() {
    local family="$1"
    local dependency="$2"
    [[ "$family" == "gr00t" && "$dependency" == "qwen3vl" ]]
}

families=()
for directory in "$model_root"/*; do
    [[ -d "$directory" && -f "$directory/mod.rs" ]] || continue
    family="$(basename "$directory")"
    is_shared_dir "$family" && continue
    families+=("$family")
done

violations=0
for family in "${families[@]}"; do
    family_dir="$model_root/$family"
    for other in "${families[@]}"; do
        [[ "$family" == "$other" ]] && continue
        is_allowed_family_dependency "$family" "$other" && continue
        pattern="(crate::${other}|super::super::${other})([^[:alnum:]_]|$)|^[[:space:]]*use[[:space:]]+crate::\\{[^;]*${other}::"
        if command -v rg >/dev/null 2>&1; then
            if rg -n -g '*.rs' "$pattern" "$family_dir"; then
                violations=1
            fi
        elif grep -R -n -E --include='*.rs' "$pattern" "$family_dir"; then
            violations=1
        fi
    done
done

if ((violations)); then
    echo 'model-family boundary violation: a family references another family directory' >&2
    echo 'copy architecture code locally or extract an explicitly reviewed shared module' >&2
    exit 1
fi

echo 'model-family boundary checks passed'
