#!/usr/bin/env bash
# Re-derive the vendored FMHA header tree from public checkouts.
#
# The 218 headers under `mate/data/include` are byte-identical to two public
# trees: MATE tag `v0.2.7` at `include/mate/{attention/fmha,common}`, and the
# MUTLASS commit that tag pins, at `include/{mutlass,mute}`. This script proves
# both directions — every vendored file matches, and the pinned subtrees hold
# nothing the vendored tree lacks — or shows exactly what moved.
#
# usage:
#   MATE_SRC=<mate checkout> MUTLASS_SRC=<mutlass checkout> tools/verify_pins.sh
#
# The two revisions default to the pins recorded in PINS.md. Override either to
# run the reverse control — pointing the script at a branch head must report
# differences, or it is not checking anything:
#   MATE_REV=v0.2.8 MUTLASS_REV=435707c2 ... tools/verify_pins.sh
#
# Reads only: `git show` per file, no fetch, no checkout, no writes.

set -u -o pipefail

readonly MATE_REV_DEFAULT=v0.2.7
readonly MUTLASS_REV_DEFAULT=78b349514831ed54a797b717f2fc0534646acff2

readonly MAT_REV="${MATE_REV:-$MATE_REV_DEFAULT}"
readonly MUT_REV="${MUTLASS_REV:-$MUTLASS_REV_DEFAULT}"

root=$(cd -- "$(dirname -- "$0")/.." && pwd)
readonly root

die() { printf '%s\n' "$*" >&2; exit 2; }

[ -n "${MATE_SRC:-}" ] || die "MATE_SRC is not set: point it at a MooreThreads/mate checkout"
[ -n "${MUTLASS_SRC:-}" ] || die "MUTLASS_SRC is not set: point it at a MooreThreads/mutlass checkout"
git -C "$MATE_SRC" rev-parse --git-dir >/dev/null 2>&1 || die "MATE_SRC is not a git checkout: $MATE_SRC"
git -C "$MUTLASS_SRC" rev-parse --git-dir >/dev/null 2>&1 || die "MUTLASS_SRC is not a git checkout: $MUTLASS_SRC"
git -C "$MATE_SRC" cat-file -e "${MAT_REV}^{commit}" 2>/dev/null || die "MATE_SRC has no revision $MAT_REV"
git -C "$MUTLASS_SRC" cat-file -e "${MUT_REV}^{commit}" 2>/dev/null || die "MUTLASS_SRC has no revision $MUT_REV"

# The vendored tree mirrors the wheel's `mate/data/`, not the source layout:
# `mate/data/include/**` is `include/**` of the two repositories.
#
#   label   source checkout   revision       vendored root   upstream subtrees...
#
# The upstream subtrees are the directories the vendored root is supposed to be
# the whole of. They are counted, not just compared, so a re-vendor that drops a
# file fails here instead of passing as a subset — the `builders/` incident was
# seven headers that went missing without anything noticing.
compare() {
    local label=$1 src=$2 rev=$3 vendored=$4
    shift 4
    local -a upstream_paths=("$@")
    local checked=0 missing=0 differing=0
    local external=1 public_count
    local own_fp pub_fp own_list public_list f rel own_sum public_sum

    echo "== $label @ $rev"

    own_list=$(mktemp) || die "mktemp failed"
    public_list=$(mktemp) || die "mktemp failed"
    # shellcheck disable=SC2064
    trap "rm -f '$own_list' '$public_list'" RETURN

    while IFS= read -r f; do
        rel=${f#"$vendored"/}
        own_sum=$(sha256sum "$f")
        own_sum=${own_sum%% *}
        public_sum=$(git -C "$src" show "$rev:include/$rel" 2>/dev/null | sha256sum)
        public_sum=${public_sum%% *}
        printf '%s  %s\n' "$own_sum" "$rel" >>"$own_list"
        if [ "$public_sum" = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 ]; then
            missing=$((missing + 1))
            printf '   missing upstream: %s\n' "$rel"
            continue
        fi
        printf '%s  %s\n' "$public_sum" "$rel" >>"$public_list"
        if [ "$public_sum" != "$own_sum" ]; then
            differing=$((differing + 1))
            printf '   differs: %s\n' "$rel"
        fi
        checked=$((checked + 1))
    done < <(find "$vendored" -type f | sort)

    own_fp=$(sort "$own_list" | sha256sum)
    own_fp=${own_fp%% *}
    pub_fp=$(sort "$public_list" | sha256sum)
    pub_fp=${pub_fp%% *}

    public_count=$(git -C "$src" ls-tree -r --name-only "$rev" -- \
        "${upstream_paths[@]}" | wc -l)
    public_count=${public_count//[[:space:]]/}
    [ -n "$public_count" ] || public_count=-1
    if [ "$public_count" -ne "$((checked + missing))" ]; then
        external=0
        printf '   upstream has %s files under %s, the vendored tree has %s\n' \
            "$public_count" "${upstream_paths[*]}" "$((checked + missing))"
    fi

    echo "   $checked compared, $differing differ, $missing missing, $public_count upstream"
    echo "   vendored fingerprint: $own_fp"
    echo "   public   fingerprint: $pub_fp"
    [ "$own_fp" = "$pub_fp" ] || echo "   FINGERPRINT MISMATCH"

    [ "$differing" -eq 0 ] && [ "$missing" -eq 0 ] && [ "$external" -eq 1 ] &&
        [ "$own_fp" = "$pub_fp" ]
}

mate_ok=0
mutlass_ok=0
compare "MATE headers" "$MATE_SRC" "$MAT_REV" "$root/mate/data/include" \
    include/mate/attention/fmha include/mate/common && mate_ok=1
compare "MUTLASS headers" "$MUTLASS_SRC" "$MUT_REV" "$root/mate/data/mutlass/include" \
    include/mutlass include/mute && mutlass_ok=1

if [ "$mate_ok" -eq 1 ] && [ "$mutlass_ok" -eq 1 ]; then
    echo "OK: the vendored tree is exactly the pinned public trees."
    exit 0
fi
echo "FAILED: the vendored tree is not $MAT_REV + $MUT_REV." >&2
exit 1
