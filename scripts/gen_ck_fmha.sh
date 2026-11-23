#!/usr/bin/env bash
# shellcheck shell=bash
# Regenerate apxinf-hip's vendored Composable Kernel (CK) FMHA subset.
#
#   scripts/gen_ck_fmha.sh <composable_kernel checkout>
#
# The checkout must be at the commit pinned in
# crates/apxinf-hip/kernels/ck/VENDOR.md. This script
#   1. runs CK's FMHA code generator for the pi05 attention instances, into
#      kernels/ck/generated/, and
#   2. when hipcc is available, recomputes the exact header subset those
#      instances include (hipcc -MM) and copies it, byte for byte, into
#      kernels/ck/{include,example}.
# To move to a newer CK, update the pin in VENDOR.md and here, rerun, rebuild
# and re-run the apxinf-hip tests and the pi05 accuracy gates.
set -euo pipefail

PIN=f176bf5b118309cb475b20841daf3f17e5231017
CK=${1:?usage: $0 <composable_kernel checkout>}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENDOR=$REPO/crates/apxinf-hip/kernels/ck
FMHA=$CK/example/ck_tile/01_fmha

actual=$(git -C "$CK" rev-parse HEAD)
if [[ "$actual" != "$PIN" ]]; then
  echo "CK checkout is at $actual, expected the pinned $PIN" >&2
  exit 1
fi

# BF16 batch-mode, no bias/LSE/dropout/sink:
#   fwd:         d128 (vision, head_dim 72 padded) and d256 (prefix), unmasked;
#   fwd_masked:  d128 causal, for GR00T's text backbone -- its 16 layers are
#                the only causal attention either model issues, and without a
#                masked instance they stay on apxinf's own kernel at 190 us
#                each against CK's 20 us;
#   fwd_splitkv: d256 split-KV (action expert) and every d256 combine
#                kernel the split-KV dispatcher references.
#
# Both causal alignments come from one instance: `mask_top_left` and
# `mask_bottom_right` differ in the `mask_type` argument, not in the generated
# kernel (see `mask.hpp`'s `decode`, where both set left = -1, right = 0).
# `-f` takes one pattern per direction, comma-separated, matched positionally
# against `-d`. Within a direction only fwd_splitkv and bwd split further on
# `@`; fwd passes its pattern straight to one `fnmatch` call. So the masked and
# unmasked fwd instances have to come from a single pattern -- `*mask` covers
# both `mask` and `nmask`. That forces the logits position to be pinned to
# `nlogits` (apxinf runs with `has_logits_soft_cap = false`, and the wrapper
# never sets a soft cap): a bare `*logits*` would also pull in the soft-cap
# variants, and the dispatcher would then pick those for every call.
#
# Generating in two passes is not an option: each pass also emits
# `fmha_fwd_api.cpp`, the dispatcher, and a dispatcher only knows the instances
# of its own pass.
# `d[1265][2456]` selects exactly head dim 64, 128 and 256 by their first two
# digits, and excludes the 32 and 192 tiles the generator also offers: nothing
# either model runs has a head dim of 32 or in (128, 192].
#
# The 64 tile matters more than its name suggests. GR00T's vision and VL
# attention have head dim 64 and its DiT has 48, and on the 128 tile both pay
# twice: the head dimension is padded to 128, and the tile's 128-row M extent
# sits two-thirds empty at the DiT's 41 query rows. Measured, the 64 tile takes
# vision from 23% of this device's compute ceiling to 57% and VL from 13% to
# 36% (`devlocal/gr00t-hip-port/perf/attention_probe.log`).
FWD="fmha_fwd_d[1265][2456]*_bf16_batch_*_nlogits_nbias_*mask_nlse_ndropout_nskip_nqscale_ntrload_nsink"
SPLITKV="fmha_fwd_splitkv_combine_d256_bf16_batch_*@fmha_fwd_splitkv_d256_bf16_batch_*_nlogits_nbias_nmask_lse_nsquant_npagedkv_nsink"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python3 "$FMHA/generate.py" --targets gfx1151 -d fwd,fwd_splitkv -f "$FWD,$SPLITKV" -o "$tmp"

# Checked before the existing vendoring is replaced. A filter that matches
# nothing is silent: the generator exits 0, the dispatcher is emitted with no
# instances behind it, and `rm -rf` has already removed the working set. That
# happened once here -- a stray `@` in the fwd pattern took out all four
# fwd instances while splitkv survived, and the crate still built because the
# wrapper falls back by design.
for required in \
  "fmha_fwd_api.cpp" \
  "fmha_fwd_d64_*_nmask_*" "fmha_fwd_d64_*_nbias_mask_*" \
  "fmha_fwd_d128_*_nmask_*" "fmha_fwd_d128_*_nbias_mask_*" \
  "fmha_fwd_d256_*_nmask_*" "fmha_fwd_d256_*_nbias_mask_*" \
  "fmha_fwd_splitkv_api.cpp" "fmha_fwd_splitkv_d256_*"; do
  # shellcheck disable=SC2086
  if ! compgen -G "$tmp"/$required > /dev/null; then
    echo "generator produced nothing matching '$required'; keeping the existing" \
         "kernels/ck/generated and leaving it untouched" >&2
    exit 1
  fi
done

rm -rf "$VENDOR/generated"
mkdir -p "$VENDOR/generated"
cp "$tmp"/*.cpp "$VENDOR/generated/"
echo "generated $(ls "$VENDOR/generated" | wc -l) files into kernels/ck/generated" \
     "($(ls "$VENDOR/generated" | grep -c '_mask_') masked)"

HIPCC=${ROCM_PATH:-/opt/rocm}/bin/hipcc
if [[ ! -x "$HIPCC" ]]; then
  echo "no hipcc: header subset not recomputed; run this on a ROCm host to refresh it" >&2
  exit 0
fi
defs="-DCK_TILE_FMHA_FWD_APPENDKV_API=0 -DCK_TILE_FMHA_FWD_SPLITKV_API=1 -DCK_TILE_FMHA_FWD_PAGEDKV_API=0"
deps=$(for f in "$VENDOR"/generated/*.cpp; do
  "$HIPCC" -MM -std=c++20 --offload-arch=gfx1151 -I"$CK/include" -I"$FMHA" $defs "$f"
done | tr ' \\' '\n\n' | grep -F "$CK/" | sed "s|^$CK/||" | sort -u)
rm -rf "$VENDOR/include" "$VENDOR/example"
while read -r f; do
  mkdir -p "$VENDOR/$(dirname "$f")"
  cp "$CK/$f" "$VENDOR/$f"
done <<< "$deps"
cp "$CK/LICENSE" "$VENDOR/LICENSE"
echo "copied $(wc -l <<< "$deps") headers into kernels/ck"
