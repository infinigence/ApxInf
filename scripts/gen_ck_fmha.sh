#!/usr/bin/env bash
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

# BF16 batch-mode, no mask/bias/LSE/dropout/sink:
#   fwd:         d128 (vision, head_dim 72 padded) and d256 (prefix);
#   fwd_splitkv: d256 split-KV (action expert) and every d256 combine
#                kernel the split-KV dispatcher references.
FWD="fmha_fwd_d[12][25][86]_bf16_batch_*_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale_ntrload_nsink"
SPLITKV="fmha_fwd_splitkv_combine_d256_bf16_batch_*@fmha_fwd_splitkv_d256_bf16_batch_*_nlogits_nbias_nmask_lse_nsquant_npagedkv_nsink"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python3 "$FMHA/generate.py" --targets gfx1151 -d fwd,fwd_splitkv -f "$FWD,$SPLITKV" -o "$tmp"
rm -rf "$VENDOR/generated"
mkdir -p "$VENDOR/generated"
cp "$tmp"/*.cpp "$VENDOR/generated/"
echo "generated $(ls "$VENDOR/generated" | wc -l) files into kernels/ck/generated"

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
