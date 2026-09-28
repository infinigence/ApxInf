#!/usr/bin/env bash
# Run cargo against the apxinf-cuda-new crate:
#   bash crates/apxinf-cuda-new/test-new.sh test -p apxinf-cuda-new -- --nocapture
#
# The crate now carries its real name and lives in the workspace, so this is
# only a thin environment wrapper. The symlink swap that used to substitute
# this crate for `apxinf-cuda` is gone; both crates build side by side.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
: "${CUDA_PATH:=/usr/local/cuda}"
: "${APXINF_CUDA_ARCH:=sm_110}"
: "${CARGO_TARGET_DIR:=$repo_root/target-gemm-pilot}"
export CUDA_PATH APXINF_CUDA_ARCH CARGO_TARGET_DIR
cargo "$@"
