#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
# Hold the original directory intact and restore it even when Cargo fails.
exec 9>crates/.gemm-pilot-link.lock
flock -n 9
original=crates/apxinf-cuda
saved=crates/apxinf-cuda.pilot-original
[[ -d "$original" && ! -L "$original" && ! -e "$saved" ]]
restore() {
 if [[ -L "$original" && "$(readlink "$original")" == apxinf-cuda-new ]]; then
  unlink "$original"
  mv "$saved" "$original"
 fi
}
trap restore EXIT HUP INT TERM
mv "$original" "$saved"
ln -s apxinf-cuda-new "$original"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
: "${CUDA_PATH:=/usr/local/cuda}"
: "${APXINF_CUDA_ARCH:=sm_110}"
: "${CARGO_TARGET_DIR:=$repo_root/target-gemm-pilot}"
export CUDA_PATH APXINF_CUDA_ARCH CARGO_TARGET_DIR
cargo "$@"
