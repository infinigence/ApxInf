#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/../.." && pwd)
target_dir=${CARGO_TARGET_DIR:-"$repo_dir/target/qwen25-omni-sm89"}

export APXINF_CUDA_ARCH=sm_89
export APXINF_CUDA_OPERATOR_SET=${APXINF_CUDA_OPERATOR_SET:-core-fa2}
export CARGO_TARGET_DIR=$target_dir

cd "$repo_dir"
cargo build --release --features cuda

binary=$CARGO_TARGET_DIR/release/apxinf
test -x "$binary"
sha256sum "$binary"
