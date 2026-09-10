#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 {bf16|fp8|int8} {1|2} OUTPUT_JSON [WARMUP] [ITERATIONS]" >&2
    echo "required env: APXINF_GR00T_CHECKPOINT APXINF_GR00T_COSMOS APXINF_GR00T_FIXTURE APXINF_GR00T_BF16_TACTICS_FILE" >&2
    echo "FP8 additionally requires APXINF_GR00T_CALIBRATION and may set APXINF_GR00T_TACTICS" >&2
}

if [[ $# -lt 3 || $# -gt 5 ]]; then
    usage
    exit 2
fi

tier=$1
views=$2
output_json=$3
warmup=${4:-5}
iterations=${5:-20}

repo_root=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || true)
if [[ -n $repo_root ]]; then
    export APXINF_BENCH_SOURCE_DIR=$repo_root
    export APXINF_BENCH_SOURCE_REVISION=$(git -C "$repo_root" rev-parse HEAD)
    if [[ -n $(git -C "$repo_root" status --porcelain) ]]; then
        export APXINF_BENCH_SOURCE_DIRTY=1
    else
        export APXINF_BENCH_SOURCE_DIRTY=0
    fi
else
    : "${APXINF_BENCH_SOURCE_DIR:?set APXINF_BENCH_SOURCE_DIR for an exported source tree}"
    : "${APXINF_BENCH_SOURCE_REVISION:?set APXINF_BENCH_SOURCE_REVISION for an exported source tree}"
    : "${APXINF_BENCH_SOURCE_DIRTY:?set APXINF_BENCH_SOURCE_DIRTY to 0 or 1}"
fi

: "${APXINF_GR00T_CHECKPOINT:?set APXINF_GR00T_CHECKPOINT to the GR00T N1.7 checkpoint directory}"
: "${APXINF_GR00T_COSMOS:?set APXINF_GR00T_COSMOS to the Cosmos-Reason2-2B directory}"
: "${APXINF_GR00T_FIXTURE:?set APXINF_GR00T_FIXTURE to the prepared fixture directory}"
: "${APXINF_GR00T_BF16_TACTICS_FILE:?set APXINF_GR00T_BF16_TACTICS_FILE to the accepted tactic database}"
checkpoint=$APXINF_GR00T_CHECKPOINT
cosmos=$APXINF_GR00T_COSMOS
fixture=$APXINF_GR00T_FIXTURE
bf16_tactics=$APXINF_GR00T_BF16_TACTICS_FILE
add_bf16_packed4_override=${APXINF_ADD_BF16_PACKED4:-}
bias_residual_bf16_packed4_override=${APXINF_BIAS_RESIDUAL_BF16_PACKED4:-}
precomputed_qwen_mrope_override=${APXINF_GR00T_PRECOMPUTED_QWEN_MROPE:-}
silu_mul_separate_packed4_override=${APXINF_SILU_MUL_SEPARATE_BF16_PACKED4:-}
bias_activation_block_cap_override=${APXINF_BIAS_ACTIVATION_BLOCK_CAP:-}
bias_activation_specialized_override=${APXINF_BIAS_ACTIVATION_SPECIALIZED:-}
precomputed_vision_rope_override=${APXINF_GR00T_PRECOMPUTED_VISION_ROPE:-}

device_model=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)
if [[ $device_model == *Orin* ]]; then
    platform=orin
else
    platform=thor
fi
binary=${APXINF_GR00T_BENCH_BIN:-$repo_root/target/release/examples/gr00t_fixture_bench}

case $views in
    1|2) ;;
    *)
        usage
        exit 2
        ;;
esac

fixture_views=$(python3 - "$fixture/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
shape = manifest["tensors"]["image_grid_thw"]["shape"]
if len(shape) != 2 or shape[1] != 3:
    raise SystemExit(f"invalid image_grid_thw shape in fixture manifest: {shape}")
print(shape[0])
PY
)
if [[ $fixture_views != "$views" ]]; then
    echo "fixture contains $fixture_views views, but benchmark requested $views" >&2
    exit 2
fi

if [[ ! -x $binary ]]; then
    echo "benchmark binary is missing or not executable: $binary" >&2
    exit 1
fi

# Prevent inherited experiment switches from silently changing the benchmark.
unset APXINF_GR00T_FUSED_FP8_VISION_FFN_HANDOFF
unset APXINF_GR00T_FUSED_QWEN_LINEAR
unset APXINF_GR00T_FUSED_QWEN_QKV
unset APXINF_GR00T_FUSED_QWEN_GATE_UP
unset APXINF_GR00T_FP8_FUSED_LINEAR_BIAS
unset APXINF_GR00T_PRECOMPUTED_QWEN_MROPE
unset APXINF_SILU_MUL_SEPARATE_BF16_PACKED4
unset APXINF_GR00T_PRECOMPUTED_VISION_ROPE
unset APXINF_BIAS_ACTIVATION_BLOCK_CAP
unset APXINF_BIAS_ACTIVATION_SPECIALIZED
unset APXINF_ADD_BF16_PACKED4
unset APXINF_BIAS_RESIDUAL_BF16_PACKED4
unset APXINF_GR00T_BF16_TACTICS
unset APXINF_CUBLASLT_TACTICS_FILE

# Canonical runs clear diagnostic kernel selectors. Controlled same-binary
# A/B tests can opt in explicitly without duplicating the benchmark command.
if [[ ${APXINF_GR00T_BENCH_EXPERIMENTS:-0} == 1 ]]; then
    if [[ -n $add_bf16_packed4_override ]]; then
        export APXINF_ADD_BF16_PACKED4=$add_bf16_packed4_override
    fi
    if [[ -n $bias_residual_bf16_packed4_override ]]; then
        export APXINF_BIAS_RESIDUAL_BF16_PACKED4=$bias_residual_bf16_packed4_override
    fi
    if [[ -n $precomputed_qwen_mrope_override ]]; then
        export APXINF_GR00T_PRECOMPUTED_QWEN_MROPE=$precomputed_qwen_mrope_override
    fi
    if [[ -n $silu_mul_separate_packed4_override ]]; then
        export APXINF_SILU_MUL_SEPARATE_BF16_PACKED4=$silu_mul_separate_packed4_override
    fi
    if [[ -n $bias_activation_block_cap_override ]]; then
        export APXINF_BIAS_ACTIVATION_BLOCK_CAP=$bias_activation_block_cap_override
    fi
    if [[ -n $bias_activation_specialized_override ]]; then
        export APXINF_BIAS_ACTIVATION_SPECIALIZED=$bias_activation_specialized_override
    fi
    if [[ -n $precomputed_vision_rope_override ]]; then
        export APXINF_GR00T_PRECOMPUTED_VISION_ROPE=$precomputed_vision_rope_override
    fi
fi

case $tier in
    bf16)
        export APXINF_GR00T_PRECISION=bf16
        unset APXINF_GR00T_FP8_CALIBRATION
        export APXINF_GR00T_BF16_TACTICS=$bf16_tactics
        if [[ $platform == thor ]]; then
            export APXINF_BIAS_ACTIVATION_BLOCK_CAP=80
            export APXINF_SILU_MUL_SEPARATE_BF16_PACKED4=1
            export APXINF_GR00T_PRECOMPUTED_VISION_ROPE=1
        else
            export APXINF_GR00T_PRECOMPUTED_QWEN_MROPE=1
            export APXINF_SILU_MUL_SEPARATE_BF16_PACKED4=1
            export APXINF_GR00T_PRECOMPUTED_VISION_ROPE=1
            export APXINF_BIAS_ACTIVATION_SPECIALIZED=1
        fi
        ;;
    fp8)
        : "${APXINF_GR00T_CALIBRATION:?set APXINF_GR00T_CALIBRATION for FP8}"
        export APXINF_GR00T_PRECISION=fp8
        export APXINF_GR00T_FP8_CALIBRATION=$APXINF_GR00T_CALIBRATION
        export APXINF_GR00T_BF16_TACTICS=${APXINF_GR00T_TACTICS:-$bf16_tactics}
        export APXINF_GR00T_FP8_FUSED_LINEAR_BIAS=1
        ;;
    int8)
        if [[ $platform != orin ]]; then
            echo "int8 is supported by this release only on Jetson AGX Orin" >&2
            exit 2
        fi
        export APXINF_GR00T_PRECISION=int8
        unset APXINF_GR00T_FP8_CALIBRATION
        export APXINF_GR00T_BF16_TACTICS=$bf16_tactics
        ;;
    *)
        usage
        exit 2
        ;;
esac

mkdir -p "$(dirname "$output_json")"
exec "$binary" "$checkpoint" "$cosmos" "$fixture" 0 "$warmup" "$iterations" graph "$output_json"
