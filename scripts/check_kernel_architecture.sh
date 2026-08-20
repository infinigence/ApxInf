#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if find crates/apxinf-cuda/src -type f -name '._*' -print -quit | grep -q .; then
    echo 'kernel architecture violation: AppleDouble ._* files must not exist under src/' >&2
    exit 1
fi

fail_if_present() {
    local pattern="$1"
    local label="$2"
    shift 2
    if search_pattern "$pattern" "$@"; then
        echo "kernel architecture violation: $label" >&2
        exit 1
    fi
}

search_pattern() {
    local pattern="$1"
    shift
    if command -v rg >/dev/null 2>&1; then
        rg -n -g '*.rs' "$pattern" "$@"
        return
    fi

    local paths=()
    while (($#)); do
        if [[ "$1" == '-g' ]]; then
            shift 2
        else
            paths+=("$1")
            shift
        fi
    done
    grep -R -n -E --include='*.rs' "$pattern" "${paths[@]}"
}

search_cuda_pattern() {
    local pattern="$1"
    shift
    if command -v rg >/dev/null 2>&1; then
        rg -n -g '*.cu' -g '*.cuh' "$pattern" "$@"
        return
    fi
    grep -R -n -E --include='*.cu' --include='*.cuh' "$pattern" "$@"
}

fail_if_cuda_present() {
    local pattern="$1"
    local label="$2"
    shift 2
    if search_cuda_pattern "$pattern" "$@"; then
        echo "kernel architecture violation: $label" >&2
        exit 1
    fi
}

if [[ -d crates/apxinf-cuda/src/launch ]] &&
    find crates/apxinf-cuda/src/launch -type f -name '*.rs' -print -quit | grep -q .; then
    echo "kernel architecture violation: src/launch must not contain Rust modules" >&2
    exit 1
fi

for legacy_module in bf16 fp8 w8a8 decode; do
    if [[ -e "crates/apxinf-cuda/src/kernels/${legacy_module}.rs" ]]; then
        echo "kernel architecture violation: public kernels/${legacy_module}.rs uses a precision/stage classification" >&2
        exit 1
    fi
done

for legacy_header in \
    core_operators.cuh static_bf16.cuh pointwise.cuh \
    fp8_quantization.cuh w8a8.cuh common.cuh softmax.cuh; do
    if [[ -e "crates/apxinf-cuda/kernels/custom/${legacy_header}" ]]; then
        echo "kernel architecture violation: custom/${legacy_header} is not classified by physical operation" >&2
        exit 1
    fi
done

if [[ -d crates/apxinf-cuda/src/native ]]; then
    echo 'kernel architecture violation: Rust src/native forwarding layer must not exist' >&2
    exit 1
fi
if [[ -d crates/apxinf-cuda/native ]]; then
    echo 'kernel architecture violation: CUDA host adapters belong under adapters/' >&2
    exit 1
fi
fail_if_present 'crate::native|native_contracts!' \
    'removed Rust native forwarding layer is still referenced' \
    crates/apxinf-cuda/src -g '*.rs'
fail_if_present '^[[:space:]]*pub[[:space:]]+(unsafe[[:space:]]+fn|fn[^;]*(\*const|\*mut)[[:space:]])' \
    'safe public kernel contracts must not expose unsafe functions or raw pointers' \
    crates/apxinf-cuda/src/kernels -g '*.rs'
fail_if_present 'std::env::|env::var(_os)?' \
    'kernel execution paths must not read environment variables' \
    crates/apxinf-cuda/src/kernels -g '*.rs'
fail_if_present 'crate::launch|launch::|mod launch' \
    'removed launch layer is still referenced' \
    crates/apxinf-cuda/src crates/apxinf-model/src/pi05 -g '*.rs'
fail_if_present 'kernels::(bf16|fp8|w8a8|decode)(::|\b)' \
    'callers must use physical operator modules, not precision/stage modules' \
    crates/apxinf-cuda/src crates/apxinf-model/src -g '*.rs'

if find crates/apxinf-cuda/kernels -maxdepth 1 -type f -name '*.cu' -print -quit |
    grep -q .; then
    echo 'kernel architecture violation: kernels root must not contain host adapter .cu files' >&2
    exit 1
fi
fail_if_cuda_present 'extern[[:space:]]+"C"' 'pure CUDA/CUTLASS operator sources must not export C ABI symbols' crates/apxinf-cuda/kernels/custom crates/apxinf-cuda/kernels/cutlass/*.cu
fail_if_cuda_present 'cublasLt' 'cuBLASLt vendor planning belongs under adapters/, not kernels/' crates/apxinf-cuda/kernels/custom

echo 'kernel architecture checks passed'
