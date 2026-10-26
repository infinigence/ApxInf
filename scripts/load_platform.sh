# Source from a Bash recipe before invoking the build tools.
# Preset values take precedence over the corresponding inherited variables.
apxinf_load_platform() {
    local selected="${APXINF_PLATFORM:-}"
    [[ -n "$selected" ]] || return 0

    local table="${BASH_SOURCE[0]%/*}/../configs/platforms.tsv"
    local platform arch parallelism extra
    while read -r platform arch parallelism extra; do
        [[ -n "$platform" && "$platform" != \#* ]] || continue
        [[ "$platform" == "$selected" ]] || continue
        if [[ ! "$arch" =~ ^sm_[0-9]+$ || -n "$extra" ]] ||
           [[ "$parallelism" != - && ! "$parallelism" =~ ^[1-9][0-9]*$ ]]; then
            echo "Invalid build preset for platform: $selected" >&2
            return 2
        fi
        export APXINF_CUDA_ARCH="$arch"
        if [[ "$parallelism" != - ]]; then
            export APXINF_NVCC_PARALLELISM="$parallelism"
        fi
        return 0
    done < "$table"
    echo "Unknown APXINF_PLATFORM: $selected (see configs/platforms.tsv)" >&2
    return 2
}

apxinf_load_platform
