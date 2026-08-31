"""Hardware tactic-database routing for source-checkout deployments."""

from __future__ import annotations

import ctypes
from pathlib import Path


_DEFAULT_TACTICS = {
    87: "orin-sm87",
    89: "rtx4090-sm89",
    101: "thor-sm101",
    110: "thor-sm110",
}
_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def cuda_sm(device: str) -> int | None:
    """Return CUDA's integer compute capability for ``cuda:N`` (e.g. 110)."""
    if device == "cuda":
        device_index = 0
    elif device.startswith("cuda:"):
        try:
            device_index = int(device.removeprefix("cuda:"))
        except ValueError as error:
            raise ValueError(f"invalid CUDA device {device!r}; expected cuda:N") from error
    else:
        return None

    try:
        cudart = ctypes.CDLL("libcudart.so")
    except OSError as error:
        raise RuntimeError(f"cannot query {device}: failed to load CUDA runtime: {error}") from error

    # cudaDevAttrComputeCapabilityMajor/Minor from cuda_runtime_api.h.
    def attribute(code: int) -> int:
        value = ctypes.c_int()
        status = cudart.cudaDeviceGetAttribute(ctypes.byref(value), code, device_index)
        if status != 0:
            raise RuntimeError(
                f"cannot query {device} compute capability: cudaDeviceGetAttribute "
                f"returned {status}"
            )
        return value.value

    return attribute(75) * 10 + attribute(76)


def resolve_pi05_tactics(
    device: str,
    precision: str,
    *,
    model_dir: Path | None = None,
    override: Path | None = None,
    allow_missing: bool = False,
) -> Path | None:
    """Resolve explicit, checkpoint-local, then source-tree PI0.5 tactics."""
    if override is not None:
        return Path(override)
    if model_dir is not None:
        checkpoint_tactics = Path(model_dir) / "tactics.json"
        if checkpoint_tactics.is_file():
            return checkpoint_tactics
    sm = cuda_sm(device)
    directory = _DEFAULT_TACTICS.get(sm)
    if directory is None:
        return None
    path = _SOURCE_ROOT / "configs" / "tuning" / "nvidia" / directory / "tactics.json"
    if not path.is_file() and not allow_missing:
        raise FileNotFoundError(f"default tactics for SM{sm} are missing: {path}")
    return path
