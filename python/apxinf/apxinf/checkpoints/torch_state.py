"""Read small, tensor-only PyTorch state archives without importing torch.

This is deliberately narrower than ``torch.load``.  It accepts the modern zip
serialization used by WallOSS normalizer sidecars, permits only an
``OrderedDict`` of contiguous one-dimensional CPU float32 tensors, and rejects
every other pickle global.  Model weights continue to be loaded by Rust from
SafeTensors; this reader is only for small processor metadata.
"""

from __future__ import annotations

import collections
import io
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["TorchStateError", "load_torch_state_file"]

_MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_TENSORS = 4096
_MAX_STORAGE_ELEMENTS = 1_000_000
_MAX_TOTAL_TENSOR_ELEMENTS = 4_000_000
_FLOAT32 = np.dtype("<f4")


class TorchStateError(ValueError):
    """The file is not a supported tensor-only PyTorch state archive."""


@dataclass(frozen=True)
class _StorageRef:
    key: str
    count: int


@dataclass(frozen=True)
class _TensorRef:
    storage: _StorageRef
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def _rebuild_tensor_v2(storage, offset, shape, stride, requires_grad, backward_hooks):
    if not isinstance(storage, _StorageRef):
        raise pickle.UnpicklingError("tensor does not reference an allowed storage")
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise pickle.UnpicklingError(f"tensor has invalid storage offset {offset!r}")
    if not isinstance(shape, tuple) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in shape
    ):
        raise pickle.UnpicklingError(f"tensor has invalid shape {shape!r}")
    if not isinstance(stride, tuple) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in stride
    ):
        raise pickle.UnpicklingError(f"tensor has invalid stride {stride!r}")
    if requires_grad is not False:
        raise pickle.UnpicklingError(
            "processor-state tensors must not require gradients"
        )
    if not isinstance(backward_hooks, collections.OrderedDict) or backward_hooks:
        raise pickle.UnpicklingError(
            "processor-state tensor has unexpected backward hooks"
        )
    return _TensorRef(storage, offset, shape, stride)


class _RestrictedStateUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch", "FloatStorage"): _FLOAT32,
            ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
        }
        try:
            return allowed[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"forbidden pickle global {module}.{name}"
            ) from error

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) != 5:
            raise pickle.UnpicklingError(f"malformed storage reference {pid!r}")
        kind, dtype, key, location, count = pid
        if kind != "storage" or dtype != _FLOAT32 or location != "cpu":
            raise pickle.UnpicklingError(
                f"unsupported storage kind={kind!r} dtype={dtype!r} location={location!r}"
            )
        if not isinstance(key, str) or not key.isdecimal():
            raise pickle.UnpicklingError(f"unsafe storage key {key!r}")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 0 <= count <= _MAX_STORAGE_ELEMENTS
        ):
            raise pickle.UnpicklingError(f"invalid storage element count {count!r}")
        return _StorageRef(key, count)


def _archive_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [
        name
        for name in archive.namelist()
        if name == suffix or name.endswith("/" + suffix)
    ]
    if not matches:
        raise TorchStateError(f"PyTorch archive has no {suffix}")
    if len(matches) != 1:
        raise TorchStateError(f"PyTorch archive has ambiguous {suffix} entries")
    return matches[0]


def load_torch_state_file(path) -> dict[str, np.ndarray]:
    """Load a WallOSS-style processor state dict into copied NumPy arrays."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as error:
        raise TorchStateError(f"read {path}: {error}") from error
    if size > _MAX_ARCHIVE_BYTES:
        raise TorchStateError(
            f"{path}: {size} bytes exceeds processor-state limit {_MAX_ARCHIVE_BYTES}"
        )

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise TorchStateError(
                    f"{path}: archive contains duplicate member names"
                )
            expanded = sum(member.file_size for member in archive.infolist())
            if expanded > _MAX_UNCOMPRESSED_BYTES:
                raise TorchStateError(
                    f"{path}: expanded archive size {expanded} exceeds "
                    f"processor-state limit {_MAX_UNCOMPRESSED_BYTES}"
                )
            pickle_name = _archive_member(archive, "data.pkl")
            prefix = pickle_name[: -len("data.pkl")]
            byteorder_name = prefix + "byteorder"
            if byteorder_name in names:
                byteorder = archive.read(byteorder_name).decode(
                    "ascii", errors="strict"
                )
                if byteorder != "little":
                    raise TorchStateError(
                        f"{path}: unsupported byte order {byteorder!r}"
                    )
            payload = _RestrictedStateUnpickler(
                io.BytesIO(archive.read(pickle_name))
            ).load()
            if not isinstance(payload, collections.OrderedDict):
                raise TorchStateError(
                    f"{path}: state payload is {type(payload).__name__}, expected OrderedDict"
                )
            if len(payload) > _MAX_TENSORS:
                raise TorchStateError(
                    f"{path}: state contains too many tensors ({len(payload)})"
                )

            tensors: dict[str, np.ndarray] = {}
            storages: dict[str, np.ndarray] = {}
            total_elements = 0
            for name, tensor in payload.items():
                if not isinstance(name, str) or not isinstance(tensor, _TensorRef):
                    raise TorchStateError(
                        f"{path}: state entry {name!r} is not a named tensor"
                    )
                if len(tensor.shape) != 1 or tensor.stride != (1,):
                    raise TorchStateError(
                        f"{path}: tensor {name!r} must be contiguous rank-1, got "
                        f"shape={tensor.shape} stride={tensor.stride}"
                    )
                width = tensor.shape[0]
                if not isinstance(width, int) or isinstance(width, bool) or width < 0:
                    raise TorchStateError(
                        f"{path}: tensor {name!r} has invalid shape {tensor.shape}"
                    )
                total_elements += width
                if total_elements > _MAX_TOTAL_TENSOR_ELEMENTS:
                    raise TorchStateError(
                        f"{path}: state tensor outputs exceed the processor-state "
                        f"limit of {_MAX_TOTAL_TENSOR_ELEMENTS} float32 values"
                    )
                start, end = tensor.offset, tensor.offset + width
                if tensor.offset < 0 or end > tensor.storage.count:
                    raise TorchStateError(
                        f"{path}: tensor {name!r} exceeds its storage"
                    )
                storage = storages.get(tensor.storage.key)
                if storage is None:
                    storage_name = prefix + "data/" + tensor.storage.key
                    try:
                        raw = archive.read(storage_name)
                    except KeyError as error:
                        raise TorchStateError(
                            f"{path}: missing storage {storage_name}"
                        ) from error
                    expected_bytes = tensor.storage.count * _FLOAT32.itemsize
                    if len(raw) != expected_bytes:
                        raise TorchStateError(
                            f"{path}: storage {tensor.storage.key} has {len(raw)} bytes, "
                            f"expected {expected_bytes}"
                        )
                    storage = np.frombuffer(
                        raw, dtype=_FLOAT32, count=tensor.storage.count
                    )
                    storages[tensor.storage.key] = storage
                elif storage.size != tensor.storage.count:
                    raise TorchStateError(
                        f"{path}: storage {tensor.storage.key} has inconsistent "
                        f"element counts {storage.size} and {tensor.storage.count}"
                    )
                tensors[name] = np.array(
                    storage[start:end], dtype=np.float32, copy=True
                )
            return tensors
    except (OSError, UnicodeError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
        raise TorchStateError(
            f"{path}: unsupported PyTorch state archive: {error}"
        ) from error
