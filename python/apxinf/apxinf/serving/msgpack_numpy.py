"""OpenPI-compatible MessagePack + NumPy codec.

Mirrors ``openpi_client.msgpack_numpy`` byte-for-byte so the server interoperates
with the unmodified official client **without importing OpenPI**. NumPy arrays and
scalars are encoded as tagged maps; every other value passes through untouched.
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np

__all__ = ["pack_numpy", "unpack_numpy", "packer", "unpackb"]


def pack_numpy(value: Any) -> Any:
    """Encode arrays exactly like ``openpi_client.msgpack_numpy``."""
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise ValueError(f"unsupported NumPy dtype {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def unpack_numpy(value: dict) -> Any:
    """Decode arrays exactly like ``openpi_client.msgpack_numpy``."""
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def packer() -> "msgpack.Packer":
    """A fresh ``Packer`` wired to :func:`pack_numpy` (one per connection)."""
    return msgpack.Packer(default=pack_numpy)


def unpackb(payload: bytes) -> Any:
    """Decode one MessagePack frame, rebuilding NumPy arrays via :func:`unpack_numpy`."""
    return msgpack.unpackb(payload, object_hook=unpack_numpy)
