from __future__ import annotations

import collections
import io
import pickle
import struct
import sys
import types
import zipfile

import numpy as np
import pytest


class _Storage:
    def __init__(self, key: str, values):
        self.key = key
        self.values = tuple(float(value) for value in values)


class _Tensor:
    def __init__(self, rebuild, storage: _Storage, *, offset=0, stride=(1,)):
        self.rebuild = rebuild
        self.storage = storage
        self.offset = offset
        self.stride = stride

    def __reduce__(self):
        return (
            self.rebuild,
            (
                self.storage,
                self.offset,
                (len(self.storage.values),),
                self.stride,
                False,
                collections.OrderedDict(),
            ),
        )


def _torch_pickle(entries, *, offset=0):
    torch = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch._utils")
    storage_type = type("FloatStorage", (), {"__module__": "torch"})

    def rebuild_tensor_v2(
        *args,
    ):  # pragma: no cover - identified, never executed while dumping
        return args

    rebuild_tensor_v2.__name__ = "_rebuild_tensor_v2"
    rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    rebuild_tensor_v2.__module__ = "torch._utils"
    torch.FloatStorage = storage_type
    torch._utils = torch_utils
    torch_utils._rebuild_tensor_v2 = rebuild_tensor_v2

    previous = {name: sys.modules.get(name) for name in ("torch", "torch._utils")}
    sys.modules["torch"] = torch
    sys.modules["torch._utils"] = torch_utils
    try:

        class StatePickler(pickle.Pickler):
            def persistent_id(self, value):
                if isinstance(value, _Storage):
                    return (
                        "storage",
                        storage_type,
                        value.key,
                        "cpu",
                        len(value.values),
                    )
                return None

        buffer = io.BytesIO()
        pickler = StatePickler(buffer, protocol=2)
        pickler.dump(
            collections.OrderedDict(
                (
                    name,
                    _Tensor(rebuild_tensor_v2, storage, offset=offset, stride=stride),
                )
                for name, storage, stride in entries
            )
        )
        return buffer.getvalue()
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _write_archive(path, entries, *, payload=None):
    storages = {storage.key: storage for _, storage, _ in entries}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("normalizer/data.pkl", payload or _torch_pickle(entries))
        archive.writestr("normalizer/byteorder", "little")
        archive.writestr("normalizer/version", "3\n")
        for key, storage in storages.items():
            archive.writestr(
                f"normalizer/data/{key}",
                struct.pack(f"<{len(storage.values)}f", *storage.values),
            )


def test_load_torch_state_file_reads_contiguous_float32(tmp_path):
    from apxinf.checkpoints import load_torch_state_file

    entries = [
        ("min.x2_normal", _Storage("0", [-1.0, 2.5, 0.25]), (1,)),
        ("delta.x2_normal", _Storage("1", [2.0, 4.0, 8.0]), (1,)),
    ]
    path = tmp_path / "normalizer.pth"
    _write_archive(path, entries)

    state = load_torch_state_file(path)

    assert set(state) == {"min.x2_normal", "delta.x2_normal"}
    assert state["min.x2_normal"].dtype == np.float32
    np.testing.assert_array_equal(state["min.x2_normal"], [-1.0, 2.5, 0.25])
    np.testing.assert_array_equal(state["delta.x2_normal"], [2.0, 4.0, 8.0])


def test_load_torch_state_file_rejects_noncontiguous_tensor(tmp_path):
    from apxinf.checkpoints import TorchStateError, load_torch_state_file

    entries = [("min.x", _Storage("0", [1.0, 2.0]), (2,))]
    path = tmp_path / "normalizer.pth"
    _write_archive(path, entries)

    with pytest.raises(TorchStateError, match="contiguous rank-1"):
        load_torch_state_file(path)


def test_load_torch_state_file_rejects_non_integer_tensor_metadata(tmp_path):
    from apxinf.checkpoints import TorchStateError, load_torch_state_file

    storage = _Storage("0", [1.0, 2.0])
    path = tmp_path / "normalizer.pth"
    entries = [("min.x", storage, (1,))]
    payload = _torch_pickle(entries, offset=0.0)
    _write_archive(path, entries, payload=payload)

    with pytest.raises(TorchStateError, match="invalid storage offset"):
        load_torch_state_file(path)


def test_load_torch_state_file_rejects_pickle_globals(tmp_path):
    from apxinf.checkpoints import TorchStateError, load_torch_state_file

    path = tmp_path / "normalizer.pth"
    _write_archive(path, [], payload=pickle.dumps(eval, protocol=2))

    with pytest.raises(TorchStateError, match="forbidden pickle global"):
        load_torch_state_file(path)


def test_load_torch_state_file_limits_aliased_output_size(tmp_path, monkeypatch):
    from apxinf.checkpoints import TorchStateError, load_torch_state_file
    from apxinf.checkpoints import torch_state

    storage = _Storage("0", [1.0, 2.0])
    entries = [("min.x", storage, (1,)), ("delta.x", storage, (1,))]
    path = tmp_path / "normalizer.pth"
    _write_archive(path, entries)
    monkeypatch.setattr(torch_state, "_MAX_TOTAL_TENSOR_ELEMENTS", 3)

    with pytest.raises(TorchStateError, match="tensor outputs exceed"):
        load_torch_state_file(path)


def test_load_torch_state_file_reads_aliased_storage_once(tmp_path, monkeypatch):
    from apxinf.checkpoints import load_torch_state_file

    storage = _Storage("0", [1.0, 2.0])
    entries = [("min.x", storage, (1,)), ("delta.x", storage, (1,))]
    path = tmp_path / "normalizer.pth"
    _write_archive(path, entries)

    reads = 0
    original_read = zipfile.ZipFile.read

    def counting_read(archive, name, *args, **kwargs):
        nonlocal reads
        if str(name).endswith("/data/0"):
            reads += 1
        return original_read(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", counting_read)
    state = load_torch_state_file(path)

    assert reads == 1
    np.testing.assert_array_equal(state["min.x"], [1.0, 2.0])
    np.testing.assert_array_equal(state["delta.x"], [1.0, 2.0])
