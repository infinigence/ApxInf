"""Read an openpi PyTorch export's ``metadata.pt`` **without torch**.

``scripts/train_pytorch.py`` in openpi ends a run with::

    torch.save({"global_step": ..., "config": dataclasses.asdict(train_config),
                "timestamp": ...}, checkpoint_dir / "metadata.pt")

so the checkpoint carries its own serving contract: action width, chunk length,
token budget, whether state is discretized, which cameras the client sends, and
— the field that matters most here — the ``asset_id`` that says where the real
``norm_stats.json`` lives. Nothing in apxinf's serving path read it, which is
why an openpi export silently fell back to :func:`Pi05Config::default` for its
architecture and to a flat ``norm_stats.json`` for its statistics.

**Why not just call torch.load.** ``metadata.pt`` is a config object graph, not
weights, so needing a 2 GB dependency to read 30 kB of nested dicts is
backwards: the preflight checks are supposed to run *before* anything heavy, and
``scripts/openpi_metadata_to_apxinf.py`` is supposed to run on a laptop.
``torch.save``'s modern format is a plain zip whose ``<archive>/data.pkl``
member is an ordinary pickle, so the standard library can read it — provided the
unpickler refuses to import the openpi/flax/jax classes the stream names, which
are not installed here and would be arbitrary code execution if they were.

**Security.** :class:`_RestrictedUnpickler` never imports anything outside
:data:`_ALLOWED`, which is a set of explicit ``(module, name)`` pairs. Allowing a
whole module — ``builtins`` especially — would resolve ``builtins.eval`` and
hand a hostile checkpoint the process. Everything else becomes an inert
:class:`_Opaque` subclass that records what it was and drops what it held.
"""

from __future__ import annotations

import importlib
import io
import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = [
    "MetadataError",
    "read_metadata_pt",
    "repack_structure",
    "train_config_facts",
]

#: openpi's flow-matching step count when ``model.num_steps`` is absent, which
#: is every upstream export (only forks serialize it). Matches the Rust
#: ``Pi05Config::default``, so an absent field is left for the loader to fill.
DEFAULT_NUM_FLOW_STEPS = 10
#: pi05's trained view-slot count when the repack transform names no cameras.
#: Also the Rust default, for the same reason.
DEFAULT_NUM_VIEWS = 3


class MetadataError(RuntimeError):
    """``metadata.pt`` is absent, is not a torch archive, or is not openpi's."""


# Classes the pickle stream is allowed to actually instantiate. Explicit pairs,
# never whole modules: see the module docstring.
_ALLOWED = frozenset(
    {
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "tuple"),
        ("pathlib", "PosixPath"),
        ("pathlib", "PurePosixPath"),
        ("pathlib", "WindowsPath"),
        ("pathlib", "PureWindowsPath"),
    }
)


class _Opaque:
    """Inert stand-in for a class we refuse to import.

    Must be a **class**, not a factory function: the ``NEWOBJ`` opcode calls
    ``cls.__new__(cls, *args)`` and rejects a non-type. The reduce arguments are
    kept because ``Enum`` pickles as ``EnumClass(value)`` — so an enum a caller
    does care about is still readable through :func:`unwrap`.
    """

    _qualname = "?"

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._args = args
        obj._state: Any = None
        return obj

    def __init__(self, *args, **kwargs) -> None:  # noqa: D107 - see __new__
        pass

    def __setstate__(self, state: Any) -> None:
        self._state = state
        if isinstance(state, dict):
            # Keep the recorded reduce args reachable even if the state shadows
            # them, so unwrap() has something to fall back on.
            self.__dict__.update({k: v for k, v in state.items() if k != "_args"})

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<opaque {self._qualname}>"


_STUBS: Dict[Tuple[str, str], type] = {}


def _stub(module: str, name: str) -> type:
    key = (module, name)
    cls = _STUBS.get(key)
    if cls is None:
        cls = type(
            name, (_Opaque,), {"__module__": module, "_qualname": f"{module}.{name}"}
        )
        _STUBS[key] = cls
    return cls


def unwrap(value: Any) -> Any:
    """Reduce an :class:`_Opaque` to its payload where that is unambiguous.

    An ``Enum`` reduces to ``(cls, (value,))``, so a single positional argument
    is the enum's value. Anything else stays as-is; callers treat it as absent.
    """
    if isinstance(value, _Opaque) and len(getattr(value, "_args", ())) == 1:
        return value._args[0]
    return value


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: D102
        if (module, name) in _ALLOWED:
            return getattr(importlib.import_module(module), name)
        return _stub(module, name)

    def persistent_load(self, pid):  # noqa: D102
        # Tensor storages resolve through the persistent-id table. metadata.pt
        # holds a config graph rather than weights, so nothing we read ever
        # dereferences one; returning None keeps the stream parseable.
        return None


def read_metadata_pt(path) -> Dict[str, Any]:
    """Return the whole ``metadata.pt`` payload dict (``global_step``/``config``/…)."""
    path = Path(path)
    if not path.is_file():
        raise MetadataError(f"{path} does not exist")
    try:
        with zipfile.ZipFile(path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name == "data.pkl" or name.endswith("/data.pkl")
            ]
            if not members:
                raise MetadataError(
                    f"{path} is a zip but has no data.pkl member, so it is not a "
                    f"torch archive; it holds {archive.namelist()[:10]}"
                )
            # Shortest name = the top-level record, not something nested.
            raw = archive.read(min(members, key=len))
    except zipfile.BadZipFile as exc:
        raise MetadataError(
            f"{path} is not a zip-format torch archive ({exc}). It was probably "
            f"written with _use_new_zipfile_serialization=False, which this "
            f"torch-free reader does not support; read it with torch instead."
        ) from exc

    payload = _RestrictedUnpickler(io.BytesIO(raw)).load()
    if not isinstance(payload, Mapping):
        raise MetadataError(
            f"{path} unpickled to {type(payload).__name__}, expected a dict"
        )
    return dict(payload)


def repack_structure(data: Mapping[str, Any]) -> Dict[str, Any]:
    """The wire keys openpi's client sends, from ``repack_transforms.inputs``.

    openpi's repack transform is written *inbound* — ``{wire_key: dataset_column}``
    — so its keys are exactly what the client puts on the network, which is what
    an apxinf preset's ``slots`` and ``state_key`` have to reproduce.
    """
    inputs = (data.get("repack_transforms") or {}).get("inputs") or ()
    for entry in inputs:
        structure = entry.get("structure") if isinstance(entry, Mapping) else None
        if isinstance(structure, Mapping):
            return dict(structure)
    return {}


def _as_int(value: Any) -> Optional[int]:
    value = unwrap(value)
    # bool is an int subclass; a flag is never a width.
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _as_text(value: Any) -> Optional[str]:
    value = unwrap(value)
    if value is None or isinstance(value, _Opaque):
        return None
    text = str(value)
    return text or None


def train_config_facts(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Split the serialized ``TrainConfig`` into architecture + deployment facts.

    Returns ``{"arch": {...}, ...}`` where ``arch`` is already in the JSON shape
    ``Pi05Config::from_json_str`` accepts (it takes openpi's field names as
    aliases), and the remaining keys are what preflight and the preset diff need.
    """
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise MetadataError(
            f"metadata.pt has no 'config' dict (keys: {sorted(payload)}); it was "
            f"not written by openpi's train_pytorch.py"
        )
    model = config.get("model") if isinstance(config.get("model"), Mapping) else {}
    data = config.get("data") if isinstance(config.get("data"), Mapping) else {}

    # --- guards. These are architecture facts apxinf's pi05 runtime hard-codes,
    #     so a checkpoint that disagrees cannot be served at all -- better to say
    #     so here than to load 7.5 GB of weights into the wrong graph.
    pi05 = unwrap(model.get("pi05"))
    if pi05 is not None and not isinstance(pi05, _Opaque) and not pi05:
        raise MetadataError(
            "metadata.pt says pi05=False: this is a pi0 checkpoint, and apxinf's "
            "pi05 path (discrete state in the prompt, quantile norm) cannot serve it"
        )
    for field, expected in (
        ("paligemma_variant", "gemma_2b"),
        ("action_expert_variant", "gemma_300m"),
    ):
        got = _as_text(model.get(field))
        if got is not None and got != expected:
            raise MetadataError(
                f"metadata.pt says {field}={got!r}, but apxinf's pi05 runtime is "
                f"built for {expected!r}; the weights would not match the graph"
            )

    # --- architecture, in Pi05Config::from_json_str's vocabulary. Only fields
    #     the checkpoint actually states go in here: an absent one has to fall
    #     through to the loader's own default rather than be asserted, or the
    #     report would present a guess as a fact.
    arch: Dict[str, Any] = {}
    for field in ("action_dim", "action_horizon", "max_token_len"):
        value = _as_int(model.get(field))
        if value is not None:
            arch[field] = value
    # ``num_steps`` is the flow-matching Euler step count. Upstream openpi does
    # not serialize it (it is DEFAULT_NUM_FLOW_STEPS by construction, which is
    # also the loader's default); some forks do.
    num_steps = _as_int(model.get("num_steps"))
    if num_steps is not None:
        arch["num_flow_steps"] = num_steps

    structure = repack_structure(data)
    images = structure.get("images")
    image_keys = tuple(images) if isinstance(images, Mapping) else ()
    if image_keys:
        arch["num_views"] = len(image_keys)

    discrete = unwrap(model.get("discrete_state_input"))
    if isinstance(discrete, bool):
        arch["discrete_state_input"] = discrete

    # --- where the statistics live. openpi's own resolution order, verbatim:
    #     `asset_id = data.assets.asset_id or data.repo_id` (openpi config.py).
    assets = data.get("assets") if isinstance(data.get("assets"), Mapping) else {}
    asset_id = _as_text(assets.get("asset_id"))
    asset_id_source = "data.assets.asset_id"
    if asset_id is None:
        asset_id = _as_text(data.get("repo_id"))
        asset_id_source = "data.repo_id"
    if asset_id is None:
        asset_id_source = ""

    return {
        "arch": arch,
        "asset_id": asset_id,
        "asset_id_source": asset_id_source,
        # Where openpi computed them on the *training* machine. Useless as a
        # path here, but it is the string to quote when asking for the file.
        "assets_dir": _as_text(assets.get("assets_dir")),
        "image_keys": image_keys,
        "state_key": _as_text(structure.get("state")),
        "adapt_to_pi": unwrap(data.get("adapt_to_pi")),
        "use_delta_joint_actions": unwrap(data.get("use_delta_joint_actions")),
        "discrete_state_input": discrete if isinstance(discrete, bool) else None,
        "default_prompt": _as_text(data.get("default_prompt")),
        "exp_name": _as_text(config.get("exp_name")) or _as_text(config.get("name")),
        "global_step": _as_int(payload.get("global_step")),
    }
