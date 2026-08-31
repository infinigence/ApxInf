"""What shape is this checkpoint directory, and where is each file?

Five directory layouts turn up in the field and only two of them matter here:

``openpi_pytorch``
    What ``scripts/train_pytorch.py`` writes and what ships to a customer:
    ``model.safetensors`` + ``metadata.pt`` (+ ``optimizer.pt``) with the
    statistics under ``assets/<asset_id>/norm_stats.json``. **There is no
    config.json**, so every architecture constant has to come out of
    ``metadata.pt``.

``lerobot``
    A HuggingFace robot-policy directory: ``config.json`` with ``"type": "pi05"``
    plus ``model.safetensors``. LeRobot keeps normalization statistics inside
    ``policy_preprocessor_step_<N>_normalizer_processor.safetensors`` rather than
    a JSON file (and the official pi05 repo deleted even that), so a servable
    LeRobot checkpoint needs a ``norm_stats.json`` supplied alongside it.

The tensor names are **identical** between the two — 812 of them, exact set
equality — so nothing about weight loading changes. Only the sidecar metadata
differs, which is the entire job of this module.

Why this is its own layer rather than a few lines in ``Pi05Policy.from_pretrained``:
three callers need the same answer and only one of them may import the CUDA
binding. :mod:`apxinf.robots.preflight` runs *before* anything heavy loads, and
``scripts/openpi_metadata_to_apxinf.py`` runs on a laptop. Previously each of
them hard-coded its own guess at the layout — ``norm_stats.json`` alone was
spelled out in three places and ``metadata.pt`` was read in none of the serving
ones.

The failure this exists to prevent: a checkpoint directory that contains a
*syntactically valid but semantically wrong* ``norm_stats.json`` loads without a
single error and unnormalizes every action through the wrong physical range. So
the resolver names every path it tried, logs the one it settled on, and refuses
to guess when nothing matches.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .metadata import MetadataError, read_metadata_pt, train_config_facts

__all__ = [
    "CheckpointError",
    "CheckpointLayout",
    "LEROBOT",
    "NORM_STATS_NAME",
    "OPENPI_PYTORCH",
    "TOKENIZER_NAMES",
    "detect_checkpoint",
    "has_layout_metadata",
    "require_norm_stats",
    "resolve_tokenizer",
]

LOGGER = logging.getLogger("apxinf.checkpoints")

#: openpi's PyTorch training export. Authority: ``metadata.pt``.
OPENPI_PYTORCH = "openpi_pytorch"
#: A LeRobot / HuggingFace robot-policy directory. Authority: ``config.json``.
LEROBOT = "lerobot"
#: Accepted values of ``checkpoint_format=`` / ``--ckpt-format``.
FORMATS = ("auto", OPENPI_PYTORCH, LEROBOT)

NORM_STATS_NAME = "norm_stats.json"
WEIGHTS_NAME = "model.safetensors"
METADATA_NAME = "metadata.pt"
CONFIG_NAME = "config.json"

#: SentencePiece filenames a checkpoint may carry, in preference order. apxinf
#: never downloads one: openpi pulls it from ``gs://big_vision`` and LeRobot from
#: a gated HF repo, and a serving box is not assumed to reach either.
TOKENIZER_NAMES = ("tokenizer.model", "paligemma_tokenizer.model")

_TOKENIZER_HELP = (
    "The tokenizer is not distributed with any pi05 checkpoint — the same ~4 MB "
    "SentencePiece model serves the whole PaliGemma family, so openpi downloads it "
    "from gs://big_vision/paligemma_tokenizer.model at runtime and LeRobot pulls "
    "google/paligemma-3b-pt-224 from the HF Hub (a gated repo: 401 without an "
    "accepted licence). apxinf never downloads anything, so the file has to be put "
    "in place once, by hand."
)


class CheckpointError(RuntimeError):
    """The checkpoint directory is not a layout apxinf can serve, as-is."""


@dataclass(frozen=True)
class CheckpointLayout:
    """Every path and constant resolved from one checkpoint directory."""

    #: :data:`OPENPI_PYTORCH` or :data:`LEROBOT`.
    format: str
    root: Path
    weights: Optional[Path]
    config_json: Optional[Path]
    metadata_pt: Optional[Path]
    #: openpi's ``data.assets.asset_id or data.repo_id``; ``None`` for LeRobot.
    asset_id: Optional[str] = None
    asset_id_source: str = ""
    #: The statistics file, or ``None`` when nothing matched — see
    #: :func:`require_norm_stats` for the error a caller should raise.
    norm_stats: Optional[Path] = None
    #: Every candidate considered, in order, whether or not it existed.
    norm_stats_tried: Tuple[Path, ...] = ()
    #: True when the asset-path candidates all missed and the root file was used.
    norm_stats_is_fallback: bool = False
    #: ``Pi05Config::from_json_str``-shaped architecture overrides. Empty for
    #: LeRobot, where the Rust loader reads ``config.json`` itself.
    arch: Mapping[str, Any] = field(default_factory=dict)
    #: Deployment facts from ``metadata.pt`` (cameras, delta convention, ...).
    openpi: Mapping[str, Any] = field(default_factory=dict)
    #: Human-readable observations worth surfacing but not worth failing on.
    notes: Tuple[str, ...] = ()

    def config_json_text(self) -> Optional[str]:
        """The ``config_json=`` string to hand ``apxinf_py.Model.load``.

        ``None`` means "let the Rust loader read ``config.json`` as before",
        which is exactly right for a LeRobot directory.
        """
        return json.dumps(dict(self.arch), sort_keys=True) if self.arch else None

    def describe(self) -> str:
        """A multi-line report for a log line or the standalone inspector."""
        lines = [
            f"format:       {self.format}",
            f"root:         {self.root}",
            f"weights:      {self.weights if self.weights else '(missing)'}",
        ]
        if self.metadata_pt:
            lines.append(f"metadata.pt:  {self.metadata_pt}")
        if self.config_json:
            lines.append(f"config.json:  {self.config_json}")
        if self.asset_id:
            lines.append(f"asset_id:     {self.asset_id!r} (from {self.asset_id_source})")
        stats = self.norm_stats or "(none found)"
        suffix = "  [ROOT FALLBACK]" if self.norm_stats_is_fallback else ""
        lines.append(f"norm_stats:   {stats}{suffix}")
        if len(self.norm_stats_tried) > 1:
            for candidate in self.norm_stats_tried:
                mark = "*" if candidate == self.norm_stats else ("+" if candidate.is_file() else "-")
                lines.append(f"                {mark} {candidate}")
        if self.arch:
            rendered = ", ".join(f"{k}={v}" for k, v in sorted(self.arch.items()))
            lines.append(f"architecture: {rendered}")
        for key in ("exp_name", "global_step", "image_keys", "state_key",
                    "adapt_to_pi", "use_delta_joint_actions", "default_prompt"):
            value = self.openpi.get(key)
            if value not in (None, (), ""):
                lines.append(f"{(key + ':').ljust(13)} {value!r}")
        for note in self.notes:
            lines.append(f"note:         {note}")
        return "\n".join(lines)


def _asset_parts(asset_id: str) -> Tuple[str, ...]:
    """Split an ``asset_id`` into path components, refusing traversal.

    openpi's asset ids are often ``org/dataset``, which becomes nested
    directories on disk. They come from a checkpoint we did not write, so a
    ``..`` component would read outside the checkpoint directory.
    """
    parts = tuple(p for p in asset_id.split("/") if p)
    if not parts or any(p in (".", "..") for p in parts):
        raise CheckpointError(
            f"asset_id {asset_id!r} is not a usable relative path; pass "
            f"--norm-stats explicitly"
        )
    return parts


def _resolve_norm_stats(
    root: Path, asset_id: Optional[str], explicit
) -> Tuple[Optional[Path], Tuple[Path, ...], bool, Tuple[str, ...]]:
    """Find the statistics file. Returns ``(path, tried, is_fallback, notes)``."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise CheckpointError(f"norm_stats path {path} does not exist")
        return path, (path,), False, ()

    candidates = []
    if asset_id:
        parts = _asset_parts(asset_id)
        # openpi's official convention, written by train_pytorch.py.
        candidates.append(root.joinpath("assets", *parts, NORM_STATS_NAME))
        # Some exporters also drop a copy without the assets/ prefix.
        candidates.append(root.joinpath(*parts, NORM_STATS_NAME))
    flat = root / NORM_STATS_NAME
    tried = (*candidates, flat)

    notes = []
    for candidate in candidates:
        if candidate.is_file():
            if flat.is_file():
                notes.append(
                    f"{flat} exists but is ignored; the asset path {candidate} wins"
                )
            return candidate, tried, False, tuple(notes)

    if flat.is_file():
        return flat, tried, bool(candidates), tuple(notes)
    return None, tried, False, tuple(notes)


def has_layout_metadata(model_dir) -> bool:
    """True when the directory declares its own layout (``metadata.pt``/``config.json``).

    Callers that must keep serving the hand-assembled flat directories that
    predate this module — ``model.safetensors`` + ``norm_stats.json`` and nothing
    that says what the checkpoint is — use this to decide whether
    :func:`detect_checkpoint` has anything to work with. Detection itself refuses
    to guess, which is right for a real checkpoint and wrong as a hard regression
    for a directory that used to load.
    """
    root = Path(model_dir)
    return (root / METADATA_NAME).is_file() or (root / CONFIG_NAME).is_file()


def detect_checkpoint(
    model_dir,
    *,
    checkpoint_format: Optional[str] = None,
    asset_id: Optional[str] = None,
    norm_stats=None,
) -> CheckpointLayout:
    """Identify ``model_dir``'s layout and resolve every file it implies.

    ``checkpoint_format`` pins the answer instead of sniffing it (``"auto"`` and
    ``None`` both sniff). ``asset_id`` overrides what ``metadata.pt`` says, for a
    checkpoint whose assets were reorganized after export. ``norm_stats`` is an
    explicit path that outranks every convention.

    Raises :class:`CheckpointError` when the directory matches neither layout, or
    when a pinned format's authoritative file is absent. A **missing
    norm_stats.json is not raised here** — the layout records what was tried and
    :func:`require_norm_stats` turns it into an error, so preflight can report it
    alongside its other findings instead of one crash at a time.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise CheckpointError(f"checkpoint directory {root} does not exist")

    metadata_pt = root / METADATA_NAME
    config_json = root / CONFIG_NAME
    has_metadata, has_config = metadata_pt.is_file(), config_json.is_file()

    resolved = (checkpoint_format or "auto").lower()
    if resolved not in FORMATS:
        raise CheckpointError(
            f"unknown checkpoint format {checkpoint_format!r}; expected one of {list(FORMATS)}"
        )
    if resolved == "auto":
        if has_metadata:
            resolved = OPENPI_PYTORCH
        elif has_config:
            resolved = LEROBOT
        else:
            present = sorted(p.name for p in root.iterdir())[:12]
            raise CheckpointError(
                f"{root} matches neither checkpoint layout apxinf can load:\n"
                f"  {OPENPI_PYTORCH}: needs {METADATA_NAME} "
                f"(+ assets/<asset_id>/{NORM_STATS_NAME})\n"
                f"  {LEROBOT}: needs {CONFIG_NAME} (+ {NORM_STATS_NAME})\n"
                f"the directory holds: {present}"
            )

    notes = []
    arch: Dict[str, Any] = {}
    facts: Dict[str, Any] = {}

    if resolved == OPENPI_PYTORCH:
        if not has_metadata:
            raise CheckpointError(
                f"checkpoint_format={OPENPI_PYTORCH!r} but {metadata_pt} does not "
                f"exist; an openpi PyTorch export always carries it (see openpi "
                f"scripts/train_pytorch.py). Pass checkpoint_format='{LEROBOT}' if "
                f"this is a LeRobot directory."
            )
        try:
            facts = train_config_facts(read_metadata_pt(metadata_pt))
        except MetadataError as exc:
            raise CheckpointError(f"{metadata_pt}: {exc}") from exc
        arch = dict(facts["arch"])
        if has_config:
            # A shipped directory can be exactly this: a hand-added config.json
            # from a *different* training run sitting next to the real metadata.
            notes.append(
                f"{config_json} is ignored — for an openpi export metadata.pt is the "
                f"authority for the architecture"
            )
    else:
        if not has_config:
            raise CheckpointError(
                f"checkpoint_format={LEROBOT!r} but {config_json} does not exist"
            )
        if has_metadata:
            notes.append(
                f"{metadata_pt} is present but ignored because checkpoint_format="
                f"{LEROBOT!r} was requested"
            )
        # arch stays empty on purpose: the Rust loader reads config.json itself,
        # which is the behaviour LeRobot checkpoints already had.

    effective_asset_id = asset_id or facts.get("asset_id")
    asset_id_source = (
        "explicit override" if asset_id else facts.get("asset_id_source", "") or ""
    )

    stats, tried, is_fallback, stats_notes = _resolve_norm_stats(
        root, effective_asset_id, norm_stats
    )
    notes.extend(stats_notes)

    if is_fallback:
        LOGGER.warning(
            "norm_stats: %s (openpi's path for asset_id=%r, from %s) does not exist — "
            "falling back to the checkpoint root %s. Verify those statistics belong to "
            "this robot: a wrong-but-valid file unnormalizes silently.",
            tried[0],
            effective_asset_id,
            asset_id_source or "unknown",
            stats,
        )
    elif stats is not None:
        LOGGER.info("norm_stats: using %s", stats)

    weights = root / WEIGHTS_NAME
    if not weights.is_file():
        notes.append(f"{weights} not found; pass an explicit checkpoint= path")

    return CheckpointLayout(
        format=resolved,
        root=root,
        weights=weights if weights.is_file() else None,
        config_json=config_json if has_config else None,
        metadata_pt=metadata_pt if has_metadata else None,
        asset_id=effective_asset_id,
        asset_id_source=asset_id_source,
        norm_stats=stats,
        norm_stats_tried=tried,
        norm_stats_is_fallback=is_fallback,
        arch=arch,
        openpi={k: v for k, v in facts.items() if k != "arch"},
        notes=tuple(notes),
    )


def require_norm_stats(layout: CheckpointLayout) -> Path:
    """The statistics path, or a :class:`CheckpointError` naming every candidate."""
    if layout.norm_stats is not None:
        return layout.norm_stats

    tried = "\n".join(f"  {path}" for path in layout.norm_stats_tried)
    if layout.format == OPENPI_PYTORCH:
        asset = layout.asset_id or "<unknown>"
        where = layout.openpi.get("assets_dir")
        origin = f" (openpi computed them under {where}{asset}/ on the training machine)" if where else ""
        hint = (
            f"openpi writes this file to assets/<asset_id>/{NORM_STATS_NAME} when "
            f"training finishes (scripts/train_pytorch.py). This checkpoint's asset_id "
            f"is {asset!r}, from {layout.asset_id_source or 'metadata.pt'}{origin}. Ask "
            f"whoever exported it for assets/{asset}/{NORM_STATS_NAME}, place it there, "
            f"or point --norm-stats at it. Do not substitute another run's file: the "
            f"statistics are what map the model's output into the robot's physical "
            f"range, so a wrong one produces in-range, meaningless actions and no error."
        )
    else:
        example = '{"actions": {"q01": [...], "q99": [...]}, "state": {...}}'
        hint = (
            f"LeRobot does not ship a {NORM_STATS_NAME}: it keeps the statistics inside "
            f"policy_preprocessor_step_<N>_normalizer_processor.safetensors — and the "
            f"official pi05 repo deleted even that on 2025-09-24 — or in the source "
            f"dataset's meta/stats.json. apxinf parses neither. Convert them into "
            f"{example} and put it in the checkpoint root, or pass --norm-stats."
        )
    raise CheckpointError(
        f"no {NORM_STATS_NAME} for {layout.root}; tried:\n{tried}\n{hint}"
    )


def resolve_tokenizer(model_dir, tokenizer_path=None, *, env: Optional[Mapping[str, str]] = None) -> Path:
    """Locate the SentencePiece model: explicit path, then env, then the directory.

    ``APXINF_TOKENIZER`` exists so one hand-placed file can serve every
    checkpoint on a box without copying it into each directory.
    """
    import os

    environ = os.environ if env is None else env
    if tokenizer_path is not None:
        path = Path(tokenizer_path)
        if not path.is_file():
            raise CheckpointError(f"tokenizer {path} does not exist")
        return path

    from_env = environ.get("APXINF_TOKENIZER")
    if from_env:
        path = Path(from_env)
        if not path.is_file():
            raise CheckpointError(f"APXINF_TOKENIZER={from_env} does not exist")
        return path

    root = Path(model_dir)
    candidates = [root / name for name in TOKENIZER_NAMES]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = "\n".join(f"  {path}" for path in candidates)
    raise CheckpointError(
        f"no SentencePiece tokenizer for {root}; tried:\n{rendered}\n{_TOKENIZER_HELP}\n"
        f"Put the file at {root / TOKENIZER_NAMES[1]}, or set APXINF_TOKENIZER, or pass "
        f"--tokenizer. Both servers in a comparison must use the *same* file or their "
        f"token ids are not comparable."
    )


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m apxinf.checkpoints <dir>`` — inspect a checkpoint, load nothing."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m apxinf.checkpoints",
        description=(
            "Report how apxinf will read a checkpoint directory: which layout it is, "
            "which norm_stats.json will actually be used, and what architecture the "
            "loader will be given. Reads no weights and needs neither torch nor CUDA."
        ),
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--ckpt-format", choices=FORMATS, default="auto")
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--norm-stats", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        layout = detect_checkpoint(
            args.model_dir,
            checkpoint_format=args.ckpt_format,
            asset_id=args.asset_id,
            norm_stats=args.norm_stats,
        )
    except CheckpointError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(layout.describe())
    print(f"config_json:  {layout.config_json_text() or '(read config.json in the loader)'}")

    status = 0
    for label, resolve in (
        ("norm_stats", lambda: require_norm_stats(layout)),
        ("tokenizer", lambda: resolve_tokenizer(args.model_dir, args.tokenizer)),
    ):
        try:
            resolve()
        except CheckpointError as exc:
            print(f"\nFAIL [{label}]: {exc}")
            status = 1
    if status == 0:
        print("\nOK: this checkpoint has everything apxinf needs to serve it.")
    return status


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_main())
