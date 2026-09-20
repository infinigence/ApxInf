"""Detect supported checkpoint layouts and resolve their assets.

Supported layouts:

``openpi_pytorch``
    What ``scripts/train_pytorch.py`` writes and what ships to a customer:
    ``model.safetensors`` + ``metadata.pt`` (+ ``optimizer.pt``) with the
    statistics under ``assets/<asset_id>/norm_stats.json``. **There is no
    config.json**, so every architecture constant has to come out of
    ``metadata.pt``.

``lerobot``
    A HuggingFace robot-policy directory: ``config.json`` with ``"type": "pi05"``
    plus ``model.safetensors``. Current repositories serialize their processor
    topology in ``policy_{pre,post}processor.json`` and, for fine-tunes, point to
    normalization state in separate SafeTensors sidecars. Base repositories may
    deliberately omit that state, in which case LeRobot uses identity transforms.

The resolver is shared by policy loading, preflight, and offline inspection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .descriptor import IDENTITY_MISSING_STATS, NormalizationPlan, TokenizerSpec
from .lerobot import LeRobotProcessorError, has_processor_layout, load_processor_plan
from .metadata import MetadataError, read_metadata_pt, train_config_facts
from .openpi import OpenPINormalizationError, load_normalization_plan

__all__ = [
    "CheckpointError",
    "CheckpointLayout",
    "LEROBOT",
    "NORM_STATS_NAME",
    "OPENPI_PYTORCH",
    "TOKENIZER_NAMES",
    "detect_checkpoint",
    "has_layout_metadata",
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
    #: The selected OpenPI-style statistics file. ``None`` when a serialized
    #: processor is authoritative or when no statistics were found.
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
    #: Layout-neutral processor facts. Present for a serialized LeRobot pipeline.
    normalization: Optional[NormalizationPlan] = None
    #: Upstream tokenizer identity, which an offline caller maps to a local file.
    tokenizer: Optional[TokenizerSpec] = None

    def config_json_text(self) -> Optional[str]:
        """The ``config_json=`` string to hand ``apxinf_py.ModelRunner.load``.

        ``None`` lets the Rust loader read ``config.json``.
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
        if self.normalization is not None:
            state = self.normalization.state
            state_text = (
                "(not declared)"
                if state is None
                else f"{state.mode}/{state.status} width={state.width}"
            )
            action = self.normalization.action
            lines.append(f"state norm:   {state_text}")
            lines.append(
                f"action norm:  {action.mode}/{action.status} width={action.width}"
            )
        if self.tokenizer is not None:
            lines.append(f"tokenizer id: {self.tokenizer.name}")
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
    """True when the directory declares a supported external checkpoint layout.

    Native flat directories with an architecture-only ``config.json`` do not
    declare an external layout and return ``False``.
    """
    root = Path(model_dir)
    if (root / METADATA_NAME).is_file():
        return True
    config_path = root / CONFIG_NAME
    if not config_path.is_file():
        return False
    try:
        document = json.loads(config_path.read_text())
    except (OSError, ValueError):
        # Let detect_checkpoint produce the actionable parse error.
        return True
    if not isinstance(document, dict):
        return True
    if document.get("type") == "pi05":
        return True
    # ApxInf native directories use this compact Rust architecture config. It is
    # consumed by Model.load and is not a LeRobot manifest.
    native_markers = {"action_dim", "action_horizon", "paligemma_variant"}
    return not native_markers.issubset(document)


def detect_checkpoint(
    model_dir,
    *,
    checkpoint_format: Optional[str] = None,
    asset_id: Optional[str] = None,
    norm_stats=None,
    norm_key: str = "actions",
    state_norm_key: Optional[str] = "state",
) -> CheckpointLayout:
    """Identify ``model_dir``'s layout and resolve every file it implies.

    ``checkpoint_format`` pins the answer instead of sniffing it (``"auto"`` and
    ``None`` both sniff). ``asset_id`` overrides what ``metadata.pt`` says, for a
    checkpoint whose assets were reorganized after export. ``norm_stats`` is an
    explicit path that outranks every convention. ``norm_key`` and
    ``state_norm_key`` select the OpenPI feature entries translated into the
    canonical :class:`NormalizationPlan`; ``None`` omits the state transform.

    Raises :class:`CheckpointError` when the directory matches neither layout, or
    when a pinned format's authoritative file is absent. A **missing
    norm_stats.json is not raised here** — the layout records what was tried so
    policy loading and preflight can apply their own missing-statistics policy.
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
                f"  {LEROBOT}: needs {CONFIG_NAME} (processor sidecars are optional)\n"
                f"the directory holds: {present}"
            )

    notes = []
    arch: Dict[str, Any] = {}
    facts: Dict[str, Any] = {}
    normalization = None
    tokenizer = None

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
            notes.append(
                f"{config_json} is ignored — for an openpi export metadata.pt is the "
                f"authority for the architecture"
            )
    else:
        if not has_config:
            raise CheckpointError(
                f"checkpoint_format={LEROBOT!r} but {config_json} does not exist"
            )
        try:
            config_document = json.loads(config_json.read_text())
        except (OSError, ValueError) as exc:
            raise CheckpointError(f"{config_json}: {exc}") from exc
        if not isinstance(config_document, dict) or config_document.get("type") != "pi05":
            declared = config_document.get("type") if isinstance(config_document, dict) else None
            raise CheckpointError(
                f"{config_json} is not a supported LeRobot PI0.5 config "
                f"(expected type='pi05', got {declared!r})"
            )
        if has_metadata:
            notes.append(
                f"{metadata_pt} is present but ignored because checkpoint_format="
                f"{LEROBOT!r} was requested"
            )
        # arch stays empty on purpose: the Rust loader reads config.json itself,
        # which is the behaviour LeRobot checkpoints already had.
        if has_processor_layout(root):
            try:
                processor_normalization, tokenizer = load_processor_plan(root, config_document)
            except LeRobotProcessorError as exc:
                raise CheckpointError(str(exc)) from exc
            # An explicit norm_stats path overrides normalization only. The
            # processor manifests remain authoritative for independent facts
            # such as tokenizer identity and must still be validated.
            if norm_stats is None:
                normalization = processor_normalization
                if any(
                    spec is not None and spec.status == IDENTITY_MISSING_STATS
                    for spec in (normalization.state, normalization.action)
                ):
                    notes.append(
                        "LeRobot processor state is absent for one or more transforms; "
                        "matching upstream identity passthrough for those transforms. "
                        "This is not checkpoint-equivalent to an embodiment with stats."
                    )

    effective_asset_id = asset_id or facts.get("asset_id")
    asset_id_source = (
        "explicit override" if asset_id else facts.get("asset_id_source", "") or ""
    )

    if normalization is None:
        stats, tried, is_fallback, stats_notes = _resolve_norm_stats(
            root, effective_asset_id, norm_stats
        )
    else:
        # The LeRobot processor sidecars already selected and serialized their
        # own state. A stray root norm_stats.json is not an alternative source.
        stats, tried, is_fallback, stats_notes = None, (), False, ()
    notes.extend(stats_notes)

    # A serialized LeRobot processor is authoritative for that family.  Every
    # other PI0.5 norm_stats.json uses OpenPI's quantile semantics and becomes
    # the same canonical plan before any policy sees it.
    if normalization is None and stats is not None:
        try:
            normalization = load_normalization_plan(
                stats, action_key=norm_key, state_key=state_norm_key
            )
        except OpenPINormalizationError as exc:
            raise CheckpointError(str(exc)) from exc

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
        normalization=normalization,
        tokenizer=tokenizer,
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
            "which normalization source will actually be used, and what architecture "
            "the loader will be given. Reads no model weights and needs neither torch "
            "nor CUDA."
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
    missing_normalization = layout.normalization is None or any(
        spec is not None and spec.status == IDENTITY_MISSING_STATS
        for spec in (
            None if layout.normalization is None else layout.normalization.state,
            None if layout.normalization is None else layout.normalization.action,
        )
    )
    if missing_normalization:
        print(
            "\nWARN [normalization]: statistics are missing for one or more "
            "transforms; those values use identity passthrough. This is "
            "load-compatible, not evidence of "
            "embodiment-level parity."
        )
    for label, resolve in (
        ("tokenizer", lambda: resolve_tokenizer(args.model_dir, args.tokenizer)),
    ):
        try:
            resolve()
        except CheckpointError as exc:
            print(f"\nFAIL [{label}]: {exc}")
            status = 1
    if status == 0:
        if missing_normalization:
            print("\nOK: checkpoint is loadable; review the normalization warning above.")
        else:
            print("\nOK: this checkpoint has everything apxinf needs to serve it.")
    return status


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_main())
