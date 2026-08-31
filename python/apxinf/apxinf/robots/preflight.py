"""Startup checks: does this checkpoint actually match the preset serving it?

A checkpoint directory and a ``--robot`` preset are two independent claims about
the same robot, and nothing forces them to agree. Serve a Unitree G1 checkpoint
with a LIBERO ``norm_stats.json`` and ``--action-dim 7`` and every layer accepts
it: the weights load, the tokenizer runs, the unnormalizer finds ``q01``/``q99``
of *some* width, the server publishes an action shape, and the robot receives 7
plausible-looking floats per step that mean nothing. The only visible symptom is
that the model "got worse" — which reads as a model problem, gets escalated as a
model problem, and is not one.

That exact combination shipped. This module is the check that would have refused
it, phrased as a list of :class:`Finding` s rather than an exception, so a caller
can report all of them at once instead of fixing them one crash at a time.

**What is checked here** is what can be read from the checkpoint *directory* with
the standard library: which layout it is, where its ``norm_stats.json`` actually
lives, that file's widths and keys, and the tokenizer. Nothing loads a weight and
nothing imports torch or the CUDA binding, so these checks run before the
expensive part of startup — and on a laptop. The richer cross-check against the
preset's own wire keys and delta convention lives in
``scripts/openpi_metadata_to_apxinf.py``, which calls this module and adds to it.

**What is deliberately not checked here** is anything already enforced elsewhere:
``len(image_keys) == model.num_views`` raises inside
:class:`~apxinf.policies.impls.pi05.Pi05Policy`, and the unnormalizer's own width
rules live in :mod:`apxinf.processors.normalize`. Duplicating them would mean two
places to keep in step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .presets import RobotPreset, get_robot_preset
from ..checkpoints import (
    CheckpointError,
    CheckpointLayout,
    TOKENIZER_NAMES,
    detect_checkpoint,
    has_layout_metadata,
    require_norm_stats,
)

__all__ = ["Finding", "FAIL", "WARN", "INFO", "check_checkpoint", "format_findings"]

#: A mismatch that makes the served actions meaningless. Refuse to start.
FAIL = "FAIL"
#: A mismatch that is survivable but is very likely not what the operator meant.
WARN = "WARN"
#: Context worth printing, not a problem.
INFO = "INFO"

_ORDER = {FAIL: 0, WARN: 1, INFO: 2}


@dataclass(frozen=True)
class Finding:
    """One check result: its severity, what it looked at, and what to do."""

    level: str
    check: str
    detail: str
    #: What the operator should change. Empty for :data:`INFO`.
    remedy: str = ""

    def __str__(self) -> str:
        line = f"[{self.level:4}] {self.check}: {self.detail}"
        return f"{line}\n         -> {self.remedy}" if self.remedy else line


def _width(stats: dict, key: str) -> Optional[int]:
    """Length of any one of the four stat vectors, or ``None`` if unreadable."""
    for name in ("q01", "q99", "mean", "std"):
        value = stats.get(name)
        if isinstance(value, (list, tuple)):
            return len(value)
    return None


def _check_layout(layout: CheckpointLayout) -> List[Finding]:
    """Report what the directory says it is, before anything reads it.

    The layout decides which ``norm_stats.json`` gets used, so an operator who
    can see the chosen path can spot the wrong-statistics failure by eye — which
    is the one failure mode this whole module exists for.
    """
    findings = [Finding(INFO, "checkpoint layout", f"{layout.format} ({layout.root})")]
    if layout.asset_id:
        findings.append(
            Finding(
                INFO,
                "asset_id",
                f"{layout.asset_id!r} (from {layout.asset_id_source or 'metadata.pt'})",
            )
        )
    if layout.arch:
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(layout.arch.items()))
        findings.append(Finding(INFO, "architecture", f"from metadata.pt: {rendered}"))
    for note in layout.notes:
        findings.append(Finding(INFO, "checkpoint layout", note))
    return findings


def _check_norm_stats(
    model_dir: Path,
    preset: RobotPreset,
    *,
    norm_key: str,
    discrete_state: bool,
    layout: Optional[CheckpointLayout] = None,
    norm_stats=None,
) -> List[Finding]:
    """Are the shipped statistics this robot's statistics?

    ``layout`` supplies the resolved path. An openpi export keeps its statistics
    under ``assets/<asset_id>/``, so reading ``<model_dir>/norm_stats.json``
    unconditionally is how the wrong file got checked *and* served: whatever a
    previous run happened to leave in the root. With no layout — a flat directory
    that declares nothing — ``norm_stats`` still names the file directly, which
    is what :meth:`Pi05Policy.from_pretrained` does in the same situation.
    """
    findings: List[Finding] = []
    if layout is not None:
        try:
            path = require_norm_stats(layout)
        except CheckpointError as exc:
            return [
                Finding(
                    FAIL,
                    "norm_stats.json",
                    str(exc),
                    "unnormalization has no statistics to use; ship the file with the "
                    "checkpoint or point --norm-stats at it",
                )
            ]
        if layout.norm_stats_is_fallback:
            findings.append(
                Finding(
                    WARN,
                    "norm_stats.json",
                    f"using the checkpoint root {path} because {layout.norm_stats_tried[0]} "
                    f"(openpi's path for asset_id={layout.asset_id!r}) does not exist",
                    "verify these statistics belong to this robot. A file from another "
                    "run is syntactically valid and unnormalizes silently; the width "
                    "check below is the only thing that would catch it.",
                )
            )
        else:
            findings.append(Finding(INFO, "norm_stats.json", str(path)))
    elif norm_stats is not None:
        path = Path(norm_stats)
        if not path.is_file():
            return [
                Finding(
                    FAIL,
                    "norm_stats.json",
                    f"--norm-stats {path} does not exist",
                    "fix the path",
                )
            ]
        findings.append(Finding(INFO, "norm_stats.json", f"{path} (explicit)"))
    else:
        path = model_dir / "norm_stats.json"
        if not path.exists():
            return [
                Finding(
                    FAIL,
                    "norm_stats.json",
                    f"missing from {model_dir}",
                    "unnormalization has no statistics to use; ship the file with the checkpoint",
                )
            ]
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return [Finding(FAIL, "norm_stats.json", f"unreadable: {exc}", "fix or re-export the file")]

    stats = document.get("norm_stats", document)

    # Width per key against what the preset says the robot is. This is the check
    # that catches a G1 checkpoint carrying LIBERO statistics: 16-dim robot,
    # 8/7-dim stats. Both directions are fatal -- too narrow and the unnormalizer
    # refuses the array outright, too wide and it silently unnormalizes dimensions
    # the robot does not have.
    for key, expected in (
        (norm_key, preset.action_width),
        ("state", preset.state_dim if discrete_state else None),
    ):
        if expected is None:
            continue
        entry = stats.get(key)
        if not isinstance(entry, dict):
            findings.append(
                Finding(
                    FAIL,
                    f"norm_stats[{key!r}]",
                    f"absent; file has {sorted(stats)}",
                    f"{preset.name} needs {key!r} statistics"
                    + ("" if key == norm_key else " because state is discretized into the prompt"),
                )
            )
            continue
        got = _width(entry, key)
        if got is None:
            findings.append(
                Finding(FAIL, f"norm_stats[{key!r}]", "has no vector-valued stat", "re-export"),
            )
        elif got != expected:
            findings.append(
                Finding(
                    FAIL,
                    f"norm_stats[{key!r}] width",
                    f"{got}, but preset {preset.name!r} is a {expected}-dim robot",
                    f"these statistics are not this robot's. Serving them maps the "
                    f"model's output through the wrong physical range -- the actions "
                    f"stay in range and stay wrong. Point --model-dir at a checkpoint "
                    f"whose norm_stats.json is {expected} wide.",
                )
            )
        else:
            findings.append(Finding(INFO, f"norm_stats[{key!r}] width", f"{got}, matches preset"))

        # pi05 is always quantile-normalized (openpi derives this from the model
        # type, not from any file: `use_quantile_norm = model_type != PI0`), so
        # q01/q99 are the stats that actually get used. openpi asserts their
        # presence; apxinf's Unnormalizer would raise a KeyError instead.
        if isinstance(entry, dict) and not ("q01" in entry and "q99" in entry):
            findings.append(
                Finding(
                    FAIL,
                    f"norm_stats[{key!r}] quantiles",
                    f"no q01/q99; has {sorted(entry)}",
                    "pi05 always unnormalizes with quantiles regardless of what any "
                    "config says (openpi training/config.py: use_quantile_norm = "
                    "model_type != PI0). mean/std alone cannot serve this checkpoint.",
                )
            )
    return findings


def _check_tokenizer(model_dir: Path, tokenizer_path) -> List[Finding]:
    """Is there a tokenizer to load, and is it the *same* one both servers use?"""
    if tokenizer_path is not None:
        path = Path(tokenizer_path)
        if not path.exists():
            return [
                Finding(FAIL, "tokenizer", f"--tokenizer {path} does not exist", "fix the path")
            ]
        return [Finding(INFO, "tokenizer", f"{path} (explicit)")]

    found = [name for name in TOKENIZER_NAMES if (model_dir / name).exists()]
    if found:
        return [Finding(INFO, "tokenizer", f"{model_dir / found[0]}")]
    return [
        Finding(
            FAIL,
            "tokenizer",
            f"none of {list(TOKENIZER_NAMES)} in {model_dir}",
            "the checkpoint does not carry its tokenizer. openpi downloads "
            "paligemma_tokenizer.model at runtime, so a checkpoint exported from it "
            "will not have one. Copy it into the checkpoint directory or pass "
            "--tokenizer. Both servers must use the *same* file or their token ids "
            "are not comparable.",
        )
    ]


def _check_cameras(preset: RobotPreset, image_keys: Sequence[str]) -> List[Finding]:
    """Camera count and ordering against the preset's declared view slots."""
    keys = tuple(image_keys)
    if keys == preset.image_keys:
        return [Finding(INFO, "cameras", f"{len(keys)} views, preset keys unchanged")]
    if len(keys) != preset.num_views:
        return [
            Finding(
                WARN,
                "cameras",
                f"serving {len(keys)} views {list(keys)} but preset {preset.name!r} "
                f"declares {preset.num_views} {list(preset.image_keys)}",
                "the checkpoint was trained with a fixed number of view slots; pass "
                "--num-views to load it for fewer, or drop the --image-keys override",
            )
        ]
    return [
        Finding(
            WARN,
            "cameras",
            f"--image-keys overrides the preset: {list(keys)} instead of "
            f"{list(preset.image_keys)}",
            "entry i fills model view slot i, so a reordered tuple silently feeds "
            "the wrong camera to each slot. Confirm the order matches "
            f"{[slot for slot, _ in preset.slots]}.",
        )
    ]


def check_checkpoint(
    model_dir,
    robot: str,
    *,
    norm_key: str = "actions",
    discrete_state: Optional[bool] = None,
    image_keys: Optional[Sequence[str]] = None,
    action_dim: Optional[int] = None,
    tokenizer_path=None,
    checkpoint_format: Optional[str] = None,
    asset_id: Optional[str] = None,
    norm_stats=None,
) -> Tuple[Finding, ...]:
    """Check a checkpoint directory against the preset that will serve it.

    Takes the *resolved* serving knobs (what the server settled on after applying
    its overrides), not the raw command line, so what is checked is what will
    actually run. Returns findings sorted most-severe first; an empty tuple is
    impossible because passing checks are reported as :data:`INFO`.

    ``checkpoint_format`` / ``asset_id`` / ``norm_stats`` are the same knobs
    :meth:`~apxinf.policies.impls.pi05.Pi05Policy.from_pretrained` takes, and must
    be passed through identically — this is a preflight for *that* load, so
    checking a different file than the one that will be served defeats the point.
    """
    model_dir = Path(model_dir)
    preset = get_robot_preset(robot)
    discrete = preset.discrete_state if discrete_state is None else bool(discrete_state)
    keys = preset.image_keys if image_keys is None else tuple(image_keys)

    # A directory that declares nothing about itself is the hand-assembled flat
    # layout that predates this check; it still loads, so it is still checked,
    # just against <model_dir>/norm_stats.json (or --norm-stats, which names one
    # file rather than asserting a directory shape). A detection failure is
    # reported rather than raised: preflight's whole contract is to list every
    # problem. This mirrors Pi05Policy.from_pretrained exactly — a preflight that
    # resolves a different file than the load would is worse than none.
    layout: Optional[CheckpointLayout] = None
    layout_findings: List[Finding] = []
    if checkpoint_format or asset_id or has_layout_metadata(model_dir):
        try:
            layout = detect_checkpoint(
                model_dir,
                checkpoint_format=checkpoint_format,
                asset_id=asset_id,
                norm_stats=norm_stats,
            )
        except CheckpointError as exc:
            layout_findings.append(
                Finding(
                    FAIL,
                    "checkpoint layout",
                    str(exc),
                    "apxinf cannot tell what this directory is, so it cannot tell "
                    "which files to read; pass --ckpt-format, or fix the directory",
                )
            )
        else:
            layout_findings.extend(_check_layout(layout))

    findings = [
        *layout_findings,
        *_check_norm_stats(
            model_dir,
            preset,
            norm_key=norm_key,
            discrete_state=discrete,
            layout=layout,
            norm_stats=norm_stats,
        ),
        *_check_tokenizer(model_dir, tokenizer_path),
        *_check_cameras(preset, keys),
    ]

    if action_dim is not None and preset.action_width is not None and action_dim != preset.action_width:
        findings.append(
            Finding(
                WARN,
                "action_dim",
                f"--action-dim {action_dim} overrides preset {preset.name!r}'s "
                f"{preset.action_width}",
                "a width the robot does not have gets truncated or padded somewhere "
                "downstream without an error. Drop the flag unless the deployed "
                "client genuinely expects this width.",
            )
        )
    if discrete and not preset.discrete_state:
        findings.append(
            Finding(INFO, "discrete_state", "enabled by override; preset default is off")
        )
    elif not discrete and preset.discrete_state:
        findings.append(
            Finding(
                WARN,
                "discrete_state",
                f"disabled by override, but preset {preset.name!r} discretizes state "
                "into the prompt",
                "state is not merely un-discretized, it is *dropped*: the model sees "
                "no proprioception, and any delta->absolute output step becomes a "
                "no-op because it has no base to add.",
            )
        )

    return tuple(sorted(findings, key=lambda f: _ORDER.get(f.level, 3)))


def format_findings(findings: Sequence[Finding], *, include_info: bool = True) -> str:
    """Render findings for a log or a terminal, most severe first."""
    shown = [f for f in findings if include_info or f.level != INFO]
    return "\n".join(str(f) for f in shown)
