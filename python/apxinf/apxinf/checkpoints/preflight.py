"""Inspect a checkpoint directory without loading weights.

This is the half of the startup preflight that depends only on the
**checkpoint**: which layout the directory is, where its normalization
statistics resolved from and how wide they are, whether pi05's mandatory
quantiles are present, and whether a tokenizer is there to load. Nothing here
imports torch, touches CUDA, or reads a weight.

What is deliberately *not* here is the other half: whether those facts match the
body being served. "This checkpoint's actions are 7 wide" is a checkpoint fact;
"...but this robot is 16-DoF" is a claim about a robot, and the engine has no
robots. A caller that has an expected width compares it against
:attr:`CheckpointReport.norm` and emits its own :class:`Finding` -- which is why
:data:`FAIL` / :data:`WARN` / :data:`INFO`, :class:`Finding`,
:func:`sort_findings` and :func:`format_findings` are exported: the checkpoint
half and the body half have to produce one report, so the vocabulary lives on
the engine side and the caller borrows it.

Model view-count validation remains in
:class:`~apxinf.policies.impls.pi05.Pi05Policy`; normalization value validation
remains in :mod:`apxinf.processors.normalize`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .descriptor import IDENTITY_MISSING_STATS
from .layout import (
    TOKENIZER_NAMES,
    CheckpointError,
    CheckpointLayout,
    detect_checkpoint,
    has_layout_metadata,
)
from .norm_stats import read_norm_stats

__all__ = [
    "FAIL",
    "WARN",
    "INFO",
    "Finding",
    "NormFacts",
    "CheckpointReport",
    "inspect_checkpoint",
    "sort_findings",
    "format_findings",
]

#: A mismatch that makes the served actions meaningless. Refuse to start.
FAIL = "FAIL"
#: A mismatch that is survivable but is very likely not what the operator meant.
WARN = "WARN"
#: Context worth printing, not a problem.
INFO = "INFO"

_ORDER = {FAIL: 0, WARN: 1, INFO: 2}

#: Role -> the human label a normalization finding is filed under when the
#: statistics came from a declared layout plan rather than a raw JSON file.
_PLAN_LABEL = {"action": "action normalization", "state": "state normalization"}


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


@dataclass(frozen=True)
class NormFacts:
    """What the checkpoint says about one normalization key.

    Facts, not judgments: a caller with an expected width turns these into
    findings. :attr:`check` is the name that caller should file its finding
    under, so that a plan-derived finding reads ``action normalization width``
    and a file-derived one reads ``norm_stats['actions'] width``.
    """

    #: ``"action"`` or ``"state"`` — which of the two the caller asked about.
    role: str
    #: The wire key these statistics were looked up under.
    key: str
    #: Base check name for findings about this key, in this layout.
    check: str
    #: Whether any statistics entry exists for this key at all.
    present: bool
    #: Observed width. ``None`` when the entry is absent, or present but carries
    #: no vector-valued stat — either way a finding was already emitted, so a
    #: caller comparing against an expected width should skip a ``None``.
    width: Optional[int] = None
    #: ``"quantile"`` / ``"mean_std"`` / ... when a plan resolved it.
    mode: Optional[str] = None
    #: Where the resolved statistics came from, for logs.
    source: Optional[str] = None
    #: Whether this key will fall through as identity (no usable statistics).
    identity: bool = False


@dataclass(frozen=True)
class CheckpointReport:
    """Everything :func:`inspect_checkpoint` could determine from disk."""

    #: Checkpoint-only findings, in emission order (not sorted — a caller merges
    #: its own findings in and sorts once, via :func:`sort_findings`).
    findings: Tuple[Finding, ...]
    #: The detected layout, or ``None`` for a flat directory that declares none.
    layout: Optional[CheckpointLayout]
    #: Role -> observed statistics facts, for the roles that were inspected.
    norm: Mapping[str, NormFacts]
    #: The resolved tokenizer file, or ``None`` if none was found.
    tokenizer: Optional[Path]
    #: The statistics file that was actually read, when one was.
    norm_stats_path: Optional[Path]

    @property
    def worst_level(self) -> Optional[str]:
        """The most severe level present, or ``None`` when there are no findings."""
        if not self.findings:
            return None
        return min((f.level for f in self.findings), key=lambda lvl: _ORDER.get(lvl, 3))


def _width(stats: Mapping) -> Optional[int]:
    """Length of any one of the four stat vectors, or ``None`` if unreadable."""
    for name in ("q01", "q99", "mean", "std"):
        value = stats.get(name)
        if isinstance(value, (list, tuple)):
            return len(value)
    return None


def _check_layout(layout: CheckpointLayout) -> List[Finding]:
    """Report the detected layout and selected checkpoint assets."""
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


def _check_tokenizer(model_dir: Path, tokenizer_path) -> Tuple[List[Finding], Optional[Path]]:
    """Is there a tokenizer to load, and is it the *same* one both servers use?"""
    if tokenizer_path is not None:
        path = Path(tokenizer_path)
        if not path.exists():
            return (
                [Finding(FAIL, "tokenizer", f"--tokenizer {path} does not exist", "fix the path")],
                None,
            )
        return [Finding(INFO, "tokenizer", f"{path} (explicit)")], path

    found = [name for name in TOKENIZER_NAMES if (model_dir / name).exists()]
    if found:
        path = model_dir / found[0]
        return [Finding(INFO, "tokenizer", f"{path}")], path
    return (
        [
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
        ],
        None,
    )


def _plan_facts(
    layout: CheckpointLayout, roles: Sequence[Tuple[str, str]]
) -> Tuple[List[Finding], Dict[str, NormFacts]]:
    """Read a declared layout's normalization plan into findings + facts."""
    findings: List[Finding] = []
    facts: Dict[str, NormFacts] = {}
    plan = layout.normalization
    specs = {"action": plan.action, "state": plan.state}
    for role, key in roles:
        label = _PLAN_LABEL[role]
        spec = specs.get(role)
        if spec is None:
            findings.append(
                Finding(
                    WARN,
                    label,
                    "identity passthrough; no statistics were declared",
                    "this is load-compatible but does not establish embodiment-level "
                    "parity; supply matching statistics for deployment claims",
                )
            )
            facts[role] = NormFacts(role, key, label, present=False, identity=True)
            continue
        if spec.status == IDENTITY_MISSING_STATS:
            findings.append(
                Finding(
                    WARN,
                    label,
                    f"identity passthrough at width {spec.width}; processor state is absent",
                    "this matches LeRobot's load behavior but does not establish "
                    "embodiment-level parity; supply a fine-tuned checkpoint with "
                    "processor state for deployment claims",
                )
            )
        else:
            findings.append(
                Finding(INFO, label, f"{spec.mode}, width {spec.width}, from {spec.source}")
            )
        facts[role] = NormFacts(
            role,
            key,
            label,
            present=True,
            width=spec.width,
            mode=spec.mode,
            source=spec.source,
            identity=spec.status == IDENTITY_MISSING_STATS,
        )
    return findings, facts


def _file_facts(
    stats: Mapping, roles: Sequence[Tuple[str, str]]
) -> Tuple[List[Finding], Dict[str, NormFacts]]:
    """Read a raw ``norm_stats.json`` into findings + facts."""
    findings: List[Finding] = []
    facts: Dict[str, NormFacts] = {}
    for role, key in roles:
        check = f"norm_stats[{key!r}]"
        entry = stats.get(key)
        if not isinstance(entry, dict):
            # Actions are mandatory; state falls through to identity, which is
            # survivable. That split is a property of the model's input contract,
            # not of whatever body is on the other end of the wire.
            is_action = role == "action"
            findings.append(
                Finding(
                    FAIL if is_action else WARN,
                    check,
                    f"absent; file has {sorted(stats)}"
                    + ("" if is_action else "; state will use identity passthrough"),
                    (
                        f"the checkpoint needs {key!r} action statistics to unnormalize"
                        if is_action
                        else "this is load-compatible but does not establish "
                        "embodiment-level parity"
                    ),
                )
            )
            facts[role] = NormFacts(role, key, check, present=False, identity=True)
            continue
        got = _width(entry)
        if got is None:
            findings.append(
                Finding(FAIL, check, "has no vector-valued stat", "re-export"),
            )
        # pi05 is always quantile-normalized (openpi derives this from the model
        # type, not from any file: `use_quantile_norm = model_type != PI0`), so
        # q01/q99 are the stats that actually get used. openpi asserts their
        # presence; apxinf's Unnormalizer would raise a KeyError instead.
        if not ("q01" in entry and "q99" in entry):
            findings.append(
                Finding(
                    FAIL,
                    f"{check} quantiles",
                    f"no q01/q99; has {sorted(entry)}",
                    "pi05 always unnormalizes with quantiles regardless of what any "
                    "config says (openpi training/config.py: use_quantile_norm = "
                    "model_type != PI0). mean/std alone cannot serve this checkpoint.",
                )
            )
        facts[role] = NormFacts(
            role,
            key,
            check,
            present=True,
            width=got,
            mode="quantile" if "q01" in entry and "q99" in entry else "mean_std",
        )
    return findings, facts


def _plan_provenance(layout: CheckpointLayout) -> List[Finding]:
    """Where a declared layout's already-resolved plan read its numbers from.

    Nothing is reported when there is no file: the plan's own per-key findings
    already say "identity passthrough", and a second line saying the file is
    missing only repeats it.
    """
    if layout.norm_stats is None:
        return []
    if layout.norm_stats_is_fallback:
        return [
            Finding(
                WARN,
                "norm_stats.json",
                f"using the checkpoint root {layout.norm_stats} because "
                f"{layout.norm_stats_tried[0]} (openpi's path for "
                f"asset_id={layout.asset_id!r}) does not exist",
                "verify these statistics belong to this robot",
            )
        ]
    return [Finding(INFO, "norm_stats.json", str(layout.norm_stats))]


def _locate_stats_file(
    model_dir: Path, layout: Optional[CheckpointLayout], norm_stats
) -> Tuple[List[Finding], Optional[Path]]:
    """Find the statistics file to read, mirroring how the policy would load it.

    A ``None`` path means there is nothing further to read; the returned findings
    say why (missing, so identity passthrough — or an explicit path that is not
    there, which is fatal).
    """
    if layout is not None:
        if layout.norm_stats is None:
            return (
                [
                    Finding(
                        WARN,
                        "norm_stats.json",
                        "missing; state and actions will use identity passthrough",
                        "this matches stateless checkpoint loading but does not establish "
                        "embodiment-level parity; supply matching statistics for "
                        "deployment claims",
                    )
                ],
                None,
            )
        path = layout.norm_stats
        if layout.norm_stats_is_fallback:
            return (
                [
                    Finding(
                        WARN,
                        "norm_stats.json",
                        f"using the checkpoint root {path} because {layout.norm_stats_tried[0]} "
                        f"(openpi's path for asset_id={layout.asset_id!r}) does not exist",
                        "verify these statistics belong to this robot. A file from another "
                        "run is syntactically valid and unnormalizes silently; the width "
                        "check below is the only thing that would catch it.",
                    )
                ],
                path,
            )
        return [Finding(INFO, "norm_stats.json", str(path))], path

    if norm_stats is not None:
        path = Path(norm_stats)
        if not path.is_file():
            return (
                [
                    Finding(
                        FAIL,
                        "norm_stats.json",
                        f"--norm-stats {path} does not exist",
                        "fix the path",
                    )
                ],
                None,
            )
        return [Finding(INFO, "norm_stats.json", f"{path} (explicit)")], path

    path = model_dir / "norm_stats.json"
    if not path.exists():
        return (
            [
                Finding(
                    WARN,
                    "norm_stats.json",
                    f"missing from {model_dir}; state and actions will use identity passthrough",
                    "this is load-compatible but does not establish embodiment-level parity",
                )
            ],
            None,
        )
    return [], path


def inspect_checkpoint(
    model_dir,
    *,
    norm_key: Optional[str] = "actions",
    state_norm_key: Optional[str] = None,
    tokenizer_path=None,
    checkpoint_format: Optional[str] = None,
    asset_id: Optional[str] = None,
    norm_stats=None,
) -> CheckpointReport:
    """Read everything a checkpoint directory states about itself.

    ``norm_key`` / ``state_norm_key`` name the statistics keys to inspect;
    ``None`` means "this deployment does not use that one", and nothing about it
    is reported. A server that drops proprioception passes
    ``state_norm_key=None`` and gets no state findings, which is the point:
    a warning about statistics nobody will read is noise.

    ``checkpoint_format`` / ``asset_id`` / ``norm_stats`` must match the values
    passed to :meth:`~apxinf.policies.impls.pi05.Pi05Policy.from_pretrained`, so
    that what is inspected is what will actually load.

    Findings come back in emission order, not sorted: a caller with its own
    checks merges them in and calls :func:`sort_findings` once.
    """
    model_dir = Path(model_dir)
    roles: List[Tuple[str, str]] = []
    if norm_key is not None:
        roles.append(("action", norm_key))
    if state_norm_key is not None:
        roles.append(("state", state_norm_key))

    # Match policy loading: declared layouts are detected; flat native
    # directories use a root or explicitly named norm_stats file.
    layout: Optional[CheckpointLayout] = None
    findings: List[Finding] = []
    if checkpoint_format or asset_id or has_layout_metadata(model_dir):
        try:
            layout = detect_checkpoint(
                model_dir,
                checkpoint_format=checkpoint_format,
                asset_id=asset_id,
                norm_stats=norm_stats,
                norm_key=norm_key or "actions",
                state_norm_key=state_norm_key,
            )
        except CheckpointError as exc:
            findings.append(
                Finding(
                    FAIL,
                    "checkpoint layout",
                    str(exc),
                    "apxinf cannot tell what this directory is, so it cannot tell "
                    "which files to read; pass --ckpt-format, or fix the directory",
                )
            )
        else:
            findings.extend(_check_layout(layout))

    facts: Dict[str, NormFacts] = {}
    path: Optional[Path] = None
    if layout is not None and layout.normalization is not None:
        # The layout already resolved its own plan; the file behind it is
        # reported for provenance and never re-read.
        path = layout.norm_stats
        findings.extend(_plan_provenance(layout))
        plan_findings, facts = _plan_facts(layout, roles)
        findings.extend(plan_findings)
    else:
        located, path = _locate_stats_file(model_dir, layout, norm_stats)
        findings.extend(located)
        if path is not None:
            try:
                stats = read_norm_stats(path)
            except ValueError as exc:
                findings.append(
                    Finding(
                        FAIL,
                        "norm_stats.json",
                        f"unreadable: {exc}",
                        "fix or re-export the file",
                    )
                )
            else:
                file_findings, facts = _file_facts(stats, roles)
                findings.extend(file_findings)

    tokenizer_findings, tokenizer = _check_tokenizer(model_dir, tokenizer_path)
    findings.extend(tokenizer_findings)

    return CheckpointReport(
        findings=tuple(findings),
        layout=layout,
        norm=facts,
        tokenizer=tokenizer,
        norm_stats_path=path,
    )


def sort_findings(findings: Sequence[Finding]) -> Tuple[Finding, ...]:
    """Order findings most-severe first, preserving emission order within a level."""
    return tuple(sorted(findings, key=lambda f: _ORDER.get(f.level, 3)))


def format_findings(findings: Sequence[Finding], *, include_info: bool = True) -> str:
    """Render findings for a log or a terminal, most severe first."""
    shown = [f for f in findings if include_info or f.level != INFO]
    return "\n".join(str(f) for f in shown)
