"""Checkpoint layout detection and canonical metadata descriptors.

    from apxinf.checkpoints import detect_checkpoint

    layout = detect_checkpoint(model_dir)
    normalization = layout.normalization  # None means identity passthrough
    config_json = layout.config_json_text() # -> apxinf_py.Model.load(config_json=...)

:func:`inspect_checkpoint` is the same information rendered as a preflight
report -- what resolved, from where, and what is missing -- for a server to log
before it loads anything.

Run ``python -m apxinf.checkpoints <dir>`` to see the same answer as a report.
Nothing here loads weights, imports torch, or touches the network.
"""

from __future__ import annotations

from .descriptor import NormalizationPlan, TokenizerSpec, TransformSpec

from .layout import (
    FORMATS,
    LEROBOT,
    NORM_STATS_NAME,
    OPENPI_PYTORCH,
    TOKENIZER_NAMES,
    CheckpointError,
    CheckpointLayout,
    detect_checkpoint,
    has_layout_metadata,
    resolve_tokenizer,
)
from .metadata import (
    MetadataError,
    read_metadata_pt,
    repack_structure,
    train_config_facts,
)
from .norm_stats import NormStatsError, read_norm_stats
from .openpi import OpenPINormalizationError, load_normalization_plan
from .preflight import (
    FAIL,
    INFO,
    WARN,
    CheckpointReport,
    Finding,
    NormFacts,
    format_findings,
    inspect_checkpoint,
    sort_findings,
)

__all__ = [
    "CheckpointError",
    "CheckpointLayout",
    "CheckpointReport",
    "FAIL",
    "FORMATS",
    "Finding",
    "INFO",
    "LEROBOT",
    "MetadataError",
    "NormFacts",
    "NormStatsError",
    "NORM_STATS_NAME",
    "NormalizationPlan",
    "OPENPI_PYTORCH",
    "OpenPINormalizationError",
    "TOKENIZER_NAMES",
    "TokenizerSpec",
    "TransformSpec",
    "WARN",
    "detect_checkpoint",
    "format_findings",
    "has_layout_metadata",
    "inspect_checkpoint",
    "read_metadata_pt",
    "read_norm_stats",
    "repack_structure",
    "resolve_tokenizer",
    "sort_findings",
    "train_config_facts",
    "load_normalization_plan",
]
