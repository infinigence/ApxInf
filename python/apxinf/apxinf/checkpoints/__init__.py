"""Checkpoint-directory layout: which format, and where is each file.

One place that answers "what shape is this checkpoint" for all three callers who
need to know — the policy loader, the preflight checks, and the offline auditor —
so a directory convention is written down once rather than guessed three times.

    from apxinf.checkpoints import detect_checkpoint, require_norm_stats

    layout = detect_checkpoint("~/airs/airs-model")
    stats = require_norm_stats(layout)      # raises, naming every path tried
    config_json = layout.config_json_text() # -> apxinf_py.Model.load(config_json=...)

Run ``python -m apxinf.checkpoints <dir>`` to see the same answer as a report.
Nothing here loads weights, imports torch, or touches the network.
"""

from __future__ import annotations

from .layout import (
    FORMATS,
    LEROBOT,
    NORM_STATS_NAME,
    OPENPI_PYTORCH,
    TOKENIZER_NAMES,
    CheckpointError,
    CheckpointLayout,
    detect_checkpoint,
    require_norm_stats,
    resolve_tokenizer,
)
from .metadata import (
    MetadataError,
    read_metadata_pt,
    repack_structure,
    train_config_facts,
)

__all__ = [
    "CheckpointError",
    "CheckpointLayout",
    "FORMATS",
    "LEROBOT",
    "MetadataError",
    "NORM_STATS_NAME",
    "OPENPI_PYTORCH",
    "TOKENIZER_NAMES",
    "detect_checkpoint",
    "read_metadata_pt",
    "repack_structure",
    "require_norm_stats",
    "resolve_tokenizer",
    "train_config_facts",
]
