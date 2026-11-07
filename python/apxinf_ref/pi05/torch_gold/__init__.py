"""Deterministic Pi0.5 PyTorch eager references.

Two modules, split by what they need rather than by size.
:mod:`torch_gold.artifact` is the fixture, the stored encoding, the gate an
artifact has to pass and the preflight a replay runs before scoring it; it
imports no torch at module scope, so it loads -- and is
tested -- on a host with no device stack.  :mod:`torch_gold.capture` drives a
model and does need torch, so it may only be imported after ``bootstrap_torch``.
``generate_pi05_torch_gold.py`` beside this package is the CLI.

``SOURCE_ROOT`` is the directory holding this package and the ``_vendor``
snapshot it loads; putting it on ``sys.path`` is what lets both modules import
that snapshot.  The entry point installs the same directory before it can import
us at all, so the guard only stops it happening twice.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCHEMA = "apxinf.pi05.torch-eager-gold.v1"

SOURCE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SOURCE_ROOT.parents[2]  # .../apxinf_ref/pi05 -> repository root
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
