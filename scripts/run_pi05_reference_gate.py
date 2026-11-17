#!/usr/bin/env python3
"""Compatibility entry point; the PI0.5 regression suite lives under tests/."""
import runpy
import sys
from pathlib import Path

suite = Path(__file__).resolve().parents[1] / "crates/apxinf-model/tests/pi05_reference"
sys.path.insert(0, str(suite))
runpy.run_path(str(suite / "run.py"), run_name="__main__")
