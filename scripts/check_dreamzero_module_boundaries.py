#!/usr/bin/env python3
"""Enforce the DreamZero model/runner/weights ownership documented in the port."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / "crates/apxinf-model/src/dreamzero"


def source(path):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)


def main():
    rules = [
        (list((ROOT / "model").rglob("*.rs")),
         r"\b(VlaRequest|VlaRuntime|InferenceSpec|ExecutionPolicy|PreparationStatus|PreparedInference|CapturedGraph|model_runner)\b"),
        (list((ROOT / "weights").rglob("*.rs")),
         r"\b(DreamZeroModel|DreamZeroModelRunner|Workspace|Profile|VlaRequest)\b|\b(model|model_runner)\s*::"),
        ([ROOT / "model_runner.rs"], r"\b(weights|dit|dit_block|text|clip|vae|unipc)\s*::"),
        ([ROOT / "config.rs", ROOT / "schedule.rs"], r"\b(model|model_runner|weights)\s*::"),
    ]
    violations = []
    for paths, forbidden in rules:
        if not paths:
            violations.append("missing declared DreamZero module")
        for path in paths:
            if re.search(forbidden, source(path)):
                violations.append(f"{path.relative_to(ROOT)}: forbidden upward dependency")
    for path in ROOT.rglob("*.rs"):
        if path.name != "backend.rs" and re.search(r"\bapxinf_cuda(?:_new)?\s*::", source(path)):
            violations.append(f"{path.relative_to(ROOT)}: use the family backend.rs seam")
        if re.search(r"\bunsafe\b|\bextern\s+\"C\"", source(path)):
            violations.append(f"{path.relative_to(ROOT)}: device FFI belongs in the backend")
    if re.search(r"DreamZeroModelRunner\s*\{", source(ROOT / "load.rs")):
        violations.append("load.rs: construct ModelRunner through its constructor")
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("DreamZero module dependency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
