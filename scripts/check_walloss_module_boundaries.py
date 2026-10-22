#!/usr/bin/env python3
"""Check WallOSS's internal dependency direction; Rust privacy enforces field access."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / "crates/apxinf-model/src/walloss"
RULES = {
    "model": r"\b(model_runner|WallossModelRunner|WallossPreparedInference|CapturedGraph|ExecutionPolicy|VlaRequest|InferenceSpec)\b",
    "weights": r"\b(WallossModel|ModelVariant|WallossModelRunner|WallossPreparedInference)\b|\b(model_runner|model)\s*::|::(?:model_runner|model)\b|\buse\s+[^;]*\{[^;]*\b(?:model_runner|model)\b",
    "model_runner": r"ModelVariant\s*::|WallossDeviceWeights\s*::|\b(blocks|weights)\s*::",
}
# Leaf modules shared by model and model_runner must not depend on either.
LEAVES = ("geometry.rs", "schedule.rs", "config.rs", "workspace.rs")
LEAF_FORBIDDEN = r"\b(model_runner|model|weights)\s*::"

def source(path):
    # Ignore prose; this guard checks explicit Rust dependencies, not a full AST.
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)

def main():
    violations = []
    for module, forbidden in RULES.items():
        paths = sorted((ROOT / module).rglob("*.rs"))
        if not paths:
            violations.append(f"{module}: expected Rust module sources are missing")
        for path in paths:
            code = source(path)
            if re.search(forbidden, code):
                violations.append(f"{path.relative_to(ROOT)}: forbidden dependency for {module}")
            if re.search(r"use\s+crate::walloss::\*", code):
                violations.append(f"{path.relative_to(ROOT)}: root wildcard hides dependencies")
    for leaf in LEAVES:
        if re.search(LEAF_FORBIDDEN, source(ROOT / leaf)):
            violations.append(f"{leaf}: leaf module depends on model, model_runner or weights")
    load = source(ROOT / "load.rs")
    if re.search(r"WallossModelRunner\s*\{|\.prepared\b", load):
        violations.append("load.rs: construct ModelRunner through its interface, not its fields")
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("WallOSS module dependency checks passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
