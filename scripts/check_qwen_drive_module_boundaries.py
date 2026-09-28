#!/usr/bin/env python3
"""Check Qwen-Drive's internal dependency direction; Rust privacy enforces field access."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / "crates/apxinf-model/src/qwen_drive"
RULES = {
    "model": r"\b(model_runner|QwenDriveModelRunner|QwenDrivePreparedInference|ExecStrategy|CapturedGraph|ExecutionPolicy|VlaRequest|InferenceSpec)\b",
    "weights": r"\b(QwenDriveModel|QwenDriveModelRunner|QwenDrivePreparedInference)\b|\b(model_runner|model)\s*::|::(?:model_runner|model)\b|\buse\s+[^;]*\{[^;]*\b(?:model_runner|model)\b",
    "model_runner": r"\b(Bf16Blocks|BackboneBf16|PlannerBf16)\b|ModelVariant\s*::",
}

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
            if re.search(r"use\s+crate::qwen_drive::\*", code):
                violations.append(f"{path.relative_to(ROOT)}: root wildcard hides dependencies")
    if sorted(p.name for p in (ROOT / "model/blocks").iterdir()) != ["bf16.rs", "mod.rs"]:
        violations.append("model/blocks: expected one BF16 implementation and its execution seam")
    load = source(ROOT / "load.rs")
    if re.search(r"QwenDriveModelRunner\s*\{|\.prepared\b", load):
        violations.append("load.rs: construct ModelRunner through its interface, not its fields")
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("Qwen-Drive module dependency checks passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
