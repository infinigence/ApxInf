#!/usr/bin/env python3
"""Check PI0.5's internal dependency direction; Rust privacy enforces field access."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / "crates/apxinf-model/src/pi05"
RULES = {
    "network": r"\b(execution|Pi05Session|Pi05PreparedInference|ExecStrategy|CapturedGraph|ExecutionPolicy|VlaRequest|InferenceSpec)\b",
    "weights": r"\b(execution|network|Pi05Network|LoadedCompute|Pi05Session|Pi05PreparedInference)\b",
    "execution": r"\b(Bf16Blocks|Fp8StaticBlocks|Int8DynamicBlocks)\b|LoadedCompute\s*::",
}

def source(path):
    # Ignore prose; this guard checks explicit Rust dependencies, not a full AST.
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)

def main():
    violations = []
    for module, forbidden in RULES.items():
        for path in sorted((ROOT / module).rglob("*.rs")):
            code = source(path)
            if re.search(forbidden, code):
                violations.append(f"{path.relative_to(ROOT)}: forbidden dependency for {module}")
            if re.search(r"use\s+crate::pi05::\*", code):
                violations.append(f"{path.relative_to(ROOT)}: root wildcard hides dependencies")
    load = source(ROOT / "load.rs")
    if re.search(r"Pi05Session\s*\{|\.prepared\b", load):
        violations.append("load.rs: construct Session through its interface, not its fields")
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("PI0.5 module dependency checks passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
