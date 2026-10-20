#!/usr/bin/env python3
"""Check Fast-WAM computation, checkpoint and execution ownership."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / "crates/apxinf-model/src/fastwam"


def source(path):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)


def main():
    errors = []
    execution_types = r"\b(model_runner|ModelRunner|PreparedInference|ExecutionSession|PreparedPhase|ExecutionPolicy|VlaRequest|InferenceSpec)\b"
    for name in ("model", "text", "vae", "dit"):
        # Asset-backed tests exercise captures; production computation owns none.
        text = source(ROOT / f"{name}.rs").split("#[cfg(test)]")[0]
        if re.search(execution_types, text):
            errors.append(f"{name}.rs: computation depends on execution ownership")
    for path in [ROOT / "weights.rs", *sorted((ROOT / "weights").glob("*.rs"))]:
        text = source(path)
        if re.search(execution_types, text) or re.search(
            r"\b(model|text|vae|dit)\s*::", text
        ):
            errors.append(
                f"{path.relative_to(ROOT)}: weights depend on computation or runner"
            )
    for path in ROOT.rglob("*.rs"):
        text = source(path)
        if re.search(r"\b(unsafe|apxinf_cuda)\b|::ffi\b", text):
            errors.append(
                f"{path.relative_to(ROOT)}: family must use the safe cuda-new seam"
            )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Fast-WAM module dependency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
