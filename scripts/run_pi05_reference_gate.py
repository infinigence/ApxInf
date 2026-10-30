#!/usr/bin/env python3
"""Build and run one pinned GPU lane, then apply the cosine ratchet policy."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from check_pi05_reference_gate import MANIFEST, POLICY, ROOT, check_report, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True, choices=("thor", "orin"))
    parser.add_argument("--variant", required=True, choices=("bf16", "fp8_static", "int8_dynamic"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration", default="")
    parser.add_argument("--tuning", type=Path, help="Override recipes with a JSON bank or directory")
    parser.add_argument("--autotune", action="store_true", help="Explicit recipe qualification only")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if not args.checkpoint.is_file():
        parser.error("--checkpoint must name an existing model.safetensors file")
    if args.variant == "fp8_static" and (not args.calibration or not Path(args.calibration).is_file()):
        parser.error("FP8 requires an existing --calibration file")
    policy = json.loads(POLICY.read_text())
    validate(policy, MANIFEST.read_bytes())
    lane = args.board + "/" + args.variant
    if lane not in policy["lanes"]:
        parser.error("unsupported board/precision pair")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    recipes = args.tuning or MANIFEST.parent / "recipes" / (args.board + ".json")
    if recipes.is_file():
        bank = json.loads(recipes.read_text())
    elif recipes.is_dir():
        bank = {p.name: p.read_text() for p in recipes.glob("*.recipe")}
    else:
        parser.error("tuning bank or directory is missing")
    (output / "recipes").mkdir()
    recipe_hash = hashlib.sha256()
    for name, content in sorted(bank.items()):
        if Path(name).name != name or not name.endswith(".recipe"):
            parser.error("invalid recipe filename")
        recipe_hash.update(name.encode() + b"\0" + content.encode())
        (output / "recipes" / name).write_text(content)
    if not list((output / "recipes").glob("*.recipe")) and not args.autotune:
        parser.error("empty recipe bank; qualification requires explicit --autotune")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    provenance = {"revision": revision, "dirty": bool(status), "lane": lane,
                  "policy_sha256": hashlib.sha256(POLICY.read_bytes()).hexdigest(),
                  "initial_recipe_sha256": recipe_hash.hexdigest(), "autotune": args.autotune}
    (output / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    subprocess.run([sys.executable, str(MANIFEST.with_name("materialize.py")),
                    "--output", str(output / "inputs")], check=True)
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "apxinf-model",
                    "--example", "pi05_reference_gate", "--features", "cuda,openblas"], check=True)
    metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"]))
    binary = Path(metadata["target_directory"]) / "release/examples/pi05_reference_gate"
    provenance["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (output / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report_path = output / "report.json"
    command = [str(binary), str(args.checkpoint.resolve()), str(output / "inputs/manifest.json"),
               args.variant, str(report_path), "--tuning", str(output / "recipes"), "--measure"]
    if args.autotune:
        command.append("--autotune")
    if args.variant == "fp8_static":
        command += ["--calibration", str(Path(args.calibration).resolve())]
    with (output / "execution.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    failures = check_report(policy, lane, json.loads(report_path.read_text()))
    result = {**provenance, "required": policy["lanes"][lane]["required"],
              "passed": not failures, "failures": failures}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(bool(failures) and result["required"])


if __name__ == "__main__":
    raise SystemExit(main())
