#!/usr/bin/env python3
"""Executable model comparison checks; run `just parity --help`.

Receipts bind observed files and tensors, not the honesty of a caller. Model
adapters must capture at the consuming boundary and be reviewed separately.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

try:
    from .agent_acceptance import digest, source_digest
except ImportError:
    from agent_acceptance import digest, source_digest


def write_json(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def array(value):
    value = np.asarray(value)
    if value.dtype.kind not in "biuf" or not value.size:
        raise ValueError("tensors must be nonempty real numeric arrays")
    if not np.isfinite(value).all():
        raise ValueError("tensor contains NaN or infinity")
    return np.ascontiguousarray(value)


def tensor_digest(value):
    value = array(value)
    header = json.dumps([value.dtype.str, list(value.shape)]).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def load_contract(path):
    contract = read_json(path)
    if contract.get("version") != 1 or not contract.get("cases"):
        raise ValueError("contract requires version=1 and nonempty cases")
    ids = []
    for case in contract["cases"]:
        name = case["id"]
        if not isinstance(name, str) or not name or not all(c.isalnum() or c == '-' for c in name):
            raise ValueError("case ids must contain letters, digits or hyphens")
        ids.append(name)
        for key in ("input_names", "output_shape", "mode", "domain"):
            if not case.get(key):
                raise ValueError(f"{name}: missing {key}")
        names = case["input_names"]
        if (not isinstance(names, list) or len(set(names)) != len(names)
                or not all(isinstance(n, str) and n for n in names)):
            raise ValueError("input_names must be unique nonempty strings")
        grid = np.asarray(case["grid"])
        if case["grid"] != [] and (grid.dtype.kind not in "iu" or grid.ndim != 2 or grid.shape[1] != 3 or (grid <= 0).any()):
            raise ValueError("grid must be positive integer [views, 3] THW")
        if not all(type(n) is int and n > 0 for n in case["output_shape"]):
            raise ValueError("output_shape must contain positive integers")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids")
    metric = contract["comparison"]
    if metric["kind"] == "tolerance":
        cosine, relative = metric["min_cosine"], metric["max_relative_l2"]
        if (not math.isfinite(cosine) or not -1 <= cosine <= 1
                or not math.isfinite(relative) or relative < 0):
            raise ValueError("invalid numerical thresholds")
    elif metric["kind"] != "byte":
        raise ValueError("comparison kind must be byte or tolerance")
    for pair in contract.get("roundtrips", []):
        if len(pair) != 2 or pair[0] == pair[1] or any(x not in ids for x in pair):
            raise ValueError("roundtrips must name two distinct declared cases")
        a, b = (next(c for c in contract["cases"] if c["id"] == name) for name in pair)
        if any(a[k] != b[k] for k in ("input_names", "grid", "output_shape", "mode", "domain")):
            raise ValueError("roundtrip endpoints must have identical contracts except id")
    return contract


class Capture:
    """Start immediately before inference; finish with consumer-observed data.

    Pass the actual tensors, including initial noise, at the shared comparison
    boundary. For differently shaped preprocessing paths choose a common boundary
    first; do not compare unrelated tensors just because their names match.
    """

    def __init__(self, destination, contract, case, inputs, loaded_artifact):
        self.destination = Path(destination)
        self.contract_path = Path(contract)
        self.contract = load_contract(contract)
        self.case = next(c for c in self.contract["cases"] if c["id"] == case)
        if set(inputs) != set(self.case["input_names"]):
            raise ValueError("consumed inputs must exactly match contract input_names")
        self.inputs = {name: array(value).copy() for name, value in inputs.items()}
        self.contract_hash = digest(self.contract_path)
        self.artifact = Path(loaded_artifact).resolve()
        self.artifact_hash = digest(self.artifact)

    def finish(self, output, *, effective_grid, actual_mode, output_domain,
               board, arch, checkpoint, entry_point):
        """Write a new run only after inference, using runtime observations."""
        if digest(self.contract_path) != self.contract_hash or digest(self.artifact) != self.artifact_hash:
            raise ValueError("contract or loaded artifact changed during inference")
        if not all(isinstance(x, str) and x.strip() for x in
                   (actual_mode, output_domain, board, arch, checkpoint, entry_point)):
            raise ValueError("runtime identity fields must be nonempty strings")
        output = array(output)
        grid = np.asarray(effective_grid)
        if grid.tolist() != [] and (grid.dtype.kind not in "iu" or grid.ndim != 2 or grid.shape[1] != 3 or (grid <= 0).any()):
            raise ValueError("effective_grid must be observed positive integer THW rows")
        self.destination.mkdir(parents=True, exist_ok=False)
        tensors = {f"input-{i}": self.inputs[name] for i, name in enumerate(self.case["input_names"])}
        tensors["output"] = output
        np.savez(self.destination / "tensors.npz", **tensors)
        receipt = {
            "version": 1, "case": self.case["id"], "contract_sha256": self.contract_hash,
            "tensors_sha256": digest(self.destination / "tensors.npz"),
            "input_names": self.case["input_names"], "grid": grid.tolist(),
            "mode": actual_mode, "domain": output_domain,
            "loaded_artifact": str(self.artifact), "loaded_sha256": self.artifact_hash,
            "board": board, "arch": arch, "checkpoint": checkpoint, "entry_point": entry_point,
        }
        write_json(self.destination / "run.json", receipt)
        return receipt


def load_run(directory, contract_hash, case):
    directory = Path(directory)
    receipt = read_json(directory / "run.json")
    if receipt["version"] != 1 or receipt["case"] != case["id"]:
        raise ValueError("run schema or case mismatch")
    if receipt["contract_sha256"] != contract_hash:
        raise ValueError("contract changed after capture")
    if digest(directory / "tensors.npz") != receipt["tensors_sha256"]:
        raise ValueError("captured tensors changed")
    if receipt["input_names"] != case["input_names"]:
        raise ValueError("captured input names differ from contract")
    for key in ("grid", "mode", "domain"):
        if receipt[key] != case[key]:
            raise ValueError(f"effective {key} differs: {receipt[key]!r} != {case[key]!r}")
    with np.load(directory / "tensors.npz", allow_pickle=False) as tensors:
        expected = {"output"} | {f"input-{i}" for i in range(len(case["input_names"]))}
        if set(tensors.files) != expected:
            raise ValueError("missing or extra captured tensors")
        values = {name: array(tensors[name]) for name in expected}
    if list(values["output"].shape) != case["output_shape"]:
        raise ValueError("full output shape differs from contract")
    return receipt, values


def compare_arrays(reference, candidate, metric):
    reference, candidate = array(reference), array(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("output shapes differ")
    identical = tensor_digest(reference) == tensor_digest(candidate)
    # Scale both together before norms to avoid overflow on large finite values.
    a, b = reference.astype(np.float64).ravel(), candidate.astype(np.float64).ravel()
    scale = max(float(np.abs(a).max()), float(np.abs(b).max())) or 1.0
    a, b = a / scale, b / scale
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    delta = float(np.linalg.norm(a - b))
    cosine = float(np.clip(np.dot(a / an, b / bn), -1, 1)) if an and bn else (1.0 if delta == 0 else 0.0)
    relative = delta / an if an else (0.0 if delta == 0 else None)
    passed = identical if metric["kind"] == "byte" else (
        relative is not None and relative <= metric["max_relative_l2"] and cosine >= metric["min_cosine"])
    return {"status": "PASS" if passed else "FAIL", "byte_identical": identical,
            "cosine": cosine, "relative_l2": relative,
            "note": "relative_l2 null means nonzero candidate against zero reference"}


def identity(receipt, build_path):
    """Never upgrade timestamps or a lone object hash into build provenance."""
    if build_path is None:
        return {"status": "UNVERIFIED", "reason": "no observed fresh-build receipt"}
    build = read_json(build_path)
    if build.get("version") != 1 or build.get("kind") != "fresh-cargo-build" or build.get("exit_code") != 0:
        raise ValueError("invalid or unsuccessful build receipt")
    if build["source_before"] != build["source_after"]:
        raise ValueError("source changed while building")
    if build["artifact_sha256"] != receipt["loaded_sha256"]:
        raise ValueError("loaded bytes do not match recorded build")
    directory = Path(build_path).resolve().parent
    if digest(directory / "build.log") != build["log_sha256"]:
        raise ValueError("build log changed")
    if digest(Path(build["artifact"])) != build["artifact_sha256"]:
        raise ValueError("built artifact changed")
    if digest(Path(receipt["loaded_artifact"])) != receipt["loaded_sha256"]:
        raise ValueError("loaded artifact file changed since capture")
    if source_digest(Path(build["source_root"])) != build["source_after"]:
        raise ValueError("source changed since build")
    return {"status": "PASS", "reason": "recorded fresh build matches loaded bytes and current source fingerprint",
            "scope": "local recorded build; external dependencies, toolchain and adapter require review"}


def compare(contract_path, baseline, candidate, baseline_build=None, candidate_build=None):
    contract = load_contract(contract_path)
    contract_hash = digest(contract_path)
    checks, loaded = [], {}
    for case in contract["cases"]:
        name = case["id"]
        try:
            a, av = load_run(Path(baseline) / name, contract_hash, case)
            b, bv = load_run(Path(candidate) / name, contract_hash, case)
            loaded[name] = (av, bv)
            if a["checkpoint"] != b["checkpoint"] or a["board"] != b["board"] or a["arch"] != b["arch"]:
                raise ValueError("checkpoint or device differs across comparison")
            for tensor in av.keys() - {"output"}:
                if tensor_digest(av[tensor]) != tensor_digest(bv[tensor]):
                    raise ValueError(f"consumed tensor differs: {case['input_names'][int(tensor[6:])]}")
            checks.append({"case": name, "check": "outputs", **compare_arrays(av["output"], bv["output"], contract["comparison"])})
            for side, receipt, build in (("baseline", a, baseline_build), ("candidate", b, candidate_build)):
                checks.append({"case": name, "check": f"{side}-identity", **identity(receipt, build)})
        except FileNotFoundError as error:
            checks.append({"case": name, "status": "UNVERIFIED", "reason": str(error)})
        except (ValueError, KeyError, TypeError) as error:
            checks.append({"case": name, "status": "FAIL", "reason": str(error)})
    for first, last in contract.get("roundtrips", []):
        try:
            for side in range(2):
                a, b = loaded[first][side], loaded[last][side]
                if a.keys() != b.keys() or any(tensor_digest(a[k]) != tensor_digest(b[k]) for k in a):
                    raise ValueError("roundtrip inputs or full output are not byte-identical")
            checks.append({"check": f"roundtrip:{first}:{last}", "status": "PASS"})
        except (KeyError, ValueError) as error:
            checks.append({"check": f"roundtrip:{first}:{last}", "status": "FAIL", "reason": str(error)})
    return result(checks)


def result(checks):
    statuses = {check["status"] for check in checks}
    status = "FAIL" if "FAIL" in statuses else "UNVERIFIED" if "UNVERIFIED" in statuses else "PASS"
    return {"status": status, "checks": checks,
            "scope": "declared checks only; not GPU, Graph, performance or human certification"}


def tuning(before, after):
    """Compare actual persisted DB records normalized by a format-specific adapter."""
    for records in (before, after):
        if not isinstance(records, dict) or not all(isinstance(k, str) and k and isinstance(v, dict) and v for k, v in records.items()):
            raise ValueError("tuning snapshots must map canonical keys to nonempty record objects")
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    return {"status": "PASS" if added and not removed else "FAIL", "new_keys": added,
            "removed_keys": removed, "changed_keys": changed,
            "scope": "persisted key growth only; adapter must validate record legality and actual execution"}


def tuning_database(document):
    """Read native ApxInf persisted records without an agent-written summary."""
    header = {}
    for key in ("schema", "device_name", "sm", "cuda_version", "cublas_version", "kernel_build_id"):
        if document.get(key) is None or document[key] == "":
            raise ValueError(f"tuning DB missing identity field: {key}")
        header[key] = document[key]
    if header["schema"] != "apxinf.cuda.tuning.v1" or type(header["sm"]) is not int or header["sm"] <= 0:
        raise ValueError("unsupported tuning DB schema or device architecture")
    records = document["records"]
    if not isinstance(records, list):
        raise ValueError("native tuning DB records must be a list")
    normalized = {}
    for record in records:
        key, tactic = record["key"], record["tactic"]
        if not isinstance(key, dict) or not key:
            raise ValueError("tactic key must be a nonempty object")
        if not isinstance(tactic.get("backend"), str) or not tactic["backend"] or type(tactic.get("id")) is not int:
            raise ValueError("invalid tactic identity")
        if type(tactic.get("implementation_version")) is not int or tactic["implementation_version"] < 1:
            raise ValueError("missing tactic implementation version")
        duration = record["milliseconds"]
        if duration is not None and (not math.isfinite(duration) or duration < 0):
            raise ValueError("invalid tactic timing")
        canonical = json.dumps(key, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if canonical in normalized:
            raise ValueError("duplicate persisted tactic key")
        normalized[canonical] = record
    return header, normalized


def persisted_tuning(before, after):
    a_header, a = tuning_database(before)
    b_header, b = tuning_database(after)
    if a_header != b_header:
        raise ValueError("tuning DB device/library/build identity changed")
    return tuning(a, b)


def shapes(contract, observations):
    """Check an ordered sequence, allowing A -> B -> A to repeat a grid."""
    checks = []
    if not isinstance(observations, list) or len(observations) != len(contract["cases"]):
        return result([{"status": "FAIL", "reason": "missing or extra shape observations"}])
    for case, observation in zip(contract["cases"], observations):
        mismatches = [key for key in ("id", "grid", "output_shape", "mode", "domain")
                      if observation.get(key) != case[key]]
        checks.append({"case": case["id"], "status": "FAIL" if mismatches else "PASS",
                       "mismatches": mismatches})
    output = result(checks)
    output["scope"] = "observed shapes/modes only; not numerical parity or source provenance"
    return output


def build(destination, artifact, cargo_args):
    """Run cargo with a newly created target directory, retaining its evidence."""
    destination = Path(destination).resolve()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    if not destination.is_relative_to(root / "devlocal"):
        raise ValueError("build evidence must live under worktree devlocal/")
    relative = Path(artifact)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact must be relative to the fresh cargo target directory")
    if any(arg == "--" or arg.startswith(("--target-dir", "--manifest-path", "--config")) for arg in cargo_args):
        raise ValueError("target, manifest and cargo configuration overrides are not supported")
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / "target"
    before = source_digest(root)
    command = ["cargo", "build", "--target-dir", str(target), *cargo_args]
    env = dict(os.environ, CARGO_INCREMENTAL="0")
    with (destination / "build.log").open("x") as log:
        code = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    after = source_digest(root)
    output = (target / relative).resolve()
    if code != 0 or before != after or not output.is_file() or not output.is_relative_to(target):
        write_json(destination / "failed.json", {"command": command, "exit_code": code,
                   "source_unchanged": before == after})
        raise ValueError(f"build failed, source changed or artifact absent; inspect {destination / 'build.log'}")
    receipt = {"version": 1, "kind": "fresh-cargo-build", "source_root": str(root),
               "source_before": before, "source_after": after, "command": command,
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               "exit_code": code, "artifact": str(output), "artifact_sha256": digest(output),
               "log_sha256": digest(destination / "build.log")}
    write_json(destination / "build.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    cmp = commands.add_parser("compare", help="check recorded full tensors, shapes, modes and build bindings")
    cmp.add_argument("contract", type=Path)
    cmp.add_argument("baseline", type=Path, help="directory containing one capture per case id")
    cmp.add_argument("candidate", type=Path)
    cmp.add_argument("--baseline-build", type=Path)
    cmp.add_argument("--candidate-build", type=Path)
    tune = commands.add_parser("tuning", help="check new persisted keys; run separately for each tuning stage")
    tune.add_argument("before", type=Path)
    tune.add_argument("after", type=Path)
    shape = commands.add_parser("shapes", help="check actual ordered grid/mode observations against a contract")
    shape.add_argument("contract", type=Path)
    shape.add_argument("observations", type=Path)
    ident = commands.add_parser("identity", help="check a native artifact against an observed fresh-build receipt")
    ident.add_argument("artifact", type=Path)
    ident.add_argument("--build", type=Path)
    builder = commands.add_parser("build", help="capture a fresh cargo build into a new devlocal directory")
    builder.add_argument("destination", type=Path)
    builder.add_argument("artifact", help="artifact path relative to target, e.g. release/libapxinf_py.so")
    builder.add_argument("cargo_args", nargs=argparse.REMAINDER, help="cargo build arguments following --")
    args = parser.parse_args()
    try:
        if args.action == "build":
            extra = args.cargo_args[1:] if args.cargo_args[:1] == ["--"] else args.cargo_args
            output = build(args.destination, args.artifact, extra)
        elif args.action == "tuning":
            output = persisted_tuning(read_json(args.before), read_json(args.after))
        elif args.action == "shapes":
            output = shapes(load_contract(args.contract), read_json(args.observations))
        elif args.action == "identity":
            output = identity({"loaded_artifact": str(args.artifact.resolve()),
                               "loaded_sha256": digest(args.artifact)}, args.build)
        else:
            output = compare(args.contract, args.baseline, args.candidate, args.baseline_build, args.candidate_build)
        print(json.dumps(output, indent=2, allow_nan=False))
        return {"PASS": 0, "FAIL": 1, "UNVERIFIED": 2}.get(output.get("status", "PASS"), 1)
    except FileNotFoundError as error:
        print(json.dumps({"status": "UNVERIFIED", "reason": str(error)}))
        return 2
    except (OSError, ValueError, KeyError, TypeError, StopIteration, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "FAIL", "reason": str(error)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
