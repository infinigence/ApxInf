#!/usr/bin/env python3
"""Record task acceptance commands and reject stale or incomplete evidence.

This is an audit aid, not a sandbox or a numerical oracle. The task owner
chooses commands whose assertions establish the declared acceptance criteria.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(root):
    names = subprocess.check_output(
        ["git", "ls-files", "-cz", "--others", "--exclude-standard"], cwd=root
    ).split(b"\0")
    result = hashlib.sha256()
    for name in sorted(set(names) - {b""}):
        path = root / name.decode()
        result.update(name + b"\0")
        if path.is_symlink():
            content = str(path.readlink()).encode()
        elif path.is_file():
            content = path.read_bytes()
        else:
            content = b"<deleted-or-submodule>"
        result.update(hashlib.sha256(content).digest())
        result.update(str(path.lstat().st_mode if path.exists() else 0).encode())
    return result.hexdigest()


def load_plan(path):
    path = path.resolve()
    root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=path.parent, text=True
    ).strip())
    if not path.is_relative_to(root / "devlocal"):
        raise ValueError("plan must live under this worktree's devlocal/<feature>/")
    plan = json.loads(path.read_text())
    if plan.get("version") != 1 or not plan.get("cases"):
        raise ValueError("plan requires version=1 and a nonempty cases list")
    ids = set()
    for case in plan["cases"]:
        name = case.get("id", "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or name in ids:
            raise ValueError(f"invalid or duplicate case id: {name}")
        ids.add(name)
        if not case.get("criterion") or not case.get("entry_point"):
            raise ValueError(f"{name}: criterion and entry_point are required")
        command = case.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(arg, str) and arg for arg in command
        ):
            raise ValueError(f"{name}: command must be a nonempty argv list")
        # Private harnesses may be ignored by Git. Pin them explicitly.
        for dependency in case.get("inputs", []):
            if not (root / dependency).is_file():
                raise ValueError(f"{name}: missing input {dependency}")
    return root, plan


def input_hashes(root, case):
    return {name: digest(root / name) for name in case.get("inputs", [])}


def records_dir(path):
    return path.parent / (path.stem + "-evidence")


def record(path, name, reason=None):
    root, plan = load_plan(path)
    case = next((item for item in plan["cases"] if item["id"] == name), None)
    if case is None:
        raise ValueError(f"unknown acceptance case {name}")
    directory = records_dir(path)
    directory.mkdir(exist_ok=True)
    before = source_digest(root)
    plan_hash = digest(path)
    inputs = input_hashes(root, case)
    receipt = {
        "id": name, "source": before, "plan": plan_hash, "inputs": inputs,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "command": case["command"],
    }
    if reason is not None:
        if not reason.strip():
            raise ValueError("blocked case requires a concrete reason")
        receipt.update(status="blocked", reason=reason)
        code = 2
    else:
        # An interrupted rerun must not leave its previous success as current.
        receipt["status"] = "running"
        (directory / f"{name}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        log = directory / f"{name}-{uuid.uuid4().hex}.log"
        with log.open("w") as output:
            try:
                code = subprocess.run(case["command"], cwd=root, stdout=output,
                                      stderr=subprocess.STDOUT).returncode
            except OSError as error:
                output.write(str(error) + "\n")
                code = 127
        unchanged = (before == source_digest(root) and plan_hash == digest(path)
                     and inputs == input_hashes(root, case))
        receipt.update(status="passed" if code == 0 and unchanged else "failed",
                       exit_code=code, log=log.name, log_sha256=digest(log))
        if not unchanged:
            receipt["reason"] = "source, plan or harness changed during command"
            code = 1
        print(f"{name}: {receipt['status']}; log: {log}")
    (directory / f"{name}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return code


def inspect(path):
    root, plan = load_plan(path)
    current = source_digest(root)
    plan_hash = digest(path)
    failures, blocked = [], []
    for case in plan["cases"]:
        name = case["id"]
        try:
            receipt = json.loads((records_dir(path) / f"{name}.json").read_text())
            if (receipt["source"] != current or receipt["plan"] != plan_hash
                    or receipt["inputs"] != input_hashes(root, case)
                    or receipt["command"] != case["command"]):
                raise ValueError("stale evidence: rerun against current source/plan/inputs")
            if receipt["status"] == "blocked" and receipt.get("reason", "").strip():
                blocked.append(f"{name}: {receipt['reason']}")
                continue
            if receipt["status"] != "passed" or receipt["exit_code"] != 0:
                raise ValueError("command failed; fix or record a concrete blocker")
            log = (records_dir(path) / receipt["log"]).resolve()
            if not log.is_relative_to(records_dir(path).resolve()) or digest(log) != receipt["log_sha256"]:
                raise ValueError("evidence log missing or modified")
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"{name}: {error}")
    return failures, blocked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("run", "block", "check"):
        command = sub.add_parser(action)
        command.add_argument("plan", type=Path)
        if action != "check":
            command.add_argument("case")
        if action == "block":
            command.add_argument("--reason", required=True)
    args = parser.parse_args()
    try:
        if args.action != "check":
            return record(args.plan.resolve(), args.case, getattr(args, "reason", None))
        failures, blocked = inspect(args.plan.resolve())
        if failures or blocked:
            print("INCOMPLETE\n" + "\n".join(failures + blocked))
            return 1 if failures else 2
        print("PASS: every declared acceptance command passed on the current source")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"Acceptance error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
