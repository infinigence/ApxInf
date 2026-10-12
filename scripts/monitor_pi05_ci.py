#!/usr/bin/env python3
"""Audit GPU users and compilers, or monitor a CI process with an approved auditor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


SCHEMA = "apxinf.pi05.activity.v1"
COMPILERS = {"cargo", "rustc", "nvcc", "cicc", "ptxas", "cc1", "cc1plus",
             "cmake", "ninja", "make"}
GPU_DEVICES = {"/dev/nvhost-gpu", "/dev/nvhost-as-gpu", "/dev/nvidia0", "/dev/nvidia1",
               "/dev/nvgpu/igpu0/ctrl"}
GPU_CHANNEL = re.compile(r"anon_inode:(?:nvgpu-[\w.-]+-tsg|nvhost-[\w.-]+\.gpu-fd)\d+")


def audit(proc: Path = Path("/proc")) -> dict:
    # Install a root-owned copy for this read-only operation. Partial process
    # visibility cannot establish exclusive use of a shared board.
    compatible = Path("/proc/device-tree/compatible").read_bytes()
    if b"tegra264" in compatible:
        output = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=15)
        gpus = ET.fromstring(output).findall("gpu")
        if not gpus or any(gpu.find("processes") is None or
                (gpu.findtext("processes") or "").strip() == "N/A" for gpu in gpus):
            raise ValueError("GPU process inventory unavailable")
        # The compute-apps query omits persistent graphics clients.
        gpu_pids = {int(process.findtext("pid")) for gpu in gpus
                    for process in gpu.findall("processes/process_info")}
    elif b"tegra234" in compatible:
        gpu_pids = None
    else:
        raise ValueError("unsupported activity audit hardware")
    records = []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            name = (directory / "comm").read_text().strip()
            if gpu_pids is None:
                devices = {os.readlink(fd) for fd in (directory / "fd").iterdir()}
                # R39's CUDA contexts retain anonymous GPU channel descriptors.
                gpu = bool(devices & GPU_DEVICES) or any(GPU_CHANNEL.fullmatch(fd) for fd in devices)
            else:
                gpu = int(directory.name) in gpu_pids
            if gpu or name in COMPILERS:
                records.append({"pid": int(directory.name), "group": int(fields[2]),
                                "name": name, "gpu": gpu, "compiler": name in COMPILERS})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return {"time": time.time(), "processes": records}


def foreign_processes(sample: dict, group: int | None) -> list[dict]:
    return [record for record in sample["processes"] if record["group"] != group]


def validate_receipt(receipt: dict) -> None:
    if receipt.get("schema") != SCHEMA or receipt.get("passed") is not True:
        raise ValueError("activity monitoring failed or missing")
    samples = receipt["samples"]
    if not all(math.isfinite(sample["time"]) for sample in samples) or not all(
            math.isfinite(receipt[key]) for key in ("start", "finish")):
        raise ValueError("invalid activity timestamps")
    if len(samples) < 2 or samples[0]["time"] > receipt["start"] or samples[-1]["time"] < receipt["finish"]:
        raise ValueError("activity monitoring does not cover the run")
    if any(a["time"] > b["time"] for a, b in zip(samples, samples[1:])):
        raise ValueError("activity timestamps out of order")
    if any(foreign_processes(sample, receipt["group"]) for sample in samples):
        raise ValueError("foreign GPU client or compiler during run")


def monitor(args: argparse.Namespace) -> int:
    auditor = Path(os.environ["APXINF_CI_ACTIVITY_AUDIT"])
    digest = hashlib.sha256(auditor.read_bytes()).hexdigest()
    if digest != os.environ["APXINF_CI_ACTIVITY_AUDIT_SHA256"]:
        raise ValueError("unapproved activity auditor")
    receipt = {"schema": SCHEMA, "auditor_sha256": digest, "samples": [], "passed": False}
    child = None
    status = 1
    try:
        def sample() -> dict:
            value = json.loads(subprocess.check_output([str(auditor)], text=True, timeout=15))
            receipt["samples"].append(value)
            if foreign_processes(value, child.pid if child else None):
                raise ValueError("foreign GPU client or compiler; measurement rejected")
            return value
        sample()
        receipt["start"] = time.time()
        child = subprocess.Popen(args.command, start_new_session=True)
        receipt["group"] = child.pid
        while child.poll() is None:
            sample()
            time.sleep(1)
        status = child.returncode
        receipt["finish"] = time.time()
        sample()
        # Numerical/performance failures are separate from environment validity.
        receipt["passed"] = True
        validate_receipt(receipt)
    except (ValueError, OSError, KeyError, ET.ParseError, subprocess.SubprocessError) as error:
        receipt["error"] = str(error)
        status = 1
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "monitor"))
    parser.add_argument("--output", type=Path)
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    args.command = argv[split + 1:]
    if args.mode == "audit":
        print(json.dumps(audit()))
        return 0
    if args.output is None or not args.command:
        parser.error("monitor requires --output and a command after --")
    return monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())
