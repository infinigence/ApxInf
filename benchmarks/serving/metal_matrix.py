#!/usr/bin/env python3
"""Measure nine serial profiles while a parent owns the Metal device lock."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time

import http_benchmark as benchmark


ROOT = Path(__file__).resolve().parents[2]
PROFILES = [(chunk, batch) for chunk in (64, 256, 1024) for batch in (1, 4, 8)]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def require_parent_lock(path, acknowledged):
    if not acknowledged:
        raise RuntimeError("Supply --lock-owned-by-parent after the parent reserves the device.")
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(handle, fcntl.LOCK_UN)
    raise RuntimeError("The Metal lock is free. No model may start without the parent reservation.")


def command_snapshot(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def memory_snapshot():
    return {"created_at": datetime.now(timezone.utc).isoformat(),
            "swap": command_snapshot(["/usr/sbin/sysctl", "vm.swapusage"]),
            "vm_stat": command_snapshot(["/usr/bin/vm_stat"])}


def group_members(pgid):
    result = subprocess.run(["/bin/ps", "-axo", "pid=,ppid=,pgid=,stat=,command="],
                            capture_output=True, text=True, timeout=10, check=True)
    members = []
    for line in result.stdout.splitlines():
        fields = line.split(None, 4)
        if len(fields) == 5 and fields[2] == str(pgid):
            members.append({"pid": int(fields[0]), "ppid": int(fields[1]),
                            "pgid": int(fields[2]), "state": fields[3], "command": fields[4]})
    return members


def stop_owned_group(child):
    """Signal only the new session created by this driver, then confirm exit."""
    report = {"pgid": child.pid, "signals": []}

    def inspect():
        try:
            return group_members(child.pid)
        except (OSError, subprocess.SubprocessError):
            # Preserve cleanup when process inspection fails. Do not start another profile.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
            raise

    for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 10)):
        child.poll()
        members = inspect()
        live = [member for member in members if not member["state"].startswith("Z")]
        if not live:
            break
        report["signals"].append({"signal": sig.name, "members": members})
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            child.poll()
            members = inspect()
            if not any(not member["state"].startswith("Z") for member in members):
                break
            time.sleep(0.2)
    report["returncode"] = child.wait(timeout=2)
    report["remaining_members"] = inspect()
    if any(not member["state"].startswith("Z") for member in report["remaining_members"]):
        raise RuntimeError("A matrix process remains alive. Do not start another model.")
    return report


def require_free_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def wait_ready(child, base_url, lock, timeout):
    started = time.perf_counter()
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"The matrix server exited during startup: {child.returncode}")
        require_parent_lock(lock, True)
        try:
            status, ready = benchmark.request_json(base_url, "/readyz", timeout=2)
            if status == 200 and ready.get("ready") is True:
                return ready, time.perf_counter() - started
            last_error = f"HTTP {status}"
        except (OSError, ValueError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise RuntimeError(f"The server did not become ready within {timeout} seconds: {last_error}")


def validate_ready(ready, model, chunk, batch, memory_budget, reference=None):
    if ready.get("ready") is not True or ready.get("model") != "apxinf-local":
        raise ValueError("Readiness does not describe the matrix model alias.")
    worker = ready["worker"]
    if worker["model_path"] != str(model):
        raise ValueError("The ready worker uses a different model path.")
    execution = worker["model_manifest"]["execution"]
    expected = {"provider": "mlx-lm", "precision": "bundle", "prefill_step_size": chunk,
                "output_batch_tokens": batch, "memory_limit_bytes": memory_budget}
    if execution != expected:
        raise ValueError("The ready worker uses different execution settings.")
    artifacts = worker["model_manifest"]["artifacts"]
    if not artifacts:
        raise ValueError("The worker supplied no model artifact identities.")
    invariant = dict(worker["model_manifest"])
    invariant["execution"] = {key: value for key, value in execution.items()
                              if key not in ("prefill_step_size", "output_batch_tokens")}
    identity = {"manifest": invariant, "capabilities": worker["capabilities"],
                "runtime": worker["runtime"], "model_path": worker["model_path"]}
    if reference is not None and identity != reference:
        raise ValueError("A model artifact, runtime, adapter, or fixed capability changed across profiles.")
    return identity


def output_comparison(profiles):
    comparison = {"basis": "HTTP visible output SHA-256 by workload index",
                  "token_id_parity": "not measured", "reference_profile": None, "profiles": []}
    if not profiles:
        return comparison
    comparison["reference_profile"] = profiles[0]["profile"]
    baseline = {item["index"]: item for item in profiles[0]["samples"]}
    for profile in profiles:
        mismatches, missing, failures = [], [], []
        actual = {item["index"]: item for item in profile["samples"]}
        for index in sorted(set(baseline) | set(actual)):
            first, current = baseline.get(index), actual.get(index)
            if first is None or current is None:
                missing.append(index)
            elif not first["success"] or not current["success"]:
                failures.append(index)
            elif (first["prompt_sha256"], first["output_sha256"]) != (current["prompt_sha256"], current["output_sha256"]):
                mismatches.append(index)
        comparison["profiles"].append({"profile": profile["profile"], "compared_requests": len(actual),
                                       "mismatched_indices": mismatches, "missing_indices": missing,
                                       "failed_indices": failures, "equal": not (mismatches or missing or failures)})
    return comparison


def profile_run(args, chunk, batch, reference):
    name = f"prefill-{chunk}-output-{batch}"
    directory = args.output_dir / name
    directory.mkdir()
    base_url = f"http://127.0.0.1:{args.port}"
    command = [str(args.server), "--model", str(args.model), "--model-id", "apxinf-local",
               "--host", "127.0.0.1", "--port", str(args.port),
               "--prefill-step-size", str(chunk), "--output-batch-tokens", str(batch),
               "--memory-budget-bytes", str(args.memory_budget_bytes)]
    report = {"profile": name, "command": command, "cwd": str(ROOT),
              "swap_before": memory_snapshot(), "startup_to_ready_s": None}
    child = None
    try:
        require_parent_lock(args.metal_lock, args.lock_owned_by_parent)
        require_free_port(args.port)
        if hashlib.sha256(args.server.read_bytes()).hexdigest() != args.server_sha256:
            raise RuntimeError("The server binary changed during the matrix.")
        with (directory / "server.log").open("wb") as log:
            launched = time.perf_counter()
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            report["server_pid"] = child.pid
            report["pgid"] = child.pid
            save(directory / "run.json", report)
            ready, _ = wait_ready(child, base_url, args.metal_lock, args.ready_timeout)
            report["startup_to_ready_s"] = time.perf_counter() - launched
            save(directory / "ready.json", ready)
            identity = validate_ready(ready, args.model, chunk, batch, args.memory_budget_bytes, reference)
            report["members_at_ready"] = group_members(child.pid)
            report["worker_pids"] = [member["pid"] for member in report["members_at_ready"]
                                     if member["pid"] != child.pid]
            if len(report["worker_pids"]) != 1:
                raise RuntimeError("The matrix requires exactly one owned worker process.")
            save(directory / "run.json", report)
            params = argparse.Namespace(base_url=base_url, model="apxinf-local", api="anthropic",
                                        samples=args.samples, warmup=2, concurrency=[1], workload="mixed",
                                        max_tokens=32, timeout=120, slo_ttft_ms=None, slo_e2e_ms=None,
                                        metal_lock=None, output=directory / "benchmark.json")
            with (directory / "benchmark-summary.txt").open("w") as text:
                with contextlib.redirect_stdout(text):
                    benchmark_status = benchmark.run(params)
            require_parent_lock(args.metal_lock, True)
            benchmark_report = json.loads(params.output.read_text())
            report["benchmark_exit_status"] = benchmark_status
            report["warmup_elapsed_s"] = sum(item["e2e_s"] for item in benchmark_report["warmup"])
            report["warmup_success"] = all(item["success"] for item in benchmark_report["warmup"])
            report["summary"] = benchmark_report["runs"][0]["summary"]
            if not report["warmup_success"] or benchmark_status:
                raise RuntimeError("A matrix request failed. Inspect the raw benchmark report.")
            return identity, {"profile": name, "samples": benchmark_report["runs"][0]["samples"],
                              "summary": report["summary"], "model_revision": ready["worker"]["model_revision"]}
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if child is not None:
                try:
                    report["shutdown"] = stop_owned_group(child)
                except BaseException as error:
                    report["shutdown_error"] = f"{type(error).__name__}: {error}"
                    raise
        finally:
            report["swap_after"] = memory_snapshot()
            save(directory / "run.json", report)


def run(args):
    if args.samples < 1 or args.ready_timeout <= 0 or not 1 <= args.port <= 65535 or args.memory_budget_bytes < 1:
        raise ValueError("Sample count, timeout, and port must be valid positive values.")
    args.model = args.model.resolve(strict=True)
    args.server = args.server.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    if not args.model.is_dir() or not args.server.is_file() or not os.access(args.server, os.X_OK):
        raise ValueError("Supply a model directory and an executable server file.")
    require_parent_lock(args.metal_lock, args.lock_owned_by_parent)
    group_members(os.getpgrp())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.server_sha256 = hashlib.sha256(args.server.read_bytes()).hexdigest()
    report = {"format": "apxinf-metal-matrix-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "host": platform.platform(), "driver_python": sys.version, "driver_pid": os.getpid(),
              "parent_pid": os.getppid(), "metal_lock": str(args.metal_lock.resolve()),
              "lock_ownership": "External owner acknowledged by caller, lock contention checked.",
              "hardware": command_snapshot(["/usr/sbin/sysctl", "hw.model", "hw.memsize", "machdep.cpu.brand_string"]),
              "model": str(args.model), "server_sha256": args.server_sha256,
              "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "benchmark_sha256": hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest(),
              "profile_order": PROFILES, "samples_per_profile": args.samples, "warmup_count": 2,
              "max_output_tokens": 32, "concurrency": 1, "workload": "mixed", "profiles": [],
              "limitations": ["This exploratory matrix does not establish a tail-latency guarantee.",
                              "Output hash equality does not establish token-ID parity.",
                              "Output intervals measure HTTP content events, not individual tokens.",
                              "Startup includes process launch, model load, and identity inspection.",
                              "The caller must ensure the external lock owner has unloaded its previous model.",
                              "A fixed profile order can introduce thermal or temporal bias."]}
    reference = None
    try:
        for chunk, batch in PROFILES:
            print(f"Starting prefill={chunk}, output_batch={batch}.", flush=True)
            identity, measured = profile_run(args, chunk, batch, reference)
            if reference is None:
                reference = identity
                save(args.output_dir / "reference-identity.json", identity)
            report["profiles"].append(measured)
            report["output_comparison"] = output_comparison(report["profiles"])
            save(args.output_dir / "matrix.json", report)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["output_comparison"] = output_comparison(report["profiles"])
        save(args.output_dir / "matrix.json", report)
    return 0 if all(item["equal"] for item in report["output_comparison"]["profiles"]) else 2


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--server", type=Path, default=ROOT / "target/release/apxinf-serve")
    result.add_argument("--port", type=int, default=8081)
    result.add_argument("--samples", type=int, default=16)
    result.add_argument("--ready-timeout", type=float, default=180)
    result.add_argument("--memory-budget-bytes", type=int, default=10 * 1024**3)
    result.add_argument("--metal-lock", type=Path,
                        default=Path.home() / ".cache/enginetailor/metal-measure.lock")
    result.add_argument("--lock-owned-by-parent", action="store_true")
    return result


if __name__ == "__main__":
    def interrupt(_signal, _frame):
        raise KeyboardInterrupt("The matrix received a termination signal.")
    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(run(parser().parse_args()))
