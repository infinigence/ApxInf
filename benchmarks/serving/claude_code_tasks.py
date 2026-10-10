#!/usr/bin/env python3
"""Run small, original Claude Code tasks against a local Messages endpoint."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
import uuid

from http_benchmark import local_url, metal_measurement_lock, metrics_snapshot, request_json


def client_environment(base_url, model, config_dir, max_tokens=512, max_context=None):
    """Keep machine basics and replace all provider configuration for this child."""
    keep = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "USER", "LOGNAME", "TERM")
    environment = {key: os.environ[key] for key in keep if key in os.environ}
    environment.update({
        "CLAUDE_CONFIG_DIR": str(config_dir.resolve()),
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": "apxinf-local-test-only",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_DISABLE_THINKING": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_tokens),
        "CLAUDE_CODE_MAX_RETRIES": "0",
        "DISABLE_PROMPT_CACHING": "1",
    })
    if max_context is not None:
        environment["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(max_context)
    return environment


def prepare_task(name, directory):
    directory.mkdir(parents=True, exist_ok=False)
    if name == "read":
        marker = "APXINF_" + uuid.uuid4().hex
        (directory / "marker.txt").write_text(marker + "\n")
        return {"prompt": "Use Read to open marker.txt. Reply with only its complete marker.",
                "tools": ["Read"], "allowed": ["Read"], "marker": marker}
    if name == "edit":
        (directory / "score.py").write_text("def normalize_score(value):\n    return value / 10\n")
        return {"prompt": "Use Read to inspect score.py. Use Edit to fix normalize_score. "
                          "Divide the input by 10 and clamp the result to [0, 1]. "
                          "Keep one function with one return statement. Do not add imports. "
                          "Reply FIXED after editing the file.",
                "tools": ["Read", "Edit"], "allowed": ["Read", "Edit"]}
    if name == "check":
        (directory / "check_fixture.py").write_text(
            "def queue_has_space(active, waiting, limit):\n"
            "    return active + waiting < limit\n\n"
            "assert queue_has_space(1, 1, 3)\n"
            "assert not queue_has_space(1, 2, 3)\n"
            "print('APXINF_CHECK_PASS')\n")
        return {"prompt": "Use Read to inspect check_fixture.py. "
                          "Use Bash to run exactly: python3 check_fixture.py. "
                          "Report the output marker after the command finishes.",
                "tools": ["Read", "Bash"],
                "allowed": ["Read", "Bash(python3 check_fixture.py)"]}
    if name == "cancel":
        return {"prompt": "Write a detailed list of 200 different short sentences about bounded queues. "
                          "Number every sentence. Continue until you reach item 200.",
                "tools": [], "allowed": []}
    raise ValueError(f"Unknown task: {name}")


def check_edit(path):
    """Evaluate only a small expression grammar, without running the edited file."""
    try:
        module = ast.parse(path.read_text())
        if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
            return False
        function = module.body[0]
        if (function.name != "normalize_score" or function.decorator_list or
                len(function.args.args) != 1 or function.args.args[0].arg != "value" or
                function.args.defaults or function.args.kwonlyargs or function.args.vararg or
                function.args.kwarg or function.args.posonlyargs or len(function.body) != 1 or
                not isinstance(function.body[0], ast.Return)):
            return False
        expression = ast.Expression(function.body[0].value)
        permitted = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
                     ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)
        for node in ast.walk(expression):
            if not isinstance(node, permitted):
                return False
            if isinstance(node, ast.Constant) and type(node.value) not in (int, float):
                return False
            if isinstance(node, ast.Name) and node.id not in ("value", "min", "max"):
                return False
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or
                                              node.func.id not in ("min", "max") or node.keywords):
                return False
        code = compile(expression, str(path), "eval")
        return all(abs(eval(code, {"__builtins__": {}, "min": min, "max": max}, {"value": value}) - expected) < 1e-9
                   for value, expected in [(-20, 0), (0, 0), (2.5, .25), (10, 1), (30, 1)])
    except (OSError, SyntaxError, ValueError, TypeError, ZeroDivisionError):
        return False


def inspect_events(events):
    tools, results, final = [], [], None
    for record in events:
        event = record["event"]
        kind = event.get("type")
        if kind == "result":
            final = event
        if kind in ("assistant", "user"):
            content = event.get("message", {}).get("content", [])
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "tool_use":
                    tools.append({"id": block.get("id"), "name": block.get("name"),
                                  "input": block.get("input"), "elapsed_s": record["elapsed_s"]})
                elif block.get("type") == "tool_result":
                    results.append({"tool_use_id": block.get("tool_use_id"),
                                    "is_error": block.get("is_error", False),
                                    "content": block.get("content"), "elapsed_s": record["elapsed_s"]})
    return tools, results, final


def has_content_delta(event):
    if event.get("type") != "stream_event" or not isinstance(event.get("event"), dict):
        return False
    delta = event["event"].get("delta", {})
    return isinstance(delta, dict) and any(delta.get(name) for name in ("text", "partial_json"))


def stop_child(process):
    """Stop only the process group created for this task."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait(timeout=3)


def capture_client(command, directory, environment, timeout, cancel_after_content):
    start = time.perf_counter()
    events, diagnostics = [], []
    first_content = None
    cancelled = timed_out = False
    stdout_path = directory / "client.stdout.jsonl"
    stderr_path = directory / "client.stderr.txt"
    with stdout_path.open("wb") as saved_stdout, stderr_path.open("wb") as saved_stderr:
        process = subprocess.Popen(command, cwd=directory, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, ("stdout", bytearray()))
        selector.register(process.stderr, selectors.EVENT_READ, ("stderr", bytearray()))
        try:
            while selector.get_map():
                elapsed = time.perf_counter() - start
                if process.poll() is None and elapsed > timeout:
                    timed_out = True
                    stop_child(process)
                if process.poll() is None and cancel_after_content and first_content is not None:
                    cancelled = True
                    stop_child(process)
                for key, _ in selector.select(timeout=.1):
                    kind, pending = key.data
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        if pending and kind == "stdout":
                            diagnostics.append("The client ended with an incomplete JSON line.")
                        continue
                    (saved_stdout if kind == "stdout" else saved_stderr).write(chunk)
                    if kind == "stderr":
                        continue
                    pending.extend(chunk)
                    while b"\n" in pending:
                        raw, _, rest = pending.partition(b"\n")
                        pending[:] = rest
                        if not raw.strip():
                            continue
                        try:
                            event = json.loads(raw)
                            if not isinstance(event, dict):
                                raise ValueError("Expected an object.")
                        except (json.JSONDecodeError, ValueError):
                            diagnostics.append("The client emitted a non-JSON output line.")
                            continue
                        now = time.perf_counter() - start
                        events.append({"elapsed_s": now, "event": event})
                        if has_content_delta(event):
                            first_content = now if first_content is None else first_content
            process.wait(timeout=3)
        finally:
            stop_child(process)
            selector.close()
            process.stdout.close()
            process.stderr.close()
    return {"exit_code": process.returncode, "elapsed_s": time.perf_counter() - start,
            "first_content_s": first_content, "cancel_sent": cancelled, "timed_out": timed_out,
            "events": events, "diagnostics": diagnostics,
            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}


def task_outcome(name, spec, directory, capture):
    tools, results, final = inspect_events(capture["events"])
    names = {tool["name"] for tool in tools}
    finished = bool(final and not final.get("is_error") and capture["exit_code"] == 0)
    final_text = final.get("result", "") if final else ""
    if name == "read":
        assertion = "Read" in names and spec["marker"] in final_text
    elif name == "edit":
        assertion = "Read" in names and "Edit" in names and check_edit(directory / "score.py")
    elif name == "check":
        successful_bash = {item["id"] for item in tools if item["name"] == "Bash"}
        assertion = ("Read" in names and any(
            item["tool_use_id"] in successful_bash and not item["is_error"] and
            "APXINF_CHECK_PASS" in json.dumps(item["content"]) for item in results))
    else:
        assertion = capture["cancel_sent"] and capture["first_content_s"] is not None
        finished = assertion
    return {"passed": bool(finished and assertion), "assertion_passed": bool(assertion),
            "tool_calls": tools, "tool_results": results, "result": final}


def cancellation_settlement(base_url, before, timeout=10):
    start = time.perf_counter()
    prior = before.get("values", {}).get('apxinf_requests_total{status="cancelled"}', 0)
    snapshots = []
    while time.perf_counter() - start < timeout:
        sample = metrics_snapshot(base_url)
        snapshots.append({"elapsed_s": time.perf_counter() - start, **sample})
        values = sample.get("values", {})
        if (values.get("apxinf_active_requests") == 0 and values.get("apxinf_queued_requests") == 0 and
                values.get('apxinf_requests_total{status="cancelled"}', 0) > prior):
            return {"settled": True, "elapsed_s": time.perf_counter() - start, "samples": snapshots}
        time.sleep(.2)
    return {"settled": False, "elapsed_s": time.perf_counter() - start, "samples": snapshots}


def run(args):
    local_url(args.base_url)
    if args.repeats < 1 or args.timeout <= 0 or args.max_tokens < 1:
        raise ValueError("Use positive repeats, timeout, and output limits.")
    if args.max_context is not None and args.max_context < 1:
        raise ValueError("The client context limit must be positive.")
    status, document = request_json(args.base_url, "/v1/models")
    if status != 200:
        raise RuntimeError("Model discovery failed.")
    model = args.model or document["data"][0]["id"]
    ready_status, readiness = request_json(args.base_url, "/readyz")
    if ready_status != 200:
        raise RuntimeError("The service is not ready.")
    capabilities = readiness.get("worker", {}).get("capabilities", {})
    server_context = capabilities.get("max_context")
    max_context = args.max_context
    if type(server_context) is int and server_context > 0:
        if max_context is not None and max_context > server_context:
            raise ValueError("The client context limit exceeds the deployment limit.")
        max_context = max_context or server_context
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"format": "apxinf-claude-code-tasks-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "model": model, "client": str(args.claude), "tasks": [],
              "deployment_limits": capabilities, "client_context_limit": max_context,
              "client_request_output_limit": args.max_tokens,
              "profile": "bare, restricted, local dummy credential, isolated configuration, narrow tool allowlists",
              "limitations": ["Anthropic does not officially support Claude Code with non-Claude models.",
                              "These small tasks measure observed compatibility, not general coding quality.",
                              "CLI latency includes client startup, prompt construction, and tool execution.",
                              "A cancellation signal alone does not prove that the worker released its state."]}
    with metal_measurement_lock(args.metal_lock):
        for repeat in range(args.repeats):
            for name in args.tasks:
                directory = args.output_dir / f"{name}-{repeat}-{uuid.uuid4().hex[:8]}"
                spec = prepare_task(name, directory)
                config_dir = directory / "client-config"
                config_dir.mkdir()
                environment = client_environment(args.base_url, model, config_dir, args.max_tokens, max_context)
                command = [str(args.claude), "--bare", "--restricted", "-p", spec["prompt"],
                           "--model", model, "--output-format", "stream-json", "--verbose",
                           "--include-partial-messages", "--no-session-persistence",
                           "--setting-sources", "", "--permission-mode", "dontAsk",
                           "--max-turns", "8", "--tools", ",".join(spec["tools"])]
                if spec["allowed"]:
                    command += ["--allowedTools", ",".join(spec["allowed"])]
                before = metrics_snapshot(args.base_url)
                capture = capture_client(command, directory, environment, args.timeout, name == "cancel")
                outcome = task_outcome(name, spec, directory, capture)
                if name == "cancel" and capture["cancel_sent"]:
                    settlement = cancellation_settlement(args.base_url, before)
                    outcome["settlement"] = settlement
                    outcome["passed"] = outcome["passed"] and settlement["settled"]
                after = metrics_snapshot(args.base_url)
                record = {"task": name, "repeat": repeat, "directory": str(directory),
                          "command": command, "metrics_before": before, "metrics_after": after,
                          **capture, **outcome}
                report["tasks"].append(record)
                (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"task": name, "repeat": repeat, "passed": outcome["passed"],
                                  "elapsed_s": capture["elapsed_s"], "exit_code": capture["exit_code"],
                                  "tools": [item["name"] for item in outcome["tool_calls"]]}), flush=True)
    return 0 if all(task["passed"] for task in report["tasks"]) else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-url", default="http://127.0.0.1:8080")
    result.add_argument("--model")
    result.add_argument("--claude", type=Path, default=Path("/opt/homebrew/bin/claude"))
    result.add_argument("--tasks", nargs="+", choices=("read", "edit", "check", "cancel"),
                        default=["read", "edit", "check"])
    result.add_argument("--repeats", type=int, default=1)
    result.add_argument("--timeout", type=float, default=180)
    result.add_argument("--max-tokens", type=int, default=512)
    result.add_argument("--max-context", type=int,
                        help="Set the client window. By default, use the deployment readiness limit.")
    result.add_argument("--metal-lock", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
