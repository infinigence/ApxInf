"""Compare an owned local HTTP service with executable-identified native records."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from metal_matrix import require_parent_lock


def checkout_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def file_identity(path):
    checksum = hashlib.sha256()
    with path.open("rb") as executable:
        for block in iter(lambda: executable.read(1048576), b""):
            checksum.update(block)
    return {"bytes": path.stat().st_size, "sha256": checksum.hexdigest()}


def source_binding(receipt_path, metadata, probe, worker):
    if receipt_path is None:
        return {"status": "unverified", "reason": "no external build receipt supplied"}
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict) or not isinstance(receipt.get("binaries"), dict):
        raise ValueError("build receipt must contain a binary record object")
    if receipt.get("source_dirty") is not False and receipt.get("source_dirty") != "":
        raise ValueError("build receipt must identify a clean source checkout")
    if not isinstance(receipt.get("source_commit"), str) or not receipt["source_commit"]:
        raise ValueError("build receipt lacks a source commit")
    if receipt["source_commit"] != metadata.get("checkout_commit") or receipt["source_commit"] != checkout_commit():
        raise ValueError("build receipt and observed checkout commits differ")
    if metadata.get("checkout_dirty") is not False:
        raise ValueError("native observation requires a clean checkout for build-receipt binding")
    for name, identity in (("metal_text_probe", probe), ("apxinf-native-worker", worker)):
        recorded = receipt.get("binaries", {}).get(name)
        if not isinstance(recorded, dict) or recorded.get("sha256") != identity["sha256"] or type(recorded.get("bytes")) is not int or recorded["bytes"] != identity["bytes"]:
            raise ValueError(f"{name} differs from the external build receipt")
    return {"status": "verified_external_build_receipt", "source_commit": receipt["source_commit"],
            "scope": "selected probe and native worker executable files", "receipt": str(receipt_path),
            "server_process_binding": "requires the external owner's process evidence"}


def request(base, payload):
    encoded = json.dumps(payload).encode()
    return urllib.request.Request(base + "/v1/chat/completions", data=encoded,
                                  headers={"Content-Type": "application/json"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--native-probe", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--native-worker", type=Path, required=True)
    parser.add_argument("--direct-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metal-lock", type=Path, required=True)
    parser.add_argument("--lock-owned-by-parent", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a new output path to preserve prior evidence")
    require_parent_lock(args.metal_lock, args.lock_owned_by_parent)
    corpus = json.loads(args.corpus.read_text())
    direct = [json.loads(line) for line in args.direct_jsonl.read_text().splitlines() if line.strip()]
    metadata = [row for row in direct if row.get("kind") == "metadata"]
    if len(metadata) != 1:
        raise ValueError("native probe must contain exactly one metadata record")
    probe_identity = file_identity(args.native_probe)
    worker_identity = file_identity(args.native_worker)
    if metadata[0].get("executable_sha256") != probe_identity["sha256"]:
        raise ValueError("native observation differs from the selected probe executable")
    binding = source_binding(args.build_receipt, metadata[0], probe_identity, worker_identity)
    with urllib.request.urlopen(args.base_url + "/readyz", timeout=10) as response:
        ready = json.load(response)
    worker = ready["worker"]
    if worker["runtime"]["executable_sha256"] != worker_identity["sha256"]:
        raise ValueError("HTTP worker differs from the selected native executable")
    execution = worker["model_manifest"]["execution"]
    for key in ("allocator_cache_limit_bytes", "memory_limit_bytes"):
        if execution[key] != metadata[0][key]:
            raise ValueError("native and HTTP memory policies differ")
    if not ready["ready"] or worker["protocol"] != "apxinf-worker/2.7":
        raise ValueError("select the ready native prepared-code profile")
    if Path(worker["model_path"]).resolve() != Path(metadata[0]["model_path"]).resolve():
        raise ValueError("native and HTTP model directories differ")
    cases = {case["id"]: case for case in corpus["cases"]}
    if len(cases) != len(corpus["cases"]):
        raise ValueError("case IDs must be unique")
    rows = []
    for native in direct:
        if native.get("kind") != "result":
            continue
        require_parent_lock(args.metal_lock, args.lock_owned_by_parent)
        case = cases[native["id"]]
        payload = {"model": ready["model"], "messages": case["messages"], "max_tokens": case["max_tokens"],
                   "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        started = time.perf_counter()
        first = None
        content = []
        usage = None
        finish_reason = None
        with urllib.request.urlopen(request(args.base_url, payload), timeout=args.timeout) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(event["error"])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    text = choice.get("delta", {}).get("content")
                    if text:
                        if first is None:
                            first = time.perf_counter()
                        content.append(text)
                    finish_reason = choice.get("finish_reason") or finish_reason
        elapsed = time.perf_counter() - started
        content = "".join(content)
        if usage is None or finish_reason is None:
            raise ValueError("HTTP response lacks final usage or completion reason")
        rows.append({"id": native["id"], "repeat": native["repeat"], "native": native,
                     "http": {"content": content, "usage": usage, "finish_reason": finish_reason,
                              "wall_ms": elapsed * 1000,
                              "ttft_ms": None if first is None else (first - started) * 1000},
                     "output_equal": content == native["content"],
                     "usage_equal": usage["prompt_tokens"] == native["input_tokens"]
                                    and usage["completion_tokens"] == native["completion_tokens"],
                     "finish_equal": finish_reason == native["finish_reason"]})
    if not rows:
        raise ValueError("native probe contains no request results")
    report = {"checkout_commit": checkout_commit(), "source_binding": binding,
              "model_binding": {"status": "directory_path_only", "artifact_equality_verified": False},
              "native_metadata": metadata[0], "probe_executable": probe_identity,
              "worker_executable": worker_identity, "ready": ready,
              "rows": rows, "comparison_scope": "serial diagnostic, not paired performance acceptance",
              "all_outputs_equal": all(row["output_equal"] and row["usage_equal"] and row["finish_equal"] for row in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as result_file:
        result_file.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"requests": len(rows), "all_outputs_equal": report["all_outputs_equal"], "source_binding": binding["status"]}))
    if not report["all_outputs_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
