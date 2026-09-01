#!/usr/bin/env python3
r"""Benchmark PI0.5 inference interfaces.

``--interface`` selects one or more public call boundaries:

* **model** — ``Model.infer_rgb`` from resized RGB.
* **policy** — ``Pi05Policy.infer`` with the full processor pipeline.
* **websocket** — one ``client.infer`` round trip to a running server.

Weights come from a real checkpoint (``--model-dir``) or, when it is omitted, from
deterministic **synthetic weights** — the checkpoint-free default (equivalent to an
explicit ``--random-weights``) that runs the engine with no checkpoint on disk
(latency depends on shape+dtype, not trained values). A checkpoint defaults to its
*native* config (e.g. pi05_libero_base = H50), matching the LIBERO deployment;
``--action-horizon`` overrides that (it is a sequence length, not a weight
dimension), while the remaining shape knobs (``--views/--image-size/
--num-flow-steps/--max-token-len``) are synthetic-only. In synthetic mode, the
policy interface uses a fixed-length tokenizer and identity unnormalizer, so its
actions are latency-only. The websocket interface attaches to a running server
(``--host/--port``) and needs no local weights.

    # checkpoint-free model interface
    python -m evaluation.benchmarks.pi05 --precision bf16 --views 2 --token-count 10

    # in-process interfaces against a checkpoint
    python -m evaluation.benchmarks.pi05 --model-dir /path/to/pi05 \
        --interface model,policy \
        --precision bf16 --prompt "put both moka pots on the stove"

    # same checkpoint, forced to a 10-step chunk instead of its native H=50
    python -m evaluation.benchmarks.pi05 --model-dir /path/to/pi05 \
        --interface model,policy \
        --precision bf16 --action-horizon 10

    # running websocket server
    python -m evaluation.benchmarks.pi05 --interface websocket --precision bf16 \
        --host 127.0.0.1 --port 8000 --prompt "put both moka pots on the stove"
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_APXINF_PKG = _REPO_ROOT / "python" / "apxinf"
if _APXINF_PKG.is_dir() and str(_APXINF_PKG) not in sys.path:
    sys.path.insert(0, str(_APXINF_PKG))

from apxinf._tactics import resolve_pi05_tactics

# doc/pi05-cuda-regression.md primary/worst-case LIBERO prompts
# (T = PaliGemma token count).
PROMPT_T10 = "put both moka pots on the stove"
PROMPT_T21 = (
    "put the white mug on the left plate and put the yellow and white mug "
    "on the right plate"
)

ALL_INTERFACES = ("model", "policy", "websocket")
IN_PROCESS_INTERFACES = ("model", "policy")


def _stats(samples_ms):
    ordered = sorted(samples_ms)
    n = len(ordered)
    return {
        "p50": ordered[int(0.50 * (n - 1))],
        "p95": ordered[int(0.95 * (n - 1))],
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered) if n > 1 else 0.0,
        "samples": n,
    }


def _time_loop(fn, warmup, samples):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(samples):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000.0)
    return out


def _git_commit():
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        )
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return rev.decode().strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _parse_interfaces(spec: str) -> list[str]:
    if spec == "all":
        return list(ALL_INTERFACES)
    picked = [item.strip().lower() for item in spec.split(",") if item.strip()]
    unknown = [item for item in picked if item not in ALL_INTERFACES]
    if unknown:
        raise SystemExit(
            f"unknown --interface value(s): {', '.join(unknown)} "
            "(choose from model,policy,websocket,all)"
        )
    return [interface for interface in ALL_INTERFACES if interface in picked]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--interface",
        default=None,
        help=(
            "comma list of model,policy,websocket or all "
            "(default: model,policy with --model-dir; otherwise model)"
        ),
    )
    p.add_argument("--precision", choices=("bf16", "fp8", "int8"), default="bf16")
    p.add_argument("--model", default="pi05", help="model name for the random-weights engine")
    p.add_argument("--device", default="cuda:0")

    # Websocket attaches to a running server and needs no local weights.
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("--model-dir", type=pathlib.Path, help="checkpoint dir/index (real weights)")
    source.add_argument(
        "--random-weights", action="store_true", help="checkpoint-free model/policy"
    )

    # Calibration is public for synthetic FP8 latency runs. Tactics are routed
    # internally by CUDA SM + precision below.
    p.add_argument("--calibration", help="FP8 calibration json or `uniform:SCALE` (random mode)")
    # Internal escape hatch for tactic generation/debugging. Normal benchmark
    # runs select the repository's validated JSON from CUDA SM + precision.
    p.add_argument("--tactics", type=pathlib.Path, help=argparse.SUPPRESS)

    # Architecture overrides — synthetic shapes. `--action-horizon` is the one
    # knob that also applies to a checkpoint (see below).
    p.add_argument("--views", type=int, help="num camera views (random)")
    p.add_argument("--image-size", type=int, help="square image edge (random)")
    p.add_argument(
        "--action-horizon",
        type=int,
        help="action horizon; overrides the checkpoint's config.json value",
    )
    p.add_argument(
        "--action-dim",
        type=int,
        help="model action width (random) or policy output width (checkpoint)",
    )
    p.add_argument("--num-flow-steps", type=int, help="diffusion flow steps (random)")
    p.add_argument("--max-token-len", type=int, help="max prompt tokens (random)")
    p.add_argument("--seed", type=int, default=0, help="random-weights seed")

    # Input workload.
    p.add_argument("--prompt", default=PROMPT_T10, help="prompt for policy/websocket input")
    p.add_argument("--token-count", type=int, default=10, help="synthetic token count")

    # Websocket server.
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--out", type=pathlib.Path)
    return p.parse_args()


def _run_in_process(handle, policy, interfaces, observation, rgb, token_ids, noise, warmup, samples):
    runners = {
        "model": lambda: handle.infer_rgb(rgb, "nhwc", token_ids, noise),
        "policy": lambda: policy.infer(observation),
    }
    raw = {}
    for interface in IN_PROCESS_INTERFACES:
        if interface in interfaces:
            raw[interface] = _time_loop(runners[interface], warmup, samples)
    return raw


def _run_l3(host, port, precision, prompt, warmup, samples):
    """Attach to a running websocket server and time one round trip per call."""
    from openpi_client import websocket_client_policy

    for variable in ("NO_PROXY", "no_proxy"):
        entries = [item for item in os.environ.get(variable, "").split(",") if item]
        if host not in entries:
            entries.append(host)
        os.environ[variable] = ",".join(entries)

    client = websocket_client_policy.WebsocketClientPolicy(host, port)
    metadata = client.get_server_metadata()
    actual_precision = metadata.get("precision")
    if actual_precision != precision:
        raise RuntimeError(f"server precision is {actual_precision!r}, expected {precision!r}")
    action_horizon = int(metadata.get("action_horizon", 10))

    rng = np.random.default_rng(0)
    observation = {
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": prompt,
    }

    def one_call() -> dict:
        started = time.perf_counter()
        response = client.infer(observation)
        round_trip_ms = (time.perf_counter() - started) * 1000.0
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.shape[0] != action_horizon or not np.isfinite(actions).all():
            raise RuntimeError(f"bad actions shape/values: {actions.shape}")
        policy_timing = response.get("policy_timing", {}) or {}
        server_timing = response.get("server_timing", {}) or {}
        model_ms = float(policy_timing.get("infer_ms", 0.0))
        server_total_ms = float(policy_timing.get("policy_ms", model_ms))
        server_compute_ms = float(server_timing.get("infer_ms", server_total_ms))
        return {
            "round_trip": round_trip_ms,
            "model": model_ms,
            "server_processor": max(0.0, server_total_ms - model_ms),
            "transport": max(0.0, round_trip_ms - server_compute_ms),
        }

    for _ in range(warmup):
        one_call()
    segments = {"round_trip": [], "model": [], "server_processor": [], "transport": []}
    for _ in range(samples):
        for name, value in one_call().items():
            segments[name].append(value)
    return metadata, segments


def main() -> None:
    args = parse_args()

    checkpoint = args.model_dir is not None
    random = not checkpoint
    interfaces = _parse_interfaces(
        args.interface
        if args.interface is not None
        else ("model,policy" if checkpoint else "model")
    )

    in_process = [
        interface
        for interface in interfaces
        if interface in IN_PROCESS_INTERFACES
    ]
    # Architecture overrides fix synthetic shapes. `--action-horizon` is the
    # exception: it is a sequence length rather than a weight dimension, so an
    # explicit value outranks the checkpoint's config.json and the same weights
    # run at the requested chunk length.
    override_flags = {
        "--views": args.views,
        "--image-size": args.image_size,
        "--num-flow-steps": args.num_flow_steps,
        "--max-token-len": args.max_token_len,
    }
    if checkpoint:
        stray = [name for name, value in override_flags.items() if value is not None]
        if stray:
            raise SystemExit(
                f"architecture overrides {', '.join(stray)} are only valid without "
                "--model-dir (they reshape synthetic weights; a checkpoint runs its "
                "native config apart from --action-horizon)"
            )
    # Calibration remains a synthetic FP8 knob here. Tactics are selected below
    # from CUDA SM + precision for both synthetic and checkpoint benchmarks.
    if checkpoint and args.calibration is not None:
        raise SystemExit(
            "--calibration only applies to synthetic weights (drop --model-dir); "
            "a checkpoint loads calibration.json from its model directory"
        )
    if args.calibration is not None and args.precision != "fp8":
        raise SystemExit("--calibration only applies to --precision fp8")
    if args.tactics is not None and args.precision == "int8":
        raise SystemExit("--tactics only applies to --precision bf16 or fp8")

    handle = None
    policy = None
    tactics = args.tactics
    observation = rgb = token_ids = noise = None

    if in_process:
        if random:
            from apxinf import Model

            # Random engines bypass Pi05Policy.from_pretrained, so this synthetic
            # benchmark is the sole caller that must resolve the package default.
            tactics = resolve_pi05_tactics(
                args.device, args.precision, override=args.tactics
            )
            if tactics is not None:
                print(
                    f"using {args.precision} tactics for {args.device}: {tactics}",
                    file=sys.stderr,
                )
            calibration = args.calibration
            if args.precision == "fp8" and calibration is None:
                # Synthetic FP8 has no calibration file; a uniform scale keeps the
                # FP8 path on (the kernel falls back to a default tactic).
                calibration = "uniform:1.0"
            handle = Model.random(
                model=args.model,
                device=args.device,
                precision=args.precision,
                num_views=args.views if args.views is not None else 2,
                image_size=args.image_size if args.image_size is not None else 224,
                action_horizon=args.action_horizon if args.action_horizon is not None else 10,
                action_dim=args.action_dim if args.action_dim is not None else 32,
                num_flow_steps=args.num_flow_steps if args.num_flow_steps is not None else 10,
                max_token_len=args.max_token_len if args.max_token_len is not None else 200,
                calibration=calibration,
                tactics=str(tactics) if tactics is not None else None,
                seed=args.seed,
            )
            if "policy" in interfaces:
                from apxinf import Pi05Policy
                from apxinf.processors.transforms import (
                    OBSERVATION,
                    PROMPT,
                    RGB,
                    TOKEN_IDS,
                )

                policy = Pi05Policy.from_random(
                    handle,
                    token_count=args.token_count,
                    action_dim=(args.action_dim or None),
                    seed=args.seed,
                )
                size = handle.image_size
                rng = np.random.default_rng(0)
                observation = {
                    key: rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
                    for key in policy.image_keys
                }
                observation["prompt"] = args.prompt
                data = policy.input_pipeline({OBSERVATION: observation, PROMPT: args.prompt})
                rgb = data[RGB]
                token_ids = np.asarray(data[TOKEN_IDS], dtype=np.uint32)
                noise = np.random.default_rng(args.seed).standard_normal(
                    (handle.action_horizon, handle.action_dim), dtype=np.float32
                )
                token_count = int(token_ids.size)
            else:
                views, size = handle.num_views, handle.image_size
                token_count = args.token_count
                token_ids = np.zeros(token_count, dtype=np.uint32)
                noise = np.zeros((handle.action_horizon, handle.action_dim), dtype=np.float32)
                rgb = np.zeros((views, size, size, 3), dtype=np.uint8)
        else:
            from apxinf import Pi05Policy

            policy = Pi05Policy.from_pretrained(
                args.model_dir,
                device=args.device,
                precision=args.precision,
                tactics=args.tactics,
                action_dim=(args.action_dim or None),
                action_horizon=args.action_horizon,
            )
            handle = policy.model
            from apxinf.processors.transforms import (
                OBSERVATION,
                PROMPT,
                RGB,
                TOKEN_IDS,
            )

            size = handle.image_size
            rng = np.random.default_rng(0)
            observation = {
                "observation/image": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
                "observation/wrist_image": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
                "prompt": args.prompt,
            }
            data = policy.input_pipeline({OBSERVATION: observation, PROMPT: args.prompt})
            rgb = data[RGB]
            token_ids = np.asarray(data[TOKEN_IDS], dtype=np.uint32)
            noise = np.random.default_rng(args.seed).standard_normal(
                (handle.action_horizon, handle.action_dim), dtype=np.float32
            )
            token_count = int(token_ids.size)

    in_process_raw = {}
    if in_process:
        in_process_raw = _run_in_process(
            handle, policy, interfaces, observation, rgb, token_ids, noise,
            args.warmup, args.samples,
        )

    l3_metadata = None
    l3_segments = None
    if "websocket" in interfaces:
        l3_metadata, l3_segments = _run_l3(
            args.host, args.port, args.precision, args.prompt, args.warmup, args.samples
        )

    in_process_report = {
        interface: _stats(ms) for interface, ms in in_process_raw.items()
    }
    result = {
        "schema": "apxinf.pi05.latency.v3",
        "git_commit": _git_commit(),
        "precision": args.precision,
        "device": args.device,
        "weights": "synthetic" if random else "checkpoint",
        "interfaces": interfaces,
        "warmup": args.warmup,
        "samples": args.samples,
    }
    if handle is not None:
        result["workload"] = {
            "token_count": token_count,
            "action_horizon": handle.action_horizon,
            "action_dim": handle.action_dim,
            "num_views": handle.num_views,
            "image_size": handle.image_size,
        }
        if not random:
            result["model_dir"] = str(args.model_dir)
            result["workload"]["prompt"] = args.prompt
            result["workload"]["deploy_action_dim"] = policy.action_dim
        if tactics is not None:
            result["tactics"] = str(tactics)
    if in_process_report:
        result["interfaces_ms"] = in_process_report
        result["raw_ms"] = in_process_raw
    if l3_segments is not None:
        result["websocket_segments_ms"] = {
            name: _stats(ms) for name, ms in l3_segments.items()
        }
        result["websocket_raw_ms"] = l3_segments
        result["websocket_server_metadata"] = l3_metadata

    # Console table.
    if in_process_report:
        hdr = (
            f"pi05 in-process latency  |  {args.precision}  "
            f"{'synthetic' if random else 'checkpoint'}  "
            f"H={handle.action_horizon} Dmodel={handle.action_dim} "
            f"views={handle.num_views} T={token_count}"
        )
        print(hdr)
        print("-" * len(hdr))
        print(
            f"{'interface':<16}{'p50':>9}{'p95':>9}{'mean':>9}"
            f"{'std':>9}{'min':>9}{'max':>9}"
        )
        for interface in in_process:
            s = in_process_report[interface]
            print(
                f"{interface:<16}{s['p50']:>9.2f}{s['p95']:>9.2f}{s['mean']:>9.2f}"
                f"{s['std']:>9.2f}{s['min']:>9.2f}{s['max']:>9.2f}"
            )
        if "model" in in_process_report and "policy" in in_process_report:
            overhead = (
                in_process_report["policy"]["p50"]
                - in_process_report["model"]["p50"]
            )
            print(f"\n  policy-model processor overhead: {overhead:+.2f} ms")

    if l3_segments is not None:
        stats = result["websocket_segments_ms"]
        print(
            f"\nwebsocket  views={l3_metadata.get('num_views')} precision={args.precision} "
            f"prompt={args.prompt!r} ({args.warmup} warmup + {args.samples} samples)"
        )
        header = f"{'segment':<18}{'p50':>9}{'p95':>9}{'min':>9}{'max':>9}{'mean':>9}{'std':>8}"
        print(header)
        print("-" * len(header))
        for name in ("round_trip", "model", "server_processor", "transport"):
            s = stats[name]
            print(
                f"{name:<18}{s['p50']:>9.3f}{s['p95']:>9.3f}{s['min']:>9.3f}"
                f"{s['max']:>9.3f}{s['mean']:>9.3f}{s['std']:>8.3f}"
            )

    if policy is not None:
        policy.close()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
