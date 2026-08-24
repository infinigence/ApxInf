#!/usr/bin/env python3
"""Unified layered latency benchmark for the pi05 serving stack — L0 / L1 / L2 / L3.

One entry point, driven by ``--layer × --precision × weights-source × input``,
that folds the former ``bench_pi05_layers.py`` (L0/L1/L2 in-process) and
``bench_pi05_openpi_latency.py`` (L3 websocket). The serving call peels into four
concentric shells, each the cost of the layer around the one inside it:

* **L0 model** — ``Model._infer_patches``: pure engine forward, inputs already
  patch-embedded (vision→patches skipped). The floor.
* **L1 rust** — ``Model.infer_rgb``: the ``apxinf_py`` binding from resized RGB;
  adds Rust-side vision→patches (in the CUDA graph) + PyO3 marshalling over L0.
* **L2 python api** — ``Pi05Policy.infer``: adds the numpy pre chain
  (parse/resize/tokenize) + post chain (trim/unnormalize) around L1. Its default
  latent is generated in the runtime's device buffer.
* **L3 websocket** — one ``client.infer`` round trip: adds transport (websocket +
  msgpack) + the server processor pipeline around the bare model. Measured against
  a *running* server (``--host/--port``); this script attaches, it does not spawn.

Weights come from a real checkpoint (``--model-dir``) or, when it is omitted, from
deterministic **synthetic weights** — the checkpoint-free default (equivalent to an
explicit ``--random-weights``) that runs the engine with no checkpoint on disk
(latency depends on shape+dtype, not trained values). A checkpoint defaults to its
*native* config (e.g. pi05_libero_base = H50), matching the LIBERO deployment;
``--action-horizon`` overrides that (it is a sequence length, not a weight
dimension), while the remaining shape knobs (``--views/--image-size/
--num-flow-steps/--max-token-len``) are synthetic-only. Synthetic mode covers
**L0/L1/L2**: L2 wraps the engine in synthetic processors (a fixed-length tokenizer
+ identity unnormalize, so its actions are latency-only). L3 attaches to a running
server (``--host/--port``) and needs no local weights — serve with
``--random-weights`` for a fully checkpoint-free L3.

    # checkpoint-free engine floor — the zero-config default (no download)
    python scripts/bench_pi05.py --precision bf16 --views 2 --token-count 10

    # full in-process breakdown against a checkpoint
    python scripts/bench_pi05.py --model-dir /path/to/pi05 --layer l0,l1,l2 \
        --precision bf16 --prompt "put both moka pots on the stove"

    # same checkpoint, forced to a 10-step chunk instead of its native H=50
    python scripts/bench_pi05.py --model-dir /path/to/pi05 --layer l0,l1,l2 \
        --precision bf16 --action-horizon 10

    # L3 against a running websocket server
    python scripts/bench_pi05.py --layer l3 --precision bf16 \
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

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_APXINF_PKG = _REPO_ROOT / "python" / "apxinf"
if _APXINF_PKG.is_dir() and str(_APXINF_PKG) not in sys.path:
    sys.path.insert(0, str(_APXINF_PKG))

# doc/pi05-cuda-regression.md primary/worst-case LIBERO prompts
# (T = PaliGemma token count).
PROMPT_T10 = "put both moka pots on the stove"
PROMPT_T21 = (
    "put the white mug on the left plate and put the yellow and white mug "
    "on the right plate"
)

ALL_LAYERS = ("l0", "l1", "l2", "l3")
IN_PROCESS = ("l0", "l1", "l2")


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


def _parse_layers(spec: str) -> list[str]:
    if spec == "all":
        return list(ALL_LAYERS)
    picked = [item.strip().lower() for item in spec.split(",") if item.strip()]
    unknown = [item for item in picked if item not in ALL_LAYERS]
    if unknown:
        raise SystemExit(f"unknown --layer value(s): {', '.join(unknown)} (choose from l0,l1,l2,l3,all)")
    # Preserve L0⊂L1⊂L2⊂L3 order and drop duplicates.
    return [layer for layer in ALL_LAYERS if layer in picked]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--layer",
        default=None,
        help="comma list of l0,l1,l2,l3 or `all` (default: all with --model-dir, else l0,l1)",
    )
    p.add_argument("--precision", choices=("bf16", "fp8", "int8"), default="bf16")
    p.add_argument("--model", default="pi05", help="model name for the random-weights engine")
    p.add_argument("--device", default="cuda:0")

    # Not unconditionally required: L3 is attach-only (connects to a running
    # server) and needs no local weights. A source is required only when an
    # in-process layer (l0/l1/l2) is requested — enforced in main().
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("--model-dir", type=pathlib.Path, help="checkpoint dir/index (real weights)")
    source.add_argument(
        "--random-weights", action="store_true", help="checkpoint-free engine (L0/L1/L2)"
    )

    # FP8 knobs. In random mode these feed `Model.random`; in checkpoint mode the
    # loader reads calibration.json / tactics.json from the model dir.
    p.add_argument("--calibration", help="FP8 calibration json or `uniform:SCALE` (random mode)")
    p.add_argument("--tactics", type=pathlib.Path, help="FP8 tactics json (random mode)")

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
        help="model action width (random) or L2 deploy width (checkpoint)",
    )
    p.add_argument("--num-flow-steps", type=int, help="diffusion flow steps (random)")
    p.add_argument("--max-token-len", type=int, help="max prompt tokens (random)")
    p.add_argument("--seed", type=int, default=0, help="random-weights seed")

    # Input workload.
    p.add_argument("--prompt", default=PROMPT_T10, help="prompt for checkpoint/L2/L3 tokenize")
    p.add_argument("--token-count", type=int, default=10, help="synthetic token count (random L0/L1)")

    # L3 server.
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--out", type=pathlib.Path)
    return p.parse_args()


def _run_in_process(handle, policy, layers, observation, rgb, token_ids, noise, patches, warmup, samples):
    """Time the requested subset of the in-process layers L0/L1/L2."""
    runners = {
        "l0": lambda: handle._infer_patches(patches, token_ids, noise),
        "l1": lambda: handle.infer_rgb(rgb, "nhwc", token_ids, noise),
        "l2": lambda: policy.infer(observation),
    }
    raw = {}
    for layer in IN_PROCESS:
        if layer in layers:
            raw[layer] = _time_loop(runners[layer], warmup, samples)
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

    # Resolve the weights source and the default layer set (minimal-surprise):
    #   --model-dir      -> checkpoint; default layers = all (native config, e.g. H50)
    #   --random-weights -> synthetic;  default layers = l0,l1
    #   neither          -> synthetic;  default layers = l0,l1
    # so bare `python bench_pi05.py` is a checkpoint-free L0/L1 run, while every
    # existing `--model-dir ...` invocation keeps its full L0/L1/L2(/L3) behavior.
    checkpoint = args.model_dir is not None
    random = not checkpoint
    layers = _parse_layers(args.layer if args.layer is not None else ("all" if checkpoint else "l0,l1"))

    # --- one parameter-conflict validation pass (fail fast, consistent messages) ---
    in_process = [layer for layer in layers if layer in IN_PROCESS]
    # L2 wraps the engine in synthetic processors when checkpoint-free, so it needs
    # no local weights. L3 attaches to a running server and is never gated on a
    # source here (start that server with --random-weights for a synthetic L3).
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
    # FP8 tuning knobs: synthetic + fp8 only (a checkpoint reads calibration/tactics
    # from its own dir; bf16/int8 have no per-tensor FP8 scales).
    fp8_knobs = [
        name for name, value in (("--tactics", args.tactics), ("--calibration", args.calibration))
        if value is not None
    ]
    if fp8_knobs:
        if checkpoint:
            raise SystemExit(
                f"{', '.join(fp8_knobs)} only apply to synthetic weights (drop --model-dir); "
                "a checkpoint loads calibration/tactics from its own directory"
            )
        if args.precision != "fp8":
            raise SystemExit(f"{', '.join(fp8_knobs)} only apply to --precision fp8")

    handle = None
    policy = None
    observation = rgb = token_ids = noise = patches = None

    if in_process:
        if random:
            from apxinf import Model

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
                tactics=str(args.tactics) if args.tactics is not None else None,
                seed=args.seed,
            )
            if "l2" in layers:
                # L2 needs a policy: wrap the random engine in synthetic processors
                # (checkpoint-free tokenizer + identity unnormalize). Drive L0/L1
                # from the same pipeline so all layers see one consistent input.
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
            patch_rows = handle.num_views * handle.patches_per_view
            patch_width = 3 * handle.patch_size * handle.patch_size
            patches = np.zeros((patch_rows, patch_width), dtype=np.float32)
        else:
            from apxinf import Pi05Policy

            policy = Pi05Policy.from_pretrained(
                args.model_dir,
                device=args.device,
                precision=args.precision,
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
            # Run the pre chain once so L0/L1 are driven by the real serving inputs.
            data = policy.input_pipeline({OBSERVATION: observation, PROMPT: args.prompt})
            rgb = data[RGB]
            token_ids = np.asarray(data[TOKEN_IDS], dtype=np.uint32)
            noise = np.random.default_rng(args.seed).standard_normal(
                (handle.action_horizon, handle.action_dim), dtype=np.float32
            )
            patch_rows = handle.num_views * handle.patches_per_view
            patch_width = 3 * handle.patch_size * handle.patch_size
            patches = np.zeros((patch_rows, patch_width), dtype=np.float32)
            token_count = int(token_ids.size)

    in_process_raw = {}
    if in_process:
        in_process_raw = _run_in_process(
            handle, policy, layers, observation, rgb, token_ids, noise, patches,
            args.warmup, args.samples,
        )

    l3_metadata = None
    l3_segments = None
    if "l3" in layers:
        l3_metadata, l3_segments = _run_l3(
            args.host, args.port, args.precision, args.prompt, args.warmup, args.samples
        )

    # Assemble report.
    layer_names = {"l0": "L0_model", "l1": "L1_rust", "l2": "L2_python_api"}
    in_process_report = {layer_names[layer]: _stats(ms) for layer, ms in in_process_raw.items()}
    result = {
        "schema": "apxinf.pi05.latency.v2",
        "git_commit": _git_commit(),
        "precision": args.precision,
        "device": args.device,
        "weights": "synthetic" if random else "checkpoint",
        "layers": layers,
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
    if in_process_report:
        result["layers_ms"] = in_process_report
        result["raw_ms"] = {layer_names[layer]: ms for layer, ms in in_process_raw.items()}
    if l3_segments is not None:
        result["l3_segments_ms"] = {name: _stats(ms) for name, ms in l3_segments.items()}
        result["l3_raw_ms"] = l3_segments
        result["l3_server_metadata"] = l3_metadata

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
        print(f"{'layer':<16}{'p50':>9}{'p95':>9}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
        for layer in in_process:
            name = layer_names[layer]
            s = in_process_report[name]
            print(
                f"{name:<16}{s['p50']:>9.2f}{s['p95']:>9.2f}{s['mean']:>9.2f}"
                f"{s['std']:>9.2f}{s['min']:>9.2f}{s['max']:>9.2f}"
            )
        if "L0_model" in in_process_report and "L1_rust" in in_process_report:
            d10 = in_process_report["L1_rust"]["p50"] - in_process_report["L0_model"]["p50"]
            print(f"\n  L1-L0 (rust vision->patches + PyO3): {d10:+.2f} ms")
        if "L1_rust" in in_process_report and "L2_python_api" in in_process_report:
            d21 = in_process_report["L2_python_api"]["p50"] - in_process_report["L1_rust"]["p50"]
            print(f"  L2-L1 (numpy pre/post chain):        {d21:+.2f} ms")

    if l3_segments is not None:
        stats = result["l3_segments_ms"]
        print(
            f"\nL3 websocket  views={l3_metadata.get('num_views')} precision={args.precision} "
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
