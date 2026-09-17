#!/usr/bin/env python3
"""Layered latency benchmark for the π0-FAST autoregressive token VLA — L1 / L2.

π0-FAST is ApxInf's token VLA: instead of one flow-matching pass it *decodes*
FAST action tokens one at a time, so a call's cost decomposes as::

    latency(frame) = fixed + steps(frame) * per_step

where ``fixed`` is the shape-determined prologue (vision tower + prompt prefill +
the first argmax) and ``steps`` is how many tokens that frame needs. The decode
ends at the ``|`` terminator, which lands at token 12–32 of the checkpoint's
256-token budget on LIBERO frames, so ``steps`` varies per frame and a
least-squares fit over frames with distinct step counts recovers both terms
without touching the Rust build (``--mode ar``).

Two layers, matching the serving stack's shells:

* **L1 rust** — ``Model.infer_action_tokens_rgb``: the ``apxinf_py`` binding from
  already-resized RGB, returning raw FAST token ids (includes the terminator).
* **L2 python api** — ``Pi0FastPolicy.infer``: adds prompt assembly (task +
  discretized state), FAST BPE detokenization, the orthonormal DCT and the
  checkpoint's action unnormalization around L1.

There is no L0 (π0-FAST exposes no patch-level entry point — the runtime takes
RGB) and no L3 (the websocket server is the PI0.5 stack).

Frames come from ``--frames`` — an ``.npz`` carrying ``base_<variant>`` /
``wrist_<variant>`` / ``state`` / ``task``, as
``devlocal/pi0-fast/scripts/collect_frames.py`` writes — or, when it is omitted,
from deterministic synthetic frames sized by the checkpoint. Every dimension is
the checkpoint's ``config.json``: π0-FAST has no synthetic weights and no shape
overrides, so ``--model-dir`` is required.

    # L1/L2 latency on synthetic frames
    python scripts/bench_pi0_fast.py --model-dir /path/to/pi0fast-libero \\
        --state-key observation/state

    # fixed cost vs per-step cost over real LIBERO frames
    python scripts/bench_pi0_fast.py --model-dir /path/to/pi0fast-libero \\
        --state-key observation/state --mode ar \\
        --frames devlocal/pi0-fast/results/raw/libero_frames.npz
"""

from __future__ import annotations

import argparse
import json
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

#: π0-FAST's layer set. L0/L3 belong to other families (see the module docstring).
ALL_LAYERS = ("l1", "l2")

#: A LIBERO-scale instruction, used only by the synthetic-frame default.
DEFAULT_PROMPT = "put both the alphabet soup and the tomato sauce in the basket"


def _stats(samples_ms) -> dict:
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


def _git_commit() -> str:
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
        raise SystemExit(
            f"unknown --layer value(s): {', '.join(unknown)} (choose from l1,l2,all)"
        )
    return [layer for layer in ALL_LAYERS if layer in picked]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-dir",
        type=pathlib.Path,
        required=True,
        help="π0-FAST checkpoint (LeRobot layout with config.json)",
    )
    p.add_argument("--layer", default="l1,l2", help="comma list of l1,l2 or `all`")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--precision",
        default="auto",
        help="runtime precision; π0-FAST runs bf16, so `auto` is the normal value",
    )
    p.add_argument(
        "--state-key",
        required=True,
        help="wire key of the proprioceptive vector π0-FAST discretizes into the prompt",
    )
    p.add_argument("--prompt-key", default="prompt", help="observation key of the task string")
    p.add_argument(
        "--image-keys",
        help="comma list of camera keys (default: the policy's own view slots)",
    )
    p.add_argument("--action-dim", type=int, help="override the checkpoint's deploy action width")
    p.add_argument("--action-horizon", type=int, help="override the checkpoint's action horizon")

    # Input workload.
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="task string (synthetic frames)")
    p.add_argument(
        "--frames",
        type=pathlib.Path,
        help="frames .npz (base_<variant>/wrist_<variant>/state/task); default synthetic",
    )
    p.add_argument(
        "--frame-variant",
        choices=("flipped", "raw"),
        default="flipped",
        help="which camera copies to drive when --frames carries both",
    )
    p.add_argument("--frames-limit", type=int, help="use only the first N frames")
    p.add_argument("--frames-count", type=int, default=10, help="synthetic frame count")
    p.add_argument("--seed", type=int, default=0, help="synthetic frame seed")

    # Measurement.
    p.add_argument("--mode", choices=("latency", "ar", "all"), default="latency")
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--survey",
        type=int,
        default=20,
        help="AR mode: frames surveyed (one timed call each) for distinct step counts",
    )
    p.add_argument("--repeats", type=int, default=3, help="AR mode: timed repeats per step count")
    p.add_argument(
        "--full-decode",
        action="store_true",
        help="also time L1 with no stop token (the full max_action_tokens budget)",
    )
    p.add_argument("--out", type=pathlib.Path)
    return p.parse_args()


def _load_frames(path, variant: str, limit) -> list[dict]:
    """Read the ``base_*/wrist_*/state/task`` frames ``collect_frames.py`` writes."""
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    def pick(prefix: str, *, per_frame_variant: bool) -> str:
        names = (
            (f"{prefix}_{variant}", f"{prefix}_raw", f"{prefix}_flipped")
            if per_frame_variant
            else (prefix, f"{prefix}_{variant}")
        )
        for name in names:
            if name in keys:
                return name
        raise SystemExit(f"{path}: no {prefix!r} array (have {sorted(keys)})")

    base = pick("base", per_frame_variant=True)
    wrist = pick("wrist", per_frame_variant=True)
    state = pick("state", per_frame_variant=False)
    task = pick("task", per_frame_variant=False)
    count = len(data[state])
    if limit is not None:
        count = min(count, int(limit))
    return [
        {
            "image": np.ascontiguousarray(data[base][index]),
            "wrist": np.ascontiguousarray(data[wrist][index]),
            "state": np.ascontiguousarray(data[state][index], dtype=np.float32),
            "task": str(data[task][index]),
        }
        for index in range(count)
    ]


def _synthetic_frames(policy, count: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    size = policy.model.image_size
    state_dim = int(policy.metadata["state_dim"])
    return [
        {
            "image": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
            "wrist": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
            "state": np.zeros(state_dim, dtype=np.float32),
            "task": None,
        }
        for _ in range(count)
    ]


def _observations(policy, frames, prompt: str) -> list[dict]:
    image_keys = tuple(policy.image_keys)
    if len(image_keys) != 2:
        raise SystemExit(
            f"this bench drives two cameras, but the checkpoint consumes {len(image_keys)} "
            f"({list(image_keys)}); extend _observations for more views"
        )
    return [
        {
            image_keys[0]: frame["image"],
            image_keys[1]: frame["wrist"],
            policy.state_key: frame["state"],
            policy.prompt_key: frame["task"] or prompt,
        }
        for frame in frames
    ]


def _prepare(policy, observation) -> tuple[np.ndarray, np.ndarray]:
    """The policy's own pre chain, so L1 and L2 see byte-identical inputs."""
    from apxinf.processors.transforms import OBSERVATION, PROMPT, RGB

    prompt = observation[policy.prompt_key]
    rgb = policy.input_pipeline({OBSERVATION: observation, PROMPT: prompt})[RGB]
    token_ids = np.asarray(policy._prompt_ids(observation, prompt), dtype=np.uint32)
    return rgb, token_ids


def _time_stream(fn, inputs, samples: int, warmup: int) -> tuple[list, list]:
    """Cycle frames, timing only ``fn`` — input preparation stays outside the clock."""
    for index in range(min(warmup, len(inputs))):
        fn(inputs[index])
    ms, steps = [], []
    for index in range(samples):
        payload = inputs[index % len(inputs)]
        started = time.perf_counter()
        tokens = np.asarray(fn(payload), dtype=np.uint32)
        ms.append((time.perf_counter() - started) * 1000.0)
        steps.append(int(tokens.size))
    return ms, steps


def _fit(points) -> dict:
    """Least squares over (steps, ms) -> the fixed/per-step split."""
    xs = np.asarray([p[0] for p in points], dtype=np.float64)
    ys = np.asarray([p[1] for p in points], dtype=np.float64)
    slope, intercept = np.polyfit(xs, ys, 1)
    predicted = slope * xs + intercept
    ss_res = float(((ys - predicted) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    return {
        "per_step_ms": float(slope),
        "fixed_ms": float(intercept),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot else 1.0,
        "points": [{"steps": int(s), "ms": float(m)} for s, m in points],
    }


def _run_ar(call, observations, survey: int, repeats: int) -> dict:
    """Survey frames, then time one representative per distinct step count."""
    print(f"surveying {min(survey, len(observations))} frames for distinct step counts ...")
    by_steps: dict[int, int] = {}
    table = []
    for index in range(min(survey, len(observations))):
        started = time.perf_counter()
        tokens = np.asarray(call(observations[index]), dtype=np.uint32)
        ms = (time.perf_counter() - started) * 1000.0
        steps = int(tokens.size)
        table.append({"frame": index, "steps": steps, "ms": ms})
        by_steps.setdefault(steps, index)
        print(f"  frame {index:3d}: steps={steps:3d}  {ms:8.1f} ms")

    points = []
    for steps in sorted(by_steps):
        index = by_steps[steps]
        reps = []
        for _ in range(repeats):
            started = time.perf_counter()
            call(observations[index])
            reps.append((time.perf_counter() - started) * 1000.0)
        median = statistics.median(reps)
        points.append((steps, median))
        print(
            f"  steps={steps:3d} frame={index:3d} -> median {median:8.1f} ms "
            f"of {[round(r, 1) for r in reps]}"
        )

    fit = _fit(points) if len(points) >= 2 else None
    if fit is None:
        print("only one distinct step count; the fixed/per-step split needs at least two")
    return {"survey": table, "fit": fit}


def main() -> None:
    args = parse_args()
    layers = _parse_layers(args.layer)
    image_keys = (
        tuple(item.strip() for item in args.image_keys.split(",") if item.strip())
        if args.image_keys
        else None
    )

    from apxinf import AutoPolicy

    options = {
        "precision": args.precision,
        "device": args.device,
        "state_key": args.state_key,
        "prompt_key": args.prompt_key,
        "action_dim": args.action_dim,
        "action_horizon": args.action_horizon,
    }
    if image_keys is not None:
        options["image_keys"] = image_keys
    policy = AutoPolicy.from_pretrained(
        args.model_dir, **{name: value for name, value in options.items() if value is not None}
    )
    handle = policy.model
    stop_token = int(policy.tokenizer.pipe_token_id)

    frames = (
        _load_frames(args.frames, args.frame_variant, args.frames_limit)
        if args.frames is not None
        else _synthetic_frames(policy, args.frames_count, args.seed)
    )
    observations = _observations(policy, frames, args.prompt)
    prepared = [_prepare(policy, observation) for observation in observations]

    def l1(payload):
        rgb, token_ids = payload
        return handle.infer_action_tokens_rgb(rgb, "nhwc", token_ids, stop_token=stop_token)

    def l2(observation):
        return policy.infer(observation)["action_tokens"]

    result = {
        "schema": "apxinf.pi0_fast.latency.v1",
        "git_commit": _git_commit(),
        "precision": args.precision,
        "device": args.device,
        "model_dir": str(args.model_dir),
        "layers": layers,
        "mode": args.mode,
        "frames": str(args.frames) if args.frames is not None else "synthetic",
        "frame_count": len(observations),
        "workload": {
            "state_dim": int(policy.metadata["state_dim"]),
            "action_dim": policy.action_dim,
            "action_horizon": policy.action_horizon,
            "max_action_tokens": policy.metadata.get("max_action_tokens"),
            "num_views": int(policy.metadata["num_views"]),
            "image_size": policy.metadata["image_size"],
            "image_keys": list(policy.image_keys),
            "state_key": policy.state_key,
            "stop_token": stop_token,
            "prompt": observations[0][policy.prompt_key],
        },
    }

    if args.mode in ("latency", "all"):
        layers_ms, raw_ms, tokens_seen = {}, {}, {}
        for layer, fn, stream in (
            ("l1", l1, prepared),
            ("l2", l2, observations),
        ):
            if layer not in layers:
                continue
            ms, steps = _time_stream(fn, stream, args.samples, args.warmup)
            layers_ms[f"L{layer[1]}_{'rust' if layer == 'l1' else 'python_api'}"] = _stats(ms)
            raw_ms[layer] = ms
            tokens_seen[layer] = {
                "min": int(min(steps)),
                "median": int(statistics.median(steps)),
                "max": int(max(steps)),
            }
        result["layers_ms"] = layers_ms
        result["raw_ms"] = raw_ms
        result["action_tokens"] = tokens_seen

        if args.full_decode and "l1" in layers:
            def full(payload):
                rgb, token_ids = payload
                return handle.infer_action_tokens_rgb(rgb, "nhwc", token_ids)

            ms, steps = _time_stream(full, prepared, min(args.samples, 5), 1)
            result["full_decode_ms"] = _stats(ms)
            result["full_decode_tokens"] = int(steps[0])

    if args.mode in ("ar", "all"):
        ar_layer = "l1" if "l1" in layers else "l2"
        call, stream = (l1, prepared) if ar_layer == "l1" else (l2, observations)
        result["ar"] = {"layer": ar_layer, **_run_ar(call, stream, args.survey, args.repeats)}

    print(
        f"\nin-process latency  |  {args.precision}  checkpoint  "
        f"tokens={result['workload']['max_action_tokens']} "
        f"D={result['workload']['action_dim']} H={result['workload']['action_horizon']} "
        f"views={result['workload']['num_views']}"
    )
    print("-" * 100)
    print(f"{'layer':<16}{'p50':>9}{'p95':>9}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
    for name, stats in result.get("layers_ms", {}).items():
        print(
            f"{name:<16}{stats['p50']:>9.2f}{stats['p95']:>9.2f}{stats['mean']:>9.2f}"
            f"{stats['std']:>9.2f}{stats['min']:>9.2f}{stats['max']:>9.2f}"
        )
    for layer, seen in result.get("action_tokens", {}).items():
        print(
            f"  {layer} action tokens: min={seen['min']} median={seen['median']} "
            f"max={seen['max']}"
        )
    if "full_decode_ms" in result:
        stats = result["full_decode_ms"]
        print(
            f"  L1 with no stop token: p50={stats['p50']:.1f} ms "
            f"({result['full_decode_tokens']} tokens)"
        )
    if "ar" in result and result["ar"]["fit"] is not None:
        fit = result["ar"]["fit"]
        print(
            f"\n  AR split: per_step={fit['per_step_ms']:.2f} ms  fixed={fit['fixed_ms']:.1f} ms  "
            f"r2={fit['r2']:.4f}"
        )

    policy.close()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
