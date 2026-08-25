#!/usr/bin/env python3
"""Thin CLI launcher for the OpenPI-compatible π0.5 websocket service.

All reusable logic lives in the library: the transport shell in
:mod:`apxinf.serving`, the policy in :mod:`apxinf` (``AutoPolicy`` /
``Pi05Policy``), and the per-embodiment wire contract in
:mod:`apxinf.robots.presets`. This file is only argument parsing + wiring — load
an **in-process** policy through the ``apxinf_py`` PyO3 binding and serve it.

The old subprocess + stdio hop (``ApxInfStdioEngine`` + ``pi05_libero_server``)
is gone; so are the script's private resize/tokenize/unnormalize copies.

**Embodiment:** ``--robot`` selects the wire keys and the robot pre/post steps,
the way openpi's ``serve_policy.py --policy.config <TrainConfig>`` does. It
defaults to ``franka_libero``; a checkpoint fine-tuned for another robot **must**
name it, because the wire keys, the state routing, and the action encoding all
differ and a mismatch degrades silently rather than failing. ``--image-keys`` /
``--state-key`` override individual fields for a client that already speaks a
fixed dialect.

**State:** each preset decides whether ``state`` is injected (discretized into
the prompt, normalized to [-1, 1] from ``norm_stats``) or dropped —
``--discrete-state`` / ``--no-discrete-state`` override it. ``franka_libero``
drops state to match the numerics of the prior serving link; a joint-space robot
needs it.

**Images are RGB.** Neither this server nor openpi converts colour: an
``H×W×3`` uint8 array is taken as RGB as-is. A client reading frames with
OpenCV must ``cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`` first. Resizing *is* done
here (aspect-preserving pad to the model's edge), so any resolution is fine.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

# Make ``import apxinf`` work from a source checkout without installation. The
# ``apxinf_py`` CUDA binding must still be installed separately (``maturin
# develop`` of crates/apxinf-py); the transport deps come from
# scripts/requirements-pi05-websocket.txt.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_APXINF_PKG = _REPO_ROOT / "python" / "apxinf"
if _APXINF_PKG.is_dir() and str(_APXINF_PKG) not in sys.path:
    sys.path.insert(0, str(_APXINF_PKG))

from apxinf import Pi05Policy  # noqa: E402
from apxinf.robots.presets import (  # noqa: E402
    ROBOT_PRESETS,
    available_robots,
    build_robot_policy,
    get_robot_preset,
)
from apxinf.serving import WebsocketPolicyServer  # noqa: E402

DEFAULT_ROBOT = "franka_libero"


def _split_keys(value: str) -> tuple:
    keys = tuple(part.strip() for part in value.split(",") if part.strip())
    if not keys:
        raise argparse.ArgumentTypeError("--image-keys needs at least one camera key")
    return keys


def parse_args() -> argparse.Namespace:
    robot_help = "\n".join(f"  {p.describe()}" for p in ROBOT_PRESETS.values())
    parser = argparse.ArgumentParser(
        description="Serve a ApxInf PI0.5 policy through OpenPI's websocket API "
        "(in-process; no subprocess)",
        epilog=f"robot presets (--robot):\n{robot_help}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", type=pathlib.Path, help="checkpoint directory")
    parser.add_argument(
        "--robot",
        choices=available_robots(include_aliases=True),
        default=DEFAULT_ROBOT,
        help="embodiment preset: wire keys + robot pre/post steps + action width "
        f"(default: {DEFAULT_ROBOT}). openpi's --policy.config equivalent; a "
        "checkpoint fine-tuned for another robot must name it. Presets are named "
        "<arm>_<key convention>, since the arm alone does not fix the contract.",
    )
    parser.add_argument(
        "--image-keys",
        type=_split_keys,
        default=None,
        help="comma-separated camera wire keys, overriding the preset. Order is "
        "significant: key i fills model view slot i (base, left wrist, right "
        "wrist). Nested client layouts are written as a path, e.g. "
        "'images/cam_high,images/cam_left_wrist'.",
    )
    parser.add_argument(
        "--state-key",
        default=None,
        help="observation key holding the state vector, overriding the preset",
    )
    parser.add_argument(
        "--random-weights",
        action="store_true",
        help="serve a checkpoint-free engine with deterministic random weights and "
        "synthetic processors (latency-only; actions are numerically meaningless). "
        "No --model-dir needed.",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        help="checkpoint or index (default: MODEL_DIR/model.safetensors)",
    )
    parser.add_argument(
        "--model-type",
        help="policy model_type; default reads MODEL_DIR/config.json (e.g. pi05)",
    )
    parser.add_argument(
        "--tokenizer",
        type=pathlib.Path,
        help="SentencePiece model (auto-detected under MODEL_DIR by default)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16"
    )
    parser.add_argument(
        "--calibration",
        type=pathlib.Path,
        help="FP8 activation calibration JSON; required only for --precision fp8",
    )
    parser.add_argument(
        "--tactics",
        type=pathlib.Path,
        help="FP8 GEMM tactics JSON; required only for --precision fp8",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="deployable action width to trim to, overriding the preset "
        "(LIBERO=7; 0 keeps the full vector)",
    )
    parser.add_argument("--norm-key", default="actions")
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="chunk length to serve. Default: the checkpoint's config.json value, "
        "or 50 with --random-weights. An explicit value outranks the checkpoint "
        "(the horizon is a sequence length, not a weight dimension).",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=None,
        help="serve fewer cameras than the checkpoint declares (must equal the "
        "number of image keys). Drops the trailing view slots at load time — "
        "equivalent to openpi zero-padding and masking them, minus their patch "
        "tokens. Required to be explicit: a short image_keys list on its own is "
        "an error, so a forgotten camera fails instead of degrading. Also sets "
        "the synthetic view count under --random-weights.",
    )
    # Synthetic-shape knobs, used only with --random-weights (a checkpoint runs its
    # native config). They mirror apxinf_py.Model.random.
    parser.add_argument("--image-size", type=int, default=224, help="random: image edge")
    parser.add_argument("--num-flow-steps", type=int, default=10, help="random: flow steps")
    parser.add_argument("--max-token-len", type=int, default=200, help="random: max prompt tokens")
    parser.add_argument("--token-count", type=int, default=10, help="random: synthetic prompt length")
    parser.add_argument(
        "--discrete-state",
        dest="discrete_state",
        action="store_true",
        default=None,
        help="inject discretized state into the prompt (state normalized to "
        "[-1, 1] from norm_stats), overriding the preset. Without it state is "
        "silently dropped — a joint-space robot needs this on.",
    )
    parser.add_argument(
        "--no-discrete-state",
        dest="discrete_state",
        action="store_false",
        help="drop state even if the preset injects it",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not args.random_weights and args.model_dir is None:
        raise ValueError("pass --model-dir, or --random-weights for a checkpoint-free engine")
    if args.random_weights and args.model_dir is not None:
        raise ValueError("--random-weights is checkpoint-free; do not also pass --model-dir")

    preset = get_robot_preset(args.robot)
    image_keys = args.image_keys if args.image_keys is not None else preset.image_keys
    state_key = args.state_key if args.state_key is not None else preset.state_key
    discrete_state = preset.discrete_state if args.discrete_state is None else args.discrete_state
    if args.num_views is not None and args.num_views != len(image_keys):
        raise ValueError(
            f"--num-views {args.num_views} disagrees with the {len(image_keys)} "
            f"camera keys being served ({list(image_keys)}); they name the same "
            "cameras, so they must match"
        )

    metadata = {
        "protocol": "openpi.websocket_policy",
        "precision": args.precision,
        "policy": preset.name,
    }

    if args.random_weights:
        import apxinf_py  # lazy: only the synthetic path needs the CUDA binding here

        # Synthetic FP8 has no calibration file; a uniform activation scale keeps the
        # FP8 path on. bf16/int8 need neither calibration nor tactics.
        calibration = None
        if args.precision == "fp8":
            calibration = str(args.calibration) if args.calibration is not None else "uniform:1.0"
        action_horizon = args.action_horizon if args.action_horizon is not None else 50
        # The preset's cameras define the synthetic view count unless --num-views is
        # given explicitly, so --robot alone yields a servable synthetic engine.
        num_views = args.num_views if args.num_views is not None else len(image_keys)
        logging.info(
            "serving checkpoint-free %s random-weights engine (views=%d, H=%d, T=%d) "
            "— actions are latency-only",
            args.precision,
            num_views,
            action_horizon,
            args.token_count,
        )
        handle = apxinf_py.Model.random(
            model=(args.model_type or "pi05"),
            device=args.device,
            precision=args.precision,
            num_views=num_views,
            image_size=args.image_size,
            action_horizon=action_horizon,
            action_dim=(args.action_dim or 32),
            num_flow_steps=args.num_flow_steps,
            max_token_len=args.max_token_len,
            calibration=calibration,
            tactics=(str(args.tactics) if args.tactics is not None else None),
            seed=args.seed,
        )
        policy = Pi05Policy.from_random(
            handle,
            token_count=args.token_count,
            action_dim=(args.action_dim or None),
            seed=args.seed,
            image_keys=image_keys[:num_views],
            state_key=state_key,
            metadata={**metadata, "robot": preset.name},
        )
    else:
        if args.precision == "fp8" and (args.calibration is None or args.tactics is None):
            raise ValueError("--calibration and --tactics are required for FP8")

        logging.info(
            "loading %s policy in-process from %s as robot=%s",
            args.precision,
            args.model_dir,
            preset.describe(),
        )
        policy = build_robot_policy(
            preset.name,
            args.model_dir,
            image_keys=image_keys,
            state_key=state_key,
            discrete_state=discrete_state,
            action_dim=args.action_dim,
            num_views=args.num_views,
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            device=args.device,
            precision=args.precision,
            calibration=args.calibration,
            tactics=args.tactics,
            tokenizer_path=args.tokenizer,
            norm_key=args.norm_key,
            action_horizon=args.action_horizon,
            seed=args.seed,
            metadata=metadata,
        )
    # Clients read the served wire contract off this metadata rather than assuming
    # one: a key mismatch is silent on the wire but visible here.
    logging.info(
        "serving robot=%s H=%d x D=%d image_keys=%s state=%s discrete_state=%s",
        policy.metadata.get("robot", preset.name),
        policy.metadata["action_horizon"],
        policy.metadata["action_dim"],
        policy.metadata["image_keys"],
        policy.metadata["state_key"],
        policy.metadata["discrete_state"],
    )
    server = WebsocketPolicyServer(policy, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("shutting down")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
