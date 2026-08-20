#!/usr/bin/env python3
"""Thin CLI launcher for the OpenPI-compatible π0.5 websocket service.

All reusable logic lives in the library: the transport shell in
:mod:`apxinf.serving` and the policy in :mod:`apxinf` (``AutoPolicy`` /
``Pi05Policy``). This file is only argument parsing + wiring — load an
**in-process** policy through the ``apxinf_py`` PyO3 binding and serve it.

The old subprocess + stdio hop (``ApxInfStdioEngine`` + ``pi05_libero_server``)
is gone; so are the script's private resize/tokenize/unnormalize copies.

**State gap:** ``observation/state`` is dropped by default so numerics match the
prior serving link. ``--discrete-state`` opts in and wires ``apxinf``'s
``state_normalizer`` (normalize raw state to [-1, 1] from ``norm_stats``) +
prompt discretization — see :mod:`apxinf.policies.impls.pi05`.
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

from apxinf import AutoPolicy, Pi05Policy  # noqa: E402
from apxinf.serving import WebsocketPolicyServer  # noqa: E402

DEFAULT_LIBERO_ACTION_DIM = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a ApxInf PI0.5 policy through OpenPI's websocket API "
        "(in-process; no subprocess)"
    )
    parser.add_argument("--model-dir", type=pathlib.Path, help="checkpoint directory")
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
        default=DEFAULT_LIBERO_ACTION_DIM,
        help="deployable action width to trim to (LIBERO=7; 0 keeps full vector)",
    )
    parser.add_argument("--norm-key", default="actions")
    # Synthetic-shape knobs, used only with --random-weights (a checkpoint runs its
    # native config). They mirror apxinf_py.Model.random.
    parser.add_argument("--num-views", type=int, default=2, help="random: camera views")
    parser.add_argument("--image-size", type=int, default=224, help="random: image edge")
    parser.add_argument("--action-horizon", type=int, default=50, help="random: action horizon")
    parser.add_argument("--num-flow-steps", type=int, default=10, help="random: flow steps")
    parser.add_argument("--max-token-len", type=int, default=200, help="random: max prompt tokens")
    parser.add_argument("--token-count", type=int, default=10, help="random: synthetic prompt length")
    parser.add_argument(
        "--discrete-state",
        action="store_true",
        help="inject discretized state into the prompt (state normalized to "
        "[-1, 1] from norm_stats). OFF by default to match current numerics.",
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

    metadata = {
        "protocol": "openpi.websocket_policy",
        "precision": args.precision,
        "policy": "libero",
    }

    if args.random_weights:
        import apxinf_py  # lazy: only the synthetic path needs the CUDA binding here

        # Synthetic FP8 has no calibration file; a uniform activation scale keeps the
        # FP8 path on. bf16/int8 need neither calibration nor tactics.
        calibration = None
        if args.precision == "fp8":
            calibration = str(args.calibration) if args.calibration is not None else "uniform:1.0"
        logging.info(
            "serving checkpoint-free %s random-weights engine (views=%d, H=%d, T=%d) "
            "— actions are latency-only",
            args.precision,
            args.num_views,
            args.action_horizon,
            args.token_count,
        )
        handle = apxinf_py.Model.random(
            model=(args.model_type or "pi05"),
            device=args.device,
            precision=args.precision,
            num_views=args.num_views,
            image_size=args.image_size,
            action_horizon=args.action_horizon,
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
            metadata=metadata,
        )
    else:
        if args.precision == "fp8" and (args.calibration is None or args.tactics is None):
            raise ValueError("--calibration and --tactics are required for FP8")

        logging.info("loading %s policy in-process from %s", args.precision, args.model_dir)
        policy = AutoPolicy.from_pretrained(
            args.model_dir,
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            device=args.device,
            precision=args.precision,
            calibration=args.calibration,
            tactics=args.tactics,
            tokenizer_path=args.tokenizer,
            norm_key=args.norm_key,
            action_dim=(args.action_dim or None),
            seed=args.seed,
            discrete_state=args.discrete_state,
            metadata=metadata,
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
