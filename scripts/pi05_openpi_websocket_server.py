#!/usr/bin/env python3
"""Full-surface CLI launcher for the OpenPI-compatible π0.5 websocket service.

All reusable logic lives in the library: the transport shell in
:mod:`apxinf.serving` and the policy in :mod:`apxinf` (``AutoPolicy`` /
``Pi05Policy``). This file is only argument parsing + wiring — load an
**in-process** policy through the ``apxinf_py`` PyO3 binding and serve it.

``examples/openpi_server.py`` is the same thing in twenty lines, and is the right
place to start reading. This launcher exists for the engine-domain knobs that
example deliberately omits: ``--random-weights`` (serve with no checkpoint at
all, to time the engine), ``--calibration`` and ``--autotune`` (FP8 activation
scales and GEMM tactic tuning), ``--norm-stats`` / ``--norm-key`` / ``--asset-id``
/ ``--ckpt-format`` (say which statistics and which layout to read when a
directory is ambiguous), ``--num-views``, and ``--discrete-state``. Every one of
those is a property of the *checkpoint or the engine*, not of a robot, which is
why they live here rather than downstream.

**Wire keys are yours to name.** ``--image-keys`` / ``--state-key`` say what your
client sends; omit them and the policy falls back to its own view-slot vocabulary
(``base_0_rgb``, ``left_wrist_0_rgb``, ...), which is a neutral fallback rather
than a contract. If you want *named* robot contracts — a preset that fixes the
keys, the action width, and the pre/post steps together and cross-checks them
against the checkpoint — that is ``apxinf-robo``, which builds on this.

**State** is dropped unless ``--discrete-state`` is passed, which discretizes it
into the prompt (normalized to [-1, 1] from ``norm_stats``) and then makes
``--state-key`` mandatory: there is no dataset-neutral name to guess, and
guessing one loses proprioception in silence.

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

from apxinf import AutoPolicy, Pi05Policy  # noqa: E402
from apxinf.checkpoints import FORMATS as CHECKPOINT_FORMATS  # noqa: E402
from apxinf.checkpoints.preflight import (  # noqa: E402
    FAIL,
    WARN,
    format_findings,
    inspect_checkpoint,
    sort_findings,
)
from apxinf.serving import WebsocketPolicyServer  # noqa: E402
from apxinf._tactics import resolve_pi05_tactics  # noqa: E402


def _split_keys(value: str) -> tuple:
    keys = tuple(part.strip() for part in value.split(",") if part.strip())
    if not keys:
        raise argparse.ArgumentTypeError("--image-keys needs at least one camera key")
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a ApxInf PI0.5 policy through OpenPI's websocket API "
        "(in-process; no subprocess)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", type=pathlib.Path, help="checkpoint directory")
    parser.add_argument(
        "--image-keys",
        type=_split_keys,
        default=None,
        help="comma-separated camera wire keys. Order is significant: key i fills "
        "model view slot i (base, left wrist, right wrist). Nested client layouts "
        "are written as a path, e.g. 'images/cam_high,images/cam_left_wrist'. "
        "Omitted, the policy names them after its own view slots.",
    )
    parser.add_argument(
        "--state-key",
        default=None,
        help="observation key holding the state vector; required with --discrete-state",
    )
    parser.add_argument(
        "--random-weights",
        action="store_true",
        help="serve a checkpoint-free engine with deterministic random weights and "
        "synthetic processors (latency-only; actions are numerically meaningless). "
        "The wire keys and view count are real; the tokenizer emits a fixed token "
        "stream and never reads state, and the unnormalizer is the identity map. "
        "No --model-dir needed.",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        help="checkpoint or index (default: MODEL_DIR/model.safetensors)",
    )
    parser.add_argument(
        "--model-type",
        help="policy model_type; default reads MODEL_DIR/config.json, or metadata.pt "
        "for an openpi export, which has no config.json (e.g. pi05)",
    )
    parser.add_argument(
        "--tokenizer",
        type=pathlib.Path,
        help="SentencePiece model (auto-detected under MODEL_DIR, or APXINF_TOKENIZER)",
    )
    parser.add_argument(
        "--ckpt-format",
        choices=CHECKPOINT_FORMATS,
        default="auto",
        help="how to read MODEL_DIR. 'auto' (default) picks openpi_pytorch when a "
        "metadata.pt is present and lerobot when a config.json is. Pin it when a "
        "directory carries both and the wrong one wins.",
    )
    parser.add_argument(
        "--asset-id",
        default=None,
        help="override the asset_id the checkpoint names, which is what selects "
        "assets/<asset_id>/norm_stats.json. For assets reorganized after export.",
    )
    parser.add_argument(
        "--norm-stats",
        type=pathlib.Path,
        default=None,
        help="explicit OpenPI-style norm_stats.json, outranking every path convention. "
        "LeRobot processor state is discovered from policy_preprocessor.json and "
        "policy_postprocessor.json instead; a base checkpoint with no declared "
        "processor state uses LeRobot-compatible identity transforms.",
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
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="tune missing exact GEMM tactics from real requests and persist them",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="deployable action width to trim the checkpoint's action transform to "
        "(LIBERO=7; 0 or omitted keeps the full vector)",
    )
    parser.add_argument("--norm-key", default="actions")
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="chunk length to serve. Default: the checkpoint's own value (config.json, "
        "or metadata.pt for an openpi export), or 50 with --random-weights. An "
        "explicit value outranks the checkpoint "
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
        "an error, so a forgotten camera fails instead of degrading. Under "
        "--random-weights it also sets the synthetic view count, and is required "
        "there when --image-keys is not given.",
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
        "[-1, 1] from norm_stats). Without it state is dropped, so a joint-space "
        "robot needs this on — and with it --state-key becomes mandatory.",
    )
    parser.add_argument(
        "--no-discrete-state",
        dest="discrete_state",
        action="store_false",
        help="drop state explicitly (the default, stated for symmetry)",
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

    image_keys = args.image_keys
    state_key = args.state_key
    discrete_state = bool(args.discrete_state)
    if args.num_views is not None and image_keys is not None:
        if args.num_views != len(image_keys):
            raise ValueError(
                f"--num-views {args.num_views} disagrees with the {len(image_keys)} "
                f"camera keys being served ({list(image_keys)}); they name the same "
                "cameras, so they must match"
            )
    if discrete_state and state_key is None:
        # Caught here rather than inside the policy so the message names the flag
        # the operator actually typed. State that is read but never addressed is
        # not an error anywhere downstream — it is simply absent from the prompt.
        raise ValueError("--discrete-state reads state; name its wire key with --state-key")

    metadata = {
        "protocol": "openpi.websocket_policy",
        "precision": args.precision,
        "policy": "pi05",
        "autotune": args.autotune,
    }
    if args.random_weights:
        import apxinf_py  # lazy: only the synthetic path needs the CUDA binding here

        # Random engines bypass Pi05Policy.from_pretrained, so this synthetic
        # server is the sole caller that must resolve the package default.
        tactics = resolve_pi05_tactics(
            args.device,
            args.precision,
            override=args.tactics,
            allow_missing=args.autotune,
        )
        if tactics is not None:
            logging.info("using %s tactics for %s: %s", args.precision, args.device, tactics)
        # Synthetic FP8 has no calibration file; a uniform activation scale keeps the
        # FP8 path on. bf16/int8 need neither calibration nor tactics.
        calibration = None
        if args.precision == "fp8":
            calibration = str(args.calibration) if args.calibration is not None else "uniform:1.0"
        action_horizon = args.action_horizon if args.action_horizon is not None else 50
        # There is no checkpoint to read a camera count off, so it has to be said:
        # either name the keys and the count follows, or state the count and the
        # policy names the slots itself. Defaulting would silently time the wrong
        # number of vision towers, which is the one number this mode exists to time.
        if args.num_views is not None:
            num_views = args.num_views
        elif image_keys is not None:
            num_views = len(image_keys)
        else:
            raise ValueError(
                "--random-weights has no checkpoint to read a view count from; "
                "pass --num-views, or --image-keys and it follows"
            )
        # --action-dim sets both the synthetic model's own width and the width it
        # trims to; without it the model is 32-wide and nothing is trimmed.
        model_dim = args.action_dim or 32
        trim_dim = args.action_dim
        logging.warning(
            "--random-weights serves real wire keys and a real view count and "
            "nothing else: the tokenizer emits a fixed token stream and never "
            "reads state, and the unnormalizer is the identity map. The actions "
            "are latency-only. Use a checkpoint to preview the full contract."
        )
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
            action_dim=model_dim,
            num_flow_steps=args.num_flow_steps,
            max_token_len=args.max_token_len,
            calibration=calibration,
            tactics=(str(tactics) if tactics is not None else None),
            autotune=args.autotune,
            seed=args.seed,
        )
        policy = Pi05Policy.from_random(
            handle,
            token_count=args.token_count,
            action_dim=(trim_dim or None),
            seed=args.seed,
            image_keys=(image_keys[:num_views] if image_keys is not None else None),
            state_key=state_key,
            metadata=metadata,
        )
    else:
        # Read what the checkpoint says about itself before any weight is loaded:
        # an unreadable layout or missing statistics should cost a second, not a
        # multi-gigabyte load followed by a failure deep in the pipeline.
        report = inspect_checkpoint(
            args.model_dir,
            norm_key=args.norm_key,
            state_norm_key=("state" if discrete_state else None),
            tokenizer_path=args.tokenizer,
            checkpoint_format=args.ckpt_format,
            asset_id=args.asset_id,
            norm_stats=args.norm_stats,
        )
        findings = sort_findings(report.findings)
        if any(finding.level == FAIL for finding in findings):
            raise SystemExit(
                f"preflight: {args.model_dir} cannot be served as configured\n"
                + format_findings(findings, include_info=False)
            )
        for finding in findings:
            level = logging.ERROR if finding.level == FAIL else (
                logging.WARNING if finding.level == WARN else logging.INFO
            )
            logging.log(level, "preflight %s", finding)

        logging.info("loading %s policy in-process from %s", args.precision, args.model_dir)
        options = {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "device": args.device,
            "precision": args.precision,
            "calibration": args.calibration,
            "tactics": args.tactics,
            "autotune": args.autotune,
            "tokenizer_path": args.tokenizer,
            "checkpoint_format": args.ckpt_format,
            "asset_id": args.asset_id,
            "norm_stats": args.norm_stats,
            "norm_key": args.norm_key,
            "action_horizon": args.action_horizon,
            "num_views": args.num_views,
            "action_dim": (args.action_dim or None),
            "discrete_state": discrete_state,
            "seed": args.seed,
            "metadata": metadata,
        }
        # Omitted keys keep the policy's own fallback (view-slot image keys, no
        # state key), which is not the same as passing None: passing None would
        # be this launcher asserting a dialect it was never told.
        if image_keys is not None:
            options["image_keys"] = image_keys
        if state_key is not None:
            options["state_key"] = state_key
        policy = AutoPolicy.from_pretrained(args.model_dir, **options)
    # Clients read the served wire contract off this metadata rather than assuming
    # one: a key mismatch is silent on the wire but visible here. A null state_key
    # is not a gap — it says this policy drops state, so there is no key to send
    # one under; rendered as "(dropped)" so the log line cannot read as an omission.
    served_state_key = policy.metadata["state_key"]
    logging.info(
        "serving H=%d x D=%d image_keys=%s state=%s discrete_state=%s",
        policy.metadata["action_horizon"],
        policy.metadata["action_dim"],
        policy.metadata["image_keys"],
        served_state_key if served_state_key is not None else "(dropped)",
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
