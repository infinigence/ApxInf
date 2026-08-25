#!/usr/bin/env python3
"""Functional smoke test for the Unitree G1 pi05 adapter.

Runs the integrator's ``pi05_UnitreeG1`` serving config through our interface: a
G1-style observation (3 cameras ``cam_high`` / ``cam_left_wrist`` /
``cam_right_wrist``, 16-DoF state, a prompt) is fed to a
:func:`~apxinf.robots.build_unitree_g1_policy` policy and must come back
as a ``[action_horizon, 16]`` action chunk — exercising every adapter step
(decode-state → tokenize → model → unnormalize → delta→absolute → 32→16 encode).

Since we hold no G1 checkpoint/norm_stats, this validates **plumbing and shape**
on a stand-in pi05 checkpoint: the action unnormalizer is a full-model-width
identity, so values are not G1-calibrated (that needs the integrator's gripper
limits + the G1 weights). It proves the config runs end-to-end through the
interface, which is the compatibility question.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_APXINF_PKG = _REPO_ROOT / "python" / "apxinf"
if _APXINF_PKG.is_dir() and str(_APXINF_PKG) not in sys.path:
    sys.path.insert(0, str(_APXINF_PKG))

from apxinf import build_unitree_g1_policy  # noqa: E402
from apxinf.processors import Unnormalizer  # noqa: E402
from apxinf.processors.transforms import Unnormalize  # noqa: E402
from apxinf.robots.unitree_g1 import G1_CAMERAS, G1_STATE_KEY  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True, type=pathlib.Path)
    p.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prompt", default="pick up the cup")
    p.add_argument("--chw", action="store_true", help="feed CHW images (ParseImage must transpose)")
    return p.parse_args()


def main():
    args = parse_args()

    # Full-model-width identity unnormalizer: [50,32] passes through unchanged so
    # the 32->16 robot truncation and delta->absolute run on the raw model output.
    identity = None  # built after we know the model width

    # Build once to learn the model width, then rebuild with a matching identity.
    # (Cheap: model load dominates; we load a single time by peeking metadata.)
    import apxinf_py

    model = apxinf_py.Model.load(
        "pi05", str(args.model_dir / "model.safetensors"), device=args.device, precision=args.precision
    )
    width = model.action_dim
    identity = Unnormalize(
        Unnormalizer(mean=np.zeros(width, np.float32), std=np.ones(width, np.float32), mode="mean_std")
    )

    policy = build_unitree_g1_policy(
        args.model_dir,
        model=model,  # reuse the already-loaded handle
        use_delta_joint_actions=True,
        adapt_to_pi=True,
        state_key=G1_STATE_KEY,
        unnormalizer=identity,
        precision=args.precision,
        device=args.device,
        metadata={"config": "pi05_UnitreeG1_groundwire", "note": "stand-in ckpt, shape-only"},
    )

    print("input pipeline :", policy.input_pipeline.names)
    print("output pipeline:", policy.output_pipeline.names)
    print("metadata       :", {k: policy.metadata[k] for k in ("robot", "action_dim", "model_action_dim", "num_views", "action_horizon")})

    S = model.image_size
    rng = np.random.default_rng(0)
    if args.chw:
        cam = lambda: rng.integers(0, 256, (3, S, S), dtype=np.uint8)  # noqa: E731
    else:
        cam = lambda: rng.integers(0, 256, (S, S, 3), dtype=np.uint8)  # noqa: E731

    # The exact layout an unmodified openpi G1 client sends: cameras nested one
    # level under "images", state flat. The policy's image_keys are the paths
    # ("images/cam_high"), which lookup_key walks into this dict.
    groups: dict = {}
    for key in G1_CAMERAS:
        group, _, camera = key.partition("/")
        assert camera, f"expected a nested camera path, got {key!r}"
        groups.setdefault(group, {})[camera] = cam()
    observation = dict(groups)
    observation[G1_STATE_KEY] = rng.standard_normal(16).astype(np.float32)
    observation["prompt"] = args.prompt

    out = policy.infer(observation)
    actions = np.asarray(out["actions"])
    H = model.action_horizon

    assert actions.shape == (H, 16), f"expected ({H}, 16), got {actions.shape}"
    assert np.isfinite(actions).all(), "non-finite actions"

    print(f"\nOK  actions {actions.shape}  dtype={actions.dtype}")
    print(f"    timing: model={out['timing']['model_ms']:.1f}ms  total={out['timing']['total_ms']:.1f}ms")
    print(f"    action[0] = {np.array2string(actions[0], precision=3, max_line_width=120)}")
    print(f"    gripper dims [7,15] over horizon in [0,1]? "
          f"{bool((actions[:, [7, 15]] >= 0).all() and (actions[:, [7, 15]] <= 1).all())}")
    print("\nG1 ADAPTER SMOKE TEST PASSED (shape/plumbing)")


if __name__ == "__main__":
    main()
