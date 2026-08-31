#!/usr/bin/env python3
"""Read an openpi checkpoint's own ``metadata.pt`` and say what apxinf must run.

openpi keeps the serving contract in a Python registry: ``serve_policy.py
--policy.config <TrainConfig name>`` selects a ``TrainConfig``, and that
object decides the wire keys, the delta convention, the state handling and the
action width. When a checkpoint is exported, openpi serializes that
``TrainConfig`` into ``metadata.pt`` — so the checkpoint carries its own answer to
"how must this be served", and nothing reads it.

apxinf's equivalent is a ``--robot`` preset, which is a *hand* transcription of
the same facts. This script closes the loop: it reads ``metadata.pt``, prints the
apxinf launch command it implies, and diffs every field against the preset the
operator named. Then it runs the checkpoint-directory checks from
:mod:`apxinf.robots.preflight` (norm_stats widths, tokenizer) on top.

Why this exists: the delivered configuration was a Unitree G1 checkpoint served
with LIBERO norm_stats and ``--action-dim 7``. Nothing errored. The weights
loaded, the tokenizer ran, the unnormalizer found quantiles of *some* width, and
the robot got seven plausible floats per step that meant nothing. Every one of
those facts contradicts ``metadata.pt``, which says 16-dim G1, three cameras,
delta joint actions, discretized state. This script prints that contradiction in
one screen.

It does *not* generate a config file. apxinf's counterpart to ``TrainConfig`` is
``robots/presets.py`` plus a builder — code, reviewed once, not generated per
checkpoint. The value here is the assertion, not the transcription.

Usage::

    python scripts/openpi_metadata_to_apxinf.py --model-dir CKPT_DIR --robot unitree_g1

Exit status is 1 when any check is fatal, so it can gate a deployment.
``metadata.pt`` is optional: without it the checkpoint-directory checks still run
and the openpi cross-check is reported as skipped. Nothing here needs torch, CUDA
or a loaded weight -- it is a seconds-long, laptop-runnable check.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_APXINF_PKG = _REPO_ROOT / "python" / "apxinf"
if _APXINF_PKG.is_dir() and str(_APXINF_PKG) not in sys.path:
    sys.path.insert(0, str(_APXINF_PKG))

from apxinf.checkpoints import (  # noqa: E402
    FORMATS as CHECKPOINT_FORMATS,
    MetadataError,
    read_metadata_pt,
    repack_structure,
)
from apxinf.robots.preflight import (  # noqa: E402
    FAIL,
    INFO,
    WARN,
    Finding,
    check_checkpoint,
    format_findings,
)
from apxinf.robots.presets import (  # noqa: E402
    ROBOT_PRESETS,
    available_robots,
    get_robot_preset,
)


def load_train_config(model_dir: pathlib.Path):
    """Return openpi's serialized ``TrainConfig`` dict, or ``(None, reason)``.

    Read through :func:`apxinf.checkpoints.read_metadata_pt`, which parses the
    torch archive with the standard library alone. That matters here: this script
    is meant to run on a laptop, before anything heavy is installed, and
    ``metadata.pt`` is 30 kB of nested dicts — needing torch to see them made the
    check skippable exactly when it was most useful. A missing or unparseable
    file is still reported rather than raised, because the directory checks are
    worth running either way.
    """
    path = model_dir / "metadata.pt"
    if not path.exists():
        return None, f"{path} not present (apxinf does not require it)"
    try:
        payload = read_metadata_pt(path)
    except MetadataError as exc:
        return None, str(exc)
    config = payload.get("config")
    if not isinstance(config, dict):
        return None, f"metadata.pt has no 'config' dict (keys: {sorted(payload)})"
    return config, ""


def compare_to_preset(config: dict, robot: str) -> list:
    """Diff openpi's own TrainConfig against the named apxinf preset."""
    preset = get_robot_preset(robot)
    model = config.get("model") or {}
    data = config.get("data") or {}
    structure = repack_structure(data)
    out = []

    exp = config.get("exp_name") or config.get("name")
    if exp:
        out.append(Finding(INFO, "metadata.pt", f"exp_name={exp!r}, serving as --robot {robot}"))

    # --- cameras: count and, where openpi names them, the keys themselves.
    images = structure.get("images")
    if isinstance(images, dict):
        openpi_keys = tuple(images)
        preset_keys = tuple(key.split("/")[-1] for key in preset.image_keys)
        if openpi_keys == preset_keys:
            out.append(
                Finding(INFO, "cameras", f"{len(openpi_keys)} views {list(openpi_keys)}, match")
            )
        else:
            out.append(
                Finding(
                    FAIL,
                    "cameras",
                    f"openpi sends {list(openpi_keys)}, preset {robot!r} expects "
                    f"{list(preset_keys)}",
                    "camera keys are order-significant -- entry i fills model view "
                    "slot i. A mismatch either drops a camera or feeds the wrong one "
                    "to a slot, and neither raises. Fix the preset's slots.",
                )
            )
    else:
        out.append(Finding(WARN, "cameras", "metadata.pt has no repack image structure"))

    # --- state: is it sent at all, and does apxinf discretize it the same way?
    discrete = model.get("discrete_state_input")
    if discrete is not None:
        if bool(discrete) == preset.discrete_state:
            out.append(Finding(INFO, "discrete_state", f"{bool(discrete)}, match"))
        else:
            out.append(
                Finding(
                    FAIL,
                    "discrete_state",
                    f"checkpoint trained with discrete_state_input={bool(discrete)}, "
                    f"preset {robot!r} serves {preset.discrete_state}",
                    "with it off the model receives no proprioception at all (state is "
                    "dropped, not merely left continuous) and any delta->absolute "
                    "output step silently becomes a no-op",
                )
            )

    # --- widths. action_dim here is the *model's* padded width (32); the robot's
    #     physical width is not in metadata.pt -- it is in the norm_stats, which
    #     the directory check covers.
    for field, ours, label in (
        ("action_dim", 32, "model action width"),
        ("action_horizon", None, "action horizon"),
        ("max_token_len", 200, "max token len"),
    ):
        value = model.get(field)
        if value is None:
            continue
        if ours is not None and value != ours:
            out.append(
                Finding(
                    WARN,
                    label,
                    f"metadata.pt says {value}, apxinf's pi05 path assumes {ours}",
                    "apxinf reads the real value from the checkpoint at load time "
                    "(config.json, or metadata.pt for an openpi export); this only "
                    "flags that the assumption baked into the preset is stale",
                )
            )
        else:
            out.append(Finding(INFO, label, str(value)))

    # --- the two robot conventions the G1 output chain implements.
    for field, label, why in (
        (
            "adapt_to_pi",
            "adapt_to_pi",
            "the joint-flip / gripper decode+encode steps; off means raw robot space",
        ),
        (
            "use_delta_joint_actions",
            "use_delta_joint_actions",
            "whether the model emits joint deltas that must have the current state "
            "added back. Getting this wrong is not subtle in the logs and is "
            "catastrophic on the robot: absolute targets treated as deltas double "
            "every joint command.",
        ),
    ):
        value = data.get(field)
        if value is None:
            continue
        wanted = preset.builder_kwargs.get(field)
        if wanted is None:
            out.append(Finding(INFO, label, f"metadata.pt={value}; preset does not set it"))
        elif bool(value) == bool(wanted):
            out.append(Finding(INFO, label, f"{bool(value)}, match"))
        else:
            out.append(
                Finding(
                    FAIL,
                    label,
                    f"checkpoint trained with {bool(value)}, preset serves {bool(wanted)}",
                    why,
                )
            )

    prompt = data.get("default_prompt")
    if prompt is not None:
        out.append(Finding(INFO, "default_prompt", repr(prompt)))

    # --- where the *real* norm_stats live, which is the whole question.
    assets = data.get("assets") or {}
    asset_id, assets_dir = assets.get("asset_id"), assets.get("assets_dir")
    if asset_id or assets_dir:
        out.append(
            Finding(
                INFO,
                "norm_stats source",
                f"openpi computed them under {assets_dir}{asset_id}/norm_stats.json",
                "",
            )
        )

    # openpi derives the normalization mode from the model type, not from a file:
    # `use_quantile_norm = model_config.model_type != ModelType.PI0`. pi05 is
    # therefore always quantile, whatever a LeRobot-style config.json says.
    if model.get("pi05"):
        out.append(
            Finding(INFO, "normalization", "pi05 -> quantile (q01/q99), per openpi's model-type rule")
        )
    return out


def describe_launch(config: dict, robot: str, model_dir: pathlib.Path) -> str:
    """The apxinf launch line this checkpoint's own metadata implies."""
    preset = get_robot_preset(robot)
    data = config.get("data") or {}
    model = config.get("model") or {}
    flags = [
        "python scripts/pi05_openpi_websocket_server.py",
        f"  --model-dir {model_dir}",
        f"  --robot {preset.name}",
        # bf16 explicitly: --precision auto resolves to int8 on SM87 (Jetson
        # Orin), which is a different numeric contract than openpi's.
        "  --precision bf16",
    ]
    if model.get("discrete_state_input") and not preset.discrete_state:
        flags.append("  --discrete-state")
    note = (
        f"# metadata.pt: action_horizon={model.get('action_horizon')} "
        f"adapt_to_pi={data.get('adapt_to_pi')} "
        f"delta_joints={data.get('use_delta_joint_actions')} "
        f"discrete_state={model.get('discrete_state_input')}"
    )
    return " \\\n".join(flags) + "\n\n" + note


def _check_lerobot_config(model_dir: pathlib.Path, robot: str, *, authoritative: bool) -> list:
    """Flag a ``config.json`` that describes a different robot than the weights.

    ``authoritative`` says whether apxinf will actually load the model shape from
    this file. For a LeRobot directory it will, so a stale ``input_features``
    silently decides ``num_views``. For an openpi export ``metadata.pt`` wins and
    the file is inert — but a config.json describing a *different* robot sitting
    next to the weights is still the loudest available signal that the directory
    was assembled from more than one training run, which is exactly how a wrong
    ``norm_stats.json`` gets in.
    """
    path = model_dir / "config.json"
    if not path.exists():
        if not authoritative:
            return [
                Finding(
                    INFO,
                    "config.json",
                    "absent, as an openpi PyTorch export always is; the architecture "
                    "comes from metadata.pt",
                )
            ]
        return [Finding(WARN, "config.json", f"missing from {model_dir}")]
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return [Finding(FAIL, "config.json", f"unreadable: {exc}", "fix or re-export")]

    preset = get_robot_preset(robot)
    out = []
    features = config.get("input_features") or {}
    views = sum(1 for f in features.values() if isinstance(f, dict) and f.get("type") == "VISUAL")
    if views and views != preset.num_views:
        out.append(
            Finding(
                FAIL if authoritative else WARN,
                "config.json num_views",
                f"{views} VISUAL features, preset {robot!r} sends {preset.num_views}",
                "apxinf reads the view count from this file, so it decides how many "
                "camera slots the model has. Pass --num-views to serve fewer."
                if authoritative
                else "metadata.pt outranks this file for an openpi export, so this "
                "does not decide the view count -- but it says the file came from "
                "another run, so treat everything else in the directory the same way.",
            )
        )
    else:
        out.append(Finding(INFO, "config.json num_views", str(views)))

    state = (features.get("observation.state") or {}).get("shape")
    action = ((config.get("output_features") or {}).get("action") or {}).get("shape")
    for label, shape, expected in (
        ("observation.state", state, preset.state_dim),
        ("action", action, preset.action_width),
    ):
        if not shape or expected is None:
            continue
        got = shape[-1]
        if got != expected:
            out.append(
                Finding(
                    WARN,
                    f"config.json {label}",
                    f"{got}-dim, but preset {robot!r} is a {expected}-dim robot "
                    f"(repo_id={config.get('repo_id')!r})",
                    "apxinf does not read these fields, so this cannot break serving "
                    "on its own -- but a config.json describing a different robot "
                    "than the weights means the file was copied from another run, "
                    "and whatever else came with it (norm_stats.json above) is "
                    "suspect for the same reason",
                )
            )
    return out


def main() -> int:
    robot_help = "\n".join(f"  {p.describe()}" for p in ROBOT_PRESETS.values())
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"robot presets (--robot):\n{robot_help}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--robot",
        default="unitree_g1",
        choices=available_robots(include_aliases=True),
        help="the preset the checkpoint would be served under",
    )
    parser.add_argument("--norm-key", default="actions")
    parser.add_argument("--tokenizer", type=pathlib.Path, default=None)
    parser.add_argument(
        "--ckpt-format",
        choices=CHECKPOINT_FORMATS,
        default="auto",
        help="how to read the directory; must match what the server will be given",
    )
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--norm-stats", type=pathlib.Path, default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument(
        "--discrete-state", dest="discrete_state", action="store_true", default=None
    )
    parser.add_argument("--no-discrete-state", dest="discrete_state", action="store_false")
    parser.add_argument("--quiet", action="store_true", help="show only WARN and FAIL")
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir.is_dir():
        parser.error(f"--model-dir {model_dir} is not a directory")

    findings = list(
        check_checkpoint(
            model_dir,
            args.robot,
            norm_key=args.norm_key,
            discrete_state=args.discrete_state,
            action_dim=args.action_dim,
            tokenizer_path=args.tokenizer,
            checkpoint_format=args.ckpt_format,
            asset_id=args.asset_id,
            norm_stats=args.norm_stats,
        )
    )
    findings += _check_lerobot_config(
        model_dir,
        args.robot,
        # metadata.pt outranks config.json, so the file only decides the
        # architecture when there is no metadata.pt to outrank it.
        authoritative=not (model_dir / "metadata.pt").is_file(),
    )

    config, reason = load_train_config(model_dir)
    if config is None:
        findings.append(
            Finding(
                WARN,
                "metadata.pt",
                reason,
                "without it the openpi-side contract (cameras, delta convention, "
                "discrete state) cannot be cross-checked and has to be trusted",
            )
        )
    else:
        findings += compare_to_preset(config, args.robot)

    print(f"checkpoint: {model_dir}")
    print(f"preset:     {get_robot_preset(args.robot).describe()}\n")
    # Re-sort: check_checkpoint returns sorted, but the metadata.pt and
    # config.json findings are appended after it. Severity order across the whole
    # report is what makes it readable at a glance.
    order = {FAIL: 0, WARN: 1, INFO: 2}
    findings.sort(key=lambda f: order.get(f.level, 3))
    print(format_findings(findings, include_info=not args.quiet))

    if config is not None:
        print("\nimplied launch:\n")
        print(describe_launch(config, args.robot, model_dir))

    fatal = sum(1 for f in findings if f.level == FAIL)
    warned = sum(1 for f in findings if f.level == WARN)
    print(f"\n{fatal} fatal, {warned} warning(s)")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
