#!/usr/bin/env python3
"""PI0.5 on MUSA: the four serving shells timed, and the one that is scored.

A latency number is only comparable with another one when both runs agree on
what they computed, so this tool does both in one session and one report:

    L0_model       _infer_patches(patches, token_ids, noise)   <- timed AND scored
    L1_rust        infer_rgb(rgb_u8, "nhwc", tokens, noise)    <- timed
    L2_python_api  Pi05Policy.infer(observation)               <- timed
    L3_websocket   one client.infer round trip                 <- timed, attach-only

**L0 is the scored call when a bundle is given.** A ``torch-eager-gold.v1``
bundle (``--reference``) fixes the weights, the configuration, the prefix, the
noise and the solver schedule; the engine is loaded with the bundle's own
``num_views``, ``num_flow_steps`` and ``flow_start_time``, so the actions the
gate reads are the actions of a call this run also timed -- the cosine comes from
the first timed sample and costs no extra inference. Nothing is scored that was
not measured.

Without ``--reference`` there is nothing to score against, and the run is
latency only: the same layers are timed on synthetic zero-filled inputs sized
from the checkpoint itself (``--views``/``--token-count``, or the loader's own
defaults for the solver schedule). Shapes and dtypes are unchanged, so the
milliseconds still describe the same work; the actions do not describe anything.

L1 and L2 are latency only. They take the bundle's frames through a lossy
float32 -> uint8 requantization (about 3.9e-3 relative on the images), which is
what a real client sends but is not what the bundle recorded: scoring them would
measure the requantization, not the engine. L3 attaches to a running server
(``--host/--port``) through this repository's own split-deployment client
(``scripts/pi05_remote_client.py``), which reads the server's metadata for its
key vocabulary and sends a latent seeded by ``--noise-seed``; two calls with one
seed have to come back bit-identical, and that is the wire-integrity check. L3
is not a shell around L0.

Refusals are stated, not scored: a bundle the preflight rejects (an unknown
schema, a schedule the engine would integrate differently, padded tokens, a
fixture the manifest disagrees with), a checkpoint whose bytes differ from the
one the bundle was recorded from, an execution policy the engine did not take, or
repeats that are not bit-identical -- with one fixed input a repeated call has to
produce the same bits or the run is not a measurement. Exit status is 0
measured-and-passed (or measured-only), 1 compared-and-failed, 2 refused.

``--autotune`` measures the exact GEMM shapes this workload uses and persists
them under ``configs/tuning/musa/<device>-mp<N>/`` -- resolved against the
working directory, so run from the repository root. ``--tactics`` names that
directory, not the ``tactics.json`` inside it. ``--reference`` decides both what
is measured and whether it is scored: given, the bundle's frames are the
workload and the gate reads the L0 call; omitted, the layers run on synthetic
inputs and the report says only how long they took.

    # daily: the scored call and the Rust boundary, gate on
    .venv/bin/python scripts/bench_pi05_musa.py --layer l0,l1 \
        --reference <gold-bundle>

    # latency only, no bundle: synthetic inputs, nothing scored
    .venv/bin/python scripts/bench_pi05_musa.py --layer l0,l1

    # the captured path, same gate: the two cosines must be bit-identical
    .venv/bin/python scripts/bench_pi05_musa.py --layer l0 --execution require-graph \
        --reference <gold-bundle>

    # L2 needs a tokenizer no checkpoint ships, and LIBERO's own statistics
    .venv/bin/python scripts/bench_pi05_musa.py --layer l0,l1,l2 \
        --reference <gold-bundle> \
        --tokenizer <path-to-paligemma_tokenizer.model> \
        --norm-stats <ckpt>/assets/physical-intelligence/libero/norm_stats.json

    # the round trip against a server already running
    .venv/bin/python scripts/bench_pi05_musa.py --layer l3 --host 127.0.0.1 --port 8000
    # ... with a seeded latent, so the two determinism calls must match bit for bit
    .venv/bin/python scripts/bench_pi05_musa.py --layer l3 --noise-seed 0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _package in ("python/apxinf", "python/apxinf_ref"):
    _path = _REPO_ROOT / _package
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from apxinf._bench import (
    IN_PROCESS,
    env_snapshot,
    git_commit,
    parse_layers,
    stats,
    time_loop,
)
from pi05.torch_gold import artifact

SCHEMA = "apxinf.pi05.latency.v2"
LAYER_NAMES = {"l0": "L0_model", "l1": "L1_rust", "l2": "L2_python_api", "l3": "L3_websocket"}
L3_SEGMENTS = ("round_trip", "model", "server_processor", "transport")

DEFAULT_CHECKPOINT = pathlib.Path("~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch")
# A synthetic run's workload: the artifact module's own fixture defaults, so a
# latency-only run stands in for the scored one without restating its shapes.
DEFAULT_VIEWS, DEFAULT_TOKEN_COUNT = artifact.DEFAULT_VIEWS, artifact.DEFAULT_TOKEN_COUNT
# This port's regression gate, not the artifact module's descriptive bands:
# 0.9999 means "only the accumulation order differs" and 0.99 is the
# cross-implementation floor; acceptance sits between them.
DEFAULT_GATE = 0.999
# Exactly ten PaliGemma tokens, the fixture's prefix length; L2 re-tokenizes it,
# so the layers stay comparable in work.
PROMPT_T10 = "put both moka pots on the stove"

# Every switch that changes what a run of this tool measures: the backend's own
# `env::var` reads plus the two that decide whether a trace is labelled.
ENV_WATCH = (
    "APXINF_MUSA_ALLOC_CACHE", "APXINF_MUSA_ALLOC_CACHE_MB", "APXINF_MUSA_ARCH",
    "APXINF_MUSA_ASSEMBLY_ARCH", "APXINF_MUSA_BF16_MHA", "APXINF_MUSA_D256_ROUTE",
    "APXINF_MUSA_FA_REPLAY_DIR", "APXINF_MUSA_OUTPUT_FILL",
    "APXINF_MUSA_POISON_ALLOC", "APXINF_MUSA_VENDOR_FMHA_ARCH", "APXINF_MUSA_WORKSPACE_SLACK",
    "MUSA_INFLIGHT_SUBMISSION_LIMIT", "MTTX_INJECTION64_PATH",
    # Switches that change what an INT8 run computes or how fast: the
    # equalization table, the load-time exclusion knob, and the two row-level
    # diagnostics (which read activations back and must stay off while timing).
    "APXINF_INT8_EQUALIZATION", "APXINF_INT8_DYNAMIC_EXCLUDE",
    "APXINF_PI05_INT8_STATS", "APXINF_PI05_INT8_CHANNELS",
)


def _refuse(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--layer", default=None,
                        help="comma list of l0,l1,l2,l3 or `all` (default: l0,l1)")
    parser.add_argument("--execution", default="prefer-graph",
                        choices=("eager", "prefer-graph", "require-graph"),
                        help="policy requested at load; the mode actually taken is recorded "
                        "per layer. `prefer-graph` captures and falls back to eager, which "
                        "the report marks as a fallback (default: prefer-graph)")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT,
                        help="checkpoint directory holding model.safetensors")
    parser.add_argument("--device", default="musa:0", help="MUSA device (default: musa:0)")
    parser.add_argument("--model-variant", default="bf16",
                        choices=("bf16", "int8_dynamic", "int8_dynamic_no_encode"),
                        help="PI0.5 variant. `int8_dynamic` quantizes every projection an "
                             "embedded s8mma tile can serve (the ViT patch embedding and the "
                             "action head stay BF16) and needs APXINF_INT8_EQUALIZATION for "
                             "the language layers to clear the gate; "
                             "`int8_dynamic_no_encode` keeps those four language projections "
                             "in BF16 instead")
    parser.add_argument("--reference", type=pathlib.Path, default=None,
                        help="torch-eager-gold.v1 bundle; given, its frames are the workload "
                        "and the L0 call is scored against it, omitted, the layers run on "
                        "synthetic inputs and nothing is scored")
    parser.add_argument("--views", type=int, default=None,
                        help="cameras for a synthetic run (default: 2); refused with "
                        "--reference, whose bundle fixes the frame count")
    parser.add_argument("--token-count", type=int, default=None,
                        help="prefix tokens for a synthetic run (default: 10); refused with "
                        "--reference, whose bundle fixes the prefix")
    parser.add_argument("--gate", type=float, default=DEFAULT_GATE,
                        help=f"minimum raw_actions cosine (default: {DEFAULT_GATE})")
    parser.add_argument("--allow-checkpoint-mismatch", action="store_true",
                        help="score a checkpoint the bundle was not recorded from")
    parser.add_argument("--tactics", type=pathlib.Path, default=None,
                        help="tactic-store DIRECTORY (the loader appends tactics.json)")
    parser.add_argument("--autotune", action="store_true",
                        help="measure missing exact GEMM shapes and persist them under "
                        "configs/tuning/musa/<device>-mp<N>/")
    parser.add_argument("--tokenizer", type=pathlib.Path, default=None,
                        help="SentencePiece model for L2 (else APXINF_TOKENIZER, else beside "
                        "the checkpoint)")
    parser.add_argument("--norm-stats", type=pathlib.Path, default=None,
                        help="L2 normalization statistics (omitted: the checkpoint's lookup, "
                        "else identity)")
    parser.add_argument("--action-dim", type=int, default=None,
                        help="trim L2's deployable action width")
    parser.add_argument("--prompt", default=PROMPT_T10, help="L2/L3 prompt")
    parser.add_argument("--warmup", type=int, default=10, help="discarded calls (default: 10)")
    parser.add_argument("--samples", type=int, default=30, help="timed calls (default: 30)")
    parser.add_argument("--host", default="127.0.0.1", help="L3 server host")
    parser.add_argument("--port", type=int, default=8000, help="L3 server port")
    parser.add_argument("--noise-seed", type=int, default=0,
                        help="L3 latent seed; two calls with one fixed seed must match "
                             "bit for bit, and a negative value sends no latent at all")
    parser.add_argument("--engine-actions", type=pathlib.Path, default=None,
                        help="score a stored actions npz instead of running; no device, no timing")
    parser.add_argument("--export", type=pathlib.Path, default=None,
                        help="write the L0 actions here")
    parser.add_argument("--label", default=None, help="free-form label recorded in the report")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="write the json report")
    return parser.parse_args()


def _bundle(reference: pathlib.Path):
    """Read and preflight the reference bundle; its fixture is the workload."""

    if not (reference / "debug.npz").is_file() or not (reference / "manifest.json").is_file():
        _refuse(f"no reference bundle at {reference} (inputs.npz, debug.npz, manifest.json)")
    try:
        fixture = artifact.load_fixture(reference / "inputs.npz")
        expected, manifest = artifact.reference_actions(reference)
    except (OSError, ValueError, KeyError) as error:
        _refuse(f"{reference} is not a readable bundle: {error}")
    problems = artifact.replay_problems(reference, fixture, manifest)
    if problems:
        _refuse(f"{reference} cannot be scored:\n  " + "\n  ".join(problems))
    return fixture, expected, manifest


def _identity(reference: pathlib.Path, fixture, manifest) -> tuple[dict, dict]:
    """What the bundle says it is, and what it holds.

    The three file hashes say a bundle changed; the per-input ones, computed with
    the artifact module's own digest, say *which input* did -- and they are
    comparable with the ``sha256`` the manifest records for the same array.
    """

    bundle = {
        "path": str(reference),
        "schema": manifest.get("schema"),
        "implementation": manifest.get("implementation"),
        "runtime": manifest.get("runtime"),
        "fixture": manifest.get("fixture"),
        "checkpoint": manifest.get("checkpoint"),
        "inputs_sha256": artifact.sha256(reference / "inputs.npz"),
        "debug_sha256": artifact.sha256(reference / "debug.npz"),
        "manifest_sha256": artifact.sha256(reference / "manifest.json"),
    }
    inputs = {
        name: {"shape": list(array.shape), "dtype": str(array.dtype),
               "sha256": artifact.array_digest(array)}
        for name, array in sorted(fixture.items())
    }
    return bundle, inputs


def _declared_config(checkpoint_dir: pathlib.Path) -> dict:
    """The checkpoint's own ``config.json`` -- the file the loader reads, read again.

    Reported, never used to load: the loader applies these values itself, and a
    field the file omits is a loader default, not something this tool can claim
    to know. ``pi05_libero_pytorch`` declares the two action shapes and nothing
    else, so its solver schedule is recorded as unknown rather than assumed.
    """

    path = checkpoint_dir / "config.json"
    if not path.is_file():
        return {}
    try:
        declared = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return declared if isinstance(declared, dict) else {}


def _load_engine(args, checkpoint_dir: pathlib.Path, fixture, manifest, verify: bool):
    """Load PI0.5 with the workload's own configuration pinned, and check what loaded.

    With a bundle every pin is the bundle's. Without one only the view count is
    pinned (``--views``): the solver schedule stays the checkpoint's, which is a
    loader default this tool records as unknown rather than guesses at.
    """

    import apxinf_py

    weights = checkpoint_dir / "model.safetensors"
    if not weights.is_file():
        _refuse(f"no checkpoint weights at {weights}")
    identity = {"path": str(checkpoint_dir), "model_sha256": None,
                "config_sha256": None, "matches_reference": None}
    if verify:
        # A bundle fixes the weights as much as the inputs, and the asset names
        # are not identities -- `pi05-base` and `pi05_libero` differ in their bytes
        # under the same kind of path -- so the recorded hash is the only thing
        # that says both sides ran the same model.
        identity["model_sha256"] = artifact.sha256(weights)
        recorded = (manifest.get("checkpoint") or {}).get("sha256")
        mismatch = recorded is not None and identity["model_sha256"] != recorded
        identity["matches_reference"] = None if recorded is None else not mismatch
        if mismatch and not args.allow_checkpoint_mismatch:
            _refuse(
                f"{weights} has sha256 {identity['model_sha256']}, but the bundle was recorded "
                f"from {recorded}; pass --allow-checkpoint-mismatch to score it anyway"
            )
        if mismatch:
            print("WARNING: scoring a checkpoint the bundle was not recorded from",
                  file=sys.stderr)
        if (checkpoint_dir / "config.json").is_file():
            identity["config_sha256"] = artifact.sha256(checkpoint_dir / "config.json")

    if args.tactics is not None:
        if args.tactics.is_file():
            _refuse(f"--tactics names the file {args.tactics}; this loader appends "
                    "`tactics.json` to the directory it is given, so pass the directory")
        if not (args.tactics / "tactics.json").is_file() and not args.autotune:
            _refuse(f"{args.tactics} holds no tactics.json; run once with --autotune to write it")

    if fixture is not None:
        pins = {"num_views": int(fixture["images"].shape[0]),
                "num_flow_steps": int(fixture["num_flow_steps"]),
                "flow_start_time": float(fixture["flow_start_time"])}
    else:
        pins = {"num_views": args.views if args.views is not None else DEFAULT_VIEWS}
    handle = apxinf_py.ModelRunner.load(
        "pi05", str(weights), device=args.device, model_variant=args.model_variant,
        execution=args.execution, **pins,
        tactics=None if args.tactics is None else str(args.tactics), autotune=args.autotune,
    )
    declared = (manifest or {}).get("model_config", {})
    for attribute in ("action_horizon", "action_dim", "num_views"):
        actual = getattr(handle, attribute)
        if declared.get(attribute) is not None and actual != declared[attribute]:
            _refuse(f"the checkpoint's {attribute}={actual} is not the bundle's "
                    f"{declared[attribute]}")
    if fixture is not None and handle.image_size != fixture["images"].shape[2]:
        _refuse(f"the checkpoint's image_size {handle.image_size} is not the fixture's "
                f"{fixture['images'].shape[2]}")
    return handle, identity


def _check_mode(args, handle, layer: str) -> str:
    """The engine reports the mode it took; a request not honoured is a refusal."""

    mode = handle.execution_mode
    expected = {"eager": "eager", "require-graph": "graph"}.get(args.execution)
    if expected is not None and mode != expected:
        _refuse(f"--execution {args.execution} ran as {mode!r} on {layer}")
    return mode


def _requantize_to_u8(images):
    """``(views, 3, H, W)`` float32 in [-1, 1] -> ``(views, H, W, 3)`` uint8.

    What a client actually sends: RGB bytes in [0, 255] where the fixture holds
    normalized floats. Lossy, hence latency-only downstream. Same arithmetic as
    ``scripts/pi05_remote_client._requantize_to_u8``;
    ``tests/test_bench_pi05_musa.py`` pins the two together by value.
    """
    return np.ascontiguousarray(
        np.rint((images.transpose(0, 2, 3, 1).astype(np.float32) + 1.0) * 127.5)
        .clip(0.0, 255.0)
        .astype(np.uint8)
    )


def _inputs(fixture, handle, args, checkpoint_dir: pathlib.Path):
    """The layers' inputs, built once: L0 exactly, L1/L2 through uint8.

    A bundle supplies every byte. Without one the arrays are zeros at the
    checkpoint's own shapes, which changes what the actions mean and nothing
    about what they cost; the solver schedule is then the checkpoint's, reported
    only where ``config.json`` states it.
    """

    if fixture is not None:
        images = fixture["images"].astype(np.float32)
        tokens, noise = fixture["token_ids"], fixture["noise"]
        schedule = {"num_flow_steps": int(fixture["num_flow_steps"]),
                    "flow_start_time": float(fixture["flow_start_time"]),
                    "dt": float(fixture["dt"])}
        source = "bundle"
    else:
        images = np.zeros((handle.num_views, 3, handle.image_size, handle.image_size),
                          dtype=np.float32)
        # A latent drawn per call would make the repeat check fire; zeros are the
        # same bits every call, so the check still says something about the engine.
        tokens = np.zeros(args.token_count if args.token_count is not None
                          else DEFAULT_TOKEN_COUNT, dtype=np.uint32)
        noise = np.zeros((handle.action_horizon, handle.action_dim), dtype=np.float32)
        declared = _declared_config(checkpoint_dir)
        steps, start = declared.get("num_flow_steps"), declared.get("flow_start_time")
        schedule = {"num_flow_steps": steps, "flow_start_time": start,
                    "dt": None if not (steps and start) else -float(start) / float(steps)}
        source = "synthetic"

    inputs = {
        "tokens": np.ascontiguousarray(tokens, dtype=np.uint32),
        "noise": np.ascontiguousarray(noise, dtype=np.float32),
        "patches": artifact.patchify(images, handle.patch_size),
        # What a client actually sends: RGB bytes in [0, 255] where the frames are
        # normalized floats. Lossy, hence latency-only downstream.
        "rgb": _requantize_to_u8(images),
    }
    workload = {"source": source, "token_count": int(inputs["tokens"].size),
                "action_horizon": handle.action_horizon, "action_dim": handle.action_dim,
                "num_views": handle.num_views, "image_size": handle.image_size,
                "patch_size": handle.patch_size, **schedule}
    return inputs, workload


def _build_policy(args, checkpoint_dir: pathlib.Path, handle):
    """L2 around the handle already in memory: same weights, same plan, plus processors."""

    from apxinf import Pi05Policy

    options = {"model_runner": handle, "tokenizer_path": args.tokenizer}
    if args.norm_stats is not None:
        options["norm_stats"] = str(args.norm_stats)
    if args.action_dim is not None:
        options["action_dim"] = args.action_dim
    return Pi05Policy.from_pretrained(checkpoint_dir, **options)


def _time_layers(args, layers, handle, inputs, policy, observation):
    """Time the requested in-process layers; L0's first sample is the scored one."""

    runners = {
        "l0": lambda: handle._infer_patches(inputs["patches"], inputs["tokens"], inputs["noise"]),
        "l1": lambda: handle.infer_rgb(inputs["rgb"], "nhwc", inputs["tokens"], inputs["noise"]),
        "l2": lambda: policy.infer(observation),
    }
    timings, first, modes, repeats = {}, {}, {}, {}
    for layer in IN_PROCESS:
        if layer not in layers:
            continue
        timings[layer], outputs = time_loop(runners[layer], args.warmup, args.samples)
        modes[layer] = _check_mode(args, handle, layer)
        if layer == "l2":
            # The policy samples its latent on device: its repeats are meant to
            # differ, which is why it is not a layer anything is scored against.
            continue
        actions = [np.asarray(output, dtype=np.float32) for output in outputs]
        repeats[layer] = max(
            (float(np.abs(actions[0] - other).max()) for other in actions[1:]), default=0.0
        )
        if repeats[layer] != 0.0:
            _refuse(f"{LAYER_NAMES[layer]} was given the same input twice and moved by "
                    f"{repeats[layer]:g}; a repeated call has to produce the same bits")
        first[layer] = actions[0]
    return timings, first, modes, repeats


def _score(args, actions, expected) -> dict:
    """The gate: one cosine over the whole chunk, with the band that describes it."""

    if actions.shape != expected.shape:
        print(f"FAIL: shape mismatch, engine {actions.shape} vs reference {expected.shape}",
              file=sys.stderr)
        raise SystemExit(1)
    if not np.isfinite(actions).all():
        print("FAIL: the engine produced a non-finite action", file=sys.stderr)
        raise SystemExit(1)
    score = artifact.cosine_similarity(actions, expected)
    if score >= artifact.ALIGNMENT_TIGHT:
        band = "aligned: only the accumulation order differs"
    elif score >= artifact.ALIGNMENT_FLOOR:
        band = "aligned within cross-implementation tolerance"
    else:
        band = "DIVERGENT"
    return {"gate": args.gate, "cosine": score,
            "max_abs": float(np.abs(actions - expected).max()), "band": band,
            "passed": score >= args.gate}


def _remote_client():
    """The repository's split-deployment client, imported from beside this file.

    ``scripts/`` is a plain directory, not a package, so the same import has two
    spellings: from the repository root the namespace package works, and run as
    ``python scripts/bench_pi05_musa.py`` the script's own directory is on
    ``sys.path``.
    """
    try:
        from scripts.pi05_remote_client import RemotePolicy, timing_segments
    except ImportError:
        from pi05_remote_client import RemotePolicy, timing_segments
    return RemotePolicy, timing_segments


def _run_l3(args, model_variant: str, action_horizon: int):
    """One round trip per call against a server this process does not own.

    Everything the client sends is read off the server's metadata, so the
    observation uses the server's own view keys rather than a hardcoded set.
    With a seeded latent the same request has to come back bit-identical; that
    comparison is the one wire-integrity check this path has, because the engine
    draws a fresh latent per request by default.
    """
    RemotePolicy, timing_segments = _remote_client()
    noise_seed = None if args.noise_seed < 0 else args.noise_seed
    client = RemotePolicy(
        args.host,
        args.port,
        expect_model_variant=model_variant,
        noise_seed=noise_seed,
    )
    try:
        metadata = client.metadata
        horizon = int(metadata.get("action_horizon", action_horizon))
        view = int((metadata.get("image_size") or (224, 224))[0])
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (view, view, 3), dtype=np.uint8)
                  for _ in client.image_keys]
        observation = client.build_observation(
            frames, state=np.zeros(8, dtype=np.float32), prompt=args.prompt)

        def infer() -> tuple:
            started = time.perf_counter()
            response = client.infer(observation)
            round_trip = (time.perf_counter() - started) * 1000.0
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.shape[0] != horizon or not np.isfinite(actions).all():
                _refuse(f"the server returned actions this tool cannot use: {actions.shape}")
            return response, actions, round_trip

        determinism = None
        if noise_seed is not None:
            pair = [infer()[1] for _ in range(2)]
            delta = float(np.abs(pair[0] - pair[1]).max())
            determinism = {
                "latent": f"seeded:{noise_seed}",
                "identical": bool(np.array_equal(pair[0], pair[1])),
                "max_abs_delta": delta,
            }
            if not determinism["identical"]:
                _refuse("two requests carrying the same fixed latent returned different "
                        f"actions (max|delta|={delta:.3e}); the wire is not intact")

        for _ in range(args.warmup):
            infer()
        segments = {name: [] for name in L3_SEGMENTS}
        for _ in range(args.samples):
            response, _, round_trip = infer()
            for name, value in timing_segments(response, round_trip).items():
                segments[name].append(value)
        return metadata, segments, determinism
    finally:
        client.close()


def _print_execution(report) -> None:
    """Which mode the timed calls ran in -- a latency-only run needs it too."""

    if "execution" not in report:
        return
    modes = ", ".join(f"{layer}={mode}"
                      for layer, mode in report["execution"]["layers"].items())
    print(f"execution: {report['execution']['requested']} -> {modes}")


def _print_report(report, prompt: str) -> None:
    """The console half: the layer table and its deltas, the gate, the round trip."""

    workload = report.get("workload")
    if workload is not None:
        header = (f"in-process latency  |  {report['model_variant']}  "
                  f"H={workload['action_horizon']} Dmodel={workload['action_dim']} "
                  f"views={workload['num_views']} T={workload['token_count']}")
        print(header)
        print("-" * len(header))
        print(f"{'layer':<16}{'p50':>9}{'p95':>9}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
        for layer in IN_PROCESS:
            record = report["layers_ms"].get(LAYER_NAMES[layer])
            if record is None:
                continue
            print(f"{LAYER_NAMES[layer]:<16}{record['p50']:>9.2f}{record['p95']:>9.2f}"
                  f"{record['mean']:>9.2f}{record['std']:>9.2f}{record['min']:>9.2f}"
                  f"{record['max']:>9.2f}")
        measured = report["layers_ms"]
        for inner, outer, label in (("L0_model", "L1_rust", "L1-L0 (rust vision->patches + PyO3)"),
                                    ("L1_rust", "L2_python_api", "L2-L1 (numpy pre/post chain)")):
            if inner in measured and outer in measured:
                print(f"\n  {label:<38} {measured[outer]['p50'] - measured[inner]['p50']:+.2f} ms")
    if "l3_segments_ms" in report:
        print(f"\nL3 websocket  prompt={prompt!r} "
              f"({report['warmup']} warmup + {report['samples']} samples)")
        print(f"{'segment':<18}{'p50':>9}{'p95':>9}{'min':>9}{'max':>9}{'mean':>9}{'std':>8}")
        for name, record in report["l3_segments_ms"].items():
            print(f"{name:<18}{record['p50']:>9.3f}{record['p95']:>9.3f}{record['min']:>9.3f}"
                  f"{record['max']:>9.3f}{record['mean']:>9.3f}{record['std']:>8.3f}")
    accuracy = report["accuracy"]
    if not accuracy["scored"]:
        _print_execution(report)
        advice = ("pass --reference <gold-bundle> to score the L0 call"
                  if any(layer in IN_PROCESS for layer in report.get("layers", ()))
                  else "L3 is attach-only, so add l0/l1/l2 to score this run")
        print(f"\naccuracy: not scored ({accuracy['reason']}) -- latency only; {advice}")
        return
    print(f"\nengine vs {pathlib.Path(accuracy['reference']['path']).name}: "
          f"cosine {accuracy['cosine']:.8f}   max_abs {accuracy['max_abs']:.6g}")
    print(f"band: {accuracy['band']}")
    _print_execution(report)
    print(f"gate: cosine >= {accuracy['gate']} {'PASS' if accuracy['passed'] else 'FAIL'}")


def main() -> int:
    args = parse_args()
    if args.device.split(":")[0] != "musa":
        _refuse(f"{args.device} is not a MUSA device; scripts/bench_pi05.py covers other backends")
    if args.warmup < 0 or args.samples < 1:
        _refuse("--warmup must be >= 0 and --samples >= 1")
    if args.views is not None and args.views < 1:
        _refuse("--views must be >= 1")
    if args.token_count is not None and args.token_count < 1:
        _refuse("--token-count must be >= 1")
    if args.engine_actions is not None:
        if args.reference is None:
            _refuse("--engine-actions scores stored actions; it needs --reference")
        if args.layer is not None:
            _refuse("--engine-actions measures no layer; drop --layer")

    verify = args.reference is not None
    layers = parse_layers(args.layer if args.layer is not None else "l0,l1")
    in_process = [layer for layer in layers if layer in IN_PROCESS]
    if not in_process and verify:
        _refuse("--layer l3 alone has nothing in-process to score; add l0, or drop --reference")
    if verify and (args.views is not None or args.token_count is not None):
        _refuse("--views/--token-count size a synthetic workload; with --reference the "
                "bundle's frames already fix both")
    if args.token_count is not None and "l2" in layers:
        # L2 re-tokenizes --prompt, so a prefix flag that moved only L0/L1 would
        # make the printed L2-L1 delta compare two different prefixes.
        _refuse("--token-count sizes the L0/L1 prefix; L2 re-tokenizes --prompt, so its "
                "prefix length is the tokenizer's")
    if args.export is not None and not verify:
        _refuse("--export writes the L0 actions; without --reference they are computed "
                "from synthetic inputs")

    report = {"schema": SCHEMA, "backend": "musa",
              "report_producer": "scripts/bench_pi05_musa.py", "run_label": args.label,
              "git_commit": git_commit(_REPO_ROOT), "model_variant": args.model_variant,
              "device": args.device, "weights": "checkpoint", "layers": layers,
              "warmup": args.warmup, "samples": args.samples, "autotune": args.autotune,
              "env": env_snapshot(ENV_WATCH)}
    reference = args.reference.expanduser().resolve() if args.reference is not None else None
    fixture = expected = manifest = None
    if reference is not None:
        fixture, expected, manifest = _bundle(reference)
    accuracy = {"scored": verify}
    if verify:
        bundle_identity, inputs_identity = _identity(reference, fixture, manifest)
        accuracy.update({"layer": "l0", "reference": bundle_identity, "inputs": inputs_identity})
    else:
        accuracy["reason"] = "no --reference"

    if args.engine_actions is not None:
        actions = np.load(args.engine_actions, allow_pickle=False)["actions"]
        print(f"replaying {args.engine_actions}")
        accuracy.update(_score(args, np.asarray(actions, dtype=np.float32), expected))
        accuracy.update(timed=False, actions=str(args.engine_actions))
        report["accuracy"] = accuracy
        _print_report(report, args.prompt)
        return _finish(args, report, accuracy)

    checkpoint_dir = args.checkpoint.expanduser()
    handle = policy = inputs = workload = None
    if in_process:
        handle, report["checkpoint"] = _load_engine(
            args, checkpoint_dir, fixture, manifest, verify
        )
        report["model_dir"] = str(checkpoint_dir)
        inputs, workload = _inputs(fixture, handle, args, checkpoint_dir)
        workload["prompt"] = args.prompt if "l2" in layers else None
        report["workload"] = workload
    observation = None
    if "l2" in layers:
        try:
            policy = _build_policy(args, checkpoint_dir, handle)
        except Exception as error:  # noqa: BLE001 - any load failure is a refusal with its reason
            _refuse(str(error))
        observation = {key: inputs["rgb"][index] for index, key in enumerate(policy.image_keys)}
        observation[policy.prompt_key] = args.prompt
        workload["deploy_action_dim"] = policy.action_dim

    timings, first, modes, repeats = {}, {}, {}, {}
    if in_process:
        timings, first, modes, repeats = _time_layers(
            args, layers, handle, inputs, policy, observation
        )
        report["execution"] = {"requested": args.execution, "layers": modes,
                               "fallback": args.execution == "prefer-graph"
                               and "graph" not in modes.values()}
        # The loader resolves this root against the working directory; naming the
        # absolute path makes "run from the repository root" checkable.
        report["tuning"] = {"root": str((pathlib.Path.cwd() / "configs" / "tuning").resolve()),
                            "tactics": str(args.tactics) if args.tactics is not None else None,
                            "autotune": args.autotune}
        report["tactics"] = report["tuning"]["tactics"]
        report["layers_ms"] = {LAYER_NAMES[layer]: stats(ms) for layer, ms in timings.items()}
        report["raw_ms"] = {LAYER_NAMES[layer]: ms for layer, ms in timings.items()}

    if "l3" in layers:
        metadata, segments, determinism = _run_l3(
            args, args.model_variant, handle.action_horizon if handle is not None else 10
        )
        report["l3_client"] = "scripts/pi05_remote_client.py"
        report["l3_server_metadata"] = metadata
        report["l3_determinism"] = determinism
        report["l3_segments_ms"] = {name: stats(ms) for name, ms in segments.items()}
        report["l3_raw_ms"] = segments

    scored = first.get("l0")
    if verify:
        if scored is None:
            # The gate is a property of the checkpoint and the engine, not of the
            # layer selection: one L0 call costs less than a latency report with no
            # accuracy statement attached to it.
            scored = np.asarray(
                handle._infer_patches(inputs["patches"], inputs["tokens"], inputs["noise"]),
                dtype=np.float32,
            )
        accuracy.update(_score(args, scored, expected))
        accuracy.update(timed="l0" in first, repeat_max_abs=repeats.get("l0"),
                        # Only an L0 run measured repeats; an l1/l2-only run
                        # states "not measured" rather than "deterministic".
                        deterministic=(None if "l0" not in repeats
                                       else repeats["l0"] == 0.0))
    if args.export is not None:
        if scored is None:
            _refuse("--export writes the L0 call, which --layer does not include")
        args.export.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.export, actions=scored)
        print(f"actions written to {args.export}")

    report["accuracy"] = accuracy
    _print_report(report, args.prompt)
    if policy is not None:
        policy.close()
    return _finish(args, report, accuracy)


def _finish(args, report, accuracy) -> int:
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    if not accuracy["scored"]:
        return 0
    return 0 if accuracy["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
