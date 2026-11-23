#!/usr/bin/env python3
"""Generate a deterministic Pi0.5 PyTorch eager reference.

The model is the vendored ``PI0Pytorch`` snapshot under ``python/apxinf_ref/pi05/_vendor``,
and the reference comes from its own end-to-end entry point,
``PI0Pytorch.sample_actions`` -- this command does not re-implement the Euler
loop.  ``_preprocess_observation`` is replaced so tokenizer and camera
preprocessing stay out of the comparison: images are the fixture's normalized
NCHW tensors, token ids are the fixture's ids, and noise enters the Euler loop
directly, so a CUDA capture and a MUSA replay see exactly the same input bytes.

Every tensor on the model's own forward boundary is captured in full, with
sampled layer signatures beside them to locate a divergence.  A run that replays
another artifact's fixture (``--input``) also prints the cosine similarity of its
own capture against that artifact, so the command producing the numbers is the
command that reports how close they landed.  Nothing is published unless it
verifies: the staged files are read back off disk and re-checked against the
manifest before the output directory appears, and every captured tensor is read
until its reads settle on one value -- on this host a device->host read can come
back holding another tensor's bytes.  Such a run fails loudly (exit status 3,
nothing published).  Timing covers the un-instrumented path only, after warmup,
and stays out of the manifest, so two runs over one fixture produce byte-identical
manifest bytes.

Examples::

    python python/apxinf_ref/pi05/generate_pi05_torch_gold.py --device cuda:0 \
        --output devlocal/pi05-gold/orin
    python python/apxinf_ref/pi05/generate_pi05_torch_gold.py --device musa:0 \
        --input <path-to-inputs.npz> \
        --output devlocal/pi05-gold/musa
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))  # `torch_gold`, and the `_vendor` snapshot it loads


def parse_args(argv=None, *, description: str | None = None):
    """The command line: every default this tool ships with, and the parser.

    The description comes from the entry point's docstring, so ``--help`` shows
    the tool's own explanation of what an artifact is and how it is checked.  It
    imports only the torch-free half of the package, which is what lets ``--help``
    run on a host with no device stack.
    """

    from torch_gold import REPO_ROOT
    from torch_gold.artifact import DEFAULT_NUM_FLOW_STEPS, DEFAULT_TOKEN_COUNT, DEFAULT_VIEWS

    default_checkpoint = Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
    default_warmup_runs, default_timed_runs = 3, 3
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(os.environ.get("APXINF_PI05_CHECKPOINT", default_checkpoint)))
    parser.add_argument("--device", default="musa:0", help="cpu, cuda:N, or musa:N (default: musa:0)")
    parser.add_argument("--input", type=Path, help="existing inputs.npz fixture to replay; outputs in it are ignored")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "devlocal" / "pi05-gold" / "run")
    parser.add_argument("--seed", type=int, default=0, help="seed for a newly generated canonical fixture")
    parser.add_argument("--views", type=int, default=DEFAULT_VIEWS, help="views for a new fixture (default: 2)")
    parser.add_argument("--token-count", type=int, default=DEFAULT_TOKEN_COUNT)
    parser.add_argument("--num-flow-steps", type=int, default=None,
                        help=f"Euler steps; defaults to the fixture's own value, else {DEFAULT_NUM_FLOW_STEPS}")
    parser.add_argument("--precision", choices=("checkpoint", "bfloat16", "float32"), default="checkpoint")
    parser.add_argument("--sample-count", type=int, default=256,
                        help="points sampled per stage hook call; 0 disables stage vectors "
                             "(boundary tensors are stored in full either way)")
    parser.add_argument("--warmup-runs", type=int, default=default_warmup_runs,
                        help=f"un-instrumented runs discarded before timing (default: {default_warmup_runs})")
    parser.add_argument("--timed-runs", type=int, default=default_timed_runs,
                        help=f"un-instrumented runs timed for the reported number (default: {default_timed_runs})")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def bootstrap_torch(device_spec: str) -> None:
    """Import torchada before anything imports torch.

    torchada patches torch as it is imported, so an unpatched torch anywhere
    earlier in the process would silently cost this host its MUSA kernels.  That
    is why the driving half of the package is imported inside :func:`main`, after
    this call, rather than at module scope -- and why ``--help`` still works
    without torch.
    """

    if device_spec.split(":", 1)[0].strip().lower() == "musa":
        try:
            import torchada  # noqa: F401  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("device 'musa' requires torchada; install the MUSA torchada extra") from error
    import torch  # noqa: F401


def main(argv=None) -> int:
    args = parse_args(argv, description=__doc__)
    bootstrap_torch(args.device)
    from torch_gold.capture import run

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
