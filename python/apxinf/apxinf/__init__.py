"""ApxInf Python frontend.

Public modules:

* :mod:`apxinf.processors` — pure-numpy pre/post-processing *steps* (resize,
  tokenize, normalize, noise) plus a :class:`~apxinf.processors.Pipeline`
  container. Each step is independently instantiable and callable on its natural
  input, with no GPU / no Rust dependency, so it unit-tests offline. Every step
  here is determined by the **checkpoint**; steps determined by a robot body live
  outside this package.
* **policies** — the **L2** layer (:mod:`apxinf.policies`).
  :class:`~apxinf.policies.impls.pi05.Pi05Policy` composes a pre pipeline + a
  bare-model handle (L1) + a post pipeline into a single
  ``infer(obs_dict, noise=None) -> {actions, timing, ...}`` call.
  :class:`~apxinf.policies.auto.AutoPolicy` dispatches a checkpoint to its concrete
  policy by ``config.json`` model type; :class:`~apxinf.policies.base.Policy` is
  the structural contract they all satisfy.
  :class:`~apxinf.policies.base.ComposablePolicy` is the capability an *outer*
  layer needs to wrap its own steps around a policy's chain.
* :mod:`apxinf.checkpoints` — what a checkpoint directory declares about itself:
  layout detection, metadata, norm stats, and
  :func:`~apxinf.checkpoints.inspect_checkpoint`, which reports whether a
  directory is self-consistent and servable.
* **bindings** — :class:`Model` re-exports the ``apxinf_py`` PyO3 handle (L1
  bare-model inference; an internal L0 patches path exists but is private). It is
  the single public surface; you never import ``apxinf_py`` directly.
* :mod:`apxinf.serving` — the websocket policy server (a thin, model-agnostic
  transport shell over any :class:`Policy`, with an openpi-compatible wire).
  Imported only on demand (``from apxinf.serving import WebsocketPolicyServer``)
  so its ``msgpack`` / ``websockets`` deps stay out of offline processor use.

This package holds **no dataset's wire contract and no robot's body**. What a
client calls its cameras, how many joints an arm has, which action components are
deltas — none of that is determined by the weights, so a caller names its own
keys through ``image_keys=`` / ``state_key=`` / ``prompt_key=`` and wraps its own
body steps through :meth:`~apxinf.policies.base.ComposablePolicy.with_adapter`.
:data:`CANONICAL_IMAGE_KEYS` / :data:`CANONICAL_STATE_KEY` /
:data:`CANONICAL_PROMPT_KEY` name the neutral fallback for callers that want to
address it without restating string literals. A robot/dataset/simulator
adaptation layer builds on top of these seams; none of it lives here.

``import apxinf`` never touches CUDA: only ``apxinf.Model`` (accessed lazily) and a
policy's ``from_pretrained`` pull in the ``apxinf_py`` binding.
"""

from __future__ import annotations

from . import processors
from .calibration import (
    CalibrationContext,
    CalibrationPlan,
    CalibrationRunner,
    ConsumerContract,
    CaptureSite,
    Fp8ExecutionPlan,
    QuantizationSpec,
    QuantizedOperator,
    adapt_records,
)
from .policies import (
    CANONICAL_IMAGE_KEYS,
    CANONICAL_PROMPT_KEY,
    CANONICAL_STATE_KEY,
    VIEW_SLOTS,
    AutoPolicy,
    ComposablePolicy,
    Pi05Policy,
    Policy,
    WallossPolicy,
)
from .processors import (
    GaussianNoise,
    Normalizer,
    ParseImage,
    Pipeline,
    ProcessorStep,
    PromptTokenizer,
    ResizeWithPad,
    Unnormalizer,
)

__all__ = [
    "processors",
    # policy contract (outward); BareModel (inward) lives in apxinf.policies
    "Policy",
    "ComposablePolicy",
    # L2 policies
    "Pi05Policy",
    "WallossPolicy",
    "AutoPolicy",
    # offline calibration framework
    "CalibrationContext",
    "CalibrationPlan",
    "CalibrationRunner",
    "ConsumerContract",
    "CaptureSite",
    "Fp8ExecutionPlan",
    "QuantizationSpec",
    "QuantizedOperator",
    "adapt_records",
    # model vocabulary + the neutral fallback wire keys (see module docstring)
    "VIEW_SLOTS",
    "CANONICAL_IMAGE_KEYS",
    "CANONICAL_STATE_KEY",
    "CANONICAL_PROMPT_KEY",
    # bindings (lazy)
    "Model",
    # processor steps
    "ProcessorStep",
    "Pipeline",
    "ParseImage",
    "ResizeWithPad",
    "PromptTokenizer",
    "Normalizer",
    "Unnormalizer",
    "GaussianNoise",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    # Re-export the compiled binding's Model under the single ``apxinf`` facade,
    # lazily — so ``import apxinf`` (processor / offline use) never imports
    # ``apxinf_py`` / touches CUDA. Only ``apxinf.Model`` access pulls it in.
    if name == "Model":
        from apxinf_py import Model

        return Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
