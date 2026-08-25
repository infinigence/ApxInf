"""ApxInf Python frontend.

Three layers, kept deliberately decoupled:

* :mod:`apxinf.processors` — pure-numpy pre/post-processing *steps* (resize,
  tokenize, normalize, noise) plus a :class:`~apxinf.processors.Pipeline`
  container. Each step is independently instantiable and callable on its natural
  input, with no GPU / no Rust dependency, so it unit-tests offline.
  Robot-specific steps (varying by robot body, not by model) live under
  :mod:`apxinf.processors.robots`.
* **policies** — the **L2** layer (:mod:`apxinf.policies`).
  :class:`~apxinf.policies.impls.pi05.Pi05Policy` composes a pre pipeline + a
  bare-model handle (L1) + a post pipeline into a single
  ``infer(obs_dict, noise=None) -> {actions, timing, ...}`` call.
  :class:`~apxinf.policies.auto.AutoPolicy` dispatches a checkpoint to its concrete
  policy by ``config.json`` model type; :class:`~apxinf.policies.base.Policy` is
  the structural contract they all satisfy.
* **robots** — the assembly layer (:mod:`apxinf.robots`). A ``build_*`` factory
  binds one robot to a model policy by wiring its
  :mod:`apxinf.processors.robots` steps into the policy pipelines. Depends
  downward on both ``policies`` and ``processors``; neither depends back.
* **bindings** — :class:`Model` re-exports the ``apxinf_py`` PyO3 handle (L1
  bare-model inference; an internal L0 patches path exists but is private). It is
  the single public surface; you never import ``apxinf_py`` directly.
* :mod:`apxinf.serving` — the websocket policy server (a thin, model-agnostic
  transport shell over any :class:`Policy`, with an openpi-compatible wire).
  Imported only on demand (``from apxinf.serving import WebsocketPolicyServer``)
  so its ``msgpack`` / ``websockets`` deps stay out of offline processor use.

``import apxinf`` never touches CUDA: only ``apxinf.Model`` (accessed lazily) and a
policy's ``from_pretrained`` pull in the ``apxinf_py`` binding.
"""

from __future__ import annotations

from . import processors
from .policies import AutoPolicy, Pi05Policy, Policy
from .robots import (
    ROBOT_PRESETS,
    RobotPreset,
    available_robots,
    build_robot_policy,
    build_unitree_g1_policy,
    get_robot_preset,
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
    # L2 policies
    "Pi05Policy",
    "AutoPolicy",
    # robot adapters
    "build_unitree_g1_policy",
    # robot presets (embodiment -> wire keys + pipelines), openpi's TrainConfig analogue
    "RobotPreset",
    "ROBOT_PRESETS",
    "available_robots",
    "get_robot_preset",
    "build_robot_policy",
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
