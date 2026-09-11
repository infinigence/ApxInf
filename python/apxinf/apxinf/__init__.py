"""ApxInf Python frontend.

Public modules:

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
* :mod:`apxinf.conventions` — dataset **recording dialects**: the camera and
  state wire keys a client sends, and whether state was recorded into the prompt.
  A dialect is a property of the data, not of the arm or the weights, so it
  depends on neither and both can change under it.
* **robots** — the assembly layer (:mod:`apxinf.robots`). A ``build_*`` factory
  binds one robot to a model policy by wrapping its
  :mod:`apxinf.processors.robots` steps *around* the policy's own chain, through
  :class:`~apxinf.policies.base.ComposablePolicy`. A
  :class:`~apxinf.robots.presets.RobotPreset` pairs one body with one convention
  and is the only deployable unit.
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

from . import conventions, processors
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
from .conventions import (
    Convention,
    available_conventions,
    get_convention,
    register_convention,
)
from .policies import AutoPolicy, ComposablePolicy, Pi05Policy, Pi0FastPolicy, Policy, WallossPolicy
from .robots import (
    ROBOT_PRESETS,
    Embodiment,
    RobotPreset,
    available_robots,
    build_robot_policy,
    build_unitree_g1_policy,
    get_robot_preset,
    register_robot_preset,
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
    "conventions",
    # policy contract (outward); BareModel (inward) lives in apxinf.policies
    "Policy",
    "ComposablePolicy",
    # L2 policies
    "Pi05Policy",
    "Pi0FastPolicy",
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
    # robot adapters
    "build_unitree_g1_policy",
    # dataset dialects (wire keys + state routing), independent of any body
    "Convention",
    "available_conventions",
    "get_convention",
    "register_convention",
    # robot presets (body x convention -> wire contract), openpi's TrainConfig analogue
    "Embodiment",
    "RobotPreset",
    "ROBOT_PRESETS",
    "available_robots",
    "get_robot_preset",
    "register_robot_preset",
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
