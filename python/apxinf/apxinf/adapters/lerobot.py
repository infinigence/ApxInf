"""Drive an :class:`apxinf.Policy` from a lerobot-shaped control loop.

:class:`ApxInfPolicy` stands in for a lerobot policy object (``PI0Policy`` &c.) in
a **hand-written control loop**, so a lerobot user keeps their robot, cameras,
dataset-feature plumbing and action dispatch and swaps only the policy::

    - from lerobot.policies.pi0 import PI0Policy
    - from lerobot.policies import make_pre_post_processors
    - model = PI0Policy.from_pretrained(model_id)
    - preprocess, postprocess = make_pre_post_processors(model.config, model_id)
    + from apxinf.adapters.lerobot import ApxInfPolicy
    + model = ApxInfPolicy.from_pretrained(ckpt_dir, device="cuda:0", precision="bf16")
    + preprocess, postprocess = model.make_pre_post_processors()

    - obs_frame = build_inference_frame(obs, ds_features=feats, device=device, task=task)
    + obs_frame = model.build_inference_frame(obs, ds_features=feats, task=task)

      obs    = preprocess(obs_frame)          # unchanged
      action = model.select_action(obs)       # unchanged
      action = postprocess(action)            # unchanged
      robot.send_action(make_robot_action(action, feats))   # unchanged

**Supported** — a user-constructed policy in a loop the user owns: lerobot
``robots`` / ``cameras`` / ``teleoperators``, ``hw_to_dataset_features``,
``build_dataset_frame``, ``make_robot_action``, ``LeRobotDataset`` recording,
hand-written gym eval loops, and chunk-at-a-time consumption via
:meth:`ApxInfPolicy.predict_action_chunk`.

**Not supported** — anything where lerobot *constructs* the policy from a string
or trains it:

* ``lerobot-eval`` / ``lerobot-rollout`` CLI, ``make_policy``,
  ``make_pre_post_processors(policy.config, ...)`` — these resolve a policy class
  out of the draccus ``ChoiceRegistry`` and build pre/post from a
  ``PreTrainedConfig``. Becoming reachable that way means shipping a
  ``lerobot_policy_*`` plugin distribution; see ``doc/`` for that trade-off.
* training / fine-tuning / PEFT (``forward``, ``loss.backward()``,
  ``lerobot-train``) — structurally impossible: the engine underneath is a Rust
  inference runtime with no autograd.
* RTC (``--inference.type=rtc``) and lerobot's async-inference policy server —
  new engine capabilities, not a wrapper concern (apxinf ships its own
  :mod:`apxinf.serving` websocket server).
* ``torch.compile`` / AMP autocast — meaningless here (the CUDA graph lives in
  Rust); accepted and ignored rather than honoured.

**Where the seam sits.** lerobot's ``build_inference_frame`` is two steps:
``build_dataset_frame`` (key regrouping, still numpy ``HWC`` ``uint8``) then
``prepare_observation_for_inference`` (H2D, ``/255``, ``permute`` to ``CHW``, batch
dim). :meth:`ApxInfPolicy.build_inference_frame` runs **only the first** and
renames keys, because that layer is exactly what :meth:`Policy.infer` already
eats. Passing a frame that went through the second step also works — see
:func:`observation_to_apxinf`, which undoes it — but it costs a device→host copy
per tick and a lossy ``*255`` round-trip, so the numpy seam is the default.

**Whose pre/post runs.** apxinf's own :class:`~apxinf.processors.Pipeline` does
*all* pre/post (resize, tokenize, noise, unnormalize) — it is what the checkpoint's
numerics are anchored to by the golden tests. lerobot splits the same work
differently (resize and prior-noise sampling live *inside* its policy, normalize
lives in its processor pipeline), so the pipelines from
``make_pre_post_processors`` are **not** interchangeable with ours; feeding their
output here would drop resize/noise and double-normalize. The
:meth:`make_pre_post_processors` on this class therefore returns near-identity
pipelines that preserve the call shape and nothing more.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ApxInfPolicy",
    "observation_to_apxinf",
    "IdentityProcessor",
    "LEROBOT_IMAGE_PREFIX",
    "LEROBOT_STATE_KEY",
    "LEROBOT_TASK_KEY",
]

# lerobot's flat dotted observation keys (``lerobot.utils.constants``).
LEROBOT_IMAGE_PREFIX = "observation.images."
LEROBOT_STATE_KEY = "observation.state"
LEROBOT_TASK_KEY = "task"

# apxinf's slash-separated keys (openpi-shaped; see apxinf.processors.transforms).
APXINF_STATE_KEY = "observation/state"
APXINF_PROMPT_KEY = "prompt"


def _require_torch():
    """Import torch on demand, with an actionable message when it is absent.

    Only the tensor boundary needs it, so ``import apxinf`` and the numpy-only
    processor tests stay torch-free.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "apxinf.adapters.lerobot needs PyTorch for its tensor boundary; "
            "install it with `pip install apxinf[lerobot]`."
        ) from exc
    return torch


def _lerobot_frame_builder():
    """Resolve lerobot's ``build_dataset_frame`` + ``OBS_STR`` across versions.

    The function itself has been stable, but its home has moved
    (``lerobot.datasets.utils`` in 0.4.x, ``lerobot.utils.feature_utils`` on
    later main), so try both rather than pinning one import path. Deliberately
    reused instead of reimplemented: it decides which observation keys are read
    and — via ``ft["names"]`` — the order the state vector is assembled in, which
    must stay bit-identical to the lerobot path the checkpoint was used with.
    """
    try:
        from lerobot.utils.constants import OBS_STR
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "apxinf.adapters.lerobot.build_inference_frame needs lerobot installed "
            "(it reuses lerobot's own build_dataset_frame). Install lerobot, or "
            "translate frames yourself with observation_to_apxinf()."
        ) from exc

    last_error: Exception | None = None
    for module_name in ("lerobot.utils.feature_utils", "lerobot.datasets.utils"):
        try:
            module = __import__(module_name, fromlist=["build_dataset_frame"])
            return module.build_dataset_frame, OBS_STR
        except (ImportError, AttributeError) as exc:  # pragma: no cover
            last_error = exc
    raise ImportError(
        "apxinf.adapters.lerobot: could not find lerobot's build_dataset_frame in "
        "lerobot.utils.feature_utils or lerobot.datasets.utils. Pass frames through "
        "observation_to_apxinf() instead, which needs no lerobot import."
    ) from last_error


class IdentityProcessor:
    """A pass-through stand-in for a lerobot ``PolicyProcessorPipeline``.

    Preserves the ``preprocess(frame)`` / ``postprocess(action)`` call shape of a
    lerobot loop while doing nothing: apxinf policies own their whole pre/post
    chain (see the module docstring). ``steps`` is empty and ``reset`` is a no-op,
    the two attributes lerobot loops touch on a pipeline.
    """

    steps: Tuple[Any, ...] = ()

    def __call__(self, value):
        return value

    def reset(self) -> None:
        return None


def observation_to_apxinf(
    frame: Mapping[str, Any],
    *,
    image_keys: Sequence[str],
    task: Optional[str] = None,
    state_key: str = LEROBOT_STATE_KEY,
    task_key: str = LEROBOT_TASK_KEY,
) -> Dict[str, Any]:
    """Translate one lerobot observation frame into an apxinf observation dict.

    Accepts a frame from either seam: raw numpy ``HWC`` ``uint8`` images (the
    output of lerobot's ``build_dataset_frame`` — the cheap path) *or* tensors that
    already went through ``prepare_observation_for_inference``, which are undone
    here (batch dim dropped, ``CHW -> HWC``, float ``[0, 1] -> uint8``, device ->
    host).

    ``image_keys`` are the apxinf camera keys **in the order the checkpoint
    expects**; each is matched to a frame key by its trailing camera name, so
    ``observation.images.base_0_rgb`` feeds ``observation/image`` when the caller
    maps them in order. ``task`` overrides the frame's ``task`` entry.
    """
    images = _ordered_frame_images(frame)
    if len(images) != len(image_keys):
        raise ValueError(
            f"lerobot adapter: policy expects {len(image_keys)} cameras "
            f"{tuple(image_keys)} but the frame carries {len(images)}: "
            f"{tuple(name for name, _ in images)}. Supply exactly the "
            f"checkpoint's cameras (real views only, no padding)."
        )

    observation: Dict[str, Any] = {
        apxinf_key: _to_hwc_uint8(value, source_key)
        for apxinf_key, (source_key, value) in zip(image_keys, images)
    }

    if state_key in frame:
        observation[APXINF_STATE_KEY] = _to_numpy(frame[state_key]).astype(
            np.float32, copy=False
        ).reshape(-1)

    prompt = task if task is not None else frame.get(task_key, "")
    if not isinstance(prompt, str):
        raise TypeError(
            f"lerobot adapter: task/prompt must be a string, got {type(prompt)!r}"
        )
    observation[APXINF_PROMPT_KEY] = prompt
    return observation


class ApxInfPolicy:
    """A lerobot-shaped facade over any :class:`apxinf.Policy`.

    Wraps an apxinf policy (``Pi05Policy`` today, any registered ``Policy``
    tomorrow) and exposes the four things a hand-written lerobot loop calls on a
    policy: :meth:`build_inference_frame`, :meth:`make_pre_post_processors`,
    :meth:`select_action` and :meth:`predict_action_chunk`. Actions come back as
    ``torch`` tensors with a leading batch dim, which is what lerobot's
    ``make_robot_action`` consumes.

    Like lerobot's chunking policies, :meth:`select_action` serves one step per
    call out of an internal queue and only re-runs the model when the queue drains.
    Wrap an existing policy directly, or build one from a checkpoint with
    :meth:`from_pretrained`.
    """

    def __init__(
        self,
        policy: Any,
        *,
        image_keys: Optional[Sequence[str]] = None,
        n_action_steps: Optional[int] = None,
        state_key: str = LEROBOT_STATE_KEY,
        task_key: str = LEROBOT_TASK_KEY,
    ):
        self.policy = policy
        self.image_keys = tuple(
            image_keys if image_keys is not None else getattr(policy, "image_keys", ())
        )
        if not self.image_keys:
            raise ValueError(
                "lerobot adapter: cannot infer the policy's camera keys; pass "
                "image_keys= in the order the checkpoint expects."
            )
        horizon = int(policy.action_horizon)
        self.n_action_steps = min(int(n_action_steps), horizon) if n_action_steps else horizon
        self.state_key = state_key
        self.task_key = task_key
        self._queue: deque = deque(maxlen=self.n_action_steps)

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        model_type: Optional[str] = None,
        n_action_steps: Optional[int] = None,
        state_key: str = LEROBOT_STATE_KEY,
        task_key: str = LEROBOT_TASK_KEY,
        **policy_kwargs: Any,
    ) -> "ApxInfPolicy":
        """Build the policy for ``model_dir`` and wrap it.

        Dispatches through :class:`~apxinf.policies.auto.AutoPolicy`, so the
        concrete policy comes from the checkpoint's ``config.json`` model type.
        Extra keyword arguments (``device`` / ``precision`` / ``action_dim`` /
        ``image_keys`` / ...) pass straight through to that policy's
        ``from_pretrained``.

        Unlike lerobot's ``from_pretrained``, this takes a **local checkpoint
        directory**, not a Hub repo id: apxinf checkpoints are engine-specific and
        are not published under lerobot's Hub layout.
        """
        from ..policies import AutoPolicy

        policy = AutoPolicy.from_pretrained(model_dir, model_type=model_type, **policy_kwargs)
        return cls(
            policy,
            n_action_steps=n_action_steps,
            state_key=state_key,
            task_key=task_key,
        )

    # --- lerobot loop surface ----------------------------------------------

    def build_inference_frame(
        self,
        observation: Mapping[str, Any],
        *,
        ds_features: Mapping[str, Mapping],
        task: Optional[str] = None,
        robot_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Raw robot observation -> apxinf observation dict (all numpy, host).

        The counterpart of lerobot's ``build_inference_frame``, stopping one step
        earlier: it runs lerobot's own ``build_dataset_frame`` (so ``ds_features``
        keeps deciding which keys are read and in which order the state vector is
        assembled — identical to the lerobot path) and then only renames keys. No
        tensor conversion, no host->device copy: apxinf's pipeline wants exactly
        this ``HWC`` ``uint8`` form.

        ``robot_type`` is accepted for call-shape compatibility and ignored — it
        exists in lerobot for multi-embodiment datasets, which the apxinf prompt
        does not carry.
        """
        build_dataset_frame, obs_prefix = _lerobot_frame_builder()

        frame = build_dataset_frame(dict(ds_features), dict(observation), prefix=obs_prefix)
        return observation_to_apxinf(
            frame,
            image_keys=self.image_keys,
            task=task,
            state_key=self.state_key,
            task_key=self.task_key,
        )

    def make_pre_post_processors(self, *args: Any, **kwargs: Any):
        """Return ``(preprocess, postprocess)`` pass-throughs for the loop shape.

        apxinf's policy owns its whole pre/post chain, so there is nothing left for
        a lerobot-side pipeline to do; these keep the two call sites in an existing
        loop valid. Arguments are accepted and ignored so a call copied from
        ``make_pre_post_processors(model.config, model_id)`` still works.
        """
        return IdentityProcessor(), IdentityProcessor()

    def predict_action_chunk(self, observation: Mapping[str, Any], **kwargs: Any):
        """Run one inference and return the whole chunk as ``[1, horizon, dim]``.

        This is the shape apxinf is native to (:meth:`Policy.infer` predicts a full
        chunk), so prefer it over :meth:`select_action` when the loop can consume a
        chunk: it does not re-run pre-processing per step.

        Accepts either an apxinf observation dict or a lerobot frame (converted via
        :func:`observation_to_apxinf`). Extra keyword arguments are rejected rather
        than silently dropped — they are how lerobot passes RTC state, which this
        adapter does not implement.
        """
        if kwargs:
            raise TypeError(
                f"lerobot adapter: predict_action_chunk got unsupported kwargs "
                f"{sorted(kwargs)}; RTC / prefix-action inference is not implemented."
            )
        torch = _require_torch()
        actions = self._infer_chunk(observation)
        return torch.from_numpy(np.ascontiguousarray(actions)).unsqueeze(0)

    def select_action(self, observation: Mapping[str, Any], **kwargs: Any):
        """Return one action step as ``[1, action_dim]``, queueing the chunk.

        Mirrors lerobot's chunking policies: the model runs only when the queue is
        empty, and each call pops the next step. Call :meth:`reset` (or
        :meth:`drop_queued_actions`) on an episode boundary or a task change so
        actions computed for the old context are not dispatched.
        """
        if kwargs:
            raise TypeError(
                f"lerobot adapter: select_action got unsupported kwargs "
                f"{sorted(kwargs)}; RTC / prefix-action inference is not implemented."
            )
        torch = _require_torch()
        if not self._queue:
            chunk = self._infer_chunk(observation)
            self._queue.extend(chunk[: self.n_action_steps])
        step = self._queue.popleft()
        return torch.from_numpy(np.ascontiguousarray(step)).unsqueeze(0)

    def drop_queued_actions(self) -> None:
        """Discard actions still queued from an earlier observation / task."""
        self._queue.clear()

    def reset(self) -> None:
        """Clear per-episode state. The apxinf policy itself is stateless."""
        self.drop_queued_actions()

    def eval(self) -> "ApxInfPolicy":
        """Accepted for call-shape compatibility; inference-only already."""
        return self

    def close(self) -> None:
        """Release the underlying model (see :meth:`apxinf.Policy.close`)."""
        self.policy.close()

    # --- introspection -----------------------------------------------------

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.policy.metadata

    @property
    def action_dim(self) -> int:
        return int(self.policy.action_dim)

    @property
    def action_horizon(self) -> int:
        return int(self.policy.action_horizon)

    def __repr__(self) -> str:
        return (
            f"ApxInfPolicy({type(self.policy).__name__}, "
            f"cameras={self.image_keys}, n_action_steps={self.n_action_steps})"
        )

    # --- helpers -----------------------------------------------------------

    def _infer_chunk(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Normalize the caller's dict to apxinf keys and run the policy."""
        if not isinstance(observation, Mapping):
            raise TypeError(
                f"lerobot adapter: observation must be a mapping, got {type(observation)!r}"
            )
        if APXINF_PROMPT_KEY not in observation:
            # A frame that skipped ``build_inference_frame`` (e.g. it came straight
            # from lerobot's own, tensor-producing one): convert it here.
            observation = observation_to_apxinf(
                observation,
                image_keys=self.image_keys,
                state_key=self.state_key,
                task_key=self.task_key,
            )
        result = self.policy.infer(observation)
        return np.asarray(result["actions"], dtype=np.float32)


def _ordered_frame_images(frame: Mapping[str, Any]):
    """The frame's camera entries as ``(key, value)``, in the frame's own order.

    Python dicts preserve insertion order and lerobot's ``build_dataset_frame``
    inserts cameras in ``ds_features`` order, so this is the caller's declared
    camera order — which is what must line up with the checkpoint's ``image_keys``.
    """
    return [
        (key, value)
        for key, value in frame.items()
        if key.startswith(LEROBOT_IMAGE_PREFIX)
    ]


def _to_numpy(value) -> np.ndarray:
    """Host numpy view of a numpy array or a (possibly GPU) torch tensor."""
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if detach is not None:  # torch tensor, without importing torch to find out
        return detach().to("cpu").numpy()
    return np.asarray(value)


def _to_hwc_uint8(value, source_key: str) -> np.ndarray:
    """Coerce one camera frame to ``HWC`` ``uint8``, undoing lerobot's tensor prep.

    Handles both seams: a raw ``HWC`` ``uint8`` frame passes through untouched,
    while a ``[1, C, H, W]`` float tensor (what
    ``prepare_observation_for_inference`` produces) is squeezed, transposed and
    scaled back.

    A float frame is taken to be in ``[0, 1]`` — lerobot's contract, since its
    conversion is a plain ``/255`` — and is range-checked rather than sniffed, so
    a frame that is really in ``[0, 255]`` fails loudly instead of silently
    saturating. The scale-back rounds before casting: ``k / 255`` is not exact in
    float32, so a truncating cast can land on ``k - 1``.
    """
    array = _to_numpy(value)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(
                f"lerobot adapter: {source_key} has batch size {array.shape[0]}; "
                f"only single-observation inference is supported."
            )
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            f"lerobot adapter: {source_key} must be a 3-D image (HWC or CHW), "
            f"got shape {array.shape}."
        )

    # Channel-first is how lerobot hands images to a policy; a channel count of
    # 1/3/4 in axis 0 that is not also a plausible channel count in axis 2
    # identifies it.
    if array.shape[0] in (1, 3, 4) and array.shape[2] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))

    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)

    scaled = array.astype(np.float32)
    peak = float(scaled.max()) if scaled.size else 0.0
    if not np.isfinite(peak) or peak > 1.0 + 1e-3:
        raise ValueError(
            f"lerobot adapter: float camera frame {source_key} peaks at {peak:.4g}, "
            f"outside the expected [0, 1] range (lerobot divides uint8 frames by "
            f"255). Pass uint8 HWC frames, or divide by 255 first."
        )
    return np.ascontiguousarray(np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8))
