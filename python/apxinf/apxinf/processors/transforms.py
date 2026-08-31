"""dict→dict processor steps for a policy's pre/post :class:`Pipeline`.

openpi composes a policy as ``inputs transforms → model → outputs transforms``,
where each transform is a ``dict -> dict`` function over a shared data dict. This
module provides that transform layer for pi05, reusing the existing single-value
steps (:class:`~apxinf.processors.resize.ResizeWithPad`,
:class:`~apxinf.processors.tokenize.PromptTokenizer`, etc.) unchanged — each class
here is a thin ``ProcessorStep`` that reads a few keys from the data dict,
delegates to a wrapped natural-signature step, and writes its output key back.

Because a policy's pre/post chain is just a :class:`~apxinf.processors.base.Pipeline`
whose flowing value is the data dict, these steps compose, reorder, and override
through the same ``Pipeline`` machinery as the image sub-chain.

**Data-dict key contract**

* pre (input) chain — reads ``observation`` / ``prompt``; writes ``rgb`` (uint8
  NHWC), ``token_ids`` (uint32), ``noise``.
* post (output) chain — reads ``normalized_actions``; writes ``trimmed`` then
  ``actions`` (unnormalized float32).

Each step returns the *same* dict, updated in place — a pre/post chain is a linear
single-owner flow, so there is no aliasing hazard, and later steps see earlier
steps' keys.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from .base import ProcessorStep

__all__ = [
    "ImageStack",
    "Tokenize",
    "SampleNoise",
    "Trim",
    "Unnormalize",
    "lookup_key",
    "has_key",
    "set_key",
]

# Canonical data-dict keys (the inter-step contract).
OBSERVATION = "observation"
PROMPT = "prompt"
RGB = "rgb"
TOKEN_IDS = "token_ids"
NOISE = "noise"
NORMALIZED = "normalized_actions"
TRIMMED = "trimmed"
ACTIONS = "actions"


def _require(data: Mapping[str, Any], key: str, step: str) -> Any:
    try:
        return data[key]
    except KeyError:
        raise KeyError(
            f"{step}: missing data key {key!r}; present keys: {sorted(data)}"
        ) from None


_ABSENT = object()
_NO_DEFAULT = object()


def lookup_key(observation: Mapping[str, Any], key: str, default: Any = _NO_DEFAULT) -> Any:
    """Resolve one **wire key** against a raw observation: flat first, then nested.

    OpenPI embodiments spell their wire keys two different ways, and both arrive
    here as one string:

    * **flat**, with the slash part of the name itself — ``data["observation/image"]``
      (LIBERO, DROID);
    * **nested** one level — ``data["images"]["cam_high"]`` (ALOHA, Unitree G1).

    A flat hit always wins, so a key that literally exists in the observation
    keeps its exact meaning; only an absent key is split on ``/`` and walked as a
    path. That makes ``"images/cam_high"`` a valid spelling of the nested form
    without changing what ``"observation/image"`` means, so one flat
    ``image_keys`` tuple can address either layout.

    Raises :class:`KeyError` when the key resolves to nothing and no ``default``
    is given.
    """
    if key in observation:
        return observation[key]
    if "/" in key:
        node: Any = observation
        for part in key.split("/"):
            if not isinstance(node, Mapping) or part not in node:
                break
            node = node[part]
        else:
            return node
    if default is _NO_DEFAULT:
        raise KeyError(key)
    return default


def has_key(observation: Mapping[str, Any], key: str) -> bool:
    """Whether :func:`lookup_key` can resolve ``key`` (flat or nested)."""
    return lookup_key(observation, key, _ABSENT) is not _ABSENT


def set_key(observation: Mapping[str, Any], key: str, value: Any) -> dict:
    """Return a copy of ``observation`` with ``key`` set, mirroring :func:`lookup_key`.

    Writes flat when the key already exists flat (or has no ``/``); otherwise
    walks the nested path, shallow-copying only the mappings along it. The
    caller's dict is never mutated — input steps hand the decoded observation
    downstream while the client's original stays untouched.
    """
    if key in observation or "/" not in key:
        return {**observation, key: value}
    head, *rest = key.split("/")
    node = observation.get(head)
    if isinstance(node, Mapping):
        return {**observation, head: set_key(node, "/".join(rest), value)}
    return {**observation, key: value}


class ImageStack(ProcessorStep):
    """Stack per-view images into one uint8 NHWC array under ``rgb``.

    Runs the single-value ``image_pipeline`` (parse + resize) on each configured
    camera key and stacks the **real** views — one row per ``image_keys`` entry,
    in order. No slot padding: the model runs the exact shape it is handed, so
    the caller must supply precisely the cameras the checkpoint expects (absent
    cameras are simply not sent, never zero-filled).

    ``image_keys`` order is what binds a camera to a model view slot: entry ``i``
    becomes ``rgb[i]``, which the checkpoint trained as its slot ``i`` (openpi's
    ``base_0_rgb`` / ``left_wrist_0_rgb`` / ``right_wrist_0_rgb``). A wrong order
    stacks cleanly and silently feeds the wrong camera to each slot, so build the
    tuple from a slot-named preset (:mod:`apxinf.robots.presets`) rather than by
    hand. Each key is resolved by :func:`lookup_key`, so flat
    (``"observation/image"``) and nested (``"images/cam_high"``) wire layouts both
    work.
    """

    def __init__(
        self,
        image_pipeline,
        image_keys: Sequence[str],
        image_size: int,
        *,
        observation_key: str = OBSERVATION,
    ):
        self.image_pipeline = image_pipeline
        self.image_keys = tuple(image_keys)
        self.image_size = int(image_size)
        self.observation_key = observation_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        observation = _require(data, self.observation_key, "ImageStack")
        views = []
        for key in self.image_keys:
            try:
                image = lookup_key(observation, key)
            except KeyError:
                raise KeyError(
                    f"ImageStack: observation has no camera {key!r}; configured "
                    f"image_keys={list(self.image_keys)}, observation keys="
                    f"{sorted(observation)}"
                ) from None
            views.append(self.image_pipeline(image))
        data[RGB] = np.ascontiguousarray(np.stack(views), dtype=np.uint8)
        return data


class Tokenize(ProcessorStep):
    """Tokenize the prompt (optionally injecting discretized state) into ``token_ids``.

    Mirrors the policy's old state routing: when the tokenizer runs in
    ``discrete_state`` mode and a ``state_normalizer`` is set, the raw state is
    first mapped to ``[-1, 1]`` before discretization; otherwise state is dropped.

    ``state_key`` has no default. It names a *dataset's* wire key, and this layer
    has no business guessing one: ``"observation/state"`` used to be the default,
    which is LIBERO's dialect quietly applied to every robot. ``None`` is allowed
    only when the tokenizer does not read state at all, and is rejected when it
    does — dropping proprioception silently is the failure this guards.
    """

    def __init__(
        self,
        tokenizer,
        state_normalizer=None,
        state_key: Optional[str] = None,
        *,
        observation_key: str = OBSERVATION,
        prompt_key: str = PROMPT,
    ):
        if state_key is None and getattr(tokenizer, "discrete_state", False):
            raise ValueError(
                "Tokenize: the tokenizer discretizes state into the prompt but no "
                "state_key was given, so there is no key to read it from. Name the "
                "wire key your client sends (see apxinf.conventions), or use a "
                "tokenizer with discrete_state=False to drop state deliberately."
            )
        self.tokenizer = tokenizer
        self.state_normalizer = state_normalizer
        self.state_key = state_key
        self.observation_key = observation_key
        self.prompt_key = prompt_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        prompt = _require(data, self.prompt_key, "Tokenize")
        if getattr(self.tokenizer, "discrete_state", False):
            observation = _require(data, self.observation_key, "Tokenize")
            state = lookup_key(observation, self.state_key, None)
            if self.state_normalizer is not None and state is not None:
                # No dtype coercion: the state only ever reaches the model as
                # prompt tokens, and openpi normalizes/discretizes it in whatever
                # dtype its own transforms produced (float64 for adapt_to_pi
                # robots). Forcing float32 here shifts values by ~1e-7, which is
                # enough to land on the other side of a discretization bin edge.
                state = self.state_normalizer(np.asarray(state))
            data[TOKEN_IDS] = self.tokenizer(prompt, state=state)
        else:
            data[TOKEN_IDS] = self.tokenizer(prompt)
        return data


class SampleNoise(ProcessorStep):
    """Draw the flow-matching prior noise into ``noise``."""

    def __init__(self, noise):
        self.noise = noise

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        data[NOISE] = self.noise()
        return data


class Trim(ProcessorStep):
    """Trim the model's normalized action to the deployable width, under ``trimmed``."""

    PARAMS = ("action_dim",)

    def __init__(self, action_dim: int):
        self.action_dim = int(action_dim)

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        normalized = _require(data, NORMALIZED, "Trim")
        data[TRIMMED] = np.ascontiguousarray(normalized[:, : self.action_dim])
        return data


class Unnormalize(ProcessorStep):
    """Unnormalize ``trimmed`` (or ``normalized_actions``) into ``actions``.

    Delegates to a wrapped :class:`~apxinf.processors.normalize.Unnormalizer`. Reads
    ``trimmed`` when present (the usual post chain ``Trim -> Unnormalize``), else
    falls back to ``normalized_actions`` so the step is usable standalone.
    """

    def __init__(self, unnormalizer):
        self.unnormalizer = unnormalizer

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        array = data.get(TRIMMED)
        if array is None:
            array = _require(data, NORMALIZED, "Unnormalize")
        data[ACTIONS] = self.unnormalizer(array)
        return data
