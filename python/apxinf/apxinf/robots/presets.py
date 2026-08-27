"""Robot presets: the wire contract of one embodiment, as a named table entry.

OpenPI keeps this information in a Python registry of ``TrainConfig``s: which
``DataTransformFn`` pair runs, and therefore which **wire keys** the client must
send. ``serve_policy.py --policy.config pi05_UnitreeG1_...`` selects an entry;
the client cannot negotiate it and must match by hand.

This module is the same idea with the same shape, so switching embodiments is a
launch flag on both sides rather than a code edit on ours:

    openpi:  serve_policy.py --policy.config pi05_UnitreeG1_groundwire
    apxinf:  pi05_openpi_websocket_server.py --robot unitree_g1

**Why a table and not a default.** ``Pi05Policy``'s ``_DEFAULT_IMAGE_KEYS`` is
LIBERO's wire contract. As *the* default it silently applied to every checkpoint,
so a G1 checkpoint served without extra arguments ran LIBERO's keys, dropped
state, and skipped the G1 delta→absolute and 32→16 steps — every symptom of a
"model accuracy problem" with nothing in the logs. A preset makes the embodiment
an explicit, named choice.

**Why slot names.** ``image_keys`` is order-significant: entry ``i`` is stacked
into model view slot ``i``, which the checkpoint trained as openpi's
``base_0_rgb`` / ``left_wrist_0_rgb`` / ``right_wrist_0_rgb``. A tuple written in
the wrong order still stacks, still has the right shape, and silently feeds the
wrong camera to each slot. Pairing every wire key with the slot it fills makes
that order reviewable instead of positional.

**Naming rule: ``<arm>_<convention>``.** The arm alone does not determine the
contract — LIBERO and DROID are both Franka Panda, yet LIBERO sends
``observation/image`` with a 7-dim EEF-delta action while DROID sends
``observation/exterior_image_1_left`` with a different action space. So a preset
is named for the arm *and* the dataset convention whose keys it implements:
``franka_libero``, not ``libero`` (a benchmark, not a robot) and not ``franka``
(ambiguous). Single-embodiment robots that own their convention need no suffix —
``unitree_g1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from ..policies.auto import AutoPolicy
from ..policies.base import Policy
from ..processors.robots.unitree_g1 import G1_CAMERAS, G1_STATE_KEY
from .unitree_g1 import build_unitree_g1_policy

__all__ = [
    "VIEW_SLOTS",
    "RobotPreset",
    "ROBOT_PRESETS",
    "ROBOT_ALIASES",
    "available_robots",
    "get_robot_preset",
    "build_robot_policy",
]

#: pi05 model view slots in order, as named by openpi's ``model.IMAGE_KEYS``.
#: A checkpoint's ``num_views`` is how many of these its weights were trained on;
#: the names are openpi's convention, the *order* is baked into the weights.
VIEW_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _build_generic(model_dir, **kwargs) -> Policy:
    """Default builder: the stock policy, no robot-specific pre/post steps."""
    return AutoPolicy.from_pretrained(model_dir, **kwargs)


@dataclass(frozen=True)
class RobotPreset:
    """One embodiment's serving contract: wire keys + action width + builder."""

    #: Registry name, used as ``--robot <name>``.
    name: str
    #: ``(view slot, wire key)`` pairs in model slot order. The slot name is
    #: documentation and validation; only the wire key goes on the network.
    slots: Tuple[Tuple[str, str], ...]
    #: Wire key of the proprioceptive state vector.
    state_key: str
    #: Wire key of the task instruction.
    prompt_key: str = "prompt"
    #: Deployable action width. ``None`` keeps the model's full vector — correct
    #: when a robot output step does the truncation itself (G1's 32→16 encode).
    action_dim: Optional[int] = None
    #: Whether state is discretized into the prompt. Off means state is *dropped*.
    discrete_state: bool = False
    #: Factory that loads the checkpoint and wires this robot's pre/post steps.
    builder: Callable[..., Policy] = _build_generic
    #: One-line description for ``--help`` and the served metadata.
    summary: str = ""
    #: Extra keyword arguments the builder always receives.
    builder_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError(f"RobotPreset {self.name!r}: at least one camera slot is required")
        if len(self.slots) > len(VIEW_SLOTS):
            raise ValueError(
                f"RobotPreset {self.name!r}: {len(self.slots)} slots exceeds pi05's "
                f"{len(VIEW_SLOTS)} view slots {VIEW_SLOTS}"
            )
        expected = VIEW_SLOTS[: len(self.slots)]
        declared = tuple(slot for slot, _ in self.slots)
        if declared != expected:
            raise ValueError(
                f"RobotPreset {self.name!r}: slots must be a prefix of {VIEW_SLOTS} in "
                f"order (a checkpoint fills view slots from 0 up), got {declared}"
            )
        wire = [key for _, key in self.slots]
        if len(set(wire)) != len(wire):
            raise ValueError(f"RobotPreset {self.name!r}: duplicate camera wire keys {wire}")

    @property
    def image_keys(self) -> Tuple[str, ...]:
        """Camera wire keys in model slot order — what ``ImageStack`` consumes."""
        return tuple(key for _, key in self.slots)

    @property
    def num_views(self) -> int:
        """Cameras this preset sends; must equal the checkpoint's ``num_views``."""
        return len(self.slots)

    def describe(self) -> str:
        """Human-readable slot→wire-key mapping, for ``--help`` and startup logs."""
        rendered = ", ".join(f"{slot}={key}" for slot, key in self.slots)
        return f"{self.name}: {rendered}; state={self.state_key}"


#: Franka Panda under LIBERO's key convention (the 7-DoF sim benchmark):
#: 2 cameras, 7-dim action. Mirrors openpi ``LiberoInputs``. State is dropped by
#: default, matching the numerics of the existing serving link.
FRANKA_LIBERO = RobotPreset(
    name="franka_libero",
    slots=(
        ("base_0_rgb", "observation/image"),
        ("left_wrist_0_rgb", "observation/wrist_image"),
    ),
    state_key="observation/state",
    action_dim=7,
    discrete_state=False,
    summary="Franka Panda, LIBERO keys: 2 cameras, 7-dim action (6 EEF deltas + gripper)",
)

#: Unitree G1 (humanoid, dual-arm + 2 dexterous hands): 3 cameras, 16-dim action.
#: Mirrors the ``pi05_UnitreeG1`` fine-tune's ``unitreeG1Inputs``/``Outputs``: the
#: client sends the nested ``obs["images"][...]`` layout and a flat ``"state"``.
#: ``action_dim`` stays ``None`` because ``UnitreeG1EncodeActions`` does the 32→16
#: truncation after delta→absolute has seen the full-width action.
UNITREE_G1 = RobotPreset(
    name="unitree_g1",
    slots=tuple(zip(VIEW_SLOTS, G1_CAMERAS)),
    state_key=G1_STATE_KEY,
    action_dim=None,
    discrete_state=True,
    builder=build_unitree_g1_policy,
    summary="Unitree G1: 3 cameras, 16-DoF state, delta joint actions, 32->16 encode",
    builder_kwargs={"use_delta_joint_actions": True, "adapt_to_pi": True},
)

#: Name -> preset. Add an embodiment here once its steps and factory exist; that
#: is the whole registration step (openpi's ``training/config.py`` equivalent).
ROBOT_PRESETS: Dict[str, RobotPreset] = {p.name: p for p in (FRANKA_LIBERO, UNITREE_G1)}

#: Accepted spellings that are not canonical names. ``libero`` names a benchmark
#: rather than a robot, but it is what the deployed launch commands and docs say,
#: so it keeps resolving instead of failing a running deployment on a rename.
ROBOT_ALIASES: Dict[str, str] = {"libero": "franka_libero"}


def available_robots(*, include_aliases: bool = False) -> Tuple[str, ...]:
    """Registered preset names, for ``--robot`` choices and error messages."""
    names = tuple(ROBOT_PRESETS)
    return names + tuple(ROBOT_ALIASES) if include_aliases else names


def get_robot_preset(name: str) -> RobotPreset:
    """Look up a preset by canonical name or alias, with a listing in the error."""
    canonical = ROBOT_ALIASES.get(name, name)
    try:
        return ROBOT_PRESETS[canonical]
    except KeyError:
        raise KeyError(
            f"unknown robot preset {name!r}; known: {list(available_robots())}"
            f" (aliases: {list(ROBOT_ALIASES)})"
        ) from None


def build_robot_policy(
    robot: str,
    model_dir,
    *,
    image_keys: Optional[Sequence[str]] = None,
    state_key: Optional[str] = None,
    action_dim: Optional[int] = None,
    discrete_state: Optional[bool] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Policy:
    """Load ``model_dir`` under the named preset, with per-argument overrides.

    Each override defaults to the preset's value; passing one replaces just that
    field. ``image_keys`` and ``state_key`` exist because a deployed client may
    already speak a fixed dialect — they let a server match an installed robot
    stack without editing the preset (and without touching the client).

    The resulting wire contract is published in the policy ``metadata``
    (``robot`` / ``image_keys`` / ``state_key`` / ``discrete_state``), which the
    server pushes on connect, so a client can assert it rather than guess.
    """
    preset = get_robot_preset(robot)
    keys = tuple(image_keys) if image_keys is not None else preset.image_keys
    state = state_key if state_key is not None else preset.state_key
    discrete = preset.discrete_state if discrete_state is None else bool(discrete_state)
    width = preset.action_dim if action_dim is None else action_dim

    return preset.builder(
        model_dir,
        image_keys=keys,
        state_key=state,
        discrete_state=discrete,
        action_dim=width,
        metadata={
            "robot": preset.name,
            "robot_slots": [list(pair) for pair in zip(VIEW_SLOTS, keys)],
            **(dict(metadata) if metadata else {}),
        },
        **{**dict(preset.builder_kwargs), **kwargs},
    )
