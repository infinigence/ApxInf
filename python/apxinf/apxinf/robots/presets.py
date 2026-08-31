"""Robot presets: the wire contract of one embodiment, as a named table entry.

OpenPI keeps this information in a Python registry of ``TrainConfig``s: which
``DataTransformFn`` pair runs, and therefore which **wire keys** the client must
send. ``serve_policy.py --policy.config pi05_UnitreeG1_...`` selects an entry;
the client cannot negotiate it and must match by hand.

This module is the same idea with the same shape, so switching embodiments is a
launch flag on both sides rather than a code edit on ours:

    openpi:  serve_policy.py --policy.config pi05_UnitreeG1_groundwire
    apxinf:  pi05_openpi_websocket_server.py --robot unitree_g1

**Why a table and not a default.** ``Pi05Policy`` used to default ``image_keys``
to ``("observation/image", "observation/wrist_image")`` — LIBERO's wire contract,
living in the model layer as *the* default for every checkpoint. A G1 checkpoint
served without extra arguments therefore ran LIBERO's keys, dropped state, and
skipped the G1 delta→absolute and 32→16 steps — every symptom of a "model
accuracy problem" with nothing in the logs. The policy now names its cameras
after its own :data:`~apxinf.policies.base.VIEW_SLOTS` when the caller names
none, so no dataset's convention can pass for a model default; a preset makes the
embodiment an explicit, named choice.

**Why slot names.** ``image_keys`` is order-significant: entry ``i`` is stacked
into model view slot ``i``, which the checkpoint trained as openpi's
``base_0_rgb`` / ``left_wrist_0_rgb`` / ``right_wrist_0_rgb``. A tuple written in
the wrong order still stacks, still has the right shape, and silently feeds the
wrong camera to each slot. Pairing every wire key with the slot it fills makes
that order reviewable instead of positional. The slot vocabulary itself is a
*model* fact, so it is imported from :mod:`apxinf.policies.base` rather than
restated here.

**Naming rule: ``<arm>_<convention>``.** The arm alone does not determine the
contract — LIBERO and DROID are both Franka Panda, yet LIBERO sends
``observation/image`` with a 7-dim EEF-delta action while DROID sends
``observation/exterior_image_1_left`` with a different action space. So a preset
is named for the arm *and* the dataset convention whose keys it implements:
``franka_libero``, not ``libero`` (a benchmark, not a robot) and not ``franka``
(ambiguous). Single-embodiment robots that own their convention need no suffix —
``unitree_g1``.

**Two halves, one flag.** That naming rule is an admission, so the dataclasses
follow it: an :class:`Embodiment` is the body (camera count, action width, which
pre/post steps its actions need) and a
:class:`~apxinf.conventions.Convention` is a dataset's recording dialect (the
wire keys, and whether state was recorded into the prompt). Neither knows the
other; they vary independently, which is the point — the same Franka under
DROID's keys is a second convention, not a second body, and re-recording the G1
changes no :class:`Embodiment` field. Conventions live in
:mod:`apxinf.conventions` rather than here, because a dialect is not a robot's
property: this module imports them, never the reverse.

A :class:`RobotPreset` names one *pairing*, and only the pairing is deployable.
``--robot`` stays a single flag over that registry rather than becoming
``--robot`` + ``--convention``: separate flags would let an operator spell a
combination nobody ever recorded, and a body served under a convention it was not
recorded on is exactly the silent mismatch this whole mechanism exists to
prevent. The pairing is validated when the module loads — a convention naming
three cameras cannot be attached to a two-camera body.

**Registering from outside.** :func:`register_robot_preset` adds an entry from
another package, so a third-party robot does not have to patch this file to
become ``--robot <name>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .. import conventions
from ..conventions import Convention
from ..policies.auto import AutoPolicy
from ..policies.base import VIEW_SLOTS, Policy
from ..processors.robots.unitree_g1 import G1_ROBOT_DIM
from .unitree_g1 import build_unitree_g1_policy

__all__ = [
    "VIEW_SLOTS",
    "Embodiment",
    "Convention",
    "RobotPreset",
    "ROBOT_PRESETS",
    "ROBOT_ALIASES",
    "available_robots",
    "get_robot_preset",
    "register_robot_preset",
    "build_robot_policy",
]


def _build_generic(model_dir, **kwargs) -> Policy:
    """Default builder: the stock policy, no robot-specific pre/post steps."""
    return AutoPolicy.from_pretrained(model_dir, **kwargs)


@dataclass(frozen=True)
class Embodiment:
    """A robot body: how many cameras it has, and what its actions mean.

    Everything here is a fact about hardware. It survives a change of dataset:
    re-record the G1 under a different key convention and this is unchanged.
    Nothing here is a string that goes on the network.
    """

    #: Robot name, used in the served metadata and in preset names.
    name: str
    #: Cameras this body carries; must equal the checkpoint's ``num_views``.
    num_cameras: int
    #: Deployable action width. ``None`` keeps the model's full vector — correct
    #: when a robot output step does the truncation itself (G1's 32→16 encode).
    action_dim: Optional[int] = None
    #: Width of this body's state vector, i.e. how wide ``norm_stats["state"]``
    #: has to be. A fact about the hardware, so it does not follow
    #: :attr:`action_dim`: a robot's state and action spaces need not match
    #: (Franka under LIBERO has 8-dim state and a 7-dim EEF-delta action).
    #: ``None`` means this body makes no claim and the check is skipped.
    state_dim: Optional[int] = None
    #: Width of this body's action vector, i.e. how wide ``norm_stats["actions"]``
    #: has to be. Distinct from :attr:`action_dim`, which is a *loading* knob
    #: saying what to trim the model down to: the G1 declares ``action_dim=None``
    #: because ``UnitreeG1EncodeActions`` owns the truncation, but its actions are
    #: still 16 wide and its statistics must be. ``None`` skips the check.
    action_width: Optional[int] = None
    #: Factory that loads the checkpoint and wires this robot's pre/post steps.
    builder: Callable[..., Policy] = _build_generic
    #: Extra keyword arguments the builder always receives.
    builder_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 1 <= self.num_cameras <= len(VIEW_SLOTS):
            raise ValueError(
                f"Embodiment {self.name!r}: num_cameras={self.num_cameras} is outside "
                f"1..{len(VIEW_SLOTS)}, the view slots {VIEW_SLOTS} a checkpoint fills"
            )
        if (
            self.action_dim is not None
            and self.action_width is not None
            and self.action_dim > self.action_width
        ):
            # Trimming is a slice of the statistics, so it cannot ask for more
            # columns than the body's actions have. Getting this pair backwards
            # would make the preflight check the wrong width and pass a
            # checkpoint that the unnormalizer then fails on, mid-serve.
            raise ValueError(
                f"Embodiment {self.name!r}: action_dim={self.action_dim} trims wider "
                f"than action_width={self.action_width}, but action_width is how many "
                "columns this body's actions (and its norm_stats) have"
            )

    @property
    def has_robot_steps(self) -> bool:
        """Whether this body needs pre/post arithmetic beyond naming its keys.

        ``False`` for a body the generic builder serves, whose entire contract is
        its wire keys and action width. A caller that *cannot* run :attr:`builder`
        — the checkpoint-free synthetic path — uses this to say what it dropped
        rather than publishing the preset name unqualified.
        """
        return self.builder is not _build_generic


@dataclass(frozen=True)
class RobotPreset:
    """One deployable pairing: a body recorded under a convention.

    The two halves are separable because they vary independently — the same
    Franka arm under LIBERO's and DROID's keys is one :class:`Embodiment` and two
    :class:`Convention`\\ s — but they are validated *together*, because only the
    pair is deployable: a convention naming three cameras cannot serve a two-camera
    body, and nothing else in the system will notice if it tries.

    This stays one ``--robot`` flag. Splitting the flag too would let an operator
    spell combinations that were never recorded, which is the silent mismatch the
    flag exists to prevent; the registry below decides which pairings are real.
    """

    #: Registry name, used as ``--robot <name>``. Spelled ``<arm>_<convention>``
    #: when the arm is shared, bare when the body owns its convention.
    name: str
    #: The body being served.
    embodiment: Embodiment
    #: The key convention its client speaks.
    convention: Convention
    #: One-line description for ``--help`` and the served metadata.
    summary: str = ""

    def __post_init__(self) -> None:
        if self.convention.num_cameras != self.embodiment.num_cameras:
            raise ValueError(
                f"RobotPreset {self.name!r}: convention {self.convention.name!r} names "
                f"{self.convention.num_cameras} cameras but embodiment "
                f"{self.embodiment.name!r} has {self.embodiment.num_cameras}; a "
                "convention can only be paired with a body it was recorded on"
            )

    # --- the pairing's flat contract ---------------------------------------
    #
    # Delegating properties rather than a rename: this is what the server, the
    # metadata and the tests read, and which half a field came from is not their
    # business. They also keep every existing caller working unchanged.

    @property
    def image_keys(self) -> Tuple[str, ...]:
        """Camera wire keys in model slot order — what ``ImageStack`` consumes."""
        return self.convention.image_keys

    @property
    def state_key(self) -> str:
        return self.convention.state_key

    @property
    def prompt_key(self) -> str:
        return self.convention.prompt_key

    @property
    def discrete_state(self) -> bool:
        return self.convention.discrete_state

    @property
    def slots(self) -> Tuple[Tuple[str, str], ...]:
        """``(view slot, wire key)`` pairs in model slot order."""
        return self.convention.slots

    @property
    def action_dim(self) -> Optional[int]:
        return self.embodiment.action_dim

    @property
    def state_dim(self) -> Optional[int]:
        return self.embodiment.state_dim

    @property
    def action_width(self) -> Optional[int]:
        return self.embodiment.action_width

    @property
    def builder(self) -> Callable[..., Policy]:
        return self.embodiment.builder

    @property
    def builder_kwargs(self) -> Mapping[str, Any]:
        return self.embodiment.builder_kwargs

    @property
    def num_views(self) -> int:
        """Cameras this preset sends; must equal the checkpoint's ``num_views``."""
        return self.embodiment.num_cameras

    @property
    def has_robot_steps(self) -> bool:
        """Whether this preset wires robot-specific pre/post steps."""
        return self.embodiment.has_robot_steps

    def synthetic_gaps(
        self, *, discrete_state: bool, served_action_dim: int
    ) -> Tuple[str, ...]:
        """What a checkpoint-free (random-weights) server cannot honour here.

        The synthetic path reproduces this preset's wire keys and view count
        exactly and nothing else: its tokenizer emits a fixed token stream and
        never reads state, and :attr:`builder` never runs. Each returned string
        names one gap, for a startup warning — a synthetic server publishing this
        preset's name unqualified would be the silent embodiment mismatch
        ``--robot`` exists to prevent.
        """
        gaps = []
        if discrete_state:
            gaps.append(
                "discrete_state=True — the synthetic tokenizer ignores state, so the "
                "served metadata says discrete_state=False"
            )
        if self.has_robot_steps:
            gap = (
                f"its robot pre/post steps — {self.builder.__name__} never runs, so no "
                "state decode, delta->absolute or action re-encode is wired"
            )
            if self.action_dim is None:
                # action_dim=None means an output step owns the truncation (G1's
                # 32->16 encode). Without that step the full model vector ships.
                gap += (
                    f"; the served action_dim is the model's full {served_action_dim}, "
                    "not the width that skipped step would emit"
                )
            gaps.append(gap)
        return tuple(gaps)

    def describe(self) -> str:
        """Human-readable slot→wire-key mapping, for ``--help`` and startup logs."""
        rendered = ", ".join(f"{slot}={key}" for slot, key in self.slots)
        return f"{self.name}: {rendered}; state={self.state_key}"


#: --- bodies -----------------------------------------------------------------

#: Franka Emika Panda, 7-DoF arm + parallel gripper, 2-camera rig. Its whole port
#: is a table row: the checkpoint already emits absolute actions at the deployable
#: width, so no robot pre/post step is needed and the generic builder serves it.
#: ``state_dim`` is 8 rather than 7 — LIBERO records EEF pose + gripper on the
#: state side and 6 EEF deltas + gripper on the action side.
FRANKA = Embodiment(name="franka", num_cameras=2, action_dim=7, state_dim=8, action_width=7)

#: Unitree G1 humanoid: dual-arm + 2 dexterous hands, 16 DoF laid out
#: ``[L-arm 7, L-gripper 1, R-arm 7, R-gripper 1]``, 3 cameras. ``action_dim``
#: stays ``None`` because ``UnitreeG1EncodeActions`` does the 32→16 truncation
#: after delta→absolute has seen the full-width action; ``action_width`` is still
#: 16, because that is how wide the body's actions and its statistics are.
UNITREE_G1_BODY = Embodiment(
    name="unitree_g1",
    num_cameras=3,
    action_dim=None,
    state_dim=G1_ROBOT_DIM,
    action_width=G1_ROBOT_DIM,
    builder=build_unitree_g1_policy,
    builder_kwargs={
        "use_delta_joint_actions": True,
        "adapt_to_pi": True,
        # openpi parses norm_stats.json into float64 and never demotes it, so its
        # whole G1 chain — normalize state, discretize it into the prompt,
        # unnormalize the action — runs in float64. Ours follows the input dtype
        # (float32) unless told otherwise. The output-side gap is ~2e-7 rad and
        # harmless, but on the input side the very next thing after normalizing
        # is a comparison against bin edges 1/128 apart: an element sitting near
        # an edge lands in a different bin, which changes the prompt text, the
        # token ids, and the whole rollout. Match openpi rather than explain the
        # divergence later.
        "norm_dtype": "float64",
    },
)

#: --- deployable pairings -----------------------------------------------------

#: Franka Panda under LIBERO's keys: 2 cameras, 7-dim action.
FRANKA_LIBERO = RobotPreset(
    name="franka_libero",
    embodiment=FRANKA,
    convention=conventions.LIBERO,
    summary="Franka Panda, LIBERO keys: 2 cameras, 7-dim action (6 EEF deltas + gripper)",
)

#: Unitree G1 under its own convention — a single-embodiment robot that owns its
#: keys, so body and convention share a name and the preset needs no suffix.
UNITREE_G1 = RobotPreset(
    name="unitree_g1",
    embodiment=UNITREE_G1_BODY,
    convention=conventions.UNITREE_G1,
    summary="Unitree G1: 3 cameras, 16-DoF state, delta joint actions, 32->16 encode",
)

#: Name -> preset. Add an embodiment here once its steps and factory exist; that
#: is the whole registration step (openpi's ``training/config.py`` equivalent).
#: A preset defined outside this package joins the same dict through
#: :func:`register_robot_preset`.
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


def register_robot_preset(
    preset: RobotPreset,
    *,
    aliases: Iterable[str] = (),
    replace: bool = False,
) -> RobotPreset:
    """Add ``preset`` to the registry and return it, for use at module scope.

    The registry above is this repository's table; a robot that lives in someone
    else's package should not have to patch it to become ``--robot <name>``.
    Importing that package is enough::

        MY_ROBOT = register_robot_preset(
            RobotPreset(name="myarm_mydataset", embodiment=MY_ARM, convention=MY_KEYS),
            aliases=("myarm",),
        )

    Re-registering a name is an error unless ``replace=True``: a silent overwrite
    would change what a launch command already in production resolves to, which
    is the same class of invisible failure ``--robot`` exists to prevent. An alias
    may never shadow a canonical name, for the same reason.

    Registration is process-local and not persisted — ``--robot`` on the shipped
    server only sees presets whose module was imported.
    """
    if not isinstance(preset, RobotPreset):
        raise TypeError(f"expected a RobotPreset, got {type(preset).__name__}")
    existing = ROBOT_PRESETS.get(preset.name)
    if existing is not None and not replace:
        raise ValueError(
            f"robot preset {preset.name!r} is already registered ({existing.describe()}); "
            "pass replace=True to override it"
        )
    for alias in aliases:
        if alias in ROBOT_PRESETS:
            raise ValueError(
                f"alias {alias!r} for {preset.name!r} is already a canonical preset name; "
                "an alias that shadows a real preset would silently redirect --robot"
            )
        target = ROBOT_ALIASES.get(alias)
        if target is not None and target != preset.name and not replace:
            raise ValueError(
                f"alias {alias!r} already resolves to {target!r}; pass replace=True to move it"
            )
    ROBOT_PRESETS[preset.name] = preset
    for alias in aliases:
        ROBOT_ALIASES[alias] = preset.name
    return preset


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
    (``robot`` / ``robot_steps`` / ``image_keys`` / ``state_key`` /
    ``discrete_state``), which the server pushes on connect, so a client can
    assert it rather than guess. ``robot_steps`` says whether this robot's
    pre/post steps are actually wired, so a server that serves the preset's keys
    without its arithmetic cannot pass for the real thing.
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
            "robot_steps": preset.has_robot_steps,
            "robot_slots": [list(pair) for pair in zip(VIEW_SLOTS, keys)],
            **(dict(metadata) if metadata else {}),
        },
        **{**dict(preset.builder_kwargs), **kwargs},
    )
