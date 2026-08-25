"""Robot adapters: assemble robot-specific steps with a model policy.

A model policy (:class:`~apxinf.policies.impls.pi05.Pi05Policy`) is robot-agnostic:
it maps ``images + prompt (+ state) -> normalized action -> unnormalized action``.
A *robot adapter* binds that generic core to one robot by wiring the robot's
:mod:`apxinf.processors.robots` steps into the policy's pre/post
:class:`~apxinf.processors.Pipeline` and loading its checkpoint / norm_stats.

This is the top assembly layer: it depends *downward* on both
:mod:`apxinf.policies` (the model) and :mod:`apxinf.processors` (the steps).
Neither depends back on it, so adding a robot never touches the policy or
processor packages — you write steps under ``processors/robots/`` and a
``build_*`` factory here, then register both in :mod:`apxinf.robots.presets` so
they are reachable as ``--robot <name>`` at serving time.
"""

from __future__ import annotations

from .presets import (
    ROBOT_PRESETS,
    VIEW_SLOTS,
    RobotPreset,
    available_robots,
    build_robot_policy,
    get_robot_preset,
)
from .unitree_g1 import build_unitree_g1_policy

__all__ = [
    "build_unitree_g1_policy",
    "RobotPreset",
    "ROBOT_PRESETS",
    "VIEW_SLOTS",
    "available_robots",
    "get_robot_preset",
    "build_robot_policy",
]
