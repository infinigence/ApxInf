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
``build_*`` factory here.
"""

from __future__ import annotations

from .unitree_g1 import build_unitree_g1_policy

__all__ = ["build_unitree_g1_policy"]
