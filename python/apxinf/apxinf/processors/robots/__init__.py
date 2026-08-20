"""Robot-specific processing steps, varying by *robot body* not by model.

These are ordinary :class:`~apxinf.processors.base.ProcessorStep` s — pure-numpy
``dict -> dict`` transforms with no policy / model / CUDA dependency — that
encode a particular robot's state and action conventions (DoF layout, joint-flip
sign, gripper calibration, delta<->absolute actions). They live alongside the
generic steps (``parse_image``, ``tokenize``, ...) because they vary along the
same axis: the data shape, not the model. A robot *adapter* in
:mod:`apxinf.robots` assembles these steps with a concrete model policy.
"""

from __future__ import annotations

from .unitree_g1 import (
    UnitreeG1AbsoluteActions,
    UnitreeG1DecodeState,
    UnitreeG1EncodeActions,
)

__all__ = [
    "UnitreeG1DecodeState",
    "UnitreeG1AbsoluteActions",
    "UnitreeG1EncodeActions",
]
