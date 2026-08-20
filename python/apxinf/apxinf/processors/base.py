"""Processor steps and the :class:`Pipeline` container.

Design (mirrors the PRD's "narrow interface, thick module" goal):

* A :class:`ProcessorStep` is a plain callable. Its ``__call__`` takes the
  step's *natural* input (an image, a prompt, an action array, nothing) and
  returns its natural output. That is the whole public contract, so every step
  can be used standalone — ``resize(img)``, ``tokenizer(prompt)``,
  ``unnormalizer(action)`` — without a pipeline and without touching the model.
* :meth:`ProcessorStep.with_overrides` returns a reconfigured *copy* of a step,
  tweaking only the knobs the step advertises in ``PARAMS``. Heavy state (a
  loaded SentencePiece model) is shared by shallow copy, not rebuilt.
* :class:`Pipeline` chains named steps left-to-right, threading one value
  through, and supports whole-step replacement and per-step parameter override
  while leaving the other steps untouched.
"""

from __future__ import annotations

import abc
import copy
from typing import Any, Iterable, Sequence, Tuple, Union

__all__ = ["ProcessorStep", "Pipeline"]


class ProcessorStep(abc.ABC):
    """A single, independently-callable pre/post-processing step.

    Subclasses implement :meth:`__call__` with their natural signature and list
    the names of runtime-tweakable knobs in the class attribute :attr:`PARAMS`.
    Those knobs are stored as public instance attributes so that
    :meth:`with_overrides` can patch a copy generically.
    """

    #: Names of attributes that :meth:`with_overrides` may replace.
    PARAMS: Tuple[str, ...] = ()

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_config(self) -> dict:
        """Return the current value of every overridable knob."""
        return {name: getattr(self, name) for name in self.PARAMS}

    def with_overrides(self, **overrides: Any) -> "ProcessorStep":
        """Return a copy of this step with the given knobs replaced.

        Only names listed in :attr:`PARAMS` are accepted; anything else raises
        ``KeyError`` naming the step and its known knobs. The original step is
        never mutated.
        """
        if not overrides:
            return self
        unknown = set(overrides) - set(self.PARAMS)
        if unknown:
            raise KeyError(
                f"{type(self).__name__}.with_overrides: unknown params "
                f"{sorted(unknown)}; known: {sorted(self.PARAMS)}"
            )
        clone = copy.copy(self)
        clone._apply_overrides(overrides)
        return clone

    def _apply_overrides(self, overrides: dict) -> None:
        """Apply already-validated overrides to ``self`` (a fresh copy).

        Steps whose knobs feed derived state (e.g. an RNG seeded from a seed)
        override this to recompute that state.
        """
        for name, value in overrides.items():
            setattr(self, name, value)


# A pipeline entry is either a bare step (named by its class) or a (name, step)
# pair. Internally everything is normalized to ``(name, step)``.
StepSpec = Union[ProcessorStep, Tuple[str, ProcessorStep]]


class Pipeline:
    """An ordered, named chain of steps applied left-to-right.

    The output of each step is fed as the sole input to the next, so a pipeline
    is itself a callable ``value -> value``. This suits homogeneous chains such
    as ``parse -> resize``. Heterogeneous fan-out (routing an image here, a
    prompt there) is the job of a policy, not of a single pipeline.
    """

    def __init__(self, steps: Sequence[StepSpec]):
        normalized = [self._normalize(spec) for spec in steps]
        names = [name for name, _ in normalized]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"Pipeline step names must be unique; duplicates: {sorted(duplicates)}")
        self._steps: list[Tuple[str, ProcessorStep]] = normalized

    @staticmethod
    def _normalize(spec: StepSpec) -> Tuple[str, ProcessorStep]:
        if isinstance(spec, tuple):
            name, step = spec
        else:
            step = spec
            name = type(step).__name__
        if not isinstance(step, ProcessorStep):
            raise TypeError(f"Pipeline step {name!r} is not a ProcessorStep: {type(step)!r}")
        return str(name), step

    def __call__(self, value: Any) -> Any:
        for _, step in self._steps:
            value = step(value)
        return value

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self._steps]

    @property
    def steps(self) -> list[ProcessorStep]:
        return [step for _, step in self._steps]

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self) -> Iterable[Tuple[str, ProcessorStep]]:
        return iter(self._steps)

    def __getitem__(self, name: str) -> ProcessorStep:
        for step_name, step in self._steps:
            if step_name == name:
                return step
        raise KeyError(f"Pipeline has no step named {name!r}; have {self.names}")

    def _index(self, name: str) -> int:
        for i, (step_name, _) in enumerate(self._steps):
            if step_name == name:
                return i
        raise KeyError(f"Pipeline has no step named {name!r}; have {self.names}")

    def replace(self, name: str, step: ProcessorStep) -> "Pipeline":
        """Return a new pipeline with the whole step ``name`` swapped out."""
        if not isinstance(step, ProcessorStep):
            raise TypeError(f"replacement for {name!r} is not a ProcessorStep: {type(step)!r}")
        index = self._index(name)
        steps = list(self._steps)
        steps[index] = (name, step)
        return Pipeline(steps)

    def override(self, name: str, **params: Any) -> "Pipeline":
        """Return a new pipeline with only step ``name``'s knobs overridden."""
        index = self._index(name)
        current = self._steps[index][1]
        steps = list(self._steps)
        steps[index] = (name, current.with_overrides(**params))
        return Pipeline(steps)

    def insert_before(self, name: str, spec: StepSpec) -> "Pipeline":
        """Return a new pipeline with ``spec`` inserted just before step ``name``."""
        index = self._index(name)
        steps = list(self._steps)
        steps.insert(index, self._normalize(spec))
        return Pipeline(steps)

    def insert_after(self, name: str, spec: StepSpec) -> "Pipeline":
        """Return a new pipeline with ``spec`` inserted just after step ``name``."""
        index = self._index(name)
        steps = list(self._steps)
        steps.insert(index + 1, self._normalize(spec))
        return Pipeline(steps)

    def remove(self, name: str) -> "Pipeline":
        """Return a new pipeline with step ``name`` removed."""
        index = self._index(name)
        steps = list(self._steps)
        del steps[index]
        return Pipeline(steps)

    def reorder(self, names: Sequence[str]) -> "Pipeline":
        """Return a new pipeline whose steps follow ``names``.

        ``names`` must be a permutation of the current step names (same set, no
        duplicates); otherwise a ``ValueError`` explains the mismatch.
        """
        requested = list(names)
        current = self.names
        if sorted(requested) != sorted(current):
            raise ValueError(
                f"reorder must be a permutation of {current}; got {requested}"
            )
        by_name = {step_name: step for step_name, step in self._steps}
        return Pipeline([(step_name, by_name[step_name]) for step_name in requested])

    def __repr__(self) -> str:
        return f"Pipeline({self.names})"
