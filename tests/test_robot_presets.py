"""Robot presets, wire-key resolution, and the G1 serving path end-to-end.

Covers the layer that decides *which keys a client must send*, which is where a
deployment silently degrades rather than fails: a checkpoint served under the
wrong embodiment still returns well-shaped actions.

Four groups:

* :class:`KeyResolutionTest` — ``lookup_key`` / ``has_key`` / ``set_key``, the
  flat-vs-nested wire layouts (``"observation/image"`` vs
  ``obs["images"]["cam_high"]``).
* :class:`RobotPresetTest` / :class:`SyntheticContractTest` /
  :class:`BuildRobotPolicyTest` — the preset table's invariants, its overrides,
  and the contract it publishes (including what a checkpoint-free server must
  admit it cannot honour).
* :class:`PipelineCompositionTest` / :class:`PolicyCompositionTest` /
  :class:`UnitreeG1AdapterTest` — the robot/model seam: a robot's steps wrap a
  model's chain from outside, without either layer naming the other.
* :class:`UnitreeG1ServingTest` — an **unmodified openpi G1 observation** driven
  through the real websocket transport into a G1-wired policy, asserting camera
  →slot binding, state routing, and the delta→absolute / 32→16 output chain.

Runs offline against a mock model; no CUDA, no checkpoint.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import pathlib
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

import numpy as np
from openpi_client import websocket_client_policy
import websockets.asyncio.server as websocket_server

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "python" / "apxinf"))
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

from apxinf import Pi05Policy  # noqa: E402
from apxinf.policies.base import ComposablePolicy  # noqa: E402
from apxinf.processors import (  # noqa: E402
    Normalizer,
    Pipeline,
    ProcessorStep,
    PromptTokenizer,
    Unnormalizer,
)
from apxinf import conventions as conventions_module  # noqa: E402
from apxinf.conventions import (  # noqa: E402
    CONVENTIONS,
    LIBERO as LIBERO_CONVENTION,
    UNITREE_G1 as G1_CONVENTION,
    Convention,
    available_conventions,
    get_convention,
    register_convention,
)
from apxinf.processors.robots.unitree_g1 import (  # noqa: E402
    G1_ROBOT_DIM,
    UnitreeG1AbsoluteActions,
    UnitreeG1DecodeState,
    UnitreeG1EncodeActions,
)

# The G1 client's wire keys are a recording convention, not a fact about the
# body, so the tests read them from the convention the preset pairs with.
G1_CAMERAS = G1_CONVENTION.image_keys
G1_STATE_KEY = G1_CONVENTION.state_key
from apxinf.processors.transforms import (  # noqa: E402
    Tokenize,
    has_key,
    lookup_key,
    set_key,
)
from apxinf.policies.impls import pi05 as pi05_module  # noqa: E402
from apxinf.robots import unitree_g1 as robot_adapter  # noqa: E402
from apxinf.robots.presets import (  # noqa: E402
    ROBOT_ALIASES,
    ROBOT_PRESETS,
    VIEW_SLOTS,
    Embodiment,
    RobotPreset,
    available_robots,
    build_robot_policy,
    get_robot_preset,
    register_robot_preset,
)
from apxinf.robots.unitree_g1 import build_unitree_g1_policy  # noqa: E402
from apxinf.serving import WebsocketPolicyServer  # noqa: E402
from apxinf.serving.websocket import health_check  # noqa: E402

HORIZON = 4
MODEL_DIM = 32
IMAGE_SIZE = 224


class KeyResolutionTest(unittest.TestCase):
    """A wire key must resolve the same way everywhere it is used."""

    OBSERVATION = {
        "observation/image": "flat-base",
        "images": {"cam_high": "nested-base", "cam_left_wrist": "nested-left"},
        "state": np.zeros(4, dtype=np.float32),
        "explicitly_none": None,
    }

    def test_flat_key_containing_a_slash_is_not_split(self) -> None:
        # LIBERO/DROID names are flat even though they contain "/". Splitting
        # them would silently look in the wrong place.
        self.assertEqual(lookup_key(self.OBSERVATION, "observation/image"), "flat-base")

    def test_nested_key_is_walked_as_a_path(self) -> None:
        # ALOHA/G1 clients send obs["images"]["cam_high"].
        self.assertEqual(lookup_key(self.OBSERVATION, "images/cam_high"), "nested-base")

    def test_flat_hit_wins_over_a_nested_path(self) -> None:
        observation = {"a/b": "flat", "a": {"b": "nested"}}
        self.assertEqual(lookup_key(observation, "a/b"), "flat")

    def test_missing_key_raises_without_a_default(self) -> None:
        with self.assertRaises(KeyError):
            lookup_key(self.OBSERVATION, "images/cam_right_wrist")
        self.assertIsNone(lookup_key(self.OBSERVATION, "images/cam_right_wrist", None))

    def test_has_key_distinguishes_absent_from_a_none_value(self) -> None:
        # A present-but-None state must not be reported as missing, or the
        # required-key check would reject a valid observation.
        self.assertTrue(has_key(self.OBSERVATION, "explicitly_none"))
        self.assertFalse(has_key(self.OBSERVATION, "images/cam_right_wrist"))

    def test_set_key_writes_nested_without_mutating_the_caller(self) -> None:
        updated = set_key(self.OBSERVATION, "images/cam_high", "decoded")
        self.assertEqual(updated["images"]["cam_high"], "decoded")
        self.assertEqual(updated["images"]["cam_left_wrist"], "nested-left")
        self.assertEqual(self.OBSERVATION["images"]["cam_high"], "nested-base")

    def test_set_key_writes_flat_for_a_flat_key(self) -> None:
        updated = set_key(self.OBSERVATION, "state", np.ones(4, dtype=np.float32))
        self.assertTrue(np.all(updated["state"] == 1))
        self.assertTrue(np.all(self.OBSERVATION["state"] == 0))


class RobotPresetTest(unittest.TestCase):
    """The preset table is the only place a wire contract is declared."""

    def test_shipped_presets_are_well_formed(self) -> None:
        for name, preset in ROBOT_PRESETS.items():
            self.assertEqual(name, preset.name)
            self.assertEqual(preset.num_views, len(preset.image_keys))
            self.assertLessEqual(preset.num_views, len(VIEW_SLOTS))

    def test_franka_libero_matches_openpi_libero_inputs(self) -> None:
        preset = get_robot_preset("franka_libero")
        self.assertEqual(
            preset.image_keys, ("observation/image", "observation/wrist_image")
        )
        self.assertEqual(preset.state_key, "observation/state")
        self.assertEqual(preset.action_dim, 7)

    def test_the_libero_alias_still_resolves(self) -> None:
        # Deployed launch commands say --robot libero; a rename must not break a
        # running install, so the old spelling keeps resolving to the same entry.
        self.assertIs(get_robot_preset("libero"), get_robot_preset("franka_libero"))
        self.assertNotIn("libero", available_robots())
        self.assertIn("libero", available_robots(include_aliases=True))

    def test_g1_matches_an_unmodified_openpi_g1_client(self) -> None:
        preset = get_robot_preset("unitree_g1")
        self.assertEqual(preset.image_keys, G1_CAMERAS)
        self.assertEqual(preset.state_key, G1_STATE_KEY)
        # State must be routed, not dropped: without it delta->absolute is a no-op.
        self.assertTrue(preset.discrete_state)
        # Full model width; UnitreeG1EncodeActions does the 32->16 truncation.
        self.assertIsNone(preset.action_dim)

    def test_slots_are_derived_in_view_slot_order(self) -> None:
        # A checkpoint fills view slots from 0 up, so "base + right wrist" is not
        # expressible. The old table let you *write* the pairs and validated the
        # order; deriving them removes the failure mode instead of checking it.
        convention = Convention(
            name="two_cam", image_keys=("a", "b"), state_key="state"
        )
        self.assertEqual(convention.slots, (("base_0_rgb", "a"), ("left_wrist_0_rgb", "b")))

    def test_duplicate_camera_keys_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Convention(name="bad", image_keys=("a", "a"), state_key="state")

    def test_more_cameras_than_view_slots_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Convention(
                name="bad",
                image_keys=tuple(f"cam{i}" for i in range(len(VIEW_SLOTS) + 1)),
                state_key="state",
            )

    def test_a_convention_cannot_be_paired_with_the_wrong_body(self) -> None:
        # The one check only the pairing can make: a 3-camera dialect against a
        # 2-camera body would stack the wrong number of views, and nothing
        # downstream distinguishes that from a checkpoint mismatch.
        with self.assertRaises(ValueError) as caught:
            RobotPreset(
                name="mismatched",
                embodiment=Embodiment(name="two_cam_body", num_cameras=2),
                convention=Convention(
                    name="three_cam", image_keys=("a", "b", "c"), state_key="state"
                ),
            )
        message = str(caught.exception)
        self.assertIn("three_cam", message)
        self.assertIn("two_cam_body", message)

    def test_the_two_halves_vary_independently(self) -> None:
        # The reason for the split: a second key convention on the same body is a
        # new Convention, not a new body. Re-pairing must need no builder edit.
        franka = get_robot_preset("franka_libero").embodiment
        droid_like = Convention(
            name="droid_like",
            image_keys=("observation/exterior_image_1_left", "observation/wrist_image_left"),
            state_key="observation/joint_position",
        )
        repaired = RobotPreset(name="franka_droid_like", embodiment=franka, convention=droid_like)
        self.assertEqual(repaired.image_keys, droid_like.image_keys)
        self.assertEqual(repaired.action_dim, 7)
        self.assertIs(repaired.builder, get_robot_preset("franka_libero").builder)

    def test_unknown_robot_names_the_known_ones(self) -> None:
        with self.assertRaises(KeyError) as caught:
            get_robot_preset("g1")
        for name in available_robots(include_aliases=True):
            self.assertIn(name, str(caught.exception))


class SyntheticContractTest(unittest.TestCase):
    """A checkpoint-free server may serve a preset's keys, not its arithmetic.

    ``--random-weights --robot unitree_g1`` runs ``Pi05Policy.from_random``, which
    never calls ``preset.builder``: the wire keys and view count are real, the
    action semantics are absent. Publishing the preset name unqualified would be
    the silent embodiment mismatch ``--robot`` exists to prevent, so every gap is
    named at startup and ``robot_steps`` goes on the wire.
    """

    def test_only_a_builder_preset_reports_robot_steps(self) -> None:
        self.assertFalse(get_robot_preset("franka_libero").has_robot_steps)
        self.assertTrue(get_robot_preset("unitree_g1").has_robot_steps)

    def test_a_generic_preset_that_drops_state_has_nothing_to_report(self) -> None:
        preset = get_robot_preset("franka_libero")
        self.assertEqual(
            preset.synthetic_gaps(discrete_state=False, served_action_dim=7), ()
        )

    def test_dropped_state_is_reported_even_for_a_generic_preset(self) -> None:
        # --discrete-state on franka_libero: the synthetic tokenizer ignores it.
        preset = get_robot_preset("franka_libero")
        gaps = preset.synthetic_gaps(discrete_state=True, served_action_dim=7)
        self.assertEqual(len(gaps), 1)
        self.assertIn("discrete_state", gaps[0])

    def test_g1_reports_both_its_state_and_its_skipped_steps(self) -> None:
        preset = get_robot_preset("unitree_g1")
        gaps = preset.synthetic_gaps(discrete_state=True, served_action_dim=MODEL_DIM)
        self.assertEqual(len(gaps), 2)
        joined = " ".join(gaps)
        self.assertIn("discrete_state", joined)
        # The gap must name the factory that was skipped and the concrete symptom:
        # a 32-wide action where the real server truncates to 16.
        self.assertIn("build_unitree_g1_policy", joined)
        self.assertIn(str(MODEL_DIM), joined)

    def test_a_preset_whose_builder_owns_no_truncation_omits_the_width_gap(self) -> None:
        # action_dim set means the width is the preset's, not a skipped step's, so
        # the synthetic server serves the right one and must not claim otherwise.
        preset = RobotPreset(
            name="stub",
            embodiment=Embodiment(
                name="stub_body",
                num_cameras=1,
                action_dim=7,
                builder=lambda model_dir, **kwargs: None,
            ),
            convention=Convention(name="stub_keys", image_keys=("cam",), state_key="state"),
        )
        gaps = preset.synthetic_gaps(discrete_state=False, served_action_dim=7)
        self.assertEqual(len(gaps), 1)
        self.assertNotIn("action_dim", gaps[0])


class BuildRobotPolicyTest(unittest.TestCase):
    """``build_robot_policy`` resolves overrides and publishes the contract."""

    def _register(self, preset: RobotPreset) -> None:
        ROBOT_PRESETS[preset.name] = preset
        self.addCleanup(ROBOT_PRESETS.pop, preset.name, None)

    def test_preset_defaults_and_builder_kwargs_reach_the_builder(self) -> None:
        seen: dict = {}
        preset = RobotPreset(
            name="stub_robot",
            embodiment=Embodiment(
                name="stub_body",
                num_cameras=2,
                action_dim=None,
                builder=lambda model_dir, **kwargs: seen.update(kwargs, model_dir=model_dir),
                builder_kwargs={"use_delta_joint_actions": True},
            ),
            convention=Convention(
                name="stub_keys",
                image_keys=("cam/high", "cam/wrist"),
                state_key="joints",
                discrete_state=True,
            ),
        )
        self._register(preset)

        build_robot_policy("stub_robot", "/nowhere", metadata={"precision": "bf16"})

        self.assertEqual(seen["model_dir"], "/nowhere")
        self.assertEqual(seen["image_keys"], ("cam/high", "cam/wrist"))
        self.assertEqual(seen["state_key"], "joints")
        self.assertIsNone(seen["action_dim"])
        self.assertTrue(seen["discrete_state"])
        self.assertTrue(seen["use_delta_joint_actions"])

        published = seen["metadata"]
        self.assertEqual(published["robot"], "stub_robot")
        # A robot-step preset loaded from a checkpoint gets its arithmetic, so the
        # flag the synthetic path clears is set here.
        self.assertTrue(published["robot_steps"])
        self.assertEqual(
            published["robot_slots"],
            [["base_0_rgb", "cam/high"], ["left_wrist_0_rgb", "cam/wrist"]],
        )
        self.assertEqual(published["precision"], "bf16")

    def test_overrides_replace_only_the_named_fields(self) -> None:
        seen: dict = {}
        preset = RobotPreset(
            name="stub_generic",
            embodiment=Embodiment(
                name="stub_body",
                num_cameras=1,
                action_dim=7,
                builder=lambda model_dir, **kwargs: seen.update(kwargs),
            ),
            convention=Convention(
                name="stub_keys",
                image_keys=("observation/image",),
                state_key="observation/state",
            ),
        )
        self._register(preset)

        build_robot_policy(
            "stub_generic", "/nowhere", image_keys=["rgb/front"], state_key="q"
        )

        self.assertEqual(seen["image_keys"], ("rgb/front",))
        self.assertEqual(seen["state_key"], "q")
        self.assertEqual(seen["action_dim"], 7)
        # robot_slots reports the *served* keys against the slots they fill, so an
        # override stays reviewable instead of hiding behind the preset name.
        self.assertEqual(seen["metadata"]["robot_slots"], [["base_0_rgb", "rgb/front"]])


class _Tag(ProcessorStep):
    """Trivial step: append its own label to the flowing list."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __call__(self, value):
        return list(value) + [self.label]


class PipelineCompositionTest(unittest.TestCase):
    """``prepend``/``append``: the only editing verbs that name no inner step.

    Every other verb (``insert_before``, ``replace``, ...) needs a step name, so
    an *outer* layer using them has to know the inner chain's private vocabulary.
    A robot adapter's requirement is only "run outside the model's steps", and
    these two say exactly that.
    """

    def _chain(self) -> Pipeline:
        return Pipeline([("a", _Tag("a")), ("b", _Tag("b"))])

    def test_wrapping_runs_outside_the_existing_steps_in_order(self) -> None:
        chain = self._chain().prepend(("pre1", _Tag("pre1")), ("pre2", _Tag("pre2")))
        chain = chain.append(("post1", _Tag("post1")), ("post2", _Tag("post2")))
        self.assertEqual(chain([]), ["pre1", "pre2", "a", "b", "post1", "post2"])

    def test_the_original_pipeline_is_untouched(self) -> None:
        original = self._chain()
        original.prepend(("pre", _Tag("pre")))
        original.append(("post", _Tag("post")))
        self.assertEqual(original.names, ["a", "b"])

    def test_a_colliding_name_is_rejected_rather_than_shadowing(self) -> None:
        # A wrapper that silently replaced an inner step would be the worst
        # possible failure: the model's chain quietly loses a step.
        with self.assertRaises(ValueError) as caught:
            self._chain().prepend(("b", _Tag("b2")))
        self.assertIn("'b'", str(caught.exception))


class MockModel:
    """In-process ``BareModel`` stand-in returning a constant normalized action."""

    def __init__(self, num_views: int, value: float = 0.0) -> None:
        self.action_horizon = HORIZON
        self.action_dim = MODEL_DIM
        self.num_views = num_views
        self.image_size = IMAGE_SIZE
        self.max_token_len = 200
        self.images: list = []
        self.states: list = []
        self._value = value

    def infer_rgb(self, rgb_u8, layout, token_ids, noise):
        assert layout == "nhwc"
        self.images.append(np.asarray(rgb_u8).copy())
        return np.full((HORIZON, MODEL_DIM), self._value, dtype=np.float32)


class RecordingTokenizer(PromptTokenizer):
    """Discrete-state tokenizer that records the state it was handed."""

    def __init__(self) -> None:
        self.max_token_len = 200
        self.discrete_state = True
        self.seen_states: list = []

    def __call__(self, prompt, state=None):
        self.seen_states.append(None if state is None else np.asarray(state).copy())
        return np.asarray([2, 108, 12], dtype=np.uint32)


def build_g1_mock_policy() -> Pi05Policy:
    """A G1-wired policy over a mock model: the real pipelines, no checkpoint.

    Takes the same route ``build_unitree_g1_policy`` does — a stock policy, then
    ``with_adapter`` — but injects the model and an identity unnormalizer, so the
    assertions isolate the adapter's arithmetic from checkpoint norm_stats.
    """
    model = MockModel(num_views=len(G1_CAMERAS))
    tokenizer = RecordingTokenizer()
    identity = Unnormalizer(
        q01=[-1.0] * MODEL_DIM, q99=[1.0] * MODEL_DIM, dims=MODEL_DIM, eps=0.0
    )
    input_pipeline, output_pipeline = Pi05Policy.default_pipelines(
        model,
        tokenizer=tokenizer,
        unnormalizer=identity,
        image_keys=G1_CAMERAS,
        state_key=G1_STATE_KEY,
    )
    base = Pi05Policy(
        model,
        input_pipeline=input_pipeline,
        output_pipeline=output_pipeline,
        image_keys=G1_CAMERAS,
        state_key=G1_STATE_KEY,
        metadata={"protocol": "openpi.websocket_policy"},
    )
    policy = base.with_adapter(
        before=[("g1_decode_state", UnitreeG1DecodeState(G1_STATE_KEY))],
        after=[
            ("g1_absolute", UnitreeG1AbsoluteActions(G1_STATE_KEY)),
            ("g1_encode", UnitreeG1EncodeActions()),
        ],
        action_dim=G1_ROBOT_DIM,
        metadata={"robot": "unitree_g1"},
    )
    policy.tokenizer = tokenizer
    return policy


def _identity_unnormalizer() -> Unnormalizer:
    return Unnormalizer(
        q01=[-1.0] * MODEL_DIM, q99=[1.0] * MODEL_DIM, dims=MODEL_DIM, eps=0.0
    )


class PolicyCompositionTest(unittest.TestCase):
    """``Pi05Policy.with_adapter``: the seam a robot adapter wraps.

    Its contract is nesting — ``before`` outside the model's whole input chain,
    ``after`` outside its whole output chain — plus republishing what the wrapped
    policy actually serves. Getting the second part wrong is the failure this
    layer exists to prevent: a policy advertising an ``action_dim`` its steps do
    not emit degrades silently on the wire.
    """

    def _base(self) -> Pi05Policy:
        model = MockModel(num_views=len(G1_CAMERAS))
        input_pipeline, output_pipeline = Pi05Policy.default_pipelines(
            model,
            tokenizer=RecordingTokenizer(),
            unnormalizer=_identity_unnormalizer(),
            image_keys=G1_CAMERAS,
            state_key=G1_STATE_KEY,
        )
        return Pi05Policy(
            model,
            input_pipeline=input_pipeline,
            output_pipeline=output_pipeline,
            image_keys=G1_CAMERAS,
            state_key=G1_STATE_KEY,
            metadata={"protocol": "openpi.websocket_policy"},
        )

    def test_steps_land_outside_the_model_chain_in_both_directions(self) -> None:
        base = self._base()
        wrapped = base.with_adapter(
            before=[("g1_decode_state", UnitreeG1DecodeState(G1_STATE_KEY))],
            after=[("g1_encode", UnitreeG1EncodeActions())],
            action_dim=G1_ROBOT_DIM,
        )
        self.assertEqual(wrapped.input_pipeline.names[0], "g1_decode_state")
        self.assertEqual(wrapped.output_pipeline.names[-1], "g1_encode")
        # The model's own steps keep their order and their names; the wrapper
        # never had to learn what they are.
        self.assertEqual(wrapped.input_pipeline.names[1:], base.input_pipeline.names)
        self.assertEqual(wrapped.output_pipeline.names[:-1], base.output_pipeline.names)

    def test_the_original_policy_is_not_mutated(self) -> None:
        base = self._base()
        base.with_adapter(
            after=[("g1_encode", UnitreeG1EncodeActions())], action_dim=G1_ROBOT_DIM
        )
        self.assertNotIn("g1_encode", base.output_pipeline.names)
        self.assertEqual(base.action_dim, MODEL_DIM)

    def test_the_published_contract_describes_the_wrapped_chain(self) -> None:
        wrapped = self._base().with_adapter(
            after=[("g1_encode", UnitreeG1EncodeActions())],
            action_dim=G1_ROBOT_DIM,
            metadata={"robot": "unitree_g1"},
        )
        # The derived half is recomputed: inheriting the pre-adapter action_dim
        # would publish a width no step in this chain produces.
        self.assertEqual(wrapped.metadata["action_dim"], G1_ROBOT_DIM)
        self.assertIn("g1_encode", wrapped.metadata["output_pipeline"])
        # The caller's half is carried forward and extended.
        self.assertEqual(wrapped.metadata["protocol"], "openpi.websocket_policy")
        self.assertEqual(wrapped.metadata["robot"], "unitree_g1")

    def test_an_unchanged_width_is_inherited(self) -> None:
        # An appended step that does not narrow the action needs no declaration.
        wrapped = self._base().with_adapter(
            after=[("g1_absolute", UnitreeG1AbsoluteActions(G1_STATE_KEY))]
        )
        self.assertEqual(wrapped.action_dim, MODEL_DIM)

    def test_the_model_handle_is_shared_not_reloaded(self) -> None:
        # Rewiring, not a second load: two handles on one GPU allocation would
        # double the checkpoint's memory and make close() order matter.
        base = self._base()
        self.assertIs(base.with_adapter().model, base.model)

    def test_a_wrapper_cannot_shadow_a_step_it_does_not_own(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._base().with_adapter(
                before=[("tokenize", UnitreeG1DecodeState(G1_STATE_KEY))]
            )
        self.assertIn("tokenize", str(caught.exception))

    def test_pi05_satisfies_the_composable_contract(self) -> None:
        self.assertIsInstance(self._base(), ComposablePolicy)


class _StubComposable:
    """Minimal ``ComposablePolicy``: records what an adapter wrapped it with."""

    def __init__(self) -> None:
        self.wrapped: dict = {}

    def with_adapter(self, *, before=(), after=(), action_dim=None, metadata=None):
        self.wrapped = {
            "before": [name for name, _ in before],
            "after": [name for name, _ in after],
            "action_dim": action_dim,
            "metadata": dict(metadata or {}),
        }
        return self


class UnitreeG1AdapterTest(unittest.TestCase):
    """The G1 builder wraps a policy it never names, and hands load flags on.

    These run against a stub loader, so they assert the *adapter's* behaviour
    without a checkpoint: which steps it wraps, on which side, what width it
    claims, and — the bug this split removes — that every keyword it forwards is
    one the real loading path accepts.
    """

    def _patched_loader(self, captured: dict, policy):
        def load(model_dir, *, model_type=None, **kwargs):
            # Mirror AutoPolicy's real split: it consumes model_type itself and
            # forwards the rest to a concrete from_pretrained that has **no**
            # **kwargs. Binding against that signature reproduces the TypeError
            # serving would raise, which is why this stub is not a bare Mock.
            inspect.signature(Pi05Policy.from_pretrained).bind(model_dir, **kwargs)
            captured.update(kwargs, model_dir=model_dir, model_type=model_type)
            return policy

        return mock.patch.object(
            robot_adapter, "AutoPolicy", types.SimpleNamespace(from_pretrained=load)
        )

    def test_server_load_flags_reach_the_concrete_loader(self) -> None:
        # --model-type travels server -> build_robot_policy -> this builder. It
        # is AutoPolicy's argument, not the concrete policy's, so a builder that
        # forwards it blindly raises TypeError the moment anyone serves a G1
        # checkpoint. Only binding the real signature catches that.
        captured: dict = {}
        with self._patched_loader(captured, _StubComposable()):
            build_robot_policy(
                "unitree_g1", "/nowhere", model_type="pi05", precision="bf16"
            )
        self.assertEqual(captured["model_type"], "pi05")
        self.assertEqual(captured["precision"], "bf16")
        self.assertEqual(captured["state_key"], G1_STATE_KEY)
        # Loaded at full model width: delta->absolute must see the whole action
        # before g1_encode truncates it to 16.
        self.assertIsNone(captured["action_dim"])

    def test_the_g1_steps_wrap_the_model_chain_from_outside(self) -> None:
        stub = _StubComposable()
        with self._patched_loader({}, stub):
            build_robot_policy("unitree_g1", "/nowhere")
        self.assertEqual(stub.wrapped["before"], ["g1_decode_state"])
        self.assertEqual(stub.wrapped["after"], ["g1_absolute", "g1_encode"])
        self.assertEqual(stub.wrapped["action_dim"], G1_ROBOT_DIM)
        self.assertEqual(stub.wrapped["metadata"]["robot"], "unitree_g1")

    def test_without_the_truncating_step_no_width_is_claimed(self) -> None:
        # adapt_to_pi=False drops g1_encode, so nothing narrows the action to 16.
        # Advertising 16 regardless would publish a width no step produces.
        stub = _StubComposable()
        with self._patched_loader({}, stub):
            build_unitree_g1_policy(
                "/nowhere",
                state_key=G1_STATE_KEY,
                image_keys=G1_CAMERAS,
                adapt_to_pi=False,
            )
        self.assertEqual(stub.wrapped["before"], [])
        self.assertEqual(stub.wrapped["after"], ["g1_absolute"])
        self.assertIsNone(stub.wrapped["action_dim"])

    def test_the_wire_keys_are_required_arguments(self) -> None:
        # They are a recording convention, not a fact about this body. A default
        # here would rebuild the robot<->dataset coupling one layer up: the same
        # G1 re-recorded under different keys would silently serve the old ones.
        with self._patched_loader({}, _StubComposable()):
            with self.assertRaises(TypeError):
                build_unitree_g1_policy("/nowhere", image_keys=G1_CAMERAS)
            with self.assertRaises(TypeError):
                build_unitree_g1_policy("/nowhere", state_key=G1_STATE_KEY)

    def test_a_policy_that_cannot_be_wrapped_is_named_in_the_error(self) -> None:
        class Unwrappable:
            pass

        with self._patched_loader({}, Unwrappable()):
            with self.assertRaises(TypeError) as caught:
                build_unitree_g1_policy(
                    "/nowhere", state_key=G1_STATE_KEY, image_keys=G1_CAMERAS
                )
        message = str(caught.exception)
        self.assertIn("Unwrappable", message)
        self.assertIn("with_adapter", message)

    def test_the_adapter_names_no_model_class(self) -> None:
        # The whole point of the split: this module knows a body, not a model.
        # A concrete policy reappearing here is the regression to catch.
        source = pathlib.Path(robot_adapter.__file__).read_text()
        self.assertNotIn("Pi05Policy", source)


class ModelLayerHoldsNoWireKeysTest(unittest.TestCase):
    """The policy layer must not carry any dataset's wire contract.

    ``Pi05Policy`` used to default ``image_keys`` to LIBERO's
    ``observation/image`` / ``observation/wrist_image``. As *the* default those
    two keys silently applied to every checkpoint, so a G1 checkpoint served bare
    ran LIBERO's contract and looked like an accuracy problem. The fallback is
    now the model's own view slots, which cannot masquerade as anyone's keys.
    """

    def _policy(self, num_views: int) -> Pi05Policy:
        model = MockModel(num_views=num_views)
        # A state key has to be named because RecordingTokenizer discretizes
        # state; these tests are about the *camera* fallback, so it is a neutral
        # spelling rather than any dataset's.
        input_pipeline, output_pipeline = Pi05Policy.default_pipelines(
            model,
            tokenizer=RecordingTokenizer(),
            unnormalizer=_identity_unnormalizer(),
            state_key="state",
        )
        return Pi05Policy(
            model,
            input_pipeline=input_pipeline,
            output_pipeline=output_pipeline,
            state_key="state",
        )

    def test_unnamed_cameras_fall_back_to_the_models_own_slots(self) -> None:
        policy = self._policy(2)
        self.assertEqual(policy.image_keys, VIEW_SLOTS[:2])
        # Same list in the published contract, so a client reading metadata sees
        # exactly what the ImageStack step resolves.
        self.assertEqual(policy.metadata["image_keys"], list(VIEW_SLOTS[:2]))

    def test_the_fallback_is_not_any_datasets_convention(self) -> None:
        policy = self._policy(2)
        for key in ("observation/image", "observation/wrist_image", "images/cam_high"):
            self.assertNotIn(key, policy.image_keys)

    def test_a_libero_client_against_the_fallback_fails_loudly(self) -> None:
        # The point of dropping the default: a mismatch is an error naming both
        # sides, not a silent run on the wrong cameras.
        policy = self._policy(2)
        observation = {
            "observation/image": np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8),
            "observation/wrist_image": np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8),
            "prompt": "pick up the block",
        }
        with self.assertRaises(KeyError) as caught:
            policy.infer(observation)
        message = str(caught.exception)
        self.assertIn(VIEW_SLOTS[0], message)

    def test_the_fallback_stays_total_beyond_the_declared_slots(self) -> None:
        # default_pipelines requires len(image_keys) == model.num_views, so the
        # fallback has to name *every* view a model claims, slots or not.
        policy = self._policy(len(VIEW_SLOTS) + 1)
        self.assertEqual(len(policy.image_keys), len(VIEW_SLOTS) + 1)
        self.assertEqual(policy.image_keys[: len(VIEW_SLOTS)], VIEW_SLOTS)
        self.assertEqual(len(set(policy.image_keys)), len(policy.image_keys))

    def test_no_constructor_defaults_a_camera_key(self) -> None:
        # Asserted on the signatures rather than by grepping the source, so the
        # prose explaining the old LIBERO default does not trip the check. Every
        # entry point that takes image_keys must leave them unnamed.
        for func in (
            pi05_module.Pi05Policy.__init__,
            pi05_module.Pi05Policy.default_pipelines,
            pi05_module.Pi05Policy.from_pretrained,
            pi05_module.Pi05Policy.from_random,
        ):
            with self.subTest(entry_point=func.__qualname__):
                default = inspect.signature(func).parameters["image_keys"].default
                self.assertIsNone(default)


class RunningServer:
    def __init__(self, policy) -> None:
        self._policy_server = WebsocketPolicyServer(policy, "127.0.0.1", 0)
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise TimeoutError("websocket test server did not start")

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)

        async def start() -> None:
            self._server = await websocket_server.serve(
                self._policy_server.handler,
                "127.0.0.1",
                0,
                compression=None,
                max_size=None,
                process_request=health_check,
            )
            self.port = self._server.sockets[0].getsockname()[1]
            self._ready.set()

        self._loop.run_until_complete(start())
        self._loop.run_forever()

    def close(self) -> None:
        async def stop() -> None:
            self._server.close()
            await self._server.wait_closed()

        asyncio.run_coroutine_threadsafe(stop(), self._loop).result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        self._loop.close()


def g1_observation(state: np.ndarray, prompt: str = "pull down the black switch"):
    """Exactly what an unmodified openpi G1 client sends."""
    def cam(fill: int) -> np.ndarray:
        return np.full((480, 640, 3), fill, dtype=np.uint8)

    return {
        "images": {
            "cam_high": cam(10),
            "cam_left_wrist": cam(20),
            "cam_right_wrist": cam(30),
        },
        "state": state,
        "prompt": prompt,
    }


class UnitreeG1ServingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = build_g1_mock_policy()
        self.server = RunningServer(self.policy)

    def tearDown(self) -> None:
        self.server.close()

    def test_unmodified_openpi_g1_observation_round_trips(self) -> None:
        client = websocket_client_policy.WebsocketClientPolicy(
            "127.0.0.1", self.server.port
        )
        metadata = client.get_server_metadata()
        # The wire contract is published, so a client can assert it instead of
        # discovering a mismatch as degraded behaviour.
        self.assertEqual(metadata["robot"], "unitree_g1")
        self.assertEqual(metadata["image_keys"], list(G1_CAMERAS))
        self.assertEqual(metadata["state_key"], G1_STATE_KEY)
        self.assertTrue(metadata["discrete_state"])
        self.assertEqual(metadata["action_dim"], G1_ROBOT_DIM)

        state = np.arange(G1_ROBOT_DIM, dtype=np.float32) / 100.0
        try:
            result = client.infer(g1_observation(state))
        finally:
            client._ws.close()

        actions = result["actions"]
        self.assertEqual(actions.shape, (HORIZON, G1_ROBOT_DIM))

        # Cameras land in declared slot order, resized to the model edge.
        stacked = self.policy.model.images[0]
        self.assertEqual(stacked.shape, (3, IMAGE_SIZE, IMAGE_SIZE, 3))
        for slot, fill in enumerate((10, 20, 30)):
            self.assertEqual(int(stacked[slot].max()), fill, f"slot {slot}")

        # State reached the tokenizer (dropped state is the silent failure mode).
        seen = self.policy.tokenizer.seen_states[-1]
        self.assertIsNotNone(seen)
        np.testing.assert_allclose(seen[:7], state[:7], atol=1e-6)

        # Model emitted normalized 0 -> identity-unnormalized 0, so every arm dim
        # is exactly the current joint angle (delta 0 made absolute).
        np.testing.assert_allclose(actions[0][:7], state[:7], atol=1e-6)
        np.testing.assert_allclose(actions[0][8:15], state[8:15], atol=1e-6)
        # Gripper dims are absolute already, and clipped into [0, 1].
        self.assertTrue(np.all(actions[:, [7, 15]] >= 0.0))
        self.assertTrue(np.all(actions[:, [7, 15]] <= 1.0))

    def test_libero_keys_against_a_g1_server_fail_loudly(self) -> None:
        # The delivered-Orin failure mode: a client sending LIBERO-shaped keys
        # must be rejected with the served contract in the message, not served a
        # plausible-looking action built from the wrong cameras.
        client = websocket_client_policy.WebsocketClientPolicy(
            "127.0.0.1", self.server.port
        )
        try:
            with self.assertRaises(Exception) as caught:
                client.infer(
                    {
                        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
                        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
                        "observation/state": np.zeros(G1_ROBOT_DIM, dtype=np.float32),
                        "prompt": "pick up the cup",
                    }
                )
        finally:
            client._ws.close()
        message = str(caught.exception)
        self.assertIn("images/cam_high", message)
        self.assertIn("missing observation keys", message)


class ImageSlotOrderTest(unittest.TestCase):
    """``image_keys`` order binds a camera to a model view slot."""

    def test_reordering_image_keys_reorders_the_stacked_views(self) -> None:
        model = MockModel(num_views=3)
        reversed_keys = tuple(reversed(G1_CAMERAS))
        input_pipeline, output_pipeline = Pi05Policy.default_pipelines(
            model,
            tokenizer=RecordingTokenizer(),
            unnormalizer=Unnormalizer(
                q01=[-1.0] * MODEL_DIM, q99=[1.0] * MODEL_DIM, dims=MODEL_DIM, eps=0.0
            ),
            image_keys=reversed_keys,
            state_key=G1_STATE_KEY,
        )
        policy = Pi05Policy(
            model,
            input_pipeline=input_pipeline,
            output_pipeline=output_pipeline,
            image_keys=reversed_keys,
            state_key=G1_STATE_KEY,
            action_dim=MODEL_DIM,
        )
        policy.infer(g1_observation(np.zeros(G1_ROBOT_DIM, dtype=np.float32)))
        # Same observation, reversed config -> reversed slots, and nothing warns.
        # That silence is why presets pair each wire key with its slot name.
        stacked = model.images[0]
        for slot, fill in enumerate((30, 20, 10)):
            self.assertEqual(int(stacked[slot].max()), fill, f"slot {slot}")

    def test_view_count_must_match_the_checkpoint(self) -> None:
        model = MockModel(num_views=3)
        with self.assertRaises(ValueError) as caught:
            Pi05Policy.default_pipelines(
                model,
                tokenizer=RecordingTokenizer(),
                unnormalizer=Unnormalizer(q01=[-1.0], q99=[1.0], dims=1, eps=0.0),
                image_keys=("observation/image", "observation/wrist_image"),
            )
        message = str(caught.exception)
        self.assertIn("3 camera views", message)
        # Serving fewer cameras is legitimate; the error must name the way to do
        # it rather than leaving the operator to pad with black frames.
        self.assertIn("num_views=2", message)

    def test_extra_image_keys_are_rejected_without_offering_a_fix(self) -> None:
        # A checkpoint cannot grow a camera, so there is no num_views to suggest.
        model = MockModel(num_views=1)
        with self.assertRaises(ValueError) as caught:
            Pi05Policy.default_pipelines(
                model,
                tokenizer=RecordingTokenizer(),
                unnormalizer=Unnormalizer(q01=[-1.0], q99=[1.0], dims=1, eps=0.0),
                image_keys=("observation/image", "observation/wrist_image"),
            )
        message = str(caught.exception)
        self.assertNotIn("num_views=", message)
        self.assertIn("cannot serve more cameras", message)


    def test_num_views_disagreeing_with_a_passed_in_model_is_rejected(self) -> None:
        # An already-loaded handle has its view count baked in, so honouring the
        # argument is impossible; ignoring it would serve a shape nobody asked for.
        with self.assertRaises(ValueError) as caught:
            Pi05Policy.from_pretrained(
                "/nonexistent", model=MockModel(num_views=3), num_views=2
            )
        self.assertIn("pass num_views to the load call", str(caught.exception))


class ConventionsAreTheirOwnLayerTest(unittest.TestCase):
    """A recording dialect is not a robot's property, so it has its own module.

    The G1's wire keys used to live in ``processors/robots/unitree_g1.py`` — a
    *dataset* fact defined inside a *body*'s processing steps, so re-recording the
    same robot meant editing its steps. They now live in :mod:`apxinf.conventions`,
    which depends on neither a body nor a policy implementation.
    """

    def test_the_shipped_conventions_are_registered_under_their_names(self) -> None:
        self.assertIs(get_convention("libero"), LIBERO_CONVENTION)
        self.assertIs(get_convention("unitree_g1"), G1_CONVENTION)
        for name in ("libero", "unitree_g1"):
            self.assertIn(name, available_conventions())

    def test_a_preset_reuses_the_convention_object_rather_than_restating_it(self) -> None:
        # If a preset copied the keys instead, the two could drift and only a
        # client would notice — on the wire, silently.
        self.assertIs(get_robot_preset("unitree_g1").convention, G1_CONVENTION)
        self.assertIs(get_robot_preset("franka_libero").convention, LIBERO_CONVENTION)

    def test_the_g1_steps_no_longer_hold_the_g1s_wire_keys(self) -> None:
        # The regression to catch: a body's processing steps naming a dataset's
        # keys again. Checked on the module's exports, not by grepping prose.
        from apxinf.processors.robots import unitree_g1 as g1_steps

        self.assertFalse(hasattr(g1_steps, "G1_CAMERAS"))
        self.assertFalse(hasattr(g1_steps, "G1_STATE_KEY"))
        for step in (UnitreeG1DecodeState, UnitreeG1AbsoluteActions):
            with self.subTest(step=step.__name__):
                default = inspect.signature(step).parameters["state_key"].default
                self.assertIs(default, inspect.Parameter.empty)

    def test_conventions_import_no_body_and_no_policy_implementation(self) -> None:
        # The layering rule: a dialect may read the model's VIEW_SLOTS vocabulary
        # and nothing else. Anything more and the third axis is coupled again.
        source = pathlib.Path(conventions_module.__file__).read_text()
        for forbidden in ("Embodiment", "RobotPreset", "Pi05Policy", "AutoPolicy"):
            with self.subTest(name=forbidden):
                self.assertNotIn(f"import {forbidden}", source)
                self.assertNotIn(f"{forbidden}(", source)


class RegistrationFromOutsideTest(unittest.TestCase):
    """A third-party robot or dialect must not have to patch our files.

    Both registries are process-local dicts, so each test restores them; the
    point under test is the guard rails, not the mutation.
    """

    def setUp(self) -> None:
        self._presets = dict(ROBOT_PRESETS)
        self._aliases = dict(ROBOT_ALIASES)
        self._conventions = dict(CONVENTIONS)

    def tearDown(self) -> None:
        ROBOT_PRESETS.clear()
        ROBOT_PRESETS.update(self._presets)
        ROBOT_ALIASES.clear()
        ROBOT_ALIASES.update(self._aliases)
        CONVENTIONS.clear()
        CONVENTIONS.update(self._conventions)

    def _preset(self, name: str = "myarm_mydataset") -> RobotPreset:
        return RobotPreset(
            name=name,
            embodiment=Embodiment(name="myarm", num_cameras=1, action_dim=6),
            convention=Convention(name="mydataset", image_keys=("rgb",), state_key="q"),
        )

    def test_a_registered_preset_is_reachable_as_a_robot_flag(self) -> None:
        preset = register_robot_preset(self._preset(), aliases=("myarm",))
        self.assertIs(get_robot_preset("myarm_mydataset"), preset)
        self.assertIs(get_robot_preset("myarm"), preset)
        self.assertIn("myarm_mydataset", available_robots())
        self.assertIn("myarm", available_robots(include_aliases=True))

    def test_re_registering_a_name_needs_an_explicit_replace(self) -> None:
        # A silent overwrite would change what a launch command already in
        # production resolves to — the invisible failure --robot exists to prevent.
        register_robot_preset(self._preset())
        with self.assertRaises(ValueError) as caught:
            register_robot_preset(self._preset())
        self.assertIn("replace=True", str(caught.exception))
        replacement = self._preset()
        self.assertIs(
            register_robot_preset(replacement, replace=True),
            get_robot_preset("myarm_mydataset"),
        )

    def test_an_alias_may_never_shadow_a_canonical_preset(self) -> None:
        # Even with replace=True: an alias winning over a real preset would
        # silently redirect a --robot spelling that already names something.
        with self.assertRaises(ValueError) as caught:
            register_robot_preset(self._preset(), aliases=("unitree_g1",), replace=True)
        self.assertIn("unitree_g1", str(caught.exception))
        self.assertIs(get_robot_preset("unitree_g1").embodiment.builder, build_unitree_g1_policy)

    def test_moving_an_existing_alias_needs_an_explicit_replace(self) -> None:
        with self.assertRaises(ValueError) as caught:
            register_robot_preset(self._preset(), aliases=("libero",))
        self.assertIn("franka_libero", str(caught.exception))
        self.assertIs(get_robot_preset("libero"), get_robot_preset("franka_libero"))
        register_robot_preset(self._preset(), aliases=("libero",), replace=True)
        self.assertEqual(get_robot_preset("libero").name, "myarm_mydataset")

    def test_a_failed_registration_leaves_both_tables_untouched(self) -> None:
        # The alias check runs before either dict is written, so a rejected call
        # cannot half-register a preset under its canonical name.
        with self.assertRaises(ValueError):
            register_robot_preset(self._preset(), aliases=("unitree_g1",))
        self.assertNotIn("myarm_mydataset", ROBOT_PRESETS)

    def test_registering_something_that_is_not_a_preset_is_a_type_error(self) -> None:
        with self.assertRaises(TypeError):
            register_robot_preset(Convention(name="x", image_keys=("a",), state_key="q"))

    def test_a_registered_convention_pairs_with_a_shipped_body(self) -> None:
        droid_like = register_convention(
            Convention(
                name="franka_droid_like",
                image_keys=("observation/exterior_image_1_left", "observation/wrist_image_left"),
                state_key="observation/joint_position",
            )
        )
        self.assertIs(get_convention("franka_droid_like"), droid_like)
        # The payoff of the split: a new dialect on an existing body needs no
        # new Embodiment, no builder, and no edit to either shipped module.
        preset = RobotPreset(
            name="franka_droid_like",
            embodiment=get_robot_preset("franka_libero").embodiment,
            convention=droid_like,
        )
        self.assertEqual(preset.action_dim, 7)
        self.assertEqual(preset.state_key, "observation/joint_position")

    def test_re_registering_a_convention_needs_an_explicit_replace(self) -> None:
        with self.assertRaises(ValueError) as caught:
            register_convention(
                Convention(name="libero", image_keys=("a", "b"), state_key="q")
            )
        self.assertIn("replace=True", str(caught.exception))
        self.assertIs(get_convention("libero"), LIBERO_CONVENTION)

    def test_registering_something_that_is_not_a_convention_is_a_type_error(self) -> None:
        with self.assertRaises(TypeError):
            register_convention(Embodiment(name="myarm", num_cameras=1))


class StateKeyIsRequiredWhenItIsReadTest(unittest.TestCase):
    """``state_key`` has no default, and is demanded exactly when state is read.

    The model layer used to default it to LIBERO's ``observation/state``, the same
    dataset-in-the-model coupling the camera keys had. Dropping the default leaves
    an asymmetry with ``image_keys``: a wrong camera key raises ``KeyError`` on the
    first inference, but a missing state key is silent — the tokenizer just
    injects nothing and the policy publishes ``state_key=null`` while proprioception
    quietly disappears. So it is required whenever the chain discretizes state, and
    allowed to stay ``None`` only when state is deliberately dropped.
    """

    def test_the_model_layer_defaults_no_state_key(self) -> None:
        # Asserted on the signatures, not by grepping the source, so the prose
        # explaining the old LIBERO default does not trip the check.
        for func in (
            pi05_module.Pi05Policy.__init__,
            pi05_module.Pi05Policy.default_pipelines,
            pi05_module.Pi05Policy.from_pretrained,
            pi05_module.Pi05Policy.from_random,
            Tokenize.__init__,
        ):
            with self.subTest(entry_point=func.__qualname__):
                self.assertIsNone(inspect.signature(func).parameters["state_key"].default)

    def test_a_discretizing_tokenizer_without_a_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            Tokenize(RecordingTokenizer(), None, None)
        self.assertIn("state_key", str(caught.exception))

    def test_a_dropping_tokenizer_without_a_key_is_fine(self) -> None:
        # discrete_state=False means state is dropped on purpose; demanding a key
        # for a value nothing reads would be noise.
        tokenizer = RecordingTokenizer()
        tokenizer.discrete_state = False
        self.assertIsNone(Tokenize(tokenizer, None, None).state_key)

    def test_a_policy_whose_chain_reads_state_is_rejected_without_a_key(self) -> None:
        model = MockModel(num_views=2)
        input_pipeline, output_pipeline = Pi05Policy.default_pipelines(
            model,
            tokenizer=RecordingTokenizer(),
            unnormalizer=_identity_unnormalizer(),
            state_key="state",
        )
        with self.assertRaises(ValueError) as caught:
            Pi05Policy(
                model,
                input_pipeline=input_pipeline,
                output_pipeline=output_pipeline,
                state_key=None,
            )
        self.assertIn("state_key", str(caught.exception))

    def test_from_pretrained_refuses_discrete_state_without_a_key(self) -> None:
        # Checked before the checkpoint is touched, so the message can name the
        # flags the caller actually passed rather than failing deep in a load.
        with self.assertRaises(ValueError) as caught:
            Pi05Policy.from_pretrained("/nonexistent", discrete_state=True)
        message = str(caught.exception)
        self.assertIn("state_key", message)
        self.assertIn("discrete_state=False", message)

    def test_the_published_contract_may_say_no_state_key(self) -> None:
        # A policy that drops state has no state key to publish, and null is the
        # honest answer — a client reading metadata sees that state is unused.
        policy = build_g1_mock_policy()
        self.assertEqual(policy.metadata["state_key"], G1_STATE_KEY)
        tokenizer = RecordingTokenizer()
        tokenizer.discrete_state = False
        model = MockModel(num_views=2)
        input_pipeline, output_pipeline = Pi05Policy.default_pipelines(
            model, tokenizer=tokenizer, unnormalizer=_identity_unnormalizer()
        )
        stateless = Pi05Policy(
            model, input_pipeline=input_pipeline, output_pipeline=output_pipeline
        )
        self.assertIsNone(stateless.metadata["state_key"])
        self.assertFalse(stateless.metadata["discrete_state"])


class NormalizationDtypeTest(unittest.TestCase):
    """A G1 checkpoint has to be normalized in float64, the way openpi does it.

    openpi parses ``norm_stats.json`` with the stdlib JSON decoder and keeps the
    result in float64. Subtracting a float64 ``q01`` from a robot's float32 state
    promotes, so openpi's whole G1 chain -- normalize the state, discretize it
    into the prompt, unnormalize the action -- runs in float64 whatever the robot
    sent. Ours follows the input dtype unless told otherwise.

    The gap is ~3e-7. On the output side that is nothing: a joint angle nobody
    can measure. On the *input* side the very next operation compares the
    normalized state against bin edges 1/128 apart, so an element sitting near
    one lands in a different bin, writes a different number into the prompt
    string, produces different token ids, and sends the rollout somewhere else.

    Under the robot/model split the pin is not something the adapter reaches in
    and applies -- it is a load flag the body declares (``norm_dtype`` in
    :attr:`Embodiment.builder_kwargs`) and the model's loader honours. These
    tests cover that as two links plus the reason it is load-bearing:
    the preset asks for it, ``from_pretrained`` applies it, and float32 really
    does cross a bin edge.
    """

    # --- link 1: the body declares it, and the loader accepts the keyword -----

    def test_the_g1_preset_asks_the_loader_for_float64(self) -> None:
        captured: dict = {}

        def load(model_dir, *, model_type=None, **kwargs):
            # Bind against the real signature, as UnitreeG1AdapterTest does: a
            # flag the preset spells but ``from_pretrained`` does not accept is a
            # TypeError the first time anyone serves a G1 checkpoint, and a
            # kwargs-swallowing mock would never show it.
            inspect.signature(Pi05Policy.from_pretrained).bind(model_dir, **kwargs)
            captured.update(kwargs)
            return _StubComposable()

        with mock.patch.object(
            robot_adapter, "AutoPolicy", types.SimpleNamespace(from_pretrained=load)
        ):
            build_robot_policy("unitree_g1", "/nowhere")

        self.assertEqual(captured.get("norm_dtype"), "float64")

    def test_libero_does_not_ask_for_it(self) -> None:
        # The pin is scoped to the body that needs it. LIBERO's numbers are
        # unchanged by this work, and a global default would have moved them.
        self.assertNotIn("norm_dtype", get_robot_preset("franka_libero").builder_kwargs)

    # --- link 2: the loader turns the flag into pinned normalizers ------------

    def _checkpoint(self) -> pathlib.Path:
        path = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(path, ignore_errors=True))
        (path / "norm_stats.json").write_text(
            json.dumps(
                {
                    "norm_stats": {
                        "actions": {
                            "q01": [-1.0] * MODEL_DIM,
                            "q99": [1.0] * MODEL_DIM,
                            "mean": [0.0] * MODEL_DIM,
                            "std": [1.0] * MODEL_DIM,
                        },
                        "state": {
                            "q01": [-1.0] * G1_ROBOT_DIM,
                            "q99": [1.0] * G1_ROBOT_DIM,
                            "mean": [0.0] * G1_ROBOT_DIM,
                            "std": [1.0] * G1_ROBOT_DIM,
                        },
                    }
                }
            )
        )
        (path / "tokenizer.model").write_bytes(b"not a real sentencepiece model")
        return path

    def _load(self, **kwargs) -> Pi05Policy:
        """Run the real ``from_pretrained``, stubbing only the tokenizer.

        Everything the pin touches -- reading norm_stats, building both
        normalizers, assembling the pipelines -- is the shipped code path. Only
        SentencePiece is replaced, because it is the one step that needs a real
        binary model file and it has nothing to do with dtypes.
        """
        with mock.patch.object(pi05_module, "PromptTokenizer", lambda *a, **k: RecordingTokenizer()):
            return Pi05Policy.from_pretrained(
                self._checkpoint(),
                model=MockModel(num_views=len(G1_CAMERAS)),
                discrete_state=True,
                state_key=G1_STATE_KEY,
                image_keys=G1_CAMERAS,
                **kwargs,
            )

    def test_norm_dtype_pins_both_normalizers(self) -> None:
        policy = self._load(norm_dtype="float64")
        self.assertEqual(
            policy.input_pipeline["tokenize"].state_normalizer.dtype, np.dtype("float64")
        )
        self.assertEqual(
            policy.output_pipeline["unnormalize"].unnormalizer.dtype, np.dtype("float64")
        )

    def test_the_default_still_follows_the_input(self) -> None:
        # Without the flag nothing is pinned, so this work changes no numbers for
        # any checkpoint that does not ask -- which is what keeps LIBERO's
        # float32 results bit-identical.
        policy = self._load()
        self.assertIsNone(policy.input_pipeline["tokenize"].state_normalizer.dtype)
        self.assertIsNone(policy.output_pipeline["unnormalize"].unnormalizer.dtype)

    # --- why it is load-bearing ----------------------------------------------

    def test_float32_normalization_lands_in_a_different_bin(self) -> None:
        """The reason for the pin, stated as a number rather than an assertion of faith."""
        from apxinf.processors import discretize_state

        rng = np.random.default_rng(11)
        q01 = rng.uniform(-2.6, -0.4, G1_ROBOT_DIM)
        q99 = q01 + rng.uniform(0.7, 4.1, G1_ROBOT_DIM)

        # States that normalize *onto the bin edges*, by inverting openpi's
        # normalize. Uniformly sampled states would not show anything: a 3e-7
        # shift only changes a bin for an element already within 3e-7 of an edge
        # 1/128 apart, which is a ~4e-5 chance each. That rarity is not safety --
        # a G1 has 16 joints polled at 30 Hz, so "one in 25000 elements" is a
        # corrupted prompt every minute or so, on a random joint, with no symptom
        # other than the rollout going somewhere else.
        edges = np.linspace(-1.0, 1.0, 257)[:-1]
        targets = np.resize(edges, (edges.size // G1_ROBOT_DIM, G1_ROBOT_DIM))
        states = (q01 + (targets + 1.0) / 2.0 * (q99 - q01 + 1e-6)).astype(np.float32)

        pinned = Normalizer(q01=q01, q99=q99, dims=G1_ROBOT_DIM, dtype="float64")
        loose = Normalizer(q01=q01, q99=q99, dims=G1_ROBOT_DIM)

        wide = np.stack([pinned(s) for s in states])
        narrow = np.stack([loose(s) for s in states])
        self.assertEqual(wide.dtype, np.float64)
        self.assertEqual(narrow.dtype, np.float32, "the unpinned default follows the input")

        # openpi's own arithmetic, for the record: float64 stats promote the state.
        reference = (states - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        np.testing.assert_array_equal(wide, reference)

        moved = int((discretize_state(wide) != discretize_state(narrow)).sum())
        self.assertGreater(
            moved, 0, "if float32 never crossed a bin edge the pin would be cosmetic"
        )


if __name__ == "__main__":
    unittest.main()
