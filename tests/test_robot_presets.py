"""Robot presets, wire-key resolution, and the G1 serving path end-to-end.

Covers the layer that decides *which keys a client must send*, which is where a
deployment silently degrades rather than fails: a checkpoint served under the
wrong embodiment still returns well-shaped actions.

Three groups:

* :class:`KeyResolutionTest` — ``lookup_key`` / ``has_key`` / ``set_key``, the
  flat-vs-nested wire layouts (``"observation/image"`` vs
  ``obs["images"]["cam_high"]``).
* :class:`RobotPresetTest` — the preset table's invariants and overrides.
* :class:`UnitreeG1ServingTest` — an **unmodified openpi G1 observation** driven
  through the real websocket transport into a G1-wired policy, asserting camera
  →slot binding, state routing, and the delta→absolute / 32→16 output chain.

Runs offline against a mock model; no CUDA, no checkpoint.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import threading
import unittest

import numpy as np
from openpi_client import websocket_client_policy
import websockets.asyncio.server as websocket_server

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "python" / "apxinf"))
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

from apxinf import Pi05Policy  # noqa: E402
from apxinf.processors import Pipeline, PromptTokenizer, Unnormalizer  # noqa: E402
from apxinf.processors.robots.unitree_g1 import (  # noqa: E402
    G1_CAMERAS,
    G1_ROBOT_DIM,
    G1_STATE_KEY,
    UnitreeG1AbsoluteActions,
    UnitreeG1DecodeState,
    UnitreeG1EncodeActions,
)
from apxinf.processors.transforms import (  # noqa: E402
    Unnormalize,
    has_key,
    lookup_key,
    set_key,
)
from apxinf.robots.presets import (  # noqa: E402
    ROBOT_PRESETS,
    VIEW_SLOTS,
    RobotPreset,
    available_robots,
    get_robot_preset,
)
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

    def test_slots_must_be_a_prefix_of_the_model_view_slots(self) -> None:
        # A checkpoint fills view slots from 0 up, so declaring "base + right
        # wrist" is not expressible and must be rejected at table-definition
        # time rather than mis-binding a camera at inference time.
        with self.assertRaises(ValueError):
            RobotPreset(
                name="bad",
                slots=(("base_0_rgb", "a"), ("right_wrist_0_rgb", "b")),
                state_key="state",
            )

    def test_duplicate_camera_keys_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RobotPreset(
                name="bad",
                slots=(("base_0_rgb", "a"), ("left_wrist_0_rgb", "a")),
                state_key="state",
            )

    def test_unknown_robot_names_the_known_ones(self) -> None:
        with self.assertRaises(KeyError) as caught:
            get_robot_preset("g1")
        for name in available_robots(include_aliases=True):
            self.assertIn(name, str(caught.exception))


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

    Mirrors ``build_unitree_g1_policy`` but injects the model and an identity
    unnormalizer, so the assertions isolate the adapter's arithmetic from
    checkpoint norm_stats.
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
    input_pipeline = input_pipeline.insert_before(
        "tokenize", ("g1_decode_state", UnitreeG1DecodeState(G1_STATE_KEY))
    )
    output_pipeline = Pipeline(
        [
            ("unnormalize", Unnormalize(identity)),
            ("g1_absolute", UnitreeG1AbsoluteActions(G1_STATE_KEY)),
            ("g1_encode", UnitreeG1EncodeActions()),
        ]
    )
    policy = Pi05Policy(
        model,
        input_pipeline=input_pipeline,
        output_pipeline=output_pipeline,
        image_keys=G1_CAMERAS,
        state_key=G1_STATE_KEY,
        action_dim=G1_ROBOT_DIM,
        metadata={"robot": "unitree_g1", "protocol": "openpi.websocket_policy"},
    )
    policy.tokenizer = tokenizer
    return policy


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


if __name__ == "__main__":
    unittest.main()
