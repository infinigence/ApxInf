from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import threading
import urllib.request
import unittest

import numpy as np
from openpi_client import websocket_client_policy
import websockets.asyncio.server as websocket_server


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "python" / "apxinf"))
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

from apxinf import Pi05Policy  # noqa: E402
from apxinf.processors import PromptTokenizer, Unnormalizer  # noqa: E402
from apxinf.serving import WebsocketPolicyServer  # noqa: E402
from apxinf.serving.websocket import health_check  # noqa: E402


HORIZON = 10
MODEL_DIM = 32
LIBERO_DIM = 7
IMAGE_SIZE = 224
NUM_VIEWS = 2


class MockModel:
    """In-process ``BareModel`` stand-in: normalized action is all-zeros.

    Replaces the old subprocess ``FakeEngine``; the server now calls the model
    in-process through ``Pi05Policy`` instead of over a stdio pipe.
    """

    def __init__(self) -> None:
        self.action_horizon = HORIZON
        self.action_dim = MODEL_DIM
        self.num_views = NUM_VIEWS
        self.image_size = IMAGE_SIZE
        self.max_token_len = 200
        self.images: list[np.ndarray] = []
        self.noises: list[np.ndarray] = []
        self.sampling_draw = 0

    def infer_rgb(self, rgb_u8, layout, token_ids, noise=None):
        assert layout == "nhwc"
        self.images.append(np.asarray(rgb_u8).copy())
        if noise is None:
            noise = np.full((HORIZON, MODEL_DIM), self.sampling_draw, dtype=np.float32)
            self.sampling_draw += 1
        self.noises.append(np.asarray(noise).copy())
        return np.zeros((HORIZON, MODEL_DIM), dtype=np.float32)

    def infer_patches(self, patches, token_ids, noise):  # pragma: no cover - unused
        return np.zeros((HORIZON, MODEL_DIM), dtype=np.float32)


class ConstTokenizer(PromptTokenizer):
    """A tokenizer needing no SentencePiece model (bypasses ``__init__``)."""

    def __init__(self, tokens=(2, 108, 12)) -> None:
        self.max_token_len = 200
        self.discrete_state = False
        self._tokens = np.asarray(tokens, dtype=np.uint32)

    def __call__(self, prompt, state=None):
        return self._tokens


def build_policy() -> Pi05Policy:
    # q99 - q01 == 2 everywhere, so unnormalize(0) == q01 + 1 == arange(7) + 1.
    q01 = np.arange(LIBERO_DIM, dtype=np.float32)
    q99 = q01 + 2
    model = MockModel()
    input_pipeline, output_pipeline = Pi05Policy.default_pipelines(
        model, tokenizer=ConstTokenizer(), unnormalizer=Unnormalizer(q01=q01, q99=q99)
    )
    return Pi05Policy(
        model,
        input_pipeline=input_pipeline,
        output_pipeline=output_pipeline,
        metadata={"precision": "int8", "protocol": "openpi.websocket_policy"},
    )


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


class WebsocketServerCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = build_policy()
        self.server = RunningServer(self.policy)

    def tearDown(self) -> None:
        self.server.close()
        self.policy.close()

    def test_official_openpi_client_round_trip_and_health(self) -> None:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(
            f"http://127.0.0.1:{self.server.port}/healthz", timeout=5
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"OK\n")

        client = websocket_client_policy.WebsocketClientPolicy(
            "127.0.0.1", self.server.port
        )
        metadata = client.get_server_metadata()
        # Policy metadata (model_type, shapes) merged with the server-injected tags.
        self.assertEqual(metadata["precision"], "int8")
        self.assertEqual(metadata["model_type"], "pi05")
        observation = {
            "observation/image": np.full((3, 224, 224), 0.5, dtype=np.float32),
            "observation/wrist_image": np.full((112, 224, 3), 64, dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "pick_up\nthe bowl",
        }
        try:
            first = client.infer(observation)
            second = client.infer(observation)
        finally:
            client._ws.close()

        expected = np.broadcast_to(np.arange(7) + 1, (10, 7))
        for response in (first, second):
            actions = response["actions"]
            self.assertEqual(actions.shape, (10, 7))
            self.assertEqual(actions.dtype, np.dtype("float32"))
            np.testing.assert_allclose(actions, expected, rtol=0, atol=1e-5)
            self.assertIn("infer_ms", response["policy_timing"])
            self.assertIn("infer_ms", response["server_timing"])
            # Library-only keys must never reach the wire.
            self.assertNotIn("normalized_actions", response)
            self.assertNotIn("token_ids", response)
            self.assertNotIn("noise", response)
        self.assertNotIn("prev_total_ms", first["server_timing"])
        self.assertIn("prev_total_ms", second["server_timing"])

        # The in-process model saw two NHWC calls, one row per configured camera.
        self.assertEqual(len(self.policy.model.images), 2)
        self.assertEqual(self.policy.model.images[0].shape, (NUM_VIEWS, 224, 224, 3))
        self.assertEqual(self.policy.model.images[0].dtype, np.dtype("uint8"))
        # The float 0.5 base image was parsed to uint8 127.
        self.assertTrue(np.all(self.policy.model.images[0][0] == 127))
        # Seeded noise advances between calls, so the two draws differ.
        self.assertFalse(
            np.array_equal(self.policy.model.noises[0], self.policy.model.noises[1])
        )


if __name__ == "__main__":
    unittest.main()
