"""Contract tests for scripts/pi05_remote_client.py — no GPU, no checkpoint.

Two things are under test. The first is the codec: the client embeds a copy of
``apxinf/serving/msgpack_numpy.py`` so that it runs on a host with no checkout,
and the only thing that makes that copy legal is that it is *byte-identical* —
so that is asserted, not assumed. The second is the client's behaviour against a
real ``WebsocketPolicyServer``: the dialect check at connect, the missing-key
error, the latent passthrough, the timeout that must close the socket, and the
reconnect that must revalidate the contract.

The server harness is written here rather than imported from
``tests/test_pi05_openpi_websocket.py`` because that module imports
``openpi_client`` at module scope, and this client exists precisely so a host
need not have that package.
"""

from __future__ import annotations

import asyncio
import threading
import time
import unittest

import numpy as np

import websockets.asyncio.server as websocket_server  # noqa: E402

from apxinf.serving import WebsocketPolicyServer  # noqa: E402
from apxinf.serving.msgpack_numpy import packer as library_packer  # noqa: E402
from apxinf.serving.websocket import health_check  # noqa: E402
from scripts import pi05_remote_client  # noqa: E402
from scripts.pi05_remote_client import (  # noqa: E402
    RemoteDialectError,
    RemotePolicy,
    RemotePolicyError,
    RemotePolicyTimeout,
    timing_segments,
)

HORIZON = 10
ACTION_DIM = 7
MODEL_DIM = 32
LIBERO_IMAGE_KEYS = ("observation/image", "observation/wrist_image")


# ── The codec ────────────────────────────────────────────────────────────────


def _codec_samples() -> list:
    """Values that pin the *shape* of the encoding, not just its round trip.

    Deterministic on purpose — no RNG — so a golden frame digest could not drift
    with a numpy version.
    """
    return [
        np.arange(3 * 4 * 5, dtype=np.uint8).reshape(4, 5, 3),
        np.linspace(-1.0, 1.0, 7, dtype=np.float32),
        np.arange(HORIZON * MODEL_DIM, dtype=np.float32).reshape(HORIZON, MODEL_DIM),
        np.float32(0.25),
        np.int64(-3),
        np.uint32(7),
        "put the bowl\non the plate",
        None,
        True,
        7,
        2.5,
        {"nested": {"x": np.zeros(2, dtype=np.float32)}},
        [np.zeros(2, dtype=np.uint8), "text"],
    ]


class CodecParityTest(unittest.TestCase):
    """The embedded codec must equal the library's, frame byte for frame byte."""

    def test_every_sample_encodes_to_identical_bytes(self):
        for value in _codec_samples():
            with self.subTest(value=type(value).__name__):
                self.assertEqual(
                    pi05_remote_client.packer().pack(value),
                    library_packer().pack(value),
                )

    def test_a_whole_observation_encodes_to_identical_bytes(self):
        observation = {
            "observation/image": np.arange(256 * 256 * 3, dtype=np.uint8).reshape(
                256, 256, 3
            ),
            "observation/wrist_image": np.full((256, 256, 3), 7, dtype=np.uint8),
            "observation/state": np.linspace(-1.0, 1.0, 8, dtype=np.float32),
            "prompt": "pick up the block",
            "noise": np.zeros((HORIZON, MODEL_DIM), dtype=np.float32),
        }
        ours = pi05_remote_client.packer().pack(observation)
        self.assertEqual(ours, library_packer().pack(observation))
        # ~400 KiB of camera frames really do go on the wire; a frame that
        # compressed or truncated would be a different length.
        self.assertGreater(len(ours), 380_000)

    def test_the_two_implementations_round_trip_each_other(self):
        """A frame packed by the library decodes through the embedded unpacker."""
        value = {"image": np.arange(12, dtype=np.uint8).reshape(3, 4)}
        decoded = pi05_remote_client.unpackb(library_packer().pack(value))
        np.testing.assert_array_equal(decoded["image"], value["image"])

    def test_upstream_openpi_agrees_when_it_is_installed(self):
        if not _module_available("openpi_client.msgpack_numpy"):
            self.skipTest("the upstream openpi_client package is not installed")
        from openpi_client.msgpack_numpy import packer as upstream_packer

        for value in _codec_samples():
            with self.subTest(value=type(value).__name__):
                self.assertEqual(
                    pi05_remote_client.packer().pack(value), upstream_packer().pack(value)
                )

    def test_unsupported_dtypes_are_refused_rather_than_mangled(self):
        for dtype in (np.dtype("V4"), np.dtype("O"), np.dtype("c8")):
            with self.subTest(dtype=str(dtype)):
                with self.assertRaises(ValueError):
                    pi05_remote_client.pack_numpy(np.zeros(2, dtype=dtype))


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


# ── Timing map ───────────────────────────────────────────────────────────────


class TimingSegmentsTest(unittest.TestCase):
    def test_the_map_is_the_bench_scripts_arithmetic(self):
        """`scripts/bench_pi05.py:251-261`, spelled out so the two cannot drift.

        That block is: model_ms = policy_timing["infer_ms"], server_total_ms =
        policy_timing["policy_ms"], server_compute_ms = server_timing["infer_ms"],
        then model = model_ms, server_processor = max(0, total - model) and
        transport = max(0, round_trip - compute).
        """
        response = {
            "policy_timing": {"infer_ms": 10.0, "policy_ms": 12.0},
            "server_timing": {"infer_ms": 11.0},
        }
        self.assertEqual(
            timing_segments(response, 20.0),
            {
                "round_trip": 20.0,
                "model": 10.0,
                "server_processor": 2.0,
                "transport": 9.0,
            },
        )

    def test_missing_fields_degrade_to_the_coarser_one(self):
        # A server that reports nothing still gets served: model falls to zero,
        # and the two coarse segments fall back to round_trip - model.
        self.assertEqual(
            timing_segments({}, 20.0),
            {
                "round_trip": 20.0,
                "model": 0.0,
                "server_processor": 0.0,
                "transport": 20.0,
            },
        )
        # policy_ms absent -> server_total_ms is infer_ms, so the processor
        # segment is zero rather than negative.
        self.assertEqual(
            timing_segments({"policy_timing": {"infer_ms": 5.0}}, 20.0)[
                "server_processor"
            ],
            0.0,
        )

    def test_the_segments_are_never_negative(self):
        response = {
            "policy_timing": {"infer_ms": 30.0, "policy_ms": 10.0},
            "server_timing": {"infer_ms": 100.0},
        }
        segments = timing_segments(response, 5.0)
        for name, value in segments.items():
            with self.subTest(segment=name):
                self.assertGreaterEqual(value, 0.0)


# ── The server harness ───────────────────────────────────────────────────────


class FakePolicy:
    """A duck-typed ``apxinf.Policy``: the shape the server needs, nothing more.

    ``actions`` are derived from the latent so that a fixed latent is
    observable in the response and a fresh one is observable as a difference.
    """

    def __init__(
        self,
        *,
        model_variant: str = "bf16",
        image_keys=LIBERO_IMAGE_KEYS,
        state_key=None,
        discrete_state: bool = False,
        delay_s: float = 0.0,
        error: BaseException | None = None,
    ) -> None:
        self.image_keys = tuple(image_keys)
        self.state_key = state_key
        self.prompt_key = "prompt"
        self.delay_s = delay_s
        self.error = error
        self.noises: list = []
        self.draws = 0
        self.metadata = {
            "protocol": "openpi.websocket_policy",
            "model_type": "pi05",
            "model_variant": model_variant,
            "action_horizon": HORIZON,
            "action_dim": ACTION_DIM,
            "model_action_dim": MODEL_DIM,
            "num_views": len(self.image_keys),
            "image_size": [224, 224],
            "image_keys": list(self.image_keys),
            "state_key": state_key,
            "prompt_key": "prompt",
            "discrete_state": discrete_state,
            "normalization": {"action": "quantile/resolved"},
        }

    def infer(self, observation, *, noise=None):
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        latent = observation.get("noise")
        if latent is None:
            latent = np.full((HORIZON, MODEL_DIM), float(self.draws), dtype=np.float32)
            self.draws += 1
        latent = np.asarray(latent, dtype=np.float32)
        self.noises.append(latent.copy())
        actions = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
        actions[:, :] = latent[:, :1]
        return {"actions": actions, "timing": {"model_ms": 1.5, "total_ms": 2.5}}

    def close(self) -> None:  # pragma: no cover - the server owns the policy
        pass


class RunningServer:
    """Serve a policy on a background thread on a real port."""

    def __init__(self, policy, *, port: int = 0) -> None:
        self._policy_server = WebsocketPolicyServer(policy, "127.0.0.1", port)
        self.port = port
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise TimeoutError("the test websocket server did not start")

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)

        async def start() -> None:
            self._server = await websocket_server.serve(
                self._policy_server.handler,
                "127.0.0.1",
                self.port,
                compression=None,
                max_size=None,
                process_request=health_check,
            )
            self.port = self._server.sockets[0].getsockname()[1]
            self._ready.set()

        self._loop.run_until_complete(start())
        self._loop.run_forever()

    def close(self) -> None:
        # Idempotent: the reconnect tests stop a server by hand and then let the
        # cleanup stop it again.
        if self._closed:
            return
        self._closed = True

        async def stop() -> None:
            self._server.close()
            await self._server.wait_closed()

        asyncio.run_coroutine_threadsafe(stop(), self._loop).result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        self._loop.close()


def _observation(**overrides) -> dict:
    observation = {
        LIBERO_IMAGE_KEYS[0]: np.zeros((8, 8, 3), dtype=np.uint8),
        LIBERO_IMAGE_KEYS[1]: np.ones((8, 8, 3), dtype=np.uint8),
        "prompt": "pick up the block",
    }
    observation.update(overrides)
    return observation


class RemotePolicyAddressTest(unittest.TestCase):
    """Which address the object says it is talking to.

    A client built from a `uri` used to keep the keyword defaults for its host
    and port, so `health()` — which builds its URL from those — probed
    `127.0.0.1` while the websocket went to the URI's host. On a split deployment
    that is the difference between a working server and `main()` exiting 2 with
    UNREACHABLE.

    The constructor connects, so these run against the loopback server and make
    the *keywords* the wrong answer: passing them something that is not listening
    proves the URI is what decided.
    """

    def setUp(self) -> None:
        self.server = RunningServer(FakePolicy())
        self.addCleanup(self.server.close)

    def _client(self, *args, **kwargs) -> RemotePolicy:
        client = RemotePolicy(*args, **kwargs)
        self.addCleanup(client.close)
        return client

    def test_a_uri_decides_the_host_and_port(self):
        client = self._client(
            "10.0.0.5", 8123, uri=f"ws://127.0.0.1:{self.server.port}"
        )
        self.assertEqual(client._host, "127.0.0.1")
        self.assertEqual(client._port, self.server.port)

    def test_the_keywords_still_work_when_no_uri_is_given(self):
        client = self._client("127.0.0.1", self.server.port)
        self.assertEqual(client._host, "127.0.0.1")
        self.assertEqual(client._port, self.server.port)

    def test_a_uri_wins_over_the_keywords(self):
        """`uri` is what `__init__` connects to, so it is what health must ask."""
        client = self._client(
            "not-a-real-host.invalid", 1, uri=f"ws://127.0.0.1:{self.server.port}"
        )
        self.assertEqual(client._host, "127.0.0.1")
        self.assertEqual(client._port, self.server.port)

    def test_the_health_url_is_the_one_that_answers(self):
        """The regression this class exists for, stated end to end."""
        client = self._client(
            "10.0.0.5", 8123, uri=f"ws://127.0.0.1:{self.server.port}"
        )
        self.assertTrue(client.health(timeout_s=5.0))


class RemotePolicyLoopbackTest(unittest.TestCase):
    """The client against a real server, with no device involved."""

    def setUp(self) -> None:
        self.policy = FakePolicy()
        self.server = RunningServer(self.policy)
        self.addCleanup(self._close_server, self.server)

    @staticmethod
    def _close_server(server) -> None:
        server.close()

    def _client(self, **kwargs) -> RemotePolicy:
        client = RemotePolicy("127.0.0.1", self.server.port, **kwargs)
        self.addCleanup(client.close)
        return client

    def test_metadata_is_read_on_connect(self):
        client = self._client()
        self.assertEqual(client.metadata["model_type"], "pi05")
        self.assertEqual(client.image_keys, LIBERO_IMAGE_KEYS)
        self.assertEqual(client.action_horizon, HORIZON)
        self.assertEqual(client.action_dim, ACTION_DIM)
        self.assertEqual(client.model_action_dim, MODEL_DIM)
        self.assertIsNone(client.state_key)
        self.assertTrue(client.health())

    def test_a_declared_dialect_that_the_server_does_not_serve_fails_at_connect(self):
        with self.assertRaises(RemoteDialectError) as caught:
            self._client(image_keys=("base_0_rgb", "left_wrist_0_rgb"))
        message = str(caught.exception)
        self.assertIn("observation/image", message)
        self.assertIn("base_0_rgb", message)

    def test_a_declared_state_key_the_server_serves_differently_fails_at_connect(self):
        server = RunningServer(FakePolicy(state_key="observation/state"))
        self.addCleanup(server.close)
        with self.assertRaises(RemoteDialectError):
            RemotePolicy("127.0.0.1", server.port, state_key="state")

    def test_a_server_that_drops_state_is_accepted_and_says_so(self):
        with self.assertLogs("pi05_remote_client", level="WARNING") as captured:
            client = self._client(state_key="observation/state")
        self.assertIsNone(client.state_key)
        self.assertTrue(any("drops state" in line for line in captured.output))

    def test_an_expected_variant_that_does_not_match_fails_at_connect(self):
        with self.assertRaises(RemoteDialectError):
            self._client(expect_model_variant="fp8_static")

    def test_the_declared_dialect_is_accepted_when_it_agrees(self):
        client = self._client(
            image_keys=LIBERO_IMAGE_KEYS, expect_model_variant="bf16"
        )
        self.assertEqual(client.image_keys, LIBERO_IMAGE_KEYS)

    def test_a_round_trip_returns_a_writable_float32_chunk(self):
        client = self._client()
        result = client.infer(_observation())
        actions = result["actions"]
        self.assertEqual(actions.shape, (HORIZON, ACTION_DIM))
        self.assertEqual(actions.dtype, np.dtype("float32"))
        # Decoded arrays view the frame's bytes and are read-only; a control
        # loop that scales a chunk in place must not hit that.
        self.assertTrue(actions.flags.writeable)
        actions[:, 0] = 1.0
        self.assertIn("model_ms", result["timing"])
        self.assertGreater(result["timing"]["round_trip_ms"], 0.0)

    def test_a_missing_observation_key_is_named(self):
        client = self._client()
        with self.assertRaises(RemotePolicyError) as caught:
            client.infer({"prompt": "pick up the block", LIBERO_IMAGE_KEYS[0]: 1})
        self.assertIn(LIBERO_IMAGE_KEYS[1], str(caught.exception))

    def test_the_latent_reaches_the_policy_unchanged(self):
        client = self._client()
        noise = np.arange(HORIZON * MODEL_DIM, dtype=np.float32).reshape(
            HORIZON, MODEL_DIM
        )
        client.infer(_observation(), noise=noise)
        np.testing.assert_array_equal(self.policy.noises[-1], noise)
        # The caller's observation is not mutated on the way out.
        observation = _observation()
        client.infer(observation, noise=noise)
        self.assertNotIn("noise", observation)

    def test_a_wrong_latent_shape_is_refused_client_side(self):
        """The server raises on a bad latent and drops the connection, so the
        shape is checked here where the message can name it."""
        client = self._client()
        with self.assertRaises(ValueError) as caught:
            client.infer(_observation(), noise=np.zeros((HORIZON, ACTION_DIM), np.float32))
        self.assertIn("model_action_dim", str(caught.exception))

    def test_a_noise_seed_fixes_one_latent_for_every_request(self):
        """One latent, drawn once and reused.

        A seeded *stream* would be reproducible across a whole sequence but
        would not make two calls comparable on their own, which is the only
        reason to fix a latent at all.
        """
        client = self._client(noise_seed=11)
        client.infer(_observation())
        client.infer(_observation())
        first, second = self.policy.noises[-2], self.policy.noises[-1]
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (HORIZON, MODEL_DIM))
        # Drawn, not zeros: a latent of all zeros would pass the equality above
        # by accident.
        self.assertGreater(float(np.abs(first).max()), 0.0)

    def test_two_clients_with_the_same_seed_send_the_same_latent(self):
        for client in (self._client(noise_seed=7), self._client(noise_seed=7)):
            client.infer(_observation())
        np.testing.assert_array_equal(self.policy.noises[-2], self.policy.noises[-1])

    def test_a_fixed_latent_makes_two_identical_requests_bit_identical(self):
        client = self._client()
        noise = np.full((HORIZON, MODEL_DIM), 3.5, dtype=np.float32)
        first = client.infer(_observation(), noise=noise)["actions"]
        second = client.infer(_observation(), noise=noise)["actions"]
        np.testing.assert_array_equal(first, second)

    def test_without_a_latent_two_identical_requests_differ(self):
        """The engine draws a fresh latent per request; equal answers would mean
        something is caching a result and calling it an inference."""
        client = self._client()
        first = client.infer(_observation())["actions"]
        second = client.infer(_observation())["actions"]
        self.assertFalse(np.array_equal(first, second))

    def test_building_an_observation_checks_the_view_count(self):
        client = self._client()
        with self.assertRaises(RemotePolicyError):
            client.build_observation([np.zeros((4, 4, 3), np.uint8)], prompt="x")
        built = client.build_observation(
            [np.zeros((4, 4, 3), np.uint8)] * 2, prompt="x", state=np.zeros(8, np.float32)
        )
        self.assertEqual(sorted(built), sorted([*LIBERO_IMAGE_KEYS, "prompt"]))

    def test_act_returns_the_segments_alongside_the_actions(self):
        client = self._client()
        result = client.act(
            [np.zeros((4, 4, 3), np.uint8)] * 2, prompt="pick up the block"
        )
        self.assertEqual(result.actions.shape, (HORIZON, ACTION_DIM))
        self.assertEqual(
            set(result.segments), {"round_trip", "model", "server_processor", "transport"}
        )
        self.assertAlmostEqual(result.segments["model"], 1.5, places=5)


class RemotePolicyFailureTest(unittest.TestCase):
    """What the client does when the far side is not answering correctly."""

    def test_a_raising_policy_comes_back_as_the_servers_traceback(self):
        server = RunningServer(FakePolicy(error=RuntimeError("engine exploded")))
        self.addCleanup(server.close)
        client = RemotePolicy("127.0.0.1", server.port)
        self.addCleanup(client.close)
        with self.assertRaises(RemotePolicyError) as caught:
            client.infer(_observation())
        message = str(caught.exception)
        self.assertIn("engine exploded", message)
        self.assertIn("Traceback", message)

    def test_a_timeout_closes_the_socket_so_no_late_frame_is_mistaken_for_an_answer(self):
        server = RunningServer(FakePolicy(delay_s=1.5))
        self.addCleanup(server.close)
        client = RemotePolicy("127.0.0.1", server.port, request_timeout_s=0.2)
        self.addCleanup(client.close)
        with self.assertRaises(RemotePolicyTimeout):
            client.infer(_observation())
        self.assertIsNone(client._connection)

    def test_an_unreachable_server_says_so_without_retrying(self):
        # A port nobody is listening on. 1 and 9 are IANA-reserved; the timeout
        # is short so a machine that silently drops packets still fails fast.
        client = None
        with self.assertRaises(OSError):
            client = RemotePolicy("127.0.0.1", 1, connect_timeout_s=1.0)
        self.assertIsNone(client)

    def test_zero_reconnects_makes_a_dropped_connection_fatal(self):
        policy = FakePolicy()
        server = RunningServer(policy)
        client = RemotePolicy(
            "127.0.0.1", server.port, max_reconnects=0, request_timeout_s=2.0
        )
        self.addCleanup(client.close)
        client.infer(_observation())
        server.close()
        with self.assertRaises(RemotePolicyError):
            client.infer(_observation())  # the failure that drops the connection
        with self.assertRaises(RemotePolicyError) as caught:
            client.infer(_observation())  # and then: no reconnect is allowed
        self.assertIn("reconnect", str(caught.exception))


class RemotePolicyReconnectTest(unittest.TestCase):
    """A reconnect must revalidate the contract, not silently adopt a new one."""

    def _restart(self, server, policy) -> RunningServer:
        port = server.port
        server.close()
        restarted = RunningServer(policy, port=port)
        self.addCleanup(restarted.close)
        return restarted

    def test_a_restarted_server_with_the_same_contract_is_adopted(self):
        server = RunningServer(FakePolicy())
        self.addCleanup(server.close)
        client = RemotePolicy("127.0.0.1", server.port, request_timeout_s=5.0)
        self.addCleanup(client.close)
        client.infer(_observation())

        self._restart(server, FakePolicy())
        # The first call after the restart fails on the dead socket; the call
        # after that reconnects and succeeds.
        with self.assertRaises(RemotePolicyError):
            client.infer(_observation())
        result = client.infer(_observation())
        self.assertEqual(result["actions"].shape, (HORIZON, ACTION_DIM))

    def test_a_restarted_server_with_another_contract_is_refused(self):
        server = RunningServer(FakePolicy())
        self.addCleanup(server.close)
        client = RemotePolicy("127.0.0.1", server.port, request_timeout_s=5.0)
        self.addCleanup(client.close)
        client.infer(_observation())

        self._restart(
            server,
            FakePolicy(image_keys=("observation/image", "observation/right_wrist")),
        )
        with self.assertRaises(RemotePolicyError):
            client.infer(_observation())
        with self.assertRaises(RemoteDialectError) as caught:
            client.infer(_observation())
        self.assertIn("reconnect", str(caught.exception))

    def test_an_in_flight_request_is_never_resent(self):
        """A resend would draw a different latent and answer with different
        actions, so a dropped request must surface rather than be retried."""
        policy = FakePolicy()
        server = RunningServer(policy)
        self.addCleanup(server.close)
        client = RemotePolicy("127.0.0.1", server.port, request_timeout_s=5.0)
        self.addCleanup(client.close)
        server.close()
        with self.assertRaises(RemotePolicyError):
            client.infer(_observation())
        self.assertEqual(policy.draws, 0)


if __name__ == "__main__":
    unittest.main()
