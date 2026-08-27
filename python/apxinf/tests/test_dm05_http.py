from __future__ import annotations

import json

import numpy as np
import pytest

from test_dm05_policy import FakeBackend, observation, policy


def service(backend=None):
    from apxinf.serving import Dm05HttpService

    return Dm05HttpService(policy(backend or FakeBackend()))


def request_body():
    value = observation()
    sampling = value.pop("sampling")
    return json.dumps({"observation": value, "sampling": sampling}).encode()


def test_dm05_native_http_health_and_infer():
    api = service()
    status, health = api.handle("GET", "/health")
    assert status == 200
    assert health["schema"] == "apxinf.dm05.libero.http.v2"
    assert health["policy"]["backend"] == "apxinf-native"
    assert health["policy"]["state_conditioned"] is False
    assert health["policy"]["max_prefix_len"] == 768
    assert health["policy"]["sampling_rng"] == "apxinf-philox-box-muller-v1"
    assert "path_proof" not in health["policy"]
    assert "execution_backend" not in health["policy"]

    status, response = api.handle("POST", "/v1/infer", request_body())
    assert status == 200
    assert len(response["actions"]) == 10
    assert all(len(row) == 7 for row in response["actions"])
    metadata = response["metadata"]
    assert metadata["schema"] == "apxinf.dm05.libero.response.v2"
    assert metadata["backend"] == "apxinf-native"
    assert metadata["precision"] == "bf16"
    assert "path_proof" not in metadata


def test_dm05_http_accepts_exact_noise_for_reference_replay():
    backend = FakeBackend()
    body = json.loads(request_body())
    noise = np.arange(320, dtype=np.float32).reshape(10, 32)
    body["noise"] = noise.tolist()
    status, response = service(backend).handle(
        "POST", "/v1/infer", json.dumps(body).encode()
    )
    assert status == 200
    assert np.array_equal(backend.calls[0]["noise"], noise)
    assert len(response["actions"]) == 10


def test_dm05_http_rejects_wrong_wire_contract():
    api = service()
    assert api.handle("GET", "/v1/infer")[0] == 405
    assert api.handle("POST", "/unknown", b"{}")[0] == 404
    assert api.handle("POST", "/v1/infer", b"not json")[0] == 400
    assert (
        api.handle(
            "POST", "/v1/infer", b"{}", content_type="application/octet-stream"
        )[0]
        == 415
    )


def test_dm05_http_rejects_out_of_range_json_numbers_as_client_errors():
    body = json.loads(request_body())
    body["observation"]["state"][0] = 10**1000
    status, response = service().handle(
        "POST", "/v1/infer", json.dumps(body).encode()
    )
    assert status == 400
    assert "finite numbers" in response["error"]

    body = json.loads(request_body())
    body["sampling"]["seed"] = 1 << 64
    status, response = service().handle(
        "POST", "/v1/infer", json.dumps(body).encode()
    )
    assert status == 400
    assert "inclusive range" in response["error"]


def test_dm05_http_does_not_mislabel_backend_failure_as_client_error():
    backend = FakeBackend()

    def fail(*args, **kwargs):
        raise OverflowError("internal arithmetic failure")

    backend.infer = fail
    with pytest.raises(OverflowError, match="internal arithmetic failure"):
        service(backend).handle("POST", "/v1/infer", request_body())
