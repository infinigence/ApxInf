from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gold import SCHEMA, artifact  # noqa: E402


ENCODINGS = {
    np.dtype(np.float32): ("torch.float32", artifact.NATIVE),
    np.dtype(np.int64): ("torch.int64", artifact.NATIVE),
    np.dtype(np.bool_): ("torch.bool", artifact.NATIVE),
}


def _describe(name, array, *, group="boundary", stage=None):
    """The manifest entries a stored array gets from the writer, recomputed here."""

    dtype, kind = ENCODINGS[array.dtype]
    meta = {"group": group, "stage": stage, "shape": list(array.shape), "dtype": dtype,
            "encoding": kind, "sha256": artifact.array_digest(array)}
    return meta, artifact.statistics_from_array(array, dtype=dtype, encoding=kind)


def _document():
    """A miniature but fully formed artifact: the arrays and the manifest describing them.

    The arrays are small, but the keys and the relationships between them are the
    real ones -- pad masks, the prefix mask, position ids and one step's masks --
    so the structural checks have something to check and the writer's gate is
    exercised end to end without a model or a device.
    """

    arrays = {
        "prefix.embs": np.zeros((1, 4, 8), dtype=np.float32),
        "prefix.pad_masks": np.ones((1, 4), dtype=bool),
        "prefix.att_masks": np.zeros((1, 4), dtype=bool),
        "prefix.position_ids": np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        "prefix.att_2d_masks": np.ones((1, 4, 4), dtype=bool),
        "step.00.suffix_pad_masks": np.ones((1, 2), dtype=bool),
        "step.00.suffix_att_masks": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "step.00.att_2d_masks": np.ones((1, 2, 6), dtype=bool),
        "raw_actions": (np.arange(64, dtype=np.float32) / 64.0).reshape(2, 32),
    }
    debug = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    meta, statistics = {}, {}
    for name, array in debug.items():
        meta[name], statistics[name] = _describe(name, array)
    inputs = artifact.make_fixture(seed=3, views=2, token_count=4, horizon=2)
    manifest = {"schema": SCHEMA, "checks": {}, "statistics": statistics, "debug": meta,
                "stages": {}, "stage_arrays": []}
    return {name: np.asarray(value) for name, value in inputs.items()}, debug, manifest


def _refresh(debug, manifest, name, array):
    """Replace one stored array the way the generator would have: everything re-derived."""

    debug[name] = array
    manifest["debug"][name], manifest["statistics"][name] = _describe(name, array)


def _publish(tmp_path):
    """Write a miniature artifact, and hand back both sides plus the manifest on disk."""

    inputs, debug, manifest = _document()
    artifact.write_output(tmp_path, inputs=inputs, debug=debug, manifest=manifest, force=False)
    return inputs, debug, manifest, json.loads((tmp_path / "manifest.json").read_text())


def test_the_verifying_side_loads_without_a_device_stack():
    """torch_gold.artifact must not drag torch in.

    Keeping torch out of it is what lets the gate be tested anywhere, and it is
    the reason the CLI installs torchada and imports torch before it imports the
    package: nothing on this side may import torch first.
    """

    program = (f"import sys; sys.path.insert(0, {str(ROOT)!r});"
               " import torch_gold.artifact;"
               " print('torch' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "importing the verifying side pulled torch in"


def test_fixture_generation_is_cpu_seeded_and_reproducible():
    first = artifact.make_fixture(seed=17, views=2, token_count=10, horizon=10)
    second = artifact.make_fixture(seed=17, views=2, token_count=10, horizon=10)

    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert first["images"].shape == (2, 3, 224, 224)
    assert first["images"].dtype == np.float32
    assert first["token_ids"].shape == (10,)
    assert first["token_ids"].dtype == np.int64
    assert first["noise"].shape == (10, 32)
    assert first["noise"].dtype == np.float32
    assert first["state"].shape == (32,)
    assert first["state"].dtype == np.float32
    assert np.all(first["images"] >= -1.0)
    assert np.all(first["images"] <= 1.0)

    # The flow-matching scalars travel with the fixture so a replay host never
    # has to guess the Euler schedule.
    assert int(first["num_flow_steps"]) == 10
    assert np.float32(first["dt"]) == np.float32(-1.0 / 10)
    assert np.float32(first["flow_start_time"]) == np.float32(1.0)
    assert artifact.flow_steps(first, None) == 10
    assert artifact.flow_steps(first, 4) == 4
    assert artifact.flow_steps({}, None) == artifact.DEFAULT_NUM_FLOW_STEPS


def test_fixture_loader_rejects_wrong_shape(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, images=np.zeros((2, 224, 224, 3), dtype=np.float32), token_ids=np.zeros(10, dtype=np.int64), noise=np.zeros((10, 32), dtype=np.float32))

    try:
        artifact.load_fixture(path)
    except ValueError as error:
        assert "images must have shape" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("invalid fixture was accepted")


def test_debug_encoding_round_trips():
    torch = pytest.importorskip("torch")
    values = torch.tensor([[1.0, -2.0, 0.0], [3.140625, -0.0, 65504.0]], dtype=torch.bfloat16)
    array, description = artifact.encode_tensor(values)
    assert array.dtype == np.uint16
    assert description == {"dtype": "torch.bfloat16", "encoding": artifact.BFLOAT16_BITS}

    restored = artifact.decode_tensor(array, description["dtype"], description["encoding"])
    assert restored.dtype == np.float32
    assert np.array_equal(restored, values.float().numpy())
    # Bit-exact: widening bfloat16 to float32 keeps the original pattern on top.
    assert np.array_equal((restored.view(np.uint32) >> 16).astype(np.uint16), array)

    # The digest the capture-time witness records is over those same bytes.
    assert artifact.tensor_digest(values) == artifact.array_digest(array)
    # Two bool tensors that compare equal can still hold different bytes, and the
    # witness is byte-level on purpose: that is how a stray byte in a mask is
    # caught at all.
    assert artifact.tensor_digest(torch.tensor([1, 2], dtype=torch.uint8).view(torch.bool)) != \
        artifact.tensor_digest(torch.tensor([True, True]))

    for tensor, expected in (
        (torch.zeros(2, dtype=torch.float32), np.float32),
        (torch.zeros(2, dtype=torch.int64), np.int64),
        (torch.zeros(2, dtype=torch.bool), np.bool_),
    ):
        native, description = artifact.encode_tensor(tensor)
        assert native.dtype == expected
        assert description["encoding"] == artifact.NATIVE
        assert np.array_equal(artifact.decode_tensor(native, description["dtype"], description["encoding"]), native)

    with pytest.raises(ValueError):
        artifact.decode_tensor(np.zeros(2, dtype=np.uint16), "torch.float32", artifact.BFLOAT16_BITS)
    with pytest.raises(ValueError):
        artifact.decode_tensor(np.zeros(2, dtype=np.float32), "torch.bfloat16", artifact.BFLOAT16_BITS)
    with pytest.raises(ValueError):
        artifact.decode_tensor(np.zeros(2, dtype=np.float32), "torch.float32", "mystery")


def test_statistics_are_recomputable_from_the_stored_array():
    """The manifest formula is the reader's formula, for the bfloat16 encoding the artifact has no array for."""

    bits = np.asarray([0x3F80, 0xC000, 0x0000], dtype=np.uint16)  # 1.0, -2.0, 0.0
    summary = artifact.statistics_from_array(bits, dtype="torch.bfloat16", encoding=artifact.BFLOAT16_BITS)
    assert (summary["sum"], summary["abs_sum"], summary["max_abs"]) == (-1.0, 3.0, 2.0)
    assert summary["elements"] == 3 and summary["shape"] == [3]

    # int64 narrows to float32 before summing, exactly as a reader does.
    summary = artifact.statistics_from_array(np.asarray([[0, 1, 2, 3]], dtype=np.int64),
                                             dtype="torch.int64", encoding=artifact.NATIVE)
    assert summary["sum"] == 6.0 and summary["dtype"] == "torch.int64"


def test_output_contains_inputs_debug_and_manifest(tmp_path):
    inputs, debug, _, saved = _publish(tmp_path)

    with np.load(tmp_path / "inputs.npz", allow_pickle=False) as document:
        assert document["images"].shape == (2, 3, 224, 224)
        assert document["state"].shape == (32,)
        assert int(document["num_flow_steps"]) == 10
    with np.load(tmp_path / "debug.npz", allow_pickle=False) as document:
        assert document["raw_actions"].shape == (2, 32)
        assert document["prefix.embs"].shape == (1, 4, 8)
    assert saved["schema"] == SCHEMA
    assert saved["files"]["inputs.npz"]["sha256"]
    assert saved["files"]["debug.npz"]["sha256"]
    assert saved["debug"]["raw_actions"]["sha256"] == artifact.array_digest(debug["raw_actions"])
    assert saved["checks"] == {"written_arrays_match_manifest": True, "content_invariants_hold": True}
    # The staging directory is renamed into place, so nothing is left beside it.
    assert not list(tmp_path.parent.glob(f"{tmp_path.name}.partial-*"))

    try:
        artifact.write_output(tmp_path, inputs=inputs, debug=debug, manifest=saved, force=False)
    except FileExistsError:
        pass
    else:  # pragma: no cover - assertion guard
        raise AssertionError("existing output was replaced without --force")


def test_published_manifest_matches_an_independent_recomputation(tmp_path):
    """Recompute every statistic the way an external verifier would, from the file."""

    _, _, _, saved = _publish(tmp_path)

    with np.load(tmp_path / "debug.npz", allow_pickle=False) as document:
        assert set(document.files) == set(saved["statistics"]) == set(saved["debug"])
        for key, recorded in saved["statistics"].items():
            values = document[key]
            values = values.astype(bool, copy=False) if saved["debug"][key]["dtype"] == "torch.bool" \
                else values.astype(np.float32, copy=False)
            flat = values.reshape(-1)
            assert recorded == {
                "elements": int(flat.size),
                "shape": list(values.shape),
                "dtype": saved["debug"][key]["dtype"],
                "sum": float(np.sum(flat, dtype=np.float64)),
                "abs_sum": float(np.sum(np.abs(flat), dtype=np.float64)),
                "l2": float(np.sqrt(np.sum(np.square(flat, dtype=np.float64), dtype=np.float64))),
                "max_abs": float(np.max(np.abs(flat))),
            }


def _disagree_with_a_statistic(debug, manifest):
    manifest["statistics"]["raw_actions"]["sum"] += 1e-9
    return "raw_actions.sum"


def _break_the_mask_algebra(debug, manifest):
    """A stray False is self-consistent -- its statistics agree with it -- and still must not ship."""

    mask = debug["step.00.att_2d_masks"].copy()
    mask[0, 0, 5] = False
    _refresh(debug, manifest, "step.00.att_2d_masks", mask)
    return "mask algebra"


def _store_a_noncanonical_bool_byte(debug, manifest):
    broken = debug["prefix.pad_masks"].copy()
    broken.view(np.uint8)[0, 0] = 7
    _refresh(debug, manifest, "prefix.pad_masks", broken)
    return "outside 0/1"


@pytest.mark.parametrize("mutate", [_disagree_with_a_statistic, _break_the_mask_algebra,
                                    _store_a_noncanonical_bool_byte])
def test_write_output_refuses_a_capture_that_fails_a_check(tmp_path, mutate):
    """Every refusal leaves the output directory untouched and says why, in a file."""

    inputs, debug, manifest = _document()
    expected = mutate(debug, manifest)

    with pytest.raises(artifact.GoldVerificationError) as error:
        artifact.write_output(tmp_path, inputs=inputs, debug=debug, manifest=manifest, force=False)

    assert expected in str(error.value)
    assert not any(tmp_path.iterdir())  # the output directory was never populated
    failures = list(tmp_path.parent.glob(f"{tmp_path.name}.partial-*/verify-failure.json"))
    assert len(failures) == 1
    assert failures[0].read_text().count(expected) == 1


def test_write_output_keeps_the_previous_artifact_on_refusal(tmp_path):
    """--force clears an existing artifact only once its replacement has been verified."""

    output = tmp_path / "run"
    output.mkdir()
    (output / "manifest.json").write_text('{"schema": "previous"}\n')
    inputs, debug, manifest = _document()
    manifest["statistics"]["raw_actions"]["l2"] += 1.0

    with pytest.raises(artifact.GoldVerificationError):
        artifact.write_output(output, inputs=inputs, debug=debug, manifest=manifest, force=True)

    assert json.loads((output / "manifest.json").read_text()) == {"schema": "previous"}


def _replayable(tmp_path, mutate=None):
    """A published artifact plus the two manifest blocks a replay reads.

    The writer does not emit ``inputs`` (the fixture is summarized where the CLI
    assembles the manifest) and knows nothing of ``model_config``, so both are
    added here -- and ``mutate`` is what edits them, so a case that stores its
    own inconsistency also leaves the bytes on disk saying so.
    """

    inputs, debug, _, saved = _publish(tmp_path)
    saved["inputs"] = artifact.describe_inputs(inputs)
    saved["model_config"] = {"action_horizon": int(inputs["noise"].shape[0]),
                             "action_dim": int(inputs["noise"].shape[1])}
    if mutate is not None:
        mutate(inputs, debug, saved)
        np.savez(tmp_path / "debug.npz", **debug)
    return inputs, saved


def _masked_a_prefix_token(inputs, debug, saved):
    masks = debug["prefix.pad_masks"].copy()
    masks.reshape(-1)[1] = False
    debug["prefix.pad_masks"] = masks


def _resized_the_noise(inputs, debug, saved):
    inputs["noise"] = np.zeros((2, 8), dtype=np.float32)


def _retimed_the_schedule(inputs, debug, saved):
    inputs["dt"] = np.asarray(-0.25, dtype=np.float32)


def _dropped_the_action_description(inputs, debug, saved):
    del saved["debug"]["raw_actions"]


def test_a_published_bundle_replays_with_nothing_to_report(tmp_path):
    inputs, saved = _replayable(tmp_path)

    assert artifact.replay_problems(tmp_path, inputs, saved) == []


@pytest.mark.parametrize("mutate, expected", [
    (_masked_a_prefix_token, "pad_masks"),
    (_resized_the_noise, "noise"),
    (_retimed_the_schedule, "dt"),
    (_dropped_the_action_description, "raw_actions"),
])
def test_replay_names_the_way_a_bundle_disagrees_with_the_engine(tmp_path, mutate, expected):
    """A score taken across one of these is a different computation, not a regression."""

    inputs, saved = _replayable(tmp_path, mutate)

    problems = artifact.replay_problems(tmp_path, inputs, saved)

    assert any(expected in problem for problem in problems), problems


def test_a_captured_tensor_is_only_accepted_when_its_reads_settle(monkeypatch):
    """The witness around the read path: reads do not become evidence by agreeing twice.

    Reading a captured tensor back to the host is the step this host has been
    observed to get wrong, and the wrong value repeats, so the value the reads
    agree on has to be a majority -- and a tensor whose reads never settle is
    refused rather than shipped.
    """

    pytest.importorskip("torch")
    from torch_gold import capture

    def reader(arrays):
        remaining = iter(arrays)

        def encode(_tensor):
            return np.asarray(next(remaining), dtype=np.float32), {"dtype": "torch.float32",
                                                                   "encoding": artifact.NATIVE}

        return encode

    zeros, ones, twos = (np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32),
                         np.full(2, 2.0, dtype=np.float32))

    # Unanimous: the first two reads are enough.
    monkeypatch.setattr(capture, "encode_tensor", reader([zeros, zeros, ones]))
    array, description, digest, reads = capture.stable_encode(object())
    assert reads == 2 and array.tolist() == [0.0, 0.0] and digest == artifact.array_digest(zeros)

    # The outlier is the first read, and the shipped array must be the majority's bytes.
    monkeypatch.setattr(capture, "encode_tensor", reader([ones, zeros, zeros]))
    array, description, digest, reads = capture.stable_encode(object())
    assert reads == 3 and array.tolist() == [0.0, 0.0] and digest == artifact.array_digest(zeros)

    # Three different values: nothing settles, so nothing is shipped.
    monkeypatch.setattr(capture, "encode_tensor", reader([zeros, ones, twos]))
    with pytest.raises(artifact.GoldVerificationError) as error:
        capture.stable_encode(object())
    assert "never settled" in str(error.value)


def test_alignment_matches_a_replay_against_the_artifact_it_replayed(tmp_path):
    """A replay's own numbers say how far it landed from the capture it replayed.

    This is the reading a replay prints at the end of a run: the fixture names the
    artifact it came from, so the directory holding it is the reference, and the
    cosine between the two captures is the answer -- per tensor, because one
    number over the whole thing would hide the layer that moved.
    """

    reference = tmp_path / "reference"
    _, debug, _, manifest = _publish(reference)

    assert artifact.reference_artifact(reference / "inputs.npz") == reference
    report = artifact.compare_artifact(debug, manifest["debug"], reference)

    # Five of the nine stored arrays are masks, and a mask is compared for
    # equality rather than by direction.
    assert report["compared"] == 4
    assert report["raw_actions"] == 1.0
    # Everything ties at 1.0, so which key is named is just the sort order.
    assert report["minimum"]["cosine"] == 1.0
    assert report["below_floor"] == []
    assert report["only_in_run"] == [] and report["only_in_reference"] == []
    assert report["masks"]["identical"] == report["masks"]["total"] == 5

    block = artifact.format_alignment(report)
    assert "aligned" in block and "DIVERGENT" not in block
    assert "5 bool tensor(s), all identical" in block
    # The criterion travels with the reading, so a number is never read bare.
    assert f"cos >= {artifact.ALIGNMENT_FLOOR}" in block
    # A readout, not a record: the manifest is about this artifact alone.
    assert "alignment" not in manifest


def test_alignment_names_the_tensor_a_divergence_comes_from(tmp_path):
    """A run that computed something else has to be visible, and visible *where*.

    The total is printed beside the worst tensor precisely so a single high
    reading cannot cover for one that moved.
    """

    reference = tmp_path / "reference"
    _, debug, _, manifest = _publish(reference)

    divergent = dict(debug)
    divergent["raw_actions"] = np.ascontiguousarray(debug["raw_actions"][:, ::-1])
    report = artifact.compare_artifact(divergent, manifest["debug"], reference)

    assert report["raw_actions"] < artifact.ALIGNMENT_FLOOR
    assert report["below_floor"] == ["raw_actions"]
    assert report["minimum"]["key"] == "raw_actions"
    block = artifact.format_alignment(report)
    assert "DIVERGENT" in block and "raw_actions" in block


def test_alignment_separates_a_resized_tensor_from_a_divergent_one(tmp_path):
    """A key present on both sides at different sizes is reported as such, not compared."""

    reference = tmp_path / "reference"
    _, debug, _, manifest = _publish(reference)

    resized = dict(debug)
    resized["raw_actions"] = np.ascontiguousarray(debug["raw_actions"].reshape(4, 16))
    report = artifact.compare_artifact(resized, manifest["debug"], reference)

    assert report["resized"] == ["raw_actions"]
    assert "raw_actions" not in report["below_floor"]
    assert "size differs" in artifact.format_alignment(report)


def test_alignment_has_nothing_to_compare_against_without_a_reference(tmp_path):
    """A fixture copied out on its own replays fine; it just has no earlier capture."""

    bare = tmp_path / "bare" / "inputs.npz"
    bare.parent.mkdir()
    bare.write_bytes(b"a fixture with no artifact beside it")

    assert artifact.reference_artifact(bare) is None


def test_the_alignment_readout_never_fails_the_run_it_describes(tmp_path):
    """The gold is already on disk when this runs, so a reference it cannot read is a message."""

    pytest.importorskip("torch")
    from torch_gold import capture

    reference = tmp_path / "reference"
    _, debug, _, manifest = _publish(reference)
    fixture = reference / "inputs.npz"

    def block_for(input_path, outputs):
        return capture.alignment_block(types.SimpleNamespace(input=input_path), debug, manifest["debug"], outputs)

    # No --input at all: a generated fixture replays nothing.
    assert block_for(None, tmp_path / "elsewhere") is None

    # A reference is only worth reading when it is not the artifact being written.
    assert block_for(fixture, reference) is None

    (reference / "debug.npz").write_bytes(b"truncated by a copy that did not finish")
    message = block_for(fixture, tmp_path / "elsewhere")
    assert message.startswith(f"alignment: could not read the reference artifact at {reference}")


def test_timing_report_states_the_warmup_basis():
    block = artifact.timing_report([3.0, 2.0, 4.0], warmup_seconds=[9.0, 5.0, 4.0], recording_seconds=7.5)

    assert (block["p50"], block["min"], block["max"], block["samples"]) == (3.0, 2.0, 4.0, 3)
    assert block["warmup_runs"] == 3 and block["warmup_seconds"] == [9.0, 5.0, 4.0]
    assert block["timed_seconds"] == [3.0, 2.0, 4.0]
    assert block["recording_run_seconds"] == 7.5
    assert block["instrumented"] is False
    assert "3 warmup run(s)" in block["basis"] and "un-instrumented" in block["basis"]

    # Printed as one reading per line: milliseconds, and the Hz of the same number.
    lines = artifact.format_timing(block).splitlines()
    assert lines[0] == "timing: un-instrumented sample_actions, 3 warmup + 3 timed runs"
    assert lines[1:] == ["  min   2000 ms   (0.50 Hz)", "  p50   3000 ms   (0.33 Hz)", "  max   4000 ms   (0.25 Hz)"]
    assert "7500" not in artifact.format_timing(block)  # the recording run stays in the JSON record

    with pytest.raises(ValueError):
        artifact.timing_report([], warmup_seconds=[])
