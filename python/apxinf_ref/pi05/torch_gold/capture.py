"""The other half: drive the model, record its boundary, and publish the result.

``sample_actions`` is the driver throughout.  Nothing here re-implements the
Euler schedule, the prefix pass or the KV cache -- :class:`BoundaryRecorder`
wraps four model methods and records while forwarding, and :func:`inference`
feeds the model the fixture's own tensors by replacing ``_preprocess_observation``.

Everything torch-facing lives here so :mod:`torch_gold.artifact` can stay
importable without a device stack, and this module may only be imported after
``bootstrap_torch`` has run in the entry script -- torchada has to patch torch
before the first ``import torch`` anywhere in the process.

Full tensors come off the model's forward boundary; sampled signatures come from
forward hooks on the transformer layers, enough to locate a divergent layer
before the boundary tensors localize it element by element.  Nothing is read
back while the model runs, and afterwards each captured tensor is read until its
reads settle on one value -- see :func:`stable_encode` for why agreeing reads
alone would prove nothing here.
"""

from __future__ import annotations

import contextlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import SCHEMA
from .artifact import (
    DEFAULT_ACTION_DIM,
    DEFAULT_FLOW_START_TIME,
    EXIT_REFUSED,
    NATIVE,
    GoldVerificationError,
    array_digest,
    compare_artifact,
    describe_inputs,
    encode_tensor,
    format_alignment,
    format_timing,
    host_read,
    reference_artifact,
    resolve_inputs,
    sha256,
    statistics_from_array,
    summarize,
    synchronize,
    timing_report,
    validate_fixture,
    write_output,
)


LAYER_SCAN = 64
_VISION = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
VISION_EMBEDDINGS = _VISION + ".embeddings"
VISION_LAYERS = _VISION + ".encoder.layers"
VISION_PROJECTOR = "paligemma_with_expert.paligemma.model.multi_modal_projector"
PREFIX_LAYERS = "paligemma_with_expert.paligemma.model.language_model.layers"
DECODER_LAYERS = "paligemma_with_expert.gemma_expert.model.layers"
CAPTURE_GROUPS = {"prefix": 0, "kv": 1, "step": 2}
MAX_CAPTURE_READS = 3


class ObservationShim:
    """Stand-in for openpi's ``Observation``.

    ``sample_actions`` reads ``observation.state.shape[0]`` before handing the
    object to ``_preprocess_observation``, which this package replaces, so the
    shim never needs the image or tokenizer fields.
    """

    def __init__(self, state):
        self.state = state


def fixture_inputs(arrays: dict[str, np.ndarray], model_config, device):
    """The fixture's canonical tensors, arranged as the model's forward boundary."""

    images = torch.from_numpy(arrays["images"]).to(device=device)
    token_ids = torch.from_numpy(arrays["token_ids"]).to(device=device)
    noise = torch.from_numpy(arrays["noise"]).to(device=device)
    state = torch.from_numpy(arrays.get("state", np.zeros(model_config.action_dim, dtype=np.float32))).to(device=device)

    image_list = [images[index : index + 1] for index in range(images.shape[0])]
    image_masks = [torch.ones((1,), dtype=torch.bool, device=device) for _ in image_list]
    lang_tokens = token_ids.unsqueeze(0)
    lang_masks = torch.ones((1, token_ids.numel()), dtype=torch.bool, device=device)
    return (image_list, image_masks, lang_tokens, lang_masks, state.unsqueeze(0)), noise


@contextlib.contextmanager
def fixture_observation(model, boundary: tuple):
    """Hand ``sample_actions`` the fixture tensors instead of camera/tokenizer output."""

    model._preprocess_observation = lambda _observation, *, train=False: boundary
    try:
        yield
    finally:
        model.__dict__.pop("_preprocess_observation", None)


def inference(model, *, boundary: tuple, noise, device, num_steps: int):
    """One end-to-end ``sample_actions`` call -- the whole point of this tool."""

    with fixture_observation(model, boundary):
        with torch.inference_mode():
            # Passing noise keeps the global RNG out of the reference.
            return model.sample_actions(
                device, ObservationShim(boundary[-1]), noise=noise.unsqueeze(0), num_steps=num_steps
            )


def benchmark(model, *, boundary, noise, device, num_steps: int, warmup_runs: int, timed_runs: int):
    """Warm the device up, then time the plain path.

    Nothing is recorded and no hook is installed: the stage hooks read whole
    tensors back to the host, so an instrumented number would describe this tool
    instead of the model.  The warmup runs are returned too -- they are the
    evidence that the machine settled before the timed runs started.
    """

    def once() -> float:
        synchronize(device)
        started = time.perf_counter()
        inference(model, boundary=boundary, noise=noise, device=device, num_steps=num_steps)
        synchronize(device)
        return time.perf_counter() - started

    warmup_seconds = [once() for _ in range(warmup_runs)]
    return warmup_seconds, [once() for _ in range(timed_runs)]


def resolve_device(spec: str):
    """Parse a device string, and refuse one this host cannot offer."""

    device = torch.device(spec)
    if device.type == "cpu":
        return device
    module = getattr(torch, device.type, None)
    if device.type not in ("cuda", "musa"):
        raise ValueError(f"unsupported device {spec!r}; choose cpu, cuda:N, or musa:N")
    if module is None or not module.is_available():
        raise RuntimeError(f"requested {device.type} device but {device.type}.is_available() is false")
    index = 0 if device.index is None else device.index
    count = module.device_count()
    if index < 0 or index >= count:
        raise RuntimeError(f"requested {device}, but this host has {count} {device.type} device(s)")
    return torch.device(device.type, index)


def checkpoint_config(path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Locate a checkpoint's weights and config, and parse the config."""

    root = path.expanduser().resolve()
    if root.is_dir():
        weights, config_path = root / "model.safetensors", root / "config.json"
    else:
        weights, config_path, root = root, root.parent / "config.json", root.parent
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint weights not found: {weights}")
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint config is not a JSON object: {config_path}")
    return root, weights, config


def config_value(config: dict[str, Any], *names: str, default: Any) -> Any:
    """The first present spelling of a config key, so older checkpoints keep working."""

    for name in names:
        if name in config:
            return config[name]
    return default


def pin_eager(model) -> None:
    """Force the vendored attention path: eager is the reference, not an optimization.

    transformers 4.53 already defaults ``_attn_implementation`` to ``eager``, so
    on today's pin this changes nothing.  It stays because the vendored
    ``sample_actions`` pins only the two language models, immediately before each
    forward call (``pi0_pytorch.py``): the SigLIP vision tower is covered here
    and nowhere else, so a future transformers default would otherwise swap that
    tower's attention kernel and move every recorded number without a word.
    """

    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"


def load_model(weights: Path, config: dict[str, Any], device, precision: str):
    """Build the vendored PI0Pytorch model and load the checkpoint into it.

    A state-dict mismatch means the vendored snapshot and the checkpoint have
    drifted apart, which would make every recorded number suspect, so it refuses
    rather than loading what it can.
    """

    from _vendor.openpi_pi0_pytorch.transformers_overlay import install_transformers_overlay

    install_transformers_overlay()
    from _vendor.openpi_pi0_pytorch.pi0_config import Pi0Config
    from _vendor.openpi_pi0_pytorch.pi0_pytorch import PI0Pytorch
    from safetensors.torch import load_file

    dtype_name = str(config.get("precision", "bfloat16")) if precision == "checkpoint" else precision
    if dtype_name not in ("bfloat16", "float32"):
        raise ValueError(f"unsupported checkpoint precision {dtype_name!r}")
    model_config = Pi0Config(
        dtype=dtype_name,
        paligemma_variant=str(config.get("paligemma_variant", "gemma_2b")),
        action_expert_variant=str(config.get("action_expert_variant", "gemma_300m")),
        action_dim=int(config_value(config, "action_dim", "max_action_dim", default=DEFAULT_ACTION_DIM)),
        action_horizon=int(config_value(config, "action_horizon", "chunk_size", default=10)),
        max_token_len=int(config_value(config, "max_token_len", "tokenizer_max_length", default=200)),
        pi05=True,
        pytorch_compile_mode=None,
    )
    model = PI0Pytorch(model_config)
    state = load_file(str(weights), device="cpu")
    alias = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    target = "paligemma_with_expert.paligemma.lm_head.weight"
    if alias not in state and target in state:
        state[alias] = state[target]
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = sorted(set(missing) - {alias})
    unexpected = sorted(unexpected)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint/model state mismatch; refusing to produce a gold file. "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    del state
    model.to(device=device)
    model.eval()
    pin_eager(model)
    return model, model_config, dtype_name


def first_tensor(value):
    """The first tensor in a hook's output, wherever the module buried it."""

    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def stage_record(tensor, *, sample_count: int) -> dict[str, Any]:
    """Full-tensor scalars plus a decimated sample for one layer-hook call.

    The reduction stays on the host in float64 numpy on purpose: these numbers
    are part of the published manifest, so moving the accumulation to the device
    would change every one of them and make them unrecomputable off-device.  The
    whole tensor is read back for the same reason -- the sample alone would be
    cheap, the scalars are not optional.
    """

    values = host_read(tensor).float().numpy().reshape(-1)
    return summarize(values, shape=list(tensor.shape), dtype=str(tensor.dtype), sample_count=sample_count)


class StageCollector:
    """Sampled signatures from forward hooks on the transformer layer modules."""

    def __init__(self, *, sample_count: int):
        self.sample_count = sample_count
        self.step: int | None = None
        self.calls: dict[str, list[dict[str, Any]]] = {}

    def hook(self, name: str):
        def collect(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is None:
                return
            stage = name
            if name.startswith("Pi05.decoder.layer_"):
                if self.step is None:
                    return
                stage = name.replace("Pi05.decoder.", f"Pi05.decoder.step_{self.step:02d}.")
            self.calls.setdefault(stage, []).append(stage_record(tensor, sample_count=self.sample_count))

        return collect


def register_stages(model, collector: StageCollector):
    """Attach the collector to every layer this snapshot is expected to have."""

    modules = dict(model.named_modules())
    families = (
        ((VISION_EMBEDDINGS, "Pi05.vision.patch_embed"),),
        tuple((f"{VISION_LAYERS}.{i}", f"Pi05.vision.layer_{i:02d}") for i in range(LAYER_SCAN)),
        ((VISION_PROJECTOR, "Pi05.vision.projected"),),
        tuple((f"{PREFIX_LAYERS}.{i}", f"Pi05.prefix.layer_{i:02d}") for i in range(LAYER_SCAN)),
        tuple((f"{DECODER_LAYERS}.{i}", f"Pi05.decoder.layer_{i:02d}") for i in range(LAYER_SCAN)),
    )
    return [
        modules[name].register_forward_hook(collector.hook(stage))
        for family in families
        for name, stage in family
        if name in modules
    ]


class BoundaryRecorder:
    """Capture the tensors on the model's own forward boundary.

    The wrappers sit between ``sample_actions`` and the model: they record and
    forward, and none of the vendor's computation is re-implemented here.
    ``_preprocess_observation`` is not among them -- :func:`fixture_observation`
    owns it, so the recording and benchmark runs supply the same boundary.
    """

    WRAPPERS = {
        "embed_prefix": "_embed_prefix",
        "embed_suffix": "_embed_suffix",
        "denoise_step": "_denoise_step",
        "_prepare_attention_masks_4d": "_attention_masks_4d",
    }

    def __init__(self, model, *, collector: StageCollector):
        self.model = model
        self.collector = collector
        self.step: int | None = None
        self.tensors: dict[str, Any] = {}
        self.meta: dict[str, dict[str, Any]] = {}
        self.steps: list[dict[str, Any]] = []
        self._originals: dict[str, Any] = {}

    def record(self, key: str, tensor, *, group: str, stage: int | None = None, derived_from: str | None = None):
        if tensor is None:
            return
        # Snapshot rather than reference: sample_actions hands out views into storage
        # it keeps mutating (time.expand(bsize) aliases the time it decrements in
        # place), so holding the tensor would read back as the final step's value.
        # Nothing is read back here either: a device->host read taken while the model
        # is still running is the least reliable read on this host, and the value is
        # not needed until the run is over.
        snapshot = tensor.detach().clone()
        self.tensors[key] = snapshot
        self.meta[key] = {"group": group, "stage": stage, "shape": list(snapshot.shape), "dtype": str(snapshot.dtype)}
        if derived_from is not None:
            self.meta[key]["derived_from"] = derived_from

    def _lead(self, field: str) -> str | None:
        return None if self.step is None else f"step.{self.step:02d}.{field}"

    def _embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        result = self._originals["embed_prefix"](images, img_masks, lang_tokens, lang_masks)
        self.record("prefix.embs", result[0], group="prefix")
        self.record("prefix.pad_masks", result[1], group="prefix")
        self.record("prefix.att_masks", result[2], group="prefix")
        # sample_actions derives the prefix positions inline from the pad mask.
        self.record(
            "prefix.position_ids",
            torch.cumsum(result[1], dim=1) - 1,
            group="prefix",
            derived_from="cumsum(prefix.pad_masks, dim=1) - 1",
        )
        return result

    def _attention_masks_4d(self, att_2d_masks):
        # The 4-D float mask is where(2-D mask, 0.0, -2.3819763e38), so the 2-D
        # bool mask is the part worth storing.  The first call is the prefix pass.
        key = self._lead("att_2d_masks")
        self.record(key or "prefix.att_2d_masks", att_2d_masks, group="step" if key else "prefix", stage=self.step)
        return self._originals["_prepare_attention_masks_4d"](att_2d_masks)

    def _embed_suffix(self, state, noisy_actions, timestep):
        result = self._originals["embed_suffix"](state, noisy_actions, timestep)
        fields = ("suffix_embs", "suffix_pad_masks", "suffix_att_masks", "adarms_cond")
        for field, tensor in zip(fields, result, strict=True):
            key = self._lead(field)
            if key is not None:
                self.record(key, tensor, group="step", stage=self.step)
        return result

    def _denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        index = self.step = len(self.steps)
        self.collector.step = index
        lead = f"step.{index:02d}"
        self.steps.append({"time": float(host_read(timestep.reshape(-1)[0]))})
        self.record(f"{lead}.x_t", x_t, group="step", stage=index)
        self.record(f"{lead}.time", timestep.reshape(-1)[0], group="step", stage=index)
        if index == 0:
            self._capture_prefix_kv(past_key_values)
        v_t = self._originals["denoise_step"](state, prefix_pad_masks, past_key_values, x_t, timestep)
        self.record(f"{lead}.v_t", v_t, group="step", stage=index)
        return v_t

    def _capture_prefix_kv(self, past_key_values) -> None:
        keys = getattr(past_key_values, "key_cache", None)
        values = getattr(past_key_values, "value_cache", None)
        if keys is None or values is None or len(keys) != len(values):
            raise RuntimeError(
                "sample_actions produced no usable prefix KV cache; the vendored PaliGemma cache API changed"
            )
        for layer, (k, v) in enumerate(zip(keys, values, strict=True)):
            self.record(f"prefix.kv.{layer:02d}.k", k, group="kv", stage=layer)
            self.record(f"prefix.kv.{layer:02d}.v", v, group="kv", stage=layer)

    def install(self) -> None:
        self._originals = {name: getattr(self.model, name) for name in self.WRAPPERS}
        for name, wrapper in self.WRAPPERS.items():
            setattr(self.model, name, getattr(self, wrapper))

    def restore(self) -> None:
        # Every wrapper went in as an instance attribute, so dropping it restores
        # the class method underneath.
        for name in self._originals:
            self.model.__dict__.pop(name, None)
        self._originals = {}


def check_run(recorder: BoundaryRecorder, *, num_steps: int) -> dict[str, bool]:
    """Guard against silent drift in the vendored entry point.

    The wrappers sit between ``sample_actions`` and the model, so this is the
    only place that can notice if the vendored loop changes shape.
    """

    expected = [DEFAULT_FLOW_START_TIME - index / num_steps for index in range(num_steps)]
    observed = [step["time"] for step in recorder.steps]
    checks = {
        "denoise_calls_match_num_flow_steps": len(observed) == num_steps,
        "prefix_kv_captured": "prefix.kv.00.k" in recorder.tensors,
        "time_schedule_matches_euler": len(observed) == len(expected)
        and all(abs(a - b) <= 1e-6 for a, b in zip(observed, expected, strict=True)),
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise RuntimeError(
            f"vendored sample_actions drifted from the recorded contract: {', '.join(failed)}; "
            f"observed timesteps={observed[:4]} expected={expected[:4]}"
        )
    return checks


def run_recorded(model, *, boundary, noise, device, num_steps: int, sample_count: int):
    """The run whose tensors become the artifact, with every hook installed.

    It runs after the warmup runs, so the capture sees a settled device; the time
    it takes is reported separately because the hooks are not free.
    """

    collector = StageCollector(sample_count=sample_count)
    handles = register_stages(model, collector) if sample_count > 0 else []
    recorder = BoundaryRecorder(model, collector=collector)
    try:
        recorder.install()
        synchronize(device)
        started = time.perf_counter()
        actions = inference(model, boundary=boundary, noise=noise, device=device, num_steps=num_steps)
        synchronize(device)
        recording_seconds = time.perf_counter() - started
        checks = check_run(recorder, num_steps=num_steps)
    finally:
        for handle in handles:
            handle.remove()
        recorder.restore()
    raw_actions = actions.detach().float()
    if raw_actions.shape[0] == 1:
        raw_actions = raw_actions[0]
    recorder.record("raw_actions", raw_actions, group="output")
    return recorder, collector.calls, checks, recording_seconds


def capture_order(key: str) -> tuple[int, str]:
    """Sort captured keys by the phase that produced them, then by name."""

    group = "kv" if key.startswith("prefix.kv.") else key.split(".", 1)[0]
    return (CAPTURE_GROUPS.get(group, 3), key)


def stable_encode(tensor, *, attempts: int = MAX_CAPTURE_READS):
    """Encode a captured tensor, and prove the read is reproducible.

    A single device->host read is not evidence on this host: runs have come back
    with a mask whose bytes belong to no tensor the model produced, and the wrong
    value is the *same* value every time -- a host buffer read before the copy
    landed, or one a later copy had already recycled.  Because the wrong value
    repeats, agreeing reads alone would settle nothing, so the value most reads
    agree on wins and a tensor whose reads never settle is refused rather than
    shipped.  Returns the array, its description, its digest, and how many reads
    it took; a third read happens only when the first two disagreed.
    """

    counts: dict[str, int] = {}
    values: dict[str, tuple[np.ndarray, dict[str, str]]] = {}
    needed = attempts // 2 + 1  # a strict majority of the reads this tensor gets
    for read in range(1, attempts + 1):
        # The digest is taken from this read's array rather than through
        # tensor_digest, which would encode -- and so read the device -- again.
        array, description = encode_tensor(tensor)
        digest = array_digest(array)
        counts[digest] = counts.get(digest, 0) + 1
        values.setdefault(digest, (array, description))
        settled = [seen for seen, count in counts.items() if count >= needed]
        if settled:
            array, description = values[settled[0]]
            return array, description, settled[0], read
    raise GoldVerificationError(
        f"refusing to produce gold: {attempts} reads of a captured tensor never settled on one value "
        f"(digests {[digest[:16] for digest in counts]}); the device->host path is not trustworthy on this host"
    )


def encode_capture(recorder: BoundaryRecorder):
    """Encode every captured tensor, each one verified by agreeing reads.

    Statistics come from the encoded array itself, so they cannot disagree with
    ``debug.npz``; the reads are what needed proving, not the arithmetic.
    """

    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, dict[str, Any]] = {}
    statistics: dict[str, Any] = {}
    retried: list[str] = []
    for key in sorted(recorder.tensors, key=capture_order):
        array, description, digest, reads = stable_encode(recorder.tensors[key])
        if reads > 2:  # only a disagreement can cost a third read
            retried.append(key)
        arrays[key] = array
        meta[key] = {**recorder.meta[key], **description, "sha256": digest}
        statistics[key] = statistics_from_array(array, dtype=description["dtype"], encoding=description["encoding"])
    return arrays, meta, statistics, retried


def collect_stages(stages, debug, debug_meta) -> list[dict[str, Any]]:
    """Write the sampled stage vectors, and describe them as stored.

    ``shape``/``dtype`` here describe the tensor the hook fired on; the stored
    array is the decimated float32 sample.  Both are recorded so a reader can see
    where the sample came from, which is why these entries stay out of
    ``statistics`` -- that block only ever describes arrays that are stored whole.
    """

    stage_arrays = []
    for index, (stage, calls) in enumerate(stages.items()):
        for call_index, record in enumerate(calls):
            key = f"stage_{index:04d}_call_{call_index:02d}"
            debug[key] = np.asarray(record["sample"], dtype=np.float32)
            debug_meta[key] = {"group": "stage", "stage": stage, "call": call_index,
                               "shape": list(record["shape"]), "dtype": record["dtype"], "encoding": NATIVE,
                               "sha256": array_digest(debug[key])}
            stage_arrays.append({"stage": stage, "call": call_index, "array": key})
    return stage_arrays


def reference_environment() -> dict[str, Any]:
    """The runtime versions a numeric baseline cannot be read without.

    `transformers` is the one that matters: the vendored replace is pinned to a
    version, and an unconditional gate would leave a baseline produced under one
    version indistinguishable from one produced under another. `numpy` decides what the decoding
    arithmetic runs through, and `transformers_replace_ok` is the vendored
    check's own verdict — recorded rather than enforced, so a mismatch is
    visible in the manifest instead of stopping the run.
    """
    import transformers

    from _vendor.openpi_pi0_pytorch import transformers_overlay

    return {
        "transformers": transformers.__version__,
        "transformers_replace_ok": transformers_overlay.REPLACE_STATUS.get("ok"),
        "numpy": np.__version__,
    }


def run(args) -> int:
    """The composition root: arguments in, a published artifact out.

    The order is the point.  The fixture and the checkpoint are read first, then
    the model is warmed up on the un-instrumented path, then one instrumented run
    records the tensors that become the artifact, and only then is anything
    encoded, verified and published -- so a run that cannot produce a trustworthy
    artifact ends in a refusal, not in a half-written directory.
    """

    if args.sample_count < 0 or args.sample_count > 4096:
        raise SystemExit("--sample-count must be in 0..4096")
    if args.num_flow_steps is not None and args.num_flow_steps <= 0:
        raise SystemExit("--num-flow-steps must be positive")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must not be negative")
    if args.timed_runs < 1:
        raise SystemExit("--timed-runs must be at least 1")

    device = resolve_device(args.device)
    checkpoint_root, weights, config = checkpoint_config(args.checkpoint)
    checkpoint_sha256 = sha256(weights)
    action_dim = int(config_value(config, "action_dim", "max_action_dim", default=DEFAULT_ACTION_DIM))
    action_horizon = int(config_value(config, "action_horizon", "chunk_size", default=10))
    max_token_len = int(config_value(config, "max_token_len", "tokenizer_max_length", default=200))

    arrays, num_steps, fixture_source = resolve_inputs(args, action_dim, action_horizon)
    validate_fixture(arrays, max_token_len=max_token_len,
                     views=None if args.input else args.views, horizon=action_horizon, action_dim=action_dim)

    model, model_config, dtype_name = load_model(weights, config, device, args.precision)
    boundary, noise = fixture_inputs(arrays, model_config, device)
    warmup_seconds, timed_seconds = benchmark(model, boundary=boundary, noise=noise, device=device,
                                              num_steps=num_steps, warmup_runs=args.warmup_runs,
                                              timed_runs=args.timed_runs)
    # The artifact comes from this run: warm by now, but carrying the hooks.
    recorder, stages, checks, recording_seconds = run_recorded(
        model, boundary=boundary, noise=noise, device=device, num_steps=num_steps, sample_count=args.sample_count
    )
    timing = timing_report(timed_seconds, warmup_seconds=warmup_seconds, recording_seconds=recording_seconds)

    try:
        debug, debug_meta, statistics, retried = encode_capture(recorder)
    except GoldVerificationError as error:
        return _refused(error)
    stage_arrays = collect_stages(stages, debug, debug_meta)
    # Derived rather than asserted. `stable_encode` proved each tensor settled
    # (it refuses the publication otherwise), so "stable" means "no tensor
    # needed a third read" -- a literal `True` here could not be checked against
    # anything inside the artifact, and would have read the same on a capture
    # whose re-reads the note below complains about.
    checks = {**checks, "capture_reads_stable": not retried}
    outputs = args.output.expanduser().resolve()
    manifest = {
        "schema": SCHEMA,
        "implementation": "vendored_openpi_pi0_pytorch",
        "execution": "torch_eager",
        "entrypoint": "PI0Pytorch.sample_actions",
        "model": "pi05",
        "observation_boundary": "PI0Pytorch._preprocess_observation is replaced: images are the fixture NCHW tensors, "
                                "token ids are the fixture ids, and no tokenizer or camera preprocessing runs",
        "derived_tensors": "prefix.position_ids is cumsum(prefix.pad_masks, dim=1) - 1; the 4-D attention masks are "
                           "where(step.*.att_2d_masks, 0.0, -2.3819763e38)",
        "statistics_contract": "each statistics entry is recomputed from the decoded array in debug.npz with "
                               "float64 accumulation (sum, abs_sum, l2 from float64 squares, max_abs in the stored "
                               "precision); the writer refuses to publish when the two disagree",
        "checkpoint": {"root": str(checkpoint_root), "weights": str(weights), "sha256": checkpoint_sha256,
                       "config": config},
        "runtime": {"device": str(device), "device_type": device.type, "torch": str(torch.__version__),
                    "python": platform.python_version(), "platform": platform.platform(),
                    "precision": dtype_name, **reference_environment()},
        "fixture": fixture_source,
        "inputs": describe_inputs(arrays),
        "outputs": {"raw_actions": {**debug_meta["raw_actions"], "file": "debug.npz"}},
        "sampling": {"num_flow_steps": num_steps, "dt": -1.0 / num_steps, "flow_start_time": DEFAULT_FLOW_START_TIME,
                     "euler_time": "float32 time starts at 1 and subtracts 1/num_flow_steps",
                     "sample_count": args.sample_count},
        "model_config": {"action_dim": model_config.action_dim, "action_horizon": model_config.action_horizon,
                         "max_token_len": model_config.max_token_len, "num_views": int(arrays["images"].shape[0]),
                         "paligemma_variant": model_config.paligemma_variant,
                         "action_expert_variant": model_config.action_expert_variant},
        "checks": checks,
        "statistics": statistics,
        "debug": debug_meta,
        "stages": stages,
        "stage_arrays": stage_arrays,
    }
    try:
        write_output(outputs, inputs={name: np.asarray(value) for name, value in arrays.items()},
                     debug=debug, manifest=manifest, force=args.force)
    except GoldVerificationError as error:
        return _refused(error)

    if retried:
        # Not a failure, but worth saying out loud: the host needed more than one
        # read to settle on these values.
        print(f"note: {len(retried)} captured tensor(s) needed a repeated read to agree: "
              f"{', '.join(retried[:5])}" + ("" if len(retried) <= 5 else f" and {len(retried) - 5} more"),
              file=sys.stderr)
    print(json.dumps({"schema": SCHEMA, "device": str(device), "output": str(outputs), "stages": len(stages),
                      "boundary_tensors": len(recorder.tensors),
                      "raw_actions_shape": recorder.meta["raw_actions"]["shape"],
                      "timing": timing}, indent=2, sort_keys=True))
    print(format_timing(timing))
    if alignment := alignment_block(args, debug, debug_meta, outputs):
        print(alignment)
    return 0


def alignment_block(args, debug, meta, outputs: Path) -> str | None:
    """How far this capture landed from the artifact its fixture was replayed from.

    ``--input`` names an ``inputs.npz`` that lives inside the artifact being
    replayed, so a replay can compare itself against that artifact without being
    told where it is -- which is the whole point of printing it here: the run
    that produces the numbers is also the run that reports them, and no second
    command has to know the two paths.

    This is a readout, not part of the artifact.  It is printed after the files
    have been published and stays out of the manifest, both because the reference
    is another artifact rather than part of this one's identity and because two
    runs over one fixture have to keep producing byte-identical manifest bytes.
    A reference that cannot be read degrades to a warning: a run whose gold is
    already on disk must not fail over a courtesy to whoever is watching.
    """

    if args.input is None:  # a generated fixture has no earlier capture to compare against
        return None
    reference = reference_artifact(args.input.expanduser().resolve())
    if reference is None or reference.resolve() == outputs:
        return None
    try:
        return format_alignment(compare_artifact(debug, meta, reference))
    except Exception as error:  # noqa: BLE001 -- see the docstring: a readout must not fail the run
        return f"alignment: could not read the reference artifact at {reference}: {error}"


def _refused(error: GoldVerificationError) -> int:
    """Report a refusal on stderr and hand back the process exit status."""

    print(str(error), file=sys.stderr)
    return EXIT_REFUSED
