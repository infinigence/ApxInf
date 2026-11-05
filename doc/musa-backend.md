# MUSA backend

`crates/apxinf-musa` runs ApxInf on Moore Threads MUSA devices. Its runtime was
ported from the former `apxinf-cuda` backend, preserving its allocation,
tensor and workspace contracts while using MUSA's own APIs (`musaMalloc`,
`musart` and `mublas`). The current CUDA runtime is `apxinf-cuda-new`; the two
backends evolve independently.

PI0.5 reaches it through `crates/apxinf-model/src/musa_pi05/`, a
backend-native port that **copies** what it needs instead of sharing the CUDA
family's code. That placement is the rule in
[model-layer-architecture](model-layer-architecture.md#where-backend-native-model-code-lives),
and it is what makes an upstream change to `pi05/` unable to alter MUSA's
numbers.

What is measured on M1000 (mp_22), PI0.5 BF16, 2 views, H = 10, 10 flow steps,
against the `torch-eager-gold.v1` bundle recorded on Orin:

| | captured graph | eager |
|---|---|---|
| p50, 15 samples after 3 warm-ups | 507.6 ± 1.3 ms | 512.6 ± 9.5 ms |
| vs Orin `raw_actions` | cosine 0.99995867, max abs 0.0128611 | same |

Both policies are bit-identical to each other, and repeated replays are
bit-identical to the first. Read the numbers with
[the benchmark caveats](#measuring-a-change) below; they move with host load,
which is why every figure here names the run it came from.

## Enabling it

`musa` is an additive feature. It coexists with `cuda` and `hip` in one binary:
each family owns its own top-level port, so nothing has to choose between them
at compile time.

```bash
export PATH="$HOME/.cargo/bin:$PATH"          # if cargo is not already on PATH
cargo build -p apxinf-musa                    # the backend alone

CARGO_TARGET_DIR=target/wheel maturin build --release --features musa \
  --auditwheel skip -m crates/apxinf-py/Cargo.toml
.venv/bin/pip install --force-reinstall --no-deps target/wheel/wheels/apxinf_py-*.whl
```

That wheel carries the backend without the MTTX range markers. Build with
`--features musa-mttx` instead to link them; `apxinf_musa::MTTX_AVAILABLE` says
which build a process got, and [profiling](pi05-mttx.md) covers the rest.

A device is `musa` or `musa:<index>` anywhere a device string is taken —
`ModelRunner.load(..., device="musa:0")`, `--device`, the Python API. The CLI
(`src/main.rs`) and the binding (`crates/apxinf-py/src/lib.rs`) parse it
separately and agree on what they accept.

The model name is `pi05`. `pi05-musa` is accepted as the backend-qualified
spelling, the way `pi05-cuda` and `pi05-hip` are, and for the same reason: a
caller that names the backend it means still reaches the family. The name is
admission only — it does not bind the device, which is what decides which
backend runs.

`ModelRunner.load(..., execution=...)` accepts `eager`, `prefer-graph` and
`require-graph` for MUSA PI0.5. Omitting it keeps `prefer-graph`. The separate
`weight_layout_budget_bytes` option is implemented by CUDA PI0.5 BF16/FP8;
it does not control MUSA allocations.

`crates/apxinf-musa` builds without a toolkit: with `MUSA_PATH` unset or
pointing nowhere, the build compiles the no-device configuration, warns, and
exposes `MUSA_AVAILABLE = false`, which is how a caller can tell. That is
deliberate — it is the configuration a CI job or a checkout on another machine
gets, and it still type-checks every feature combination, which is the one
property that would otherwise only be checked on a machine that has the
toolkit.

## Offload target

The kernels are compiled for one generation, chosen at build time:
`APXINF_MUSA_ARCH` if set, else a probe of the installed toolkit. A machine can
hold two generations (M1000 and an S5000, say), and launching one generation's
image on another is not a degraded run — it is a driver-level failure with no
attribution, or a plausible wrong answer.

So the build **refuses to guess**: an `APXINF_MUSA_ARCH` value that is neither
blank nor a known name is an error naming the setting, rather than silently
falling back to detection and surfacing the mismatch at run time. The probe
itself is advisory — when it cannot run, the build warns and keeps the default,
because a probe that fails hard turns a toolkit problem into a disabled backend.

`MusaContext::new` re-checks at run time: a binary built for `mp_22` opened
against a different generation is refused with both values in the message.

## Generations

[`MusaArchFamily`](../crates/apxinf-musa/src/device_caps.rs) classifies a
compute-capability number: `Mp22` (M1000; BF16 MMA uses a `32x32x16` atom, no
TMA or cluster unit), `Mp31` (S5000/S4000, which add TME and SQMMA), and
`Other(n)`.

`Other` is not "unknown hardware". It is the arm a generation lands in before
this backend has measured what it can run, and its capability questions answer
`false` on purpose: the conservative answer, and the **silent** one. Opening a
context on such a device prints one line per process naming the generation, so
that "a kernel that exists for this generation is unused rather than absent"
stops being invisible.

## Relationship to the CUDA backends

The legacy `apxinf-cuda` tree was removed when all CUDA families migrated to
`apxinf-cuda-new`. Its source at
[`01232a9a`](https://github.com/infinigence/ApxInf/tree/01232a9a8220977a76094907eaed6da2ef1ba0db/crates/apxinf-cuda)
is the historical origin of MUSA's runtime, not a second live dependency.
The current relationship is:

- **Surface serves the MUSA port.** The typed `ops/` contract, `ExecutionSession`
  and `graph::capture` were shaped after `apxinf-cuda-new`. They cover the
  independent `musa_pi05` BF16/INT8 paths; CUDA-only additions such as
  `RepackedWeight` do not automatically become MUSA requirements.
- **Interior follows legacy CUDA.** `kernels/`, `ffi/`, `buffer`/`context`/
  `stream`/`device_caps` and the flat `adapters/*.mu` are ported from
  `apxinf-cuda`, which is the tree this backend was taken from.
- **Kernels are a subset, rewritten where the hardware forced it.** 35 device
  kernels against the original CUDA tree's 221; the 16 that keep the same name
  are the ones PI0.5 uses. Their bodies differ where M1000 made the upstream shape wrong:
  row-grid launches (64-bit integer division is a software routine here), a
  precomputed RoPE frequency table, register-cached W8A8 quantization, FP16
  scales with a 512-entry zero pad for the `s8mma` tiles, and `copysign`
  rounding in place of `roundf`. Each rewrite states the measurement behind it.
- **The comment density is deliberate.** These files carry roughly three times
  the comment-to-code ratio of the CUDA originals. That prose is what makes a
  rewritten kernel reviewable against its origin; it is not something to trim
  into line with upstream.

Relative to that historical source, the `kernels/custom` file set maps one-for-one
onto `apxinf-cuda`'s, `custom_kernels.mu` uses the same anonymous-namespace plus
single-C-ABI translation-unit pattern, and every ported `extern "C"` signature
has the same parameter order and meaning. Where a signature does differ —
`quantize_rows_bf16_int8` grew the per-channel equalization vector — the adapter
README says so.

`scripts/check_ops_contract_surface.py` checks the operators imported by the
MUSA model port, including its INT8 path. `crates/apxinf-musa/tests/backend_parity.rs`
checks literal environment reads in the current CUDA Rust/native sources and
MUSA backend/model sources. Its table documents the current differences;
removed legacy CUDA allocation switches are not MUSA counterparts.

## PI0.5 on MUSA

`crates/apxinf-model/src/musa_pi05/` is self-contained: 18 files, no import of
`crate::pi05`. `scripts/check_model_family_boundaries.sh` enforces that in both
directions. Differences from the upstream file a copy came from belong to the
port; upstream changes to `pi05/` are audited against each copy's import
watermark with [copy-drift-audit](../skills/copy-drift-audit/SKILL.md), so a fix
upstream cannot pass the port by unnoticed.

The layers follow `pi05/model/blocks/bf16.rs` operator for operator, with the
same rounding points, so a stage-by-stage comparison is meaningful. Differences
that do not change a value:

- Per-step modulation is a constant of the model (the flow schedule comes from
  config, the projections from the checkpoint), so it is built once at load and
  held by the model. That is also what makes it safe to capture: the tensors
  live as long as the graph does, unlike a temporary built before the region.
- Decode GEMMs run on the MP22 assembly tiles, and the encode path takes the
  registered variant for its shapes. Both are prepared at load.
- Immutable weights are registered once at load (`prepare_assembly_weights`),
  held for the context's lifetime, and must not be mutated afterwards. There is
  no per-call candidate cache, so there is no weight-version token to key one.
- The prefix cache reserves the action rows, so language layers write K/V
  straight into it.

### INT8 (W8A8)

`model_variant=int8_dynamic` and `int8_dynamic_no_encode` run the route upstream
and the integration branch define. `auto` still resolves to BF16 here: whether a projection
can be quantized is a per-shape question, and MP22's `mp` number is not a compute
capability. What the port adds to the device path:

- The tiles read their dequantization scales as **FP16 with a 512-entry zero
  pad**, not f32, so the weight packer renders that vector once at load and the
  row quantizer writes the activation half in the same form. Nothing converts
  scales per GEMM.
- `geglu(gate_up)` and its row quantization run as one kernel for the down
  projection, reproducing the materialized chain's intermediate BF16 rounding in
  registers. The fused output is byte-identical; a device test checks it across
  150..16392 columns and falls back for views the kernel cannot take.
- The action traversal marks itself as quantized decode, and its short-row GEMMs
  take NN/NT images rendered from the BF16 GEMM template; gate/up's KN weight
  copy is packed once at load, so no transpose runs per call.
- A projection no tile serves keeps BF16 arithmetic — the ViT patch embedding
  (`k = 588`), the action input and the action head — and the ViT MLP is zero
  padded from 4304 to 4352 so that it does not.

Full-encode quantization needs per-channel equalization to clear the 0.999 gate:
the encode layers' shared residual stream has outlier channels one `amax/127`
row scale cannot cover. `APXINF_INT8_EQUALIZATION` names the checked-in
`configs/pi05_int8_equalization_q50_g60_d60.json`; the fold costs one multiply
per lane inside the quantizer. That table is built by
`scripts/build_pi05_int8_equalization.py` from `APXINF_PI05_INT8_STATS` row
levels, and the script's docstring records how to rebuild it and why a schedule
change invalidates it. `int8_dynamic_no_encode` keeps those four language
projections in BF16 instead and needs no calibration.

Measured on M1000, two views / ten PaliGemma tokens / ten flow steps, against the
Orin bundle (3 warm-up, 15 samples, equalization set, frozen checkpoint copy):

| variant | execution | L1 p50 | cosine |
|---|---|---:|---|
| `bf16` | eager | 512.6 ms | 0.99995867 |
| `bf16` | graph | 507.6 ms | 0.99995867 |
| `int8_dynamic` | eager | 361.5 ms | 0.99936157 |
| `int8_dynamic` | graph | 363.2 ms | 0.99936157 |
| `int8_dynamic_no_encode` | eager | 502.2 ms | 0.99963635 |
| `int8_dynamic_no_encode` | graph | 502.9 ms | 0.99963635 |

The integration branch on the same machine measures 513.4 / 510.1 ms for `bf16`, 406.4 /
363.2 for `int8_dynamic` and 545.1 / 502.9 for `int8_dynamic_no_encode` (eager /
graph), with the same cosine to the last digit in every row. Both captured routes
are at parity. Eager is 11% ahead on `int8_dynamic` for one reason: the port
builds every step's modulation once, when the model is assembled, because the
flow schedule and the projection weights are constants and so is what they
project to. A traversal that recomputes it pays 390 `m=1` GEMMs — and inside a
captured graph it paid them on every replay, which is what the graph column used
to trail dev by.

`int8_dynamic_no_encode` is the slower of the two INT8 routes because its BF16
language layers do not reuse the BF16 route's assembly registrations; it lands
where plain BF16 does (502.9 against 507.6 ms captured) and exists for accuracy,
not speed. The capture's arena is not CUDA's either — a measured INT8 capture
touches 3.84 GB, so the reservation is the BF16 one rather than the doubled one
upstream keeps for its INT32 accumulator and verifier.

FP8 is **not ported**, and the port says so rather than silently degrading: an
FP8 calibration input is refused by name, and `ModelVariantChoice` rejects
`fp8_static`.

## Serving on MUSA

The websocket policy server runs unchanged on this backend: `WebsocketPolicyServer`
is transport for any `Policy`, and `--device musa:0` reaches the same
`ModelRunner.load` everything else does.

One caveat about this repository's checkpoint. `pi05_libero_pytorch` is an OpenPI
export whose `config.json` has no `type` field and which carries no `metadata.pt`,
so `scripts/pi05_openpi_websocket_server.py`'s preflight refuses it under either
`--ckpt-format` — it cannot tell what the directory is. The example server has no
preflight and serves the same model:

```bash
.venv/bin/python python/apxinf/examples/openpi_server.py \
  --model-dir <checkpoint> --device musa:0 --model-variant bf16 \
  --tokenizer <path-to-paligemma_tokenizer.model> \
  --image-keys observation/image,observation/wrist_image \
  --state-key observation/state \
  --policy-options '{"model_type":"pi05","num_views":2,
     "norm_stats":"<checkpoint>/assets/physical-intelligence/libero/norm_stats.json"}'
```

Every flag there is load-bearing on this checkpoint: without `model_type` the
policy cannot be dispatched; the tokenizer is not in the checkpoint; without
`norm_stats` the statistics resolve to identity and actions come back in the
normalized domain, with only a warning; and the checkpoint declares three
cameras, so `num_views` and the key list have to agree with what the client
sends.

Three things about the server are worth knowing before deploying it, and none of
them are visible from the wire.

**Inference is serial, one request at a time.** The engine handle is
`unsendable` — its context belongs to the thread that built it — so the server
runs the policy on its event loop and concurrent clients queue. One server per
card; `--device musa:N` picks the card.

**Consecutive identical requests do not return identical actions.** The engine
draws a fresh flow-matching latent per request from its own RNG stream, which is
a stochastic policy behaving normally. To make a call reproducible, send the
latent: it travels in the observation under `"noise"`, a key `Pi05Policy` already
reads, so the wire needs no extension and a client that sends none is unaffected.
With a fixed latent two calls must come back **bit-identical** — that is the one
wire-integrity check this path has.

**The device index reaches every layer.** `--device musa:N` sets the driver
context, and every allocation and launch takes its device from that context;
`cargo run -p apxinf-musa --example device_report N` answers "would this build
serve card N?" without loading a model.

For the far side of a split deployment — render and drive on a robot host,
infer on the card — `scripts/pi05_remote_client.py` is the client. It is
self-contained (numpy, msgpack, websockets, plus a byte-identical copy of the
`apxinf.serving` codec pinned by `tests/test_pi05_remote_client.py`), so it runs
on a host with no `openpi_client` and no checkout. Its `RemotePolicy` reads the
server's metadata at connect and sends exactly the keys the server published, and
`--noise-seed` makes a request reproducible:

```bash
.venv/bin/python scripts/pi05_remote_client.py --host <server> --port 8000 \
  --noise-seed 0 --requests 5 --dump-actions actions.npz
```

The benchmark's L3 attaches to a running server through the same client and times
one round trip per call. It cannot share a 32 GB host with the server when
in-process layers are also selected — that is a second model in a second process.

Measured on M1000, `bf16`, two views, 224×224 synthetic frames, one warm-up and
ten samples: round trip p50 **514.0 ms**, of which the model is 510.8, the
policy's own processor chain 0.75 and the socket 1.97.

**Bit-identity across processes is not something this path promises.** Inside one
server, two requests carrying the same latent come back identical, and an
in-process load of the same checkpoint reproduced the second server's actions
byte for byte. A first server process of the same build returned actions
differing by 3.3e-3 — the size of a provider-selection difference, not of a
transport one. Read the seeded comparison as "the same server reproduces
itself", not as "every process computes the same bits"; a cross-host comparison
needs the tolerance the accuracy gate uses, not equality.

## Capture

Capture is the contract to read before changing any operator. Vendor state
(per-stream handles, per-shape descriptors and plans) is created on first use
and **only outside capture**, so a graph records a traversal that has already
run once on the same stream. `ExecutionSession::prepare` is that pass;
`ExecutionSession::run` refuses a GEMM or attention shape the prepare pass never
reached, naming the shape, and `prepared_shapes()` lists what it covered.

The stream is the context's own, and that is what keeps warm-up and capture
together: `ExecutionSession::prepare` runs on `MusaContext::stream` and so does
the capture that follows, so there is no second stream for a prepared resource
to be launched into.

A capture that fails is not a recoverable failure on this device: the runtime
leaves the stream in `INVALIDATED`, where every later launch is refused and even
`musaDeviceSynchronize` fails for as long as it exists, and the error slot keeps
a status that every adapter would report as its own verdict. Nothing clears that
stream state. `graph::capture` therefore hands the context a fresh stream when
leaving the region fails, and empties the error slot with it, so a caller that
falls back to eager — which is what `PreferGraph` asks for — has a stream to
fall back onto.

## Tuning database

A GEMM shape's provider is chosen from a MUSA tactics database, looked up in
two steps: the toolkit-qualified directory wins when it holds a `tactics.json`,
and the hardware-only directory is the fallback.

```text
configs/tuning/musa/<device-name-lowercased>-mp<mp>/
├── tactics.json
├── tuning_report.json
└── musa<major.minor>-mublaslt<major.minor>/
    ├── tactics.json
    └── tuning_report.json
```

`TuningPaths::resolve_for_device` derives that name from the running device's
caps and library versions, and resolves `configs/tuning` against the **process
working directory** — run from the repository root. A database is written by
`scripts/bench_pi05_musa.py --autotune` (or `ModelRunner.load(autotune=True)`),
and `--tactics DIR` names one explicitly — the directory, since the loader
appends `tactics.json` itself.

Records are bound to the kernel build id, the MP generation, the device name and
the vendor library fingerprints. A database that does not match is ignored with
a printed reason rather than silently used: a stale one measures another build,
and copying one between machines is the usual way that happens.

No MUSA database is checked into this repository today. M1000's shapes are
already covered closely enough that autotuning produces no measurable gain, so
the directory above is an artifact of a local tuning run rather than a shipped
file. A checkout without `configs/tuning/musa` is normal: every shape uses the
safe default, and `cargo test -p apxinf-musa --lib` validates a database only
when one is present. The same test covers an S5000 asset if one is added later.

Tuning itself is a load-time setting: `--tactics` and `--autotune` live in
`LoadOptions` and are installed once before prepare or capture. They are not
part of the per-call operator policy.

## Operator policies

`GemmPolicy` and `AttentionPolicy` are enforced at the operator entry point,
not carried and ignored:

- `accumulation_dtype` must be F32 — the adapter's `MUBLAS_COMPUTE_32F`
  contract. Anything else is refused.
- `workspace_limit` bounds the bytes the operator allocates for itself: none for
  a plain GEMM or attention, `rows × gate_cols × 2` for `gemm_geglu`. Vendor
  libraries allocate their own internal workspaces outside this accounting.
- `allow_fallback = false` requires a tactic record measured at the exact shape.
  A bucket extrapolation or the muBLAS default is refused; autotuning may
  measure the shape and write the record. On attention it requires the packed
  decode route for a decode-shaped call, per build: the assembly build's packed
  decode kernel, or the MUTLASS build's packed tile 128 (9..=16 query rows).
  The general D256 assembly, and the MUTLASS unpacked tile 192 for 17..=256
  rows, are slower substitutes and are refused; from 257 rows the unpacked tile
  is MATE's own choice, and a prefix pass uses the general route by
  construction. The assembly GEMM route is a platform optimisation and does not
  count as a fallback.
- `deterministic = true` is refused: there is no bitwise-determinism contract
  across the available providers.
- `graph_safe` is accepted either way: MUSA's routes are capture-safe
  unconditionally, so `false` is a weaker request that is satisfied too.

## Measuring a change

```bash
# Latency and the accuracy gate in one run: L0 is both the timed and the scored
# call, so the cosine and the p50 describe one configuration.
.venv/bin/python scripts/bench_pi05_musa.py --layer l0,l1 \
  --reference <gold-bundle> --execution eager
.venv/bin/python scripts/bench_pi05_musa.py --layer l0,l1 \
  --reference <gold-bundle> --execution require-graph

# Latency only, no bundle: synthetic zero inputs at the loaded checkpoint's shapes.
.venv/bin/python scripts/bench_pi05_musa.py --layer l0,l1
.venv/bin/python scripts/bench_pi05_musa.py --layer l0,l1 --views 2 --token-count 10
```

`execution=` on `ModelRunner.load` — the same policy the benchmark selects with
`--execution` — is read only by this port. The CUDA and HIP runners hardcode
`PreferGraph`, so a non-default value is silently ignored there; set it only
when the device is `musa`.

The benchmark's own `--execution` defaults to `prefer-graph`, so a run without
the flag times the captured path when capture succeeds; `execution.fallback` in
the report says whether it did. Pass `--execution eager` for the eager baseline
and `--execution require-graph` to turn a capture failure into a refusal.

```bash

# L2 needs a tokenizer no checkpoint ships, and LIBERO's own statistics
.venv/bin/python scripts/bench_pi05_musa.py --layer l0,l1,l2 \
  --reference <gold-bundle> \
  --tokenizer <path-to-paligemma_tokenizer.model> \
  --norm-stats <path-to-pi05_libero_base>/assets/physical-intelligence/libero/norm_stats.json
```

The gate scores the engine's `raw_actions` against a `torch-eager-gold.v1`
bundle named by `--reference` — a directory holding `inputs.npz`, `debug.npz`
and `manifest.json`. Bundles are not committed, so each host records one with
`python/apxinf_ref/pi05/generate_pi05_torch_gold.py` and passes it on every
scored call. `--reference` decides both what is measured and whether it is
scored: its frames are the workload, which is why omitting it switches the run
to synthetic zero inputs (`--views`, `--token-count`, and the checkpoint's own
solver schedule) and to a report that says only how long the layers took — the
console prints `accuracy: not scored (no --reference)` and the exit status is 0.
The gate refuses a bundle it cannot compare against — a checkpoint whose
bytes differ from the one the bundle records, a flow schedule other than the
one the engine derives, padded tokens, a mis-shaped noise tensor, or repeats
that are not bit-identical — instead of scoring it. Exit status is 0
measured-and-passed (or measured only), 1 compared-and-failed (cosine <
`--gate`, default 0.999), 2 refused.

L0 is the scored layer when a bundle is given, because it is the call the bundle
fixes exactly; without one nothing is scored and only the timings are reported.
With a bundle, L1 and L2 take its frames through a lossy float32 -> uint8
requantization, which is what a client sends but not what was recorded, so they
are latency only: scoring them would measure the requantization rather than the
engine. Their deltas are not boundary promises either — L0 marshals a host patch
buffer while L1 copies uint8, so L1−L0 can come out negative. A synthetic run
keeps the same shapes and dtypes, so its deltas mean the same thing; its actions
mean nothing, which is why `--export` is refused without `--reference`.
The cosine is computed in float64 over the whole chunk (the artifact module's
own), so a value recorded at float32 precision can differ in its eighth digit
without the actions differing; the band and the gate are what it is read for.

Two caveats for any latency number. First, **eager and captured must be measured
in the same session and compared as p50 with their spread**, because this device
is memory-bound and its spread moves with host load: runs an hour apart can
differ by several percent with no code change. Second, the `.venv` wheel has to
be rebuilt after any change to `crates/apxinf-musa` or the port — a stale wheel
measures the previous build, and nothing else tells you that.

## Verifying the port's own tests

```bash
export MUSA_PATH=<path-to-musa-toolkit>
cargo test -p apxinf-model --features musa --lib musa_pi05::
APXINF_PI05_TEST_CHECKPOINT=<checkpoint dir> \
  cargo test -p apxinf-model --features musa --lib -- --ignored
```

The first runs without a checkpoint. The second needs a real one: it includes
the capture acceptance test, which builds an eager and a captured plan from one
loaded model and compares raw action bits under a changed latent.
