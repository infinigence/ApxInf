# ApxInf Review Checklists

Simplified Chinese version: [checklists.zh-CN.md](checklists.zh-CN.md).
Historical evidence behind the expanded checks:
[historical-patterns.md](historical-patterns.md).

Use the sections that match the changed paths. High-risk sections require full
inspection of changed production code, integration callers, and relevant tests.

## Risk routing

High risk:

- Metal, CUDA, HIP, CoreML, MLX kernels and `unsafe` or FFI boundaries.
- Quantization, weight decoding, tensor layout, KV cache, attention, recurrence,
  sampling, model loading, tokenizer-to-embedding boundaries, multimodal input.
- Runtime traits, model dispatch, cache publication, concurrency and lifecycle.
- Execution sessions, CUDA Graph capture, prepared profiles, cuda-new autotuning
  and exact-key recipe persistence, native instantiation/build fingerprints,
  persistent caches, and device-memory arenas.
- Backend migrations, feature-graph changes, public ABI changes, and code that
  selects providers or fallbacks.

Medium risk:

- Configuration parsing, Python and CLI surfaces, benchmark harnesses, build
  features, platform selection, serialization and conversion tools.
- Documentation or policy changes that weaken architecture checks, validation
  gates, supported-device claims, or compatibility guarantees.

Low risk:

- Documentation, examples, formatting and mechanical renames, unless they alter
  commands, compatibility claims, unsafe contracts, or public behavior.

## Kernels, tensors, FFI, and devices

- Recompute every grid, group, index, byte offset, stride, alignment, padding,
  broadcast, and output extent with checked host arithmetic.
- Test zero, one, boundary tiles, non-multiple tails, maximum representable
  extents, mixed batch or head counts, and partial rotary dimensions.
- Verify shape and dtype guards dominate every cast and dispatch. Check signed
  narrowing and `usize` multiplication before native calls.
- Confirm buffer lifetime, stream or device ownership, synchronization, aliasing,
  temporary allocation, and asynchronous publication semantics.
- Re-derive ordering when a stream changes between blocking/default and
  non-blocking. Plain `cudaMemcpy`, `cudaMemset`, host reads, cache clear, buffer
  reuse, and pageable uploads do not automatically inherit another stream's
  order.
- Test output buffers with zero, dirty, and poison initialization when kernels
  claim full overwrite. Model-output sensitivity to initialization is a
  correctness failure until the first partial writer or ordering gap is found.
- Trace native-runtime ownership through every execution, session, graph, and
  eager handle. Do not rely on struct field declaration order for destruction;
  the owner must outlive every dependent native object on success and errors.
- Ensure capture begin/end and native resource preparation clean up correctly on
  every early return. Resource creation, tuning, allocation, and host callbacks
  must not occur unexpectedly inside capture.
- Compare numerical order, accumulation dtype, rounding points, epsilon, NaN and
  infinity behavior against the claimed reference.
- Inspect unsupported architectures, feature-disabled builds, CPU fallbacks, and
  error propagation. Compilation alone is not device qualification.

## Quantization and weights

- Verify packing order, group size, scale and bias layout, transpose convention,
  row slicing, padding, endianness and dtype conversions.
- Match parameter names, shapes and optional tensors against real checkpoint
  metadata. Account for sharding, tied weights and missing or extra keys.
- Ensure temporary decoded weights do not create unexpected residency or repeated
  hot-path conversion. Validate format and arithmetic variants independently.
- Bind calibration and execution-selection artifacts to every
  inference-relevant asset, not only safetensors. For cuda-new this means recipe
  identity; apply legacy tactic-database rules only to a path that still consumes
  that format. Processor configs, statistics, tokenizer assets, sharding indexes
  and auxiliary bundles can change model behavior.
- Make Rust/Python identity rules byte-for-byte canonical: identical shard
  selection, relative-path normalization, ordering, encoding and schema version.
- Treat legacy artifact migration and rejected profiles deliberately. Invalid or
  unsupported artifacts should fail closed or produce an explicit, documented
  fallback; never silently select a numerically different route.
- Check that fused and decomposed paths do not double-observe, double-quantize,
  or retain duplicate weights unless the memory cost is admitted and measured.

## Models, tokenizers, and multimodal inputs

- Range-check tokenizer and generated IDs before signed casts and embedding
  lookup. Verify special-token IDs and vocabulary agreement with the checkpoint.
- Trace prompt templates, schema or decision serialization, stop conditions,
  maximum generation limits, output parsing and malformed-model behavior.
- For images, audio or video, verify advertised capabilities reach preprocessing,
  encoder, projection, batching and runtime dispatch; do not infer support from
  upstream model capability.
- Check cache reset and reuse across requests, batch sizes and conversations.
- Validate capacity, divisors, dimensions and request limits before expensive
  vision/audio/device work or state mutation. Include zero, overflow and exact
  boundary values.
- Load optional modality assets conditionally. Derive advertised capabilities
  from assets actually loaded, and preserve text-only or reduced checkpoints.
- Verify every public trait shape and semantic contract, not only the shared
  generation caller's needs; returning final-row logits for a full-sequence API
  is still a contract violation.
- Trace user overrides and defaults end to end: Python/CLI -> load options ->
  admission -> runtime call. Metadata must not report an option that execution
  ignores.

## Runtime and API design

- Keep model category, loading protocol and execution protocol conceptually
  separate. A shared trait should express a real substitutable contract rather
  than force unrelated models to implement meaningless methods.
- Preserve the current contract boundaries: `LlmTrait` owns autoregressive
  LLM/VLM generation, `VlaRuntime` owns observation-to-action inference, and a
  decision-model protocol must be justified by its own substitutable semantics.
  Shared language or image inputs alone do not justify one execution trait.
- Check object safety, ownership, thread safety, error types, capability discovery,
  downcasting, public enum exhaustiveness and compatibility with existing loaders.
- Public names should reveal specialized layouts, dtypes and fixed geometry when
  callers must know them. Avoid generic names for hard-coded multiples or shapes.
- Ensure modules group one responsibility; do not hide multi-stage calling
  protocols behind broad names such as `runtime`, `utils` or `gemm`.
- Keep model mathematics in the model and execution/capture selection in the
  runner. Backend layers expose model-neutral operations and support queries;
  model code must not duplicate provider architecture predicates.
- Keep model families and backend-native ports isolated. A new native port
  belongs in its own current-architecture family module, must not import another
  family's private config, weights, model, runner or graph, and must preserve the
  shared user-facing registry/load contract. Run and inspect the repository's
  boundary guards; do not add an allowlist merely to make a cross-family import
  pass.
- Include all selection-relevant state in execution/profile/cache identity:
  exact dynamic shape, dtype, layout, pointer identity when required, precision,
  epilogue/fusion, device capability, policy, variant and generation.
- Instance configuration must not mutate process-global atomics or environment
  variables in a way that changes already-loaded models. Bind it to the instance
  or reject conflicting process-wide configurations explicitly.
- Keep registry, aliases and admission in sync. Every accepted model name and
  device route must pass the same `model_variant`, capability and option checks.
- Reject unsupported explicit options. Never silently ignore tuning paths,
  variants, precision, sampling, or backend choices supplied by a caller.
- Treat readiness reports as observations, not execution proof. Successful MLX
  preparation must report actual compiled scopes and admitted shapes through
  `TextPreparationStatus`; VLA `PreparationStatus` is separate, and
  `RuntimeManaged` does not establish graph or compiled readiness.

## Execution planning, graph capture, autotuning, and recipe persistence

- Distinguish logical capacity from exact execution shape. Reuse a prepared
  session only when every native execution key and operator sequence remains
  valid; changing `M`, token count or grid inside one padded capacity may still
  require re-prepare.
- Reconcile cache sharing with native state scope. A process/context-shared Rust
  cache cannot skip preparation that lives in `thread_local`, stream-local, or
  device-local native state.
- Prevent key aliasing between fused/decomposed operations, epilogues, providers,
  dynamic/static decode and shape-pinned/bucketed implementations. A cuda-new
  recipe key must cover every selection-relevant fact, and a persisted winner
  must be resolved and revalidated before preparing an execution.
- Preparation and tuning must be impossible during graph capture. Cache misses in
  capture should fail loudly, not lazily create descriptors or benchmark
  candidates.
- Bound execution/profile caches and device allocations. Check eviction,
  invalidation, pointer stability, changed-shape requests, cold-cache behavior
  and memory pressure rather than only warm steady state.
- Current cuda-new persistence is one exact-key `.recipe` file, not the legacy
  merged `tactics.json` database. Do not apply JSON record-merge, version-filter,
  or stale-lock rules unless a changed path actually reads or writes that format.
  For recipes, verify the complete key is stored and checked despite hashed file
  names, corrupted or obsolete winners miss safely, and publication handles
  same-process and cross-process writers without exposing partial content.
- Fast-path admission must include compiled/backend availability and provide an
  always-correct fallback. Unsupported provider combinations should be rejected
  during prepare, not at the first launch.

## MLX and Metal prepared execution

- Respect thread confinement: MLX `Array`, `Stream`, and `Compiled` ownership
  must not cross threads or acquire invented `Send`/`Sync` guarantees. Account
  separately for thread-local compiler state and process-wide allocator state.
- MLX operations are lazy. Evaluate every output and newly published KV/state
  array before committing request position or cache metadata; synchronization
  alone is not proof that a lazy expression was evaluated.
- Verify the actual compiled scope and shape profile. Variant names and a
  non-null compiled handle do not prove whole-model readiness; preparation must
  exercise admitted callables, changing inputs, retained state, and failure
  recovery.
- Treat any present `MLX_DISABLE_COMPILE` value, including `0`, according to the
  backend's presence-based disable rule. Do not report an eager callable as a
  compiled path.
- Measure bounded warm resource use after synchronization in an isolated process
  when making memory claims. Distinguish lazy temporary arrays, retained model
  state, compiler cache, and allocator cache.

## Memory and hot-path behavior

- Count allocations, retained arenas, duplicate weights, scratch buffers and
  execution objects by layer/profile. Verify the documented memory requirement
  includes prepared and fallback paths.
- Look for unconditional device-to-host readbacks, digest/statistics work,
  synchronizations, host interpolation, repeated conversion or plan creation
  hidden behind disabled printing or diagnostics macros.
- Do not launch a kernel for metadata-only reshape/reinterpret operations. Reuse
  existing views when element order is unchanged.
- Make fallback-only buffers and plans lazy. A fused winner should not allocate
  or tune resources used only by the decomposed path.
- Benchmark cold and warm starts, alternating profiles, cache misses, constrained
  memory and repeated requests. A warm single-shape benchmark cannot establish
  bounded production behavior.

## Concurrency, caches, and files

- Analyze same-process threads, multiple processes, crashes and retries separately.
- Temporary and staging paths must be unique per writer. Publication should be
  atomic, validated, and safe when another writer wins the race.
- Check lock scope, stale locks, partial artifacts, cleanup ownership, permissions,
  cache-key completeness and use-after-eviction.
- Validate before mutating shared metadata. Rejected calls must not update mapped
  decode state, cache generation, process configuration or published artifacts.
- Separate recoverable request validation from poisoned runtime state. One bad
  request must not permanently block later requests unless device/model state was
  actually mutated and cannot be rolled back safely.

## Build, features, ABI, native instantiation, and vendor code

- Exercise the real supported feature matrix: default/CPU, each device feature,
  workspace builds and relevant combinations. Hosted CI passing is irrelevant if
  it never compiles the changed CUDA/HIP/Metal path.
- Analyze Cargo feature unification across workspace members. Gate enum variants,
  match arms and dependencies on the feature that actually defines them, not a
  correlated model/backend feature.
- For backend migrations, compare lost dtype support, fast paths, configuration
  plumbing, environment gates, profiling and architecture checks against the old
  path. A net speedup does not excuse a broken default user route.
- For cuda-new, verify architecture selection, provider source inclusion,
  generated/template instantiation coverage, and target-SM compile/link behavior.
  Build fingerprints must change for every candidate/kernel/ABI input that can
  invalidate a recipe, without being coupled to unrelated documentation or
  model-only edits.
- Match FFI declarations on both sides, including status type, integer width,
  signedness, struct layout/version, optional pointers and ownership. Keep one
  error convention per operation family where possible.
- Record third-party kernel provenance: upstream repository and revision,
  original path, license, content hash when useful, and local modifications.
- Respect repository layer ownership: device kernels in kernel layers, adapters
  as validation/resource/dispatch, and model-specific policy outside shared
  backend APIs.

## Python, CLI, and cross-language contracts

- Preserve positional API compatibility when adding Python parameters; prefer
  trailing keyword-only options for public bindings.
- Keep CLI/Python defaults, aliases and explicit overrides consistent with Rust
  admission. Test default values as well as non-default overrides.
- Follow compatibility option names through their current semantics. In
  particular, a Python `tactics` or Rust `tuning_path` value reaching cuda-new
  may select a recipe directory or sibling `.recipes` directory; the name alone
  is not evidence that `tactics.json` is read.
- Reject missing explicit paths and synthetic-load leakage from the current
  working directory. User-specified artifacts must not silently become `None` or
  unrelated fallback files.
- Check EOS and budget boundaries after the same token-accounting order used by
  generation. A stop token in the final budget slot is completion, not truncation.
- Avoid machine-specific paths and PR checkout instructions in committed docs.
  User instructions must work from a normal repository checkout.

## Tests and benchmarks

- Require reference comparisons for numerical kernels and end-to-end model tests
  for model support. Include rejected shapes, dtypes and malformed configuration.
- Confirm ignored or hardware tests were actually run on named devices and exact
  revisions. Historical evidence must retain its original scope.
- Benchmark records should include revision, hardware, OS and SDK, model and
  quantization, data, command, warmup, repetitions, statistic, baseline, raw
  output and correctness checks.
- Prefer independent references. Two kernels changed to share the same rounding
  are not an oracle for each other; repeated output from the same binary is not
  an expected-token regression gate.
- Verify route attestation: report the provider, stream, binary/library hash,
  model variant, environment gates and loaded artifact actually used. Labels
  inferred from environment-variable presence are insufficient.
- Compare end-to-end outputs after backend or operator-stack migrations, not only
  isolated kernels. Include real checkpoints, non-square/multi-shape inputs,
  alternate requests, stop boundaries and state reset.
- Test errors behaviorally. Source-text grep tests can pass on comments and can
  freeze temporary architecture exceptions. Device tests should skip/ignore
  cleanly on unsupported hosts rather than `unwrap()` device creation.
- Run the exact-commit architecture and family-boundary guards affected by the
  change, then inspect their scope. Passing a guard that does not enumerate the
  new family, operator layer, feature, or native source is not coverage.
- Re-run affected hardware evidence after the final repair head. Source continuity
  can preserve narrow evidence but does not turn an old run into exact-head device
  qualification.

## Scope, documentation, and dead paths

- Remove probes, one-off experiment reports and intermediate artifacts from the
  production diff; keep them under `devlocal/` unless they are durable user docs.
- Trace every new operator through Rust, FFI, adapter, kernel and a production
  caller. Test-only or unreachable stacks should be removed or split into the PR
  that activates them.
- Reject unrelated family edits and opportunistic public renames in performance
  or migration PRs unless necessary for the stated path.
- Reconcile PR description, README, architecture docs, environment-variable
  tables, test names and shipped code. Policy-rule relaxations are substantive
  changes and must be reviewed explicitly, not hidden as cleanup.
- Treat very large delete-and-restore histories as a reviewability risk. Prefer
  runtime migration, per-model migration and legacy deletion as separate steps.

## AI-generated-code diagnostics

- Verify referenced APIs and symbols exist in the pinned dependency version.
- Compare near-duplicate paths for missing branches or divergent validation.
- Check comments and safety proofs against the implementation, not intent.
- Watch for tests that duplicate the implementation, assert only shape, or avoid
  the failing integration path.
- Check for comments that overstate core guarantees while validation actually
  lives in a backend, or document thread/device isolation inconsistent with the
  real ownership type.
- Search for options that are accepted, stored or reported but never reach the
  production call, and for capabilities advertised unconditionally despite
  optional weights or unavailable providers.
