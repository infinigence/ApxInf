# `apxinf-cuda-new` Architecture Contract

`apxinf-cuda-new` provides stable CUDA L3 semantic interfaces to the model layer and selects and prepares a provider kernel for the same semantic in the native layer. GEMM and Attention are the two operator families currently integrated, not the full set of types supported by the framework.

- Current public interfaces and mathematical semantics: [L3 operator catalog](cuda-operator.md)
- Workflow for adding or extending a kernel: [Adding New Kernels](../../doc/adding-new-kernels.md)

## Operator Layers: L3 to L0

L0-L3 here describe only the CUDA operators inside `apxinf-cuda-new`, not the model, policy, and serving layers in the repository root documentation.

| Layer | Responsibility | Input → Output | Main Interfaces and Directories |
| --- | --- | --- | --- |
| L3 Semantic (Rust) | Define the complete model-visible mathematical semantic and tensor contract | `CudaContext + Args` → `Result<()>` | `ops::<semantic>`; `src/ops/<operator>/` |
| L2 Execution (Rust) | Validate and normalize L3 calls; manage the execution cache, session, storage, and graph lifecycle | L3 Args → `Spec + Policy + Bindings` → opaque native handle | `normalize`, `prepare/execute`, Rust FFI declaration; `src/ops/`, `src/workspace.rs`, `src/graph.rs` |
| L1 Native operator (C++) | Implement the C ABI; perform recipe lookup, kernel selection, autotune, fallback, candidate/provider prepare, and enqueue | `Spec + Policy + Bindings` → native `Execution` | `*_prepare/*_enqueue/*_destroy`, registry and candidate callbacks; `native/adapters/`, `native/framework/` |
| L0 Kernel | Perform the actual GPU computation | provider launch arguments → GPU work | custom CUDA, FA2, CUTLASS, cuBLAS/cuBLASLt; `native/kernels/` or vendor API |

```text
L3 Rust → L2 Rust │ C ABI │ L1 C++ → L0 CUDA

L3: no provider       L2: no candidate selection
L1: owns selection    L0: no recipe/fallback/model logic
```

## Call Chain

```text
L3  Rust semantic API
L2  normalize → ExecutionKey/session cache → Rust FFI call
                     ───── C ABI boundary ─────
L1  recipe/selection → candidate/provider prepare → native Execution
L0  CUDA kernel or vendor API
```

```text
prepare: L2 → L1 *_prepare → recipe/selection → native Execution
run:     L2 → L1 *_enqueue ───────────────────→ L0 launch
capture: prepare_with_session once → capture/with_session reuses Execution
```

## Six Core Objects

| Object | Layer | Composition | Relationship to Other Objects |
| --- | --- | --- | --- |
| Semantic | L3 | Public mathematical semantic, tensor contract, and semantic ID | One Semantic has one independent candidate registry; L3 exposes it publicly |
| Provider | L1 (adapts L0) | Implementation-stack identity, such as FA2, CUTLASS, cuBLAS, or custom CUDA | One Provider can supply multiple Candidates; the Provider name does not enter the L3 API |
| Candidate / Implementation | L1 | Provider ID + implementation ID/version + capability attributes + callbacks | Registered under one Semantic; can enumerate multiple Configurations |
| Configuration | L1 | A stable configuration number defined by the Candidate | Has meaning only under its owning Candidate |
| Recipe | L1 | The Candidate's stable identity + one Configuration | Value in the recipe cache; records the winner but does not store pointers, handles, or provider state |
| Prepared Execution | L1; L2 holds an opaque handle | Current `Spec + Bindings + device + Candidate + Configuration + policy-derived limits + provider state/resources` | Prepared again from a Recipe or the current selection result; can be enqueued multiple times and is valid only in the current process |

Composition:

```text
Semantic → Registry<Candidate>
Provider + implementation identity + callbacks → Candidate
Candidate identity + Configuration → Recipe
Recipe key → Recipe
Current Spec + Policy + Bindings + resolved Recipe → Prepared Execution
Prepared Execution → Candidate enqueue → Provider → L0 kernel
```

```text
Registry = Candidate container          # not a seventh core object
Recipe   = winner identity              # not executable
Recipe hit → resolve Candidate → validate → prepare → Execution
```

## Three Kinds of Cross-Layer Data

| Data | Contents | Bound to the Current Call |
| --- | --- | --- |
| `Spec` | Normalized problem description: semantic, shape, dtype, mask, layout, alignment class, and so on | No; equivalent calls can share it |
| `Policy` | Constraints such as workspace, graph-safe, deterministic, and whether tune/fallback is allowed | Some fields affect selection or prepared state |
| `Bindings` | Addresses, stream, and actual dynamic values for this call, such as `alpha` and attention scale | Yes |

## Key Interfaces

| Interface | Layer | Responsibility |
| --- | --- | --- |
| `ops::<semantic>(ctx, args)` | L3 Rust | The only model entry point; expresses the complete mathematical semantic without exposing a provider |
| `normalize(ctx, args)` | L2 Rust | Validate the L3 contract, produce `Spec + Policy + Bindings`, and keep storage alive |
| Rust `prepare/execute` | L2 Rust | Look up the execution cache; create or enqueue an opaque native execution through FFI |
| `*_prepare(..., &execution)` | L1 C++ | Perform recipe/selection and create all provider state/resources outside capture |
| `*_enqueue(execution)` | L1 C++ | Submit only prepared work to the bound stream; must not select, allocate, or synchronize |
| `*_destroy(execution)` | L1 C++ | Release provider state and resources |
| `registry(spec.semantic)` | L1 C++ | Return the candidates eligible for the semantic |
| `supports(spec)` | L1 C++ | Declare the candidate's correctness domain, not a performance hint |
| `alignment_requirements/resource_requirements` | L1 C++ | Declare address and resource constraints before prepare |
| `enumerate_configs` | L1 C++ | Return every configuration that requires independent selection or benchmarking |
| `framework::autotune(problem, report)` | L1 C++ | Only iterate, time, and return a winner; does not interpret semantic or fallback |

```text
*_prepare → *_enqueue [0..N] → *_destroy
```

## Keys and Caches

| Cache | Layer | Key → Value | Lifetime |
| --- | --- | --- | --- |
| Execution cache | L2 Rust | `ExecutionKey → Rc<Execution wrapper>` | One `ExecutionSession` |
| Recipe cache | L1 C++ | `exact recipe key → Recipe` | Native runtime memory; optional disk persistence |

### Execution Key

| Component | Attention | GEMM |
| --- | --- | --- |
| Problem | Complete normalized `Spec` | Complete normalized `Spec` |
| Device and execution location | device, Q/K/V/offsets/output addresses, stream | device, A/B/bias/scales/output addresses, stream |
| Actual dynamic values | Bit patterns of `scale` and `output_scale` | Bit patterns of `alpha` and `output_scale`; B version/immutable flag |
| Policy affecting prepared state | workspace limit, graph-safe, deterministic | workspace limit, graph-safe, deterministic |
| Excluded from key | `online_tune`, `allow_fallback`, `cache_dir` | `online_tune`, `allow_fallback`, `cache_dir` |

### Exact Recipe Key

| Component | Contents |
| --- | --- |
| Cache/build namespace | recipe schema version, operator build fingerprint, compiled CUDA toolkit |
| Runtime environment | GPU compute capability, SM count; compatible CUDA runtime/driver version; GEMM also includes the cuBLASLt version |
| Attention equivalence class | semantic, input/output dtype, mask, batch, Q/K tokens, key capacity, Q/KV heads, head dim, query start, segments/max segment, default-scale predicate |
| GEMM equivalence class | semantic, M/N/K, A/B/accumulation/output dtype, quantization, B immutable, unit-alpha/unit-output-scale predicates |
| Candidate eligibility | Alignment class for each binding, workspace limit, graph-safe, deterministic |
| Excluded from key | Raw pointers, stream, input contents, actual scale, B version, device UUID, cache dir, tune/fallback switches, Attention offsets contents |

### Other Identities Are Not a Second Recipe Key

| Name | Purpose |
| --- | --- |
| Build fingerprint | One field of the recipe key; changes to candidate/kernel/ABI/framework build inputs make old recipes miss |
| Recipe value | `(provider_id, implementation_id, implementation_version, configuration)`, the winner, not a key |
| `.recipe` filename | FNV-1a hash of the exact recipe key, used only to locate the file; the complete key is still stored and compared inside the file |
| Typed cache `TypeId` | Separates different Rust `ExecutionKey/Execution` types; does not express operator equivalence classes |
| Prepared sequence identity | Verifies that prepare and capture use the same operator order; does not participate in recipe lookup |

### Recipe Hit Rules

| Condition | Behavior |
| --- | --- |
| Key mode | Exact key only; no compatible/bucket key |
| hit and identity, supports, configuration, policy, and prepare are all valid | Create the execution directly without benchmarking |
| miss, corruption, or failed validation after a hit, with `online_tune=true` | Fully benchmark every legal candidate/configuration |
| no usable recipe and fallback allowed | Use the baseline explicitly marked for the semantic; do not persist it as a tuned winner |
| no usable recipe and both tune/fallback unavailable | Return cache miss/unsupported |
| Persistence condition | Real prepare of the tune winner succeeds |

## Session / CUDA Graph State Machine

| Stage | Entry Point | Allowed | Forbidden/Failure Conditions | Result |
| --- | --- | --- | --- | --- |
| Prepare traversal | `prepare_with_session(&session, forward)` | recipe lookup, autotune, native prepare, provider resource creation, execute and record order | nested session | session typed execution cache + prepared sequence |
| Capture traversal | `capture(&ctx, || with_session(&session, forward))` | Hit executions in the same order and enqueue | cache miss, order change, device/stream change, native prepare | `CapturedGraph` |
| Eager reuse | `with_session(&session, forward)` | Reuse the same executions outside capture | cache miss or order change | asynchronous enqueue |
| Replay | `CapturedGraph::replay()` | Launch the instantiated graph | changing the captured structure | GPU work |

The graph retains the executions and storage used during capture. The recipe key does not contain raw pointers; fixed addresses and the stream belong to the Rust execution cache key.

## Directory Responsibilities

| Layer | Path | Sole Responsibility |
| --- | --- | --- |
| L3 Rust | `cuda-operator.md`, public APIs in `src/ops/<operator>/` | semantic, Args, and model-visible contract |
| L2 Rust | normalize/execution in `src/ops/<operator>/`, `src/workspace.rs`, `src/graph.rs` | ABI lowering, execution cache, session and CUDA Graph lifecycle |
| L2/L1 ABI | `src/ffi/abi/`, `native/include/apxinf_cuda/` | Stable Rust/C boundary and opaque execution handle |
| L1 C++ | `native/adapters/<operator>/` | recipe key, registry, selection, fallback, candidate/provider state, and prepared execution |
| L1 Shared | `native/framework/` | Operator-independent registry, autotune, recipe I/O, and error boundary |
| L0 | `native/kernels/` | Kernel source, template instances, and vendor tree |
| Build | `build_support/`, `build.rs` | Build target, build fingerprint, and native compilation |

## Change Matrix

| Change Type | Required Changes | Possible Changes | Must Not Change |
| --- | --- | --- | --- |
| Reuse an existing semantic | Model call site | Args/policy | registry, framework, ABI |
| Extend a candidate shape/dtype | candidate `supports`; provider/kernel; all-candidate tests | alignment/resource/config; build inputs | add a semantic, framework |
| Add a candidate | identity/version; all callbacks; correct semantic registry; tests | provider file, kernel, build.rs/fingerprint inputs | Rust L3 API, common autotune |
| Add a provider | provider callbacks/state; dependency and build integration; candidate registration | vendor provenance/patch | framework provider branch, model provider branch |
| Add a semantic | Rust Args/normalize/export; ABI enum/Spec; independent registry; fallback; catalog/tests | new fields, provider/kernel, ABI version; exact key/autotune when tuning is required | reuse another semantic's selection domain |
| Change candidate behavior/performance | implementation version or a build fingerprint covering the change; regression tests | recipe schema/key version | continue reusing old recipes that are no longer equivalent |

Every candidate must be tested on the inputs it claims to support, not only when it becomes the final autotune winner. See [Adding New Kernels](../../doc/adding-new-kernels.md) for complete testing and acceptance steps.
