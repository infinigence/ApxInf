---
name: analyze-model-impl
description: Trace an inference model's current implementation from model computations through Rust operations and backend dispatch to CUDA kernels. Use for implementation walkthroughs, kernel-graph analysis, fusion analysis, and tables explaining which kernels an active model configuration actually executes.
---

**Analyze model implementation**

Produce an implementation-backed execution map and explain its consequences for
precision, memory, graph capture, and performance. Analyze the requested build
and execution path; kernel availability and design proposals alone do not prove
that a model uses a kernel. An analysis request does not by itself require model
changes, a rebuild, remote GPU execution, or new benchmarks.

**Establish the implementation being analyzed**

- Resolve the repository and model from the request and session. Record the
  source revision and relevant working-tree changes; identify the checkpoint,
  target architecture, and runtime configuration from available evidence.
- Distinguish the default path, the configuration used in an accepted run, and
  experimental alternatives. Use the configuration under discussion when clear;
  otherwise state which path is being analyzed. Follow feature gates, environment
  defaults, precedence, shape restrictions, and architecture dispatch.
- Read relevant model/design/evaluation documents for context, then verify their
  claims against executable code. Label older timings, proposals, and stale
  comments instead of incorporating them as current implementation facts.
- Preserve the user's precision and checkpoint choices. Identify unresolved
  dispatch branches or missing artifacts explicitly; continue mapping paths that
  can be established locally.

**Trace the execution chain**

Start at the public loader and forward entry point. Follow the concrete model
runtime through the Rust operation, backend/provider selection, C ABI adapter,
and device implementation. Record the condition selecting each branch.

For ApxInf, useful starting locations relative to the selected checkout are:

- `crates/apxinf-model/src/<model>/`: configuration, weights, public integration,
  and runtime scheduling.
- `crates/apxinf-cuda/src/kernels/`: typed operation contracts and dispatch.
- `crates/apxinf-cuda/src/ffi/`, `src/cublas.rs`, `src/graph.rs`, and
  `src/backend.rs` within that crate: ABI, vendor calls, and graph lifecycle.
- `crates/apxinf-cuda/adapters/`: exported launchers and template selection.
- `crates/apxinf-cuda/kernels/`: custom and vendored device implementations.

Use `rg --files` to locate the real files before assuming a directory layout.
Search exact symbols in the relevant subtree, then expand into vendor code only
when dispatch leads there. Read call sites and launcher bodies as well as
definitions. An operation name such as `fused_qkv` does not establish whether
the projection itself, only its epilogue, or several launches are fused.

For custom kernels, identify the selected `__global__` function and meaningful
template parameters. For vendor libraries, record the API, operand/output
datatypes, and algorithm selection contract. Do not invent an exact cuBLAS
device-kernel symbol from a source call or reuse an old profiler name as proof
of current dispatch. Identify host computation or another backend explicitly
when a stage has no CUDA kernel.

**Map computations and data representation**

Verify model dimensions from the actual configuration or attributable checkpoint
evidence. Do not assume query width equals hidden width. Track projection shapes,
head grouping, expert count/top-k, intermediate width, and quantization group size
where they explain kernel selection.

Separate initialization from repeated execution:

- Loading: concatenation/transposition, quantized layouts, repacking, scale/zero
  transformations, dtype conversion, persistent weight caches, lookup tables,
  device versus mapped placement, and workspace allocation.
- Forward execution: embedding, normalization, projections, position encoding,
  cache updates, attention, routing, expert products/activation, weighted combine,
  residual operations, final normalization, and output head as applicable.
- Generation boundary: distinguish model logits from sampling, tokenization, and
  transfers performed by the caller.

For each stage, trace weight storage, activation type, accumulator type, output
type, and important rounding boundaries. INT4 weight storage does not imply
integer matrix arithmetic or INT4 computation throughout the model. A fused
operation can preserve intermediate rounding; verify the implementation before
claiming that fusion changes precision.

Keep initialization, common entry/output work, and prefill/decode tables
separate when their graphs differ. For other model types, use the actual phases
or modalities. Make operation frequency explicit: once per model load, request,
chunk, layer, expert assignment, or generated token. Locate norms and reductions
that are folded into the preceding layer rather than adding duplicate rows.

**Inspect graph capture and memory behavior**

Follow the runtime's actual loop and capture lifecycle, including:

- Prefill/decode selection for an empty cache, a single token, and an existing
  cache; chunk size, partial chunks, and expert-tile padding.
- Cache capacity versus actual used prefix length; which values are captured
  constants and which are read from mutable device buffers.
- Graph keys, graph reuse/eviction, stable buffer addresses, and the capture-miss
  path. Inspect graph helper implementations to distinguish eager execution,
  recording, instantiation, and replay.
- Host uploads, downloads, synchronizations, CPU routing, and allocations inside
  or outside the captured body.
- Residual/KV/weight storage versus activation/routing scratch. Explain which
  allocations grow with chunk size, total context, or graph count.
- Work whose result is overwritten or unused, such as an output head evaluated
  on every chunk when only the final chunk's logits are returned.

A graph replay submits many dependent kernels. Count launches inside adapters:
one Rust operation can launch several kernels, while a single kernel can perform
several model computations. If counts help, report explicit device launches and
vendor-library calls separately, label them as static counts, and state their
scope. Do not treat a library call as exactly one device launch or include
load/capture work in steady-state counts without saying so.

**Present the mapping and assessment**

Lead with the main implementation choice and the source/configuration scope.
Use tables in execution order. The full mapping can use this schema; merge the
ABI column into the CUDA column for a compact answer:

| Model computation and frequency | Rust op | C ABI / vendor call | Selected CUDA kernel | Shape, precision, fusion, or memory detail |
|---|---|---|---|---|

Use actual symbols. Define aliases once when repeated template names would make
the table unreadable. Include navigable source locations for the orchestration,
Rust contracts, and launchers/device definitions. A compact graph is useful when
it clarifies dependencies beyond the ordered tables.

Explain what the map establishes: active fusions, remaining materializations or
conversions, routing location, weight reuse, repeated work at longer context,
and the graph's remaining host dependencies. Distinguish implemented and enabled
paths from disabled experiments and proposed optimizations. Treat bottleneck
claims as hypotheses unless supported by attributable measurements; source-level
counts alone do not establish runtime cost or predict a speedup.

When discussing acceptance, separate numerical correctness, byte equality,
generation quality, and benchmark milestones. State which fixtures/configuration
were validated and which remain unresolved. Do not infer long-context numerical
acceptance from short fixtures or matching greedy tokens.

Follow the project root [AGENTS.md](../../AGENTS.md#development-artifacts)
for all generated artifacts. Save task-specific reports under
`<project-root>/devlocal/<feat-name>/reports/` and link the result in the response.
Promote a report to maintained documentation only when that is part of the task. Keep machine paths,
checkpoint-specific flags, measured speeds, and this session's kernel choices in
the report rather than treating them as universal skill requirements.

**Check the result**

Verify that every selected path follows the recorded configuration and that
every CUDA name is a device kernel or clearly labeled library call. Check that
tables cover entry/output operations, layer repetitions, and fused next-layer
norms without double counting. Validate source links, table formatting, and any
reported counts. State whether evidence came from source inspection, recovered
measurements, or newly executed tests. Do not claim runtime validation for a
static analysis.
