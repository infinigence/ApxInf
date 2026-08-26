# Workflow for Adding New Hardware, Models, and Missing Operators to ApxInf

This guide covers the vertical path for a genuinely missing operator. Before
adding one for a model port, use
[Model Execution Wiring](model-execution-wiring.md) to check maintained fused
interfaces, device compositions, and runtime lifetime requirements. Absence
from the portable backend trait alone is not evidence that a CUDA capability is
missing.

This document is intended for an agent responsible for porting a working PyTorch reference model to ApxInf. It assumes that the model already runs on the target hardware and that an initial scan has identified operators, data types, shapes, or hardware implementations missing from ApxInf.

The goal is not to translate the PyTorch graph node by node. The goal is to:

1. Confirm that the ApxInf backend, build system, and runtime capability detection support the target hardware.
2. Map PyTorch computations to existing model-neutral ApxInf operators.
3. Add operators or implementation paths only for genuine capability gaps.
4. Connect the model layer through the safe Rust kernel, FFI, host adapter, and GPU implementation.
5. Validate individual operators, layers, and the complete model against the PyTorch reference.
6. Consider fusion and autotuning only after correctness is established and profiling data is available.

## 0. Execution Principles

- Read the existing code and call paths before editing. Do not infer the architecture from filenames alone.
- Model runtimes and executors may call only safe Rust operator APIs. They must not call raw FFI, cuBLAS, cuBLASLt, or CUDA kernels directly.
- Organize `src/kernels/` by logical or physical operator and `src/ffi/` by the underlying provider.
- Organize Custom CUDA sources by physical operation, not by model, executor, inference stage, or precision.
- Prefer existing operators, reshapes, and vendor libraries. Do not add a kernel for every PyTorch expression.
- The first implementation should be correct, verifiable, and have a safe fallback. Perform fusion and tuning only after profiling.
- A CPU fallback may serve as a temporary correctness oracle while a target
  accelerator path is under development. After replay proves the semantics,
  continue through safe Rust API, FFI, adapter, CUDA implementation, workspace,
  dispatch, and target-hardware replay. Repeated model-layer D2H/H2D is not a
  completed accelerator implementation and must not be reclassified as generic
  performance debt merely because end-to-end values pass.
- CUDA compilation cost affects iteration strategy, not the required execution
  path. Batch related kernel changes, use focused compile checks where the build
  permits, and pay the full target build when validation requires it.
- When no matching tactic exists, an operator must use its explicitly defined safe default or return a clear unsupported error. A model-level precision policy may explicitly require calibration or tuning artifacts.
- Kernel launchers must not call `cudaStreamSynchronize`; doing so breaks asynchronous execution and CUDA Graph capture.
- Unsupported hardware, dtype, shape, or alignment must use a correct fallback or return a clear error. It must never silently produce an incorrect result.

## 1. Collect the Required Inputs

Before coding, collect and preserve:

- Target GPU model and compute capability.
- CUDA Toolkit and driver versions.
- PyTorch, CUDA, Transformers, and other reference environment versions.
- Model configuration, weight format, tokenizer, and preprocessor.
- Fixed inputs, random seeds, and complete model outputs.
- Important intermediate tensors, including dtype, shape, stride/layout, and values.
- Actual ranges of dynamic dimensions such as batch size, sequence length, image size, and head dimension.
- The initial operator gap list.

At minimum, save reference values at these points:

```text
model input
embedding/preprocessing output
per-layer normalization output
Q/K/V
attention logits/output
MLP output
final normalization
final model output
```

Save debug tensors as FP32 when practical. Never label raw BF16 bytes as NumPy FP16.

## 2. Confirm Target Hardware Support

### 2.1 Backend-level support

Start by inspecting:

```text
crates/apxinf-core/src/lib.rs
crates/apxinf-model/src/accelerator.rs
```

The main device types are currently CPU and CUDA.

| Target | Required work |
| --- | --- |
| Supported NVIDIA SM80/SM100-family GPU | Usually proceed directly to operator and model integration |
| New NVIDIA SM architecture | Extend architecture detection, NVCC configuration, device capability classification, and kernel cfgs |
| AMD GPU | Implement a HIP/ROCm backend first; this is not merely an operator task |
| CPU | Use or extend the CPU backend |

Do not put AMD-versus-NVIDIA decisions in a model runtime. Vendor-level dispatch belongs to the backend layer.

### 2.2 NVIDIA architecture-level support

Inspect and update as needed:

```text
crates/apxinf-cuda/build_support/cuda_arch.rs
crates/apxinf-cuda/build.rs
crates/apxinf-cuda/src/device_caps.rs
crates/apxinf-cuda/src/context.rs
```

Confirm that:

- `APXINF_CUDA_ARCH` and automatic detection recognize the target SM.
- NVCC receives the correct `-arch=sm_XX` value.
- The new architecture belongs to an existing `CudaArchFamily`, or a new family is genuinely required.
- CUTLASS, FA2, cuBLASLt, and Custom kernels support the target architecture.
- Compile-time cfgs agree with runtime capability checks.
- Unavailable implementations have a fallback or return a clear error.

Keep three dispatch levels separate:

```text
Compile-time dispatch
  Decides which adapters and kernels are included in the build.

Runtime hardware dispatch
  Uses CudaDeviceCaps to choose implementations available on the current GPU.

Runtime input dispatch
  Uses dtype, shape, layout, and alignment to choose a concrete variant.
```

The current build is not a universal fat binary containing independent SM80, SM100, and other implementations. Each adapter is normally compiled with a single `-arch`, and automatic detection requires all visible GPUs to have the same compute capability. Build and test separately for every target SM. Set `APXINF_CUDA_ARCH` explicitly when cross-compiling. CUTLASS may also use `APXINF_CUDA_ARCH_CUTLASS` for an appropriate architecture-feature target.

Adding genuine multi-architecture or fatbin support is a separate build-system task. A new runtime `CudaArchFamily` branch alone does not provide it.

## 3. Build an Operator Gap Table

Classify every relevant PyTorch graph node as one of the following:

1. Already supported by ApxInf and directly reusable.
2. Logical operator exists, but the required dtype is missing.
3. Logical operator exists, but the required shape or layout is missing.
4. Logical operator exists, but the target hardware implementation is missing.
5. Expressible as existing operators plus a reshape or view.
6. Requires a new model-neutral operator.
7. Can initially use a composition, but may deserve fusion after profiling.

Record at least these fields for each gap:

| Field | Example |
| --- | --- |
| Logical operator | RMSNorm |
| PyTorch semantics | `x * rsqrt(mean(x²) + eps) * weight` |
| Input/output shape | `[B, S, H]` |
| Kernel view | `[B*S, H]` |
| Input/output dtype | BF16 |
| Accumulation precision | FP32 |
| Layout | Contiguous row-major |
| Dynamic dimensions | Dynamic `B/S`, fixed `H` |
| Target hardware | SM87 |
| Call frequency | Twice per layer |
| Performance importance | High |
| ApxInf status | Missing BF16, missing shape, or entirely missing |
| Initial implementation | Custom CUDA |
| Fallback | Existing composition, ordinary cuBLAS, or none |

Compare full semantics rather than operator names alone:

- Dtype and accumulation precision.
- Shape, broadcasting, and layout.
- Causal versus non-causal masking.
- Head layout and GQA/MQA rules.
- RoPE pairing and position IDs.
- Epsilon, approximation formula, and activation variant.
- Weight transposition and physical storage layout.

## 4. Choose an Implementation Path

ApxInf CUDA operators primarily use four implementation paths:

| Operator characteristics | Implementation path | Examples |
| --- | --- | --- |
| Elementwise, normalization, RoPE, quantization, cache, or project-specific logic | Custom CUDA | RMSNorm, SiLU, RoPE, KV-cache write |
| Standard matrix multiplication | Ordinary cuBLAS | BF16 GEMM, strided batched GEMM |
| GEMM requiring epilogue, heuristic, workspace, plan, or autotuning | cuBLASLt | FP8 GEMM, GEMM + bias + GELU |
| Mature third-party CUDA template implementation exists | Vendored kernel | CUTLASS FP8 GEMM, CUTLASS FMHA, FA2 |

Use this preference order:

```text
Reuse an existing ApxInf operator
→ compose existing operators with reshape/view
→ use ordinary cuBLAS or cuBLASLt
→ use a mature vendored kernel with an acceptable license
→ write a Custom CUDA kernel
```

Record the selected backend, fallback, hardware constraints, and rationale in the gap table.

When the selected path is a temporary host scaffold, also record its **exit
criterion**: the safe device API that will replace it, the required workspace
lifetime, and the operator replay that removes the scaffold. A passing model
fixture starts that replacement work; it does not close the gap.

## 5. Define the Safe Rust Operator Contract First

Regardless of the implementation path, first expose a model-neutral API under:

```text
crates/apxinf-cuda/src/kernels/
```

For example:

```rust
pub fn rms_bf16_into(
    ctx: &CudaContext,
    input: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()>
```

This layer owns:

- Parameter, dtype, shape, buffer-size, and device-consistency checks.
- Checked integer conversion and size arithmetic.
- Output and workspace allocation.
- Compile-time capability, runtime hardware, and runtime shape dispatch.
- Persisted tactic lookup and the safe default path.
- Conversion of lower-level errors to `apxinf_core::Result`.

Outputs and temporary buffers that must support CUDA Graphs should use the active workspace policy, such as `workspace::output_buffer()`. Do not allocate dynamically during capture or replay.

Choose the Rust file by operator semantics:

| Operator category | Safe Rust kernel file |
| --- | --- |
| Activation | `src/kernels/activation.rs` |
| Attention | `src/kernels/attention.rs` |
| KV cache | `src/kernels/cache.rs` |
| Elementwise | `src/kernels/elementwise.rs` |
| Embedding | `src/kernels/embedding.rs` |
| Fused operator | `src/kernels/fused.rs` |
| GEMM | `src/kernels/gemm/*.rs` |
| Normalization | `src/kernels/norm.rs` |
| Preprocessing | `src/kernels/preprocess.rs` |
| Quantization | `src/kernels/quantization.rs` |
| RoPE | `src/kernels/rope.rs` |

Add a Rust module and export it from `src/kernels/mod.rs` only when introducing a new logical category.

## 6. Connect the Lower-level Implementation

### 6.1 Path A: Custom CUDA kernel

Use this path for normalization, activation, elementwise, RoPE, embedding, preprocessing, quantization, KV cache, selection, and project-specific fused computation.

Call path:

```text
src/kernels/<operator>.rs
→ src/ffi/custom.rs
→ adapters/custom_kernels.cu
→ kernels/custom/<physical_type>.cuh
→ GPU
```

#### 6.1.1 Select the device-side file

Organize Custom CUDA sources by physical operation:

| Operator | Device-side file |
| --- | --- |
| GELU, SiLU, GeGLU | `kernels/custom/activation.cuh` |
| Mask, softmax, custom attention | `kernels/custom/attention.cuh` |
| KV-cache write | `kernels/custom/cache.cuh` |
| Add, multiply, scale, bias, concatenate | `kernels/custom/elementwise.cuh` |
| Embedding lookup | `kernels/custom/embedding.cuh` |
| Residual + normalization, QKV split + RoPE | `kernels/custom/fused.cuh` |
| RMSNorm, LayerNorm, AdaNorm | `kernels/custom/normalization.cuh` |
| Image preprocessing, patchification | `kernels/custom/preprocess.cuh` |
| FP8/INT8 quantization and dequantization | `kernels/custom/quantization.cuh` |
| RoPE | `kernels/custom/rope.cuh` |
| Argmax, token selection | `kernels/custom/selection.cuh` |
| Shared mathematical helpers | `kernels/custom/math.cuh` |
| Warp/block reductions | `kernels/custom/reduction.cuh` |

Add a kernel to an existing `.cuh` whenever the physical category already exists. Create a new `.cuh` only for a genuinely new physical category.

Whenever adding a launcher to `custom_kernels.cu`, verify that the corresponding `.cuh` is included by that translation unit. Do not assume that a header is visible merely because a legacy adapter includes it. At present, `rope.cuh` and `selection.cuh` are included only by the legacy `core_kernels_adapter.cu`. A new ABI for those categories in `custom_kernels.cu` must add the include there or deliberately follow a controlled legacy-ABI extension strategy.

Do not create aggregate headers named after a model, executor, or precision. Express precision through function suffixes or template specialization, for example `rms_norm_bf16_kernel`.

#### 6.1.2 Add the host adapter

New stable C ABIs for Custom operators should normally be added to:

```text
crates/apxinf-cuda/adapters/custom_kernels.cu
```

The adapter owns:

- Raw-pointer and scalar-parameter validation.
- Grid, block, and dynamic shared-memory selection.
- Vectorized versus scalar dispatch based on shape and alignment.
- Kernel launch on the supplied CUDA stream.
- Returning `cudaGetLastError()`.

`core_kernels_adapter.cu`, `static_bf16_adapter.cu`, and `w8a8_adapter.cu` primarily preserve legacy C ABIs. Do not use them as the default location for new Custom operators.

#### 6.1.3 Add the Rust FFI declaration

Declare the symbol exported by the adapter in:

```text
crates/apxinf-cuda/src/ffi/custom.rs
```

The FFI layer describes only the raw C ABI. It does not validate tensors, allocate memory, or choose hardware policies.

#### 6.1.4 Handle the build

Changing an existing `.cuh` and `custom_kernels.cu` normally does not require a `build.rs` change. Add a file to `kernel_files` only when introducing a separate `.cu` translation unit. Do not create an independent adapter for every Custom operator.

### 6.2 Path B: ordinary cuBLAS

Use this path for standard GEMM, strided batched GEMM, and matrix multiplications that do not need a custom epilogue or explicit heuristic selection.

Call path:

```text
src/kernels/gemm/*.rs
→ src/cublas.rs
→ src/ffi/cublas.rs
→ libcublas.so
→ NVIDIA kernel
```

No project-owned GPU `.cu` kernel is required.

#### 6.2.1 Existing `CublasHandle` method is sufficient

If `CublasHandle::gemm()`, `batched_gemm()`, or another existing method already expresses the operation, modify only the relevant `src/kernels/gemm/*.rs` call site. Normally do not change `src/cublas.rs`, `src/ffi/cublas.rs`, adapters, or `build.rs`.

#### 6.2.2 cuBLAS supports it, but Rust does not yet wrap it

Add a high-level method to:

```text
crates/apxinf-cuda/src/cublas.rs
```

Centralize the following there:

- `cublasHandle_t` lifetime and stream binding.
- Rust dtype to CUDA/cuBLAS type conversion.
- Row-major to column-major semantic conversion.
- M/N, A/B, transpose, and leading-dimension handling.
- Compute type, alpha/beta, and status checking.

If the raw NVIDIA function is not declared yet, add the corresponding `extern "C"` declaration to `src/ffi/cublas.rs`.

Do not put an ordinary cuBLAS operation in `kernels/custom/` or `adapters/custom_kernels.cu`.

#### 6.2.3 When to use `cublas_adapter.cu`

Use the C++ adapter only when one logical call requires multiple cuBLAS operations, C++-managed workspace, a stable composite ABI, or an interface too complex for direct Rust FFI:

```text
src/kernels/<operator>.rs
→ src/ffi/cublas.rs
→ adapters/cublas_adapter.cu
→ cuBLAS
```

The project's cuBLAS MQA implementation is such a composite adapter. This is not the default path for ordinary GEMM.

### 6.3 Path C: cuBLASLt

Use this path for GEMM with bias, GELU, or residual epilogues; FP8 GEMM; and GEMM requiring descriptors, layouts, plans, workspace, heuristics, or autotuning.

Call path:

```text
src/kernels/gemm/*.rs or src/kernels/fused.rs
→ src/ffi/cublaslt.rs
→ adapters/cublaslt_adapter.cu
→ libcublasLt.so
→ NVIDIA kernel
```

cuBLASLt normally does not pass through `src/cublas.rs`, because the C++ adapter manages its complex objects and plans.

#### 6.3.1 Existing C ABI is sufficient

If an existing `apxinf_static_*gemm*` interface already supports the new call, add only the safe Rust call and dispatch. Do not duplicate an adapter or FFI declaration.

#### 6.3.2 Add an epilogue or GEMM variant

Modify:

```text
crates/apxinf-cuda/adapters/cublaslt_adapter.cu
crates/apxinf-cuda/src/ffi/cublaslt.rs
crates/apxinf-cuda/src/kernels/gemm/*.rs or src/kernels/fused.rs
```

`cublaslt_adapter.cu` owns:

- Creating and caching `cublasLtMatmulDesc_t` and matrix layouts.
- Setting transpose flags, compute type, bias, and epilogue.
- Setting workspace preferences.
- Querying and selecting heuristics.
- Caching plans and calling `cublasLtMatmul`.
- Exporting a stable, simple C ABI.

Provide a `prepare` API when native resources must be created before CUDA Graph capture. Provide heuristic setters and autotune APIs when tuning is required. The safe Rust kernel decides when to prepare resources, how to query persisted tactics, and which default path to use.

Changing the existing `cublaslt_adapter.cu` normally does not require a `build.rs` change. NVIDIA provides the actual GPU kernel; do not reimplement the same GEMM under `kernels/custom/`.

### 6.4 Path D: vendored third-party kernel

Use this path for CUTLASS GEMM/FMHA, FlashAttention/FA2, or another mature third-party CUDA kernel optimized for the target architecture.

Call path:

```text
src/kernels/<operator>.rs
→ src/ffi/cutlass.rs or src/ffi/fa2.rs
→ adapters/<vendor>_adapter.cu
→ kernels/cutlass/<operator>.cu or vendored source
→ GPU
```

#### 6.4.1 Store the third-party implementation

Place CUTLASS, FMHA, and FA2 sources under:

```text
crates/apxinf-cuda/kernels/cutlass/
```

Filenames should identify the physical operation, dtype or quantization scheme, and target architecture, for example:

```text
fp8_gemm_sm100.cu
w8a8_gemm_sm80.cu
fmha_sm100.cu
grouped_gemm_bf16_sm100.cu
```

Preserve the upstream repository, commit/version, license, and local modification notes. Update `kernels/cutlass/README.md`, `VENDOR.md`, and `licenses/` as needed.

#### 6.4.2 Add a stable C ABI adapter

Extend an existing `cutlass_*_adapter.cu` or `fa2_adapter.cu` under `crates/apxinf-cuda/adapters/`, or add an adapter for a genuinely new provider or operator family.

The adapter hides C++ template types, converts raw arguments, passes workspace and tactics, invokes the vendored implementation, and returns a stable error code. The vendored operator implementation itself should not export the Rust-facing C ABI.

#### 6.4.3 Add Rust FFI and the safe kernel

- Put CUTLASS declarations in `src/ffi/cutlass.rs`.
- Put FA2 declarations in `src/ffi/fa2.rs`.
- Keep GEMM safe APIs in `src/kernels/gemm/*.rs`.
- Keep FMHA and FA2 safe APIs in `src/kernels/attention.rs`.
- Place other operators according to their logical category.

The model must not call raw CUTLASS or FA2 FFI. It calls a logical operator, and the safe kernel selects the third-party implementation or fallback.

#### 6.4.4 Update the build and cfgs

Vendored kernels normally require changes to `crates/apxinf-cuda/build.rs`:

1. Validate that sources, headers, include directories, and adapters exist.
2. Follow the existing translation-unit pattern and avoid compiling the same operator twice.
3. Add third-party include paths.
4. Compile only for supported target SMs.
5. Emit the corresponding `cargo:rustc-cfg`.
6. Add `rerun-if-changed` entries.
7. Ensure cfg-disabled builds use a fallback or return a clear unsupported error.

The current CUTLASS GEMM/FMHA/W8A8 pattern has the adapter directly `#include` the operator `.cu`; only the adapter is added to `kernel_files`. Do not separately compile the included operator. FA2 instead compiles `fa2_adapter.cu` plus separate head-dimension instantiation `.cu` files. Compare against the existing implementation of the same category before adding a source.

For every new cfg:

- Add a matching `cargo:rustc-check-cfg=cfg(...)` declaration to `build.rs`.
- Emit `cargo:rustc-cfg=...` only when the relevant object is actually compiled.
- Use the same cfg to guard both the raw FFI declaration and the safe-kernel call.

## 7. Integrate CUDA Graph and Workspace Lifecycles

If the model uses CUDA Graphs, validate eager execution, preparation, capture, and replay. A successful standalone kernel launch is not sufficient.

The basic ApxInf lifecycle is:

```text
create GraphWorkspace
→ run eager preflight with prepare_with_workspace()
→ create native plans/resources and validate workspace capacity
→ begin_capture()
→ capture through with_workspace()
→ end_capture()/instantiate
→ replay repeatedly
```

Requirements:

- Allocate outputs and temporary buffers through `workspace::output_buffer()` or the relevant workspace helper. Without an active workspace, the helper may use eager allocation.
- Include every temporary buffer introduced by the operator in the model's `GraphWorkspace` capacity calculation.
- Create cuBLASLt plans and other native resources only when `workspace::may_prepare_native_resources()` is true, normally during preflight preparation.
- Use stable addresses during capture and replay. Do not allocate dynamically, read back to the host, or synchronize the stream.
- If capture fails, correctly end or invalidate it so that the stream remains usable.

Validate at least:

1. Eager execution works without an active workspace.
2. Preparation works and detects shape, artifact, and capacity errors before capture.
3. Capture does not create native plans or dynamic resources.
4. Insufficient workspace returns a clear error.
5. The graph can replay repeatedly.
6. Updating the input changes replay output as expected.
7. The stream remains usable after a capture failure.

## 8. Add Hardware, Shape, and Backend Dispatch

Dispatch responsibilities belong to different layers:

| Dispatch type | Owner |
| --- | --- |
| CPU/CUDA/HIP backend | Accelerator/registry |
| Model precision and calibration/tuning artifact policy | Model loader/runtime |
| CUTLASS/cuBLASLt/cuBLAS/Custom implementation selection | Safe kernel |
| Grid/block/vectorized kernel variant | Host adapter |

The model layer may use device capabilities and artifact availability to choose an FP8, W8A8, BF16, or other model-level precision policy. It must not choose a specific CUTLASS tactic, cuBLASLt heuristic, or grid/block shape.

Example safe-kernel implementation dispatch:

```text
logical GEMM
├── persisted CUTLASS tactic
├── persisted cuBLASLt heuristic
└── ordinary cuBLAS safe default
```

Use this decision order:

1. Was the implementation included at compile time?
2. Does `CudaDeviceCaps` support it on the current device?
3. Do dtype, shape, layout, and alignment satisfy its constraints?
4. Does the tuning database contain a match allowed by this backend?
5. Execute the selected implementation; otherwise use the safe default.

Use compute capability and `CudaDeviceCaps` for primary hardware decisions. Do not rely primarily on GPU name strings.

## 9. Decide Whether to Add Autotuning

Autotuning is appropriate when:

- Multiple CUTLASS tiles or tactics exist.
- Multiple cuBLASLt heuristics exist.
- One logical operator has multiple backends.
- Performance depends strongly on M/N/K.
- Different hardware has different optimal paths.

Autotuning is usually unnecessary when:

- A simple elementwise or normalization operator has one path.
- The shape is fixed and the launch configuration is clearly optimal.
- The first implementation only needs to establish correctness.

The tuning key should include at least the device fingerprint/SM, dtype, important shapes, layout, and operator variant. Follow the matching semantics already defined by each backend:

- The cuBLASLt backend and heuristic require an exact physical key.
- CUTLASS may check exact first, then use a compatible bucket defined by the existing `GemmBucketKey`.
- Do not invent an undefined fuzzy-shape match.
- If no legal tactic exists, use the operator's explicit default policy or return a clear error.

When adding persisted GEMM tactics, inspect and update as needed:

```text
crates/apxinf-cuda/src/tuning/key.rs
crates/apxinf-cuda/src/tuning/db.rs
crates/apxinf-cuda/src/tuning/store.rs
crates/apxinf-cuda/src/tuning/mod.rs
crates/apxinf-cuda/src/kernels/gemm/fp8.rs
the corresponding tuning generator/tool
```

Verify `GemmOp`, epilogue, layout, legal backend/tactic ranges, lookup semantics, and the database header. The database must also match the device name, SM, kernel build ID, and CUDA/cuBLAS versions. The current tuning store is installed globally; do not assume that a process can freely install multiple different stores.

Tuning workflow:

```text
enumerate legal tactics
→ warm up
→ time with CUDA events
→ have the tuning tool or an independent test validate every tactic
→ select the fastest correct tactic
→ persist backend + tactic ID
→ look up according to backend rules
  cuBLASLt: exact
  CUTLASS: exact → compatible bucket → default
```

The current low-level autotune APIs mainly measure time and reject tactics that fail to launch. They do not prove numerical correctness. The tuning tool or an independent test must explicitly compare every candidate against a reference.

## 10. Integrate the Model Runtime and Executor

Modify the model configuration, weights, runtime/executor, and registration only after the lower-level operators pass their own correctness tests.

The model layer owns:

- Model structure and inference flow.
- Weight-name mapping, transposition, quantization, and upload.
- Tensor-shape flow.
- KV-cache and workspace lifetimes.
- Calls into `apxinf-cuda/src/kernels/`.

The model layer does not own:

- Raw FFI.
- Concrete kernel cfg, backend tactic, or launch-parameter decisions.
- CUDA grid/block configuration.
- cuBLAS handle lifetime.
- cuBLASLt descriptors and plans.
- Low-level CUTLASS template and tactic execution.

Use the existing registry and backend-suffix mechanism for model registration. Do not reimplement CPU/CUDA dispatch inside a model.

Quantized models also need an artifact workflow:

```text
generate and validate calibration data
→ convert and validate quantized weights, scale mode, and physical layout
→ generate tactics on the target hardware and current kernel build
→ verify artifacts against weights, device fingerprint, SM, build ID, and library versions
→ install the tuning store
→ create the runtime for the selected precision
```

An explicitly selected precision mode may require calibration or tuning artifacts and refuse to load when they are absent. An `Auto` mode should choose another supported precision according to the model's policy or return a clear explanation. Do not confuse an operator's default tactic with optional model-level quantization artifacts.

## 11. Validate in Layers

### 11.1 Individual operator

Compare every new operator against the PyTorch reference. Cover:

- Minimum, representative, and maximum shapes.
- Irregular shapes.
- Aligned and unaligned cases.
- Dynamic-dimension boundaries.
- Zeros, random values, and extremes.
- Every supported dtype.
- Every hardware/backend branch.
- Every fallback branch.

Record maximum absolute error, maximum relative error, mean error, and NaN/Inf occurrences. Use appropriate tolerances for BF16, FP16, FP8, and INT8; do not require bitwise equality for all of them.

### 11.2 Complete layer

Validate a complete model layer:

```text
Norm → QKV → RoPE → Attention → Projection → Residual → MLP
```

Compare intermediate results and identify the earliest point where error begins.

### 11.3 Complete model

Cover complete outputs, multi-step inference, KV-cache updates, dynamic batch/sequence lengths, eager and CUDA Graph paths, and the target hardware.

### 11.4 Performance

Record individual kernel latency, end-to-end latency, throughput, memory and workspace use, temporary allocations, host/device synchronization, graph compatibility, and comparisons with PyTorch and fallback implementations.

Add fusion or a more specialized kernel only after profiling identifies an actual hotspot.

### 11.5 Build and link matrix

- Perform a clean build for each target SM and run it on the corresponding physical GPU.
- Verify that every new C ABI links successfully; Rust type-checking alone is insufficient.
- Test both cfg-enabled and cfg-disabled builds to cover optimized and fallback/unsupported paths.
- Compare every tuning candidate against a reference. A successful kernel return code does not imply numerical correctness.

## 12. File-selection Reference

| Addition | Safe Rust API | Raw FFI | Host adapter | Actual implementation | `build.rs` |
| --- | --- | --- | --- | --- | --- |
| Custom RMSNorm | `src/kernels/norm.rs` | `src/ffi/custom.rs` | `custom_kernels.cu` | `custom/normalization.cuh` | Usually unchanged |
| Custom SiLU | `src/kernels/activation.rs` | `src/ffi/custom.rs` | `custom_kernels.cu` | `custom/activation.cuh` | Usually unchanged |
| Ordinary BF16 GEMM | `src/kernels/gemm/bf16.rs` | Reuse or add to `src/ffi/cublas.rs` | None | `libcublas.so` | Unchanged |
| cuBLAS batched GEMM | `src/kernels/gemm/*.rs` | `src/ffi/cublas.rs` | Usually none | `libcublas.so` | Unchanged |
| cuBLASLt GEMM + epilogue | `src/kernels/gemm/*.rs` or `fused.rs` | `src/ffi/cublaslt.rs` | `cublaslt_adapter.cu` | `libcublasLt.so` | Usually unchanged |
| CUTLASS GEMM | `src/kernels/gemm/*.rs` | `src/ffi/cutlass.rs` | `cutlass_*_adapter.cu` | `kernels/cutlass/*.cu` | Required |
| CUTLASS FMHA | `src/kernels/attention.rs` | `src/ffi/cutlass.rs` | `cutlass_fmha_adapter.cu` | `kernels/cutlass/fmha*.cu` | Required |
| FA2 | `src/kernels/attention.rs` | `src/ffi/fa2.rs` | `fa2_adapter.cu` | `kernels/cutlass/fa2/` | Required |

## 13. Required Agent Deliverables

At completion, submit or report:

1. **Hardware support conclusion:** target GPU/SM, architecture family, compile-time cfgs, unavailable implementations, and fallbacks.
2. **Operator mapping table:** PyTorch operator, ApxInf safe API, lower-level path, shape/dtype/hardware constraints, and status.
3. **Changed-file list:** grouped by model layer, safe kernel, FFI, adapter, GPU/vendored source, build, tuning, and tests.
4. **New operator call paths:** from model runtime to the actual GPU implementation.
5. **Correctness results:** operator, layer, and model errors, plus covered shapes, dtypes, and hardware.
6. **Performance results:** hotspot latency, end-to-end results, fallback comparison, and tuning results.
7. **Remaining risks:** uncovered hardware, shapes, graph capture, precision, or licensing concerns.

Recommended operator mapping format:

| PyTorch operator | ApxInf safe API | Lower-level implementation | Hardware restriction | Fallback | Status |
| --- | --- | --- | --- | --- | --- |
| `F.linear` | `gemm::bf16()` | Ordinary cuBLAS | CUDA | Not needed | Reused |
| `rms_norm` | `norm::rms_*()` | Custom CUDA | CUDA | None | Added |
| `scaled_dot_product_attention` | `attention::*()` | FA2 | Specific SM/head dimension | Custom/error | Pending validation |

## 14. Recommended Execution Order and Acceptance Gates

```text
confirm target hardware and build architecture
→ freeze the PyTorch reference
→ build the complete operator gap table
→ reuse existing operators and composition paths
→ select one of the four lower-level implementations for each remaining gap
→ define the safe Rust kernel contract
→ connect FFI, adapter, and actual implementation
→ add compile-time, hardware, shape, and backend dispatch
→ pass individual-operator correctness tests
→ integrate workspace and validate prepare/capture/replay
→ integrate the model runtime/executor
→ pass layer and complete-model correctness tests
→ profile
→ optimize only measured hotspots
→ add autotuning where justified
→ complete target-hardware regression tests
```

Final acceptance criteria:

- The model runtime depends only on safe Rust operator APIs.
- Every new operator has a clear call path, hardware constraints, and fallback.
- When no matching tactic exists, the operator uses a defined safe strategy or returns a clear error. Required model calibration/tuning artifacts have an explicit and verifiable loading policy.
- Unsupported hardware, dtype, shape, and alignment never enter an invalid implementation silently.
- Individual operator and complete model outputs agree with the PyTorch reference within the appropriate precision tolerance.
- CUDA Graph paths perform no illegal resource creation, synchronization, or dynamic allocation.
- Performance optimizations are supported by profiling data collected on the target hardware.
