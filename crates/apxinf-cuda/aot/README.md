# Native CuTe operators

This directory owns offline export and native integration for the SM110 BF16
operators. The engine links AArch64 object files and the static CuTe runtime.
It does not import Python, Torch, FlashAttention, FlashInfer, or Quack at runtime.

## Files

- `manifest.json`: pinned upstream versions, exported symbols and tensor contracts.
- `build.rs`: Rust export driver. Runs the exporters and packages checksummed artifacts.
- `bundle.rs`: artifact integrity, target, toolkit and header checks.
- `link.rs`: Cargo integration and compatibility checks against the maintained recipes.
- `exporters/`: Python entry points required to instantiate and compile CuTe DSL.

The maintained C++ adapters are in `../adapters/`; safe Rust dispatch belongs in
`../src/kernels/`. Model code calls these operators through the kernel facade.

## Produce a bundle

Export on a Linux AArch64 SM110 host with CUDA 13.2 and CuTe DSL 4.7.0.
Use the versions in `manifest.json`; the exporters also check the kernel source
hashes. Upstream trees and Python dependencies stay outside this repository.
Supply a JSON file containing these paths:

```json
{
  "python": "/path/to/export-venv/bin/python",
  "gdn_source": "/path/to/pinned/flashinfer/gdn_kernels",
  "fa4_source": "/path/to/pinned/flash-attention",
  "quack_dependencies": "/path/to/quack/site-packages",
  "cutlass_dependencies": "/path/to/cutlass/site-packages",
  "extra_dependencies": "/path/to/fa4/dependencies",
  "runtime_archive": "/path/to/libcute_runtime.a"
}
```

Input paths are relative to that JSON file unless absolute. The output directory
must be new, preserving evidence from earlier builds.

```sh
cargo run -p apxinf-cuda --example build-aot -- \
  --inputs /path/to/source-paths.json \
  --out /path/to/new-bundle
```

The bundle contains `manifest.json`, one directory per operator with its `.h`
and `.o`, and `lib/libcute_runtime.a`. Logs and exporter metadata accompany the
objects. Upstream sources and their accompanying notices remain external build
dependencies; keep those notices with any redistributed operator artifacts.
Generated artifacts belong in an external artifact store or ignored
`devlocal/`, never in the source diff.

## Consume a bundle

Pass the manifest of the actual exported artifact directory as a build input.
There is no reserved bundle directory in the source tree:

```sh
APXINF_CUDA_AOT_MANIFEST=/path/to/bundle/manifest.json \
  cargo build --release -p apxinf-py --features cuda
```

This is a build input, not a runtime optimization switch. Cargo checks the
checksums, AArch64 ELF type, CUDA/DSL versions, symbols and tensor contracts
and compiler specialization before linking. It also compares the bundle exporter
SHA256 with the reviewed recipe and the committed exporter file. Editing an
exporter and merely rehashing the bundle does not satisfy this check. A changed artifact invalidates the kernel build identity.
The adapters own module initialization and stream-aware launches. Preparation
happens before CUDA graph capture.

The current bundle covers GDN prefill, language attention, split action
attention, SwiGLU GEMM, and both fixed vision attention groups. Unsupported
hardware and tensor shapes use the existing generic kernels. Direct planning
pads only logical lengths 3383–3387 to 3387 tokens; this window uses masked
split attention. Shorter prompts keep their original length and use the generic
operators. Longer prompts are not truncated. Reasoning retains variable length
execution.

## What an exporter contains

For example, GDN's Python entry instantiates the pinned kernel, creates typed
example tensors with the required layouts, and asks CuTe for a native export:

```python
# Simplified: gdn and args are constructed from the fixed recipe above.
compiled = cute.compile(gdn, *args, options="--gpu-arch sm_110a --opt-level 2")
compiled.export_to_c(
    file_path=str(out), file_name=symbol, function_prefix=symbol,
)
```

The generated header declares a module handle, tensor descriptors and a C launch
wrapper. The `.o` contains the native wrapper and embedded device code. The
maintained C++ adapter fills those descriptors from device pointers and passes
the caller's CUDA stream. Rust checks tensor types, sizes and device ownership
before calling that adapter. Changes to a kernel specialization therefore touch
the recipe/exporter and its adapter; they do not add Python to model execution.

### Action attention specialization

The action exporter compiles a 1718-row example. Its native tensor descriptors
accept a 1720-row runtime view with a 1718-row batch stride, allowing the five
logical prompt lengths above. Keep these two shapes distinct: compiling a
1720-row example changes BF16 rounding on real model inputs even when random
input comparisons pass. The bundle records the compiler specialization separately
from the runtime tensor contract and Cargo checks both before linking.

## Verification and build scope

Run `cargo test -p apxinf-cuda --test aot_bundle` for bundle corruption,
exporter/specification mismatch and shared geometry checks. This is a normal
integration-test target; testing the build script does not run these checks.
`sm_110` and `sm_110a` both accept the same pinned SM110 bundle.

AOT remains an optional build input because this CUDA crate also supports other
models and generic operators. Absence on SM110 produces a build warning; it does
not silently claim the qualified Qwen-Drive performance. The benchmark build
must supply the manifest explicitly. Source content hashes in the exporters
pin upstream files even when the upstream revision is a package version.

Reuse one Cargo target directory and preserve unchanged source/artifact paths
for incremental builds. NVCC objects are reused only when their command and
all recorded dependencies are unchanged. Generated include directories currently
enter all legacy NVCC commands, so relocating a bundle can trigger broad
recompilation; reuse the validated artifacts during ordinary Rust changes.

For GEMM heuristic comparisons, use the supported tuner through the safe GEMM
API and its vendor-versus-winner report. Calling the raw cuBLASLt plan API from
an isolated test omits required context setup and does not yield a valid sweep.
