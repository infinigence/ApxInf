# Building ApxInf on Windows with CUDA

Native Windows (MSVC + CUDA Toolkit) build of the CUDA runtime, the `apxinf` binary and the
`apxinf_py` Python package. Developed and checked on an RTX 5090 Laptop (sm_120, CUDA 13.0,
VS 2022 Build Tools, Rust 1.99, Python 3.10).

## Status

| Item | State |
|---|---|
| `apxinf.exe` (`--features cuda-no-nvtx`) | see "Verification" below |
| `apxinf_py` wheel (`--features cuda`, release) | see "Verification" below |
| GDN (FlashInfer Cake) prefill on sm_120 | **not supported**: stub returns an unsupported status |
| fp8 / int8, GPUs other than sm_120 | not tested |

## Prerequisites

1. NVIDIA driver with CUDA 13 support (`nvidia-smi` lists the GPU).
2. CUDA Toolkit 13.x. On Windows the libraries are in `lib\x64` and the DLLs in `bin\x64`.
3. Visual Studio 2022 Build Tools: "Desktop development with C++" (MSVC v143, Windows SDK,
   C++ CMake tools).
4. Rust (`stable-x86_64-pc-windows-msvc`).
5. **Git on PATH.** `crates/apxinf-cuda-new/build.rs` applies the FlashAttention2 patch with
   `git apply` on Windows (`patch` on Linux).
6. Miniconda, only for the Python package.

## Build

From a normal PowerShell:

```powershell
# apxinf.exe only
powershell -NoProfile -File scripts\build-win.ps1

# plus the Python package (conda env, wheel, editable python/apxinf)
powershell -NoProfile -File scripts\build-win.ps1 -Wheel

# behind a network that cannot reach crates.io / PyPI / Anaconda: rsproxy + Tsinghua mirrors
powershell -NoProfile -File scripts\build-win.ps1 -Wheel -Mirror
```

Options: `-TargetDir` (default `C:\apx-target`, keep it short), `-CudaPath`, `-CudaArch`
(default from `nvidia-smi`), `-EnvName`, `-Conda`, `-Jobs` (nvcc/cargo parallelism, default 2),
`-DebugProfile`, `-Mirror`.

## What the Windows port changes

Source (all gated so Linux behavior is unchanged, except the GDN row):

| File | Change |
|---|---|
| `crates/apxinf-cuda-new/build.rs` | `lib\x64` search dir; `nvcc.exe`; `.obj` objects and `lib.exe` instead of `ar`; no `-fPIC`, `-MD -MF`, `stdc++`, `dl`; `\` normalized before the `/gemm/` test; FA2 patch through `git apply --unsafe-paths -p0` |
| `build.rs` (GDN) | the tcgen05 launch kernel is compiled only for sm_100/101/110 targets; other targets, **including sm_120 on any OS**, link `flashinfer_gdn_stub.cpp` |
| `native/framework/tuning_db.cpp` | `getpid` via `<process.h>` on `_WIN32` |
| `native/kernels/fa2/flash_attn/softmax.h` | define `M_LOG2E` on `_WIN32` |
| `native/kernels/flashinfer_gdn/gdn_prefill_generated.cuh` | the 64-bit `uint64_t` assertion also accepts `_WIN32` (LLP64) |
| `python/apxinf/apxinf/_tactics.py` | load `cudart64_13.dll` / `cudart64_12.dll` on Windows instead of `libcudart.so` |

Build environment (done by `scripts/build-win.ps1`, not by the repo):

| Setting | Why |
|---|---|
| short `CARGO_TARGET_DIR` | CMake object paths for `sentencepiece-sys` exceed 260 characters (`C1083`) |
| `CMAKE_GENERATOR=NMake Makefiles` | the Visual Studio generator cannot find the C++ compiler from `cmake-rs` |
| `CXXFLAGS=/MT` and a patch of the cached `sentencepiece-sys` `CMakeLists.txt` (`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`) | nvcc objects and `esaxx-rs` use the static CRT; a mix with `/MD` fails with `LNK2038`/`LNK1169` |
| `apxinf_cuda_dll.pth` in the env's `site-packages` | Python >= 3.8 ignores `PATH` for DLLs; without it `import apxinf_py` fails with `DLL load failed` |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `apply FA2 direct-E4M3 patch: program not found` | git is not on PATH |
| `ptxas ... 'tcgen05.alloc' not supported on .target 'sm_120a'` | GDN launch kernel compiled for a non-tcgen05 target; fixed by the `build.rs` gate |
| unresolved `cudaFree`, `cublasDestroy_v2`, ... at link | CUDA `lib\x64` not searched (old `build.rs`) |
| `fatal error C1083` in sentencepiece | target dir path too long |
| `No CMAKE_CXX_COMPILER could be found` | `CMAKE_GENERATOR` not set to `NMake Makefiles` |
| `LNK2038 MT_StaticRelease / MD_DynamicRelease` | sentencepiece CRT patch missing |
| `DLL load failed while importing apxinf_py` | CUDA DLL directory not registered (`.pth`) |
| cargo `schannel: server closed abruptly` | proxy breaks the TLS to `static.crates.io`; use `-Mirror` (rsproxy hosts the downloads too) |

## Verification

Run on 2026-11-16 from `origin/main` `caca33f` plus this change, with
`scripts\build-win.ps1 -Wheel -Mirror`, a fresh target dir (`C:\apx-target`), a fresh conda env,
and a real `git` (so the FA2 `git apply` step ran):

| Check | Result |
|---|---|
| `cargo build --release --features cuda-no-nvtx` | passes; `apxinf.exe --help` runs |
| `apxinf.exe test` | passes (CPU matmul only, it does not exercise the GPU) |
| `maturin build --release --features cuda` | passes, wheel `apxinf_py-0.1.0-cp310-cp310-win_amd64.whl` |
| `import apxinf_py` in the new env with no extra setup | passes (the `.pth` registers the CUDA DLL dirs) |
| `python scripts/bench_pi05.py --random-weights --model-variant bf16 --layer l1 --samples 5` | passes: `L1_rust` p50 96.4 ms (synthetic, tiny model) |
| `scripts/check_cuda_new_architecture.py` | passes (needs `PYTHONUTF8=1` on Windows; it reads files with the default GBK codec otherwise) |

Not covered: checkpoint inference with real weights, fp8/int8, GPUs other than sm_120, the Linux
build after the `build.rs` change, and `-Debug` / `--features cuda` (NVTX) binaries.
