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

## LIBERO closed loop

Closed-loop simulation of `pi05_libero_base` on the same machine, in-process (no server): the
policy is built by `AutoPolicy` inside `scripts/eval_libero.py`, which drives LIBERO/MuJoCo. Only this
repository is needed; there is no dependency on APXinf-robo.

### Extra prerequisites

- Everything in [Prerequisites](#prerequisites), plus network access to GitHub and Hugging Face
  (LIBERO is cloned; the checkpoint is downloaded).
- An interactive Windows desktop session. MuJoCo renders through `glfw` (Windows has no EGL), which
  needs an OpenGL context; it does not work from a Windows service or a session without a display.
- About 5 GB free disk (LIBERO with assets 0.4 GB, torch and the checkpoint the rest) and 24 GB of GPU memory
  is comfortable for bf16.

### 1. Install the simulator

```powershell
# builds apxinf_py, then installs the pinned LIBERO dependencies and clones LIBERO at a fixed commit
powershell -NoProfile -File scripts\build-win.ps1 -Wheel -Libero [-Mirror]
```

`-Libero` installs [scripts/requirements-libero-win.txt](../scripts/requirements-libero-win.txt)
(Python 3.10: robosuite 1.4.1, mujoco 2.3.7, numpy 1.26.4, torch 2.14.1, ...), clones
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) at `-LiberoRev` into `-LiberoDir`
(default `C:\apx-libero`), puts it on `sys.path` through a `.pth` file (its own `requirements.txt` pins `numpy==1.22.4`, so it is not pip-installed),
generates a non-interactive LIBERO config under `<LiberoDir>\_cfg`, and checks that the `libero_10`
suite imports. `-Mirror` only affects crates, conda and pip; the LIBERO clone always uses GitHub.

### 2. Get the checkpoint

```powershell
pip install -U "huggingface_hub[cli]"
huggingface-cli download lerobot/pi05_libero_base --local-dir D:\models\pi05_libero_base
curl.exe -fL https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json -o D:\models\pi05_libero_base\norm_stats.json
```

The `norm_stats.json` step is required: the Hugging Face repository lost its statistics, and without them
actions are not de-normalized and every episode fails.

### 3. Run

```powershell
# smoke check: libero_10, task 0, 10 trials, bf16
powershell -NoProfile -File scripts\run-libero-win.ps1 -ModelDir D:\models\pi05_libero_base
```

Options: `-Suite`, `-Tasks` (comma list or `all`), `-Trials`, `-Precision`, `-Seed` (default 7),
`-OutDir` (default `devlocal\libero-win\<suite>-<timestamp>`), `-LiberoDir`, `-EnvName`, `-CudaPath`,
`-MujocoGl`. The script wraps `scripts/eval_libero.py --backend in-process`
(see the README "LIBERO evaluation" section for every flag); the published protocol is
`-Tasks all -Trials 50`.

Output in `-OutDir`: `results.jsonl` (one row per trial, fsync'd, so a re-run with the same `-OutDir`
resumes), `summary.json` (success rate and per-segment latency) and `run.log`. The script exits
non-zero and prints `INCOMPLETE` if fewer episodes finished than requested.

`-Libero` also applies three Windows fixes inside the conda env, so nothing has to be done by hand:

| Fix | Why |
|---|---|
| copies `mujoco\mujoco.dll` to `robosuite\utils\mujoco.dll` | robosuite 1.4.1 loads that path with `ctypes` on Windows |
| writes `robosuite\macros_private.py` with `MUJOCO_GPU_RENDERING = False` | robosuite otherwise forces `MUJOCO_GL=egl`, which is invalid on Windows |
| writes `apxinf_libero_src.pth` pointing at the LIBERO checkout | LIBERO's own `pip install -e` is not importable |

A `[Warning]: datasets path ... does not exist!` line at start-up is harmless: rollouts use the
bundled init states, not the LIBERO demonstration datasets.

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
| `git clone` of LIBERO fails with `RPC failed` / `early EOF` | flaky GitHub connection; the script retries 4 times, re-run it if all fail |
| `ImportError: native LIBERO observations require ...` | a LIBERO dependency is missing; re-run `build-win.ps1 -Wheel -Libero`, its self-check imports `OffScreenRenderEnv` |
| every episode fails, success 0% | `norm_stats.json` missing from the model dir (actions are not de-normalized) |

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

LIBERO closed loop, run on 2026-11-16 on an RTX 5090 Laptop GPU (sm_120, driver 610.91) in a fresh
conda env (`build-win.ps1 -Wheel -Libero -Mirror`, LIBERO `8f1084e`), checkpoint `lerobot/pi05_libero_base`
with OpenPI `norm_stats.json`, `run-libero-win.ps1` defaults (libero_10, task 0, 10 trials, bf16, seed 7):

| Metric | Result |
|---|---|
| Episodes finished | 10 / 10 |
| Successes | 10 / 10 (task 0: "put both the alphabet soup and the tomato sauce in the basket") |
| Mean inference per replan | 78.6 ms (model 77.0 ms, preprocess 0.5 ms), 576 calls |
| Episode length / wall time | 250-342 action steps, 12.8-19.2 s each |

This is a smoke check of the closed loop on one task, not an accuracy benchmark; 10 trials of a single
task say nothing about the other nine tasks. The published protocol is `-Tasks all -Trials 50`.

Not covered: fp8/int8, GPUs other than sm_120, the Linux
build after the `build.rs` change, and `-Debug` / `--features cuda` (NVTX) binaries.
