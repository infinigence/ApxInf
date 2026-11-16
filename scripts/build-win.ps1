<#
.SYNOPSIS
  Build ApxInf on native Windows with CUDA (NVIDIA GPU, tested on RTX 5090 / sm_120).

.DESCRIPTION
  Default      : cargo build of apxinf.exe (--features cuda-no-nvtx).
  -Wheel       : additionally create a conda env, build the apxinf_py CUDA wheel, install it
                 together with the python/apxinf frontend, and import-check it.
  Details and troubleshooting: doc/windows-cuda.md

  Run from a normal PowerShell (not a VS developer prompt):
    powershell -NoProfile -File scripts\build-win.ps1 -Wheel -Mirror *> build-win.log

  The file is ASCII on purpose: Windows PowerShell 5.1 misreads BOM-less UTF-8.
#>
param(
  [string]$TargetDir = 'C:\apx-target',   # KEEP SHORT: CMake object paths exceed the 260-char limit otherwise
  [string]$CudaPath  = $(if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0' }),
  [string]$CudaArch  = '',                # default: derived from nvidia-smi (12.0 -> sm_120)
  [switch]$Wheel,
  [string]$EnvName   = 'apxinf-win',
  [string]$Conda     = '',
  [int]$Jobs         = 2,                 # nvcc / cargo parallelism
  [switch]$DebugProfile,                       # debug profile for apxinf.exe (the wheel is always release)
  [switch]$Mirror                         # use rsproxy (crates) and Tsinghua (conda, pip) mirrors
)
# Native tools (cargo, conda) write progress to stderr; PS 5.1 would treat that as fatal under 'Stop'.
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['*:ErrorAction'] = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Repo = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) '..')).Path

function Step($m) { Write-Host "`n=== $m" -ForegroundColor Cyan }
function Need($ok, $msg) { if (-not $ok) { throw "PREREQUISITE MISSING: $msg" } }
function Run($exe, [string[]]$a) { & $exe @a; if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $exe $($a -join ' ')" } }

# ------------------------------------------------------------ prerequisites
Step 'prerequisites'
Need (Test-Path "$CudaPath\bin\nvcc.exe") "nvcc.exe not found under $CudaPath (install CUDA Toolkit 13.x or pass -CudaPath)"
Need (Test-Path "$CudaPath\lib\x64") "$CudaPath\lib\x64 not found (not a Windows CUDA Toolkit?)"
$vc = Get-ChildItem "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022", "$env:ProgramFiles\Microsoft Visual Studio\2022" -Filter vcvars64.bat -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
Need ($null -ne $vc) 'Visual Studio 2022 (Build Tools) with "Desktop development with C++": MSVC v143, Windows SDK, C++ CMake tools'
Need ($null -ne (Get-Command cargo -ErrorAction SilentlyContinue)) 'Rust (rustup, stable-x86_64-pc-windows-msvc)'
Need ($null -ne (Get-Command git -ErrorAction SilentlyContinue)) 'git on PATH (build.rs runs "git apply" to patch FlashAttention2)'
Need (Test-Path "$Repo\Cargo.toml") "not an ApxInf checkout: $Repo"
if ($TargetDir.Length -gt 14) { Write-Warning "TargetDir '$TargetDir' is long; sentencepiece/CMake may hit the 260-char path limit. Use e.g. C:\apx-target." }
if (-not $CudaArch) {
  $cap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim()
  Need ($cap -match '^\d+\.\d+$') 'nvidia-smi did not report a compute capability (NVIDIA driver installed?)'
  $CudaArch = 'sm_' + ($cap -replace '\.', '')
}
$CutlassArch = if ($CudaArch -in 'sm_120', 'sm_121') { $CudaArch + 'a' } else { $CudaArch }
Write-Host "repo: $Repo`nGPU arch: $CudaArch (cutlass: $CutlassArch)   CUDA: $CudaPath   target: $TargetDir"

# ------------------------------------------------------------ build environment
$vars = cmd.exe /d /c "`"$($vc.FullName)`" >nul 2>nul && set"   # import the MSVC environment (cl, link, lib, ...)
foreach ($l in $vars) { if ($l -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }
$env:CUDA_PATH = $CudaPath
$env:PATH = "$CudaPath\bin\x64;$CudaPath\bin;$env:PATH"
$env:APXINF_CUDA_ARCH = $CudaArch
$env:APXINF_CUDA_ARCH_CUTLASS = $CutlassArch
$env:APXINF_NVCC_PARALLELISM = "$Jobs"
$env:CARGO_BUILD_JOBS = "$Jobs"
$env:CARGO_TARGET_DIR = $TargetDir
$env:CMAKE_GENERATOR = 'NMake Makefiles'   # cmake-rs picks the Visual Studio generator by default and cannot find the C++ compiler
$env:CXXFLAGS = '/MT'                      # cc-rs objects must use the same static CRT as the nvcc objects
if ($Mirror) {
  $env:CARGO_HOME = Join-Path (Split-Path -Parent $TargetDir) 'apx-cargo-home'
  New-Item -ItemType Directory -Force $env:CARGO_HOME | Out-Null
@'
[source.crates-io]
replace-with = "rsproxy-sparse"
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
[http]
multiplexing = false
timeout = 120
[net]
retry = 10
'@ | Set-Content "$env:CARGO_HOME\config.toml" -Encoding ascii
}

# ------------------------------------------------------------ apxinf.exe
Step 'cargo build apxinf.exe (--features cuda-no-nvtx)'
Push-Location $Repo
try {
  $cfg = if ($DebugProfile) { @() } else { @('--release') }
  Run cargo (@('build', '--locked', '--features', 'cuda-no-nvtx') + $cfg)
} finally { Pop-Location }
$exe = Join-Path $TargetDir ($(if ($DebugProfile) { 'debug' } else { 'release' }) + '\apxinf.exe')
Need (Test-Path $exe) "apxinf.exe not produced at $exe"
Run $exe @('--help')
Write-Host "apxinf.exe OK: $exe" -ForegroundColor Green

if (-not $Wheel) { Write-Host "`nDone. Re-run with -Wheel to build the Python package."; return }

# ------------------------------------------------------------ python package
Step "conda env '$EnvName' (python 3.10)"
if (-not $Conda) {
  $cands = @($env:CONDA_EXE, "$env:USERPROFILE\miniconda3\Scripts\conda.exe", "$env:USERPROFILE\anaconda3\Scripts\conda.exe", 'C:\ProgramData\miniconda3\Scripts\conda.exe', 'C:\ProgramData\anaconda3\Scripts\conda.exe')
  $Conda = ($cands | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1)
  if (-not $Conda) { $g = Get-Command conda -ErrorAction SilentlyContinue; if ($g) { $Conda = $g.Source } }
}
Need ($Conda -and (Test-Path $Conda)) 'conda (Miniconda): pass -Conda C:\path\to\conda.exe'
$envDir = (& $Conda info --base | Select-Object -First 1).Trim() + "\envs\$EnvName"
$py = "$envDir\python.exe"
$pipIdx = @()
if (-not (Test-Path $py)) {
  $ch = @()
  if ($Mirror) {
    $t = 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs'
    $ch = @('--override-channels', '-c', "$t/main", '-c', "$t/msys2")
    $env:CONDA_REMOTE_READ_TIMEOUT_SECS = '300'
  }
  Run $Conda (@('create', '-n', $EnvName, 'python=3.10', 'pip', '-y') + $ch)
}
Need (Test-Path $py) "python.exe missing in $envDir"
if ($Mirror) { $pipIdx = @('-i', 'https://pypi.tuna.tsinghua.edu.cn/simple') }
Run $py (@('-m', 'pip', 'install') + $pipIdx + @('maturin==1.16.0', 'numpy==1.26.4'))

# sentencepiece-sys (pulled in by apxinf-py) builds its C++ core with CMake. Its /MT option is
# broken (comma typo in option(...), and CMP0091 NEW needs CMAKE_MSVC_RUNTIME_LIBRARY), so force the
# static CRT in the cached crate source. Idempotent; the link fails with LNK2038/LNK1169 without it.
Step 'fetch crates and patch sentencepiece-sys CRT'
Push-Location $Repo
try { Run cargo @('fetch', '--locked') } finally { Pop-Location }
$cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { "$env:USERPROFILE\.cargo" }
$cm = Get-ChildItem "$cargoHome\registry\src" -Recurse -Filter CMakeLists.txt -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -match 'sentencepiece-sys-[0-9.]+\\source\\CMakeLists\.txt$' } | Select-Object -First 1
Need ($null -ne $cm) 'sentencepiece-sys source not found in the cargo registry'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$txt = [System.IO.File]::ReadAllText($cm.FullName)
if ($txt -notmatch 'CMAKE_MSVC_RUNTIME_LIBRARY') {
  $needle = 'option(SPM_ENABLE_MSVC_MT_BUILD, "Use /MT flag in MSVC build" OFF)'
  Need ($txt.Contains($needle)) "sentencepiece CMakeLists layout changed ($($cm.FullName)); update this patch"
  $txt = $txt.Replace($needle, $needle + "`nset(SPM_ENABLE_MSVC_MT_BUILD ON)`nset(CMAKE_MSVC_RUNTIME_LIBRARY `"MultiThreaded`")")
  [System.IO.File]::WriteAllText($cm.FullName, $txt, $utf8)
  Write-Host "patched $($cm.FullName)"
} else { Write-Host 'sentencepiece already patched' }

Step 'build apxinf_py wheel (release, --features cuda)'
$wheels = Join-Path (Split-Path -Parent $TargetDir) 'apx-wheels'
New-Item -ItemType Directory -Force $wheels | Out-Null
$env:PATH = "$envDir;$envDir\Scripts;$env:PATH"
Push-Location $Repo
try {
  Run "$envDir\Scripts\maturin.exe" @('build', '--release', '--features', 'cuda', '--auditwheel', 'skip',
    '-m', 'crates/apxinf-py/Cargo.toml', '-i', $py, '--out', $wheels)
} finally { Pop-Location }
$whl = Get-ChildItem "$wheels\apxinf_py-*.whl" | Sort-Object LastWriteTime | Select-Object -Last 1
Need ($null -ne $whl) 'no wheel produced'

Step 'install'
Run $py (@('-m', 'pip', 'install', '--force-reinstall', '--no-deps', $whl.FullName))
Run $py (@('-m', 'pip', 'install') + $pipIdx + @('-e', "$Repo\python\apxinf"))
# Python >= 3.8 ignores PATH for DLL lookup, and CUDA 13 keeps cublas/cudart in bin\x64.
# A .pth line runs at interpreter start, so `import apxinf_py` works without extra setup.
$sp = (& $py -c "import sysconfig;print(sysconfig.get_paths()['purelib'])" | Select-Object -First 1).Trim()
$dlls = @("$CudaPath\bin\x64", "$CudaPath\bin") | Where-Object { Test-Path $_ }
$line = 'import os; [os.add_dll_directory(d) for d in (' + (($dlls | ForEach-Object { "r'$_'" }) -join ', ') + ') if os.path.isdir(d)]'
Set-Content "$sp\apxinf_cuda_dll.pth" $line -Encoding ascii
$ver = & $py -c "import apxinf_py;print(apxinf_py.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'import apxinf_py failed' }
Write-Host "`nWHEEL OK: apxinf_py $ver   env: $envDir   wheel: $($whl.FullName)" -ForegroundColor Green
