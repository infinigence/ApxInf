# 远程成员本地开发环境

本文件描述成员2、成员3在自己电脑上的开发环境。成员1的服务器 GPU、模型权重、
Xid、显存和正式评测流程见 workflows/server-gpu-validation.md。启动 prompt 通过聊天
分别发送，不提交到仓库。

## 结论

| 成员 | 首批任务是否需要 NVIDIA GPU | 推荐系统 | 必需组件 |
| --- | --- | --- | --- |
| 成员2 | 不需要 | Linux、macOS 或 Windows/WSL2 | Git、Rust stable、Python 3.9+、curl |
| 成员3 | 不需要完成 M3-E0 | Linux 最方便；macOS/Windows 可做脚本和静态检查 | Git、Rust stable、Python 3.9+、可选 CUDA toolkit |
| 成员1 | 需要正式服务器 GPU | 服务器 Linux x86_64 | RTX 4090、NVIDIA driver/CUDA、Rust、Python、Git |

成员2的协议 stub、fake runtime、schema、loader manifest 单元测试和 oracle 格式可以
完全在 CPU 上完成。成员3的 campaign 目录、paired harness、shape inventory、配置
解析和静态检查也可以完全在 CPU 上完成。没有本地 GPU 时，不要把任务标为 blocked；
把需要 CUDA 的部分交给成员1在服务器重放。

## 共同软件基线

仓库使用 Rust 2021 edition，但没有提交固定的 rust-toolchain.toml。因此使用 rustup
安装当前 stable，并以仓库的 Cargo.lock 做依赖锁定。

需要：

- Git 2.30 或更新版本；
- Rust stable、cargo、rustfmt；
- Python 3.9 或更新版本，建议 3.10/3.11；
- python3 -m venv、pip；
- curl；jq、sha256sum 或 shasum 为方便审计的可选工具；
- 网络只用于 clone 和下载依赖。不要把 GitHub token、模型凭据或私钥写入仓库。

按操作系统补齐本地编译工具：

- Debian/Ubuntu/WSL2：`sudo apt install build-essential pkg-config`；
- Fedora/RHEL：`sudo dnf groupinstall "Development Tools"`，并安装 `pkgconf-pkg-config`；
- macOS：先运行 `xcode-select --install`，确认 Command Line Tools 已可用；
- Windows 原生：安装 Visual Studio Build Tools 的“Desktop development with C++”工作负载，
  并使用 PowerShell 或 Developer Command Prompt；如果使用 WSL2，按 Debian/Ubuntu 方式安装
  Linux 工具链即可。

首次提交前配置自己的 Git 身份（不要使用共享账号）：

~~~bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --get user.name
git config --get user.email
~~~

每名成员必须拥有 GitHub 仓库的个人 push 权限，并使用自己的 SSH key 或个人 PAT 完成
认证。推荐 SSH：把个人公钥加入 GitHub 后使用 `git@github.com:mygqdmmss/ApxInf.git`；
使用 HTTPS 时只通过 Git credential manager 或系统密码库提供 PAT。不要共享服务器账号、
SSH 私钥、PAT，也不要把凭据写入 remote URL、脚本、日志或仓库文件。

不需要：

- Docker；
- 成员1的服务器账号；
- Qwen 权重或真实 hidden 数据来完成 M2-P0/M3-E0；
- 本地启动正式评测服务。

如果使用 Windows，推荐 Git Bash 或 WSL2 执行仓库命令。成员2的纯协议工作也可以
直接使用 PowerShell；成员3若要编译 CUDA 源文件，应使用 Linux/WSL2，而不是原生
Windows 工具链。

## 通用初始化

把以下命令中的分支替换为自己的分支；不要直接在集成分支开发。

~~~bash
git clone https://github.com/mygqdmmss/ApxInf.git
cd ApxInf
git fetch origin --prune
git switch -c feat/protocol-stub origin/APXinf-Contest-2026

rustup toolchain install stable
rustup component add rustfmt --toolchain stable

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip

python3 benchmarks/qwen38_4090/evaluation/test.py check
cargo +stable check --workspace --locked
git status --short --branch
~~~

成员3使用 exp/w4-gemv 或其他 exp/bonus-context 等 exp/* 分支；成员2使用
feat/protocol-stub 或 feat/oracle-loader。如果本机已有 stable，可以省略 rustup 安装，
但必须在启动报告中记录 rustc --version、cargo --version、python --version 和操作系统。

Windows PowerShell 激活虚拟环境的命令是：

~~~powershell
.venv\Scripts\Activate.ps1
~~~

不要把 .venv/、target/、模型目录或评测生成物提交到 Git。

## 成员2环境：协议与 oracle

### M2-P0 最小环境

M2-P0 只依赖 Rust workspace 和 CPU fake runtime。先完成通用初始化，再按任务需要
安装测试依赖：

~~~bash
python -m pip install pytest
~~~

如果要测试仓库 Python frontend，可使用其声明的可选依赖；这不是协议 stub 的必需项：

~~~bash
python -m pip install -e "python/apxinf[test,tokenizer]"
~~~

M2-P0 的验收重点是：

~~~bash
cargo +stable fmt --all -- --check
cargo +stable check --workspace --locked
cargo +stable test -p apxinf-loader --locked
python -m pytest -q python/apxinf/tests
python3 benchmarks/qwen38_4090/evaluation/test.py check
~~~

协议 PR 新增的测试命令以实际 crate 或 package 为准，必须写入 PR；不要为了让全量
测试变绿而修改 evaluator、scorer 或无关示例。

### M2-O0 oracle 编写与服务器执行

成员2负责在本地编写可移植的 oracle/manifest/golden 生成器、参数校验和 synthetic
fixture；成员1负责在服务器执行需要真实 checkpoint 的 oracle。成员2不需要、也不应在
自己的电脑下载约 19.57 GiB 的 checkpoint、展开完整 BF16 权重或运行 8K/16K 逐层 oracle。
当前 `transformers`/`vllm` 也没有原生 `qwen3_5` runtime，Qwen3Next 只能作为适配起点，
不能把本地相近模型输出当作权威答案。

当前服务器 SafeTensors header 的估算是：packed W4 逻辑参数约 24.30B，若全部展开为
BF16 约 55.6 GB（不含 CUDA context、workspace、KV、GDN state 和临时 buffer）。这是
容量预算证据，不是要求任何远程成员下载的环境要求；完整 BF16 副本也不是 oracle 的强制
artifact，优先保存选择性 dequant block/golden，最终以服务器 manifest 的实际峰值为准。

成员2本地只安装脚本实际需要的轻量依赖，并记录版本：

~~~bash
python -m pip install safetensors numpy
~~~

不把 `transformers`、模型权重或 tokenizer 私有文件列为成员2的必需环境。oracle CLI
必须接受显式 `--model-dir`、`--output-dir`、`--revision` 和 `--layers/--stages` 参数，
使成员1可以在服务器重放同一个 commit；本地用 synthetic W4 fixture 和 tiny input
完成脚本/格式测试。

需要真实 checkpoint 的一次性执行由成员1排入服务器 GPU 队列：

~~~bash
exec 9>/tmp/apxinf-gpu-job.lock
flock -n 9 || exit 2
export CUDA_VISIBLE_DEVICES=GPU-343bc895-b011-22fa-4449-97207aa2bdec
export APXINF_SHARED_ARTIFACT_ROOT=/mnt/chuangxin/team2/artifacts/apxinf
python tools/oracle/generate_golden.py \
  --model-dir /mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4 \
  --output-dir "$APXINF_SHARED_ARTIFACT_ROOT/oracle/<revision>/<commit-sha>" \
  --revision 63768c10df38c0395e12ef49edac1bd539eaeeea \
  --layers 0,1,3,63
~~~

实际命令、层选择和显存峰值必须写入 PR；`tools/oracle/generate_golden.py` 是成员2应创建
的稳定入口。服务器输出至少包括 manifest、reference token IDs、selected layer
hidden/state/logit golden、生成参数和 SHA256。原始权重、完整 BF16 展开副本和大日志只留
在受控共享路径，不进 Git；远程成员消费 manifest、哈希和经批准导出的 golden artifact，
不复制模型权重。

上述命令中的 Python 依赖只在服务器 P0 job 的隔离环境中按生成器实际 imports 安装；成员2
的电脑不需要安装 `transformers`、vLLM、`huggingface_hub` 或任何模型 serving runtime。
若服务器侧借用 `transformers`/Qwen3Next 代码，只能作为适配参考；checkpoint-specific
实现、逐层对拍和最终 artifact identity 仍由服务器 job 负责。成员1把可导出的最小 bundle
通过批准的项目 artifact/release 通道提供给远程成员，并在 oracle handoff 中登记方式和
SHA256；原始共享目录不作为远程成员的必需挂载点。

### 成员2不需要配置的内容

- 不需要 CUDA toolkit、nvcc 或 NVIDIA GPU；
- 不需要访问服务器端口；
- 不需要修改 src/main.rs 才能运行 fake stub；
- 不需要安装 vLLM、Transformers serving 或其他推理 runtime。

如果协议实现确实需要新增 Cargo.toml、Cargo.lock 依赖，先在 PR 说明最小变更，由
成员1负责集成入口文件；成员2可以在自己的分支验证该变更。

## 成员3环境：benchmark 与实验

### M3-E0 最小环境

M3-E0 的 harness、manifest、shape inventory 和文档不需要本地 GPU：

~~~bash
python -m pip install pytest numpy
python -m compileall -q benchmarks scripts
cargo +stable fmt --all -- --check
cargo +stable check --workspace --locked
~~~

如果现有脚本没有使用 numpy，可以不安装它；不要为了实验方便引入未使用的重型依赖。
benchmark 脚本必须能在没有服务、没有模型权重时完成 manifest/schema 静态检查。

### 可选本地 CUDA 环境

只有需要本地编译或运行实验 kernel 时才配置：

- Linux x86_64；
- NVIDIA driver 与 CUDA 12.x toolkit；
- nvcc、nvidia-smi；
- 能够编译 SM89 的 CUDA 工具链；
- 足够的本地 GPU 显存用于短 kernel smoke。

自检：

~~~bash
nvidia-smi
nvcc --version
APXINF_CUDA_ARCH=sm_89 cargo +stable check --workspace --locked --features cuda-no-nvtx
~~~

cuda-no-nvtx 用于没有 NVTX 库的本地环境；如果本地安装了可用 NVTX，也可以按
仓库已有配置使用 --features cuda。不要为适配本地机器修改 crates/apxinf-cuda/build.rs
或把 experimental kernel 加入默认 adapter 列表。

本地 CUDA 结果只作为 development evidence。提交给成员1时必须给出：

- 完整 commit SHA；
- APXINF_CUDA_ARCH、CUDA/driver 版本和实际 GPU 信息；
- baseline/candidate paired 命令；
- warmup、repeat、CV、显存、correctness、recovery 和 raw artifact SHA256；
- 服务器 GPU lane 和一条可复制 replay 命令。

## 本地验证和服务器验证的边界

| 项目 | 成员2/3本地 | 成员1服务器 |
| --- | --- | --- |
| Rust 编译、格式、单元测试 | 可以作为 PR 证据 | 合入前重跑 |
| fake protocol 七项 gate | 成员2必须完成 | 接真实 runtime 后重跑 |
| loader manifest/oracle schema | 可以完成 | 生产 loader 接线后重跑 |
| W4/GEMV/Graph microbenchmark | 可做开发信号 | GPU0/指定 replay lane 复现 |
| 端到端 TTFT/TPOT、C4/C8 | 不能作为正式成绩 | 只能由服务器复现 |
| /health 真实 capability、fallback | stub 只能标 fixture | GPU0 正式检查 |
| Xid、正式 OOM/reliability、最终 eligibility | 不能伪造或替代 | 只能由服务器证据决定 |

本地没有 RTX 4090 时，成员3仍然可以提交 harness 和候选代码；“没有本地 GPU”
不是理由去放宽 feature-off 默认、跳过 paired A/B，或宣称性能收益。

## 分支、记录和共享文件

成员2、成员3只能在自己的 feature/experiment 分支提交，不能直接 push
APXinf-Contest-2026。日常进度写自己的 PR 描述或 task-specific progress log；
不要并发修改 docs/collaboration/records/PROGRESS.md。聚合 PROGRESS、集成 SHA 和
服务器 replay 结果由成员1合并后统一回填。

成员2如果需要可运行 stub 的新 HTTP 依赖，先在 task-spec/PR 中列出最小 Cargo.toml
和 Cargo.lock 变更。成员1确认后，成员2可以在自己的分支验证；src/main.rs、生产
入口和真实 runtime 接线仍由成员1落地。这样不会因为 Cargo ownership 阻塞协议测试。

## 首次启动报告应记录

成员2、成员3发送启动报告时至少包括：

~~~text
OS/ARCH:
git --version:
rustc --version:
cargo --version:
python --version:
GPU/CUDA: none | <nvidia-smi and nvcc summary>
BRANCH:
BASE_SHA:
COMMANDS_RUN:
RESULTS:
MISSING_OPTIONAL_COMPONENTS:
NEXT_ACTION:
~~~

若某个可选组件不可用，写明它影响的具体任务和替代方案。不要把“本地环境不完整”
笼统写成 blocked；先完成所有 CPU 可做的工作，并把服务器重放要求写进 PR。

## 已知基线现象

python3 benchmarks/qwen38_4090/evaluation/test.py check 和 cargo check --workspace
--locked 是所有成员的共同基线。当前仓库的完整 cargo test --workspace --locked 可能
被既有 pi05_integrity_probe 示例的默认 feature/CUDA 符号问题阻塞；遇到该现象时保存
完整命令和输出，在 PR 中标注为基线问题，不要修改 evaluator 或无关代码来绕过它。
