# Claude Code 本地接入验证：2026-10-09

最终构建通过真实 Claude Code 的读取、修改、执行检查和取消四项验证。最终 HTTP 检查 37 项全部通过，直接断开连接取消、队列溢出和空闲 Worker 故障检查也通过。此前发现的取消指标分类错误已修复并重新验证，原始失败证据保留。

这些任务实际使用 Claude Code 2.1.289，通过 ApxInf 的 Anthropic Messages 接口调用本地 Qwen3.5-2B，没有使用付费 Claude 模型替代本地模型。记录证明指定受限配置可以完成这些工具循环，不证明全部 Claude Code 功能兼容，也不代表复杂代码任务的成功率。每轮、每类任务仅有一次样本。

## 环境与实际能力

| 项目 | 本轮值 |
| --- | --- |
| 机器 | Apple M4，Mac16,10，16 GiB 统一内存 |
| 系统 | macOS 27.0.1，build 26A434 |
| Claude Code | 2.1.289，来自真实 `system/init` 事件 |
| HTTP 地址 | 初轮与补充实验为 `http://127.0.0.1:8080`，最终验证为 `http://127.0.0.1:8081` |
| 对外模型名 | `apxinf-local` |
| 模型 | 本地 `Qwen/Qwen3.5-2B`，使用已有权重 |
| 权重快照 | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 服务版本标识 | `apxinf-serial-v0.1`，Rust release 构建 |
| Worker 协议 | `apxinf-worker/2.0` |
| Python / MLX / MLX-LM | 3.14.3 / 0.32.1 / 0.31.3 |
| 服务上下文 / 单次输出上限 | 16384 / 2048 token |
| 本轮客户端单次输出上限 | 512 token |
| 执行配置 | greedy，单序列，prefill chunk 256，输出事件批量 1 token |

能力与模型清单来自[服务就绪快照](../../benchmarks/serving/results/ready-256-1.json)。该快照包含权重文件散列、运行库版本和模型 revision。机器与运行环境见[环境记录](../../benchmarks/serving/results/environment-before.json)。

测试启动器持有已有 Metal 测量锁，没有并行执行另一组 GPU 测试。桌面活动仍然存在，测试并非完全隔离的实验环境。服务报告不支持 batching、图片、公共会话状态和 prompt cache，本轮不作相关能力声明。

## 初轮真实任务结果

| 任务 | 实际工具调用 | 模型请求数 | CLI 总时长 | 输入 token 合计 | 输出 token 合计 | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 读取随机 marker | Read | 2 | 6.856 秒 | 1341 | 105 | 通过 |
| 修复分数归一化函数 | Read、Edit | 3 | 11.521 秒 | 2919 | 201 | 通过 |
| 运行本地检查脚本 | Read、Bash | 3 | 9.036 秒 | 3421 | 121 | 通过 |

请求数来自任务前后服务完成计数器的差值。token 数来自真实客户端累计 usage，包含工具结果回传后的请求。CLI 总时长包括进程启动、提示构造、模型推理和工具执行，不能直接当作单次 HTTP 延迟或模型 TPOT。

读取任务在提示中只提供文件名。测试脚本事先创建随机 marker，Claude Code 实际调用 Read，再返回对应内容。

修改任务的初始函数只做 `value / 10`。Claude Code 实际读取并修改 `score.py`，将结果限制到 `[0, 1]`。独立检查器使用受限表达式求值，检查负数、正常值和超出范围的输入；只回复 `FIXED` 不算通过。

执行任务实际运行 `python3 check_fixture.py`。工具结果包含 `APXINF_CHECK_PASS`，并且对应已记录的 Bash 调用。没有工具调用而声称检查成功，会被判定为失败。

初轮每个任务的进程退出码均为零。初轮每个任务完成后的指标均为 active=0、queued=0、failed=0。三项任务期间最后观察到的 Worker 分配器峰值为 4038199082 字节，约 3.76 GiB。它不等于进程 RSS，也不等于整机物理内存用量。

原始证据：

- [完整 JSON 报告](../../benchmarks/serving/results/claude-20261009-initial/report.json)。
- [Read 原始事件](../../benchmarks/serving/results/claude-20261009-initial/read-0-318505bd/client.stdout.jsonl)。
- [Edit 原始事件](../../benchmarks/serving/results/claude-20261009-initial/edit-0-ba6b2d9a/client.stdout.jsonl)与[实际修改文件](../../benchmarks/serving/results/claude-20261009-initial/edit-0-ba6b2d9a/score.py)。
- [Bash 原始事件](../../benchmarks/serving/results/claude-20261009-initial/check-0-097950b1/client.stdout.jsonl)。

这些路径是本机测量产物，Git 忽略原始结果和客户端状态。将文档移至其他机器时，需要另外携带对应证据目录。

## 隔离配置与协议检查

每个任务使用独立目录和 `CLAUDE_CONFIG_DIR`。子进程只继承基本机器环境，并使用本地占位 API key，不继承用户的 provider credential 或代理配置。`HOME` 保持原值，全局 Claude Code 设置未修改。

所有实际调用保留客户端默认系统提示，没有替换或截断。共同 CLI 参数为：

```text
--bare --restricted --model apxinf-local
--output-format stream-json --verbose --include-partial-messages
--no-session-persistence --setting-sources ""
--permission-mode dontAsk --max-turns 8
```

工具列表按任务收窄：Read；Read、Edit；Read、Bash。Bash 许可精确指定为 `Bash(python3 check_fixture.py)`。真实初始化事件报告没有 MCP server，并列出两个客户端内置插件；本报告不将 `--bare` 解释成客户端内部没有任何插件记录。

本轮进程环境还配置了 `CLAUDE_CODE_DISABLE_THINKING=1`、`CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`、`DISABLE_PROMPT_CACHING=1`、`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`、`CLAUDE_CODE_DISABLE_TERMINAL_TITLE=1` 和 `CLAUDE_CODE_MAX_RETRIES=0`。这些值来自官方[环境变量说明](https://code.claude.com/docs/en/env-vars)，作用于该子进程。

真实 HTTP 检查共 29 项，全部通过，耗时 3.296 秒。其中六个测试方法访问真实服务，包含 Anthropic 和 OpenAI 两种实际流式生成、token counting、健康状态、指标及错误响应。[完整测试日志](../../benchmarks/serving/results/http-live-20261009-initial.txt)保留结果。

## 首次测试暴露的问题

第一次 Claude Code 启动产生 `system/ui_invalidate` 事件，其中 `event` 为字符串 `ui.render`。测试工具错误地假定每个 `event` 都是对象，因此在模型请求之前退出。这个失败属于测试工具解析错误，不是模型任务质量失败，也不是服务端故障。

修正后，解析器只读取 `stream_event` 对象中的内容增量，并增加原创回归测试。随后上述三个任务全部通过。[首次尝试的原始输出](../../benchmarks/serving/results/claude-20261009-initial/read-0-7b4a02ea/client.stdout.jsonl)仍然保留。

最初的受限环境也禁止连接 loopback。获得本机网络执行许可后，真实 HTTP 测试通过。该权限限制不计为服务错误率。

## 客户端模型元数据与配置修正

初次实测报告中的 `modelUsage` 显示 `contextWindow=200000`、`maxOutputTokens=32000`、`provider=firstParty` 和 `costUSD`。这些值是客户端对未知模型别名的默认处理，不能用来声明本地服务能力、费用或计费来源。服务的实际限制仍为 16384/2048。

官方支持 `CLAUDE_CODE_MAX_CONTEXT_TOKENS=16384`。对于不含 `[1m]`、不映射为已知 Claude 模型的 `apxinf-local`，该变量直接设置客户端假定的上下文窗口，并保留主动压缩行为。[官方模型配置](https://code.claude.com/docs/en/model-config#correct-the-window-for-a-gateway-or-custom-model-id)

`CLAUDE_CODE_MAX_OUTPUT_TOKENS=2048` 设置大多数请求的输出上限。本轮任务使用更小的 512。官方文档没有保证它同时重写最终 `modelUsage.maxOutputTokens` 字段，因此本报告不宣称该显示值已修正。[官方输出配置](https://code.claude.com/docs/en/env-vars)

`modelOverrides` 的结构是“Anthropic 模型 ID → provider 模型 ID 字符串”。它没有自定义上下文或输出上限字段，不适合作为 Qwen 的能力注册表。把 Qwen 映射成某个 Claude 型号会引入错误的能力假设，本方案不采用该办法。[官方映射规则](https://code.claude.com/docs/en/model-config#override-model-ids-per-version)

后续测试工具从 `/readyz` 读取部署上下文上限，并为子进程设置 `CLAUDE_CODE_MAX_CONTEXT_TOKENS`。可用 `--max-context` 指定更小的窗口。报告分别保存部署限制、客户端窗口和请求输出上限。这项配置修改的真实验证单独记录如下，没有混入初次三个任务的实测结论。

修改后的离线检查为 31 项：25 项通过，六项真实服务测试未启用。这里不重复计算初次真实 HTTP 检查。

## 补充实测：上下文与取消

补充实验使用同一服务和模型，设置 `--max-context 16384 --max-tokens 512`，依次执行 read 与 cancel。启动器继续持有 Metal 锁，测试工具没有重复获取该锁。[补充报告](../../benchmarks/serving/results/claude-20261009-context-cancel/report.json)保存全部事件、客户端输出和指标采样。

读取任务通过，实际调用 Read，CLI 总时长 6.921 秒。两次模型请求合计输入 1343 token、输出 107 token。真实结果中的 `modelUsage.contextWindow` 从初轮的 200000 变为 16384，证明该客户端配置在这次运行中生效。`modelUsage.maxOutputTokens` 仍显示 32000，不能把它解释成实际请求或服务输出上限。

取消任务没有工具权限。它要求生成长列表，测试工具在首个文本增量之后向自己的 Claude Code 进程组发送 SIGINT。客户端在启动后 0.506 秒收到文本 `I`，0.522 秒退出，并报告 `terminal_reason=aborted_streaming`。请求 ID 为 `e1f4d49f-a072-4cac-b089-4a292c5c9e64`。

服务日志记录该请求 `status=cancelled`、输入 150 token、输出 2 token、请求耗时 381 毫秒。客户端退出后的首次指标采样仍有一个活跃请求；0.201 秒处的下一次采样显示 active=0、queued=0、reserved bytes=0。因此这次实验观察到断开连接后停止生成和资源释放。[服务日志](../../benchmarks/serving/results/service-release-256-1.log)

该轮严格取消断言失败：公开指标中的 cancelled 保持 0，而 failed 从 0 增为 1。这与服务日志的 cancelled 状态不一致。测试工具等待约 10.113 秒后仍未观察到取消计数增长，因此没有把该实验记为通过。最终构建修复了指标分类并重新验证，结果单独记录如下。

这次证据证明单次客户端取消引发了实际中断，并观察到资源归还。它不证明取消分类正确，也不覆盖排队时取消、prefill 时取消、连续取消或全部竞态。原始失败报告保留，不因资源释放成功而改写为通过。

## 最终构建验证

最终服务使用端口 8081、服务 PID 81226、Worker PID 81227。Worker epoch 为 `4a5da30a-d28e-4c43-8107-cddd9b8a2f65`。执行配置仍为 prefill chunk 256、输出事件批量 1、上下文 16384、请求输出上限 2048；客户端任务输出上限为 512。测试按 HTTP、Claude Code、生命周期、Worker 故障的顺序执行，没有并行 GPU 测试。

[最终 HTTP 日志](../../benchmarks/serving/results/http-live-20261009-final.txt)记录 37 项全部通过，耗时 1.689 秒。这包含新增的生命周期目标检查与真实服务方法，没有把离线夹具当成硬件运行。

| 最终客户端任务 | 实际工具 | 模型请求 | CLI 总时长 | 输入 / 输出 token | 结果 |
| --- | --- | ---: | ---: | --- | --- |
| Read | Read | 2 | 6.142 秒 | 1329 / 99 | 通过 |
| Edit | Read、Edit | 3 | 11.674 秒 | 2919 / 201 | 通过 |
| Check | Read、Bash | 3 | 9.038 秒 | 3421 / 121 | 通过 |
| Cancel | 无工具 | 1 次被取消 | 0.509 秒 | 服务计数差值 150 / 2 | 通过 |

完成任务中的 `modelUsage.contextWindow` 均为 16384，`maxOutputTokens` 仍为客户端静态值 32000。取消任务的客户端 usage 没有完整结算，因此表格明确使用服务计数器差值，不能把客户端显示的零解释为未执行推理。[最终 Claude Code 报告](../../benchmarks/serving/results/claude-20261009-final/report.json)

最终 Claude 取消请求 ID 为 `6b0dcf79-9c12-447b-bbfd-52939590897f`。客户端在 0.498 秒收到首个文本增量，随后测试工具发送 SIGINT，客户端报告 `aborted_streaming`。退出后约 0.211 秒处，取消计数增加 1、失败计数不增加，active、queued 和预留字节均归零。严格取消断言通过，修复前的失败报告仍然保留。

[直接生命周期报告](../../benchmarks/serving/results/lifecycle-20261009-final.json)记录两项通过结果：

- 直接断开 HTTP 连接：请求 `652088da-7f22-40f8-a16b-44f21aeed0c2` 在首个内容后断开，约 0.054 秒处观察到取消计数增长和资源归零。
- 队列溢出：活跃请求 `ceb6dbfc-b148-4f70-ac2f-b8213dda2800` 持续运行，等待队列达到 16。额外请求收到 HTTP 429，错误码为 `queue_full`。关闭全部测试连接后，约 0.058 秒处资源归零，readiness 保持 200。

排队连接在收到响应头前关闭，因此没有服务器分配的公开 request ID。报告保留它们各自的本地 probe ID，没有伪造服务器 ID。等待队列归零说明请求已移出队列，不代表每个未运行请求都计入已执行请求的 cancelled 指标。

[空闲 Worker 故障报告](../../benchmarks/serving/results/worker-failure-20261009-final.json)记录最后一项破坏性检查。测试工具核对显式 PID 的用户、父子关系、启动时间、Worker 命令、模型路径、epoch 和服务监听端口，然后只向 Worker 81227 发送 SIGTERM。约 0.051 秒处 readiness 变为 HTTP 503，health 仍为 HTTP 200，服务进程身份不变，active、queued 和预留字节为零。后续只针对 PID 81227 的 `ps` 查询没有发现该进程。

macOS 的 `lsof -Fp` 输出还包含 `f9` 之类的文件描述符字段。最终测试前，工具修正为比较 `p` 开头的进程字段，并加入原创回归测试。该修正只影响测试目标识别，没有改动服务或 Worker。

所有最终请求均已结束。Worker 故障检查有意使该测试实例不可用，没有自动重启服务，也没有终止其他应用。服务操作方随后关闭最终网关和旧网关，并通过 `ps` 确认其管理的五个相关进程全部退出，持有 Metal 锁的启动器也已结束。当前不保留测试服务实例，复现前需要重新启动服务。

## 可复现命令

在仓库根目录运行以下命令，可重复三个任务。实际服务需要先就绪。服务启动器若已持有 Metal 锁，任务工具不再次获取同一把锁。

```sh
python3 benchmarks/serving/claude_code_tasks.py \
  --base-url http://127.0.0.1:8080 --model apxinf-local \
  --tasks read edit check --repeats 1 \
  --max-context 16384 --max-tokens 512 --timeout 180 \
  --output-dir /tmp/apxinf-claude-repeat
```

下面是仅允许 Read 的直接客户端接入示例。它使用新的临时目录，保留默认系统提示，并显式限制上下文和输出。上下文 16384 已通过上述真实读取任务验证；直接命令的输出上限 2048 与服务部署一致，但补充实测仍使用 512。

```sh
APXINF_CLIENT_DIR="$(mktemp -d /tmp/apxinf-claude.XXXXXX)"
mkdir -p "$APXINF_CLIENT_DIR/config"
printf 'APXINF_LOCAL_READ_OK\n' > "$APXINF_CLIENT_DIR/marker.txt"
(
  cd "$APXINF_CLIENT_DIR" || exit 1
  /usr/bin/env -i PATH="$PATH" HOME="$HOME" \
    CLAUDE_CONFIG_DIR="$APXINF_CLIENT_DIR/config" \
    ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
    ANTHROPIC_API_KEY=apxinf-local-test-only \
    ANTHROPIC_MODEL=apxinf-local \
    ANTHROPIC_DEFAULT_SONNET_MODEL=apxinf-local \
    ANTHROPIC_DEFAULT_OPUS_MODEL=apxinf-local \
    ANTHROPIC_DEFAULT_HAIKU_MODEL=apxinf-local \
    CLAUDE_CODE_MAX_CONTEXT_TOKENS=16384 \
    CLAUDE_CODE_MAX_OUTPUT_TOKENS=2048 \
    CLAUDE_CODE_MAX_RETRIES=0 \
    CLAUDE_CODE_DISABLE_THINKING=1 \
    CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    CLAUDE_CODE_DISABLE_TERMINAL_TITLE=1 \
    DISABLE_PROMPT_CACHING=1 \
    /opt/homebrew/bin/claude --bare --restricted \
      --model apxinf-local --permission-mode dontAsk \
      --tools Read --allowedTools Read --max-turns 8 \
      --setting-sources "" --no-session-persistence \
      --output-format stream-json --verbose --include-partial-messages \
      -p 'Use Read to open marker.txt. Reply with only its complete marker.'
)
```

这个示例没有授权修改任意工程或执行任意 shell 命令。需要扩展工具范围时，应同时扩展任务断言和可观察证据。

## 结论边界

当前证据覆盖本地文本推理、三种基础工具循环，以及最终构建上的取消、队列溢出和空闲 Worker 故障处理。没有覆盖长会话压缩、复杂仓库任务、MCP、子代理、图像、强制工具选择或完整交互式体验。每个故障场景仅执行一次，不构成长期稳定性或全部竞态的证明。

Anthropic 官方不将非 Claude 模型作为 Claude Code 的受支持配置。本次结论是指定版本、指定本地配置下的实际兼容结果。[官方 gateway 说明](https://code.claude.com/docs/en/llm-gateway)

本文是中文验证记录，不声称符合 ASD-STE100 英文词典或获得任何认证。
