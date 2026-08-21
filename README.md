# ApxInf 任务：单张 RTX 4090 上的 Qwen3.8-27B INT4 推理优化

[English](README_EN.md) | 中文

## 1. 目标

在 ApxInf 中实现并优化 `cyankiwi/Qwen3.8-27B-AWQ-INT4`，使它能在单张
NVIDIA RTX 4090 上完成稳定、正确且高效的文本推理。

提交物是一个实现 PR 和一份 `REPORT.md`。模型权重及未公开评测数据不提交到仓库。

## 2. 固定条件

- GPU：单张 RTX 4090，compute capability 8.9；
- 模型 revision：`63768c10df38c0395e12ef49edac1bd539eaeeea`；
- 权重：W4A16、group size 32、asymmetric；
- 输入：预分词 `input_ids`；
- 解码：greedy，`temperature=0`，thinking 关闭；
- 基础场景：一次一个请求，输出 128 token；
- 不允许切换到 vLLM、Transformers、CPU、其他模型或其他 GPU 作为 fallback；
- [`contract-v1.json`](benchmarks/qwen38_4090/evaluation/contract-v1.json) 是 workload、门槛和分数的唯一机器合同。

可以修改 ApxInf 的 Rust、CUDA、调度、内存管理和服务实现，但不得修改
`benchmarks/qwen38_4090/evaluation/` 中的合同、公开数据生成器或评测程序来提高结果。

## 3. 服务接口

### 健康检查

`GET /health` 在服务可用时返回 HTTP 200：

```json
{
  "status": "ok",
  "evaluation_contract": "apxinf.qwen38_27b.inference_interface.v1",
  "model_revision": "63768c10df38c0395e12ef49edac1bd539eaeeea",
  "max_model_len": 32768,
  "parallel_requests": 1,
  "fallback_active": false,
  "capabilities": {
    "pretokenized_input_ids": true,
    "token_id_output": true,
    "multimodal": false
  }
}
```

### 生成请求

`POST /v1/evaluations/generate` 接收：

```json
{
  "input_ids": [151644, 872, 198],
  "max_new_tokens": 128,
  "temperature": 0.0,
  "ignore_eos": true,
  "stream": true
}
```

流式响应的每个 token 使用一个 SSE event：

```text
data: {"type":"token","request_id":"req-1","index":0,"token_id":198}
```

结束时返回：

```text
data: {"type":"done","request_id":"req-1","usage":{"prompt_tokens":3,"completion_tokens":128,"total_tokens":131}}

data: [DONE]
```

要求：

- token index 从 0 连续递增；
- 并发请求的 `request_id` 互不混淆；
- 性能用例必须输出完整预算；
- 非法参数返回 HTTP 400 和 JSON `error`；
- 容量失败后，健康检查和小请求仍应成功；
- TTFT 和 TPOT 以客户端接收时间计算，服务端自报时间不计分。

图片能力是可选加分项。支持时，`capabilities.multimodal` 必须为 `true`，并通过
`POST /v1/chat/completions` 接收一个 `data:image/png;base64` 图片 part 和一个文本 part；
请求必须由提交的 ApxInf 实现执行且不能 fallback。不支持时保持 `false`，图片探测应以
HTTP 400、415、422 或 501 及 `error.type=unsupported_capability` 明确失败。

## 4. 评分

基础分 100 分：

| 项目 | 分值 | 要求 |
|---|---:|---|
| Correctness | 30 | 协议、公开用例及未公开用例 |
| TTFT | 35 | 1K、2K、4K、8K、16K prompt |
| TPOT | 25 | 1K 与 8K prompt |
| Reliability | 10 | 成功率、无 fallback/OOM/NaN/Xid、失败后恢复 |

可选加分：

- 长上下文 0-10 分：从 32K 以上逐级验证，最高到模型原生 262,144 positions；
- 多请求 0-10 分：C4/C8 closed-loop correct goodput，同时约束成功率、公平性和尾延迟；
- 图片能力 0-10 分：公开正确性 2 分、私有正确性 6 分，两个 split 的完整集成与稳定性各
  1 分。图片能力缺失得 0 分，但不影响合格的纯文本提交。

基础分上限为 100，加分上限为 30，排行榜总分上限为 130。图片报告由评测平台生成并
绑定实现 identity、合同 hash 和 split，`submission.json` 中的图片字段不能手工填写。

性能 cell 先 warm-up 1 次，再测 5 次，取中位数；TTFT/TPOT 的 CV 不得超过 10%。
同一轮中每个 cell 的最好有效结果作为该 cell 的满分参考。

进入性能排名前必须满足：公开正确率 100%、未公开正确率至少 11/12、请求成功率至少
99%，并且协议与可靠性门槛通过。公开运行仅用于本地调试，不代表正式成绩。

## 5. 统一测试脚本

本任务只有一个测试入口：`test.py`。

先检查代码和任务包：

```bash
python3 benchmarks/qwen38_4090/evaluation/test.py check
```

准备公开数据，需要本地模型目录及 `transformers`：

```bash
python3 benchmarks/qwen38_4090/evaluation/test.py prepare \
  --model-dir /path/to/Qwen3.8-27B-AWQ-INT4
```

启动自己的服务后运行公开测试：

```bash
python3 benchmarks/qwen38_4090/evaluation/test.py run \
  --model-dir /path/to/Qwen3.8-27B-AWQ-INT4 \
  --base-url http://127.0.0.1:8001
```

默认产物位于 `benchmarks/qwen38_4090/evaluation/runs/`。脚本会生成原始请求记录、环境
记录和 `submission.json`。不要手工填写或修改任何汇总结果。

## 6. 建议过程

1. 让 `test.py check` 通过；
2. 实现 `/health` 和生成接口，完成公开 correctness；
3. 记录初始 TTFT、TPOT、显存和失败边界；
4. 每次只验证一个性能假设，并保留负结果；
5. 优先优化端到端瓶颈，不用孤立 kernel 数字代替服务结果；
6. 修改共享 CUDA/Rust 边界后重新运行统一测试；
7. 最后从 clean checkout 重放构建、服务和公开测试。

## 7. 提交要求

PR 需要包含：

- 设计变化及影响的执行阶段；
- `test.py check` 与 `test.py run` 的命令和结果；
- 至少一个负控制或回归测试；
- correctness、性能、稳定性和显存之间的取舍；
- 已知限制、失败实验和回滚方法；
- `REPORT.md`：baseline、假设、实现、测量、结果和复现步骤。

验收以 PR 的完整 commit SHA 为准。不得针对 case ID、公开 token 序列或已知答案硬编码
输出，也不得提交模型权重、凭据、机器地址或未公开评测数据。

## 8. 最低完成标准

- clean checkout 能构建并启动；
- 健康检查声明真实；
- 公开功能用例全部通过；
- 基础性能 cell 输出完整且不 OOM、不 fallback；
- 非法请求和容量失败后服务仍可用；
- 统一测试脚本能生成完整结果；
- PR 中的报告足以让其他人独立复现。
