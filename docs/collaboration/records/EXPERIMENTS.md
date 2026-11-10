# 实验索引

本文件只做索引；完整结果使用 experiment-record 模板，raw artifact 存服务器或共享存储，不进 Git。

| Experiment ID | Owner | Candidate | Fixed commit | GPU UUID | Status | Result/artifact |
| --- | --- | --- | --- | --- | --- | --- |
| M1-R0-CAPTURE | 成员1 | 环境/合同/模型基线采集（无 GPU model job） | `d1d0b0aec2f9545eb0e2195e2e7ea0af1babbff1` | `GPU-d074a13d-dbb6-fceb-4caf-a45be9be9281` | blocked | `/mnt/chuangxin/team2/artifacts/apxinf/r0/d1d0b0aec2f9545eb0e2195e2e7ea0af1babbff1/`; Xid evidence unavailable and M2-O0 handoff missing |
| M3-E0-W4-GEMV-001 | member3 | packed-W4 decode GEMV scaffolding | `81dad4753f2aa72b77f8deddbe7fb290b3d1789e` | pending RTX 4090 replay | planned | `benchmarks/campaign/manifests/w4-gemv-baseline.json` |

## 记录要求

每条实验必须绑定：

- model revision、contract SHA256、完整 commit SHA；
- GPU UUID、driver/CUDA、时钟/温度/功耗；
- 输入 manifest、服务命令、warmup/repeat/timeout；
- baseline/candidate 和唯一变量；
- correctness、success、latency、CV、显存、fallback/OOM/NaN/Xid、recovery；
- 接受/拒绝结论、限制和回滚 SHA；
- raw artifact 路径和 SHA256。

## 结果状态

- planned：已登记但未运行；
- active：正在运行；
- accepted：满足 feature-off/on 和端到端接收门；
- rejected：有明确失败原因和回滚点；
- superseded：被新实验替代，但原始证据保留。
