# 服务器 GPU 验证流程

本流程适用于成员1在服务器、成员2/3在本地的协作方式。服务器只有一个账号，所有任务串行运行；GPU 标签是逻辑 lane，不表示允许并发。

## GPU、锁和目录

每次启动前核对 UUID：

~~~bash
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv
~~~

~~~text
GPU0 GPU-d074a13d-dbb6-fceb-4caf-a45be9be9281  # 正式集成/最终成绩
GPU1 GPU-343bc895-b011-22fa-4449-97207aa2bdec  # oracle/protocol replay
GPU2 GPU-f4efcc89-d74e-d37b-caf1-52cde9f0582e  # W4/GEMV/Graph replay
GPU3 GPU-ea64faa4-13fb-ce41-1180-d6edbfb6be2f  # context/C4/C8/vision/MTP replay
~~~

GPU0 是唯一正式成绩来源；GPU1-3 的结果必须标记为 development/replay evidence。

所有 CUDA 服务、profile 和长 benchmark 先获取全局锁，不要同时启动两个模型进程：

~~~bash
exec 9>/tmp/apxinf-gpu-job.lock
flock -n 9 || { echo "another ApxInf GPU job is running" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=GPU-d074a13d-dbb6-fceb-4caf-a45be9be9281
export APXINF_GPU_LABEL=GPU0
export APXINF_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)
nvidia-smi -L
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv
~~~

确认进程报告的 `gpu_uuid` 与绑定 UUID 一致；若 `nvidia-smi --query-compute-apps` 没有运行中进程，记录“启动前无进程”，并在服务启动后再次执行该命令。进程内的 ordinal 0 不等于物理 GPU0，记录原始 UUID。正式服务默认 127.0.0.1:8000，
回放可用 8001、8002、8003，但仍须串行。启动前检查端口：

~~~bash
ss -ltnp | rg ':800[0-3]\\b' || true
~~~

### 队列优先级与登记

逻辑 GPU lane 只是记录目标 UUID；服务器一次只允许一个真实 GPU/model job。这里的
“队列”是登记、优先级和人工重试协议，不是后台调度 daemon；拿不到锁的 job 必须退出并
按优先级重新登记，不能绕过锁或占用另一张卡。成员1在取得锁前先登记一行 queue record：

~~~text
QUEUE_ID:
PRIORITY: P0-oracle | P1-base | P2-kernel | P3-bonus
SUBMITTED_AT_UTC:
OWNER:
COMMIT_SHA:
MODEL_REVISION:
TARGET_GPU_UUID:
EXPECTED_MINUTES:
ARTIFACT_DIR:
DEPENDENCY_OR_EXIT_CONDITION:
~~~

优先级固定为 `P0 oracle`（成员2生成器的真实 checkpoint 选择性 golden）、`P1 base`
（成员1 GPU0 runtime/protocol/correctness/reliability）、`P2 kernel`（成员3单变量
paired profile）、`P3 bonus`（context/C4/C8/vision/MTP）。同一优先级按提交顺序执行；
运行中的 job 不抢占。oracle 完成后释放 GPU1 logical lane，不能以“GPU1 空闲”为理由绕过
全局锁。

模型权重放在受控目录，不进 Git。raw artifact 放在：

~~~text
/mnt/chuangxin/team2/artifacts/apxinf/<YYYYMMDD>/<full-commit-sha>/<run-id>/
~~~

每个 run 至少保存 environment.json、command.txt、health.json、protocol.json、
metrics.json、stderr.log、manifest.sha256；大文件只记录路径和 SHA256。

## 标准顺序

1. 记录完整 commit SHA、工作树状态、model revision、contract SHA256、GPU UUID、driver/CUDA。
2. 验证 `journalctl -k --since '-10 min'` 或等价 Xid 证据命令可读；不可读就标记 R0 blocked，不启动正式评分。
3. 记录启动前显存、温度、功耗、时钟和 Xid。
4. 启动服务，先检查 /health 的 identity、真实 max_model_len、parallel_requests
   和 fallback_active=false。
5. 运行短请求和全部 protocol gate；失败就停服务并保存日志。
6. 运行 correctness/recovery，再运行 latency 或 bonus campaign。
7. 记录结束后的 health、显存、温度、功耗、Xid 和进程状态。
8. 停止服务，确认进程、端口和显存回收，再释放锁。

GPU0 正式 campaign 前清理其他卡的残留任务；能锁频就固定并记录，不能锁频就保留环境波动。
latency cell 固定 warmup 1、measured 5，CV > 10% 的候选不接受。正式报告只引用 GPU0；
成员2/3本地 GPU 或 CPU 结果只能作为开发反馈。

## 失败恢复检查

非法请求、容量拒绝、客户端断开和 CUDA error 后都要确认：

~~~text
/health.status == "ok"
下一次 8-token stream=false 请求为 HTTP 200/type=result
显存回到基线附近，无持续增长
无新 Xid、NaN、unexpected OOM 或 fallback
~~~

如果 CUDA context 疑似损坏，停止服务、保存 Xid/日志并创建 incident；不得用虚假的
/health 继续通过测试。

## 远程成员的 GPU 重放包

PR 必须给出 exact commit、GPU lane、CUDA_VISIBLE_DEVICES、启动命令、base URL、
model/contract/input manifest SHA256、warmup/repeat/timeout/接受阈值和期望 artifact 文件名。
成员1按原命令重放并回填结果；服务器环境不同的结果分开记录，正式结论以 GPU0 为准。
