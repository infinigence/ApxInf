# PR Checklist

PR URL：
Source branch：
Source HEAD SHA：
Base：APXinf-Contest-2026
Owner：
Task ID：
Status：review

## Scope

- [ ] 修改文件都在 owner 范围内，或共享文件变更已获批准。
- [ ] 未修改 benchmarks/qwen38_4090/evaluation/。
- [ ] 未提交模型权重、凭据、hidden data、runs、logs 或 Nsight 大文件。
- [ ] 没有 case ID、答案、token 序列或输出位置硬编码。
- [ ] 没有 vLLM/Transformers/CPU/其他 GPU/其他模型外部 fallback。

## Tests

- [ ] git diff --check
- [ ] python3 benchmarks/qwen38_4090/evaluation/test.py check
- [ ] cargo test --workspace
- [ ] 相关 unit/integration/fake-runtime tests
- [ ] protocol gate 逐项结果（如适用）
- [ ] health、取消、容量拒绝和失败恢复（如适用）
- [ ] GPU0 replay 命令和结果（如适用）

## Evidence

- [ ] 完整 commit SHA、model revision、contract SHA256。
- [ ] GPU UUID、driver/CUDA、warmup/repeat、CV 和时钟/环境记录。
- [ ] raw artifact 路径和 manifest SHA256。
- [ ] baseline/candidate、接受阈值和负结果。
- [ ] feature flag、默认值和一条回滚命令。
- [ ] 未运行测试和原因已写明。

## Review decision

Reviewer：

Decision：accept | request changes | reject

Blocking findings：

Integration commit：

Post-merge smoke result：
