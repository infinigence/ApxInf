# Git、PR 与服务器同步流程

当前仓库是 `https://github.com/mygqdmmss/ApxInf.git`，集成分支为
`APXinf-Contest-2026`。成员1在服务器上维护集成分支；成员2、3在自己的电脑
上开发并推送 feature/experiment 分支。不要共享服务器凭据、token 或私钥。

## 分支命名

```text
feat/protocol-stub       # 成员2
feat/oracle-loader       # 成员2
feat/qwen35-runtime      # 成员1
exp/w4-gemv              # 成员3
exp/graph-benchmark      # 成员3
exp/bonus-context        # 成员3
integrate/<pr-number>    # 成员1临时组合分支
```

## 远程成员的本地流程

```bash
git clone https://github.com/mygqdmmss/ApxInf.git
cd ApxInf
git fetch origin --prune
git switch -c feat/protocol-stub origin/APXinf-Contest-2026
git status --short --branch
python3 benchmarks/qwen38_4090/evaluation/test.py check

git add <declared-files>
git diff --cached --check
git commit -m "feat(protocol): add admission checks"
git push -u origin feat/protocol-stub
```

PR 的 base 固定为 `APXinf-Contest-2026`。不要直接 push 集成分支，不要提交
模型权重、日志、`evaluation/runs/`、Nsight 文件或本地凭据。

## 成员1在服务器检出 PR

成员1的主工作树保持在 `/mnt/chuangxin/team2/ApxInf`，不在主工作树切换
远程分支；每个 PR 使用独立 sibling worktree：

```bash
cd /mnt/chuangxin/team2/ApxInf
git fetch origin --prune
git worktree add ../review-protocol-stub origin/feat/protocol-stub
cd ../review-protocol-stub
git status --short --branch
git diff origin/APXinf-Contest-2026...HEAD --stat
```

验证结束后只清理自己创建的 worktree：

```bash
cd /mnt/chuangxin/team2/ApxInf
git worktree remove ../review-protocol-stub
git worktree prune
```

不要用 `git reset --hard` 覆盖主工作树的未提交内容。

## Review 和合并顺序

1. 作者填写 [pr-checklist.md](../templates/pr-checklist.md)，状态改为
   `review`。
2. 成员1检查 diff 范围、合同只读、测试和复现命令。
3. 成员1在 review worktree 跑本地/fake 测试；需要 GPU 的命令进入服务器
   验证流程。
4. 协议 PR 先于真实 runtime 接入；loader/oracle 先于完整模型 correctness；
   实验 PR 先保留在隔离分支。
5. 原始证据和回滚点完整后，成员1才合并。合并后在集成分支重跑最小 smoke。

若 GitHub UI 不可用，可由成员1执行：

```bash
cd /mnt/chuangxin/team2/ApxInf
git switch APXinf-Contest-2026
git pull --ff-only origin APXinf-Contest-2026
git merge --no-ff origin/feat/protocol-stub -m "merge: protocol stub"
git push origin APXinf-Contest-2026
```

合并前保留 PR URL、源分支 SHA 和合并 commit SHA；不要用 squash 丢失可回滚边界，
除非 PR 明确给出 source SHA 映射。

## 更新和冲突

作者在自己的电脑上基于最新集成分支更新：

```bash
git fetch origin
git rebase origin/APXinf-Contest-2026
git push --force-with-lease origin feat/protocol-stub
```

只允许作者对自己的分支使用 `--force-with-lease`。冲突触及另一 owner 的文件
时先暂停并在 PR 留言，不要静默删除对方代码。

## PR 最低要求

每个 PR 必须包含：

- 目的、非目标、修改文件和接口影响；
- task ID、状态和 owner；
- 本地命令、服务器重放命令（如需要）及结果；
- protocol/correctness/reliability/performance 证据和 artifact manifest；
- 风险、限制、feature flag 和回滚 SHA；
- 未运行的测试及原因。

成员1合并后，成员2/3执行 `git fetch origin --prune` 和
`git rebase origin/APXinf-Contest-2026`。新的任务从最新集成分支切出。
