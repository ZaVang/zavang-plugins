# git-flow — 怎么触发与使用

git-flow 给单个 repo 装上"issue → 分支 → 提交 → PR/MR、人来 merge"的纪律。**按 repo opt-in**：只有你 setup 过的 repo 才生效。

---

## 一、一次性 setup（每个 repo 做一次）

在目标 repo 里**自然语言触发** git-flow skill，例如：

```
给这个 repo 设置 git-flow
让 agent 以后都开分支、提 PR，别再直接改 main
enforce git discipline in this repo
```

或显式调用（前缀视安装而定）：`/git-flow:git-flow`

setup 会自动：

1. 检测 forge —— 看 `git remote get-url origin`：
   - 含 `github` → **GitHub**（`gh`，PR）
   - 含 `gitlab` → **GitLab**（`glab`，MR）
   - 无远程 → 本地模式（只分支 + 你本地 merge，不提 issue/PR）
2. 验登录（`gh auth status` / `glab auth status`），缺了会提示你先登录
3. 拷 3 个 hook 脚本进 `.claude/hooks/`，并把 hooks 块并入 `.claude/settings.json`

> 前置：装了 `gh`（GitHub）或 `glab`（GitLab）并已登录。纯本地 repo 不需要。

---

## 二、日常使用（不用触发，全自动）

setup 后**没有命令要记**。每个新 session：

- SessionStart hook 自动把协议注入上下文 → agent 照做：
  **读/建 issue → 开分支 → 实现 → 规范 commit → 提 PR/MR → 停（你 merge）**
- 你只管正常派活，例如 "修下导出 CSV 的乱码"。

issue 只在**任务非 trivial 且你没给现成 issue** 时才建（typo/一行改会跳过）。

---

## 三、两个护栏（机械兜底，偷懒也偷不成）

| 你做的事 | 结果 |
|---|---|
| 在 `main`/`master` 上让 agent 改文件 | **被拦**，提示先开分支 |
| agent 想 `gh pr merge` / `glab mr merge` | **被拦**，merge 留给你 |

**逃生口**：某 repo 想临时放开（直接改 main / 允许 agent merge）→ 建空文件
`.claude/gitflow.allow-main`；删掉即恢复护栏。

---

## 四、GitHub vs GitLab

同一插件，按 forge 自动切换，无需配置：

| | GitHub（`gh`） | GitLab（`glab`） |
|---|---|---|
| review 单元 | PR | MR |
| 建 issue | `gh issue create --title --body` | `glab issue create --title --description` |
| 提 review | `gh pr create --base main` | `glab mr create --target-branch main` |
| 关 issue | 描述里 `Closes #n` | 描述里 `Closes #n` |

---

## 五、装完做个冒烟

在 setup 过的 repo 里：

```
git rev-parse --abbrev-ref HEAD     # 若是 main/master，下一步应被拦
```
然后让 agent 改任意文件 —— 应看到 `[git-flow] Blocked: you are on 'main'...`，并自动改为先开分支。

---

## 六、和 loop 的关系

git-flow 是包在 multi-ralph / product-loop 外面的 git 壳：loop 在分支上干活、Evaluator 当本地验证 gate、完成提 PR/MR、你 merge。护栏对 loop 同样生效。详见 [git-flow-guide.md](../docs/git-flow-guide.md)。
