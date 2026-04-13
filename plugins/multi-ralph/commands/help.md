在运行 `/multi-ralph:multi-ralph` 之前，请先阅读此指南，确认所有前置条件就绪。

---

## 什么是 Multi-Ralph Loop？

Multi-Ralph 是一个**三角色 Sprint 执行工作流**：

```
Orchestrator（你当前的 Claude 会话）
 ├── Planner   → 读 Sprint 合同，写任务计划（只规划 WHAT，不写代码）
 ├── Generator → 读计划，自主实现代码，运行验收命令
 └── Evaluator → 独立重跑验收命令，输出 COMPLETE / CONTINUE / BLOCKED
```

三个 subagent **互不共享上下文**，只通过文件通信——这是它的核心设计，确保每个角色视角干净、无污染。

---

## ⚠️ 前置条件（缺少任何一项都会导致运行失败）

### 必须存在：`docs/plans/SPRINT.md`

这是 Multi-Ralph 的**"合同"**，没有它无法运行。

文件必须包含两个部分：

**① 任务清单**（用 `[ ]` 标记未完成任务）
```markdown
## 任务清单

- [ ] 任务 1：实现 XXX 功能
  - 目标：...
  - 验收标准：...

- [ ] 任务 2：添加 YYY 模块
  - 目标：...
  - 验收标准：...
```

**② 验收命令**（Evaluator 会独立重跑这些命令来验证）
```markdown
## 验收命令

```bash
pytest tests/ -v --tb=short
python -m mypy src/ --ignore-missing-imports
```
```

> **⚠️ 如果没有验收命令，Evaluator 无法独立验证，Loop 会变成空转。**

### 建议存在：`docs/plans/pitfalls.md`

陷阱知识库——Planner、Generator、Evaluator 三个角色都会读取它，避免重蹈覆辙。

第一次运行前留空也可以，后续 Sprint 结束后 Loop 会自动追加新发现的陷阱。

```markdown
# Pitfalls

<!-- 初次创建可留空，Loop 运行后会自动追加 -->
```

---

## 快速开始

### Step 1：创建项目规划文件

```bash
mkdir -p docs/plans docs/orch

# 创建 SPRINT.md（必须）
# 创建 pitfalls.md（推荐，可留空）
touch docs/plans/pitfalls.md
```

### Step 2：编写 SPRINT.md

参考上方格式，填写任务清单 + 验收命令。

> **提示**：使用 `/claude-init:init-project` 初始化项目时会自动创建 `docs/` 目录结构。

### Step 3：启动 Loop

```
/multi-ralph:multi-ralph
```

可选参数：
```
/multi-ralph:multi-ralph --max-iter 3
/multi-ralph:multi-ralph --sprint docs/plans/MY_SPRINT.md --max-iter 5
```

---

## Loop 的文件协议（了解即可）

| 文件 | 谁写 | 谁读 | 作用 |
|------|------|------|------|
| `docs/plans/SPRINT.md` | 你（预先准备） | 全部三个角色 | Sprint 合同 |
| `docs/plans/pitfalls.md` | Loop 自动追加 | 全部三个角色 | 陷阱知识库 |
| `docs/orch/plan.md` | Planner | Generator | 本轮任务计划 |
| `docs/orch/gen_status.md` | Generator | Evaluator | 实现结果自报 |
| `docs/orch/eval.md` | Evaluator | Planner（下轮） | 验收报告 + 决策 |

---

## 常见问题

**Q：Loop 跑完了但任务还是 FAILED？**
A：查看 `docs/orch/eval.md` 中的「失败原因分析」，手动修复后可再次运行 Loop。

**Q：BLOCKED 是什么意思？**
A：Evaluator 判断问题无法通过修改代码解决（如环境依赖缺失、外部服务不可用），需要人工介入。

**Q：可以只跑一次不循环吗？**
A：`/multi-ralph:multi-ralph --max-iter 1`

**Q：如何在多个 Sprint 之间复用陷阱知识？**
A：`docs/plans/pitfalls.md` 是跨 Sprint 持久化的，Loop 每次结束都会追加新陷阱，无需手动维护。
