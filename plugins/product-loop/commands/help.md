在运行 `/product-loop:product-loop` 之前，请先阅读此指南，确认所有前置条件就绪。

---

## 什么是 Product-Loop？

Product-Loop 是 multi-ralph 的增强版，在内部研发循环外层加入独立的 Product Experience Reviewer：

```
Orchestrator（你当前的 Claude 会话）
 ├── Planner   → 读 Sprint + 体验官上轮报告 + 协商记录 → 写计划 + 协商回应
 ├── Generator → 读计划 → 自主实现 → 勾 checkbox → 写状态报告
 ├── Evaluator → 独立重跑验收命令 → 写评估报告（信息性）
 └── Product Experience Reviewer → 以首次用户体验产品 → 写审计报告
```

四个 subagent **互不共享上下文**，只通过文件通信。体验官与产品方通过 negotiation.md 进行结构化协商。

**核心隐喻：Planner/Generator/Evaluator 是内部研发团队，Reviewer 是挑剔的外部用户。双方通过文档协商，永远迭代到上限。**

---

## ⚠️ 前置条件

### 必须存在：`docs/plans/SPRINT.md`

与 multi-ralph 格式相同：

```markdown
## 任务清单

- [ ] 任务 1：实现 XXX 功能
  - 目标：...
  - 验收标准：...

## 验收命令

```bash
pytest tests/ -v --tb=short
```
```

### 必须存在：系统级 agent `product-experience-reviewer`

通过 `/agents` 创建。Reviewer 以该 agent 身份运行，拥有持久化记忆。

### 建议存在：`docs/plans/pitfalls.md`

初建可留空，后续自动追加。

---

## 快速开始

```
/product-loop:product-loop
```

可选参数：
```
/product-loop:product-loop --max-iter 3
/product-loop:product-loop --sprint docs/plans/MY_SPRINT.md --max-iter 5
```

---

## 与 multi-ralph 的区别

| | multi-ralph | product-loop |
|---|---|---|
| 停止条件 | Evaluator COMPLETE 即停 | 永远跑满 max_iter |
| 默认 max_iter | 2 | 1 |
| 外部视角 | 无 | 体验官独立审计 |
| 协商机制 | 无 | negotiation.md 闭环 |

---

## 文件协议

| 文件 | 谁写 | 谁读 | 作用 |
|------|------|------|------|
| `docs/plans/SPRINT.md` | 用户 + Generator | P, G, E | Sprint 合同 |
| `docs/plans/pitfalls.md` | G + E（追加） | P, G, E, R | 陷阱知识库 |
| `docs/orch/plan.md` | Planner | Generator | 本轮任务计划 |
| `docs/orch/gen_status.md` | Generator | Evaluator | 实现结果自报 |
| `docs/orch/eval.md` | Evaluator | Planner（下轮） | 验收报告（信息性） |
| `docs/orch/product-audit-report.md` | Reviewer | Planner + Reviewer（下轮） | 体验官审计报告 |
| `docs/orch/negotiation.md` | Planner | Reviewer（下轮） | 产品方逐条回应 |
