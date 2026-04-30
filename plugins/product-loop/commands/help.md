在运行 `/product-loop:product-loop` 之前，请先阅读此指南，确认所有前置条件就绪。

---

## 什么是 Product-Loop？

Product-Loop 在内部研发循环外层加入独立的 Product Experience Reviewer，体验官先审计，研发团队再响应：

```
Orchestrator（你当前的 Claude 会话）
 ├── Reviewer  → 以首次用户体验产品 → 写审计报告
 ├── Planner   → 读审计报告 → 逐条回应 + 拆解 Sprint 任务
 ├── Generator → 读计划 → 自主实现 → 勾 checkbox → 写状态报告
 └── Evaluator → 独立重跑验收命令 → 写评估报告（信息性）
```

四个 subagent **互不共享上下文**，只通过文件通信。体验官与产品方通过 negotiation.md 进行结构化协商。

**核心隐喻：Reviewer 是挑剔的外部体验官，P/G/E 是内部研发团队。体验官先挑刺，研发团队再决定怎么改。**

---

## ⚠️ 前置条件

### 必须存在：`docs/plans/SPRINT.md`

至少包含产品基本信息和验收命令。任务清单可以很简洁，Planner 会在首轮基于体验官报告追加任务。

```markdown
# Sprint: 产品名称

## 产品信息
- 名称：XXX
- 简介：XXX
- 启动方式：npm run dev --port 5173
- 访问地址：http://localhost:5173

## 任务清单

- [ ] 初始任务（可选，Planner 会自动追加）

## 验收命令

```bash
npm run build
npm run lint
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
| 角色数 | 3（P/G/E） | 4（R/P/G/E） |
| 第一步 | Planner 读 Sprint | Reviewer 体验产品 |
| 停止条件 | Evaluator COMPLETE 即停 | 永远跑满 max_iter |
| 默认 max_iter | 2 | 1 |
| 外部视角 | 无 | 体验官独立审计 |
| 协商机制 | 无 | negotiation.md 闭环 |
| SPRINT.md | 用户预先写好完整任务 | Planner 每轮基于体验官报告追加任务 |

---

## 文件协议

| 文件 | 谁写 | 谁读 | 作用 |
|------|------|------|------|
| `docs/plans/SPRINT.md` | 用户 + Planner + Generator | R, P, G, E | Sprint 合同 |
| `docs/plans/pitfalls.md` | G + E（追加） | R, P, G, E | 陷阱知识库 |
| `docs/orch/product-audit-report.md` | Reviewer（Step A） | Planner（Step B） | 体验官审计报告 |
| `docs/orch/negotiation.md` | Planner（Step B） | Reviewer（下轮 Step A） | 产品方逐条回应 |
| `docs/orch/plan.md` | Planner（Step B） | Generator（Step C） | 本轮任务计划 |
| `docs/orch/gen_status.md` | Generator（Step C） | Evaluator（Step D） | 实现结果自报 |
| `docs/orch/eval.md` | Evaluator（Step D） | Planner（下轮） | 验收报告（信息性） |
