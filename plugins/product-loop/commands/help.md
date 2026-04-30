在运行 `/product-loop:product-loop` 之前，请先阅读此指南，确认所有前置条件就绪。

---

## 什么是 Product-Loop？

Product-Loop 在内部研发循环外层加入独立的 Reviewer，Reviewer 先审计，研发团队再响应。支持三种 Reviewer 模式：

```
Orchestrator（你当前的 Claude 会话）
 ├── Reviewer  → 审计产品 → 写审计报告（--mode 决定类型）
 │   ├── experience: Product Experience Reviewer（体验官）
 │   ├── evolution:  Product Evolution Reviewer（进化策略师）
 │   └── all:        两者并行
 ├── Planner   → 读审计报告 → 逐条回应 + 拆解 Sprint 任务
 ├── Generator → 读计划 → 自主实现 → 勾 checkbox → 写状态报告
 └── Evaluator → 独立重跑验收命令 → 写评估报告（信息性）
```

四个 subagent **互不共享上下文**，只通过文件通信。Reviewer 与产品方通过 negotiation.md 进行结构化协商。

**两种 Reviewer**：
- **Product Experience Reviewer**（体验官）：挑剔的外部体验者，从功能体验、审美品味、产品想象力审视产品
- **Product Evolution Reviewer**（进化策略师）：产品策略师，从核心完整性、竞争差距、功能深度、差异化提出功能进化方案

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

通过 `/agents` 创建。experience / all 模式需要。

### 必须存在（evolution / all 模式）：系统级 agent `product-evolution-reviewer`

通过 `/agents` 创建。evolution / all 模式需要。

### 建议存在：`docs/plans/pitfalls.md`

初建可留空，后续自动追加。

---

## 快速开始

```
# 默认：体验模式
/product-loop:product-loop

# 进化策略模式（专注功能进化）
/product-loop:product-loop --mode evolution

# 双审模式（体验 + 进化并行）
/product-loop:product-loop --mode all

# 可选参数组合
/product-loop:product-loop --mode experience --max-iter 3
/product-loop:product-loop --mode evolution --sprint docs/plans/MY_SPRINT.md --max-iter 5
```

---

## 与 multi-ralph 的区别

| | multi-ralph | product-loop |
|---|---|---|
| 角色数 | 3（P/G/E） | 4（R/P/G/E） |
| Reviewer 类型 | 无 | 体验官 / 进化策略师 / 两者并行 |
| 第一步 | Planner 读 Sprint | Reviewer 审计产品 |
| 停止条件 | Evaluator COMPLETE 即停 | 永远跑满 max_iter |
| 默认 max_iter | 2 | 1 |
| 外部视角 | 无 | Reviewer 独立审计 |
| 协商机制 | 无 | negotiation.md 闭环 |
| SPRINT.md | 用户预先写好完整任务 | Planner 每轮基于 Reviewer 报告追加任务 |

---

## 文件协议

| 文件 | 谁写 | 谁读 | 作用 |
|------|------|------|------|
| `docs/plans/SPRINT.md` | 用户 + Planner + Generator | R, P, G, E | Sprint 合同 |
| `docs/plans/pitfalls.md` | G + E（追加） | R, P, G, E | 陷阱知识库 |
| `docs/orch/product-audit-report.md` | Experience Reviewer（Step A） | Planner（Step B） | 体验官审计报告 |
| `docs/orch/evolution-audit-report.md` | Evolution Reviewer（Step A） | Planner（Step B） | 进化策略审计报告 |
| `docs/orch/negotiation.md` | Planner（Step B） | Reviewer（下轮 Step A） | 产品方逐条回应 |
| `docs/orch/plan.md` | Planner（Step B） | Generator（Step C） | 本轮任务计划 |
| `docs/orch/gen_status.md` | Generator（Step C） | Evaluator（Step D） | 实现结果自报 |
| `docs/orch/eval.md` | Evaluator（Step D） | Planner（下轮） | 验收报告（信息性） |
